"""`policy.upstream_identity_encoding` tests.

The key makes one change to one request: the client's `Accept-Encoding` is
replaced with `identity` on the request the router forwards upstream, so the
upstream replies in plain bytes and the token counts in that reply can be read.
It is off by default, and it is a request header only - nothing on a response is
touched.

Every test here drives the same in-process harness the other proxy tests use: an
upstream mock on loopback that records the exact request it received, and a
client that posts to the proxy and reads what comes back. Nothing here starts a
server of its own, runs the router as a subprocess or contacts a real upstream.

Config is `tools/sample-config-ACTIVE-test.yaml` with the policy key set as the
test needs. Prices are the dummy numbers from that file; the model ids are
invented.
"""
from __future__ import annotations

import contextlib
import gzip
import http.client
import json
import threading
import time
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from router.config import ConfigError, PolicySpec, RouterConfig, load_config
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.proxy import (
    IDENTITY_ENCODING,
    LOOPBACK_HOST,
    ProxySettings,
    create_server,
    upstream_identity_encoding,
)
from router.state import SessionStore

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-ACTIVE-test.yaml"

MODEL_MID = "test-mid"

OPENAI_PATH = "/v1/chat/completions"

#: What the client asks for in every test. Never the value the upstream sees when
#: the feature is on, which is what makes the two cases distinguishable.
CLIENT_ACCEPT_ENCODING = "gzip, br"

FAKE_AUTHORIZATION = "Bearer sk-ant-fake-token-DO-NOT-LOG-444"
FAKE_API_KEY = "sk-ant-api03-fake-key-DO-NOT-LOG-555"

#: How long the mock waits for the test to release the remaining stream chunks, so
#: a proxy that buffers the whole response fails the test instead of hanging it.
STREAM_GATE_TIMEOUT = 10.0

#: Client-side timeout. A proxy that buffers the whole streaming response trips
#: this instead of hanging the test run.
CLIENT_TIMEOUT = 5.0

INPUT_TOKENS = 31
OUTPUT_TOKENS = 7
CACHE_READ_TOKENS = 2

STREAM_CHUNKS: tuple[bytes, ...] = (
    b'data: {"type": "message_start", "message": {"model": "test-mid", '
    b'"usage": {"input_tokens": 31, "cache_read_input_tokens": 2}}}\n\n',
    b'data: {"type": "content_block_delta", "delta": {"text": "hi"}}\n\n',
    b'data: {"type": "message_delta", "usage": {"output_tokens": 7}}\n\n',
)
STREAM_BODY = b"".join(STREAM_CHUNKS)


def identity_config(
    *,
    enabled: bool | None = None,
    mode: str = "shadow",
) -> RouterConfig:
    """The sample config, with the policy key set as the test needs.

    `enabled=None` leaves the key out of the policy entirely, which is what a
    config written before the key existed looks like.
    """
    base = load_config(ACTIVE_CONFIG_PATH)
    assert base.policy is not None
    policy = base.policy
    if enabled is not None:
        policy = replace(base.policy, upstream_identity_encoding=enabled)
    return replace(
        base,
        mode=mode,
        policy=policy,
    )


def messages_body(model: str = MODEL_MID, stream: bool = False) -> bytes:
    """A Messages request no rule acts on, so the plan is a STAY."""
    return json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode()


def messages_reply(model: str = MODEL_MID) -> bytes:
    """A Messages response carrying usage, so a usage row can be checked."""
    return json.dumps(
        {
            "id": "msg_ie_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": INPUT_TOKENS,
                "output_tokens": OUTPUT_TOKENS,
                "cache_read_input_tokens": CACHE_READ_TOKENS,
                "cache_creation_input_tokens": 4,
            },
        }
    ).encode()


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def accept_encoding(self) -> str | None:
        return self.headers.get("accept-encoding")


@dataclass
class MockState:
    requests: list[Recorded] = field(default_factory=list)
    #: When set, the mock stops after its first SSE chunk and waits for
    #: `release_remaining`. Only the incremental-delivery test needs that; every
    #: other test wants the whole stream at once.
    hold_stream: bool = False
    release_remaining: threading.Event = field(default_factory=threading.Event)
    last_chunk_sent: threading.Event = field(default_factory=threading.Event)
    #: When true the mock ignores `identity` and compresses anyway, which is the
    #: response path the policy must leave completely alone.
    always_gzip: bool = False


class IdentityUpstream(BaseHTTPRequestHandler):
    """Records what it received, and compresses only when it was invited to.

    `Accept-Encoding: identity` means "do not compress", so a request carrying it
    is answered in plain bytes - which is the whole point of the policy. A
    request that asked for gzip gets a gzipped answer, byte for byte as sent.
    """

    protocol_version = "HTTP/1.1"
    server_version = "identity-upstream"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def state(self) -> MockState:
        return self.server.state  # type: ignore[attr-defined]

    @property
    def route(self) -> str:
        return self.path.split("?", 1)[0]

    def _record(self, body: bytes) -> None:
        self.state.requests.append(
            Recorded(
                method=self.command,
                path=self.path,
                headers={name.lower(): value for name, value in self.headers.items()},
                body=body,
            )
        )

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        length = int(raw_length) if raw_length else 0
        return self.rfile.read(length) if length else b""

    def _accepted(self, encoding: str) -> bool:
        """Whether the request's `Accept-Encoding` invited this encoding.

        `identity` never invites compression, whether it arrived on its own or
        beside `gzip`, so a header the router replaced cannot be read as a
        request for gzip it also left in place.
        """
        tokens = {
            token.strip().lower()
            for token in (self.headers.get("Accept-Encoding") or "").split(",")
            if token.strip()
        }
        if IDENTITY_ENCODING in tokens:
            return False
        return encoding in tokens

    def _send(self, status: int, payload: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Upstream", "identity-mock")
        if self.state.always_gzip or self._accepted("gzip"):
            payload = gzip.compress(payload)
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _not_found(self) -> None:
        payload = b"upstream said 404\n"
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for index, chunk in enumerate(STREAM_CHUNKS):
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
            if index == 0 and self.state.hold_stream:
                self.state.release_remaining.wait(timeout=STREAM_GATE_TIMEOUT)
        self.state.last_chunk_sent.set()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_POST(self) -> None:
        body = self._read_body()
        self._record(body)
        if self.route not in (MESSAGES_PATH, OPENAI_PATH):
            self._not_found()
            return

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            payload = {}
        if isinstance(payload, dict) and payload.get("stream") is True:
            self._stream()
            return
        self._send(200, messages_reply())

    def do_GET(self) -> None:
        self._record(b"")
        if self.route == "/static":
            payload = b"static body bytes\n"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            return
        self._not_found()


@dataclass
class Harness:
    proxy_port: int
    recorded: list[Recorded]
    state: MockState

    @property
    def last(self) -> Recorded:
        assert self.recorded, "the upstream received nothing"
        return self.recorded[-1]


@contextlib.contextmanager
def running_proxy(
    config: RouterConfig | None,
    db_path: Path | None = None,
    *,
    hold_stream: bool = False,
    always_gzip: bool = False,
) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), IdentityUpstream)
    upstream.daemon_threads = True
    state = MockState(hold_stream=hold_stream, always_gzip=always_gzip)
    upstream.state = state  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    decisions = DecisionLog(db_path) if db_path is not None else None
    server = create_server(
        ProxySettings(
            listen_port=0,
            upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
            mode="shadow" if config is None else config.mode,
            decisions=decisions,
            config=config,
            state=SessionStore() if config is not None else None,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Harness(
            proxy_port=server.server_address[1],
            recorded=state.requests,
            state=state,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def post(
    port: int,
    body: bytes,
    path: str = MESSAGES_PATH,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """One request; returns `(status, lowercased headers, body)`."""
    sent = {"Content-Type": "application/json", "Accept-Encoding": CLIENT_ACCEPT_ENCODING}
    sent.update(headers or {})
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        connection.request("POST", path, body=body, headers=sent)
        response = connection.getresponse()
        return (
            response.status,
            {name.lower(): value for name, value in response.getheaders()},
            response.read(),
        )
    finally:
        connection.close()


def get(port: int, path: str = "/static") -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        connection.request("GET", path, headers={"Accept-Encoding": CLIENT_ACCEPT_ENCODING})
        response = connection.getresponse()
        return (
            response.status,
            {name.lower(): value for name, value in response.getheaders()},
            response.read(),
        )
    finally:
        connection.close()


def log_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DecisionLog:
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    return DecisionLog.from_env()


def wait_for_usage(log: DecisionLog, expected: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while log.usage_count() < expected and time.monotonic() < deadline:
        time.sleep(0.01)


# --- the header ---------------------------------------------------------------


@pytest.mark.parametrize("path", [MESSAGES_PATH, OPENAI_PATH])
def test_the_clients_accept_encoding_is_replaced_when_the_policy_asks_for_it(path: str):
    with running_proxy(identity_config(enabled=True)) as harness:
        post(harness.proxy_port, messages_body(), path=path)

        recorded = harness.last
        assert recorded.accept_encoding == IDENTITY_ENCODING
        # One header, and the client's value is nowhere in it.
        assert recorded.accept_encoding != CLIENT_ACCEPT_ENCODING
        assert "gzip" not in (recorded.accept_encoding or "")


def test_a_client_that_sent_no_accept_encoding_still_gets_one_when_the_policy_asks():
    with running_proxy(identity_config(enabled=True)) as harness:
        connection = http.client.HTTPConnection(
            LOOPBACK_HOST, harness.proxy_port, timeout=CLIENT_TIMEOUT
        )
        try:
            connection.putrequest("POST", MESSAGES_PATH, skip_host=True, skip_accept_encoding=True)
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(messages_body())))
            connection.endheaders(messages_body())
            connection.getresponse().read()
        finally:
            connection.close()

        assert harness.last.accept_encoding == IDENTITY_ENCODING


@pytest.mark.parametrize("enabled", [None, False])
def test_the_clients_accept_encoding_is_forwarded_unchanged_otherwise(enabled: bool | None):
    with running_proxy(identity_config(enabled=enabled)) as harness:
        post(harness.proxy_port, messages_body())

        assert harness.last.accept_encoding == CLIENT_ACCEPT_ENCODING


def test_a_request_with_no_config_forwards_the_clients_accept_encoding_unchanged():
    """A config-less proxy is the shipped default: off, and the client's value."""
    with running_proxy(None) as harness:
        post(harness.proxy_port, messages_body())

        assert harness.last.accept_encoding == CLIENT_ACCEPT_ENCODING


def test_the_replacement_is_the_same_in_active_mode():
    with running_proxy(identity_config(enabled=True, mode="active")) as harness:
        status, _, payload = post(harness.proxy_port, messages_body())

        assert status == 200
        assert harness.last.accept_encoding == IDENTITY_ENCODING
        assert json.loads(payload)["model"] == MODEL_MID


# --- everything else about the request ----------------------------------------


def test_every_other_request_header_reaches_the_upstream_unchanged():
    body = messages_body()
    with running_proxy(identity_config(enabled=True)) as harness:
        post(
            harness.proxy_port,
            body,
            path=f"{MESSAGES_PATH}?marker=plain",
            headers={
                "Authorization": FAKE_AUTHORIZATION,
                "x-api-key": FAKE_API_KEY,
                "X-Custom-Trace": "trace-abc-123",
            },
        )

        recorded = harness.last
        assert recorded.method == "POST"
        assert recorded.path == f"{MESSAGES_PATH}?marker=plain"
        assert recorded.headers["authorization"] == FAKE_AUTHORIZATION
        assert recorded.headers["x-api-key"] == FAKE_API_KEY
        assert recorded.headers["x-custom-trace"] == "trace-abc-123"
        assert recorded.headers["content-type"] == "application/json"
        assert recorded.headers["content-length"] == str(len(body))
        # The one difference from the client's request, and nothing else.
        assert recorded.accept_encoding == IDENTITY_ENCODING


def test_the_body_reaches_the_upstream_byte_for_byte():
    payload = bytes(range(256)) * 4
    with running_proxy(identity_config(enabled=True)) as harness:
        post(harness.proxy_port, payload, headers={"Content-Type": "application/octet-stream"})

        recorded = harness.last
        assert recorded.body == payload
        assert recorded.headers["content-length"] == str(len(payload))


def test_a_get_is_forwarded_the_same_way():
    with running_proxy(identity_config(enabled=True)) as harness:
        status, _, payload = get(harness.proxy_port)

        assert status == 200
        assert payload == b"static body bytes\n"
        assert harness.last.method == "GET"
        assert harness.last.accept_encoding == IDENTITY_ENCODING


def test_an_unknown_path_is_forwarded_unchanged_too():
    """The key is a transport header, not a routing decision: no path is special."""
    body = messages_body()
    with running_proxy(identity_config(enabled=True)) as harness:
        status, _, _ = post(harness.proxy_port, body, path="/v1/other")

        assert status == 404
        assert harness.last.path == "/v1/other"
        assert harness.last.body == body
        assert harness.last.accept_encoding == IDENTITY_ENCODING


# --- nothing on the response --------------------------------------------------


def test_a_response_is_relayed_exactly_as_the_upstream_sent_it():
    """The upstream ignores `identity` and compresses anyway.

    The client must get the gzipped bytes and the header that says so: the
    router decodes for counting and forwards what it was given, never its own
    re-encoded copy.
    """
    with running_proxy(identity_config(enabled=True), always_gzip=True) as harness:
        status, headers, payload = post(harness.proxy_port, messages_body())

        assert status == 200
        assert headers["content-encoding"] == "gzip"
        assert payload[:2] == b"\x1f\x8b"
        assert json.loads(gzip.decompress(payload))["model"] == MODEL_MID
        assert harness.last.accept_encoding == IDENTITY_ENCODING


def test_no_accept_encoding_is_ever_added_to_a_response():
    with running_proxy(identity_config(enabled=True)) as harness:
        _, headers, _ = post(harness.proxy_port, messages_body())

        assert "accept-encoding" not in headers


def test_the_headers_the_upstream_sent_are_relayed_unchanged():
    with running_proxy(identity_config(enabled=True)) as harness:
        _, headers, _ = post(harness.proxy_port, messages_body())

        assert headers["x-upstream"] == "identity-mock"
        assert headers["content-type"] == "application/json"
        assert headers["content-length"] == str(len(messages_reply()))


# --- streaming ----------------------------------------------------------------


def test_a_stream_is_still_delivered_incrementally():
    with running_proxy(identity_config(enabled=True), hold_stream=True) as harness:
        connection = http.client.HTTPConnection(
            LOOPBACK_HOST, harness.proxy_port, timeout=CLIENT_TIMEOUT
        )
        try:
            body = messages_body(stream=True)
            connection.request(
                "POST",
                MESSAGES_PATH,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept-Encoding": CLIENT_ACCEPT_ENCODING,
                },
            )
            response = connection.getresponse()

            assert response.status == 200
            first = response.read1(65536)

            assert first == STREAM_CHUNKS[0]
            assert not harness.state.last_chunk_sent.is_set(), "proxy buffered the whole response"

            harness.state.release_remaining.set()
            received = bytearray(first)
            while True:
                piece = response.read1(65536)
                if not piece:
                    break
                received += piece
        finally:
            connection.close()

        assert bytes(received) == STREAM_BODY
        assert harness.last.accept_encoding == IDENTITY_ENCODING


# --- usage --------------------------------------------------------------------


def test_usage_is_read_from_the_uncompressed_response_it_asked_for(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(identity_config(enabled=True), log.db_path) as harness:
        status, headers, payload = post(harness.proxy_port, messages_body())

        assert harness.last.accept_encoding == IDENTITY_ENCODING
        assert status == 200
        # The upstream honoured `identity`, so the answer really is uncompressed.
        assert "content-encoding" not in headers
        assert payload == messages_reply()
        wait_for_usage(log, 1)

        row = log.usage_rows(1)[0]
        assert row.status == "OK"
        assert row.input_tokens == INPUT_TOKENS
        assert row.output_tokens == OUTPUT_TOKENS
        assert row.cache_read_tokens == CACHE_READ_TOKENS
        assert row.model_reported == MODEL_MID
        assert row.decision_id == log.recent(1)[0].decision_id


def test_a_compressed_response_is_relayed_compressed_and_still_counted(tmp_path, monkeypatch):
    """With the key off the upstream gzips, and nothing about that changes."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(identity_config(enabled=False), log.db_path) as harness:
        status, headers, payload = post(harness.proxy_port, messages_body())

        assert harness.last.accept_encoding == CLIENT_ACCEPT_ENCODING
        assert status == 200
        assert headers["content-encoding"] == "gzip"
        assert payload[:2] == b"\x1f\x8b"
        assert gzip.decompress(payload) == messages_reply()
        wait_for_usage(log, 1)

        assert log.usage_rows(1)[0].input_tokens == INPUT_TOKENS


def test_the_key_records_nothing_of_its_own(tmp_path, monkeypatch):
    """The header is transport. It adds no reason code and no signal value."""
    log = log_at(tmp_path, monkeypatch)
    # The downgrade rule is off so the row is a plain STAY and nothing the policy
    # did on its own can be mistaken for something the header did.
    base = identity_config(enabled=True)
    assert base.policy is not None
    config = replace(base, policy=replace(base.policy, downgrade_enabled=False))

    with running_proxy(config, log.db_path) as harness:
        post(harness.proxy_port, messages_body())
        deadline = time.monotonic() + 5.0
        while log.count() < 1 and time.monotonic() < deadline:
            time.sleep(0.01)

        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.reason_codes == ["NO_RULE_MATCHED"]
        assert "upstream_identity_encoding" not in row.signal_values
        assert "accept" not in json.dumps(row.signal_values).lower()


# --- the default is off -------------------------------------------------------


def test_the_reader_is_off_for_a_config_that_never_heard_of_the_key():
    base = load_config(ACTIVE_CONFIG_PATH)
    assert base.policy is not None
    older = replace(base.policy, upstream_identity_encoding=False)

    assert upstream_identity_encoding(base) is False
    assert upstream_identity_encoding(replace(base, policy=older)) is False
    assert upstream_identity_encoding(replace(base, policy=None)) is False


@pytest.mark.parametrize("config", [None, object()])
def test_the_reader_is_off_without_a_config(config: Any):
    assert upstream_identity_encoding(config) is False


@pytest.mark.parametrize("value", ["true", 1, None, "identity", [True]])
def test_a_value_that_is_not_boolean_true_does_not_enable_the_key(value: Any):
    base = load_config(ACTIVE_CONFIG_PATH)
    assert base.policy is not None

    config = replace(base, policy=replace(base.policy, upstream_identity_encoding=value))

    assert upstream_identity_encoding(config) is False


def test_the_reader_follows_the_policy_when_it_is_boolean_true():
    assert upstream_identity_encoding(identity_config(enabled=True)) is True


def test_a_policy_spec_built_by_hand_defaults_the_key_off():
    policy = PolicySpec(
        escalate_consecutive_errors=2,
        escalate_repeated_tool_calls=3,
        downgrade_enabled=True,
        downgrade_max_context_tokens=2000,
        downgrade_max_turn_index=1,
        dwell_requests=5,
        hysteresis_requests=3,
    )

    assert policy.upstream_identity_encoding is False


def test_the_sample_config_leaves_the_key_off():
    config = load_config(ACTIVE_CONFIG_PATH)

    assert config.policy is not None
    assert config.policy.upstream_identity_encoding is False


def test_the_shipped_config_leaves_the_key_off():
    config = load_config(REPO_ROOT / "router" / "config.yaml")

    assert config.policy is not None
    assert config.policy.upstream_identity_encoding is False


# --- the config loader --------------------------------------------------------


def _policy_config(tmp_path: Path, **policy: Any) -> Path:
    """A config file carrying `policy` on top of the sample's own block.

    Built from the sample rather than hand-written, so the fixture cannot drift
    from the shape the loader accepts. It is written into `tmp_path`: the shipped
    `router/config.yaml` is hand-edited and never dumped (Rule 9).
    """
    data = yaml.safe_load(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert isinstance(data["policy"], dict)
    data["policy"].update(policy)
    path = tmp_path / "identity-policy.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_the_loader_defaults_the_key_off_when_it_is_absent(tmp_path):
    config = load_config(_policy_config(tmp_path))

    assert config.policy is not None
    assert config.policy.upstream_identity_encoding is False


def test_the_loader_reads_the_key(tmp_path):
    config = load_config(_policy_config(tmp_path, upstream_identity_encoding=True))

    assert config.policy is not None
    assert config.policy.upstream_identity_encoding is True


@pytest.mark.parametrize("value", ["yes", 1, None, "true", 0])
def test_the_loader_refuses_an_illegal_value(tmp_path, value: Any):
    with pytest.raises(ConfigError, match="upstream_identity_encoding"):
        load_config(_policy_config(tmp_path, upstream_identity_encoding=value))