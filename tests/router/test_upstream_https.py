"""Forwarding to an https upstream, including one with a path prefix.

Every test here drives `router.proxy.ProxyHandler` in process with an injected
fake connection factory, so no socket is opened and no name is resolved: the
fake stands in for `http.client.HTTPSConnection` and answers with bytes the test
chose. Nothing in this file contacts an upstream, real or fake-over-a-socket.

What is asserted is the wire the router builds and the answers it maps to:

* the upstream URL's path prefix is put in front of the request path and the
  client's query follows it, with and without a prefix and with trailing slashes;
* an https upstream gets `HTTPSConnection` and `ssl.create_default_context()`,
  with certificate and hostname verification on and port 443 by default;
* a plain http upstream is refused unless the host is loopback, and credentials,
  a query or a fragment are refused at all;
* the `Host` header is the upstream's authority;
* a 3xx is relayed as it arrived and never followed;
* a refused connection or a failed name is a 502, a timeout is a 504, and a TLS
  failure is a 502 carrying the code alone;
* a streaming response over https is still relayed one chunk at a time;
* no credential, header value or certificate detail reaches the log;
* active mode refuses a `FILL_FROM_` model id, exactly as it refuses `TODO_`.

`FAKE_AUTHORIZATION` and friends exist so the "never logged" assertions are
about real header values rather than about the absence of a test fixture.
"""
from __future__ import annotations

import contextlib
import io
import json
import socket
import ssl
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from router import cli as cli_module
from router.config import (
    LOOPBACK_UPSTREAM_HOSTS,
    ConfigError,
    load_config,
    parse_upstream,
)
from router.decisions import DB_ENV_VAR
from router.proxy import (
    DEFAULT_UPSTREAM_TIMEOUT,
    LOOPBACK_HOST,
    ProxyError,
    ProxyHandler,
    ProxySettings,
    UpstreamTarget,
    create_server,
    parse_upstream as parse_target,
)

FAKE_AUTHORIZATION = "Bearer sk-or-v1-fake-DO-NOT-LOG-444"
FAKE_API_KEY = "sk-or-v1-key-DO-NOT-LOG-555"
FAKE_PASSWORD = "hunter2-DO-NOT-LOG-666"

STREAM_CHUNKS = [
    b'data: {"index": 0}\n\n',
    b'data: {"index": 1}\n\n',
    b'data: {"index": 2}\n\n',
]
STREAM_BODY = b"".join(STREAM_CHUNKS)

OPENROUTER = "https://openrouter.ai/api"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point the decision log, and the kill switch that sits beside it, at tmp.

    The proxy reads the kill switch on every request it recognises. Left alone it
    would look next to the repository's own database, so a `tamias-router off`
    run on this machine would change what these tests see.
    """
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))


# --- the fake upstream -----------------------------------------------------


class Sink:
    """What the client would have received, byte for byte and write by write."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    @property
    def buffer(self) -> bytes:
        return b"".join(self.writes)


class FakeResponse:
    """The upstream's answer: headers, a status and a body in pieces."""

    def __init__(
        self,
        status: int = 200,
        reason: str = "OK",
        headers: list[tuple[str, str]] | None = None,
        chunks: list[bytes] | None = None,
        chunked: bool = False,
        sink: Sink | None = None,
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = list(headers) if headers is not None else []
        self.chunks = list(chunks) if chunks is not None else []
        self.chunked = chunked
        self.sink = sink
        self.index = 0
        #: What the client already had when the proxy asked for the second chunk.
        #: Set during the relay, and empty if the proxy asked for it up front,
        #: which would mean it buffered the whole response first.
        self.before_second_chunk = b""

    def getheaders(self) -> list[tuple[str, str]]:
        return list(self.headers)

    def getheader(self, name: str, default: str | None = None) -> str | None:
        for header, value in self.headers:
            if header.lower() == name.lower():
                return value
        return default

    def read1(self, amount: int) -> bytes:
        if self.index >= len(self.chunks):
            return b""
        if self.index == 1 and self.sink is not None:
            self.before_second_chunk = self.sink.buffer
        piece = self.chunks[self.index]
        self.index += 1
        return piece


class FakeConnection:
    """What the proxy uses in place of an `http.client` connection."""

    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.request_line: tuple[str, str] | None = None
        self.headers: list[tuple[str, str]] = []
        self.body = b""
        self.closed = False

    def putrequest(
        self,
        method: str,
        url: str,
        skip_host: bool = False,
        skip_accept_encoding: bool = False,
    ) -> None:
        self.request_line = (method, url)

    def putheader(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def endheaders(self, body: bytes | None = None) -> None:
        self.body = body or b""

    def getresponse(self) -> FakeResponse:
        if self.error is not None:
            raise self.error
        assert self.response is not None, "the test built a connection with no answer"
        return self.response

    def close(self) -> None:
        self.closed = True

    def header(self, name: str) -> str | None:
        for header, value in self.headers:
            if header.lower() == name.lower():
                return value
        return None


class FailsMidBody(FakeResponse):
    """Answers, then fails the way a dropped TLS connection does."""

    def read1(self, amount: int) -> bytes:
        if self.index >= 1:
            raise ssl.SSLError("the tls stream broke mid-body")
        return super().read1(amount)


@dataclass
class FakeUpstream:
    """The injected factory: what it was asked for, and what it handed back."""

    response: FakeResponse | None = None
    error: Exception | None = None
    opened: list[tuple[UpstreamTarget, float]] = field(default_factory=list)
    connections: list[FakeConnection] = field(default_factory=list)
    sink: Sink | None = None

    def __call__(
        self, target: UpstreamTarget, timeout: float
    ) -> FakeConnection:
        self.opened.append((target, timeout))
        connection = FakeConnection(self.response, self.error)
        self.connections.append(connection)
        return connection

    @property
    def only(self) -> FakeConnection:
        assert len(self.connections) == 1, (
            f"expected one upstream connection, got {len(self.connections)}"
        )
        return self.connections[0]


@dataclass
class _FakeServer:
    """The two things `ProxyHandler` reads off its server."""

    settings: ProxySettings
    target: UpstreamTarget


class Harness(ProxyHandler):
    """`ProxyHandler` fed from a buffer, with no socket on either side.

    `BaseHTTPRequestHandler.__init__` is what would set up a socket, so it is not
    called: `handle_one_request` - the stdlib's own request parser - is, against
    `rfile` as a `BytesIO` and `wfile` as a `Sink`. Everything the router does
    with the request and the response therefore runs exactly as it would on a
    real connection.
    """

    def __init__(self, server: _FakeServer, request: bytes, wfile: Sink) -> None:
        self.server = server
        self.request = None
        self.client_address = (LOOPBACK_HOST, 0)
        self.rfile = io.BytesIO(request)
        self.wfile = wfile
        self.connection = wfile
        self.raw_requestline = b""
        self.requestline = ""
        self.request_version = ""
        self.command = ""
        self.close_connection = False

    def finish(self) -> None:
        """Nothing to close: both ends are buffers."""


@dataclass
class Drove:
    """One request, end to end: what went upstream and what came back."""

    status: int
    reason: str
    headers: list[tuple[str, str]]
    body: bytes
    raw: bytes
    connection: FakeConnection
    upstream: FakeUpstream
    stderr: str

    @property
    def url(self) -> str:
        assert self.connection.request_line is not None, "nothing was sent upstream"
        return self.connection.request_line[1]

    @property
    def json(self) -> Any:
        return json.loads(self.body)

    def header(self, name: str) -> str | None:
        for header, value in self.headers:
            if header.lower() == name.lower():
                return value
        return None


def raw_request(
    method: str = "GET",
    path: str = "/v1/models",
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> bytes:
    lines = [f"{method} {path} HTTP/1.1", f"Host: {LOOPBACK_HOST}:8787"]
    lines.extend(f"{name}: {value}" for name, value in (headers or {}).items())
    lines.append(f"Content-Length: {len(body)}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body


def answer(
    status: int = 200,
    reason: str = "OK",
    body: bytes = b"upstream said so\n",
    headers: list[tuple[str, str]] | None = None,
) -> FakeResponse:
    all_headers = [("Content-Type", "text/plain")]
    all_headers.extend(headers or [])
    all_headers.append(("Content-Length", str(len(body))))
    return FakeResponse(
        status=status,
        reason=reason,
        headers=all_headers,
        chunks=[body] if body else [],
    )


def drive(
    upstream: str = OPENROUTER,
    request: bytes | None = None,
    response: FakeResponse | None = None,
    error: Exception | None = None,
    settings: ProxySettings | None = None,
) -> Drove:
    """Run one request through the handler against the injected fake."""
    target = parse_target(upstream)
    sink = Sink()
    if response is None and error is None:
        response = answer()
    fake = FakeUpstream(response=response, error=error, sink=sink)
    if response is not None and response.sink is None:
        # The response records what the client already had, so it has to see the
        # sink the client is reading from.
        response.sink = sink
    base = settings if settings is not None else ProxySettings()
    server = _FakeServer(
        settings=replace(base, upstream=upstream, connection_factory=fake),
        target=target,
    )
    log = io.StringIO()
    with contextlib.redirect_stderr(log):
        handler = Harness(
            server, request if request is not None else raw_request(), sink
        )
        handler.handle_one_request()

    head, _, body = sink.buffer.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    parts = lines[0].split(b" ", 2)
    status = int(parts[1])
    reason = parts[2].decode("latin-1") if len(parts) > 2 else ""
    headers = [
        (name.decode("latin-1").strip(), value.decode("latin-1").strip())
        for name, _, value in (line.partition(b":") for line in lines[1:])
    ]
    return Drove(
        status=status,
        reason=reason,
        headers=headers,
        body=body,
        raw=sink.buffer,
        connection=fake.only,
        upstream=fake,
        stderr=log.getvalue(),
    )


def dechunk(raw: bytes) -> bytes:
    """The body a client would decode from chunked framing."""
    out = bytearray()
    rest = raw
    while True:
        size_line, _, rest = rest.partition(b"\r\n")
        size = int(size_line.split(b";", 1)[0], 16)
        if size == 0:
            return bytes(out)
        out += rest[:size]
        rest = rest[size + 2 :]


# --- the URL the router builds ---------------------------------------------


def test_the_prefix_from_the_upstream_url_comes_first():
    drove = drive(OPENROUTER, raw_request("POST", "/v1/chat/completions", b"{}"))

    assert drove.url == "/api/v1/chat/completions"


def test_an_upstream_without_a_prefix_leaves_the_path_alone():
    drove = drive("https://openrouter.ai", raw_request("POST", "/v1/messages", b"{}"))

    assert drove.url == "/v1/messages"


@pytest.mark.parametrize(
    ("upstream", "expected"),
    [
        ("https://openrouter.ai/api", "/api/v1/models"),
        ("https://openrouter.ai/api/", "/api/v1/models"),
        ("https://openrouter.ai/api///", "/api/v1/models"),
        ("https://openrouter.ai/", "/v1/models"),
        ("https://openrouter.ai", "/v1/models"),
        ("http://localhost:8931/gw/v2", "/gw/v2/v1/models"),
    ],
)
def test_a_trailing_slash_on_the_prefix_never_doubles(upstream, expected):
    drove = drive(upstream, raw_request("GET", "/v1/models"))

    assert drove.url == expected


def test_the_clients_query_is_forwarded_after_the_prefix():
    drove = drive(
        OPENROUTER, raw_request("POST", "/v1/chat/completions?stream=true&beta=1", b"{}")
    )

    assert drove.url == "/api/v1/chat/completions?stream=true&beta=1"


def test_a_prefix_that_is_only_slashes_forwards_the_path_unchanged():
    drove = drive("https://openrouter.ai/", raw_request("GET", "/v1/models?limit=1"))

    assert drove.url == "/v1/models?limit=1"


# --- https, and verification that cannot be turned off ----------------------


def test_an_https_upstream_gets_a_tls_connection_with_verification_on(monkeypatch):
    """The real `open_upstream`, with the connection classes swapped out.

    `HTTPS_CONNECTION` and `HTTP_CONNECTION` are bound in `router.proxy` exactly
    so this can be done: the arguments the router would have passed are what the
    test asserts on, and no socket is opened.
    """
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class Fake:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append((type(self).__name__, args, kwargs))

    class FakeHTTPS(Fake):
        pass

    class FakeHTTP(Fake):
        pass

    monkeypatch.setattr("router.proxy.HTTPS_CONNECTION", FakeHTTPS)
    monkeypatch.setattr("router.proxy.HTTP_CONNECTION", FakeHTTP)

    parse_target(OPENROUTER).connect(DEFAULT_UPSTREAM_TIMEOUT)
    parse_target("http://127.0.0.1:8931").connect(DEFAULT_UPSTREAM_TIMEOUT)

    https_name, https_args, https_kwargs = calls[0]
    assert https_name == "FakeHTTPS", "an https upstream was not dialled over TLS"
    assert https_args[0] == "openrouter.ai"
    assert https_args[1] == 443, "the default https port is not 443"
    assert https_kwargs["timeout"] == DEFAULT_UPSTREAM_TIMEOUT

    context = https_kwargs["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True

    http_name, http_args, http_kwargs = calls[1]
    assert http_name == "FakeHTTP", "a loopback http upstream was dialled over TLS"
    assert http_args == ("127.0.0.1", 8931)
    assert "context" not in http_kwargs


def test_an_explicit_port_is_used_as_the_url_gave_it(monkeypatch):
    calls: list[tuple[Any, ...]] = []

    class FakeHTTPS:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(args)

    monkeypatch.setattr("router.proxy.HTTPS_CONNECTION", FakeHTTPS)

    parse_target("https://gateway.internal:8443/api").connect(1.0)

    assert calls[0] == ("gateway.internal", 8443)


def test_the_factory_is_handed_the_https_target_and_the_timeout():
    drove = drive(OPENROUTER)

    target, timeout = drove.upstream.opened[0]
    assert target.is_tls is True
    assert target.host == "openrouter.ai"
    assert target.port is None, "the URL named no port, so none is invented"
    assert timeout == DEFAULT_UPSTREAM_TIMEOUT


# --- what a URL is allowed to be -------------------------------------------


@pytest.mark.parametrize("host", ["example.com", "10.0.0.5", "openrouter.ai"])
def test_plain_http_to_a_non_loopback_host_is_refused(host):
    with pytest.raises(ProxyError, match="refusing plain http upstream"):
        parse_target(f"http://{host}/api")


@pytest.mark.parametrize("host", sorted(LOOPBACK_UPSTREAM_HOSTS))
def test_plain_http_to_a_loopback_host_is_allowed(host):
    target = parse_target(f"http://{host}:8931/api")

    assert target.is_tls is False
    assert target.host == host


def test_a_config_with_a_non_loopback_http_upstream_will_not_load(tmp_path):
    path = tmp_path / "plain-http.yaml"
    path.write_text(
        "\n".join(
            [
                "listen: 127.0.0.1:8787",
                "upstream: http://api.example.com",
                "mode: shadow",
                "models:",
                "  - id: vendor/real-low",
                "    legal_efforts: [a]",
                "    cost_tier: low",
                "default_model: vendor/real-low",
                "default_effort: a",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="refusing plain http upstream"):
        load_config(path)


@pytest.mark.parametrize(
    ("url", "match"),
    [
        (f"https://user:{FAKE_PASSWORD}@openrouter.ai/api", "must not carry credentials"),
        ("https://user@openrouter.ai/api", "must not carry credentials"),
        ("https://openrouter.ai/api?key=abc", "query string or fragment"),
        ("https://openrouter.ai/api#frag", "query string or fragment"),
        ("ftp://openrouter.ai/api", "scheme must be http or https"),
        ("https:///api", "must include a host"),
    ],
)
def test_an_upstream_that_carries_anything_it_should_not_is_refused(url, match):
    with pytest.raises(ProxyError, match=match) as caught:
        parse_target(url)

    assert FAKE_PASSWORD not in str(caught.value), "a credential was quoted back"


def test_a_config_with_credentials_in_the_upstream_will_not_load(tmp_path):
    path = tmp_path / "credentials.yaml"
    path.write_text(
        "\n".join(
            [
                "listen: 127.0.0.1:8787",
                f"upstream: https://user:{FAKE_PASSWORD}@openrouter.ai/api",
                "mode: shadow",
                "models:",
                "  - id: vendor/real-low",
                "    legal_efforts: [a]",
                "    cost_tier: low",
                "default_model: vendor/real-low",
                "default_effort: a",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as caught:
        load_config(path)

    assert "credentials" in str(caught.value)
    assert FAKE_PASSWORD not in str(caught.value)


def test_start_refuses_a_non_loopback_http_override(capsys, tmp_path, monkeypatch):
    """`--upstream` is validated by the same rules the config load uses."""
    config = tmp_path / "shadow.yaml"
    config.write_text(
        "\n".join(
            [
                "listen: 127.0.0.1:8787",
                "upstream: https://openrouter.ai/api",
                "mode: shadow",
                "models:",
                "  - id: vendor/real-low",
                "    legal_efforts: [a]",
                "    cost_tier: low",
                "default_model: vendor/real-low",
                "default_effort: a",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli_module,
        "serve",
        lambda settings: pytest.fail("the proxy was started on a refused upstream"),
    )

    code = cli_module.main(
        ["--config", str(config), "start", "--upstream", "http://api.example.com"]
    )

    assert code == 1
    assert "refusing plain http upstream" in capsys.readouterr().err


# --- the Host header -------------------------------------------------------


def test_the_host_header_is_the_upstream_host_without_a_default_port():
    drove = drive(OPENROUTER)

    assert drove.connection.header("Host") == "openrouter.ai"


def test_the_host_header_carries_a_port_the_url_named():
    drove = drive("https://gateway.internal:8443/api")

    assert drove.connection.header("Host") == "gateway.internal:8443"


def test_the_host_header_for_a_loopback_http_upstream_carries_its_port():
    drove = drive("http://127.0.0.1:8931")

    assert drove.connection.header("Host") == "127.0.0.1:8931"


# --- redirects are relayed, never followed ----------------------------------


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_relayed_as_it_arrived_and_never_followed(status):
    reason = {301: "Moved Permanently", 302: "Found", 303: "See Other"}.get(status, "Redirect")
    upstream = answer(
        status=status,
        reason=reason,
        body=b"",
        headers=[("Location", "https://elsewhere.example/v1/messages")],
    )

    drove = drive(OPENROUTER, raw_request("POST", "/v1/messages", b"{}"), upstream)

    assert drove.status == status
    assert drove.header("Location") == "https://elsewhere.example/v1/messages"
    assert drove.url == "/api/v1/messages", "the router followed the Location itself"
    assert len(drove.upstream.connections) == 1, "a second request was sent"


# --- failures, and what the client is told ---------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError("connection refused"),
        socket.gaierror(-2, "name or service not known"),
        OSError("upstream is down"),
    ],
)
def test_an_unreachable_upstream_is_a_502(error):
    drove = drive(OPENROUTER, error=error)

    assert drove.status == 502
    assert drove.json["error"] == "upstream_unreachable"
    assert drove.json["message"]


def test_a_timeout_is_a_504():
    drove = drive(OPENROUTER, error=TimeoutError("timed out"))

    assert drove.status == 504
    assert drove.json["error"] == "upstream_timeout"


def test_a_tls_failure_is_a_502_that_says_only_the_code():
    """A certificate failure must not describe the certificate.

    The exception message carries the peer's details - which certificate, whose
    issuer, which hostname - and none of it belongs in a response the client can
    read or in anything written down.
    """
    error = ssl.SSLCertVerificationError(
        "certificate verify failed: self-signed certificate (_ssl.c:1)"
    )
    drove = drive(OPENROUTER, error=error)

    assert drove.status == 502
    assert drove.json == {"error": "upstream_tls_error"}
    text = (drove.body + drove.stderr.encode()).decode("utf-8", "replace")
    for leak in ("certificate", "self-signed", "_ssl.c", FAKE_AUTHORIZATION):
        assert leak not in text


def test_a_plain_tls_failure_is_mapped_the_same_way():
    drove = drive(OPENROUTER, error=ssl.SSLError("handshake failure"))

    assert drove.status == 502
    assert drove.json == {"error": "upstream_tls_error"}


def test_a_tls_failure_after_the_head_was_sent_does_not_write_a_second_head():
    """The client's response is never replaced by one the router invented."""
    response = FailsMidBody(
        status=200,
        headers=[("Content-Type", "text/plain")],
        chunks=[b"partial body, never finished"],
    )

    drove = drive(OPENROUTER, response=response)

    assert drove.status == 200
    assert drove.raw.startswith(b"HTTP/1.1 200 OK")
    assert drove.raw.count(b"HTTP/1.1 ") == 1
    assert drove.raw.endswith(b"partial body, never finished")


# --- streaming over https --------------------------------------------------


def test_a_streaming_response_over_https_is_still_relayed_chunk_by_chunk():
    """Chunk at a time, not buffered: the first chunk is out before the next.

    `before_second_chunk` is what the client already held at the moment the proxy
    asked the upstream for the following chunk. If the response were buffered,
    it would be empty and the assertion below would fail.
    """
    response = FakeResponse(
        status=200,
        headers=[
            ("Content-Type", "text/event-stream"),
            ("Transfer-Encoding", "chunked"),
        ],
        chunks=STREAM_CHUNKS,
        chunked=True,
    )

    drove = drive(OPENROUTER, raw_request("GET", "/api/v1/chat/completions"), response)

    assert drove.header("Transfer-Encoding") == "chunked"
    assert dechunk(drove.body) == STREAM_BODY
    framed_first = b"%x\r\n" % len(STREAM_CHUNKS[0]) + STREAM_CHUNKS[0] + b"\r\n"
    assert response.before_second_chunk.endswith(
        framed_first
    ), "the first chunk was not on the wire before the second was requested"


# --- credentials are forwarded and never logged ----------------------------


def test_credentials_are_forwarded_upstream_and_never_written_to_the_log():
    drove = drive(
        OPENROUTER,
        raw_request(
            "POST",
            "/v1/messages",
            b"{}",
            headers={"Authorization": FAKE_AUTHORIZATION, "x-api-key": FAKE_API_KEY},
        ),
        answer(),
    )

    assert drove.connection.header("Authorization") == FAKE_AUTHORIZATION
    assert drove.connection.header("x-api-key") == FAKE_API_KEY

    assert drove.stderr.strip(), "the proxy logged nothing for a request"
    assert "ms" in drove.stderr
    assert "POST /v1/messages 200" in drove.stderr, "the query is dropped from the log"
    for secret in (FAKE_AUTHORIZATION, FAKE_API_KEY, FAKE_PASSWORD):
        assert secret not in drove.stderr


def test_a_tls_failure_is_not_logged_with_the_peer_s_details():
    drove = drive(
        OPENROUTER,
        error=ssl.SSLCertVerificationError(
            "hostname 'openrouter.ai' doesn't match 'CN=api.example.com'"
        ),
    )

    assert "openrouter.ai" not in drove.stderr
    assert "api.example.com" not in drove.stderr


# --- the startup guard -----------------------------------------------------


ACTIVE_CONFIG = "\n".join(
    [
        "listen: 127.0.0.1:8787",
        "upstream: https://openrouter.ai/api",
        "mode: {mode}",
        "models:",
        "  - id: {first}",
        "    legal_efforts: [a]",
        "    cost_tier: low",
        "  - id: vendor/real-low",
        "    legal_efforts: [a]",
        "    cost_tier: low",
        "default_model: {first}",
        "default_effort: a",
    ]
)


def write_config(tmp_path: Path, mode: str, first_model: str) -> Path:
    path = tmp_path / f"case-{abs(hash((mode, first_model)))}.yaml"
    path.write_text(
        ACTIVE_CONFIG.format(mode=mode, first=first_model),
        encoding="utf-8",
    )
    return path


def test_active_mode_refuses_a_fill_from_model_id(tmp_path):
    path = write_config(tmp_path, "active", "FILL_FROM_MODEL_TIER_LOW")

    blocker = cli_module.active_start_blocker(load_config(path))

    assert blocker is not None
    assert "FILL_FROM_MODEL_TIER_LOW" in blocker


def test_active_start_exits_non_zero_on_a_fill_from_model_id(tmp_path, capsys, monkeypatch):
    path = write_config(tmp_path, "active", "FILL_FROM_MODEL_TIER_LOW")

    monkeypatch.setattr(
        cli_module,
        "serve",
        lambda settings: pytest.fail("active mode started on a placeholder id"),
    )

    code = cli_module.main(["--config", str(path), "start"])

    assert code == 1
    error = capsys.readouterr().err
    assert "active" in error
    assert "FILL_FROM_" in error


def test_active_mode_still_refuses_a_todo_model_id(tmp_path):
    path = write_config(tmp_path, "active", "TODO_MODEL_TIER_LOW")

    blocker = cli_module.active_start_blocker(load_config(path))

    assert blocker is not None
    assert "TODO_MODEL_TIER_LOW" in blocker


def test_shadow_mode_accepts_a_fill_from_model_id(tmp_path):
    path = write_config(tmp_path, "shadow", "FILL_FROM_MODEL_TIER_LOW")

    config = load_config(path)

    assert config.mode == "shadow"
    assert cli_module.active_start_blocker(config) is None


def test_active_mode_is_allowed_when_every_id_is_real(tmp_path):
    path = write_config(tmp_path, "active", "vendor/real-mid")
    config = load_config(path)

    assert cli_module.active_start_blocker(config) is None


# --- the listener and the shipped defaults ---------------------------------


def test_the_loopback_listener_is_unchanged_by_any_of_this():
    with pytest.raises(ProxyError, match="refusing to listen"):
        create_server(
            ProxySettings(
                listen_host="0.0.0.0", listen_port=0, upstream="http://127.0.0.1:1"
            )
        )


def test_the_shipped_config_still_loads_and_still_names_an_https_upstream():
    config = load_config()

    assert parse_upstream(config.upstream).scheme == "https"