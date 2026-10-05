"""Token usage: what a response reported, priced or not.

Two things are tested here, and they are deliberately tested separately.

`UsageParser` on its own: does it read the four counts and the model out of a
JSON body and out of an SSE stream, including when an event is split across two
chunks, and does it say UNKNOWN rather than guess when it cannot.

The proxy around it: the bytes, headers and framing the client receives are
exactly what the upstream sent, chunks still arrive one at a time, and a parser
that raises changes nothing. The last of those is the one that matters most, so
it is tested by making the parser fail on purpose.

The secret phrase test is the Rule 1 test for responses. The mock upstream puts
a unique phrase in the `content_block_delta` of its streaming response - that
is, in the assistant's own text - and the assertion is that the phrase appears
nowhere in the database bytes, the log lines or the cost report.
"""
from __future__ import annotations

import contextlib
import gzip
import http.client
import importlib.util
import io
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from router.config import RouterConfig, load_config
from router.cost_report import format_cost_report, build_cost_report
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog, UsageRow
from router.proxy import LOOPBACK_HOST, ProxySettings, create_server
from router.state import SessionStore
from router.usage import (
    STATUS_NO_USAGE,
    STATUS_OK,
    STATUS_UNKNOWN,
    UsageParser,
    UsageRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-ACTIVE-test.yaml"

MODEL_LOW = "test-low"
MODEL_MID = "test-mid"

CLIENT_TIMEOUT = 5.0


def mock_module() -> Any:
    """`tools/mock_upstream.py`, loaded as a module so the tests drive the real
    mock rather than a copy of it."""
    spec = importlib.util.spec_from_file_location(
        "mock_upstream_for_usage", REPO_ROOT / "tools" / "mock_upstream.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOCK = mock_module()

#: The phrase the mock puts in a streamed assistant response.
TEST_PHRASE = MOCK.TEST_PHRASE


def parse(status: int = 200, encoding: str = "", content_type: str = "") -> UsageParser:
    return UsageParser(status, content_encoding=encoding, content_type=content_type)


def json_body(**overrides: Any) -> bytes:
    payload: dict[str, Any] = {
        "model": "mock-model",
        "usage": {
            "input_tokens": 120,
            "output_tokens": 45,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 200,
        },
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


def feed_all(parser: UsageParser, body: bytes, size: int | None = None) -> UsageRecord:
    """Feed `body` in `size`-byte pieces (all of it at once when None)."""
    if size is None:
        parser.feed(body)
        return parser.finish()
    for start in range(0, len(body), size):
        parser.feed(body[start : start + size])
    return parser.finish()


# --- the extractor on its own ------------------------------------------------


def test_non_streaming_json_reads_all_four_counts_and_the_model():
    parser = parse(content_type="application/json")

    record = feed_all(parser, json_body())

    assert record.status == STATUS_OK
    assert record.model_reported == "mock-model"
    assert record.input_tokens == 120
    assert record.output_tokens == 45
    assert record.cache_read_tokens == 800
    assert record.cache_write_tokens == 200


def test_non_streaming_json_ignores_everything_except_the_counts():
    """Rule 1 for responses: the assistant text is read past, never kept."""
    parser = parse(content_type="application/json")
    body = json.dumps(
        {
            "model": "mock-model",
            "content": [{"type": "text", "text": TEST_PHRASE}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 6},
            "thinking": TEST_PHRASE,
        }
    ).encode()

    record = feed_all(parser, body)

    assert record.input_tokens == 5
    assert record.output_tokens == 6
    assert record.cache_read_tokens is None
    assert record.cache_write_tokens is None
    assert not hasattr(record, "text")
    assert TEST_PHRASE not in repr(record)


def test_streaming_reads_input_from_start_and_output_from_the_last_delta():
    parser = parse(content_type="text/event-stream")

    record = feed_all(parser, b"".join(MOCK.MESSAGES_STREAM_CHUNKS))

    assert record.status == STATUS_OK
    assert record.model_reported == MOCK.MOCK_MODEL
    assert record.input_tokens == MOCK.MOCK_INPUT_TOKENS
    assert record.output_tokens == MOCK.MOCK_OUTPUT_TOKENS
    assert record.cache_read_tokens == MOCK.MOCK_CACHE_READ_TOKENS
    assert record.cache_write_tokens == MOCK.MOCK_CACHE_WRITE_TOKENS


def test_streaming_reads_an_event_that_is_split_across_two_chunks():
    """One event straddling the chunk boundary must still be read."""
    body = b"".join(MOCK.MESSAGES_STREAM_CHUNKS)
    start = body.index(b"event: message_start")
    end = body.index(b"\n\n", start) + 2

    for size in (1, 3, 7, 13, 64, len(body) - 1):
        parser = parse(content_type="text/event-stream")
        record = feed_all(parser, body, size=size)

        assert record.status == STATUS_OK, f"chunk size {size} lost the stream"
        assert record.input_tokens == MOCK.MOCK_INPUT_TOKENS, f"chunk size {size}"
        assert record.output_tokens == MOCK.MOCK_OUTPUT_TOKENS, f"chunk size {size}"


def test_streaming_ignores_the_content_delta_that_carries_the_text():
    parser = parse(content_type="text/event-stream")

    record = feed_all(parser, b"".join(MOCK.MESSAGES_STREAM_CHUNKS))

    assert TEST_PHRASE not in repr(record)
    assert record.input_tokens == MOCK.MOCK_INPUT_TOKENS


def test_a_gzipped_body_is_decompressed_on_the_parsers_copy():
    parser = parse(encoding="gzip", content_type="application/json")

    record = feed_all(parser, gzip.compress(json_body()))

    assert record.status == STATUS_OK
    assert record.input_tokens == 120
    assert record.output_tokens == 45


def test_a_gzipped_stream_is_decompressed_incrementally():
    body = gzip.compress(b"".join(MOCK.MESSAGES_STREAM_CHUNKS))
    parser = parse(encoding="gzip", content_type="text/event-stream")

    # One byte at a time: a decompressor that buffered the whole thing would
    # still pass, so the point here is only that it works piece by piece.
    record = feed_all(parser, body, size=1)

    assert record.status == STATUS_OK
    assert record.input_tokens == MOCK.MOCK_INPUT_TOKENS
    assert record.output_tokens == MOCK.MOCK_OUTPUT_TOKENS


def test_an_unsupported_encoding_is_unknown_and_never_a_guess():
    parser = parse(encoding="br", content_type="application/json")

    record = feed_all(parser, json_body())

    assert record.status == STATUS_UNKNOWN
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.model_reported is None


def test_a_body_that_is_not_the_json_it_claims_to_be_is_unknown():
    parser = parse(content_type="application/json")

    record = feed_all(parser, b"{not json at all")

    assert record.status == STATUS_UNKNOWN
    assert record.input_tokens is None


def test_a_body_with_no_usage_at_all_is_unknown():
    parser = parse(content_type="application/json")

    record = feed_all(parser, json.dumps({"model": "mock-model"}).encode())

    assert record.status == STATUS_UNKNOWN


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 503])
def test_a_status_of_four_hundred_or_above_is_no_usage(status):
    """Decided from the status alone, before a byte is parsed."""
    parser = parse(status=status, content_type="application/json")

    record = feed_all(parser, json_body())

    assert record.status == STATUS_NO_USAGE


def test_finish_is_idempotent():
    parser = parse(content_type="application/json")
    feed_all(parser, json_body())

    assert parser.finish() == parser.finish()


def test_the_record_has_no_field_that_could_hold_text():
    """The type itself is the guarantee: no field can receive a string of
    content, so there is nowhere for assistant text to go."""
    text_fields = [
        name
        for name in UsageRecord.__dataclass_fields__
        if "text" in name or "content" in name or "message" in name
    ]

    assert text_fields == []


# --- the proxy: the client is unaffected -------------------------------------


@dataclass
class MockState:
    requests: list[bytes] = field(default_factory=list)
    #: When `hold_stream` is set, the mock stops after its first SSE chunk and
    #: waits for `release_remaining`. Only the incremental-delivery test needs
    #: that; every other test wants the whole sequence at once.
    hold_stream: bool = False
    release_remaining: threading.Event = field(default_factory=threading.Event)
    last_chunk_sent: threading.Event = field(default_factory=threading.Event)
    #: The bytes the mock handed the proxy, so a test can compare them.
    sent: list[bytes] = field(default_factory=list)


class UsageUpstream(BaseHTTPRequestHandler):
    """A mock that answers `/v1/messages` with the mock upstream's own shape:
    JSON with usage, or the three-chunk SSE sequence when the request streams.

    `Content-Encoding` and the HTTP status are settable so one harness covers the
    gzip, unsupported-encoding and error-status cases.
    """

    protocol_version = "HTTP/1.1"
    server_version = "usage-upstream"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def state(self) -> MockState:
        return self.server.state  # type: ignore[attr-defined]

    @property
    def route(self) -> str:
        return self.path.split("?", 1)[0]

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        length = int(raw_length) if raw_length else 0
        return self.rfile.read(length) if length else b""

    def _json(self, payload: bytes, encoding: str = "") -> None:
        if encoding == "gzip":
            payload = gzip.compress(payload)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Upstream", "usage-mock")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _status(self, code: int) -> None:
        payload = f"upstream said {code}\n".encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for index, chunk in enumerate(MOCK.MESSAGES_STREAM_CHUNKS):
            self.state.sent.append(chunk)
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
            if index == 0 and self.state.hold_stream:
                self.state.release_remaining.wait(timeout=STREAM_GATE_TIMEOUT)
        self.state.last_chunk_sent.set()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_POST(self) -> None:
        body = self._read_body()
        self.state.requests.append(body)

        upstream_status = self.server.upstream_status  # type: ignore[attr-defined]
        if upstream_status is not None:
            self._status(int(upstream_status))
            return
        if self.route != MESSAGES_PATH:
            self._status(404)
            return

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            payload = {}

        if isinstance(payload, dict) and payload.get("stream") is True:
            self._sse()
            return

        report = json.dumps(
            {
                "model": MOCK.MOCK_MODEL,
                "usage": MOCK._usage_payload(),
            }
        ).encode()
        self._json(report, encoding=self.server.content_encoding)  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.route == MESSAGES_PATH:
            self._status(200)
            return
        self._status(404)


#: Long enough that a proxy which buffers the whole response fails the test
#: rather than hanging the run.
STREAM_GATE_TIMEOUT = 10.0


@dataclass
class Harness:
    proxy_port: int
    upstream_port: int
    state: MockState


@contextlib.contextmanager
def running(
    db_path: Path | None,
    content_encoding: str = "",
    upstream_status: int | None = None,
    config: RouterConfig | None = None,
    hold_stream: bool = False,
) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), UsageUpstream)
    upstream.daemon_threads = True
    state = MockState(hold_stream=hold_stream)
    upstream.state = state  # type: ignore[attr-defined]
    upstream.content_encoding = content_encoding  # type: ignore[attr-defined]
    upstream.upstream_status = upstream_status  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    authority = f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}"
    log = DecisionLog(db_path) if db_path is not None else None
    server = create_server(
        ProxySettings(
            listen_port=0,
            upstream=authority,
            mode="shadow",
            decisions=log,
            config=config,
            state=SessionStore() if config is not None else None,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Harness(
            proxy_port=server.server_address[1],
            upstream_port=upstream.server_address[1],
            state=state,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def log_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DecisionLog:
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    return DecisionLog.from_env()


def messages_body(model: str = MODEL_MID, stream: bool = False) -> bytes:
    return json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode()


def post(port: int, path: str, body: bytes) -> tuple[int, bytes, dict[str, str]]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        headers = {name.lower(): value for name, value in response.getheaders()}
        return response.status, response.read(), headers
    finally:
        connection.close()


def drain(response: http.client.HTTPResponse) -> bytes:
    buffer = bytearray()
    while True:
        piece = response.read1(65536)
        if not piece:
            break
        buffer += piece
    return bytes(buffer)


def wait_for_usage(log: DecisionLog, expected: int, timeout: float = 5.0) -> int:
    """The usage row is written after the response ends, so it can trail it."""
    deadline = time.monotonic() + timeout
    count = log.usage_count()
    while count < expected and time.monotonic() < deadline:
        time.sleep(0.01)
        count = log.usage_count()
    return count


def wait_for_text(stream: io.StringIO, fragment: str, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = stream.getvalue()
        if fragment in text:
            return text
        time.sleep(0.01)
    return stream.getvalue()


def test_a_non_streaming_response_produces_one_usage_row(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path) as harness:
        status, payload, _ = post(harness.proxy_port, MESSAGES_PATH, messages_body())

        assert status == 200
        assert json.loads(payload)["usage"] == MOCK._usage_payload()
        assert wait_for_usage(log, 1) == 1

        row = log.usage_rows(1)[0]
        assert row.status == STATUS_OK
        assert row.model_reported == MOCK.MOCK_MODEL
        assert row.input_tokens == MOCK.MOCK_INPUT_TOKENS
        assert row.output_tokens == MOCK.MOCK_OUTPUT_TOKENS
        assert row.cache_read_tokens == MOCK.MOCK_CACHE_READ_TOKENS
        assert row.cache_write_tokens == MOCK.MOCK_CACHE_WRITE_TOKENS
        assert row.decision_id == log.recent(1)[0].decision_id


def test_a_streaming_response_produces_one_usage_row(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path) as harness:
        status, payload, _ = post(
            harness.proxy_port, MESSAGES_PATH, messages_body(stream=True)
        )

        assert status == 200
        assert payload == b"".join(MOCK.MESSAGES_STREAM_CHUNKS)
        assert wait_for_usage(log, 1) == 1

        row = log.usage_rows(1)[0]
        assert row.status == STATUS_OK
        assert row.model_reported == MOCK.MOCK_MODEL
        assert row.input_tokens == MOCK.MOCK_INPUT_TOKENS
        assert row.output_tokens == MOCK.MOCK_OUTPUT_TOKENS


def test_a_gzipped_response_is_still_parsed_and_the_client_still_gets_gzip(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path, content_encoding="gzip") as harness:
        status, payload, headers = post(harness.proxy_port, MESSAGES_PATH, messages_body())

        assert status == 200
        assert headers["content-encoding"] == "gzip"
        assert payload[:2] == b"\x1f\x8b", "the client was given plain bytes, not the gzip"
        assert wait_for_usage(log, 1) == 1

        row = log.usage_rows(1)[0]
        assert row.status == STATUS_OK
        assert row.input_tokens == MOCK.MOCK_INPUT_TOKENS


def test_an_unsupported_encoding_is_recorded_as_unknown(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path, content_encoding="br") as harness:
        status, payload, headers = post(harness.proxy_port, MESSAGES_PATH, messages_body())

        assert status == 200
        assert headers["content-encoding"] == "br"
        assert json.loads(payload)["model"] == MOCK.MOCK_MODEL, "body relayed unchanged"
        assert wait_for_usage(log, 1) == 1

        assert log.usage_rows(1)[0].status == STATUS_UNKNOWN


def test_an_upstream_error_is_recorded_as_no_usage(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path, upstream_status=500) as harness:
        status, payload, _ = post(harness.proxy_port, MESSAGES_PATH, messages_body())

        assert status == 500
        assert payload == b"upstream said 500\n"
        assert wait_for_usage(log, 1) == 1

        row = log.usage_rows(1)[0]
        assert row.status == STATUS_NO_USAGE
        assert row.cost_usd is None
        assert row.baseline_cost_usd is None


def test_a_request_that_is_not_messages_records_no_usage_at_all(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path) as harness:
        post(harness.proxy_port, "/v1/complete", messages_body())
        post(harness.proxy_port, MESSAGES_PATH, messages_body())

        assert wait_for_usage(log, 1) == 1
        time.sleep(0.2)
        assert log.usage_count() == 1, "only the /v1/messages response is priced"


def test_the_relayed_bytes_headers_and_framing_are_untouched(tmp_path, monkeypatch):
    """What the upstream sent is what the client gets, byte for byte."""
    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path) as harness:
        status, payload, headers = post(harness.proxy_port, MESSAGES_PATH, messages_body())

        assert status == 200
        assert headers["x-upstream"] == "usage-mock"
        assert payload == json.dumps(
            {"model": MOCK.MOCK_MODEL, "usage": MOCK._usage_payload()}
        ).encode()
        assert wait_for_usage(log, 1) == 1


def test_streaming_chunks_still_arrive_one_at_a_time(tmp_path, monkeypatch):
    """The parser must not wait for the whole response, and neither must the
    client: the first chunk is delivered before the mock sends the last."""
    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path, hold_stream=True) as harness:
        connection = http.client.HTTPConnection(
            LOOPBACK_HOST, harness.proxy_port, timeout=CLIENT_TIMEOUT
        )
        try:
            connection.request(
                "POST",
                MESSAGES_PATH,
                body=messages_body(stream=True),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            assert response.status == 200

            first = response.read1(65536)
            assert first == MOCK.MESSAGES_STREAM_CHUNKS[0]
            assert not harness.state.last_chunk_sent.is_set(), "proxy buffered the response"
            assert log.usage_count() == 0, "the usage row was written before the response ended"

            harness.state.release_remaining.set()
            received = bytearray(first)
            while True:
                piece = response.read1(65536)
                if not piece:
                    break
                received += piece
            assert bytes(received) == b"".join(MOCK.MESSAGES_STREAM_CHUNKS)
            assert wait_for_usage(log, 1) == 1
        finally:
            connection.close()


def test_a_parser_that_raises_does_not_break_forwarding(tmp_path, monkeypatch):
    """Rule 3 at the level of the usage parser: a bug here can cost a row and
    nothing else."""
    log = log_at(tmp_path, monkeypatch)
    expected = json.dumps(
        {"model": MOCK.MOCK_MODEL, "usage": MOCK._usage_payload()}
    ).encode()

    class Exploding(UsageParser):
        def feed(self, chunk: bytes) -> None:
            raise RuntimeError("parser is broken")

        def finish(self) -> UsageRecord:
            raise RuntimeError("parser is broken")

    monkeypatch.setattr("router.proxy.UsageParser", Exploding)

    with running(log.db_path) as harness:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            status, payload, headers = post(harness.proxy_port, MESSAGES_PATH, messages_body())
            wait_for_text(captured, "POST /v1/messages 200")

        assert status == 200
        assert payload == expected, "the client did not get the upstream's bytes"
        assert headers["x-upstream"] == "usage-mock"
        assert wait_for_usage(log, 1) == 1
        assert log.usage_rows(1)[0].status == STATUS_UNKNOWN
        assert "parser is broken" not in captured.getvalue(), "a traceback reached the log"


def test_a_usage_row_that_cannot_be_written_does_not_break_the_request(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_bytes(b"this is a file, not a directory")
    monkeypatch.setenv(DB_ENV_VAR, str(blocker / "router.sqlite3"))
    log = DecisionLog.from_env()

    with running(log.db_path) as harness:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            status, payload, _ = post(harness.proxy_port, MESSAGES_PATH, messages_body())
            wait_for_text(captured, "POST /v1/messages 200")

        assert status == 200
        assert json.loads(payload)["usage"] == MOCK._usage_payload()
        assert "decision log write failed" in captured.getvalue()

        # And the proxy is still serving.
        status, _, _ = post(harness.proxy_port, MESSAGES_PATH, messages_body())
        assert status == 200


def test_the_mock_upstream_answers_a_streaming_request_with_three_chunks():
    """The mock's own shape, so the harness above and a manual run agree."""
    chunks = MOCK.MESSAGES_STREAM_CHUNKS

    assert len(chunks) == 3
    assert b"event: message_start" in chunks[0]
    assert b"event: content_block_delta" in chunks[1]
    assert b"event: message_delta" in chunks[2]
    assert TEST_PHRASE.encode() in chunks[1]


# --- Rule 1: response content is never stored --------------------------------


def test_the_phrase_in_the_streamed_assistant_text_is_stored_nowhere(
    tmp_path, monkeypatch
):
    """The one assertion that matters for responses, on all three sinks."""
    log = log_at(tmp_path, monkeypatch)
    config = load_config(ACTIVE_CONFIG_PATH)

    with running(log.db_path, config=config) as harness:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            status, payload, _ = post(
                harness.proxy_port, MESSAGES_PATH, messages_body(stream=True)
            )
            assert status == 200
            assert TEST_PHRASE.encode() in payload, "the phrase never reached the client"
            assert wait_for_usage(log, 1) == 1
        logs = wait_for_text(captured, "POST /v1/messages 200")

        report = format_cost_report(build_cost_report(log.db_path))

    database_bytes = log.db_path.read_bytes()
    assert database_bytes, "the database file was not written"
    assert TEST_PHRASE.encode() not in database_bytes
    assert TEST_PHRASE not in logs
    assert TEST_PHRASE not in report

    # And the row itself is nothing but numbers plus a model id.
    row = log.usage_rows(1)[0]
    assert (row.input_tokens, row.output_tokens) == (
        MOCK.MOCK_INPUT_TOKENS,
        MOCK.MOCK_OUTPUT_TOKENS,
    )
    assert row.notes is None or "MODEL_MISMATCH" in row.notes


def test_the_usage_table_has_no_content_columns(tmp_path, monkeypatch):
    from router.decisions import forbidden_columns

    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path) as harness:
        post(harness.proxy_port, MESSAGES_PATH, messages_body())
        assert wait_for_usage(log, 1) == 1

    columns = log.usage_column_names()
    assert forbidden_columns(columns) == []
    assert columns == [
        "usage_id",
        "decision_id",
        "model_reported",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "status",
        "cost_usd",
        "baseline_cost_usd",
        "notes",
    ]


def test_the_usage_table_is_added_without_touching_the_decisions_table(tmp_path, monkeypatch):
    """A database written before `router_usage` existed gains the table and
    keeps every row it had."""
    log = log_at(tmp_path, monkeypatch)

    with running(log.db_path) as harness:
        post(harness.proxy_port, MESSAGES_PATH, messages_body())
        assert wait_for_usage(log, 1) == 1
        decisions_before = log.recent(10)

    # A second log object re-applies both schemas to the existing file.
    reopened = DecisionLog(log.db_path)
    assert reopened.count() == len(decisions_before)
    assert reopened.recent(10) == decisions_before
    assert reopened.usage_count() == 1


def test_an_unknown_figure_is_stored_as_null_and_never_as_zero(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_bytes(b"this is a file, not a directory")
    monkeypatch.setenv(DB_ENV_VAR, str(blocker / "router.sqlite3"))

    path = tmp_path / "manual.sqlite3"
    log = DecisionLog(path)
    log.record_usage(UsageRow(status=STATUS_UNKNOWN))
    row = log.usage_rows(1)[0]

    assert row.status == STATUS_UNKNOWN
    assert row.input_tokens is None
    assert row.cost_usd is None
    assert row.baseline_cost_usd is None