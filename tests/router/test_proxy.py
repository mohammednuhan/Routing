from __future__ import annotations

import contextlib
import http.client
import io
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import pytest

from router.proxy import LOOPBACK_HOST, ProxyError, ProxySettings, create_server

STATIC_BODY = b"static body bytes\n"
STREAM_CHUNKS = [b'data: {"index": 0}\n\n', b'data: {"index": 1}\n\n', b'data: {"index": 2}\n\n']
STREAM_BODY = b"".join(STREAM_CHUNKS)

#: How long the mock waits for the test to release the remaining chunks, so a
#: proxy that buffers the whole response fails the test instead of hanging it.
STREAM_GATE_TIMEOUT = 10.0

#: Client-side timeout. A proxy that buffers the whole streaming response trips
#: this instead of hanging the test run.
CLIENT_TIMEOUT = 5.0

FAKE_AUTHORIZATION = "Bearer sk-ant-fake-token-DO-NOT-LOG-111"
FAKE_API_KEY = "sk-ant-api03-fake-key-DO-NOT-LOG-222"
FAKE_QUERY_MARKER = "query-secret-DO-NOT-LOG-333"


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class MockState:
    requests: list[RecordedRequest] = field(default_factory=list)
    release_remaining: threading.Event = field(default_factory=threading.Event)
    last_chunk_sent: threading.Event = field(default_factory=threading.Event)


class MockUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mock-upstream"
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
            RecordedRequest(
                method=self.command,
                path=self.path,
                headers={name.lower(): value for name, value in self.headers.items()},
                body=body,
            )
        )

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError:
            return b""
        return self.rfile.read(length) if length > 0 else b""

    def _send(self, status: int, payload: bytes, content_type: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def do_GET(self) -> None:
        self._record(b"")
        if self.route == "/static":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Upstream", "mock")
            self.send_header("Content-Length", str(len(STATIC_BODY)))
            self.end_headers()
            self.wfile.write(STATIC_BODY)
            self.wfile.flush()
            return
        if self.route == "/stream":
            self._stream()
            return
        if self.route.startswith("/status/"):
            status = int(self.route.rsplit("/", 1)[1])
            self._send(status, f"upstream said {status}\n".encode())
            return
        self._send(404, b"upstream said 404\n")

    def do_POST(self) -> None:
        body = self._read_body()
        self._record(body)
        if self.route == "/echo":
            payload = json.dumps({"ok": True, "received_bytes": len(body)}).encode()
            self._send(200, payload, "application/json")
            return
        if self.route == "/sink":
            self._send(200, b"sunk\n")
            return
        self._send(404, b"upstream said 404\n")

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for index, chunk in enumerate(STREAM_CHUNKS):
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
            if index == 0:
                self.state.release_remaining.wait(timeout=STREAM_GATE_TIMEOUT)
        self.state.last_chunk_sent.set()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


@dataclass
class RunningServer:
    port: int
    thread: threading.Thread
    server: Any


@dataclass
class UnderTest:
    proxy: RunningServer
    mock: RunningServer
    state: MockState

    @property
    def upstream_authority(self) -> str:
        return f"{LOOPBACK_HOST}:{self.mock.port}"


@contextlib.contextmanager
def running_mock_upstream() -> Iterator[tuple[RunningServer, MockState]]:
    server = ThreadingHTTPServer((LOOPBACK_HOST, 0), MockUpstreamHandler)
    server.daemon_threads = True
    state = MockState()
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield RunningServer(server.server_address[1], thread, server), state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def running_proxy(upstream: str) -> Iterator[RunningServer]:
    server = create_server(ProxySettings(listen_port=0, upstream=upstream))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield RunningServer(server.server_address[1], thread, server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def proxy_for_mock() -> Iterator[UnderTest]:
    with running_mock_upstream() as (mock, state):
        upstream = f"http://{LOOPBACK_HOST}:{mock.port}"
        with running_proxy(upstream) as proxy:
            yield UnderTest(proxy=proxy, mock=mock, state=state)


@contextlib.contextmanager
def client(port: int) -> Iterator[http.client.HTTPConnection]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        yield connection
    finally:
        connection.close()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((LOOPBACK_HOST, 0))
        return probe.getsockname()[1]


def drain(response: http.client.HTTPResponse) -> bytes:
    buffer = bytearray()
    while True:
        piece = response.read1(65536)
        if not piece:
            break
        buffer += piece
    return bytes(buffer)


def wait_for_log(log: io.StringIO, fragment: str, timeout: float = 5.0) -> str:
    """The proxy logs after the response is flushed, so the write can trail the
    client by a moment. Wait for the line rather than racing it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log.getvalue()
        if fragment in text:
            return text
        time.sleep(0.01)
    return log.getvalue()


def test_non_streaming_response_is_byte_identical():
    with proxy_for_mock() as under:
        with client(under.proxy.port) as connection:
            connection.request("GET", "/static?beta=true")
            response = connection.getresponse()

            assert response.status == 200
            assert response.getheader("X-Upstream") == "mock"
            assert drain(response) == STATIC_BODY


@pytest.mark.parametrize("status", [404, 500])
def test_upstream_error_status_is_passed_through(status):
    with proxy_for_mock() as under:
        with client(under.proxy.port) as connection:
            connection.request("GET", f"/status/{status}")
            response = connection.getresponse()

            assert response.status == status
            assert drain(response) == f"upstream said {status}\n".encode()


def test_streaming_response_arrives_incrementally_and_is_byte_identical():
    with proxy_for_mock() as under:
        with client(under.proxy.port) as connection:
            connection.request("GET", "/stream")
            response = connection.getresponse()

            assert response.status == 200
            first = response.read1(65536)

            assert first == STREAM_CHUNKS[0]
            assert not under.state.last_chunk_sent.is_set(), "proxy buffered the whole response"

            under.state.release_remaining.set()
            received = bytearray(first)
            while True:
                piece = response.read1(65536)
                if not piece:
                    break
                received += piece

            assert bytes(received) == STREAM_BODY


def test_unreachable_upstream_returns_502_and_server_keeps_running():
    with running_proxy(f"http://{LOOPBACK_HOST}:{free_port()}") as proxy:
        for _ in range(2):
            with client(proxy.port) as connection:
                connection.request("GET", "/static")
                response = connection.getresponse()

                assert response.status == 502
                payload = json.loads(drain(response))
                assert payload["error"] == "upstream_unreachable"
                assert payload["message"]

        assert proxy.thread.is_alive()


def test_headers_are_forwarded_but_credential_values_are_never_logged():
    with proxy_for_mock() as under:
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            with client(under.proxy.port) as connection:
                connection.putrequest(
                    "POST", f"/sink?marker={FAKE_QUERY_MARKER}", skip_host=True
                )
                connection.putheader("Authorization", FAKE_AUTHORIZATION)
                connection.putheader("x-api-key", FAKE_API_KEY)
                connection.putheader("Content-Type", "application/json")
                connection.putheader("Content-Length", "2")
                connection.endheaders(b"{}")
                response = connection.getresponse()
                assert response.status == 200
                assert drain(response) == b"sunk\n"
            logs = wait_for_log(log, "POST /sink 200")

        recorded = under.state.requests[-1]
        assert recorded.method == "POST"
        assert recorded.path == f"/sink?marker={FAKE_QUERY_MARKER}"
        assert recorded.headers["authorization"] == FAKE_AUTHORIZATION
        assert recorded.headers["x-api-key"] == FAKE_API_KEY
        assert recorded.headers["host"] == under.upstream_authority

        assert logs.strip(), "the proxy logged nothing for a request"
        assert "ms" in logs
        assert FAKE_AUTHORIZATION not in logs
        assert FAKE_API_KEY not in logs
        assert FAKE_QUERY_MARKER not in logs


def test_request_body_reaches_upstream_unchanged():
    payload = bytes(range(256)) * 4
    with proxy_for_mock() as under:
        with client(under.proxy.port) as connection:
            connection.request(
                "POST",
                "/echo",
                body=payload,
                headers={"Content-Type": "application/octet-stream"},
            )
            response = connection.getresponse()

            assert response.status == 200
            echoed = json.loads(drain(response))

        assert under.state.requests[-1].body == payload
        assert echoed == {"ok": True, "received_bytes": len(payload)}


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "192.168.1.10", "::", "example.internal"])
def test_non_loopback_listen_host_is_refused(host):
    settings = ProxySettings(listen_host=host, listen_port=0, upstream="http://127.0.0.1:1")

    with pytest.raises(ProxyError, match="refusing to listen"):
        create_server(settings)
