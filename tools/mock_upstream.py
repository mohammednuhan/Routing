"""A tiny local upstream for manually exercising the tamias-router proxy.

Standard library only. Routes:

    POST /echo         -> {"ok": true, "received_bytes": N}
    POST /sink         -> "ok", after recording the request
    POST /v1/messages  -> {"ok", "received_bytes", "content_length", "model",
                            "effort", "stream", "messages"}
    GET  /static       -> a fixed non-streaming body
    GET  /stream       -> 3 SSE chunks, one per second, chunked
    GET  /status/<n>   -> HTTP <n> with a small body
    anything else      -> HTTP 404

Run it, then point the proxy at it:

    python tools/mock_upstream.py 8931
    tamias-router start --upstream http://127.0.0.1:8931

This tool is for manual testing only. It is not part of the router package and
prints request lines to stderr so a human can see what arrived; the proxy
itself never logs headers or bodies.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STREAM_CHUNKS = [
    b"data: {\"index\": 0}\n\n",
    b"data: {\"index\": 1}\n\n",
    b"data: {\"index\": 2}\n\n",
]
CHUNK_INTERVAL_SECONDS = 1.0


def _effort_of(payload: dict[str, object]) -> object:
    """The effort a request asked for, as the API expresses it.

    `effort` may be a plain field or the budget of a `thinking` block. Anything
    unexpected yields None rather than a guess.
    """
    direct = payload.get("effort")
    if direct is not None:
        return direct
    thinking = payload.get("thinking")
    if isinstance(thinking, dict):
        return thinking.get("budget_tokens")
    return None


class MockUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mock-upstream"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"mock-upstream: {self.command} {self.path}\n")
        sys.stderr.flush()

    @property
    def route(self) -> str:
        return self.path.split("?", 1)[0]

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError:
            return b""
        return self.rfile.read(length) if length > 0 else b""

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.route == "/static":
            self._send(200, b"hello from the mock upstream\n", "text/plain")
            return
        if self.route == "/stream":
            self._stream()
            return
        if self.route.startswith("/status/"):
            tail = self.route[len("/status/") :]
            try:
                status = int(tail)
            except ValueError:
                status = 500
            self._send(status, f"mock upstream returned {status}\n".encode(), "text/plain")
            return
        self._send(404, b"mock upstream: no such route\n", "text/plain")

    def do_POST(self) -> None:
        body = self._read_body()
        if self.route == "/echo":
            payload = json.dumps({"ok": True, "received_bytes": len(body)}).encode()
            self._send(200, payload, "application/json")
            return
        if self.route == "/sink":
            self._send(200, b"ok\n", "text/plain")
            return
        if self.route == "/v1/messages":
            self._messages(body)
            return
        self._send(404, b"mock upstream: no such route\n", "text/plain")

    def _messages(self, body: bytes) -> None:
        """Report what a `/v1/messages` request carried, as JSON.

        Echoes only metadata plus the fields a router would act on (`model`,
        `effort`, `stream`) and their received length. The message list is not
        returned: an upstream that echoed prompts would make this tool a place
        where request content is stored.
        """
        try:
            payload = json.loads(body.decode("utf-8"))
        except ValueError:
            payload = None

        report: dict[str, object] = {
            "ok": True,
            "received_bytes": len(body),
            "content_length": self.headers.get("Content-Length"),
        }
        if isinstance(payload, dict):
            report["model"] = payload.get("model")
            report["effort"] = _effort_of(payload)
            report["stream"] = payload.get("stream")
            report["messages"] = len(payload.get("messages") or [])
        else:
            report["json"] = False

        self._send(200, json.dumps(report).encode(), "application/json")

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in STREAM_CHUNKS:
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
            time.sleep(CHUNK_INTERVAL_SECONDS)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mock_upstream",
        description="Local upstream for manual tamias-router proxy testing",
    )
    parser.add_argument("port", nargs="?", type=int, default=8931)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), MockUpstreamHandler)
    server.daemon_threads = True
    host, port = server.server_address[:2]
    print(f"mock upstream on http://{host}:{port}/ (ctrl+c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("mock upstream: stopping", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
