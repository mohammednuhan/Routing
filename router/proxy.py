"""Transparent loopback proxy for the Tamias router.

Standard library only: `http.server.ThreadingHTTPServer` for the listener,
`http.client` for the upstream connection. No other dependency is added.

The proxy is transparent. It forwards the method, path, query, headers and body
to the upstream named in the config and returns the upstream status, headers and
body unchanged. It makes no routing decision and changes no model. The bytes it
forwards are never altered, not even to read metadata out of them: a request
body is parsed read-only, after the original bytes have already been sent.

For `POST /v1/messages` it appends one row to the decision log: requested and
chosen model, effort, mode, reason codes and an error class. Metadata only - see
`router/decisions.py` and Rule 1.

Streaming responses are relayed chunk by chunk as they arrive and flushed after
each chunk; the whole response is never buffered.

It never logs request headers or bodies, and never logs credential values. The
only per-request log line is:

    <method> <path> <status> <duration>ms

Rules 1-5 of `router/AGENTS.md` apply. Rule 3 needs one honest caveat: when the
upstream cannot be reached there is no upstream response to pass through, so
the proxy returns a minimal JSON error of its own (502 or 504) instead of
inventing an answer, and never drops the connection without a response.
"""
from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .decisions import MESSAGES_PATH, DecisionLog, DecisionRecord, read_request_metadata
from .signals import compute_signals

#: The only host this proxy will bind. Rule 4.
LOOPBACK_HOST = "127.0.0.1"

#: Seconds allowed for the upstream connection and for each read from it.
DEFAULT_UPSTREAM_TIMEOUT = 60.0

#: Largest request body the proxy will buffer in order to forward it.
MAX_REQUEST_BODY = 64 * 1024 * 1024

_READ_CHUNK = 65536
_MAX_LINE = 65536

#: Headers that describe a single hop and must not be forwarded (RFC 9110).
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

#: Request headers the proxy sets itself rather than copying. `host` is replaced
#: with the upstream authority, `content-length` is recomputed from the body it
#: actually forwards, and `expect` is satisfied locally before forwarding.
REQUEST_HEADERS_DROPPED = HOP_BY_HOP_HEADERS | {"host", "content-length", "expect"}

#: Methods whose empty body still needs an explicit `Content-Length: 0`.
_METHODS_EXPECTING_BODY = frozenset({"POST", "PUT", "PATCH"})

#: Statuses that carry no body and therefore no framing headers.
_BODYLESS_STATUSES = frozenset({204, 304})

_DEFAULT_PORTS = {"http": 80, "https": 443}


class ProxyError(Exception):
    """The proxy cannot be configured or started."""


@dataclass(frozen=True)
class UpstreamTarget:
    """A validated upstream origin, with any base path already stripped."""

    scheme: str
    host: str
    port: int | None
    path_prefix: str

    @property
    def is_tls(self) -> bool:
        return self.scheme == "https"

    @property
    def authority(self) -> str:
        """The value to send as the upstream `Host` header."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        if self.port is None or self.port == _DEFAULT_PORTS[self.scheme]:
            return host
        return f"{host}:{self.port}"

    @property
    def safe_label(self) -> str:
        """A loggable origin. Never includes any userinfo from the URL."""
        return f"{self.scheme}://{self.authority}"

    def request_target(self, request_path: str) -> str:
        """The upstream request target for a received path (query included)."""
        if not request_path.startswith("/"):
            request_path = "/" + request_path
        return self.path_prefix + request_path

    def connect(self, timeout: float) -> http.client.HTTPConnection:
        if self.is_tls:
            return http.client.HTTPSConnection(self.host, self.port, timeout=timeout)
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)


def parse_upstream(url: str) -> UpstreamTarget:
    """Validate an upstream URL. Raises `ProxyError` with a clear message."""
    text = url.strip()
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in _DEFAULT_PORTS:
        raise ProxyError(f"upstream scheme must be http or https, got {parts.scheme or text!r}")
    if not parts.hostname:
        raise ProxyError(f"upstream must include a host, got {text!r}")
    try:
        port = parts.port
    except ValueError:
        raise ProxyError(f"upstream port is not a number in {parts.netloc!r}") from None
    if parts.query or parts.fragment:
        raise ProxyError("upstream URL must not carry a query string or fragment")
    return UpstreamTarget(
        scheme=parts.scheme,
        host=parts.hostname,
        port=port,
        path_prefix=parts.path.rstrip("/"),
    )


@dataclass(frozen=True)
class ProxySettings:
    """Everything the proxy needs. No routing decision is held here."""

    listen_host: str = LOOPBACK_HOST
    listen_port: int = 8787
    upstream: str = ""
    timeout: float = DEFAULT_UPSTREAM_TIMEOUT
    mode: str = "shadow"
    decisions: DecisionLog | None = None


class _RequestError(Exception):
    """A malformed request; answered locally without touching the upstream."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class ProxyHandler(BaseHTTPRequestHandler):
    """Forwards every request to the upstream and relays the response back."""

    protocol_version = "HTTP/1.1"
    server_version = "tamias-router"
    sys_version = ""
    disable_nagle_algorithm = True

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default request/error logging (Rule 5)."""

    @property
    def settings(self) -> ProxySettings:
        return self.server.settings  # type: ignore[attr-defined]

    @property
    def target(self) -> UpstreamTarget:
        return self.server.target  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> Any:
        """Route any verb to the proxy, not just the ones listed below."""
        if name.startswith("do_") and name[3:].isupper():
            return self._proxy
        raise AttributeError(name)

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_PATCH(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    def do_HEAD(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self._proxy()

    def _proxy(self) -> None:
        started = time.monotonic()
        status = 502
        sent_head = False
        body = b""
        error_class: str | None = None
        connection: http.client.HTTPConnection | None = None
        try:
            try:
                body = self._read_request_body()
            except _RequestError as exc:
                status = exc.status
                error_class = "malformed_request"
                self._reply_json(exc.status, exc.code, exc.message)
                sent_head = True
                self.close_connection = True
                return

            headers = self._forwarded_headers(body)
            connection = self.target.connect(self.settings.timeout)
            connection.putrequest(
                self.command,
                self.target.request_target(self.path),
                skip_host=True,
                skip_accept_encoding=True,
            )
            for name, value in headers:
                connection.putheader(name, value)
            connection.endheaders(body if body else None)

            response = connection.getresponse()
            status = response.status
            self._relay(response)
            sent_head = True
        except TimeoutError:
            status = 504
            error_class = "upstream_timeout"
            if not sent_head:
                self._reply_json(504, "upstream_timeout", "the upstream did not respond in time")
        except (OSError, http.client.HTTPException):
            status = 502
            error_class = "upstream_unreachable"
            if not sent_head:
                self._reply_json(502, "upstream_unreachable", "the upstream could not be reached")
            else:
                self.close_connection = True
        finally:
            if connection is not None:
                connection.close()
            self._record_decision(body, status, error_class)
            self._log_request(status, started)

    def _record_decision(self, body: bytes, status: int, error_class: str | None) -> None:
        """Append one metadata-only row for a POST to /v1/messages.

        Every other method and path is forwarded and not recorded. A failure
        here is reported as one short line and never affects the request that
        was already forwarded.
        """
        log = self.settings.decisions
        if log is None or self.command != "POST":
            return
        if self.path.split("?", 1)[0] != MESSAGES_PATH:
            return

        try:
            metadata = read_request_metadata(body)
            signal_values: dict[str, Any] = {}
            sig_error: str | None = None
            try:
                parsed = json.loads(body.decode("utf-8"))
                sigs = compute_signals(parsed)
                signal_values = sigs.to_dict()
            except Exception:
                signal_values = {}
                sig_error = "signals_failed"

            record = DecisionRecord(
                session_hint=log.session_hint(metadata.first_user_text),
                requested_model=metadata.model,
                chosen_model=metadata.model,
                chosen_effort=None,
                mode=self.settings.mode,
                reason_codes=["PASSTHROUGH"],
                signal_values=signal_values,
                action="STAY",
                applied=0,
                error=metadata.error or sig_error or error_class,
            )
            log.record(record)
        except Exception as exc:
            print(
                f"tamias-router: decision log write failed ({type(exc).__name__})",
                file=sys.stderr,
                flush=True,
            )

    def _read_request_body(self) -> bytes:
        encoding = self.headers.get("Transfer-Encoding", "")
        if "chunked" in encoding.lower():
            return self._read_chunked_body()

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError:
            raise _RequestError(
                400, "bad_request", f"Content-Length is not an integer: {raw_length!r}"
            ) from None
        if length < 0:
            raise _RequestError(400, "bad_request", f"Content-Length is negative: {length}")
        if length > MAX_REQUEST_BODY:
            raise _RequestError(
                413,
                "payload_too_large",
                f"request body exceeds {MAX_REQUEST_BODY} bytes",
            )
        return self._read_exactly(length)

    def _read_exactly(self, count: int) -> bytes:
        buffer = bytearray()
        while len(buffer) < count:
            piece = self.rfile.read(min(count - len(buffer), _READ_CHUNK))
            if not piece:
                raise _RequestError(400, "bad_request", "request body ended early")
            buffer += piece
        return bytes(buffer)

    def _read_chunked_body(self) -> bytes:
        body = bytearray()
        while True:
            line = self.rfile.readline(_MAX_LINE)
            if not line:
                raise _RequestError(400, "bad_request", "chunked request body ended early")
            try:
                size = int(line.split(b";", 1)[0].strip().decode("ascii"), 16)
            except (ValueError, UnicodeDecodeError):
                raise _RequestError(400, "bad_request", "malformed chunk size") from None
            if size < 0:
                raise _RequestError(400, "bad_request", "negative chunk size")
            if size == 0:
                while True:
                    trailer = self.rfile.readline(_MAX_LINE)
                    if trailer in (b"\r\n", b"\n", b""):
                        break
                break
            if len(body) + size > MAX_REQUEST_BODY:
                raise _RequestError(
                    413, "payload_too_large", f"request body exceeds {MAX_REQUEST_BODY} bytes"
                )
            body += self._read_exactly(size)
            self.rfile.readline(_MAX_LINE)
        return bytes(body)

    def _forwarded_headers(self, body: bytes) -> list[tuple[str, str]]:
        """Copy the client's headers, minus hop-by-hop and credential routing."""
        connection_tokens = {
            token.strip().lower()
            for value in self.headers.get_all("Connection") or []
            for token in value.split(",")
            if token.strip()
        }
        headers: list[tuple[str, str]] = []
        for name, value in self.headers.items():
            key = name.lower()
            if key in REQUEST_HEADERS_DROPPED or key.startswith("proxy-"):
                continue
            if key in connection_tokens:
                continue
            headers.append((name, value))

        headers.append(("Host", self.target.authority))
        if body:
            headers.append(("Content-Length", str(len(body))))
        elif self.command in _METHODS_EXPECTING_BODY:
            headers.append(("Content-Length", "0"))
        return headers

    def _relay(self, response: http.client.HTTPResponse) -> None:
        """Send the upstream status and headers, then stream the body through."""
        self.send_response(response.status, response.reason)
        for name, value in response.getheaders():
            key = name.lower()
            if key in HOP_BY_HOP_HEADERS or key.startswith("proxy-"):
                continue
            self.send_header(name, value)

        bodyless = (
            self.command == "HEAD"
            or response.status in _BODYLESS_STATUSES
            or 100 <= response.status < 200
        )
        if bodyless:
            framing = "none"
        elif response.chunked:
            framing = "chunked"
            self.send_header("Transfer-Encoding", "chunked")
        elif response.getheader("Content-Length") is not None:
            framing = "length"
        else:
            framing = "close"
            self.send_header("Connection", "close")
            self.close_connection = True

        self.end_headers()
        if framing == "none":
            return

        try:
            while True:
                piece = response.read1(_READ_CHUNK)
                if not piece:
                    break
                if framing == "chunked":
                    self.wfile.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
                else:
                    self.wfile.write(piece)
                self.wfile.flush()
            if framing == "chunked":
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _reply_json(self, status: int, code: str, message: str) -> None:
        """A short JSON error. Carries no header, body or credential material."""
        payload = json.dumps({"error": code, "message": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _log_request(self, status: int, started: float) -> None:
        """The one permitted log line: method, path, status, duration in ms.

        The query string is dropped so that nothing a client put in it can be
        written to the log. No header or body value is ever logged.
        """
        duration_ms = (time.monotonic() - started) * 1000.0
        path = self.path.split("?", 1)[0]
        print(
            f"{self.command} {path} {status} {duration_ms:.0f}ms",
            file=sys.stderr,
            flush=True,
        )


class ProxyServer(ThreadingHTTPServer):
    """Loopback-only threaded listener carrying its own settings."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        settings: ProxySettings,
        target: UpstreamTarget,
    ) -> None:
        self.settings = settings
        self.target = target
        super().__init__(server_address, handler_class)

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Report the exception type only: a traceback could carry header text."""
        exc_type = type(sys.exc_info()[1]).__name__
        print(
            f"tamias-router: request handler error ({exc_type})",
            file=sys.stderr,
            flush=True,
        )


def create_server(settings: ProxySettings) -> ProxyServer:
    """Build the listener. Refuses any host except 127.0.0.1 (Rule 4)."""
    if settings.listen_host != LOOPBACK_HOST:
        raise ProxyError(
            f"refusing to listen on {settings.listen_host!r}: "
            f"the router listens on {LOOPBACK_HOST} only"
        )
    target = parse_upstream(settings.upstream)
    return ProxyServer((settings.listen_host, settings.listen_port), ProxyHandler, settings, target)


def serve(settings: ProxySettings) -> None:
    """Run the proxy until interrupted. Ctrl+C stops it cleanly."""
    server = create_server(settings)
    host, port = server.server_address[:2]
    print(f"listening on http://{host}:{port}/ -> {server.target.safe_label}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("tamias-router: stopping", flush=True)
    finally:
        server.server_close()
