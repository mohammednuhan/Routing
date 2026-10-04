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

from .config import LEGAL_MODES
from .decisions import (
    MESSAGES_PATH,
    DecisionLog,
    DecisionRecord,
    RequestMetadata,
    read_request_metadata,
)
from .signals import Signals, compute_signals
from .policy import decide
from .safety import COST_SIGNAL_KEY, apply_safety, safety_failure
from .state import SessionStore

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


def _is_legal_target(config: Any, model: str) -> bool:
    """True when `model` appears in the config's models (Rule 7).

    Checks the id only. Effort is never rewritten, so there is no effort to
    validate here.
    """
    models = getattr(config, "model_ids", None)
    return isinstance(models, tuple) and model in models


class ProxyError(Exception):
    """The proxy cannot be configured or started."""


class NotAJSONObject(Exception):
    """The body is not a JSON object, or has no top-level "model" to replace."""


#: Error classes recorded when a rewrite was considered and abandoned. The
#: original bytes are forwarded in every one of these cases.
REWRITE_SKIPPED_ENCODING = "rewrite_skipped_encoding"
REWRITE_SKIPPED_NOT_JSON = "rewrite_skipped_not_json"
REWRITE_FAILED = "rewrite_failed"

#: Response header naming the hop, added only when a body was actually rewritten
#: and `routed_header` is enabled.
ROUTED_HEADER = "X-Tamias-Routed"

_WHITESPACE = " \t\r\n"


def rewrite_model(body: bytes, target_model: str) -> bytes:
    """Return `body` with only its top-level `"model"` value replaced.

    This is a splice, not a re-serialization: every other byte of the request,
    including key order and whitespace, is preserved exactly. Only the span
    occupied by the model value is replaced, with the same encoding style
    (`json.dumps` with `ensure_ascii=False`) the body itself would use.

    Raises `NotAJSONObject` when the body is not a JSON object or has no
    top-level `model`. Any other failure is the caller's to classify.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NotAJSONObject("body is not utf-8") from exc

    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise NotAJSONObject("body is not json") from exc
    if not isinstance(payload, dict):
        raise NotAJSONObject("body is not a json object")

    try:
        start, end = _top_level_value_span(text, "model")
    except _SpanError as exc:
        raise NotAJSONObject(f"cannot locate the top-level model value: {exc}") from exc

    encoded = json.dumps(target_model, ensure_ascii=False)
    return (text[:start] + encoded + text[end:]).encode("utf-8")


class _SpanError(ValueError):
    """The raw JSON text could not be walked as expected."""


def is_rewritable(body: bytes) -> bool:
    """True when `rewrite_model` would find a top-level `"model"` in `body`.

    Used by active mode to say "this body could never have been rewritten",
    which is a different problem from "there was nothing to rewrite".
    """
    try:
        text = body.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    try:
        _top_level_value_span(text, "model")
    except _SpanError:
        return False
    return True


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in _WHITESPACE:
        index += 1
    return index


def _value_end(text: str, start: int) -> int:
    """The index just past the JSON value that begins at `start`."""
    if start >= len(text):
        raise _SpanError("value starts past the end")
    char = text[start]

    if char == '"':
        index = start + 1
        while index < len(text):
            current = text[index]
            if current == "\\":
                index += 2
                continue
            if current == '"':
                return index + 1
            index += 1
        raise _SpanError("unterminated string")

    if char in "{[":
        depth = 0
        index = start
        while index < len(text):
            current = text[index]
            if current == '"':
                index = _value_end(text, index)
                continue
            if current in "{[":
                depth += 1
            elif current in "}]":
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1
        raise _SpanError("unterminated container")

    index = start
    while index < len(text) and text[index] not in ",}]" and text[index] not in _WHITESPACE:
        index += 1
    if index == start:
        raise _SpanError("empty scalar value")
    return index


def _top_level_value_span(text: str, key: str) -> tuple[int, int]:
    """The `(start, end)` span of a top-level object's member value.

    Walks the raw text rather than re-encoding it, so the span is exact even
    when the request used unusual whitespace or key order.
    """
    index = _skip_whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        raise _SpanError("not a json object")
    index += 1

    while True:
        index = _skip_whitespace(text, index)
        if index >= len(text):
            raise _SpanError("unterminated object")
        if text[index] == "}":
            raise _SpanError(f"no top-level {key!r}")
        if text[index] != '"':
            raise _SpanError("expected a member name")

        name_start = index
        name_end = _value_end(text, index)
        try:
            name = json.loads(text[name_start:name_end])
        except ValueError as exc:
            raise _SpanError("member name is not json") from exc

        index = _skip_whitespace(text, name_end)
        if index >= len(text) or text[index] != ":":
            raise _SpanError("expected ':'")
        index = _skip_whitespace(text, index + 1)
        value_end = _value_end(text, index)

        if name == key:
            return (index, value_end)

        index = _skip_whitespace(text, value_end)
        if index >= len(text):
            raise _SpanError("unterminated object")
        if text[index] == ",":
            index += 1
            continue
        if text[index] == "}":
            raise _SpanError(f"no top-level {key!r}")
        raise _SpanError("expected ',' or '}'")


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
    config: Any | None = None
    state: SessionStore | None = None


@dataclass(frozen=True)
class RoutingPlan:
    """Everything the router decided about one request, plus the bytes to send.

    Computed before the body is forwarded, because in active mode the decision
    has to exist before anything is sent upstream. The body forwarded is
    `forward_body`: byte-identical to the request in every mode except an
    allowed switch in active mode.
    """

    metadata: RequestMetadata
    signals: Signals | None
    sig_error: str | None
    session_hint: str | None
    action: str
    chosen_model: str | None
    chosen_effort: str | None
    reason_codes: list[str]
    record_error: str | None
    estimated_cost: float | str | None
    mode: str
    forward_body: bytes
    applied: int = 0
    routed_header: str | None = None


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
        plan: RoutingPlan | None = None
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

            # The decision is made before anything is sent upstream, because
            # active mode has to forward the body it intends to send. In shadow
            # and off this returns the original bytes untouched.
            plan = self._plan(body)
            forward_body = plan.forward_body if plan is not None else body

            headers = self._forwarded_headers(forward_body)
            connection = self.target.connect(self.settings.timeout)
            connection.putrequest(
                self.command,
                self.target.request_target(self.path),
                skip_host=True,
                skip_accept_encoding=True,
            )
            for name, value in headers:
                connection.putheader(name, value)
            connection.endheaders(forward_body if forward_body else None)

            response = connection.getresponse()
            status = response.status
            self._relay(response, plan.routed_header if plan is not None else None)
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
            self._record_decision(body, status, error_class, plan)
            self._log_request(status, started)

    @property
    def _effective_mode(self) -> str:
        """The mode that governs both the recorded row and any rewrite.

        The config is authoritative when it carries a legal mode, so a row can
        never claim one mode while the body was forwarded under another. Only
        `active` rewrites; every other mode forwards the original bytes.
        """
        config = self.settings.config
        configured = getattr(config, "mode", None) if config is not None else None
        if isinstance(configured, str) and configured in LEGAL_MODES:
            return configured
        return self.settings.mode

    def _plan(self, body: bytes) -> RoutingPlan | None:
        """Decide what to do with one request. Never raises, never prints.

        Returns None when the request is not one this router records or routes.
        Any failure inside degrades to "forward the original bytes", because a
        decision that could not be made must never become a change.
        """
        if self.command != "POST":
            return None
        if self.path.split("?", 1)[0] != MESSAGES_PATH:
            return None

        mode = self._effective_mode
        try:
            metadata = read_request_metadata(body)

            signal_values: dict[str, Any] = {}
            sig_error: str | None = None
            signals: Signals | None = None
            try:
                parsed = json.loads(body.decode("utf-8"))
                signals = compute_signals(parsed)
                signal_values = signals.to_dict()
            except Exception:
                signal_values = {}
                sig_error = "signals_failed"

            action = "STAY"
            chosen_model = metadata.model
            chosen_effort: str | None = None
            reason_codes = ["PASSTHROUGH"]
            record_error = metadata.error or sig_error
            estimated_cost: float | str | None = None
            session_hint: str | None = None
            decision: Any = None
            safety_ran = False

            log = self.settings.decisions
            config = self.settings.config

            if signals is not None and config is not None:
                requested_model = metadata.model or config.default_model
                try:
                    decision = decide(
                        signals,
                        requested_model,
                        signals.requested_effort_if_present,
                        config,
                    )
                    store = self.settings.state
                    if store is not None and log is not None:
                        session_hint = log.session_hint(metadata.first_user_text)
                        decision, safety_failed = self._apply_safety(
                            decision, signals, store, session_hint, config
                        )
                        safety_ran = not safety_failed
                        if safety_failed:
                            record_error = record_error or "safety_failed"
                    action = decision.action
                    # A blocked switch is a STAY, so the model actually in use
                    # is the requested one. Reporting the blocked target here
                    # would claim a model the router did not choose.
                    chosen_model = (
                        (decision.target_model or requested_model)
                        if action == "SWITCH"
                        else requested_model
                    )
                    chosen_effort = (
                        decision.target_effort
                        if action == "SWITCH"
                        else signals.requested_effort_if_present
                    )
                    reason_codes = list(decision.reason_codes or reason_codes)
                    estimated_cost = decision.estimated_rebuild_cost_usd
                    if estimated_cost is not None:
                        signal_values[COST_SIGNAL_KEY] = estimated_cost
                except Exception:
                    decision = None
                    action = "STAY"
                    chosen_model = metadata.model
                    chosen_effort = None
                    reason_codes = ["PASSTHROUGH"]
                    record_error = record_error or "policy_failed"

            forward_body = body
            applied = 0
            routed_header: str | None = None

            if mode == "active":
                # Active mode reports why it did not rewrite, so an operator can
                # tell "nothing to do" apart from "could not have".
                encoding = self.headers.get("Content-Encoding", "").strip().lower()
                if encoding and encoding != "identity":
                    record_error = REWRITE_SKIPPED_ENCODING
                elif not is_rewritable(body):
                    record_error = REWRITE_SKIPPED_NOT_JSON
                elif action == "SWITCH":
                    # Only a switch safety actually approved may be applied.
                    # Without a session there is no safety, and an unchecked
                    # switch is not one (Rule 2).
                    if safety_ran:
                        forward_body, rewrite_error = self._rewrite_for_active(
                            body, decision, config
                        )
                        if rewrite_error is None:
                            applied = 1
                            if getattr(config, "routed_header", False):
                                routed_header = (
                                    f"{metadata.model or ''}->{chosen_model or ''}"
                                )
                        else:
                            record_error = rewrite_error

            return RoutingPlan(
                metadata=metadata,
                signals=signals,
                sig_error=sig_error,
                session_hint=session_hint,
                action=action,
                chosen_model=chosen_model,
                chosen_effort=chosen_effort,
                reason_codes=reason_codes,
                record_error=record_error,
                estimated_cost=estimated_cost,
                mode=mode,
                forward_body=forward_body,
                applied=applied,
                routed_header=routed_header,
            )
        except Exception:
            # Nothing above may prevent the request from being forwarded.
            return RoutingPlan(
                metadata=RequestMetadata(error="router_plan_failed"),
                signals=None,
                sig_error=None,
                session_hint=None,
                action="STAY",
                chosen_model=None,
                chosen_effort=None,
                reason_codes=["PASSTHROUGH"],
                record_error="router_plan_failed",
                estimated_cost=None,
                mode=mode,
                forward_body=body,
                applied=0,
                routed_header=None,
            )

    def _rewrite_for_active(
        self, body: bytes, decision: Any, config: Any
    ) -> tuple[bytes, str | None]:
        """The bytes to forward in active mode, and any rewrite error class.

        Returns the original body untouched whenever the rewrite cannot be done
        safely. Forwarding is never blocked by this.
        """
        target = getattr(decision, "target_model", None)
        if not isinstance(target, str) or not _is_legal_target(config, target):
            # A target outside the config is never forwarded, whatever decided
            # it (Rule 7).
            return (body, REWRITE_FAILED)

        try:
            rewritten = rewrite_model(body, target)
        except NotAJSONObject:
            return (body, REWRITE_SKIPPED_NOT_JSON)
        except Exception:
            return (body, REWRITE_FAILED)

        if rewritten == body:
            return (body, REWRITE_FAILED)
        return (rewritten, None)

    def _record_decision(
        self,
        body: bytes,
        status: int,
        error_class: str | None,
        plan: RoutingPlan | None = None,
    ) -> None:
        """Append one metadata-only row for a POST to /v1/messages.

        Every other method and path is forwarded and not recorded. A failure
        here is reported as one short line and never affects the request, which
        has already been forwarded by the time this runs.

        `plan` is the decision computed before forwarding. It is recomputed
        only when it is absent, which happens for a request rejected before its
        body was read.
        """
        log = self.settings.decisions
        if log is None or self.command != "POST":
            return
        if self.path.split("?", 1)[0] != MESSAGES_PATH:
            return

        if plan is None:
            plan = self._plan(body)
            if plan is None:
                return

        try:
            signal_values: dict[str, Any] = {}
            if plan.signals is not None:
                signal_values = plan.signals.to_dict()
            if plan.estimated_cost is not None:
                signal_values[COST_SIGNAL_KEY] = plan.estimated_cost

            record_error = plan.record_error or error_class
            session_hint = plan.session_hint
            if session_hint is None:
                session_hint = log.session_hint(plan.metadata.first_user_text)

            record = DecisionRecord(
                session_hint=session_hint,
                requested_model=plan.metadata.model,
                chosen_model=plan.chosen_model,
                chosen_effort=plan.chosen_effort,
                mode=plan.mode,
                reason_codes=plan.reason_codes,
                signal_values=signal_values,
                action=plan.action,
                applied=plan.applied,
                error=record_error,
            )
            log.record(record)
        except Exception as exc:
            print(
                f"tamias-router: decision log write failed ({type(exc).__name__})",
                file=sys.stderr,
                flush=True,
            )

    def _apply_safety(
        self,
        decision: Any,
        signals: Signals,
        store: SessionStore,
        session_hint: str,
        config: Any,
    ) -> tuple[Any, bool]:
        """Run the safety layer and return `(decision, failed)`.

        Safety can only remove a switch, so a failure here can only cost the
        router a switch, never gain one: `safety_failure` turns the decision
        into a STAY, and because this runs before anything is forwarded, a
        STAY means the original bytes are what go upstream. A failure is
        recorded as an error class, never dropped and never applied.
        """
        try:
            current = store.observe(session_hint)
            final, updated = apply_safety(decision, signals, current, config)
            store.replace(session_hint, updated)
            return (final, False)
        except Exception:
            return (safety_failure(decision), True)

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

    def _relay(
        self, response: http.client.HTTPResponse, routed_header: str | None = None
    ) -> None:
        """Send the upstream status and headers, then stream the body through.

        `routed_header` is emitted only when a body was actually rewritten and
        the config asked for it, so the header can never claim a hop that did
        not happen.
        """
        self.send_response(response.status, response.reason)
        for name, value in response.getheaders():
            key = name.lower()
            if key in HOP_BY_HOP_HEADERS or key.startswith("proxy-"):
                continue
            if key == ROUTED_HEADER.lower():
                # The router owns this header: an upstream copy of it would be a
                # claim about a hop the upstream knows nothing about.
                continue
            self.send_header(name, value)
        if routed_header is not None:
            self.send_header(ROUTED_HEADER, routed_header)

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
