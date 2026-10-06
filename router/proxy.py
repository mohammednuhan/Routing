"""Transparent loopback proxy for the Tamias router.

Standard library only: `http.server.ThreadingHTTPServer` for the listener,
`http.client` for the upstream connection. No other dependency is added.

The proxy is transparent. It forwards the method, path, query, headers and body
to the upstream named in the config and returns the upstream status, headers and
body unchanged. It makes no routing decision and changes no model. The bytes it
forwards are never altered, not even to read metadata out of them: a request
body is parsed read-only, after the original bytes have already been sent.

The upstream may be `http` or `https` and may name a path prefix. A request for
`/v1/chat/completions` reaches `https://openrouter.ai/api` as
`/api/v1/chat/completions?` followed by the query the client sent. https is
connected with `http.client.HTTPSConnection` and
`ssl.create_default_context()`, so certificate and hostname verification are
always on and nothing in this package can turn either off. Redirects are not
followed: a 3xx is relayed to the client as the upstream sent it, headers and
body alike, because deciding to follow one would be the router choosing to send
the request somewhere the client did not ask for. `parse_upstream` is what
refuses a plain http upstream that is not on loopback; see `router/config.py`.

For a `POST` to a request path it recognises, it appends one row to the decision
log: requested and chosen model, effort, mode, reason codes and an error class.
Metadata only - see `router/decisions.py` and Rule 1. Two request formats are
recognised, decided by the path and nothing else: a path ending in
`/v1/messages` is Anthropic Messages format, and one ending in
`/chat/completions` is OpenAI chat-completions format. Both are read, decided
on and logged by the same pipeline; `signal_values` carries the format under
`api_format` so a row says which one it was. Any other path is forwarded and
recorded nowhere, exactly as before.

Each recognised response also produces one `router_usage` row, written after the
response has ended: the token counts the upstream reported and what they cost at
the config's prices. The extractor is picked by the same `api_format`, so an
OpenAI response is read as an OpenAI response, and both paths write the same
columns with the same meanings - see `router/usage.py` and `router/cost.py`. The
parser is fed a copy of each chunk after those exact bytes are on the wire, so it
can cost a row and nothing else.

Streaming responses are relayed chunk by chunk as they arrive and flushed after
each chunk; the whole response is never buffered.

One request header the proxy can replace, and only when the policy says so:
`policy.upstream_identity_encoding` turns the client's `Accept-Encoding` into
`identity` on the forwarded upstream request, so the upstream answers in plain
bytes and the counts in that answer can be read. It is off by default, and it is
a request header only: nothing on a response is added, removed or decoded on the
way out, and every other request header and the body are forwarded exactly as
they arrived. See `upstream_identity_encoding` below.

One exception to the streaming rule, and it is not an exception to transparency:
when the policy enables it, a request the upstream answered with a retryable
status is sent again before anything is relayed, with only the top-level `model`
value replaced by the next configured model in the same cost tier. The upstream
status is known before the first byte reaches the client, so the decision to
retry is taken while the client has been sent nothing: a response that has
started streaming is never retried. See `REASON_FALLBACK_NEXT_IN_TIER` below.

It never logs request headers or bodies, and never logs credential values. The
only per-request log line is:

    <method> <path> <status> <duration>ms

Rules 1-5 of `router/AGENTS.md` apply. Rule 3 needs one honest caveat: when the
upstream cannot be reached there is no upstream response to pass through, so
the proxy returns a minimal JSON error of its own (502 or 504) instead of
inventing an answer, and never drops the connection without a response.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import ssl
import sys
import time
import urllib.parse
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Protocol

from .breaker import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_THRESHOLD,
    REASON_CIRCUIT_OPEN,
    CircuitBreaker,
    is_internal_error,
)
from .config import (
    DEFAULT_FALLBACK_MAX_ATTEMPTS,
    DEFAULT_FALLBACK_STATUSES,
    LEGAL_MODES,
    ConfigError,
    parse_upstream as _parse_upstream_url,
)
from .decisions import (
    MESSAGES_PATH,
    DecisionLog,
    DecisionRecord,
    RequestMetadata,
    UsageRow,
    default_db_path,
    read_request_metadata,
)
from .killswitch import REASON_KILL_SWITCH, read_kill_switch
from .hold import HELD_SIGNAL_KEY, apply_hold
from .classifier import classify_prompt
from .cost import estimate_cost
from .usage import OpenAIUsageParser, UsageParser, UsageRecord
from .signals import Signals, compute_signals
from .signals_openai import compute_signals_openai
from .policy import decide
from .safety import COST_SIGNAL_KEY, apply_safety, safety_failure
from .state import SessionStore

#: The only host this proxy will bind. Rule 4.
LOOPBACK_HOST = "127.0.0.1"

#: The request formats the router knows how to read. Anything else is forwarded
#: and recorded nowhere, so an unknown route can never acquire a decision.
API_FORMAT_ANTHROPIC = "anthropic"
API_FORMAT_OPENAI = "openai"

#: Path suffixes that name a format. Matched as a suffix on the path with the
#: query string removed, so a gateway that prefixes its routes
#: (`/anthropic/v1/messages`, `/openai/v1/chat/completions`) still names the
#: format it is serving. The Anthropic suffix is the one
#: `router/decisions.py` has always logged.
ANTHROPIC_PATH_SUFFIX = MESSAGES_PATH
OPENAI_PATH_SUFFIX = "/chat/completions"

#: Key the format is recorded under in `signal_values`. A constant the tests and
#: the reports read it by, so the name is written down once.
API_FORMAT_SIGNAL_KEY = "api_format"

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

#: The one request header the policy can replace, and the one value it is
#: replaced with. `identity` is the encoding that means "do not compress", so an
#: upstream that honours it replies in plain bytes.
ACCEPT_ENCODING = "Accept-Encoding"
IDENTITY_ENCODING = "identity"

#: Statuses that carry no body and therefore no framing headers.
_BODYLESS_STATUSES = frozenset({204, 304})

_DEFAULT_PORTS = {"http": 80, "https": 443}

#: The upstream connection classes, bound here as module names so that
#: `open_upstream` reads as the shipped path it is, and so a test can replace
#: one of them without patching the `http.client` module every other test in
#: the suite shares.
HTTP_CONNECTION = http.client.HTTPConnection
HTTPS_CONNECTION = http.client.HTTPSConnection


class ConnectionFactory(Protocol):
    """`(target, timeout) -> connection`: how the proxy reaches the upstream.

    Production passes nothing and `open_upstream` is used. A test passes a fake,
    which is what lets the whole forwarding path be exercised without opening a
    socket.
    """

    def __call__(
        self, target: UpstreamTarget, timeout: float
    ) -> http.client.HTTPConnection: ...


def detect_api_format(path: str) -> str | None:
    """The format `path` names, or None when it names no format at all.

    The query string is dropped first: `?beta=true` says nothing about which API
    a route belongs to, and a query the client controls must never be able to
    change what the router believes it is looking at.

    The only caller that reads a request body picks its signal reader from this,
    so an unrecognised path is the only way to guarantee no decision is made.
    """
    route = path.split("?", 1)[0]
    if route.endswith(ANTHROPIC_PATH_SUFFIX):
        return API_FORMAT_ANTHROPIC
    if route.endswith(OPENAI_PATH_SUFFIX):
        return API_FORMAT_OPENAI
    return None


def usage_parser_class(api_format: str | None) -> type[UsageParser] | None:
    """The extractor that reads `api_format`'s response, or None for no format.

    The two formats carry their counts in differently named fields and neither can
    be read as the other, so the request's `api_format` - the path, and nothing
    else - is what selects one. Both extractors return the same `UsageRecord` and
    are written into the same `router_usage` row, so no report downstream of this
    can tell which format answered.

    Looked up when the response arrives rather than bound at import, so the class
    in force is the one this module holds at the moment it is asked to read a
    body.
    """
    if api_format == API_FORMAT_ANTHROPIC:
        return UsageParser
    if api_format == API_FORMAT_OPENAI:
        return OpenAIUsageParser
    return None


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

#: Error class recorded when classification was attempted and failed. Both
#: classifier signals stay None, the policy layer runs exactly as it would with
#: no classifier, and the original bytes are forwarded. Rule 3.
CLASSIFIER_FAILED = "classifier_failed"


def with_classifier(body_json: Any, signals: Signals | None, config: Any) -> Signals | None:
    """`signals` with the classifier's score and tier filled in.

    Read-only on the parsed body: nothing here writes to it, and the classifier
    keeps no copy of the prompt it read. `signals` comes back unchanged when the
    section is disabled or the conversation holds no human prompt, which are
    ordinary states and not failures.
    """
    if signals is None or config is None:
        return signals
    task = classify_prompt(body_json, config)
    if task is None:
        return signals
    return replace(signals, classifier_score=task.score, classifier_tier=task.tier)


def without_classifier(signals: Signals | None) -> Signals | None:
    """`signals` with both classifier values cleared, after a failure."""
    if signals is None:
        return None
    return replace(signals, classifier_score=None, classifier_tier=None)


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


#: Reason code added to a row whose request was retried onto another model. The
#: row is still one row for one client request; the code is what says the hop
#: was a fallback and not a switch the policy proposed.
REASON_FALLBACK_NEXT_IN_TIER = "FALLBACK_NEXT_IN_TIER"

#: Key the model the first attempt used is recorded under in `signal_values`.
FALLBACK_FROM_SIGNAL_KEY = "fallback_from"

#: Key the number of retries is recorded under in `signal_values`. A count, so a
#: row can show how far a request got without carrying anything from it.
FALLBACK_ATTEMPTS_SIGNAL_KEY = "fallback_attempts"


@dataclass(frozen=True)
class FallbackSettings:
    """What the policy says about retrying onto the next model in a tier.

    The shipped defaults never enable a retry and are what a config without a
    policy, or with a policy that does not carry the keys, resolves to.
    """

    enabled: bool = False
    statuses: tuple[int, ...] = DEFAULT_FALLBACK_STATUSES
    max_attempts: int = DEFAULT_FALLBACK_MAX_ATTEMPTS


def fallback_settings(config: Any) -> FallbackSettings:
    """The fallback settings `config` carries, defaults everywhere else.

    Read defensively rather than trusted: `ProxySettings.config` is whatever the
    caller passed, and a value that is not what the loader would have produced
    falls back to the shipped default rather than being used. Only
    `enabled is True` ever enables a retry, so a missing key, a policy that is
    absent and a policy whose flag is not the boolean `true` all mean off.
    """
    policy = getattr(config, "policy", None)
    statuses = getattr(policy, "fallback_statuses", None)
    if not isinstance(statuses, (list, tuple)) or not all(
        isinstance(status, int)
        and not isinstance(status, bool)
        and 100 <= status <= 599
        for status in statuses
    ):
        statuses = DEFAULT_FALLBACK_STATUSES
    attempts = getattr(policy, "fallback_max_attempts", None)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        attempts = DEFAULT_FALLBACK_MAX_ATTEMPTS
    return FallbackSettings(
        enabled=getattr(policy, "fallback_enabled", False) is True,
        statuses=tuple(statuses),
        max_attempts=attempts,
    )


def upstream_identity_encoding(config: Any) -> bool:
    """Whether the policy asks for `Accept-Encoding: identity` on the upstream hop.

    Read the way `fallback_settings` reads its own: the policy section, and only
    `is True` ever turns the feature on. A config with no policy, a config whose
    policy predates the key and a policy whose value is not the boolean `true`
    all mean off, so a fresh install and an old config both forward the client's
    own header exactly as it arrived.

    The flag is not a routing decision and is not gated on the mode: it is read
    for every request the proxy forwards, and it cannot change a model, an effort
    or a body. Rule 2.
    """
    policy = getattr(config, "policy", None)
    return getattr(policy, "upstream_identity_encoding", False) is True


def forwarded_model(body: bytes) -> str | None:
    """The top-level `model` value in the bytes about to be sent, or None.

    Read from the bytes rather than from the plan, because a plan's
    `chosen_model` can name a switch whose rewrite was refused - the model in
    the body is then still the one that really went upstream, and a retry has to
    start from that model. Only the model value is read; the rest of the body is
    parsed and dropped. Rule 1.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return model if isinstance(model, str) else None


def next_in_cost_tier(config: Any, model: str, tried: Iterable[str]) -> str | None:
    """The next model to try after `model`, within `model`'s own cost tier.

    Config order is the order, starting at the model after `model` and wrapping
    around at the end, skipping every model in `tried`. Only models that share
    `model`'s `cost_tier` are eligible: a fallback moves within a price band, so
    it can never quietly change what a request costs.

    None when `model` is not a model the config lists, when its tier holds no
    other model, or when every model in the tier has already been tried. Rule 7:
    nothing outside the config is ever returned.
    """
    specs = getattr(config, "models", None)
    if not specs:
        return None
    pairs = [(spec.id, spec.cost_tier) for spec in specs]
    ids = [model_id for model_id, _ in pairs]
    if model not in ids:
        return None
    tier = pairs[ids.index(model)][1]
    skip = set(tried)
    for offset in range(1, len(ids)):
        candidate = ids[(ids.index(model) + offset) % len(ids)]
        if candidate in skip:
            continue
        if pairs[ids.index(candidate)][1] == tier:
            return candidate
    return None


def _is_legal_target(config: Any, model: str) -> bool:
    """True when `model` is a model id the config lists.

    The test for Rule 7. A model that is not in the config has no tier, so no
    fallback may be planned from it, whatever the request asked for.
    """
    specs = getattr(config, "models", None) or ()
    return any(spec.id == model for spec in specs)


def _retry_body(body: bytes, target: str) -> bytes | None:
    """`body` with only its model value replaced, or None when it cannot be.

    The same raw-text splice active mode uses, so a retry is the request the
    client sent on a different model: every other byte, and the framing header
    computed from them, is identical. None means the body cannot carry a
    different model, in which case nothing is retried and the upstream's own
    response is relayed unchanged (Rule 3).
    """
    try:
        rewritten = rewrite_model(body, target)
    except Exception:
        return None
    return None if rewritten == body else rewritten


def _discard(response: http.client.HTTPResponse) -> None:
    """Close a response that will not be relayed, without reading its body.

    The body of a status the router is about to replace is never sent to the
    client, never parsed for usage and never kept. Closing it is the whole of
    what happens to it, and a close that fails changes nothing, because the
    response is being dropped either way.
    """
    with contextlib.suppress(Exception):
        response.close()


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
        """The upstream request target for a received path (query included).

        The prefix the upstream URL named, then the path and query that arrived,
        unchanged: `https://openrouter.ai/api` sends a request for
        `/v1/chat/completions?stream=true` upstream as
        `/api/v1/chat/completions?stream=true`. The query is the client's and is
        forwarded as it came, so a prefix can never replace or drop it.
        """
        if not request_path.startswith("/"):
            request_path = "/" + request_path
        return self.path_prefix + request_path

    def connect(
        self, timeout: float, factory: ConnectionFactory | None = None
    ) -> http.client.HTTPConnection:
        """The connection one request will be sent over.

        `factory` is how a test gets a fake in here so no socket is ever
        opened; with nothing passed the upstream is reached the shipped way,
        through `open_upstream`.
        """
        opener = factory if factory is not None else open_upstream
        return opener(self, timeout)


def open_upstream(
    target: UpstreamTarget, timeout: float
) -> http.client.HTTPConnection:
    """The connection to the upstream. Nothing is sent until a request is put on it.

    An https upstream gets an `HTTPSConnection` carrying
    `ssl.create_default_context()`, so certificate verification *and* hostname
    verification are both on and stay on: there is no argument, flag, config
    key or environment variable anywhere in this package that turns either off,
    and a self-signed or wrongly named certificate fails the handshake. The port
    is 443 unless the URL named another.

    An http upstream gets a plain connection, which `parse_upstream` only
    permits for a loopback host, so no credential is ever put on a wire in clear
    text across a network.

    `http.client` opens the socket when the first request is put on the
    connection, so building one here touches nothing.
    """
    if target.is_tls:
        return HTTPS_CONNECTION(
            target.host,
            target.port if target.port is not None else _DEFAULT_PORTS["https"],
            timeout=timeout,
            context=ssl.create_default_context(),
        )
    return HTTP_CONNECTION(target.host, target.port, timeout=timeout)


def parse_upstream(url: str) -> UpstreamTarget:
    """Validate an upstream URL. Raises `ProxyError` with a clear message.

    The rules themselves live in `router/config.py`, which the config load and
    this function share, so an upstream the config refuses is the same one
    `--upstream` refuses. Only the exception type differs: a config error
    belongs to `ConfigError`, a proxy error to this.
    """
    try:
        parsed = _parse_upstream_url(url)
    except ConfigError as exc:
        raise ProxyError(str(exc)) from None
    return UpstreamTarget(
        scheme=parsed.scheme,
        host=parsed.host,
        port=parsed.port,
        path_prefix=parsed.path_prefix,
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
    breaker: CircuitBreaker | None = None
    #: How to reach the upstream: `(target, timeout) -> connection`. `None` is
    #: the shipped `open_upstream`. A test passes a fake so that no connection,
    #: and no socket, is ever opened.
    connection_factory: ConnectionFactory | None = None


@dataclass(frozen=True)
class FallbackHop:
    """One request's fallback: where it came from, where it ended up, how far.

    `from_model` is the first model tried, `to_model` the model whose response
    the client got, and `attempts` the number of retries that took. All three
    are metadata; nothing here came out of a request or a response body. Rule 1.
    """

    from_model: str
    to_model: str
    attempts: int


@dataclass(frozen=True)
class RoutingPlan:
    """Everything the router decided about one request, plus the bytes to send.

    Computed before the body is forwarded, because in active mode the decision
    has to exist before anything is sent upstream. The body forwarded is
    `forward_body`: byte-identical to the request in every mode except an
    allowed switch in active mode.

    `fallback` is set afterwards, once the upstream has answered, and is None on
    every request that was not retried.
    """

    metadata: RequestMetadata
    api_format: str
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
    held_requests: int | None = None
    fallback: FallbackHop | None = None


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

    #: The `router_usage` record for the response being relayed, set once the
    #: response has ended. `None` means "not parsed, or nothing to record".
    _usage_record: UsageRecord | None = None

    #: True once the status line and headers have been written to the client. A
    #: failure after that point can only end the connection: the response that
    #: started is the client's, and a second one cannot be put in front of it.
    _head_sent = False

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
        body = b""
        error_class: str | None = None
        plan: RoutingPlan | None = None
        connection: http.client.HTTPConnection | None = None
        self._usage_record = None
        self._head_sent = False
        try:
            try:
                body = self._read_request_body()
            except _RequestError as exc:
                status = exc.status
                error_class = "malformed_request"
                self._reply_json(exc.status, exc.code, exc.message)
                self.close_connection = True
                return

            # The decision is made before anything is sent upstream, because
            # active mode has to forward the body it intends to send. In shadow
            # and off this returns the original bytes untouched.
            plan = self._plan(body)
            forward_body = plan.forward_body if plan is not None else body

            # The upstream status is known here, before a single byte has been
            # relayed, so this is the last point at which a retryable status can
            # be retried: once `_relay` has written to the client, that response
            # is the client's and is never retried.
            connection, response, hop = self._forward_upstream(forward_body, plan)
            if hop is not None and plan is not None:
                plan = self._with_fallback(plan, hop)
            status = response.status
            self._relay(response, plan.routed_header if plan is not None else None)
        except TimeoutError:
            status = 504
            error_class = "upstream_timeout"
            if not self._head_sent:
                self._reply_json(504, "upstream_timeout", "the upstream did not respond in time")
        except ssl.SSLError:
            # A TLS failure says things about the peer that are not the
            # client's to be told: which certificate, which hostname, which
            # issuer. The body carries the code alone, the metadata carries the
            # class, and nothing about the certificate or the request is
            # written anywhere (Rule 5). Checked before `OSError`, because
            # `ssl.SSLError` is an `OSError`.
            status = 502
            error_class = "upstream_tls_error"
            if not self._head_sent:
                self._reply_json(502, "upstream_tls_error")
            else:
                self.close_connection = True
        except (OSError, http.client.HTTPException):
            # A refused connection and a name that does not resolve arrive here,
            # and both are the client getting a 502 rather than a dropped
            # connection. The message is fixed and says neither host nor port.
            status = 502
            error_class = "upstream_unreachable"
            if not self._head_sent:
                self._reply_json(502, "upstream_unreachable", "the upstream could not be reached")
            else:
                self.close_connection = True
        finally:
            if connection is not None:
                connection.close()
            decision_id = self._record_decision(body, status, error_class, plan)
            self._record_usage(decision_id, plan)
            self._log_request(status, started)

    def _forward_upstream(
        self, forward_body: bytes, plan: RoutingPlan | None
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse, FallbackHop | None]:
        """Send the request upstream, retrying it onto the next model in its tier.

        Returns the live connection - the caller's to close once the response has
        been relayed - the response to relay, and the hop the request made, if it
        made one.

        A retry sends the bytes that were already read, with only the top-level
        `model` value replaced, so every attempt is the request the client sent,
        on the next configured model sharing the tier of the one just tried. The
        framing headers are recomputed from the bytes each attempt sends, so a
        longer or shorter model value cannot leave a wrong `Content-Length`
        behind.

        Each attempt that is retried is closed unread and never relayed, and the
        whole loop runs before `_relay` is called: nothing reaches the client
        until the final response is known. A response that has started streaming
        is therefore never retried.
        """
        settings = fallback_settings(self.settings.config)
        model = forwarded_model(forward_body)
        budget = settings.max_attempts if self._may_fallback(plan, settings, model) else 0
        if budget < 1 or model is None:
            connection = self._send_upstream(forward_body)
            return (connection, connection.getresponse(), None)

        current_body = forward_body
        current_model = model
        tried: list[str] = []
        retries = 0
        while True:
            tried.append(current_model)
            connection = self._send_upstream(current_body)
            response = connection.getresponse()
            hop = None if retries == 0 else FallbackHop(model, current_model, retries)
            target = self._retry_target(response, settings, current_model, tried, retries, budget)
            if target is None:
                return (connection, response, hop)
            retry_body = _retry_body(current_body, target)
            if retry_body is None:
                # The bytes cannot carry a different model, so the next model in
                # the tier cannot be tried. The upstream's own response is
                # relayed unchanged. Rule 3.
                return (connection, response, hop)
            _discard(response)
            connection.close()
            current_model = target
            current_body = retry_body
            retries += 1

    def _send_upstream(self, body: bytes) -> http.client.HTTPConnection:
        """Open a connection and send one request. Returns it open.

        The headers are built from `body`, so `Content-Length` always describes
        the bytes this attempt actually sends. A failure while sending closes the
        connection before it propagates, so a request that never reached the
        upstream cannot leave a socket behind.
        """
        headers = self._forwarded_headers(body)
        connection = self.target.connect(
            self.settings.timeout, self.settings.connection_factory
        )
        try:
            connection.putrequest(
                self.command,
                self.target.request_target(self.path),
                skip_host=True,
                skip_accept_encoding=True,
            )
            for name, value in headers:
                connection.putheader(name, value)
            connection.endheaders(body if body else None)
        except Exception:
            connection.close()
            raise
        return connection

    def _may_fallback(
        self, plan: RoutingPlan | None, settings: FallbackSettings, model: str | None
    ) -> bool:
        """Whether this request may be retried onto the next model in its tier.

        Every condition is checked here, per request, so none of them can be
        missed by a code path that forgets to look:

        * `plan` is not None, which is a POST on a path naming one of the two
          formats the router reads - anything else is forwarded and recorded
          nowhere, and is never retried either;
        * the plan's mode is `active`, and only `active` may change a request.
          Shadow and off are excluded, and so is a request the kill switch or an
          open circuit is passing through: both of those produce a plan whose
          mode is `off`, so neither switch is read a second time here;
        * the router evaluated the request without an internal error. A request
          the router failed on is passed through unchanged, retried or not
          (Rule 3);
        * the policy switched the feature on;
        * the model that really went upstream is a model the config lists, so the
          tier it belongs to is one the config knows (Rule 7).
        """
        if plan is None or plan.mode != "active" or plan.record_error is not None:
            return False
        if not settings.enabled:
            return False
        return model is not None and _is_legal_target(self.settings.config, model)

    def _retry_target(
        self,
        response: http.client.HTTPResponse,
        settings: FallbackSettings,
        model: str | None,
        tried: list[str],
        retries: int,
        budget: int,
    ) -> str | None:
        """The model to retry on, or None to relay this response unchanged.

        Three things stop a retry: a status that is not retryable, a budget
        already spent, and a tier with no model left that has not been tried. A
        status the client must see - a 400, a 401, a 404 - is never retried,
        because it is the upstream's answer to this request and not a sign that
        the model is busy.
        """
        if response.status not in settings.statuses:
            return None
        if retries >= budget or model is None:
            return None
        return next_in_cost_tier(self.settings.config, model, tried)

    def _with_fallback(self, plan: RoutingPlan, hop: FallbackHop) -> RoutingPlan:
        """The row for a request that was retried onto another model.

        `chosen_model` names the model whose response the client got, because a
        row naming the model that failed would claim a model produced nothing.
        The reason code says the hop was a fallback, so it can never be read as
        a switch the policy proposed, and `applied` is 1 because the body really
        did go to a model the client did not name.

        The routed header, when the config asks for one, is rebuilt from the same
        two ends: after a fallback the hop ends at the model that answered, not
        at the switch target, and a header that named the target would be a claim
        about a hop that did not happen.
        """
        reason_codes = list(plan.reason_codes)
        if REASON_FALLBACK_NEXT_IN_TIER not in reason_codes:
            reason_codes.append(REASON_FALLBACK_NEXT_IN_TIER)
        routed_header = None
        if getattr(self.settings.config, "routed_header", False):
            routed_header = f"{plan.metadata.model or ''}->{hop.to_model}"
        return replace(
            plan,
            chosen_model=hop.to_model,
            reason_codes=reason_codes,
            applied=1,
            routed_header=routed_header,
            fallback=hop,
        )

    @property
    def _db_path(self) -> Any:
        """Where the kill switch's flag file lives."""
        log = self.settings.decisions
        return log.db_path if log is not None else default_db_path()

    def _passthrough_plan(self, body: bytes, reason: str, api_format: str) -> RoutingPlan:
        """The plan for a request the router is not allowed to change.

        Used by the kill switch and by an open circuit. The bytes are the ones
        that arrived, the row is written with `reason`, and `mode` is `off` so
        the log says the request was handled as if the router were disabled.

        Any hold for this session is released: while the router is not deciding,
        a hold would be a claim about a request it never evaluated.
        """
        try:
            metadata = read_request_metadata(body)
        except Exception:
            metadata = RequestMetadata(error="router_plan_failed")
        self._release_hold(metadata)
        return RoutingPlan(
            metadata=metadata,
            api_format=api_format,
            signals=None,
            sig_error=None,
            session_hint=None,
            action="STAY",
            chosen_model=metadata.model,
            chosen_effort=None,
            reason_codes=[reason],
            record_error=metadata.error,
            estimated_cost=None,
            mode="off",
            forward_body=body,
            applied=0,
            routed_header=None,
        )

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
        api_format = detect_api_format(self.path)
        if api_format is None:
            return None

        # The kill switch and the breaker are checked here, per request, so
        # neither needs a restart and neither can be missed by a code path that
        # forgets to look. Both pass the request straight through.
        switch = read_kill_switch(self._db_path)
        if switch.on:
            return self._passthrough_plan(body, REASON_KILL_SWITCH, api_format)

        breaker = self.settings.breaker
        if breaker is not None and breaker.is_open():
            return self._passthrough_plan(body, REASON_CIRCUIT_OPEN, api_format)

        mode = self._effective_mode
        try:
            metadata = read_request_metadata(body)

            signal_values: dict[str, Any] = {API_FORMAT_SIGNAL_KEY: api_format}
            sig_error: str | None = None
            signals: Signals | None = None
            parsed: Any = None
            try:
                parsed = json.loads(body.decode("utf-8"))
                # The path named the format, so the format decides how the body
                # is read. Both readers return the same dataclass, and nothing
                # below this line knows which one ran.
                signals = (
                    compute_signals_openai(parsed)
                    if api_format == API_FORMAT_OPENAI
                    else compute_signals(parsed)
                )
            except Exception:
                parsed = None
                signals = None
                sig_error = "signals_failed"

            # Classification reads the same parsed body, in memory, and only
            # writes two metadata values back. A failure clears both signals and
            # is counted as an error class; nothing below can forward differently
            # because of it.
            try:
                signals = with_classifier(parsed, signals, self.settings.config)
            except Exception:
                signals = without_classifier(signals)
                sig_error = sig_error or CLASSIFIER_FAILED

            signal_values = signals.to_dict() if signals is not None else {}

            action = "STAY"
            chosen_model = metadata.model
            chosen_effort: str | None = None
            reason_codes = ["PASSTHROUGH"]
            record_error = metadata.error or sig_error
            estimated_cost: float | str | None = None
            held_requests: int | None = None
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
                        decision, safety_failed, held_state = self._apply_safety(
                            decision,
                            signals,
                            store,
                            session_hint,
                            config,
                            requested_model,
                            mode,
                        )
                        safety_ran = not safety_failed
                        if safety_failed:
                            record_error = record_error or "safety_failed"
                        if held_state is not None:
                            # Only a session holding something reports a count, so
                            # an absent value means "no hold", not "a hold of zero".
                            held_requests = held_state.held_requests if held_state.has_hold else None
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

            breaker = self.settings.breaker
            if breaker is not None:
                # Only requests the router actually evaluated can move the
                # breaker. One the kill switch or an open circuit short-circuited
                # is not evidence that the router is broken.
                breaker.record(is_internal_error(record_error, reason_codes))

            return RoutingPlan(
                metadata=metadata,
                api_format=api_format,
                signals=signals,
                sig_error=sig_error,
                session_hint=session_hint,
                action=action,
                chosen_model=chosen_model,
                chosen_effort=chosen_effort,
                reason_codes=reason_codes,
                record_error=record_error,
                estimated_cost=estimated_cost,
                held_requests=held_requests,
                mode=mode,
                forward_body=forward_body,
                applied=applied,
                routed_header=routed_header,
            )
        except Exception:
            # Nothing above may prevent the request from being forwarded.
            return RoutingPlan(
                metadata=RequestMetadata(error="router_plan_failed"),
                api_format=api_format,
                signals=None,
                sig_error=None,
                session_hint=None,
                action="STAY",
                chosen_model=None,
                chosen_effort=None,
                reason_codes=["PASSTHROUGH"],
                record_error="router_plan_failed",
                estimated_cost=None,
                held_requests=None,
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

    def _record_usage(self, decision_id: int | None, plan: RoutingPlan | None) -> None:
        """Append one `router_usage` row for a POST in either recognised format.

        Runs after the response has ended and after the decision row exists, so
        the usage row can link to it. Counts come from the upstream, prices from
        the config, and neither is ever invented: an unknown figure is written
        as NULL, which is a different claim from 0.0.

        The row is written exactly as it is for Anthropic. Both extractors return
        the same `UsageRecord` and the same columns mean the same thing on both
        paths, so nothing below this line can tell which format answered.

        A failure here is reported as one short line and never affects anything:
        the response is already complete and the decision row is already
        written, so the worst a bad usage row can cost is itself.
        """
        log = self.settings.decisions
        if log is None or not self._records_usage:
            return

        record = self._usage_record
        if record is None:
            # The upstream never answered, or the request was rejected before
            # any body was read: there is no response to have reported usage.
            record = UsageRecord()

        chosen = plan.chosen_model if plan is not None else None
        requested = plan.metadata.model if plan is not None else None
        try:
            priced = estimate_cost(record, chosen, requested, self.settings.config)
        except Exception:
            priced = None

        notes = ",".join(priced.notes) if priced is not None and priced.notes else None
        cost = priced.cost_usd if priced is not None else None
        baseline = priced.baseline_cost_usd if priced is not None else None

        try:
            log.record_usage(
                UsageRow(
                    status=record.status,
                    decision_id=decision_id,
                    model_reported=record.model_reported,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cache_read_tokens=record.cache_read_tokens,
                    cache_write_tokens=record.cache_write_tokens,
                    cost_usd=cost,
                    baseline_cost_usd=baseline,
                    notes=notes,
                )
            )
        except Exception as exc:
            print(
                f"tamias-router: usage log write failed ({type(exc).__name__})",
                file=sys.stderr,
                flush=True,
            )

    def _record_decision(
        self,
        body: bytes,
        status: int,
        error_class: str | None,
        plan: RoutingPlan | None = None,
    ) -> int | None:
        """Append one metadata-only row for a POST the router recognises.

        Both recognised routes, Anthropic and OpenAI, get one row each, and the
        format is recorded in `signal_values` so a row can be read back without
        guessing which body produced it. Every other method and path is
        forwarded and not recorded. A failure here is reported as one short line
        and never affects the request, which has already been forwarded by the
        time this runs.

        `plan` is the decision computed before forwarding. It is recomputed
        only when it is absent, which happens for a request rejected before its
        body was read.

        Returns the new row's `decision_id`, or None when nothing was written,
        so the usage row can link to it.
        """
        log = self.settings.decisions
        if log is None or self.command != "POST":
            return None
        if detect_api_format(self.path) is None:
            return None

        if plan is None:
            plan = self._plan(body)
            if plan is None:
                return None

        try:
            signal_values: dict[str, Any] = {API_FORMAT_SIGNAL_KEY: plan.api_format}
            if plan.signals is not None:
                signal_values.update(plan.signals.to_dict())
            if plan.estimated_cost is not None:
                signal_values[COST_SIGNAL_KEY] = plan.estimated_cost
            if plan.held_requests is not None:
                signal_values[HELD_SIGNAL_KEY] = plan.held_requests
            if plan.fallback is not None:
                signal_values[FALLBACK_FROM_SIGNAL_KEY] = plan.fallback.from_model
                signal_values[FALLBACK_ATTEMPTS_SIGNAL_KEY] = plan.fallback.attempts

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
            return log.record(record)
        except Exception as exc:
            print(
                f"tamias-router: decision log write failed ({type(exc).__name__})",
                file=sys.stderr,
                flush=True,
            )
            return None

    def _release_hold(self, metadata: RequestMetadata) -> None:
        """Drop this session's hold without counting the request.

        Used on the paths that never reach a decision: the kill switch, an open
        circuit and mode `off`. `store.get` rather than `store.observe`, because
        a request the router did not evaluate must not move `requests_seen`,
        dwell or the switch counter.
        """
        store = self.settings.state
        log = self.settings.decisions
        if store is None or log is None:
            return
        try:
            hint = log.session_hint(metadata.first_user_text)
            current = store.get(hint)
            if current.has_hold:
                store.replace(hint, current.released_hold())
        except Exception:
            # Nothing about a hold may affect the request, which is already
            # being forwarded either way.
            pass

    def _apply_safety(
        self,
        decision: Any,
        signals: Signals,
        store: SessionStore,
        session_hint: str,
        config: Any,
        requested_model: str,
        mode: str = "active",
    ) -> tuple[Any, bool, SessionState | None]:
        """Run the safety layer and the hold; return `(decision, failed, state)`.

        Safety can only remove a switch, so a failure here can only cost the
        router a switch, never gain one: `safety_failure` turns the decision
        into a STAY, and because this runs before anything is forwarded, a
        STAY means the original bytes are what go upstream. A failure is
        recorded as an error class, never dropped and never applied.

        The hold runs afterwards, on safety's final answer, so it can only
        replace a STAY. It is given this request's `human_prompt_count`, so a
        hold recorded for one human prompt is released rather than served when
        the request in front of it carries a new one. Nothing is stored and no
        hold moves when safety failed: a layer that could not check the request
        does not get to act on it.
        """
        try:
            current = store.observe(session_hint)
            final, updated = apply_safety(decision, signals, current, config)
            final, updated = apply_hold(
                final,
                requested_model,
                signals.requested_effort_if_present,
                updated,
                config,
                mode,
                signals.human_prompt_count,
            )
            store.replace(session_hint, updated)
            return (final, False, updated)
        except Exception:
            return (safety_failure(decision), True, None)

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
        """Copy the client's headers, minus hop-by-hop and credential routing.

        The one header the policy can change is `Accept-Encoding`, and only when
        `upstream_identity_encoding` is on: the client's value is dropped and
        `identity` is sent in its place, so the upstream answers uncompressed and
        the counts in that answer can be read. With the key absent or false -
        which is every shipped config - the client's own value is forwarded
        exactly as it arrived.

        Every other header is copied unchanged, including the credentials the
        client sent: they are read in memory and forwarded, never logged (Rule 5).
        """
        connection_tokens = {
            token.strip().lower()
            for value in self.headers.get_all("Connection") or []
            for token in value.split(",")
            if token.strip()
        }
        identity_encoding = upstream_identity_encoding(self.settings.config)
        headers: list[tuple[str, str]] = []
        for name, value in self.headers.items():
            key = name.lower()
            if key in REQUEST_HEADERS_DROPPED or key.startswith("proxy-"):
                continue
            if key in connection_tokens:
                continue
            if identity_encoding and key == ACCEPT_ENCODING.lower():
                # Replaced below by a single header the router chose. Whatever
                # the client asked for, every copy of it, goes.
                continue
            headers.append((name, value))

        headers.append(("Host", self.target.authority))
        if identity_encoding:
            headers.append((ACCEPT_ENCODING, IDENTITY_ENCODING))
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

        # The response is rewound before anything is written, so the bytes that
        # reach the client are the whole body, including the first chunk. It
        # cannot change what is relayed, only that nothing is lost before the
        # first write, and a body left partly unread by an earlier reader cannot
        # shorten the response the client receives. A stream that cannot seek -
        # a socket, a chunked body - is left exactly as it was, which is the
        # behaviour this always had.
        with contextlib.suppress(Exception):
            response._fp.seek(0)  # type: ignore[attr-defined]

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
        self._head_sent = True
        if framing == "none":
            return

        # The parser sees a copy of what the client got, never the bytes
        # themselves: `piece` is relayed above, and only then is a copy handed
        # over. A parser that is slow, raises or is absent cannot delay or
        # change what the client receives.
        parser = self._usage_parser(response)

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
                self._feed_usage(parser, piece)
            if framing == "chunked":
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

        self._finish_usage(parser)

    def _usage_parser(self, response: http.client.HTTPResponse) -> UsageParser | None:
        """A parser for this response, or None when nothing is worth parsing.

        Only a POST in one of the two formats the router prices gets a parser, so
        every other request gets none at all and pays nothing for the feature. The
        extractor is picked by the request's format, because a chat-completions
        response and a Messages response are read differently and neither can be
        read as the other. Construction can raise on an odd header, so a failure
        here means "no row", never a changed response.
        """
        if not self._records_usage:
            return None
        parser_class = usage_parser_class(detect_api_format(self.path))
        if parser_class is None:
            return None
        try:
            return parser_class(
                response.status,
                content_encoding=response.getheader("Content-Encoding") or "",
                content_type=response.getheader("Content-Type") or "",
            )
        except Exception:
            return None

    def _feed_usage(self, parser: UsageParser | None, piece: bytes) -> None:
        """Hand the parser a copy of one chunk, and ignore anything it does.

        Every failure is swallowed. The bytes are already on the wire by this
        point, so a parser that cannot cope costs one row, never a response
        (Rule 3).
        """
        if parser is None:
            return
        try:
            parser.feed(bytes(piece))
        except Exception:
            pass

    def _finish_usage(self, parser: UsageParser | None) -> None:
        """Read the final record and queue it for writing after the response.

        The row is written by `_record_usage`, once the response has ended and
        the decision row exists to link to.
        """
        if parser is None:
            return
        try:
            record = parser.finish()
        except Exception:
            record = UsageRecord()
        self._usage_record = record

    @property
    def _records_usage(self) -> bool:
        """True only for a POST in a format the router prices.

        Both recognised formats are priced: Anthropic Messages and OpenAI chat
        completions. What decides that is `detect_api_format` and nothing else, so
        it is exactly the set of requests that get a decision row, and a path that
        names no format is forwarded and recorded nowhere - no decision row and no
        usage row, rather than a half-filled one.
        """
        return self.command == "POST" and detect_api_format(self.path) is not None

    def _reply_json(self, status: int, code: str, message: str | None = None) -> None:
        """A short JSON error. Carries no header, body or credential material.

        `message` is left out entirely when it is None, which is what a failure
        whose cause cannot be summarised safely gets: `{"error": ...}` then says
        what happened and nothing about why.
        """
        error: dict[str, str] = {"error": code}
        if message is not None:
            error["message"] = message
        payload = json.dumps(error).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._head_sent = True
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
    if settings.breaker is None:
        # Every proxy gets a breaker, so a router can never run without one by
        # forgetting to pass it. The threshold and cooldown come from the config
        # when it has a policy block, and from the shipped defaults otherwise.
        policy = getattr(settings.config, "policy", None)
        settings = replace(
            settings,
            breaker=CircuitBreaker(
                threshold=getattr(
                    policy, "breaker_error_threshold", DEFAULT_THRESHOLD
                )
                or DEFAULT_THRESHOLD,
                cooldown_seconds=getattr(
                    policy, "breaker_cooldown_seconds", DEFAULT_COOLDOWN_SECONDS
                )
                or DEFAULT_COOLDOWN_SECONDS,
            ),
        )
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
