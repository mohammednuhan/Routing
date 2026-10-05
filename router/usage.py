"""Token usage reported by a response. Numbers only, nothing else.

This module reads what an upstream response *says* it cost, and keeps nothing
but the numbers. Rule 1 applies to responses exactly as it applies to requests:
the assistant text in a response is response content, so it is parsed, counted
and dropped. `UsageRecord` has no field that could hold text, and the parser
keeps no buffer that survives the event it came from.

How it is fed
-------------

The proxy never hands this module the bytes it relays. It reads a chunk from the
upstream, writes those exact bytes to the client, and only then feeds a copy of
them here. So a bug here can change nothing the client receives: at worst the
row says UNKNOWN. `feed` treats its argument as read-only and never keeps a
reference to it.

Two shapes of response are understood:

* a single JSON body, read at the end, whose `usage` object carries the counts;
* an SSE stream, parsed event by event as the chunks arrive, so an event split
  across two chunks is still read once it completes.

Both response formats are read, each by its own extractor: `UsageParser` for
Anthropic Messages and `OpenAIUsageParser` for chat completions. The choice is
made by the proxy from the request's `api_format` and by nothing else. Both
produce the same `UsageRecord`, so `router/cost.py` and every report downstream
are unchanged by which format answered.

Statuses
--------

`OK`        counts were read.
`UNKNOWN`   they could not be read: an unsupported `Content-Encoding`, a body
            that is not the JSON it claims to be, or a response that simply
            carries no usage at all. UNKNOWN is never reported as zero.
`NO_USAGE`  the upstream answered with an HTTP status of 400 or above. The
            request was rejected, so no tokens were billed and there is nothing
            to look for. This is decided before a single byte is parsed.

`Content-Encoding: gzip` is decompressed here, on this module's copy only, with
`zlib`, incrementally so a stream is never assembled. Any other encoding is
left alone and the status becomes UNKNOWN rather than a guess.

Nothing here touches the network, the filesystem or any configuration.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import Any, Final

#: Counts were read.
STATUS_OK: Final = "OK"

#: Counts could not be read. Never zero, never free.
STATUS_UNKNOWN: Final = "UNKNOWN"

#: The upstream answered 4xx or 5xx: nothing was billed.
STATUS_NO_USAGE: Final = "NO_USAGE"

#: Every status this module can report.
LEGAL_STATUSES: Final = (STATUS_OK, STATUS_UNKNOWN, STATUS_NO_USAGE)

#: The lowest HTTP status that means "no usage": 400 and above.
NO_USAGE_FROM_STATUS: Final = 400

#: `Content-Encoding` values that need no decompression.
IDENTITY_ENCODINGS: Final = frozenset({"", "identity"})

#: The only transfer encoding this parser decodes.
SUPPORTED_ENCODINGS: Final = frozenset({"gzip", "x-gzip"})

#: A JSON body larger than this is not parsed at all: a usage object is small,
#: and a body this big is response content this module has no business holding.
#: The SSE path has no such limit, because its buffers are per event.
MAX_JSON_BYTES: Final = 4 * 1024 * 1024

#: Longest SSE event held while waiting for its blank-line terminator. A single
#: event larger than this is not the streaming response this parser understands.
MAX_SSE_EVENT_BYTES: Final = 1024 * 1024

#: SSE events carrying counts. Every other event is discarded unparsed.
EVENT_MESSAGE_START: Final = "message_start"
EVENT_MESSAGE_DELTA: Final = "message_delta"

#: `Content-Type` that means the body is a stream of SSE events.
SSE_CONTENT_TYPE: Final = "text/event-stream"

#: The `data:` payload that ends an SSE stream. It is not JSON, so it is
#: recognized rather than parsed: the end of a stream is not a parse failure.
SSE_DONE: Final = b"[DONE]"

#: Names a provider may give the cache-write count inside
#: `usage.prompt_tokens_details`. OpenAI's own schema has no cache-write field, so
#: these are the names the gateways that report one use; the first that is present
#: and numeric is taken, and a provider reporting none leaves the count unknown.
CACHE_WRITE_DETAIL_KEYS: Final = (
    "cache_write_tokens",
    "cache_creation_tokens",
    "cache_creation_input_tokens",
)


@dataclass(frozen=True)
class UsageRecord:
    """The counts a response reported, and how sure the extractor is.

    Every field except `status` is optional because every count is optional: an
    upstream may report some and not others, and a missing count stays None
    rather than becoming 0. There is no field here that could hold text.
    """

    model_reported: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    status: str = STATUS_UNKNOWN


def _as_int(value: Any) -> int | None:
    """A token count, or None. A bool or a string is not a count."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _as_text(value: Any) -> str | None:
    """A model id, or None. Never anything else, and never anything longer."""
    return value if isinstance(value, str) and value else None


class UsageParser:
    """Reads usage out of one response, incrementally, keeping only numbers.

    Create one per response, `feed` it a copy of each chunk as the chunk arrives,
    then call `finish` once. The record is final at `finish` and the parser
    should not be fed afterwards.

    Every method is total: a malformed body, an unsupported encoding or a
    decompressor that fails leaves the parser in the UNKNOWN state instead of
    raising. The caller can therefore feed it without a try/except, though the
    proxy wraps the call anyway (Rule 3).
    """

    def __init__(
        self,
        status: int,
        content_encoding: str = "",
        content_type: str = "",
    ) -> None:
        """`status` is the upstream HTTP status; the headers are its own.

        `content_encoding` and `content_type` are header *values*. They are read
        in memory to decide how to parse and are never logged.
        """
        self.http_status = status
        encoding = content_encoding.strip().lower()
        media_type = content_type.split(";", 1)[0].strip().lower()

        self._streaming = SSE_CONTENT_TYPE in media_type
        self._usable = self._decodable(encoding)
        self._decompressor: zlib.decompressobj | None = None
        if encoding in SUPPORTED_ENCODINGS:
            self._decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)

        self._body = bytearray()
        self._pending = bytearray()
        self._failed = False
        self._done = False

        self.model_reported: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cache_read_tokens: int | None = None
        self.cache_write_tokens: int | None = None
        self._saw_counts = False

    # --- public interface ---------------------------------------------------

    def feed(self, chunk: bytes) -> None:
        """Take a copy of one chunk of the response body. Never raises."""
        if self._done:
            return
        if not chunk:
            return
        try:
            if self._decompressor is not None:
                # `max_length` bounds a single decompression, so a compressed
                # body cannot expand into an unbounded buffer here.
                plain = self._decompressor.decompress(bytes(chunk), MAX_JSON_BYTES)
                if self._decompressor.unconsumed_tail:
                    # More output than this parser will ever hold.
                    self._fail()
                    return
            else:
                plain = bytes(chunk)
            if plain:
                self._consume(plain)
        except Exception:
            self._fail()

    def finish(self) -> UsageRecord:
        """The final record. Idempotent, and never raises."""
        if self._done:
            return self._record()
        try:
            self._finish()
        except Exception:
            self._fail()
        self._done = True
        return self._record()

    # --- status -------------------------------------------------------------

    def _status(self) -> str:
        if self.http_status >= NO_USAGE_FROM_STATUS:
            return STATUS_NO_USAGE
        if not self._usable or self._failed:
            return STATUS_UNKNOWN
        # OK means counts were read. A body that named a model but reported no
        # counts at all is not OK: pricing it would come out at 0.0, and 0.0 is
        # a claim that the response was free.
        return STATUS_OK if self._saw_counts else STATUS_UNKNOWN

    def _record(self) -> UsageRecord:
        return UsageRecord(
            model_reported=self.model_reported,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            status=self._status(),
        )

    @staticmethod
    def _decodable(encoding: str) -> bool:
        """False for an encoding this parser will not decode."""
        return encoding in IDENTITY_ENCODINGS or encoding in SUPPORTED_ENCODINGS

    def _fail(self) -> None:
        """Give up on reading counts, keeping whatever was read before.

        Dropped on purpose: once a body has failed to parse, the numbers already
        taken from it cannot be trusted as a complete set, and a partial set
        priced as if it were complete would understate the cost.
        """
        self._failed = True
        self._body.clear()
        self._pending.clear()
        self.model_reported = None
        self.input_tokens = None
        self.output_tokens = None
        self.cache_read_tokens = None
        self.cache_write_tokens = None
        self._saw_counts = False

    # --- body routing -------------------------------------------------------

    def _consume(self, plain: bytes) -> None:
        if self._streaming:
            self._feed_stream(plain)
        else:
            self._feed_json(plain)

    def _feed_json(self, plain: bytes) -> None:
        if len(self._body) + len(plain) > MAX_JSON_BYTES:
            self._fail()
            return
        self._body += plain

    def _feed_stream(self, plain: bytes) -> None:
        # A carriage return can only be SSE line framing or a raw CR, which JSON
        # forbids inside a string, so dropping it makes \r\n and \n equivalent
        # without touching any payload.
        self._pending += plain.replace(b"\r", b"")
        while True:
            end = self._pending.find(b"\n\n")
            if end < 0:
                if len(self._pending) > MAX_SSE_EVENT_BYTES:
                    self._fail()
                return
            event = bytes(self._pending[:end])
            del self._pending[: end + 2]
            self._handle_event(event)

    def _finish(self) -> None:
        if not self._usable:
            return
        if self._decompressor is not None:
            tail = self._decompressor.flush()
            if tail:
                self._consume(tail)
        if self._streaming:
            # A stream that ended without its final blank line still carries a
            # whole event: the terminator is transport, not content.
            if self._pending:
                self._handle_event(bytes(self._pending))
                self._pending.clear()
        elif self._body:
            self._read_json_body(bytes(self._body))

    # --- JSON ---------------------------------------------------------------

    def _read_json_body(self, body: bytes) -> None:
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            return
        self._read_usage(payload.get("usage"))
        self.model_reported = _as_text(payload.get("model"))

    def _read_usage(self, usage: Any, include_output: bool = True) -> None:
        """Take the counts from a `usage` object, and nothing else.

        `include_output` is False for `message_start`, which is not where the
        final output count lives: on a stream that count arrives with the last
        `message_delta`, and reading it earlier would price a partial response.
        """
        if not isinstance(usage, dict):
            return
        self._take("input", _as_int(usage.get("input_tokens")))
        if include_output:
            self._take("output", _as_int(usage.get("output_tokens")))
        self._take("cache_read", _as_int(usage.get("cache_read_input_tokens")))
        self._take("cache_write", _as_int(usage.get("cache_creation_input_tokens")))

    # --- SSE ----------------------------------------------------------------

    def _handle_event(self, event: bytes) -> None:
        """Read one complete SSE event, or discard it.

        Only `message_start` and `message_delta` are parsed. Every other event -
        the content deltas, which carry the assistant text - is dropped without
        ever being decoded, so no response text exists as a Python object here.
        """
        name: str | None = None
        data: list[bytes] = []
        for line in event.split(b"\n"):
            if not line or line.startswith(b":"):
                # A blank line or a comment (often a keep-alive ping).
                continue
            field, separator, value = line.partition(b":")
            if not separator:
                continue
            if value.startswith(b" "):
                value = value[1:]
            if field == b"event":
                name = value.decode("utf-8", "replace")
            elif field == b"data":
                data.append(value)

        if not data:
            return
        kind = self._kind_of(name, b"\n".join(data))
        if kind == EVENT_MESSAGE_START:
            self._read_start(b"\n".join(data))
        elif kind == EVENT_MESSAGE_DELTA:
            self._read_delta(b"\n".join(data))
        # Anything else is discarded here, unparsed and unreferenced.

    @staticmethod
    def _kind_of(name: str | None, payload: bytes) -> str | None:
        """Which counting event this is, by name or - failing that - by type.

        A stream that omits the `event:` field still labels its data with
        `"type"`. That is matched against the raw bytes rather than by decoding
        the payload, so an event that carries text is still never decoded.
        """
        if name in (EVENT_MESSAGE_START, EVENT_MESSAGE_DELTA):
            return name
        if name is not None:
            return None
        if b'"message_start"' in payload:
            return EVENT_MESSAGE_START
        if b'"message_delta"' in payload:
            return EVENT_MESSAGE_DELTA
        return None

    def _read_start(self, payload: bytes) -> None:
        event = json.loads(payload.decode("utf-8"))
        if not isinstance(event, dict):
            return
        message = event.get("message")
        if isinstance(message, dict):
            self._read_usage(message.get("usage"), include_output=False)
            self.model_reported = _as_text(message.get("model"))
        self._read_usage(event.get("usage"), include_output=False)

    def _read_delta(self, payload: bytes) -> None:
        """The last `message_delta` carries the final output count."""
        event = json.loads(payload.decode("utf-8"))
        if not isinstance(event, dict):
            return
        usage = event.get("usage")
        if isinstance(usage, dict):
            self._take("output", _as_int(usage.get("output_tokens")))

    def _take(self, name: str, value: int | None) -> None:
        if value is None:
            return
        setattr(self, f"{name}_tokens", value)
        self._saw_counts = True


class OpenAIUsageParser(UsageParser):
    """Reads usage out of a chat-completions response, keeping only numbers.

    Everything `UsageParser` already guarantees is inherited unchanged: gzip is
    decompressed on this module's copy and nowhere else, an unsupported encoding
    or a status of 400 and above settles the status before a byte is parsed, a
    body that names a model but reports no counts is UNKNOWN rather than 0, and
    no method raises. Only the shape of the response differs.

    Two shapes are read:

    * one JSON body whose `usage` object carries `prompt_tokens`,
      `completion_tokens` and, optionally, `prompt_tokens_details.cached_tokens`;
    * an SSE stream of `data: {json}` chunks terminated by `data: [DONE]`, where
      the counts are taken from whichever chunk reports a non-null `usage` object
      - conventionally a final chunk whose `choices` list is empty - and the
      model from the first chunk that names one.

    Normalization
    -------------
    OpenAI reports `prompt_tokens` as a total that *includes* the tokens served
    from cache, while the `router_usage` columns mean something narrower:
    `input_tokens` is the tokens that were not read from cache and
    `cache_read_tokens` is the ones that were. So the total is split here, once,
    on the way in:

        input_tokens      = prompt_tokens - cached_tokens
        cache_read_tokens = cached_tokens
        output_tokens     = completion_tokens

    Splitting rather than storing the total is what keeps `router/cost.py`
    correct unchanged: it prices `input_tokens` at the input rate and
    `cache_read_tokens` at the much lower cache-read rate, and pricing a total
    that already contains the cached tokens would bill them twice. When the
    provider reports no `cached_tokens` the split cannot be done, so
    `input_tokens` is the whole total and `cache_read_tokens` stays NULL: the
    figure is recorded as nobody having told us, which is what happened.

    The router never asks for usage. Whether a stream carries a `usage` object at
    all is the client's decision - it depends on `stream_options` in the request -
    and the request is forwarded byte for byte, so this extractor reads what
    arrived and says UNKNOWN when that is nothing. It never adds a field to the
    request to improve its own answer.
    """

    def _read_json_body(self, body: bytes) -> None:
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            return
        self._read_openai_usage(payload.get("usage"))
        self.model_reported = _as_text(payload.get("model"))

    def _read_openai_usage(self, usage: Any) -> None:
        """Take the counts from a chat-completions `usage` object, and nothing else.

        Every count is optional and read independently, so a provider that reports
        two of the four is stored as two known counts and two NULLs rather than as
        four guesses.
        """
        if not isinstance(usage, dict):
            return
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, dict) else {}

        prompt = _as_int(usage.get("prompt_tokens"))
        cached = _as_int(details.get("cached_tokens"))
        if cached is not None and prompt is not None and cached > prompt:
            # `prompt_tokens` is inclusive, so a cache read larger than the total
            # cannot be right and neither figure can be trusted. UNKNOWN is the
            # honest reading; a subtraction that came out negative is not.
            self._fail()
            return

        # The split described in the class docstring. Done here, on the copy, so
        # that every consumer of the columns downstream prices cache tokens once.
        # A cache read reported without a total cannot be subtracted from, so the
        # input count stays NULL rather than becoming the cache read itself.
        if prompt is None:
            input_value = None
        elif cached is None:
            input_value = prompt
        else:
            input_value = prompt - cached

        self._take("input", input_value)
        self._take("output", _as_int(usage.get("completion_tokens")))
        self._take("cache_read", cached)
        self._take("cache_write", self._cache_write(details))

    @staticmethod
    def _cache_write(details: dict[str, Any]) -> int | None:
        """The cache-write count from `prompt_tokens_details`, or None.

        Optional in this format: OpenAI's own schema has no such field, so a
        provider that does not report one leaves `cache_write_tokens` NULL.
        """
        for name in CACHE_WRITE_DETAIL_KEYS:
            value = _as_int(details.get(name))
            if value is not None:
                return value
        return None

    def _handle_event(self, event: bytes) -> None:
        """Read one complete SSE event's `data:` payloads.

        A chat-completions event carries no `event:` name to filter on, so every
        event is read and the counts are taken from whichever of them reports a
        `usage` object. That means the chunk holding the assistant delta is
        decoded - it has to be, to reach the `usage` beside it - and dropped
        whole: the only things taken from it are the model id and the counts, so
        no response text survives the event that carried it.
        """
        data: list[bytes] = []
        for line in event.split(b"\n"):
            if not line or line.startswith(b":"):
                # A blank line or a comment (often a keep-alive ping).
                continue
            field, separator, value = line.partition(b":")
            if not separator:
                continue
            if value.startswith(b" "):
                value = value[1:]
            if field == b"data":
                data.append(value)

        if data:
            self._read_chunk(b"\n".join(data))

    def _read_chunk(self, payload: bytes) -> None:
        """Read one SSE `data:` payload: a JSON chunk, or the end marker.

        A payload that is not the JSON it claims to be is discarded and reading
        continues. A stream is many payloads and one unreadable one must not throw
        away the counts already read, so this returns rather than failing.
        """
        if payload.strip() == SSE_DONE:
            return
        try:
            chunk = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return
        if not isinstance(chunk, dict):
            return
        # A null `usage` is the ordinary shape of every chunk before the last, so
        # only a non-null one is read. `choices` is never looked at: the assistant
        # text lives there and it is not this module's business.
        self._read_openai_usage(chunk.get("usage"))
        if self.model_reported is None:
            self.model_reported = _as_text(chunk.get("model"))


__all__ = [
    "CACHE_WRITE_DETAIL_KEYS",
    "EVENT_MESSAGE_DELTA",
    "EVENT_MESSAGE_START",
    "LEGAL_STATUSES",
    "MAX_JSON_BYTES",
    "MAX_SSE_EVENT_BYTES",
    "NO_USAGE_FROM_STATUS",
    "SSE_DONE",
    "STATUS_NO_USAGE",
    "STATUS_OK",
    "STATUS_UNKNOWN",
    "OpenAIUsageParser",
    "UsageParser",
    "UsageRecord",
]