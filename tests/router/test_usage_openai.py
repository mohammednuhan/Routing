"""Token usage reported by a chat-completions response. Numbers only, nothing else.

Rule 1 is the point of this file. A chat-completions response carries the
assistant's text in every chunk it streams, so an extractor that reads it for
token counts is one step away from storing it. Every test here that puts a
sentence into a response also asserts that sentence is nowhere in the database, so
the counting and the keeping are asserted together rather than one at a time.

Coverage, in the order it is read:

* the extractor on its own, on a single JSON body and on a stream, with and
  without cached tokens;
* the normalization, which is the whole reason the row is correct: OpenAI's
  `prompt_tokens` is a total that includes the cached tokens, and the
  `router_usage` columns mean something narrower;
* the same safeguards the Messages extractor has, because it is the same class:
  gzip on the module's copy only, an unsupported encoding, a status of 400 and
  above, a response that reports no counts, a model that is not the one asked for;
* the proxy, which selects the extractor from the request's path and writes one row
  per response, and which relays the request byte for byte while it does;
* the reports that read `router_usage`, so the row is worth having;
* a vendor model id carrying a slash and a colon, end to end.

Anthropic is not exercised here. `tests/router/test_usage.py` owns it, unchanged.
"""
from __future__ import annotations

import contextlib
import gzip
import http.client
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from router.config import RouterConfig, load_config
from router.cost import estimate_cost
from router.cost_report import build_cost_report, format_cost_report
from router.decisions import DB_ENV_VAR, DecisionLog, UsageRow
from router.killswitch import KILL_SWITCH_ENV
from router.proxy import (
    API_FORMAT_ANTHROPIC,
    API_FORMAT_OPENAI,
    ProxySettings,
    create_server,
    detect_api_format,
    usage_parser_class,
)
from router.state import SessionStore
from router.usage import (
    STATUS_NO_USAGE,
    STATUS_OK,
    STATUS_UNKNOWN,
    OpenAIUsageParser,
    UsageParser,
    UsageRecord,
)
from router.watch import WatchOptions, poll_once

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-OPENAI-test.yaml"

MODEL_LOW = "test-low"
MODEL_MID = "test-mid"
MODEL_HIGH = "test-high"

#: Vendor-style ids: a slash and a colon are ordinary characters in the ids real
#: gateways serve, so both ends of a hop have to carry them verbatim.
VENDOR_MID = "vendor/model-a:free"
VENDOR_HIGH = "vendor/model-b:free"

OPENAI_VERSIONED_PATH = "/v1/chat/completions"
LOOPBACK_HOST = "127.0.0.1"
CLIENT_TIMEOUT = 5.0

#: A phrase that must never reach the database, a log line or a report. It travels
#: in the request's prompt and in the response's streamed delta.
SECRET_PHRASE = "citrine-marmalade-9052"

#: Dummy write 3.75, dummy read 0.30, dummy benefit 0.001: the benefit covers a
#: rebuild below 0.001 * 1_000_000 / 3.45 = 289 tokens of context. The bodies here
#: are deliberately small so a switch is affordable and is never quietly blocked
#: on cost instead.
CROSSOVER_BYTES = 289 * 4

#: The counts the fake upstream reports. Invented numbers in shapes real providers
#: use, and never read by any test as a real price.
PROMPT_TOKENS = 920
CACHED_TOKENS = 800
COMPLETION_TOKENS = 45

#: "Build a `usage` object for me" as distinct from "here is a `usage` object that
#: does not parse". Spelled out because `None` is a value this format really does
#: report - it is every chunk of a stream before the last one.
MISSING = object()


# --- bodies -----------------------------------------------------------------


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def tool_call(name: str, index: int) -> dict[str, Any]:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def tool_result(index: int, text: str = "ok") -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": f"call_{index}", "content": text}


def stay_body(model: str = MODEL_MID, text: str = "plain question") -> bytes:
    """A body nothing acts on: one turn, no tools, a prompt that suits test-mid.

    The classifier is off in `config()`, so this is a STAY for the plain reason
    that no rule proposed anything - which is the state most requests are in and
    the one worth pricing.
    """
    return json.dumps({"model": model, "messages": [user_message(text)]}).encode()


def repeat_tool_body(model: str = MODEL_MID) -> bytes:
    """A body that escalates: one tool called three times in a row.

    `escalate_repeated_tool_calls: 3` in the sample, so this proposes test-high.
    It ends on a tool result, so nothing is pending and `block_in_tool_loop` does
    not stop it.
    """
    body = json.dumps(
        {
            "model": model,
            "max_completion_tokens": 16,
            "messages": [
                user_message("go"),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("search", 1), tool_call("search", 2), tool_call("search", 3)],
                },
                tool_result(1),
                tool_result(2),
                tool_result(3),
            ],
        }
    ).encode()
    assert len(body) < CROSSOVER_BYTES, "repeat_tool_body must stay cheap enough to be allowed"
    return body


def usage_object(cached: int | None = CACHED_TOKENS) -> dict[str, Any]:
    """A chat-completions `usage` object, with or without a cached count."""
    usage: dict[str, Any] = {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
    }
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached, "audio_tokens": 0}
    return usage


def json_response(
    model: str = MODEL_MID,
    cached: int | None = CACHED_TOKENS,
    cache_write: int | None = None,
    usage: Any = MISSING,
) -> bytes:
    """A whole chat-completions body carrying counts, and assistant text."""
    body: dict[str, Any] = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
    }
    if usage is MISSING:
        reported = usage_object(cached)
        if cache_write is not None:
            reported["prompt_tokens_details"]["cache_write_tokens"] = cache_write
        body["usage"] = reported
    else:
        body["usage"] = usage
    return json.dumps(body).encode()


def stream_chunks(
    model: str = MODEL_MID,
    cached: int | None = CACHED_TOKENS,
    cache_write: int | None = None,
    done: bool = True,
) -> list[bytes]:
    """A chat-completions stream, one SSE event per entry, as separate chunks.

    Shaped like a real one: the assistant delta first, then the finish reason, then
    a final chunk whose `choices` is empty and whose `usage` carries the counts,
    then `data: [DONE]`. `done=False` drops the terminator, which is a stream that
    was cut off.
    """
    usage_block = usage_object(cached)
    if cache_write is not None:
        usage_block["prompt_tokens_details"]["cache_write_tokens"] = cache_write

    events = [
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [],
            "usage": usage_block,
        },
    ]
    chunks = [b"data: " + json.dumps(event).encode() + b"\n\n" for event in events]
    if done:
        chunks.append(b"data: [DONE]\n\n")
    return chunks


# --- extractor-level tests --------------------------------------------------


def parsed(chunks: list[bytes], **kwargs: Any) -> UsageRecord:
    parser = OpenAIUsageParser(kwargs.pop("status", 200), **kwargs)
    for chunk in chunks:
        parser.feed(chunk)
    return parser.finish()


def test_a_whole_body_with_cached_tokens_is_split_into_the_two_columns():
    """The split, once, on the way in: 920 total, 800 of it cached, 120 not.

    `prompt_tokens` is a total, so storing it in `input_tokens` and 800 in
    `cache_read_tokens` would price the 800 twice - once at the input rate and
    once at the cache-read rate. The extractor cannot know any price; it can only
    report the two figures the columns mean.
    """
    record = parsed([json_response(cached=CACHED_TOKENS)])

    assert record.status == STATUS_OK
    assert record.model_reported == MODEL_MID
    assert record.input_tokens == PROMPT_TOKENS - CACHED_TOKENS == 120
    assert record.cache_read_tokens == CACHED_TOKENS == 800
    assert record.output_tokens == COMPLETION_TOKENS == 45
    assert record.cache_write_tokens is None


def test_a_whole_body_with_no_cached_count_keeps_the_total_and_nulls_the_cache():
    """With no `cached_tokens` there is no split to make, so none is invented.

    `input_tokens` is the whole total and `cache_read_tokens` is NULL. The
    alternative - reading NULL as zero - would claim nobody was served from cache,
    which nobody told us.
    """
    record = parsed([json_response(cached=None)])

    assert record.status == STATUS_OK
    assert record.input_tokens == PROMPT_TOKENS == 920
    assert record.cache_read_tokens is None
    assert record.output_tokens == COMPLETION_TOKENS


def test_a_cached_count_of_zero_is_read_and_not_dropped():
    """Zero is a count. A provider that reports it reported no cache reads, and
    `input_tokens` is then the whole total - which is the same number the no
    `cached_tokens` case gives, for a stated reason rather than a missing field."""
    record = parsed([json_response(cached=0)])

    assert record.status == STATUS_OK
    assert record.cache_read_tokens == 0
    assert record.input_tokens == PROMPT_TOKENS


def test_the_totals_line_up_with_the_sums_whatever_the_cache_read():
    """`input_tokens + cache_read_tokens == prompt_tokens` at every cached value.

    The invariant that makes the row safe to price: no cache token is double
    counted and none is dropped.
    """
    for cached in (0, 1, 120, 800, 919, PROMPT_TOKENS):
        record = parsed([json_response(cached=cached)])

        assert record.input_tokens is not None
        assert record.cache_read_tokens is not None
        assert record.input_tokens + record.cache_read_tokens == PROMPT_TOKENS


def test_a_cache_read_larger_than_the_total_is_unknown_rather_than_negative():
    """`prompt_tokens` is inclusive, so a cache read above it cannot be right.

    Neither figure can be trusted once the two contradict each other, so both are
    dropped and the row is UNKNOWN. Subtracting anyway would store a negative
    input count, and `router/cost.py` would price it as though it were a refund.
    """
    record = parsed([json_response(cached=PROMPT_TOKENS + 1)])

    assert record.status == STATUS_UNKNOWN
    assert record.input_tokens is None
    assert record.cache_read_tokens is None
    assert record.output_tokens is None


def test_a_cache_read_reported_without_a_total_is_recorded_and_input_stays_null():
    """Every count is optional and read independently.

    A provider that reports a cache read and no `prompt_tokens` has told us one
    figure it knows. There is nothing to subtract from, so `input_tokens` stays
    NULL - it does not become the cache read, and the arithmetic does not raise
    and throw away the counts already read from this chunk.
    """
    record = parsed([json_response(usage={"prompt_tokens_details": {"cached_tokens": 800}})])

    assert record.status == STATUS_OK
    assert record.cache_read_tokens == 800
    assert record.input_tokens is None
    assert record.output_tokens is None


def test_a_cache_write_count_is_taken_from_the_first_name_a_provider_reports():
    """OpenAI's own schema has no cache-write field, so the names are all
    aliases. Each is read in turn and the first numeric one wins."""
    for name in ("cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens"):
        record = parsed([json_response(cached=None, cache_write=None, usage={
            "prompt_tokens": 920,
            "completion_tokens": 45,
            "prompt_tokens_details": {"cached_tokens": 800, name: 200},
        })])

        assert record.cache_write_tokens == 200, name
        assert record.cache_read_tokens == 800


def test_a_string_count_is_not_a_count():
    """`"45"` is not 45. Reading it as a number would be inventing a figure, so a
    count that arrives as anything else is NULL and the status reflects whatever
    else was readable."""
    record = parsed([json_response(usage={"prompt_tokens": "920", "completion_tokens": 45})])

    assert record.input_tokens is None
    assert record.output_tokens == COMPLETION_TOKENS
    assert record.status == STATUS_OK


def test_a_bool_is_not_a_count():
    """`True` is an int in Python and 1 token is not what it means."""
    record = parsed([json_response(usage={"prompt_tokens": True, "completion_tokens": 45})])

    assert record.input_tokens is None
    assert record.output_tokens == COMPLETION_TOKENS


def test_a_null_usage_object_is_ignored_rather_than_read_as_zero():
    """`"usage": null` is the shape of a provider that declined to report. Every
    chunk of some streams carries it until the last one."""
    record = parsed([json_response(usage=None)])

    assert record.status == STATUS_UNKNOWN
    assert record.input_tokens is None
    assert record.output_tokens is None


def test_a_body_that_reports_no_usage_at_all_is_unknown_and_keeps_the_model():
    """The model is metadata and is kept; the counts are the missing part. The
    status is UNKNOWN rather than OK with zeros, because pricing zero tokens
    against a model that was billed would say the response was free."""
    body = json.dumps({"id": "chatcmpl-mock", "model": MODEL_MID, "choices": []}).encode()

    record = parsed([body])

    assert record.status == STATUS_UNKNOWN
    assert record.model_reported == MODEL_MID
    assert record.input_tokens is None


def test_a_body_that_is_not_json_is_unknown_and_does_not_raise():
    parser = OpenAIUsageParser(200)

    parser.feed(b"<html>gateway timeout</html>")

    assert parser.finish().status == STATUS_UNKNOWN


def test_a_status_of_400_and_above_is_no_usage_before_anything_is_parsed():
    """The request was rejected, so nothing was billed and there is nothing to look
    for.

    The status is settled from the HTTP code alone. Counts that a body carries
    anyway are read but never priced - `estimate_cost` prices nothing unless the
    status is OK - so a body that reports its own usage on an error cannot put a
    dollar figure in the report.
    """
    record = parsed([json_response()], status=429)

    assert record.status == STATUS_NO_USAGE

    priced = estimate_cost(record, MODEL_MID, MODEL_MID, config())
    assert priced.cost_usd is None
    assert priced.baseline_cost_usd is None
    assert priced.is_unknown


def test_an_unsupported_encoding_is_unknown_and_the_body_is_left_alone():
    """`br` is not decoded, and guessing at it would be inventing counts."""
    record = parsed([json_response()], content_encoding="br")

    assert record.status == STATUS_UNKNOWN
    assert record.input_tokens is None


def test_an_encoding_this_parser_never_sees_still_records_no_counts():
    """A header it does not know must not become a crash. Rule 3: the client still
    gets every byte, and the row says UNKNOWN."""
    record = parsed([json_response()], content_encoding="brotli-ish")

    assert record.status == STATUS_UNKNOWN


# --- the extractor on a stream ----------------------------------------------


def test_a_stream_is_read_and_the_counts_come_from_the_final_chunk():
    record = parsed(stream_chunks(), content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.model_reported == MODEL_MID
    assert record.input_tokens == 120
    assert record.cache_read_tokens == 800
    assert record.output_tokens == 45


def test_a_stream_works_with_the_parameters_a_proxy_actually_sends():
    """Real SSE responses carry parameters, and they are not what tells the parser
    anything: the counts are found by `usage`, not by the media type spelled a
    particular way."""
    record = parsed(stream_chunks(), content_type="text/event-stream; charset=utf-8")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


def test_an_event_split_across_two_chunks_is_still_read():
    """TCP does not respect events. The whole usage event arrives in two writes
    here, cut in the middle of the JSON, and it must be read once it completes."""
    whole = b"".join(stream_chunks())
    cut = whole.index(b'"usage"') + 30

    record = parsed([whole[:cut], whole[cut:]], content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120
    assert record.cache_read_tokens == 800
    assert record.output_tokens == 45


def test_an_event_split_one_byte_at_a_time_is_still_read():
    """The worst case the buffering exists for: every byte its own chunk."""
    chunks = [bytes([byte]) for byte in b"".join(stream_chunks())]

    record = parsed(chunks, content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


def test_a_crlf_framed_stream_is_read_like_a_lf_one():
    """`\\r\\n` is the framing many servers send. It must not change the counts."""
    whole = b"".join(stream_chunks()).replace(b"\n", b"\r\n")

    record = parsed([whole], content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


def test_the_done_marker_is_recognized_rather_than_parsed():
    """`data: [DONE]` is not JSON. A stream that ends with it is a stream that
    worked, so this must not be what turns the row UNKNOWN."""
    parser = OpenAIUsageParser(200, content_type="text/event-stream")

    for chunk in stream_chunks():
        parser.feed(chunk)
    record = parser.finish()

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


def test_a_stream_cut_off_before_its_done_marker_still_reports_what_it_carried():
    """The counts arrive before the terminator. A connection that drops after
    them does not un-count them."""
    record = parsed(stream_chunks(done=False), content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120
    assert record.output_tokens == 45


def test_a_stream_whose_last_event_has_no_blank_line_after_it_is_still_read():
    """The blank line is transport, not content. A final event that arrives without
    its terminator is a whole event."""
    whole = b"".join(stream_chunks()).rstrip(b"\n")

    record = parsed([whole], content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


def test_one_unreadable_chunk_does_not_throw_away_the_counts_already_read():
    """A stream is many payloads and one bad payload is not a reason to forget the
    rest. This counts in the assistant delta, reports its counts, then receives
    something that is not JSON; the counts survive."""
    before = stream_chunks()[:2]
    counts = stream_chunks()[2]

    record = parsed([*before, b"data: {not json\n\n", counts], content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120
    assert record.output_tokens == 45


def test_a_keep_alive_ping_between_chunks_is_not_an_error():
    """A `:` comment line is transport. Servers send them to hold a connection
    open, and one must not cost us the counts."""
    chunks: list[bytes] = []
    for index, chunk in enumerate(stream_chunks()):
        if index == 2:
            chunks.append(b": ping\n\n")
        chunks.append(chunk)

    record = parsed(chunks, content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


def test_the_model_is_the_first_one_named_not_the_last():
    """A gateway may label the first chunk with the model it routed to and the
    last with the one it actually served. The row is about the model that answered,
    which is the first id the stream carried."""
    first, second = stream_chunks()[0], stream_chunks()[2]
    assert b'"model": "test-mid"' in first

    record = parsed(
        [
            json.dumps({"model": "router-front", "choices": [{"delta": {}}]}).encode().join((b"data: ", b"\n\n")),
            first,
            second.replace(b'"model": "test-mid"', b'"model": "served-model"'),
        ],
        content_type="text/event-stream",
    )

    assert record.model_reported == "router-front"
    assert record.input_tokens == 120


def test_an_empty_model_string_is_not_a_model():
    """`""` is falsy and carries no id, so it does not overwrite a real one or
    stand in for one."""
    record = parsed(
        [b'data: {"model": "", "usage": null}\n\n', b"".join(stream_chunks())],
        content_type="text/event-stream",
    )

    assert record.model_reported == MODEL_MID


def test_a_stream_with_no_usage_chunk_is_unknown():
    """The client did not ask for usage, so the stream does not carry it. That is
    the client's decision, made in the request the router forwarded unchanged."""
    chunks = [chunk for chunk in stream_chunks() if b'"usage"' not in chunk]

    record = parsed(chunks, content_type="text/event-stream")

    assert record.status == STATUS_UNKNOWN
    assert record.input_tokens is None
    assert record.output_tokens is None


def test_a_stream_of_zero_choices_and_no_usage_is_unknown_not_free():
    """An empty choices list with no counts is not a free response."""
    chunk = b'data: {"model": "test-mid", "choices": []}\n\n'

    record = parsed([chunk], content_type="text/event-stream")

    assert record.status == STATUS_UNKNOWN


def test_choices_are_never_read_and_the_text_in_them_is_not_kept():
    """The counts come from `usage`, and `choices` is where the assistant's words
    are. Nothing in the record can hold them, and this is the test that says so.

    The phrase below is in the delta of a real-shaped chunk; it must not appear in
    any attribute of the record.
    """
    chunk = json.dumps(
        {
            "model": MODEL_MID,
            "choices": [{"delta": {"content": SECRET_PHRASE}, "finish_reason": None}],
            "usage": usage_object(cached=CACHED_TOKENS),
        }
    ).encode()

    record = parsed([b"data: " + chunk + b"\n\n"], content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120
    assert SECRET_PHRASE not in json.dumps(record.__dict__, default=str)


def test_a_stream_that_is_not_sse_is_read_as_whole_json():
    """A path says chat-completions but the response came back as one body. The
    media type decides how it is read, not the request."""
    record = parsed([json_response()], content_type="application/json")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


def test_a_chat_completions_body_arriving_as_sse_yields_unknown_not_a_crash():
    """The mirror of the above, and the awkward one: the whole body read as a
    stream is one payload that is not a `data:` event, so there is nothing to read
    and the row says UNKNOWN. It must not raise."""
    record = parsed([json_response()], content_type="text/event-stream")

    assert record.status == STATUS_UNKNOWN


# --- gzip -------------------------------------------------------------------


def test_a_gzipped_body_is_read_from_this_module_s_copy():
    """gzip is decoded here and nowhere else: the proxy relayed the compressed
    bytes to the client exactly as they arrived."""
    body = gzip.compress(json_response(cached=CACHED_TOKENS))

    record = parsed([body], content_encoding="gzip")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120
    assert record.cache_read_tokens == 800


def test_a_gzipped_stream_is_read_across_chunk_boundaries():
    """The compressed bytes are cut wherever the network cut them, and the counts
    are still there. Compressing the whole stream and splitting it at an awkward
    point is the case a one-shot decompressor would get wrong."""
    packed = gzip.compress(b"".join(stream_chunks()))

    record = parsed(
        [packed[:20], packed[20:]],
        content_encoding="gzip",
        content_type="text/event-stream",
    )

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


def test_a_corrupt_gzip_body_is_unknown_rather_than_an_exception():
    """A truncated body is what a dropped connection looks like here."""
    record = parsed([b"\x1f\x8b\x08\x00garbage"], content_encoding="gzip")

    assert record.status == STATUS_UNKNOWN


def test_x_gzip_is_the_same_encoding():
    record = parsed([gzip.compress(json_response())], content_encoding="x-gzip")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


def test_identity_is_left_alone():
    record = parsed([json_response()], content_encoding="identity")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120


# --- the proxy --------------------------------------------------------------


def config() -> RouterConfig:
    """The sample with the classifier off, so every switch here is a policy switch.

    Otherwise the classifier could move a request for reasons that have nothing to
    do with pricing, and a test asserting a chosen model would be asserting a
    keyword list.
    """
    loaded = load_config(CONFIG_PATH)
    assert loaded.classifier is not None
    return replace(loaded, classifier=replace(loaded.classifier, enabled=False))


def priced_config() -> RouterConfig:
    """`config()` with test-high priced lower than test-mid.

    Only `input_per_million` and `output_per_million` move. The safety layer's cost
    check reads the target's cache-write and cache-read rates, so leaving those
    alone keeps the switch affordable and this test still about pricing.
    """
    base = config()
    models = tuple(
        replace(spec, input_per_million=1.0, output_per_million=3.0)
        if spec.id == MODEL_HIGH
        else spec
        for spec in base.models
    )
    return replace(base, models=models)


def shipped_config() -> RouterConfig:
    """The shipped file: shadow mode, and every price null.

    Read, never written. It is the standing proof that a response nobody has a
    price for is recorded as unpriced rather than as free.
    """
    return load_config(REPO_ROOT / "router" / "config.yaml")


def log_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DecisionLog:
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    return DecisionLog.from_env()


@dataclass
class Recorded:
    body: bytes
    headers: dict[str, str]


@dataclass
class MockState:
    """What the fake upstream was asked for and what it is allowed to answer."""

    requests: list[Recorded] = field(default_factory=list)
    hold_stream: bool = False
    #: Set by the test to let the upstream send the rest of a held stream.
    release_stream: threading.Event = field(default_factory=threading.Event)
    sent_chunks: list[bytes] = field(default_factory=list)


class OpenAIUpstream(BaseHTTPRequestHandler):
    """Answers `/v1/chat/completions` with a body or a stream the test chose.

    A local fake on the loopback. It never reaches a provider, and the body it
    answers with is the test's own bytes.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass

    def _send(self, status: int, payload: bytes, content_type: str, encoding: str = "") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        server = self.server
        state: MockState = server.state  # type: ignore[attr-defined]
        state.requests.append(Recorded(body=raw, headers={k.lower(): v for k, v in self.headers.items()}))

        if not self.path.endswith("/chat/completions"):
            self._send(404, b'{"error":"no such route"}', "application/json")
            return

        stream = b'"stream":true' in raw.replace(b" ", b"")
        if stream:
            self._send_stream()
            return

        payload = server.json_body  # type: ignore[attr-defined]
        if server.content_encoding == "gzip":  # type: ignore[attr-defined]
            payload = gzip.compress(payload)
        self._send(
            server.upstream_status or 200,  # type: ignore[attr-defined]
            payload,
            "application/json",
            server.content_encoding,  # type: ignore[attr-defined]
        )
    def _send_stream(self) -> None:
        server = self.server
        state: MockState = server.state  # type: ignore[attr-defined]
        chunks: list[bytes] = server.stream_body  # type: ignore[attr-defined]
        encoding: str = server.content_encoding  # type: ignore[attr-defined]

        if state.hold_stream:
            self.send_response(server.upstream_status or 200)  # type: ignore[attr-defined]
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            if encoding:
                self.send_header("Content-Encoding", encoding)
            self.end_headers()

            for index, chunk in enumerate(chunks):
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
                state.sent_chunks.append(chunk)
                if index == 0:
                    state.release_stream.wait(timeout=CLIENT_TIMEOUT)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        body = b"".join(chunks)
        if encoding == "gzip":
            body = gzip.compress(body)
        self._send(server.upstream_status or 200, body, "text/event-stream", encoding)  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        self._send(404, b'{"error":"no such route"}', "application/json")


@dataclass
class Harness:
    proxy_port: int
    state: MockState
    upstream_port: int
    respond: Any
    streamed: Any


@contextlib.contextmanager
def running_proxy(
    settings_config: RouterConfig,
    log: DecisionLog,
    json_body: bytes | None = None,
    stream_body: list[bytes] | None = None,
    content_encoding: str = "",
    upstream_status: int | None = None,
) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), OpenAIUpstream)
    upstream.daemon_threads = True
    upstream.state = MockState()  # type: ignore[attr-defined]
    upstream.json_body = json_body if json_body is not None else json_response()  # type: ignore[attr-defined]
    upstream.stream_body = stream_body if stream_body is not None else stream_chunks()  # type: ignore[attr-defined]
    upstream.content_encoding = content_encoding  # type: ignore[attr-defined]
    upstream.upstream_status = upstream_status  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    store = SessionStore()
    server = create_server(
        ProxySettings(
            listen_port=0,
            upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
            mode=settings_config.mode,
            decisions=log,
            config=settings_config,
            state=store,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def set_json(payload: bytes) -> None:
        upstream.json_body = payload  # type: ignore[attr-defined]

    def set_stream(chunks: list[bytes]) -> None:
        upstream.stream_body = chunks  # type: ignore[attr-defined]

    try:
        yield Harness(
            proxy_port=server.server_address[1],
            state=upstream.state,  # type: ignore[attr-defined]
            upstream_port=upstream.server_address[1],
            respond=set_json,
            streamed=set_stream,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def request(
    port: int,
    body: bytes,
    path: str = OPENAI_VERSIONED_PATH,
    method: str = "POST",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        send = {"Content-Type": "application/json"}
        send.update(headers or {})
        connection.request(method, path, body=body, headers=send)
        response = connection.getresponse()
        payload = response.read()
        return (
            response.status,
            {name.lower(): value for name, value in response.getheaders()},
            payload,
        )
    finally:
        connection.close()


def wait_for_usage(log: DecisionLog, expected: int = 1, timeout: float = 5.0) -> int:
    """The usage row is written after the response has ended, so it can trail it."""
    deadline = time.monotonic() + timeout
    count = log.usage_count()
    while count < expected and time.monotonic() < deadline:
        time.sleep(0.01)
        count = log.usage_count()
    return count


def usage_row(log: DecisionLog) -> Any:
    rows = log.usage_rows(1)
    assert rows, "no usage row was written"
    return rows[0]


def database_bytes(db_path: Path) -> bytes:
    """Every byte of the database file, including any journal beside it.

    A secret can hide in an unallocated page that a query would never look at, so
    the assertion is over the file rather than over a `SELECT`.
    """
    blobs = [db_path.read_bytes()]
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            blobs.append(sidecar.read_bytes())
    return b"".join(blobs)


def decision_rows(log: DecisionLog) -> list[Any]:
    return log.recent(10)


def applied_switch_count(log: DecisionLog) -> int:
    return sum(1 for row in decision_rows(log) if row.applied)


def chosen_models(log: DecisionLog) -> list[str | None]:
    return [row.chosen_model for row in decision_rows(log)]


def requested_models(log: DecisionLog) -> list[str | None]:
    return [row.requested_model for row in decision_rows(log)]


def usage_rows_via_sqlite(db_path: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(db_path)
    try:
        connection.row_factory = sqlite3.Row
        return list(connection.execute("SELECT * FROM router_usage ORDER BY usage_id"))
    finally:
        connection.close()


# --- the proxy writes one row ----------------------------------------------


def test_the_path_chooses_the_extractor_and_nothing_else_does():
    """The format comes from the path and only the path.

    The body is not inspected to decide, a query string does not change it, and
    the same body posted to the other route is read by the other extractor. This is
    why a chat-completions response cannot have its `prompt_tokens` read as
    `input_tokens` by mistake.
    """
    assert detect_api_format("/v1/chat/completions") == API_FORMAT_OPENAI
    assert detect_api_format("/chat/completions") == API_FORMAT_OPENAI
    assert detect_api_format("/v1/chat/completions?beta=true") == API_FORMAT_OPENAI
    assert detect_api_format("/v1/messages") == API_FORMAT_ANTHROPIC
    assert detect_api_format("/v1/embeddings") is None

    assert usage_parser_class(API_FORMAT_OPENAI) is OpenAIUsageParser
    assert usage_parser_class(API_FORMAT_ANTHROPIC) is UsageParser
    assert usage_parser_class(None) is None


def test_the_two_extractors_are_not_interchangeable():
    """Neither can read the other, which is what makes choosing by format necessary
    rather than merely tidy: a Messages body has no `prompt_tokens`, and a
    chat-completions body has no `input_tokens`.

    Both are read as UNKNOWN by the wrong extractor, so a mispairing would show up
    as a missing row rather than as wrong numbers.
    """
    from router.usage import UsageParser

    chat = json_response(cached=CACHED_TOKENS)
    messages = json.dumps(
        {
            "model": MODEL_MID,
            "usage": {
                "input_tokens": 120,
                "output_tokens": 45,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 200,
            },
        }
    ).encode()

    anthropic = UsageParser(200)
    anthropic.feed(chat)
    assert anthropic.finish().status == STATUS_UNKNOWN

    chat_parser = OpenAIUsageParser(200)
    chat_parser.feed(messages)
    assert chat_parser.finish().status == STATUS_UNKNOWN


def test_one_chat_completions_response_records_one_row_with_the_split_counts(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        status, _, payload = request(harness.proxy_port, stay_body())
        assert status == 200
        assert wait_for_usage(log) == 1

    row = usage_row(log)
    assert row.status == STATUS_OK
    assert row.input_tokens == 120
    assert row.cache_read_tokens == 800
    assert row.output_tokens == 45
    assert row.cache_write_tokens is None
    assert json.loads(payload)["model"] == MODEL_MID


def test_the_response_is_relayed_byte_for_byte(tmp_path, monkeypatch):
    """The client gets the upstream's bytes unchanged, including the counts in the
    shape the upstream sent them. The row is normalized; the wire is not."""
    log = log_at(tmp_path, monkeypatch)
    body = json_response(cached=CACHED_TOKENS)

    with running_proxy(config(), log, json_body=body) as harness:
        status, headers, payload = request(harness.proxy_port, stay_body())

    assert status == 200
    assert payload == body
    assert headers["content-type"] == "application/json"
    # Normalized on the way in, and not on the way out.
    assert json.loads(payload)["usage"]["prompt_tokens"] == PROMPT_TOKENS


def test_the_request_is_relayed_byte_for_byte(tmp_path, monkeypatch):
    """Rule 2 and Rule 3: what goes upstream is exactly what the client sent.

    No `stream_options` is added, no field is reordered, no whitespace changes.
    That is what lets the row say UNKNOWN when a stream carries no counts: the
    client decided that, not the router.
    """
    log = log_at(tmp_path, monkeypatch)
    body = stay_body()

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, body)
        assert wait_for_usage(log) == 1
        assert harness.state.requests

    forwarded = harness.state.requests[-1]
    assert forwarded.body == body
    assert b"stream_options" not in forwarded.body


def test_the_request_is_relayed_unchanged_even_when_the_router_rewrote_the_model(tmp_path, monkeypatch):
    """The one mode in which a body may change, and the row is still the upstream's.

    The rewrite is visible in the forwarded body; the counts are the upstream's
    whatever the client asked for.
    """
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, repeat_tool_body())
        assert wait_for_usage(log) == 1
        assert harness.state.requests

    forwarded = json.loads(harness.state.requests[-1].body)
    assert forwarded["model"] == MODEL_HIGH
    row = usage_row(log)
    assert row.status == STATUS_OK
    assert row.input_tokens == 120


def test_a_stream_records_one_row_and_relays_every_chunk_incrementally(tmp_path, monkeypatch):
    """A streamed response is relayed chunk by chunk as it arrives, not buffered
    to the end. The first chunk must be readable by the client while the upstream
    is still holding the rest."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        harness.state.hold_stream = True
        chunks = harness.state
        connection = http.client.HTTPConnection(LOOPBACK_HOST, harness.proxy_port, timeout=CLIENT_TIMEOUT)
        try:
            stream_body = json.dumps(
                {"model": MODEL_MID, "stream": True, "messages": [user_message("go")]}
            ).encode()
            connection.request("POST", OPENAI_VERSIONED_PATH, body=stream_body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            assert response.status == 200
            first = response.read1(4096)
            assert first, "the client received no first chunk"

            # The upstream is still holding chunks 2..n while the client already
            # has the first one.
            chunks.release_stream.set()
            rest = response.read()
            payload = first + rest
        finally:
            connection.close()

        assert wait_for_usage(log) == 1
        assert len(chunks.sent_chunks) == len(stream_chunks())

    assert payload == b"".join(stream_chunks())
    assert payload.rstrip().endswith(b"data: [DONE]")
    row = usage_row(log)
    assert row.status == STATUS_OK
    assert row.input_tokens == 120
    assert row.output_tokens == 45


def test_a_gzipped_response_is_relayed_gzipped_and_still_recorded(tmp_path, monkeypatch):
    """The client gets the compressed bytes and the header that says so. The row is
    read from a decompressed copy the client never sees."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log, content_encoding="gzip") as harness:
        status, headers, payload = request(harness.proxy_port, stay_body())
        assert status == 200
        assert wait_for_usage(log) == 1

    assert headers["content-encoding"] == "gzip"
    assert json.loads(gzip.decompress(payload))["usage"]["prompt_tokens"] == PROMPT_TOKENS
    assert usage_row(log).status == STATUS_OK


def test_a_gzipped_stream_is_relayed_gzipped_and_still_recorded(tmp_path, monkeypatch):
    """The same on the streaming path, where the compressed bytes are cut wherever
    the network cut them and the header has to be forwarded with them."""
    log = log_at(tmp_path, monkeypatch)
    chunks = stream_chunks()

    with running_proxy(config(), log, stream_body=chunks, content_encoding="gzip") as harness:
        status, headers, payload = request(
            harness.proxy_port,
            json.dumps({"model": MODEL_MID, "stream": True, "messages": [user_message("go")]}).encode(),
        )
        assert status == 200
        assert wait_for_usage(log) == 1

    assert headers["content-encoding"] == "gzip"
    assert gzip.decompress(payload) == b"".join(chunks)
    row = usage_row(log)
    assert row.status == STATUS_OK
    assert row.input_tokens == 120
    assert row.cache_read_tokens == 800


def test_an_unsupported_encoding_is_still_relayed_and_the_row_is_unknown(tmp_path, monkeypatch):
    """Rule 3. The router cannot read `br`, so it says so in the row and does not
    touch the response: the bytes the client gets are the upstream's."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log, content_encoding="br") as harness:
        status, headers, payload = request(harness.proxy_port, stay_body())
        assert status == 200
        assert wait_for_usage(log) == 1

    assert headers["content-encoding"] == "br"
    assert json.loads(payload)["usage"]["prompt_tokens"] == PROMPT_TOKENS
    row = usage_row(log)
    assert row.status == STATUS_UNKNOWN
    assert row.input_tokens is None
    assert row.cost_usd is None


def test_a_500_from_the_upstream_is_relayed_and_the_row_says_no_usage(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log, upstream_status=500) as harness:
        status, _, payload = request(harness.proxy_port, stay_body())
        assert status == 500
        assert wait_for_usage(log) == 1
        assert harness.state.requests

    assert json.loads(payload)["usage"]["prompt_tokens"] == PROMPT_TOKENS
    row = usage_row(log)
    assert row.status == STATUS_NO_USAGE
    # The counts are read but never priced: the row is NO_USAGE, so both money
    # columns are NULL and nothing enters a total.
    assert row.cost_usd is None
    assert row.baseline_cost_usd is None


def test_an_upstream_error_still_produces_one_row(tmp_path, monkeypatch):
    """One row per response, whatever the response was. A request that failed
    still costs a decision row, and a row saying NO_USAGE is what keeps the cost
    report from reading it as a request that was never made."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log, upstream_status=429) as harness:
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    assert usage_row(log).status == STATUS_NO_USAGE


def test_a_model_that_is_not_the_one_asked_for_is_noted_and_still_priced(tmp_path, monkeypatch):
    """The upstream answered with a model nobody configured. The counts are real
    and are kept, the row says so in its notes, and the cost is still worked out
    against the model the router chose - because that is what the router would have
    paid for the work, and guessing a price for a model nobody priced would be
    inventing one."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log, json_body=json_response(model="some-other-model")) as harness:
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    row = usage_row(log)
    assert row.status == STATUS_OK
    assert row.model_reported == "some-other-model"
    assert "MODEL_MISMATCH" in row.notes
    # 120 * 3.0 / 1e6 + 45 * 15.0 / 1e6 + 800 * 0.30 / 1e6
    assert row.cost_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)


def test_a_model_that_matches_the_chosen_one_carries_no_note(tmp_path, monkeypatch):
    """A gateway that answered with the model the router asked for is the ordinary
    case, and it is recorded without an annotation."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log, json_body=json_response(model=MODEL_MID)) as harness:
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    row = usage_row(log)
    assert row.model_reported == MODEL_MID
    assert not row.notes


def test_a_model_named_without_counts_is_unknown_and_noted(tmp_path, monkeypatch):
    """The mismatch is worth recording even when there is nothing to price. The
    metadata survives; the numbers do not."""
    log = log_at(tmp_path, monkeypatch)
    body = json.dumps({"id": "chatcmpl-mock", "model": "some-other-model", "choices": []}).encode()

    with running_proxy(config(), log, json_body=body) as harness:
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    row = usage_row(log)
    assert row.status == STATUS_UNKNOWN
    assert row.model_reported == "some-other-model"
    assert row.cost_usd is None


def test_a_parser_that_raises_does_not_break_forwarding(tmp_path, monkeypatch):
    """Rule 3, applied to this module: a bug in counting cannot cost the client a
    byte. The parser is swapped for one that throws on every call and the response
    still arrives whole, with a row that says UNKNOWN."""
    log = log_at(tmp_path, monkeypatch)

    class Exploding(OpenAIUsageParser):
        def feed(self, chunk: bytes) -> None:
            raise RuntimeError("counting failed")

        def finish(self) -> UsageRecord:
            raise RuntimeError("counting failed")

    monkeypatch.setattr("router.proxy.OpenAIUsageParser", Exploding)
    body = json_response(cached=CACHED_TOKENS)

    with running_proxy(config(), log, json_body=body) as harness:
        status, _, payload = request(harness.proxy_port, stay_body())
        assert status == 200
        assert wait_for_usage(log) == 1

    assert payload == body
    row = usage_row(log)
    assert row.status == STATUS_UNKNOWN
    assert row.input_tokens is None
    assert row.cost_usd is None


def test_a_parser_that_raises_on_a_stream_does_not_break_forwarding(tmp_path, monkeypatch):
    """The same on the streaming path, where the parser is fed a chunk at a time
    and a raise would happen mid-response with bytes already on the wire."""
    log = log_at(tmp_path, monkeypatch)

    class Exploding(OpenAIUsageParser):
        def feed(self, chunk: bytes) -> None:
            raise RuntimeError("counting failed")

    monkeypatch.setattr("router.proxy.OpenAIUsageParser", Exploding)

    with running_proxy(config(), log) as harness:
        status, _, payload = request(
            harness.proxy_port,
            json.dumps({"model": MODEL_MID, "stream": True, "messages": [user_message("go")]}).encode(),
        )
        assert status == 200
        assert wait_for_usage(log) == 1

    assert payload == b"".join(stream_chunks())
    assert usage_row(log).status == STATUS_UNKNOWN


def test_a_usage_row_that_cannot_be_written_does_not_break_forwarding(tmp_path, monkeypatch, capsys):
    """The row is written after the response has ended, so a failure here can cost
    the row and nothing else. Rule 3 again."""
    log = log_at(tmp_path, monkeypatch)
    body = json_response()

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(log, "record_usage", refuse)

    with running_proxy(config(), log, json_body=body) as harness:
        status, _, payload = request(harness.proxy_port, stay_body())
        assert status == 200

    assert payload == body
    assert log.usage_count() == 0
    # The failure is counted and reported, and it names the failure and not the row.
    assert "usage log write failed" in capsys.readouterr().err


def test_a_get_is_not_priced(tmp_path, monkeypatch):
    """Only a POST carries a response worth counting. A GET has no request body to
    route and no usage object to read, so it is forwarded and not priced."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        status, _, _ = request(harness.proxy_port, None, method="GET")
        assert status == 404

    assert log.usage_count() == 0


def test_two_responses_record_two_rows_and_the_link_to_the_decision_holds(tmp_path, monkeypatch):
    """One row per response, each linked to the decision it belongs to. This is
    what makes the cost report's own figures add up."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, stay_body(text="first"))
        request(harness.proxy_port, stay_body(text="second"))
        assert wait_for_usage(log, 2) == 2

    rows = log.usage_rows(2)
    assert [row.input_tokens for row in rows] == [120, 120]
    assert [row.output_tokens for row in rows] == [45, 45]
    assert [row.cache_read_tokens for row in rows] == [800, 800]
    # Two decisions, two usage rows, each pointing at one of them.
    assert sorted(row.decision_id for row in rows) == sorted(
        row.decision_id for row in decision_rows(log)
    )


# --- pricing ----------------------------------------------------------------


def test_a_free_model_costs_exactly_zero():
    """`test-low` carries 0.0 for all four prices on purpose. Zero is a price, not
    an absence, so a real response on a free model is priced and costs 0.0 - it is
    not rounded to NULL and not reported as unknown.

    The response names test-mid and the router chose test-low, so the row also
    carries a mismatch note. That is separate: the counts were read and priced, and
    the note is only saying they came from a model nobody asked for.
    """
    record = parsed([json_response(cached=CACHED_TOKENS)])
    priced = estimate_cost(record, MODEL_LOW, MODEL_MID, config())

    assert record.status == STATUS_OK
    assert priced.cost_usd == 0.0
    assert priced.baseline_cost_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)
    assert not priced.is_unknown
    assert "MODEL_MISMATCH" in priced.notes


def test_a_null_price_with_real_counts_is_unpriced_and_never_zero():
    """The shipped config prices nothing, because nobody has verified a price.

    A response that really was billed must not be recorded as costing 0.0: that
    would be a free response that was not free. Both figures are None, the counts
    are kept, and nothing is invented to fill the gap.
    """
    record = parsed([json_response(model="TODO_MODEL_TIER_MID", cached=CACHED_TOKENS)])
    priced = estimate_cost(record, "TODO_MODEL_TIER_MID", "TODO_MODEL_TIER_MID", shipped_config())

    assert record.status == STATUS_OK
    assert record.input_tokens == 120
    assert priced.cost_usd is None
    assert priced.baseline_cost_usd is None
    assert priced.is_unknown
    # The response named the model it was sent to, so there is nothing to note.
    assert not priced.notes


def test_the_counts_are_kept_when_the_price_is_unknown(tmp_path, monkeypatch):
    """Unpriced is about the price, not about the tokens.

    The same response against the shipped config keeps every count it reported: the
    row is `OK` because the counts were read, and the two money columns are NULL
    because nobody has verified a price. The status describes the reading, and the
    NULLs describe the pricing; they are not the same question.
    """
    log = log_at(tmp_path, monkeypatch)
    shadow = shipped_config()

    with running_proxy(shadow, log) as harness:
        request(harness.proxy_port, stay_body(model="TODO_MODEL_TIER_MID"))
        assert wait_for_usage(log) == 1

    row = usage_row(log)
    assert row.status == STATUS_OK
    assert row.input_tokens == 120
    assert row.cache_read_tokens == 800
    assert row.output_tokens == 45
    assert row.cost_usd is None
    assert row.baseline_cost_usd is None


def test_a_stay_costs_the_same_as_the_baseline():
    """No switch, so the two figures are the same number, and the row says so by
    having them equal rather than by having one of them missing."""
    record = parsed([json_response(cached=CACHED_TOKENS)])
    priced = estimate_cost(record, MODEL_MID, MODEL_MID, config())

    assert priced is not None
    assert priced.cost_usd == priced.baseline_cost_usd
    # 120 * 3.0 + 45 * 15.0 + 800 * 0.30, over a million.
    assert priced.cost_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)


def test_a_routed_request_is_priced_against_the_model_it_was_sent_to():
    """A switch makes the two figures different, and the difference is the saving
    or the extra spend the switch bought.

    The record here is test-high's answer to a request the router sent to test-high
    after being asked for test-mid, whose input rate is three times test-high's and
    whose output rate is five times it.
    """
    record = parsed([json_response(model=MODEL_HIGH, cached=CACHED_TOKENS)])
    priced = estimate_cost(record, MODEL_HIGH, MODEL_MID, priced_config())

    assert priced.cost_usd == pytest.approx((120 * 1.0 + 45 * 3.0 + 800 * 0.30) / 1_000_000)
    assert priced.baseline_cost_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)
    assert priced.cost_usd < priced.baseline_cost_usd
    assert not priced.notes


def test_the_saving_a_switch_bought_is_recorded_end_to_end(tmp_path, monkeypatch):
    """The same comparison through the proxy, on a real decision.

    The client asked for test-mid and the router sent it to test-high, so the row
    carries both figures: what the chosen model cost, and what the model the client
    named would have cost. The difference is the point of the whole table.
    """
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(
        priced_config(), log, json_body=json_response(model=MODEL_HIGH)
    ) as harness:
        request(harness.proxy_port, repeat_tool_body(model=MODEL_MID))
        assert wait_for_usage(log) == 1
        assert harness.state.requests

    assert json.loads(harness.state.requests[-1].body)["model"] == MODEL_HIGH

    row = usage_row(log)
    assert row.status == STATUS_OK
    assert row.cost_usd == pytest.approx((120 * 1.0 + 45 * 3.0 + 800 * 0.30) / 1_000_000)
    assert row.baseline_cost_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)
    assert row.cost_usd < row.baseline_cost_usd
    # The response named the model the router chose, so nothing needs noting.
    assert not row.notes


def test_a_cache_read_is_priced_at_the_cache_rate_and_not_the_input_rate():
    """The whole reason the total is split on the way in.

    Priced as one input figure the 800 cached tokens would be billed at 3.0 per
    million; split as they are, they are billed at 0.30. The difference is asserted
    rather than described, because this is the arithmetic that would be wrong.
    """
    record = parsed([json_response(cached=CACHED_TOKENS)])
    priced = estimate_cost(record, MODEL_MID, MODEL_MID, config())

    assert priced is not None
    as_one_figure = (920 * 3.0 + 45 * 15.0) / 1_000_000
    assert priced.cost_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)
    assert priced.cost_usd == pytest.approx(as_one_figure - 800 * 2.70 / 1_000_000)


def test_an_unknown_row_costs_nothing_rather_than_nothing_known(tmp_path, monkeypatch):
    """UNKNOWN is not 0.0. The report must show a row it cannot price as unknown,
    so a total over the rows that can be priced is not understated by one that
    cannot."""
    log = log_at(tmp_path, monkeypatch)
    unpriceable = json.dumps({"id": "chatcmpl-mock", "model": MODEL_MID, "choices": []}).encode()

    with running_proxy(config(), log) as harness:
        harness.respond(unpriceable)
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    row = usage_row(log)
    assert row.status == STATUS_UNKNOWN
    assert row.cost_usd is None
    assert row.baseline_cost_usd is None

    priced = estimate_cost(UsageRecord(status=STATUS_UNKNOWN), MODEL_MID, MODEL_MID, config())
    assert priced is not None
    assert priced.cost_usd is None


# --- the reports -----------------------------------------------------------


def test_the_cost_report_counts_and_totals_a_chat_completions_row(tmp_path, monkeypatch):
    """`tamias-router cost` reads `router_usage`, so a chat-completions row has to
    appear in it with the figures the table already knows how to display."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    report = build_cost_report(log.db_path)
    text = format_cost_report(report)

    assert report.rows == 1
    assert report.unknown == 0
    assert report.totals.requests == 1
    assert report.totals.input_tokens == 120
    assert report.totals.output_tokens == 45
    assert report.totals.cache_read_tokens == 800
    assert report.totals.cache_write_tokens == 0
    assert report.totals.cost_known_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)
    assert MODEL_MID in text
    # Displayed to six places, like every other figure in the report.
    assert f"{report.totals.cost_known_usd:.6f}" in text


def test_the_cost_report_totals_two_formats_together(tmp_path, monkeypatch):
    """The table is format-blind by design: an Anthropic row and a chat-completions
    row are two requests in one total, and neither the schema nor the report can
    tell which was which."""
    log = log_at(tmp_path, monkeypatch)
    log.record_usage(
        UsageRow(
            status=STATUS_OK,
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=5,
            cache_write_tokens=1,
            cost_usd=0.0,
            baseline_cost_usd=0.0,
        )
    )

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log, 2) == 2

    report = build_cost_report(log.db_path)

    assert report.rows == 2
    assert report.totals.input_tokens == 130
    assert report.totals.output_tokens == 65
    assert report.totals.cache_read_tokens == 805
    assert report.totals.cache_write_tokens == 1
    assert report.totals.cost_known_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)


def test_the_cost_report_leaves_an_unpriceable_row_out_of_the_known_total_and_says_so(tmp_path, monkeypatch):
    """A row with counts but no price is counted, not totalled. `cost_known_usd` is
    the known part of the money and `unknown` is how many rows are not in it."""
    log = log_at(tmp_path, monkeypatch)
    unpriceable = json.dumps({"id": "chatcmpl-mock", "model": MODEL_MID, "choices": []}).encode()

    with running_proxy(config(), log) as harness:
        harness.respond(unpriceable)
        request(harness.proxy_port, stay_body(text="unpriceable one"))
        assert wait_for_usage(log) == 1

    report = build_cost_report(log.db_path)
    text = format_cost_report(report)

    assert report.rows == 1
    assert report.unknown == 1
    assert report.totals.cost_unknown == 1
    assert report.totals.cost_known_usd == 0.0
    assert "UNKNOWN" in text


def watch_lines(db_path: Path) -> list[str]:
    """One `watch` poll, rendered. Read-only: it opens `mode=ro` and creates
    nothing, so it can be pointed at a database the proxy is still writing."""
    return poll_once(db_path, WatchOptions(once=True))


def test_watch_shows_a_chat_completions_row_with_its_chosen_model(tmp_path, monkeypatch):
    """`watch` reads `router_usage` as a left join onto the decisions, so a
    chat-completions request is one line like any other: the model it went to, and
    the money it cost."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    lines = watch_lines(log.db_path)
    text = "\n".join(lines)

    assert f"{MODEL_MID} -> {MODEL_MID}" in text
    assert f"{(120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000:.6f}" in text


def test_watch_shows_the_split_counts(tmp_path, monkeypatch):
    """The split is what makes the line worth reading: 120 in, 800 from cache, 45
    out, rather than one 920 that could not be priced."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    row = log.usage_rows(1)[0]
    text = "\n".join(watch_lines(log.db_path))

    assert row.input_tokens == 120
    assert row.cache_read_tokens == 800
    assert f"{row.input_tokens}" in text


def test_watch_leaves_an_unpriced_row_as_unknown_rather_than_zero(tmp_path, monkeypatch):
    """A row nobody has a price for is counted beside the money and added to
    neither. `watch` keeps `UNKNOWN` and `pending` apart; this asserts the first."""
    log = log_at(tmp_path, monkeypatch)
    unpriceable = json.dumps({"id": "chatcmpl-mock", "model": MODEL_MID, "choices": []}).encode()

    with running_proxy(config(), log) as harness:
        harness.respond(unpriceable)
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    text = "\n".join(watch_lines(log.db_path))

    assert "UNKNOWN" in text
    assert "free" not in text


def test_watch_counts_an_unpriced_row_beside_a_priced_one(tmp_path, monkeypatch):
    """Two requests, one priced and one not. The footer is the known part of the
    money plus how many rows are not in it, so an unpriced row can never quietly
    become a 0.0 that makes the total look complete."""
    log = log_at(tmp_path, monkeypatch)
    unpriceable = json.dumps({"id": "chatcmpl-mock", "model": MODEL_MID, "choices": []}).encode()

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, stay_body(text="priced"))
        assert wait_for_usage(log) == 1
        harness.respond(unpriceable)
        request(harness.proxy_port, stay_body(text="not priced"))
        assert wait_for_usage(log, 2) == 2

    text = "\n".join(watch_lines(log.db_path))

    assert f"{(120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000:.6f}" in text
    assert "UNKNOWN" in text


def test_watch_reads_nothing_and_writes_nothing(tmp_path, monkeypatch):
    """`watch` is read-only. Pointing it at the database the proxy just wrote must
    leave that file byte-identical, or a viewer would be able to change the
    ledger it is only allowed to look at."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    before = database_bytes(log.db_path)
    watch_lines(log.db_path)
    after = database_bytes(log.db_path)

    assert before == after
    assert log.usage_count() == 1


def test_watch_cannot_see_the_response_text(tmp_path, monkeypatch):
    """The phrase is in the response body the proxy relayed, and `watch` and `cost`
    render the same table. Nothing either of them prints may contain it."""
    log = log_at(tmp_path, monkeypatch)
    phrase = SECRET_PHRASE
    body = json_response(model=MODEL_MID).replace(b'"ok"', json.dumps(phrase).encode())

    with running_proxy(config(), log, json_body=body) as harness:
        _, _, payload = request(harness.proxy_port, stay_body())
        assert wait_for_usage(log) == 1

    # The client got it: the router forwards, it does not filter.
    assert phrase.encode() in payload
    assert phrase not in "\n".join(watch_lines(log.db_path))
    assert phrase not in format_cost_report(build_cost_report(log.db_path))


# --- Rule 1 -----------------------------------------------------------------


def test_a_phrase_in_the_streamed_response_never_reaches_the_database(tmp_path, monkeypatch):
    """Rule 1, on the path where it is hardest: the phrase arrives in the response,
    spread across the chunks the router reads to count them.

    The assertion is over the bytes of the database file and any journal beside it,
    not over a query, because a value that was written and never deleted would still
    be in the file.
    """
    log = log_at(tmp_path, monkeypatch)
    phrase = SECRET_PHRASE.encode()
    chunks = [
        json.dumps(
            {
                "model": MODEL_MID,
                "choices": [{"delta": {"content": phrase.decode()}, "finish_reason": None}],
            }
        ).encode(),
        json.dumps(
            {
                "model": MODEL_MID,
                "choices": [{"delta": {"content": " and " + phrase.decode()}, "finish_reason": "stop"}],
            }
        ).encode(),
        json.dumps({"model": MODEL_MID, "choices": [], "usage": usage_object(cached=CACHED_TOKENS)}).encode(),
    ]
    streamed = [b"data: " + chunk + b"\n\n" for chunk in chunks] + [b"data: [DONE]\n\n"]

    with running_proxy(config(), log, stream_body=streamed) as harness:
        _, _, payload = request(
            harness.proxy_port,
            json.dumps({"model": MODEL_MID, "stream": True, "messages": [user_message("go")]}).encode(),
        )
        assert wait_for_usage(log) == 1

    # The client did get it: the router forwards, it does not filter.
    assert phrase in payload
    # The counts were read anyway.
    assert usage_row(log).input_tokens == 120
    # And nowhere in the database.
    assert phrase not in database_bytes(log.db_path)


def test_a_phrase_in_the_request_prompt_never_reaches_the_database(tmp_path, monkeypatch, capsys):
    """The same for the request side, through the classifier's keyword match.

    "refactor" is a strong keyword, so the classifier reads this prompt and scores
    it. Reading a prompt is not keeping it.
    """
    log = log_at(tmp_path, monkeypatch)
    phrase = SECRET_PHRASE.encode()
    body = json.dumps(
        {
            "model": MODEL_MID,
            "messages": [user_message(f"refactor {phrase.decode()} and check the tests")],
        }
    ).encode()

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, body)
        assert wait_for_usage(log) == 1

    assert phrase in harness.state.requests[-1].body
    assert phrase not in database_bytes(log.db_path)
    assert phrase not in capsys.readouterr().err.encode()


def test_the_whole_body_of_a_chat_completions_response_is_not_stored(tmp_path, monkeypatch):
    """Not the assistant text, not the id, not the tool arguments.

    Every column the row has is asserted to be a number, a model id, a status or a
    note - which is what `router_usage` is. This is the schema-level statement of
    Rule 1 for this format.
    """
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(config(), log) as harness:
        request(harness.proxy_port, stay_body(text=SECRET_PHRASE))
        assert wait_for_usage(log) == 1

    row = usage_rows_via_sqlite(log.db_path)[0]
    for column, value in dict(row).items():
        assert value is None or isinstance(value, (int, float, str)), column
        assert not (isinstance(value, str) and SECRET_PHRASE in value), column

    assert SECRET_PHRASE.encode() not in database_bytes(log.db_path)


# --- ids with a slash and a colon ------------------------------------------


def vendor_config() -> RouterConfig:
    """The sample with vendor-style ids: a slash and a colon in every id.

    A colon and a slash are ordinary characters in the ids real gateways serve -
    `openrouter/vendor/model:free` is the shape - and they are also the two
    characters most likely to be mishandled on the way into a database or a report.
    """
    base = config()
    remap = {MODEL_LOW: "vendor/model-z:free", MODEL_MID: VENDOR_MID, MODEL_HIGH: VENDOR_HIGH}
    models = tuple(
        replace(spec, id=remap[spec.id]) if spec.id in remap else spec for spec in base.models
    )
    return replace(
        base,
        models=models,
        default_model=VENDOR_MID,
        default_effort=next(spec.legal_efforts[1] for spec in models if spec.id == VENDOR_MID),
    )


def test_a_vendor_id_survives_the_whole_hop_end_to_end(tmp_path, monkeypatch):
    """Slash and colon from the request, through the config, into the decision row,
    the usage row, the cost report and watch."""
    log = log_at(tmp_path, monkeypatch)
    settings_config = vendor_config()

    with running_proxy(settings_config, log, json_body=json_response(model=VENDOR_MID)) as harness:
        request(harness.proxy_port, stay_body(model=VENDOR_MID))
        assert wait_for_usage(log) == 1

    assert requested_models(log) == [VENDOR_MID]
    assert chosen_models(log) == [VENDOR_MID]

    row = usage_row(log)
    assert row.model_reported == VENDOR_MID
    assert row.status == STATUS_OK
    assert row.input_tokens == 120

    report = build_cost_report(log.db_path)
    assert report.totals.cost_known_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)
    assert VENDOR_MID in format_cost_report(report)
    assert VENDOR_MID in "\n".join(watch_lines(log.db_path))


def test_a_vendor_id_survives_a_switch_and_the_rewrite(tmp_path, monkeypatch):
    """The id also crosses the rewrite, where it is a string spliced into a body.

    A slash in a path is a path separator and a colon is a port, so an id like this
    would break any code that assumed an id is a bare word.
    """
    log = log_at(tmp_path, monkeypatch)
    settings_config = vendor_config()

    with running_proxy(settings_config, log, json_body=json_response(model=VENDOR_HIGH)) as harness:
        request(harness.proxy_port, repeat_tool_body(model=VENDOR_MID))
        assert wait_for_usage(log) == 1
        assert harness.state.requests

    forwarded = json.loads(harness.state.requests[-1].body)
    assert forwarded["model"] == VENDOR_HIGH
    assert applied_switch_count(log) == 1
    assert chosen_models(log) == [VENDOR_HIGH]

    row = usage_row(log)
    assert row.status == STATUS_OK
    assert row.model_reported == VENDOR_HIGH
    # The rewrite happened, so the row prices the chosen model against the model
    # the client named - two different vendor ids, both carried whole.
    assert row.cost_usd == pytest.approx((120 * 3.0 + 45 * 15.0 + 800 * 0.30) / 1_000_000)
    assert row.baseline_cost_usd == pytest.approx(row.cost_usd)
    assert "MODEL_MISMATCH" not in (row.notes or "")


def test_a_vendor_id_in_a_stream_is_recorded_whole(tmp_path, monkeypatch):
    """The model id arrives in the first chunk of a stream, and a parser that
    treated it as anything but a string would mangle it."""
    log = log_at(tmp_path, monkeypatch)
    settings_config = vendor_config()

    with running_proxy(
        settings_config, log, stream_body=stream_chunks(model=VENDOR_MID)
    ) as harness:
        request(
            harness.proxy_port,
            json.dumps({"model": VENDOR_MID, "stream": True, "messages": [user_message("go")]}).encode(),
        )
        assert wait_for_usage(log) == 1

    row = usage_row(log)
    assert row.model_reported == VENDOR_MID
    assert row.status == STATUS_OK
    assert row.input_tokens == 120


# --- the fake upstream ------------------------------------------------------


def test_the_mock_upstreams_openai_chunks_are_the_shape_this_file_reads():
    """The route the manual walkthrough uses answers with the same shape of stream
    the tests here assert on, so the manual and the tests cannot drift apart."""
    from tools.mock_upstream import CHAT_STREAM_CHUNKS

    record = parsed([bytes(chunk) for chunk in CHAT_STREAM_CHUNKS], content_type="text/event-stream")

    assert record.status == STATUS_OK
    assert record.input_tokens == 120
    assert record.cache_read_tokens == 0
    assert record.output_tokens == 45
    assert len(CHAT_STREAM_CHUNKS) == 4
    assert CHAT_STREAM_CHUNKS[-1].strip() == b"data: [DONE]"


def test_the_mock_upstreams_openai_usage_totals_line_up():
    """`total_tokens` in the mock is the sum of the two counts it reports, so a
    provider's own arithmetic and the router's agree.

    The mock serves no cache, so its `cached_tokens` is 0 and its `prompt_tokens`
    is the whole prompt. That is the zero case of the normalization, and it is why
    the end-to-end tests can use it: `input_tokens` comes out as `prompt_tokens`.
    """
    from tools.mock_upstream import MOCK_OUTPUT_TOKENS, MOCK_PROMPT_TOKENS, _chat_usage_payload

    usage = _chat_usage_payload()

    assert usage["prompt_tokens"] == MOCK_PROMPT_TOKENS
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}
    assert usage["total_tokens"] == MOCK_PROMPT_TOKENS + MOCK_OUTPUT_TOKENS
    # And the router reads it back as the prompt it was told about.
    assert parsed([json.dumps({"model": "mock-model", "usage": usage}).encode()]) == UsageRecord(
        model_reported="mock-model",
        input_tokens=MOCK_PROMPT_TOKENS,
        output_tokens=MOCK_OUTPUT_TOKENS,
        cache_read_tokens=0,
        cache_write_tokens=None,
        status=STATUS_OK,
    )
