"""Tests for OpenAI chat-completions support.

A client that posts to `/chat/completions` gets the same treatment a client that
posts to `/v1/messages` gets: the same signals, classifier, policy, safety layer,
hold, decision log and model rewrite. The only thing that differs is which reader
turns the body into signals, and the path is the only thing that decides that.

The tests here fall into three groups:

* the pure functions - format detection, the OpenAI signal reader, and the
  classifier over an OpenAI body;
* the proxy, driven through the same in-process harness the other proxy tests use,
  with a recording fake upstream on 127.0.0.1 and a temporary database;
* the mock upstream's new route.

Config comes from `tools/sample-config-OPENAI-test.yaml`, a copy of the CLASSIFIER
sample with `TEST VALUES ONLY, NOT REAL MODELS OR PRICES` in its first line. The
shipped config keeps its `TODO_` ids, its null prices and `mode: shadow`, so
nothing here can change a request a real install would send.

Rule 1 is asserted directly: no test sends prompt text that the router must not
keep, and two of them check that the phrase never reaches the database or a log
line. Rule 7 is asserted too: a rewrite only ever targets a model the config
lists, and one test drives the whole path with vendor-style ids containing `/`
and `:`.
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import threading
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from router.classifier import HIGH, LOW, classify_prompt, latest_prompt_text
from router.config import RouterConfig, load_config
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog, read_request_metadata
from router.hold import HELD_SIGNAL_KEY, REASON_HELD_MODEL
from router.killswitch import KILL_SWITCH_ENV, REASON_KILL_SWITCH
from router.proxy import (
    API_FORMAT_ANTHROPIC,
    API_FORMAT_OPENAI,
    API_FORMAT_SIGNAL_KEY,
    LOOPBACK_HOST,
    REWRITE_SKIPPED_NOT_JSON,
    ROUTED_HEADER,
    ProxySettings,
    create_server,
    detect_api_format,
)
from router.safety import is_blocked
from router.signals import Signals, compute_signals
from router.signals_openai import compute_signals_openai
from router.state import SessionStore

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENAI_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-OPENAI-test.yaml"

MODEL_LOW = "test-low"
MODEL_MID = "test-mid"
MODEL_HIGH = "test-high"

#: Vendor-style ids: a slash and a colon are ordinary characters in the ids real
#: gateways serve, so both ends of a hop have to carry them verbatim.
VENDOR_MID = "vendor/model-a:free"
VENDOR_HIGH = "vendor/model-b:free"

#: The routes this format is reached through.
OPENAI_BARE_PATH = "/chat/completions"
OPENAI_VERSIONED_PATH = "/v1/chat/completions"

#: A phrase that must never reach the database or any log line.
SECRET_PHRASE = "citrine-marmalade-9052"

CLIENT_TIMEOUT = 5.0

#: Dummy write 3.75, dummy read 0.30, dummy benefit 0.001: the benefit covers a
#: rebuild below 0.001 * 1_000_000 / 3.45 = 289 tokens of context. The bodies here
#: are deliberately small so a switch is affordable and is never quietly blocked
#: on cost instead.
CROSSOVER_BYTES = 289 * 4

#: One session for every request in the hold test: the session hint is a salted
#: digest of the first user message, so that text must not change.
SESSION_TEXT = "one long openai session"

#: A reply carrying counts, so the two usage tests differ only in the route they
#: post to. Invented numbers, and never read by any test as a real price.
USAGE_RESPONSE = json.dumps(
    {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": MODEL_MID,
        "content": [{"type": "text", "text": "ok"}],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 45,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 200,
        },
    }
).encode()


def openai_config() -> RouterConfig:
    return load_config(OPENAI_CONFIG_PATH)


def shadow_config() -> RouterConfig:
    return replace(openai_config(), mode="shadow")


def no_classifier(config: RouterConfig) -> RouterConfig:
    """The same config with the classifier switched off, to isolate other rules."""
    assert config.classifier is not None
    return replace(config, classifier=replace(config.classifier, enabled=False))


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


def repeat_tool_body(model: str = MODEL_MID, text: str = "go") -> bytes:
    """A body that escalates: one tool called three times in a row.

    This is the escalation the OpenAI format can express. It ends on a tool
    result, so nothing is pending and the switch is not blocked as mid tool call,
    and `block_in_tool_loop` is false in this config.
    """
    body = json.dumps(
        {
            "model": model,
            "max_completion_tokens": 16,
            "messages": [
                user_message(text),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("search", 1), tool_call("search", 2),
                                   tool_call("search", 3)],
                },
                tool_result(1),
                tool_result(2),
                tool_result(3),
            ],
        }
    ).encode()
    assert len(body) < CROSSOVER_BYTES, "repeat_tool_body must stay cheap enough to be allowed"
    return body


def classifier_switch_body(model: str = MODEL_MID, phrase: str = SECRET_PHRASE) -> bytes:
    """A body the classifier escalates on: a hard prompt in text parts.

    "refactor" is a strong keyword and the prompt carries none of the cheap ones,
    so the score clears `strong_from` and the tier is high.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "be brief"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"refactor {phrase} and check the tests"}
                    ],
                },
            ],
        }
    ).encode()
    assert len(body) < CROSSOVER_BYTES, "classifier_switch_body must stay cheap enough"
    return body


def stay_body(model: str = MODEL_MID, text: str = SESSION_TEXT) -> bytes:
    """A body nothing acts on: no repeat, one turn, a prompt that suits test-mid.

    "one long openai session" holds no strong or cheap keyword, so it scores the
    base and lands in `mid`, which is the tier of the requested model: the
    classifier agrees and the policy has nothing else to propose.
    """
    return json.dumps({"model": model, "messages": [user_message(text)]}).encode()


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


class RecordingUpstream(BaseHTTPRequestHandler):
    """Records what arrived and replies with `server.response`."""

    protocol_version = "HTTP/1.1"
    server_version = "recording-upstream"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle(self) -> None:
        raw_length = self.headers.get("Content-Length")
        length = int(raw_length) if raw_length else 0
        body = self.rfile.read(length) if length else b""
        self.server.recorded.append(  # type: ignore[attr-defined]
            Recorded(
                method=self.command,
                path=self.path,
                headers={name.lower(): value for name, value in self.headers.items()},
                body=body,
            )
        )
        payload = self.server.response  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    do_GET = _handle
    do_POST = _handle


@dataclass
class Harness:
    proxy_port: int
    recorded: list[Recorded]
    store: SessionStore
    #: Swap the fake upstream's reply, for the tests that need a body with counts
    #: in it.
    respond: Any

    @property
    def forwarded(self) -> Recorded:
        assert self.recorded, "the upstream received nothing"
        return self.recorded[-1]


@contextlib.contextmanager
def running_proxy(config: RouterConfig, db_path: Path) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), RecordingUpstream)
    upstream.daemon_threads = True
    upstream.recorded = []  # type: ignore[attr-defined]
    upstream.response = b'{"ok":true}'  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    store = SessionStore()
    server = create_server(
        ProxySettings(
            listen_port=0,
            upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
            mode=config.mode,
            decisions=DecisionLog(db_path),
            config=config,
            state=store,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Harness(
            proxy_port=server.server_address[1],
            recorded=upstream.recorded,  # type: ignore[attr-defined]
            store=store,
            respond=lambda payload: setattr(upstream, "response", payload),
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
    """One request; returns `(status, headers, body)` with headers lowercased."""
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


def wait_for_rows(log: DecisionLog, expected: int, timeout: float = 5.0) -> int:
    import time

    deadline = time.monotonic() + timeout
    count = log.count()
    while count < expected and time.monotonic() < deadline:
        time.sleep(0.01)
        count = log.count()
    return count


def log_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DecisionLog:
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    return DecisionLog.from_env()


# --- the sample config ------------------------------------------------------


def test_the_sample_config_says_it_is_test_data():
    first = OPENAI_CONFIG_PATH.read_text(encoding="utf-8").splitlines()[0]

    assert first == "# TEST VALUES ONLY, NOT REAL MODELS OR PRICES"


def test_the_sample_config_is_the_classifier_sample_with_a_different_header():
    """Every configured value is the CLASSIFIER sample's, so a test that passes
    against one is testing the shipped rules rather than a second set of them.

    `upstream` is the only value that differs, and it differs because this file
    exists to send a chat-completions body somewhere: naming the Anthropic host
    in a chat sample would describe a request that could not succeed. No test
    reads it - each one points the proxy at its own local fake.
    """
    classifier_lines = [
        line.strip()
        for line in (REPO_ROOT / "tools" / "sample-config-CLASSIFIER-test.yaml")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    openai_lines = [
        line.strip()
        for line in OPENAI_CONFIG_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert len(openai_lines) == len(classifier_lines)
    for mine, theirs in zip(openai_lines, classifier_lines):
        if theirs.startswith("upstream:"):
            assert mine.startswith("upstream:")
        else:
            assert mine == theirs, f"{mine!r} is not the sample's {theirs!r}"


def test_the_sample_config_activates_the_classifier_in_active_mode():
    config = openai_config()

    assert config.mode == "active"
    assert config.classifier is not None and config.classifier.enabled is True


# --- format detection -------------------------------------------------------


def test_a_path_ending_in_v1_messages_is_the_anthropic_format():
    assert detect_api_format(MESSAGES_PATH) == API_FORMAT_ANTHROPIC


@pytest.mark.parametrize(
    "path",
    [
        OPENAI_BARE_PATH,
        OPENAI_VERSIONED_PATH,
        "/openai/v1/chat/completions",
        "/api/openai/chat/completions",
    ],
)
def test_a_path_ending_in_chat_completions_is_the_openai_format(path: str):
    assert detect_api_format(path) == API_FORMAT_OPENAI


@pytest.mark.parametrize(
    "path",
    [
        "/v1/messages?beta=true",
        "/v1/chat/completions?api-version=2024-08-01",
        "/chat/completions?model=vendor/model-a:free",
    ],
)
def test_the_query_string_cannot_change_the_format(path: str):
    """A client-controlled query must never decide what the router reads."""
    assert detect_api_format(path) == detect_api_format(path.split("?", 1)[0])


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/",
        "/v1/embeddings",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages/",
        "/chat/completions/extra",
        "/messages",
    ],
)
def test_a_path_that_names_no_format_names_no_format(path: str):
    assert detect_api_format(path) is None


# --- the signal reader ------------------------------------------------------


def test_an_ordinary_chat_request_scores_the_way_an_ordinary_request_does():
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message("first"),
            {"role": "assistant", "content": "answer"},
            user_message("second"),
        ],
    }

    signals = compute_signals_openai(body)

    assert signals.message_count == 3
    assert signals.turn_index == 1
    assert signals.tool_result_count == 0
    assert signals.context_tokens_estimate == len(json.dumps(body).encode()) // 4


def test_the_two_readers_agree_on_the_size_heuristic():
    body = {"model": MODEL_MID, "messages": [user_message("hello")]}

    assert (
        compute_signals_openai(body).context_tokens_estimate
        == compute_signals(body).context_tokens_estimate
    )


def test_tool_results_are_counted_from_the_tool_messages():
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message("go"),
            {"role": "assistant", "tool_calls": [tool_call("search", 1)]},
            tool_result(1),
            tool_result(2),
        ],
    }

    signals = compute_signals_openai(body)

    assert signals.tool_result_count == 2
    assert signals.message_count == 4


def test_the_tool_error_signals_are_none_so_tool_error_escalation_cannot_fire():
    """The format has no error flag, and reading tool output would be Rule 1."""
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message("go"),
            {"role": "assistant", "tool_calls": [tool_call("search", 1), tool_call("search", 2)]},
            tool_result(1, "Error: it failed"),
            tool_result(2, "Error: it failed again"),
        ],
    }

    signals = compute_signals_openai(body)

    assert signals.last_tool_result_is_error is None
    assert signals.consecutive_tool_errors is None


def test_thinking_is_undetermined_rather_than_off():
    """None means "this format cannot say", which is not the same as False."""
    assert compute_signals_openai({"messages": [user_message("hi")]}).has_thinking_enabled is None


def test_repeated_tool_calls_are_counted_by_function_name():
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message("go"),
            {
                "role": "assistant",
                "tool_calls": [
                    tool_call("search", 1),
                    tool_call("search", 2),
                    tool_call("read", 3),
                    tool_call("read", 4),
                    tool_call("read", 5),
                ],
            },
            tool_result(1),
        ],
    }

    assert compute_signals_openai(body).repeated_tool_call_count == 3


def test_a_run_of_calls_is_counted_across_several_assistant_messages():
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message("go"),
            {"role": "assistant", "tool_calls": [tool_call("search", 1)]},
            tool_result(1),
            {"role": "assistant", "tool_calls": [tool_call("search", 2)]},
            tool_result(2),
        ],
    }

    assert compute_signals_openai(body).repeated_tool_call_count == 2


def test_two_different_tools_in_a_row_are_not_a_repeat():
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message("go"),
            {"role": "assistant", "tool_calls": [tool_call("search", 1), tool_call("read", 2)]},
            tool_result(1),
        ],
    }

    assert compute_signals_openai(body).repeated_tool_call_count == 1


def test_a_tool_call_with_no_name_ends_a_run_instead_of_counting():
    """Names run search, unknown, search: the last run of `search` is one call."""
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message("go"),
            {
                "role": "assistant",
                "tool_calls": [
                    tool_call("search", 1),
                    {"type": "function"},
                    tool_call("search", 2),
                ],
            },
        ],
    }

    assert compute_signals_openai(body).repeated_tool_call_count == 1


def test_a_conversation_ending_on_tool_calls_has_one_pending():
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message("go"),
            {"role": "assistant", "tool_calls": [tool_call("search", 1)]},
        ],
    }

    signals = compute_signals_openai(body)

    assert signals.tool_use_pending is True
    assert signals.in_tool_loop is False


def test_a_conversation_ending_on_a_tool_result_is_inside_a_tool_loop():
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message("go"),
            {"role": "assistant", "tool_calls": [tool_call("search", 1)]},
            tool_result(1),
        ],
    }

    signals = compute_signals_openai(body)

    assert signals.tool_use_pending is False
    assert signals.in_tool_loop is True


def test_an_ordinary_answer_is_neither_pending_nor_in_a_loop():
    signals = compute_signals_openai(
        {
            "model": MODEL_MID,
            "messages": [user_message("hi"), {"role": "assistant", "content": "yo"}],
        }
    )

    assert signals.tool_use_pending is False
    assert signals.in_tool_loop is False


def test_no_messages_leaves_the_tool_state_undetermined():
    """An empty list has no final message, which is not the same as a clean one."""
    signals = compute_signals_openai({"model": MODEL_MID, "messages": []})

    assert signals.message_count == 0
    assert signals.tool_use_pending is None
    assert signals.in_tool_loop is None


def test_the_effort_is_read_from_reasoning_effort():
    signals = compute_signals_openai({"reasoning_effort": "high"})

    assert signals.requested_effort_if_present == "high"


def test_the_effort_is_read_from_a_nested_reasoning_effort():
    body = {"reasoning": {"effort": "low", "summary": "auto"}}

    assert compute_signals_openai(body).requested_effort_if_present == "low"


def test_the_top_level_effort_wins_over_the_nested_one():
    body = {"reasoning_effort": "high", "reasoning": {"effort": "low"}}

    assert compute_signals_openai(body).requested_effort_if_present == "high"


def test_an_effort_that_is_not_a_string_is_ignored_rather_than_coerced():
    assert compute_signals_openai({"reasoning_effort": 3}).requested_effort_if_present is None
    assert compute_signals_openai({"reasoning": "high"}).requested_effort_if_present is None

    nested = compute_signals_openai({"reasoning": {"effort": None}})
    assert nested.requested_effort_if_present is None


def test_no_effort_field_means_no_effort_signal():
    assert compute_signals_openai({"messages": []}).requested_effort_if_present is None


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        "text",
        7,
        3.5,
        {"messages": "x"},
        {"messages": {"role": "user"}},
        {"messages": [None, 5, "x"]},
        {"messages": [{"role": "assistant", "tool_calls": "no"}]},
        {"messages": [{"role": "assistant", "tool_calls": [None, {}, {"function": 5}]}]},
        {"messages": [{"role": "user", "content": {"nested": "object"}}]},
        {"messages": [{"role": "tool", "tool_calls": 7}]},
        {"reasoning": 5},
        {"reasoning": []},
        {"reasoning_effort": {"nested": "object"}},
    ],
)
def test_no_openai_body_shape_can_make_the_reader_raise(body: Any):
    signals = compute_signals_openai(body)

    assert isinstance(signals, Signals)
    assert signals.last_tool_result_is_error is None
    assert signals.consecutive_tool_errors is None


def test_a_malformed_message_is_counted_and_skipped_rather_than_raised():
    signals = compute_signals_openai({"messages": [None, 5, "x", user_message("hi")]})

    assert signals.message_count == 4
    assert signals.turn_index == 0
    assert signals.tool_result_count == 0


def test_the_reader_never_returns_any_message_text():
    """Rule 1: a signal is a number, a boolean or an effort, never content."""
    body = {
        "model": MODEL_MID,
        "messages": [
            user_message(SECRET_PHRASE),
            {"role": "assistant", "content": f"the argument was {SECRET_PHRASE}"},
            {"role": "assistant", "tool_calls": [tool_call(SECRET_PHRASE, 1)]},
            tool_result(1, SECRET_PHRASE),
        ],
    }

    rendered = json.dumps(compute_signals_openai(body).to_dict())

    assert SECRET_PHRASE not in rendered
    for value in compute_signals_openai(body).to_dict().values():
        assert not isinstance(value, (list, dict))


def test_a_value_that_cannot_be_serialized_still_counts_as_zero_bytes():
    signals = compute_signals_openai({"messages": [], "bad": {1, 2, object()}})

    assert signals.context_tokens_estimate == 0
    assert signals.message_count == 0


# --- the classifier over an openai body --------------------------------------


def test_the_classifier_reads_the_latest_user_message_and_only_that_one():
    body = {
        "model": MODEL_MID,
        "messages": [
            {"role": "system", "content": "refactor debug architecture optimize"},
            {"role": "user", "content": "please refactor the parser"},
            {"role": "assistant", "content": "refactor debug architecture optimize"},
            {"role": "tool", "tool_call_id": "c1", "content": "refactor debug architecture"},
            {"role": "user", "content": "hello"},
        ],
    }

    task = classify_prompt(body, openai_config())

    assert task is not None
    assert task.tier == LOW, "only the last user message may be read"


def test_the_classifier_reads_text_parts():
    body = {
        "model": MODEL_MID,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "refactor this"}]}],
    }

    task = classify_prompt(body, openai_config())

    assert task is not None and task.tier == HIGH


def test_the_latest_user_message_wins_even_when_an_older_one_is_louder():
    body = {
        "model": MODEL_MID,
        "messages": [
            {"role": "user", "content": "refactor debug architecture optimize implement"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "just a typo in a comment"},
        ],
    }

    assert latest_prompt_text(body) == "just a typo in a comment"
    task = classify_prompt(body, openai_config())
    # `typo` and `comment` are both cheap words, and two of them take the score
    # from 50 to 10: the loud older prompt above would have scored 100.
    assert task is not None and task.tier == LOW


def test_an_openai_body_with_no_user_message_classifies_as_nothing():
    body = {
        "model": MODEL_MID,
        "messages": [{"role": "assistant", "content": "refactor debug architecture"}],
    }

    assert latest_prompt_text(body) is None
    assert classify_prompt(body, openai_config()) is None


# --- the session hint -------------------------------------------------------


def test_the_same_first_user_text_gives_the_same_session_in_either_format(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    anthropic = read_request_metadata(
        json.dumps({"model": MODEL_MID, "messages": [user_message(SESSION_TEXT)]}).encode()
    )
    openai = read_request_metadata(
        json.dumps(
            {
                "model": MODEL_MID,
                "messages": [
                    {"role": "system", "content": "ignored"},
                    {"role": "user", "content": [{"type": "text", "text": SESSION_TEXT}]},
                ],
            }
        ).encode()
    )

    assert log.session_hint(anthropic.first_user_text) == log.session_hint(
        openai.first_user_text
    )


def test_a_different_first_user_text_gives_a_different_session(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    assert log.session_hint(SESSION_TEXT) != log.session_hint(SESSION_TEXT + " and more")


def test_the_first_user_message_is_found_past_system_assistant_and_tool_messages():
    body = json.dumps(
        {
            "model": MODEL_MID,
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "assistant", "content": "assistant"},
                {"role": "tool", "tool_call_id": "c1", "content": "tool"},
                user_message(SESSION_TEXT),
                user_message("a later user message"),
            ],
        }
    ).encode()

    assert read_request_metadata(body).first_user_text == SESSION_TEXT


# --- the proxy --------------------------------------------------------------


def test_a_chat_completions_stay_forwards_the_original_bytes(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = stay_body()

    with running_proxy(openai_config(), log.db_path) as harness:
        status, headers, _ = request(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        assert ROUTED_HEADER.lower() not in headers

        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.chosen_model == MODEL_MID
        assert row.signal_values[API_FORMAT_SIGNAL_KEY] == API_FORMAT_OPENAI


def test_an_escalation_rewrites_only_the_model_and_applies_one(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = repeat_tool_body()

    with running_proxy(openai_config(), log.db_path) as harness:
        status, headers, payload = request(harness.proxy_port, body)
        assert status == 200
        assert json.loads(payload) == {"ok": True}
        assert wait_for_rows(log, 1) == 1

        forwarded = harness.forwarded
        sent = json.loads(forwarded.body)
        original = json.loads(body)
        assert sent["model"] == MODEL_HIGH
        assert {name: value for name, value in sent.items() if name != "model"} == {
            name: value for name, value in original.items() if name != "model"
        }
        assert sent["messages"] == original["messages"]
        assert forwarded.headers["content-length"] == str(len(forwarded.body))
        assert headers[ROUTED_HEADER.lower()] == f"{MODEL_MID}->{MODEL_HIGH}"

        row = log.recent(1)[0]
        assert row.action == "SWITCH"
        assert row.applied == 1
        assert row.requested_model == MODEL_MID
        assert row.chosen_model == MODEL_HIGH
        assert row.reason_codes == ["ESCALATE_REPEATED_TOOLS"]
        assert not is_blocked(row.reason_codes)


def test_the_row_records_the_signals_the_openai_reader_produced(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = repeat_tool_body()

    with running_proxy(openai_config(), log.db_path) as harness:
        request(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1

        signals = log.recent(1)[0].signal_values
        assert signals[API_FORMAT_SIGNAL_KEY] == API_FORMAT_OPENAI
        assert signals["tool_result_count"] == 3
        assert signals["repeated_tool_call_count"] == 3
        assert signals["turn_index"] == 1
        assert signals["consecutive_tool_errors"] is None
        assert signals["has_thinking_enabled"] is None


def test_shadow_mode_forwards_identical_bytes_and_applies_nothing(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = repeat_tool_body()

    with running_proxy(shadow_config(), log.db_path) as harness:
        status, headers, _ = request(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body, "shadow mode changed the bytes"
        assert ROUTED_HEADER.lower() not in headers

        row = log.recent(1)[0]
        assert row.mode == "shadow"
        assert row.action == "SWITCH"
        assert row.applied == 0


def test_the_classifier_can_escalate_an_openai_request(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = classifier_switch_body()

    with running_proxy(openai_config(), log.db_path) as harness:
        status, _, _ = request(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH

        row = log.recent(1)[0]
        assert row.action == "SWITCH"
        assert row.applied == 1
        assert row.reason_codes == ["CLASSIFIER_HIGH"]
        assert row.signal_values["classifier_tier"] == HIGH


def test_the_format_is_recorded_on_every_row(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    anthropic_body = json.dumps({"model": MODEL_MID, "max_tokens": 8, "messages": []}).encode()

    with running_proxy(openai_config(), log.db_path) as harness:
        request(harness.proxy_port, anthropic_body, path=MESSAGES_PATH)
        assert wait_for_rows(log, 1) == 1
        request(harness.proxy_port, repeat_tool_body())
        assert wait_for_rows(log, 2) == 2

        rows = log.recent(2)
        assert rows[0].signal_values[API_FORMAT_SIGNAL_KEY] == API_FORMAT_OPENAI
        assert rows[1].signal_values[API_FORMAT_SIGNAL_KEY] == API_FORMAT_ANTHROPIC


def test_the_hold_serves_the_same_model_on_the_next_request(tmp_path, monkeypatch):
    """The client keeps sending the model it configured; the hold keeps the hop."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(openai_config(), log.db_path) as harness:
        first_status, _, _ = request(harness.proxy_port, repeat_tool_body(text=SESSION_TEXT))
        assert first_status == 200
        assert wait_for_rows(log, 1) == 1

        second_status, _, _ = request(harness.proxy_port, stay_body(text=SESSION_TEXT))
        assert second_status == 200
        assert wait_for_rows(log, 2) == 2

        rows = log.recent(2)  # newest first: [0] is the held request
        assert rows[0].applied == 1
        assert rows[1].applied == 1
        assert rows[0].reason_codes == [REASON_HELD_MODEL]
        assert rows[0].chosen_model == MODEL_HIGH
        assert rows[0].signal_values[HELD_SIGNAL_KEY] == 1
        assert rows[1].session_hint == rows[0].session_hint
        assert [json.loads(record.body)["model"] for record in harness.recorded] == [
            MODEL_HIGH,
            MODEL_HIGH,
        ]


def test_the_kill_switch_passes_an_openai_request_through_unchanged(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    body = repeat_tool_body()

    with running_proxy(openai_config(), log.db_path) as harness:
        status, headers, _ = request(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        assert ROUTED_HEADER.lower() not in headers

        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.reason_codes == [REASON_KILL_SWITCH]
        assert row.mode == "off"
        assert row.signal_values[API_FORMAT_SIGNAL_KEY] == API_FORMAT_OPENAI


def test_a_blocked_switch_forwards_the_original_bytes(tmp_path, monkeypatch):
    """An expensive escalation is blocked on cost, so nothing is rewritten."""
    log = log_at(tmp_path, monkeypatch)
    config = replace(
        openai_config(),
        policy=replace(openai_config().policy, safety_margin_usd=0.0, dwell_requests=5),
    )
    body = json.dumps(
        {
            "model": MODEL_MID,
            "messages": [
                user_message("x" * 8_000),
                {
                    "role": "assistant",
                    "tool_calls": [tool_call("search", 1), tool_call("search", 2),
                                   tool_call("search", 3)],
                },
                tool_result(1),
            ],
        }
    ).encode()

    with running_proxy(config, log.db_path) as harness:
        status, headers, _ = request(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == body
        assert ROUTED_HEADER.lower() not in headers

        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert is_blocked(row.reason_codes)
        assert row.chosen_model == MODEL_MID


def test_the_downgrade_rule_cannot_fire_on_an_openai_request(tmp_path, monkeypatch):
    """It needs a known count of zero tool errors, and this format has none."""
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(no_classifier(openai_config()), log.db_path) as harness:
        status, _, _ = request(harness.proxy_port, stay_body(model=MODEL_HIGH))
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert "SIGNAL_UNKNOWN" in row.reason_codes
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH


def test_an_openai_path_records_a_row_even_when_the_body_is_not_json(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(openai_config(), log.db_path) as harness:
        status, _, _ = request(harness.proxy_port, b"this is not json")
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert harness.forwarded.body == b"this is not json"
        row = log.recent(1)[0]
        assert row.applied == 0
        assert row.error == REWRITE_SKIPPED_NOT_JSON
        assert row.signal_values[API_FORMAT_SIGNAL_KEY] == API_FORMAT_OPENAI


def test_an_openai_request_records_no_usage_row(tmp_path, monkeypatch):
    """Pricing stays Anthropic-only: no row, not a half-filled one.

    The fake upstream replies with a response that carries counts in it, so this
    is not the absence of something to record but the router declining to price
    a format it has no price sheet for.
    """
    log = log_at(tmp_path, monkeypatch)
    body = repeat_tool_body()

    with running_proxy(openai_config(), log.db_path) as harness:
        harness.respond(USAGE_RESPONSE)
        request(harness.proxy_port, body)
        assert wait_for_rows(log, 1) == 1

        assert log.usage_count() == 0
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH


def test_an_anthropic_request_still_records_its_usage_row(tmp_path, monkeypatch):
    """The contrast the test above needs: same proxy, same reply, other route."""
    log = log_at(tmp_path, monkeypatch)
    body = json.dumps({"model": MODEL_MID, "max_tokens": 8, "messages": []}).encode()

    with running_proxy(openai_config(), log.db_path) as harness:
        harness.respond(USAGE_RESPONSE)
        status, _, _ = request(harness.proxy_port, body, path=MESSAGES_PATH)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert log.usage_count() == 1
        stored = log.usage_rows(1)[0]
        assert stored.input_tokens == 120
        assert stored.output_tokens == 45
        assert stored.cache_read_tokens == 800
        assert stored.cache_write_tokens == 200


def test_an_unknown_path_is_forwarded_and_recorded_nowhere(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = repeat_tool_body()

    with running_proxy(openai_config(), log.db_path) as harness:
        status, _, _ = request(harness.proxy_port, body, path="/v1/embeddings")
        assert status == 200

        assert harness.forwarded.body == body
        assert log.count() == 0


def test_a_get_to_chat_completions_is_forwarded_and_recorded_nowhere(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(openai_config(), log.db_path) as harness:
        status, _, _ = request(harness.proxy_port, b"", path=OPENAI_VERSIONED_PATH, method="GET")
        assert status == 200

        assert log.count() == 0


def test_a_query_string_on_the_openai_route_still_routes(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(openai_config(), log.db_path) as harness:
        status, headers, _ = request(
            harness.proxy_port,
            repeat_tool_body(),
            path=f"{OPENAI_VERSIONED_PATH}?api-version=2024-08-01",
        )
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        assert headers[ROUTED_HEADER.lower()] == f"{MODEL_MID}->{MODEL_HIGH}"
        assert log.recent(1)[0].signal_values[API_FORMAT_SIGNAL_KEY] == API_FORMAT_OPENAI


def test_no_openai_request_text_reaches_the_database_or_the_log(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    body = classifier_switch_body()
    tool_body = repeat_tool_body()

    with running_proxy(openai_config(), log.db_path) as harness:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            request(harness.proxy_port, body)
            request(harness.proxy_port, tool_body)
            assert wait_for_rows(log, 2) == 2

        assert SECRET_PHRASE.encode() not in log.db_path.read_bytes()
        assert SECRET_PHRASE not in captured.getvalue()


# --- model ids that carry a slash and a colon -------------------------------


def vendor_config() -> RouterConfig:
    """The active config with the two ids a real gateway serves."""
    config = openai_config()
    prices = {spec.cost_tier: spec for spec in config.models}
    models = tuple(
        replace(
            prices[tier],
            id=model_id,
        )
        for tier, model_id in (
            ("low", "vendor/model-low:free"),
            ("mid", VENDOR_MID),
            ("high", VENDOR_HIGH),
        )
    )
    return replace(config, models=models, default_model=VENDOR_MID)


def test_config_validation_accepts_an_id_with_a_slash_and_a_colon(tmp_path):
    path = tmp_path / "vendor-ids.yaml"
    path.write_text(
        "\n".join(
            [
                "listen: 127.0.0.1:8787",
                "upstream: https://api.openai.com",
                "mode: shadow",
                "models:",
                f'  - id: "{VENDOR_MID}"',
                "    legal_efforts: [test-effort-medium]",
                "    cost_tier: mid",
                f"default_model: \"{VENDOR_MID}\"",
                "default_effort: test-effort-medium",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model_ids == (VENDOR_MID,)
    assert config.is_legal(VENDOR_MID, "test-effort-medium")


def test_a_vendor_id_survives_the_rewrite_the_header_and_the_log(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = vendor_config()
    body = repeat_tool_body(model=VENDOR_MID)

    with running_proxy(config, log.db_path) as harness:
        status, headers, _ = request(harness.proxy_port, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1

        assert json.loads(harness.forwarded.body)["model"] == VENDOR_HIGH
        assert headers[ROUTED_HEADER.lower()] == f"{VENDOR_MID}->{VENDOR_HIGH}"

        row = log.recent(1)[0]
        assert row.requested_model == VENDOR_MID
        assert row.chosen_model == VENDOR_HIGH
        assert row.applied == 1


def test_the_classifier_can_escalate_to_a_vendor_id(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)

    with running_proxy(vendor_config(), log.db_path) as harness:
        request(harness.proxy_port, classifier_switch_body(model=VENDOR_MID))
        assert wait_for_rows(log, 1) == 1

        row = log.recent(1)[0]
        assert row.reason_codes == ["CLASSIFIER_HIGH"]
        assert row.chosen_model == VENDOR_HIGH
        assert json.loads(harness.forwarded.body)["model"] == VENDOR_HIGH


# --- the mock upstream ------------------------------------------------------


@contextlib.contextmanager
def running_mock_upstream() -> Iterator[int]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mock_upstream", REPO_ROOT / "tools" / "mock_upstream.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    server = ThreadingHTTPServer((LOOPBACK_HOST, 0), module.MockUpstreamHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_the_mock_upstream_answers_the_chat_completions_route():
    with running_mock_upstream() as port:
        body = json.dumps(
            {
                "model": VENDOR_MID,
                "reasoning_effort": "high",
                "stream": False,
                "messages": [user_message(f"the phrase is {SECRET_PHRASE}")],
            }
        ).encode()
        status, _, payload = request(port, body, path=OPENAI_VERSIONED_PATH)

        assert status == 200
        report = json.loads(payload)
        assert report["ok"] is True
        assert report["model"] == VENDOR_MID
        assert report["effort"] == "high"
        assert report["messages"] == 1
        assert report["received_bytes"] == len(body)
        assert report["content_length"] == str(len(body))
        assert SECRET_PHRASE.encode() not in payload, "the mock echoed request content"


def test_the_mock_upstream_reads_a_nested_reasoning_effort():
    with running_mock_upstream() as port:
        body = json.dumps({"model": MODEL_MID, "reasoning": {"effort": "low"}}).encode()
        status, _, payload = request(port, body, path=OPENAI_VERSIONED_PATH)

        assert status == 200
        assert json.loads(payload)["effort"] == "low"


def test_the_mock_upstream_survives_a_chat_body_that_is_not_json():
    with running_mock_upstream() as port:
        status, _, payload = request(port, b"not json", path=OPENAI_VERSIONED_PATH)

        assert status == 200
        report = json.loads(payload)
        assert report["json"] is False
        assert report["received_bytes"] == 8
        for absent in ("model", "effort", "stream", "messages", "usage"):
            assert absent not in report, f"reported {absent} from a body it could not parse"
