"""Tests for the per-prompt hold: one human prompt, not the whole session.

The hold exists so that the request after an approved switch stays on the model
the router moved to, because the client never learns it moved. A hold that
lived for the whole session would also answer the client's NEXT human prompt -
a different task - with the model the previous task needed, which is exactly
what the policy exists to decide.

The rule these tests pin down: a hold protects one human prompt, meaning the
tool loop that prompt opens. The request's own `human_prompt_count` is compared
with the count the hold was recorded with. An equal count is the same prompt
continuing, so the hold serves it. A higher count is a new prompt, so the hold
is released before it could be served and this request is decided by the
policy. A count that cannot be determined is None, and None never reads as
"new": the hold behaves exactly as it did before the count existed.

Every test drives the same in-process harness the other proxy tests use, or
calls the pure functions directly. No server is started by hand and the router
is never run as a subprocess.

Config comes from `tools/sample-config-ACTIVE-test.yaml` (no classifier),
`tools/sample-config-CLASSIFIER-test.yaml` (classifier on, so the easy-then-hard
prompt pair can be exercised end to end) and `tools/sample-config-OPENAI-test.yaml`
(the same rules through the chat-completions route).
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

from router.config import RouterConfig, load_config
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.hold import REASON_HELD_MODEL, apply_hold
from router.killswitch import KILL_SWITCH_ENV, REASON_KILL_SWITCH
from router.policy import CLASSIFIER_MATCHES_TIER, Decision
from router.proxy import LOOPBACK_HOST, ProxySettings, create_server
from router.safety import BLOCKED_SWITCH_CAP
from router.signals import compute_signals, count_human_prompts
from router.signals_openai import compute_signals_openai
from router.state import UP, SessionState, SessionStore

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-ACTIVE-test.yaml"
CLASSIFIER_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-CLASSIFIER-test.yaml"
OPENAI_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-OPENAI-test.yaml"

MODEL_LOW = "test-low"
MODEL_MID = "test-mid"
MODEL_HIGH = "test-high"

#: Vendor-style ids: a slash and a colon are ordinary characters in the ids real
#: gateways serve, so a hold on one has to carry it verbatim.
VENDOR_MID = "vendor/model-a:free"
VENDOR_HIGH = "vendor/model-b:free"

#: The chat-completions route.
OPENAI_PATH = "/v1/chat/completions"

CLIENT_TIMEOUT = 5.0

#: Dummy write 3.75, dummy read 0.30, dummy benefit 0.001 covers a rebuild below
#: ~289 tokens, so the bodies here stay well under 1156 bytes and an escalation
#: is never quietly blocked on cost instead.
CROSSOVER_BYTES = 289 * 4

#: One session for every request unless a test says otherwise: the session hint
#: is derived from the first user message, so the text must not change.
SESSION_TEXT = "one long per prompt session"

#: The two prompts of the classifier pair. The first scores below `cheap_below`
#: because it carries the cheap keyword `hello`; the second carries the strong
#: keyword `refactor` and lands above `strong_from`.
EASY_PROMPT = "hello there"
HARD_PROMPT = "refactor the parser"

#: The text a second human prompt brings with it. No keyword in it, so the
#: classifier agrees with test-mid and has nothing else to propose.
NEW_PROMPT_TEXT = "a different task"

#: A phrase that must never reach the database or any log line.
SECRET_PHRASE = "quartz-lantern-4417"


# --- the config --------------------------------------------------------------


def active_config(hold_max: int = 50, **policy_overrides: Any) -> RouterConfig:
    """The ACTIVE sample: rules only, no classifier, a hold that outlives one prompt."""
    config = load_config(ACTIVE_CONFIG_PATH)
    return replace(
        config,
        policy=replace(
            config.policy,
            hold_max_requests=hold_max,
            **policy_overrides,
        ),
    )


def classifier_config(hold_max: int = 50, **policy_overrides: Any) -> RouterConfig:
    """The CLASSIFIER sample. Dwell and hysteresis default to 5 and 3, which would
    block the second switch of an easy-then-hard prompt pair; a test that needs
    both prompts decided on their own merits passes 1 for each."""
    config = load_config(CLASSIFIER_CONFIG_PATH)
    return replace(
        config,
        policy=replace(
            config.policy,
            hold_max_requests=hold_max,
            **policy_overrides,
        ),
    )


def openai_config(hold_max: int = 50, **policy_overrides: Any) -> RouterConfig:
    """The chat-completions sample: the CLASSIFIER sample's rules and keywords."""
    config = load_config(OPENAI_CONFIG_PATH)
    return replace(
        config,
        policy=replace(
            config.policy,
            hold_max_requests=hold_max,
            **policy_overrides,
        ),
    )


def vendor_config() -> RouterConfig:
    """The ACTIVE sample with the ids a real gateway serves, prices unchanged."""
    config = active_config()
    prices = {spec.cost_tier: spec for spec in config.models}
    models = tuple(
        replace(prices[tier], id=model_id)
        for tier, model_id in (
            ("low", "vendor/model-low:free"),
            ("mid", VENDOR_MID),
            ("high", VENDOR_HIGH),
        )
    )
    return replace(config, models=models, default_model=VENDOR_MID)


# --- the bodies --------------------------------------------------------------


def switch_body(model: str = MODEL_MID, text: str = SESSION_TEXT) -> bytes:
    """A body that escalates: two tool errors in a row, cheap enough to afford.

    It carries one human prompt, which is what the hold it starts is recorded
    against.
    """
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": text},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "search"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "is_error": True},
                        {"type": "tool_result", "tool_use_id": "t2", "is_error": True},
                    ],
                },
            ],
        }
    ).encode()
    assert len(body) < CROSSOVER_BYTES, "switch_body must stay cheap enough to be allowed"
    return body


def continuation_body(model: str = MODEL_MID, text: str = SESSION_TEXT) -> bytes:
    """The same prompt's tool loop carrying on: two clean results, no new prompt.

    Two results below `escalate_consecutive_errors`, a tail of two calls below
    `escalate_repeated_tool_calls`, and a second turn above
    `downgrade_max_turn_index`, so the policy has nothing to propose and the
    hold - if there is one - is the only thing that can move the model.
    """
    return json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": text},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "search"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t2", "name": "search"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t2"}],
                },
            ],
        }
    ).encode()


def new_prompt_body(
    model: str = MODEL_MID, text: str = SESSION_TEXT, new_text: str = NEW_PROMPT_TEXT
) -> bytes:
    """The same session after the human asked for something else: two prompts.

    The tool result in between is a single error, so neither escalation nor the
    downgrade rule fires: with no hold in the way, the policy answers STAY.
    """
    return json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": text},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "search"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "is_error": True}
                    ],
                },
                {"role": "user", "content": new_text},
            ],
        }
    ).encode()


def easy_body() -> bytes:
    """The first human prompt: `hello` is a cheap keyword, so the tier is low."""
    return json.dumps(
        {
            "model": MODEL_MID,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": EASY_PROMPT}],
        }
    ).encode()


def classifier_body(text: str) -> bytes:
    """A later human prompt in the easy session, whatever that prompt asks for.

    Its first message is the first prompt's text, so the session hint - and with
    it the session, and with it the hold - carries over.
    """
    return json.dumps(
        {
            "model": MODEL_MID,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": EASY_PROMPT},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": text},
            ],
        }
    ).encode()


def hard_body() -> bytes:
    """A second human prompt carrying `refactor`, so its tier is high."""
    return classifier_body(HARD_PROMPT)


def openai_switch_body(model: str = MODEL_MID, text: str = SESSION_TEXT) -> bytes:
    """A chat body that escalates: one tool called three times in a row."""
    body = json.dumps(
        {
            "model": model,
            "max_completion_tokens": 16,
            "messages": [
                {"role": "user", "content": text},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call("search", index) for index in (1, 2, 3)],
                },
                _tool_result(1),
                _tool_result(2),
                _tool_result(3),
            ],
        }
    ).encode()
    assert len(body) < CROSSOVER_BYTES, "openai_switch_body must stay cheap enough"
    return body


def openai_continuation_body(model: str = MODEL_MID, text: str = SESSION_TEXT) -> bytes:
    """The same prompt's chat-completions tool loop: two calls, then two results.

    Below the repeat threshold, and the latest prompt still scores `mid`, so the
    classifier agrees with test-mid and only a hold can move the model.
    """
    return json.dumps(
        {
            "model": model,
            "max_completion_tokens": 16,
            "messages": [
                {"role": "user", "content": text},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call("search", 1), _tool_call("search", 2)],
                },
                _tool_result(1),
                _tool_result(2),
            ],
        }
    ).encode()


def openai_new_prompt_body(
    model: str = MODEL_MID, text: str = SESSION_TEXT, new_text: str = NEW_PROMPT_TEXT
) -> bytes:
    """The same chat session after a second human prompt: two prompts."""
    return json.dumps(
        {
            "model": model,
            "max_completion_tokens": 16,
            "messages": [
                {"role": "user", "content": text},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call("search", 1)],
                },
                _tool_result(1),
                {"role": "user", "content": new_text},
            ],
        }
    ).encode()


def _tool_call(name: str, index: int) -> dict[str, Any]:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _tool_result(index: int) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": f"call_{index}", "content": "ok"}


# --- the harness -------------------------------------------------------------


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


class RecordingUpstream(BaseHTTPRequestHandler):
    """Records what arrived and replies with a fixed JSON body."""

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
        payload = b'{"ok":true}'
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
    db_path: Path

    @property
    def forwarded(self) -> Recorded:
        assert self.recorded, "the upstream received nothing"
        return self.recorded[-1]


@contextlib.contextmanager
def running_proxy(config: RouterConfig, db_path: Path) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), RecordingUpstream)
    upstream.daemon_threads = True
    upstream.recorded = []  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    store = SessionStore()
    settings = ProxySettings(
        listen_port=0,
        upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
        mode=config.mode,
        decisions=DecisionLog(db_path),
        config=config,
        state=store,
    )
    server = create_server(settings)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Harness(
            proxy_port=server.server_address[1],
            recorded=upstream.recorded,  # type: ignore[attr-defined]
            store=store,
            db_path=db_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def post_once(
    port: int, body: bytes, path: str = MESSAGES_PATH
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        connection.request(
            "POST", path, body=body, headers={"Content-Type": "application/json"}
        )
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


def hold_of(store: SessionStore, log: DecisionLog, newest: int = 1) -> SessionState:
    """The stored state of the session the row `newest` rows back belongs to."""
    rows = log.recent(newest)
    return store.get(rows[-1].session_hint)


# --- the count ---------------------------------------------------------------


def test_the_count_reads_only_the_user_messages_that_carry_text():
    body = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [{"type": "text", "text": "an answer"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second"},
                    {"type": "tool_result", "tool_use_id": "t2"},
                ],
            },
        ]
    }

    assert count_human_prompts(body["messages"]) == 2
    assert compute_signals(body).human_prompt_count == 2


def test_the_openai_count_skips_system_tool_and_assistant_messages():
    body = {
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("search", 1)],
            },
            _tool_result(1),
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
    }

    assert compute_signals_openai(body).human_prompt_count == 2


def test_an_empty_conversation_has_zero_human_prompts():
    assert compute_signals({"messages": []}).human_prompt_count == 0
    assert compute_signals_openai({"messages": []}).human_prompt_count == 0


def test_a_conversation_that_cannot_be_read_has_no_count():
    assert compute_signals(None).human_prompt_count is None
    assert compute_signals("bad").human_prompt_count is None
    assert compute_signals({}).human_prompt_count is None
    assert compute_signals({"messages": None}).human_prompt_count is None
    assert compute_signals({"messages": "nope"}).human_prompt_count is None
    assert (
        compute_signals({"messages": [{"role": "user", "content": 7}]}).human_prompt_count
        is None
    )
    assert compute_signals_openai({"messages": None}).human_prompt_count is None
    assert (
        compute_signals_openai({"messages": [{"role": "user", "content": 7}]}).human_prompt_count
        is None
    )


def test_the_count_keeps_no_text():
    """Rule 1: block types are inspected to count, and nothing else is read."""
    signals = compute_signals(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": SECRET_PHRASE}]}]}
    )

    assert signals.human_prompt_count == 1
    assert SECRET_PHRASE not in json.dumps(signals.to_dict())
    assert SECRET_PHRASE not in json.dumps(
        compute_signals(
            {"messages": [{"role": "user", "content": f"{SECRET_PHRASE} and the rest"}]}
        ).to_dict()
    )


# --- the hold, as a value ----------------------------------------------------


def switch_decision(target: str = MODEL_HIGH) -> Decision:
    return Decision(
        action="SWITCH",
        target_model=target,
        reason_codes=["ESCALATE_TOOL_ERRORS"],
        direction=UP,
    )


def stay_decision(*reasons: str) -> Decision:
    return Decision(
        action="STAY",
        target_model=MODEL_MID,
        reason_codes=list(reasons) or ["NO_RULE_MATCHED"],
    )


def test_a_fresh_session_has_no_prompt_count():
    assert SessionState().held_prompt_count is None
    assert SessionState().recorded_hold(MODEL_HIGH, MODEL_MID).held_prompt_count is None


def test_recording_a_hold_keeps_the_prompt_count_it_was_given():
    state = SessionState(requests_seen=3).recorded_hold(MODEL_HIGH, MODEL_MID, 4)

    assert state.held_prompt_count == 4
    assert state.has_hold is True


def test_releasing_a_hold_clears_the_prompt_count():
    state = SessionState(
        held_model=MODEL_HIGH, held_from_model=MODEL_MID, held_prompt_count=4
    )

    assert state.released_hold().held_prompt_count is None
    assert state.released_hold().has_hold is False


def test_an_approved_switch_is_held_with_this_request_s_count():
    final, state = apply_hold(
        switch_decision(),
        MODEL_MID,
        None,
        SessionState(requests_seen=1),
        active_config(),
        "active",
        3,
    )

    assert final.action == "SWITCH"
    assert state.held_model == MODEL_HIGH
    assert state.held_prompt_count == 3
    assert state.held_requests == 0


def test_the_same_prompt_count_is_served_from_the_hold():
    held = SessionState(
        requests_seen=2,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
        held_prompt_count=1,
    )

    final, state = apply_hold(
        stay_decision(), MODEL_MID, None, held, active_config(), "active", 1
    )

    assert final.action == "SWITCH"
    assert final.reason_codes == [REASON_HELD_MODEL]
    assert state.held_requests == 1
    assert state.held_prompt_count == 1, "the same prompt keeps the count it was held with"


def test_a_higher_prompt_count_releases_the_hold_before_it_can_be_served():
    held = SessionState(
        requests_seen=2,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
        held_prompt_count=1,
    )

    final, state = apply_hold(
        stay_decision("ESCALATE_TOOL_ERRORS"),
        MODEL_MID,
        None,
        held,
        active_config(),
        "active",
        2,
    )

    assert final.action == "STAY"
    assert final.reason_codes == ["ESCALATE_TOOL_ERRORS"], "the policy's own answer stands"
    assert REASON_HELD_MODEL not in final.reason_codes
    assert state.has_hold is False
    assert state.held_prompt_count is None


def test_a_request_that_cannot_be_counted_is_still_served_from_the_hold():
    """None cannot say the prompt is new, so the hold behaves as it always did."""
    held = SessionState(
        requests_seen=2,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
        held_prompt_count=1,
    )

    final, state = apply_hold(
        stay_decision(), MODEL_MID, None, held, active_config(), "active", None
    )

    assert final.reason_codes == [REASON_HELD_MODEL]
    assert state.has_hold is True
    assert state.held_requests == 1


def test_a_hold_recorded_without_a_count_is_still_served():
    held = SessionState(
        requests_seen=2,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
        held_prompt_count=None,
    )

    final, state = apply_hold(
        stay_decision(), MODEL_MID, None, held, active_config(), "active", 5
    )

    assert final.reason_codes == [REASON_HELD_MODEL]
    assert state.held_requests == 1


def test_a_new_prompt_does_not_stop_a_switch_becoming_the_new_hold():
    held = SessionState(
        requests_seen=4,
        switch_count=1,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
        held_prompt_count=1,
    )

    final, state = apply_hold(
        switch_decision(MODEL_LOW),
        MODEL_MID,
        None,
        held,
        active_config(),
        "active",
        2,
    )

    assert final.reason_codes == ["ESCALATE_TOOL_ERRORS"], "safety's switch, not a hold"
    assert state.held_model == MODEL_LOW
    assert state.held_prompt_count == 2, "the new hold protects the new prompt"
    assert state.held_requests == 0


# --- the hold, through the proxy --------------------------------------------


def test_a_tool_loop_continuation_is_still_served_from_the_hold(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=5)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        first = log.recent(1)[0]
        assert first.reason_codes == ["ESCALATE_TOOL_ERRORS"]
        assert first.signal_values["human_prompt_count"] == 1

        post_once(harness.proxy_port, continuation_body(MODEL_MID))
        assert wait_for_rows(log, 2) == 2

        held = log.recent(1)[0]
        assert held.reason_codes == [REASON_HELD_MODEL]
        assert held.applied == 1
        assert held.signal_values["human_prompt_count"] == 1, "the loop added no prompt"
        assert held.session_hint == first.session_hint
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH

        state = hold_of(harness.store, log)
        assert state.held_requests == 1
        assert state.held_prompt_count == 1


def test_a_new_human_prompt_releases_the_hold(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=5)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        assert hold_of(harness.store, log).held_prompt_count == 1

        body = new_prompt_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body, "the policy found nothing to do"
        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.reason_codes == ["NO_RULE_MATCHED"]
        assert REASON_HELD_MODEL not in row.reason_codes
        assert row.signal_values["human_prompt_count"] == 2
        assert row.session_hint == log.recent(2)[1].session_hint, "one session, two prompts"

        state = hold_of(harness.store, log)
        assert state.has_hold is False
        assert state.held_prompt_count is None


def test_the_hold_still_expires_on_the_prompt_it_protects(tmp_path, monkeypatch):
    """Expiry counts held requests, so it still fires inside one prompt."""
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=2, dwell_requests=5)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1

        for expected in (2, 3):
            post_once(harness.proxy_port, continuation_body(MODEL_MID))
            assert wait_for_rows(log, expected) == expected
            assert log.recent(1)[0].reason_codes == [REASON_HELD_MODEL]

        body = continuation_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 4) == 4

        assert harness.forwarded.body == body, "the hold expired"
        expired = log.recent(1)[0]
        assert expired.action == "STAY"
        assert expired.applied == 0
        assert REASON_HELD_MODEL not in expired.reason_codes
        assert hold_of(harness.store, log).has_hold is False


def test_the_session_cap_still_ends_the_hold_inside_one_prompt(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=1, max_switches_per_session=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        assert hold_of(harness.store, log).switch_count == 1

        blocked = switch_body(MODEL_MID)
        post_once(harness.proxy_port, blocked)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == blocked, "the cap blocked the switch"
        row = log.recent(1)[0]
        assert BLOCKED_SWITCH_CAP in row.reason_codes
        assert REASON_HELD_MODEL not in row.reason_codes
        assert hold_of(harness.store, log).has_hold is False

        post_once(harness.proxy_port, continuation_body(MODEL_MID))
        assert wait_for_rows(log, 3) == 3
        assert REASON_HELD_MODEL not in log.recent(1)[0].reason_codes


def test_the_kill_switch_beats_a_live_hold_on_a_new_prompt(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=5)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        assert hold_of(harness.store, log).has_hold is True

        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        body = new_prompt_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.reason_codes == [REASON_KILL_SWITCH]
        assert row.applied == 0
        assert hold_of(harness.store, log).has_hold is False

        monkeypatch.delenv(KILL_SWITCH_ENV)
        post_once(harness.proxy_port, continuation_body(MODEL_MID))
        assert wait_for_rows(log, 3) == 3
        assert REASON_HELD_MODEL not in log.recent(1)[0].reason_codes


def test_a_hold_on_a_vendor_id_is_released_by_a_new_prompt(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = vendor_config()

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(VENDOR_MID))
        assert wait_for_rows(log, 1) == 1
        first = log.recent(1)[0]
        assert first.chosen_model == VENDOR_HIGH
        assert json.loads(harness.forwarded.body)["model"] == VENDOR_HIGH
        assert hold_of(harness.store, log).held_prompt_count == 1

        post_once(harness.proxy_port, continuation_body(VENDOR_MID))
        assert wait_for_rows(log, 2) == 2
        held = log.recent(1)[0]
        assert held.reason_codes == [REASON_HELD_MODEL]
        assert held.chosen_model == VENDOR_HIGH
        assert json.loads(harness.forwarded.body)["model"] == VENDOR_HIGH

        body = new_prompt_body(VENDOR_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 3) == 3

        assert json.loads(harness.forwarded.body)["model"] == VENDOR_MID
        released = log.recent(1)[0]
        assert released.requested_model == VENDOR_MID
        assert released.chosen_model == VENDOR_MID
        assert REASON_HELD_MODEL not in released.reason_codes
        assert "/" in released.chosen_model and ":" in released.chosen_model
        assert hold_of(harness.store, log).has_hold is False


def test_the_second_prompt_phrase_never_reaches_the_database_or_a_log_line(
    tmp_path, monkeypatch
):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=5)

    with running_proxy(config, log.db_path) as harness:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            post_once(harness.proxy_port, switch_body(MODEL_MID, SECRET_PHRASE))
            assert wait_for_rows(log, 1) == 1
            post_once(harness.proxy_port, new_prompt_body(MODEL_MID, SECRET_PHRASE))
            assert wait_for_rows(log, 2) == 2

        assert log.recent(1)[0].reason_codes == ["NO_RULE_MATCHED"], "the release ran"
        assert SECRET_PHRASE.encode() not in log.db_path.read_bytes()
        assert SECRET_PHRASE not in captured.getvalue()
        for row in log.recent(2):
            assert SECRET_PHRASE not in str(row.signal_values)
            assert row.session_hint not in (None, SECRET_PHRASE)


# --- two human prompts, one session ------------------------------------------


def test_the_second_prompt_is_classified_on_its_own_and_is_not_held(
    tmp_path, monkeypatch
):
    """The easy prompt holds the session; the hard prompt that follows does not."""
    log = log_at(tmp_path, monkeypatch)
    config = classifier_config(dwell_requests=1, hysteresis_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, easy_body())
        assert wait_for_rows(log, 1) == 1
        first = log.recent(1)[0]
        assert first.reason_codes == ["CLASSIFIER_LOW"]
        assert first.chosen_model == MODEL_LOW
        assert first.applied == 1
        assert first.signal_values["human_prompt_count"] == 1
        assert json.loads(harness.forwarded.body)["model"] == MODEL_LOW
        assert hold_of(harness.store, log).held_model == MODEL_LOW
        assert hold_of(harness.store, log).held_prompt_count == 1

        post_once(harness.proxy_port, hard_body())
        assert wait_for_rows(log, 2) == 2

        second = log.recent(1)[0]
        assert second.reason_codes == ["CLASSIFIER_HIGH"]
        assert second.chosen_model == MODEL_HIGH
        assert second.applied == 1
        assert REASON_HELD_MODEL not in second.reason_codes
        assert second.signal_values["human_prompt_count"] == 2
        assert second.session_hint == first.session_hint, "one session, two prompts"
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH

        state = hold_of(harness.store, log)
        assert state.held_model == MODEL_HIGH, "the second switch is the new hold"
        assert state.held_prompt_count == 2


def test_a_second_prompt_that_fits_the_model_is_not_answered_with_the_first(
    tmp_path, monkeypatch
):
    """The first prompt held test-low. A second prompt that suits test-mid must
    reach test-mid: the router does not answer a new task with the model the
    previous task needed."""
    log = log_at(tmp_path, monkeypatch)
    config = classifier_config(dwell_requests=1, hysteresis_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, easy_body())
        assert wait_for_rows(log, 1) == 1
        first = log.recent(1)[0]
        assert first.chosen_model == MODEL_LOW
        assert hold_of(harness.store, log).held_model == MODEL_LOW

        post_once(harness.proxy_port, classifier_body(SESSION_TEXT))
        assert wait_for_rows(log, 2) == 2

        second = log.recent(1)[0]
        assert second.action == "STAY"
        assert second.applied == 0
        assert second.reason_codes == [CLASSIFIER_MATCHES_TIER]
        assert REASON_HELD_MODEL not in second.reason_codes
        assert second.chosen_model == MODEL_MID
        assert second.signal_values["human_prompt_count"] == 2
        assert second.session_hint == first.session_hint
        assert json.loads(harness.forwarded.body)["model"] == MODEL_MID

        state = hold_of(harness.store, log)
        assert state.has_hold is False
        assert state.held_prompt_count is None


def test_a_tool_loop_continuation_is_still_held_in_the_openai_format(
    tmp_path, monkeypatch
):
    log = log_at(tmp_path, monkeypatch)
    config = openai_config(hold_max=50)

    with running_proxy(config, log.db_path) as harness:
        status, _, _ = post_once(harness.proxy_port, openai_switch_body(), OPENAI_PATH)
        assert status == 200
        assert wait_for_rows(log, 1) == 1
        first = log.recent(1)[0]
        assert first.reason_codes == ["ESCALATE_REPEATED_TOOLS"]
        assert first.signal_values["human_prompt_count"] == 1
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH

        status, _, _ = post_once(
            harness.proxy_port, openai_continuation_body(), OPENAI_PATH
        )
        assert status == 200
        assert wait_for_rows(log, 2) == 2

        held = log.recent(1)[0]
        assert held.reason_codes == [REASON_HELD_MODEL]
        assert held.applied == 1
        assert held.signal_values["human_prompt_count"] == 1
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH


def test_a_new_human_prompt_releases_the_hold_in_the_openai_format(
    tmp_path, monkeypatch
):
    log = log_at(tmp_path, monkeypatch)
    config = openai_config(hold_max=50)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, openai_switch_body(), OPENAI_PATH)
        assert wait_for_rows(log, 1) == 1
        assert hold_of(harness.store, log).held_prompt_count == 1

        body = openai_new_prompt_body()
        status, _, _ = post_once(harness.proxy_port, body, OPENAI_PATH)
        assert status == 200
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body, "the classifier agreed with test-mid"
        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert row.reason_codes == [CLASSIFIER_MATCHES_TIER]
        assert REASON_HELD_MODEL not in row.reason_codes
        assert row.signal_values["human_prompt_count"] == 2
        assert row.session_hint == log.recent(2)[1].session_hint

        state = hold_of(harness.store, log)
        assert state.has_hold is False
        assert state.held_prompt_count is None
