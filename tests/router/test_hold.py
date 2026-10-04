"""Tests for the model hold.

The problem the hold solves: the client keeps sending the model it configured,
so the request after an approved switch arrives naming the ORIGINAL model again.
Without a hold the router hands the session straight back to that model on the
next request, and the switch it just approved is undone.

Every test here drives the same in-process harness the other proxy tests use, or
calls the pure functions directly. No server is started by hand and the router is
never run as a subprocess.

Config comes from `tools/sample-config-ACTIVE-test.yaml` (`mode: active`, the
non-placeholder ids `test-low`/`test-mid`/`test-high`, dummy prices) and from a
shadow copy of it for the mode contrast. `hold_max_requests` is set to 2 in most
tests so that expiry is testable without sending fifty requests.
"""
from __future__ import annotations

import contextlib
import dataclasses
import http.client
import json
import threading
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from router import proxy as proxy_module
from router.breaker import REASON_CIRCUIT_OPEN, CircuitBreaker
from router.config import DEFAULT_HOLD_MAX_REQUESTS, ConfigError, RouterConfig, load_config
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.hold import (
    HELD_SIGNAL_KEY,
    REASON_HELD_MODEL,
    apply_hold,
    hold_max_requests,
)
from router.killswitch import KILL_SWITCH_ENV, REASON_KILL_SWITCH
from router.policy import Decision
from router.proxy import LOOPBACK_HOST, ProxySettings, create_server
from router.safety import BLOCKED_DWELL, BLOCKED_SWITCH_CAP, apply_safety
from router.signals import Signals
from router.state import UP, SessionState, SessionStore

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-ACTIVE-test.yaml"

MODEL_LOW = "test-low"
MODEL_MID = "test-mid"
MODEL_HIGH = "test-high"

CLIENT_TIMEOUT = 5.0

#: Dummy write 3.75, dummy read 0.30, dummy benefit 0.001 covers a rebuild below
#: ~289 tokens, so the bodies here stay well under 1156 bytes and an escalation
#: is never quietly blocked on cost instead.
CROSSOVER_BYTES = 289 * 4

#: One session for every request unless a test says otherwise: the session hint
#: is derived from the first user message, so the text must not change.
SESSION_TEXT = "one long hold session"


def active_config(hold_max: int = 2, **policy_overrides: Any) -> RouterConfig:
    """The active config with a short hold, so expiry needs two requests."""
    config = load_config(ACTIVE_CONFIG_PATH)
    return replace(
        config,
        mode="shadow" if policy_overrides.pop("shadow", False) else config.mode,
        policy=replace(
            config.policy,
            hold_max_requests=hold_max,
            **policy_overrides,
        ),
    )


def shadow_config(hold_max: int = 2, **policy_overrides: Any) -> RouterConfig:
    """The same config in shadow mode: nothing may be rewritten."""
    config = active_config(hold_max, **policy_overrides)
    return replace(config, mode="shadow")


def switch_body(model: str = MODEL_MID, text: str = SESSION_TEXT) -> bytes:
    """A body that escalates: two tool errors in a row, cheap enough to afford."""
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


def stay_body(model: str = MODEL_MID, text: str = SESSION_TEXT) -> bytes:
    """A body no rule acts on: one tool error, no repeat, one turn.

    `escalate_consecutive_errors` is 2, so a single error is not an escalation,
    and the downgrade rule needs zero errors. The policy must STAY here, which is
    exactly the case the hold has to cover.
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
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True}],
                },
            ],
        }
    ).encode()


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
def running_proxy(
    config: RouterConfig,
    db_path: Path,
    store: SessionStore | None = None,
    breaker: CircuitBreaker | None = None,
) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), RecordingUpstream)
    upstream.daemon_threads = True
    upstream.recorded = []  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    sessions = SessionStore() if store is None else store
    settings = ProxySettings(
        listen_port=0,
        upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
        mode=config.mode,
        decisions=DecisionLog(db_path),
        config=config,
        state=sessions,
        breaker=breaker,
    )
    server = create_server(settings)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Harness(
            proxy_port=server.server_address[1],
            recorded=upstream.recorded,  # type: ignore[attr-defined]
            store=sessions,
            db_path=db_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def post_once(port: int, body: bytes) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=CLIENT_TIMEOUT)
    try:
        connection.request(
            "POST", MESSAGES_PATH, body=body, headers={"Content-Type": "application/json"}
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
    """The stored state of the session the row `newest` rows back belongs to.

    Keyed off the row the router actually recorded, so the lookup cannot drift
    from the session the proxy used.
    """
    rows = log.recent(newest)
    return store.get(rows[-1].session_hint)


# --- the state --------------------------------------------------------------


def test_a_fresh_session_holds_nothing():
    state = SessionState()

    assert state.held_model is None
    assert state.held_from_model is None
    assert state.held_requests == 0
    assert state.has_hold is False


def test_recording_a_hold_starts_its_count_at_zero():
    state = SessionState(requests_seen=3, switch_count=1).recorded_hold(MODEL_HIGH, MODEL_MID)

    assert state.held_model == MODEL_HIGH
    assert state.held_from_model == MODEL_MID
    assert state.held_requests == 0
    assert state.has_hold is True


def test_recording_a_hold_changes_no_switch_counter():
    before = SessionState(
        requests_seen=7,
        last_switch_request_index=5,
        last_switch_direction=UP,
        opposite_streak=2,
        switch_count=3,
    )

    after = before.recorded_hold(MODEL_HIGH, MODEL_MID)

    assert after.requests_seen == before.requests_seen
    assert after.last_switch_request_index == before.last_switch_request_index
    assert after.last_switch_direction == before.last_switch_direction
    assert after.opposite_streak == before.opposite_streak
    assert after.switch_count == before.switch_count
    assert before.held_model is None, "the original state is untouched"


def test_releasing_a_hold_clears_all_three_fields():
    state = SessionState(held_model=MODEL_HIGH, held_from_model=MODEL_MID, held_requests=7)

    released = state.released_hold()

    assert released.held_model is None
    assert released.held_from_model is None
    assert released.held_requests == 0
    assert released.switch_count == state.switch_count


def test_state_still_holds_only_ints_and_strings():
    """Rule 1. Nothing here may become a container or any request content."""
    for field in dataclasses.fields(SessionState):
        annotation = str(field.type)
        assert "str" in annotation or "int" in annotation, (
            f"{field.name} is annotated {field.type!r}, which is neither int nor str"
        )
        assert "list" not in annotation
        assert "dict" not in annotation
        assert "bytes" not in annotation

    state = SessionState(
        requests_seen=1,
        last_switch_request_index=1,
        last_switch_direction=UP,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
        held_requests=2,
    )
    for value in dataclasses.astuple(state):
        assert value is None or isinstance(value, (int, str)), f"{value!r} is not int or str"


# --- the config key ---------------------------------------------------------


def test_the_shipped_default_is_fifty():
    assert DEFAULT_HOLD_MAX_REQUESTS == 50
    assert load_config().policy.hold_max_requests == 50


def test_the_shipped_config_documents_the_key():
    assert "hold_max_requests: 50" in (REPO_ROOT / "router" / "config.yaml").read_text(
        encoding="utf-8"
    )


def test_the_key_is_read_when_present(tmp_path):
    import yaml

    data = yaml.safe_load(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    data["policy"]["hold_max_requests"] = 7
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert load_config(path).policy.hold_max_requests == 7


@pytest.mark.parametrize("value", [0, -1, "5", 1.5, True, None])
def test_the_key_must_be_a_positive_integer(tmp_path, value):
    import yaml

    data = yaml.safe_load(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    data["policy"]["hold_max_requests"] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="hold_max_requests"):
        load_config(path)


def test_an_unknown_policy_key_is_still_rejected(tmp_path):
    import yaml

    data = yaml.safe_load(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    data["policy"]["hold_max_request"] = 50
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown policy key"):
        load_config(path)


def test_hold_max_requests_falls_back_to_the_default():
    config = replace(active_config(), policy=None)

    assert hold_max_requests(config) == DEFAULT_HOLD_MAX_REQUESTS


# --- the hold, as a value ---------------------------------------------------


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


def test_an_approved_switch_is_held():
    final, state = apply_hold(
        switch_decision(), MODEL_MID, None, SessionState(requests_seen=1), active_config()
    )

    assert final.action == "SWITCH"
    assert state.held_model == MODEL_HIGH
    assert state.held_from_model == MODEL_MID
    assert state.held_requests == 0


def test_a_live_hold_replaces_a_stay():
    held = SessionState(requests_seen=1, held_model=MODEL_HIGH, held_from_model=MODEL_MID)

    final, state = apply_hold(stay_decision(), MODEL_MID, None, held, active_config())

    assert final.action == "SWITCH"
    assert final.target_model == MODEL_HIGH
    assert final.reason_codes == [REASON_HELD_MODEL]
    assert state.held_requests == 1
    assert state.held_model == MODEL_HIGH


def test_a_hold_is_not_used_when_the_request_names_another_model():
    held = SessionState(requests_seen=1, held_model=MODEL_HIGH, held_from_model=MODEL_MID)

    final, state = apply_hold(stay_decision(), MODEL_LOW, None, held, active_config())

    assert final.action == "STAY"
    assert final.reason_codes == ["NO_RULE_MATCHED"]
    assert state.has_hold is False


def test_a_hold_is_not_used_when_the_model_is_not_in_the_config():
    held = SessionState(requests_seen=1, held_model="not-a-model", held_from_model=MODEL_MID)

    final, state = apply_hold(stay_decision(), MODEL_MID, None, held, active_config())

    assert final.action == "STAY"
    assert state.has_hold is False


def test_a_hold_outside_the_config_is_never_recorded():
    final, state = apply_hold(
        switch_decision("not-a-model"), MODEL_MID, None, SessionState(), active_config()
    )

    assert final.action == "SWITCH", "safety still decides; the hold does not"
    assert state.has_hold is False


def test_a_hold_survives_a_dwell_block_because_it_never_reaches_dwell():
    held = SessionState(
        requests_seen=2,
        last_switch_request_index=1,
        last_switch_direction=UP,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
    )

    final, state = apply_hold(
        stay_decision("ESCALATE_TOOL_ERRORS", BLOCKED_DWELL),
        MODEL_MID,
        None,
        held,
        active_config(),
    )

    assert final.action == "SWITCH"
    assert final.reason_codes == [REASON_HELD_MODEL]
    assert state.held_requests == 1


def test_a_hold_does_not_reset_dwell_or_count_a_switch():
    held = SessionState(
        requests_seen=2,
        last_switch_request_index=1,
        last_switch_direction=UP,
        switch_count=1,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
    )

    final, state = apply_hold(stay_decision(), MODEL_MID, None, held, active_config())

    assert final.action == "SWITCH"
    assert state.switch_count == 1, "a held request is not a switch"
    assert state.last_switch_request_index == 1, "dwell is not reset"
    assert state.last_switch_direction == UP


def test_a_hold_carries_the_requested_effort_unchanged():
    held = SessionState(requests_seen=1, held_model=MODEL_HIGH, held_from_model=MODEL_MID)

    final, _ = apply_hold(
        stay_decision(), MODEL_MID, "test-effort-medium", held, active_config()
    )

    assert final.target_effort == "test-effort-medium"


def test_a_hold_reports_no_direction_for_safety_to_rate_limit():
    held = SessionState(requests_seen=1, held_model=MODEL_HIGH, held_from_model=MODEL_MID)

    final, _ = apply_hold(stay_decision(), MODEL_MID, None, held, active_config())

    assert final.direction is None


def test_a_hold_expires_after_hold_max_requests():
    held = SessionState(
        requests_seen=9,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
        held_requests=2,
    )

    final, state = apply_hold(stay_decision(), MODEL_MID, None, held, active_config(hold_max=2))

    assert final.action == "STAY"
    assert final.reason_codes == ["NO_RULE_MATCHED"]
    assert state.has_hold is False


def test_the_last_allowed_hold_is_still_served():
    held = SessionState(
        requests_seen=8,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
        held_requests=1,
    )

    final, state = apply_hold(stay_decision(), MODEL_MID, None, held, active_config(hold_max=2))

    assert final.action == "SWITCH"
    assert state.held_requests == 2


def test_a_new_approved_switch_replaces_the_hold():
    held = SessionState(
        requests_seen=4,
        switch_count=1,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
        held_requests=2,
    )

    final, state = apply_hold(switch_decision(MODEL_LOW), MODEL_MID, None, held, active_config())

    assert final.target_model == MODEL_LOW
    assert state.held_model == MODEL_LOW
    assert state.held_requests == 0, "a new switch starts a new hold"


def test_a_cap_block_is_not_held():
    """The per-session cap bounds how often a session is moved, hold included."""
    held = SessionState(
        requests_seen=4,
        switch_count=10,
        held_model=MODEL_HIGH,
        held_from_model=MODEL_MID,
    )

    final, state = apply_hold(
        stay_decision("ESCALATE_TOOL_ERRORS", BLOCKED_SWITCH_CAP),
        MODEL_MID,
        None,
        held,
        active_config(max_switches_per_session=10),
    )

    assert final.action == "STAY"
    assert BLOCKED_SWITCH_CAP in final.reason_codes
    assert state.has_hold is False


def test_mode_off_releases_the_hold():
    held = SessionState(requests_seen=1, held_model=MODEL_HIGH, held_from_model=MODEL_MID)

    final, state = apply_hold(
        stay_decision("MODE_OFF"), MODEL_MID, None, held, active_config(), "off"
    )

    assert final.action == "STAY"
    assert state.has_hold is False


def test_nothing_is_held_without_a_hold():
    final, state = apply_hold(stay_decision(), MODEL_MID, None, SessionState(), active_config())

    assert final.action == "STAY"
    assert state == SessionState()


def test_the_hold_runs_after_safety_and_only_on_its_final_answer():
    """A switch safety approved is held; a switch it blocked is not."""
    config = active_config(hold_max=2, dwell_requests=1)
    signals = Signals(context_tokens_estimate=100, consecutive_tool_errors=2)

    approved, state = apply_safety(switch_decision(), signals, SessionState(requests_seen=1), config)
    assert approved.action == "SWITCH"
    final, held = apply_hold(approved, MODEL_MID, None, state, config)
    assert final.reason_codes == ["ESCALATE_TOOL_ERRORS"]
    assert held.held_model == MODEL_HIGH

    fresh = SessionState(requests_seen=1)
    blocked, blocked_state = apply_safety(
        switch_decision(), signals, replace(fresh, last_switch_request_index=1), config
    )
    assert blocked.action == "STAY"
    final, held = apply_hold(blocked, MODEL_MID, None, blocked_state, config)
    assert final.action == "STAY"
    assert held.has_hold is False


# --- the hold, through the proxy -------------------------------------------


def test_the_next_request_in_active_mode_is_rewritten_to_the_held_model(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=3, dwell_requests=1)
    store = SessionStore()

    with running_proxy(config, log.db_path, store=store) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH

        post_once(harness.proxy_port, stay_body(MODEL_MID))
        assert wait_for_rows(log, 2) == 2

        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        held = log.recent(1)[0]
        assert held.action == "SWITCH"
        assert held.applied == 1
        assert held.reason_codes == [REASON_HELD_MODEL]
        assert held.requested_model == MODEL_MID
        assert held.chosen_model == MODEL_HIGH
        assert held.mode == "active"


def test_the_held_request_keeps_every_other_byte(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=3, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1

        body = stay_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        forwarded = harness.forwarded.body
        assert forwarded != body, "the model must have been rewritten"
        assert forwarded.replace(b'"model": "test-high"', b'"model": "test-mid"') == body


def test_shadow_mode_logs_the_hold_without_rewriting(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = shadow_config(hold_max=3, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        body = stay_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body, "shadow mode changes nothing"
        held = log.recent(1)[0]
        assert held.action == "SWITCH"
        assert held.applied == 0
        assert held.reason_codes == [REASON_HELD_MODEL]
        assert held.chosen_model == MODEL_HIGH
        assert held.mode == "shadow"


def test_shadow_mode_reports_the_hold_as_would_switch(tmp_path, monkeypatch, capsys):
    from router.cli import format_rows

    log = log_at(tmp_path, monkeypatch)
    config = shadow_config(hold_max=3, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        post_once(harness.proxy_port, stay_body(MODEL_MID))
        assert wait_for_rows(log, 2) == 2

    table = format_rows(log.recent(1))
    assert REASON_HELD_MODEL in table
    assert "would SWITCH" in table


def test_held_requests_is_recorded_in_signal_values(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=4, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        assert log.recent(1)[0].signal_values[HELD_SIGNAL_KEY] == 0

        for expected in (1, 2):
            post_once(harness.proxy_port, stay_body(MODEL_MID))
            assert wait_for_rows(log, expected + 1) == expected + 1
            assert log.recent(1)[0].signal_values[HELD_SIGNAL_KEY] == expected


def test_no_held_requests_key_when_nothing_is_held(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=3, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, stay_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1

        assert HELD_SIGNAL_KEY not in log.recent(1)[0].signal_values


def test_the_hold_does_not_change_switch_count_or_dwell(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=4, dwell_requests=5)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        after_switch = hold_of(harness.store, log)
        assert after_switch.switch_count == 1
        assert after_switch.last_switch_request_index == 1

        for index in range(2, 5):
            post_once(harness.proxy_port, stay_body(MODEL_MID))
            assert wait_for_rows(log, index) == index

        state = hold_of(harness.store, log)
        assert state.switch_count == 1, "three held requests are not three switches"
        assert state.last_switch_request_index == 1, "dwell was not reset"
        assert state.held_requests == 3

        # Dwell is still in force: with the hold expired, a fresh switch is
        # refused exactly where it would have been before any hold existed.
        assert log.recent(4)[0].signal_values[HELD_SIGNAL_KEY] == 3


def test_dwell_still_blocks_a_new_switch_while_a_hold_is_live(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=4, dwell_requests=5)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1

        # A second escalating request inside the dwell window: the hold serves
        # it, and the switch the policy proposed stays blocked.
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 2) == 2

        row = log.recent(1)[0]
        assert row.action == "SWITCH"
        assert row.reason_codes == [REASON_HELD_MODEL]
        assert BLOCKED_DWELL not in row.reason_codes
        assert hold_of(harness.store, log).switch_count == 1


def test_the_hold_expires_after_hold_max_requests(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=2, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1

        for expected in (2, 3):
            post_once(harness.proxy_port, stay_body(MODEL_MID))
            assert wait_for_rows(log, expected) == expected
            assert log.recent(1)[0].reason_codes == [REASON_HELD_MODEL]

        body = stay_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 4) == 4

        assert harness.forwarded.body == body, "the hold expired, so nothing rewrites"
        expired = log.recent(1)[0]
        assert expired.action == "STAY"
        assert expired.applied == 0
        assert REASON_HELD_MODEL not in expired.reason_codes
        assert hold_of(harness.store, log).has_hold is False


def test_the_hold_is_replaced_by_a_new_approved_switch(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        # Low escalates to mid, and mid is held.
        post_once(harness.proxy_port, switch_body(MODEL_LOW, "escalating session"))
        assert wait_for_rows(log, 1) == 1
        assert json.loads(harness.forwarded.body)["model"] == MODEL_MID
        assert hold_of(harness.store, log).held_model == MODEL_MID
        assert hold_of(harness.store, log).held_from_model == MODEL_LOW

        # The client follows the router to mid, which escalates again. The
        # approved switch to high is the new truth and replaces the hold.
        post_once(harness.proxy_port, switch_body(MODEL_MID, "escalating session"))
        assert wait_for_rows(log, 2) == 2
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH

        state = hold_of(harness.store, log)
        assert state.held_model == MODEL_HIGH, "the new switch replaced the hold"
        assert state.held_from_model == MODEL_MID
        assert state.held_requests == 0
        assert state.switch_count == 2

        # And the replaced hold serves the next quiet request from the new model.
        post_once(harness.proxy_port, stay_body(MODEL_MID, "escalating session"))
        assert wait_for_rows(log, 3) == 3
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        assert log.recent(1)[0].reason_codes == [REASON_HELD_MODEL]
        assert hold_of(harness.store, log).held_requests == 1


def test_the_hold_is_not_used_when_the_request_names_another_model(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1

        # The client asks for a third model: the hold was about test-mid.
        body = stay_body(MODEL_LOW)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert REASON_HELD_MODEL not in row.reason_codes
        assert hold_of(harness.store, log).has_hold is False


def test_the_hold_is_not_used_when_the_model_is_not_in_the_config(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1

        # Same session, a model the config does not list.
        body = stay_body("model-that-is-not-configured")
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.reason_codes == ["UNKNOWN_MODEL"]
        assert row.applied == 0
        assert REASON_HELD_MODEL not in row.reason_codes


def test_the_kill_switch_releases_the_hold(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        assert hold_of(harness.store, log).has_hold is True

        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        body = stay_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.reason_codes == [REASON_KILL_SWITCH]
        assert row.applied == 0
        assert hold_of(harness.store, log).has_hold is False, "the hold was dropped"

        # With the switch off again the session is no longer held.
        monkeypatch.delenv(KILL_SWITCH_ENV)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 3) == 3

        assert harness.forwarded.body == body
        assert REASON_HELD_MODEL not in log.recent(1)[0].reason_codes


def test_an_open_circuit_releases_the_hold(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=1)
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60)

    healthy = proxy_module.compute_signals

    def break_signals(*args: Any, **kwargs: Any) -> Signals:
        raise RuntimeError("signals exploded")

    with running_proxy(config, log.db_path, breaker=breaker) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        assert hold_of(harness.store, log).has_hold is True

        # This request fails inside the router, which opens the circuit.
        monkeypatch.setattr(proxy_module, "compute_signals", break_signals)
        post_once(harness.proxy_port, stay_body(MODEL_MID))
        assert wait_for_rows(log, 2) == 2
        assert breaker.is_open() is True
        assert hold_of(harness.store, log).has_hold is True, "no decision, no release"

        # The next request is passed through by the open circuit, and that is
        # what releases the hold.
        monkeypatch.setattr(proxy_module, "compute_signals", healthy)
        body = stay_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 3) == 3

        assert harness.forwarded.body == body
        assert log.recent(1)[0].reason_codes == [REASON_CIRCUIT_OPEN]
        assert log.recent(1)[0].applied == 0
        assert hold_of(harness.store, log).has_hold is False


def test_a_cap_block_releases_the_hold(tmp_path, monkeypatch):
    """A cap-blocked stay is a change of model the cap said no to, so no holding."""
    log = log_at(tmp_path, monkeypatch)
    config = active_config(
        hold_max=50,
        dwell_requests=1,
        max_switches_per_session=1,
    )

    with running_proxy(config, log.db_path) as harness:
        # The session spends its whole switching budget on one approved switch.
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH
        state = hold_of(harness.store, log)
        assert state.has_hold is True
        assert state.switch_count == 1

        # The next switch is refused by the cap, so the hold goes with it.
        blocked = switch_body(MODEL_MID)
        post_once(harness.proxy_port, blocked)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == blocked
        row = log.recent(1)[0]
        assert row.action == "STAY"
        assert row.applied == 0
        assert BLOCKED_SWITCH_CAP in row.reason_codes
        assert hold_of(harness.store, log).has_hold is False

        # And nothing is held afterwards: a quiet request is not served from it.
        quiet = stay_body(MODEL_MID)
        post_once(harness.proxy_port, quiet)
        assert wait_for_rows(log, 3) == 3

        assert harness.forwarded.body == quiet
        assert REASON_HELD_MODEL not in log.recent(1)[0].reason_codes
        assert hold_of(harness.store, log).has_hold is False


def test_a_signal_failure_leaves_the_hold_alone(tmp_path, monkeypatch):
    """A request the router could not analyse neither serves nor drops a hold."""
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=1)

    def break_signals(*args: Any, **kwargs: Any) -> Signals:
        raise RuntimeError("signals exploded")

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1

        monkeypatch.setattr(proxy_module, "compute_signals", break_signals)
        body = stay_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.error == "signals_failed"
        assert row.applied == 0
        assert REASON_HELD_MODEL not in row.reason_codes

        state = hold_of(harness.store, log)
        assert state.has_hold is True
        assert state.held_requests == 0, "the failed request was not served from it"
        assert state.requests_seen == 1, "an unanalysed request is not counted"


def test_sessions_do_not_share_holds(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=1)

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID, "first session"))
        assert wait_for_rows(log, 1) == 1
        post_once(harness.proxy_port, stay_body(MODEL_MID, "first session"))
        assert wait_for_rows(log, 2) == 2
        assert json.loads(harness.forwarded.body)["model"] == MODEL_HIGH

        # A different first message is a different session, with no hold of its own.
        body = stay_body(MODEL_MID, "second session")
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 3) == 3

        assert harness.forwarded.body == body
        row = log.recent(1)[0]
        assert row.reason_codes == ["NO_RULE_MATCHED"]
        assert row.applied == 0
        assert row.session_hint != log.recent(3)[2].session_hint
        assert hold_of(harness.store, log, newest=3).has_hold is True
        assert hold_of(harness.store, log).has_hold is False


def test_mode_off_releases_the_hold_and_rewrites_nothing(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = replace(active_config(hold_max=50, dwell_requests=1), mode="off")

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID))
        assert wait_for_rows(log, 1) == 1
        assert log.recent(1)[0].reason_codes == ["MODE_OFF"]
        assert hold_of(harness.store, log).has_hold is False

        body = stay_body(MODEL_MID)
        post_once(harness.proxy_port, body)
        assert wait_for_rows(log, 2) == 2

        assert harness.forwarded.body == body
        assert log.recent(1)[0].applied == 0


def test_the_hold_never_leaks_request_text(tmp_path, monkeypatch):
    log = log_at(tmp_path, monkeypatch)
    config = active_config(hold_max=50, dwell_requests=1)
    phrase = "purple-orchid-trombone-7731"

    with running_proxy(config, log.db_path) as harness:
        post_once(harness.proxy_port, switch_body(MODEL_MID, phrase))
        assert wait_for_rows(log, 1) == 1
        post_once(harness.proxy_port, stay_body(MODEL_MID, phrase))
        assert wait_for_rows(log, 2) == 2

        assert phrase.encode() not in log.db_path.read_bytes()
        for row in log.recent(2):
            assert row.reason_codes == (
                ["ESCALATE_TOOL_ERRORS"] if row.applied == 1 and row.signal_values.get(
                    HELD_SIGNAL_KEY
                ) == 0
                else [REASON_HELD_MODEL]
            )
            assert row.session_hint not in (None, phrase)