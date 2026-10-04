"""Safety-layer tests.

These are pure-function tests: `apply_safety` is called directly with a
decision, signals, a session state and a config. The only network test is the
last one, which drives the proxy through the in-process harness the other proxy
tests already use.

Prices come from `tools/sample-config-DUMMY-prices.yaml`, never from the shipped
config, whose prices are deliberately null. No server is started here.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import threading
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from router.config import RouterConfig, load_config
from router.decisions import DB_ENV_VAR, MESSAGES_PATH, DecisionLog
from router.policy import Decision
from router import proxy as proxy_module
from router.proxy import LOOPBACK_HOST, ProxySettings, create_server
from router.safety import (
    BLOCKED_COST,
    BLOCKED_COST_UNKNOWN,
    BLOCKED_DWELL,
    BLOCKED_HYSTERESIS,
    BLOCKED_TOOL_LOOP,
    BLOCKED_TOOL_USE_PENDING,
    COST_SIGNAL_KEY,
    apply_safety,
    is_blocked,
)
from router.signals import Signals, compute_signals
from router.state import DOWN, UP, MAX_SESSIONS, SessionState, SessionStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMMY_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-DUMMY-prices.yaml"

MODEL_LOW = "TODO_MODEL_TIER_LOW"
MODEL_MID = "TODO_MODEL_TIER_MID"
MODEL_HIGH = "TODO_MODEL_TIER_HIGH"

# Dummy write 3.75, dummy read 0.30 -> a rebuild costs (3.75 - 0.30) per million
# tokens, i.e. USD 0.00000345 per token.
DUMMY_SPREAD = 3.75 - 0.30
DUMMY_BENEFIT = 0.001

# The dummy benefit of USD 0.001 is outweighed above
# 0.001 * 1_000_000 / 3.45 = 289 tokens of context.
CHEAP_TOKENS = 200
EXPENSIVE_TOKENS = 1_000
# Just under the crossover, so a small margin tip is enough to block it.
MARGINAL_TOKENS = 280
SMALL_MARGIN_USD = 0.0001


def dummy_config() -> RouterConfig:
    """The shipped config with dummy prices substituted for the null ones."""
    return load_config(DUMMY_CONFIG_PATH)


def switch_to(model: str, direction: str, effort: str | None = None) -> Decision:
    """A SWITCH decision as `decide` would have produced."""
    return Decision(
        action="SWITCH",
        target_model=model,
        target_effort=effort,
        reason_codes=["ESCALATE_TOOL_ERRORS"],
        direction=direction,
    )


def clean_signals(tokens: int = CHEAP_TOKENS, **overrides: Any) -> Signals:
    """Signals that trip no safety check on their own.

    The default context is deliberately cheap enough that the dummy benefit
    covers it, so a test that is about dwell or hysteresis is not quietly
    blocked by the cost check instead.
    """
    base = Signals(
        context_tokens_estimate=tokens,
        tool_use_pending=False,
        in_tool_loop=False,
        consecutive_tool_errors=2,
    )
    return replace(base, **overrides)


def fresh_state(requests_seen: int = 1, direction: str | None = None, index: int | None = None) -> SessionState:
    return SessionState(
        requests_seen=requests_seen,
        last_switch_request_index=index,
        last_switch_direction=direction,
    )


def blocking_code(decision: Decision) -> str | None:
    codes = [c for c in (decision.reason_codes or []) if is_blocked([c])]
    assert len(codes) <= 1, f"exactly one blocking code expected, got {codes}"
    return codes[0] if codes else None


def blocking_code_reason(reason_codes: list[str]) -> str | None:
    """The single `BLOCKED_*` code in a row's reasons, if there is one."""
    for code in reason_codes:
        if is_blocked([code]):
            return code
    return None


# --- non-SWITCH decisions pass through untouched ---------------------------


def test_stay_decision_passes_through_untouched():
    config = dummy_config()
    stay = Decision(action="STAY", target_model=MODEL_MID, reason_codes=["NO_RULE_MATCHED"])
    state = fresh_state(requests_seen=3)

    final, new_state = apply_safety(stay, Signals(), state, config)

    assert final == stay
    assert new_state == state


def test_safety_never_creates_a_switch():
    """Whatever the signals and state, the result is never a new switch."""
    config = dummy_config()
    stay = Decision(action="STAY", target_model=MODEL_MID, reason_codes=["NO_RULE_MATCHED"])

    for signals in (
        Signals(),
        clean_signals(),
        clean_signals(tool_use_pending=True),
        clean_signals(context_tokens_estimate=10_000_000),
    ):
        final, _ = apply_safety(stay, signals, fresh_state(), config)
        assert final.action == "STAY"
        assert not is_blocked(final.reason_codes)


def test_switch_is_allowed_when_nothing_blocks_it():
    config = dummy_config()

    final, state = apply_safety(switch_to(MODEL_HIGH, UP), clean_signals(), fresh_state(), config)

    assert final.action == "SWITCH"
    assert not is_blocked(final.reason_codes)
    assert state.last_switch_direction == UP
    assert state.last_switch_request_index == 1


# --- a. pending tool use ---------------------------------------------------


def test_pending_tool_use_blocks():
    config = dummy_config()

    final, _ = apply_safety(
        switch_to(MODEL_HIGH, UP),
        clean_signals(tool_use_pending=True),
        fresh_state(),
        config,
    )

    assert final.action == "STAY"
    assert blocking_code(final) == BLOCKED_TOOL_USE_PENDING


def test_blocked_keeps_original_reasons_and_adds_exactly_one_code():
    config = dummy_config()
    decision = switch_to(MODEL_HIGH, UP)
    decision = replace(decision, reason_codes=["ESCALATE_TOOL_ERRORS", "RULE_EXTRA"])

    final, _ = apply_safety(decision, clean_signals(tool_use_pending=True), fresh_state(), config)

    assert final.reason_codes == [
        "ESCALATE_TOOL_ERRORS",
        "RULE_EXTRA",
        BLOCKED_TOOL_USE_PENDING,
    ]


def test_undeterminable_tool_use_pending_is_not_treated_as_pending():
    config = dummy_config()

    final, _ = apply_safety(
        switch_to(MODEL_HIGH, UP),
        clean_signals(tool_use_pending=None),
        fresh_state(),
        config,
    )

    assert final.action == "SWITCH"


# --- b. tool loop ----------------------------------------------------------


def test_tool_loop_blocks_only_when_configured():
    config = dummy_config()
    config = replace(config, policy=replace(config.policy, block_in_tool_loop=False))
    signals = clean_signals(in_tool_loop=True)

    allowed, _ = apply_safety(switch_to(MODEL_HIGH, UP), signals, fresh_state(), config)
    assert allowed.action == "SWITCH"

    config = replace(config, policy=replace(config.policy, block_in_tool_loop=True))
    blocked, _ = apply_safety(switch_to(MODEL_HIGH, UP), signals, fresh_state(), config)
    assert blocked.action == "STAY"
    assert blocking_code(blocked) == BLOCKED_TOOL_LOOP


# --- c. dwell --------------------------------------------------------------


def test_dwell_blocks_a_second_switch_then_releases_after_n():
    config = dummy_config()
    dwell = config.policy.dwell_requests

    # First switch of the session: nothing to rate-limit against.
    first, state = apply_safety(switch_to(MODEL_HIGH, UP), clean_signals(), fresh_state(), config)
    assert first.action == "SWITCH"
    assert state.last_switch_direction == UP

    # Requests that arrive too soon are blocked; the last one allowed is at
    # exactly `dwell` requests after the switch.
    for requests_seen in range(1, dwell):
        blocked, _ = apply_safety(
            switch_to(MODEL_HIGH, UP),
            clean_signals(),
            fresh_state(requests_seen=1 + requests_seen, direction=UP, index=1),
            config,
        )
        assert blocked.action == "STAY"
        assert blocking_code(blocked) == BLOCKED_DWELL

    released, _ = apply_safety(
        switch_to(MODEL_HIGH, UP),
        clean_signals(),
        fresh_state(requests_seen=1 + dwell, direction=UP, index=1),
        config,
    )
    assert released.action == "SWITCH"


def test_dwell_does_not_block_when_no_switch_has_happened():
    config = dummy_config()
    state = fresh_state(requests_seen=99)

    final, _ = apply_safety(switch_to(MODEL_HIGH, UP), clean_signals(), state, config)

    assert final.action == "SWITCH"


# --- d. hysteresis ---------------------------------------------------------


def test_hysteresis_blocks_an_immediate_reversal():
    config = dummy_config()
    needed = config.policy.hysteresis_requests
    state = fresh_state(requests_seen=1 + config.policy.dwell_requests, direction=UP, index=1)

    for attempt in range(1, needed):
        blocked, _ = apply_safety(switch_to(MODEL_LOW, DOWN), clean_signals(), state, config)
        assert blocked.action == "STAY"
        assert blocking_code(blocked) == BLOCKED_HYSTERESIS


def test_hysteresis_allows_after_enough_consecutive_proposals():
    config = dummy_config()
    needed = config.policy.hysteresis_requests
    state = SessionState(
        requests_seen=1 + config.policy.dwell_requests,
        last_switch_request_index=1,
        last_switch_direction=UP,
        opposite_streak=needed - 1,
    )

    final, updated = apply_safety(switch_to(MODEL_LOW, DOWN), clean_signals(), state, config)

    assert final.action == "SWITCH"
    assert updated.last_switch_direction == DOWN
    assert updated.opposite_streak == 0


def test_hysteresis_streak_counts_up_across_consecutive_proposals():
    config = dummy_config()
    needed = config.policy.hysteresis_requests
    state = SessionState(
        requests_seen=1 + config.policy.dwell_requests,
        last_switch_request_index=1,
        last_switch_direction=UP,
    )

    for attempt in range(1, needed + 1):
        final, state = apply_safety(switch_to(MODEL_LOW, DOWN), clean_signals(), state, config)
        if attempt < needed:
            assert final.action == "STAY"
            assert blocking_code(final) == BLOCKED_HYSTERESIS
            assert state.opposite_streak == attempt
        else:
            assert final.action == "SWITCH"
            assert state.opposite_streak == 0
            assert state.last_switch_direction == DOWN


def test_hysteresis_streak_resets_when_the_same_direction_is_proposed_again():
    config = dummy_config()
    needed = config.policy.hysteresis_requests
    state = SessionState(
        requests_seen=1 + config.policy.dwell_requests,
        last_switch_request_index=1,
        last_switch_direction=UP,
        opposite_streak=needed - 1,
    )

    # Proposing "up" again is not a reversal, so the streak resets. This one is
    # too expensive to be allowed, which keeps dwell released for the next
    # proposal rather than re-arming it.
    blocked_up, after_same = apply_safety(
        switch_to(MODEL_HIGH, UP), clean_signals(EXPENSIVE_TOKENS), state, config
    )
    assert blocking_code(blocked_up) == BLOCKED_COST
    assert after_same.opposite_streak == 0

    # The next reversal therefore starts from zero again.
    blocked, _ = apply_safety(switch_to(MODEL_LOW, DOWN), clean_signals(), after_same, config)
    assert blocked.action == "STAY"
    assert blocking_code(blocked) == BLOCKED_HYSTERESIS


def test_hysteresis_does_not_apply_to_a_repeat_of_the_same_direction():
    config = dummy_config()
    state = SessionState(
        requests_seen=1 + config.policy.dwell_requests,
        last_switch_request_index=1,
        last_switch_direction=UP,
    )

    final, _ = apply_safety(switch_to(MODEL_HIGH, UP), clean_signals(), state, config)

    assert final.action == "SWITCH"


# --- e. cost ---------------------------------------------------------------


def rebuild_cost(tokens: int) -> float:
    return tokens / 1_000_000 * DUMMY_SPREAD


#: Pads the request past the cost crossover. Plain filler, no secrets.
FILLER_TEXT = "padding " * 400


def test_cost_blocks_when_cost_exceeds_benefit():
    config = dummy_config()
    tokens = EXPENSIVE_TOKENS
    assert rebuild_cost(tokens) > DUMMY_BENEFIT

    final, _ = apply_safety(switch_to(MODEL_HIGH, UP), clean_signals(tokens), fresh_state(), config)

    assert final.action == "STAY"
    assert blocking_code(final) == BLOCKED_COST
    assert final.estimated_rebuild_cost_usd == pytest.approx(rebuild_cost(tokens))


def test_cost_allows_when_benefit_exceeds_cost_plus_margin():
    config = dummy_config()
    tokens = CHEAP_TOKENS
    assert DUMMY_BENEFIT > rebuild_cost(tokens) + config.policy.safety_margin_usd

    final, state = apply_safety(switch_to(MODEL_HIGH, UP), clean_signals(tokens), fresh_state(), config)

    assert final.action == "SWITCH"
    assert final.estimated_rebuild_cost_usd == pytest.approx(rebuild_cost(tokens))
    assert state.last_switch_direction == UP


def test_safety_margin_can_block_what_would_otherwise_pass():
    config = dummy_config()
    tokens = MARGINAL_TOKENS
    # without the margin this passes
    assert DUMMY_BENEFIT > rebuild_cost(tokens)
    policy = replace(config.policy, safety_margin_usd=SMALL_MARGIN_USD)
    config = replace(config, policy=policy)
    # with it, the benefit no longer clears
    assert not DUMMY_BENEFIT > rebuild_cost(tokens) + policy.safety_margin_usd

    final, _ = apply_safety(switch_to(MODEL_HIGH, UP), clean_signals(tokens), fresh_state(), config)

    assert final.action == "STAY"
    assert blocking_code(final) == BLOCKED_COST


def test_null_prices_block_as_cost_unknown():
    config = load_config()  # the shipped config: every price is null

    final, _ = apply_safety(switch_to(MODEL_HIGH, UP), clean_signals(), fresh_state(), config)

    assert final.action == "STAY"
    assert blocking_code(final) == BLOCKED_COST_UNKNOWN
    assert final.estimated_rebuild_cost_usd == "UNKNOWN"


def test_null_benefit_blocks_as_cost_unknown_but_reports_the_cost():
    config = dummy_config()
    policy = replace(config.policy, escalate_benefit_usd=None)
    config = replace(config, policy=policy)
    tokens = CHEAP_TOKENS

    final, _ = apply_safety(switch_to(MODEL_HIGH, UP), clean_signals(tokens), fresh_state(), config)

    assert final.action == "STAY"
    assert blocking_code(final) == BLOCKED_COST_UNKNOWN
    assert final.estimated_rebuild_cost_usd == pytest.approx(rebuild_cost(tokens))


def test_null_benefit_downgrade_blocks_as_cost_unknown():
    config = dummy_config()
    policy = replace(config.policy, downgrade_benefit_usd=None)
    config = replace(config, policy=policy)

    final, _ = apply_safety(switch_to(MODEL_LOW, DOWN), clean_signals(), fresh_state(), config)

    assert final.action == "STAY"
    assert blocking_code(final) == BLOCKED_COST_UNKNOWN


def test_missing_context_tokens_blocks_as_cost_unknown():
    config = dummy_config()

    final, _ = apply_safety(
        switch_to(MODEL_HIGH, UP),
        clean_signals(context_tokens_estimate=None),
        fresh_state(),
        config,
    )

    assert final.action == "STAY"
    assert blocking_code(final) == BLOCKED_COST_UNKNOWN
    assert final.estimated_rebuild_cost_usd == "UNKNOWN"


def test_cost_check_disabled_skips_the_cost_block():
    config = dummy_config()
    policy = replace(config.policy, cost_check_enabled=False)
    config = replace(config, policy=policy)

    final, _ = apply_safety(
        switch_to(MODEL_HIGH, UP),
        clean_signals(EXPENSIVE_TOKENS),
        fresh_state(),
        config,
    )

    assert final.action == "SWITCH"
    assert final.estimated_rebuild_cost_usd is None


# --- order of checks -------------------------------------------------------


def test_pending_tool_use_beats_every_other_block():
    config = dummy_config()
    state = SessionState(
        requests_seen=1,
        last_switch_request_index=1,
        last_switch_direction=UP,
        opposite_streak=0,
    )

    # Everything below would also block; the first check must win.
    final, _ = apply_safety(
        switch_to(MODEL_LOW, DOWN),
        clean_signals(EXPENSIVE_TOKENS, tool_use_pending=True, in_tool_loop=True),
        state,
        replace(config, policy=replace(config.policy, block_in_tool_loop=True)),
    )

    assert blocking_code(final) == BLOCKED_TOOL_USE_PENDING


def test_tool_loop_beats_dwell():
    config = dummy_config()
    state = fresh_state(requests_seen=1, direction=UP, index=1)

    final, _ = apply_safety(
        switch_to(MODEL_HIGH, UP),
        clean_signals(in_tool_loop=True),
        state,
        replace(config, policy=replace(config.policy, block_in_tool_loop=True)),
    )

    assert blocking_code(final) == BLOCKED_TOOL_LOOP


def test_dwell_beats_hysteresis():
    config = dummy_config()
    state = fresh_state(requests_seen=2, direction=UP, index=1)

    final, _ = apply_safety(
        switch_to(MODEL_LOW, DOWN),
        clean_signals(),
        state,
        replace(config, policy=replace(config.policy, dwell_requests=50)),
    )

    assert blocking_code(final) == BLOCKED_DWELL


def test_hysteresis_beats_cost():
    config = dummy_config()
    state = SessionState(
        requests_seen=1 + config.policy.dwell_requests,
        last_switch_request_index=1,
        last_switch_direction=UP,
        opposite_streak=0,
    )

    # A reversal that is also far too expensive to be worth it.
    final, _ = apply_safety(
        switch_to(MODEL_LOW, DOWN),
        clean_signals(EXPENSIVE_TOKENS),
        state,
        config,
    )

    assert blocking_code(final) == BLOCKED_HYSTERESIS


# --- state -----------------------------------------------------------------


def test_store_counts_requests_per_session():
    store = SessionStore()

    assert store.observe("a").requests_seen == 1
    assert store.observe("a").requests_seen == 2
    assert store.observe("b").requests_seen == 1


def test_store_is_bounded_to_1000_sessions():
    store = SessionStore()

    assert MAX_SESSIONS == 1000
    assert store.max_sessions == 1000

    for index in range(1500):
        store.observe(f"session-{index}")

    assert len(store) == 1000
    # the most recent are kept, the oldest are evicted
    assert "session-1499" in store
    assert "session-0" not in store


def test_store_eviction_keeps_the_bound_under_churn():
    store = SessionStore(max_sessions=10)

    for index in range(100):
        store.observe(f"session-{index % 5}")
        store.observe(f"filler-{index}")

    assert len(store) == 10


def test_store_holds_no_request_content():
    store = SessionStore()
    store.observe("abc123")

    state = store.get("abc123")
    for value in vars(state).values():
        assert isinstance(value, (int, str, type(None)))


def test_store_is_threadsafe():
    store = SessionStore(max_sessions=50)
    errors: list[BaseException] = []

    def worker(offset: int) -> None:
        try:
            for index in range(200):
                store.observe(f"session-{(offset + index) % 20}")
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store) == 20


# --- config ----------------------------------------------------------------


def test_dummy_config_prices_are_dummy_and_non_null():
    config = dummy_config()

    assert config.policy.escalate_benefit_usd == 0.001
    assert config.policy.downgrade_benefit_usd == 0.001
    for spec in config.models:
        assert spec.cache_write_per_million == 3.75
        assert spec.cache_read_per_million == 0.30


def test_shipped_config_prices_are_null():
    config = load_config()

    assert config.policy.escalate_benefit_usd is None
    assert config.policy.downgrade_benefit_usd is None
    for spec in config.models:
        assert spec.cache_write_per_million is None
        assert spec.cache_read_per_million is None


def test_dummy_config_stays_in_shadow_mode():
    assert dummy_config().mode == "shadow"


# --- proxy -----------------------------------------------------------------


@dataclass
class Recorded:
    method: str
    path: str
    body: bytes


class RecordingUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "recording-upstream"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle(self) -> None:
        raw_length = self.headers.get("Content-Length")
        length = int(raw_length) if raw_length else 0
        body = self.rfile.read(length) if length else b""
        self.server.recorded.append(Recorded(self.command, self.path, body))
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


@contextlib.contextmanager
def running_proxy(db_path: Path, config: RouterConfig) -> Iterator[Harness]:
    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), RecordingUpstream)
    upstream.daemon_threads = True
    upstream.recorded = []  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    decisions = DecisionLog(db_path)
    server = create_server(
        ProxySettings(
            listen_port=0,
            upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
            mode="shadow",
            decisions=decisions,
            config=config,
            state=SessionStore(),
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Harness(
            proxy_port=server.server_address[1],
            recorded=upstream.recorded,  # type: ignore[attr-defined]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def post(port: int, path: str, body: bytes) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=5.0)
    try:
        connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
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


def two_tool_errors_body(model: str = MODEL_MID) -> bytes:
    """A body that escalates and is expensive enough to be blocked on cost.

    The user text is padded so that `context_tokens_estimate` (serialized bytes
    // 4) lands well past the ~289 token point where the dummy benefit of
    USD 0.001 stops covering a rebuild. Without the padding the switch would be
    legitimately allowed and there would be no block to observe.
    """
    return json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": FILLER_TEXT}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "search"},
                    ],
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


def test_body_is_actually_expensive_enough_to_block():
    """Guards the premise of the proxy tests below."""
    signals = compute_signals(json.loads(two_tool_errors_body()))

    assert signals.consecutive_tool_errors == 2
    assert signals.tool_use_pending is False
    assert signals.in_tool_loop is True
    assert signals.context_tokens_estimate is not None
    assert rebuild_cost(signals.context_tokens_estimate) > DUMMY_BENEFIT


def test_proxy_logs_a_blocked_switch_with_code_zero_and_identical_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    log = DecisionLog.from_env()
    config = dummy_config()
    body = two_tool_errors_body()

    with running_proxy(log.db_path, config) as harness:
        status, _ = post(harness.proxy_port, MESSAGES_PATH, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1
        row = log.recent(1)[0]

        # cost is far too high for the dummy benefit, so the switch is blocked
        assert row.action == "STAY"
        assert row.applied == 0
        assert is_blocked(row.reason_codes)
        assert blocking_code_reason(row.reason_codes) == BLOCKED_COST
        assert COST_SIGNAL_KEY in row.signal_values
        # the bytes the upstream received are exactly the bytes we sent
        assert harness.recorded[-1].body == body


def test_proxy_blocked_switch_keeps_the_escalation_reason(tmp_path, monkeypatch):
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    log = DecisionLog.from_env()

    with running_proxy(log.db_path, dummy_config()) as harness:
        post(harness.proxy_port, MESSAGES_PATH, two_tool_errors_body())
        assert wait_for_rows(log, 1) == 1
        row = log.recent(1)[0]

        assert row.reason_codes[:1] == ["ESCALATE_TOOL_ERRORS"]
        assert row.reason_codes[-1] == BLOCKED_COST
        assert len(row.reason_codes) == 2


def test_proxy_safety_never_rewrites_the_model_or_the_body(tmp_path, monkeypatch):
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    log = DecisionLog.from_env()

    with running_proxy(log.db_path, dummy_config()) as harness:
        post(harness.proxy_port, MESSAGES_PATH, two_tool_errors_body())
        assert wait_for_rows(log, 1) == 1
        row = log.recent(1)[0]

        assert row.requested_model == MODEL_MID
        assert row.chosen_model == MODEL_MID
        assert row.applied == 0
        assert row.error is None


def test_proxy_safety_failure_is_recorded_and_request_still_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    log = DecisionLog.from_env()
    body = two_tool_errors_body()

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("safety exploded")

    monkeypatch.setattr(proxy_module, "apply_safety", boom)

    with running_proxy(log.db_path, dummy_config()) as harness:
        status, payload = post(harness.proxy_port, MESSAGES_PATH, body)

        # Rule 3: a router failure never changes the response.
        assert status == 200
        assert json.loads(payload) == {"ok": True}
        assert wait_for_rows(log, 1) == 1
        row = log.recent(1)[0]

        assert row.action == "STAY"
        assert row.applied == 0
        assert row.reason_codes == ["SAFETY_ERROR"]
        assert row.error == "safety_failed"
        assert row.chosen_model == MODEL_MID
        assert harness.recorded[-1].body == body


def test_proxy_without_a_state_store_still_records_the_decision(tmp_path, monkeypatch):
    """No session store means no safety, not a failure."""
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "router.sqlite3"))
    log = DecisionLog.from_env()
    body = two_tool_errors_body()

    upstream = ThreadingHTTPServer((LOOPBACK_HOST, 0), RecordingUpstream)
    upstream.daemon_threads = True
    upstream.recorded = []  # type: ignore[attr-defined]
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    server = create_server(
        ProxySettings(
            listen_port=0,
            upstream=f"http://{LOOPBACK_HOST}:{upstream.server_address[1]}",
            mode="shadow",
            decisions=log,
            config=dummy_config(),
            state=None,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = post(server.server_address[1], MESSAGES_PATH, body)
        assert status == 200
        assert wait_for_rows(log, 1) == 1
        row = log.recent(1)[0]
        assert row.action == "SWITCH"
        assert row.applied == 0
        assert row.error is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)