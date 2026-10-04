"""Safety rules that can only BLOCK a switch.

`apply_safety` takes the decision the policy layer produced and may turn a
SWITCH into a STAY. It can never do the opposite: there is no path through this
module that turns a STAY into a SWITCH, and none may be added. Rule 2.

The checks run in a fixed order and the first one that fails is the one that is
reported. A blocked decision keeps every reason the policy layer produced and
gains exactly one `BLOCKED_*` code, so a row always says what the router wanted
and then why it did not do it.

Cost is UNKNOWN unless every input is present. An unknown price is never
treated as zero and never treated as free: it blocks, because a switch whose
cost cannot be established is a switch nobody can justify.

This module is pure. It performs no I/O, keeps no state of its own, and does
not mutate the `Decision` or the `SessionState` it is given; both come back as
new values.
"""
from __future__ import annotations

from dataclasses import replace

from .config import ModelSpec, PolicySpec, RouterConfig
from .policy import Decision
from .signals import Signals
from .state import DOWN, UP, SessionState

#: Every code this module can add, in the order the checks run.
BLOCKED_TOOL_USE_PENDING = "BLOCKED_TOOL_USE_PENDING"
BLOCKED_TOOL_LOOP = "BLOCKED_TOOL_LOOP"
BLOCKED_DWELL = "BLOCKED_DWELL"
BLOCKED_SWITCH_CAP = "BLOCKED_SWITCH_CAP"
BLOCKED_HYSTERESIS = "BLOCKED_HYSTERESIS"
BLOCKED_COST = "BLOCKED_COST"
BLOCKED_COST_UNKNOWN = "BLOCKED_COST_UNKNOWN"

#: Every blocking code, as a prefix test for callers such as `tamias-router log`.
BLOCKED_PREFIX = "BLOCKED_"
BLOCKED_CODES: tuple[str, ...] = (
    BLOCKED_TOOL_USE_PENDING,
    BLOCKED_TOOL_LOOP,
    BLOCKED_DWELL,
    BLOCKED_SWITCH_CAP,
    BLOCKED_HYSTERESIS,
    BLOCKED_COST,
    BLOCKED_COST_UNKNOWN,
)

#: Recorded in `signal_values` when a cost could not be computed at all.
COST_UNKNOWN = "UNKNOWN"

#: Key the proxy writes the estimate into `signal_values` under.
COST_SIGNAL_KEY = "estimated_rebuild_cost_usd"

#: Tokens per million, the unit the configured prices are quoted in.
TOKENS_PER_MILLION = 1_000_000


def is_blocked(reason_codes: object) -> bool:
    """True when any reason code is a `BLOCKED_*` safety block."""
    if not isinstance(reason_codes, (list, tuple)):
        return False
    return any(
        isinstance(code, str) and code.startswith(BLOCKED_PREFIX) for code in reason_codes
    )


def safety_failure(decision: Decision) -> Decision:
    """The decision to record when the safety layer itself raised.

    STAY, with `SAFETY_ERROR` and no cost claim. A router that cannot check a
    switch must not make it.
    """
    return replace(decision, action="STAY", reason_codes=["SAFETY_ERROR"])


def apply_safety(
    decision: Decision,
    signals: Signals,
    session_state: SessionState,
    config: RouterConfig,
) -> tuple[Decision, SessionState]:
    """Apply every safety check to `decision`.

    Returns the final decision and the state to store. A decision that is not a
    SWITCH is returned untouched, with the state it was given.
    """
    if decision.action != "SWITCH":
        return (decision, session_state)

    policy = config.policy
    if policy is None:
        return (decision, session_state)

    # Confirming a reversal is a separate axis from rate-limiting switches, so
    # the streak is advanced for every switch proposal, including one that a
    # later check blocks. Reset it whenever the proposal is not a reversal.
    state = _advance_opposite_streak(decision, session_state)

    # a. A tool call with no result yet: switching models mid tool call changes
    #    what the pending call was issued against.
    if signals.tool_use_pending is True:
        return (_block(decision, BLOCKED_TOOL_USE_PENDING), state)

    # b. Optional guard for requests that arrive inside an active tool loop.
    if policy.block_in_tool_loop and signals.in_tool_loop is True:
        return (_block(decision, BLOCKED_TOOL_LOOP), state)

    # c. Dwell: do not switch more often than the config allows.
    since = session_state.requests_since_last_switch
    if since is not None and since < policy.dwell_requests:
        return (_block(decision, BLOCKED_DWELL), state)

    # d. Per-session cap: a session may only be moved this many times, however
    #    long it lives. Checked after dwell so a request that is merely too
    #    early still reports the dwell block it hit first.
    if session_state.switch_count >= policy.max_switches_per_session:
        return (_block(decision, BLOCKED_SWITCH_CAP), state)

    # e. Hysteresis: a reversal needs the same proposal several times running.
    if _is_reversal(decision, session_state) and state.opposite_streak < policy.hysteresis_requests:
        return (_block(decision, BLOCKED_HYSTERESIS), state)

    # f. Cost: allow only when the benefit clearly beats the rebuild cost.
    if policy.cost_check_enabled:
        benefit = _benefit_for(policy, decision)
        target = _find_spec(config.models, decision.target_model)
        estimate, blocked = _cost_verdict(policy, signals, target, benefit)
        if estimate is None:
            estimate = COST_UNKNOWN
        if blocked is not None:
            return (_block(decision, blocked, estimate), state)
        return (_allow(decision, estimate), state.recorded_switch(_direction(decision)))

    return (_allow(decision, None), state.recorded_switch(_direction(decision)))


def _block(
    decision: Decision, code: str, estimate: float | str | None = None
) -> Decision:
    """A STAY that keeps the original reasons and adds exactly one code."""
    reasons = list(decision.reason_codes or [])
    reasons.append(code)
    return replace(
        decision,
        action="STAY",
        reason_codes=reasons,
        estimated_rebuild_cost_usd=estimate,
    )


def _allow(decision: Decision, estimate: float | str | None) -> Decision:
    """The switch survives; carry the cost estimate that justified it."""
    return replace(decision, estimated_rebuild_cost_usd=estimate)


def _direction(decision: Decision) -> str | None:
    """Which way this switch goes, as recorded by the policy layer."""
    direction = decision.direction
    return direction if direction in (UP, DOWN) else None


def _is_reversal(decision: Decision, session_state: SessionState) -> bool:
    """True when this switch proposes the opposite of the last allowed one."""
    direction = _direction(decision)
    if direction is None or session_state.last_switch_direction is None:
        return False
    return direction != session_state.last_switch_direction


def _advance_opposite_streak(decision: Decision, session_state: SessionState) -> SessionState:
    """Track consecutive proposals of the opposite direction."""
    if not _is_reversal(decision, session_state):
        return replace(session_state, opposite_streak=0)
    return replace(session_state, opposite_streak=session_state.opposite_streak + 1)


def _benefit_for(policy: PolicySpec, decision: Decision) -> float | None:
    """The configured benefit for this direction, or None when unknown."""
    if _direction(decision) == UP:
        return policy.escalate_benefit_usd
    if _direction(decision) == DOWN:
        return policy.downgrade_benefit_usd
    return None


def _cost_verdict(
    policy: PolicySpec,
    signals: Signals,
    target: ModelSpec | None,
    benefit: float | None,
) -> tuple[float | None, str | None]:
    """`(estimate, blocking_code)` for the cost check.

    `estimate` is None when it could not be computed at all; `blocking_code` is
    None when the switch may proceed.
    """
    write = target.cache_write_per_million if target is not None else None
    read = target.cache_read_per_million if target is not None else None
    tokens = signals.context_tokens_estimate

    if tokens is None or write is None or read is None:
        return (None, BLOCKED_COST_UNKNOWN)
    if benefit is None:
        # The rebuild cost is known; the benefit is not, so the switch cannot
        # be shown to pay for itself.
        return (tokens / TOKENS_PER_MILLION * (write - read), BLOCKED_COST_UNKNOWN)

    estimate = tokens / TOKENS_PER_MILLION * (write - read)
    if benefit > estimate + policy.safety_margin_usd:
        return (estimate, None)
    return (estimate, BLOCKED_COST)


def _find_spec(models: tuple[ModelSpec, ...], model_id: str | None) -> ModelSpec | None:
    if not isinstance(model_id, str):
        return None
    for spec in models:
        if spec.id == model_id:
            return spec
    return None