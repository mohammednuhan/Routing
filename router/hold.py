"""The model hold: remembering the model an allowed switch chose.

In active mode the client keeps sending the model it configured, because nothing
told it the router moved. So the request after an approved switch arrives naming
the original model again, and a policy that looks only at that request and its
immediate signals would put the session straight back where it started.

A hold closes that gap. When safety approves a switch, the target is held for
that session. While the hold is live, a request that names the model the switch
came FROM is answered with the held model instead, under the reason
`HELD_MODEL`. The session stays where the router put it until the hold is
released.

A hold is not a new switch, and it is deliberately not treated as one:

* it does not pass the dwell, hysteresis or cost checks, because the first
  switch already paid the rebuild cost and already passed every one of them;
* it does not increment `switch_count`, so the per-session cap counts switches
  and not held requests;
* it does not reset dwell, so the dwell window the first switch opened still
  runs.

It is still a switch, and still has to be one the rules allow: the held model
must be a model the config lists (Rule 7), and it can only be produced from a
STAY the policy would otherwise have made. An unknown model is never held.

Release happens when the hold stops describing reality:

* the policy proposes a switch of its own that safety approves - the new switch
  replaces the old hold;
* the hold has served `hold_max_requests` requests, so it cannot outlive the
  conditions that justified it;
* the client asks for a different model, so the hold was about a model this
  request is not using;
* the session has used up `max_switches_per_session` and the cap blocked the
  switch the policy proposed - a cap on how often a session may be moved is not
  sidestepped by moving it anyway;
* the kill switch is on, the circuit breaker is open, or the mode is `off` for
  that request - while the router is not allowed to act, nothing is held, and a
  hold kept across such a request would be a claim about a request the router
  never made.

Nothing here stores request content. Rule 1.
"""
from __future__ import annotations

from dataclasses import replace

from .config import DEFAULT_HOLD_MAX_REQUESTS, RouterConfig
from .policy import Decision
from .safety import BLOCKED_SWITCH_CAP
from .state import SessionState

#: Reason code recorded on every row produced by a hold.
REASON_HELD_MODEL = "HELD_MODEL"

#: Key the proxy writes the served-request count into `signal_values` under.
HELD_SIGNAL_KEY = "held_requests"

#: The one mode in which nothing may be held.
MODE_OFF = "off"


def hold_max_requests(config: RouterConfig) -> int:
    """How many requests one hold may serve. From the policy, else the default."""
    policy = getattr(config, "policy", None)
    value = getattr(policy, "hold_max_requests", None)
    return value if isinstance(value, int) and value > 0 else DEFAULT_HOLD_MAX_REQUESTS


def apply_hold(
    decision: Decision,
    requested_model: str,
    requested_effort: str | None,
    state: SessionState,
    config: RouterConfig,
    mode: str = "active",
) -> tuple[Decision, SessionState]:
    """The decision a live hold implies, and the state to store.

    Called only after the safety layer has run, so `decision` is safety's final
    answer: a SWITCH it approved, or a STAY it did not turn into a switch.
    A decision that is neither leaves the hold alone.
    """
    if mode == MODE_OFF:
        # The router is off for this request. Whatever it was holding describes
        # a hop it is not making now.
        return (decision, state.released_hold())

    if decision.action == "SWITCH":
        target = decision.target_model
        if not isinstance(target, str) or not _is_legal(config, target):
            # A switch to a model the config does not list is not one this
            # router may make (Rule 7), so there is nothing to hold.
            return (decision, state.released_hold())
        # An approved switch is the truth about this session, so it becomes the
        # hold and the previous one, if any, is gone.
        return (decision, state.recorded_hold(target, requested_model))

    if not state.has_hold:
        return (decision, state)
    if requested_model != state.held_from_model:
        # The client moved on to another model. The hold was about this one.
        return (decision, state.released_hold())
    if state.held_requests >= hold_max_requests(config):
        return (decision, state.released_hold())
    if _cap_blocked(decision):
        # A BLOCKED_SWITCH_CAP stay releases the hold, deliberately. A hold is a
        # change of model, so serving one here would make the session's switching
        # budget unenforceable for exactly the sessions it exists to bound.
        return (decision, state.released_hold())
    if not _is_legal(config, state.held_model) or not _is_legal(config, requested_model):
        # Rule 7, on both ends: nothing outside the config is ever held, and a
        # hold is never served onto a request naming a model outside it.
        return (decision, state.released_hold())

    held = Decision(
        action="SWITCH",
        target_model=state.held_model,
        target_effort=requested_effort,
        reason_codes=[REASON_HELD_MODEL],
        # `direction` stays None: it exists so the safety layer can rate-limit a
        # switch without re-deriving the tiers, and the safety layer has already
        # run and deliberately declined to switch here.
        direction=None,
    )
    return (held, _served(state))


def _served(state: SessionState) -> SessionState:
    """The state after this request was served from the hold."""
    return replace(state, held_requests=state.held_requests + 1)


def _cap_blocked(decision: Decision) -> bool:
    """True when safety blocked this STAY because the session is out of budget."""
    codes = decision.reason_codes or []
    return isinstance(codes, (list, tuple)) and BLOCKED_SWITCH_CAP in codes


def _is_legal(config: RouterConfig, model: object) -> bool:
    """True when `model` is one of the config's model ids."""
    if not isinstance(model, str):
        return False
    models = getattr(config, "model_ids", None)
    return isinstance(models, tuple) and model in models