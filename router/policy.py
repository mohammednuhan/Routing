"""Routing policy (shadow behavior only)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import ModelSpec, PolicySpec, RouterConfig
from .signals import Signals

#: Recorded when the classifier's tier is the tier already in use. A STAY, and
#: the only outcome the classifier may produce without switching: agreeing with
#: the request is what Rule 2 asks for by default.
CLASSIFIER_MATCHES_TIER = "CLASSIFIER_MATCHES_TIER"

#: Recorded when the classifier wants a tier no configured model carries. Rule 7
#: forbids inventing a model to satisfy it, so the router stays put.
CLASSIFIER_NO_TARGET = "CLASSIFIER_NO_TARGET"

#: Prefix of the per-tier switch reasons: CLASSIFIER_LOW, CLASSIFIER_MID and
#: CLASSIFIER_HIGH. Derived from the tier so a new tier cannot produce a code
#: that says something else.
CLASSIFIER_REASON_PREFIX = "CLASSIFIER_"


@dataclass(frozen=True)
class Decision:
    """Routing decision.

    `direction` is "up", "down" or None and is recorded here so that the safety
    layer can rate-limit switches without re-deriving the tiers. `policy` owns
    it; only the safety layer reads it.

    `estimated_rebuild_cost_usd` is set by the safety layer, not here, and is a
    number or the string "UNKNOWN". None means the cost was never evaluated.
    """

    action: str  # "STAY" or "SWITCH"
    target_model: str | None = None
    target_effort: str | None = None
    reason_codes: list[str] | None = None
    direction: str | None = None
    estimated_rebuild_cost_usd: float | str | None = None


def _direction_for(specs: list[ModelSpec], source: str, target: str | None) -> str | None:
    """Which way a switch from `source` to `target` goes."""
    if target is None:
        return None
    source_tier = _tier_order_for(specs, source)
    target_tier = _tier_order_for(specs, target)
    if source_tier is None or target_tier is None:
        return None
    if target_tier > source_tier:
        return "up"
    if target_tier < source_tier:
        return "down"
    return None


def _tier_order_for(specs: list[ModelSpec], model_id: str) -> int | None:
    tier_order = {"low": 0, "mid": 1, "high": 2}
    for s in specs:
        if s.id == model_id:
            return tier_order.get(s.cost_tier, -1)
    return None


def _find_spec(specs: list[ModelSpec], model_id: str) -> ModelSpec | None:
    for s in specs:
        if s.id == model_id:
            return s
    return None


def _next_higher(specs: list[ModelSpec], model_id: str) -> ModelSpec | None:
    tier_order = {"low": 0, "mid": 1, "high": 2}
    cur_t = None
    for s in specs:
        if s.id == model_id:
            cur_t = tier_order.get(s.cost_tier, -10**9)
            break
    if cur_t is None:
        return None
    best = None
    best_t = cur_t
    for s in specs:
        t = tier_order.get(s.cost_tier, 10**9)
        if t > cur_t and (best is None or t < best_t):
            best = s
            best_t = t
    return best


def _next_lower(specs: list[ModelSpec], model_id: str) -> ModelSpec | None:
    tier_order = {"low": 0, "mid": 1, "high": 2}
    cur_t = None
    for s in specs:
        if s.id == model_id:
            cur_t = tier_order.get(s.cost_tier, 10**9)
            break
    if cur_t is None:
        return None
    best = None
    best_t = cur_t
    for s in specs:
        t = tier_order.get(s.cost_tier, -10**9)
        if t < cur_t and (best is None or t > best_t):
            best = s
            best_t = t
    return best


def _first_with_tier(specs: list[ModelSpec], tier: str) -> ModelSpec | None:
    """The first model in config order whose `cost_tier` is `tier`.

    Config order, not cheapest and not strongest: the config's order is the
    operator's preference, and the router never re-ranks models itself (Rule 7).
    """
    for spec in specs:
        if spec.cost_tier == tier:
            return spec
    return None


def classifier_decision(
    specs: list[ModelSpec],
    signals: Signals,
    config: RouterConfig,
    requested_model: str,
    requested_effort: str | None,
) -> Decision | None:
    """The decision the classifier implies, or None when it does not apply.

    None means "the classifier did not decide", which is what lets the old
    downgrade rule run exactly as it did before. It is returned when the section
    is disabled, when no prompt could be classified, and when the config holds
    no model of the classified tier at all.

    This rule sits between tool-error escalation and the downgrade rule: a model
    that keeps failing is the stronger evidence, and the old downgrade rule is
    what the classifier replaces while it is enabled. Agreement is a STAY, so
    the common case - a prompt that suits the model already in use - changes
    nothing. Rule 2.
    """
    spec = getattr(config, "classifier", None)
    if spec is None or not getattr(spec, "enabled", False):
        return None
    tier = signals.classifier_tier
    if tier is None:
        return None

    requested = _find_spec(specs, requested_model)
    if requested is None:
        return None

    if requested.cost_tier == tier:
        return Decision(
            action="STAY",
            target_model=requested_model,
            target_effort=requested_effort,
            reason_codes=[CLASSIFIER_MATCHES_TIER],
        )

    target = _first_with_tier(specs, tier)
    if target is None:
        # No configured model carries this tier. Inventing one is forbidden, so
        # the request is passed through unchanged and the reason says why.
        return Decision(
            action="STAY",
            target_model=requested_model,
            target_effort=requested_effort,
            reason_codes=[CLASSIFIER_NO_TARGET],
        )

    if requested_effort is not None and not config.is_legal(target.id, requested_effort):
        return Decision(
            action="STAY",
            target_model=requested_model,
            target_effort=requested_effort,
            reason_codes=["ILLEGAL_PAIR"],
        )

    return Decision(
        action="SWITCH",
        target_model=target.id,
        target_effort=requested_effort,
        reason_codes=[f"{CLASSIFIER_REASON_PREFIX}{str(tier).upper()}"],
        direction=_direction_for(specs, requested_model, target.id),
    )


def decide(signals: Signals, requested_model: str, requested_effort: str | None, config: RouterConfig) -> Decision:
    mode = config.mode
    if mode == "off":
        return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["MODE_OFF"])
    specs_list = list(config.models)
    req_spec = _find_spec(specs_list, requested_model)
    if req_spec is None:
        return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["UNKNOWN_MODEL"])
    if signals is None:
        return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["NO_RULE_MATCHED", "SIGNAL_UNKNOWN"])
    policy = config.policy
    if policy is None:
        return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["NO_RULE_MATCHED"])
    if signals.consecutive_tool_errors is not None and signals.consecutive_tool_errors >= policy.escalate_consecutive_errors:
        higher = _next_higher(specs_list, requested_model)
        if higher is None:
            return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["ALREADY_HIGHEST"])
        target_model = higher.id
        target_effort = requested_effort
        if target_effort is not None and not config.is_legal(target_model, target_effort):
            return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["ILLEGAL_PAIR"])
        if target_effort is None:
            return Decision(action="SWITCH", target_model=target_model, target_effort=None, reason_codes=["ESCALATE_TOOL_ERRORS"], direction=_direction_for(specs_list, requested_model, target_model))
        return Decision(action="SWITCH", target_model=target_model, target_effort=target_effort, reason_codes=["ESCALATE_TOOL_ERRORS"], direction=_direction_for(specs_list, requested_model, target_model))
    if signals.repeated_tool_call_count is not None and signals.repeated_tool_call_count >= policy.escalate_repeated_tool_calls:
        higher = _next_higher(specs_list, requested_model)
        if higher is None:
            return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["ALREADY_HIGHEST"])
        target_model = higher.id
        target_effort = requested_effort
        if target_effort is not None and not config.is_legal(target_model, target_effort):
            return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["ILLEGAL_PAIR"])
        if target_effort is None:
            return Decision(action="SWITCH", target_model=target_model, target_effort=None, reason_codes=["ESCALATE_REPEATED_TOOLS"], direction=_direction_for(specs_list, requested_model, target_model))
        return Decision(action="SWITCH", target_model=target_model, target_effort=target_effort, reason_codes=["ESCALATE_REPEATED_TOOLS"], direction=_direction_for(specs_list, requested_model, target_model))
    classified = classifier_decision(
        specs_list, signals, config, requested_model, requested_effort
    )
    if classified is not None:
        return classified
    if policy.downgrade_enabled:
        if (signals.context_tokens_estimate is not None and signals.turn_index is not None and signals.consecutive_tool_errors is not None):
            if (signals.context_tokens_estimate <= policy.downgrade_max_context_tokens and
                signals.turn_index <= policy.downgrade_max_turn_index and
                signals.consecutive_tool_errors == 0):
                lower = _next_lower(specs_list, requested_model)
                if lower is None:
                    return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["ALREADY_LOWEST"])
                target_model = lower.id
                target_effort = requested_effort
                if target_effort is not None and not config.is_legal(target_model, target_effort):
                    return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["ILLEGAL_PAIR"])
                if target_effort is None:
                    return Decision(action="SWITCH", target_model=target_model, target_effort=None, reason_codes=["DOWNGRADE_SMALL_CONTEXT"], direction=_direction_for(specs_list, requested_model, target_model))
                return Decision(action="SWITCH", target_model=target_model, target_effort=target_effort, reason_codes=["DOWNGRADE_SMALL_CONTEXT"], direction=_direction_for(specs_list, requested_model, target_model))
        else:
            return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["NO_RULE_MATCHED", "SIGNAL_UNKNOWN"])
    return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["NO_RULE_MATCHED"])
