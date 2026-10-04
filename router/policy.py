"""Routing policy (shadow behavior only)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import ModelSpec, PolicySpec, RouterConfig
from .signals import Signals


@dataclass(frozen=True)
class Decision:
    """Routing decision."""

    action: str  # "STAY" or "SWITCH"
    target_model: str | None = None
    target_effort: str | None = None
    reason_codes: list[str] | None = None


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
            return Decision(action="SWITCH", target_model=target_model, target_effort=None, reason_codes=["ESCALATE_TOOL_ERRORS"])
        return Decision(action="SWITCH", target_model=target_model, target_effort=target_effort, reason_codes=["ESCALATE_TOOL_ERRORS"])
    if signals.repeated_tool_call_count is not None and signals.repeated_tool_call_count >= policy.escalate_repeated_tool_calls:
        higher = _next_higher(specs_list, requested_model)
        if higher is None:
            return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["ALREADY_HIGHEST"])
        target_model = higher.id
        target_effort = requested_effort
        if target_effort is not None and not config.is_legal(target_model, target_effort):
            return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["ILLEGAL_PAIR"])
        if target_effort is None:
            return Decision(action="SWITCH", target_model=target_model, target_effort=None, reason_codes=["ESCALATE_REPEATED_TOOLS"])
        return Decision(action="SWITCH", target_model=target_model, target_effort=target_effort, reason_codes=["ESCALATE_REPEATED_TOOLS"])
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
                    return Decision(action="SWITCH", target_model=target_model, target_effort=None, reason_codes=["DOWNGRADE_SMALL_CONTEXT"])
                return Decision(action="SWITCH", target_model=target_model, target_effort=target_effort, reason_codes=["DOWNGRADE_SMALL_CONTEXT"])
        else:
            return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["NO_RULE_MATCHED", "SIGNAL_UNKNOWN"])
    return Decision(action="STAY", target_model=requested_model, target_effort=requested_effort, reason_codes=["NO_RULE_MATCHED"])
