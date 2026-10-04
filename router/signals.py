"""Difficulty signals for a request.

Compute difficulty signals from a parsed /v1/messages request (Anthropic
Messages format). This module has no network I/O and never retains request text.
All returned values are numbers or booleans, or None when they cannot be
determined. No text from the request may be retained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace as dataclasses_replace
from typing import Any


@dataclass(frozen=True)
class Signals:
    """Computed difficulty signals. Metadata only."""

    context_tokens_estimate: int | None = None
    message_count: int | None = None
    tool_result_count: int | None = None
    last_tool_result_is_error: bool | None = None
    consecutive_tool_errors: int | None = None
    repeated_tool_call_count: int | None = None
    turn_index: int | None = None
    has_thinking_enabled: bool | None = None
    requested_effort_if_present: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bytes_len(obj: Any) -> int:
    try:
        import json

        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def compute_signals(body_json: Any) -> Signals:
    """Compute signals from parsed request body."""
    try:
        if body_json is None:
            return Signals()
        # For malformed non-dict inputs, return empty signals
        if not isinstance(body_json, dict):
            return Signals()

        signals = Signals()
        # context_tokens_estimate: len of the serialized JSON bytes // 4
        try:
            signals = dataclasses_replace(signals, context_tokens_estimate=_bytes_len(body_json) // 4)
        except Exception:
            pass

        messages = _get(body_json, "messages")
        if isinstance(messages, list):
            signals = dataclasses_replace(signals, message_count=len(messages))
            try:
                assistant_count = sum(1 for m in messages if isinstance(m, dict) and _get(m, "role") == "assistant")
                signals = dataclasses_replace(signals, turn_index=assistant_count)
            except Exception:
                pass

            tool_results = []
            tool_uses = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                content = _get(m, "content")
                blocks = []
                if isinstance(content, list):
                    blocks = content
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    btype = _get(b, "type")
                    if btype == "tool_result":
                        tool_results.append(_get(b, "is_error") is True)
                    elif btype == "tool_use":
                        name = _get(b, "name")
                        if isinstance(name, str):
                            tool_uses.append(name)
                        else:
                            tool_uses.append(None)

            if tool_results:
                signals = dataclasses_replace(signals, tool_result_count=len(tool_results))
                last = tool_results[-1]
                signals = dataclasses_replace(signals, last_tool_result_is_error=bool(last))
                consec = 0
                for tr in reversed(tool_results):
                    if tr is True:
                        consec += 1
                    else:
                        break
                signals = dataclasses_replace(signals, consecutive_tool_errors=consec)
            else:
                signals = dataclasses_replace(signals, tool_result_count=0, last_tool_result_is_error=None, consecutive_tool_errors=0)

            if tool_uses:
                last_name = tool_uses[-1]
                if last_name is not None:
                    repeated = 0
                    for n in reversed(tool_uses):
                        if n == last_name:
                            repeated += 1
                        else:
                            break
                    signals = dataclasses_replace(signals, repeated_tool_call_count=repeated)
                else:
                    signals = dataclasses_replace(signals, repeated_tool_call_count=0)
            else:
                signals = dataclasses_replace(signals, repeated_tool_call_count=0)

        thinking = _get(body_json, "thinking")
        if isinstance(thinking, dict):
            ttype = _get(thinking, "type")
            if ttype in ("enabled", "adaptive"):
                signals = dataclasses_replace(signals, has_thinking_enabled=True)
            else:
                signals = dataclasses_replace(signals, has_thinking_enabled=False)
        else:
            signals = dataclasses_replace(signals, has_thinking_enabled=False)

        effort = _get(body_json, "effort")
        if isinstance(effort, str):
            signals = dataclasses_replace(signals, requested_effort_if_present=effort)
        else:
            out_cfg = _get(body_json, "output_config")
            if isinstance(out_cfg, dict):
                e2 = _get(out_cfg, "effort")
                if isinstance(e2, str):
                    signals = dataclasses_replace(signals, requested_effort_if_present=e2)

        return signals
    except Exception:
        return Signals()
