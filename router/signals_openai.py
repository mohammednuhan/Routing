"""Difficulty signals for a request in OpenAI chat-completions format.

The OpenAI counterpart of `router/signals.py`: it reads a parsed
`/v1/chat/completions` body and returns the same `Signals` dataclass, so the
policy, safety, hold, breaker and log layers downstream cannot tell which
format produced them. No network I/O, and no request text is read or retained:
every value returned is a number, a boolean, an effort string or None.

Where the two formats differ:

* a tool result is its own message with role `tool`, so `tool_result_count` is
  the number of `tool` messages;
* a tool call is a `tool_calls` entry on an assistant message, each carrying
  `function.name`, so `repeated_tool_call_count` counts consecutive calls by
  name backwards from the most recent one;
* there is no error flag on a tool result in this format. The Anthropic
  Messages API marks a failed `tool_result` with `is_error`; OpenAI does not,
  and the only place a failure could be read from is the tool's own output
  text, which is request content and is never parsed here (Rule 1).
  `last_tool_result_is_error` and `consecutive_tool_errors` are therefore None
  in OpenAI mode, and tool-error escalation consequently cannot fire here: the
  policy escalates only on a count of consecutive tool errors, and an
  undeterminable count is never read as zero errors;
* there is no thinking switch either, so `has_thinking_enabled` is None rather
  than False. None means "this format cannot say", not "thinking was off".

`human_prompt_count` is not one of the differences: it is read by the same
`router/signals.count_human_prompts` helper, because a role `user` message
carrying a string or a `text` part is a human prompt in this format exactly as
it is in the other, and a role `tool` message carries no human text and is not
counted. Both readers therefore answer the same question the same way.

Everything else is the same heuristic as the Anthropic reader, so a request is
scored the same way whichever format it arrived in.
"""
from __future__ import annotations

from dataclasses import replace as dataclasses_replace
from typing import Any

from .signals import Signals, count_human_prompts


def _bytes_len(obj: Any) -> int:
    """The serialized size of the parsed body in bytes.

    The same heuristic `router/signals.py` uses, and the same 0 answer when the
    body cannot be serialized.
    """
    try:
        import json

        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _tool_call_names(messages: list[Any]) -> list[Any]:
    """Every assistant tool call in order, as its `function.name`, or None.

    Only the name is read. Arguments, ids and everything else a tool call
    carries are request content and are never touched (Rule 1). A call that
    does not have the shape this format defines contributes None, which ends a
    run of equal names rather than being counted as one of them.
    """
    names: list[Any] = []
    for message in messages:
        if not isinstance(message, dict) or _get(message, "role") != "assistant":
            continue
        tool_calls = _get(message, "tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            function = _get(call, "function") if isinstance(call, dict) else None
            name = _get(function, "name") if isinstance(function, dict) else None
            names.append(name if isinstance(name, str) else None)
    return names


def _consecutive_tail(names: list[Any]) -> int:
    """How many trailing entries equal the last one. 0 when there are none."""
    if not names:
        return 0
    last = names[-1]
    if not isinstance(last, str):
        return 0
    count = 0
    for name in reversed(names):
        if name == last:
            count += 1
        else:
            break
    return count


def _last_message_tool_state(messages: list[Any]) -> dict[str, bool | None]:
    """`tool_use_pending` and `in_tool_loop` for the final message.

    A call is pending when the conversation ends on an assistant message that
    carries tool calls, because no result can have arrived yet. The loop is
    open when it ends on a tool result, because the model has not answered it.
    Any other ending is neither.

    There is no final message to examine, or it is not one this format defines,
    when the pair is None - the same answer `router/signals.py` gives. None
    means "not determined" and is never read as False.
    """
    last = messages[-1] if messages else None
    if not isinstance(last, dict):
        return {"tool_use_pending": None, "in_tool_loop": None}
    role = _get(last, "role")
    calls = _get(last, "tool_calls")
    return {
        "tool_use_pending": role == "assistant" and isinstance(calls, list) and bool(calls),
        "in_tool_loop": role == "tool",
    }


def _requested_effort(body_json: dict[str, Any]) -> str | None:
    """The effort this format asks for, or None.

    Read from the top-level `reasoning_effort`, or from `reasoning.effort` for
    the providers that nest it. A non-string value is ignored rather than
    coerced: an effort is only ever a value the config already lists (Rule 7),
    so anything else is not one.
    """
    effort = _get(body_json, "reasoning_effort")
    if isinstance(effort, str):
        return effort
    reasoning = _get(body_json, "reasoning")
    if isinstance(reasoning, dict):
        nested = _get(reasoning, "effort")
        if isinstance(nested, str):
            return nested
    return None


def compute_signals_openai(body_json: Any) -> Signals:
    """Compute signals from a parsed OpenAI-format request body.

    Never raises: anything unreadable leaves its signal at None, which the
    policy reads as "not determined" rather than as a zero. Rule 3.
    """
    try:
        if body_json is None or not isinstance(body_json, dict):
            return Signals()

        signals = Signals(
            context_tokens_estimate=_bytes_len(body_json) // 4,
            # The OpenAI format carries no error flag on a tool result and no
            # thinking switch. Both stay None, which is why tool-error
            # escalation cannot fire in OpenAI mode: see the module docstring.
            last_tool_result_is_error=None,
            consecutive_tool_errors=None,
            has_thinking_enabled=None,
            requested_effort_if_present=_requested_effort(body_json),
        )

        messages = _get(body_json, "messages")
        if not isinstance(messages, list):
            return signals

        return dataclasses_replace(
            signals,
            message_count=len(messages),
            human_prompt_count=count_human_prompts(messages),
            turn_index=sum(
                1 for m in messages if isinstance(m, dict) and _get(m, "role") == "assistant"
            ),
            tool_result_count=sum(
                1 for m in messages if isinstance(m, dict) and _get(m, "role") == "tool"
            ),
            repeated_tool_call_count=_consecutive_tail(_tool_call_names(messages)),
            **_last_message_tool_state(messages),
        )
    except Exception:
        return Signals()