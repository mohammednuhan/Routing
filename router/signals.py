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
    """Computed difficulty signals. Metadata only.

    `classifier_score` and `classifier_tier` are filled by the proxy from
    `router/classifier.py`, not by `compute_signals`, because only the proxy
    holds the parsed body and the config. Both stay None when the classifier
    section is disabled, when there is no human prompt to classify, and when
    classification failed: an undeterminable value is never guessed.

    `human_prompt_count` is how many messages are human prompts carrying text:
    a message with role `user` whose content is a string or holds a `text`
    block/part. A message made only of `tool_result` blocks and a role `tool`
    message carry no human text and are not counted, so a tool-loop
    continuation scores the same count as the prompt that opened the loop and
    a later human prompt scores a higher one. It is None when the conversation
    cannot be read at all, and no text from any of those messages is kept.
    """

    context_tokens_estimate: int | None = None
    message_count: int | None = None
    human_prompt_count: int | None = None
    tool_result_count: int | None = None
    last_tool_result_is_error: bool | None = None
    consecutive_tool_errors: int | None = None
    repeated_tool_call_count: int | None = None
    turn_index: int | None = None
    has_thinking_enabled: bool | None = None
    requested_effort_if_present: str | None = None
    tool_use_pending: bool | None = None
    in_tool_loop: bool | None = None
    classifier_score: int | None = None
    classifier_tier: str | None = None

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


def _content_has_text(content: Any) -> bool | None:
    """Whether one message's content carries human text, or None.

    A string body carries it, and so does a list holding a `text` block/part -
    the two shapes both accepted formats use for a human prompt. A list with
    no `text` entry in it - an Anthropic message made only of `tool_result`
    blocks, for instance - carries none. Any other shape cannot be read, and
    an unreadable shape is None rather than an answer: an undeterminable
    signal is never guessed (Rule 2). Only block *types* are inspected; no
    block text is read or kept (Rule 1).
    """
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and _get(block, "type") == "text" for block in content
        )
    return None


def count_human_prompts(messages: Any) -> int | None:
    """How many messages are human prompts containing text, or None.

    Only role `user` counts, so an assistant message, an OpenAI `tool` message
    and a `system` message are all skipped, in either request format. The count
    is a number and nothing else: no message text is read beyond its block
    types, and none is returned or kept (Rule 1).

    None means the count could not be determined - `messages` is not a list,
    or a message is not an object this format defines, or a user message's
    content has a shape that cannot be read. None is a different claim from 0
    and must never be read as one.
    """
    if not isinstance(messages, list):
        return None
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            return None
        if _get(message, "role") != "user":
            continue
        has_text = _content_has_text(_get(message, "content"))
        if has_text is None:
            return None
        if has_text:
            count += 1
    return count


def last_message_tool_state(messages: Any) -> tuple[bool | None, bool | None]:
    """`(tool_use_pending, in_tool_loop)` for the final message.

    Both are None when the conversation cannot be examined at all, which is not
    the same as False: an undeterminable signal must never be read as "no tool
    is pending". Only block *types* are inspected; no block text is read or kept.
    """
    if not isinstance(messages, list) or not messages:
        return (None, None)
    last = messages[-1]
    if not isinstance(last, dict):
        return (None, None)
    content = _get(last, "content")
    if not isinstance(content, list):
        # A plain-text message carries no tool blocks of either kind.
        return (False, False)

    has_result = any(
        isinstance(block, dict) and _get(block, "type") == "tool_result" for block in content
    )
    has_use = any(
        isinstance(block, dict) and _get(block, "type") == "tool_use" for block in content
    )
    if has_result:
        return (False, True)
    if has_use and _get(last, "role") == "assistant":
        # The tool call is the last thing in the conversation, so no
        # tool_result can follow it.
        return (True, False)
    return (False, False)


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
        pending, in_loop = last_message_tool_state(messages)
        signals = dataclasses_replace(
            signals, tool_use_pending=pending, in_tool_loop=in_loop
        )
        if isinstance(messages, list):
            signals = dataclasses_replace(
                signals,
                message_count=len(messages),
                human_prompt_count=count_human_prompts(messages),
            )
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
