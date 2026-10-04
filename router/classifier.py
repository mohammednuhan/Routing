"""Rule-based task classification from the latest human prompt.

The prompt is read from the already-parsed request body, in memory, for the
length of one call. It is never stored, never logged, never put in an exception
message and never returned: `TaskClass` holds a score, a tier and reason codes
only, so there is nothing in it that could carry request content. Rule 1.

Every rule is a plain, named weight read from the `classifier` section of
`config.yaml`. Nothing here is learned, measured or guessed, and a rule that
cannot be evaluated leaves the score alone rather than substituting a value for
it.

Classification alone never changes a request. The score becomes a signal, and
only `router/policy.py` may act on that signal, and only when the section is
enabled. Rule 2.

This module makes no network calls, reads no file and holds no state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import ClassifierSpec

#: Reason codes. The two keyword codes carry a count; the rest are flags.
#: None of them ever carries the word that matched: that would be request text.
STRONG_KW = "STRONG_KW"
CHEAP_KW = "CHEAP_KW"
LONG_PROMPT = "LONG_PROMPT"
VERY_LONG_PROMPT = "VERY_LONG_PROMPT"
MANY_FILES = "MANY_FILES"
QUESTION_ONLY = "QUESTION_ONLY"

#: The tier labels a score maps to. They are the same `cost_tier` labels the
#: models in the config already carry, so a classified tier names a tier a
#: configured model can already have.
LOW = "low"
MID = "mid"
HIGH = "high"

#: The score is clamped to this range, so a rule can never produce a score that
#: a tier comparison has to reason about.
MIN_SCORE = 0
MAX_SCORE = 100

#: Punctuation stripped from a token's edges before a file name is looked for.
_EDGE_PUNCTUATION = "\"'`()[]{}<>,;:!?*"

#: A file-like token: any run of name characters, a dot, then letters. The
#: extension must start with a letter, so "v1.2.3" and a sentence's trailing
#: full stop are not mistaken for file names.
_FILE_LIKE = re.compile(r"[A-Za-z0-9_\-/\\.]*\.[A-Za-z][A-Za-z0-9]{0,7}")


@dataclass(frozen=True)
class TaskClass:
    """What the rules said about one prompt. Numbers and codes, never text."""

    score: int
    tier: str
    reason_codes: tuple[str, ...]


def classifier_spec(config: Any) -> ClassifierSpec | None:
    """The classifier section to read, or None when it is absent or disabled.

    None means "no classifier": the caller must then leave every signal unset
    and every decision untouched, which is what a disabled section means.
    """
    spec = getattr(config, "classifier", None)
    if spec is None:
        return None
    if not getattr(spec, "enabled", False):
        return None
    return spec


def latest_prompt_text(body_json: Any) -> str | None:
    """The latest human prompt in the body, or None. Held in memory only.

    The last message with role `user` that carries text: a string body, or a
    `text` block. A message made only of `tool_result` blocks carries no human
    text and is skipped, so the prompt being classified during a tool loop is
    the human message that started it and not the tool output that followed.

    Callers must not store, log or return the result. `classify_prompt` is the
    only intended caller, and it keeps nothing.
    """
    if not isinstance(body_json, dict):
        return None
    messages = body_json.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _content_text(message.get("content"))
        if text is not None and text.strip():
            return text
    return None


def _content_text(content: Any) -> str | None:
    """The text of one message: a string body, or its `text` blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        joined = "\n".join(parts)
        return joined or None
    return None


def classify_prompt(body_json: Any, config: Any) -> TaskClass | None:
    """Classify the latest human prompt, or None when there is nothing to do.

    None means one of exactly two things: the classifier is absent or disabled
    in the config, or the conversation holds no human prompt. Both are ordinary
    states, not failures, and neither is allowed to change anything.
    """
    spec = classifier_spec(config)
    if spec is None:
        return None
    text = latest_prompt_text(body_json)
    if text is None:
        return None

    score, reasons = _score(text, spec)
    return TaskClass(score=score, tier=_tier_for(score, spec), reason_codes=reasons)


def _score(text: str, spec: ClassifierSpec) -> tuple[int, tuple[str, ...]]:
    """Apply every rule, and report which ones fired. Rules are independent."""
    points = spec.points
    lowered = text.casefold()
    reasons: list[str] = []
    score = points.base

    strong_hits = _count_keywords(spec.strong_keywords, lowered)
    if strong_hits:
        score += min(strong_hits * points.strong, points.strong_cap)
        reasons.append(f"{STRONG_KW}:{strong_hits}")

    cheap_hits = _count_keywords(spec.cheap_keywords, lowered)
    if cheap_hits:
        # `cheap` and `cheap_cap` are negative, so the cap is the floor the
        # total penalty cannot pass.
        score += max(cheap_hits * points.cheap, points.cheap_cap)
        reasons.append(f"{CHEAP_KW}:{cheap_hits}")

    length = len(text)
    if length > points.long_chars:
        score += points.long_points
        reasons.append(LONG_PROMPT)
    if length > points.very_long_chars:
        score += points.very_long_points
        reasons.append(VERY_LONG_PROMPT)

    if file_like_count(text) >= points.many_files:
        score += points.many_files_points
        reasons.append(MANY_FILES)

    if is_question_only(text):
        score += points.question_points
        reasons.append(QUESTION_ONLY)

    return max(MIN_SCORE, min(MAX_SCORE, score)), tuple(reasons)


def _tier_for(score: int, spec: ClassifierSpec) -> str:
    """The tier a score falls in. A boundary score belongs to the tier it names:
    `cheap_below` is exclusive, `strong_from` is inclusive."""
    if score < spec.cheap_below:
        return LOW
    if score >= spec.strong_from:
        return HIGH
    return MID


def _count_keywords(keywords: Any, lowered_text: str) -> int:
    """How many distinct keywords appear in the text.

    A keyword matches when its characters appear anywhere in the prompt,
    ignoring case. An empty keyword is ignored: it would match everything.
    """
    if not isinstance(keywords, (list, tuple)):
        return 0
    seen: set[str] = set()
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        folded = keyword.casefold().strip()
        if not folded or folded in seen:
            continue
        seen.add(folded)
        if folded in lowered_text:
            yield_hits = True  # noqa: F841
    return sum(
        1
        for folded in seen
        if folded in lowered_text
    ) if False else _hits(seen, lowered_text)


def _hits(seen: set[str], lowered_text: str) -> int:
    return sum(1 for folded in seen if folded in lowered_text)


def file_like_count(text: str) -> int:
    """How many whitespace-separated tokens look like a file name.

    A token counts when it is made of name characters, a dot, and an extension
    that starts with a letter: `router/cli.py`, `config.yaml` and `notes.md`
    count; `v1.2.3`, a bare number and a word do not.
    """
    count = 0
    for raw in text.split():
        token = raw.strip(_EDGE_PUNCTUATION)
        if token and _FILE_LIKE.fullmatch(token):
            count += 1
    return count


def is_question_only(text: str) -> bool:
    """True when the prompt is one question and nothing else.

    It ends with `?` and carries no other sentence mark, so "what does this do?"
    is a question and "fix it. what does this do?" is not.
    """
    stripped = text.strip()
    if not stripped.endswith("?"):
        return False
    body = stripped[:-1]
    return not any(mark in body for mark in (".", "!", ":"))