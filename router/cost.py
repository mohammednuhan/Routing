"""What a response cost, priced from the config's list prices.

Pure arithmetic. No I/O, no state, no configuration beyond the `RouterConfig` it
is handed, and no network. Every number here is an estimate and is labelled as
one wherever it is shown: the prices are list prices from `config.yaml` and the
token counts are whatever the upstream reported.

Two figures come out of one response:

* `cost_usd` - the counts priced at the **chosen** model's prices. This is what
  the request actually cost.
* `baseline_cost_usd` - the same counts priced at the **requested** model's
  prices. This is what it would have cost with no router in the path. It is an
  assumption: the other model would have produced a different number of tokens.
  In shadow mode, and for any STAY, the chosen model is the requested one and
  the two figures are equal, because nothing was moved.

UNKNOWN is never zero
---------------------

A price of `null` means nobody has verified that figure. It makes the cost
UNKNOWN when - and only when - the token count it prices is greater than zero:
a response that used no cache tokens is fully priced even if the cache price is
unknown, because there is no cache spend to price. A price of `0.0` is a real
price: a free model costs exactly 0.0, and it is never rounded to UNKNOWN.

UNKNOWN propagates: one unpriced nonzero count makes the whole total UNKNOWN,
and a report that contains an UNKNOWN row prints UNKNOWN rather than a dollar
figure (see `router/cost_report.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from .config import ModelSpec, RouterConfig
from .usage import STATUS_OK, UsageRecord

#: Tokens per million, the unit every configured price is quoted in.
TOKENS_PER_MILLION: Final = 1_000_000

#: Written into the usage row's `notes` when the upstream answered with a model
#: other than the one the router chose. The status stays OK: the counts are real,
#: they are simply for a different model than the row claims.
NOTE_MODEL_MISMATCH: Final = "MODEL_MISMATCH"


@dataclass(frozen=True)
class Priced:
    """One response's cost, its baseline, and anything worth saying about it.

    Either figure is None for UNKNOWN. A None is never to be read, summed or
    reported as 0.0.
    """

    cost_usd: float | None = None
    baseline_cost_usd: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_unknown(self) -> bool:
        """True when either figure could not be priced."""
        return self.cost_usd is None or self.baseline_cost_usd is None


def price_tokens(
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read_tokens: int | None,
    cache_write_tokens: int | None,
    spec: ModelSpec | None,
) -> float | None:
    """The four counts at one model's prices, or None when that is unknown.

    `spec` is None for a model that is not in the config. Such a model has no
    price for anything, so any nonzero count makes the result UNKNOWN.
    """
    prices = (
        (input_tokens, _price(spec, "input_per_million")),
        (output_tokens, _price(spec, "output_per_million")),
        (cache_read_tokens, _price(spec, "cache_read_per_million")),
        (cache_write_tokens, _price(spec, "cache_write_per_million")),
    )
    total = 0.0
    for count, price in prices:
        if count is None or count == 0:
            # Nothing was spent here, so an unknown price for it cannot make the
            # cost unknown.
            continue
        if price is None:
            return None
        total += count * price
    return total / TOKENS_PER_MILLION


def estimate_cost(
    usage: UsageRecord,
    chosen_model: str | None,
    requested_model: str | None,
    config: RouterConfig | None,
) -> Priced:
    """Price one response at the chosen model's rates and at the requested one's.

    `usage.status` is read and never written: a mismatch is a note, not a status.
    Anything other than `OK` means no counts were read, so there is nothing to
    price and both figures are UNKNOWN.
    """
    if usage.status != STATUS_OK:
        return Priced()

    counts = (
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
    )
    cost = price_tokens(*counts, _find_spec(config, chosen_model))
    baseline = price_tokens(*counts, _find_spec(config, requested_model))

    notes: list[str] = []
    if _mismatched(usage.model_reported, chosen_model):
        notes.append(NOTE_MODEL_MISMATCH)
    return Priced(cost_usd=cost, baseline_cost_usd=baseline, notes=tuple(notes))


def _price(spec: ModelSpec | None, name: str) -> float | None:
    """One price off a spec. An absent spec has no prices at all."""
    if spec is None:
        return None
    value: Any = getattr(spec, name, None)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _find_spec(config: RouterConfig | None, model: str | None) -> ModelSpec | None:
    """The spec for `model`, or None. Never guesses an id (Rule 7)."""
    if config is None or not isinstance(model, str):
        return None
    for spec in config.models:
        if spec.id == model:
            return spec
    return None


def _mismatched(reported: str | None, chosen: str | None) -> bool:
    """True when the upstream named a model, and it was not the chosen one.

    An absent report is not a mismatch: the upstream simply did not say, which is
    no more of a contradiction than a row with no model id.
    """
    return reported is not None and chosen is not None and reported != chosen


__all__ = [
    "NOTE_MODEL_MISMATCH",
    "TOKENS_PER_MILLION",
    "Priced",
    "estimate_cost",
    "price_tokens",
]