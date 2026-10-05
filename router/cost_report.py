"""`tamias-router cost`: what the observed token counts cost, priced and read-only.

This reads `router_usage`, which holds token counts and prices but no request or
response content, and it opens the database through `open_read_only`: a report
never creates a database, never writes a row and never migrates a schema. It is
a reader, and the tables it reads are the only ones it can see.

Two figures come out, and the difference between them is the whole point:

* `routed cost` is what the responses cost at the **chosen** model's prices.
* `baseline cost` is the same token counts priced at the **requested** model's
  prices: what they would have cost with no router in the path.

In shadow mode, and for every STAY, the chosen model is the requested one, so
the two are equal and the difference is 0.0. That is not a saving. It is the
absence of one, and the header says so.

UNKNOWN is never zero
---------------------

An unknown figure is not a small figure. A row whose cost could not be priced
- a null price in the config, a token count nobody reported, a model that is
not in the config at all - is counted as UNKNOWN and is never added into a
dollar total as if it were free. So every money figure in this report is a
pair: the subtotal of the rows that *are* priced, and whether any row in it was
not. When any row is unknown the report prints UNKNOWN and then the known
subtotal beside it, so a partial number is never mistaken for a complete one.

Grouping is by the **chosen** model, taken from the decision row the usage row
links to. A usage row whose decision row is missing (the decision write failed)
is grouped under `-` rather than being attributed to a model nobody chose.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cost import NOTE_MODEL_MISMATCH
from .decisions import USAGE_TABLE
from .readonly import ReadOnlyError, open_read_only
from .report import ReportFilters, parse_since
from .usage import STATUS_NO_USAGE, STATUS_UNKNOWN

#: Printed first, always. Everything this report prints is an estimate, and the
#: two limitations that matter are stated rather than left to be discovered.
COST_NOTE = (
    "Estimated from list prices in router/config.yaml and observed token counts. "
    "Baseline assumes the same token counts on the requested model, which may "
    "differ in reality. Not a bill. Savings only mean something in active mode."
)

#: Printed when no row matches. An absent database reads the same way: a report
#: never brings one into existence.
NOTHING_RECORDED = "no usage recorded"

#: The decision table the chosen model and the filters come from.
_DECISIONS = "router_decisions"

#: Shown for a chosen model that is unknown: a row with no id, or a usage row
#: whose decision row could not be written.
BLANK = "-"

#: Read from `router_usage` ...
_USAGE_COLUMNS = (
    "usage_id",
    "decision_id",
    "model_reported",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "status",
    "cost_usd",
    "baseline_cost_usd",
    "notes",
)

#: ... and from the decision row it links to. Only the chosen model is taken
#: from there, plus the two columns the filters compare against.
_DECISION_COLUMNS = ("decision_id", "timestamp", "session_hint", "chosen_model")

_SELECT = f"""
SELECT
    {', '.join(f'u.{name}' for name in _USAGE_COLUMNS)},
    {', '.join(f'd.{name}' for name in _DECISION_COLUMNS if name != 'decision_id')}
FROM {USAGE_TABLE} AS u
LEFT JOIN {_DECISIONS} AS d ON d.decision_id = u.decision_id
"""


@dataclass
class ModelCosts:
    """Every number for one chosen model.

    `cost_unknown` and `baseline_unknown` count the rows in this group whose
    figure could not be priced. They are rows, not money: a row is either priced
    or it is not, and the known subtotal covers the rest.

    Mutable because the report builds it up one row at a time. Nothing outside
    this module constructs one, so nothing can put a figure in it that the rows
    did not support.
    """

    model: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_known_usd: float = 0.0
    cost_unknown: int = 0
    baseline_known_usd: float = 0.0
    baseline_unknown: int = 0

    @property
    def cost_is_unknown(self) -> bool:
        return self.cost_unknown > 0

    @property
    def baseline_is_unknown(self) -> bool:
        return self.baseline_unknown > 0

    @property
    def cache_tokens(self) -> int:
        """Both cache counts added. A sum is safe here: both are token counts."""
        return self.cache_read_tokens + self.cache_write_tokens

    def add(self, row: "UsageRowView") -> None:
        self.requests += 1
        self.input_tokens += row.input_tokens
        self.output_tokens += row.output_tokens
        self.cache_read_tokens += row.cache_read_tokens
        self.cache_write_tokens += row.cache_write_tokens
        if row.cost_usd is None:
            self.cost_unknown += 1
        else:
            self.cost_known_usd += row.cost_usd
        if row.baseline_cost_usd is None:
            self.baseline_unknown += 1
        else:
            self.baseline_known_usd += row.baseline_cost_usd

    def merge(self, other: "ModelCosts") -> None:
        """Fold `other` into this group, in place."""
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cost_known_usd += other.cost_known_usd
        self.cost_unknown += other.cost_unknown
        self.baseline_known_usd += other.baseline_known_usd
        self.baseline_unknown += other.baseline_unknown

    def as_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_known_usd": self.cost_known_usd,
            "cost_unknown": self.cost_unknown,
            "baseline_known_usd": self.baseline_known_usd,
            "baseline_unknown": self.baseline_unknown,
        }


@dataclass(frozen=True)
class UsageRowView:
    """One usage row as read, joined to the decision it belongs to."""

    chosen_model: str | None
    timestamp: str | None
    session_hint: str | None
    status: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None
    baseline_cost_usd: float | None
    notes: str
    model_reported: str | None

    @property
    def is_mismatch(self) -> bool:
        return NOTE_MODEL_MISMATCH in self.note_codes

    @property
    def note_codes(self) -> tuple[str, ...]:
        return tuple(
            code for code in (part.strip() for part in self.notes.split(",")) if code
        )


@dataclass(frozen=True)
class CostReport:
    """Every number the report prints."""

    source: str
    filters: ReportFilters
    rows: int
    by_model: tuple[ModelCosts, ...]
    totals: ModelCosts
    unknown: int
    no_usage: int
    model_mismatch: int
    first: str | None = None
    last: str | None = None
    not_readable: int = field(default=0, compare=False)


def build_cost_report(
    path: Path | str, filters: ReportFilters = ReportFilters()
) -> CostReport:
    """Read `path` read-only and price what it holds."""
    with open_read_only(path) as connection:
        return collect_cost_report(connection, str(Path(path)), filters)


def collect_cost_report(
    connection: sqlite3.Connection,
    source: str = "",
    filters: ReportFilters = ReportFilters(),
) -> CostReport:
    """Count the usage rows in an open connection."""
    if not _has_table(connection, USAGE_TABLE):
        return CostReport(
            source=source,
            filters=filters,
            rows=0,
            by_model=(),
            totals=ModelCosts(model=BLANK),
            unknown=0,
            no_usage=0,
            model_mismatch=0,
        )

    rows = _read_rows(connection, filters)

    groups: dict[str, ModelCosts] = {}
    totals = ModelCosts(model="TOTAL")
    unknown = 0
    no_usage = 0
    mismatch = 0
    first: str | None = None
    last: str | None = None

    for row in rows:
        key = BLANK if row.chosen_model is None else row.chosen_model
        group = groups.setdefault(key, ModelCosts(model=key))
        group.add(row)
        totals.add(row)
        if row.status == STATUS_UNKNOWN:
            unknown += 1
        elif row.status == STATUS_NO_USAGE:
            no_usage += 1
        if row.is_mismatch:
            mismatch += 1
        if row.timestamp is not None:
            if first is None or row.timestamp < first:
                first = row.timestamp
            if last is None or row.timestamp > last:
                last = row.timestamp

    return CostReport(
        source=source,
        filters=filters,
        rows=len(rows),
        by_model=tuple(_by_requests_desc(groups)),
        totals=totals,
        unknown=unknown,
        no_usage=no_usage,
        model_mismatch=mismatch,
        first=first,
        last=last,
    )


def _has_table(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _read_rows(connection: sqlite3.Connection, filters: ReportFilters) -> list[UsageRowView]:
    """The rows to count, oldest first.

    Filters come from the decision row, because that is where a timestamp and a
    session hint live. A usage row with no decision row cannot be filtered by
    either, so it counts only when no filter narrows the report.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if filters.since is not None:
        clauses.append("d.timestamp >= ?")
        params.append(filters.since)
    if filters.session is not None:
        clauses.append("d.session_hint = ?")
        params.append(filters.session)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    if filters.last is None:
        raw = connection.execute(f"{_SELECT}{where} ORDER BY u.usage_id", params).fetchall()
        return [_to_row(row) for row in raw]

    params.append(filters.last)
    newest_first = connection.execute(
        f"{_SELECT}{where} ORDER BY u.usage_id DESC LIMIT ?", params
    ).fetchall()
    return [_to_row(row) for row in reversed(newest_first)]


def _to_row(row: tuple[Any, ...]) -> UsageRowView:
    values = dict(
        zip((*_USAGE_COLUMNS, *(n for n in _DECISION_COLUMNS if n != "decision_id")), row, strict=True)
    )
    return UsageRowView(
        chosen_model=values["chosen_model"],
        timestamp=values["timestamp"],
        session_hint=values["session_hint"],
        status=str(values["status"]),
        input_tokens=_count(values["input_tokens"]),
        output_tokens=_count(values["output_tokens"]),
        cache_read_tokens=_count(values["cache_read_tokens"]),
        cache_write_tokens=_count(values["cache_write_tokens"]),
        cost_usd=_amount(values["cost_usd"]),
        baseline_cost_usd=_amount(values["baseline_cost_usd"]),
        notes=str(values["notes"] or ""),
        model_reported=values["model_reported"],
    )


def _count(value: Any) -> int:
    """A reported token count. NULL means nobody reported it, which is 0 for a
    *sum*: a count that was never given cannot make a total larger, and the row
    it belongs to is already counted as UNKNOWN by its status."""
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _amount(value: Any) -> float | None:
    """A money figure. NULL is UNKNOWN and stays None; it is never 0.0."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _by_requests_desc(groups: dict[str, ModelCosts]) -> list[ModelCosts]:
    """Groups ordered by size, then by name, so the order never wobbles."""
    return [
        groups[key]
        for key in sorted(groups, key=lambda k: (-groups[k].requests, k))
    ]


def money(known_usd: float, unknown_rows: int) -> str:
    """One money figure: a dollar amount, or UNKNOWN with its known subtotal.

    `unknown_rows` is how many rows in this figure could not be priced. Any
    non-zero value means the dollar amount alone would understate the cost, so
    the amount is never printed on its own.
    """
    amount = f"${known_usd:.6f}"
    if unknown_rows:
        return f"UNKNOWN ({unknown_rows} row(s) unpriced, known part {amount})"
    return amount


def difference(routed: ModelCosts, baseline: ModelCosts) -> str:
    """Baseline minus routed, or UNKNOWN.

    The known difference is printed alongside UNKNOWN, never instead of it: the
    difference of two partial sums is itself only a partial figure.
    """
    known = baseline.baseline_known_usd - routed.cost_known_usd
    amount = f"${known:.6f}"
    if routed.cost_is_unknown or baseline.baseline_is_unknown:
        unpriced = routed.cost_unknown + baseline.baseline_unknown
        return f"UNKNOWN ({unpriced} unpriced figure(s), known part {amount})"
    return amount


def format_cost_report(report: CostReport) -> str:
    """Render the report. The note is always the first line."""
    header = [
        COST_NOTE,
        f"source: {report.source} (read-only)",
        f"filters: {report.filters.describe()}",
        f"responses: {report.rows}",
    ]

    if not report.by_model:
        return "\n".join([*header, "", NOTHING_RECORDED])

    sections = [
        "",
        "By chosen model",
        *_table(report.by_model),
        "",
        "Totals",
        f"  routed cost     {money(report.totals.cost_known_usd, report.totals.cost_unknown)}",
        f"  baseline cost   {money(report.totals.baseline_known_usd, report.totals.baseline_unknown)}",
        f"  difference      {difference(report.totals, report.totals)}",
        f"  UNKNOWN rows    {report.unknown}",
        f"  NO_USAGE rows   {report.no_usage}",
        f"  MODEL_MISMATCH  {report.model_mismatch}",
        f"  time range      {_time_range(report.first, report.last)}",
    ]
    return "\n".join([*header, *sections])


def _table(groups: tuple[ModelCosts, ...]) -> list[str]:
    header = ("chosen model", "reqs", "input", "output", "cache", "cost")
    body = [
        (
            group.model,
            str(group.requests),
            str(group.input_tokens),
            str(group.output_tokens),
            str(group.cache_tokens),
            money(group.cost_known_usd, group.cost_unknown),
        )
        for group in groups
    ]
    widths = [
        max(len(header[index]), *(len(row[index]) for row in body))
        for index in range(len(header))
    ]
    return [
        "  " + "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
        for row in (header, *body)
    ]


def _time_range(first: str | None, last: str | None) -> str:
    if first is None or last is None:
        return BLANK
    return first if first == last else f"{first} .. {last}"


def cost_as_json(report: CostReport) -> str:
    """The same numbers as one JSON object. `null` is how UNKNOWN is carried."""
    return json.dumps(_payload(report), indent=2, sort_keys=True)


def _payload(report: CostReport) -> dict[str, Any]:
    return {
        "note": COST_NOTE,
        "source": report.source,
        "read_only": True,
        "filters": report.filters.as_json(),
        "totals": {
            "responses": report.rows,
            "routed_cost_usd": None if report.totals.cost_is_unknown else report.totals.cost_known_usd,
            "routed_cost_known_usd": report.totals.cost_known_usd,
            "baseline_cost_usd": (
                None if report.totals.baseline_is_unknown else report.totals.baseline_known_usd
            ),
            "baseline_cost_known_usd": report.totals.baseline_known_usd,
            "difference_usd": (
                None
                if report.totals.cost_is_unknown or report.totals.baseline_is_unknown
                else report.totals.baseline_known_usd - report.totals.cost_known_usd
            ),
            "input_tokens": report.totals.input_tokens,
            "output_tokens": report.totals.output_tokens,
            "cache_read_tokens": report.totals.cache_read_tokens,
            "cache_write_tokens": report.totals.cache_write_tokens,
            "first": report.first,
            "last": report.last,
        },
        "by_model": [group.as_json() for group in report.by_model],
        "rows": {
            STATUS_UNKNOWN: report.unknown,
            STATUS_NO_USAGE: report.no_usage,
            NOTE_MODEL_MISMATCH: report.model_mismatch,
        },
    }


__all__ = [
    "COST_NOTE",
    "NOTHING_RECORDED",
    "CostReport",
    "ModelCosts",
    "ReadOnlyError",
    "UsageRowView",
    "build_cost_report",
    "collect_cost_report",
    "cost_as_json",
    "format_cost_report",
    "money",
    "parse_since",
]