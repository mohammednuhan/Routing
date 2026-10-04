"""`tamias-router report`: counts over the router's own decision log.

Everything here is computed from `router_decisions`, which holds metadata only:
model ids, modes, counts, timings, flags and reason codes. No token counts and
no prices were ever recorded, so this report prints no money and claims no
saving. It says what the router decided and how often, never what that was
worth.

The database is opened read-only through `open_read_only`: a report never
creates one, never writes a row and never migrates a schema.

Counting rules, so the numbers mean one thing:

* an applied switch is a row with `action = 'SWITCH'` and `applied = 1`, except
  a row held from an earlier switch (`HELD_MODEL`), which is the same model
  being served again and is not a new switch;
* `would-switch` is a proposed switch that was not applied, which is what
  shadow mode records;
* a blocked switch is any row carrying a `BLOCKED_*` code, counted once per
  code, so one row can be counted under two of them;
* a kill-switch or open-circuit row is a pass-through the router did not
  choose. It is an override, not a decision, and is counted in its own section.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .breaker import REASON_CIRCUIT_OPEN
from .hold import REASON_HELD_MODEL
from .killswitch import REASON_KILL_SWITCH
from .readonly import ReadOnlyError, open_read_only
from .safety import BLOCKED_PREFIX

#: The one table this report reads.
TABLE_NAME = "router_decisions"

#: The only timestamp format the log writes, so `--since` compares as text.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Printed first in text, and carried in JSON. The log cannot support a cost
#: claim, so the report must not make one.
REPORT_NOTE = "Router decisions only (metadata). This is not cost and not a causal effect."

#: Printed when no row matches. An absent database reads the same way: there is
#: nothing recorded either way, and a read never creates one.
NOTHING_RECORDED = "no decisions recorded"

#: The columns the report reads. Nothing else is selected, so a field that is
#: not counted cannot reach the output.
_COLUMNS = (
    "timestamp",
    "session_hint",
    "requested_model",
    "chosen_model",
    "mode",
    "action",
    "applied",
    "reason_codes",
    "error",
)

#: Shown instead of a missing model id or an absent value.
BLANK = "-"


def parse_since(value: str) -> str:
    """A `--since` timestamp in the only format the log writes.

    Raises `ValueError` with the expected shape, so a typo cannot silently
    filter every row away.
    """
    try:
        datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError:
        raise ValueError(
            f"expected a UTC ISO-8601 timestamp like 2026-10-01T00:00:00Z, got {value!r}"
        ) from None
    return value


@dataclass(frozen=True)
class ReportFilters:
    """Which rows the report counts. Absent filters count everything."""

    since: str | None = None
    last: int | None = None
    session: str | None = None

    def describe(self) -> str:
        """One line naming the filters, or saying there were none."""
        parts = []
        if self.since is not None:
            parts.append(f"since {self.since}")
        if self.last is not None:
            parts.append(f"last {self.last}")
        if self.session is not None:
            parts.append(f"session {self.session}")
        return ", ".join(parts) if parts else "none"

    def as_json(self) -> dict[str, Any]:
        return {"since": self.since, "last": self.last, "session": self.session}


@dataclass(frozen=True)
class Report:
    """Every number the report prints. Metadata counts only."""

    source: str
    filters: ReportFilters
    requests: int
    by_mode: dict[str, int]
    first: str | None
    last: str | None
    model_flow: tuple[tuple[str, str, int], ...]
    stay: int
    switch_proposed: int
    switch_applied: int
    would_switch: int
    blocked: dict[str, int]
    held: int
    kill_switch: int
    circuit_open: int
    errors: dict[str, int]
    sessions: int
    max_applied_switches: int
    median_requests_per_session: float


def build_report(path: Path | str, filters: ReportFilters = ReportFilters()) -> Report:
    """Read `path` read-only and count what it holds."""
    with open_read_only(path) as connection:
        return collect_report(connection, str(Path(path)), filters)


def collect_report(
    connection: sqlite3.Connection,
    source: str = "",
    filters: ReportFilters = ReportFilters(),
) -> Report:
    """Count the decision rows in an open connection."""
    rows = _read_rows(connection, filters)
    return _count(rows, source, filters)


def _read_rows(connection: sqlite3.Connection, filters: ReportFilters) -> list[tuple[Any, ...]]:
    """The rows to count, oldest first. No table means nothing recorded."""
    if not _has_table(connection):
        return []

    clauses: list[str] = []
    params: list[Any] = []
    if filters.since is not None:
        clauses.append("timestamp >= ?")
        params.append(filters.since)
    if filters.session is not None:
        clauses.append("session_hint = ?")
        params.append(filters.session)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"SELECT {', '.join(_COLUMNS)} FROM {TABLE_NAME}{where}"

    if filters.last is None:
        return list(connection.execute(f"{query} ORDER BY decision_id", params))

    params.append(filters.last)
    newest_first = connection.execute(
        f"{query} ORDER BY decision_id DESC LIMIT ?", params
    ).fetchall()
    return list(reversed(newest_first))


def _has_table(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (TABLE_NAME,)
    ).fetchone()
    return row is not None


def _count(
    rows: list[tuple[Any, ...]], source: str, filters: ReportFilters
) -> Report:
    """Every number in the report, from the rows read. Nothing else is consulted."""
    modes: Counter[str] = Counter()
    flow: Counter[tuple[str, str]] = Counter()
    blocked: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    requests_per_session: Counter[str] = Counter()
    switches_per_session: Counter[str] = Counter()

    stay = 0
    proposed = 0
    applied = 0
    would_switch = 0
    held = 0
    kill_switch = 0
    circuit_open = 0
    first: str | None = None
    last: str | None = None

    for row in rows:
        values = dict(zip(_COLUMNS, row, strict=True))
        timestamp = str(values["timestamp"])
        session = str(values["session_hint"])
        requested = values["requested_model"]
        chosen = values["chosen_model"]
        mode = str(values["mode"])
        action = str(values["action"])
        was_applied = int(values["applied"]) == 1
        reasons = _reason_codes(values["reason_codes"])

        if first is None:
            first = timestamp
        last = timestamp
        modes[mode] += 1
        requests_per_session[session] += 1
        flow[(BLANK if requested is None else str(requested), BLANK if chosen is None else str(chosen))] += 1

        for code in reasons:
            if code.startswith(BLOCKED_PREFIX):
                blocked[code] += 1
        if REASON_HELD_MODEL in reasons:
            held += 1
        if REASON_KILL_SWITCH in reasons:
            kill_switch += 1
        if REASON_CIRCUIT_OPEN in reasons:
            circuit_open += 1

        error = values["error"]
        if error is not None:
            errors[str(error)] += 1

        if action == "STAY":
            stay += 1
        else:
            proposed += 1
            if was_applied:
                applied += 1
                if REASON_HELD_MODEL not in reasons:
                    switches_per_session[session] += 1
            else:
                would_switch += 1

    counts = list(requests_per_session.values())
    return Report(
        source=source,
        filters=filters,
        requests=len(rows),
        by_mode=_by_count_desc(modes),
        first=first,
        last=last,
        model_flow=tuple(
            (requested, chosen, count)
            for (requested, chosen), count in _by_count_desc(flow).items()
        ),
        stay=stay,
        switch_proposed=proposed,
        switch_applied=applied,
        would_switch=would_switch,
        blocked=_by_count_desc(blocked),
        held=held,
        kill_switch=kill_switch,
        circuit_open=circuit_open,
        errors=_by_count_desc(errors),
        sessions=len(counts),
        max_applied_switches=max(switches_per_session.values(), default=0),
        median_requests_per_session=statistics.median(counts) if counts else 0.0,
    )


def _reason_codes(raw: Any) -> list[str]:
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(code) for code in parsed]


def _by_count_desc(counter: Counter[str]) -> dict[str, int]:
    """Counts ordered by size, then by name, so the order never wobbles."""
    return {key: counter[key] for key in sorted(counter, key=lambda k: (-counter[k], k))}


def as_json(report: Report) -> str:
    """The same numbers as one JSON object."""
    return json.dumps(_payload(report), indent=2, sort_keys=True)


def _payload(report: Report) -> dict[str, Any]:
    return {
        "note": REPORT_NOTE,
        "source": report.source,
        "read_only": True,
        "filters": report.filters.as_json(),
        "totals": {
            "requests": report.requests,
            "by_mode": dict(report.by_mode),
            "first": report.first,
            "last": report.last,
        },
        "model_flow": [
            {"requested": requested, "chosen": chosen, "count": count}
            for requested, chosen, count in report.model_flow
        ],
        "actions": {
            "stay": report.stay,
            "switch_proposed": report.switch_proposed,
            "switch_applied": report.switch_applied,
            "would_switch": report.would_switch,
        },
        "blocked": dict(report.blocked),
        "held": {REASON_HELD_MODEL: report.held},
        "overrides": {
            REASON_KILL_SWITCH: report.kill_switch,
            REASON_CIRCUIT_OPEN: report.circuit_open,
        },
        "errors": dict(report.errors),
        "sessions": {
            "distinct": report.sessions,
            "max_applied_switches": report.max_applied_switches,
            "median_requests": report.median_requests_per_session,
        },
    }


def format_report(report: Report) -> str:
    """Render the report as sections. The note is always the first line."""
    flow_width = _flow_width(report)
    flow_lines = [
        f"  {f'{requested} -> {chosen}':<{flow_width}}  {count}"
        for requested, chosen, count in report.model_flow
    ] or ["  none"]

    sections = [
        _section(
            "Totals",
            (
                _pair("requests", str(report.requests)),
                _pair("by mode", _counts(report.by_mode)),
                _pair("time range", _time_range(report.first, report.last)),
            ),
        ),
        _section("Model flow (requested -> chosen)", flow_lines),
        _section(
            "Actions",
            (
                _pair("STAY", str(report.stay)),
                _pair("SWITCH proposed", str(report.switch_proposed)),
                _pair("SWITCH applied", str(report.switch_applied)),
                _pair("would-switch", str(report.would_switch)),
            ),
        ),
        _section("Blocked switches", _coded(report.blocked)),
        _section("Held", _coded({REASON_HELD_MODEL: report.held})),
        _section(
            "Overrides",
            (
                _pair(REASON_KILL_SWITCH, str(report.kill_switch)),
                _pair(REASON_CIRCUIT_OPEN, str(report.circuit_open)),
            ),
        ),
        _section("Errors", _coded(report.errors)),
        _section(
            "Sessions",
            (
                _pair("distinct", str(report.sessions)),
                _pair("max applied switches", str(report.max_applied_switches)),
                _pair("median requests", _number(report.median_requests_per_session)),
            ),
        ),
    ]
    header = [
        REPORT_NOTE,
        f"source: {report.source} (read-only)",
        f"filters: {report.filters.describe()}",
        f"rows: {report.requests}",
    ]
    return "\n".join([*header, "", *(line for section in sections for line in section)])


def _section(name: str, lines: tuple[str, ...] | list[str]) -> list[str]:
    return ["", name, *lines]


def _pair(label: str, value: str) -> str:
    return f"  {label:<{_LABEL_WIDTH}} {value}"


def _coded(counts: dict[str, int]) -> list[str]:
    """One line per code, or `none` so an empty section still shows."""
    if not counts:
        return ["  none"]
    return [f"  {code:<{_CODE_WIDTH}} {count}" for code, count in counts.items()]


def _counts(counts: dict[str, int]) -> str:
    if not counts:
        return BLANK
    return ", ".join(f"{name} {count}" for name, count in counts.items())


def _time_range(first: str | None, last: str | None) -> str:
    if first is None or last is None:
        return BLANK
    return first if first == last else f"{first} .. {last}"


def _number(value: float) -> str:
    """A median of whole requests reads as a whole number."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _flow_width(report: Report) -> int:
    return max(
        (len(f"{requested} -> {chosen}") for requested, chosen, _ in report.model_flow),
        default=0,
    )


_LABEL_WIDTH = 20
_CODE_WIDTH = 26


__all__ = [
    "NOTHING_RECORDED",
    "REPORT_NOTE",
    "ReadOnlyError",
    "Report",
    "ReportFilters",
    "TABLE_NAME",
    "TIMESTAMP_FORMAT",
    "as_json",
    "build_report",
    "collect_report",
    "format_report",
    "parse_since",
]