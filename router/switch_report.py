"""`tamias-router switch-report`: what the Observer ledger saw around a switch.

For each approved router switch, this prints the ledger rows observed in a time
window around it. It is a time correlation and nothing more.

What it is not: it is not a cost, and it is not a cause. Two clocks that happen
to agree do not prove one produced the other, so every switch is printed with
`cause: UNKNOWN (time correlation only)` and the model check is reported as
observed rather than argued. The token counts are the ledger's own numbers; no
price is looked up and no dollar figure is printed anywhere.

The ledger is another tool's database. Nothing here imports the Tamias
Observer, and nothing here interprets the ledger: only the columns listed in
`LEDGER_COLUMNS` are read, `NULL` is reported as `UNKNOWN` because it means
unknown rather than zero, and a timestamp that cannot be parsed is counted
rather than guessed at. Both databases are opened read-only, so neither is
created nor changed by reading it.

Timestamp handling. The router writes `YYYY-MM-DDTHH:MM:SSZ`. The ledger's
`ts_utc` is read as ISO-8601, with or without a `Z` and with or without
fractional seconds; a value with no offset is read as UTC, because that is what
the column is named. Anything else is counted as unparseable and left out of
every window rather than being guessed into one.
"""
from __future__ import annotations

import json
import sqlite3
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .hold import REASON_HELD_MODEL
from .readonly import open_read_only
from .report import TABLE_NAME as ROUTER_TABLE

#: Default ledger. The Tamias Observer's own file, read but never written.
DEFAULT_LEDGER_PATH = Path.home() / ".tamias" / "ledger.sqlite3"

#: The one ledger table this reads, and the filter that makes it the main chain.
LEDGER_TABLE = "request_record"
MAIN_CHAIN = "main"

#: How far either side of a switch the ledger is searched, unless overridden.
DEFAULT_WINDOW_SECONDS = 120
DEFAULT_SKEW_SECONDS = 5

#: How far back the "previous request" comparison looks.
PREVIOUS_LOOKBACK_SECONDS = 600

#: Switches shown when `--last` is not given.
DEFAULT_LAST = 10

#: Characters of a ledger session id printed. A session id is not a secret, but
#: it is not what the report is about either.
SESSION_ID_SHOWN = 8

#: Printed when a value is NULL, or absent. Never a zero.
UNKNOWN = "UNKNOWN"

#: Printed under every switch, always. The same words, every time.
CAUSE = "UNKNOWN (time correlation only)"

MATCH = "MATCH"
MISMATCH = "MISMATCH"

NO_LEDGER_ROWS = "NO LEDGER ROWS IN WINDOW"

#: Printed when two or more sessions fall in one window, so time alone cannot
#: say which one the switch belongs to.
AMBIGUOUS = "AMBIGUOUS: {sessions} sessions overlap in this window; the match is by time only"

#: Printed when the ledger has nothing to correlate with.
NO_REQUEST_RECORDS = (
    "ledger has no request records: run a real Claude Code session, then the Observer scan"
)

#: The first line of the report, in text and in JSON.
SWITCH_REPORT_NOTE = (
    "Time correlation between router switches and observed ledger tokens. This is NOT "
    "the cost of the switch and NOT a causal effect. Token counts are observed values; "
    "no dollar figures are shown."
)

#: The only ledger columns read. Nothing else is selected, so nothing else can
#: reach the output. `seq` orders rows that share a timestamp and is not printed.
LEDGER_COLUMNS = (
    "ts_utc",
    "session_id",
    "model_id",
    "input_tokens",
    "cache_read_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "cache_write_total_tokens",
)

_SELECTED_LEDGER_COLUMNS = ("seq", *LEDGER_COLUMNS)

#: The only router columns read.
_SWITCH_COLUMNS = (
    "decision_id",
    "timestamp",
    "session_hint",
    "requested_model",
    "chosen_model",
    "reason_codes",
)


class LedgerProblem(Exception):
    """The ledger cannot be read as a store of request records."""


class NoRequestRecords(LedgerProblem):
    """The ledger is absent, or holds no `request_record` rows."""


class MissingRequestRecordTable(LedgerProblem):
    """The ledger exists but has no `request_record` table."""


def parse_timestamp(raw: Any) -> datetime | None:
    """An ISO-8601 timestamp as an aware UTC datetime, or None if unparseable.

    Accepts a trailing `Z` or a numeric offset, with or without fractional
    seconds. A value with no offset is read as UTC. Anything else is None: a
    timestamp this tool cannot read is counted, never rounded towards a window.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text[-1] in ("Z", "z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class LedgerRow:
    """One main-chain ledger row, with its timestamp read."""

    at: datetime
    ts_utc: str
    session_id: str | None
    model_id: str | None
    input_tokens: int | None
    cache_read_tokens: int | None
    cache_write_5m_tokens: int | None
    cache_write_1h_tokens: int | None
    cache_write_total_tokens: int | None

    @property
    def session_shown(self) -> str:
        if self.session_id is None:
            return UNKNOWN
        return self.session_id[:SESSION_ID_SHOWN]

    def value(self, column: str) -> Any:
        return getattr(self, column)

    def as_json(self) -> dict[str, Any]:
        """The printed columns, with NULL kept as null.

        JSON keeps the value as it is stored; the text form renders it as
        `UNKNOWN`, because a machine reader can tell null from zero and a
        person should not be shown a zero the ledger never claimed.
        """
        return {column: self.value(column) for column in LEDGER_COLUMNS}


@dataclass(frozen=True)
class LedgerIndex:
    """Every main-chain ledger row, read once and ordered by time.

    Windows are then found by binary search, so one pass over the ledger serves
    every switch in the report no matter how many there are.
    """

    rows: tuple[LedgerRow, ...]
    unparseable: int
    times: tuple[datetime, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", tuple(row.at for row in self.rows))

    def window(self, start: datetime, end: datetime) -> tuple[LedgerRow, ...]:
        """Rows whose timestamp falls in `[start, end]`, oldest first."""
        return self.rows[bisect_left(self.times, start) : bisect_right(self.times, end)]

    def previous(self, moment: datetime, lookback_seconds: int) -> LedgerRow | None:
        """The last row strictly before `moment`, within the lookback."""
        index = bisect_left(self.times, moment) - 1
        if index < 0:
            return None
        row = self.rows[index]
        if row.at < moment - timedelta(seconds=lookback_seconds):
            return None
        return row


@dataclass(frozen=True)
class SwitchEntry:
    """One approved switch and what the ledger held around it."""

    decision_id: int
    router_timestamp: str
    requested_model: str | None
    chosen_model: str | None
    session_hint: str
    reason_codes: tuple[str, ...]
    rows: tuple[LedgerRow, ...]
    previous: LedgerRow | None
    model_check: str
    sessions_in_window: int

    @property
    def ambiguous(self) -> bool:
        return self.sessions_in_window > 1

    @property
    def empty(self) -> bool:
        return not self.rows

    def as_json(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "router_timestamp": self.router_timestamp,
            "requested_model": self.requested_model,
            "chosen_model": self.chosen_model,
            "session_hint": self.session_hint,
            "reason_codes": list(self.reason_codes),
            "cause": CAUSE,
            "ledger_rows": [row.as_json() for row in self.rows],
            "previous_request": self.previous.as_json() if self.previous else None,
            "model_check": self.model_check,
            "sessions_in_window": self.sessions_in_window,
            "ambiguity": ambiguous_note(self.sessions_in_window) if self.ambiguous else None,
            "no_ledger_rows": self.empty,
        }


@dataclass(frozen=True)
class SwitchReportOptions:
    """What to correlate, and how wide a window to look in."""

    ledger: Path = DEFAULT_LEDGER_PATH
    window_seconds: int = DEFAULT_WINDOW_SECONDS
    skew_seconds: int = DEFAULT_SKEW_SECONDS
    last: int = DEFAULT_LAST
    since: datetime | None = None


@dataclass(frozen=True)
class SwitchReport:
    """Every switch shown, and the counts that describe them."""

    router_database: str
    ledger: str
    window_seconds: int
    skew_seconds: int
    switches: tuple[SwitchEntry, ...]
    unparseable_router_timestamps: int
    unparseable_ledger_timestamps: int

    @property
    def shown(self) -> int:
        return len(self.switches)

    @property
    def without_rows(self) -> int:
        return sum(1 for entry in self.switches if entry.empty)

    @property
    def ambiguous_windows(self) -> int:
        return sum(1 for entry in self.switches if entry.ambiguous)

    def model_checks(self) -> dict[str, int]:
        return {
            label: sum(1 for entry in self.switches if entry.model_check == label)
            for label in (MATCH, MISMATCH, UNKNOWN)
        }


def ambiguous_note(sessions: int) -> str:
    return AMBIGUOUS.format(sessions=sessions)


def read_ledger(path: Path | str) -> LedgerIndex:
    """Read every main-chain ledger row, read-only.

    Raises `NoRequestRecords` when the file is absent or holds no rows, and
    `MissingRequestRecordTable` when it has no `request_record` table.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise NoRequestRecords(f"{NO_REQUEST_RECORDS}\nledger: {resolved} (not found)")

    with open_read_only(resolved) as connection:
        if not _has_table(connection, LEDGER_TABLE):
            raise MissingRequestRecordTable(
                f"ledger has no {LEDGER_TABLE} table: {resolved}"
            )
        if _count(connection, LEDGER_TABLE) == 0:
            raise NoRequestRecords(f"{NO_REQUEST_RECORDS}\nledger: {resolved}")
        rows, unparseable = _read_main_chain(connection)

    return LedgerIndex(rows=tuple(rows), unparseable=unparseable)


def _has_table(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _count(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()
    return int(row[0])


def _read_main_chain(connection: sqlite3.Connection) -> tuple[list[LedgerRow], int]:
    """Main-chain rows with a readable timestamp, oldest first, plus a tally.

    The main-chain filter is exact, so it belongs in SQL: a NULL `is_sidechain`
    is unknown rather than zero and is therefore not a main-chain row. The
    timestamp filter cannot be, because `ts_utc` may be written in any ISO form
    and text comparison would only be right for one of them.
    """
    cursor = connection.execute(
        f"SELECT {', '.join(_SELECTED_LEDGER_COLUMNS)} FROM {LEDGER_TABLE} "
        f"WHERE request_bucket = ? AND is_sidechain = 0",
        (MAIN_CHAIN,),
    )
    rows: list[LedgerRow] = []
    unparseable = 0
    for raw in cursor:
        values = dict(zip(_SELECTED_LEDGER_COLUMNS, raw, strict=True))
        at = parse_timestamp(values["ts_utc"])
        if at is None:
            unparseable += 1
            continue
        rows.append(
            LedgerRow(
                at=at,
                ts_utc=str(values["ts_utc"]),
                session_id=values["session_id"],
                model_id=values["model_id"],
                input_tokens=values["input_tokens"],
                cache_read_tokens=values["cache_read_tokens"],
                cache_write_5m_tokens=values["cache_write_5m_tokens"],
                cache_write_1h_tokens=values["cache_write_1h_tokens"],
                cache_write_total_tokens=values["cache_write_total_tokens"],
            )
        )
    rows.sort(key=lambda row: row.at)
    return rows, unparseable


def read_switches(connection: sqlite3.Connection, limit: int) -> tuple[list[dict[str, Any]], int]:
    """The newest approved switches, newest first, plus unparseable timestamps.

    Approved means `applied = 1`, and a held row is not a switch: it is an
    earlier switch being served again, so it is excluded here as well as in the
    report's own switch count.
    """
    if not _has_table(connection, ROUTER_TABLE):
        return [], 0

    cursor = connection.execute(
        f"SELECT {', '.join(_SWITCH_COLUMNS)} FROM {ROUTER_TABLE} "
        "WHERE applied = 1 AND reason_codes NOT LIKE ? ORDER BY decision_id DESC LIMIT ?",
        (f"%{REASON_HELD_MODEL}%", max(1, int(limit))),
    )
    switches: list[dict[str, Any]] = []
    unparseable = 0
    for raw in cursor:
        values = dict(zip(_SWITCH_COLUMNS, raw, strict=True))
        reasons = _reason_codes(values["reason_codes"])
        # Authoritative: the SQL filter is a pre-filter that lets the LIMIT work.
        if REASON_HELD_MODEL in reasons:
            continue
        if parse_timestamp(values["timestamp"]) is None:
            unparseable += 1
            continue
        switches.append(values)
    return switches, unparseable


def build_switch_report(
    router_database: Path | str,
    options: SwitchReportOptions = SwitchReportOptions(),
    ledger: LedgerIndex | None = None,
) -> SwitchReport:
    """Correlate the newest approved switches with the ledger around them."""
    index = read_ledger(options.ledger) if ledger is None else ledger

    with open_read_only(router_database) as connection:
        switches, unparseable = read_switches(connection, options.last)

    entries: list[SwitchEntry] = []
    for values in switches:
        at = parse_timestamp(values["timestamp"])
        if at is None:
            continue
        if options.since is not None and at < options.since:
            continue
        entries.append(_correlate(values, at, index, options))

    return SwitchReport(
        router_database=str(Path(router_database)),
        ledger=str(Path(options.ledger)),
        window_seconds=options.window_seconds,
        skew_seconds=options.skew_seconds,
        switches=tuple(entries),
        unparseable_router_timestamps=unparseable,
        unparseable_ledger_timestamps=index.unparseable,
    )


def _correlate(
    values: dict[str, Any], at: datetime, index: LedgerIndex, options: SwitchReportOptions
) -> SwitchEntry:
    start = at - timedelta(seconds=options.skew_seconds)
    end = at + timedelta(seconds=options.window_seconds)
    rows = index.window(start, end)
    chosen = values["chosen_model"]
    return SwitchEntry(
        decision_id=int(values["decision_id"]),
        router_timestamp=str(values["timestamp"]),
        requested_model=values["requested_model"],
        chosen_model=chosen,
        session_hint=str(values["session_hint"]),
        reason_codes=tuple(_reason_codes(values["reason_codes"])),
        rows=rows,
        previous=index.previous(at, PREVIOUS_LOOKBACK_SECONDS),
        model_check=model_check(rows, at, chosen),
        # A NULL session id is unknown, not a session of its own, so it is not
        # counted here: one unknown must not invent an ambiguity.
        sessions_in_window=len({row.session_id for row in rows if row.session_id is not None}),
    )


def model_check(rows: tuple[LedgerRow, ...], at: datetime, chosen_model: str | None) -> str:
    """Whether the first row at or after the switch used the chosen model.

    Reported as observed. UNKNOWN when the ledger said nothing: no row at or
    after the switch, a NULL `model_id`, or a switch that chose no model.
    """
    if chosen_model is None:
        return UNKNOWN
    for row in rows:
        if row.at < at:
            continue
        if row.model_id is None:
            return UNKNOWN
        return MATCH if row.model_id == chosen_model else MISMATCH
    return UNKNOWN


def _reason_codes(raw: Any) -> list[str]:
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(code) for code in parsed]


def quote_identifier(name: str) -> str:
    """A SQL identifier quoted for use as one."""
    return '"' + str(name).replace('"', '""') + '"'


def switch_report_as_json(report: SwitchReport) -> str:
    """The same numbers as one JSON object."""
    return json.dumps(_payload(report), indent=2, sort_keys=True)


def _payload(report: SwitchReport) -> dict[str, Any]:
    checks = report.model_checks()
    return {
        "note": SWITCH_REPORT_NOTE,
        "router_database": report.router_database,
        "ledger": report.ledger,
        "read_only": True,
        "cause": CAUSE,
        "window": {
            "skew_seconds": report.skew_seconds,
            "window_seconds": report.window_seconds,
            "previous_lookback_seconds": PREVIOUS_LOOKBACK_SECONDS,
        },
        "totals": {
            "switches_shown": report.shown,
            "switches_with_no_ledger_rows": report.without_rows,
            "ambiguous_windows": report.ambiguous_windows,
            "model_checks": checks,
            "unparseable_router_timestamps": report.unparseable_router_timestamps,
            "unparseable_ledger_timestamps": report.unparseable_ledger_timestamps,
        },
        "switches": [entry.as_json() for entry in report.switches],
    }


def format_switch_report(report: SwitchReport) -> str:
    """Render the report. The note is always the first line."""
    lines = [
        SWITCH_REPORT_NOTE,
        f"router database: {report.router_database} (read-only)",
        f"ledger: {report.ledger} (read-only)",
        f"window: {report.skew_seconds}s before and {report.window_seconds}s after each "
        f"switch; previous request looks back {PREVIOUS_LOOKBACK_SECONDS}s",
    ]

    if not report.switches:
        lines.extend(["", "no approved switches to report"])
    for number, entry in enumerate(report.switches, start=1):
        lines.extend(["", f"switch {number} of {report.shown}"])
        lines.extend(_entry_lines(entry))

    checks = report.model_checks()
    lines.extend(
        [
            "",
            "totals",
            _total("switches shown", str(report.shown)),
            _total("switches with no ledger rows", str(report.without_rows)),
            _total("ambiguous windows", str(report.ambiguous_windows)),
            _total(
                "model checks",
                f"{MATCH} {checks[MATCH]}, {MISMATCH} {checks[MISMATCH]}, {UNKNOWN} {checks[UNKNOWN]}",
            ),
            _total(
                "unparseable timestamps",
                f"router {report.unparseable_router_timestamps}, "
                f"ledger {report.unparseable_ledger_timestamps}",
            ),
        ]
    )
    return "\n".join(lines)


def _total(label: str, value: str) -> str:
    return f"  {label:<{_WIDTH}} {value}"


def _entry_lines(entry: SwitchEntry) -> list[str]:
    lines = [
        f"  router timestamp: {entry.router_timestamp}",
        f"  requested: {_flow(entry)}",
        f"  session hint: {entry.session_hint}",
        f"  reason codes: {','.join(entry.reason_codes) or UNKNOWN}",
        f"  cause: {CAUSE}",
    ]
    if entry.rows:
        lines.append(
            f"  ledger rows in window (session_id shown as its first "
            f"{SESSION_ID_SHOWN} characters):"
        )
        lines.extend(_table(entry.rows))
    if entry.previous is None:
        lines.append("  previous request: none found")
    else:
        lines.append("  previous request, the last main-chain row before the switch:")
        lines.extend(_table([entry.previous]))
    lines.append(f"  model_check: {entry.model_check}")
    if entry.empty:
        lines.append(f"  {NO_LEDGER_ROWS}")
    elif entry.ambiguous:
        lines.append(f"  {ambiguous_note(entry.sessions_in_window)}")
    return lines


def _flow(entry: SwitchEntry) -> str:
    requested = entry.requested_model if entry.requested_model is not None else UNKNOWN
    chosen = entry.chosen_model if entry.chosen_model is not None else UNKNOWN
    return f"{requested} -> {chosen}"


def _table(rows: tuple[LedgerRow, ...] | list[LedgerRow]) -> list[str]:
    """A fixed-width table of the printed ledger columns only."""
    headers = list(LEDGER_COLUMNS)
    body = [[_shown(row, column) for column in headers] for row in rows]
    widths = [
        max([len(headers[index]), *(len(line[index]) for line in body)])
        for index in range(len(headers))
    ]

    def render(line: list[str]) -> str:
        return "  ".join(
            line[index].ljust(widths[index]) for index in range(len(line))
        ).rstrip()

    return [render(headers), *(render(line) for line in body)]


def _shown(row: LedgerRow, column: str) -> str:
    """One cell: a session id cut to eight characters, or UNKNOWN for a NULL."""
    if column == "session_id":
        return row.session_shown
    value = row.value(column)
    return UNKNOWN if value is None else str(value)


_WIDTH = 30


__all__ = [
    "AMBIGUOUS",
    "CAUSE",
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_LAST",
    "DEFAULT_SKEW_SECONDS",
    "DEFAULT_WINDOW_SECONDS",
    "LEDGER_TABLE",
    "MATCH",
    "MISMATCH",
    "NO_LEDGER_ROWS",
    "NO_REQUEST_RECORDS",
    "PREVIOUS_LOOKBACK_SECONDS",
    "SWITCH_REPORT_NOTE",
    "UNKNOWN",
    "LedgerProblem",
    "MissingRequestRecordTable",
    "NoRequestRecords",
    "SwitchReport",
    "SwitchReportOptions",
    "build_switch_report",
    "format_switch_report",
    "model_check",
    "parse_timestamp",
    "read_ledger",
    "switch_report_as_json",
]