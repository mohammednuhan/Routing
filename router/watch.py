"""`tamias-router watch`: a live, read-only view of decisions and estimated cost.

This polls the router's own database and prints one line per request as it
arrives, with a running footer under each batch. It opens the file through
`open_read_only`, so SQLite itself refuses to create it, write to it, or leave a
journal beside it: a view with nothing to read changes nothing on disk.

What it reads
-------------

`router_decisions` for what was requested, what was chosen, why, and what the
router did; `router_usage` for the token counts the upstream reported and what
those counts cost. Both hold metadata only, so what is printed here is metadata
too: a time, a session prefix, two model ids, an action, a reason code, a
classifier score, four numbers and a money figure. No prompt text, no headers,
no keys and no response content is printed, because none of it is in those tables
and none of it is selected. Rule 1.

Prices are not computed here. Every usage row already carries `cost_usd` and
`baseline_cost_usd`, priced by the proxy from the config it was running with, so
this command never loads `router/config.yaml` and never re-prices anything. It
reports what was recorded and it writes nothing.

UNKNOWN is never zero
---------------------

An unpriced figure is not a small one. A usage row whose cost is NULL is counted
in `unknown_rows` and never added into a dollar total as if it were free, and a
decision whose usage row has not been written yet is counted in `pending_rows` and
contributes nothing either. So the footer is a running pair: the known part of the
money, and how many rows are not in it.

`pending` and `UNKNOWN` are kept apart because only one of them can still change.
`pending` is a response that had not ended when the row was read, and its usage
row arrives later.

Late usage
----------

`router_usage` is written after the response ends, so a decision read on one poll
frequently has no usage row yet. Those rows are not lost and not reprinted as if
nothing happened: each is shown once with `pending`, and when its usage row
arrives it is printed again as an update line carrying the real figures, and the
running totals grow by exactly that amount. Late usage is found by `usage_id`
rather than by remembering which decisions are pending, so the work is bounded by
the table's own sequence and not by how much this process remembers.

An absent database is not an error. Nothing has been logged yet, so there is
nothing to show, and the view says `waiting for database...` and keeps polling: a
watch started before the proxy is expected to catch up on its own.

ASCII only
----------

Every line is plain ASCII, using `->` rather than a typographic arrow and no box
characters, so it renders the same in any Windows console, in a redirected file
and in CI logs. Colour is off by default, off whenever the output is not a
terminal, and turned off explicitly by `--no-color`. When it is on, the only
non-ASCII byte it adds is the SGR escape, never a box-drawing or arrow character.

Time and sleeping are injected. `now` supplies the clock and the local timezone,
`sleep` supplies the pause between polls, and `write` receives each finished line.
Nothing here calls `datetime.now` or `time.sleep` directly, so a test can run many
polls with no real sleeping and a fixed clock.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Callable

from .decisions import USAGE_TABLE
from .hold import REASON_HELD_MODEL
from .killswitch import REASON_KILL_SWITCH
from .readonly import ReadOnlyError, open_read_only
from .report import TABLE_NAME
from .safety import is_blocked
from .switch_report import parse_timestamp

#: Printed first, once. Every number below it is an estimate, and the two
#: limitations that matter are stated rather than left to be discovered.
WATCH_NOTE = (
    "Router decisions and estimated cost. Estimates use list prices from "
    "router/config.yaml and observed token counts. Not a bill."
)

#: Printed when the database is not there yet. Not an error: nothing has been
#: logged, and the view keeps polling so it catches up on its own.
WAITING = "waiting for database..."

#: Printed when the watch was stopped with Ctrl+C.
STOPPED = "stopped"

#: The cost of a row in a database with no usage table. Nothing can be priced
#: without one, so this is said outright rather than shown as $0.000000.
NO_USAGE_TABLE_COST = "n/a (no usage table)"

#: The cost of a decision whose response had not ended when the row was read.
COST_PENDING = "pending"

#: A usage row that reported counts no price could be applied to.
COST_UNKNOWN = "UNKNOWN"

#: A price of zero, which is a price and not the absence of one.
COST_FREE = "free"

#: Seconds between polls when `--interval` is not given.
DEFAULT_INTERVAL = 1.0

#: Existing rows shown at startup, oldest first, when `--since` is not given.
DEFAULT_TAIL = 5

#: Characters of a session hint printed. A hint is a salted digest and not a
#: secret, but it is not what this view is about either.
SESSION_SHOWN = 6

#: The local time format: enough to place a request in a day, no timezone noise.
TIME_FORMAT = "%H:%M:%S"

#: The keys read out of `signal_values`. The proxy writes them; nothing here
#: computes a score, and nothing here reads any other key out of that mapping.
CLASSIFIER_SCORE_KEY = "classifier_score"
CLASSIFIER_TIER_KEY = "classifier_tier"

#: Only these decision columns are selected, so nothing else can be printed.
_DECISION_COLUMNS = (
    "decision_id",
    "timestamp",
    "session_hint",
    "requested_model",
    "chosen_model",
    "action",
    "applied",
    "reason_codes",
    "signal_values",
)

#: Only these usage columns are selected. `usage_id` comes first so the joined
#: row carries the identity of the usage row it was priced from, which is what
#: lets a later poll tell an already-printed figure from a genuinely new one.
_USAGE_COLUMNS = (
    "usage_id",
    "status",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "baseline_cost_usd",
)

#: The join takes the newest usage row per decision rather than every one, so a
#: decision can never be printed twice because it accumulated two usage rows.
_JOINED = f"""
FROM {TABLE_NAME} AS d
LEFT JOIN {USAGE_TABLE} AS u ON u.usage_id = (
    SELECT MAX(u2.usage_id) FROM {USAGE_TABLE} AS u2 WHERE u2.decision_id = d.decision_id
)
"""

_DECISIONS_ONLY = f"FROM {TABLE_NAME} AS d"

_SELECTED = (
    f"{', '.join(f'd.{name}' for name in _DECISION_COLUMNS)}, "
    f"{', '.join(f'u.{name}' for name in _USAGE_COLUMNS)}"
)

#: The usage columns as literal NULLs, under the same names, when there is no
#: usage table to join. The rows a decision query returns then have exactly the
#: shape `_to_row` already expects, and `has_usage` comes out False for all of
#: them, which is the honest reading of a database that cannot price anything.
_SELECTED_UNPRICED = (
    f"{', '.join(f'd.{name}' for name in _DECISION_COLUMNS)}, "
    f"{', '.join(f'NULL AS {name}' for name in _USAGE_COLUMNS)}"
)

_SELECT_TAIL = f"SELECT {_SELECTED} {_JOINED} ORDER BY d.decision_id DESC LIMIT ?"
_SELECT_SINCE = f"SELECT {_SELECTED} {_JOINED} WHERE d.timestamp >= ? ORDER BY d.decision_id"
_SELECT_NEW = f"SELECT {_SELECTED} {_JOINED} WHERE d.decision_id > ? ORDER BY d.decision_id"

_SELECT_TAIL_UNPRICED = (
    f"SELECT {_SELECTED_UNPRICED} {_DECISIONS_ONLY} ORDER BY d.decision_id DESC LIMIT ?"
)
_SELECT_SINCE_UNPRICED = (
    f"SELECT {_SELECTED_UNPRICED} {_DECISIONS_ONLY} WHERE d.timestamp >= ? ORDER BY d.decision_id"
)
_SELECT_NEW_UNPRICED = (
    f"SELECT {_SELECTED_UNPRICED} {_DECISIONS_ONLY} WHERE d.decision_id > ? ORDER BY d.decision_id"
)

#: The left join is what makes a late-arriving usage row printable even when its
#: decision row could not be written at all.
_SELECT_LATE_USAGE = f"""
SELECT u.usage_id, u.decision_id, u.status, u.input_tokens, u.output_tokens,
       u.cost_usd, u.baseline_cost_usd, d.timestamp, d.session_hint
FROM {USAGE_TABLE} AS u
LEFT JOIN {TABLE_NAME} AS d ON d.decision_id = u.decision_id
WHERE u.usage_id > ?
ORDER BY u.usage_id
"""

_SELECT_MAX_USAGE_ID = f"SELECT MAX(usage_id) FROM {USAGE_TABLE}"
_SELECT_MAX_DECISION_ID = f"SELECT MAX(decision_id) FROM {TABLE_NAME}"


@dataclass(frozen=True)
class WatchOptions:
    """How the view polls and what it shows at startup."""

    interval: float = DEFAULT_INTERVAL
    since: str | None = None
    tail: int = DEFAULT_TAIL
    once: bool = False
    max_iterations: int | None = None
    color: bool = False


@dataclass(frozen=True)
class WatchRow:
    """One decision row, with its usage row if one had arrived.

    `has_usage` distinguishes "the usage row has not been written yet" from "the
    usage row says the cost is unknown". They are different facts and they print
    differently, because only one of them can still change.
    """

    decision_id: int
    timestamp: str
    session_hint: str
    requested_model: str | None
    chosen_model: str | None
    action: str
    applied: int
    reason_codes: tuple[str, ...]
    classifier_score: int | None
    classifier_tier: str | None
    has_usage: bool
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    baseline_cost_usd: float | None
    usage_id: int | None = None

    @property
    def session_shown(self) -> str:
        """The first six characters of the session hint."""
        return str(self.session_hint)[:SESSION_SHOWN]

    @property
    def main_reason(self) -> str:
        """The first reason code: the one that decided this row."""
        return self.reason_codes[0] if self.reason_codes else "-"

    @property
    def action_label(self) -> str:
        """What happened, as the router recorded it.

        An override is labelled by the override rather than by the action, because
        a kill-switch row is a STAY the router never chose and a held row is a
        switch being served again. Neither is a routing decision, so neither is
        printed as one.
        """
        if REASON_KILL_SWITCH in self.reason_codes:
            return "kill-switch"
        if is_blocked(self.reason_codes):
            return "blocked"
        if REASON_HELD_MODEL in self.reason_codes:
            return "held"
        if self.action == "SWITCH":
            return "SWITCH applied" if int(self.applied) == 1 else "would-switch"
        return "STAY"

    @property
    def has_classifier(self) -> bool:
        """True when the proxy recorded a classifier score for this request."""
        return self.classifier_score is not None


@dataclass(frozen=True)
class UsageUpdate:
    """A usage row that arrived after its decision row was already printed."""

    usage_id: int
    decision_id: int | None
    timestamp: str | None
    session_hint: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    baseline_cost_usd: float | None
    usage_status: str | None

    @property
    def session_shown(self) -> str:
        return str(self.session_hint or "-")[:SESSION_SHOWN]


@dataclass
class WatchTotals:
    """The running figures, built only from rows that have been printed.

    `cost_known_usd` is the known part and nothing else: an unpriced row is
    counted in `unknown_rows` and a row whose usage has not arrived is counted in
    `pending_rows`, and neither contributes a cent. `requests` counts decision
    lines only, so an update line never inflates it.
    """

    requests: int = 0
    cost_known_usd: float = 0.0
    baseline_known_usd: float = 0.0
    unknown_rows: int = 0
    pending_rows: int = 0
    usage_table: bool = True

    def add_row(self, row: WatchRow) -> None:
        """Fold one printed decision row in."""
        self.requests += 1
        if not self.usage_table:
            return
        if not row.has_usage:
            self.pending_rows += 1
            return
        self.add_money(row.cost_usd, row.baseline_cost_usd)

    def add_money(self, cost_usd: float | None, baseline_cost_usd: float | None) -> None:
        """Fold one row's two money figures in, or count the row as UNKNOWN.

        A row counts as UNKNOWN unless both figures are known. One known figure
        beside one unknown is not a partial total, it is a row nobody can price,
        and counting it twice would let a real cost look like an error.
        """
        if cost_usd is None or baseline_cost_usd is None:
            self.unknown_rows += 1
            return
        self.cost_known_usd += cost_usd
        self.baseline_known_usd += baseline_cost_usd

    @property
    def difference_usd(self) -> float:
        """Baseline minus routed, over the rows both figures are known for."""
        return self.baseline_known_usd - self.cost_known_usd


@dataclass
class WatchState:
    """Everything the loop carries between polls.

    The watermarks are what let each poll read only what is new, and holding them
    here is what lets `_poll` stay a function of a connection and this state. A
    test can drive `_poll` by hand, or hand the whole state to `run`.
    """

    last_decision_id: int = 0
    last_usage_id: int = 0
    started: bool = False
    totals: WatchTotals = field(default_factory=WatchTotals)


# --- reading -----------------------------------------------------------------


def has_table(connection: sqlite3.Connection, table: str) -> bool:
    """True when the database holds that table."""
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def has_usage_table(connection: sqlite3.Connection) -> bool:
    """True when usage rows can be priced at all.

    A database written before `router_usage` existed has decisions and no costs.
    That is not a failure and not an empty result: the decisions are real and are
    still shown, with their cost stated as unavailable rather than as zero.
    """
    return has_table(connection, USAGE_TABLE)


def has_decision_table(connection: sqlite3.Connection) -> bool:
    """True when the database holds the decision table at all."""
    return has_table(connection, TABLE_NAME)


def read_initial(
    connection: sqlite3.Connection, options: WatchOptions
) -> tuple[list[WatchRow], int]:
    """The rows shown at startup, and the newest `decision_id` among them.

    `--since` selects everything from that timestamp; otherwise the last `tail`
    rows, oldest first, so the newest line on screen is the newest request. A
    database with no rows returns the current maximum instead, so the first poll
    after the proxy starts does not replay everything from the beginning.

    The queries are chosen from the tables the database actually has. A database
    with no usage table still has decisions, and they are still read.
    """
    priced = has_usage_table(connection)
    since_query = _SELECT_SINCE if priced else _SELECT_SINCE_UNPRICED
    tail_query = _SELECT_TAIL if priced else _SELECT_TAIL_UNPRICED
    if options.since is not None:
        rows = [_to_row(raw) for raw in connection.execute(since_query, (options.since,))]
    else:
        newest_first = connection.execute(tail_query, (max(1, int(options.tail)),)).fetchall()
        rows = [_to_row(raw) for raw in reversed(newest_first)]
    newest = rows[-1].decision_id if rows else _max_id(connection, _SELECT_MAX_DECISION_ID)
    return rows, newest


def read_new(connection: sqlite3.Connection, after: int) -> list[WatchRow]:
    """Decision rows written after `after`, oldest first."""
    query = _SELECT_NEW if has_usage_table(connection) else _SELECT_NEW_UNPRICED
    return [_to_row(raw) for raw in connection.execute(query, (int(after),))]


def read_late_usage(connection: sqlite3.Connection, after: int) -> list[UsageUpdate]:
    """Usage rows written after `after`, oldest first.

    A database with no usage table has no late usage to find, which is a fact
    about the schema rather than an error: nothing was written that could be
    priced.
    """
    if not has_usage_table(connection):
        return []
    return [_to_update(raw) for raw in connection.execute(_SELECT_LATE_USAGE, (int(after),))]


def _max_id(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _to_row(raw: tuple[Any, ...]) -> WatchRow:
    values = dict(zip((*_DECISION_COLUMNS, *_USAGE_COLUMNS), raw, strict=True))
    score, tier = _classifier(values["signal_values"])
    status = values["status"]
    usage_id = values["usage_id"]
    return WatchRow(
        decision_id=int(values["decision_id"]),
        timestamp=str(values["timestamp"]),
        session_hint=str(values["session_hint"]),
        requested_model=values["requested_model"],
        chosen_model=values["chosen_model"],
        action=str(values["action"]),
        applied=int(values["applied"] or 0),
        reason_codes=_reason_codes(values["reason_codes"]),
        classifier_score=score,
        classifier_tier=tier,
        has_usage=status is not None,
        input_tokens=_count(values["input_tokens"]),
        output_tokens=_count(values["output_tokens"]),
        cost_usd=_amount(values["cost_usd"]),
        baseline_cost_usd=_amount(values["baseline_cost_usd"]),
        usage_id=None if usage_id is None else int(usage_id),
    )


def _to_update(raw: tuple[Any, ...]) -> UsageUpdate:
    status = raw[2]
    return UsageUpdate(
        usage_id=int(raw[0]),
        decision_id=None if raw[1] is None else int(raw[1]),
        timestamp=None if raw[7] is None else str(raw[7]),
        session_hint=None if raw[8] is None else str(raw[8]),
        input_tokens=_count(raw[3]),
        output_tokens=_count(raw[4]),
        cost_usd=_amount(raw[5]),
        baseline_cost_usd=_amount(raw[6]),
        usage_status=None if status is None else str(status),
    )


def _classifier(raw: Any) -> tuple[int | None, str | None]:
    """The classifier score and tier the proxy recorded, or `(None, None)`.

    Only these two keys are read out of `signal_values`. The mapping holds
    difficulty signals and nothing else, but nothing here iterates it or prints
    it wholesale: a field that is never read cannot be leaked by it.
    """
    if not isinstance(raw, str) or not raw:
        return None, None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    score = parsed.get(CLASSIFIER_SCORE_KEY)
    tier = parsed.get(CLASSIFIER_TIER_KEY)
    return (
        score if isinstance(score, int) and not isinstance(score, bool) else None,
        tier if isinstance(tier, str) and tier else None,
    )


def _reason_codes(raw: Any) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(code) for code in parsed)


def _count(value: Any) -> int | None:
    """A token count, or None. None means nobody reported it, which is not 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _amount(value: Any) -> float | None:
    """A money figure, or None. NULL is unknown and stays None; it is never 0.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# --- formatting --------------------------------------------------------------


def money(value: float) -> str:
    """One money figure at six decimals, matching `tamias-router cost`."""
    return f"${value:.6f}"


def format_time(timestamp: str, local: tzinfo | None) -> str:
    """A UTC timestamp as local wall-clock time, or `-` when unreadable.

    An unreadable timestamp shows as `-` rather than as the current time. A
    timestamp this view cannot read is unknown, and substituting the clock for it
    would put a plausible-looking time on a row that has none.
    """
    moment = parse_timestamp(timestamp)
    if moment is None:
        return "-"
    if local is not None:
        moment = moment.astimezone(local)
    return moment.strftime(TIME_FORMAT)


def cost_text(cost_usd: float | None, has_usage: bool, usage_table: bool) -> str:
    """The cost of one request, as it may honestly be shown.

    Four states, four words. `pending` is a response that had not ended;
    `UNKNOWN` is a row that arrived and could not be priced; `free` is a real
    price of zero; a dollar figure is a figure. None is ever rendered as another,
    and a missing usage table is stated outright rather than priced at zero.
    """
    if not usage_table:
        return NO_USAGE_TABLE_COST
    if not has_usage:
        return COST_PENDING
    if cost_usd is None:
        return COST_UNKNOWN
    if cost_usd == 0.0:
        return COST_FREE
    return money(cost_usd)


def format_row(row: WatchRow, local: tzinfo | None, usage_table: bool) -> str:
    """One request as one line. ASCII only."""
    parts = [
        format_time(row.timestamp, local),
        row.session_shown,
        f"{row.requested_model or '-'} -> {row.chosen_model or '-'}",
        row.action_label,
        row.main_reason,
    ]
    if row.has_classifier:
        parts.append(f"classifier {row.classifier_score}/{row.classifier_tier or '-'}")
    parts.append(f"in {_shown(row.input_tokens)} out {_shown(row.output_tokens)}")
    parts.append(cost_text(row.cost_usd, row.has_usage, usage_table))
    return "  ".join(parts)


def format_update(update: UsageUpdate, local: tzinfo | None) -> str:
    """A usage row that arrived after its decision row was printed.

    The decision line is not reprinted: it is still on screen, and only the
    figures that were missing are new.
    """
    return "  ".join(
        [
            "updated",
            format_time(update.timestamp, local) if update.timestamp else "-",
            update.session_shown,
            f"in {_shown(update.input_tokens)} out {_shown(update.output_tokens)}",
            cost_text(update.cost_usd, update.usage_status is not None, True),
        ]
    )


def format_footer(totals: WatchTotals) -> str:
    """The running footer under each batch.

    The two money figures are labelled as the known part, because they are: a row
    that is unpriced or still pending is counted beside them and added to neither.
    Without that count a partial total reads as a complete one, which is the
    failure this whole module exists to avoid.
    """
    if not totals.usage_table:
        return f"-- requests {totals.requests}  cost {NO_USAGE_TABLE_COST}"
    return (
        f"-- requests {totals.requests}  "
        f"routed {money(totals.cost_known_usd)} (known part)  "
        f"baseline {money(totals.baseline_known_usd)} (known part)  "
        f"difference {money(totals.difference_usd)}  "
        f"UNKNOWN rows {totals.unknown_rows}"
    )


def format_batch(
    rows: list[WatchRow],
    updates: list[UsageUpdate],
    totals: WatchTotals,
    local: tzinfo | None,
    usage_table: bool,
) -> list[str]:
    """One poll's output: the new rows, then any late usage, then the footer.

    The footer prints even when the batch was empty. A view that goes quiet
    without saying why reads as a crashed process rather than as a router with
    nothing to report.
    """
    lines = [format_row(row, local, usage_table) for row in rows]
    lines.extend(format_update(update, local) for update in updates)
    lines.append(format_footer(totals))
    return lines


def _shown(value: int | None) -> str:
    """A token count, or `-` when nobody reported it."""
    return "-" if value is None else str(value)


def _dim(text: str, color: bool) -> str:
    """Wrap in an SGR dim sequence, when colour is on.

    Only SGR escapes are added, never any other byte, so a colour line is still
    plain text apart from the escape and still carries no arrow or box character.
    """
    return f"\x1b[2m{text}\x1b[0m" if color else text


# --- polling and the loop ----------------------------------------------------


def _poll(
    connection: sqlite3.Connection,
    options: WatchOptions,
    state: WatchState,
    local: tzinfo | None,
) -> list[str]:
    """Read once, render, and advance the watermarks.

    The first poll takes the opening batch; every later poll reads only what is
    newer than the last one printed. Its first pass also lifts the usage watermark
    past whatever usage already exists, because those rows are shown in the
    opening batch already and must not be printed a second time as updates.

    Later polls do the same thing for each batch. A short request can have both
    its decision row and its usage row written between two polls, in which case the
    decision line already carries the price. Those usage rows are excluded from
    the update pass by their own `usage_id`, because replaying them would count
    the same request twice and overstate the cost of everything short in the
    session. Usage for an older row that was pending when it was printed is not
    excluded, because that row was printed without any figures and needs them.
    """
    usage_table = has_usage_table(connection)
    state.totals.usage_table = usage_table
    updates: list[UsageUpdate] = []

    if state.started:
        rows = read_new(connection, state.last_decision_id)
        if rows:
            state.last_decision_id = rows[-1].decision_id
        updates = read_late_usage(connection, state.last_usage_id)
        if updates:
            state.last_usage_id = updates[-1].usage_id
            priced_already = {row.usage_id for row in rows if row.usage_id is not None}
            if priced_already:
                updates = [update for update in updates if update.usage_id not in priced_already]
    else:
        rows, state.last_decision_id = read_initial(connection, options)
        state.last_usage_id = _max_id(connection, _SELECT_MAX_USAGE_ID) if usage_table else 0

    for row in rows:
        state.totals.add_row(row)
    if usage_table:
        for update in updates:
            state.totals.add_money(update.cost_usd, update.baseline_cost_usd)

    state.started = True
    return format_batch(rows, updates, state.totals, local, usage_table)


def poll_once(
    path: Path | str,
    options: WatchOptions = WatchOptions(),
    state: WatchState | None = None,
    local: tzinfo | None = None,
) -> list[str]:
    """One read and render of `path`, read-only. The unit the loop is built from.

    Opens and closes the database for this poll alone, so nothing holds a read
    lock across a sleep and a proxy writing at the same moment is never blocked by
    a viewer.
    """
    tracker = state if state is not None else WatchState()
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return [WAITING]
    try:
        with open_read_only(resolved) as connection:
            return _poll(connection, options, tracker, local)
    except ReadOnlyError as exc:
        return [f"{WAITING} ({exc})"]


def local_zone(clock: Callable[[], datetime]) -> tzinfo | None:
    """The local timezone, according to the injected clock."""
    try:
        moment = clock()
    except Exception:
        return None
    return moment.tzinfo if isinstance(moment, datetime) else None


def run(
    path: Path | str,
    options: WatchOptions = WatchOptions(),
    *,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
    write: Callable[[str], None] | None = None,
    color: bool | None = None,
) -> int:
    """Watch `path` until stopped. Returns 0.

    Three collaborators are injected so a test never sleeps and never depends on
    the wall clock: `now` supplies the current time and the local timezone,
    `sleep` supplies the pause between polls, and `write` receives each finished
    line. The defaults are the real clock, the real sleep and `print`.

    An absent or unreadable database is printed and then waited out, because a
    watch is expected to outlive the proxy it is watching. `KeyboardInterrupt` is
    the normal way to stop this and exits 0: the command was asked to stop, and a
    non-zero code would report a failure that never happened.
    """
    clock = now if now is not None else datetime.now
    pause = sleep if sleep is not None else time.sleep
    emit = write if write is not None else print
    use_color = options.color if color is None else color
    state = WatchState()
    zone = local_zone(clock)

    emit(_dim(WATCH_NOTE, use_color))
    iterations = 0
    try:
        while True:
            iterations += 1
            for line in poll_once(path, options, state, zone):
                emit(line)

            if options.once:
                return 0
            if options.max_iterations is not None and iterations >= options.max_iterations:
                return 0
            pause(options.interval)
    except KeyboardInterrupt:
        emit(_dim(STOPPED, use_color))
        return 0


__all__ = [
    "COST_FREE",
    "COST_PENDING",
    "COST_UNKNOWN",
    "DEFAULT_INTERVAL",
    "DEFAULT_TAIL",
    "NO_USAGE_TABLE_COST",
    "SESSION_SHOWN",
    "STOPPED",
    "WATCH_NOTE",
    "WAITING",
    "UsageUpdate",
    "WatchOptions",
    "WatchRow",
    "WatchState",
    "WatchTotals",
    "cost_text",
    "format_batch",
    "format_footer",
    "format_row",
    "format_time",
    "format_update",
    "has_usage_table",
    "local_zone",
    "money",
    "poll_once",
    "read_initial",
    "read_late_usage",
    "read_new",
    "run",
]