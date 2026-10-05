"""`tamias-router watch`: a read-only live view of decisions and estimated cost.

The order the tests follow is the order the numbers depend on each other.

`cost_text`: one request's cost as one of four words. This is the rule that can be
broken silently, because a `pending` row or an unpriced row rendered as `$0.000000`
looks exactly like a free request. So the tests assert on which words appear, and
on which dollar figure appears, rather than only that "something" was printed.

`WatchTotals` and `format_footer`: the running arithmetic. An UNKNOWN row must
never be added into a dollar total, so the footer tests use a hand-computed
figure with a free model and an UNKNOWN row side by side and check the exact
cents. If the arithmetic were wrong, every other figure in this file would still
look right.

The reader and the loop: initial rows, rows appearing between polls, usage
arriving after its decision row, a database with no usage table, and an absent
database. Time and sleeping are injected, so nothing here waits for anything.

Read-only: the database's size and mtime must be unchanged after every path,
including the ones that read a lot.

The output itself: ASCII only, and the secret phrase never appears in it.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from router import cli
from router.cost import estimate_cost
from router.decisions import (
    SCHEMA,
    USAGE_SCHEMA,
    USAGE_TABLE,
    DecisionLog,
    DecisionRecord,
    UsageRow,
    default_db_path,
)
from router.readonly import open_read_only
from router.usage import STATUS_OK, STATUS_UNKNOWN, UsageRecord
from router.watch import (
    COST_FREE,
    COST_PENDING,
    COST_UNKNOWN,
    NO_USAGE_TABLE_COST,
    WATCH_NOTE,
    WAITING,
    WatchOptions,
    WatchRow,
    WatchState,
    WatchTotals,
    cost_text,
    format_footer,
    format_row,
    format_time,
    format_update,
    has_usage_table,
    money,
    poll_once,
    read_initial,
    read_late_usage,
    run,
)

MODEL_LOW = "test-low"
MODEL_MID = "test-mid"
MODEL_HIGH = "test-high"

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-ACTIVE-test.yaml"

#: Dummy rates from `tools/sample-config-ACTIVE-test.yaml`, per million tokens.
RATE_INPUT = 3.00
RATE_OUTPUT = 15.00
RATE_CACHE_WRITE = 3.75
RATE_CACHE_READ = 0.30

#: A value that would be unmistakable in the output if a request ever reached it.
SECRET = "violet-marmalade-9143-not-metadata"

#: A fixed clock, so a local-time assertion is about the conversion and not about
#: when the suite happened to run.
FIXED_NOW = datetime(2026, 10, 5, 14, 30, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

#: Two hours ahead of UTC, which is what a `+02:00` timestamp prints as locally.
AFTERNOON_UTC = "2026-10-05T12:00:00Z"


# --- fixtures and helpers ---------------------------------------------------


def build(tmp_path) -> Path:
    """A router database with both tables present and no rows in either.

    The schema is created directly rather than through a first `record()`, so a
    test that counts rows is counting only the rows that test added.
    """
    path = tmp_path / "router.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA + USAGE_SCHEMA)
    connection.close()
    return path


def add_decision(
    log: DecisionLog,
    *,
    session: str = "aaaa000000000001",
    requested: str = MODEL_MID,
    chosen: str = MODEL_MID,
    timestamp: str = AFTERNOON_UTC,
    action: str = "STAY",
    applied: int = 0,
    reasons: list[str] | None = None,
    classifier_score: int | None = None,
    classifier_tier: str | None = None,
) -> int:
    """Append one decision row, optionally carrying a classifier score."""
    signals: dict[str, Any] = {"message_count": 3}
    if classifier_score is not None:
        signals["classifier_score"] = classifier_score
        signals["classifier_tier"] = classifier_tier
    return log.record(
        DecisionRecord(
            session_hint=session,
            requested_model=requested,
            chosen_model=chosen,
            mode="active",
            reason_codes=reasons if reasons is not None else ["PASSTHROUGH"],
            signal_values=signals,
            action=action,
            applied=applied,
            timestamp=timestamp,
        )
    )


def add_usage(
    log: DecisionLog,
    decision_id: int,
    *,
    input_tokens: int = 1_000_000,
    output_tokens: int = 1_000_000,
    status: str = STATUS_OK,
    cost_usd: float | None = None,
    baseline_cost_usd: float | None = None,
) -> int:
    """Append one usage row for a decision, priced unless told otherwise."""
    return log.record_usage(
        UsageRow(
            status=status,
            decision_id=decision_id,
            model_reported="mock-model",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            baseline_cost_usd=baseline_cost_usd,
        )
    )


def priced(chosen: str = MODEL_MID, requested: str = MODEL_MID) -> tuple[float, float]:
    """`(cost_usd, baseline_cost_usd)` from the ACTIVE sample's dummy rates."""
    from router.config import load_config

    result = estimate_cost(
        UsageRecord(
            status=STATUS_OK,
            model_reported="mock-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=1_000_000,
            cache_write_tokens=1_000_000,
        ),
        chosen,
        requested,
        load_config(ACTIVE_CONFIG_PATH),
    )
    assert result.cost_usd is not None
    assert result.baseline_cost_usd is not None
    return result.cost_usd, result.baseline_cost_usd


def hand_computed_cost() -> float:
    """One million of each count, priced by hand from the dummy rates."""
    return 3.00 + 15.00 + 0.30 + 3.75


def no_sleep() -> Callable[[float], None]:
    """A sleep that does not sleep. Every poll in this file uses it."""

    def _record(seconds: float) -> None:
        return None

    return _record


class Recorder:
    """Collects the lines `run` emits, without touching stdout."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, line: str) -> None:
        self.lines.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def batch(self, index: int) -> list[str]:
        """The lines of one poll: everything before the next footer."""
        starts = [i for i, line in enumerate(self.lines) if line.startswith("-- ")]
        if index + 1 < len(starts):
            return self.lines[starts[index] : starts[index + 1]]
        return self.lines[starts[index] :] if starts else []


# --- cost_text: four states, four words --------------------------------------


def test_pending_is_not_zero_and_not_unknown():
    assert cost_text(None, has_usage=False, usage_table=True) == COST_PENDING
    assert cost_text(None, has_usage=False, usage_table=True) != COST_UNKNOWN
    assert "0.000000" not in cost_text(None, has_usage=False, usage_table=True)


def test_an_unpriced_row_is_unknown_and_never_a_dollar_figure():
    text = cost_text(None, has_usage=True, usage_table=True)
    assert text == COST_UNKNOWN
    assert "$" not in text


def test_a_zero_price_is_free_and_is_still_a_price():
    """`free` is a claim, not the absence of one. A priced zero must not read as
    UNKNOWN, because the two say opposite things about what is known."""
    text = cost_text(0.0, has_usage=True, usage_table=True)
    assert text == COST_FREE
    assert text != COST_UNKNOWN
    assert text != COST_PENDING


def test_a_priced_row_shows_its_figure_at_six_decimals():
    assert cost_text(0.5, has_usage=True, usage_table=True) == "$0.500000"


def test_a_missing_usage_table_is_stated_outright():
    """No usage table means nothing can be priced. It must never read as free."""
    text = cost_text(0.0, has_usage=True, usage_table=False)
    assert text == NO_USAGE_TABLE_COST
    assert COST_FREE not in text
    assert "$0.000000" not in text


# --- the footer arithmetic ---------------------------------------------------


def test_the_footer_labels_both_money_figures_as_the_known_part():
    totals = WatchTotals(requests=3, cost_known_usd=1.0, baseline_known_usd=2.0)
    footer = format_footer(totals)

    assert "$1.000000 (known part)" in footer
    assert "$2.000000 (known part)" in footer
    assert "requests 3" in footer


def test_the_difference_is_baseline_minus_routed():
    totals = WatchTotals(cost_known_usd=1.25, baseline_known_usd=2.0)
    assert totals.difference_usd == pytest.approx(0.75)
    assert "$0.750000" in format_footer(totals)


def test_a_free_model_and_an_unknown_row_in_the_same_batch(tmp_path):
    """The case the whole module exists for: one row genuinely free, one row whose
    cost nobody could work out. The free row contributes a real 0.0; the unknown
    row contributes nothing at all and is counted separately. If the unknown row
    were added as a zero the total would read complete when it is not."""
    path = build(tmp_path)
    log = DecisionLog(path)
    free_id = add_decision(log, chosen=MODEL_LOW, requested=MODEL_LOW)
    priced_id = add_decision(log, chosen=MODEL_MID, requested=MODEL_HIGH)
    add_usage(log, free_id, cost_usd=0.0, baseline_cost_usd=0.0)
    add_usage(log, priced_id, cost_usd=hand_computed_cost(), baseline_cost_usd=hand_computed_cost())
    unpriced_id = add_decision(log, chosen=MODEL_HIGH)
    add_usage(log, unpriced_id, status=STATUS_UNKNOWN, cost_usd=None, baseline_cost_usd=None)

    lines = poll_once(path, WatchOptions(once=True), local=FIXED_NOW.tzinfo)
    footer = lines[-1]

    mid, high = priced(chosen=MODEL_MID, requested=MODEL_HIGH)
    # The free row adds a real 0.0, the priced row adds exactly its cost, and the
    # UNKNOWN row adds nothing at all and is counted beside the total.
    assert money(hand_computed_cost()) in footer
    assert "routed $22.050000" in footer
    assert "UNKNOWN rows 1" in footer
    assert "requests 3" in footer
    assert mid == pytest.approx(hand_computed_cost())
    assert high == pytest.approx(hand_computed_cost())


def test_an_unknown_row_never_adds_a_cent_to_either_total():
    totals = WatchTotals()
    totals.add_money(None, None)

    assert totals.cost_known_usd == 0.0
    assert totals.baseline_known_usd == 0.0
    assert totals.unknown_rows == 1
    assert "UNKNOWN rows 1" in format_footer(totals)


def test_one_known_figure_beside_one_unknown_counts_the_row_as_unknown():
    """Half a row is not a partial total. Counting it would let a real cost hide
    behind an UNKNOWN count that does not include it."""
    totals = WatchTotals()
    totals.add_money(1.0, None)

    assert totals.cost_known_usd == 0.0
    assert totals.baseline_known_usd == 0.0
    assert totals.unknown_rows == 1


def test_a_pending_row_is_counted_separately_from_an_unknown_one():
    """`pending` can still change; `UNKNOWN` cannot. The footer reports UNKNOWN
    rows only, so a pending row must not inflate that count."""
    totals = WatchTotals(usage_table=True)
    pending = WatchRow(
        decision_id=1,
        timestamp=AFTERNOON_UTC,
        session_hint="aaaa000000000001",
        requested_model=MODEL_MID,
        chosen_model=MODEL_MID,
        action="STAY",
        applied=0,
        reason_codes=(),
        classifier_score=None,
        classifier_tier=None,
        has_usage=False,
        input_tokens=None,
        output_tokens=None,
        cost_usd=None,
        baseline_cost_usd=None,
    )
    totals.add_row(pending)

    assert totals.pending_rows == 1
    assert totals.unknown_rows == 0
    assert totals.requests == 1


def test_the_footer_says_the_cost_is_unavailable_without_a_usage_table():
    footer = format_footer(WatchTotals(requests=4, usage_table=False))
    assert NO_USAGE_TABLE_COST in footer
    assert "$" not in footer


# --- the initial batch -------------------------------------------------------


def test_the_last_five_rows_are_shown_oldest_first(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    ids = [add_decision(log, timestamp=f"2026-10-05T12:0{index}:00Z") for index in range(7)]

    with _connection(path) as connection:
        rows, newest = read_initial(connection, WatchOptions())

    assert len(rows) == 5
    assert newest == ids[-1]
    # Oldest first, so the newest request is the last line on screen.
    assert [row.decision_id for row in rows] == sorted(row.decision_id for row in rows)
    assert [row.decision_id for row in rows] == ids[-5:]


def test_a_tail_smaller_than_the_table_shows_only_that_many(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    ids = [add_decision(log, timestamp=f"2026-10-05T12:0{index}:00Z") for index in range(4)]

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions(tail=2))

    assert [row.decision_id for row in rows] == ids[-2:]


def test_since_selects_everything_from_that_timestamp(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, timestamp="2026-10-05T11:00:00Z")
    add_decision(log, timestamp="2026-10-05T12:00:00Z")
    add_decision(log, timestamp="2026-10-05T13:00:00Z")

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions(since="2026-10-05T12:00:00Z"))

    assert [row.timestamp for row in rows] == [
        "2026-10-05T12:00:00Z",
        "2026-10-05T13:00:00Z",
    ]


def test_the_first_poll_of_an_empty_database_shows_no_rows(tmp_path):
    path = build(tmp_path)
    DecisionLog(path).usage_count()

    lines = poll_once(path, WatchOptions(once=True), local=FIXED_NOW.tzinfo)

    assert len(lines) == 1, "just the footer"
    assert lines[0].startswith("-- ")
    assert "requests 0" in lines[0]


# --- rows appearing between polls -------------------------------------------


def test_a_row_written_between_polls_is_shown_on_the_next_one(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    state = WatchState()

    first = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)
    assert "requests 0" in first[-1]

    add_decision(log, chosen=MODEL_HIGH, reasons=["ESCALATE_TOOL_ERRORS"], action="SWITCH", applied=1)

    second = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert len(second) == 2, "one new row and the footer"
    assert f"{MODEL_MID} -> {MODEL_HIGH}" in second[0]
    assert "SWITCH applied" in second[0]
    assert "ESCALATE_TOOL_ERRORS" in second[0]
    assert "requests 1" in second[-1]


def test_a_row_is_never_printed_twice(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    state = WatchState()
    poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    add_decision(log)
    poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)
    third = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert len(third) == 1, "an empty batch prints only the footer"
    assert "requests 1" in third[0], "the total did not move"


def test_a_would_switch_is_not_reported_as_applied(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    decision_id = add_decision(
        log, action="SWITCH", applied=0, reasons=["DOWNGRADE_SMALL_CONTEXT"]
    )

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions())

    line = format_row(rows[0], FIXED_NOW.tzinfo, usage_table=True)
    assert "would-switch" in line
    assert "SWITCH applied" not in line
    assert decision_id


def test_a_held_row_is_labelled_held_and_not_as_a_switch(tmp_path):
    """A held row is an earlier switch being served again, not a new decision."""
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, action="SWITCH", applied=1, reasons=["HELD_MODEL"])

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions())

    line = format_row(rows[0], FIXED_NOW.tzinfo, usage_table=True)
    assert "  held  " in line
    assert "SWITCH applied" not in line


def test_a_blocked_row_is_labelled_blocked(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, reasons=["STAY", "BLOCKED_COST"])

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions())

    assert "blocked" in format_row(rows[0], FIXED_NOW.tzinfo, usage_table=True)


def test_a_kill_switch_row_is_labelled_kill_switch(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, reasons=["STAY", "KILL_SWITCH"])

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions())

    assert "kill-switch" in format_row(rows[0], FIXED_NOW.tzinfo, usage_table=True)


def test_the_session_hint_is_shown_as_its_first_six_characters(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, session="abcdef0123456789")

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions())

    line = format_row(rows[0], FIXED_NOW.tzinfo, usage_table=True)
    assert "abcdef" in line
    assert "abcdef0123456789" not in line


def test_the_timestamp_is_local_time_not_utc(tmp_path):
    """`12:00:00Z` at +05:30 is `17:30:00` locally. Showing the UTC figure while
    calling it local would be a lie in the one field a person checks against the
    clock on their wall."""
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, timestamp="2026-10-05T12:00:00Z")

    lines = poll_once(path, WatchOptions(once=True), local=FIXED_NOW.tzinfo)

    assert lines[0].startswith("17:30:00 ")


def test_an_unreadable_timestamp_is_shown_as_a_dash_not_as_the_clock():
    """A timestamp this view cannot read is unknown. Printing the current time in
    its place would put a plausible-looking time on a row that has none."""
    assert format_time("not a timestamp", FIXED_NOW.tzinfo) == "-"


# --- the classifier ----------------------------------------------------------


def test_a_classifier_score_and_tier_are_shown_when_present(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, classifier_score=90, classifier_tier="high")

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions())

    line = format_row(rows[0], FIXED_NOW.tzinfo, usage_table=True)
    assert "classifier 90/high" in line


def test_no_classifier_column_when_the_score_is_absent(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log)

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions())

    assert "classifier" not in format_row(rows[0], FIXED_NOW.tzinfo, usage_table=True)


def test_a_null_classifier_score_is_not_shown_as_zero(tmp_path):
    """A score of None means the classifier did not run, which is not a score of
    zero and not a low score."""
    path = build(tmp_path)
    log = DecisionLog(path)
    decision_id = add_decision(log)
    _set_signal(log, decision_id, {"classifier_score": None, "classifier_tier": None})

    with _connection(path) as connection:
        rows, _ = read_initial(connection, WatchOptions())

    line = format_row(rows[0], FIXED_NOW.tzinfo, usage_table=True)
    assert "classifier" not in line
    assert "classifier 0/" not in line


# --- usage arriving late -----------------------------------------------------


def test_a_decision_read_before_its_usage_shows_pending(tmp_path):
    """The usage row is written after the response ends, so this is the normal
    case, not an edge one."""
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log)

    lines = poll_once(path, WatchOptions(once=True), local=FIXED_NOW.tzinfo)

    assert COST_PENDING in lines[0]
    assert "$" not in lines[0].rsplit("in ", 1)[-1].replace(COST_PENDING, "")


def test_late_usage_is_reprinted_as_an_update_line_and_not_lost(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    state = WatchState()
    decision_id = add_decision(log)

    first = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)
    assert COST_PENDING in first[0]

    cost, baseline = priced()
    add_usage(log, decision_id, cost_usd=cost, baseline_cost_usd=baseline)

    second = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert len(second) == 2, "one update line and the footer"
    assert second[0].startswith("updated ")
    assert money(cost) in second[0]
    assert "in 1000000 out 1000000" in second[0]
    # The request count does not move: the same request, priced.
    assert "requests 1" in second[-1]
    assert money(cost) in second[-1], "the running total grew by exactly that row"


def test_a_late_update_never_inflates_the_request_count(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    state = WatchState()
    decision_id = add_decision(log)
    poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    add_usage(log, decision_id, cost_usd=1.0, baseline_cost_usd=2.0)
    poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)
    third = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert "requests 1" in third[0]
    assert "$1.000000" in third[0]
    assert "$2.000000" in third[0]


def test_the_update_line_is_not_printed_twice(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    state = WatchState()
    decision_id = add_decision(log)
    poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)
    add_usage(log, decision_id, cost_usd=1.0, baseline_cost_usd=1.0)

    second = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)
    third = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert second[0].startswith("updated ")
    assert len(third) == 1, "an empty batch prints only the footer"


def test_a_decision_and_its_usage_written_between_polls_are_counted_once(tmp_path):
    """The common case for a fast request: the whole row, both tables, lands
    between two polls. The decision line already carries the price, so the usage
    row must not be replayed as an update on top of it. Counting it twice would
    overstate the cost of every short request in the whole session."""
    path = build(tmp_path)
    log = DecisionLog(path)
    state = WatchState()
    poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    decision_id = add_decision(log, chosen=MODEL_HIGH, action="SWITCH", applied=1)
    add_usage(log, decision_id, cost_usd=1.5, baseline_cost_usd=3.0)

    lines = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert len(lines) == 2, "the decision line and the footer, and nothing else"
    assert lines[0].startswith("17:30:00 "), "a decision line, not an update line"
    assert "$1.500000" in lines[0]
    assert "routed $1.500000" in lines[-1]
    assert "baseline $3.000000" in lines[-1]
    assert "difference $1.500000" in lines[-1]
    assert "requests 1" in lines[-1]
    assert "UNKNOWN rows 0" in lines[-1]


def test_several_decisions_and_their_usage_landing_together_are_counted_once(tmp_path):
    """The same thing at a batch level, which is what a burst of short requests
    looks like. The total must be the sum of the rows and not twice the sum."""
    path = build(tmp_path)
    log = DecisionLog(path)
    state = WatchState()
    poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    for index in range(3):
        decision_id = add_decision(log, timestamp=f"2026-10-05T12:0{index}:00Z")
        add_usage(
            log,
            decision_id,
            input_tokens=10 * index,
            output_tokens=20,
            cost_usd=1.0,
            baseline_cost_usd=2.0,
        )

    lines = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert len(lines) == 4, "three decision lines and one footer"
    assert not [line for line in lines if line.startswith("updated ")]
    assert "routed $3.000000" in lines[-1]
    assert "baseline $6.000000" in lines[-1]
    assert "requests 3" in lines[-1]


def test_usage_still_arriving_for_an_older_pending_row_is_shown_after_a_new_batch(tmp_path):
    """The two things happen independently. A new row must not stop an older
    pending row from being priced later, or a request that was still in flight
    when the batch was read would never be priced at all."""
    path = build(tmp_path)
    log = DecisionLog(path)
    state = WatchState()

    pending_id = add_decision(log, timestamp="2026-10-05T12:00:00Z")
    poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    second_id = add_decision(log, timestamp="2026-10-05T12:01:00Z")
    add_usage(log, second_id, cost_usd=1.0, baseline_cost_usd=2.0)
    add_usage(log, pending_id, cost_usd=4.0, baseline_cost_usd=5.0)

    lines = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert len(lines) == 3, "one new decision, one update for the older row, a footer"
    assert lines[0].startswith("17:31:00 ")
    assert lines[1].startswith("updated ")
    assert "routed $5.000000" in lines[-1]
    assert "baseline $7.000000" in lines[-1]
    assert "requests 2" in lines[-1], "the update did not become a third request"


def test_late_usage_for_a_row_nobody_printed_is_still_shown(tmp_path):
    """The usage watermark is the table's own sequence, not a memory of which
    decisions this view printed, so a usage row is found either way."""
    path = build(tmp_path)
    log = DecisionLog(path)
    log.record_usage(
        UsageRow(
            status=STATUS_OK,
            decision_id=None,
            model_reported="mock-model",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.5,
            baseline_cost_usd=0.5,
        )
    )

    with _connection(path) as connection:
        updates = read_late_usage(connection, 0)

    assert len(updates) == 1
    assert updates[0].decision_id is None
    assert updates[0].cost_usd == pytest.approx(0.5)


def test_a_usage_row_with_no_decision_row_prints_a_dash_for_the_session(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    log.record_usage(
        UsageRow(
            status=STATUS_OK,
            decision_id=None,
            model_reported="mock-model",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.5,
            baseline_cost_usd=0.5,
        )
    )

    with _connection(path) as connection:
        updates = read_late_usage(connection, 0)

    line = format_update(updates[0], FIXED_NOW.tzinfo)
    assert line.startswith("updated ")
    assert "  -  " in line


# --- a database with no usage table ------------------------------------------


def test_a_database_with_no_usage_table_is_detected(tmp_path):
    path = build(tmp_path)
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE router_usage")
    connection.commit()
    connection.close()

    with _connection(path) as opened:
        assert has_usage_table(opened) is False


def test_decisions_are_still_shown_without_a_usage_table(tmp_path):
    """A database written before `router_usage` existed has real decisions and no
    prices. Those decisions must still appear."""
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, chosen=MODEL_HIGH, action="SWITCH", applied=1)
    connection = sqlite3.connect(path)
    connection.execute(f"DROP TABLE {USAGE_TABLE}")
    connection.commit()
    connection.close()

    lines = poll_once(path, WatchOptions(once=True), local=FIXED_NOW.tzinfo)

    assert len(lines) == 2, "the decision and the footer"
    assert f"{MODEL_MID} -> {MODEL_HIGH}" in lines[0]
    assert NO_USAGE_TABLE_COST in lines[0]
    assert NO_USAGE_TABLE_COST in lines[1]
    assert "$" not in lines[1]


def test_a_missing_usage_table_is_not_reported_as_free(tmp_path):
    path = build(tmp_path)
    connection = sqlite3.connect(path)
    connection.execute(f"DROP TABLE {USAGE_TABLE}")
    connection.commit()
    connection.close()

    lines = poll_once(path, WatchOptions(once=True), local=FIXED_NOW.tzinfo)

    assert COST_FREE not in "".join(lines)


# --- an absent database ------------------------------------------------------


def test_an_absent_database_says_it_is_waiting(tmp_path):
    missing = tmp_path / "no-such-router.sqlite3"

    lines = poll_once(missing, WatchOptions(once=True), local=FIXED_NOW.tzinfo)

    assert lines == [WAITING]


def test_an_absent_database_is_not_created(tmp_path):
    """A view must not bring a database into existence. `mode=ro` refuses to open a
    missing file at all, so the path is never created and no parent directory is
    made either."""
    missing = tmp_path / "nested" / "router.sqlite3"

    poll_once(missing, WatchOptions(once=True), local=FIXED_NOW.tzinfo)

    assert not missing.exists()
    assert not missing.parent.exists()


def test_a_database_that_appears_later_is_picked_up(tmp_path):
    """A watch started before the proxy is expected to catch up on its own."""
    path = tmp_path / "router.sqlite3"
    state = WatchState()

    first = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)
    assert first == [WAITING]

    log = DecisionLog(path)
    add_decision(log, chosen=MODEL_HIGH, action="SWITCH", applied=1)

    second = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert second[0] != WAITING
    assert f"{MODEL_MID} -> {MODEL_HIGH}" in second[0]


# --- read-only ---------------------------------------------------------------


def test_every_connection_is_opened_read_only(tmp_path, monkeypatch):
    """Enforced by SQLite rather than by discipline, so it is asserted rather than
    assumed."""
    path = build(tmp_path)
    calls: list[tuple[Any, ...]] = []
    connect = sqlite3.connect

    def spy(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    try:
        poll_once(path, WatchOptions(once=True), local=FIXED_NOW.tzinfo)
    finally:
        monkeypatch.setattr(sqlite3, "connect", connect)

    assert calls, "the watch opened no connection at all"
    for args, kwargs in calls:
        assert kwargs.get("uri") is True, f"opened without a URI: {args}"
        assert "mode=ro" in str(args[0]), f"opened writable: {args}"


def test_reading_does_not_change_the_file_size_or_mtime(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    decision_id = add_decision(log)
    add_usage(log, decision_id, cost_usd=1.0, baseline_cost_usd=2.0)

    before = path.stat()
    state = WatchState()
    for _ in range(3):
        poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)
    after = path.stat()

    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert not list(path.parent.glob("*-journal"))
    assert not list(path.parent.glob("*-wal"))


def test_polling_never_writes_a_row(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log)

    def count() -> int:
        with open_read_only(path) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM router_decisions").fetchone()[0])

    state = WatchState()
    for _ in range(4):
        poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert count() == 1, "exactly the row the test added, and nothing the watch wrote"


def test_a_watch_leaves_no_journal_even_after_many_polls(tmp_path):
    path = build(tmp_path)
    add_decision(DecisionLog(path))
    state = WatchState()
    for _ in range(10):
        poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert [p.name for p in path.parent.iterdir() if p.name != path.name] == []


# --- --once, --max-iterations, Ctrl+C ---------------------------------------


def test_once_prints_the_current_state_and_exits_zero(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, chosen=MODEL_HIGH, action="SWITCH", applied=1)
    add_decision(log)
    recorder = Recorder()

    code = run(path, WatchOptions(once=True), now=_fixed, sleep=no_sleep(), write=recorder.write)

    assert code == 0
    assert recorder.lines[0] == WATCH_NOTE
    assert len(recorder.lines) == 4, "the note, two rows and a footer"
    assert recorder.lines[-1].startswith("-- requests 2")


def test_once_does_not_sleep_at_all(tmp_path):
    """A test that really waited would be a slow test, and this one must be able
    to poll as often as it likes."""
    path = build(tmp_path)
    slept: list[float] = []

    run(
        path,
        WatchOptions(once=True),
        now=_fixed,
        sleep=slept.append,
        write=lambda line: None,
    )

    assert slept == [], "--once prints and exits, so it never pauses"


def test_max_iterations_stops_after_that_many_polls(tmp_path):
    path = build(tmp_path)
    recorder = Recorder()

    code = run(
        path,
        WatchOptions(max_iterations=3),
        now=_fixed,
        sleep=no_sleep(),
        write=recorder.write,
    )

    assert code == 0
    footers = [line for line in recorder.lines if line.startswith("-- ")]
    assert len(footers) == 3


def test_max_iterations_shows_rows_written_during_the_watch(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    recorder = Recorder()
    slept = 0

    def sleep_then_add(seconds: float) -> None:
        nonlocal slept
        slept += 1
        if slept == 1:
            add_decision(log, chosen=MODEL_HIGH, action="SWITCH", applied=1)

    run(
        path,
        WatchOptions(max_iterations=2),
        now=_fixed,
        sleep=sleep_then_add,
        write=recorder.write,
    )

    assert f"{MODEL_MID} -> {MODEL_HIGH}" in recorder.text
    assert "requests 1" in recorder.text


def test_ctrl_c_exits_zero_and_says_it_stopped(tmp_path):
    """Ctrl+C is the normal way to stop a watch, so it is not a failure and must
    not report one. An exit code of 130 would say the command broke."""
    path = build(tmp_path)
    recorder = Recorder()

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    code = run(path, WatchOptions(), now=_fixed, sleep=interrupt, write=recorder.write)

    assert code == 0
    assert recorder.lines[-1] == "stopped"


def test_ctrl_c_before_any_database_is_still_a_clean_exit(tmp_path):
    recorder = Recorder()

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    code = run(
        tmp_path / "missing.sqlite3",
        WatchOptions(),
        now=_fixed,
        sleep=interrupt,
        write=recorder.write,
    )

    assert code == 0
    assert WAITING in recorder.text
    assert recorder.lines[-1] == "stopped"


def test_the_interval_is_passed_to_the_sleep(tmp_path):
    path = build(tmp_path)
    seen: list[float] = []

    def record(seconds: float) -> None:
        seen.append(seconds)

    run(
        path,
        WatchOptions(interval=2.5, max_iterations=2),
        now=_fixed,
        sleep=record,
        write=lambda line: None,
    )

    assert seen == [2.5], "one pause between two polls"


def test_a_non_positive_interval_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "router.sqlite3"))

    code = cli.main(["watch", "--interval", "0"])
    assert code == 2
    assert "--interval" in capsys.readouterr().err


def test_a_zero_max_iterations_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "router.sqlite3"))

    code = cli.main(["watch", "--max-iterations", "0"])
    assert code == 2
    assert "--max-iterations" in capsys.readouterr().err


def test_a_bad_since_is_refused_before_anything_is_printed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "router.sqlite3"))

    code = cli.main(["watch", "--since", "yesterday", "--once"])

    assert code == 2
    assert "--since" in capsys.readouterr().err


def test_the_command_once_exits_zero(tmp_path, monkeypatch, capsys):
    """End to end through `main`, so the argument names are exercised too."""
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, chosen=MODEL_HIGH, action="SWITCH", applied=1)
    monkeypatch.setenv("ROUTER_DB", str(path))

    code = cli.main(["watch", "--once", "--no-color"])

    assert code == 0
    out = capsys.readouterr().out
    assert WATCH_NOTE in out
    assert f"{MODEL_MID} -> {MODEL_HIGH}" in out
    assert "requests 1" in out


def test_the_command_max_iterations_exits_zero(tmp_path, monkeypatch, capsys):
    path = build(tmp_path)
    monkeypatch.setenv("ROUTER_DB", str(path))

    code = cli.main(["watch", "--max-iterations", "2", "--no-color"])

    assert code == 0
    out = capsys.readouterr().out
    assert out.count("-- requests") == 2


def test_the_command_reads_the_database_from_the_environment(tmp_path, monkeypatch, capsys):
    path = build(tmp_path)
    DecisionLog(path)
    monkeypatch.setenv("ROUTER_DB", str(path))
    assert default_db_path() == path

    cli.main(["watch", "--once", "--no-color"])

    assert WATCH_NOTE in capsys.readouterr().out


# --- the output itself -------------------------------------------------------


def test_every_line_is_plain_ascii(tmp_path, monkeypatch):
    """No typographic arrow and no box characters, so it renders the same in any
    Windows console, in a file, and in CI logs. The model flow uses `->`."""
    path = build(tmp_path)
    log = DecisionLog(path)
    decision_id = add_decision(log, classifier_score=90, classifier_tier="high")
    add_usage(log, decision_id, cost_usd=1.0, baseline_cost_usd=2.0)
    recorder = Recorder()

    run(
        path,
        WatchOptions(max_iterations=2),
        now=_fixed,
        sleep=no_sleep(),
        write=recorder.write,
        color=False,
    )

    assert recorder.lines
    for line in recorder.lines:
        line.encode("ascii")
        assert "\u2192" not in line, "a typographic arrow would not render everywhere"
        for box in "\u2502\u250c\u2510\u2514\u2500\u2550\u2588\u2591":
            assert box not in line


def test_the_model_flow_uses_an_ascii_arrow(tmp_path):
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log, requested=MODEL_LOW, chosen=MODEL_HIGH, action="SWITCH", applied=1)

    lines = poll_once(path, WatchOptions(once=True), local=FIXED_NOW.tzinfo)

    assert f"{MODEL_LOW} -> {MODEL_HIGH}" in lines[0]


def test_no_colour_escapes_when_colour_is_off(tmp_path):
    path = build(tmp_path)
    recorder = Recorder()

    run(
        path,
        WatchOptions(once=True),
        now=_fixed,
        sleep=no_sleep(),
        write=recorder.write,
        color=False,
    )

    assert "\x1b[" not in recorder.text


def test_the_no_color_flag_produces_the_same_bytes_as_the_default(tmp_path, monkeypatch, capsys):
    """Colour is already off for a non-terminal, so the flag changes nothing there.
    That is the point of the default: a redirected watch is always plain."""
    path = build(tmp_path)
    log = DecisionLog(path)
    add_decision(log)
    monkeypatch.setenv("ROUTER_DB", str(path))

    cli.main(["watch", "--once", "--no-color"])
    without = capsys.readouterr().out

    cli.main(["watch", "--once"])
    with_default = capsys.readouterr().out

    assert without == with_default
    assert "\x1b[" not in without


def test_colour_adds_only_sgr_escapes_and_no_box_characters(tmp_path):
    path = build(tmp_path)
    recorder = Recorder()

    run(
        path,
        WatchOptions(once=True, color=True),
        now=_fixed,
        sleep=no_sleep(),
        write=recorder.write,
        color=True,
    )

    text = recorder.text
    assert text.startswith("\x1b[2m"), "colour is on, so the note is dimmed"
    # Stripping SGR sequences must leave the same plain text as an uncoloured run.
    plain = Recorder()
    run(
        path,
        WatchOptions(once=True, color=False),
        now=_fixed,
        sleep=no_sleep(),
        write=plain.write,
        color=False,
    )
    assert _strip_sgr(text) == plain.text


def test_the_header_names_the_two_limitations(tmp_path):
    """Every number under it is an estimate, and the two things that make it an
    estimate are stated rather than left to be discovered."""
    assert WATCH_NOTE == (
        "Router decisions and estimated cost. Estimates use list prices from "
        "router/config.yaml and observed token counts. Not a bill."
    )
    assert "Not a bill" in WATCH_NOTE
    assert "router/config.yaml" in WATCH_NOTE
    assert "observed token counts" in WATCH_NOTE


def test_the_footer_prints_even_when_a_batch_is_empty(tmp_path):
    """A view that goes quiet without saying why reads as a crashed process."""
    path = build(tmp_path)
    state = WatchState()
    poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    lines = poll_once(path, WatchOptions(), state, local=FIXED_NOW.tzinfo)

    assert len(lines) == 1
    assert lines[0].startswith("-- ")


# --- the secret phrase never leaves -----------------------------------------


def test_the_secret_phrase_never_appears_in_the_output(tmp_path):
    """Rule 1. A prompt phrase is not metadata, and the watch reads a database
    that has no column it could have come from anyway."""
    path = build(tmp_path)
    log = DecisionLog(path)
    decision_id = add_decision(log, classifier_score=90, classifier_tier="high")
    add_usage(log, decision_id, cost_usd=1.0, baseline_cost_usd=2.0)
    # A session hint derived from a message that contained the phrase.
    log2 = DecisionLog(path)
    log2.record(
        DecisionRecord(
            session_hint=DecisionLog(path).session_hint(f"please refactor {SECRET}"),
            requested_model=MODEL_MID,
            chosen_model=MODEL_MID,
            mode="active",
            reason_codes=["PASSTHROUGH"],
            timestamp="2026-10-05T12:05:00Z",
        )
    )
    recorder = Recorder()

    run(
        path,
        WatchOptions(max_iterations=3),
        now=_fixed,
        sleep=no_sleep(),
        write=recorder.write,
    )

    assert recorder.lines
    assert SECRET not in recorder.text


def test_the_secret_phrase_is_not_in_the_database_bytes_either(tmp_path):
    """So a reader of this view cannot recover it from the source it read."""
    path = build(tmp_path)
    log = DecisionLog(path)
    log.session_hint(f"please refactor {SECRET}")
    add_decision(log)
    log.record_usage(
        UsageRow(status=STATUS_OK, decision_id=1, model_reported="mock-model", cost_usd=0.0)
    )

    assert SECRET.encode() not in path.read_bytes()


# --- helpers used by the tests above ----------------------------------------


def _connection(path: Path):
    """A read-only connection to `path`, for reading rows directly."""
    return open_read_only(path)


def _fixed() -> datetime:
    """The injected clock. Fixed, so nothing here depends on the wall clock."""
    return FIXED_NOW


def _set_signal(log: DecisionLog, decision_id: int, signals: dict[str, Any]) -> None:
    """Overwrite one row's `signal_values`, for the null-classifier case.

    Written through a normal connection: this is the test arranging the fixture,
    not the watch. Only the watch is required to read read-only.
    """
    import json

    connection = sqlite3.connect(log.db_path)
    connection.execute(
        "UPDATE router_decisions SET signal_values = ? WHERE decision_id = ?",
        (json.dumps(signals), decision_id),
    )
    connection.commit()
    connection.close()


def _strip_sgr(text: str) -> str:
    """Remove SGR escape sequences, leaving the characters a person would read."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)
