"""`tamias-router switch-report`: time correlation, and nothing more.

The ledger here is created with exactly the columns the Tamias Observer's
`request_record` declares, so the report reads a table shaped like the real one
rather than one shaped to suit the code.
"""
from __future__ import annotations

import json
import re
import sqlite3

from router.cli import main
from router.decisions import DB_ENV_VAR, DecisionLog, DecisionRecord
from router.switch_report import (
    AMBIGUOUS,
    CAUSE,
    MATCH,
    MISMATCH,
    NO_LEDGER_ROWS,
    NO_REQUEST_RECORDS,
    PREVIOUS_LOOKBACK_SECONDS,
    SWITCH_REPORT_NOTE,
    UNKNOWN,
)

#: The Observer's `request_record` columns, exactly as its schema declares.
#: Every column is nullable: NULL means unknown, never zero.
LEDGER_SCHEMA = """
CREATE TABLE request_record (
    record_uid                TEXT,
    session_id                TEXT,
    seq                       INTEGER,
    ts_utc                    TEXT,
    request_id                TEXT,
    is_sidechain              INTEGER,
    request_bucket            TEXT,
    agent                     TEXT,
    agent_version             TEXT,
    model_id                  TEXT,
    effort                    TEXT,
    input_tokens              INTEGER,
    output_tokens             INTEGER,
    cache_read_tokens         INTEGER,
    cache_write_tokens        INTEGER,
    cache_write_5m_tokens     INTEGER,
    cache_write_1h_tokens     INTEGER,
    cache_write_total_tokens  INTEGER,
    is_zero_usage             INTEGER,
    is_synthetic              INTEGER,
    dedup_conflict            INTEGER
)
"""

#: In an unrelated column, so it can only appear in the output if the report is
#: reading columns it was not asked to read.
UNRELATED = "zebra-quetzal-4417-not-a-printed-column"

SESSION = "sess-aaaa1111bbbb2222"
OTHER_SESSION = "sess-cccc3333dddd4444"

#: ts_utc, session_id, seq, model_id, input_tokens, overrides.
ROWS = (
    ("2026-10-01T09:00:00Z", SESSION, 1, "test-low", 100, {}),
    ("2026-10-01T10:00:20Z", SESSION, 2, "test-high", 200, {}),
    ("2026-10-01T10:00:30Z", SESSION, 3, "test-high", 300, {"request_bucket": "sidechain"}),
    ("2026-10-01T10:00:35Z", SESSION, 4, "test-high", 400, {"is_sidechain": 1}),
    ("2026-10-01T10:00:40Z", SESSION, 5, None, None, {}),
    ("2026-10-01T10:01:30Z", SESSION, 6, "test-mid", 500, {}),
    ("2026-10-01T10:03:00Z", SESSION, 7, "test-high", 600, {}),
    ("2026-10-01T10:10:00Z", SESSION, 8, "test-high", 700, {}),
    ("2026-10-01T10:20:00Z", OTHER_SESSION, 1, "test-high", 800, {}),
    ("2026-10-01T10:30:00Z", SESSION, 9, "test-high", 900, {"agent": UNRELATED}),
    ("not-a-timestamp", SESSION, 10, "test-high", 1000, {}),
)


def make_ledger(path, rows=ROWS, schema=LEDGER_SCHEMA):
    """A ledger holding `rows`, written with the Observer's column order."""
    connection = sqlite3.connect(path)
    try:
        if schema is not None:
            connection.executescript(schema)
        for ts_utc, session_id, seq, model_id, input_tokens, overrides in rows:
            values = {
                "record_uid": f"rec-{(session_id or 'unknown')[:8]}-{seq}",
                "session_id": session_id,
                "seq": seq,
                "ts_utc": ts_utc,
                "request_id": f"req-{seq}",
                "is_sidechain": 0,
                "request_bucket": "main",
                "agent": "claude-code",
                "agent_version": "2.0.0",
                "model_id": model_id,
                "effort": "high",
                "input_tokens": input_tokens,
                "output_tokens": 50,
                "cache_read_tokens": 10,
                "cache_write_tokens": 5,
                "cache_write_5m_tokens": 3,
                "cache_write_1h_tokens": 2,
                "cache_write_total_tokens": 5,
                "is_zero_usage": 0,
                "is_synthetic": 0,
                "dedup_conflict": 0,
            }
            values.update(overrides)
            columns = ", ".join(values)
            connection.execute(
                f"INSERT INTO request_record ({columns}) VALUES ({', '.join('?' * len(values))})",
                tuple(values.values()),
            )
        connection.commit()
    finally:
        connection.close()
    return path


def make_router(tmp_path, monkeypatch, switches=("2026-10-01T10:00:00Z",)):
    """A decision log holding one approved switch per timestamp given."""
    path = tmp_path / "router.sqlite3"
    monkeypatch.setenv(DB_ENV_VAR, str(path))
    log = DecisionLog(path)
    for index, timestamp in enumerate(switches, start=1):
        log.record(
            DecisionRecord(
                session_hint="aaaa000000000001",
                requested_model="test-low",
                chosen_model="test-high",
                mode="active",
                reason_codes=["ESCALATE_TOOL_ERRORS"],
                action="SWITCH",
                applied=1,
                request_index_in_session=index,
                timestamp=timestamp,
            )
        )
    return path


def held_row(log, timestamp="2026-10-01T10:00:50Z"):
    log.record(
        DecisionRecord(
            session_hint="aaaa000000000001",
            requested_model="test-low",
            chosen_model="test-high",
            mode="active",
            reason_codes=["HELD_MODEL"],
            action="SWITCH",
            applied=1,
            request_index_in_session=99,
            timestamp=timestamp,
        )
    )


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def report_for(ledger, *extra):
    return ["switch-report", "--ledger", str(ledger), *extra]


def totals(text):
    """The footer's label and value pairs, so spacing never matters."""
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"^  ([a-z ]+?)\s{2,}(\S.*?)\s*$", text, re.MULTILINE)
    }


def newest_switch(capsys, ledger, *extra):
    """The newest switch entry of a JSON report, and the whole object."""
    code, out, err = run(capsys, *report_for(ledger, "--json", *extra))
    assert code == 0, err
    data = json.loads(out)
    return data["switches"][0], data


def window_of(entry):
    return [row["ts_utc"] for row in entry["ledger_rows"]]


# --- header, cause, and the absence of money --------------------------------


def test_the_header_says_this_is_not_cost_and_not_cause(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert out.splitlines()[0] == SWITCH_REPORT_NOTE
    assert "NOT the cost of the switch" in SWITCH_REPORT_NOTE
    assert "NOT a causal effect" in SWITCH_REPORT_NOTE
    assert "$" not in out
    assert "usd" not in out.lower()


def test_every_switch_carries_the_same_unknown_cause(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch, ("2026-10-01T10:00:00Z", "2026-10-01T10:10:00Z"))
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert out.count(f"cause: {CAUSE}") == 2


def test_an_unrelated_column_never_reaches_the_output(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, err = run(capsys, *report_for(ledger))

    assert code == 0
    assert UNRELATED not in out
    assert UNRELATED not in err
    # Neither do the other columns the report was not asked to read.
    assert "claude-code" not in out
    assert "output_tokens" not in out


# --- what a switch shows ---------------------------------------------------


def test_a_switch_shows_its_own_metadata(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert re.search(r"^  router timestamp: 2026-10-01T10:00:00Z$", out, re.MULTILINE)
    assert "  requested: test-low -> test-high" in out
    assert "  session hint: aaaa000000000001" in out
    assert "  reason codes: ESCALATE_TOOL_ERRORS" in out


def test_the_window_holds_the_ledger_rows_around_the_switch(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))
    assert code == 0
    # The table carries the ledger's own header names, and nothing else.
    for column in (
        "ts_utc",
        "session_id",
        "model_id",
        "input_tokens",
        "cache_read_tokens",
        "cache_write_5m_tokens",
        "cache_write_1h_tokens",
        "cache_write_total_tokens",
    ):
        assert column in out

    entry, _ = newest_switch(capsys, ledger)
    assert window_of(entry) == [
        "2026-10-01T10:00:20Z",
        "2026-10-01T10:00:40Z",
        "2026-10-01T10:01:30Z",
    ]


def test_a_session_id_is_shown_as_its_first_eight_characters(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert SESSION[:8] in out, "the first eight characters are shown"
    assert SESSION not in out, "the whole session id must not be printed"
    assert OTHER_SESSION not in out

    # A window wide enough to reach the other session shows it the same way.
    code, wide, _ = run(capsys, *report_for(ledger, "--window-seconds", "1200"))
    assert code == 0
    assert OTHER_SESSION[:8] in wide
    assert OTHER_SESSION not in wide


# --- window boundaries and skew --------------------------------------------


def test_the_skew_reaches_back_before_the_switch(tmp_path, monkeypatch, capsys):
    """A row 3s before the switch is inside a 5s skew and outside a 0s one."""
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(("2026-10-01T09:59:57Z", SESSION, 1, "test-low", 100, {}),),
    )

    entry, _ = newest_switch(capsys, ledger, "--skew-seconds", "5")
    assert window_of(entry) == ["2026-10-01T09:59:57Z"]

    entry, _ = newest_switch(capsys, ledger, "--skew-seconds", "0")
    assert window_of(entry) == []


def test_the_window_end_is_inclusive_and_bounded(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(
            ("2026-10-01T10:02:00Z", SESSION, 1, "test-high", 100, {}),
            ("2026-10-01T10:02:01Z", SESSION, 2, "test-high", 100, {}),
        ),
    )

    entry, _ = newest_switch(capsys, ledger)

    assert window_of(entry) == ["2026-10-01T10:02:00Z"], "the window end is inclusive"


def test_a_wider_window_finds_more_rows(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    entry, _ = newest_switch(capsys, ledger, "--window-seconds", "30")
    assert "2026-10-01T10:10:00Z" not in window_of(entry)

    entry, _ = newest_switch(capsys, ledger, "--window-seconds", "1200")
    assert "2026-10-01T10:10:00Z" in window_of(entry)
    assert "2026-10-01T10:20:00Z" in window_of(entry)


# --- main chain only --------------------------------------------------------


def test_sidechain_rows_are_excluded(tmp_path, monkeypatch, capsys):
    """Neither a sidechain bucket nor an `is_sidechain` flag is the main chain."""
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert "2026-10-01T10:00:30Z" not in out, "request_bucket sidechain"
    assert "2026-10-01T10:00:35Z" not in out, "is_sidechain 1"

    entry, _ = newest_switch(capsys, ledger)
    assert "2026-10-01T10:00:30Z" not in window_of(entry)
    assert "2026-10-01T10:00:35Z" not in window_of(entry)


# --- NULL is UNKNOWN, never zero -------------------------------------------


def test_null_prints_unknown_and_never_zero(tmp_path, monkeypatch, capsys):
    """The 10:00:40 row has a NULL model and NULL tokens."""
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    line = next(line for line in out.splitlines() if line.startswith("2026-10-01T10:00:40Z"))
    assert UNKNOWN in line
    cells = [cell.strip() for cell in line.split()]
    assert "0" not in cells, f"a NULL was printed as zero: {line}"
    # Its input_tokens is NULL, so it must not read as a number at all.
    assert cells[3] == UNKNOWN


def test_a_null_session_id_is_unknown_and_does_not_invent_ambiguity(
    tmp_path, monkeypatch, capsys
):
    """One unknown session is not two sessions."""
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(
            ("2026-10-01T10:00:10Z", None, 1, "test-high", 100, {}),
            ("2026-10-01T10:00:20Z", SESSION, 2, "test-high", 100, {}),
        ),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert "AMBIGUOUS" not in out
    assert UNKNOWN in out


# --- the previous request ---------------------------------------------------


def test_the_previous_request_is_the_last_row_before_the_switch(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(
            ("2026-10-01T09:56:00Z", SESSION, 1, "test-low", 100, {}),
            ("2026-10-01T10:00:20Z", SESSION, 2, "test-high", 200, {}),
        ),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert "previous request, the last main-chain row before the switch:" in out

    entry, _ = newest_switch(capsys, ledger)
    # 09:56:00 is the last row before 10:00:00; the 10:00:20 row is after it.
    assert entry["previous_request"]["ts_utc"] == "2026-10-01T09:56:00Z"
    assert entry["previous_request"]["model_id"] == "test-low"
    # The previous request is not part of the window itself.
    assert window_of(entry) == ["2026-10-01T10:00:20Z"]


def test_the_previous_request_may_be_none_found(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(("2026-10-01T10:00:20Z", SESSION, 1, "test-high", 100, {}),),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert "  previous request: none found" in out


def test_the_previous_request_ignores_rows_beyond_the_lookback(
    tmp_path, monkeypatch, capsys
):
    """A row older than ten minutes is not a previous request."""
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(("2026-10-01T09:49:00Z", SESSION, 1, "test-low", 100, {}),),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert "  previous request: none found" in out
    assert "2026-10-01T09:49:00Z" not in out

    entry, _ = newest_switch(capsys, ledger)
    assert entry["previous_request"] is None
    assert PREVIOUS_LOOKBACK_SECONDS == 600


# --- the model check --------------------------------------------------------


def test_a_matching_model_is_a_match(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert "  model_check: MATCH" in out


def test_a_different_model_is_a_mismatch(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(("2026-10-01T10:00:10Z", SESSION, 1, "test-low", 100, {}),),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert f"  model_check: {MISMATCH}" in out
    assert f"  model_check: {MATCH}" not in out


def test_the_model_check_uses_the_first_row_at_or_after_the_switch(
    tmp_path, monkeypatch, capsys
):
    """A row before the switch does not decide the check."""
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(
            ("2026-10-01T09:59:58Z", SESSION, 1, "test-low", 100, {}),
            ("2026-10-01T10:00:06Z", SESSION, 2, "test-high", 100, {}),
        ),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert f"  model_check: {MATCH}" in out
    # The 09:59:58 row is in the window but before the switch, so it is skipped.
    entry, _ = newest_switch(capsys, ledger)
    assert window_of(entry) == ["2026-10-01T09:59:58Z", "2026-10-01T10:00:06Z"]


def test_a_null_model_is_an_unknown_check(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(("2026-10-01T10:00:10Z", SESSION, 1, None, 100, {}),),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert f"  model_check: {UNKNOWN}" in out


def test_a_window_with_no_row_after_the_switch_is_an_unknown_check(
    tmp_path, monkeypatch, capsys
):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(("2026-10-01T09:59:58Z", SESSION, 1, "test-low", 100, {}),),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert "  previous request, the last main-chain row before the switch:" in out
    assert f"  model_check: {UNKNOWN}" in out
    assert NO_LEDGER_ROWS not in out


# --- ambiguity --------------------------------------------------------------


def test_two_sessions_in_one_window_is_ambiguous(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(
            ("2026-10-01T10:00:10Z", SESSION, 1, "test-high", 100, {}),
            ("2026-10-01T10:00:20Z", OTHER_SESSION, 1, "test-high", 100, {}),
        ),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert AMBIGUOUS.format(sessions=2) in out
    assert totals(out)["ambiguous windows"] == "1"


def test_a_switch_with_no_rows_says_so(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(("2026-10-01T12:00:00Z", SESSION, 1, "test-high", 100, {}),),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert NO_LEDGER_ROWS in out
    assert "AMBIGUOUS" not in out
    assert totals(out)["switches with no ledger rows"] == "1"
    assert f"  model_check: {UNKNOWN}" in out


# --- held rows are not switches --------------------------------------------


def test_a_held_row_is_never_reported_as_a_switch(tmp_path, monkeypatch, capsys):
    router_database = make_router(tmp_path, monkeypatch)
    held_row(DecisionLog(router_database))
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert totals(out)["switches shown"] == "1"
    assert "10:00:50" not in out, "the held row was reported as a switch"
    assert out.count(f"cause: {CAUSE}") == 1


def test_only_applied_switches_are_reported(tmp_path, monkeypatch, capsys):
    router_database = make_router(tmp_path, monkeypatch)
    log = DecisionLog(router_database)
    log.record(
        DecisionRecord(
            session_hint="aaaa000000000001",
            requested_model="test-low",
            chosen_model="test-low",
            mode="shadow",
            reason_codes=["ESCALATE_TOOL_ERRORS"],
            action="SWITCH",
            applied=0,
            request_index_in_session=50,
            timestamp="2026-10-01T10:05:00Z",
        )
    )
    log.record(
        DecisionRecord(
            session_hint="aaaa000000000001",
            requested_model="test-low",
            chosen_model="test-low",
            mode="active",
            reason_codes=["NO_RULE_MATCHED"],
            action="STAY",
            applied=0,
            request_index_in_session=51,
            timestamp="2026-10-01T10:06:00Z",
        )
    )
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert totals(out)["switches shown"] == "1"
    assert "10:05:00Z" not in out
    assert "10:06:00Z" not in out


def test_last_limits_the_switches_shown_newest_first(tmp_path, monkeypatch, capsys):
    make_router(
        tmp_path,
        monkeypatch,
        ("2026-10-01T10:00:00Z", "2026-10-01T10:10:00Z", "2026-10-01T10:20:00Z"),
    )
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger, "--last", "2"))

    assert code == 0
    assert totals(out)["switches shown"] == "2"
    assert "switch 1 of 2" in out
    assert "switch 2 of 2" in out

    _, data = newest_switch(capsys, ledger, "--last", "2")
    shown = [entry["router_timestamp"] for entry in data["switches"]]
    assert shown == ["2026-10-01T10:20:00Z", "2026-10-01T10:10:00Z"]


def test_since_skips_older_switches(tmp_path, monkeypatch, capsys):
    make_router(
        tmp_path,
        monkeypatch,
        ("2026-10-01T10:00:00Z", "2026-10-01T10:20:00Z"),
    )
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger, "--since", "2026-10-01T10:10:00Z"))

    assert code == 0
    assert totals(out)["switches shown"] == "1"

    _, data = newest_switch(capsys, ledger, "--since", "2026-10-01T10:10:00Z")
    assert [entry["router_timestamp"] for entry in data["switches"]] == [
        "2026-10-01T10:20:00Z"
    ]


# --- timestamps that cannot be read ----------------------------------------


def test_an_unparseable_ledger_timestamp_is_counted_not_fatal(
    tmp_path, monkeypatch, capsys
):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert totals(out)["unparseable timestamps"] == "router 0, ledger 1"
    assert "not-a-timestamp" not in out


def test_ledger_timestamps_are_read_in_several_iso_forms(tmp_path, monkeypatch, capsys):
    """With or without `Z`, with fractional seconds, with an offset."""
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(
            ("2026-10-01T10:00:10", SESSION, 1, "test-high", 100, {}),
            ("2026-10-01T10:00:20.123456Z", SESSION, 2, "test-high", 100, {}),
            ("2026-10-01T12:00:30+02:00", SESSION, 3, "test-high", 100, {}),
        ),
    )

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert totals(out)["unparseable timestamps"] == "router 0, ledger 0"
    assert "2026-10-01T10:00:10" in out
    assert "2026-10-01T10:00:20.123456Z" in out
    # 12:00:30+02:00 is 10:00:30Z, so it belongs to this window.
    assert "2026-10-01T12:00:30+02:00" in out


def test_an_unparseable_router_timestamp_is_counted_not_fatal(tmp_path, monkeypatch, capsys):
    router_database = make_router(tmp_path, monkeypatch)
    connection = sqlite3.connect(router_database)
    connection.execute(
        "UPDATE router_decisions SET timestamp = 'never' WHERE decision_id = 1"
    )
    connection.commit()
    connection.close()
    log = DecisionLog(router_database)
    log.record(
        DecisionRecord(
            session_hint="aaaa000000000001",
            requested_model="test-low",
            chosen_model="test-high",
            mode="active",
            reason_codes=["ESCALATE_TOOL_ERRORS"],
            action="SWITCH",
            applied=1,
            request_index_in_session=2,
            timestamp="2026-10-01T10:10:00Z",
        )
    )
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert totals(out)["unparseable timestamps"] == "router 1, ledger 1"
    assert totals(out)["switches shown"] == "1"


# --- ledgers with nothing to correlate -------------------------------------


def test_a_missing_ledger_says_what_to_do_and_exits_zero(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)

    code, out, err = run(capsys, *report_for(tmp_path / "not-created.sqlite3"))

    assert code == 0
    assert NO_REQUEST_RECORDS in out
    assert "run a real Claude Code session" in out
    assert "Observer scan" in out
    assert err == ""
    assert not (tmp_path / "not-created.sqlite3").exists()


def test_an_empty_ledger_table_says_what_to_do_and_exits_zero(
    tmp_path, monkeypatch, capsys
):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3", rows=())

    code, out, err = run(capsys, *report_for(ledger))

    assert code == 0
    assert NO_REQUEST_RECORDS in out
    assert err == ""


def test_a_ledger_without_the_table_exits_non_zero(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(),
        schema="CREATE TABLE something_else (id INTEGER, note TEXT)",
    )

    code, out, err = run(capsys, *report_for(ledger))

    assert code == 1
    assert "no request_record table" in err
    assert out == ""


def test_a_ledger_with_no_approved_switches_says_so(tmp_path, monkeypatch, capsys):
    """A decision log that holds nothing but a held row has no switch to show."""
    router_database = tmp_path / "router.sqlite3"
    monkeypatch.setenv(DB_ENV_VAR, str(router_database))
    held_row(DecisionLog(router_database))
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert "no approved switches to report" in out
    assert totals(out)["switches shown"] == "0"


def test_a_missing_router_database_is_not_created(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "never" / "router.sqlite3"))
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger))

    assert code == 0
    assert "no decisions recorded" in out
    assert not (tmp_path / "never").exists()


# --- JSON -------------------------------------------------------------------


def test_json_carries_the_same_numbers_as_the_text(tmp_path, monkeypatch, capsys):
    make_router(
        tmp_path,
        monkeypatch,
        ("2026-10-01T10:00:00Z", "2026-10-01T11:00:00Z"),
    )
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(
            ("2026-10-01T10:00:20Z", SESSION, 1, "test-high", 100, {}),
            ("2026-10-01T11:00:20Z", OTHER_SESSION, 1, "test-high", 100, {}),
            ("2026-10-01T11:00:40Z", SESSION, 2, "test-mid", 100, {}),
        ),
    )

    code, out, _ = run(capsys, *report_for(ledger, "--json"))

    assert code == 0
    data = json.loads(out)
    assert data["note"] == SWITCH_REPORT_NOTE
    assert data["read_only"] is True
    assert data["cause"] == CAUSE
    assert data["window"] == {
        "skew_seconds": 5,
        "window_seconds": 120,
        "previous_lookback_seconds": 600,
    }
    assert data["totals"] == {
        "switches_shown": 2,
        "switches_with_no_ledger_rows": 0,
        "ambiguous_windows": 1,
        # The newest window's first at-or-after row is test-high too, so both
        # windows match; ambiguity does not make a check UNKNOWN.
        "model_checks": {"MATCH": 2, "MISMATCH": 0, "UNKNOWN": 0},
        "unparseable_router_timestamps": 0,
        "unparseable_ledger_timestamps": 0,
    }

    newest, oldest = data["switches"]
    assert newest["router_timestamp"] == "2026-10-01T11:00:00Z"
    assert newest["requested_model"] == "test-low"
    assert newest["chosen_model"] == "test-high"
    assert newest["session_hint"] == "aaaa000000000001"
    assert newest["reason_codes"] == ["ESCALATE_TOOL_ERRORS"]
    assert newest["cause"] == CAUSE
    assert newest["model_check"] == MATCH
    assert newest["sessions_in_window"] == 2
    assert newest["ambiguity"] == AMBIGUOUS.format(sessions=2)
    assert [row["session_id"] for row in newest["ledger_rows"]] == [
        OTHER_SESSION,
        SESSION,
    ]

    assert oldest["model_check"] == MATCH
    assert oldest["ambiguity"] is None
    assert oldest["no_ledger_rows"] is False
    assert oldest["previous_request"] is None


def test_json_keeps_a_null_as_null(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(
        tmp_path / "ledger.sqlite3",
        rows=(("2026-10-01T10:00:20Z", SESSION, 1, None, None, {}),),
    )

    code, out, _ = run(capsys, *report_for(ledger, "--json"))

    assert code == 0
    row = json.loads(out)["switches"][0]["ledger_rows"][0]
    assert row["model_id"] is None
    assert row["input_tokens"] is None
    assert row["cache_read_tokens"] == 10


def test_json_prints_no_dollar_figure(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, out, _ = run(capsys, *report_for(ledger, "--json"))

    assert code == 0
    assert "$" not in out
    assert "usd" not in out.lower()


# --- read-only --------------------------------------------------------------


def test_both_databases_are_never_opened_writable(tmp_path, monkeypatch, capsys):
    router_database = make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (router_database, ledger)
    }

    calls = []
    connect = sqlite3.connect

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return connect(*args, **kwargs)

    original = sqlite3.connect
    try:
        sqlite3.connect = spy
        code, _, _ = run(capsys, *report_for(ledger))
    finally:
        sqlite3.connect = original

    assert code == 0
    for path, (size, mtime) in before.items():
        assert path.stat().st_size == size, f"{path.name} changed size"
        assert path.stat().st_mtime_ns == mtime, f"{path.name} changed mtime"
    assert calls, "no connection was opened at all"
    for args, kwargs in calls:
        assert kwargs.get("uri") is True, f"opened without a URI: {args}"
        assert "mode=ro" in str(args[0]), f"opened writable: {args}"
    opened = {str(args[0]).split("?")[0] for args, _ in calls}
    assert any("router.sqlite3" in uri for uri in opened)
    assert any("ledger.sqlite3" in uri for uri in opened)
    assert not list(tmp_path.glob("*journal*"))


# --- arguments that would be guessed at ------------------------------------


def test_a_bad_since_is_refused_rather_than_guessed(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    code, _, err = run(capsys, *report_for(ledger, "--since", "last tuesday"))

    assert code == 2
    assert "2026-10-01T00:00:00Z" in err


def test_negative_windows_are_refused(tmp_path, monkeypatch, capsys):
    make_router(tmp_path, monkeypatch)
    ledger = make_ledger(tmp_path / "ledger.sqlite3")

    for flag in ("--window-seconds", "--skew-seconds"):
        code, _, err = run(capsys, *report_for(ledger, flag, "-5"))
        assert code == 2
        assert flag in err