"""Seed two TEMP databases so `tamias-router switch-report` can be seen working.

Creates, under a fresh temporary directory and nowhere else:

  * a router decision log, written through the router's own `DecisionLog`, with
    four approved switches: one whose window holds a ledger row using the chosen
    model (MATCH), one using another model (MISMATCH), one whose window holds
    two sessions at once (AMBIGUOUS), and one with no ledger row anywhere near
    it (NO LEDGER ROWS);
  * a Tamias Observer style ledger, holding `request_record` with exactly the
    columns the Observer's schema declares.

Nothing real is touched: no server is started, no home directory is read or
written, and both files live in a temporary directory that is printed so it can
be deleted. Run it, then run the command it prints.

    python tools/demo_join.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.decisions import DecisionLog, DecisionRecord  # noqa: E402

#: The Observer's `request_record` columns, exactly as its schema declares them.
#: Nothing is inferred and nothing is added: this demo writes what was verified.
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

#: switch time, requested, chosen, and the ledger rows around it.
#:
#: Every switch is `low -> high` so the model check has something to compare
#: against, and the four cases are the four the report distinguishes.
SWITCHES = (
    ("2026-10-01T10:00:00Z", "test-low", "test-high", "match"),
    ("2026-10-01T11:00:00Z", "test-low", "test-high", "mismatch"),
    ("2026-10-01T12:00:00Z", "test-low", "test-high", "ambiguous"),
    ("2026-10-01T13:00:00Z", "test-low", "test-high", "empty"),
)

SESSION = "sess-aaaa1111bbbb2222"
OTHER_SESSION = "sess-cccc3333dddd4444"


def ledger_row(
    ts_utc: str,
    session_id: str,
    seq: int,
    model_id: str | None,
    input_tokens: int | None = 1000,
    cache_read: int | None = 200,
    write_5m: int | None = 50,
    write_1h: int | None = 25,
    write_total: int | None = 75,
    request_bucket: str = "main",
    is_sidechain: int = 0,
) -> tuple:
    """One row, in the column order of `LEDGER_SCHEMA`.

    A NULL is written wherever a value is unknown, so the report prints
    `UNKNOWN` rather than a zero the ledger never claimed.
    """
    return (
        f"rec-{session_id[:8]}-{seq}",
        session_id,
        seq,
        ts_utc,
        f"req-{seq}",
        is_sidechain,
        request_bucket,
        "claude-code",
        "2.0.0",
        model_id,
        "high",
        input_tokens,
        300,
        cache_read,
        10,
        write_5m,
        write_1h,
        write_total,
        0,
        0,
        0,
    )


#: The rows, one group per case.
LEDGER_ROWS = (
    # A previous request before the MATCH switch, and the row inside its window.
    ledger_row("2026-10-01T09:58:30Z", SESSION, 1, "test-low", 800),
    ledger_row("2026-10-01T10:00:20Z", SESSION, 2, "test-high", 1200),
    # A previous request before the MISMATCH switch, and the row after it.
    ledger_row("2026-10-01T10:59:00Z", SESSION, 3, "test-low", 700),
    ledger_row("2026-10-01T11:00:30Z", SESSION, 4, "test-mid", 1500),
    # Two different sessions inside the AMBIGUOUS switch's window.
    ledger_row("2026-10-01T11:59:40Z", SESSION, 5, "test-low", 600),
    ledger_row("2026-10-01T12:00:10Z", SESSION, 6, "test-high", 1800),
    ledger_row("2026-10-01T12:00:40Z", OTHER_SESSION, 1, "test-high", 1900),
    # A timestamp this tool cannot read: counted, never guessed into a window.
    ledger_row("not-a-timestamp", SESSION, 7, "test-high", 100),
    # A sidechain row: never part of the main chain, so never correlated.
    ledger_row(
        "2026-10-01T10:00:30Z",
        SESSION,
        8,
        "test-low",
        10,
        request_bucket="sidechain",
        is_sidechain=1,
    ),
)


def seed_router_database(path: Path) -> None:
    """Four approved switches, written by the router's own decision log."""
    log = DecisionLog(path)
    for index, (timestamp, requested, chosen, _case) in enumerate(SWITCHES, start=1):
        log.record(
            DecisionRecord(
                session_hint="aaaa000000000001",
                requested_model=requested,
                chosen_model=chosen,
                mode="active",
                reason_codes=["ESCALATE_TOOL_ERRORS"],
                action="SWITCH",
                applied=1,
                request_index_in_session=index,
                timestamp=timestamp,
            )
        )
    # A held row: the same switch served again. It must not appear in the report.
    log.record(
        DecisionRecord(
            session_hint="aaaa000000000001",
            requested_model="test-low",
            chosen_model="test-high",
            mode="active",
            reason_codes=["HELD_MODEL"],
            action="SWITCH",
            applied=1,
            request_index_in_session=5,
            timestamp="2026-10-01T14:00:00Z",
        )
    )


def seed_ledger(path: Path) -> None:
    """A `request_record` table with the Observer's columns and demo rows."""
    connection = sqlite3.connect(path)
    try:
        connection.executescript(LEDGER_SCHEMA)
        placeholders = ", ".join("?" * 21)
        connection.executemany(
            f"INSERT INTO request_record VALUES ({placeholders})", LEDGER_ROWS
        )
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    directory = Path(tempfile.mkdtemp(prefix="tamias-switch-demo-"))
    router_database = directory / "router.sqlite3"
    ledger = directory / "ledger.sqlite3"

    seed_router_database(router_database)
    seed_ledger(ledger)

    print(f"temporary directory: {directory}")
    print(f"  router decision log: {router_database}")
    print(f"  observer ledger:     {ledger}")
    print()
    print("four approved switches: one MATCH, one MISMATCH, one AMBIGUOUS window,")
    print("one with no ledger rows, plus a held row that must not be reported.")
    print()
    print("run this:")
    print()
    print(f'$env:ROUTER_DB = "{router_database}"')
    print(
        f'.\\.venv\\Scripts\\tamias-router.exe switch-report --ledger "{ledger}" --last 10'
    )
    print()
    print("and the same report as JSON:")
    print(
        f'.\\.venv\\Scripts\\tamias-router.exe switch-report --ledger "{ledger}" --json'
    )
    print()
    print(f"delete the demo with: Remove-Item -LiteralPath \"{directory}\" -Recurse")
    return 0


if __name__ == "__main__":
    sys.exit(main())