"""`tamias-router ledger-info`: schema in, no row content out.

This command exists so the Tamias Observer ledger's shape can be seen before
anything claims to understand it. These tests check the shape it prints, and
that no row value can appear in its output.
"""
from __future__ import annotations

import re
import sqlite3

import pytest

from router.cli import main
from router.decisions import DecisionLog, DecisionRecord
from router.ledger import LEDGER_NOTE

#: A value that would be unmistakable in the output if a row were ever printed.
SECRET = "purple-heron-8f31c0de-not-a-schema-name"


def build(tmp_path):
    """A database with a router decision table and an observer-style table."""
    path = tmp_path / "ledger.sqlite3"
    log = DecisionLog(path)
    log.record(
        DecisionRecord(
            session_hint="aaaa000000000001",
            requested_model="test-low",
            chosen_model="test-high",
            mode="active",
            reason_codes=["ESCALATE_TOOL_ERRORS"],
            action="SWITCH",
            applied=1,
        )
    )
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE observer_events ("
        "event_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "kind TEXT NOT NULL,"
        "occurred_at TEXT,"
        "detail TEXT)"
    )
    connection.execute("INSERT INTO observer_events (kind, occurred_at, detail) VALUES (?, ?, ?)", ("session.end", "2026-10-01T10:00:00Z", SECRET))
    connection.execute("INSERT INTO observer_events (kind, occurred_at) VALUES (?, ?)", ("session.start", "2026-10-01T09:59:00Z"))
    connection.commit()
    connection.close()
    return path


def run(capsys, *argv):
    """Run the CLI and return `(exit_code, stdout, stderr)`."""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_it_lists_every_table_with_its_columns(tmp_path, capsys):
    path = build(tmp_path)

    code, out, err = run(capsys, "ledger-info", "--ledger", str(path))

    assert code == 0
    assert err == ""
    assert out.splitlines()[0] == LEDGER_NOTE
    assert f"ledger: {path}" in out
    assert "tables: 3" in out
    assert "\nobserver_events\n" in out
    assert "\nrouter_decisions\n" in out
    assert "\nrouter_usage\n" in out
    assert re.search(r"^observer_events\n  rows\s+2$", out, re.MULTILINE)
    assert re.search(r"^router_decisions\n  rows\s+1$", out, re.MULTILINE)
    assert re.search(r"^router_usage\n  rows\s+0$", out, re.MULTILINE)


def test_it_prints_column_names_and_declared_types(tmp_path, capsys):
    path = build(tmp_path)

    code, out, _ = run(capsys, "ledger-info", "--ledger", str(path))

    assert code == 0
    for line in (
        r"^    event_id\s+INTEGER$",
        r"^    kind\s+TEXT$",
        r"^    occurred_at\s+TEXT$",
        r"^    detail\s+TEXT$",
        r"^    decision_id\s+INTEGER$",
        r"^    timestamp\s+TEXT$",
        r"^    session_hint\s+TEXT$",
        r"^    applied\s+INTEGER$",
        r"^    reason_codes\s+TEXT$",
    ):
        assert re.search(line, out, re.MULTILINE), f"missing column line {line}"


def test_it_prints_no_row_contents(tmp_path, capsys):
    """The one thing this command must never do is print a row's value."""
    path = build(tmp_path)

    code, out, err = run(capsys, "ledger-info", "--ledger", str(path))

    assert code == 0
    assert SECRET not in out
    assert SECRET not in err
    # Not just the planted phrase: no value from the row is echoed either.
    assert "session.end" not in out
    assert "2026-10-01T09:59:00Z" not in out
    assert "ESCALATE_TOOL_ERRORS" not in out


def test_it_does_not_describe_what_a_column_means(tmp_path, capsys):
    """Structure only: no interpretation, no unit, no guess."""
    path = build(tmp_path)

    code, out, _ = run(capsys, "ledger-info", "--ledger", str(path))

    assert code == 0
    lowered = out.lower()
    # Words that can never be part of this schema, so their absence still
    # means the command added nothing of its own.
    for word in ("probably", "likely", "seems", "appears to", "$"):
        assert word not in lowered
    # "usd" and "cost" used to be banned outright. They no longer can be:
    # `router_usage` has columns literally named `cost_usd` and
    # `baseline_cost_usd`, so banning the words would ban the schema itself.
    # What must not happen is the command using them as a unit or a label, so
    # any line carrying one has to be a bare `name  TYPE` column entry.
    for line in out.splitlines():
        if "usd" in line.lower() or "cost" in line.lower():
            assert re.fullmatch(r"    \w+ +[A-Z]+", line), line


def test_a_missing_file_is_reported_and_exits_non_zero(tmp_path, capsys):
    missing = tmp_path / "no-such-ledger.sqlite3"

    code, out, err = run(capsys, "ledger-info", "--ledger", str(missing))

    assert code != 0
    assert code == 1
    assert "no such database file" in err
    assert str(missing) in err
    assert out == ""
    assert not missing.exists()


def test_a_file_that_is_not_a_database_exits_non_zero(tmp_path, capsys):
    not_a_db = tmp_path / "notes.txt"
    not_a_db.write_text("this is plain text, not sqlite", encoding="utf-8")

    code, _, err = run(capsys, "ledger-info", "--ledger", str(not_a_db))

    assert code == 1
    assert "notes.txt" in err


def test_the_ledger_is_never_opened_writable(tmp_path, capsys):
    path = build(tmp_path)
    before = path.stat()
    calls = []
    connect = sqlite3.connect

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return connect(*args, **kwargs)

    original = sqlite3.connect
    try:
        sqlite3.connect = spy
        code, _, _ = run(capsys, "ledger-info", "--ledger", str(path))
    finally:
        sqlite3.connect = original

    assert code == 0
    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert calls, "the command opened no connection at all"
    for args, kwargs in calls:
        assert kwargs.get("uri") is True, f"opened without a URI: {args}"
        assert "mode=ro" in str(args[0]), f"opened writable: {args}"
    assert not list(tmp_path.glob("*-journal"))


def test_an_empty_database_lists_no_tables(tmp_path, capsys):
    path = tmp_path / "empty.sqlite3"
    connection = sqlite3.connect(path)
    connection.close()

    code, out, _ = run(capsys, "ledger-info", "--ledger", str(path))

    assert code == 0
    assert "tables: none" in out


def test_sqlite_internal_tables_are_not_listed(tmp_path, capsys):
    """`sqlite_sequence` is sqlite's bookkeeping, not the ledger's schema."""
    path = tmp_path / "internal.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
    connection.execute("INSERT INTO t (v) VALUES ('x')")
    connection.commit()
    connection.close()

    code, out, _ = run(capsys, "ledger-info", "--ledger", str(path))

    assert code == 0
    assert "sqlite_sequence" not in out
    assert "\nt\n" in out


def test_the_ledger_argument_is_required(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["ledger-info"])

    assert excinfo.value.code == 2