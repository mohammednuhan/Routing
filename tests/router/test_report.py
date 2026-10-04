"""`tamias-router report`: the numbers, the filters, and the read-only promise.

Every test seeds a temporary database with rows written by the real decision
log, so the report is counting what the router would actually have written.
"""
from __future__ import annotations

import json
import re
import sqlite3

from router.cli import main
from router.decisions import DB_ENV_VAR, DecisionLog, DecisionRecord
from router.report import NOTHING_RECORDED, REPORT_NOTE

SESSION_A = "aaaa000000000001"
SESSION_B = "bbbb000000000002"
SESSION_C = "cccc000000000003"
SESSION_D = "dddd000000000004"

#: session, timestamp, requested, chosen, mode, action, applied, reasons, error.
#:
#: Ten rows chosen so every section has a number that could only be right if it
#: was counted correctly: a held row that is applied but is not a switch, a
#: shadow would-switch, two blocked stays, both overrides, an error, a row with
#: no model at all, and four sessions whose sizes make the median fractional
#: under `--last`.
SEED = (
    (
        SESSION_A,
        "2026-10-01T10:00:00Z",
        "test-low",
        "test-high",
        "active",
        "SWITCH",
        1,
        ["ESCALATE_TOOL_ERRORS"],
        None,
    ),
    (
        SESSION_A,
        "2026-10-01T10:00:10Z",
        "test-low",
        "test-high",
        "active",
        "SWITCH",
        1,
        ["HELD_MODEL"],
        None,
    ),
    (
        SESSION_A,
        "2026-10-01T10:00:20Z",
        "test-low",
        "test-low",
        "active",
        "STAY",
        0,
        ["NO_RULE_MATCHED"],
        None,
    ),
    (
        SESSION_B,
        "2026-10-02T11:00:00Z",
        "test-mid",
        "test-mid",
        "shadow",
        "SWITCH",
        0,
        ["ESCALATE_TOOL_ERRORS"],
        None,
    ),
    (
        SESSION_B,
        "2026-10-02T11:00:10Z",
        "test-mid",
        "test-mid",
        "shadow",
        "STAY",
        0,
        ["ESCALATE_TOOL_ERRORS", "BLOCKED_DWELL"],
        None,
    ),
    (
        SESSION_B,
        "2026-10-02T11:00:20Z",
        "test-mid",
        "test-mid",
        "shadow",
        "STAY",
        0,
        ["ESCALATE_TOOL_ERRORS", "BLOCKED_SWITCH_CAP"],
        None,
    ),
    (
        SESSION_C,
        "2026-10-03T12:00:00Z",
        "test-mid",
        "test-mid",
        "active",
        "STAY",
        0,
        ["KILL_SWITCH"],
        None,
    ),
    (
        SESSION_C,
        "2026-10-03T12:00:10Z",
        "test-mid",
        "test-mid",
        "active",
        "STAY",
        0,
        ["CIRCUIT_OPEN"],
        None,
    ),
    (
        SESSION_C,
        "2026-10-03T12:00:20Z",
        None,
        None,
        "active",
        "STAY",
        0,
        ["PASSTHROUGH"],
        "model_field_not_text",
    ),
    (
        SESSION_D,
        "2026-10-04T13:00:00Z",
        "test-high",
        "test-high",
        "shadow",
        "STAY",
        0,
        ["NO_RULE_MATCHED"],
        None,
    ),
)


def seed(tmp_path, monkeypatch, rows=SEED):
    """A database at `tmp_path/router.sqlite3` holding `rows`, and its path."""
    path = tmp_path / "router.sqlite3"
    monkeypatch.setenv(DB_ENV_VAR, str(path))
    log = DecisionLog(path)
    sessions: dict[str, int] = {}
    for session, timestamp, requested, chosen, mode, action, applied, reasons, error in rows:
        index = sessions.get(session, 0) + 1
        sessions[session] = index
        log.record(
            DecisionRecord(
                session_hint=session,
                requested_model=requested,
                chosen_model=chosen,
                mode=mode,
                reason_codes=list(reasons),
                action=action,
                applied=applied,
                error=error,
                request_index_in_session=index,
                timestamp=timestamp,
            )
        )
    return path


def run(capsys, *argv):
    """Run the CLI and return `(exit_code, stdout)`."""
    code = main(list(argv))
    return code, capsys.readouterr().out


def run_failing(capsys, *argv):
    """Run the CLI and return `(exit_code, stdout, stderr)`, for error paths."""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def payload(out):
    """The JSON object `report --json` printed."""
    return json.loads(out)


# --- every section, exactly -------------------------------------------------


def test_every_section_counts_the_seeded_rows_exactly(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)

    code, out = run(capsys, "report", "--json")

    assert code == 0
    assert payload(out) == {
        "note": REPORT_NOTE,
        "source": str(tmp_path / "router.sqlite3"),
        "read_only": True,
        "filters": {"since": None, "last": None, "session": None},
        "totals": {
            "requests": 10,
            "by_mode": {"active": 6, "shadow": 4},
            "first": "2026-10-01T10:00:00Z",
            "last": "2026-10-04T13:00:00Z",
        },
        "model_flow": [
            {"requested": "test-mid", "chosen": "test-mid", "count": 5},
            {"requested": "test-low", "chosen": "test-high", "count": 2},
            {"requested": "-", "chosen": "-", "count": 1},
            {"requested": "test-high", "chosen": "test-high", "count": 1},
            {"requested": "test-low", "chosen": "test-low", "count": 1},
        ],
        "actions": {
            "stay": 7,
            "switch_proposed": 3,
            "switch_applied": 2,
            "would_switch": 1,
        },
        "blocked": {"BLOCKED_DWELL": 1, "BLOCKED_SWITCH_CAP": 1},
        "held": {"HELD_MODEL": 1},
        "overrides": {"KILL_SWITCH": 1, "CIRCUIT_OPEN": 1},
        "errors": {"model_field_not_text": 1},
        "sessions": {
            "distinct": 4,
            "max_applied_switches": 1,
            "median_requests": 3,
        },
    }


def test_the_text_report_prints_every_section(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)

    code, out = run(capsys, "report")

    assert code == 0
    assert out.splitlines()[0] == REPORT_NOTE
    assert "(read-only)" in out
    assert "filters: none" in out
    assert "rows: 10" in out

    for heading in (
        "Totals",
        "Model flow (requested -> chosen)",
        "Actions",
        "Blocked switches",
        "Held",
        "Overrides",
        "Errors",
        "Sessions",
    ):
        assert f"\n{heading}\n" in out, f"missing the {heading} section"

    for line in (
        r"^  requests\s+10$",
        r"^  by mode\s+active 6, shadow 4$",
        r"^  time range\s+2026-10-01T10:00:00Z \.\. 2026-10-04T13:00:00Z$",
        r"^  test-mid -> test-mid\s+5$",
        r"^  test-low -> test-high\s+2$",
        r"^  - -> -\s+1$",
        r"^  STAY\s+7$",
        r"^  SWITCH proposed\s+3$",
        r"^  SWITCH applied\s+2$",
        r"^  would-switch\s+1$",
        r"^  BLOCKED_DWELL\s+1$",
        r"^  BLOCKED_SWITCH_CAP\s+1$",
        r"^  HELD_MODEL\s+1$",
        r"^  KILL_SWITCH\s+1$",
        r"^  CIRCUIT_OPEN\s+1$",
        r"^  model_field_not_text\s+1$",
        r"^  distinct\s+4$",
        r"^  max applied switches\s+1$",
        r"^  median requests\s+3$",
    ):
        assert re.search(line, out, re.MULTILINE), f"missing report line {line}"


def test_held_rows_are_not_counted_as_switches(tmp_path, monkeypatch, capsys):
    """Session A has two applied switches, one of them a held model."""
    seed(tmp_path, monkeypatch)

    data = payload(run(capsys, "report", "--json")[1])

    assert data["actions"]["switch_applied"] == 2, "both applied rows count as switches"
    assert data["held"] == {"HELD_MODEL": 1}
    assert (
        data["sessions"]["max_applied_switches"] == 1
    ), "a held row is the same model served again, not a new switch"


def test_no_dollar_figure_is_printed(tmp_path, monkeypatch, capsys):
    """The log has no token counts and no prices, so no money may appear."""
    seed(tmp_path, monkeypatch)

    _, text = run(capsys, "report")
    _, machine = run(capsys, "report", "--json")

    assert "$" not in text
    assert "$" not in machine
    assert "saving" not in text.lower()
    assert "usd" not in machine.lower()


# --- filters ----------------------------------------------------------------


def test_since_counts_only_rows_at_or_after_that_timestamp(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)

    code, out = run(capsys, "report", "--since", "2026-10-02T00:00:00Z", "--json")

    assert code == 0
    data = payload(out)
    assert data["filters"]["since"] == "2026-10-02T00:00:00Z"
    assert data["totals"]["requests"] == 7
    assert data["totals"]["first"] == "2026-10-02T11:00:00Z"
    assert data["totals"]["by_mode"] == {"active": 3, "shadow": 4}
    assert data["actions"] == {
        "stay": 6,
        "switch_proposed": 1,
        "switch_applied": 0,
        "would_switch": 1,
    }
    assert data["blocked"] == {"BLOCKED_DWELL": 1, "BLOCKED_SWITCH_CAP": 1}
    assert data["held"] == {"HELD_MODEL": 0}
    assert data["errors"] == {"model_field_not_text": 1}


def test_last_counts_the_newest_rows_only(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)

    code, out = run(capsys, "report", "--last", "3", "--json")

    assert code == 0
    data = payload(out)
    assert data["filters"]["last"] == 3
    assert data["totals"]["requests"] == 3
    assert data["totals"]["first"] == "2026-10-03T12:00:10Z"
    assert data["totals"]["last"] == "2026-10-04T13:00:00Z"
    assert data["actions"]["switch_proposed"] == 0
    # Two rows in one session and one in another: the median is 1.5.
    assert data["sessions"] == {
        "distinct": 2,
        "max_applied_switches": 0,
        "median_requests": 1.5,
    }
    assert data["overrides"] == {"KILL_SWITCH": 0, "CIRCUIT_OPEN": 1}


def test_session_filters_to_exactly_one_session(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)

    code, out = run(capsys, "report", "--session", SESSION_A, "--json")

    assert code == 0
    data = payload(out)
    assert data["filters"]["session"] == SESSION_A
    assert data["totals"]["requests"] == 3
    assert data["model_flow"] == [
        {"requested": "test-low", "chosen": "test-high", "count": 2},
        {"requested": "test-low", "chosen": "test-low", "count": 1},
    ]
    assert data["actions"]["switch_applied"] == 2
    assert data["sessions"]["distinct"] == 1


def test_a_session_filter_matching_nothing_says_so(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)

    code, out = run(capsys, "report", "--session", "nosuchsession")

    assert code == 0
    assert NOTHING_RECORDED in out


def test_the_filters_narrow_together(tmp_path, monkeypatch, capsys):
    """Session A's newest two rows: the held switch and the stay after it."""
    seed(tmp_path, monkeypatch)

    code, out = run(
        capsys, "report", "--session", SESSION_A, "--last", "2", "--json"
    )

    assert code == 0
    data = payload(out)
    assert data["totals"]["requests"] == 2
    assert data["actions"]["switch_applied"] == 1
    assert data["actions"]["stay"] == 1
    assert data["held"] == {"HELD_MODEL": 1}
    assert data["sessions"]["max_applied_switches"] == 0


def test_a_bad_since_is_refused_rather_than_guessed(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)

    code, _, err = run_failing(capsys, "report", "--since", "last tuesday")

    assert code == 2
    assert "2026-10-01T00:00:00Z" in err


def test_last_must_be_one_or_more(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)

    code, _, err = run_failing(capsys, "report", "--last", "0")

    assert code == 2
    assert "--last must be 1 or more" in err


# --- nothing to report ------------------------------------------------------


def test_an_empty_database_says_nothing_is_recorded(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch, rows=())

    code, out = run(capsys, "report")

    assert code == 0
    assert NOTHING_RECORDED in out


def test_a_missing_database_is_not_created(tmp_path, monkeypatch, capsys):
    """A report is a reader. It must not bring a database into existence."""
    missing = tmp_path / "never" / "made" / "router.sqlite3"
    monkeypatch.setenv(DB_ENV_VAR, str(missing))

    code, out = run(capsys, "report")

    assert code == 0
    assert NOTHING_RECORDED in out
    assert not missing.exists()
    assert not missing.parent.exists()


def test_a_database_with_no_router_table_says_nothing_is_recorded(tmp_path, monkeypatch, capsys):
    other = tmp_path / "other.sqlite3"
    connection = sqlite3.connect(other)
    connection.execute("CREATE TABLE something_else (id INTEGER)")
    connection.commit()
    connection.close()
    monkeypatch.setenv(DB_ENV_VAR, str(other))

    code, out = run(capsys, "report")

    assert code == 0
    assert NOTHING_RECORDED in out


# --- read-only --------------------------------------------------------------


def test_the_database_is_never_opened_writable(tmp_path, monkeypatch, capsys):
    """Size and mtime must not move, and every connection must be `mode=ro`."""
    path = seed(tmp_path, monkeypatch)
    before = path.stat()
    calls = []
    connect = sqlite3.connect

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return connect(*args, **kwargs)

    assert main(["report"]) == 0
    capsys.readouterr()

    monkeypatch.setattr(sqlite3, "connect", spy)
    assert main(["report"]) == 0
    capsys.readouterr()
    monkeypatch.undo()

    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert calls, "the report opened no connection at all"
    for args, kwargs in calls:
        assert kwargs.get("uri") is True, f"opened without a URI: {args}"
        assert "mode=ro" in str(args[0]), f"opened writable: {args}"
    assert DecisionLog(path).count() == len(SEED)


def test_the_report_needs_no_config(tmp_path, monkeypatch, capsys):
    """`report` counts rows; it never loads config.yaml to do it."""
    seed(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    code, out = run(capsys, "report")

    assert code == 0
    assert "requests" in out