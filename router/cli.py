"""tamias-router command line: `status`, `start`, `log`, `report`,
`ledger-info` and `switch-report`.

`status` reads the config and prints it. It makes no network calls. `start`
validates the config and runs the transparent loopback proxy; it makes no
routing decision and changes no model. `log` prints decision-log rows, which
never contain request or response content. `report` counts those same rows and
`ledger-info` prints a SQLite file's schema; both open their database read-only
and neither can write a row. `switch-report` correlates approved switches with
the rows the Tamias Observer ledger recorded around the same time, by time
alone, and says so.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from .breaker import REASON_CIRCUIT_OPEN
from .config import DEFAULT_CONFIG_PATH, RouterConfig, load_config_or_exit
from .decisions import DecisionLog, DecisionRow, default_db_path
from .killswitch import (
    KILL_SWITCH_FILENAME,
    REASON_KILL_SWITCH,
    KillSwitch,
    read_kill_switch,
    write_kill_switch,
)
from .ledger import format_ledger, inspect_ledger
from .proxy import ProxyError, ProxySettings, parse_upstream, serve
from .readonly import ReadOnlyError
from .report import (
    NOTHING_RECORDED,
    ReportFilters,
    as_json,
    build_report,
    format_report,
    parse_since,
)
from .safety import is_blocked
from .state import SessionStore
from .switch_report import (
    DEFAULT_LAST as DEFAULT_SWITCHES,
    DEFAULT_LEDGER_PATH,
    DEFAULT_SKEW_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    LedgerProblem,
    MissingRequestRecordTable,
    NoRequestRecords,
    SwitchReportOptions,
    build_switch_report,
    format_switch_report,
    parse_timestamp,
    switch_report_as_json,
)

PROG = "tamias-router"

#: Model ids in the shipped config are placeholders. Active mode refuses to run
#: on one: it would rewrite requests toward a model nobody verified (Rule 7).
PLACEHOLDER_PREFIX = "TODO_"

#: Printed once at startup when active mode is allowed to run.
ACTIVE_BANNER = "ACTIVE MODE: requests may be rewritten to other models"

LOG_COLUMNS = (
    ("id", "decision_id"),
    ("timestamp", "timestamp"),
    ("session", "session_hint"),
    ("idx", "request_index_in_session"),
    ("requested", "requested_model"),
    ("chosen", "chosen_model"),
    ("effort", "chosen_effort"),
    ("mode", "mode"),
    ("action", "action"),
    ("applied", "applied"),
    ("reasons", "reason_codes"),
    ("error", "error"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Tamias router: local, metadata-only model/effort routing",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"path to config.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("status", help="print mode, listen address, upstream, defaults (no network)")
    start = sub.add_parser("start", help="run the transparent proxy")
    start.add_argument(
        "--upstream",
        default=None,
        help="override the upstream URL from the config (for testing against a mock)",
    )
    log = sub.add_parser("log", help="print the latest decision-log rows (metadata only)")
    log.add_argument("--last", type=int, default=10, help="how many rows to show (default 10)")
    log.add_argument(
        "--signals",
        action="store_true",
        help="print signal_values JSON of each row below the table",
    )
    sub.add_parser(
        "off",
        help="turn the kill switch on: create the router.off flag file beside the log",
    )
    sub.add_parser(
        "on", help="turn the kill switch off: remove the router.off flag file"
    )

    report = sub.add_parser(
        "report",
        help="summarise the decision log (read-only, metadata only, no cost)",
    )
    report.add_argument(
        "--since",
        default=None,
        metavar="ISO_TIMESTAMP",
        help="only rows at or after this UTC timestamp, e.g. 2026-10-01T00:00:00Z",
    )
    report.add_argument(
        "--last", type=int, default=None, metavar="N", help="only the newest N rows"
    )
    report.add_argument(
        "--session",
        default=None,
        metavar="HINT",
        help="only rows whose session_hint is exactly HINT",
    )
    report.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="print one JSON object instead of the sections",
    )

    ledger = sub.add_parser(
        "ledger-info",
        help="print a SQLite file's tables, columns and row counts (read-only)",
    )
    ledger.add_argument(
        "--ledger",
        required=True,
        type=Path,
        metavar="PATH",
        help="path to the SQLite file to inspect",
    )

    switch = sub.add_parser(
        "switch-report",
        help=(
            "correlate approved switches with the ledger rows recorded around the "
            "same time (read-only, by time only)"
        ),
    )
    switch.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        metavar="PATH",
        help=f"Observer ledger to read (default: {DEFAULT_LEDGER_PATH})",
    )
    switch.add_argument(
        "--since",
        default=None,
        metavar="ISO_TIMESTAMP",
        help="only switches at or after this UTC timestamp, e.g. 2026-10-01T00:00:00Z",
    )
    switch.add_argument(
        "--last",
        type=int,
        default=DEFAULT_SWITCHES,
        metavar="N",
        help=f"how many approved switches to show, newest first (default {DEFAULT_SWITCHES})",
    )
    switch.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        metavar="SECONDS",
        help=f"how far after a switch to look (default {DEFAULT_WINDOW_SECONDS})",
    )
    switch.add_argument(
        "--skew-seconds",
        type=int,
        default=DEFAULT_SKEW_SECONDS,
        metavar="SECONDS",
        help=f"how far before a switch to look (default {DEFAULT_SKEW_SECONDS})",
    )
    switch.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="print one JSON object instead of the text report",
    )
    return parser


def format_status(config: RouterConfig, switch: KillSwitch | None = None) -> str:
    """Render the status report. Reports configuration only, never traffic."""
    fields = (
        ("mode", config.mode),
        ("kill_switch", format_kill_switch(switch)),
        ("listen", config.listen),
        ("upstream", config.upstream),
        ("default_model", config.default_model),
        ("default_effort", config.default_effort),
        ("legal_models", str(len(config.models))),
    )
    width = max(len(name) for name, _ in fields)
    lines = [f"{PROG} status"]
    lines.extend(f"{name.ljust(width)} : {value}" for name, value in fields)
    return "\n".join(lines)


def format_kill_switch(switch: KillSwitch | None) -> str:
    """`OFF`, or `ON` with the reason, so status never hides why."""
    if switch is None or not switch.on:
        return "OFF"
    return f"ON ({switch.detail})"


def format_rows(rows: list[DecisionRow]) -> str:
    """Render rows as a fixed-width table of metadata fields only."""

    def cell(row: DecisionRow, field: str) -> str:
        value = getattr(row, field)
        if value is None:
            return "-"
        if isinstance(value, list):
            return ",".join(str(item) for item in value) or "-"
        if field == "action":
            # A kill-switch or open-circuit row is a STAY the router never
            # chose, so label it as the reason it happened rather than as a
            # decision. The reason column carries the code itself.
            if REASON_KILL_SWITCH in row.reason_codes:
                return "kill switch"
            if REASON_CIRCUIT_OPEN in row.reason_codes:
                return "circuit open"
            if is_blocked(row.reason_codes):
                return "blocked"
            if value == "SWITCH" and getattr(row, "applied", 0) == 0:
                return "would SWITCH"
        return str(value)

    header = [label for label, _ in LOG_COLUMNS]
    body = [[cell(row, field) for _, field in LOG_COLUMNS] for row in rows]
    widths = [
        max([len(header[index]), *(len(row[index]) for row in body)])
        for index in range(len(header))
    ]

    def render(cells: list[str]) -> str:
        return "  ".join(
            cells[index].ljust(widths[index]) for index in range(len(cells))
        ).rstrip()

    return "\n".join([render(header), *(render(row) for row in body)])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd in ("off", "on"):
        return _kill_switch(args.cmd == "off")

    if args.cmd == "log":
        return _log(args.last, show_signals=getattr(args, "signals", False))

    if args.cmd == "report":
        return _report(args)

    if args.cmd == "ledger-info":
        return _ledger_info(args.ledger)

    if args.cmd == "switch-report":
        return _switch_report(args)

    config = load_config_or_exit(args.config)

    if args.cmd == "status":
        print(format_status(config, read_kill_switch(DecisionLog.from_env().db_path)))
        return 0

    return _start(config, args.upstream)


def _kill_switch(on: bool) -> int:
    """Create or remove the flag file. The running proxy sees it on the next
    request, so no restart is involved."""
    log = DecisionLog.from_env()
    path = log.db_path.parent / KILL_SWITCH_FILENAME
    try:
        write_kill_switch(log.db_path, on)
    except OSError as exc:
        action = "create" if on else "remove"
        print(f"{PROG}: cannot {action} {path}: {exc}", file=sys.stderr)
        return 1
    if on:
        print(f"{PROG}: kill switch ON ({path})")
    else:
        print(f"{PROG}: kill switch OFF ({path} removed if it existed)")
    return 0


def _log(last: int, show_signals: bool = False) -> int:
    if last < 1:
        print(f"{PROG}: --last must be 1 or more, got {last}", file=sys.stderr)
        return 2

    decisions = DecisionLog.from_env()
    if not decisions.db_path.exists():
        print(f"no decision log yet at {decisions.db_path}")
        return 0

    try:
        rows = decisions.recent(last)
    except (OSError, sqlite3.Error) as exc:
        print(f"{PROG}: cannot read {decisions.db_path}: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print(f"no decisions recorded yet in {decisions.db_path}")
        return 0

    print(format_rows(rows))
    if show_signals:
        import json

        for row in rows:
            print(f"\n{id if False else ''}signals for decision_id={row.decision_id}:")
            print(json.dumps(row.signal_values, indent=2, sort_keys=True))
    return 0


def _report(args: argparse.Namespace) -> int:
    """Count the decision log. Read-only, and it never creates the database."""
    if args.last is not None and args.last < 1:
        print(f"{PROG}: --last must be 1 or more, got {args.last}", file=sys.stderr)
        return 2

    since = None
    if args.since is not None:
        try:
            since = parse_since(args.since)
        except ValueError as exc:
            print(f"{PROG}: --since {exc}", file=sys.stderr)
            return 2

    filters = ReportFilters(since=since, last=args.last, session=args.session)
    path = default_db_path()
    if not path.is_file():
        print(f"{NOTHING_RECORDED} (no database at {path})")
        return 0

    try:
        report = build_report(path, filters)
    except ReadOnlyError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1

    if report.requests == 0:
        print(f"{NOTHING_RECORDED} (nothing matching in {path})")
        return 0

    print(as_json(report) if args.as_json else format_report(report))
    return 0


def _ledger_info(path: Path) -> int:
    """Print a SQLite file's schema. Read-only, and no row is ever selected."""
    try:
        tables = inspect_ledger(path)
    except ReadOnlyError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    print(format_ledger(path, tables))
    return 0


def _switch_report(args: argparse.Namespace) -> int:
    """Correlate approved switches with the ledger, by time alone.

    An absent or empty ledger is not a failure: there is simply nothing to
    correlate with, and the message says what to do about it. A ledger that is
    not a request-record store at all is a failure.
    """
    for flag, value, least in (
        ("--last", args.last, 1),
        ("--window-seconds", args.window_seconds, 0),
        ("--skew-seconds", args.skew_seconds, 0),
    ):
        if value < least:
            print(f"{PROG}: {flag} must be {least} or more, got {value}", file=sys.stderr)
            return 2

    since = None
    if args.since is not None:
        since = parse_timestamp(args.since)
        if since is None:
            print(
                f"{PROG}: --since expected a UTC ISO-8601 timestamp like "
                f"2026-10-01T00:00:00Z, got {args.since!r}",
                file=sys.stderr,
            )
            return 2

    router_database = default_db_path()
    if not router_database.is_file():
        print(f"{NOTHING_RECORDED} (no database at {router_database})")
        return 0

    options = SwitchReportOptions(
        ledger=args.ledger,
        window_seconds=args.window_seconds,
        skew_seconds=args.skew_seconds,
        last=args.last,
        since=since,
    )
    try:
        report = build_switch_report(router_database, options)
    except NoRequestRecords as exc:
        print(str(exc))
        return 0
    except MissingRequestRecordTable as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    except (LedgerProblem, ReadOnlyError) as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1

    print(switch_report_as_json(report) if args.as_json else format_switch_report(report))
    return 0


def active_start_blocker(config: RouterConfig) -> str | None:
    """Why `start` must refuse, or None when it may proceed.

    Active mode is the only mode that can change a request, so it is the only
    mode that is refused when the config still carries placeholder model ids.
    Shadow and off forward every request untouched, so a placeholder there is
    harmless and must stay loadable.
    """
    if config.mode != "active":
        return None
    placeholders = [
        spec.id for spec in config.models if spec.id.startswith(PLACEHOLDER_PREFIX)
    ]
    if not placeholders:
        return None
    return (
        f"refusing to start in active mode: {len(placeholders)} model id(s) are "
        f"placeholders beginning {PLACEHOLDER_PREFIX!r}: {', '.join(placeholders)}. "
        "Replace each with a verified id plus source and access date, or run in shadow."
    )


def _start(config: RouterConfig, upstream_override: str | None) -> int:
    blocker = active_start_blocker(config)
    if blocker is not None:
        print(f"{PROG}: {blocker}", file=sys.stderr)
        return 1

    upstream = upstream_override if upstream_override else config.upstream
    try:
        target = parse_upstream(upstream)
        decisions = DecisionLog.from_env()
        settings = ProxySettings(
            listen_host=config.listen_host,
            listen_port=config.listen_port,
            upstream=upstream,
            mode=config.mode,
            decisions=decisions,
            config=config,
            state=SessionStore(),
        )
    except ProxyError as exc:
        print(f"{PROG}: invalid upstream: {exc}", file=sys.stderr)
        return 1

    print(f"{PROG} start: mode={config.mode} listen={config.listen}")
    if config.mode == "active":
        print(ACTIVE_BANNER)
    print(f"{PROG} decisions: {decisions.db_path}")
    try:
        serve(settings)
    except OSError as exc:
        print(f"{PROG}: cannot listen on {config.listen}: {exc}", file=sys.stderr)
        return 1
    print(f"{PROG}: stopped (upstream was {target.safe_label})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
