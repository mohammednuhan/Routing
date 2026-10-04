"""tamias-router command line: `status`, `start` and `log`.

`status` reads the config and prints it. It makes no network calls. `start`
validates the config and runs the transparent loopback proxy; it makes no
routing decision and changes no model. `log` prints decision-log rows, which
never contain request or response content.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, RouterConfig, load_config_or_exit
from .decisions import DecisionLog, DecisionRow
from .proxy import ProxyError, ProxySettings, parse_upstream, serve
from .safety import is_blocked
from .state import SessionStore

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
    return parser


def format_status(config: RouterConfig) -> str:
    """Render the status report. Reports configuration only, never traffic."""
    fields = (
        ("mode", config.mode),
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


def format_rows(rows: list[DecisionRow]) -> str:
    """Render rows as a fixed-width table of metadata fields only."""

    def cell(row: DecisionRow, field: str) -> str:
        value = getattr(row, field)
        if value is None:
            return "-"
        if isinstance(value, list):
            return ",".join(str(item) for item in value) or "-"
        if field == "action":
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

    if args.cmd == "log":
        return _log(args.last, show_signals=getattr(args, "signals", False))

    config = load_config_or_exit(args.config)

    if args.cmd == "status":
        print(format_status(config))
        return 0

    return _start(config, args.upstream)


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
