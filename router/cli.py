"""tamias-router command line: `status` and `start`.

`status` reads the config and prints it. It makes no network calls. `start`
validates the config and runs the transparent loopback proxy; it makes no
routing decision and changes no model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, RouterConfig, load_config_or_exit
from .proxy import ProxyError, ProxySettings, parse_upstream, serve

PROG = "tamias-router"


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    config = load_config_or_exit(args.config)

    if args.cmd == "status":
        print(format_status(config))
        return 0

    return _start(config, args.upstream)


def _start(config: RouterConfig, upstream_override: str | None) -> int:
    upstream = upstream_override if upstream_override else config.upstream
    try:
        target = parse_upstream(upstream)
        settings = ProxySettings(
            listen_host=config.listen_host,
            listen_port=config.listen_port,
            upstream=upstream,
        )
    except ProxyError as exc:
        print(f"{PROG}: invalid upstream: {exc}", file=sys.stderr)
        return 1

    print(f"{PROG} start: mode={config.mode} listen={config.listen}")
    try:
        serve(settings)
    except OSError as exc:
        print(f"{PROG}: cannot listen on {config.listen}: {exc}", file=sys.stderr)
        return 1
    print(f"{PROG}: stopped (upstream was {target.safe_label})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
