"""tamias-router command line: `status` and `start`.

`status` reads the config and prints it. It makes no network calls. `start`
does not run a proxy yet and does not read the config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, RouterConfig, load_config_or_exit

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
    sub.add_parser("start", help="start the proxy (not implemented yet)")
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

    if args.cmd == "start":
        print("proxy not implemented yet")
        return 0

    config = load_config_or_exit(args.config)
    print(format_status(config))
    return 0


if __name__ == "__main__":
    sys.exit(main())