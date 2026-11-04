"""CLI entrypoint: validate | run | report | smoke."""

from __future__ import annotations

import argparse
import sys

from proactive_bench import __version__

_COMMANDS = ("validate", "run", "report", "smoke")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proactive-bench",
        description="Mini benchmark for proactive engineering agents (HALO-lite).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command")
    for name in _COMMANDS:
        sub.add_parser(name, help=f"{name} (implemented in later phases)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    print(
        f"command {args.command!r} is not implemented yet "
        f"(see SPEC.md phases; currently Phase 1).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
