"""CLI entrypoint: validate | run | report | smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from proactive_bench import __version__
from proactive_bench.scenarios import validate_directory

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
    validate = sub.add_parser("validate", help="validate scenario JSON files")
    validate.add_argument(
        "--dir",
        type=Path,
        default=Path("scenarios"),
        help="scenarios root directory (default: scenarios)",
    )
    for name in _COMMANDS:
        if name != "validate":
            sub.add_parser(name, help=f"{name} (implemented in later phases)")
    return parser


def _run_validate(scenarios_dir: Path) -> int:
    report = validate_directory(scenarios_dir)
    for scenario_file in report.files:
        if scenario_file.errors:
            print(f"FAIL {scenario_file.path.name}")
            for error in scenario_file.errors:
                print(f"  {error}")
        else:
            print(f"OK {scenario_file.path.name}")
    for error in report.global_errors:
        print(f"FAIL {error}")
    error_count = sum(len(f.errors) for f in report.files) + len(report.global_errors)
    print(f"{report.scenario_count} scenarios, {error_count} error(s)")
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "validate":
        return _run_validate(args.dir)
    print(
        f"command {args.command!r} is not implemented yet "
        f"(see SPEC.md phases; currently Phase 3).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
