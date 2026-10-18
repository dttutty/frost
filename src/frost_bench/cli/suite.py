from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from frost_bench.paths import REPO_ROOT
from frost_bench.suite import load_suites, parse_overrides


DEFAULT_SPEC = REPO_ROOT / "benchmarks" / "specs" / "throughput_suites.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a benchmark suite declared in YAML."
    )
    parser.add_argument("suite", nargs="?")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    suites = load_suites(args.spec)
    if args.list:
        for suite in suites.values():
            print(f"{suite.name}: {suite.description}")
        return 0
    if args.suite not in suites:
        parser.error(f"suite must be one of: {', '.join(sorted(suites))}")

    suite = suites[args.suite]
    command = suite.command(
        overrides=parse_overrides(args.set), extra_args=extra_args
    )
    if args.dry_run:
        print(suite.shell_command(overrides=parse_overrides(args.set), extra_args=extra_args))
        return 0
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
