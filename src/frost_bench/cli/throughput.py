from __future__ import annotations

import argparse
from pathlib import Path

from frost_bench.suite import load_suites, parse_overrides
from frost_bench.throughput import run_throughput_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a maintained throughput matrix.")
    parser.add_argument("suite")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    suites = load_suites(args.spec)
    if args.suite not in suites:
        parser.error(f"unknown suite {args.suite!r}")
    suite = suites[args.suite]
    if suite.profile is None:
        parser.error(f"suite {suite.name!r} has no throughput profile")
    arguments = {**suite.arguments, **parse_overrides(args.set)}
    run_dir = run_throughput_suite(
        suite_name=suite.name,
        profile=suite.profile,
        arguments=arguments,
    )
    print(f"Benchmark artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
