"""Update the README badge from a real coverage.py JSON report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    totals = json.loads(args.report.read_text())["totals"]
    total = int(totals["num_statements"])
    covered = int(totals["covered_lines"])
    if not 0 <= covered <= total or total == 0:
        raise ValueError("Coverage report must contain a nonempty Python source measurement")
    percent = 100 * covered / total
    color = "brightgreen" if percent >= 80 else "yellowgreen" if percent >= 60 else "orange"
    repository = os.environ.get("GITHUB_REPOSITORY", "dttutty/frost")
    badge = (
        f"[![CPU coverage](https://img.shields.io/badge/CPU%20coverage-"
        f"{percent:.1f}%25-{color})]"
        f"(https://github.com/{repository}/actions/workflows/coverage.yml)"
    )
    readme = Path(__file__).resolve().parents[2] / "README.md"
    updated, count = re.subn(
        r"^\[!\[CPU coverage\].*$", badge, readme.read_text(), flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError(f"Expected one CPU coverage badge in README, found {count}")
    readme.write_text(updated)
    print(f"CPU Python line coverage: {covered}/{total} = {percent:.1f}%")


if __name__ == "__main__":
    main()
