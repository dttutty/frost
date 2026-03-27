#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def parse_number(value: str) -> float:
    value = value.strip()
    if value == "":
        raise ValueError("empty numeric field")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the last CSV column equals the first column plus an offset."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file to inspect.")
    parser.add_argument(
        "--offset",
        type=float,
        default=1.0,
        help="Expected offset between the first and last column. Default: 1.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for float comparison. Default: 1e-9.",
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        print(f"file not found: {args.csv_path}", file=sys.stderr)
        return 2

    total_rows = 0
    mismatch_count = 0
    parse_error_count = 0
    first_mismatch: tuple[int, str, str] | None = None
    first_parse_error: tuple[int, str] | None = None

    with args.csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            print("empty CSV: no header row", file=sys.stderr)
            return 2
        if len(header) < 2:
            print("CSV must have at least two columns", file=sys.stderr)
            return 2

        for line_no, row in enumerate(reader, start=2):
            total_rows += 1
            if len(row) < 2:
                parse_error_count += 1
                if first_parse_error is None:
                    first_parse_error = (line_no, "row has fewer than two columns")
                continue

            try:
                first_val = parse_number(row[0])
                last_val = parse_number(row[-1])
            except ValueError as exc:
                parse_error_count += 1
                if first_parse_error is None:
                    first_parse_error = (line_no, str(exc))
                continue

            expected = first_val + args.offset
            if not math.isclose(last_val, expected, rel_tol=0.0, abs_tol=args.tolerance):
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = (line_no, row[0], row[-1])

    print(f"file: {args.csv_path}")
    print(f"first column: {header[0]}")
    print(f"last column: {header[-1]}")
    print(f"rows checked: {total_rows}")
    print(f"offset: {args.offset}")
    print(f"mismatches: {mismatch_count}")
    print(f"parse_errors: {parse_error_count}")

    if first_mismatch is not None:
        line_no, first_val, last_val = first_mismatch
        print(
            f"first mismatch at line {line_no}: first={first_val}, last={last_val}, "
            f"expected={float(first_val) + args.offset}"
        )

    if first_parse_error is not None:
        line_no, message = first_parse_error
        print(f"first parse error at line {line_no}: {message}")

    if mismatch_count == 0 and parse_error_count == 0:
        print("result: PASS")
        return 0

    print("result: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
