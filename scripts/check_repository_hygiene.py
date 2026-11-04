"""Validate repository contents for generated outputs and recorded result values."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


OUTPUT_DIRECTORIES = {
    "artifacts",
    "logs",
    "logs_old",
    "outputs",
    "previous_log",
    "results",
}
OUTPUT_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".log",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".svg",
    ".tsv",
}
OUTPUT_FILENAMES = {
    "progress.md",
    "report.md",
    "result.json",
    "results.json",
    "status.json",
    "summary.json",
}
# Published reference papers are source material, not generated experiment output.
REFERENCE_PAPERS = {
    Path("third_party_papers/DyREP.pdf"),
    Path("third_party_papers/PipeTGL.pdf"),
    Path(
        "third_party_papers/"
        "Redundancy-Free_and_Load-Balanced_TGNN_Training_With_Hierarchical_Pipeline_Parallelism (4).pdf"
    ),
}
TEXT_SUFFIXES = {
    ".cpp",
    ".h",
    ".hpp",
    ".md",
    ".py",
    ".pyi",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
RECORDED_VALUE_PATTERNS = {
    "evaluation metric": re.compile(
        r"\b(?:best|final|validation|val|test)[ _-]?(?:ap|auc|mrr)"
        r"\s*(?::|=|is)\s*[\"']?(?!0(?:\.0+)?\b)\d+(?:\.\d+)?",
        re.IGNORECASE,
    ),
    "throughput": re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:events?|edges?|samples?)/(?:s|sec|second)\b",
        re.IGNORECASE,
    ),
    "speedup": re.compile(
        r"(?:\b\d+(?:\.\d+)?x\s+(?:speedup|faster|slower)\b|"
        r"\b(?:speedup|faster|slower)\s*(?::|=|is)?\s*\d+(?:\.\d+)?x\b)",
        re.IGNORECASE,
    ),
}


def tracked_paths() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        Path(raw.decode())
        for raw in completed.stdout.split(b"\0")
        if raw
    ]


def generated_output_reason(path: Path) -> str | None:
    if path in REFERENCE_PAPERS:
        return None
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & OUTPUT_DIRECTORIES:
        return "generated-output directory"
    if path.suffix.lower() in OUTPUT_SUFFIXES:
        return "generated-output file type"
    if path.name.lower() in OUTPUT_FILENAMES:
        return "generated-output filename"
    return None


def recorded_value_reasons(path: Path) -> list[str]:
    if path.parts[0] == "tests" or path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        label
        for label, pattern in RECORDED_VALUE_PATTERNS.items()
        if pattern.search(text)
    ]


def main() -> None:
    problems: list[str] = []
    for path in tracked_paths():
        output_reason = generated_output_reason(path)
        if output_reason is not None:
            problems.append(f"{path}: {output_reason}")
        for value_reason in recorded_value_reasons(path):
            problems.append(f"{path}: recorded {value_reason} value")

    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise SystemExit(
            "Repository hygiene check failed:\n"
            f"{details}\n"
            "Keep reproducibility parameters and synthetic test fixtures, but "
            "remove generated outputs and recorded result values."
        )

    print("Repository hygiene check passed.")


if __name__ == "__main__":
    main()
