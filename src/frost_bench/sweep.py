from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from frost_bench.io import write_rows_tsv


FINAL_SUMMARY_PATTERN = re.compile(
    r"Best Val AP:\s*(?P<val_ap>[0-9eE+\-.]+)\s*@ Epoch\s*(?P<epoch>\d+)\s*"
    r"\|\s*Val AUC:\s*(?P<val_auc>[0-9eE+\-.]+)\s*"
    r"\|\s*Test AP:\s*(?P<test_ap>[0-9eE+\-.]+)\s*\|"
)


@dataclass(frozen=True)
class SweepSpec:
    name: str
    description: str
    models: tuple[str, ...]
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    epoch: int
    stable_mode: bool | None


@dataclass(frozen=True)
class SweepJob:
    index: int
    model: str
    dataset: str
    seed: int
    command: tuple[str, ...]
    log_path: Path


@dataclass(frozen=True)
class SeedMetric:
    seed: int
    epoch: int
    val_ap: float
    test_ap: float
    log_path: Path


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Sweep field {field!r} must be a list of strings")
    return tuple(value)


def _seed_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError("Sweep field 'seeds' must be a list of integers")
    return tuple(value)


def load_sweeps(path: Path) -> dict[str, SweepSpec]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    raw_sweeps = document.get("sweeps")
    if not isinstance(raw_sweeps, dict):
        raise ValueError(f"{path} must contain a 'sweeps' mapping")

    sweeps: dict[str, SweepSpec] = {}
    for name, raw in raw_sweeps.items():
        if not isinstance(raw, dict):
            raise TypeError(f"Sweep {name!r} must be a mapping")
        stable_mode = raw.get("stable_mode")
        if stable_mode is not None and not isinstance(stable_mode, bool):
            raise TypeError(f"Sweep {name!r} stable_mode must be a boolean or null")
        sweeps[str(name)] = SweepSpec(
            name=str(name),
            description=str(raw.get("description", "")),
            models=_string_tuple(raw.get("models"), field="models"),
            datasets=_string_tuple(raw.get("datasets"), field="datasets"),
            seeds=_seed_tuple(raw.get("seeds")),
            epoch=int(raw.get("epoch", 100)),
            stable_mode=stable_mode,
        )
    return sweeps


def parse_seed_metric(text: str, *, seed: int, log_path: Path) -> SeedMetric:
    final_matches = list(FINAL_SUMMARY_PATTERN.finditer(text))
    if not final_matches:
        raise ValueError(f"No final FROST metric summary found in {log_path}")
    final = final_matches[-1]
    epoch = int(final.group("epoch"))
    return SeedMetric(
        seed=seed,
        epoch=epoch,
        val_ap=float(final.group("val_ap")),
        test_ap=float(final.group("test_ap")),
        log_path=log_path,
    )


def summarize_seed_logs(
    log_paths: Iterable[tuple[int, Path]], *, output_dir: Path
) -> tuple[SeedMetric, ...]:
    metrics = tuple(
        parse_seed_metric(
            path.read_text(encoding="utf-8", errors="ignore"),
            seed=seed,
            log_path=path,
        )
        for seed, path in log_paths
        if path.is_file()
    )
    if not metrics:
        raise ValueError(f"No seed logs found in {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows_tsv(
        output_dir / "results.tsv",
        [
            {
                "seed": metric.seed,
                "epoch": metric.epoch,
                "val_ap": metric.val_ap,
                "test_ap": metric.test_ap,
                "log_path": metric.log_path,
            }
            for metric in metrics
        ],
    )
    values = [metric.test_ap for metric in metrics]
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    (output_dir / "summary.txt").write_text(
        f"seed_count={len(values)}\nmean_test_ap={mean:.6f}\nstd_test_ap={std:.6f}\n",
        encoding="utf-8",
    )
    return metrics
