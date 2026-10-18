from __future__ import annotations

import json
import re
from json import JSONDecoder


ThroughputByRank = dict[int, dict[str, float | int]]


def parse_all_json_after_prefix(
    text: str, prefix: str, *, required: bool = True
) -> list[dict]:
    rows: list[dict] = []
    decoder = JSONDecoder()
    for match in re.finditer(re.escape(prefix) + r"\s+", text):
        try:
            obj, _ = decoder.raw_decode(text[match.end() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    if required and not rows:
        raise RuntimeError(f"Could not find `{prefix}` JSON payloads in log.")
    return rows


def parse_rank_payloads(text: str, prefix: str) -> dict[int, dict]:
    """Index JSON benchmark payloads by their integer rank."""
    return {
        int(row["rank"]): row for row in parse_all_json_after_prefix(text, prefix)
    }


def summarize_distribution(values: list[float]) -> dict[str, float]:
    """Return the common nearest-rank summary used by benchmark reports."""
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    ordered = sorted(float(value) for value in values)
    last = len(ordered) - 1
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[int(0.50 * last)],
        "p90": ordered[int(0.90 * last)],
        "max": ordered[-1],
    }


def normalized_progress(values: list[object]) -> list[float]:
    """Map a sequence to an inclusive 0..100 progress axis."""
    if len(values) <= 1:
        return [0.0] * len(values)
    return [index / (len(values) - 1) * 100.0 for index in range(len(values))]


def sum_memory_curves_by_rank(
    rows_by_rank: dict[int, dict],
) -> tuple[list[float], list[float]]:
    """Sum aligned reserved/allocated memory curves across ranks."""
    if not rows_by_rank:
        raise RuntimeError("No rank-local memory curves found.")
    reserved = [
        [float(value) for value in row["reserved_mib_curve"]]
        for row in rows_by_rank.values()
    ]
    allocated = [
        [float(value) for value in row["allocated_mib_curve"]]
        for row in rows_by_rank.values()
    ]
    length = min(*(len(curve) for curve in reserved), *(len(curve) for curve in allocated))
    if length <= 0:
        raise RuntimeError("Empty memory curve.")
    return (
        [sum(curve[index] for curve in reserved) for index in range(length)],
        [sum(curve[index] for curve in allocated) for index in range(length)],
    )


def parse_frost_rank_throughput(text: str) -> ThroughputByRank:
    matches = re.findall(
        r"TRAIN_THROUGHPUT_PER_GPU\s+rank=(\d+)\s+"
        r"events_per_sec_per_gpu=([0-9eE+\-.]+).*?measured_batches=([0-9]+)",
        text,
    )
    if not matches:
        raise RuntimeError("Could not parse FROST throughput.")
    return {
        int(rank): {
            "events_per_sec_per_gpu": float(events_per_second),
            "measured_batches": int(measured_batches),
        }
        for rank, events_per_second, measured_batches in matches
    }


def parse_tgl_rank_throughput(
    text: str, *, n_gpu: int | None = None
) -> ThroughputByRank:
    by_rank: ThroughputByRank = {}
    for row in parse_all_json_after_prefix(text, "APPENDIX_TRAIN_E2E_SUMMARY"):
        if "events_per_sec_per_gpu" not in row:
            continue
        rank = int(row["rank"])
        if n_gpu is None or rank < n_gpu:
            by_rank[rank] = {
                "events_per_sec_per_gpu": float(row["events_per_sec_per_gpu"]),
                "measured_batches": int(row["interval_count"]),
            }
    if not by_rank:
        raise RuntimeError("Could not parse TGL throughput.")
    return by_rank


def parse_dyglib_summary(text: str) -> dict[str, float | int]:
    rows = parse_all_json_after_prefix(text, "APPENDIX_TRAIN_E2E_SUMMARY")
    if len(rows) != 1:
        raise RuntimeError(f"Expected one DyGLib summary row, got {len(rows)}")
    return {
        "events_per_sec_per_gpu": float(rows[0]["events_per_sec_per_gpu"]),
        "measured_batches": int(rows[0]["measured_batches"]),
    }


def parse_pipetgl_rank_throughput(text: str) -> ThroughputByRank:
    by_rank: ThroughputByRank = {}
    for row in parse_all_json_after_prefix(text, "APPENDIX_TRAIN_E2E_SUMMARY"):
        if row.get("system") != "pipetgl" or "events_per_sec_per_gpu" not in row:
            continue
        by_rank[int(row["rank"])] = {
            "events_per_sec_per_gpu": float(row["events_per_sec_per_gpu"]),
            "measured_batches": int(row["measured_batches"]),
        }
    if not by_rank:
        raise RuntimeError("Could not parse PipeTGL throughput.")
    return by_rank


def parse_pipetgl_epoch_summaries(text: str) -> list[dict[str, float | int | str]]:
    rows = parse_all_json_after_prefix(text, "PIPETGL_EPOCH_SUMMARY")
    parsed: list[dict[str, float | int | str]] = []
    for row in rows:
        item: dict[str, float | int | str] = {
            "system": "pipetgl",
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "events": int(row["events"]),
            "cumulative_events": int(row["cumulative_events"]),
            "train_time_s": float(row["train_time_s"]),
            "cumulative_train_time_s": float(row["cumulative_train_time_s"]),
        }
        if "epoch_wall_time_s" in row:
            item["epoch_wall_time_s"] = float(row["epoch_wall_time_s"])
        if "val_mrr" in row:
            item.update(evaluation_metric="mrr", val_mrr=float(row["val_mrr"]))
            if "test_mrr" in row:
                item["test_mrr"] = float(row["test_mrr"])
        else:
            item.update(
                evaluation_metric="ap",
                val_ap=float(row["val_ap"]),
                val_auc=float(row["val_auc"]),
            )
            if "test_ap" in row:
                item["test_ap"] = float(row["test_ap"])
            if "test_auc" in row:
                item["test_auc"] = float(row["test_auc"])
        parsed.append(item)
    return parsed


def parse_tgl_epoch_summaries(text: str) -> list[dict[str, float | int | str]]:
    rows = parse_all_json_after_prefix(text, "TGL_EPOCH_SUMMARY")
    parsed: list[dict[str, float | int | str]] = []
    for row in rows:
        metric = str(row["evaluation_metric"])
        item: dict[str, float | int | str] = {
            "system": "tgl",
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "evaluation_metric": metric,
            "epoch_wall_time_s": float(row["epoch_wall_time_s"]),
        }
        if metric == "mrr":
            item["val_mrr"] = float(row["val_mrr"])
            item["test_mrr"] = float(row["test_mrr"])
        elif metric == "ap":
            item["val_ap"] = float(row["val_ap"])
            item["val_auc"] = float(row["val_auc"])
            item["test_ap"] = float(row["test_ap"])
            item["test_auc"] = float(row["test_auc"])
        else:
            raise ValueError(f"Unsupported TGL evaluation metric {metric!r}")
        parsed.append(item)
    return parsed


def parse_frost_epoch_wall_summaries(text: str) -> dict[int, float]:
    rows = parse_all_json_after_prefix(text, "APPENDIX_TRAIN_EPOCH_WALL")
    return {int(row["epoch"]): float(row["wall_time_s"]) for row in rows}


def parse_pipetgl_run_summary(text: str) -> dict[str, float | int | str]:
    rows = parse_all_json_after_prefix(text, "PIPETGL_RUN_SUMMARY")
    if len(rows) != 1:
        raise RuntimeError(f"Expected one PipeTGL run summary, got {len(rows)}")
    return rows[0]


def parse_frost_epoch_summaries(text: str) -> list[dict[str, float | int | str]]:
    pattern = re.compile(
        r"\[Epoch\s+(\d+)\]\s+Train Loss:\s*([0-9eE+\-.]+)\s*\|\s*"
        r"Val AP(?:\(New high\))?:\s*([0-9eE+\-.]+)\s*\|\s*"
        r"Val AUC:\s*([0-9eE+\-.]+)\s*\|\s*"
        r"Test AUC:\s*([0-9eE+\-.]+)\s*\|\s*"
        r"Test AP:\s*([0-9eE+\-.]+)"
    )
    rows: list[dict[str, float | int | str]] = [
        {
            "system": "frost",
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "evaluation_metric": "ap",
            "val_ap": float(val_ap),
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "test_ap": float(test_ap),
        }
        for epoch, train_loss, val_ap, val_auc, test_auc, test_ap in pattern.findall(
            text
        )
    ]
    mrr_pattern = re.compile(
        r"\[Epoch\s+(\d+)\]\s+Train Loss:\s*([0-9eE+\-.]+)\s*\|\s*"
        r"Val MRR(?:\(New high\))?:\s*([0-9eE+\-.]+)\s*\|\s*"
        r"Test MRR:\s*([0-9eE+\-.]+)"
    )
    rows.extend(
        {
            "system": "frost",
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "evaluation_metric": "mrr",
            "val_mrr": float(val_mrr),
            "test_mrr": float(test_mrr),
        }
        for epoch, train_loss, val_mrr, test_mrr in mrr_pattern.findall(text)
    )
    if not rows:
        raise RuntimeError("Could not parse FROST epoch summaries.")
    rows.sort(key=lambda row: int(row["epoch"]))
    return rows


def select_bottleneck_rank(
    throughput_by_rank: ThroughputByRank,
) -> tuple[int, float, int]:
    rank, payload = min(
        throughput_by_rank.items(),
        key=lambda item: float(item[1]["events_per_sec_per_gpu"]),
    )
    return (
        rank,
        float(payload["events_per_sec_per_gpu"]),
        int(payload["measured_batches"]),
    )


def mean_events_per_sec(throughput_by_rank: ThroughputByRank) -> float:
    values = [
        float(payload["events_per_sec_per_gpu"])
        for payload in throughput_by_rank.values()
    ]
    return sum(values) / len(values)
