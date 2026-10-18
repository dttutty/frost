#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from json import JSONDecoder
from pathlib import Path

import matplotlib.pyplot as plt


FIG_DIR = Path(__file__).resolve().parent
TGN_CURVES_JSON = FIG_DIR / "tgn_lastfm_nvidia_smi_gpu_util_progress_500mb_curves.json"
FROST_2GPU_LOG = FIG_DIR / "frost_2gpu_lastfm_dygformer_mb4000.log"
DYGLIB_1GPU_LOG = FIG_DIR / "dyglib_1gpu_lastfm_dygformer_mb4000.log"
OUTPUT_PDF = FIG_DIR / "gpu_util_lastfm_tgn_dygformer_bs4000.pdf"

TGN_BATCH_SIZE = 4000
SAMPLE_SKIP = 100
SAMPLE_COUNT = 500


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_path_from_log(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}_nvidia_smi.csv")


def parse_all_json_after_prefix(text: str, prefix: str) -> list[dict]:
    rows: list[dict] = []
    decoder = JSONDecoder()
    for match in re.finditer(re.escape(prefix) + r"\s+", text):
        start = match.end()
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        rows.append(obj)
    if not rows:
        raise RuntimeError(f"Could not find `{prefix}` JSON payloads in log.")
    return rows


def parse_interval_rows(log_path: Path) -> dict[int, dict]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = parse_all_json_after_prefix(text, "APPENDIX_TRAIN_INTERVALS")
    return {int(row["rank"]): row for row in rows}


def parse_nvidia_smi_samples(sample_path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for line in sample_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("timestamp"):
            continue
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) != 4:
            continue
        timestamp_s, gpu_index_s, gpu_util_s, mem_util_s = parts
        timestamp = datetime.strptime(
            timestamp_s, "%Y/%m/%d %H:%M:%S.%f"
        ).replace(tzinfo=timezone.utc).timestamp()
        gpu_util = float(gpu_util_s)
        mem_util = float(mem_util_s)
        if not (0.0 <= gpu_util <= 100.0 and 0.0 <= mem_util <= 100.0):
            continue
        rows.append(
            {
                "timestamp_unix_s": timestamp,
                "gpu_index": int(gpu_index_s),
                "gpu_util": gpu_util,
            }
        )
    if not rows:
        raise RuntimeError(f"No nvidia-smi samples found in {sample_path}")
    return rows


def filter_samples_for_intervals(
    rows: list[dict[str, float | int]],
    *,
    gpu_index: int,
    interval_start_unix_s: list[float],
    interval_end_unix_s: list[float],
) -> list[dict[str, float | int]]:
    if len(interval_start_unix_s) != len(interval_end_unix_s):
        raise RuntimeError("Interval start/end length mismatch.")
    gpu_rows = [row for row in rows if int(row["gpu_index"]) == gpu_index]
    selected: list[dict[str, float | int]] = []
    interval_idx = 0
    for row in gpu_rows:
        ts = float(row["timestamp_unix_s"])
        while interval_idx < len(interval_end_unix_s) and ts > float(interval_end_unix_s[interval_idx]):
            interval_idx += 1
        if interval_idx >= len(interval_start_unix_s):
            break
        start_ts = float(interval_start_unix_s[interval_idx])
        end_ts = float(interval_end_unix_s[interval_idx])
        if start_ts <= ts <= end_ts:
            selected.append(row)
    if not selected:
        raise RuntimeError(
            f"No nvidia-smi samples for gpu_index={gpu_index} within measured intervals."
        )
    return selected


def pair_mean_samples(
    left: list[dict[str, float | int]],
    right: list[dict[str, float | int]],
    *,
    max_delta_s: float = 0.02,
) -> list[float]:
    paired: list[float] = []
    left_idx = 0
    right_idx = 0
    while left_idx < len(left) and right_idx < len(right):
        left_ts = float(left[left_idx]["timestamp_unix_s"])
        right_ts = float(right[right_idx]["timestamp_unix_s"])
        delta = left_ts - right_ts
        if abs(delta) <= max_delta_s:
            paired.append(
                (float(left[left_idx]["gpu_util"]) + float(right[right_idx]["gpu_util"])) / 2.0
            )
            left_idx += 1
            right_idx += 1
        elif delta < 0.0:
            left_idx += 1
        else:
            right_idx += 1
    if not paired:
        raise RuntimeError("Could not pair FROST worker samples into a mean curve.")
    return paired


def select_sample_window(values: list[float], *, skip: int, count: int) -> list[float]:
    start = skip
    end = skip + count
    if len(values) < end:
        raise RuntimeError(
            f"Need at least {end} samples after filtering, got {len(values)}."
        )
    return values[start:end]


def load_dygformer_sample_window_curves() -> dict[str, list[float]]:
    frost_sample_path = sample_path_from_log(FROST_2GPU_LOG)
    dyglib_sample_path = sample_path_from_log(DYGLIB_1GPU_LOG)

    frost_intervals = parse_interval_rows(FROST_2GPU_LOG)
    dyglib_intervals = parse_interval_rows(DYGLIB_1GPU_LOG)
    frost_samples = parse_nvidia_smi_samples(frost_sample_path)
    dyglib_samples = parse_nvidia_smi_samples(dyglib_sample_path)

    frost_rank0 = filter_samples_for_intervals(
        frost_samples,
        gpu_index=0,
        interval_start_unix_s=[float(v) for v in frost_intervals[0]["interval_start_unix_s"]],
        interval_end_unix_s=[float(v) for v in frost_intervals[0]["interval_end_unix_s"]],
    )
    frost_rank1 = filter_samples_for_intervals(
        frost_samples,
        gpu_index=1,
        interval_start_unix_s=[float(v) for v in frost_intervals[1]["interval_start_unix_s"]],
        interval_end_unix_s=[float(v) for v in frost_intervals[1]["interval_end_unix_s"]],
    )
    dyglib_rank0 = filter_samples_for_intervals(
        dyglib_samples,
        gpu_index=0,
        interval_start_unix_s=[float(v) for v in dyglib_intervals[0]["interval_start_unix_s"]],
        interval_end_unix_s=[float(v) for v in dyglib_intervals[0]["interval_end_unix_s"]],
    )

    frost_mean_gpu_util = pair_mean_samples(frost_rank0, frost_rank1)
    dyglib_gpu_util = [float(row["gpu_util"]) for row in dyglib_rank0]

    return {
        "frost_2gpu": select_sample_window(frost_mean_gpu_util, skip=SAMPLE_SKIP, count=SAMPLE_COUNT),
        "dyglib_1gpu": select_sample_window(dyglib_gpu_util, skip=SAMPLE_SKIP, count=SAMPLE_COUNT),
    }


def main() -> int:
    tgn_rows = load_json(TGN_CURVES_JSON)
    dyg_sample_curves = load_dygformer_sample_window_curves()

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.9),
        constrained_layout=True,
    )

    # Left: LastFM-TGN, bs=4000, full measured-progress axis.
    tgn_label = {"frost": "FROST", "tgl": "TGL"}
    tgn_color = {"frost": "#d62728", "tgl": "#29b6f6"}
    for row in tgn_rows:
        system = str(row["system"])
        if int(row["macro_batch_size"]) != TGN_BATCH_SIZE or system not in tgn_label:
            continue
        x = [float(v) for v in row["progress_pct_curve"]]
        y = [float(v) for v in row["gpu_util_curve"]]
        ax_left.plot(
            x,
            y,
            label=tgn_label[system],
            linewidth=1.3,
            color=tgn_color[system],
            alpha=0.95,
        )
    ax_left.set_title("TGN-LastFM", fontsize=10)
    ax_left.set_xlim(0.0, 100.0)
    ax_left.set_xticks([0, 25, 50, 75, 100])
    ax_left.set_ylim(0.0, 100.0)
    ax_left.set_xlabel("Measured progress (%)", fontsize=9)
    ax_left.set_ylabel("nvidia-smi GPU util (%)", fontsize=9)
    ax_left.grid(True, alpha=0.25)
    ax_left.legend(
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="#cccccc",
        fontsize=8,
        loc="upper right",
    )

    # Right: LastFM-DyGFormer, bs=4000, raw nvidia-smi sample window.
    dyg_label = {"frost_2gpu": "FROST", "dyglib_1gpu": "DyGLib"}
    dyg_color = {"frost_2gpu": "#d62728", "dyglib_1gpu": "#29b6f6"}
    x_curve = list(range(SAMPLE_SKIP + 1, SAMPLE_SKIP + SAMPLE_COUNT + 1))
    for system in ("frost_2gpu", "dyglib_1gpu"):
        ax_right.plot(
            x_curve,
            dyg_sample_curves[system],
            label=dyg_label[system],
            linewidth=1.3,
            color=dyg_color[system],
            alpha=0.95,
        )
    ax_right.set_title("DyGFormer-LastFM", fontsize=10)
    ax_right.set_xlim(float(SAMPLE_SKIP + 1), float(SAMPLE_SKIP + SAMPLE_COUNT))
    ax_right.set_xticks([101, 200, 300, 400, 500, 600])
    ax_right.set_ylim(0.0, 100.0)
    ax_right.set_xlabel(
        f"nvidia-smi sample index ({SAMPLE_SKIP + 1}-{SAMPLE_SKIP + SAMPLE_COUNT})",
        fontsize=9,
    )
    ax_right.grid(True, alpha=0.25)
    ax_right.legend(
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="#cccccc",
        fontsize=8,
        loc="upper right",
    )

    fig.savefig(OUTPUT_PDF)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
