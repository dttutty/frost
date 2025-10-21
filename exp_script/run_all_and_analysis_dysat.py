#!/usr/bin/env python3
import os
import sys
import time
import uuid
import csv
import re
import subprocess
from pathlib import Path
from statistics import mean, stdev
from datetime import datetime
import shutil

# ===== Fixed configuration =====
WORKDIR = Path.cwd()

CONFIG_DIR = WORKDIR / "config" / "dist"

DATASETS = ["LASTFM", "MOOC", "REDDIT", "WIKI"]
DATASETS = ["LASTFM"]
CONFIG_NAMES = ["APAN", "DySAT", "JODIE", "TGAT", "TGN"]
CONFIG_NAMES = ["DYSAT"]


OVERRIDE_EPOCHS = 11

TORCHRUN = shutil.which("torchrun") or f"{sys.executable} -m torch.distributed.run"
NPROC_PER_NODE = 3
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29500

CUDA_VISIBLE_DEVICES = "0,1"
TORCH_DISTRIBUTED_DEBUG = "DETAIL"
USE_dedup = True

# Feature sweep
FEATURE_DIMS = [64, 128, 192, 256, 512]
EDGE_DIMS = FEATURE_DIMS
NODE_DIMS = FEATURE_DIMS
CROSS_PRODUCT = False  # True => all edge x node; False => pair by index

# Sampling combination sweep (layer, neighbor_list)
SAMPLING_COMBOS = [
    # (1, [10]),
    # (1, [20]),
    # (1, [40]),
    # (1, [80]),
    (2, [10, 10]),
    (2, [10, 25]),
    (2, [25, 25]),
    # (3, [5, 5, 5]),
]

# Sampling single-parameter sweeps
SAMPLING_HISTORY = [2, 3, 4, 5, 6]
SAMPLING_DURATION = [10000, 20000, 30000, 40000, 50000]
SAMPLING_STRATEGY = ["uniform"]
BATCH_SIZE = [1000, 2000, 4000, 8000, 16000, 32000]
# =================================

# Regex patterns for parsing logs
PAT_EPOCH_TS = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+Epoch\s+(\d+)", re.M
)
PAT_TRAIN_TIME = re.compile(r"Training time:\s*([0-9.]+)")
PAT_SAMPLE_TIME = re.compile(r"sample time:\s*([0-9.]+)s", re.I)
PAT_BEST = re.compile(r"^\s*\d{4}-\d{2}-\d{2}.*Best model at epoch\s+(\d+)\.?", re.M)
PAT_BEST_TS = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+).*Best model at epoch\s+(\d+)\.?",
    re.M,
)
PAT_TEST = re.compile(r"test ap:\s*([0-9.]+)\s+test auc:\s*([0-9.]+)", re.I)

try:
    import yaml
except ImportError:
    print("Please install pyyaml: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ---------- YAML helpers ----------
def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def set_train_epoch(cfg: dict, epochs: int) -> dict:
    train = cfg.get("train")
    if isinstance(train, list) and train and isinstance(train[0], dict):
        train[0]["epoch"] = int(epochs)
    else:
        cfg["train"] = [{"epoch": int(epochs)}]
    return cfg


def ensure_sampling(cfg: dict) -> dict:
    samp = cfg.get("sampling")
    if not isinstance(samp, list) or not samp or not isinstance(samp[0], dict):
        cfg["sampling"] = [{}]
    return cfg


def override_sampling_layer_neighbor(
    cfg: dict, layer: int, neighbors: list[int]
) -> dict:
    cfg = ensure_sampling(cfg)
    s0 = cfg["sampling"][0]
    s0["layer"] = int(layer)
    s0["neighbor"] = [int(x) for x in neighbors]
    return cfg


def override_sampling_history(cfg: dict, history: int) -> dict:
    cfg = ensure_sampling(cfg)
    cfg["sampling"][0]["history"] = int(history)
    return cfg


def override_sampling_duration(cfg: dict, duration: int) -> dict:
    cfg = ensure_sampling(cfg)
    cfg["sampling"][0]["duration"] = int(duration)
    return cfg


def override_sampling_strategy(cfg: dict, strategy: str) -> dict:
    cfg = ensure_sampling(cfg)
    cfg["sampling"][0]["strategy"] = str(strategy)
    return cfg


def override_batch_size(cfg: dict, batch_size: int) -> dict:
    cfg = ensure_sampling(cfg)
    cfg["train"][0]["batch_size"] = str(batch_size)
    return cfg


def build_config_paths(config_dir: Path, names: list[str]) -> list[Path]:
    paths = []
    for n in names:
        p = config_dir / f"{n}.yml"
        if not p.is_file():
            print(f"Missing config file: {p}", file=sys.stderr)
            sys.exit(2)
        paths.append(p)
    return paths


# ---------- Logging/parse helpers ----------
def stream_and_log(proc: subprocess.Popen, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        for line in iter(proc.stdout.readline, b""):
            text = line.decode(errors="replace")
            sys.stdout.write(text)
            lf.write(text)
            lf.flush()
    return proc.wait()


def safe_mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def parse_log(text: str):
    # 1) epoch timestamps
    epoch_ts = []
    for m in PAT_EPOCH_TS.finditer(text):
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
        ep = int(m.group(2))
        epoch_ts.append((ep, ts))
    epoch_ts.sort(key=lambda x: x[0])

    # 2) training and sample time per epoch
    lines = text.splitlines()
    train_time = {}
    sample_time = {}
    current_epoch = None
    for ln in lines:
        m_ep = PAT_EPOCH_TS.search(ln)
        if m_ep:
            current_epoch = int(m_ep.group(2))
            continue
        if current_epoch is None:
            continue
        m_tt = PAT_TRAIN_TIME.search(ln)
        if m_tt:
            try:
                train_time[current_epoch] = float(m_tt.group(1))
            except ValueError:
                pass
        m_st = PAT_SAMPLE_TIME.search(ln)
        if m_st:
            try:
                sample_time[current_epoch] = float(m_st.group(1))
            except ValueError:
                pass

    # 3) best model epoch and its timestamp
    best_epoch = None
    best_ts = None
    for m in PAT_BEST_TS.finditer(text):
        best_ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
        best_epoch = int(m.group(2))
    if best_epoch is None:
        m_best = PAT_BEST.search(text)
        if m_best:
            best_epoch = int(m_best.group(1))

    # 4) delta to next epoch timestamp
    delta_next = {}
    for i in range(len(epoch_ts) - 1):
        ep_i, ts_i = epoch_ts[i]
        ep_j, ts_j = epoch_ts[i + 1]
        delta_next[ep_i] = (ts_j - ts_i).total_seconds()
    # Fill last epoch delta using "Best model ..." timestamp
    if epoch_ts:
        last_ep, last_ts = epoch_ts[-1]
        if best_ts and best_ts >= last_ts:
            delta_next[last_ep] = (best_ts - last_ts).total_seconds()

    # 5) test metrics
    test_ap = test_auc = None
    m_test = PAT_TEST.search(text)
    if m_test:
        try:
            test_ap = float(m_test.group(1))
            test_auc = float(m_test.group(2))
        except ValueError:
            pass

    return epoch_ts, train_time, sample_time, delta_next, best_epoch, test_ap, test_auc


def summarize(epoch_ts, train_time, sample_time, delta_next):
    if not epoch_ts:
        return None
    first_ep = min(ep for ep, _ in epoch_ts)

    train_vals = [v for ep, v in sorted(train_time.items()) if ep != first_ep]
    sample_vals = [v for ep, v in sorted(sample_time.items()) if ep != first_ep]
    delta_vals = [v for ep, v in sorted(delta_next.items()) if ep != first_ep]

    tm_mean, tm_std = safe_mean_std(train_vals)
    sm_mean, sm_std = safe_mean_std(sample_vals)
    dt_mean, dt_std = safe_mean_std(delta_vals)

    return {
        "exclude_first_epoch": first_ep,
        "train_time_mean": tm_mean,
        "train_time_std": tm_std,
        "sample_time_mean": sm_mean,
        "sample_time_std": sm_std,
        "delta_next_mean": dt_mean,
        "delta_next_std": dt_std,
        "n_train": len(train_vals),
        "n_sample": len(sample_vals),
        "n_delta": len(delta_vals),
    }


def get_log_dir(dataset: str, cfg_name: str) -> Path:
    return WORKDIR / "logs" / dataset / cfg_name


def append_summary_to_log(
    logfile: Path, summary, best_epoch, test_ap, test_auc, meta_lines: list[str]
):
    with logfile.open("a", encoding="utf-8") as lf:
        lf.write("\n=== Summary (auto) ===\n")
        for ln in meta_lines:
            lf.write(ln + "\n")
        if summary is None:
            lf.write("No epoch timestamps found.\n")
        else:

            def fmt(x):
                return "NA" if x is None else f"{x:.6f}"

            lf.write(f"Excluded first epoch: {summary['exclude_first_epoch']}\n")
            lf.write(
                f"Training time mean/std (s): {fmt(summary['train_time_mean'])} / {fmt(summary['train_time_std'])}\n"
            )
            lf.write(
                f"Sample time mean/std   (s): {fmt(summary['sample_time_mean'])} / {fmt(summary['sample_time_std'])}\n"
            )
            lf.write(
                f"Delta-to-next mean/std (s): {fmt(summary['delta_next_mean'])} / {fmt(summary['delta_next_std'])}\n"
            )
            lf.write(
                f"Best model epoch: {best_epoch if best_epoch is not None else 'NA'}\n"
            )
            lf.write(
                f"test ap: {test_ap if test_ap is not None else 'NA'}  test auc: {test_auc if test_auc is not None else 'NA'}\n"
            )
        lf.write("=======================\n")


def append_row_to_csv(csv_path: Path, header: list[str], row: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as cf:
        wr = csv.writer(cf)
        if write_header:
            wr.writerow(header)
        wr.writerow(row)


def classify_returncode(rc: int, log_text: str) -> str:
    """
    Returns one of: "ok", "aborted", "failed".
    - ok: rc == 0
    - aborted: terminated by signal or elastic ChildFailedError after successful end-of-run prints
    - failed: everything else
    """
    if rc == 0:
        return "ok"

    # Typical SIGINT/SIGTERM exit codes
    if rc in (-2, 130, 143):
        return "aborted"

    # torchrun often returns rc=1 with ChildFailedError on SIGINT;
    # treat as aborted if we already printed final metrics.
    text = log_text or ""
    saw_final = ("Best model at epoch" in text) or (
        re.search(r"test ap:\s*[0-9.]+\s+test auc:\s*[0-9.]+", text, re.I) is not None
    )
    if rc == 1 and "ChildFailedError" in text and saw_final:
        return "aborted"

    return "failed"


# ---------- Runner ----------
def run_one(
    dataset: str,
    base_cfg: Path,
    temp_cfg: Path,
    phase: str,
    edge_dim: int | None = None,
    node_dim: int | None = None,
    sampling_layer: int | None = None,
    sampling_neighbors: list[int] | None = None,
    sampling_history: int | None = None,
    sampling_duration: int | None = None,
    sampling_strategy: str | None = None,
    batch_size: int | None = None,
):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_name = base_cfg.stem
    log_dir = get_log_dir(dataset, cfg_name)

    # Build logfile name tag
    tag_parts = []
    if phase == "feat":
        tag_parts.append(f"ed{edge_dim}_nd{node_dim}")
    elif phase == "sampling_combo":
        nb = "x".join(str(x) for x in (sampling_neighbors or []))
        tag_parts.append(f"sl{sampling_layer}_nb{nb}")
    elif phase == "sampling_history":
        tag_parts.append(f"h{sampling_history}")
    elif phase == "sampling_duration":
        tag_parts.append(f"d{sampling_duration}")
    elif phase == "sampling_strategy":
        tag_parts.append(f"strat_{sampling_strategy}")
    elif phase == "batch_size":
        tag_parts.append(f"bs{batch_size}")
    tag = "_".join(tag_parts) if tag_parts else "run"

    logfile = log_dir / f"train_{dataset}_{cfg_name}_{phase}_{tag}_{stamp}.log"

    header = (
        "===============================\n"
        f"Start Time : {time.asctime()}\n"
        f"Phase      : {phase}\n"
        f"Dataset    : {dataset}\n"
        f"Base Config: {base_cfg}\n"
        f"Temp Config: {temp_cfg}\n"
        f"CUDA       : {os.environ.get('CUDA_VISIBLE_DEVICES','')}\n"
    )
    if phase == "feat":
        header += f"rand_edge_features: {edge_dim}\nrand_node_features: {node_dim}\n"
    if phase == "sampling_combo":
        header += f"sampling.layer: {sampling_layer}\nsampling.neighbor: {sampling_neighbors}\n"
    if phase == "sampling_history":
        header += f"sampling.history: {sampling_history}\n"
    if phase == "sampling_duration":
        header += f"sampling.duration: {sampling_duration}\n"
    if phase == "sampling_strategy":
        header += f"sampling.strategy: {sampling_strategy}\n"
    if phase == "batch_size":
        header += f"train.batch_size: {batch_size}\n"
    header += "===============================\n"

    log_dir.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as lf:
        sys.stdout.write(header)
        lf.write(header)

    cmd = [
        str(TORCHRUN),
        "--nproc_per_node",
        str(NPROC_PER_NODE),
        "--master_addr",
        MASTER_ADDR,
        "--master_port",
        str(MASTER_PORT),
        "train_dist.py",
        "--data",
        dataset,
        "--config",
        str(temp_cfg),
    ]
    if phase == "feat":
        cmd += [
            "--rand_edge_features",
            str(edge_dim),
            "--rand_node_features",
            str(node_dim),
        ]
    if USE_dedup:
        cmd.append("--dedup")

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKDIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    rc = stream_and_log(proc, logfile)
    elapsed = time.perf_counter() - start

    text = logfile.read_text(encoding="utf-8", errors="replace")
    epoch_ts, train_time_map, sample_time_map, delta_next_map, best_ep, ap, auc = (
        parse_log(text)
    )
    summary = summarize(epoch_ts, train_time_map, sample_time_map, delta_next_map)
    status = classify_returncode(rc, text)

    footer = (
        f"\nEnd Time   : {time.asctime()}\n"
        f"Elapsed(s) : {elapsed:.3f}\n"
        f"Run status : {status}\n"
        "===============================\n"
    )
    with logfile.open("a", encoding="utf-8") as lf:
        sys.stdout.write(footer)
        lf.write(footer)

    footer = (
        f"\nEnd Time   : {time.asctime()}\n"
        f"Elapsed(s) : {elapsed:.3f}\n"
        "===============================\n"
    )
    with logfile.open("a", encoding="utf-8") as lf:
        sys.stdout.write(footer)
        lf.write(footer)

    # # Parse and summarize
    # text = logfile.read_text(encoding="utf-8", errors="replace")
    # epoch_ts, train_time_map, sample_time_map, delta_next_map, best_ep, ap, auc = parse_log(text)
    # summary = summarize(epoch_ts, train_time_map, sample_time_map, delta_next_map)

    # Append to log
    meta_lines = []
    if phase == "feat":
        meta_lines += [
            f"rand_edge_features: {edge_dim}",
            f"rand_node_features: {node_dim}",
        ]
    if phase == "sampling_combo":
        meta_lines += [
            f"sampling.layer: {sampling_layer}",
            f"sampling.neighbor: {sampling_neighbors}",
        ]
    if phase == "sampling_history":
        meta_lines += [f"sampling.history: {sampling_history}"]
    if phase == "sampling_duration":
        meta_lines += [f"sampling.duration: {sampling_duration}"]
    if phase == "sampling_strategy":
        meta_lines += [f"sampling.strategy: {sampling_strategy}"]
    if phase == "batch_size":
        meta_lines += [f"train.batch_size: {batch_size}"]
    append_summary_to_log(logfile, summary, best_ep, ap, auc, meta_lines)

    # Append to CSV in logs/{dataset}/{model}/
    if phase == "feat":
        csv_path = log_dir / "feature_size_sweep.csv"
        header = [
            "run_stamp",
            "dataset",
            "config",
            "rand_edge_features",
            "rand_node_features",
            "best_epoch",
            "test_ap",
            "test_auc",
            "train_mean_s",
            "train_std_s",
            "n_train",
            "sample_mean_s",
            "sample_std_s",
            "n_sample",
            "delta_next_mean_s",
            "delta_next_std_s",
            "n_delta",
            "logfile",
        ]
        row = [
            stamp,
            dataset,
            cfg_name,
            edge_dim,
            node_dim,
            (best_ep if best_ep is not None else ""),
            (f"{ap:.6f}" if ap is not None else ""),
            (f"{auc:.6f}" if auc is not None else ""),
            (
                f"{summary['train_time_mean']:.6f}"
                if summary and summary["train_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['train_time_std']:.6f}"
                if summary and summary["train_time_std"] is not None
                else ""
            ),
            (summary["n_train"] if summary else 0),
            (
                f"{summary['sample_time_mean']:.6f}"
                if summary and summary["sample_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['sample_time_std']:.6f}"
                if summary and summary["sample_time_std"] is not None
                else ""
            ),
            (summary["n_sample"] if summary else 0),
            (
                f"{summary['delta_next_mean']:.6f}"
                if summary and summary["delta_next_mean"] is not None
                else ""
            ),
            (
                f"{summary['delta_next_std']:.6f}"
                if summary and summary["delta_next_std"] is not None
                else ""
            ),
            (summary["n_delta"] if summary else 0),
            str(logfile),
        ]
        append_row_to_csv(csv_path, header, row)

    elif phase == "sampling_combo":
        csv_path = log_dir / "sampling_combination_sweep.csv"
        nb = "-".join(str(x) for x in (sampling_neighbors or []))
        header = [
            "run_stamp",
            "dataset",
            "config",
            "sampling_layer",
            "sampling_neighbors",
            "best_epoch",
            "test_ap",
            "test_auc",
            "train_mean_s",
            "train_std_s",
            "n_train",
            "sample_mean_s",
            "sample_std_s",
            "n_sample",
            "delta_next_mean_s",
            "delta_next_std_s",
            "n_delta",
            "logfile",
        ]
        row = [
            stamp,
            dataset,
            cfg_name,
            sampling_layer,
            nb,
            (best_ep if best_ep is not None else ""),
            (f"{ap:.6f}" if ap is not None else ""),
            (f"{auc:.6f}" if auc is not None else ""),
            (
                f"{summary['train_time_mean']:.6f}"
                if summary and summary["train_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['train_time_std']:.6f}"
                if summary and summary["train_time_std"] is not None
                else ""
            ),
            (summary["n_train"] if summary else 0),
            (
                f"{summary['sample_time_mean']:.6f}"
                if summary and summary["sample_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['sample_time_std']:.6f}"
                if summary and summary["sample_time_std"] is not None
                else ""
            ),
            (summary["n_sample"] if summary else 0),
            (
                f"{summary['delta_next_mean']:.6f}"
                if summary and summary["delta_next_mean"] is not None
                else ""
            ),
            (
                f"{summary['delta_next_std']:.6f}"
                if summary and summary["delta_next_std"] is not None
                else ""
            ),
            (summary["n_delta"] if summary else 0),
            str(logfile),
        ]
        append_row_to_csv(csv_path, header, row)

    elif phase == "sampling_history":
        csv_path = log_dir / "sampling_history_sweep.csv"
        header = [
            "run_stamp",
            "dataset",
            "config",
            "sampling_history",
            "best_epoch",
            "test_ap",
            "test_auc",
            "train_mean_s",
            "train_std_s",
            "n_train",
            "sample_mean_s",
            "sample_std_s",
            "n_sample",
            "delta_next_mean_s",
            "delta_next_std_s",
            "n_delta",
            "logfile",
        ]
        row = [
            stamp,
            dataset,
            cfg_name,
            sampling_history,
            (best_ep if best_ep is not None else ""),
            (f"{ap:.6f}" if ap is not None else ""),
            (f"{auc:.6f}" if auc is not None else ""),
            (
                f"{summary['train_time_mean']:.6f}"
                if summary and summary["train_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['train_time_std']:.6f}"
                if summary and summary["train_time_std"] is not None
                else ""
            ),
            (summary["n_train"] if summary else 0),
            (
                f"{summary['sample_time_mean']:.6f}"
                if summary and summary["sample_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['sample_time_std']:.6f}"
                if summary and summary["sample_time_std"] is not None
                else ""
            ),
            (summary["n_sample"] if summary else 0),
            (
                f"{summary['delta_next_mean']:.6f}"
                if summary and summary["delta_next_mean"] is not None
                else ""
            ),
            (
                f"{summary['delta_next_std']:.6f}"
                if summary and summary["delta_next_std"] is not None
                else ""
            ),
            (summary["n_delta"] if summary else 0),
            str(logfile),
        ]
        append_row_to_csv(csv_path, header, row)

    elif phase == "sampling_duration":
        csv_path = log_dir / "sampling_duration_sweep.csv"
        header = [
            "run_stamp",
            "dataset",
            "config",
            "sampling_duration",
            "best_epoch",
            "test_ap",
            "test_auc",
            "train_mean_s",
            "train_std_s",
            "n_train",
            "sample_mean_s",
            "sample_std_s",
            "n_sample",
            "delta_next_mean_s",
            "delta_next_std_s",
            "n_delta",
            "logfile",
        ]
        row = [
            stamp,
            dataset,
            cfg_name,
            sampling_duration,
            (best_ep if best_ep is not None else ""),
            (f"{ap:.6f}" if ap is not None else ""),
            (f"{auc:.6f}" if auc is not None else ""),
            (
                f"{summary['train_time_mean']:.6f}"
                if summary and summary["train_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['train_time_std']:.6f}"
                if summary and summary["train_time_std"] is not None
                else ""
            ),
            (summary["n_train"] if summary else 0),
            (
                f"{summary['sample_time_mean']:.6f}"
                if summary and summary["sample_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['sample_time_std']:.6f}"
                if summary and summary["sample_time_std"] is not None
                else ""
            ),
            (summary["n_sample"] if summary else 0),
            (
                f"{summary['delta_next_mean']:.6f}"
                if summary and summary["delta_next_mean"] is not None
                else ""
            ),
            (
                f"{summary['delta_next_std']:.6f}"
                if summary and summary["delta_next_std"] is not None
                else ""
            ),
            (summary["n_delta"] if summary else 0),
            str(logfile),
        ]
        append_row_to_csv(csv_path, header, row)

    elif phase == "sampling_strategy":
        csv_path = log_dir / "sampling_strategy_sweep.csv"
        header = [
            "run_stamp",
            "dataset",
            "config",
            "sampling_strategy",
            "best_epoch",
            "test_ap",
            "test_auc",
            "train_mean_s",
            "train_std_s",
            "n_train",
            "sample_mean_s",
            "sample_std_s",
            "n_sample",
            "delta_next_mean_s",
            "delta_next_std_s",
            "n_delta",
            "logfile",
        ]
        row = [
            stamp,
            dataset,
            cfg_name,
            sampling_strategy,
            (best_ep if best_ep is not None else ""),
            (f"{ap:.6f}" if ap is not None else ""),
            (f"{auc:.6f}" if auc is not None else ""),
            (
                f"{summary['train_time_mean']:.6f}"
                if summary and summary["train_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['train_time_std']:.6f}"
                if summary and summary["train_time_std"] is not None
                else ""
            ),
            (summary["n_train"] if summary else 0),
            (
                f"{summary['sample_time_mean']:.6f}"
                if summary and summary["sample_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['sample_time_std']:.6f}"
                if summary and summary["sample_time_std"] is not None
                else ""
            ),
            (summary["n_sample"] if summary else 0),
            (
                f"{summary['delta_next_mean']:.6f}"
                if summary and summary["delta_next_mean"] is not None
                else ""
            ),
            (
                f"{summary['delta_next_std']:.6f}"
                if summary and summary["delta_next_std"] is not None
                else ""
            ),
            (summary["n_delta"] if summary else 0),
            str(logfile),
        ]
        append_row_to_csv(csv_path, header, row)

    elif phase == "batch_size":
        csv_path = log_dir / "batch_size_sweep.csv"
        header = [
            "run_stamp",
            "dataset",
            "config",
            "batch_size",
            "best_epoch",
            "test_ap",
            "test_auc",
            "train_mean_s",
            "train_std_s",
            "n_train",
            "sample_mean_s",
            "sample_std_s",
            "n_sample",
            "delta_next_mean_s",
            "delta_next_std_s",
            "n_delta",
            "logfile",
        ]
        row = [
            stamp,
            dataset,
            cfg_name,
            batch_size,
            (best_ep if best_ep is not None else ""),
            (f"{ap:.6f}" if ap is not None else ""),
            (f"{auc:.6f}" if auc is not None else ""),
            (
                f"{summary['train_time_mean']:.6f}"
                if summary and summary["train_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['train_time_std']:.6f}"
                if summary and summary["train_time_std"] is not None
                else ""
            ),
            (summary["n_train"] if summary else 0),
            (
                f"{summary['sample_time_mean']:.6f}"
                if summary and summary["sample_time_mean"] is not None
                else ""
            ),
            (
                f"{summary['sample_time_std']:.6f}"
                if summary and summary["sample_time_std"] is not None
                else ""
            ),
            (summary["n_sample"] if summary else 0),
            (
                f"{summary['delta_next_mean']:.6f}"
                if summary and summary["delta_next_mean"] is not None
                else ""
            ),
            (
                f"{summary['delta_next_std']:.6f}"
                if summary and summary["delta_next_std"] is not None
                else ""
            ),
            (summary["n_delta"] if summary else 0),
            str(logfile),
        ]
        append_row_to_csv(csv_path, header, row)
    return rc


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = TORCH_DISTRIBUTED_DEBUG

    WORKDIR.mkdir(parents=True, exist_ok=True)

    config_paths = build_config_paths(CONFIG_DIR, CONFIG_NAMES)

    # Feature pairs
    if CROSS_PRODUCT:
        feat_pairs = [(e, n) for e in EDGE_DIMS for n in NODE_DIMS]
    else:
        ln = min(len(EDGE_DIMS), len(NODE_DIMS))
        feat_pairs = list(zip(EDGE_DIMS[:ln], NODE_DIMS[:ln]))

    failures = []

    # Phase A: feature size sweep (with --rand_* params)
    for ds in DATASETS:
        for cfg_path in config_paths:
            for ed, nd in feat_pairs:
                try:
                    cfg_obj = load_yaml(cfg_path)
                    cfg_obj = set_train_epoch(cfg_obj, OVERRIDE_EPOCHS)
                    tmp_name = f"{cfg_path.stem}_{uuid.uuid4().hex[:8]}.yml"
                    tmp_path = WORKDIR / "temp" / ds / cfg_path.stem / tmp_name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    save_yaml(cfg_obj, tmp_path)

                    rc = run_one(
                        dataset=ds,
                        base_cfg=cfg_path,
                        temp_cfg=tmp_path,
                        phase="feat",
                        edge_dim=ed,
                        node_dim=nd,
                    )
                    if rc != 0:
                        failures.append(
                            (f"feat/{ds}/{cfg_path.name}/ed{ed}/nd{nd}", rc)
                        )
                except Exception as e:
                    failures.append(
                        (f"feat/{ds}/{cfg_path.name}/ed{ed}/nd{nd}", f"exception:{e}")
                    )

    # Phase B: sampling combination sweep (override layer+neighbor; no rand params)
    for ds in DATASETS:
        for cfg_path in config_paths:
            for layer, neigh in SAMPLING_COMBOS:
                try:
                    cfg_obj = load_yaml(cfg_path)
                    cfg_obj = set_train_epoch(cfg_obj, OVERRIDE_EPOCHS)
                    cfg_obj = override_sampling_layer_neighbor(
                        cfg_obj, layer=layer, neighbors=neigh
                    )
                    tmp_name = f"{cfg_path.stem}_{uuid.uuid4().hex[:8]}.yml"
                    tmp_path = WORKDIR / "temp" / ds / cfg_path.stem / tmp_name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    save_yaml(cfg_obj, tmp_path)

                    rc = run_one(
                        dataset=ds,
                        base_cfg=cfg_path,
                        temp_cfg=tmp_path,
                        phase="sampling_combo",
                        sampling_layer=layer,
                        sampling_neighbors=neigh,
                    )
                    if rc != 0:
                        failures.append(
                            (
                                f"samp_combo/{ds}/{cfg_path.name}/sl{layer}/nb{'-'.join(map(str,neigh))}",
                                rc,
                            )
                        )
                except Exception as e:
                    failures.append(
                        (
                            f"samp_combo/{ds}/{cfg_path.name}/sl{layer}/nb{'-'.join(map(str,neigh))}",
                            f"exception:{e}",
                        )
                    )

    # Phase C: sampling.history sweep (only override history)
    for ds in DATASETS:
        for cfg_path in config_paths:
            for h in SAMPLING_HISTORY:
                try:
                    cfg_obj = load_yaml(cfg_path)
                    cfg_obj = set_train_epoch(cfg_obj, OVERRIDE_EPOCHS)
                    cfg_obj = override_sampling_history(cfg_obj, history=h)
                    tmp_name = f"{cfg_path.stem}_{uuid.uuid4().hex[:8]}.yml"
                    tmp_path = WORKDIR / "temp" / ds / cfg_path.stem / tmp_name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    save_yaml(cfg_obj, tmp_path)

                    rc = run_one(
                        dataset=ds,
                        base_cfg=cfg_path,
                        temp_cfg=tmp_path,
                        phase="sampling_history",
                        sampling_history=h,
                    )
                    if rc != 0:
                        failures.append((f"samp_hist/{ds}/{cfg_path.name}/h{h}", rc))
                except Exception as e:
                    failures.append(
                        (f"samp_hist/{ds}/{cfg_path.name}/h{h}", f"exception:{e}")
                    )

    # Phase D: sampling.duration sweep (only override duration)
    for ds in DATASETS:
        for cfg_path in config_paths:
            for d in SAMPLING_DURATION:
                try:
                    cfg_obj = load_yaml(cfg_path)
                    cfg_obj = set_train_epoch(cfg_obj, OVERRIDE_EPOCHS)
                    cfg_obj = override_sampling_duration(cfg_obj, duration=d)
                    tmp_name = f"{cfg_path.stem}_{uuid.uuid4().hex[:8]}.yml"
                    tmp_path = WORKDIR / "temp" / ds / cfg_path.stem / tmp_name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    save_yaml(cfg_obj, tmp_path)

                    rc = run_one(
                        dataset=ds,
                        base_cfg=cfg_path,
                        temp_cfg=tmp_path,
                        phase="sampling_duration",
                        sampling_duration=d,
                    )
                    if rc != 0:
                        failures.append((f"samp_dur/{ds}/{cfg_path.name}/d{d}", rc))
                except Exception as e:
                    failures.append(
                        (f"samp_dur/{ds}/{cfg_path.name}/d{d}", f"exception:{e}")
                    )

    # Phase E: sampling.strategy sweep (only override strategy)
    for ds in DATASETS:
        for cfg_path in config_paths:
            for s in SAMPLING_STRATEGY:
                try:
                    cfg_obj = load_yaml(cfg_path)
                    cfg_obj = set_train_epoch(cfg_obj, OVERRIDE_EPOCHS)
                    cfg_obj = override_sampling_strategy(cfg_obj, strategy=s)
                    tmp_name = f"{cfg_path.stem}_{uuid.uuid4().hex[:8]}.yml"
                    tmp_path = WORKDIR / "temp" / ds / cfg_path.stem / tmp_name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    save_yaml(cfg_obj, tmp_path)

                    rc = run_one(
                        dataset=ds,
                        base_cfg=cfg_path,
                        temp_cfg=tmp_path,
                        phase="sampling_strategy",
                        sampling_strategy=s,
                    )
                    if rc != 0:
                        failures.append((f"samp_strat/{ds}/{cfg_path.name}/{s}", rc))
                except Exception as e:
                    failures.append(
                        (f"samp_strat/{ds}/{cfg_path.name}/{s}", f"exception:{e}")
                    )

    # Phase F: batch.size sweep (only override batch size)
    for ds in DATASETS:
        for cfg_path in config_paths:
            for s in BATCH_SIZE:
                try:
                    cfg_obj = load_yaml(cfg_path)
                    cfg_obj = set_train_epoch(cfg_obj, OVERRIDE_EPOCHS)
                    cfg_obj = override_batch_size(cfg_obj, batch_size=s)
                    tmp_name = f"{cfg_path.stem}_{uuid.uuid4().hex[:8]}.yml"
                    tmp_path = WORKDIR / "temp" / ds / cfg_path.stem / tmp_name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    save_yaml(cfg_obj, tmp_path)

                    rc = run_one(
                        dataset=ds,
                        base_cfg=cfg_path,
                        temp_cfg=tmp_path,
                        phase="batch_size",
                        batch_size=s,
                    )
                    if rc != 0:
                        failures.append((f"train_bs/{ds}/{cfg_path.name}/{s}", rc))
                except Exception as e:
                    failures.append(
                        (f"train_bs/{ds}/{cfg_path.name}/{s}", f"exception:{e}")
                    )

    if failures:
        print("\nSome runs failed:", file=sys.stderr)
        for name, rc in failures:
            print(f"- {name} -> {rc}", file=sys.stderr)
        sys.exit(1)

    print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()
