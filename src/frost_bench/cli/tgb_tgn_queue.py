"""Schedule single-GPU FROST and official-TGB TGN jobs over a GPU pool."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


_DATASETS = {
    # frost name, official name, sampler, eval negatives, queue weight
    "wiki": ("TGBL_WIKI", "tgbl-wiki", "rnd_dst_then_tgb_wiki", 999, 5),
    "review": ("TGBL_REVIEW", "tgbl-review", "rnd_dst_then_tgb", 100, 20),
    "coin": ("TGBL_COIN", "tgbl-coin", "rnd_dst_then_tgb", 20, 80),
    "comment": ("TGBL_COMMENT", "tgbl-comment", "rnd_dst_then_tgb", 20, 120),
    "flight": ("TGBL_FLIGHT", "tgbl-flight", "rnd_dst_then_tgb", 20, 160),
}


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    system: str
    priority: int
    command: list[str]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(_DATASETS), required=True
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("frost", "official"),
        default=["frost", "official"],
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=1)
    seed_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="run a task matrix over multiple seeds, for example --seeds 1 2 3 4 5",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def build_tasks(
    *,
    repo_root: Path,
    datasets: Sequence[str],
    systems: Sequence[str],
    epochs: int,
    batch_size: int,
    seed: int,
    output_dir: Path,
    patience: int = 5,
    tolerance: float = 1e-6,
    name_prefix: str = "",
) -> list[Task]:
    python = Path(sys.executable)
    tasks: list[Task] = []
    for short_name in datasets:
        (
            frost_name,
            official_name,
            neg_sampler,
            eval_n_neg,
            weight,
        ) = _DATASETS[short_name]
        if "official" in systems:
            name = f"{name_prefix}official-{short_name}"
            tasks.append(
                Task(
                    name=name,
                    dataset=short_name,
                    system="official",
                    priority=2 * weight,
                    command=[
                        str(python),
                        "-m",
                        "frost_bench.cli.official_tgb_tgn",
                        "--dataset",
                        official_name,
                        "--epochs",
                        str(epochs),
                        "--batch-size",
                        str(batch_size),
                        "--eval-batch-size",
                        str(batch_size),
                        "--patience",
                        str(patience),
                        "--tolerance",
                        str(tolerance),
                        "--loss-mode",
                        "corrected",
                        "--seed",
                        str(seed),
                        "--output",
                        str(output_dir / name / "metrics.json"),
                    ],
                )
            )
        if "frost" in systems:
            name = f"{name_prefix}frost-{short_name}"
            tasks.append(
                Task(
                    name=name,
                    dataset=short_name,
                    system="frost",
                    priority=weight,
                    command=[
                        str(python),
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=2",
                        str(repo_root / "run.py"),
                        "model=tgn",
                        f"dataset={frost_name}",
                        f"neg_sampler={neg_sampler}",
                        "eval_metric=mrr",
                        f"epoch={epochs}",
                        f"patience={patience}",
                        f"early_stop_tolerance={tolerance}",
                        f"macro_batch_size={batch_size}",
                        f"eval_macro_batch_size={batch_size}",
                        f"neg_sampler.eval.n_neg={eval_n_neg}",
                        f"seed={seed}",
                        "model.embedding_dim=100",
                        "d_time=100",
                        "dropout=0.1",
                        "runtime.single_gpu_resident=on",
                        "reset_eval_neg_sampler_each_epoch=true",
                        "tqdm=false",
                    ],
                )
            )
    return sorted(tasks, key=lambda task: (-task.priority, task.name))


def _resolve_gpu_ids(
    requested_gpu_ids: Sequence[str],
    inherited_visible_devices: str | None,
) -> list[str]:
    """Resolve queue GPU IDs without escaping an inherited visibility mask.

    With no inherited ``CUDA_VISIBLE_DEVICES``, queue IDs name host devices as
    before. When a launcher (for example Slurm) already supplies a visibility
    mask, numeric queue IDs are logical indexes into that mask. Non-numeric
    tokens must already be present in the mask. This keeps worker subprocesses
    inside the allocation instead of silently replacing its device selection.
    """
    requested = [gpu_id.strip() for gpu_id in requested_gpu_ids if gpu_id.strip()]
    if not requested:
        raise ValueError("--gpu-ids must select at least one GPU")
    if inherited_visible_devices is None:
        resolved = requested
    else:
        inherited = [
            gpu_id.strip()
            for gpu_id in inherited_visible_devices.split(",")
            if gpu_id.strip()
        ]
        if not inherited:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES is empty; the queue cannot select a GPU"
            )
        resolved = []
        for gpu_id in requested:
            try:
                logical_index = int(gpu_id)
            except ValueError:
                if gpu_id not in inherited:
                    raise ValueError(
                        f"GPU token {gpu_id!r} is outside inherited "
                        f"CUDA_VISIBLE_DEVICES={inherited_visible_devices!r}"
                    ) from None
                resolved.append(gpu_id)
                continue
            if logical_index < 0 or logical_index >= len(inherited):
                raise ValueError(
                    f"logical GPU index {logical_index} is outside inherited "
                    f"CUDA_VISIBLE_DEVICES={inherited_visible_devices!r}"
                )
            resolved.append(inherited[logical_index])
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"--gpu-ids resolves to duplicate devices: {resolved}")
    return resolved


def run_queue(tasks: Sequence[Task], gpu_ids: Sequence[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_queue: queue.Queue[Task] = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    statuses: list[dict[str, object]] = []
    status_lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            task_dir = output_dir / task.name
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "command.json").write_text(
                json.dumps(asdict(task), indent=2) + "\n", encoding="utf-8"
            )
            started = time.time()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env.setdefault("OMP_NUM_THREADS", "8")
            env.setdefault("MKL_NUM_THREADS", "8")
            with (task_dir / "stdout.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    task.command,
                    cwd=Path(__file__).resolve().parents[3],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            status = {
                "name": task.name,
                "dataset": task.dataset,
                "system": task.system,
                "gpu_id": gpu_id,
                "started_unix_s": started,
                "ended_unix_s": time.time(),
                "exit_code": process.returncode,
            }
            (task_dir / "status.json").write_text(
                json.dumps(status, indent=2) + "\n", encoding="utf-8"
            )
            with status_lock:
                statuses.append(status)
                print("TGB_TGN_QUEUE " + json.dumps(status, sort_keys=True), flush=True)
            task_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), daemon=False)
        for gpu_id in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = sorted(statuses, key=lambda status: str(status["name"]))
    (output_dir / "queue_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return int(any(status["exit_code"] != 0 for status in statuses))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch size, and patience must be positive")
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    gpu_ids = _resolve_gpu_ids(
        args.gpu_ids.split(","), os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    seeds = list(dict.fromkeys(args.seeds or [args.seed]))
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    repo_root = Path(__file__).resolve().parents[3]
    tasks = [
        task
        for seed in seeds
        for task in build_tasks(
            repo_root=repo_root,
            datasets=args.datasets,
            systems=args.systems,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            output_dir=args.output_dir.resolve(),
            patience=args.patience,
            tolerance=args.tolerance,
            name_prefix=f"seed-{seed}-" if len(seeds) > 1 else "",
        )
    ]
    tasks.sort(key=lambda task: (-task.priority, task.name))
    return run_queue(tasks, gpu_ids, args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
