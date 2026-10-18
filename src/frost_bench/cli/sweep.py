from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path

from frost_bench.cli.train import build_command
from frost_bench.config import PRESETS, normalize_dataset_name
from frost_bench.io import utc_timestamp
from frost_bench.paths import REPO_ROOT, artifact_root
from frost_bench.sweep import SweepJob, SweepSpec, load_sweeps, summarize_seed_logs


DEFAULT_SPEC = REPO_ROOT / "benchmarks" / "specs" / "sweeps.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a declared FROST model/dataset/seed sweep sequentially."
    )
    parser.add_argument("sweep", nargs="?")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--macro-batch-size", type=int)
    parser.add_argument("--stable-mode", choices=("true", "false"))
    parser.add_argument("--nproc-per-node", type=int, default=3)
    parser.add_argument("--master-port-base", type=int, default=29550)
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES value, for example 0,1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="HYDRA_OVERRIDE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def _canonical_models(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    aliases = {"graphmixer": "graph_mixer"}
    models = tuple(aliases.get(value.lower(), value.lower()) for value in values)
    unknown = sorted(set(models) - PRESETS.keys())
    if unknown:
        raise ValueError(
            f"Unknown models: {', '.join(unknown)}. Available: {', '.join(sorted(PRESETS))}"
        )
    return models


def _selected_spec(base: SweepSpec, args: argparse.Namespace) -> SweepSpec:
    stable_mode = base.stable_mode
    if args.stable_mode is not None:
        stable_mode = args.stable_mode == "true"
    return SweepSpec(
        name=base.name,
        description=base.description,
        models=_canonical_models(args.models or base.models),
        datasets=tuple(
            normalize_dataset_name(dataset)
            for dataset in (args.datasets or base.datasets)
        ),
        seeds=tuple(args.seeds or base.seeds),
        epoch=args.epoch if args.epoch is not None else base.epoch,
        stable_mode=stable_mode,
    )


def _output_dir(spec: SweepSpec, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return artifact_root() / "sweeps" / f"{spec.name}_{utc_timestamp()}"


def build_jobs(
    spec: SweepSpec,
    *,
    output_dir: Path,
    nproc_per_node: int,
    master_port_base: int,
    macro_batch_size: int | None,
    extra_overrides: list[str],
) -> tuple[SweepJob, ...]:
    jobs: list[SweepJob] = []
    for model in spec.models:
        for dataset in spec.datasets:
            for seed in spec.seeds:
                overrides = [
                    f"dataset={dataset}",
                    f"epoch={spec.epoch}",
                    f"seed={seed}",
                ]
                if spec.stable_mode is not None:
                    overrides.append(f"stable_mode={str(spec.stable_mode).lower()}")
                if macro_batch_size is not None:
                    overrides.append(f"macro_batch_size={macro_batch_size}")
                overrides.extend(extra_overrides)
                index = len(jobs)
                torchrun_args = build_command(
                    preset=PRESETS[model],
                    nproc_per_node=nproc_per_node,
                    master_addr="127.0.0.1",
                    master_port=master_port_base + index,
                    local_ranks_filter=None,
                    extra_overrides=overrides,
                )
                jobs.append(
                    SweepJob(
                        index=index,
                        model=model,
                        dataset=dataset,
                        seed=seed,
                        command=(
                            sys.executable,
                            "-m",
                            "torch.distributed.run",
                            *torchrun_args,
                        ),
                        log_path=output_dir
                        / model
                        / dataset.lower()
                        / f"seed_{seed}.log",
                    )
                )
    return tuple(jobs)


def _run_job(job: SweepJob, *, gpus: str | None) -> int:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    with job.log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(job.command)}\n")
        handle.flush()
        process = subprocess.Popen(
            job.command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return process.wait()


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "dataset", "seed", "exit_code", "log_path", "command"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _summarize(spec: SweepSpec, jobs: tuple[SweepJob, ...]) -> None:
    for model in spec.models:
        for dataset in spec.datasets:
            selected = [
                (job.seed, job.log_path)
                for job in jobs
                if job.model == model and job.dataset == dataset
            ]
            output_dir = selected[0][1].parent
            try:
                summarize_seed_logs(selected, output_dir=output_dir)
            except ValueError as exc:
                print(f"[summary skipped] {model}/{dataset}: {exc}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    sweeps = load_sweeps(args.spec)
    if args.list:
        for sweep in sweeps.values():
            print(f"{sweep.name}: {sweep.description}")
        return 0
    if args.sweep not in sweeps:
        parser.error(f"sweep must be one of: {', '.join(sorted(sweeps))}")

    try:
        spec = _selected_spec(sweeps[args.sweep], args)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = _output_dir(spec, args.output_dir)
    if args.summarize_only and args.output_dir is None:
        parser.error("--summarize-only requires --output-dir")
    jobs = build_jobs(
        spec,
        output_dir=output_dir,
        nproc_per_node=args.nproc_per_node,
        master_port_base=args.master_port_base,
        macro_batch_size=args.macro_batch_size,
        extra_overrides=args.set,
    )

    print(f"Sweep: {spec.name}")
    print(f"Output: {output_dir}")
    print(f"Jobs: {len(jobs)} (executed sequentially)")
    if args.summarize_only:
        _summarize(spec, jobs)
        return 0
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job.command))
        return 0

    rows: list[dict[str, object]] = []
    failures = 0
    for job in jobs:
        print(
            f"[{job.index + 1}/{len(jobs)}] {job.model}/{job.dataset}/seed={job.seed}"
        )
        exit_code = _run_job(job, gpus=args.gpus)
        rows.append(
            {
                "model": job.model,
                "dataset": job.dataset,
                "seed": job.seed,
                "exit_code": exit_code,
                "log_path": job.log_path,
                "command": shlex.join(job.command),
            }
        )
        _write_manifest(output_dir / "manifest.tsv", rows)
        if exit_code != 0:
            failures += 1
            if args.stop_on_error:
                break

    _summarize(spec, jobs)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
