from __future__ import annotations

import argparse
import os
import shlex

from torch.distributed import run

from frost.launch.affinity import mark_auto_thread_environment
from frost_bench.config import (
    DYGLIB_BACKED_PRESETS,
    PRESETS,
    Preset,
    build_dyglib_hydra_overrides,
    load_dataset_default_macro_batch_size,
)
from frost_bench.paths import REPO_ROOT


def get_override_value(overrides: tuple[str, ...] | list[str], key: str) -> str | None:
    prefix = f"{key}="
    for override in reversed(overrides):
        if override.startswith(prefix):
            return override[len(prefix) :]
    return None


def resolve_dataset_name(preset: Preset, normalized_extra_overrides: list[str]) -> str:
    dataset = get_override_value(normalized_extra_overrides, "dataset")
    dataset = dataset or get_override_value(preset.hydra_overrides, "dataset")
    if dataset is None:
        raise ValueError(f"Preset {preset.name!r} does not define a dataset override")
    return dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a common FROST training preset via torchrun."
    )
    parser.add_argument("preset", nargs="?", choices=sorted(PRESETS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--nproc-per-node", type=int, default=3)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--local-ranks-filter", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def print_presets() -> None:
    print("Available presets:")
    for preset in PRESETS.values():
        print(f"- {preset.name}: {preset.description}")
        print(f"  port={preset.master_port}")
        print(f"  overrides={' '.join(preset.hydra_overrides)}")
        if preset.name in DYGLIB_BACKED_PRESETS:
            print(
                "  note=dataset-specific parity settings come from benchmarks/specs/dyglib_parity.yaml"
            )


def build_command(
    preset: Preset,
    nproc_per_node: int,
    master_addr: str,
    master_port: int,
    local_ranks_filter: str | None,
    extra_overrides: list[str],
) -> list[str]:
    if preset.family == "tpnet":
        if nproc_per_node < 1:
            raise ValueError("TPNet requires at least one GPU process")
        return [
            f"--nproc_per_node={nproc_per_node}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            "-m",
            "frost_bench.tpnet_train",
            "--model_name",
            "TPNet",
            *extra_overrides,
        ]

    normalized_extra_overrides = [
        override.replace("dataset_name=", "dataset=", 1)
        if override.startswith("dataset_name=")
        else override
        for override in extra_overrides
    ]
    dataset_name = resolve_dataset_name(preset, normalized_extra_overrides)
    combined_overrides = [
        *preset.hydra_overrides,
        *build_dyglib_hydra_overrides(preset.name, dataset_name),
        *normalized_extra_overrides,
    ]
    if get_override_value(combined_overrides, "macro_batch_size") is None:
        default_batch_size = load_dataset_default_macro_batch_size(
            dataset_name, preset.name
        )
        combined_overrides.append(f"macro_batch_size={default_batch_size}")

    command = [
        f"--nproc_per_node={nproc_per_node}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
    ]
    if local_ranks_filter is not None:
        command.append(f"--local-ranks-filter={local_ranks_filter}")
    return [*command, "run.py", *combined_overrides]


def main() -> int:
    parser = build_parser()
    args, extra_overrides = parser.parse_known_args()
    if args.list:
        print_presets()
        return 0
    if args.preset is None:
        parser.error("preset is required unless --list is used")

    preset = PRESETS[args.preset]
    os.chdir(REPO_ROOT)
    command = build_command(
        preset=preset,
        nproc_per_node=args.nproc_per_node,
        master_addr=args.master_addr,
        master_port=args.master_port or preset.master_port,
        local_ranks_filter=args.local_ranks_filter,
        extra_overrides=extra_overrides,
    )
    if args.preset == "graph_mixer":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.dry_run:
        print(shlex.join(["python", "-m", "torch.distributed.run", *command]))
        return 0
    mark_auto_thread_environment()
    run.main(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
