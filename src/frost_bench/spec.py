from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping


class BatchSizeScope(StrEnum):
    """Meaning of ``batch_size`` for a benchmark backend."""

    GLOBAL = "global"
    PER_GPU = "per_gpu"
    PER_PROCESS = "per_process"


@dataclass(frozen=True)
class ExperimentSpec:
    model: str
    dataset: str
    seed: int
    n_gpus: int
    batch_size: int
    batch_size_scope: BatchSizeScope = BatchSizeScope.GLOBAL
    epochs: int = 1
    compile: bool = False
    tf32: bool = False
    single_gpu_resident: bool = False
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_gpus < 1:
            raise ValueError("n_gpus must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.single_gpu_resident and self.n_gpus != 1:
            raise ValueError("single_gpu_resident requires n_gpus=1")

    @property
    def global_batch_size(self) -> int:
        if self.batch_size_scope is BatchSizeScope.GLOBAL:
            return self.batch_size
        return self.batch_size * self.n_gpus

    @property
    def per_gpu_batch_size(self) -> int:
        if self.batch_size_scope is BatchSizeScope.GLOBAL:
            if self.batch_size % self.n_gpus != 0:
                raise ValueError(
                    f"global batch_size ({self.batch_size}) must be divisible by "
                    f"n_gpus ({self.n_gpus})"
                )
            return self.batch_size // self.n_gpus
        return self.batch_size


@dataclass(frozen=True)
class PreparedRun:
    system: str
    command: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = field(default_factory=dict)
    cleanup_paths: tuple[Path, ...] = ()

    def command_list(self) -> list[str]:
        return list(self.command)
