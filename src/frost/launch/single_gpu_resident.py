"""Resolution and sizing for the single-GPU resident runtime backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class SingleGpuResidentMode(StrEnum):
    OFF = "off"
    AUTO = "auto"
    ON = "on"


@dataclass(frozen=True)
class SingleGpuResidentDecision:
    enabled: bool
    mode: SingleGpuResidentMode
    required_bytes: int
    budget_bytes: int
    reason: str


def parse_single_gpu_resident_mode(raw_mode: object) -> SingleGpuResidentMode:
    try:
        return SingleGpuResidentMode(str(raw_mode).strip().lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in SingleGpuResidentMode)
        raise ValueError(
            f"runtime.single_gpu_resident must be one of: {valid}."
        ) from exc


def tensor_storage_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def estimate_resident_storage_bytes(
    *,
    node_features: torch.Tensor | None,
    edge_features: torch.Tensor | None,
    n_nodes: int,
    boxsize: int,
    d_mem: int,
    d_mail: int,
    state_dtype: torch.dtype,
    timestamp_dtype: torch.dtype,
) -> int:
    """Estimate persistent tensors only; activations/model use the remainder."""
    state_bytes = torch.empty((), dtype=state_dtype).element_size()
    timestamp_bytes = torch.empty((), dtype=timestamp_dtype).element_size()
    dynamic_bytes = 0
    if d_mem > 0:
        dynamic_bytes = n_nodes * (
            d_mem * state_bytes
            + timestamp_bytes
            + boxsize * d_mail * state_bytes
            + boxsize * timestamp_bytes
            + torch.empty((), dtype=torch.int64).element_size()
        )
    return (
        tensor_storage_bytes(node_features)
        + tensor_storage_bytes(edge_features)
        + dynamic_bytes
    )


def resolve_single_gpu_resident(
    raw_mode: object,
    *,
    n_gpu: int,
    required_bytes: int,
    total_device_bytes: int,
    memory_fraction: float = 0.7,
) -> SingleGpuResidentDecision:
    """Resolve ``off|auto|on`` without allocating CUDA memory."""
    mode = parse_single_gpu_resident_mode(raw_mode)
    if not 0.0 < memory_fraction < 1.0:
        raise ValueError(
            "runtime.single_gpu_resident_memory_fraction must be between 0 and 1."
        )

    budget_bytes = int(total_device_bytes * memory_fraction)
    if mode is SingleGpuResidentMode.OFF:
        return SingleGpuResidentDecision(
            enabled=False,
            mode=mode,
            required_bytes=required_bytes,
            budget_bytes=budget_bytes,
            reason="disabled by configuration",
        )
    if n_gpu != 1:
        reason = f"requires exactly one GPU worker, got {n_gpu}"
        if mode is SingleGpuResidentMode.ON:
            raise ValueError(f"runtime.single_gpu_resident=on {reason}.")
        return SingleGpuResidentDecision(
            enabled=False,
            mode=mode,
            required_bytes=required_bytes,
            budget_bytes=budget_bytes,
            reason=reason,
        )
    if required_bytes > budget_bytes:
        reason = (
            f"persistent storage needs {required_bytes / 2**30:.2f} GiB, "
            f"above the {budget_bytes / 2**30:.2f} GiB resident budget"
        )
        if mode is SingleGpuResidentMode.ON:
            raise ValueError(f"runtime.single_gpu_resident=on cannot be enabled: {reason}.")
        return SingleGpuResidentDecision(
            enabled=False,
            mode=mode,
            required_bytes=required_bytes,
            budget_bytes=budget_bytes,
            reason=reason,
        )
    return SingleGpuResidentDecision(
        enabled=True,
        mode=mode,
        required_bytes=required_bytes,
        budget_bytes=budget_bytes,
        reason="single-worker persistent storage fits the configured GPU budget",
    )
