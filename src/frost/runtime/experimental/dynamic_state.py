"""Policy presets for dynamic-state ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PlannerMode(StrEnum):
    PIPELINED = "pipelined"
    CURRENT_BATCH = "current_batch"
    NONE = "none"


class TransportMode(StrEnum):
    ASYNC_P2P = "async_p2p"
    DIRECT_CPU = "direct_cpu"
    GPU_LOCAL = "gpu_local"


class WritebackTiming(StrEnum):
    AFTER_MEMORY_UPDATE = "after_memory_update"
    AFTER_OPTIMIZER_STEP = "after_optimizer_step"


@dataclass(frozen=True)
class DynamicStatePolicy:
    """Resolved runtime behavior; baseline code never interprets preset names."""

    name: str
    planner: PlannerMode
    transport: TransportMode
    writeback: WritebackTiming
    filter_writeback_by_plan: bool = False
    pin_gather_buffers: bool = False

    @property
    def max_in_flight_batches(self) -> int | None:
        return None if self.transport is TransportMode.ASYNC_P2P else 1


_POLICIES = {
    "baseline": DynamicStatePolicy(
        name="baseline",
        planner=PlannerMode.PIPELINED,
        transport=TransportMode.ASYNC_P2P,
        writeback=WritebackTiming.AFTER_MEMORY_UPDATE,
    ),
    "baseline_pinned_gather": DynamicStatePolicy(
        name="baseline_pinned_gather",
        planner=PlannerMode.PIPELINED,
        transport=TransportMode.ASYNC_P2P,
        writeback=WritebackTiming.AFTER_MEMORY_UPDATE,
        pin_gather_buffers=True,
    ),
    "no_plan_no_delta_post_step_raw": DynamicStatePolicy(
        name="no_plan_no_delta_post_step_raw",
        planner=PlannerMode.NONE,
        transport=TransportMode.DIRECT_CPU,
        writeback=WritebackTiming.AFTER_OPTIMIZER_STEP,
    ),
    "no_plan_no_delta_post_mem_update_raw": DynamicStatePolicy(
        name="no_plan_no_delta_post_mem_update_raw",
        planner=PlannerMode.NONE,
        transport=TransportMode.DIRECT_CPU,
        writeback=WritebackTiming.AFTER_MEMORY_UPDATE,
    ),
    "plan_no_delta_dedup_post_mem_update_raw": DynamicStatePolicy(
        name="plan_no_delta_dedup_post_mem_update_raw",
        planner=PlannerMode.CURRENT_BATCH,
        transport=TransportMode.DIRECT_CPU,
        writeback=WritebackTiming.AFTER_MEMORY_UPDATE,
        filter_writeback_by_plan=True,
    ),
}


def single_gpu_resident_dynamic_state_policy() -> DynamicStatePolicy:
    """Return the internal policy used by the GPU-resident worker backend."""
    return DynamicStatePolicy(
        name="single_gpu_resident",
        planner=PlannerMode.NONE,
        transport=TransportMode.GPU_LOCAL,
        writeback=WritebackTiming.AFTER_MEMORY_UPDATE,
    )


def resolve_dynamic_state_policy(raw: object) -> DynamicStatePolicy:
    name = "baseline" if raw is None else str(raw).strip()
    try:
        return _POLICIES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(_POLICIES))
        raise ValueError(
            f"Unsupported dynamic_state_ablation={name!r}. Expected one of: {valid}."
        ) from exc
