import pytest

from frost.runtime.experimental.dynamic_state import (
    PlannerMode,
    TransportMode,
    WritebackTiming,
    resolve_dynamic_state_policy,
    single_gpu_resident_dynamic_state_policy,
)


@pytest.mark.parametrize(
    (
        "name",
        "planner",
        "transport",
        "writeback",
        "filter_by_plan",
        "pin_gather_buffers",
    ),
    [
        (
            "baseline",
            PlannerMode.PIPELINED,
            TransportMode.ASYNC_P2P,
            WritebackTiming.AFTER_MEMORY_UPDATE,
            False,
            False,
        ),
        (
            "baseline_pinned_gather",
            PlannerMode.PIPELINED,
            TransportMode.ASYNC_P2P,
            WritebackTiming.AFTER_MEMORY_UPDATE,
            False,
            True,
        ),
        (
            "no_plan_no_delta_post_step_raw",
            PlannerMode.NONE,
            TransportMode.DIRECT_CPU,
            WritebackTiming.AFTER_OPTIMIZER_STEP,
            False,
            False,
        ),
        (
            "no_plan_no_delta_post_mem_update_raw",
            PlannerMode.NONE,
            TransportMode.DIRECT_CPU,
            WritebackTiming.AFTER_MEMORY_UPDATE,
            False,
            False,
        ),
        (
            "plan_no_delta_dedup_post_mem_update_raw",
            PlannerMode.CURRENT_BATCH,
            TransportMode.DIRECT_CPU,
            WritebackTiming.AFTER_MEMORY_UPDATE,
            True,
            False,
        ),
    ],
)
def test_resolve_dynamic_state_policy(
    name,
    planner,
    transport,
    writeback,
    filter_by_plan,
    pin_gather_buffers,
) -> None:
    policy = resolve_dynamic_state_policy(name)

    assert policy.name == name
    assert policy.planner is planner
    assert policy.transport is transport
    assert policy.writeback is writeback
    assert policy.filter_writeback_by_plan is filter_by_plan
    assert policy.pin_gather_buffers is pin_gather_buffers


def test_resolve_dynamic_state_policy_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported dynamic_state_ablation"):
        resolve_dynamic_state_policy("unknown")


def test_single_gpu_resident_policy_is_internal_gpu_local_path() -> None:
    policy = single_gpu_resident_dynamic_state_policy()
    assert policy.name == "single_gpu_resident"
    assert policy.planner is PlannerMode.NONE
    assert policy.transport is TransportMode.GPU_LOCAL
    assert policy.writeback is WritebackTiming.AFTER_MEMORY_UPDATE
