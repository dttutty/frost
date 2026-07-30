import pytest

from frost.runtime.experimental.dynamic_state import (
    PlannerMode,
    TransportMode,
    WritebackTiming,
    resolve_dynamic_state_policy,
)


@pytest.mark.parametrize(
    ("name", "planner", "transport", "writeback", "filter_by_plan"),
    [
        (
            "baseline",
            PlannerMode.PIPELINED,
            TransportMode.ASYNC_P2P,
            WritebackTiming.AFTER_MEMORY_UPDATE,
            False,
        ),
        (
            "no_plan_no_delta_post_step_raw",
            PlannerMode.NONE,
            TransportMode.DIRECT_CPU,
            WritebackTiming.AFTER_OPTIMIZER_STEP,
            False,
        ),
        (
            "no_plan_no_delta_post_mem_update_raw",
            PlannerMode.NONE,
            TransportMode.DIRECT_CPU,
            WritebackTiming.AFTER_MEMORY_UPDATE,
            False,
        ),
        (
            "plan_no_delta_dedup_post_mem_update_raw",
            PlannerMode.CURRENT_BATCH,
            TransportMode.DIRECT_CPU,
            WritebackTiming.AFTER_MEMORY_UPDATE,
            True,
        ),
    ],
)
def test_resolve_dynamic_state_policy(
    name,
    planner,
    transport,
    writeback,
    filter_by_plan,
) -> None:
    policy = resolve_dynamic_state_policy(name)

    assert policy.name == name
    assert policy.planner is planner
    assert policy.transport is transport
    assert policy.writeback is writeback
    assert policy.filter_writeback_by_plan is filter_by_plan


def test_resolve_dynamic_state_policy_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported dynamic_state_ablation"):
        resolve_dynamic_state_policy("unknown")
