from __future__ import annotations

from dataclasses import dataclass


DYNAMIC_STATE_ABLATION_BASELINE = "baseline"
DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_STEP_RAW = (
    "no_plan_no_delta_post_step_raw"
)
DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_MEM_UPDATE_RAW = (
    "no_plan_no_delta_post_mem_update_raw"
)
DYNAMIC_STATE_ABLATION_PLAN_NO_DELTA_DEDUP_POST_MEM_UPDATE_RAW = (
    "plan_no_delta_dedup_post_mem_update_raw"
)

VALID_DYNAMIC_STATE_ABLATIONS = frozenset(
    {
        DYNAMIC_STATE_ABLATION_BASELINE,
        DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_STEP_RAW,
        DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_MEM_UPDATE_RAW,
        DYNAMIC_STATE_ABLATION_PLAN_NO_DELTA_DEDUP_POST_MEM_UPDATE_RAW,
    }
)


@dataclass(frozen=True)
class DynamicStateAblation:
    name: str

    @property
    def is_baseline(self) -> bool:
        return self.name == DYNAMIC_STATE_ABLATION_BASELINE

    @property
    def uses_async_dynamic_state_pipeline(self) -> bool:
        return self.is_baseline

    @property
    def disable_mem_mail_planner(self) -> bool:
        return self.name in {
            DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_STEP_RAW,
            DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_MEM_UPDATE_RAW,
        }

    @property
    def current_batch_dedup_plan_only(self) -> bool:
        return (
            self.name
            == DYNAMIC_STATE_ABLATION_PLAN_NO_DELTA_DEDUP_POST_MEM_UPDATE_RAW
        )

    @property
    def disable_p2p_delta(self) -> bool:
        return not self.is_baseline

    @property
    def direct_cpu_read_all_positions(self) -> bool:
        return self.disable_mem_mail_planner

    @property
    def direct_cpu_writeback_after_optimizer_step(self) -> bool:
        return (
            self.name == DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_STEP_RAW
        )

    @property
    def direct_cpu_writeback_after_mem_update(self) -> bool:
        return self.name in {
            DYNAMIC_STATE_ABLATION_NO_PLAN_NO_DELTA_POST_MEM_UPDATE_RAW,
            DYNAMIC_STATE_ABLATION_PLAN_NO_DELTA_DEDUP_POST_MEM_UPDATE_RAW,
        }

    @property
    def uses_direct_cpu_writeback(self) -> bool:
        return (
            self.direct_cpu_writeback_after_mem_update
            or self.direct_cpu_writeback_after_optimizer_step
        )

    @property
    def filter_writeback_by_plan(self) -> bool:
        return (
            self.name
            == DYNAMIC_STATE_ABLATION_PLAN_NO_DELTA_DEDUP_POST_MEM_UPDATE_RAW
        )

    @property
    def requires_tgn_boxsize_one_only(self) -> bool:
        return not self.is_baseline

    @property
    def max_in_flight_batches(self) -> int | None:
        if self.is_baseline:
            return None
        return 1


def resolve_dynamic_state_ablation(raw: object) -> DynamicStateAblation:
    if raw is None:
        return DynamicStateAblation(DYNAMIC_STATE_ABLATION_BASELINE)
    name = str(raw).strip()
    if name not in VALID_DYNAMIC_STATE_ABLATIONS:
        valid = ", ".join(sorted(VALID_DYNAMIC_STATE_ABLATIONS))
        raise ValueError(
            f"Unsupported dynamic_state_ablation={name!r}. Expected one of: {valid}."
        )
    return DynamicStateAblation(name=name)
