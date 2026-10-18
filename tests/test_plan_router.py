"""Tests for contract-driven host planner routing."""

import threading
from queue import Queue

import torch

from frost.datatypes.meta import Meta
from frost.datatypes.sampled_layout import SampledLayoutView
from frost.launch.plan import ExecutionPlan, SamplingPlan
from frost.models.contracts import (
    InteractionMailbox,
    MemoryState,
    NodeSet,
    RepresentationKind,
    RootExpansion,
)
from frost.runtime.experimental.dynamic_state import resolve_dynamic_state_policy
from frost.runtime.host import PlanRouter
from frost.runtime.host.analyzer import AccessPlanAnalyzer


def _meta(index: int) -> Meta:
    return Meta(
        split="train",
        n_src=1,
        mini_batch_idx=index,
        flat_graph=None,
        buf_len=6,
        n_neg=1,
    )


def test_identical_memory_mailbox_contract_uses_one_planner() -> None:
    state = MemoryState(
        memory_read=NodeSet.ALL_SAMPLED,
        memory_write=NodeSet.POSITIVE_ROOTS,
        mailbox_read=NodeSet.ALL_SAMPLED,
        mailbox=InteractionMailbox(),
    )
    plan = ExecutionPlan(
        model_name="test",
        embedding_dim=8,
        sampling=SamplingPlan((1,), RootExpansion.ALL_ROOTS),
        node_feature_scope=NodeSet.ALL_SAMPLED,
        dynamic_state=state,
        representation=RepresentationKind.NODE_ALIGNED,
    )
    layouts = [
        SampledLayoutView(
            nid=torch.tensor([rank, rank + 1, rank + 2, 10, 11, 12]),
            n_src=1,
            n_neg=1,
            fanout=[1],
        )
        for rank in range(2)
    ]
    mem_out = Queue()
    mail_out = Queue()
    router = PlanRouter(
        gpu_topo=[0, 1],
        analyzer=AccessPlanAnalyzer(plan),
        mem_plan_out_q=mem_out,
        mail_plan_out_q=mail_out,
        nfeat_plan_out_q=None,
        stop_ev=threading.Event(),
        dynamic_state_policy=resolve_dynamic_state_policy(
            "plan_no_delta_dedup_post_mem_update_raw"
        ),
    )

    try:
        router.dispatch([_meta(0), _meta(1)], layouts)
        memory_plans = mem_out.get(timeout=2.0)
        mailbox_plans = mail_out.get(timeout=2.0)
        assert memory_plans is mailbox_plans
        assert router.mem_planner is not None
        assert router.mail_planner is None
    finally:
        router.shutdown()
