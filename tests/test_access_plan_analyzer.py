"""Contract-level tests for semantic host access analysis."""

import torch

from frost.datatypes.sampled_layout import SampledLayoutView
from frost.launch.plan import ExecutionPlan, SamplingPlan
from frost.models.contracts import (
    InteractionMailbox,
    MemoryState,
    NeighborhoodMailbox,
    NodeSet,
    RepresentationKind,
    RootExpansion,
    Stateless,
)
from frost.runtime.host.analyzer import AccessPlanAnalyzer


def _plan(
    *,
    expansion: RootExpansion,
    feature_scope: NodeSet,
    state: Stateless | MemoryState,
) -> ExecutionPlan:
    fanout = (2,)
    return ExecutionPlan(
        model_name="test",
        embedding_dim=8,
        sampling=SamplingPlan(fanout, expansion),
        node_feature_scope=feature_scope,
        dynamic_state=state,
        representation=RepresentationKind.NODE_ALIGNED,
    )


def test_stateless_analyzer_uses_declared_feature_scope() -> None:
    layout = SampledLayoutView(
        nid=torch.tensor([0, 1, 2, 10, -1, 12, 13, 14, 15]),
        n_src=1,
        n_neg=1,
        fanout=[2],
    )
    analyzer = AccessPlanAnalyzer(
        _plan(
            expansion=RootExpansion.ALL_ROOTS,
            feature_scope=NodeSet.ALL_SAMPLED,
            state=Stateless(),
        )
    )

    assert analyzer.get_feat_plan_inputs([layout])[0].data.tolist() == [
        0,
        1,
        2,
        10,
        12,
        13,
        14,
        15,
    ]
    assert analyzer.get_mem_plan_inputs([layout]) == ([], [])
    assert analyzer.get_mail_plan_inputs([layout]) == ([], [])


def test_interaction_mail_scopes_come_from_memory_contract() -> None:
    layout = SampledLayoutView(
        nid=torch.tensor([0, 1, 2, 10, 11, 12]),
        n_src=1,
        n_neg=1,
        fanout=[1],
    )
    state = MemoryState(
        memory_read=NodeSet.ALL_SAMPLED,
        memory_write=NodeSet.POSITIVE_ROOTS,
        mailbox_read=NodeSet.ALL_SAMPLED,
        mailbox=InteractionMailbox(),
    )
    analyzer = AccessPlanAnalyzer(
        _plan(
            expansion=RootExpansion.ALL_ROOTS,
            feature_scope=NodeSet.ALL_SAMPLED,
            state=state,
        )
    )

    mem_need, mem_update = analyzer.get_mem_plan_inputs([layout])
    mail_need, mail_update = analyzer.get_mail_plan_inputs([layout])
    assert mem_need[0].data.tolist() == [0, 1, 2, 10, 11, 12]
    assert mem_update[0].data.tolist() == [0, 1]
    assert mail_need[0].data.tolist() == [0, 1, 2, 10, 11, 12]
    assert mail_update[0].data.tolist() == [0, 1]


def test_neighborhood_mail_selects_valid_positive_neighbors() -> None:
    layout = SampledLayoutView(
        nid=torch.tensor([0, 1, 2, 10, -1, 12, 13]),
        n_src=1,
        n_neg=1,
        fanout=[2],
        root_expansion=RootExpansion.POSITIVE_ROOTS,
    )
    state = MemoryState(
        memory_read=NodeSet.ROOTS,
        memory_write=NodeSet.POSITIVE_ROOTS,
        mailbox_read=NodeSet.ROOTS,
        mailbox=NeighborhoodMailbox(),
    )
    analyzer = AccessPlanAnalyzer(
        _plan(
            expansion=RootExpansion.POSITIVE_ROOTS,
            feature_scope=NodeSet.ROOTS,
            state=state,
        )
    )

    mail_need, mail_update = analyzer.get_mail_plan_inputs([layout])
    assert mail_need[0].data.tolist() == [0, 1, 2]
    assert mail_update[0].data.tolist() == [10, 12, 13]
    assert analyzer.get_feat_plan_inputs([layout])[0].data.tolist() == [0, 1, 2]
