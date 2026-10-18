"""Contract-level tests for semantic host access analysis."""

import torch
import pytest

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

    combined = analyzer.analyze_batch([layout], include_node_features=True)
    assert combined.memory is not None
    assert combined.mailbox is not None
    assert combined.node_features is not None
    assert combined.memory.needs is combined.mailbox.needs
    assert combined.memory.updates is combined.mailbox.updates
    assert combined.memory.needs is combined.node_features
    assert combined.memory.raw_need_counts == [6]
    assert analyzer.can_share_memory_mailbox_planner()


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
    assert not analyzer.can_share_memory_mailbox_planner()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_native_batch_analysis_preserves_all_scopes_and_raw_counts(dtype) -> None:
    layout = SampledLayoutView(
        nid=torch.tensor([5, 5, -1, 7, -1, 7, 8], dtype=dtype),
        n_src=1,
        n_neg=1,
        fanout=[2],
        root_expansion=RootExpansion.POSITIVE_ROOTS,
    )
    state = MemoryState(
        memory_read=NodeSet.ROOTS,
        memory_write=NodeSet.POSITIVE_ROOTS,
        mailbox_read=NodeSet.ALL_SAMPLED,
        mailbox=NeighborhoodMailbox(),
    )
    analyzer = AccessPlanAnalyzer(
        _plan(
            expansion=RootExpansion.POSITIVE_ROOTS,
            feature_scope=NodeSet.ALL_SAMPLED,
            state=state,
        )
    )

    inputs = analyzer.analyze_batch([layout], include_node_features=True)

    assert inputs.memory is not None
    assert inputs.mailbox is not None
    assert inputs.node_features is not None
    assert inputs.memory.needs[0].data.tolist() == [5]
    assert inputs.memory.updates[0].data.tolist() == [5]
    assert inputs.memory.raw_need_counts == [2]
    assert inputs.mailbox.needs[0].data.tolist() == [5, 7, 8]
    assert inputs.mailbox.updates[0].data.tolist() == [7, 8]
    assert inputs.mailbox.raw_need_counts == [5]
    assert inputs.node_features[0].data.tolist() == [5, 7, 8]
