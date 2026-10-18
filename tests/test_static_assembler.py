import pytest
import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.n_feat_nid import NodeFeatNid, NodeFeatureLookup
from frost.datatypes.plan import StaticPlan
from frost.runtime.worker._static_assembler import StaticAssembler
from frost.utils.tensor_set import TensorSet


def _make_flat_graph() -> FlatGraph:
    return FlatGraph(
        nid=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        eid=torch.full((6,), -1, dtype=torch.long),
        ts=torch.zeros(6, dtype=torch.float32),
    )


def test_fill_nfeat_strict_mode_rejects_missing_nodes():
    assembler = StaticAssembler(
        shared_nfeat=torch.zeros((8, 2), dtype=torch.float32),
        shared_efeat=None,
        device=torch.device("cpu"),
    )
    flat_graph = _make_flat_graph()
    nfeat_lookup = NodeFeatureLookup(
        fetched=NodeFeatNid(
            nid=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            nfeat=torch.tensor(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                dtype=torch.float32,
            ),
            is_sorted=True,
        ),
        cached=None,
    )

    with pytest.raises(ValueError, match="missing in the node feature lookup"):
        assembler.fill_nfeat(
            flat_graph=flat_graph,
            nfeat_lookup=nfeat_lookup,
            strict=True,
        )


def test_fill_nfeat_non_strict_keeps_unplanned_rows_zero():
    assembler = StaticAssembler(
        shared_nfeat=torch.zeros((8, 2), dtype=torch.float32),
        shared_efeat=None,
        device=torch.device("cpu"),
    )
    flat_graph = _make_flat_graph()
    nfeat_lookup = NodeFeatureLookup(
        fetched=NodeFeatNid(
            nid=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            nfeat=torch.tensor(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                dtype=torch.float32,
            ),
            is_sorted=True,
        ),
        cached=None,
    )

    out = assembler.fill_nfeat(
        flat_graph=flat_graph,
        nfeat_lookup=nfeat_lookup,
        strict=False,
    )

    assert out.nfeat is not None
    assert nfeat_lookup.fetched is not None
    assert torch.equal(out.nfeat[:4], nfeat_lookup.fetched.nfeat)
    assert torch.equal(out.nfeat[4:], torch.zeros((2, 2), dtype=torch.float32))


def test_fill_nfeat_non_strict_accepts_no_matching_rows():
    assembler = StaticAssembler(
        shared_nfeat=torch.zeros((8, 2), dtype=torch.float32),
        shared_efeat=None,
        device=torch.device("cpu"),
    )
    flat_graph = _make_flat_graph()
    nfeat_lookup = NodeFeatureLookup(
        fetched=NodeFeatNid(
            nid=torch.tensor([6, 7], dtype=torch.long),
            nfeat=torch.tensor([[7.0, 7.0], [8.0, 8.0]], dtype=torch.float32),
            is_sorted=True,
        ),
        cached=None,
    )

    out = assembler.fill_nfeat(
        flat_graph=flat_graph,
        nfeat_lookup=nfeat_lookup,
        strict=False,
    )

    assert out.nfeat is not None
    assert torch.equal(out.nfeat, torch.zeros((6, 2), dtype=torch.float32))


def test_node_feat_merge_still_rejects_overlap_on_cpu():
    left = NodeFeatNid(
        nid=torch.tensor([1, 2]),
        nfeat=torch.ones((2, 2)),
        is_sorted=True,
    )
    right = NodeFeatNid(
        nid=torch.tensor([2, 3]),
        nfeat=torch.ones((2, 2)),
        is_sorted=True,
    )

    with pytest.raises(ValueError, match="must not overlap"):
        left.merge(right)


def test_prepare_nfeat_cache_casts_compact_storage_to_float32():
    assembler = StaticAssembler(
        shared_nfeat=torch.ones((4, 2), dtype=torch.float16),
        shared_efeat=None,
        device=torch.device("cpu"),
    )
    plan = type(
        "Plan",
        (),
        {
            "to_cache": type("Ids", (), {"data": torch.tensor([0, 1])})(),
            "fetch": type("Ids", (), {"data": torch.tensor([0, 1])})(),
        },
    )()

    result = assembler.prepare_nfeat_cache(plan)

    assert result.fetched is not None
    assert result.fetched.nfeat.dtype == torch.float32


def test_prepare_nfeat_lookup_casts_staged_rows_after_same_dtype_copy():
    assembler = StaticAssembler(
        shared_nfeat=torch.ones((4, 2), dtype=torch.float16),
        shared_efeat=None,
        device=torch.device("cpu"),
    )
    plan = StaticPlan(
        fetch=TensorSet(torch.tensor([1, 3])),
        to_cache=TensorSet(torch.tensor([], dtype=torch.long)),
    )
    staged = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)

    result = assembler.prepare_nfeat_lookup(plan, staged_nfeat=staged)

    assert result.fetched is not None
    assert result.fetched.nfeat.dtype == torch.float32
    torch.testing.assert_close(result.fetched.nfeat, staged.float())


def test_prepare_nfeat_cache_keeps_fetched_and_cached_sources_separate():
    shared_nfeat = torch.tensor(
        [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]]
    )
    assembler = StaticAssembler(
        shared_nfeat=shared_nfeat,
        shared_efeat=None,
        device=torch.device("cpu"),
    )

    first = assembler.prepare_nfeat_cache(
        StaticPlan(
            fetch=TensorSet(torch.tensor([0, 1, 2, 3])),
            to_cache=TensorSet(torch.tensor([1, 3])),
        )
    )
    assert first.cached is None
    assert assembler.nfeat_delta is not None
    assert assembler.nfeat_delta.nid.tolist() == [1, 3]

    second = assembler.prepare_nfeat_cache(
        StaticPlan(
            fetch=TensorSet(torch.tensor([2, 4])),
            to_cache=TensorSet(torch.tensor([2, 3])),
        )
    )

    assert second.fetched is not None
    assert second.cached is not None
    assert second.fetched.nid.tolist() == [2, 4]
    assert second.cached.nid.tolist() == [1, 3]
    assert assembler.nfeat_delta is not None
    assert assembler.nfeat_delta.nid.tolist() == [2, 3]
    assert torch.equal(assembler.nfeat_delta.nfeat, shared_nfeat[[2, 3]])

    graph = FlatGraph(
        nid=torch.tensor([4, 1, 3, 2, -1]),
        eid=torch.full((5,), -1),
        ts=torch.zeros(5),
    )
    result = assembler.fill_nfeat(graph, second, strict=True)
    assert result.nfeat is not None
    assert torch.equal(result.nfeat[:4], shared_nfeat[[4, 1, 3, 2]])
    assert torch.equal(result.nfeat[4], torch.zeros(2))


def test_nfeat_cache_update_can_be_deferred_until_after_lookup_fill():
    shared_nfeat = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    assembler = StaticAssembler(
        shared_nfeat=shared_nfeat,
        shared_efeat=None,
        device=torch.device("cpu"),
    )
    plan = StaticPlan(
        fetch=TensorSet(torch.tensor([0, 1, 2, 3])),
        to_cache=TensorSet(torch.tensor([1, 3])),
    )

    lookup = assembler.prepare_nfeat_lookup(plan)
    assert assembler.nfeat_delta is None

    graph = FlatGraph(
        nid=torch.tensor([3, 0, 2, 1]),
        eid=torch.full((4,), -1),
        ts=torch.zeros(4),
    )
    result = assembler.fill_nfeat(graph, lookup)
    assert result.nfeat is not None
    assert torch.equal(result.nfeat, shared_nfeat[[3, 0, 2, 1]])

    assembler.update_nfeat_cache(lookup, plan.to_cache.data)
    assert assembler.nfeat_delta is not None
    assert assembler.nfeat_delta.nid.tolist() == [1, 3]
    assert torch.equal(assembler.nfeat_delta.nfeat, shared_nfeat[[1, 3]])


def test_nfeat_cache_can_retain_a_row_unused_by_the_current_batch():
    shared_nfeat = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    assembler = StaticAssembler(
        shared_nfeat=shared_nfeat,
        shared_efeat=None,
        device=torch.device("cpu"),
    )
    first = assembler.prepare_nfeat_cache(
        StaticPlan(
            fetch=TensorSet(torch.tensor([1, 2])),
            to_cache=TensorSet(torch.tensor([1, 2])),
        )
    )
    assert first.fetched is not None

    second_plan = StaticPlan(
        fetch=TensorSet(torch.tensor([3])),
        to_cache=TensorSet(torch.tensor([1, 3])),
    )
    second = assembler.prepare_nfeat_lookup(second_plan)
    current = FlatGraph(
        nid=torch.tensor([2, 3]),
        eid=torch.full((2,), -1),
        ts=torch.zeros(2),
    )
    assembler.fill_nfeat(current, second)
    assembler.update_nfeat_cache(second, second_plan.to_cache.data)

    assert assembler.nfeat_delta is not None
    assert assembler.nfeat_delta.nid.tolist() == [1, 3]
    assert torch.equal(assembler.nfeat_delta.nfeat, shared_nfeat[[1, 3]])


def test_fill_efeat_casts_compact_storage_to_float32():
    assembler = StaticAssembler(
        shared_nfeat=None,
        shared_efeat=torch.ones((3, 2), dtype=torch.bool),
        device=torch.device("cpu"),
    )
    graph = FlatGraph(
        nid=torch.arange(4),
        eid=torch.tensor([1, -1, -1, 2]),
        ts=torch.zeros(4),
    )

    result = assembler.fill_efeat(graph, n_src=1)

    assert result.efeat is not None
    assert result.efeat.dtype == torch.float32
