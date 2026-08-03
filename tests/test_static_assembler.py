import pytest
import torch

from frost.datatypes.graph import FlatGraph
from frost.datatypes.n_feat_nid import NodeFeatNid
from frost.runtime.worker._static_assembler import StaticAssembler


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
    nfeat_nid = NodeFeatNid(
        nid=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        nfeat=torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            dtype=torch.float32,
        ),
        is_sorted=True,
    )

    with pytest.raises(ValueError, match="missing in nfeat_nid.nid"):
        assembler.fill_nfeat(
            flat_graph=flat_graph,
            nfeat_nid=nfeat_nid,
            strict=True,
        )


def test_fill_nfeat_non_strict_keeps_unplanned_rows_zero():
    assembler = StaticAssembler(
        shared_nfeat=torch.zeros((8, 2), dtype=torch.float32),
        shared_efeat=None,
        device=torch.device("cpu"),
    )
    flat_graph = _make_flat_graph()
    nfeat_nid = NodeFeatNid(
        nid=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        nfeat=torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            dtype=torch.float32,
        ),
        is_sorted=True,
    )

    out = assembler.fill_nfeat(
        flat_graph=flat_graph,
        nfeat_nid=nfeat_nid,
        strict=False,
    )

    assert out.nfeat is not None
    assert torch.equal(out.nfeat[:4], nfeat_nid.nfeat)
    assert torch.equal(out.nfeat[4:], torch.zeros((2, 2), dtype=torch.float32))


def test_fill_nfeat_non_strict_accepts_no_matching_rows():
    assembler = StaticAssembler(
        shared_nfeat=torch.zeros((8, 2), dtype=torch.float32),
        shared_efeat=None,
        device=torch.device("cpu"),
    )
    flat_graph = _make_flat_graph()
    nfeat_nid = NodeFeatNid(
        nid=torch.tensor([6, 7], dtype=torch.long),
        nfeat=torch.tensor([[7.0, 7.0], [8.0, 8.0]], dtype=torch.float32),
        is_sorted=True,
    )

    out = assembler.fill_nfeat(
        flat_graph=flat_graph,
        nfeat_nid=nfeat_nid,
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

    assert result.nfeat.dtype == torch.float32


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
