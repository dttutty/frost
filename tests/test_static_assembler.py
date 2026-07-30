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
