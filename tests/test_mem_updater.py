import torch

from frost.datatypes.graph import FlatGraph
from frost.models.mem_updater import TransformerMemoryUpdater


def test_transformer_memory_updater_only_updates_root_prefix():
    updater = TransformerMemoryUpdater(
        d_mail=3,
        d_mem=4,
        d_time=0,
        boxsize=2,
        heads=2,
        dropout=0.0,
    )

    original_mem = torch.arange(16, dtype=torch.float32).view(4, 4)
    flat_graph = FlatGraph(
        nid=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        eid=torch.full((4,), -1, dtype=torch.long),
        ts=torch.tensor([10.0, 11.0, 12.0, 13.0]),
        mem=original_mem.clone(),
        mem_ts=torch.zeros(4, dtype=torch.float32),
        box=torch.arange(24, dtype=torch.float32).view(4, 2, 3),
        box_ts=torch.zeros(4, 2, dtype=torch.float32),
    )

    updated = updater(flat_graph, n_target_nodes=2, n_root_nodes=3)

    assert updated.mem is not None
    assert updated.detached_mem is not None
    assert updated.detached_ts is not None
    assert updated.detached_nid is not None
    assert not torch.equal(updated.mem[:3], original_mem[:3])
    assert torch.equal(updated.mem[3], original_mem[3])
    assert torch.equal(updated.detached_mem, updated.mem[:2])
    assert torch.equal(updated.detached_ts, flat_graph.ts[:2])
    assert torch.equal(updated.detached_nid, flat_graph.nid[:2])
