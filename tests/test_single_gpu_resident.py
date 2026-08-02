from __future__ import annotations

import pytest
import torch

from frost.datatypes.graph import FlatGraph
from frost.launch.single_gpu_resident import (
    estimate_resident_storage_bytes,
    resolve_single_gpu_resident,
)
from frost.models.contracts import NodeSet, RootExpansion
from frost.runtime.worker._static_assembler import StaticAssembler


def test_resident_storage_estimate_counts_features_and_dynamic_state() -> None:
    nfeat = torch.empty((5, 3), dtype=torch.float32)
    efeat = torch.empty((7, 2), dtype=torch.float32)
    estimated = estimate_resident_storage_bytes(
        node_features=nfeat,
        edge_features=efeat,
        n_nodes=5,
        boxsize=1,
        d_mem=4,
        d_mail=10,
        state_dtype=torch.float32,
        timestamp_dtype=torch.float32,
    )
    expected_dynamic = 5 * (4 * 4 + 4 + 10 * 4 + 4 + 8)
    assert estimated == nfeat.numel() * 4 + efeat.numel() * 4 + expected_dynamic


def test_resident_auto_falls_back_and_on_rejects_an_oversized_store() -> None:
    auto = resolve_single_gpu_resident(
        "auto",
        n_gpu=1,
        required_bytes=80,
        total_device_bytes=100,
        memory_fraction=0.7,
    )
    assert not auto.enabled
    assert "above" in auto.reason

    with pytest.raises(ValueError, match="cannot be enabled"):
        resolve_single_gpu_resident(
            "on",
            n_gpu=1,
            required_bytes=80,
            total_device_bytes=100,
            memory_fraction=0.7,
        )


def test_resident_node_features_obey_root_scope() -> None:
    features = torch.arange(30, dtype=torch.float32).view(10, 3)
    graph = FlatGraph(
        nid=torch.tensor([1, 2, 3, 4, 5, 6]),
        eid=torch.full((6,), -1),
        ts=torch.zeros(6),
    )
    assembler = StaticAssembler(features, None, torch.device("cpu"))
    assembler.fill_resident_nfeat(
        graph,
        n_src=1,
        n_neg=1,
        fanout=[1],
        root_expansion=RootExpansion.ALL_ROOTS,
        scope=NodeSet.ROOTS,
    )
    assert graph.nfeat is not None
    assert torch.equal(graph.nfeat[:3], features[torch.tensor([1, 2, 3])])
    assert torch.count_nonzero(graph.nfeat[3:]) == 0
