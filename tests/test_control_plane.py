from __future__ import annotations

import numpy as np
import pytest
import torch

from frost.datatypes.meta import Meta
from frost.datatypes.plan import DynamicPlan, PlanSet, StaticPlan
from frost.runtime.control_plane import (
    CONTROL_HEADER_SIZE,
    ControlMessageKind,
    decode_meta,
    encode_meta,
    make_control_header,
    parse_control_header,
)
from frost.shm.control_plane import control_payload_capacity
from frost.utils.tensor_set import TensorSet


def _set(*nodes: int) -> TensorSet:
    return TensorSet(
        torch.tensor(nodes, dtype=torch.int64),
        sort_now=False,
        ensure_unique=False,
    )


def _dynamic(offset: int) -> DynamicPlan:
    return DynamicPlan(
        p2p=[_set(offset + 1, offset + 2), _set(offset + 3)],
        fetch=_set(offset + 4, offset + 5),
        writeback=_set(offset + 6),
        recv_counts=np.array([2, 1], dtype=np.int32),
        send_counts=np.array([2, 1], dtype=np.int32),
    )


def test_meta_tensor_codec_round_trip() -> None:
    original = Meta(
        split="test",
        n_src=3,
        mini_batch_idx=17,
        macro_batch_idx=8,
        epoch_idx=3,
        flat_graph=None,
        buf_len=27,
        n_neg=2,
        is_last_macro_of_epoch=True,
        is_last_macro_of_split=True,
        neg_valid_mask=torch.tensor([True, False, True, True, False, False]),
        plan_set=PlanSet(
            mem_plan=_dynamic(0),
            box_plan=_dynamic(10),
            nfeat_plan=StaticPlan(to_cache=_set(21, 22), fetch=_set(21, 22, 23)),
        ),
    )

    payload = encode_meta(original, n_gpus=2)
    decoded = decode_meta(payload, n_gpus=2)

    assert payload.dtype == torch.int64
    assert payload.is_contiguous()
    assert decoded.split == original.split
    assert decoded.n_src == original.n_src
    assert decoded.mini_batch_idx == original.mini_batch_idx
    assert decoded.macro_batch_idx == original.macro_batch_idx
    assert decoded.epoch_idx == original.epoch_idx
    assert decoded.buf_len == original.buf_len
    assert decoded.n_neg == original.n_neg
    assert decoded.is_last_macro_of_epoch
    assert decoded.is_last_macro_of_split
    assert torch.equal(decoded.neg_valid_mask, original.neg_valid_mask)
    assert decoded.plan_set is not None
    for decoded_plan, original_plan in (
        (decoded.plan_set.mem_plan, original.plan_set.mem_plan),
        (decoded.plan_set.box_plan, original.plan_set.box_plan),
    ):
        assert decoded_plan is not None
        assert original_plan is not None
        assert np.array_equal(decoded_plan.recv_counts, original_plan.recv_counts)
        assert np.array_equal(decoded_plan.send_counts, original_plan.send_counts)
        assert torch.equal(decoded_plan.fetch.data, original_plan.fetch.data)
        assert decoded_plan.writeback is not None
        assert original_plan.writeback is not None
        assert torch.equal(decoded_plan.writeback.data, original_plan.writeback.data)
        assert [nodes.data.tolist() for nodes in decoded_plan.p2p] == [
            nodes.data.tolist() for nodes in original_plan.p2p
        ]
    assert decoded.plan_set.nfeat_plan is not None
    assert torch.equal(
        decoded.plan_set.nfeat_plan.fetch.data,
        original.plan_set.nfeat_plan.fetch.data,
    )


def test_meta_tensor_codec_handles_absent_plans_and_mask() -> None:
    original = Meta(
        split="train",
        n_src=2,
        mini_batch_idx=0,
        flat_graph=None,
        buf_len=6,
        n_neg=1,
    )

    decoded = decode_meta(encode_meta(original, n_gpus=4), n_gpus=4)

    assert decoded.neg_valid_mask is None
    assert decoded.plan_set == PlanSet()
    assert decoded.macro_batch_idx == -1
    assert decoded.epoch_idx == 0
    assert decoded.is_last_macro_of_split is False


def test_meta_tensor_codec_preserves_shared_dynamic_plan() -> None:
    shared_plan = _dynamic(0)
    shared = Meta(
        split="train",
        n_src=2,
        mini_batch_idx=0,
        flat_graph=None,
        buf_len=6,
        n_neg=1,
        plan_set=PlanSet(mem_plan=shared_plan, box_plan=shared_plan),
    )
    distinct = Meta(
        split="train",
        n_src=2,
        mini_batch_idx=0,
        flat_graph=None,
        buf_len=6,
        n_neg=1,
        plan_set=PlanSet(mem_plan=shared_plan, box_plan=_dynamic(0)),
    )

    shared_payload = encode_meta(shared, n_gpus=2)
    decoded = decode_meta(shared_payload, n_gpus=2)

    assert decoded.plan_set is not None
    assert decoded.plan_set.mem_plan is decoded.plan_set.box_plan
    assert shared_payload.numel() < encode_meta(distinct, n_gpus=2).numel()


@pytest.mark.parametrize("kind", list(ControlMessageKind))
def test_control_header_round_trip(kind: ControlMessageKind) -> None:
    payload_length = 13 if kind is ControlMessageKind.BATCH else 0
    header = make_control_header(kind, payload_length)

    assert header.shape == (CONTROL_HEADER_SIZE,)
    assert parse_control_header(header) == (kind, payload_length)


def test_sentinel_header_rejects_payload() -> None:
    header = torch.tensor(
        [int(ControlMessageKind.SHUTDOWN), 1], dtype=torch.int64
    )
    with pytest.raises(ValueError, match="cannot carry"):
        parse_control_header(header)


def test_shared_control_capacity_covers_plans_and_mask() -> None:
    assert control_payload_capacity(
        n_gpus=4, node_capacity=100, mask_capacity=300
    ) == 1745
