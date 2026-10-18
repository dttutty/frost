"""Tensor codec for host-to-worker batch control messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, cast

import numpy as np
import torch

from frost.datatypes.meta import Meta
from frost.datatypes.plan import DynamicPlan, PlanSet, StaticPlan
from frost.utils.tensor_set import TensorSet


class ControlMessageKind(IntEnum):
    BATCH = 0
    END_OF_EPOCH = 1
    SHUTDOWN = 2


CONTROL_HEADER_SIZE = 2
CONTROL_CODEC_VERSION = 3
_SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}
_CODE_TO_SPLIT = {value: key for key, value in _SPLIT_TO_CODE.items()}


@dataclass
class _TensorEncoder:
    parts: list[torch.Tensor] = field(default_factory=list)
    scalars: list[int] = field(default_factory=list)

    def scalar(self, value: int) -> None:
        self.scalars.append(int(value))

    def _flush_scalars(self) -> None:
        if self.scalars:
            self.parts.append(torch.tensor(self.scalars, dtype=torch.int64))
            self.scalars = []

    def tensor(self, value: torch.Tensor) -> None:
        self._flush_scalars()
        flat = value.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
        self.parts.append(flat.contiguous())

    def nodes(self, value: TensorSet) -> None:
        self.scalar(value.data.numel())
        self.tensor(value.data)

    def optional_nodes(self, value: TensorSet | None) -> None:
        if value is None:
            self.scalar(-1)
            return
        self.nodes(value)

    def finish(self) -> torch.Tensor:
        self._flush_scalars()
        if not self.parts:
            return torch.empty(0, dtype=torch.int64)
        return torch.cat(self.parts)


@dataclass
class _TensorDecoder:
    payload: torch.Tensor
    offset: int = 0

    def scalar(self) -> int:
        if self.offset >= self.payload.numel():
            raise ValueError("Truncated control-plane payload.")
        value = int(self.payload[self.offset].item())
        self.offset += 1
        return value

    def tensor(self, length: int) -> torch.Tensor:
        if length < 0 or self.offset + length > self.payload.numel():
            raise ValueError(
                f"Invalid control-plane tensor length {length} at offset {self.offset}."
            )
        value = self.payload[self.offset : self.offset + length]
        self.offset += length
        return value

    def nodes(self) -> TensorSet:
        length = self.scalar()
        if length < 0:
            raise ValueError("Required node set cannot be absent.")
        return TensorSet(
            self.tensor(length), sort_now=False, ensure_unique=False
        )

    def optional_nodes(self) -> TensorSet | None:
        length = self.scalar()
        if length < 0:
            return None
        return TensorSet(
            self.tensor(length), sort_now=False, ensure_unique=False
        )


def _encode_dynamic_plan(
    encoder: _TensorEncoder, plan: DynamicPlan | None, n_gpus: int
) -> None:
    if plan is None:
        encoder.scalar(0)
        return
    if len(plan.p2p) != n_gpus:
        raise ValueError(
            f"Dynamic plan has {len(plan.p2p)} P2P peers, expected {n_gpus}."
        )
    encoder.scalar(1)
    recv_counts = torch.as_tensor(plan.recv_counts, dtype=torch.int64).reshape(-1)
    send_counts = torch.as_tensor(plan.send_counts, dtype=torch.int64).reshape(-1)
    if recv_counts.numel() != n_gpus or send_counts.numel() != n_gpus:
        raise ValueError("Dynamic-plan send/recv counts must match worker count.")
    encoder.tensor(recv_counts)
    encoder.tensor(send_counts)
    encoder.nodes(plan.fetch)
    encoder.optional_nodes(plan.writeback)
    for nodes in plan.p2p:
        encoder.nodes(nodes)


def _decode_dynamic_plan(
    decoder: _TensorDecoder, n_gpus: int
) -> DynamicPlan | None:
    if decoder.scalar() == 0:
        return None
    recv_counts = (
        decoder.tensor(n_gpus).numpy().astype(np.int32, copy=True)
    )
    send_counts = (
        decoder.tensor(n_gpus).numpy().astype(np.int32, copy=True)
    )
    fetch = decoder.nodes()
    writeback = decoder.optional_nodes()
    p2p = [decoder.nodes() for _ in range(n_gpus)]
    return DynamicPlan(
        p2p=p2p,
        fetch=fetch,
        recv_counts=recv_counts,
        send_counts=send_counts,
        writeback=writeback,
    )


def _encode_static_plan(encoder: _TensorEncoder, plan: StaticPlan | None) -> None:
    if plan is None:
        encoder.scalar(0)
        return
    encoder.scalar(1)
    encoder.nodes(plan.to_cache)
    encoder.nodes(plan.fetch)


def _decode_static_plan(decoder: _TensorDecoder) -> StaticPlan | None:
    if decoder.scalar() == 0:
        return None
    return StaticPlan(to_cache=decoder.nodes(), fetch=decoder.nodes())


def encode_meta(meta: Meta, *, n_gpus: int) -> torch.Tensor:
    """Encode one worker's metadata and plans into a contiguous CPU tensor."""
    if meta.flat_graph is not None:
        raise ValueError("FlatGraph must remain in the shared graph ring buffer.")
    if meta.labels is not None or meta.direct_writeback_request is not None:
        raise ValueError("Worker-owned Meta fields cannot be sent by the host.")
    plan_set = meta.plan_set or PlanSet()
    encoder = _TensorEncoder()
    encoder.scalar(CONTROL_CODEC_VERSION)
    encoder.scalar(_SPLIT_TO_CODE[meta.split])
    encoder.scalar(meta.n_src)
    encoder.scalar(meta.mini_batch_idx)
    encoder.scalar(meta.macro_batch_idx)
    encoder.scalar(meta.epoch_idx)
    encoder.scalar(meta.buf_len)
    encoder.scalar(meta.n_neg)
    encoder.scalar(int(meta.is_last_macro_of_epoch))
    encoder.scalar(int(meta.is_last_macro_of_split))
    if meta.neg_valid_mask is None:
        encoder.scalar(-1)
    else:
        encoder.scalar(meta.neg_valid_mask.numel())
        encoder.tensor(meta.neg_valid_mask)
    shared_dynamic_plan = (
        plan_set.mem_plan is not None and plan_set.box_plan is plan_set.mem_plan
    )
    encoder.scalar(int(shared_dynamic_plan))
    _encode_dynamic_plan(encoder, plan_set.mem_plan, n_gpus)
    if not shared_dynamic_plan:
        _encode_dynamic_plan(encoder, plan_set.box_plan, n_gpus)
    _encode_static_plan(encoder, plan_set.nfeat_plan)
    return encoder.finish()


def decode_meta(payload: torch.Tensor, *, n_gpus: int) -> Meta:
    """Decode one tensor control payload into worker-local metadata."""
    if payload.device.type != "cpu" or payload.dtype != torch.int64:
        raise TypeError("Control payload must be a CPU int64 tensor.")
    decoder = _TensorDecoder(payload)
    version = decoder.scalar()
    if version != CONTROL_CODEC_VERSION:
        raise ValueError(f"Unsupported control-plane codec version {version}.")
    split_code = decoder.scalar()
    try:
        split = cast(Literal["train", "val", "test"], _CODE_TO_SPLIT[split_code])
    except KeyError as exc:
        raise ValueError(f"Invalid split code {split_code}.") from exc
    n_src = decoder.scalar()
    mini_batch_idx = decoder.scalar()
    macro_batch_idx = decoder.scalar()
    epoch_idx = decoder.scalar()
    buf_len = decoder.scalar()
    n_neg = decoder.scalar()
    is_last_macro_of_epoch = bool(decoder.scalar())
    is_last_macro_of_split = bool(decoder.scalar())
    mask_length = decoder.scalar()
    neg_valid_mask = (
        None if mask_length < 0 else decoder.tensor(mask_length).to(torch.bool)
    )
    shared_dynamic_plan = bool(decoder.scalar())
    mem_plan = _decode_dynamic_plan(decoder, n_gpus)
    if shared_dynamic_plan and mem_plan is None:
        raise ValueError("Shared dynamic plan cannot be absent.")
    box_plan = (
        mem_plan
        if shared_dynamic_plan
        else _decode_dynamic_plan(decoder, n_gpus)
    )
    plan_set = PlanSet(
        mem_plan=mem_plan,
        box_plan=box_plan,
        nfeat_plan=_decode_static_plan(decoder),
    )
    if decoder.offset != payload.numel():
        raise ValueError(
            f"Control payload has {payload.numel() - decoder.offset} trailing values."
        )
    return Meta(
        split=split,
        n_src=n_src,
        mini_batch_idx=mini_batch_idx,
        macro_batch_idx=macro_batch_idx,
        epoch_idx=epoch_idx,
        flat_graph=None,
        buf_len=buf_len,
        n_neg=n_neg,
        is_last_macro_of_epoch=is_last_macro_of_epoch,
        is_last_macro_of_split=is_last_macro_of_split,
        plan_set=plan_set,
        neg_valid_mask=neg_valid_mask,
    )


def make_control_header(
    kind: ControlMessageKind, payload_length: int = 0
) -> torch.Tensor:
    if payload_length < 0:
        raise ValueError("Control payload length cannot be negative.")
    return torch.tensor([int(kind), payload_length], dtype=torch.int64)


def parse_control_header(header: torch.Tensor) -> tuple[ControlMessageKind, int]:
    if header.shape != (CONTROL_HEADER_SIZE,) or header.dtype != torch.int64:
        raise ValueError("Control header must be an int64 tensor with two elements.")
    kind = ControlMessageKind(int(header[0].item()))
    payload_length = int(header[1].item())
    if payload_length < 0:
        raise ValueError("Control payload length cannot be negative.")
    if kind is not ControlMessageKind.BATCH and payload_length != 0:
        raise ValueError("Sentinel control messages cannot carry a payload.")
    return kind, payload_length
