from __future__ import annotations

import pickle
from typing import Literal, TypedDict, cast

import numpy as np
import torch

from frost.datatypes.meta import Meta
from frost.datatypes.plan import DyPlan, PlanSet, StPlan
from frost.utils.tensor_set import TensorSet


class TensorWire(TypedDict):
    dtype: str
    data: list[int | float | bool]


class DyPlanWire(TypedDict):
    p2p: list[TensorWire]
    fetch: TensorWire
    rec_cts: list[list[int]]
    send_cts: list[list[int]]
    writeback: TensorWire | None


class StPlanWire(TypedDict):
    to_cache: TensorWire
    fetch: TensorWire


class PlanSetWire(TypedDict):
    mem: DyPlanWire | None
    box: DyPlanWire | None
    nfeat: StPlanWire | None


class MetaWire(TypedDict):
    kind: Literal["meta"]
    cur_epoch: int
    cur_macro_of_split: int
    n_macro_of_split: int
    cur_macro_of_epoch: int
    split: Literal["train", "val", "test"]
    n_src: int
    batch_idx: int
    buf_len: int
    fanout: list[int] | None
    plan_set: PlanSetWire | None


class ControlWire(TypedDict):
    kind: Literal["meta", "end_of_epoch", "stop"]
    meta: MetaWire | None


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_str(dtype: str) -> torch.dtype:
    maybe_dtype = getattr(torch, dtype, None)
    if maybe_dtype is None or not isinstance(maybe_dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype in wire payload: {dtype}")
    return maybe_dtype


def tensor_to_wire(t: torch.Tensor) -> TensorWire:
    cpu_t = t.detach().to(device="cpu").contiguous()
    return TensorWire(dtype=_dtype_to_str(cpu_t.dtype), data=cpu_t.tolist())


def wire_to_tensor(w: TensorWire) -> torch.Tensor:
    return torch.tensor(w["data"], dtype=_dtype_from_str(w["dtype"]), device="cpu")


def tensor_set_to_wire(ts: TensorSet) -> TensorWire:
    return tensor_to_wire(ts.data)


def wire_to_tensor_set(w: TensorWire) -> TensorSet:
    return TensorSet(wire_to_tensor(w), sort_now=False)


def dy_plan_to_wire(plan: DyPlan) -> DyPlanWire:
    return DyPlanWire(
        p2p=[tensor_set_to_wire(ts) for ts in plan.p2p],
        fetch=tensor_set_to_wire(plan.fetch),
        rec_cts=cast(list[list[int]], plan.rec_cts.tolist()),
        send_cts=cast(list[list[int]], plan.send_cts.tolist()),
        writeback=tensor_set_to_wire(plan.writeback) if plan.writeback is not None else None,
    )


def wire_to_dy_plan(w: DyPlanWire) -> DyPlan:
    rec_cts_np = np.asarray(w["rec_cts"], dtype=np.int32)
    send_cts_np = np.asarray(w["send_cts"], dtype=np.int32)
    return DyPlan(
        p2p=[wire_to_tensor_set(ts) for ts in w["p2p"]],
        fetch=wire_to_tensor_set(w["fetch"]),
        rec_cts=rec_cts_np,
        send_cts=send_cts_np,
        writeback=wire_to_tensor_set(w["writeback"]) if w["writeback"] is not None else None,
        fetch_on_gpu=None,
    )


def st_plan_to_wire(plan: StPlan) -> StPlanWire:
    return StPlanWire(
        to_cache=tensor_set_to_wire(plan.to_cache),
        fetch=tensor_set_to_wire(plan.fetch),
    )


def wire_to_st_plan(w: StPlanWire) -> StPlan:
    return StPlan(
        to_cache=wire_to_tensor_set(w["to_cache"]),
        fetch=wire_to_tensor_set(w["fetch"]),
    )


def plan_set_to_wire(plan_set: PlanSet | None) -> PlanSetWire | None:
    if plan_set is None:
        return None
    return PlanSetWire(
        mem=dy_plan_to_wire(plan_set.mem) if plan_set.mem is not None else None,
        box=dy_plan_to_wire(plan_set.box) if plan_set.box is not None else None,
        nfeat=st_plan_to_wire(plan_set.nfeat) if plan_set.nfeat is not None else None,
    )


def wire_to_plan_set(w: PlanSetWire | None) -> PlanSet | None:
    if w is None:
        return None
    return PlanSet(
        mem=wire_to_dy_plan(w["mem"]) if w["mem"] is not None else None,
        box=wire_to_dy_plan(w["box"]) if w["box"] is not None else None,
        nfeat=wire_to_st_plan(w["nfeat"]) if w["nfeat"] is not None else None,
    )


def meta_to_wire(meta: Meta) -> MetaWire:
    return MetaWire(
        kind="meta",
        cur_epoch=meta.cur_epoch,
        cur_macro_of_split=meta.cur_macro_of_split,
        n_macro_of_split=meta.n_macro_of_split,
        cur_macro_of_epoch=meta.cur_macro_of_epoch,
        split=meta.split,
        n_src=meta.n_src,
        batch_idx=meta.batch_idx,
        buf_len=meta.buf_len,
        fanout=meta.fanout,
        plan_set=plan_set_to_wire(meta.plan_set),
    )


def wire_to_meta(w: MetaWire) -> Meta:
    return Meta(
        cur_epoch=w["cur_epoch"],
        cur_macro_of_split=w["cur_macro_of_split"],
        n_macro_of_split=w["n_macro_of_split"],
        cur_macro_of_epoch=w["cur_macro_of_epoch"],
        split=w["split"],
        n_src=w["n_src"],
        batch_idx=w["batch_idx"],
        flat_graph=None,
        buf_len=w["buf_len"],
        graph=None,
        plan_set=wire_to_plan_set(w["plan_set"]),
        access=None,
        labels=None,
        fanout=w["fanout"],
    )


def pack_wire_message(msg: ControlWire) -> bytes:
    return pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)


def unpack_wire_message(payload: bytes) -> ControlWire:
    obj = pickle.loads(payload)
    if not isinstance(obj, dict) or "kind" not in obj or "meta" not in obj:
        raise ValueError("Invalid wire message payload")
    return cast(ControlWire, obj)
