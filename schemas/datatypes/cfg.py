from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(kw_only=True)
class Config:
    g: GNNCFG
    t: TrainCFG
    s: SampleCFG  # if APAN or JODIE, do reverse sampling
    m: MemoryCFG | None = None

    def dict(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True)
class GNNCFG:
    arch: Literal["identity", "attn"] = "identity"
    layer: int = 1
    d_hid: int = 100
    d_time: int = 100
    d_in: int | None = None  # will later be set by train_dist.py
    d_out: int | None = None  # will later be set by train_dist.py
    att_h: int | None = 2
    time_transform: str | None = None
    combine: Literal["rnn"] | None = None


@dataclass(kw_only=True)
class TrainCFG:
    model: Literal["APAN", "JODIE", "TGN"]
    patience: int
    epoch: int
    batch_size: int
    reorder: int | None
    lr: float
    dropout: float
    merge_src: bool
    ndim: int
    edim: int


@dataclass(kw_only=True)
class SampleCFG:
    strategy: Literal["recent", "uniform"]
    layer: int
    history: int
    neg_samples: int
    sample_support_node: bool
    sample_support_for_neg_dst: bool
    neighbor: list[int]
    prop_time: bool
    duration: int
    num_thread: int


@dataclass(kw_only=True)
class MemoryCFG:
    d_time: int
    d_mem: int
    deliver_to: Literal["self", "neighbors"]
    mail_combine: Literal["mean", "last"]
    cell: Literal["gru", "rnn", "transformer"]
    boxsize: int
    combine_node_feature: bool
    att_h: int | None
