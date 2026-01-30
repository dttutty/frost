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
    model: Literal["APAN", "JODIE", "TGN"] = "TGN"
    patience: int = 50
    epoch: int = 10
    batch_size: int = 600
    reorder: int | None = 8
    lr: float = 0.0001
    dropout: float = 0.1
    merge_src: bool = False  # add by sqp17
    ndim: int = 0  # will later be set by train_dist.py
    edim: int = 0  # will later be set by train_dist.py


@dataclass(kw_only=True)
class SampleCFG:
    strategy: Literal["recent", "uniform"] = "recent"
    layer: int = 1
    history: int = 1
    neg_samples: int = 1
    sample_support_node: bool = True
    sample_support_for_neg_dst: bool = True
    neighbor: list[int]
    prop_time: bool = False
    duration: int = 0
    num_thread: int = 32  # valid if history > 1


@dataclass(kw_only=True)
class MemoryCFG:
    d_time: int = 0  # time dim
    d_mem: int = 128  # memory dim
    deliver_to: Literal["self", "neighbors"] = "self"
    mail_combine: Literal["mean", "last"] = "last"
    cell: Literal["gru", "rnn", "transformer"] = "gru"
    boxsize: int = 1
    combine_node_feature: bool = True
    att_h: int | None = None  # only for transformer cell
