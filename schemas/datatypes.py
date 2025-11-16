from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from threading import Event
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
from dgl.heterograph import DGLBlock
from numpy.typing import NDArray


@dataclass
class GNNCFG:
    arch: Literal["identity", "transformer_attention"] = "identity"
    dim_out: int = 100
    layer: int = 1
    dim_time: int = 100
    att_head: int = 2
    time_transform: Optional[str] = None
    combine: Optional[Literal["rnn"]] = None


@dataclass
class TrainCFG:
    epoch: Optional[int] = 10
    batch_size: Optional[int] = 600
    reorder: Optional[int] = 8
    lr: Optional[float] = 0.0001
    dropout: Optional[float] = 0.1
    att_dropout: Optional[float] = 0.1
    all_on_gpu: Optional[bool] = False
    merge_src: bool = False  # add by sqp17
    dedup: bool = True  # add by sqp17
    ndim: Optional[int] = None  # add by sqp17
    edim: Optional[int] = None  # add by sqp17


@dataclass
class SampleCFG:
    strategy: Literal["recent", "uniform"] = "recent"
    layer: Optional[int] = 1
    neighbor: Optional[List[int]] = field(default_factory=lambda: [10, 10])
    history: Optional[int] = 1
    prop_time: bool = False
    duration: Optional[int] = 0
    num_thread: Optional[int] = 32
    no_sample: Optional[bool] = False
    no_neg: Optional[bool] = False


@dataclass
class MemoryCFG:
    type: Optional[Literal["node"]] = "node"
    dim_time: Optional[int] = None  # time dim
    deliver_to: Optional[Literal["self", "neighbors"]] = "self"
    mail_combine: Optional[Literal["mean", "last"]] = "last"
    memory_update: Optional[Literal["gru", "rnn", "transformer"]] = "gru"
    mailbox_size: Optional[int] = None
    combine_node_feature: bool = False
    attention_head: Optional[int] = None
    dim_out: Optional[int] = None  # memory dim

    # shared_cfg = {
    #     'node_memory':      ([num_nodes, memory_param.dim_out], torch.float32),
    #     'node_memory_ts':   ([num_nodes], torch.float32),
    #     'mails':            ([num_nodes, memory_param.mailbox_size, 2 * memory_param.dim_out + edge_feat_dim], torch.float32),
    #     'mail_ts':          ([num_nodes, memory_param.mailbox_size], torch.float32),
    #     'next_mail_pos':    ([num_nodes], torch.long),
    #     'mail_pos_update_pending_flag':  ([num_nodes], torch.int8),
    # }


@dataclass
class ScatterResult:
    m_id: int
    total_sample_time: Optional[float] = None
    ap: Optional[List[float]] = None
    auc: Optional[List[float]] = None


@dataclass
class Config:
    s: SampleCFG
    m: MemoryCFG
    g: GNNCFG
    t: TrainCFG

    def dict(self):
        return asdict(self)


@dataclass
class MailTask:
    bid: Optional[int] = None  # batch id（用于 gate 的 batch 粒度节流）
    mid: Optional[int] = None  # microbatch id（用于 gate 的微批粒度完成信号）
    root_nodes: Optional[NDArray[np.int32]] = None
    block: Optional[object] = None
    last_updated_nid: Optional[torch.Tensor] = None
    last_updated_memory: Optional[torch.Tensor] = None
    ts: Optional[NDArray[np.float32]] = None
    mem_edge_feats: Optional[torch.Tensor] = None
    done: Optional[Event] = None
    last_updated_ts: Optional[torch.Tensor] = None


@dataclass
class SharedMailboxArrays:
    """一个专门存放所有共享内存张量的数据类"""

    node_memory: torch.Tensor
    node_memory_ts: torch.Tensor
    mails: torch.Tensor
    mail_ts: torch.Tensor
    next_mail_pos: torch.Tensor
    mail_pos_update_pending_flag: torch.Tensor

    @classmethod
    def create_from_config(
        cls, shared_cfg: dict, init_shared_func, local_rank: int
    ) -> "SharedMailboxArrays":
        """
        工厂方法：根据配置，初始化所有共享数组，并返回一个实例。
        """

        # 1. 验证配置和类的字段是否匹配 (可选，但非常安全)
        cfg_keys = set(shared_cfg.keys())
        cls_keys = {f.name for f in fields(cls)}
        if cfg_keys != cls_keys:
            raise ValueError(
                f"Config keys {cfg_keys} do not match class fields {cls_keys}"
            )

        # 2. 创建所有数组
        #    使用字典推导式 **kwargs
        init_kwargs = {
            name: init_shared_func(name, *spec) for name, spec in shared_cfg.items()
        }

        # 3. 只让 rank 0 清零
        if local_rank == 0:
            for arr in init_kwargs.values():
                arr.zero_()

        # 4. 使用关键字参数解包，实例化本类
        #    例如: SharedMailboxArrays(node_memory=tensor1, mail_ts=tensor2, ...)
        return cls(**init_kwargs)


@dataclass
class PrevCache:
    gpunfeat: Optional[torch.Tensor]
    gpuefeat: Optional[torch.Tensor]
    uniqn: Optional[torch.Tensor]
    uniqe: Optional[torch.Tensor]


@dataclass
class GPUData:
    mfgs: Optional[List[DGLBlock]]
    nfeat: Optional[torch.Tensor]
    efeat: Optional[torch.Tensor]
    gpumem: Optional[torch.Tensor]
    gpumail: Optional[torch.Tensor]


@dataclass
class DataTsNid:
    nid: torch.Tensor
    data: torch.Tensor
    ts: Optional[torch.Tensor] = None
    ct: Optional[torch.Tensor] = None


    def keep_latest():
        pass

    def keep_mean():
        


@dataclass
class MailboxTsNid:
    nid: torch.Tensor
    ts: torch.Tensor
    data: torch.Tensor


@dataclass
class Data4Train:
    mid: Optional[int]
    mfgs: Optional[List[DGLBlock]]
    nfeat: Optional[torch.Tensor]
    efeat: Optional[torch.Tensor]
    uniqn: Optional[torch.Tensor]
    uniqe: Optional[torch.Tensor]
    done_ev: Optional[torch.cuda.Event]
    gpumem: Optional[torch.Tensor] = None
    gpumail: Optional[torch.Tensor] = None


@dataclass
class TransferData:
    mid: int
    bid: int
    mfgss: List[List[DGLBlock]]
    root: NDArray[np.int32]
    ts: NDArray[np.float32]
    eid: Optional[List] = None
    block: Optional[DGLBlock] = None
    next_pos_root_nodes: Optional[List] = None  # 这是src, dst
    next_all_nodes: Optional[List] = (
        None  # 这是src, dst, neg_dst + (src, dst, neg_dst)采样得到的
    )


@dataclass
class Peer2Peer:
    memplan: torch.Tensor
    mem2disk: torch.Tensor

    mailplan: torch.Tensor
    mail2disk: torch.Tensor

    featplan: torch.Tensor


Role = Literal["worker", "host"]


@dataclass(frozen=True)
class DistEnv:
    rank: int
    world_size: int
    host_rank: int  # 约定为 world_size - 1
    role: Role  # "worker" / "host"
    ngpus: int  # = world_size - 1（每 worker 1 张）
    local_rank: int  # 用于绑定设备

    @property
    def is_host(self) -> bool:
        return self.role == "host"

    @classmethod
    def initialize(cls) -> "DistEnv":
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Process group not initialized")

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        host_rank = world_size - 1
        role: Role = "host" if rank == host_rank else "worker"

        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

        # --- [修改] 启用运行期检查 ---
        # 每进程可见 GPU 数（此时 CUDA_VISIBLE_DEVICES 应该已被 bootstrap_visible_devices 收窄）
        try:
            # import torch # torch 应该已在 train.py 中导入
            visible_gpus_this_proc = torch.cuda.device_count()
        except Exception:
            visible_gpus_this_proc = 0

        ngpus = max(world_size - 1, 0)

        # 运行期一致性检查（第二层防护）
        if role == "worker" and visible_gpus_this_proc != 1:
            # 豁免：在 CPU-only 调试模式下 (ngpus=0)，此检查不适用
            if ngpus > 0:
                raise RuntimeError(
                    f"[DistEnv Init Error] Worker rank {rank} (local_rank {local_rank}) "
                    f"应该只看到 1 张 GPU (因已被收窄)，但实际看到 {visible_gpus_this_proc} 张。\n"
                    f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}'"
                )

        if role == "host" and visible_gpus_this_proc != 0:
            raise RuntimeError(
                f"[DistEnv Init Error] Host rank {rank} (local_rank {local_rank}) "
                f"不应看到任何 GPU (应被设为空)，但实际看到 {visible_gpus_this_proc} 张。\n"
                f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}'"
            )
        # --- [修改结束] ---

        return cls(
            rank=rank,
            world_size=world_size,
            host_rank=host_rank,
            role=role,
            ngpus=ngpus,
            local_rank=local_rank,
        )
