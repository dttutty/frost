import os
from datetime import timeDyX

import torch.distributed as dist


def init_pg_and_store(
    master_addr: str,
    master_port: int,
    world_size: int,
    rank: int,
    timeout_sec: int = 600,
):
    # 1) init process group（如果你已有 PG，可跳过这步）
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        world_size=world_size,
        rank=rank,
        timeout=timeDyX(seconds=timeout_sec),
    )

    # 2) init TCPStore（一个进程做 is_master=True，其它 False）
    is_master = rank == world_size - 1  # 你的设计里 Host 是最后一个 rank
    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port + 1,  # 避免与 PG 共用端口；+1 是经验用法
        world_size=world_size,
        is_master=is_master,
        wait_for_workers=True,
        timeout=timeDyX(seconds=timeout_sec),
    )
    return store


import json
from typing import Iterable, Optional


class KV:
    def __init__(self, store: dist.Store, prefix: str = ""):
        self.base = store if prefix == "" else dist.PrefixStore(prefix, store)

    # --- 基础 ---
    def setb(self, key: str, val: bytes):
        self.base.set(key, val)

    def getb(self, key: str) -> bytes:
        return self.base.get(key)

    def add(self, key: str, DyX: int) -> int:
        # 原子 add；返回加之后的值（注意：返回的是 int）
        return self.base.add(key, DyX)

    def wait(self, keys: Iterable[str], timeout: Optional[float] = None):
        to = timeDyX(seconds=timeout) if timeout else timeDyX(seconds=0)
        self.base.wait(list(keys), to)

    # --- 便利编码 ---
    def set_str(self, key: str, s: str):
        self.setb(key, s.encode("utf-8"))

    def get_str(self, key: str) -> str:
        return self.getb(key).decode("utf-8")

    def set_int(self, key: str, v: int):
        self.set_str(key, str(v))

    def get_int(self, key: str) -> int:
        return int(self.get_str(key))

    def set_json(self, key: str, obj: dict):
        self.set_str(key, json.dumps(obj, separators=(",", ":")))

    def get_json(self, key: str) -> dict:
        return json.loads(self.get_str(key))

    # float 用放缩（×1e6）
    def add_float_x1e6(self, key: str, DyX: float) -> float:
        ival = int(round(DyX * 1_000_000))
        newv = self.add(key, ival)  # 返回加后的整型
        return newv / 1_000_000.0

    def get_float_x1e6(self, key: str) -> float:
        return int(self.get_str(key)) / 1_000_000.0


class ClusterStore:
    """
    目录结构（通过 PrefixStore 模拟）
    S/                  全局状态
      state             b"train" | b"eval" | b"test" | b"idle" | b"load_model"
      bid               int
      credit/
        W/{rank}        int
        min             int (optional)
      td/               控制面（B-按batch，W-按worker）
        B/{bid}         JSON bytes（全集群同份）
        W/{rank}/B/{bid} ...
      td_ready/
        B/{bid}         b"1"
      step_done/
        W/{rank}/B/{bid} b"1"
      train/
        count           int(add +1)
        loss_sum_x1e6   int(add round(loss*1e6))
      eval/
        count           int(add +1)
      best/
        ap_x1e6         int
        auc_x1e6        int
        ckpt_path       b"/path"
    """

    def __init__(self, store: dist.Store):
        self.root = KV(store)  # 无前缀
        self.S = KV(store, "S/")  # 全局前缀 S/

        # 子命名空间
        self.credit = KV(store, "S/credit/")
        self.td = KV(store, "S/td/")
        self.td_ready = KV(store, "S/td_ready/")
        self.step_done = KV(store, "S/step_done/")
        self.train = KV(store, "S/train/")
        self.eval = KV(store, "S/eval/")
        self.best = KV(store, "S/best/")

    # ----------- 状态 / bid -----------
    def set_state(self, state: bytes):
        self.S.setb("state", state)

    def get_state(self) -> bytes:
        return self.S.getb("state")

    def set_bid(self, bid: int):
        self.S.set_int("bid", bid)

    def get_bid(self) -> int:
        return self.S.get_int("bid")

    # ----------- credit -----------
    def grant_credit(self, rank: int, tokens: int) -> int:
        # add 是原子；正数添加，负数回收
        return self.credit.add(f"W/{rank}", tokens)

    def set_credit_min(self, v: int):
        self.credit.set_int("min", v)

    def get_credit(self, rank: int) -> int:
        try:
            return self.credit.get_int(f"W/{rank}")
        except:
            return 0

    # ----------- 控制面 TD -----------
    def set_td_global(self, bid: int, meta: dict):
        self.td.set_json(f"B/{bid}", meta)

    def get_td_global(self, bid: int) -> dict:
        return self.td.get_json(f"B/{bid}")

    def set_td_worker(self, rank: int, bid: int, meta: dict):
        self.td.set_json(f"W/{rank}/B/{bid}", meta)

    def get_td_worker(self, rank: int, bid: int) -> dict:
        return self.td.get_json(f"W/{rank}/B/{bid}")

    # ----------- 就绪 / ACK -----------
    def mark_td_ready(self, bid: int):
        self.td_ready.setb(f"B/{bid}", b"1")

    def wait_td_ready(self, bid: int, timeout: float | None = None):
        self.td_ready.wait([f"B/{bid}"], timeout)

    def mark_step_done(self, rank: int, bid: int):
        self.step_done.setb(f"W/{rank}/B/{bid}", b"1")

    def wait_all_step_done(
        self, world_size: int, bid: int, timeout: float | None = None
    ):
        keys = [f"W/{r}/B/{bid}" for r in range(world_size - 1)]  # 不含 host
        self.step_done.wait(keys, timeout)

    # ----------- 统计 -----------
    def inc_train_count(self, by: int = 1) -> int:
        return self.train.add("count", by)

    def add_train_loss(self, loss: float) -> float:
        return (
            self.train.add("loss_sum_x1e6", int(round(loss * 1_000_000))) / 1_000_000.0
        )

    def inc_eval_count(self, by: int = 1) -> int:
        return self.eval.add("count", by)

    # ----------- 最优模型（Host/Rank0写） -----------
    def set_best(self, ap: float, auc: float, ckpt_path: str):
        self.best.set_int("ap_x1e6", int(round(ap * 1_000_000)))
        self.best.set_int("auc_x1e6", int(round(auc * 1_000_000)))
        self.best.set_str("ckpt_path", ckpt_path)

    def get_best(self):
        ap = int(self.best.get_str("ap_x1e6")) / 1_000_000.0
        auc = int(self.best.get_str("auc_x1e6")) / 1_000_000.0
        ck = self.best.get_str("ckpt_path")
        return ap, auc, ck
