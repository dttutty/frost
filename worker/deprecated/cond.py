from torch.distributed import TCPStore  # type: ignore

from worker.kv import KV


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

    def __init__(self, store: TCPStore):
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
        # set_state之前之后需要barrier同步
        self.S.setb("state", state)

    def get_state(self) -> bytes:
        return self.S.getb("state")

    def set_bid(self, bid: int):
        self.S.set_int("bid", bid)

    def get_bid(self) -> int:
        return self.S.get_int("bid")

    # ----------- credit -----------
    # 每个 worker 的 credit 存在 S/credit/W/{rank} 下
    # grant_credit 正数添加，负数回收
    # get_credit 不存在则返回 0
    # set_credit_min 设置最小 credit（Host/Rank0 写）
    # credit 用于控制 worker 的并发度
    # worker 可通过 get_credit 查询当前 credit 数量

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
    # 就是数据准备好了,已经放在共享区了
    def mark_td_ready(self, bid: int):
        self.td_ready.setb(f"B/{bid}", b"1")  # 这里表示这个batch的数据已经准备好了

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
