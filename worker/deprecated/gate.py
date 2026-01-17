import time
from threading import Condition, Lock
from typing import Dict, Optional


# from torch.distributed import Store   # 可加类型注解


class GlobalMicrobatchGate:
    def __init__(self, store, m_per_b: int, gap: int = 3):
        self._gap = gap
        self._finished = -1  # last finished microbatch id
        self._cv1 = Condition(Lock())  # 保护 _finished
        self._store = store
        self._key_finished = "ctrl/last_finished"

        self._cv2 = Condition(Lock())  # 保护 _mm_counts
        self.last_b = -1  # 当前最新的 batch id
        self._key_mm_prefix = "ctrl/last_mm_count_b_"  # 每个 batch 各一把计数
        self._m_per_b = int(m_per_b)

        # 本进程内的 shadow 计数（避免每次都去 store 读）
        self._mm_counts: Dict[int, int] = {}

    def mark_mm_updated(self, b: int):
        """某个 microbatch 属于 batch b，完成一次 mailbox 更新"""
        key = f"{self._key_mm_prefix}{int(b)}"

        # 1) 跨进程：原子自增
        try:
            new_val = self._store.add(key, 1)  # 返回自增后的值（int）

        # (!!!!) 关键修复：移除了危险的、有竞态条件的 except 回退逻辑 (!!!!)
        # 必须依赖 store.add() 的原子性。
        # 如果 store 不支持 add，回退逻辑也是错的，不如直接让它失败。

        except Exception as e:
            print(
                f"FATAL: Store backend does not support atomic 'add'. Race condition possible. Error: {e}"
            )
            # 重新抛出异常，因为没有 'add' 就无法安全工作
            raise e

        # 2) 本进程：更新本地 shadow 并唤醒等待者
        with self._cv2:
            self._mm_counts[b] = new_val
            self._cv2.notify_all()

    def wait_can_update_next_mail_pos(
        self, b: int, timeout: Optional[float] = None
    ) -> bool:
        """等待 batch b 的 microbatch 全部完成（达到 m_per_b）"""
        key = f"{self._key_mm_prefix}{int(b)}"
        start_time = time.time()

        def _check_store() -> bool:
            # 检查本地缓存
            with self._cv2:
                cnt = self._mm_counts.get(b, 0)
            if cnt >= self._m_per_b:
                return True

            # 轮询 Store
            try:
                val_bytes = self._store.get(key)
                cnt_store = int.from_bytes(val_bytes, "big", signed=False)
                with self._cv2:
                    self._mm_counts[b] = cnt_store  # 更新缓存
                return cnt_store >= self._m_per_b
            except Exception:
                return False

        # (!!!!) 关键修复：使用循环轮询 (!!!!)
        while not _check_store():
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False  # 超时

            # 短暂休眠或等待
            with self._cv2:
                # 等待一个短时间或等待本地 notify，以先到者为准
                self._cv2.wait(timeout=0.05)  # 50ms 轮询间隔

        return True

    def mark_scattered(self, b: int):
        """DataMover 在 microbatch 属于 batch b 时调用"""
        with self._cv2:
            if b > self.last_b:
                self.last_b = b

    def wait_until_scattered(self, b: int, timeout: Optional[float] = None) -> bool:
        """DataMover 在 microbatch 属于 batch b 时调用；等待直到 batch b 已经开始 scatter"""
        with self._cv2:
            return self._cv2.wait_for(lambda: b <= self.last_b, timeout=timeout)

    def mark_finished(self, m: int):
        """由 MailboxMemUpdater 在 microbatch=m 完成本地更新后调用"""
        # 广而告之（跨进程）
        self._store.set(self._key_finished, int(m).to_bytes(8, "big", signed=True))
        # 本进程状态 + 唤醒
        with self._cv1:
            if m > self._finished:
                self._finished = m
                self._cv1.notify_all()

    def wait_can_start(self, m: int, timeout: Optional[float] = None) -> bool:
        """DataMover 在启动 microbatch=m 前调用；满足 m <= finished + gap 才放行"""
        with self._cv1:
            return self._cv1.wait_for(
                lambda: m <= self._finished + self._gap, timeout=timeout
            )

    def get(self) -> int:
        with self._cv1:
            return self._finished
