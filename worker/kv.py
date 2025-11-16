import json
from datetime import timedelta
from typing import Iterable, Optional

from torch.distributed import (
    PrefixStore,  # type: ignore
    Store,  # type: ignore
    TCPStore,  # type: ignore
)


class KV:
    base_store: Store

    def __init__(self, base_store: TCPStore, prefix: str = ""):
        self.base_store = (
            base_store if prefix == "" else PrefixStore(prefix, base_store)
        )

    # --- 基础 ---
    def setb(self, key: str, val: bytes):
        self.base_store.set(key, val)  # type: ignore[arg-type]

    def getb(self, key: str) -> bytes:
        return self.base_store.get(key)

    def add(self, key: str, delta: int) -> int:
        # 原子 add；返回加之后的值（注意：返回的是 int）
        return self.base_store.add(key, delta)

    def wait(self, keys: Iterable[str], timeout: Optional[float] = None):
        to = timedelta(seconds=timeout) if timeout else timedelta(seconds=0)
        self.base_store.wait(list(keys), to)

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
    def add_float_x1e6(self, key: str, delta: float) -> float:
        ival = int(round(delta * 1_000_000))
        newv = self.add(key, ival)  # 返回加后的整型
        return newv / 1_000_000.0

    def get_float_x1e6(self, key: str) -> float:
        return int(self.get_str(key)) / 1_000_000.0
