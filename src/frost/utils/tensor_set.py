import logging
import threading

import torch

logger = logging.getLogger(__name__)


_SET_OP_LOCK = threading.Lock()


class TensorSet:
    data: torch.Tensor
    """
    A Set-like wrapper for 1D PyTorch Tensors supporting operators.
    """

    def __init__(self, t: torch.Tensor, sort_now: bool = True) -> None:
        # 初始化时确保唯一性，像真正的 Set 一样
        # 保持 1D
        if t.dim() > 1:
            t = t.reshape(-1)
        self.data = torch.unique(t, sorted=sort_now)

    @classmethod
    def empty_like(cls, t: torch.Tensor) -> "TensorSet":
        """
        Create an empty TensorSet with same device and dtype as t.
        """
        empty = torch.empty(
            0,
            dtype=t.dtype,
            device=t.device,
        )
        # sort_now=False 避免多余操作
        return cls(empty, sort_now=False)

    def __repr__(self) -> str:
        return f"TensorSet({self.data})"

    def _extract_tensor(self, other: object) -> torch.Tensor:
        """Helper to handle both TensorSet and torch.Tensor inputs"""
        if isinstance(other, TensorSet):
            other_t = other.data
        elif isinstance(other, torch.Tensor):
            other_t = other if other.dim() == 1 else other.reshape(-1)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        if other_t.device != self.data.device:
            other_t = other_t.to(self.data.device)
        if other_t.dtype != self.data.dtype:
            other_t = other_t.to(self.data.dtype)
        return other_t

    def _contains_mask(self, other_t: torch.Tensor) -> torch.Tensor:
        """Return element-wise membership mask without using torch.isin."""
        if self.data.numel() == 0:
            return torch.zeros_like(self.data, dtype=torch.bool)
        if other_t.numel() == 0:
            return torch.zeros_like(self.data, dtype=torch.bool)

        sorted_other = torch.sort(other_t).values
        idx = torch.searchsorted(sorted_other, self.data)
        in_range = idx < sorted_other.numel()
        mask = torch.zeros_like(self.data, dtype=torch.bool)
        mask[in_range] = sorted_other[idx[in_range]] == self.data[in_range]
        return mask

    # -----------------------------------------------------------
    # 1. 实现 & 号 (Intersection / 交集)
    # -----------------------------------------------------------
    def __and__(self, other: object) -> "TensorSet":
        with _SET_OP_LOCK:
            other_t = self._extract_tensor(other)
            mask = self._contains_mask(other_t)
            # 返回新的 TensorSet
            return TensorSet(self.data[mask], sort_now=False)

    # -----------------------------------------------------------
    # 2. 实现 | 号 (Union / 并集)
    # -----------------------------------------------------------
    def __or__(self, other: object) -> "TensorSet":
        with _SET_OP_LOCK:
            other_t = self._extract_tensor(other)
            # Cat and unique
            return TensorSet(torch.cat((self.data, other_t)), sort_now=True)

    # -----------------------------------------------------------
    # 3. 实现 + 号 (Alias for Union / 加号也做并集)
    # -----------------------------------------------------------
    def __add__(self, other: object) -> "TensorSet":
        # 集合论里通常用 | 表示并集，但为了方便，+ 号也可以实现为并集
        return self.__or__(other)

    # -----------------------------------------------------------
    # 4. 实现 - 号 (Difference / 差集)
    # -----------------------------------------------------------
    def __sub__(self, other: object) -> "TensorSet":
        with _SET_OP_LOCK:
            other_t = self._extract_tensor(other)
            mask = ~self._contains_mask(other_t)
            return TensorSet(self.data[mask], sort_now=False)

    # -----------------------------------------------------------
    # 5. 实现 ^ 号 (Symmetric Difference / 对称差集)
    # -----------------------------------------------------------
    def __xor__(self, other: object) -> "TensorSet":
        with _SET_OP_LOCK:
            other_t = self._extract_tensor(other)
            # (A-B) U (B-A)
            # 为了速度，直接算两边的差集然后拼接
            diff1 = self.data[~self._contains_mask(other_t)]
            diff2 = other_t[
                ~TensorSet(other_t, sort_now=False)._contains_mask(self.data)
            ]
            return TensorSet(torch.cat((diff1, diff2)), sort_now=True)

    # 支持 len()
    def __len__(self) -> int:
        return len(self.data)

    # 支持 indexing 取值
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def numel(self) -> int:
        return self.data.numel()


# 使用示例
if __name__ == "__main__":
    t1 = torch.tensor([1, 2, 3, 4, 5])
    t2 = torch.tensor([4, 5, 6, 7, 8])

    s1 = TensorSet(t1)
    s2 = TensorSet(t2)

    print(f"Set 1: {s1}")
    print(f"Set 2: {s2}")

    # 1. 交集 (&)
    print(f"Intersection (&): {s1 & s2}")  # output: [4, 5]

    # 2. 并集 (|)
    print(f"Union (|): {s1 | s2}")  # output: [1, 2, 3, 4, 5, 6, 7, 8]

    # 3. 加号 (+) -> 等同于并集
    print(f"Plus (+): {s1 + s2}")  # output: [1, 2, 3, 4, 5, 6, 7, 8]

    # 4. 差集 (-)
    print(f"Difference (-): {s1 - s2}")  # output: [1, 2, 3] (在s1不在s2)

    # 5. 对称差集 (^)
    print(f"Sym Diff (^): {s1 ^ s2}")  # output: [1, 2, 3, 6, 7, 8]

    # 6. 混合使用 (TensorSet 和 Raw Tensor)
    t3 = torch.tensor([1, 100])
    print(f"Mixed (-): {s1 - t3}")  # output: [2, 3, 4, 5] (支持直接减 tensor)
