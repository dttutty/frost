import logging

import torch


logger = logging.getLogger(__name__)


class TensorSet:
    """
    A Set-like wrapper for 1D PyTorch Tensors supporting operators.
    """

    def __init__(self, t: torch.Tensor, sort_now: bool = True) -> None:
        # 初始化时确保唯一性，像真正的 Set 一样
        # 保持 1D
        if t.dim() > 1:
            t = t.reshape(-1)
        self.data = torch.unique(t, sorted=sort_now)

    def __repr__(self) -> str:
        return f"TensorSet({self.data})"

    def _extract_tensor(self, other: object) -> torch.Tensor:
        """Helper to handle both TensorSet and torch.Tensor inputs"""
        if isinstance(other, TensorSet):
            return other.data
        elif isinstance(other, torch.Tensor):
            return other if other.dim() == 1 else other.reshape(-1)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    # -----------------------------------------------------------
    # 1. 实现 & 号 (Intersection / 交集)
    # -----------------------------------------------------------
    def __and__(self, other: object) -> "TensorSet":
        other_t = self._extract_tensor(other)
        # mask = isin(self, other)
        mask = torch.isin(self.data, other_t, assume_unique=True)
        # 返回新的 TensorSet
        return TensorSet(self.data[mask], sort_now=False)

    # -----------------------------------------------------------
    # 2. 实现 | 号 (Union / 并集)
    # -----------------------------------------------------------
    def __or__(self, other: object) -> "TensorSet":
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
        other_t = self._extract_tensor(other)
        # ~isin
        mask = ~torch.isin(self.data, other_t, assume_unique=True)
        return TensorSet(self.data[mask], sort_now=False)

    # -----------------------------------------------------------
    # 5. 实现 ^ 号 (Symmetric Difference / 对称差集)
    # -----------------------------------------------------------
    def __xor__(self, other: object) -> "TensorSet":
        other_t = self._extract_tensor(other)
        # (A-B) U (B-A)
        # 为了速度，直接算两边的差集然后拼接
        diff1 = self.data[~torch.isin(self.data, other_t, assume_unique=True)]
        diff2 = other_t[~torch.isin(other_t, self.data, assume_unique=True)]
        return TensorSet(torch.cat((diff1, diff2)), sort_now=True)

    # 支持 len()
    def __len__(self) -> int:
        return len(self.data)

    # 支持 indexing 取值
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


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
