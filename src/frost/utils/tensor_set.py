import logging

import torch

logger = logging.getLogger(__name__)


def isin_sorted(
    query: torch.Tensor,
    sorted_candidates: torch.Tensor,
) -> torch.Tensor:
    """Return membership in an ascending, unique candidate tensor."""
    if sorted_candidates.numel() == 0:
        return torch.zeros_like(query, dtype=torch.bool)

    pos = torch.searchsorted(sorted_candidates, query)
    in_bounds = pos < sorted_candidates.numel()
    safe_pos = pos.clamp_max(sorted_candidates.numel() - 1)
    return in_bounds & (sorted_candidates[safe_pos] == query)


class TensorSet:
    data: torch.Tensor
    """
    A Set-like wrapper for 1D PyTorch Tensors supporting operators.
    """

    def __init__(
        self,
        t: torch.Tensor,
        sort_now: bool = True,
        ensure_unique: bool = True,
    ) -> None:
        if t.dim() > 1:
            t = t.reshape(-1)

        # clone + contiguous to avoid underlying storage risks if the external tensor is modified concurrently
        t = t.contiguous().clone()

        if ensure_unique:
            self.data = torch.unique(t, sorted=sort_now)
        elif sort_now:
            self.data = torch.sort(t).values
        else:
            self.data = t

    @classmethod
    def empty_like(cls, t: torch.Tensor) -> "TensorSet":
        empty = torch.empty(0, dtype=t.dtype, device=t.device)
        return cls(empty, sort_now=False, ensure_unique=False)

    def __repr__(self) -> str:
        return f"TensorSet({self.data})"

    def _extract_tensor(self, other: object) -> torch.Tensor:
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

    def __and__(self, other: object) -> "TensorSet":
        other_t = self._extract_tensor(other)
        return TensorSet(self.data[torch.isin(self.data, other_t)], sort_now=False, ensure_unique=False)

    def __or__(self, other: object) -> "TensorSet":
        other_t = self._extract_tensor(other)
        return TensorSet(torch.cat((self.data, other_t)), sort_now=True)

    def __add__(self, other: object) -> "TensorSet":
        return self.__or__(other)

    def __sub__(self, other: object) -> "TensorSet":
        other_t = self._extract_tensor(other)
        return TensorSet(self.data[~torch.isin(self.data, other_t)], sort_now=False, ensure_unique=False)

    def __xor__(self, other: object) -> "TensorSet":
        other_t = self._extract_tensor(other)
        diff1 = self.data[~torch.isin(self.data, other_t)]
        diff2 = other_t[~torch.isin(other_t, self.data)]
        return TensorSet(torch.cat((diff1, diff2)), sort_now=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def numel(self) -> int:
        return self.data.numel()


# Example usage
if __name__ == "__main__":
    t1 = torch.tensor([1, 2, 3, 4, 5])
    t2 = torch.tensor([4, 5, 6, 7, 8])

    s1 = TensorSet(t1)
    s2 = TensorSet(t2)

    print(f"Set 1: {s1}")
    print(f"Set 2: {s2}")

    # 1. Intersection (&)
    print(f"Intersection (&): {s1 & s2}")  # output: [4, 5]

    # 2. Union (|)
    print(f"Union (|): {s1 | s2}")  # output: [1, 2, 3, 4, 5, 6, 7, 8]

    # 3. Plus (+) -> equivalent to Union
    print(f"Plus (+): {s1 + s2}")  # output: [1, 2, 3, 4, 5, 6, 7, 8]

    # 4. Difference (-)
    print(f"Difference (-): {s1 - s2}")  # output: [1, 2, 3] (in s1 but not in s2)

    # 5. Symmetric Difference (^)
    print(f"Sym Diff (^): {s1 ^ s2}")  # output: [1, 2, 3, 6, 7, 8]

    # 6. Mixed use (TensorSet and Raw Tensor)
    t3 = torch.tensor([1, 100])
    print(f"Mixed (-): {s1 - t3}")  # output: [2, 3, 4, 5] (supports subtracting a tensor directly)
