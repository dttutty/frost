from typing import cast

import torch


class TensorSetFn:
    """
    Helper class to perform set operations using PyTorch Tensors.
    Assumes inputs are 1D tensors.
    """

    @staticmethod
    def unique(t: torch.Tensor, sorted: bool = True) -> torch.Tensor:
        # Equivalent to set(t)
        # return_inverse and return_counts can be useful for advanced stats
        return cast(torch.Tensor, torch.unique(t, sorted=sorted))

    @staticmethod
    def intersection(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Equivalent to t1 & t2
        # Use torch.isin to create a boolean mask
        # Note: assume_unique=True helps performance if t2 is already unique
        mask = torch.isin(t1, t2, assume_unique=True)
        return t1[mask]

    @staticmethod
    def union(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Equivalent to t1 | t2
        # Concatenate and then find unique elements
        return cast(torch.Tensor, torch.unique(torch.cat((t1, t2))))

    @staticmethod
    def difference(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Equivalent to t1 - t2
        # Invert the boolean mask from isin
        mask = ~torch.isin(t1, t2, assume_unique=True)
        return t1[mask]

    @staticmethod
    def symmetric_difference(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Equivalent to t1 ^ t2
        # Elements in either t1 or t2, but not both
        # Method 1: (Union) - (Intersection)
        # Method 2: (A-B) concat (B-A) -> usually faster as it avoids large sorts
        diff1 = t1[~torch.isin(t1, t2, assume_unique=True)]
        diff2 = t2[~torch.isin(t2, t1, assume_unique=True)]
        return cast(torch.Tensor, torch.cat((diff1, diff2)))


if __name__ == "__main__":
    # --- Usage Example ---

    # Create large random data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randint(0, 100, (20,), device=device)
    b = torch.randint(0, 100, (20,), device=device)

    # Ensure they are unique first to simulate "set" behavior
    set_a = torch.unique(a)
    set_b = torch.unique(b)

    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")

    # Intersection
    inter = TensorSetFn.intersection(set_a, set_b)
    print(f"Intersection: {inter}")

    # Difference
    diff = TensorSetFn.difference(set_a, set_b)
    print(f"Difference (A - B): {diff}")
