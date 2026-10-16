"""All workers must participate in every state-versioned macro-batch."""

from collections import Counter

import pandas as pd
import pytest

from frost.runtime.host._batch_balance import make_balance_plan


@pytest.mark.parametrize("n_gpu", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 7])
def test_tail_batch_keeps_every_worker_and_balances_rows(
    n_gpu: int, batch_size: int,
) -> None:
    full_size = n_gpu * batch_size
    for leftover in range(full_size):
        frame = pd.DataFrame({"event": range(full_size + leftover)})

        balanced, steps, workers = make_balance_plan(frame, batch_size, n_gpu)

        assert len(balanced) == len(steps) == len(workers)
        assert steps[:full_size] == [0] * full_size
        assert workers[:full_size] == [
            rank for rank in range(n_gpu) for _ in range(batch_size)
        ]
        assert balanced["event"].tolist()[:len(frame)] == frame["event"].tolist()
        if not leftover:
            assert len(balanced) == full_size
            continue
        counts = Counter(workers[full_size:])
        assert set(counts) == set(range(n_gpu))
        assert max(counts.values()) - min(counts.values()) <= 1
        assert len(balanced) == full_size + max(leftover, n_gpu)
        assert steps[full_size:] == [1] * max(leftover, n_gpu)
