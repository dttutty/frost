from __future__ import annotations

import numpy as np
import torch

from frost.runtime.worker._batch_convergence import BatchConvergenceRecorder


def test_disabled_batch_convergence_recorder_is_a_noop(tmp_path) -> None:
    recorder = BatchConvergenceRecorder(system="frost", rank=0, output_dir=None)
    recorder.record(events=4, loss=torch.tensor(1.25))

    assert recorder.flush_epoch(1) is None
    assert list(tmp_path.iterdir()) == []


def test_batch_convergence_recorder_writes_one_curve_per_epoch(tmp_path) -> None:
    recorder = BatchConvergenceRecorder(
        system="frost", rank=1, output_dir=tmp_path
    )
    recorder.record(events=4, loss=torch.tensor(1.25))
    recorder.record(events=3, loss=torch.tensor(0.75))

    destination = recorder.flush_epoch(2)

    assert destination == tmp_path / "frost_rank1_epoch002.npz"
    assert destination is not None
    with np.load(destination) as curve:
        assert curve["system"].item() == "frost"
        assert curve["rank"].item() == 1
        assert curve["epoch"].item() == 2
        np.testing.assert_array_equal(curve["batch"], [0, 1])
        np.testing.assert_array_equal(curve["events"], [4, 3])
        np.testing.assert_allclose(curve["loss"], [1.25, 0.75])

    assert recorder.flush_epoch(3) is None
