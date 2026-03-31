from pathlib import Path

import numpy as np
import torch

from frost.shm import dataset_mgr as dm


class TestDatasetManager:
    def test_ignore_nfeat_on_disk_skips_node_feature_loading(
        self, tmp_path: Path, monkeypatch
    ):
        dataset_dir = tmp_path / "MOCK"
        dataset_dir.mkdir()
        np.save(dataset_dir / "node_features.npy", np.ones((4, 3), dtype=np.float32))

        monkeypatch.setattr(dm, "DATA_DIR", tmp_path)

        mgr = dm.DatasetManager(
            env=None,  # type: ignore[arg-type]
            run_id="test",
            dataset="MOCK",
            index_dtype=np.int64,
            time_dtype=np.int64,
            feature_dtype=torch.float32,
            ignore_nfeat_on_disk=True,
        )

        feat, dim, dtype = mgr._load_feature_to_shm(
            dataset="MOCK",
            feat_type="node",
            shm_name="nfeat",
            expected_rows=4,
        )

        assert feat is None
        assert dim == 0
        assert dtype is None
