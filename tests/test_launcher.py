from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from frost.launch.config import (
    DEFAULT_NO_MEMORY_QUEUE_SIZE,
    DYGLIB_ALIGNED_FEATURE_DIM,
    resolve_dygformer_dummy_edge_dim,
    resolve_no_memory_queue_size,
    resolve_tgat_dummy_node_dim,
    resolve_tgat_dummy_edge_dim,
    resolve_time_scaling_factor,
)
from frost.launch.factory import build_worker_modules
from frost.launch.plan import DatasetSchema, compile_execution_plan
from frost.models.mem_updater import TransformerMemoryUpdater
from frost.models.recipes import APANRecipe
from frost.launch.storage import resolve_batch_layout


def test_apan_runtime_only_config_builds_worker_modules():
    cfg = OmegaConf.create(
        {
            "ndim": 0,
            "edim": 0,
            "loss_fn": {"_target_": "torch.nn.BCEWithLogitsLoss"},
        }
    )
    recipe = APANRecipe(
        embedding_dim=8,
        neighborhood={"fanout": [2]},
        encoder={
            "_target_": "frost.models.encoder.apan.APAN",
            "d_mem": 8,
            "n_neg": 1,
        },
        state={
            "boxsize": 2,
            "mail_combine": "last",
            "updater": {
                "_target_": "frost.models.mem_updater.TransformerMemoryUpdater",
                "d_time": 4,
                "d_mem": 8,
                "heads": 2,
                "dropout": 0.0,
            },
        },
        head={
            "_target_": "frost.models.header.AdditiveLinkHead",
            "d_in": 8,
        },
    )
    plan = compile_execution_plan(recipe, DatasetSchema("EXAMPLE", 10, 0, 0))

    mem_updater, _, _, postman, _ = build_worker_modules(
        cfg,
        recipe=recipe,
        plan=plan,
        d_mail=16,
        max_n_neg=1,
    )

    assert isinstance(mem_updater, TransformerMemoryUpdater)
    assert postman is not None


class TestResolveDyGFormerDummyEdgeDim:
    def test_uses_dyglib_alignment_floor_for_zero_feature_datasets(
        self, tmp_path: Path
    ):
        dataset_dir = tmp_path / "lastfm"
        dataset_dir.mkdir()
        np.save(dataset_dir / "ml_lastfm.npy", np.zeros((8, 1), dtype=np.float32))

        resolved = resolve_dygformer_dummy_edge_dim(
            dataset="LASTFM",
            requested_dim=None,
            processed_data_root=tmp_path,
        )

        assert resolved == DYGLIB_ALIGNED_FEATURE_DIM

    def test_prefers_larger_processed_edge_feature_width(self, tmp_path: Path):
        dataset_dir = tmp_path / "mock"
        dataset_dir.mkdir()
        np.save(dataset_dir / "ml_mock.npy", np.zeros((8, 256), dtype=np.float32))

        resolved = resolve_dygformer_dummy_edge_dim(
            dataset="MOCK",
            requested_dim=None,
            processed_data_root=tmp_path,
        )

        assert resolved == 256

    def test_explicit_config_override_wins(self, tmp_path: Path):
        resolved = resolve_dygformer_dummy_edge_dim(
            dataset="LASTFM",
            requested_dim=64,
            processed_data_root=tmp_path,
        )

        assert resolved == 64


class TestResolveTGATDummyEdgeDim:
    def test_uses_dyglib_alignment_floor_for_zero_feature_datasets(
        self, tmp_path: Path
    ):
        dataset_dir = tmp_path / "lastfm"
        dataset_dir.mkdir()
        np.save(dataset_dir / "ml_lastfm.npy", np.zeros((8, 1), dtype=np.float32))

        resolved = resolve_tgat_dummy_edge_dim(
            dataset="LASTFM",
            requested_dim=None,
            processed_data_root=tmp_path,
        )

        assert resolved == DYGLIB_ALIGNED_FEATURE_DIM

    def test_explicit_config_override_wins(self, tmp_path: Path):
        resolved = resolve_tgat_dummy_edge_dim(
            dataset="LASTFM",
            requested_dim=32,
            processed_data_root=tmp_path,
        )

        assert resolved == 32


class TestResolveTGATDummyNodeDim:
    def test_uses_dyglib_alignment_floor_for_zero_feature_datasets(
        self, tmp_path: Path
    ):
        dataset_dir = tmp_path / "lastfm"
        dataset_dir.mkdir()
        np.save(dataset_dir / "ml_lastfm_node.npy", np.zeros((8, 1), dtype=np.float32))

        resolved = resolve_tgat_dummy_node_dim(
            dataset="LASTFM",
            requested_dim=None,
            processed_data_root=tmp_path,
        )

        assert resolved == DYGLIB_ALIGNED_FEATURE_DIM

    def test_explicit_config_override_wins(self, tmp_path: Path):
        resolved = resolve_tgat_dummy_node_dim(
            dataset="LASTFM",
            requested_dim=48,
            processed_data_root=tmp_path,
        )

        assert resolved == 48


class TestResolveNoMemoryQueueSize:
    def test_uses_default_queue_depth_when_omitted(self):
        assert resolve_no_memory_queue_size(None) == DEFAULT_NO_MEMORY_QUEUE_SIZE

    def test_accepts_explicit_positive_queue_depth(self):
        assert resolve_no_memory_queue_size(6) == 6

    def test_rejects_non_positive_queue_depth(self):
        import pytest

        with pytest.raises(ValueError, match="no_memory_queue_size must be >= 1"):
            resolve_no_memory_queue_size(0)


class TestResolveBatchLayout:
    def test_resolves_train_and_eval_batch_sizes(self):
        layout = resolve_batch_layout(
            macro_batch_size=4000,
            eval_macro_batch_size=1200,
            n_gpu=2,
        )

        assert layout.train_per_gpu == 2000
        assert layout.eval_macro == 1200
        assert layout.eval_per_gpu == 600

    def test_eval_defaults_to_training_macro_batch_size(self):
        layout = resolve_batch_layout(
            macro_batch_size=600,
            eval_macro_batch_size=None,
            n_gpu=2,
        )

        assert layout.eval_macro == 600
        assert layout.eval_per_gpu == 300

    def test_rejects_non_divisible_batch_sizes(self):
        import pytest

        with pytest.raises(ValueError, match="macro_batch_size"):
            resolve_batch_layout(
                macro_batch_size=601,
                eval_macro_batch_size=None,
                n_gpu=2,
            )


class TestResolveTimeScalingFactor:
    def test_uses_default_when_omitted(self):
        assert resolve_time_scaling_factor(None) == 0.0

    def test_accepts_finite_float(self):
        assert resolve_time_scaling_factor(1.0e-6) == 1.0e-6

    def test_rejects_non_numeric_values(self):
        import pytest

        with pytest.raises(TypeError, match="time_scaling_factor must be a float"):
            resolve_time_scaling_factor("oops")

    def test_rejects_non_finite_values(self):
        import pytest

        with pytest.raises(ValueError, match="time_scaling_factor must be finite"):
            resolve_time_scaling_factor(float("inf"))
