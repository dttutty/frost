from pathlib import Path

import numpy as np

from frost.launch.config import (
    CAWN_ENCODER_TARGET,
    DEFAULT_NO_MEMORY_QUEUE_SIZE,
    DYGLIB_ALIGNED_FEATURE_DIM,
    DYGFORMER_ENCODER_TARGET,
    LINK_PREDICTOR_TARGET,
    PAIR_AWARE_PREDICTOR_TARGET,
    TCL_ENCODER_TARGET,
    TCL_PAIR_AWARE_PREDICTOR_TARGET,
    resolve_dygformer_dummy_edge_dim,
    resolve_no_memory_queue_size,
    resolve_predictor_cfg,
    resolve_tgat_dummy_node_dim,
    resolve_tgat_dummy_edge_dim,
    resolve_time_scaling_factor,
)


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


class TestResolvePredictorCfg:
    def test_switches_default_link_predictor_for_pair_aware_encoders(self):
        for encoder_target in (DYGFORMER_ENCODER_TARGET, CAWN_ENCODER_TARGET):
            resolved, switched = resolve_predictor_cfg(
                encoder_target,
                {"_target_": LINK_PREDICTOR_TARGET, "d_in": 172, "n_neg": 1},
            )

            assert switched is True
            assert resolved["_target_"] == PAIR_AWARE_PREDICTOR_TARGET

    def test_keeps_custom_predictor_for_pair_aware_encoders(self):
        resolved, switched = resolve_predictor_cfg(
            CAWN_ENCODER_TARGET,
            {
                "_target_": PAIR_AWARE_PREDICTOR_TARGET,
                "d_in": 172,
                "n_neg": 1,
            },
        )

        assert switched is False
        assert resolved["_target_"] == PAIR_AWARE_PREDICTOR_TARGET

    def test_switches_tcl_to_merge_layer_predictor(self):
        resolved, switched = resolve_predictor_cfg(
            TCL_ENCODER_TARGET,
            {"_target_": LINK_PREDICTOR_TARGET, "d_in": 172, "n_neg": 1},
        )

        assert switched is True
        assert resolved["_target_"] == TCL_PAIR_AWARE_PREDICTOR_TARGET

    def test_keeps_default_link_predictor_for_non_pair_aware_encoders(self):
        resolved, switched = resolve_predictor_cfg(
            "frost.models.encoder.tgat.TGAT",
            {"_target_": LINK_PREDICTOR_TARGET, "d_in": 172, "n_neg": 1},
        )

        assert switched is False
        assert resolved["_target_"] == LINK_PREDICTOR_TARGET


class TestResolveNoMemoryQueueSize:
    def test_uses_default_queue_depth_when_omitted(self):
        assert resolve_no_memory_queue_size(None) == DEFAULT_NO_MEMORY_QUEUE_SIZE

    def test_accepts_explicit_positive_queue_depth(self):
        assert resolve_no_memory_queue_size(6) == 6

    def test_rejects_non_positive_queue_depth(self):
        import pytest

        with pytest.raises(ValueError, match="no_memory_queue_size must be >= 1"):
            resolve_no_memory_queue_size(0)


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
