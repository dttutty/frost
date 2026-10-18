from omegaconf import OmegaConf

from frost.launch.config import (
    DEFAULT_NO_MEMORY_QUEUE_SIZE,
    resolve_no_memory_queue_size,
    resolve_time_scaling_factor,
)
from frost.launch.factory import build_worker_modules
from frost.launch.plan import compile_execution_plan
from frost.launch.storage import resolve_batch_layout
from frost.models.mem_updater import TransformerMemoryUpdater
from frost.models.recipes import APANRecipe


def test_apan_runtime_only_config_builds_worker_modules():
    cfg = OmegaConf.create({"ndim": 0, "edim": 0})
    recipe = APANRecipe(
        embedding_dim=8,
        neighborhood={"fanout": [2]},
        encoder={
            "_target_": "frost.models.encoder.apan.APAN",
            "d_mem": 8,
        },
        state={
            "boxsize": 2,
            "mail_combine": "last",
            "updater": {
                "_target_": "frost.models.mem_updater.TransformerMemoryUpdater",
                "d_time": 4,
                "d_mem": 8,
                "boxsize": 2,
                "heads": 2,
                "dropout": 0.0,
            },
        },
        head={
            "_target_": "frost.models.header.AdditiveLinkHead",
            "d_in": 8,
        },
    )
    plan = compile_execution_plan(recipe)

    mem_updater, _, _, mailbox_router = build_worker_modules(
        cfg,
        recipe=recipe,
        plan=plan,
        d_mail=16,
    )

    assert isinstance(mem_updater, TransformerMemoryUpdater)
    assert mailbox_router is not None


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
