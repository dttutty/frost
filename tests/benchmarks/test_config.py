from frost_bench.config import (
    PRESETS,
    build_dyglib_hydra_overrides,
    load_dataset_default_macro_batch_size,
    load_dyglib_parity_settings,
)


def test_presets_are_repository_owned_and_complete() -> None:
    assert set(PRESETS) == {
        "tgn",
        "jodie",
        "apan",
        "tgat",
        "graph_mixer",
        "tcl",
        "dygformer",
        "cawn",
        "tpnet",
    }


def test_dataset_defaults_preserve_memory_batch_semantics() -> None:
    assert load_dataset_default_macro_batch_size("LASTFM", "tgn") == 4000
    assert load_dataset_default_macro_batch_size("MOOC", "apan") == 1200
    assert load_dataset_default_macro_batch_size("LASTFM", "tgat") == 600


def test_jodie_uses_one_fifth_of_each_dataset_default() -> None:
    assert load_dataset_default_macro_batch_size("WIKI", "JODIE") == 120
    assert load_dataset_default_macro_batch_size("MOOC", "jodie") == 240
    assert load_dataset_default_macro_batch_size("REDDIT", "jodie") == 400
    assert load_dataset_default_macro_batch_size("LASTFM", "jodie") == 800


def test_dyglib_parity_settings_are_dataset_specific() -> None:
    dygformer = load_dyglib_parity_settings(
        preset_name="dygformer", dataset_name="REDDIT"
    )
    assert dygformer["max_input_sequence_length"] == 64
    assert dygformer["patch_size"] == 2
    assert dygformer["dropout"] == 0.2

    cawn = load_dyglib_parity_settings(preset_name="cawn", dataset_name="CANPARL")
    assert cawn["num_neighbors"] == 128
    assert cawn["dropout"] == 0.0
    assert cawn["sample_neighbor_strategy"] == "time_interval_aware"


def test_dyglib_parity_translates_to_hydra_overrides() -> None:
    overrides = build_dyglib_hydra_overrides("graph_mixer", "UNVOTE")
    assert "model.neighborhood.fanout=[2000]" in overrides
    assert "model.encoder.num_tokens=20" in overrides
    assert "ngh_sampler.strategy=uniform" in overrides
    assert "dropout=0.0" in overrides
