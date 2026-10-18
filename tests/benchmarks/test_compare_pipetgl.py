from frost_bench.cli.compare_pipetgl import (
    attach_frost_epoch_wall_times,
    resolve_macro_batch_size,
)


def test_comparison_uses_dataset_default_macro_batch_size() -> None:
    assert resolve_macro_batch_size("WIKI", None) == 600
    assert resolve_macro_batch_size("MOOC", None) == 1200
    assert resolve_macro_batch_size("REDDIT", None) == 2000
    assert resolve_macro_batch_size("LASTFM", None) == 4000


def test_comparison_explicit_batch_size_wins() -> None:
    assert resolve_macro_batch_size("WIKI", 256) == 256


def test_frost_epoch_wall_times_map_runtime_zero_based_to_printed_one_based() -> None:
    rows: list[dict[str, float | int | str]] = [
        {"system": "frost", "epoch": 1},
        {"system": "frost", "epoch": 2},
    ]

    attach_frost_epoch_wall_times(rows, {0: 1.25, 1: 1.5})

    assert [row["epoch_wall_time_s"] for row in rows] == [1.25, 1.5]
