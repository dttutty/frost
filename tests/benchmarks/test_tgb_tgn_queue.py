from __future__ import annotations

from pathlib import Path

import pytest

from frost_bench.cli.tgb_tgn_queue import (
    _parser,
    _resolve_gpu_ids,
    build_tasks,
)


def test_queue_builds_slowest_official_jobs_first(tmp_path: Path) -> None:
    tasks = build_tasks(
        repo_root=tmp_path,
        datasets=["wiki", "flight"],
        systems=["frost", "official"],
        epochs=10,
        batch_size=4000,
        seed=1,
        output_dir=tmp_path / "out",
    )

    assert [task.name for task in tasks] == [
        "official-flight",
        "frost-flight",
        "official-wiki",
        "frost-wiki",
    ]
    official = tasks[0].command
    assert "--eval-batch-size" in official
    assert official[official.index("--eval-batch-size") + 1] == "4000"
    assert official[official.index("--loss-mode") + 1] == "corrected"
    frost = tasks[1].command
    assert "runtime.single_gpu_resident=on" in frost
    assert "neg_sampler=rnd_dst_then_tgb" in frost
    assert "neg_sampler.eval.n_neg=20" in frost
    assert "early_stop_tolerance=1e-06" in frost
    assert frost[1:3] == ["-m", "torch.distributed.run"]


def test_queue_uses_wiki_variable_width_protocol(tmp_path: Path) -> None:
    task = build_tasks(
        repo_root=tmp_path,
        datasets=["wiki"],
        systems=["frost"],
        epochs=2,
        batch_size=4000,
        seed=3,
        output_dir=tmp_path / "out",
    )[0]

    assert "neg_sampler=rnd_dst_then_tgb_wiki" in task.command
    assert "eval_macro_batch_size=4000" in task.command
    assert "neg_sampler.eval.n_neg=999" in task.command


def test_queue_uses_review_candidate_width(tmp_path: Path) -> None:
    task = build_tasks(
        repo_root=tmp_path,
        datasets=["review"],
        systems=["frost"],
        epochs=2,
        batch_size=4000,
        seed=3,
        output_dir=tmp_path / "out",
    )[0]

    assert "neg_sampler.eval.n_neg=100" in task.command


def test_queue_accepts_official_five_seed_protocol() -> None:
    args = _parser().parse_args(
        [
            "--datasets",
            "wiki",
            "--seeds",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--output-dir",
            "out",
        ]
    )

    assert args.seeds == [1, 2, 3, 4, 5]


def test_queue_prefixes_multi_seed_task_names(tmp_path: Path) -> None:
    task = build_tasks(
        repo_root=tmp_path,
        datasets=["wiki"],
        systems=["official"],
        epochs=30,
        batch_size=200,
        seed=5,
        output_dir=tmp_path / "out",
        name_prefix="seed-5-",
    )[0]

    assert task.name == "seed-5-official-wiki"
    assert task.command[task.command.index("--seed") + 1] == "5"


def test_queue_preserves_physical_ids_without_inherited_visibility() -> None:
    assert _resolve_gpu_ids(["1", "3"], None) == ["1", "3"]


def test_queue_maps_logical_ids_through_inherited_visibility() -> None:
    assert _resolve_gpu_ids(["0", "1"], "3,7") == ["3", "7"]
    assert _resolve_gpu_ids(["0"], "GPU-first,GPU-second") == ["GPU-first"]


def test_queue_accepts_visible_nonnumeric_device_tokens() -> None:
    assert _resolve_gpu_ids(["GPU-second"], "GPU-first,GPU-second") == [
        "GPU-second"
    ]


@pytest.mark.parametrize(
    ("requested", "visible", "match"),
    [
        (["1"], "3", "logical GPU index 1"),
        (["GPU-other"], "GPU-only", "outside inherited"),
        (["0"], "", "is empty"),
        (["0", "0"], "3,7", "duplicate devices"),
    ],
)
def test_queue_rejects_gpu_ids_outside_inherited_visibility(
    requested: list[str], visible: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _resolve_gpu_ids(requested, visible)
