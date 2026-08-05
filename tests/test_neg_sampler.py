"""Tests for PrecomputedNegSampler, RandomNegLinkSampler, and precomputed negative generation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from frost.neg_sampler import (
    HistoricalNegLinkSampler,
    InductiveNegLinkSampler,
    PRECOMPUTED_NEGATIVE_FILENAME,
    PrecomputedNegSampler,
    RandomNegLinkSampler,
    generate_precomputed_negative_file,
)


def _write_edges_csv(path: Path, rows: list[tuple[int, int, int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eid", "src", "dst", "time", "default_split"])
        for eid, (src, dst, ts) in enumerate(rows, start=1):
            writer.writerow([eid, src, dst, ts, 0])


class TestPrecomputedNegSampler:
    def test_generate_precomputed_negative_file_uses_dst_partition(
        self, tmp_path: Path
    ):
        dataset_dir = tmp_path / "MOCK"
        dataset_dir.mkdir()
        edges_path = dataset_dir / "edges.csv"
        _write_edges_csv(
            edges_path,
            [
                (0, 10, 0),
                (0, 11, 1),
                (0, 12, 2),
                (0, 13, 3),
                (1, 10, 3),
                (1, 11, 4),
                (2, 12, 5),
                (3, 14, 6),
            ],
        )

        output_path = generate_precomputed_negative_file(edges_path, seed=7)
        generated = pd.read_csv(output_path)
        negatives = generated["neg_dst"].tolist()
        assert len(negatives) == 8
        assert generated["eid"].tolist() == list(range(1, 9))
        assert set(negatives).issubset({10, 11, 12, 13, 14})
        rows = [
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (1, 10),
            (1, 11),
            (2, 12),
            (3, 14),
        ]
        for neg, (src, dst) in zip(negatives, rows):
            assert neg != dst

    def test_generate_precomputed_negative_file_uses_all_nodes(self, tmp_path: Path):
        dataset_dir = tmp_path / "MOCK"
        dataset_dir.mkdir()
        edges_path = dataset_dir / "edges.csv"
        _write_edges_csv(
            edges_path,
            [(0, 4, 0), (1, 4, 1), (2, 4, 2), (3, 4, 3)],
        )
        output_path = dataset_dir / "neg_all.csv"

        generate_precomputed_negative_file(
            edges_path,
            output_path=output_path,
            seed=7,
            candidate_mode="all",
        )

        negatives = pd.read_csv(output_path)["neg_dst"]
        assert set(negatives).issubset(set(range(5)))
        assert any(negative != 4 for negative in negatives)

        sampler = PrecomputedNegSampler(
            n_nodes=5,
            n_neg=1,
            dataset="MOCK",
            data_root=tmp_path,
            neg_file="neg_all.csv",
            candidate_mode="all",
        )
        sampler.init_from_df(pd.read_csv(edges_path))
        assert torch.equal(
            sampler.sample(torch.tensor([0]), torch.tensor([1])),
            torch.tensor([negatives.iloc[0]]),
        )

    def test_generate_precomputed_negative_file_synthesizes_missing_eid(
        self, tmp_path: Path
    ):
        with_eid_dir = tmp_path / "WITH_EID"
        without_eid_dir = tmp_path / "WITHOUT_EID"
        with_eid_dir.mkdir()
        without_eid_dir.mkdir()
        rows = [(0, 10, 0), (0, 11, 1), (1, 12, 2), (2, 10, 3)]
        with_eid_path = with_eid_dir / "edges.csv"
        without_eid_path = without_eid_dir / "edges.csv"
        _write_edges_csv(with_eid_path, rows)
        pd.read_csv(with_eid_path).drop(columns="eid").to_csv(
            without_eid_path, index=False
        )

        with_eid = pd.read_csv(
            generate_precomputed_negative_file(with_eid_path, seed=7)
        )
        synthesized = pd.read_csv(
            generate_precomputed_negative_file(without_eid_path, seed=7)
        )

        pd.testing.assert_frame_equal(synthesized, with_eid)

    def test_sample_reads_precomputed_negatives_by_eid(self, tmp_path: Path):
        dataset_dir = tmp_path / "MOCK"
        dataset_dir.mkdir()
        edges_path = dataset_dir / "edges.csv"
        _write_edges_csv(
            edges_path,
            [
                (0, 4, 0),
                (0, 5, 1),
                (0, 6, 2),
                (1, 4, 2),
                (2, 7, 3),
            ],
        )
        generate_precomputed_negative_file(edges_path, seed=3)

        sampler = PrecomputedNegSampler(
            n_nodes=8,
            n_neg=1,
            dataset="MOCK",
            data_root=tmp_path,
        )
        sampler.init_from_df(pd.read_csv(edges_path))

        src = torch.tensor([123, 456])
        eid = torch.tensor([3, 1])
        sampled = sampler.sample(src, eid)

        expected = torch.tensor(
            [
                sampler.precomputed_negatives[2].item(),
                sampler.precomputed_negatives[0].item(),
            ],
            dtype=torch.long,
        )
        assert torch.equal(sampled, expected)

    def test_requires_single_negative_per_edge(self, tmp_path: Path):
        dataset_dir = tmp_path / "MOCK"
        dataset_dir.mkdir()
        (dataset_dir / PRECOMPUTED_NEGATIVE_FILENAME).write_text("1\n")

        try:
            PrecomputedNegSampler(
                n_nodes=5,
                n_neg=2,
                dataset="MOCK",
                data_root=tmp_path,
            )
        except ValueError as exc:
            assert "n_neg=1" in str(exc)
        else:
            raise AssertionError("Expected ValueError for n_neg != 1")

    def test_random_sampler_all_mode_samples_from_full_node_range(self, tmp_path: Path):
        dataset_dir = tmp_path / "MOCK"
        dataset_dir.mkdir()
        _write_edges_csv(
            dataset_dir / "edges.csv",
            [
                (0, 10, 0),
                (1, 11, 1),
            ],
        )

        sampler = RandomNegLinkSampler(
            n_nodes=6,
            n_neg=2,
            dataset="MOCK",
            data_root=tmp_path,
            candidate_mode="all",
            seed=123,
        )

        src = torch.tensor([7, 8, 9])
        eid = torch.tensor([1, 2, 3])
        sampled = sampler.sample(src, eid)

        assert sampled.shape == (6,)
        assert sampled.dtype == torch.long
        assert int(sampled.min().item()) >= 0
        assert int(sampled.max().item()) < 6

    def test_random_sampler_dst_partition_mode_uses_only_dst_nodes(
        self, tmp_path: Path
    ):
        dataset_dir = tmp_path / "mock"
        dataset_dir.mkdir()
        _write_edges_csv(
            dataset_dir / "edges.csv",
            [
                (10, 100, 0),
                (11, 101, 1),
                (12, 101, 2),
                (13, 103, 3),
            ],
        )

        sampler = RandomNegLinkSampler(
            n_nodes=200,
            n_neg=3,
            dataset="MOCK",
            data_root=tmp_path,
            candidate_mode="dst_partition",
            seed=7,
        )

        src = torch.tensor([1, 2])
        eid = torch.tensor([1, 2])
        sampled = sampler.sample(src, eid)

        assert sampled.shape == (6,)
        assert set(sampled.tolist()).issubset({100, 101, 103})

    def test_random_sampler_uses_preloaded_dst_candidates_without_csv(
        self, tmp_path: Path
    ):
        sampler = RandomNegLinkSampler(
            n_nodes=200,
            n_neg=2,
            dataset="MISSING",
            data_root=tmp_path,
            candidate_mode="dst_partition",
            candidate_nodes=torch.tensor([100, 101, 103]),
            seed=7,
        )

        sampled = sampler.sample(torch.tensor([1, 2]), torch.tensor([1, 2]))

        assert sampled.shape == (4,)
        assert set(sampled.tolist()).issubset({100, 101, 103})

    def test_builds_sorted_dst_candidates_from_loaded_column(self):
        candidates = RandomNegLinkSampler.candidates_from_dst_nodes(
            np.array([103, 100, 101, 101, 103], dtype=np.int64), n_nodes=200
        )

        assert torch.equal(candidates, torch.tensor([100, 101, 103]))


def test_historical_sampler_keeps_source_and_uses_its_past_destinations() -> None:
    frame = pd.DataFrame(
        {
            "eid": [1, 2, 3, 4],
            "src": [1, 2, 1, 1],
            "dst": [10, 20, 11, 12],
            "time": [1, 1, 2, 3],
            "default_split": [0, 0, 0, 1],
        }
    )
    sampler = HistoricalNegLinkSampler(n_nodes=21, n_neg=4, dataset="MOCK", seed=7)
    sampler.init_from_df(frame)
    sampler.set_batch_context(
        batch_src=np.array([1]),
        batch_dst=np.array([12]),
        batch_start_time=3,
        batch_end_time=3,
    )

    sampled = sampler.sample(torch.tensor([1]), torch.tensor([4]))

    assert set(sampled.tolist()).issubset({10, 11})


def test_inductive_sampler_resolves_train_cutoff_and_uses_post_train_history() -> None:
    frame = pd.DataFrame(
        {
            "eid": [1, 2, 3],
            "src": [1, 1, 1],
            "dst": [10, 11, 12],
            "time": [1, 2, 3],
            "default_split": [0, 1, 2],
        }
    )
    sampler = InductiveNegLinkSampler(n_nodes=13, n_neg=3, dataset="MOCK", seed=7)
    sampler.init_from_df(frame)
    sampler.set_batch_context(
        batch_src=np.array([1]),
        batch_dst=np.array([12]),
        batch_start_time=3,
        batch_end_time=3,
    )

    sampled = sampler.sample(torch.tensor([1]), torch.tensor([3]))

    assert sampler.last_observed_time == 1
    assert sampled.tolist() == [11, 11, 11]
