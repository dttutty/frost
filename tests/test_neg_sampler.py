"""Tests for PrecomputedNegSampler, RandomNegLinkSampler, and precomputed negative generation."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from frost.sampling.neg_sampler import (
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
    def test_generate_precomputed_negative_file_excludes_seen_neighbors(self, tmp_path: Path):
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
        negatives = [int(line) for line in output_path.read_text().splitlines()]
        assert len(negatives) == 8

        seen_neighbors: dict[int, set[int]] = {}
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
            seen = seen_neighbors.setdefault(src, set())
            assert neg != dst
            assert neg not in seen
            seen.add(dst)

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

    def test_random_sampler_dst_partition_mode_uses_only_dst_nodes(self, tmp_path: Path):
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
