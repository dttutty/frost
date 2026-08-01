from __future__ import annotations

from pathlib import Path

from frost.launch import affinity


def test_parse_cpu_list() -> None:
    assert affinity.parse_cpu_list("0-2,8,10-11\n") == {0, 1, 2, 8, 10, 11}


def test_auto_policy_selects_common_gpu_numa(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [1, 1]
    )
    assert affinity.resolve_numa_node(
        "auto", n_gpu=2, cuda_visible_devices="2,3"
    ) == (1, "auto-selected common visible-GPU NUMA node")


def test_auto_policy_skips_mixed_gpu_numa(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [0, 1]
    )
    node, reason = affinity.resolve_numa_node(
        "auto", n_gpu=2, cuda_visible_devices="0,1"
    )
    assert node is None
    assert "span NUMA nodes" in reason


def test_configure_process_affinity_intersects_cpuset(tmp_path: Path, monkeypatch) -> None:
    node_dir = tmp_path / "node0"
    node_dir.mkdir()
    (node_dir / "cpulist").write_text("0-3,8-9\n", encoding="utf-8")
    monkeypatch.setattr(affinity, "resolve_numa_node", lambda *_args, **_kwargs: (0, "auto"))
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _pid: {2, 3, 4, 8})
    selected: list[set[int]] = []
    monkeypatch.setattr(
        affinity.os, "sched_setaffinity", lambda _pid, cpus: selected.append(set(cpus))
    )
    memory_nodes: list[int] = []
    monkeypatch.setattr(
        affinity, "_bind_memory_to_node", lambda node: memory_nodes.append(node)
    )

    status = affinity.configure_process_affinity(
        "auto", n_gpu=2, sysfs_root=tmp_path
    )

    assert selected == [{2, 3, 8}]
    assert memory_nodes == [0]
    assert status.node == 0
    assert status.cpu_count == 3
    assert status.memory_bound


def test_off_policy_does_not_change_affinity(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity.os,
        "sched_setaffinity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not bind")),
    )
    status = affinity.configure_process_affinity("off", n_gpu=4)
    assert status.node is None
    assert "disabled" in status.reason
