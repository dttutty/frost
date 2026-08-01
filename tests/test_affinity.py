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


def test_auto_policy_places_each_worker_with_its_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [0, 1]
    )
    assert affinity.resolve_numa_node(
        "auto", n_gpu=2, local_rank=1, cuda_visible_devices="0,1"
    ) == (1, "auto-selected visible GPU 1 NUMA node")


def test_auto_policy_places_workers_when_gpus_share_numa(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [0, 0]
    )

    node, reason = affinity.resolve_numa_node(
        "auto", n_gpu=2, local_rank=0, cuda_visible_devices="0,1"
    )

    assert node == 0
    assert "visible GPU 0 NUMA node" in reason


def test_auto_policy_leaves_host_on_full_cpuset(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [0, 0]
    )
    node, reason = affinity.resolve_numa_node(
        "auto", n_gpu=2, local_rank=2, cuda_visible_devices="0,1"
    )
    assert node is None
    assert "host rank keeps the full CPU cpuset" in reason


def test_shared_memory_nodes_follow_gpus_only_in_auto(monkeypatch) -> None:
    detected: list[int | None] = [0, 1, 2]
    monkeypatch.setattr(
        affinity,
        "detect_visible_gpu_numa_nodes",
        lambda *_args, **_kwargs: detected,
    )
    assert affinity.resolve_gpu_shm_numa_nodes(
        "auto", n_gpu=3, cuda_visible_devices="0,1,2"
    ) == [0, 1, 2]
    assert affinity.resolve_gpu_shm_numa_nodes(
        "off", n_gpu=3, cuda_visible_devices="0,1,2"
    ) == [None, None, None]
    assert affinity.resolve_gpu_shm_numa_nodes(
        1, n_gpu=3, cuda_visible_devices="0,1,2"
    ) == [None, None, None]

    detected[:] = [0, 0, 0]
    assert affinity.resolve_gpu_shm_numa_nodes(
        "auto", n_gpu=3, cuda_visible_devices="0,1,2"
    ) == [0, 0, 0]

    detected[:] = [0, None, 2]
    assert affinity.resolve_gpu_shm_numa_nodes(
        "auto", n_gpu=3, cuda_visible_devices="0,1,2"
    ) == [None, None, None]


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
    assert status.cpu_bound
    assert status.memory_bound


def test_auto_worker_binds_cpu_and_memory_to_gpu_node(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [0, 1]
    )
    node_dir = tmp_path / "node1"
    node_dir.mkdir()
    (node_dir / "cpulist").write_text("2-3\n", encoding="utf-8")
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    selected: list[set[int]] = []
    monkeypatch.setattr(
        affinity.os,
        "sched_setaffinity",
        lambda _pid, cpus: selected.append(set(cpus)),
    )
    memory_nodes: list[int] = []
    monkeypatch.setattr(
        affinity, "_bind_memory_to_node", lambda node: memory_nodes.append(node)
    )

    status = affinity.configure_process_affinity(
        "auto",
        n_gpu=2,
        local_rank=1,
        cuda_visible_devices="0,1",
        sysfs_root=tmp_path,
    )

    assert selected == [{2, 3}]
    assert memory_nodes == [1]
    assert status.node == 1
    assert status.cpu_count == 2
    assert status.cpu_bound
    assert status.memory_bound


def test_shared_gpu_node_requires_common_topology(monkeypatch) -> None:
    detected: list[int | None] = [0, 0]
    monkeypatch.setattr(
        affinity,
        "detect_visible_gpu_numa_nodes",
        lambda *_args, **_kwargs: detected,
    )
    assert affinity.resolve_shared_gpu_numa_node(
        "auto", n_gpu=2, cuda_visible_devices="0,1"
    ) == 0
    detected[:] = [0, 1]
    assert (
        affinity.resolve_shared_gpu_numa_node(
            "auto", n_gpu=2, cuda_visible_devices="0,1"
        )
        is None
    )


def test_runtime_plan_keeps_same_node_featureless_workload_broad(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [0, 0]
    )
    plan = affinity.resolve_numa_runtime_plan(
        "auto",
        n_gpu=2,
        cuda_visible_devices="0,1",
        has_static_features=False,
    )
    assert not plan.localize_workers
    assert plan.graph_ring_nodes == (None, None)
    assert plan.shared_state_node is None


def test_runtime_plan_localizes_same_node_feature_workload(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [0, 0]
    )
    plan = affinity.resolve_numa_runtime_plan(
        "auto",
        n_gpu=2,
        cuda_visible_devices="0,1",
        has_static_features=True,
    )
    assert plan.localize_workers
    assert plan.graph_ring_nodes == (0, 0)
    assert plan.shared_state_node == 0


def test_runtime_plan_localizes_cross_numa_featureless_workload(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [0, 1]
    )
    plan = affinity.resolve_numa_runtime_plan(
        "auto",
        n_gpu=2,
        cuda_visible_devices="0,1",
        has_static_features=False,
    )
    assert plan.localize_workers
    assert plan.graph_ring_nodes == (0, 1)
    assert plan.shared_state_node is None


def test_auto_worker_affinity_can_remain_broad(monkeypatch) -> None:
    monkeypatch.setattr(
        affinity, "detect_visible_gpu_numa_nodes", lambda *_args, **_kwargs: [0, 0]
    )
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr(
        affinity.os,
        "sched_setaffinity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must stay broad")),
    )
    status = affinity.configure_process_affinity(
        "auto",
        n_gpu=2,
        local_rank=0,
        cuda_visible_devices="0,1",
        apply_auto_worker=False,
    )
    assert status.node is None
    assert not status.cpu_bound
    assert "full CPU cpuset" in status.reason


def test_off_policy_does_not_change_affinity(monkeypatch) -> None:
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr(
        affinity.os,
        "sched_setaffinity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not bind")),
    )
    status = affinity.configure_process_affinity("off", n_gpu=4)
    assert status.node is None
    assert status.cpu_count == 4
    assert not status.cpu_bound
    assert "disabled" in status.reason
