from __future__ import annotations

from pathlib import Path

from frost.launch import affinity


def _write_cpu_topology(
    root: Path, *, cpu: int, package: int, core: int
) -> None:
    topology = root / f"cpu{cpu}" / "topology"
    topology.mkdir(parents=True)
    (topology / "physical_package_id").write_text(
        f"{package}\n", encoding="utf-8"
    )
    (topology / "core_id").write_text(f"{core}\n", encoding="utf-8")


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


def test_count_physical_cores_deduplicates_smt_siblings(tmp_path: Path) -> None:
    for cpu in range(8):
        _write_cpu_topology(tmp_path, cpu=cpu, package=0, core=cpu % 4)

    count, exact = affinity.count_physical_cores(
        set(range(8)), cpu_sysfs_root=tmp_path
    )

    assert count == 4
    assert exact


def test_auto_worker_threads_divide_gpu_local_physical_cores(
    tmp_path: Path, monkeypatch
) -> None:
    node_root = tmp_path / "nodes"
    cpu_root = tmp_path / "cpus"
    node = node_root / "node0"
    node.mkdir(parents=True)
    (node / "cpulist").write_text("0-15\n", encoding="utf-8")
    for cpu in range(16):
        _write_cpu_topology(cpu_root, cpu=cpu, package=0, core=cpu % 8)
    monkeypatch.setattr(
        affinity,
        "detect_visible_gpu_numa_nodes",
        lambda *_args, **_kwargs: [0, 0, 0, 0],
    )
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _pid: set(range(16)))

    threads, reason = affinity.resolve_auto_process_threads(
        n_gpu=4,
        local_rank=2,
        numa_policy="auto",
        cuda_visible_devices="0,1,2,3",
        node_sysfs_root=node_root,
        cpu_sysfs_root=cpu_root,
    )

    assert threads == 2
    assert "8 physical cores" in reason
    assert "4 worker(s)" in reason


def test_auto_host_threads_are_resolved_separately(tmp_path: Path, monkeypatch) -> None:
    for cpu in range(80):
        _write_cpu_topology(tmp_path, cpu=cpu, package=cpu // 40, core=cpu % 40)
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _pid: set(range(80)))

    threads, reason = affinity.resolve_auto_process_threads(
        n_gpu=4,
        local_rank=4,
        numa_policy="auto",
        cuda_visible_devices="0,1,2,3",
        cpu_sysfs_root=tmp_path,
    )

    assert threads == 32
    assert "host rank" in reason


def test_configure_threads_replaces_only_launcher_managed_values(
    tmp_path: Path, monkeypatch
) -> None:
    for cpu in range(8):
        _write_cpu_topology(tmp_path, cpu=cpu, package=0, core=cpu)
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(
        affinity,
        "detect_visible_gpu_numa_nodes",
        lambda *_args, **_kwargs: [None, None],
    )
    environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "7",
        affinity.AUTO_THREAD_ENV_MARKER: "OMP_NUM_THREADS",
    }

    status = affinity.configure_process_threads(
        "auto",
        n_gpu=2,
        local_rank=0,
        numa_policy="auto",
        cuda_visible_devices="0,1",
        environ=environment,
        cpu_sysfs_root=tmp_path,
    )

    assert environment["OMP_NUM_THREADS"] == "4"
    assert environment["MKL_NUM_THREADS"] == "7"
    assert affinity.AUTO_THREAD_ENV_MARKER not in environment
    assert status.omp_source == "auto"
    assert status.mkl_source == "environment"


def test_fixed_thread_config_does_not_override_environment(monkeypatch) -> None:
    environment = {"OMP_NUM_THREADS": "9"}
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _pid: {0, 1})

    status = affinity.configure_process_threads(
        6,
        n_gpu=1,
        local_rank=0,
        numa_policy="off",
        cuda_visible_devices="0",
        environ=environment,
    )

    assert environment["OMP_NUM_THREADS"] == "9"
    assert environment["MKL_NUM_THREADS"] == "6"
    assert status.omp_source == "environment"
    assert status.mkl_source == "config"


def test_mark_auto_thread_environment_preserves_user_values() -> None:
    environment = {"OMP_NUM_THREADS": "12"}

    managed = affinity.mark_auto_thread_environment(environment)

    assert managed == ("MKL_NUM_THREADS",)
    assert environment["OMP_NUM_THREADS"] == "12"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment[affinity.AUTO_THREAD_ENV_MARKER] == "MKL_NUM_THREADS"
