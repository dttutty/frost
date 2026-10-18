import numpy as np

from frost.launch.gpu_topology import (
    _parse_nvidia_smi_topology,
    detect_gpu_topology,
)
from frost.runtime.host._plan_router import build_gpu_distance_matrix


def test_build_gpu_distance_matrix_three_levels() -> None:
    topo = [[[0, 1], [2]], [[3]]]
    got = build_gpu_distance_matrix(topo)
    want = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 2, 0, 3],
            [3, 3, 3, 0],
        ]
    )
    np.testing.assert_array_equal(got, want)


def test_parse_nvidia_smi_topology_table() -> None:
    topo_output = """
\tGPU0\tGPU1\tGPU2\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\tX\tNV4\tNODE\t0-15\t0\tN/A
GPU1\tNV4\tX\tNODE\t0-15\t0\tN/A
GPU2\tNODE\tNODE\tX\t16-31\t1\tN/A

Legend:
"""
    links, numa = _parse_nvidia_smi_topology(topo_output)
    assert links[0][1] == "NV4"
    assert links[0][2] == "NODE"
    assert numa == {0: 0, 1: 0, 2: 1}


def test_parse_nvidia_smi_topology_with_nic_columns() -> None:
    topo_output = """
\tGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\tX\tNV4\tNODE\t0-15\t0\tN/A
GPU1\tNV4\tX\tPIX\t0-15\t0\tN/A
NIC0\tNODE\tPIX\tX

Legend:
"""
    _, numa = _parse_nvidia_smi_topology(topo_output)
    assert numa == {0: 0, 1: 0}


def test_detect_gpu_topology_groups_nvlink_then_numa(monkeypatch) -> None:
    topo_output = """
\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\tX\tNV4\tNODE\tSYS\t0-15\t0\tN/A
GPU1\tNV4\tX\tNODE\tSYS\t0-15\t0\tN/A
GPU2\tNODE\tNODE\tX\tPIX\t0-15\t0\tN/A
GPU3\tSYS\tSYS\tPIX\tX\t16-31\t1\tN/A

Legend:
"""
    inventory_output = """
0, GPU-0000, 00000000:3B:00.0
1, GPU-0001, 00000000:5E:00.0
2, GPU-0002, 00000000:86:00.0
3, GPU-0003, 00000000:AF:00.0
"""

    def fake_run_nvidia_smi(args: list[str]) -> str | None:
        if args == ["topo", "-m"]:
            return topo_output
        if args == ["--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"]:
            return inventory_output
        raise AssertionError(args)

    monkeypatch.setattr(
        "frost.launch.gpu_topology._run_nvidia_smi", fake_run_nvidia_smi
    )
    topo = detect_gpu_topology(n_gpu=4, cuda_visible_devices="0,1,2,3")
    dist = build_gpu_distance_matrix(topo)
    want = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 2, 0, 3],
            [3, 3, 3, 0],
        ]
    )
    np.testing.assert_array_equal(dist, want)
