import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "log_gpu_users.py"
SPEC = importlib.util.spec_from_file_location("log_gpu_users", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
log_gpu_users = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = log_gpu_users
SPEC.loader.exec_module(log_gpu_users)


def test_write_snapshot_handles_unknown_gpu_index(tmp_path: Path) -> None:
    output_path = tmp_path / "gpu-users.log"
    processes = [
        log_gpu_users.GpuProcess(
            gpu_index="?",
            gpu_uuid="GPU-unknown",
            pid="30",
            username="alice",
            process_name="unknown-job",
            used_memory_mib="100",
        ),
        log_gpu_users.GpuProcess(
            gpu_index="2",
            gpu_uuid="GPU-two",
            pid="20",
            username="alice",
            process_name="second-job",
            used_memory_mib="200",
        ),
        log_gpu_users.GpuProcess(
            gpu_index="0",
            gpu_uuid="GPU-zero",
            pid="10",
            username="alice",
            process_name="first-job",
            used_memory_mib="300",
        ),
    ]

    log_gpu_users.write_snapshot(
        output_path,
        current_user="alice",
        processes=processes,
    )

    process_lines = [
        line
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("gpu=")
    ]
    assert [line.split()[0] for line in process_lines] == ["gpu=0", "gpu=2", "gpu=?"]
