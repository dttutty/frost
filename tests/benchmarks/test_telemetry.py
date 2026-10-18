from pathlib import Path

from frost_bench.telemetry import parse_nvidia_smi_samples, summarize_utilization


def test_nvidia_smi_parser_and_summary(tmp_path: Path) -> None:
    path = tmp_path / "samples.csv"
    path.write_text(
        "2026/07/29 12:00:00.000, 0, 25, 10\n"
        "2026/07/29 12:00:00.050, 0, 75, 30\n"
        "2026/07/29 12:00:00.100, 0, 999, 20\n",
        encoding="utf-8",
    )
    rows = parse_nvidia_smi_samples(path)
    assert len(rows) == 2
    summary = summarize_utilization(rows)
    assert summary["gpu_util_mean"] == 50.0
    assert summary["mem_util_max"] == 30.0
