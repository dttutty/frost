from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from frost_bench.io import run_prepared
from frost_bench.spec import PreparedRun


@pytest.mark.parametrize("returncode", [0, 3])
def test_run_prepared_always_removes_transient_files(
    tmp_path: Path, returncode: int
) -> None:
    transient = tmp_path / "shared-memory-artifact"
    transient.write_text("temporary", encoding="utf-8")
    prepared = PreparedRun(
        system="test",
        command=(sys.executable, "-c", f"raise SystemExit({returncode})"),
        cwd=tmp_path,
        cleanup_paths=(transient,),
    )

    if returncode:
        with pytest.raises(subprocess.CalledProcessError):
            run_prepared(prepared, tmp_path / "run.log")
    else:
        run_prepared(prepared, tmp_path / "run.log")

    assert not transient.exists()
