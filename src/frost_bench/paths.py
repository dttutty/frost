from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
TGL_ROOT = THIRD_PARTY_ROOT / "tgl"
DYGLIB_ROOT = THIRD_PARTY_ROOT / "DyGLib"
PIPETGL_ROOT = THIRD_PARTY_ROOT / "PipeTGL"
SPEC_ROOT = REPO_ROOT / "benchmarks" / "specs"


def artifact_root() -> Path:
    """Return the single root used for generated benchmark artifacts."""

    configured = os.environ.get("FROST_ARTIFACT_ROOT")
    return Path(configured).expanduser() if configured else REPO_ROOT / "artifacts"
