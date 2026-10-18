"""Download official TGB link prediction datasets with aria2c."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from frost.data.paths import tgb_data_root


TGB_LINKPROP_DOCS_URL = "https://tgb.complexdatalab.com/docs/linkprop/"
_TGB_OBJECT_ROOT = (
    "https://object-arbutus.alliancecan.ca/swift/v1/"
    "14c95234f6cd4a21a47deafe20cce2a7/tgb"
)


@dataclass(frozen=True)
class TGBDownloadSpec:
    name: str
    version: int
    remote_archive: str
    md5: str
    extra_required_members: tuple[str, ...] = ()

    @property
    def directory_name(self) -> str:
        return self.name.replace("-", "_")

    @property
    def url(self) -> str:
        return f"{_TGB_OBJECT_ROOT}/{self.remote_archive}"

    @property
    def required_members(self) -> tuple[str, ...]:
        version = f"_v{self.version}" if self.version > 1 else ""
        return (
            f"{self.name}_edgelist{version}.csv",
            f"{self.name}_val_ns{version}.pkl",
            f"{self.name}_test_ns{version}.pkl",
            *self.extra_required_members,
        )


# Complete dynamic link property prediction catalog from the official TGB page.
TGB_LINK_DATASETS: dict[str, TGBDownloadSpec] = {
    spec.name: spec
    for spec in (
        TGBDownloadSpec(
            "tgbl-wiki",
            2,
            "tgbl-wiki-v2.zip",
            "aa921ce124011424c5c989652d890528",
        ),
        TGBDownloadSpec(
            "tgbl-review",
            2,
            "tgbl-review-v2.zip",
            "7cd24aab5052ba5d2e2b7656e9c57157",
        ),
        TGBDownloadSpec(
            "tgbl-coin",
            2,
            "tgbl-coin-v2.zip",
            "469f19f8990bca8bf2a1767117d2fb58",
        ),
        TGBDownloadSpec(
            "tgbl-comment",
            1,
            "tgbl-comment.zip",
            "75a376fa5f77939b1401216b7fa18fea",
        ),
        TGBDownloadSpec(
            "tgbl-flight",
            2,
            "tgbl-flight-v2.zip",
            "0a5a306481a793abf22ba701a1ef87aa",
            ("airport_node_feat_v2.csv",),
        ),
    )
}


def _normalize_dataset_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    for spec in TGB_LINK_DATASETS.values():
        if normalized in {spec.name, f"{spec.name}-v{spec.version}"}:
            return spec.name
    available = ", ".join(TGB_LINK_DATASETS)
    raise ValueError(f"Unknown TGB link dataset {value!r}; available: {available}")


def select_tgb_link_datasets(
    names: Iterable[str] = (), *, all_datasets: bool = False
) -> list[TGBDownloadSpec]:
    requested = list(names)
    if all_datasets or requested == ["all"]:
        if requested and requested != ["all"]:
            raise ValueError("Do not combine explicit dataset names with --all.")
        return list(TGB_LINK_DATASETS.values())
    if not requested:
        raise ValueError("Select one or more datasets, or pass --all.")
    normalized = list(
        dict.fromkeys(_normalize_dataset_name(name) for name in requested)
    )
    return [TGB_LINK_DATASETS[name] for name in normalized]


def _missing_members(path: Path, spec: TGBDownloadSpec) -> list[str]:
    return [name for name in spec.required_members if not (path / name).is_file()]


def _aria2_command(aria2c: str, spec: TGBDownloadSpec, archive_dir: Path) -> list[str]:
    return [
        aria2c,
        "--continue=true",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=10M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        f"--checksum=md5={spec.md5}",
        f"--dir={archive_dir}",
        f"--out={spec.remote_archive}",
        spec.url,
    ]


def fetch_tgb_link_dataset(
    spec: TGBDownloadSpec,
    *,
    root: Path | None = None,
    download_only: bool = False,
    aria2c: str | None = None,
) -> dict[str, object]:
    """Download one pinned official archive and optionally extract it."""
    resolved_root = (root or tgb_data_root()).expanduser().resolve()
    archive_dir = resolved_root / "downloads"
    target = resolved_root / "raw" / spec.directory_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    if target.is_dir() and not _missing_members(target, spec) and not download_only:
        return {
            "dataset": spec.name,
            "path": str(target),
            "status": "present",
            "version": spec.version,
        }

    executable = aria2c or shutil.which("aria2c")
    if executable is None:
        raise RuntimeError(
            "aria2c is required for TGB downloads. Install the aria2 package first."
        )
    subprocess.run(  # noqa: S603
        _aria2_command(executable, spec, archive_dir),
        check=True,
    )
    archive_path = archive_dir / spec.remote_archive
    if download_only:
        return {
            "archive": str(archive_path),
            "dataset": spec.name,
            "status": "downloaded",
            "version": spec.version,
        }

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target)
    missing = _missing_members(target, spec)
    if missing:
        raise ValueError(
            f"Archive for {spec.name} is missing required files: {missing}"
        )
    return {
        "archive": str(archive_path),
        "dataset": spec.name,
        "path": str(target),
        "status": "fetched",
        "version": spec.version,
    }


def fetch_tgb_link_datasets(
    specs: Iterable[TGBDownloadSpec],
    *,
    root: Path | None = None,
    download_only: bool = False,
) -> list[dict[str, object]]:
    return [
        fetch_tgb_link_dataset(spec, root=root, download_only=download_only)
        for spec in specs
    ]


def tgb_link_dataset_catalog() -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "version": spec.version,
            "url": spec.url,
        }
        for spec in TGB_LINK_DATASETS.values()
    ]
