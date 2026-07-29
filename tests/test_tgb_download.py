from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from frost.data.cli import main
from frost.data.tgb_download import (
    TGBDownloadSpec,
    TGB_LINK_DATASETS,
    fetch_tgb_link_dataset,
    select_tgb_link_datasets,
)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return output.getvalue()


def _spec() -> TGBDownloadSpec:
    return TGBDownloadSpec(
        name="tgbl-mock",
        version=1,
        remote_archive="tgbl-mock.zip",
        md5="0123456789abcdef0123456789abcdef",
    )


def _valid_archive() -> bytes:
    return _zip_bytes(
        {
            "tgbl-mock_edgelist.csv": b"src,dst,ts\n0,1,0\n",
            "tgbl-mock_val_ns.pkl": b"val",
            "tgbl-mock_test_ns.pkl": b"test",
        }
    )


def _fake_aria2(payload: bytes, commands: list[list[str]]):
    def run(command, *, check):
        assert check is True
        commands.append(command)
        archive_dir = Path(
            next(
                arg.removeprefix("--dir=")
                for arg in command
                if arg.startswith("--dir=")
            )
        )
        archive_name = next(
            arg.removeprefix("--out=") for arg in command if arg.startswith("--out=")
        )
        (archive_dir / archive_name).write_bytes(payload)
        return subprocess.CompletedProcess(command, 0)

    return run


def test_fetch_tgb_dataset_delegates_download_to_aria2_and_extracts(
    tmp_path: Path, monkeypatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_aria2(_valid_archive(), commands))

    result = fetch_tgb_link_dataset(_spec(), root=tmp_path, aria2c="aria2c")

    target = tmp_path / "raw" / "tgbl_mock"
    assert result["status"] == "fetched"
    assert (target / "tgbl-mock_edgelist.csv").is_file()
    assert len(commands) == 1
    assert "--continue=true" in commands[0]
    assert "--split=8" in commands[0]
    assert "--checksum=md5=0123456789abcdef0123456789abcdef" in commands[0]

    result = fetch_tgb_link_dataset(_spec(), root=tmp_path, aria2c="aria2c")
    assert result["status"] == "present"
    assert len(commands) == 1


def test_fetch_tgb_dataset_supports_download_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_aria2(_valid_archive(), []))

    result = fetch_tgb_link_dataset(
        _spec(), root=tmp_path, download_only=True, aria2c="aria2c"
    )

    assert result["status"] == "downloaded"
    assert (tmp_path / "downloads" / "tgbl-mock.zip").is_file()
    assert not (tmp_path / "raw" / "tgbl_mock").exists()


def test_fetch_tgb_dataset_checks_extracted_members(
    tmp_path: Path, monkeypatch
) -> None:
    incomplete = _zip_bytes({"tgbl-mock_edgelist.csv": b"edge"})
    monkeypatch.setattr(subprocess, "run", _fake_aria2(incomplete, []))

    with pytest.raises(ValueError, match="missing required files"):
        fetch_tgb_link_dataset(_spec(), root=tmp_path, aria2c="aria2c")


def test_select_tgb_link_datasets_supports_all_and_version_aliases() -> None:
    assert [spec.name for spec in select_tgb_link_datasets(all_datasets=True)] == list(
        TGB_LINK_DATASETS
    )
    selected = select_tgb_link_datasets(["TGBL_WIKI", "tgbl-wiki-v2"])
    assert [spec.name for spec in selected] == ["tgbl-wiki"]


def test_fetch_tgb_list_prints_official_catalog(capsys) -> None:
    assert main(["fetch-tgb", "--list"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in result["datasets"]] == list(TGB_LINK_DATASETS)
