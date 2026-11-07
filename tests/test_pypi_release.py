"""Publication checks run on synthetic archives without uploading anything."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / ".github/scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(source):
    (source / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(source.iterdir())
            if path.name != "SHA256SUMS"
        )
    )


@pytest.fixture
def assets(tmp_path):
    source = tmp_path / "assets"
    source.mkdir()
    metadata = b"Metadata-Version: 2.4\nName: frost-gnn\nVersion: 0.3.1\n\nInstall with pip install frost_gnn.\n"
    with zipfile.ZipFile(
        source / "frost_gnn-0.3.1-cp312-cp312-manylinux_2_31_x86_64.whl", "w"
    ) as wheel:
        wheel.writestr("frost_gnn-0.3.1.dist-info/METADATA", metadata)
    with tarfile.open(source / "frost_gnn-0.3.1.tar.gz", "w:gz") as sdist:
        entry = tarfile.TarInfo("frost_gnn-0.3.1/PKG-INFO")
        entry.size = len(metadata)
        sdist.addfile(entry, io.BytesIO(metadata))
    (source / "BUILD_INFO.json").write_text(
        json.dumps({"version": "0.3.1", "source_commit": "1" * 40})
    )
    manifest(source)
    return source


def test_public_description_uses_current_readme():
    original = (ROOT / "README.md").read_text()
    rendered = script("build_pypi_release").public_readme(original)
    assert "recommendation systems" in rendered
    assert "python -m pip install frost_gnn" in rendered
    assert "not available yet" not in rendered
    assert "github.com/dttutty/frost_dev" not in rendered
    assert "assets/frost-banner.png" not in rendered
    assert "docs/model_training.md" in rendered
    assert "not a calibrated probability" in rendered
    assert original[original.index("```bibtex") :].split("```")[1] in rendered
    assert (ROOT / "README.md").read_text() == original


def test_release_verification_copies_only_distributions(assets, tmp_path):
    output = tmp_path / "dist"
    script("verify_pypi_release").verify(assets, output, "v0.3.1", "1" * 40)
    assert sorted(path.suffix for path in output.iterdir()) == [".gz", ".whl"]
    for path in output.iterdir():
        assert path.read_bytes() == (assets / path.name).read_bytes()


@pytest.mark.parametrize(
    "failure",
    ["checksum", "commit", "version", "path", "private_description", "missing_file"],
)
def test_release_rejects_invalid_assets(assets, tmp_path, failure):
    tag, commit = "v0.3.1", "1" * 40
    if failure == "checksum":
        next(assets.glob("*.whl")).write_bytes(b"tampered")
    elif failure == "commit":
        commit = "2" * 40
    elif failure == "version":
        tag = "v0.3.2"
        info = json.loads((assets / "BUILD_INFO.json").read_text())
        info["version"] = "0.3.2"
        (assets / "BUILD_INFO.json").write_text(json.dumps(info))
        manifest(assets)
    elif failure == "path":
        with (assets / "SHA256SUMS").open("a") as stream:
            stream.write("0" * 64 + "  ../outside\n")
    elif failure == "private_description":
        with zipfile.ZipFile(next(assets.glob("*.whl")), "w") as wheel:
            wheel.writestr(
                "frost_gnn-0.3.1.dist-info/METADATA",
                "Name: frost-gnn\nVersion: 0.3.1\n\nhttps://github.com/dttutty/frost_dev",
            )
        manifest(assets)
    elif failure == "missing_file":
        next(assets.glob("*.tar.gz")).unlink()
    output = tmp_path / "dist"
    with pytest.raises(ValueError):
        script("verify_pypi_release").verify(assets, output, tag, commit)
    assert not output.exists()
