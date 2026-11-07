"""Verify private release assets before copying public distributions to dist/."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import zipfile


def verify(source: Path, output: Path, tag: str, commit: str) -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", tag):
        raise ValueError("Expected a version tag such as v0.3.1.")
    version = tag[1:]
    hashes = {}
    for line in (source / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if (
            not re.fullmatch(r"[a-f0-9]{64}", digest)
            or Path(name).name != name
            or name in hashes
        ):
            raise ValueError("Invalid checksum manifest entry.")
        hashes[name] = digest
    files = {p.name for p in source.iterdir() if p.is_file()} - {"SHA256SUMS"}
    if files != hashes.keys():
        raise ValueError("Checksum manifest must cover every release asset.")
    for name, digest in hashes.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    info = json.loads((source / "BUILD_INFO.json").read_text())
    if info["version"] != version or info["source_commit"] != commit:
        raise ValueError("Release version or source commit does not match the tag.")
    distributions = sorted(
        p for p in source.iterdir() if p.suffix == ".whl" or p.name.endswith(".tar.gz")
    )
    if (
        not any(p.suffix == ".whl" for p in distributions)
        or sum(p.name.endswith(".tar.gz") for p in distributions) != 1
    ):
        raise ValueError("Expected at least one wheel and one source distribution.")
    for path in distributions:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as wheel:
                names = [
                    n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")
                ]
                if len(names) != 1:
                    raise ValueError("Expected one wheel metadata record.")
                metadata = wheel.read(names[0])
        else:
            with tarfile.open(path) as sdist:
                names = [
                    m
                    for m in sdist.getmembers()
                    if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")
                ]
                if len(names) != 1:
                    raise ValueError("Expected one source metadata record.")
                stream = sdist.extractfile(names[0])
                if stream is None:
                    raise ValueError("Missing source metadata.")
                metadata = stream.read()
        record = BytesParser().parsebytes(metadata)
        name = re.sub(r"[-_.]+", "-", record["Name"]).lower()
        if name != "frost-gnn" or record["Version"] != version:
            raise ValueError(f"Wrong distribution name/version: {path.name}")
        if (
            b"github.com/dttutty/frost_dev" in metadata
            or b"not available yet" in metadata
        ):
            raise ValueError(
                "Distribution description is not ready for public installation."
            )
    output.mkdir(parents=True, exist_ok=False)
    for path in distributions:
        shutil.copy2(path, output / path.name)
    print(
        f"Verified {len(distributions)} frost-gnn {version} distributions from {commit}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    verify(args.source, args.output, args.tag, args.commit)
