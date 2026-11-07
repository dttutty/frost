"""Build public distributions from a clean commit without a second README source."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib


PUBLIC_INSTALL = """## 📦 Install

On Linux x86-64 (glibc 2.31+) with Python 3.12:

```bash
python -m pip install frost-gnn
```

The Python import is `frost`; the command is `frost`. The CSV trainer runs on CPU
unless you select a GPU. To use CPU-only PyTorch, install `torch==2.9.0` from
`https://download.pytorch.org/whl/cpu` before installing FROST. CUDA execution
requires a compatible NVIDIA driver.

If upgrading from the old FROST 0.1.0 distribution, first run
`python -m pip uninstall frost` to avoid overlapping package files.

"""


def public_readme(source: str) -> str:
    """Reuse user documentation, stripping repository-only artwork and links."""
    intro = source.index("A multi-GPU framework")
    result = "# FROST\n\n" + source[intro:]
    result, count = re.subn(
        r"^## 📦 Install\n.*?(?=^## )",
        PUBLIC_INSTALL,
        result,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    if count != 1:
        raise ValueError("Expected exactly one installation section in README.md.")

    def link(match):
        label, target = match.groups()
        if "://" in target or target.startswith(("#", "mailto:")):
            return match.group(0)
        return f"{label} (`{target.split('#')[0]}` in the source distribution)"

    result = re.sub(r"\[([^\[\]]+)\]\(([^\s)]+)\)", link, result)
    if "github.com/dttutty/frost_dev" in result:
        raise ValueError(
            "The public package description still contains obsolete repository links."
        )
    return result


def build(output: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=repo, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw, assets = output / "raw", output / "assets"
    assets.mkdir()
    archive = subprocess.check_output(["git", "archive", "HEAD"], cwd=repo)
    with tempfile.TemporaryDirectory(prefix="frost-pypi-source-") as temp:
        stage = Path(temp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(stage, filter="data")
        readme = stage / "README.md"
        readme.write_text(public_readme(readme.read_text()))
        metadata = tomllib.loads((stage / "pyproject.toml").read_text())["project"]
        subprocess.run(["uv", "build", "--out-dir", str(raw)], cwd=stage, check=True)
    wheels, sdists = list(raw.glob("*.whl")), list(raw.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected one wheel and one source distribution.")
    subprocess.run(
        [
            "uvx",
            "--with",
            "patchelf",
            "auditwheel",
            "repair",
            "--plat",
            "manylinux_2_31_x86_64",
            "--wheel-dir",
            str(assets),
            str(wheels[0]),
        ],
        check=True,
    )
    shutil.copy2(sdists[0], assets / sdists[0].name)
    subprocess.run(
        ["uvx", "twine", "check", *map(str, sorted(assets.iterdir()))], check=True
    )
    info = {
        "distribution": metadata["name"],
        "version": metadata["version"],
        "source_commit": commit,
        "platform": "manylinux_2_31_x86_64",
        "python": "3.12",
        "readme": "Generated from README.md by build_pypi_release.py",
    }
    (assets / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    manifest = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
        for p in sorted(assets.iterdir())
    )
    (assets / "SHA256SUMS").write_text(manifest)
    print(f"Release artifacts: {assets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New build directory."
    )
    build(parser.parse_args().output)
