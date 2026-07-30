"""Regression tests for the model/runtime dependency boundary."""

import ast
from pathlib import Path

from frost.models.recipes import RECIPE_TYPES


MODEL_ROOT = Path(__file__).parents[1] / "src" / "frost" / "models"


def test_model_package_does_not_import_runtime_modules() -> None:
    violations: list[str] = []
    for path in MODEL_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "frost.runtime"
            ):
                violations.append(f"{path.relative_to(MODEL_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("frost.runtime"):
                        violations.append(
                            f"{path.relative_to(MODEL_ROOT)}:{node.lineno}"
                        )
    assert violations == []


def test_supported_recipes_are_explicit() -> None:
    assert set(RECIPE_TYPES) == {
        "tgn",
        "jodie",
        "apan",
        "tgat",
        "graph_mixer",
        "tcl",
        "dygformer",
        "cawn",
    }
