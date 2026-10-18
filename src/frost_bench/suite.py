from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

Scalar = str | int | float | bool
ArgumentValue = Scalar | tuple[Scalar, ...] | None


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    description: str
    source: Path
    profile: str | None
    arguments: Mapping[str, ArgumentValue]

    def command(
        self,
        *,
        overrides: Mapping[str, ArgumentValue] | None = None,
        extra_args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        arguments = {**self.arguments, **(overrides or {})}
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "frost_bench.cli.throughput",
            self.name,
            "--spec",
            str(self.source),
        ]
        for name, value in arguments.items():
            if value is None:
                continue
            encoded = list(value) if isinstance(value, tuple) else value
            command.extend(("--set", f"{name}={json.dumps(encoded)}"))
        return (*command, *extra_args)

    def shell_command(
        self,
        *,
        overrides: Mapping[str, ArgumentValue] | None = None,
        extra_args: Sequence[str] = (),
    ) -> str:
        return shlex.join(
            self.command(overrides=overrides, extra_args=extra_args)
        )


def _normalize_value(value: object) -> ArgumentValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(
        isinstance(item, (str, int, float, bool)) for item in value
    ):
        return tuple(value)
    raise TypeError(f"Unsupported suite argument value: {value!r}")


def load_suites(path: Path) -> dict[str, SuiteSpec]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    raw_suites = document.get("suites")
    if not isinstance(raw_suites, dict):
        raise ValueError(f"{path} must contain a 'suites' mapping")

    suites: dict[str, SuiteSpec] = {}
    for name, raw in raw_suites.items():
        if not isinstance(raw, dict):
            raise TypeError(f"Suite {name!r} must be a mapping")
        raw_arguments = raw.get("arguments", {})
        if not isinstance(raw_arguments, dict):
            raise TypeError(f"Suite {name!r} arguments must be a mapping")
        suites[str(name)] = SuiteSpec(
            name=str(name),
            description=str(raw.get("description", "")),
            source=path,
            profile=str(raw["profile"]) if raw.get("profile") is not None else None,
            arguments={
                str(key): _normalize_value(value)
                for key, value in raw_arguments.items()
            },
        )
    return suites


def parse_overrides(values: Sequence[str]) -> dict[str, ArgumentValue]:
    overrides: dict[str, ArgumentValue] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE, got {value!r}")
        key, raw_value = value.split("=", 1)
        parsed = yaml.safe_load(raw_value)
        overrides[key.replace("-", "_")] = _normalize_value(parsed)
    return overrides
