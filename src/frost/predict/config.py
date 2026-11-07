"""Shared, validated defaults for the simple training CLI and Python API."""

import math
from pathlib import Path

import yaml


def default_path() -> Path:
    installed = Path(__file__).resolve().parents[1] / "configs/predict.yaml"
    return (
        installed
        if installed.is_file()
        else Path(__file__).resolve().parents[3] / "configs/predict.yaml"
    )


def validate(config: dict) -> dict:
    if config["model"] not in ("tgn", "tgat"):
        raise ValueError("Model must be tgn or tgat.")
    for name in (
        "epochs",
        "batch_size",
        "fanout",
        "hidden",
        "time_dim",
        "heads",
        "threads",
    ):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if config["hidden"] % config["heads"]:
        raise ValueError("hidden must be divisible by heads.")
    for name in ("lr", "dropout"):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]):
            raise ValueError(f"{name} must be a finite number.")
    if config["lr"] <= 0 or not 0 <= config["dropout"] < 1:
        raise ValueError("lr must be positive and dropout must be in [0, 1).")
    device = config["device"]
    if not isinstance(device, str) or not (
        device == "cpu"
        or device == "cuda"
        or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("device must be cpu, cuda, or cuda:<index>.")
    return config


DEFAULTS = validate(yaml.safe_load(default_path().read_text()))


def resolve(path=None, **overrides) -> dict:
    result = DEFAULTS.copy()
    if path:
        custom = yaml.safe_load(Path(path).read_text())
        if not isinstance(custom, dict):
            raise ValueError("Training configuration must be a YAML mapping.")
        unknown = custom.keys() - DEFAULTS.keys()
        if unknown:
            raise ValueError(
                f"Unknown training configuration keys: {sorted(map(str, unknown))}."
            )
        result.update(custom)
    result.update({key: value for key, value in overrides.items() if value is not None})
    return validate(result)
