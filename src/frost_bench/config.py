from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from frost_bench.paths import SPEC_ROOT


MEMORY_BASED_PRESETS = frozenset({"tgn", "jodie", "apan"})
DYGLIB_BACKED_PRESETS = frozenset({"tgat", "cawn", "tcl", "graph_mixer", "dygformer"})


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    master_port: int
    family: str
    hydra_overrides: tuple[str, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


@lru_cache(maxsize=1)
def load_presets() -> dict[str, Preset]:
    payload = _load_yaml(SPEC_ROOT / "presets.yaml")
    raw_presets = payload.get("presets")
    if not isinstance(raw_presets, dict):
        raise ValueError("benchmarks/specs/presets.yaml must define `presets`")
    result: dict[str, Preset] = {}
    for name, raw in raw_presets.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Preset {name!r} must be a mapping")
        result[name] = Preset(
            name=name,
            description=str(raw["description"]),
            master_port=int(raw["master_port"]),
            family=str(raw["family"]),
            hydra_overrides=tuple(str(value) for value in raw["overrides"]),
        )
    return result


PRESETS = load_presets()


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().upper()
    aliases = {"WIKI": "WIKIPEDIA", "SOCIAL_EVO": "SOCIALEVO"}
    return aliases.get(normalized, normalized)


@lru_cache(maxsize=1)
def _preset_payload() -> dict[str, Any]:
    return _load_yaml(SPEC_ROOT / "presets.yaml")


def load_dataset_default_macro_batch_size(dataset_name: str, preset_name: str) -> int:
    normalized_preset = preset_name.strip().lower()
    family = "memory" if normalized_preset in MEMORY_BASED_PRESETS else "stateless"
    raw = _preset_payload()["dataset_defaults"][family]
    dataset = normalize_dataset_name(dataset_name)
    batch_size = int(raw.get("datasets", {}).get(dataset, raw["default"]))
    divisor = int(raw.get("model_divisors", {}).get(normalized_preset, 1))
    if divisor < 1:
        raise ValueError(f"Batch-size divisor for {normalized_preset!r} must be positive")
    if batch_size % divisor:
        raise ValueError(
            f"Dataset batch size {batch_size} is not divisible by {divisor} "
            f"for preset {normalized_preset!r}"
        )
    return batch_size // divisor


@lru_cache(maxsize=1)
def _dyglib_parity_payload() -> dict[str, Any]:
    return _load_yaml(SPEC_ROOT / "dyglib_parity.yaml")


def load_dyglib_parity_settings(
    *, preset_name: str, dataset_name: str
) -> dict[str, Any]:
    if preset_name not in DYGLIB_BACKED_PRESETS:
        raise ValueError(f"Preset {preset_name!r} is not DyGLib-backed")
    raw_model = _dyglib_parity_payload()["models"][preset_name]
    settings = dict(raw_model["defaults"])
    settings.update(
        raw_model.get("datasets", {}).get(normalize_dataset_name(dataset_name), {})
    )
    return settings


def load_dyglib_model_defaults(preset_name: str) -> dict[str, Any]:
    if preset_name not in DYGLIB_BACKED_PRESETS:
        raise ValueError(f"Preset {preset_name!r} is not DyGLib-backed")
    return dict(_dyglib_parity_payload()["models"][preset_name]["defaults"])


def load_dyglib_best_link_prediction_args(
    *, preset_name: str, dataset_name: str
) -> SimpleNamespace:
    """Compatibility view of the repository-owned DyGLib parity settings."""

    return SimpleNamespace(
        **load_dyglib_parity_settings(
            preset_name=preset_name, dataset_name=dataset_name
        )
    )


def build_dyglib_hydra_overrides(preset_name: str, dataset_name: str) -> list[str]:
    if preset_name not in DYGLIB_BACKED_PRESETS:
        return []

    args = load_dyglib_best_link_prediction_args(
        preset_name=preset_name, dataset_name=dataset_name
    )
    if preset_name == "tgat":
        fanout = ", ".join([str(args.num_neighbors)] * args.num_layers)
        return [
            f"model.neighborhood.fanout=[{fanout}]",
            f"ngh_sampler.strategy={args.sample_neighbor_strategy}",
            f"model.encoder.heads={args.num_heads}",
            f"d_time={args.time_feat_dim}",
            f"dropout={args.dropout}",
        ]
    if preset_name == "cawn":
        return [
            f"model.neighborhood.fanout=[{args.num_neighbors}]",
            f"model.encoder.walk_length={args.walk_length}",
            f"model.encoder.position_feat_dim={args.position_feat_dim}",
            f"model.encoder.num_walk_heads={args.num_walk_heads}",
            f"ngh_sampler.strategy={args.sample_neighbor_strategy}",
            f"ngh_sampler.time_scaling_factor={args.time_scaling_factor}",
            f"model.encoder.dropout={args.dropout}",
            f"d_time={args.time_feat_dim}",
        ]
    if preset_name == "tcl":
        return [
            f"model.neighborhood.fanout=[{args.num_neighbors}]",
            f"model.encoder.num_depths={args.num_neighbors + 1}",
            f"model.encoder.num_layers={args.num_layers}",
            f"model.encoder.num_heads={args.num_heads}",
            f"ngh_sampler.strategy={args.sample_neighbor_strategy}",
            f"d_time={args.time_feat_dim}",
            f"dropout={args.dropout}",
        ]
    if preset_name == "graph_mixer":
        return [
            f"model.neighborhood.fanout=[{args.time_gap}]",
            f"model.encoder.num_tokens={args.num_neighbors}",
            f"model.encoder.num_layers={args.num_layers}",
            f"ngh_sampler.strategy={args.sample_neighbor_strategy}",
            f"d_time={args.time_feat_dim}",
            f"dropout={args.dropout}",
        ]
    if preset_name == "dygformer":
        return [
            f"model.neighborhood.fanout=[{args.max_input_sequence_length - 1}]",
            f"model.encoder.max_input_sequence_length={args.max_input_sequence_length}",
            f"model.encoder.patch_size={args.patch_size}",
            f"model.encoder.channel_embedding_dim={args.channel_embedding_dim}",
            f"model.encoder.transformer_layers={args.num_layers}",
            f"model.encoder.num_heads={args.num_heads}",
            f"d_time={args.time_feat_dim}",
            f"dropout={args.dropout}",
        ]
    raise ValueError(f"Unhandled DyGLib-backed preset {preset_name!r}")
