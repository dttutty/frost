"""Backward-compatibility shim — re-exports from frost.neg_sampler.

All implementation has moved to the ``frost.neg_sampler`` package.
This module is kept so that existing code importing from
``frost.sampling.neg_sampler`` continues to work without modification.
"""
from frost.neg_sampler import (  # noqa: F401
    LEGACY_PRECOMPUTED_NEGATIVE_FILENAME,
    PRECOMPUTED_NEGATIVE_FILENAME,
    HistoricalNegLinkSampler,
    InductiveNegLinkSampler,
    PrecomputedNegSampler,
    RandomNegLinkSampler,
    generate_precomputed_negative_file,
    resolve_dataset_dir,
    resolve_precomputed_negative_path,
)
