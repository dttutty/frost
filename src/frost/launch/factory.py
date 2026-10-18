from __future__ import annotations

import logging

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from frost.launch.config import (
    DYGLIB_ALIGNED_FEATURE_DIM,
)
from frost.launch.plan import ExecutionPlan
from frost.models.contracts import MemoryState, NeighborhoodMailbox
from frost.models.protocols import MemoryUpdaterProto
from frost.models.protocols import MailboxRouterProto
from frost.models.recipes import ModelRecipe
from frost.runtime.worker.mailbox_router import NeighborhoodMailboxRouter


logger = logging.getLogger(__name__)


def build_worker_modules(
    cfg: DictConfig,
    *,
    recipe: ModelRecipe,
    plan: ExecutionPlan,
    d_mail: int,
) -> tuple[
    MemoryUpdaterProto | None,
    nn.Module,
    nn.Module,
    MailboxRouterProto | None,
]:
    mem_updater: MemoryUpdaterProto | None = None
    if recipe.state is not None:
        updater_cfg = recipe.state.get("updater")
        if not isinstance(updater_cfg, dict):
            raise TypeError("model.state.updater must resolve to a mapping")
        mem_updater = instantiate(
            updater_cfg,
            d_mail=d_mail,
        )

    encoder_cfg = dict(recipe.encoder)
    if (
        cfg.ndim == 0
        and bool(encoder_cfg.get("enable_dummy_node_features", False))
        and encoder_cfg.get("dummy_node_dim") is None
    ):
        encoder_cfg["dummy_node_dim"] = DYGLIB_ALIGNED_FEATURE_DIM
    if (
        cfg.edim == 0
        and bool(encoder_cfg.get("enable_dummy_edge_features", False))
        and encoder_cfg.get("dummy_edge_dim") is None
    ):
        encoder_cfg["dummy_edge_dim"] = DYGLIB_ALIGNED_FEATURE_DIM
    if (
        bool(encoder_cfg.get("align_dyglib_zero_features", False))
        and encoder_cfg.get("dyglib_feature_dim") is None
    ):
        encoder_cfg["dyglib_feature_dim"] = DYGLIB_ALIGNED_FEATURE_DIM
        logger.info(
            "Resolved model encoder dyglib_feature_dim=%d.",
            DYGLIB_ALIGNED_FEATURE_DIM,
        )
    encoder: nn.Module = instantiate(encoder_cfg)
    predictor: nn.Module = instantiate(recipe.head)
    state_contract = plan.dynamic_state
    mailbox_router: MailboxRouterProto | None = (
        NeighborhoodMailboxRouter()
        if isinstance(state_contract, MemoryState)
        and isinstance(state_contract.mailbox, NeighborhoodMailbox)
        else None
    )
    return mem_updater, encoder, predictor, mailbox_router
