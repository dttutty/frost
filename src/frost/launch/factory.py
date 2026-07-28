from __future__ import annotations

import logging

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from frost.launch.config import (
    DYGFORMER_ENCODER_TARGET,
    DYGLIB_ALIGNED_FEATURE_DIM,
    GRAPH_MIXER_ENCODER_TARGET,
    TGAT_ENCODER_TARGET,
    resolve_dygformer_dummy_edge_dim,
    resolve_dygformer_dummy_node_dim,
    resolve_predictor_cfg,
    resolve_tgat_dummy_edge_dim,
    resolve_tgat_dummy_node_dim,
    string_keyed_dict,
)
from frost.models.protocols import MemoryUpdaterProto
from frost.models.protocols import PostmanProto
from frost.runtime.worker.postman import Postman


logger = logging.getLogger(__name__)


def build_worker_modules(
    cfg: DictConfig,
    *,
    encoder_target: str,
    fanout: list[int],
    d_mail: int,
    max_n_neg: int,
    postman_enabled: bool,
) -> tuple[
    MemoryUpdaterProto | None,
    nn.Module,
    nn.Module,
    PostmanProto | None,
    nn.Module,
]:
    mem_updater_cfg = cfg.get("mem_updater")
    if mem_updater_cfg is not None:
        mem_updater_cfg = OmegaConf.to_container(mem_updater_cfg, resolve=True)
        if not isinstance(mem_updater_cfg, dict):
            raise TypeError("mem_updater must resolve to a mapping")
        mem_updater_cfg.pop("mail_recipients", None)
        mem_updater_cfg.pop("mail_combine", None)
        mem_updater_cfg.pop("postman_delivery_strategy", None)
    mem_updater: MemoryUpdaterProto | None = instantiate(mem_updater_cfg, d_mail=d_mail)

    encoder_cfg = string_keyed_dict(
        OmegaConf.to_container(cfg.encoder, resolve=True), name="encoder"
    )
    encoder_cfg.pop("out_dim", None)
    encoder_cfg.pop("fanout", None)
    encoder_cfg.pop("ngh_sampler_strategy", None)
    if (
        encoder_target == DYGFORMER_ENCODER_TARGET
        and cfg.ndim == 0
        and bool(encoder_cfg.get("enable_dummy_node_features", False))
    ):
        encoder_cfg["dummy_node_dim"] = resolve_dygformer_dummy_node_dim(
            dataset=cfg.dataset,
            requested_dim=encoder_cfg.get("dummy_node_dim"),
        )
    if (
        encoder_target == DYGFORMER_ENCODER_TARGET
        and cfg.edim == 0
        and bool(encoder_cfg.get("enable_dummy_edge_features", True))
    ):
        encoder_cfg["dummy_edge_dim"] = resolve_dygformer_dummy_edge_dim(
            dataset=cfg.dataset,
            requested_dim=encoder_cfg.get("dummy_edge_dim"),
        )
    if (
        encoder_target == TGAT_ENCODER_TARGET
        and cfg.ndim == 0
        and bool(encoder_cfg.get("enable_dummy_node_features", True))
    ):
        encoder_cfg["dummy_node_dim"] = resolve_tgat_dummy_node_dim(
            dataset=cfg.dataset,
            requested_dim=encoder_cfg.get("dummy_node_dim"),
        )
    if (
        encoder_target == TGAT_ENCODER_TARGET
        and cfg.edim == 0
        and bool(encoder_cfg.get("enable_dummy_edge_features", True))
    ):
        encoder_cfg["dummy_edge_dim"] = resolve_tgat_dummy_edge_dim(
            dataset=cfg.dataset,
            requested_dim=encoder_cfg.get("dummy_edge_dim"),
        )
    if (
        encoder_target == GRAPH_MIXER_ENCODER_TARGET
        and bool(encoder_cfg.get("align_dyglib_zero_features", False))
        and encoder_cfg.get("dyglib_feature_dim") is None
    ):
        encoder_cfg["dyglib_feature_dim"] = DYGLIB_ALIGNED_FEATURE_DIM
        logger.info(
            "Resolved GraphMixer dyglib_feature_dim=%d for encoder=%s.",
            DYGLIB_ALIGNED_FEATURE_DIM,
            GRAPH_MIXER_ENCODER_TARGET,
        )
    encoder: nn.Module = (
        instantiate(encoder_cfg, fanout=fanout)
        if encoder_target == TGAT_ENCODER_TARGET
        else instantiate(encoder_cfg)
    )

    predictor_cfg = string_keyed_dict(
        OmegaConf.to_container(cfg.predictor, resolve=True), name="predictor"
    )
    predictor_cfg, predictor_switched = resolve_predictor_cfg(
        encoder_target, predictor_cfg
    )
    if predictor_switched:
        logger.info(
            "Auto-switched predictor for pair-aware encoder=%s.", encoder_target
        )
    predictor: nn.Module = instantiate(predictor_cfg)
    postman: PostmanProto | None = Postman(n_neg=max_n_neg) if postman_enabled else None
    loss_fn: nn.Module = instantiate(cfg.loss_fn)
    return mem_updater, encoder, predictor, postman, loss_fn
