# worker/trainer.py
from threading import Event
from typing import Optional

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP

from memorys import BasicMemoryUpdater, MemoryMailbox, TransformerMemoryUpdater
from modules import GeneralModel
from schemas.datatypes import DistEnv
from train_dist import Config
from worker.d2h import MailTask, d2h
from worker.h2d import GPUData, TransferData


# -------------------- Trainer --------------------
class Trainer:
    def __init__(
        self,
        cfg: Config,
        nccl_group,
        dedup,
        efeat,
        mm: Optional[MemoryMailbox],
        mail_worker: d2h,
        env: DistEnv,
    ):
        self._cfg = cfg
        self._dedup = dedup
        self._efeat = efeat
        self._mm = mm
        self._mail_worker = mail_worker
        self.nccl_group = nccl_group
        self.env = env

        ndim, edim = cfg.t.ndim, cfg.t.edim

        model = GeneralModel(ndim, edim, cfg).cuda()
        self._model: DDP = DDP(
            model,
            device_ids=[0],
            process_group=nccl_group,
            output_device=0,  # 因为本进程视角只有一张卡
            find_unused_parameters=True if cfg.s.history > 1 else False,
        )

        self.mu: Optional[BasicMemoryUpdater | TransformerMemoryUpdater] = None
        if cfg.m.type != "none":
            self.mu = self._model.module.memory_updater

        self._loss_fn = torch.nn.BCEWithLogitsLoss()
        self._opt = torch.optim.Adam(self._model.parameters(), lr=cfg.t.lr)

        self.tot_loss = 0.0

    def train(self, gpudata: GPUData, td: TransferData, done_ev: torch.cuda.Event):
        # 等 h2d 完成
        print(f"[Rank {self.env.local_rank}] wait for mid {td.mid}")
        torch.cuda.current_stream().wait_event(done_ev)

        self._model.train()
        self._opt.zero_grad(set_to_none=True)

        pred_pos, pred_neg = self._model(gpudata)
        loss = self._loss_fn(pred_pos, torch.ones_like(pred_pos))
        loss = loss + self._loss_fn(pred_neg, torch.zeros_like(pred_neg))
        loss.backward()
        self._opt.step()

        with torch.no_grad():
            self.tot_loss += float(loss.item())

        # Mail 投递
        if (self._mm is not None) and (self.mu is not None):
            with torch.no_grad():
                mu_efeats = self._efeat[td.eid] if self._efeat is not None else None
                task = MailTask(
                    mid=td.mid,
                    root=td.root,
                    block=td.block,
                    nid=self.mu.last_updated_nid.clone(),
                    memory=self.mu.last_updated_memory.clone(),
                    ts=self.mu.last_updated_ts.clone(),
                    mem_edge_feats=mu_efeats,
                    done=Event(),
                )
                self._mail_worker.enqueue(task)

        print(f"[Rank {self.env.local_rank}], mid {td.mid}, loss {loss.item()}")

    def eval(
        self,
        gpudata: GPUData,
        td: TransferData,
        done_ev: torch.cuda.Event,
        host_rank: int,
    ):
        torch.cuda.current_stream().wait_event(done_ev)

        self._model.eval()
        with torch.no_grad():
            pred_pos, pred_neg = self._model(gpudata)

            # 评估阶段也能选择投递 mailbox（按需）
            if (self._mm is not None) and (self.mu is not None):
                mem_edge_feats = (
                    self._efeat[td.eid] if self._efeat is not None else None
                )
                task = MailTask(
                    mid=td.mid,
                    nid=self.mu.last_updated_nid.clone(),
                    memory=self.mu.last_updated_memory.clone(),
                    ts=self.mu.last_updated_ts.clone(),
                    root=td.root,
                    block=td.block,
                    mem_edge_feats=mem_edge_feats,
                    done=Event(),
                )
                self._mail_worker.enqueue(task)

            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0
            )

            ap = average_precision_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)

        return ap, auc

    # 供外部保存/加载
    def save(self, path):
        if self.env.local_rank == 0:
            torch.save(self._model.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=torch.device("cuda:0"))
        self._model.load_state_dict(state)

    def close(self):
        # 释放资源（可按需扩展）
        del self._opt
        self._model = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
