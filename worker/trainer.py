# # worker/trainer.py
# from typing import Optional

# import torch
# from torch.nn.parallel import DistributedDataParallel as DDP

# from memorys import BasicMemoryUpdater, TransformerMemoryUpdater
# from modules import GeneralModel
# from schemas.datatypes import DistEnv
# from train_dist import Config


# # -------------------- Trainer --------------------
# class Trainer:
#     def __init__(
#         self,
#         cfg: Config,
#         env: DistEnv,
#         model: DDP[GeneralModel],
#     ):
#         self.env = env

#         self._model = model

#         self.mu: Optional[BasicMemoryUpdater | TransformerMemoryUpdater] = None
#         if cfg.m.type != "none":
#             self.mu = self._model.module.memory_updater

#         self._loss_fn = torch.nn.BCEWithLogitsLoss()
#         self._opt = torch.optim.Adam(self._model.parameters(), lr=cfg.t.lr)

#         self.tot_loss = 0.0
