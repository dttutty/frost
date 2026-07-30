from __future__ import annotations

from typing import Any, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from frost.models.temporal_link_model import TemporalLinkModel


class WorkerCheckpointMixin:
    device: torch.device
    nccl: dist.ProcessGroup
    model: Any

    def _unwrap_model(self) -> TemporalLinkModel:
        """Return the bare model, stripping torch.compile and DDP wrappers."""
        model: object = self.model
        # torch.compile wraps the model in an OptimizedModule
        if hasattr(model, "_orig_mod"):
            model = getattr(model, "_orig_mod")
        # DDP wraps the model in a DistributedDataParallel
        if isinstance(model, DDP):
            model = model.module
        return cast(TemporalLinkModel, model)

    def save_model(self, path: str) -> None:
        """Save a consistent CPU snapshot of the unwrapped model (rank 0 only).

        The DDP wrapper is stripped so the checkpoint can be loaded without a
        distributed context. The snapshot is materialized on CPU in the caller
        thread before ``torch.save`` so training never overlaps with a
        background thread reading live GPU parameters.

        Args:
            path: File path where the state dict will be written.
        """
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        if rank == 0:
            torch.cuda.synchronize(self.device)
            with torch.no_grad():
                cpu_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self._unwrap_model().state_dict().items()
                }
            torch.save(cpu_state_dict, path)

    def load_model(self, path: str) -> None:
        """Load model weights into the existing model / DDP wrapper in place.

        A barrier before loading ensures the file is fully written by rank 0
        before any rank reads it.  A second barrier after loading keeps all
        ranks in sync.

        Args:
            path: File path to the state dict checkpoint.
        """
        # Barrier ensures the file is fully written before any rank reads it
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.nccl)

        state = torch.load(path, map_location="cpu")

        bare_model = self._unwrap_model()
        bare_model.load_state_dict(state, strict=True)
        bare_model.to(self.device)

        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.nccl)
