"""Shared-memory node-memory and mailbox store management.

The host process allocates the shared-memory tensors for node memory states
and mailboxes; worker processes attach to them via a distributed barrier
handshake.  All tensors are zero-initialised and reset between epochs.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from frost.shm._core import create_shared_mem_array as csm
from frost.shm._core import get_shared_mem_array as gsm


@dataclass(kw_only=True)
class MemBoxStore:
    """Dataclass holding all shared-memory tensors for node memory and mailbox."""

    mem: torch.Tensor
    mem_ts: torch.Tensor
    box: torch.Tensor
    box_ts: torch.Tensor

    def reset(self) -> None:
        """Zero out all shared-memory tensors (called at epoch boundaries)."""
        self.mem.zero_()
        self.mem_ts.zero_()
        self.box.zero_()
        self.box_ts.zero_()


class MemBoxStoreManager:
    """Manages allocation and distribution of shared-memory node-memory and mailbox tensors.

    The host calls :meth:`prepare` first (before the distributed barrier) to
    create the shared-memory segments; each worker calls :meth:`prepare` after
    the barrier to attach to them.  The resulting :class:`MemBoxStore` is
    accessible via :attr:`state`.

    Args:
        env: Distributed environment descriptor (:class:`~frost.datatypes.env.Env`).
        run_id: Unique string prefix used to name shared-memory segments.
        n_nodes: Total number of nodes in the graph.
        boxsize: Number of mailbox slots per node.
        d_mem: Node-memory vector dimensionality (must equal ``final_emb_dim``
            for identity-GNN models).
        edim: Edge-feature dimensionality; used to compute ``d_mail``.
        mdtype: PyTorch dtype for all memory / mailbox tensors.
    """

    def __init__(
        self,
        env,
        run_id: str,
        n_nodes: int,
        boxsize: int,  # from hydra config
        d_mem: int,  # from hydra config
        edim: int,  # from hydra config
        mdtype: torch.dtype,
    ):
        self.env = env
        self.run_id = run_id
        self.mdtype = mdtype
        self.n_nodes = n_nodes
        self.boxsize = boxsize
        self.d_mem = d_mem
        self.d_mail = 2 * d_mem + edim

        # Initialize internal state
        self.state: Optional[MemBoxStore] = None

    def prepare(self):
        """
        Host creates shared memory, workers attach.
        Populates internal state.
        """
        n_nodes = self.n_nodes
        boxsize = self.boxsize
        d_mem = self.d_mem
        d_mail = self.d_mail

        mail_shape = [n_nodes, boxsize, d_mail]

        # Initialize local variables to avoid UnboundLocalError
        mem = mem_ts = box = box_ts = None

        if self.env.is_host:
            mem = csm(f"{self.run_id}_mem", [n_nodes, d_mem], self.mdtype)
            mem_ts = csm(f"{self.run_id}_mem_ts", [n_nodes], self.mdtype)
            box = csm(f"{self.run_id}_box", mail_shape, self.mdtype)
            box_ts = csm(f"{self.run_id}_box_ts", [n_nodes, boxsize], self.mdtype)

            # Initialize memory states to zero
            mem.zero_()
            mem_ts.zero_()
            box.zero_()
            box_ts.zero_()

        dist.barrier()

        if not self.env.is_host:
            mem = gsm(f"{self.run_id}_mem", [n_nodes, d_mem], self.mdtype)
            mem_ts = gsm(f"{self.run_id}_mem_ts", [n_nodes], self.mdtype)
            box = gsm(f"{self.run_id}_box", mail_shape, self.mdtype)
            box_ts = gsm(f"{self.run_id}_box_ts", [n_nodes, boxsize], self.mdtype)

        # Store the dataclass as an instance attribute
        assert mem is not None
        assert mem_ts is not None
        assert box is not None
        assert box_ts is not None
        self.state = MemBoxStore(mem=mem, mem_ts=mem_ts, box=box, box_ts=box_ts)

        dist.barrier()

        # Return self for method chaining, matching DatasetManager
        return self.state
