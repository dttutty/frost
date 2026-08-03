"""Shared-memory node-memory and mailbox store management.

The host process allocates the shared-memory tensors for node memory states
and mailboxes; worker processes attach to them via a distributed barrier
handshake.  All tensors are zero-initialised and reset between epochs.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from frost.shm._core import create_shared_mem_array as csm
from frost.shm._core import get_shared_mem_array as gsm

if TYPE_CHECKING:
    from frost.datatypes.mem_mail_box.mail import MailEntry


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MemBoxStore:
    """Dataclass holding all shared-memory tensors for node memory and mailbox.

    ``box`` and ``box_ts`` use a per-node ring-buffer indexed by ``box_pos``.
    ``box_pos[n]`` points to the *next* write slot for node ``n``
    (wraps around modulo ``boxsize``).  The oldest entry is at
    ``box_pos[n]`` and the most-recent at ``(box_pos[n] - 1) % boxsize``.
    """

    mem: torch.Tensor
    mem_ts: torch.Tensor
    box: torch.Tensor  # [n_nodes, boxsize, d_mail]
    box_ts: torch.Tensor  # [n_nodes, boxsize]
    box_pos: torch.Tensor  # [n_nodes], int64 – next write position per node

    def reset(self) -> None:
        """Zero out all shared-memory tensors (called at epoch boundaries)."""
        self.mem.zero_()
        self.mem_ts.zero_()
        self.box.zero_()
        self.box_ts.zero_()
        self.box_pos.zero_()

    def write_mail(self, mail: MailEntry) -> None:
        """Insert one new mail per node into the per-node ring buffer.

        ``mail.nid`` must be unique (duplicates are not handled here).
        All ``nid`` values must already exist as valid node indices in this
        store.  The mail tensors may live on any device; they are moved to
        CPU automatically.

        The ring buffer for node ``n`` has ``boxsize`` slots.
        ``box_pos[n]`` is the *next write* position.  Each call:

        1. Reads the current write position ``pos = box_pos[nid]``.
        2. Writes ``mail.data`` and ``mail.ts`` to ``box[nid, pos]``.
        3. Advances ``box_pos[nid] = (pos + 1) % boxsize``.

        The oldest entry in a full buffer is always at the current
        ``box_pos[n]`` slot; the most recent is at ``(box_pos[n] - 1) % boxsize``.

        Args:
            mail: Deduplicated mail with unique ``nid``.  May reside on GPU.
        """
        boxsize = self.box.shape[1]

        nid = mail.nid.cpu().long()  # [N]
        data = mail.data.cpu().to(self.box.dtype)  # [N, d_mail]
        ts = mail.ts.cpu().to(self.box_ts.dtype)  # [N]

        pos = self.box_pos[nid]  # [N] – current write positions

        # Advanced indexing: box[nid[i], pos[i]] = data[i]  for each i
        self.box[nid, pos] = data
        self.box_ts[nid, pos] = ts

        self.box_pos[nid] = (pos + 1) % boxsize


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
        d_mem: Node-memory vector dimensionality (typically tied to ``encoder.out_dim``
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
        tdtype: torch.dtype,
        numa_node: int | None = None,
    ):
        self.env = env
        self.run_id = run_id
        self.mdtype = mdtype
        self.tdtype = tdtype
        self.n_nodes = n_nodes
        self.boxsize = boxsize
        self.d_mem = d_mem
        self.d_mail = 2 * d_mem + edim
        self.numa_node = numa_node

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
        mem = mem_ts = box = box_ts = box_pos = None

        if self.env.is_host:
            mem = csm(
                f"{self.run_id}_mem",
                [n_nodes, d_mem],
                self.mdtype,
                numa_node=self.numa_node,
            )
            mem_ts = csm(
                f"{self.run_id}_mem_ts",
                [n_nodes],
                self.tdtype,
                numa_node=self.numa_node,
            )
            box = csm(
                f"{self.run_id}_box",
                mail_shape,
                self.mdtype,
                numa_node=self.numa_node,
            )
            box_ts = csm(
                f"{self.run_id}_box_ts",
                [n_nodes, boxsize],
                self.tdtype,
                numa_node=self.numa_node,
            )
            box_pos = csm(
                f"{self.run_id}_box_pos",
                [n_nodes],
                torch.int64,
                numa_node=self.numa_node,
            )

            # Initialize memory states to zero
            mem.zero_()
            mem_ts.zero_()
            box.zero_()
            box_ts.zero_()
            box_pos.zero_()

        dist.barrier()

        if not self.env.is_host:
            populate_start = time.perf_counter()
            mem = gsm(
                f"{self.run_id}_mem",
                [n_nodes, d_mem],
                self.mdtype,
                populate_read=True,
            )
            mem_ts = gsm(
                f"{self.run_id}_mem_ts",
                [n_nodes],
                self.tdtype,
                populate_read=True,
            )
            box = gsm(
                f"{self.run_id}_box",
                mail_shape,
                self.mdtype,
                populate_read=True,
            )
            box_ts = gsm(
                f"{self.run_id}_box_ts",
                [n_nodes, boxsize],
                self.tdtype,
                populate_read=True,
            )
            box_pos = gsm(
                f"{self.run_id}_box_pos",
                [n_nodes],
                torch.int64,
                populate_read=True,
            )
            logger.info(
                "Dynamic shared-state page-table population completed in %.3fs",
                time.perf_counter() - populate_start,
            )

        # Store the dataclass as an instance attribute
        assert mem is not None
        assert mem_ts is not None
        assert box is not None
        assert box_ts is not None
        assert box_pos is not None
        self.state = MemBoxStore(
            mem=mem, mem_ts=mem_ts, box=box, box_ts=box_ts, box_pos=box_pos
        )

        dist.barrier()

        # Return self for method chaining, matching DatasetManager
        return self.state
