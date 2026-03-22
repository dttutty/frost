"""Task datatypes representing units of work queued between worker threads."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
from torch import Tensor

from frost.datatypes.graph import SeqGraph
from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.datatypes.plan import DynamicPlan


@dataclass(slots=True)
class MemTask:
    """Work item consumed by the memory-writeback writer thread.

    Attributes:
        batch_idx: Global batch counter used to order writes correctly.
        root_new_mem: Updated memory entries for root nodes computed during
            the forward pass.
        mem_plan: DynamicPlan carrying the node IDs and P2P routing needed to
            propagate the memory update to other workers.
        is_last: ``True`` iff this is the final batch of the current epoch;
            signals the writer to flush and reset state.
    """

    batch_idx: int
    root_new_mem: MemEntry
    mem_plan: DynamicPlan
    is_last: bool


@dataclass(slots=True)
class BoxTask:
    """Work item consumed by the mailbox-writeback writer thread.

    Attributes:
        pos_efeat: Edge features for positive destination nodes used to
            construct the new mail message; ``None`` when edge features are
            not used.
        root_new_mem: Updated memory entries for root nodes, needed to build
            the mail content.
        batch_idx: Global batch counter for ordering writes.
        box_plan: DynamicPlan carrying node IDs and P2P routing for the
            mailbox update.
        is_last: ``True`` iff this is the final batch of the current epoch.
        graph: Inflated ``Graph`` for this batch; used by the postman to
            determine which neighbor mailboxes to update.
    """

    pos_efeat: Tensor | None
    root_new_mem: MemEntry
    batch_idx: int
    box_plan: DynamicPlan
    is_last: bool
    graph: SeqGraph | None
    flat_nid_cpu: Tensor | None = None  # CPU nids for APAN writeback

    def __repr__(self) -> str:
        # if data is tensor, show its shape
        return (
            f"BoxTask(batch_idx={self.batch_idx}, "
            f"box_plan={self.box_plan}, "
            f"is_last={self.is_last}, "
            f"graph={self.graph})"
        )


@dataclass(slots=True)
class P2PTask:
    """Work item representing one round of peer-to-peer memory / mailbox exchange.

    Consumed by ``MemHandler`` / ``BoxHandler`` P2P syncer threads.  Each
    worker sends its locally-owned updates to peers and receives updates that
    peers own for nodes assigned to this worker.

    Attributes:
        send_nid: Node IDs whose state this worker is sending to peers.
        recv_nid: Node IDs whose state this worker expects to receive from
            peers.
        data2send: State tensor to send, shape ``[len(send_nid), d_state]``.
        data2recv: Pre-allocated buffer for received state, same shape layout.
        ts2send: Timestamps paired with ``data2send``.
        ts2recv: Pre-allocated buffer for received timestamps.
        recv_counts: Number of elements to receive from each peer worker.
        send_counts: Number of elements to send to each peer worker.
        batch_idx: Global batch counter for ordering P2P exchanges.
    """

    send_nid: Tensor
    recv_nid: Tensor
    data2send: Tensor
    data2recv: Tensor
    ts2send: Tensor
    ts2recv: Tensor
    recv_counts: NDArray
    send_counts: NDArray
    batch_idx: int

    def __repr__(self) -> str:
        # if data is tensor, show its shape
        return (
            f"P2PTask(send_nid_shape={self.send_nid.shape}, "
            f"recv_nid_shape={self.recv_nid.shape}, "
            f"data2send_shape={self.data2send.shape}, "
            f"data2recv_shape={self.data2recv.shape}, "
            f"ts2send_shape={self.ts2send.shape}, "
            f"ts2recv_shape={self.ts2recv.shape}, "
            f"recv_counts={self.recv_counts}, "
            f"send_counts={self.send_counts}, "
            f"batch_idx={self.batch_idx})"
        )

    def clone(self) -> P2PTask:
        """Return a deep copy of this task with all tensors and arrays cloned."""
        return P2PTask(
            send_nid=self.send_nid.clone(),
            recv_nid=self.recv_nid.clone(),
            data2send=self.data2send.clone(),
            data2recv=self.data2recv.clone(),
            ts2send=self.ts2send.clone(),
            ts2recv=self.ts2recv.clone(),
            recv_counts=self.recv_counts.copy(),
            send_counts=self.send_counts.copy(),
            batch_idx=self.batch_idx,
        )
