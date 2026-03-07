import logging
import random
import threading

import numpy as np
import torch

from frost.aliases.host import DynamicPlanInQ, DynamicPlanOutQ, MetaBundle, MetasDyNeed
from frost.datatypes.plan import DynamicPlan
from frost.runtime._thread_base import ThreadBase
from frost.utils.tensor_set import TensorSet

logger = logging.getLogger(__name__)


def compute_p2p_transfer_plan(
    needs: list[TensorSet],
    avails: list[TensorSet],
) -> tuple[
    list[TensorSet],  # leftovers_on_gpu
    list[list[TensorSet]],  # p2p_transfer_plan
]:
    """Sender-centric fresh P2P plan: each sender distributes its new updates to
    receivers that need them next batch."""
    n_gpus = len(needs)

    ref_ts = None
    if n_gpus > 0:
        if len(needs[0]) > 0:
            ref_ts = needs[0]
        elif len(avails[0]) > 0:
            ref_ts = avails[0]

    if ref_ts is not None:
        empty_tensor = torch.tensor(
            [], dtype=ref_ts.data.dtype, device=ref_ts.data.device
        )
    else:
        empty_tensor = torch.tensor([], dtype=torch.long)

    empty_ts = TensorSet(empty_tensor)

    p2p_plan: list[list[TensorSet]] = [
        [empty_ts for _ in range(n_gpus)] for _ in range(n_gpus)
    ]
    leftovers: list[TensorSet] = []

    for s in range(n_gpus):
        new_update = avails[s]
        everything_sender_sends = empty_ts

        for r in range(n_gpus):
            need = needs[r]
            to_transfer = new_update & need

            if len(to_transfer) > 0:
                p2p_plan[s][r] = to_transfer
                everything_sender_sends = everything_sender_sends | to_transfer

        leftover = new_update - everything_sender_sends
        leftovers.append(leftover)

    return leftovers, p2p_plan


def handle_remaining(
    needs: list[TensorSet],
    avails: list[TensorSet],
    gpus_distance: np.ndarray,
) -> tuple[list[TensorSet], list[list[TensorSet]]]:
    """Receiver-centric stale P2P plan: cover unfulfilled needs from stale GPU memory,
    preferring the nearest sender (greedy matching by distance).

    Args:
        needs:  per-GPU sets of nodes still needed after fresh P2P is subtracted.
        avails: per-GPU stale memory (full fetch minus fresh updates).
        gpus_distance: NxN distance matrix from Planner.

    Returns:
        remaining_needs: nodes not covered by any GPU (must be fetched from CPU).
        stale_p2p:       [src][dst] transfer plan for stale nodes.
    """
    n_gpus = len(gpus_distance)

    ref_ts = None
    if n_gpus > 0:
        if len(needs[0]) > 0:
            ref_ts = needs[0]
        elif len(avails[0]) > 0:
            ref_ts = avails[0]

    if ref_ts is not None:
        empty_tensor = torch.tensor(
            [], dtype=ref_ts.data.dtype, device=ref_ts.data.device
        )
    else:
        empty_tensor = torch.tensor([], dtype=torch.long)

    empty_ts = TensorSet(empty_tensor)

    p2p_plan: list[list[TensorSet]] = [
        [empty_ts for _ in range(n_gpus)] for _ in range(n_gpus)
    ]

    unique_distances = sorted(np.unique(gpus_distance))

    for dist in unique_distances:
        pairs = np.argwhere(gpus_distance == dist)

        for s, r in pairs:
            if len(needs[r]) == 0:
                continue
            if len(avails[s]) == 0:
                continue

            to_transfer = avails[s] & needs[r]

            if len(to_transfer) > 0:
                p2p_plan[s][r] = p2p_plan[s][r] | to_transfer
                needs[r] = needs[r] - to_transfer

    return needs, p2p_plan


def deduplicate_updates_random(
    updates: list[TensorSet], seed: int | None = None
) -> tuple[list[TensorSet], TensorSet]:
    """Random deduplication: each node is assigned to exactly one GPU chosen at random."""
    n_gpus = len(updates)

    empty_tensor = torch.tensor([], dtype=torch.long)
    if n_gpus > 0 and updates[0].data.is_cuda:
        empty_tensor = empty_tensor.to(updates[0].data.device)

    empty_ts = TensorSet(empty_tensor)
    new_updates: list[TensorSet] = [empty_ts] * n_gpus

    if n_gpus == 0:
        return new_updates, empty_ts

    all_update = empty_ts
    indices = list(range(n_gpus))
    random.shuffle(indices)

    for i in indices:
        new_updates[i] = updates[i] - all_update
        all_update = all_update | updates[i]

    return new_updates, all_update


def deduplicate_updates(
    updates: list[TensorSet], reverse: bool = True
) -> tuple[list[TensorSet], TensorSet]:
    """Deterministic deduplication: higher-priority GPU (lower index when reverse=True)
    wins conflicting node ownership."""
    n_gpus = len(updates)

    empty_tensor = torch.tensor([], dtype=torch.long)
    if n_gpus > 0 and updates[0].data.is_cuda:
        empty_tensor = empty_tensor.to(updates[0].data.device)

    empty_ts = TensorSet(empty_tensor)
    new_updates: list[TensorSet] = [empty_ts] * n_gpus

    if n_gpus == 0:
        return new_updates, empty_ts

    all_update = empty_ts
    iterator = range(n_gpus - 1, -1, -1) if reverse else range(n_gpus)

    for i in iterator:
        new_updates[i] = updates[i] - all_update
        all_update = all_update | updates[i]

    return new_updates, all_update


class DynamicPlanner(ThreadBase):
    """Per-batch P2P and writeback planner for memory/mailbox.

    Args:
        handle_stale: When True, unfulfilled next-batch needs are served from stale
            GPU memory (nearest-sender greedy match) before falling back to CPU fetch.
            When False (default), all unfulfilled needs go straight to CPU fetch.
    """

    gpus_distance: np.ndarray
    n_gpus: int

    def __init__(
        self,
        gpus_distance: np.ndarray,
        in_q: DynamicPlanInQ,
        out_q: DynamicPlanOutQ,
        stop_ev: threading.Event,
        handle_stale: bool = False,
    ) -> None:
        super().__init__(name="dynamic_planner", stop_ev=stop_ev)
        self.gpus_distance = gpus_distance
        self.n_gpus = len(self.gpus_distance)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.handle_stale = handle_stale
        self.pending: MetasDyNeed | None = None

    def logic(self) -> None:
        """Main planning loop: consume (needs, updates, metas) tuples and emit DynamicPlans.

        Uses a one-step lookahead: each incoming batch is held as ``pending`` until
        the *next* batch arrives (or an end-of-epoch / shutdown sentinel), so that
        fresh P2P transfers can be pre-computed based on what nodes the next batch
        will need.  Sentinel handling:

        - ``None``: forward ``None`` to the output queue and exit.
        - ``"end_of_epoch"``: plan the pending batch with empty next-batch sets.
        """
        while not self.stop_ev.is_set():
            in_q_data = self.in_q.get(block=True)

            if in_q_data is None:
                self.out_q.put(None)
                break

            elif in_q_data == "end_of_epoch":
                assert self.pending is not None
                needs, updates, metas = self.pending

                if needs and len(needs) > 0:
                    ref_t = needs[0].data
                    empty_tensor = torch.tensor(
                        [], dtype=ref_t.dtype, device=ref_t.device
                    )
                else:
                    empty_tensor = torch.tensor([], dtype=torch.long)
                empty_ts = TensorSet(empty_tensor)

                # Last batch of the epoch: no next-batch needs or updates.
                next_needs: list[TensorSet] = [empty_ts for _ in range(self.n_gpus)]
                next_updates: list[TensorSet] = [empty_ts for _ in range(self.n_gpus)]

                plan = self._plan(needs, updates, next_needs, next_updates, metas)
                self.out_q.put(plan)
                self.pending = None
                continue

            if self.pending is None:
                needs, updates, metas = in_q_data
                self.pending = in_q_data
                self.fetch = needs  # first batch: fetch everything from CPU
            else:
                needs, updates, metas = self.pending
                plan = self._plan(needs, updates, in_q_data[0], in_q_data[1], metas)
                self.out_q.put(plan)
                self.pending = in_q_data

    def _plan(
        self,
        all_list: list[TensorSet],
        update_list: list[TensorSet],
        next_all_list: list[TensorSet],
        next_update_list: list[TensorSet],
        metas: MetaBundle,
    ) -> list[DynamicPlan]:
        """Build per-GPU DynamicPlan for the current batch.

        Args:
            all_list:        per-GPU full fetch set for the current batch.
            update_list:     per-GPU nodes updated by the current batch.
            next_all_list:   per-GPU full fetch set for the next batch.
            next_update_list: per-GPU nodes that will be updated by the next batch.
            metas:           batch metadata (for reference).
        """
        n_gpus = self.n_gpus

        # 1. Deduplicate: each updated node is owned by exactly one GPU.
        deduplicate_update_list, all_update = deduplicate_updates(update_list)

        # 2. Union of next batch's updates — nodes that will be refreshed anyway.
        next_all_update = TensorSet.empty_like(all_update.data)
        for update in next_update_list:
            next_all_update = next_all_update | update

        # 3. Fresh P2P: send this batch's updates to GPUs that need them next batch.
        _, fresh_p2p = compute_p2p_transfer_plan(next_all_list, deduplicate_update_list)

        # 4. Writeback: nodes updated here but NOT updated next batch must be persisted
        #    to CPU shared memory now (future batches will need to re-read them).
        writeback_list = [
            per_gpu_update - next_all_update
            for per_gpu_update in deduplicate_update_list
        ]

        # 5. Prefetch for next batch and optional stale P2P.
        if self.handle_stale:
            # Nodes next batch needs that fresh updates can't cover → try stale GPU mem.
            stale_needs = [next_need - all_update for next_need in next_all_list]
            # Stale avails: what this GPU fetched minus its own fresh updates.
            stale_avails = [avail - all_update for avail in all_list]
            remaining_needs, stale_p2p = handle_remaining(
                stale_needs, stale_avails, self.gpus_distance
            )
            # After stale P2P, only truly unavailable nodes need CPU fetch.
            next_prefetch = remaining_needs

            # Merge fresh + stale P2P (disjoint by construction).
            ref = all_update
            empty_ts = TensorSet(
                torch.tensor([], dtype=ref.data.dtype, device=ref.data.device)
            )
            p2p: list[list[TensorSet]] = [[empty_ts] * n_gpus for _ in range(n_gpus)]
            for src in range(n_gpus):
                for dst in range(n_gpus):
                    p2p[src][dst] = fresh_p2p[src][dst] | stale_p2p[src][dst]
        else:
            # No stale handling: unfulfilled needs go to CPU fetch.
            next_prefetch = [next_need - all_update for next_need in next_all_list]
            p2p = fresh_p2p

        # 6. Build per-GPU count matrix and plans.
        p2p_cts = np.zeros((n_gpus, n_gpus), dtype=np.int32)
        for src in range(n_gpus):
            for dst in range(n_gpus):
                p2p_cts[src, dst] = len(p2p[src][dst])

        plans = []
        for src in range(n_gpus):
            plans.append(
                DynamicPlan(
                    p2p=p2p[src],
                    fetch=self.fetch[src],
                    writeback=writeback_list[src],
                    recv_counts=p2p_cts[:, src],
                    send_counts=p2p_cts[src, :],
                )
            )

        self.fetch = next_prefetch
        return plans
