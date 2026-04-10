import os
import random
import sys
import threading
from pathlib import Path

import numpy as np
import torch

from frost.aliases.host import DynamicPlanInQ, DynamicPlanOutQ, MetaBundle, MetasDyNeed
from frost.datatypes.plan import DynamicPlan
from frost.runtime._thread_base import ThreadBase, put_shutdown_sentinel
from frost.utils.tensor_set import TensorSet

_PLAN_DUMP_LOCK = threading.Lock()


def _debug_force_cpu_freshness_single_worker() -> bool:
    flag = os.getenv("FROST_DEBUG_CPU_FRESHNESS_SINGLE_WORKER")
    if flag is None:
        return False
    return flag.lower() in {"1", "true", "yes", "on"}


def _should_dump_plans() -> bool:
    dump_env = os.getenv("FROST_DUMP_DYNAMIC_PLANS")
    if dump_env is not None:
        return dump_env.lower() in {"1", "true", "yes", "on"}
    return sys.gettrace() is not None


def _tensor_values(ts: TensorSet | None) -> list[int] | None:
    if ts is None:
        return None
    data = ts.data.detach().cpu()
    return [int(v) for v in data.tolist()]


def _tensor_set_lines(name: str, ts: TensorSet | None) -> list[str]:
    values = _tensor_values(ts)
    if values is None:
        return [f"{name}_count: None", f"{name}: None"]
    return [f"{name}_count: {len(values)}", f"{name}: {values}"]


def _format_plan_section(
    planner_kind: str,
    meta,
    plan: DynamicPlan,
    extra_lines: list[str],
) -> str:
    lines = [
        f"[{planner_kind}]",
        f"cur_epoch: {meta.cur_epoch}",
        f"split: {meta.split}",
        f"cur_macro_of_split: {meta.cur_macro_of_split}",
        f"cur_macro_of_epoch: {meta.cur_macro_of_epoch}",
        f"n_macro_of_split: {meta.n_macro_of_split}",
        f"n_src: {meta.n_src}",
        f"fanout: {meta.fanout}",
        *extra_lines,
        *_tensor_set_lines("plan_fetch", plan.fetch),
        *_tensor_set_lines("plan_writeback", plan.writeback),
        f"plan_recv_counts: {plan.recv_counts.tolist()}",
        f"plan_send_counts: {plan.send_counts.tolist()}",
    ]
    for peer_idx, peer_plan in enumerate(plan.p2p):
        lines.extend(_tensor_set_lines(f"plan_p2p_to_gpu_{peer_idx}", peer_plan))
    lines.append("")
    return "\n".join(lines)


def _append_plan_dump(
    plan_dump_dir: Path | None,
    planner_kind: str,
    metas: MetaBundle,
    plans: list[DynamicPlan],
    extra_lines_per_gpu: list[list[str]],
) -> None:
    if plan_dump_dir is None:
        return

    with _PLAN_DUMP_LOCK:
        plan_dump_dir.mkdir(parents=True, exist_ok=True)
        for meta, plan, extra_lines in zip(
            metas, plans, extra_lines_per_gpu, strict=True
        ):
            dump_path = plan_dump_dir / f"{meta.mini_batch_idx}_plan.log"
            with dump_path.open("a", encoding="ascii") as f:
                f.write(_format_plan_section(planner_kind, meta, plan, extra_lines))


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
    updates: list[TensorSet], prefer_higher_rank: bool = True
) -> tuple[list[TensorSet], TensorSet]:
    """Deterministic deduplication: when prefer_higher_rank=True,
    higher-rank GPU wins conflicting node ownership."""
    n_gpus = len(updates)

    empty_tensor = torch.tensor([], dtype=torch.long)
    if n_gpus > 0 and updates[0].data.is_cuda:
        empty_tensor = empty_tensor.to(updates[0].data.device)

    empty_ts = TensorSet(empty_tensor)
    new_updates: list[TensorSet] = [empty_ts] * n_gpus

    if n_gpus == 0:
        return new_updates, empty_ts

    all_update = empty_ts
    iterator = range(n_gpus - 1, -1, -1) if prefer_higher_rank else range(n_gpus)

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
        output_dir: str | None = None,
        plan_kind: str = "dynamic",
        persist_all_updates_to_cpu: bool = False,
    ) -> None:
        super().__init__(name="dynamic_planner", stop_ev=stop_ev)
        self.gpus_distance = gpus_distance
        self.n_gpus = len(self.gpus_distance)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_ev = stop_ev
        self.handle_stale = handle_stale
        self.pending: MetasDyNeed | None = None
        self.plan_kind = plan_kind
        self.persist_all_updates_to_cpu = persist_all_updates_to_cpu
        self.plan_dump_dir = (
            Path(output_dir) / "detail"
            if output_dir is not None and _should_dump_plans()
            else None
        )

    def logic(self) -> None:
        """Main planning loop: consume (needs, updates, raw_need_counts, metas) tuples and emit DynamicPlans.

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
                put_shutdown_sentinel(self.out_q, None)
                break

            elif in_q_data == "end_of_epoch":
                assert self.pending is not None
                all_needs, all_pos_root, raw_need_counts, metas = self.pending

                if all_needs and len(all_needs) > 0:
                    ref_t = all_needs[0].data
                    empty_tensor = torch.tensor(
                        [], dtype=ref_t.dtype, device=ref_t.device
                    )
                else:
                    empty_tensor = torch.tensor([], dtype=torch.long)
                empty_ts = TensorSet(empty_tensor)

                # Last batch of the epoch: no next-batch needs or updates.
                next_needs: list[TensorSet] = [empty_ts for _ in range(self.n_gpus)]
                next_updates: list[TensorSet] = [empty_ts for _ in range(self.n_gpus)]

                plan = self._plan(
                    all_needs,
                    all_pos_root,
                    next_needs,
                    next_updates,
                    metas,
                    raw_need_counts=raw_need_counts,
                )
                self.out_q.put(plan)
                self.pending = None
                continue

            if self.pending is None:
                all_needs, all_pos_root, raw_need_counts, metas = in_q_data
                self.pending = in_q_data
                self.fetch = all_needs  # first batch: fetch everything from CPU
            else:
                all_needs, all_pos_root, raw_need_counts, metas = self.pending
                plan = self._plan(
                    all_needs,
                    all_pos_root,
                    in_q_data[0],
                    in_q_data[1],
                    metas,
                    raw_need_counts=raw_need_counts,
                )
                self.out_q.put(plan)
                self.pending = in_q_data

    def _plan(
        self,
        all_list: list[TensorSet],
        update_list: list[TensorSet],
        next_all_list: list[TensorSet],
        next_update_list: list[TensorSet],
        metas: MetaBundle,
        raw_need_counts: list[int] | None = None,
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
        if raw_need_counts is None:
            raw_need_counts = [len(ts) for ts in all_list]

        # Hidden debug mode for single-worker diagnosis:
        # - persist every current update to CPU
        # - fetch every next-batch need from CPU
        # - disable fresh self-P2P chaining
        # Together with a 1-batch-in-flight worker pipeline, this approximates
        # a strictly CPU-synchronized freshness baseline.
        if _debug_force_cpu_freshness_single_worker() and n_gpus == 1:
            deduplicate_update_list, _ = deduplicate_updates(update_list)
            empty_ts = TensorSet.empty_like(deduplicate_update_list[0].data)
            p2p = [[empty_ts]]
            p2p_cts = np.zeros((1, 1), dtype=np.int32)

            plans = [
                DynamicPlan(
                    p2p=p2p[0],
                    fetch=self.fetch[0],
                    writeback=deduplicate_update_list[0],
                    recv_counts=p2p_cts[:, 0],
                    send_counts=p2p_cts[0, :],
                )
            ]

            _append_plan_dump(
                self.plan_dump_dir,
                self.plan_kind,
                metas,
                plans,
                [
                    [
                        "debug_cpu_freshness_single_worker: True",
                        *_tensor_set_lines("current_need", all_list[0]),
                        *_tensor_set_lines("current_update", update_list[0]),
                        *_tensor_set_lines(
                            "deduplicated_update", deduplicate_update_list[0]
                        ),
                        *_tensor_set_lines("next_need", next_all_list[0]),
                        *_tensor_set_lines("next_update", next_update_list[0]),
                        *_tensor_set_lines("next_prefetch", next_all_list[0]),
                    ]
                ],
            )

            self.fetch = next_all_list
            return plans

        # 1. Deduplicate: each updated node is owned by exactly one GPU.
        deduplicate_update_list, all_update = deduplicate_updates(update_list)

        # 2. Union of next batch's updates — nodes that will be refreshed anyway.
        next_all_update = TensorSet.empty_like(all_update.data)
        for update in next_update_list:
            next_all_update = next_all_update | update

        # 3. Fresh P2P: send this batch's updates to GPUs that need them next batch.
        _, fresh_p2p = compute_p2p_transfer_plan(next_all_list, deduplicate_update_list)

        # 4. Writeback policy:
        #    - Shared-memory direct-fetch mode must persist every owned update now,
        #      because later readers bypass P2P and re-read remote state from CPU.
        #    - The one-step deferred writeback optimization is only sound when stale
        #      P2P is enabled: if the next batch consumes a fresh delta but does not
        #      update that node again, a later batch may need to fetch the node from
        #      CPU. Without stale-P2P coverage the CPU copy would remain stale.
        if self.persist_all_updates_to_cpu or not self.handle_stale:
            writeback_list = deduplicate_update_list
        else:
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

        _append_plan_dump(
            self.plan_dump_dir,
            self.plan_kind,
            metas,
            plans,
            [
                [
                    *_tensor_set_lines("current_need", all_list[src]),
                    *_tensor_set_lines("current_update", update_list[src]),
                    *_tensor_set_lines(
                        "deduplicated_update", deduplicate_update_list[src]
                    ),
                    *_tensor_set_lines("next_need", next_all_list[src]),
                    *_tensor_set_lines("next_update", next_update_list[src]),
                    *_tensor_set_lines("next_prefetch", next_prefetch[src]),
                ]
                for src in range(n_gpus)
            ],
        )

        self.fetch = next_prefetch
        return plans


class APANDynamicPlanner(DynamicPlanner):
    def _plan(
        self,
        box_fetch_list: list[TensorSet],
        box_update_list: list[TensorSet],
        next_box_fetch_list: list[TensorSet],
        next_box_update_list: list[TensorSet],
        metas: MetaBundle,
        raw_need_counts: list[int] | None = None,
    ) -> list[DynamicPlan]:
        n_gpus = self.n_gpus
        if raw_need_counts is None:
            raw_need_counts = [len(ts) for ts in box_fetch_list]

        # 1. Deduplicate: each updated node is owned by exactly one GPU.
        # box_update_list = nodes that postman delivers mail to (src+dst + their neighbors)
        deduplicate_update_list, all_update = deduplicate_updates(box_update_list)

        # 2. Fresh P2P: send this batch's updates to GPUs that need them next batch.
        # next_box_fetch_list = next batch's pos_root_need (src+dst that need box read)
        _, p2p = compute_p2p_transfer_plan(next_box_fetch_list, deduplicate_update_list)

        # 3. Writeback all updated nodes to keep ring buffer state consistent on CPU.
        writeback_list = deduplicate_update_list

        # 4. Build per-GPU count matrix and plans.
        p2p_cts = np.zeros((n_gpus, n_gpus), dtype=np.int32)
        for src in range(n_gpus):
            for dst in range(n_gpus):
                p2p_cts[src, dst] = len(p2p[src][dst])

        plans = []
        for src in range(n_gpus):
            plans.append(
                DynamicPlan(
                    p2p=p2p[src],
                    fetch=box_fetch_list[src],
                    writeback=writeback_list[src],
                    recv_counts=p2p_cts[:, src],
                    send_counts=p2p_cts[src, :],
                )
            )

        _append_plan_dump(
            self.plan_dump_dir,
            self.plan_kind,
            metas,
            plans,
            [
                [
                    *_tensor_set_lines("current_box_fetch", box_fetch_list[src]),
                    *_tensor_set_lines("current_box_update", box_update_list[src]),
                    *_tensor_set_lines(
                        "deduplicated_box_update", deduplicate_update_list[src]
                    ),
                    *_tensor_set_lines("next_box_fetch", next_box_fetch_list[src]),
                    *_tensor_set_lines("next_box_update", next_box_update_list[src]),
                ]
                for src in range(n_gpus)
            ],
        )

        return plans
