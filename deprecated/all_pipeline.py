#!/usr/bin/env python3
"""
Temporal-GNN multi-stage pipeline with controllable batch staleness.

Stages:
  P0  : Sampler (produces BatchSpec of 3 microbatches per batch)
  CP1 : Feature loader + data mover orchestrator
  H2D : Per-GPU host→device feeder with ring buffer (size=3 by default)
  CP2 : Trainers (3 procs = 3 GPUs), gradient accumulation over 3 micros
  CP3 : Memory aggregator + persistence + commit cursor

Staleness controller:
  committed_bid: int  (largest batch-id whose memory has been persisted)
  S            : int  ( -1: unlimited, 0: strict, k>=0: allow up to k leading uncommitted batches )

Run examples (single node):
  # Fast demo (seconds-scale sleeps):
  CUDA_VISIBLE_DEVICES=0,1,2 python tgl_pipeline_staleness_demo.py \
    --batches 8 --staleness 0

  # Throughput mode (looser consistency):
  CUDA_VISIBLE_DEVICES=0,1,2 python tgl_pipeline_staleness_demo.py \
    --batches 8 --staleness 2

  # No limits:
  CUDA_VISIBLE_DEVICES=0,1,2 python tgl_pipeline_staleness_demo.py \
    --batches 8 --staleness -1

  # If you really want to emulate your stated latencies:
  CUDA_VISIBLE_DEVICES=0,1,2 python tgl_pipeline_staleness_demo.py \
    --batches 4 --staleness 0 --t-sample 10 --t-h2d 8 --t-io 2

Notes:
  * If no CUDA is available, the script will fallback to CPU+gloo backend; timings still illustrate the pipeline.
  * For Windows, keep world_size small and ports open; mp start_method is spawn.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pathlib
import pickle
import socket
import time
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from multiprocessing.synchronize import Event as MEvent
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn


def find_free_port():
    """
    Finds and returns an available TCP port on the system.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# -----------------------------
# Config defaults
# -----------------------------
FEAT_DIM = 128
MICRO_SIZE = 2048  # number of nodes per microbatch (dummy)
WORLD_SIZE_DEFAULT = 2
RING_SIZE_DEFAULT = 2

# -----------------------------
# Data classes
# -----------------------------


@dataclass
class MicroSpec:
    bid: int
    mid: int  # 0..world_size-1 (also equals device index in this toy setup)
    node_ids: list[int]
    aux: dict[str, Any]


@dataclass
class MicroOnGPU:
    bid: int
    mid: int
    device: int
    features_dev: torch.Tensor  # (MICRO_SIZE, FEAT_DIM)
    node_ids: list[int]


@dataclass
class GradSignal:
    bid: int
    mid: int
    device: int
    new_memory_DyX: dict[int, Any]


@dataclass
class RootNodesWithTs:
    src: list[int]
    dst: list[int]
    time: list[float]


@dataclass
class BatchSpec:
    bid: int
    micros: list[MicroSpec]


# Sentinel for queues
SENTINEL = None

# -----------------------------
# Logging helpers
# -----------------------------


def ts() -> str:
    return time.strftime("%H:%M:%S")


def log(who: str, msg: str) -> None:
    logger.debug(f"[{ts()}] [{who}] {msg}", flush=True)


# -----------------------------
# Staleness controller
# -----------------------------


def can_start_bid(bid: int, committed_bid: int, S: int) -> bool:
    s = S.value
    c = committed_bid.value
    if s < 0:
        return True
    # allow up to S leading uncommitted batches → max allowed is c + S + 1
    return bid <= c + s + 1


def wait_until_can_start(
    who: str, bid: int, committed_bid: int, S: int, cond: mp.synchronize.Condition
):
    with cond:
        while not can_start_bid(bid, committed_bid, S):
            cond.wait(timeout=0.1)
        # Optional: one-shot log when gate opens
        # log(who, f"gate open for bid={bid} (committed={committed_bid.value}, S={S.value})")


# -----------------------------
# Dummy domain helpers
# -----------------------------


def sample_nodes_for_micro(
    bid: int,
    mid: int,
    src_meta: tuple[str, tuple[int, ...], Any],
    dst_meta: tuple[str, tuple[int, ...], Any],
    ts_meta: tuple[str, tuple[int, ...], Any],
    train_edge_end: int,
    val_edge_end: int,
) -> list[int]:
    # Deterministic-ish sample for visibility
    src_shm_name, src_shape, src_dtype = src_meta
    src_shm = shared_memory.SharedMemory(name=src_shm_name)
    src_values = np.ndarray(src_shape, dtype=src_dtype, buffer=src_shm.buf)

    # Process 'dst' data
    dst_shm_name, dst_shape, dst_dtype = dst_meta
    dst_shm = shared_memory.SharedMemory(name=dst_shm_name)
    dst_values = np.ndarray(dst_shape, dtype=dst_dtype, buffer=dst_shm.buf)

    # Process 'time' data
    ts_shm_name, ts_shape, ts_dtype = ts_meta
    ts_shm = shared_memory.SharedMemory(name=ts_shm_name)
    time_values = np.ndarray(ts_shape, dtype=ts_dtype, buffer=ts_shm.buf)

    # train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    # val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    # train_df = df[:train_edge_end]
    # val_df = df[train_edge_end:val_edge_end]
    # test_df = df[val_edge_end:]

    selected_df_start = (bid * WORLD_SIZE_DEFAULT + mid) * 600
    selected_df_end = (bid * WORLD_SIZE_DEFAULT + mid) * 600 + 600

    # subset_df = train_df.iloc[selected_df_start:selected_df_end]

    src_values = src_values[selected_df_start:selected_df_end]
    dst_values = dst_values[selected_df_start:selected_df_end]
    time_values = time_values[selected_df_start:selected_df_end]

    root_nodes_with_ts = RootNodesWithTs(
        src=src_values, dst=dst_values, time=time_values
    )
    return root_nodes_with_ts


def load_features_pinned(micro: MicroSpec, use_cuda: bool) -> torch.Tensor:
    # Simulate feature IO by generating random tensor in pinned memory
    x = torch.randn(MICRO_SIZE, FEAT_DIM, pin_memory=use_cuda)
    return x


def detach_to_cpu(DyX: dict[int, Any]) -> dict[int, Any]:
    # Ensure CPU-serializable content
    out = {}
    for k, v in DyX.items():
        if torch is not None and isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = v
    return out


# -----------------------------
# Processes
# -----------------------------


def sampler_proc(
    q01: mp.Queue,
    src_meta: tuple[str, tuple[int, ...], Any],
    dst_meta: tuple[str, tuple[int, ...], Any],
    ts_meta: tuple[str, tuple[int, ...], Any],
    train_edge_end: int,
    val_edge_end: int,
    cond: mp.synchronize.Condition,
    committed_bid: int,
    S: int,
    stop_ev: MEvent,
    batches: int,
    t_sample: float,
    world_size: int,
) -> None:
    who = "P0"
    for bid in range(batches):
        # If you want strict freshest-memory-aware sampling, uncomment this line:
        # wait_until_can_start(who, bid, committed_bid, S, cond)
        t0 = time.time()
        micros: list[MicroSpec] = []
        for mid in range(world_size):
            root_nodes_with_ts = sample_nodes_for_micro(
                bid, mid, src_meta, dst_meta, ts_meta, train_edge_end, val_edge_end
            )
            micros.append(
                MicroSpec(bid=bid, mid=mid, node_ids=root_nodes_with_ts, aux={})
            )
        q01.put(BatchSpec(bid=bid, micros=micros))
        log(who, f"emitted BatchSpec bid={bid}")
    # signal end
    q01.put(SENTINEL)
    log(who, "done")


def cp1_proc(
    q01: mp.Queue, q12_devs: list[mp.Queue], stop_ev: MEvent, world_size: int
) -> None:
    who = "CP1"
    while True:
        item = q01.get()
        if item is SENTINEL:
            for d in range(world_size):
                q12_devs[d].put(SENTINEL)
            log(who, "done")
            return
        assert isinstance(item, BatchSpec)
        # Distribute micros to device-specific feeders (mid==device by convention)
        for micro in item.micros:
            q12_devs[micro.mid].put(micro)
        log(who, f"dispatched bid={item.bid} → per-GPU feeders")


def h2d_feeder_proc(
    dev: int,
    q12_dev: mp.Queue,
    ring_dev: mp.Queue,
    stop_ev: MEvent,
    t_h2d: float,
    use_cuda: bool,
) -> None:
    who = f"H2D:d{dev}"
    if use_cuda:
        torch.cuda.set_device(dev)
        h2d_stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    else:
        h2d_stream = None
    while True:
        item = q12_dev.get()
        if item is SENTINEL:
            ring_dev.put(SENTINEL)
            log(who, "done")
            return
        micro: MicroSpec = item
        # Load features into pinned
        feats_pin = load_features_pinned(micro, use_cuda)
        # emulate H2D latency
        if t_h2d > 0:
            time.sleep(t_h2d)
        if use_cuda:
            with torch.cuda.stream(h2d_stream):
                feats_dev = feats_pin.to(dev, non_blocking=False)
            h2d_stream.synchronize()
        else:
            feats_dev = feats_pin  # CPU fallback
        ring_dev.put(
            MicroOnGPU(
                bid=micro.bid,
                mid=micro.mid,
                device=dev,
                features_dev=feats_dev,
                node_ids=micro.node_ids,
            )
        )
        # Keep ring_dev bounded (Queue capacity limits ring size)
        # No explicit log each micro to avoid spam


class TinyModel(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_loss_and_memory(
    out: torch.Tensor, node_ids: list[int]
) -> tuple[torch.Tensor, dict[int, Any]]:
    # Dummy supervised loss: drive outputs to 0
    target = torch.zeros_like(out)
    loss = torch.nn.functional.mse_loss(out, target)
    # Memory DyX: map a small subset of node_ids to a tiny vector (mean/std of this micro)
    # NOTE: in real TGNN you would return learned temporal memory DyXs.
    k = min(32, len(node_ids))
    sel_ids = node_ids[:k]
    mean = out.detach().mean()
    std = out.detach().std()
    DyX = {nid: torch.stack([mean, std]) for nid in sel_ids}
    return loss, DyX


def dist_init(
    backend: str, rank: int, world_size: int, master_addr: str, master_port: int
):
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def trainer_proc(
    dev: int,
    ring_dev: mp.Queue,
    q23: mp.Queue,
    cond: mp.synchronize.Condition,
    committed_bid: int,
    S: int,
    stop_ev: MEvent,
    master_addr: str,
    master_port: int,
    world_size: int,
    amp: bool,
):
    who = f"CP2:d{dev}"
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    torch.cuda.set_device(dev) if use_cuda else None

    # rank == dev by construction here
    dist_init(
        backend=backend,
        rank=dev,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    )

    model = TinyModel(FEAT_DIM)
    if use_cuda:
        model = model.to(dev)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[dev] if use_cuda else None
    )
    opt = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and use_cuda)

    current_bid = None
    accum = 0

    while True:
        item = ring_dev.get()
        if item is SENTINEL:
            break
        micro: MicroOnGPU = item

        # Gate on batch boundary
        if current_bid != micro.bid:
            wait_until_can_start(who, micro.bid, committed_bid, S, cond)
            current_bid = micro.bid
            accum = 0
            opt.zero_grad(set_to_none=True)
            log(
                who,
                f"START bid={current_bid} (committed={committed_bid.value}, S={S.value})",
            )

        # Forward/backward with optional no_sync() to accumulate across micros
        use_no_sync = accum < world_size - 1
        cm = ddp_model.no_sync() if use_no_sync else nullcontext()
        with cm:
            with torch.cuda.amp.autocast(enabled=amp and use_cuda):
                out = ddp_model(micro.features_dev)
                loss, DyX = compute_loss_and_memory(out, micro.node_ids)
            scaler.scale(loss).backward()
        accum += 1

        if accum == world_size:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            accum = 0
            log(who, f"STEP bid={current_bid}")

        # Emit memory DyX per micro
        q23.put(
            GradSignal(
                bid=micro.bid,
                mid=micro.mid,
                device=dev,
                new_memory_DyX=detach_to_cpu(DyX),
            )
        )

    # Send one sentinel to CP3 when this rank is done
    q23.put(SENTINEL)
    log(who, "done")
    dist.destroy_process_group()


from contextlib import contextmanager


@contextmanager
def nullcontext():
    yield


def merge_memory_DyXs(DyX_dict: dict[int, dict[int, Any]]) -> dict[int, Any]:
    # DyX_dict: mid -> {nid -> vec}
    merged: dict[int, Any] = {}
    # simple last-writer-wins merge by mid order
    for mid in sorted(DyX_dict.keys()):
        merged.update(DyX_dict[mid])
    return merged


def cp3_proc(
    q23: mp.Queue,
    cond: mp.synchronize.Condition,
    committed_bid: int,
    stop_ev: MEvent,
    world_size: int,
    t_io: float,
    out_dir: str,
):
    who = "CP3"
    inbox: dict[int, dict[int, Any]] = defaultdict(dict)  # bid -> mid -> DyX
    done_signals = 0
    outp = pathlib.Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    while True:
        item = q23.get()
        if item is SENTINEL:
            done_signals += 1
            if done_signals == world_size:
                # finish when all trainers signaled done and inbox drained
                break
            else:
                continue
        sig: GradSignal = item
        inbox[sig.bid][sig.mid] = sig.new_memory_DyX
        # when full set of micros for this bid arrived
        if len(inbox[sig.bid]) == world_size:
            merged = merge_memory_DyXs(inbox[sig.bid])
            # emulate IO latency
            if t_io > 0:
                time.sleep(t_io)
            # write to disk
            f = outp / f"memory_bid_{sig.bid:06d}.pkl"
            with open(f, "wb") as fh:
                pickle.dump(merged, fh)
            del inbox[sig.bid]
            # advance commit cursor (monotonic non-decreasing)
            with cond:
                committed_bid.value = max(committed_bid.value, sig.bid)
                cond.notify_all()
            log(who, f"COMMIT bid={sig.bid} → committed={committed_bid.value}")

    # Flush any stragglers (should be empty unless asymmetric termination)
    if inbox:
        log(who, f"leftover inbox keys={list(inbox.keys())}")
    log(who, "done")


# -----------------------------
# Main wiring
# -----------------------------
from multiprocessing import shared_memory

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE_DEFAULT)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument(
        "--staleness",
        type=int,
        default=0,
        help="-1=unlimited, 0=strict, k>=0 allows k leading uncommitted",
    )
    parser.add_argument("--ring-size", type=int, default=RING_SIZE_DEFAULT)
    parser.add_argument("--t-sample", type=float, default=1.0)
    parser.add_argument("--t-h2d", type=float, default=0.5)
    parser.add_argument("--t-io", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true", help="Enable autocast+GradScaler")
    parser.add_argument("--out-dir", type=str, default="./_tgmem_out")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=None)
    args = parser.parse_args()

    free_port = find_free_port()
    if args.master_port is None:
        args.master_port = free_port
    logger.debug(f"Using port {args.master_port}")
    ws = args.world_size
    assert ws >= 1
    from utils import load_graph

    g, df = load_graph("REDDIT")
    src_numpy_array = df["src"].values
    dst_numpy_array = df["dst"].values
    ts_numpy_array = df["time"].values

    train_edge_end = df[df["ext_roll"].gt(0)].index[0]
    val_edge_end = df[df["ext_roll"].gt(1)].index[0]

    num_nodes = g["indptr"].shape[0] - 1

    src_shm = shared_memory.SharedMemory(create=True, size=src_numpy_array.nbytes)
    dst_shm = shared_memory.SharedMemory(create=True, size=dst_numpy_array.nbytes)
    ts_shm = shared_memory.SharedMemory(create=True, size=ts_numpy_array.nbytes)
    shared_array_main_src = np.ndarray(
        src_numpy_array.shape, dtype=src_numpy_array.dtype, buffer=src_shm.buf
    )
    shared_array_main_dst = np.ndarray(
        dst_numpy_array.shape, dtype=dst_numpy_array.dtype, buffer=dst_shm.buf
    )
    shared_array_main_ts = np.ndarray(
        ts_numpy_array.shape, dtype=ts_numpy_array.dtype, buffer=ts_shm.buf
    )
    shared_array_main_src[:] = src_numpy_array[:]
    shared_array_main_dst[:] = dst_numpy_array[:]
    shared_array_main_ts[:] = ts_numpy_array[:]

    src_meta = (src_shm.name, src_numpy_array.shape, src_numpy_array.dtype)
    dst_meta = (dst_shm.name, dst_numpy_array.shape, dst_numpy_array.dtype)
    ts_meta = (ts_shm.name, ts_numpy_array.shape, ts_numpy_array.dtype)

    # CUDA sanity
    use_cuda = torch is not None and torch.cuda.is_available()
    if use_cuda:
        ndev = torch.cuda.device_count()
        if ndev < ws:
            raise RuntimeError(f"Need at least {ws} CUDA devices, found {ndev}")
        log("MAIN", f"CUDA devices={ndev}, using 0..{ws - 1}")
    else:
        log("MAIN", "CUDA not available → CPU fallback + gloo")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()

    # Shared state
    committed_bid = manager.Value("i", -1)
    S = manager.Value("i", int(args.staleness))
    cond = manager.Condition()
    stop_ev = manager.Event()

    # Queues (bounded → backpressure)
    q01 = mp.Queue(maxsize=8)
    q12_devs = [mp.Queue(maxsize=16) for _ in range(ws)]
    ring_devs = [mp.Queue(maxsize=args.ring_size) for _ in range(ws)]
    q23 = mp.Queue(maxsize=64)

    procs: list[mp.Process] = []

    # P0
    procs.append(
        mp.Process(
            target=sampler_proc,
            name="P0",
            args=(
                q01,
                src_meta,
                dst_meta,
                ts_meta,
                train_edge_end,
                val_edge_end,
                cond,
                committed_bid,
                S,
                stop_ev,
                args.batches,
                args.t_sample,
                ws,
            ),
        )
    )

    # CP1
    procs.append(
        mp.Process(target=cp1_proc, name="CP1", args=(q01, q12_devs, stop_ev, ws))
    )

    # H2D feeders per device
    for d in range(ws):
        procs.append(
            mp.Process(
                target=h2d_feeder_proc,
                name=f"H2D:{d}",
                args=(d, q12_devs[d], ring_devs[d], stop_ev, args.t_h2d, use_cuda),
            )
        )

    # CP2 trainers per device
    for d in range(ws):
        procs.append(
            mp.Process(
                target=trainer_proc,
                name=f"CP2:{d}",
                args=(
                    d,
                    ring_devs[d],
                    q23,
                    cond,
                    committed_bid,
                    S,
                    stop_ev,
                    args.master_addr,
                    int(args.master_port),
                    ws,
                    args.amp,
                ),
            )
        )

    # CP3
    procs.append(
        mp.Process(
            target=cp3_proc,
            name="CP3",
            args=(q23, cond, committed_bid, stop_ev, ws, args.t_io, args.out_dir),
        )
    )

    # Start
    for p in procs:
        p.start()

    # Join
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        log("MAIN", "KeyboardInterrupt → stopping")
        stop_ev.set()
        for p in procs:
            p.terminate()
    finally:
        # Final state
        log("MAIN", f"FINAL committed_bid={committed_bid.value}")


if __name__ == "__main__":
    main()
