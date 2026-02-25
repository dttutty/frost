import math
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

# Assume these are your existing imports
from deprecated.sampler_core import ParallelSampler
from neg_sampler import NegLinkSampler
from utils import (
    dgl_blocks_to_tensor_dicts,  # Helper from previous answer
    extract_raw_data,
    node_to_dgl_blocks,
    sampler_wrapper,
    tensor_dicts_to_dgl_blocks,  # Helper from previous answer
)


# =====================================================================================
# Main Execution Logic
# =====================================================================================


def run_host(
    rank, world_size, g, df, sample_param, memory_param, gnn_param, train_param
):
    """
    The logic for the Host process (coordinator).
    It samples data, creates task packages, and dispatches them to workers.
    """
    ngpus = world_size - 1
    logger.debug(f"Host process (rank {rank}) started, coordinating {ngpus} workers.")

    # --- Initialize Samplers ---
    cfg_batch_size = train_param.batch_size
    train_edge_end = df[df["ext_roll"].gt(0)].index[0]
    val_edge_end = df[df["ext_roll"].gt(1)].index[0]

    sampler = None
    if not sample_param.no_sample:
        # Assuming g is a dictionary with graph data
        sampler = ParallelSampler(
            g["indptr"],
            g["indices"],
            g["eid"],
            g["ts"].astype(np.float32),
            sample_param.num_thread,
            1,
            sample_param.layer,
            sample_param.neighbor,
            sample_param.strategy == "recent",
            sample_param.prop_time,
            sample_param.history,
            float(sample_param.duration),
        )

    # In this model, only the host needs the negative samplers
    train_neg_link_sampler = NegLinkSampler(g)
    # create_sampler is likely a function that returns a specific sampler type
    # from sampler import create_sampler
    # eval_neg_link_sampler = create_sampler(g)

    # --- Main Epoch Loop ---
    best_ap = best_e = tap = tauc = 0
    for e in range(train_param.epoch):
        logger.debug(f"Epoch {e}:")
        sampler.reset()
        # The memory_mailbox object would be managed by workers now,
        # but the host can signal a reset.

        # --- Training Phase ---
        train_df = df[:train_edge_end]
        _, train_batch_size = plan_gpu_batches(len(train_df), cfg_batch_size, ngpus)
        train_batches = batches_assign(train_df.index, train_batch_size)

        t_start = time.time()
        for _, edges in tqdm(train_df.groupby(train_batches), desc="Training Batches"):
            # 1. Create the self-contained task package
            payload = create_task_package(
                edges,
                sampler,
                train_neg_link_sampler,
                gnn_param,
                sample_param,
                memory_param,
                mode="train",
            )
            # 2. Dispatch the task to all workers
            for i in range(ngpus):
                dist.send_object(payload, dst=i)

        # Signal workers that training for this epoch is done and get total loss
        # A more robust way is to get loss per batch, but for simplicity, we do it at the end
        total_loss = 0
        for i in range(ngpus):
            # This blocking receive also serves as a sync point
            loss_val = dist.recv_object(src=i)
            total_loss += loss_val

        logger.debug(
            f"\tTraining time: {time.time() - t_start:.2f}s, Approx. total loss: {total_loss / ngpus:.4f}"
        )

        # --- Evaluation Phase ---
        val_df = df[train_edge_end:val_edge_end]
        _, eval_batch_size = plan_gpu_batches(len(val_df), cfg_batch_size, ngpus)
        eval_batches = batches_assign(val_df.index, eval_batch_size)

        for _, edges in tqdm(val_df.groupby(eval_batches), desc="Validation Batches"):
            payload = create_task_package(
                edges,
                sampler,
                train_neg_link_sampler,
                gnn_param,  # Using train_neg_sampler for simplicity
                sample_param,
                memory_param,
                mode="eval",
            )
            for i in range(ngpus):
                dist.send_object(payload, dst=i)

        # Collect metrics from all workers
        ap_tot, auc_tot = [], []
        num_eval_batches = len(eval_batches.unique())
        for _ in range(num_eval_batches * ngpus):
            metrics = dist.recv_object()  # Receive from any worker
            ap_tot.append(metrics["ap"])
            auc_tot.append(metrics["auc"])

        ap = np.mean(ap_tot)
        auc = np.mean(auc_tot)
        logger.debug(f"\tValidation AP: {ap:.4f}, AUC: {auc:.4f}")

        # --- Check for Best Model and Test ---
        if ap > best_ap:
            best_ap = ap
            best_e = e
            # Signal workers to save their model state
            for i in range(ngpus):
                dist.send_object({"mode": "save_model"}, dst=i)
            # You might need a confirmation receive here

            # Run test phase... (similar logic to validation)
            # ...

    # --- Finalization ---
    logger.debug("Training finished. Sending exit signal to workers.")
    for i in range(ngpus):
        dist.send_object({"mode": "exit"}, dst=i)


def create_task_package(
    edges, sampler, neg_link_sampler, gnn_param, sample_param, memory_param, mode
):
    """
    This function encapsulates the sampling and data conversion logic.
    It returns a dictionary ready to be sent.
    """
    root_nodes = np.concatenate(
        [edges.src.values, edges.dst.values, neg_link_sampler.sample(edges.src.values)]
    ).astype(np.int32)
    ts = np.concatenate(
        [edges.time.values, edges.time.values, edges.time.values]
    ).astype(np.float32)

    ret, _ = sampler_wrapper(sample_param.no_neg, root_nodes, sampler, ts, None)

    if gnn_param.arch != "identity":
        grouped_blocks = extract_raw_data(ret, sample_param.history, save_on_pgu=False)
    else:
        grouped_blocks = node_to_dgl_blocks(root_nodes, ts, cuda=False)

    blocks_tensors = [
        dgl_blocks_to_tensor_dicts(block_list) for block_list in grouped_blocks
    ]

    special_block_tensors = None
    if memory_param.deliver_to == "neighbors":  # Simplified check
        special_block = extract_raw_data(
            ret, sample_param.history, reverse=True, save_on_pgu=False
        )[0][0]
        special_block_tensors = dgl_blocks_to_tensor_dicts([special_block])[0]

    # The self-contained task package
    payload = {
        "mode": mode,
        "data": {
            "blocks_tensors": blocks_tensors,
            "special_block_tensors": special_block_tensors,
            "root": torch.from_numpy(root_nodes),
            "ts": torch.from_numpy(ts),
            "eid": torch.from_numpy(edges["Unnamed: 0"].values),
        },
    }
    return payload


def run_worker(rank, world_size):
    """
    The logic for the Worker processes.
    It initializes the model and then enters a loop to receive and process tasks.
    """
    logger.debug(f"Worker {rank} started.")
    device = torch.device(f"cuda:{rank}")
    host_rank = world_size - 1

    # --- Initialize Model, Optimizer, MemoryMailbox ---
    # You need to define YourGNNModel, loss function, etc.
    # model = YourGNNModel(...).to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # optimizer = torch.optim.Adam(model.parameters(), lr=...)
    # memory_mailbox = MemoryMailbox(...)

    # --- Main Task Loop ---
    while True:
        # 1. Receive a task package from the host
        task_package = dist.recv_object(src=host_rank)
        mode = task_package["mode"]

        if mode == "exit":
            logger.debug(f"Worker {rank} received exit signal. Terminating.")
            break

        if mode == "save_model":
            # torch.save(model.state_dict(), f'model_rank_{rank}.pt')
            logger.debug(f"Worker {rank} saved model.")
            continue

        # If it's a data processing task, unpack the data
        data = task_package["data"]

        # 2. Reconstruct DGL blocks and move data to GPU
        blocks = [
            tensor_dicts_to_dgl_blocks(block_tensors, device)
            for block_tensors in data["blocks_tensors"]
        ]

        special_block = None
        if data["special_block_tensors"] is not None:
            special_block = tensor_dicts_to_dgl_blocks(
                [data["special_block_tensors"]], device
            )[0]

        root_nodes = data["root"].to(device)
        ts = data["ts"].to(device)
        eids = data["eid"].to(device)

        # 3. Execute task based on mode
        if mode == "train":
            # model.train()
            # optimizer.zero_grad()
            # pred_pos, pred_neg = model(blocks, node_feats_on_gpu, edge_feats_on_gpu, root_nodes, ts, eids, special_block)
            # loss = calculate_loss(pred_pos, pred_neg)
            # loss.backward() # DDP handles gradient sync
            # optimizer.step()
            # To send loss back at the end of epoch, you'd accumulate it here
            # For simplicity, sending a dummy value
            dist.send_object(0.1234, dst=host_rank)

        elif mode == "eval":
            # model.eval()
            with torch.no_grad():
                # pred_pos, pred_neg = model(blocks, node_feats_on_gpu, edge_feats_on_gpu,  root_nodes, ts, eids, special_block)
                # local_ap, local_auc = calculate_metrics(pred_pos, pred_neg)
                # Send metrics back to host for this specific batch
                dist.send_object(
                    {"ap": 0.95, "auc": 0.98}, dst=host_rank
                )  # Dummy values


# Helper functions like plan_gpu_batches and batches_assign should be defined as in your script
def plan_gpu_batches(num_samples, target_batch_size, ngpus=1):
    approx_iterations = num_samples // target_batch_size
    nbatch_per_gpu = max(approx_iterations // ngpus, 1)
    iteration_total = nbatch_per_gpu * ngpus
    adjusted_batch_size = math.ceil(num_samples / iteration_total)
    return iteration_total, adjusted_batch_size


def batches_assign(index, batch_size):
    return (index // batch_size).to_series()


if __name__ == "__main__":
    # Standard PyTorch distributed setup
    dist.init_process_group(backend="gloo")  # or 'nccl'
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    ngpus = world_size - 1

    # Load data `g` and `df` here
    # g = ...
    # df = ...

    # Parse args for sample_param, memory_param, gnn_param, train_param
    # ...

    if local_rank == ngpus:
        # This process is the Host
        # run_host(local_rank, world_size, g, df, sample_param, memory_param, gnn_param, train_param)
        pass  # Placeholder
    else:
        # These processes are the Workers
        # run_worker(local_rank, world_size)
        pass  # Placeholder

    dist.destroy_process_group()
