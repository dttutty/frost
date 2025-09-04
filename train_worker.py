# train_worker.py (key excerpts)
import torch
import torch.distributed as dist
from dist_loader import TemporalEdgesDataset, plan_gpu_batches
from dist_sampler import DistSampler
from sampler_core import ParallelSampler
import numpy as np
from tqdm import tqdm

def run_epoch_worker(
    e,
    ps_core: ParallelSampler,
    dist_sampler: DistSampler,
    df,
    train_edge_end,
    val_edge_end,
    gnn_param,
    memory_param,
    cfg_batch_size,
    sample_param,
    train_param,
    train_neg_link_sampler,
    eval_neg_link_sampler,
    model,
    optimizer,
    loss_fn,
    local_rank,
    world_size
):
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # build partitions
    train_df = df[:train_edge_end]
    val_df = df[train_edge_end:val_edge_end]
    test_df = df[val_edge_end:]

    # plan batches (consistent across ranks)
    ngpus = world_size
    train_iters, train_bsz = plan_gpu_batches(len(train_df), cfg_batch_size, ngpus)
    val_iters, _ = plan_gpu_batches(len(val_df), train_bsz, ngpus)

    # dataset per-rank iterator
    train_ds = TemporalEdgesDataset(train_df, train_bsz, reorder=train_param.reorder)
    val_ds = TemporalEdgesDataset(val_df, train_bsz, reorder=None)

    model.train()
    total_loss_local = 0.0

    with tqdm(total=train_iters + val_iters, disable=(local_rank != 0)) as pbar:
        # ---- Train ----
        for edges in train_ds.iter_rank(local_rank, world_size):
            # root nodes and timestamps
            root_nodes = np.concatenate([
                edges.src.values, edges.dst.values,
                train_neg_link_sampler.sample(edges.src.values, edges.time.values)
            ]).astype(np.int32)
            ts = np.concatenate([edges.time.values, edges.time.values, edges.time.values]).astype(np.float32)

            # sample with shared pointers
            blocks, sparse_updates = dist_sampler.sample_with_ptr(root_nodes, ts)

            # sync and apply pointer updates after sampling
            dist_sampler.sync_and_apply_updates(sparse_updates)

            # move blocks to tensors your model expects and run forward/backward
            # (you can reuse your existing to_dgl_blocks / tensorize paths)
            logits_pos, logits_neg = model(...)  # replace with your real call
            loss = loss_fn(logits_pos, logits_neg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss_local += float(loss.detach().cpu())
            if local_rank == 0:
                pbar.update(1)

        # loss reduce
        loss_tensor = torch.tensor([total_loss_local], dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = float(loss_tensor.item())

        # ---- Val ----
        model.eval()
        with torch.no_grad():
            ap_local, auc_local = 0.0, 0.0
            cnt_local = 0
            for edges in val_ds.iter_rank(local_rank, world_size):
                root_nodes = np.concatenate([
                    edges.src.values, edges.dst.values,
                    eval_neg_link_sampler.sample(edges.src.values, edges.time.values)
                ]).astype(np.int32)
                ts = np.concatenate([edges.time.values, edges.time.values, edges.time.values]).astype(np.float32)

                blocks, sparse_updates = dist_sampler.sample_with_ptr(root_nodes, ts)
                dist_sampler.sync_and_apply_updates(sparse_updates)

                # eval forward -> ap, auc for this mini-batch
                # ap_b, auc_b = ...
                # ap_local += ap_b; auc_local += auc_b; cnt_local += 1
                if local_rank == 0:
                    pbar.update(1)

            # reduce metrics
            m = torch.tensor([ap_local, auc_local, float(cnt_local)], device=device, dtype=torch.float64)
            dist.all_reduce(m, op=dist.ReduceOp.SUM)
            ap = float(m[0].item() / max(m[2].item(), 1.0))
            auc = float(m[1].item() / max(m[2].item(), 1.0))

    return total_loss, ap, auc
