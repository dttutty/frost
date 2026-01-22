# train_dist.py
import argparse
import os
import random
import time
from pathlib import Path

import Reddit
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset


# ---------- toy dataset ----------
class ToyDS(Dataset):
    def __init__(self, n=10000, dim=128, num_classes=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, dim, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ---------- toy model ----------
class MLP(nn.Module):
    def __init__(self, dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------- utils ----------
def setup_dist(args):
    # torchrun 会注入 RANK, WORLD_SIZE, LOCAL_RANK
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend=args.backend, timeout=torch.timeDyX(seconds=1800))
    torch.cuda.set_device(args.local_rank if torch.cuda.is_available() else 0)


def cleanup_dist():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main():
    return (
        (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    )


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_ckpt(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


# ---------- train/eval ----------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    # 聚合各 rank 的统计
    t = torch.tensor([correct, total], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    acc = (t[0] / t[1]).item()
    return acc


from multiprocessing import Process, Queue, cpu_count


SENTINEL = None


def producer(q: Queue, n=20):
    for i in range(n):
        q.put(i)
        time.sleep(random.random() * 0.05)
    for _ in range(cpu_count()):
        q.put(SENTINEL)


def consumer(q: Queue):
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        # 处理
        # ... do work ...


if __name__ == "__main__":
    q = Queue(maxsize=1000)
    ps = [Process(target=consumer, args=(q,)) for _ in range(cpu_count())]
    for p in ps:
        p.start()
    producer(q)
    for p in ps:
        p.join()


def train_one_epoch(model, edge_enumerator, sampler, opt, scaler, device, epoch, args):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    step = 0
    opt.zero_grad(set_to_none=True)

    # this is standard link prediction (given src, dst, time, predict if edge exists)
    for src, dst, ts in edge_enumerator:
        # sampler = Sampler(args.hist_sampler, args.train_neg_sampler, args.eval_neg_sampler, seed=args.seed)
        neg_dst = sampler.pick_train_neg(dst)
        root_nodes = [src, dst, neg_dst]  # already each sorted by timestamp
        topo = sampler.step(all_nodes, ts)
        all_nodes = all_nodes(topo)
        all_edges = all_edges(topo)

        uniq_nodes, nidx = torch.unique(torch.cat(root_nodes), return_inverse=True)
        uniq_edges, eidx = torch.unique(torch.cat(all_edges), return_inverse=True)

        uniq_nodes_feat = load_feat(uniq_nodes)
        uniq_edges_feat = load_feat(uniq_edges)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(x)
            loss = loss_fn(logits, y) / args.grad_accum

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.grad_accum == 0:
            if args.amp:
                scaler.unscale_(opt)
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        total_loss += loss.item() * args.grad_accum
        step += 1

    # 取平均并做 all-reduce
    loss_t = torch.tensor([total_loss, step], device=device, dtype=torch.float64)
    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
    mean_loss = (loss_t[0] / loss_t[1]).item()
    if is_main():
        print(f"epoch {epoch} train_loss {mean_loss:.4f}")
    return mean_loss


def main_worker(args):
    setup_dist(args)
    device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    if is_main():
        print(f"World size={args.world_size}, backend={args.backend}, device={device}")

    set_seed(args.seed)

    # Data
    reddit_dataset = Reddit(split=[6, 2, 2], dim=args.dim, seed=args.seed)

    def Sampler():
        pass

    sampler = Sampler(
        args.hist_sampler, args.train_neg_sampler, args.eval_neg_sampler, seed=args.seed
    )

    edge_enumerator = DataLoader(
        reddit_dataset,
        batch_size=args.batch_size,
        num_replicas=args.world_size,
        rank=args.rank,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    # Model/opt
    model = MLP(dim=args.dim, num_classes=args.num_classes).to(device)
    model = DDP(
        model,
        device_ids=[args.local_rank] if device.type == "cuda" else None,
        broadcast_buffers=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best_acc = 0.0
    ckpt_path = Path(args.out) / "last.pt"
    best_path = Path(args.out) / "best.pt"

    # resume
    if args.resume and ckpt_path.exists():
        map_location = device if device.type == "cuda" else "cpu"
        ckpt = load_ckpt(ckpt_path, map_location=map_location)
        model.module.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        if is_main():
            print(f"Resumed from epoch {ckpt['epoch']} best_acc={best_acc:.4f}")

    # train loop
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        _ = train_one_epoch(
            model, edge_enumerator, sampler, opt, scaler, device, epoch, args
        )
        # eval
        acc = evaluate(model, edge_enumerator, sampler, device)
        if is_main():
            print(f"epoch {epoch} val_acc {acc:.4f}")

            # save last
            save_ckpt(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc": best_acc,
                },
                ckpt_path,
            )

            # save best
            if acc > best_acc:
                best_acc = acc
                save_ckpt(
                    {
                        "model": model.module.state_dict(),
                        "acc": best_acc,
                        "epoch": epoch,
                    },
                    best_path,
                )

        dist.barrier()  # ensure all ranks sync on checkpoint boundary

    cleanup_dist()


def parse_args():
    p = argparse.ArgumentParser(description="PyTorch DDP training template")
    p.add_argument(
        "--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"]
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/ddp_template")
    p.add_argument("--resume", action="store_true")
    # data/model toy settings
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--train_size", type=int, default=20000)
    p.add_argument("--val_size", type=int, default=5000)
    args = p.parse_args()
    return args


if __name__ == "__main__":
    # 提示：用 torchrun 启动。LOCAL_RANK 等由 torchrun 注入。
    args = parse_args()
    main_worker(args)
