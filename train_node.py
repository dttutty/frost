import argparse
import hashlib
import os


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="dataset name")
parser.add_argument("--config", type=str, default="", help="path to config file")
parser.add_argument("--batch_size", type=int, default=4000)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gpu", type=str, default="0", help="which GPU to use")
parser.add_argument(
    "--model", type=str, default="", help="name of stored model to load"
)
parser.add_argument(
    "--posneg",
    default=False,
    action="store_true",
    help="for positive negative detection, whether to sample negative nodes",
)
parser.add_argument("--dedup", action="store_true", help="enable dedup")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.data == "WIKI" or args.data == "REDDIT":
    args.posneg = True

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm

from modules import *
from sampler import *
from utils import *


ldf = pd.read_csv("DATA/{}/labels.csv".format(args.data))
role = ldf["ext_roll"].values
# train_node_end = ldf[ldf['ext_roll'].gt(0)].index[0]
# val_node_end = ldf[ldf['ext_roll'].gt(1)].index[0]
labels = ldf["label"].values.astype(np.int64)


from utils import (
    get_node_edge_dimensions,
)


def generate_temporal_embeddings(ldf, emb_file_name):
    node_np, edge_np = load_feat(args.data)
    g, df = load_graph(args.data)
    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
    train_edge_end = df[df["ext_roll"].gt(0)].index[0]
    val_edge_end = df[df["ext_roll"].gt(1)].index[0]

    _, node_feat_dim, _, _, edge_feat_dim, _ = get_node_edge_dimensions(
        node_np, edge_np
    )
    model = GeneralModel(
        node_feat_dim,
        edge_feat_dim,
        sample_param,
        memory_param,
        gnn_param,
        train_param,
        combined=bool(train_param.merge_src),
    ).cuda()
    shared_cfg = {
        "node_memory": ([num_nodes, memory_param.dim_out], torch.float32),
        "node_memory_ts": ([num_nodes], torch.float32),
        "mails": (
            [
                num_nodes,
                memory_param.mailbox_size,
                2 * memory_param.dim_out + edge_feat_dim,
            ],
            torch.float32,
        ),
        "mail_ts": ([num_nodes, memory_param.mailbox_size], torch.float32),
        "next_mail_pos": ([num_nodes], torch.long),
        "mail_pos_update_pending_flag": ([num_nodes], torch.int8),
    }
    num_nodes = g["indptr"].shape[0] - 1
    mailbox = (
        MemoryMailbox(memory_param, shared_cfg, edge_feat_dim)
        if memory_param.type != "none"
        else None
    )
    # creterion = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=train_param.lr)
    if train_param.all_on_gpu:
        node_np = node_np.cuda() if node_np is not None else None
        edge_np = edge_np.cuda() if edge_np is not None else None
        mailbox and mailbox.move_to_gpu()

    sampler = None
    if not sample_param.no_sample:
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
    neg_link_sampler = NegLinkSampler(num_nodes)

    model.load_state_dict(torch.load(args.model))

    processed_edge_id = 0

    def forward_model_to(time):
        global processed_edge_id
        if processed_edge_id >= len(df):
            return
        while df.time[processed_edge_id] < time:
            rows = df[
                processed_edge_id : min(
                    processed_edge_id + train_param.batch_size, len(df)
                )
            ]
            if processed_edge_id < train_edge_end:
                model.train()  # why
            else:
                model.eval()
            root_nodes = np.concatenate(
                [rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]
            ).astype(np.int32)
            ts = np.concatenate(
                [rows.time.values, rows.time.values, rows.time.values]
            ).astype(np.float32)

            from utils import sampler_wrapper

            if sampler is not None:
                ret, _ = sampler_wrapper(sample_param.no_neg, root_nodes, sampler, ts)
            if gnn_param.arch != "identity":
                mfgs = to_dgl_blocks(ret, sample_param.history, save_on_pgu=True)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=True)

            mfgs, node_feats_on_gpu, edge_feats_on_gpu, node_uniq, edge_uniq = (
                prepare_mfgs(
                    args.dedup,
                    mfgs,
                    node_np,
                    edge_np,
                    deduplicate_mfg_src=bool(train_param.deduplicate_mfg_src),
                )
            )
            if mailbox is not None:
                deduplicated_mem, deduplicated_mem_input = mailbox.prepmm(
                    mfgs[0], node_uniq
                )
            with torch.no_grad():
                pred_pos, pred_neg = model(mfgs)
                if mailbox is not None:
                    eid = rows["Unnamed: 0"].values
                    mem_edge_feats = edge_np[eid] if edge_np is not None else None
                    block = None
                    if memory_param.deliver_to == "neighbors":
                        block = to_dgl_blocks(
                            ret, sample_param.history, reverse=True, save_on_pgu=True
                        )[0][0]
                    mailbox.write_mail_to_mailbox(
                        model.memory_updater.last_updated_nid,
                        model.memory_updater.last_updated_memory,
                        root_nodes,
                        ts,
                        mem_edge_feats,
                        block,
                    )
                    mailbox.write_node_memory(
                        model.memory_updater.last_updated_nid,
                        model.memory_updater.last_updated_memory,
                        model.memory_updater.last_updated_ts,
                    )
            processed_edge_id += train_param.batch_size
            if processed_edge_id >= len(df):
                return

    def get_node_emb(root_nodes, ts):
        forward_model_to(ts[-1])  # not properly

        if sampler is not None:
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
        if gnn_param.arch != "identity":
            mfgs = to_dgl_blocks(ret, sample_param.history, save_on_pgu=True)
        else:  # why aggragate?
            mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=True)
        mfgs, node_feats_on_gpu, edge_feats_on_gpu, node_uniq, edge_uniq = prepare_mfgs(
            args.dedup,
            mfgs,
            node_np,
            edge_np,
            deduplicate_mfg_src=bool(train_param.deduplicate_mfg_src),
        )
        if mailbox is not None:
            deduplicated_mem, deduplicated_mem_input = mailbox.prepmm(
                mfgs[0], node_uniq
            )
        with torch.no_grad():
            ret = model.get_emb(mfgs)
        return ret.detach().cpu()

    emb = list()
    for _, rows in tqdm(ldf.groupby(ldf.index // args.batch_size)):
        emb.append(
            get_node_emb(
                rows.node.values.astype(np.int32), rows.time.values.astype(np.float32)
            )
        )
    emb = torch.cat(emb, dim=0)
    torch.save(emb, "embs/" + emb_file_name)
    print("Saved to embs/" + emb_file_name)


def get_temporal_embeddings(ldf, emb_file_name):
    emb_path = Path("embs") / emb_file_name
    emb_path.parent.mkdir(parents=True, exist_ok=True)  # ensure 'embs/' exists

    try:
        print(f"Loading temporal embeddings from {emb_path}")
        return torch.load(emb_path)
    except FileNotFoundError:
        print("Generating temporal embeddings…")
        emb = generate_temporal_embeddings(ldf, emb_file_name)
        torch.save(emb, emb_path)
        return emb


emb_file_name = (
    hashlib.md5(
        str(torch.load(args.model, map_location=torch.device("cpu"))).encode("utf-8")
    ).hexdigest()
    + ".pt"
)
emb = get_temporal_embeddings(ldf, emb_file_name)

model = NodeClassificationModel(emb.shape[1], args.dim, labels.max() + 1).cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
labels = torch.from_numpy(labels).type(torch.int32)
role = torch.from_numpy(role).type(torch.int32)
emb = emb

from torch.utils.data import DataLoader, Dataset


class EmbDataset(Dataset):
    def __init__(self, emb, labels):
        self.emb = emb
        self.labels = labels

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            emb_batch = self.emb[idx]
            label_batch = self.labels[idx]
            return emb_batch, label_batch
        else:
            return self.emb[idx], self.labels[idx]


def make_loader(emb, labels, batch_size, shuffle=True):
    ds = EmbDataset(emb, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class NodeEmbMinibatch:
    def __init__(self, emb, label, batch_size):
        self.role = role
        self.batch_size = batch_size
        self.emb = emb
        self.label = label
        self.s_idx = 0
        self.len = emb.shape[0]

    def shuffle(self):
        perm = torch.randperm(self.train_emb.shape[0])
        self.emb = self.emb[perm]
        self.label = self.label[perm]

    def __iter__(self):
        return self

    def __next__(self):
        if self.s_idx >= self.len:
            raise StopIteration
        else:
            end = min(self.s_idx + self.batch_size, self.len)
            curr_emb = self.emb[self.s_idx : end]
            curr_label = self.label[self.s_idx : end]
            self.s_idx += self.batch_size
            return curr_emb.cuda(), curr_label.cuda()


if args.posneg:
    role = role[labels == 1]
    emb_neg = emb[labels == 0].cuda()
    emb = emb[labels == 1]
    labels = torch.ones(emb.shape[0], dtype=torch.int64).cuda()
    labels_neg = torch.zeros(emb_neg.shape[0], dtype=torch.int64).cuda()
    neg_node_sampler = NegLinkSampler(emb_neg.shape[0])

train_emb = emb[role == 0]
val_emb = emb[role == 1]
test_emb = emb[role == 2]
train_label = labels[role == 0]
val_label = labels[role == 1]
test_label = labels[role == 2]
trainLoader = NodeEmbMinibatch(train_emb, train_label, args.batch_size)
valLoader = NodeEmbMinibatch(val_emb, val_label, args.batch_size)
testLoader = NodeEmbMinibatch(test_emb, test_label, args.batch_size)

if not os.path.isdir("models"):
    os.mkdir("models")
save_path = "models/node_" + args.model.split("/")[-1]
best_e = 0
best_acc = 0
for e in range(args.epoch):
    trainLoader.shuffle()
    model.train()
    for emb, label in trainLoader:
        optimizer.zero_grad()
        if args.posneg:
            neg_idx = neg_node_sampler.sample(emb.shape[0])
            emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
            label = torch.cat([label, labels_neg[neg_idx]], dim=0)
        pred = model(emb)
        loss = loss_fn(pred, label.long())
        loss.backward()
        optimizer.step()

    valLoader.shuffle()
    model.eval()
    accs = list()
    with torch.no_grad():
        for emb, label in valLoader:
            if args.posneg:
                neg_idx = neg_node_sampler.sample(emb.shape[0])
                emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
                label = torch.cat([label, labels_neg[neg_idx]], dim=0)
            pred = model(emb)
            if args.posneg:
                acc = average_precision_score(
                    label.cpu(), pred.softmax(dim=1)[:, 1].cpu()
                )
            else:
                acc = f1_score(
                    label.cpu(), torch.argmax(pred, dim=1).cpu(), average="micro"
                )
            accs.append(acc)
        acc = float(torch.tensor(accs).mean())
    print("Epoch: {}\tVal acc: {:.4f}".format(e, acc))
    if acc > best_acc:
        best_e = e
        best_acc = acc
        torch.save(model.state_dict(), save_path)
print("Loading model at epoch {}...".format(best_e))
model.load_state_dict(torch.load(save_path))

testLoader.shuffle()
model.eval()
accs = list()
with torch.no_grad():
    for emb, label in testLoader:
        if args.posneg:
            neg_idx = neg_node_sampler.sample(emb.shape[0])
            emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
            label = torch.cat([label, labels_neg[neg_idx]], dim=0)
        pred = model(emb)
        if args.posneg:
            acc = average_precision_score(label.cpu(), pred.softmax(dim=1)[:, 1].cpu())
        else:
            acc = f1_score(
                label.cpu(), torch.argmax(pred, dim=1).cpu(), average="micro"
            )
        accs.append(acc)
    acc = float(torch.tensor(accs).mean())
print("Testing acc: {:.4f}".format(acc))
