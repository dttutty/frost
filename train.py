import argparse
import os
import multiprocessing as mp
import threading
import queue
from etc import get_staleness_constraint,  adaptive_split, batch_producer
from tqdm import tqdm

from utils import mfgs_to_cuda
parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=128, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=128, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
parser.add_argument('--early_stop_patience', type=int, default=5, help='patience for early stopping. 0 to disable.')
parser.add_argument('--train_neg_link_sampler', type=str, default='tgl_random', choices=['tgl_random', 'real', 'avoid_info_leak'], help='Type of negative link sampler to use for training')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ["OMP_NUM_THREADS"] = "4"

import torch
import time
import random
import numpy as np
from modules import GeneralModel
from sampler import ParallelSampler, NegLinkSampler, NegLinkInductiveSampler, create_sampler, AvoidInfoLeakNegLinkSampler
from utils import load_feat, load_graph, parse_config, to_dgl_blocks, node_to_dgl_blocks, prepare_mfgs
from sklearn.metrics import average_precision_score, roc_auc_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# set_seed(0)

node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
g, df = load_graph(args.data)
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
# ext_roll有三种 0 1 2
# 0: train, 1: val, 2: test

def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]
    
    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    train_node_set = set(np.unique(np.hstack([train_df['src'].values, train_df['dst'].values])))
    new_node_set = total_node_set - train_node_set
    
    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end+index)
    
    print('Inductive links', len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds

if args.use_inductive:
    inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
    df = df.iloc[inductive_inds]
    
# gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
# gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

from utils import get_node_edge_dimensions, get_node_count_from_graph
_, node_feat_dim, _, _, edge_feat_dim, _ = get_node_edge_dimensions(node_feats, edge_feats)
gnn_dim_node = node_feat_dim
gnn_dim_edge = edge_feat_dim
num_nodes = get_node_count_from_graph(g)

shared_cfg = {
    'node_memory':      ([num_nodes, memory_param.dim_out], torch.float32),
    'node_memory_ts':   ([num_nodes], torch.float32),
    'mails':            ([num_nodes, memory_param.mailbox_size, 2 * memory_param.dim_out + edge_feat_dim], torch.float32),
    'mail_ts':          ([num_nodes, memory_param.mailbox_size], torch.float32),
    'next_mail_pos':    ([num_nodes], torch.long),
    'mail_pos_update_pending_flag':  ([num_nodes], torch.int8),
}
    # def __init__(self, 
    #              memory_param: MemoryParams, 
    #              shared_cfg, 
    #              dim_edge_feat, 
    #              shared_arrays = None):
from memorys import MemoryMailbox
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=bool(train_param.combine_neighs)).cuda()
mailbox = MemoryMailbox(memory_param, shared_cfg, edge_feat_dim) if memory_param.type != 'none' else None
creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param.lr)
if  train_param.all_on_gpu:
    if node_feats is not None:
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()

sampler = None
if not sample_param.no_sample:
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param.num_thread, 1, sample_param.layer, sample_param.neighbor,
                              sample_param.strategy=='recent', sample_param.prop_time,
                              sample_param.history, float(sample_param.duration))

if args.use_inductive:
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.dst.values) #fixed by sqp17
    print("inductive nodes", len(inductive_nodes))
    train_neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
    eval_neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    if args.train_neg_link_sampler == 'tgl_random':
        train_neg_link_sampler = NegLinkSampler(g)
    elif args.train_neg_link_sampler == 'real':
        train_neg_link_sampler = create_sampler(g)
    elif args.train_neg_link_sampler == 'avoid_info_leak':
        train_neg_link_sampler = AvoidInfoLeakNegLinkSampler(sampler)
    # eval_neg_link_sampler = NegLinkSampler(g)
    eval_neg_link_sampler = create_sampler(g)
    

def eval(mode='val'):
    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        for _, edges in eval_df.groupby(eval_df.index // train_param.batch_size):
            root_nodes = np.concatenate([edges.src.values, edges.dst.values, eval_neg_link_sampler.sample(edges.src.values, edges.time.values)]).astype(np.int32)
            ts = np.tile(edges.time.values, neg_samples + 2).astype(np.float32)
            if sampler is not None:
                if sample_param.no_neg:
                    pos_root_end = len(edges) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param.arch != 'identity':
                mfgs_same_device_as_feat = to_dgl_blocks(ret, sample_param.history, cuda=train_param.all_on_gpu)
            else:
                mfgs_same_device_as_feat = node_to_dgl_blocks(root_nodes, ts, cuda=train_param.all_on_gpu)
            # show mfgs device
            mfgs_on_gpu = mfgs_to_cuda(mfgs_same_device_as_feat)
            _, node_feats_on_gpu, edge_feats_on_gpu = prepare_mfgs(mfgs_same_device_as_feat, mfgs_on_gpu, node_feats, edge_feats, deduplicate_mfg_src=bool(train_param.deduplicate_mfg_src))
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs_on_gpu[0])
            pred_pos, pred_neg = model(mfgs_on_gpu, node_feats_on_gpu, edge_feats_on_gpu,  neg_samples=neg_samples)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = edges['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param.deliver_to == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param.history, reverse=True, cuda=train_param.all_on_gpu)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
        if mode == 'val':
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

if not os.path.isdir('models'):
    os.mkdir('models')
if args.model_name == '':
    path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
else:
    path_saver = 'models/{}.pkl'.format(args.model_name)
best_ap = 0
best_e = 0
val_losses = list()
group_indexes = list()
group_indexes.append(np.array(df[:train_edge_end].index // train_param.batch_size))
if train_param.reorder is not None:
    # random chunk shceduling
    reorder = train_param.reorder
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), train_param.batch_size // reorder)
    group_idx = np.tile(group_idx, train_edge_end // train_param.batch_size + 1)[:train_edge_end]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, train_param.reorder):
        additional_idx = np.zeros(train_param.batch_size // train_param.reorder * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])

for e in range(train_param.epoch):
    print('Epoch {:d}:'.format(e))
    total_sampling_time = 0
    time_prep = 0
    time_tot = 0
    total_loss = 0
    # training
    model.train()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
        
    train_index = np.array(df[:train_edge_end].index)
    train_data = np.stack([df[:train_edge_end].src, df[:train_edge_end].dst]).T
    batch_plan_queue = queue.Queue(maxsize=5)
    producer_thread = threading.Thread(
        target=batch_producer,
        args=(train_data, train_param.batch_size, batch_plan_queue)
    )
    producer_thread.start()
    print("[Sampler] Ready to start sampling. Waiting for first batch...")
    with tqdm(desc="[Sampler] Processing Batches") as pbar:

        batch_num = 0
        
        while True:
            current_batch_indices = batch_plan_queue.get()
            

            if current_batch_indices is None:
                tqdm.write("[Sampler] Received completion signal. Finishing sampling.")
                break
            
            batch_num += 1
            pbar.update(1)
            pbar.set_postfix(
                batch=batch_num,
                q_size=batch_plan_queue.qsize(),
                batch_size=len(current_batch_indices)
            )
            t0 = time.time()
            edge_src_values = df.loc[current_batch_indices].src.values
            edge_dst_values = df.loc[current_batch_indices].dst.values
            neg = train_neg_link_sampler.sample(edge_src_values, edge_dst_values)
            root_nodes = np.concatenate([edge_src_values, edge_dst_values, neg]).astype(np.int32)
            tims_value = df.loc[current_batch_indices].time.values
            ts = np.concatenate([tims_value, tims_value, tims_value]).astype(np.float32)
            t_tot_s = time.time()
            # root_nodes = np.concatenate([edges.src.values, edges.dst.values, train_neg_link_sampler.sample(edges.src.values, edges.time.values)]).astype(np.int32)
            # ts = np.concatenate([edges.time.values, edges.time.values, edges.time.values]).astype(np.float32)
            if sampler is not None:
                if sample_param.no_neg:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
                total_sampling_time += ret[0].sample_time()
            t_prep_s = time.time()
            if gnn_param.arch != 'identity':
                mfgs_same_device_as_feat = to_dgl_blocks(ret, sample_param.history, cuda=train_param.all_on_gpu)
            else:
                mfgs_same_device_as_feat = node_to_dgl_blocks(root_nodes, ts, cuda=train_param.all_on_gpu)
            mfgs_on_gpu = mfgs_to_cuda(mfgs_same_device_as_feat)
            _, node_feats_on_gpu, edge_feats_on_gpu = prepare_mfgs(mfgs_same_device_as_feat, mfgs_on_gpu, node_feats, edge_feats, deduplicate_mfg_src=bool(train_param.deduplicate_mfg_src))
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs_on_gpu[0])
            time_prep += time.time() - t_prep_s
            optimizer.zero_grad()
            pred_pos, pred_neg = model(mfgs_on_gpu, node_feats_on_gpu, edge_feats_on_gpu)
            loss = creterion(pred_pos, torch.ones_like(pred_pos))
            loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * train_param.batch_size
            loss.backward()
            optimizer.step()
            t_prep_s = time.time()
            if mailbox is not None:
                eid = current_batch_indices
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param.deliver_to == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param.history, reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
            time_prep += time.time() - t_prep_s
            time_tot += time.time() - t_tot_s
        producer_thread.join()
    ap, auc = eval('val')
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
    print('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s'.format(time_tot, total_sampling_time, time_prep))
    
    # if e > 2 and ap > best_ap:
    if ap > best_ap:
        best_e = e
        best_ap = ap
        epochs_without_improvement = 0
        print(f'\tNew best validation AP: {best_ap:.4f}. Saving model to {path_saver}')
        torch.save(model.state_dict(), path_saver)
    else:
        epochs_without_improvement += 1
        print(f'\tValidation AP did not improve for {epochs_without_improvement} epoch(s).')
        # Trigger early stopping if patience is reached
    if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
        print(f'\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement.')
        print(f'The best model was from epoch {best_e} with validation AP {best_ap:.4f}.')
        break # Exit the training loop



print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(path_saver))
model.eval()
if sampler is not None:
    sampler.reset()
if mailbox is not None:
    mailbox.reset()
    model.memory_updater.last_updated_nid = None
    eval('train')
    eval('val')
ap, auc = eval('test')
if args.eval_neg_samples > 1:
    print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
else:
    print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
