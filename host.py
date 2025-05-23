
import torch
import random
import math
import numpy as np
import time
from tqdm import tqdm
from sampler import NegLinkSampler, ParallelSampler
from utils import  parse_config, to_dgl_blocks, node_to_dgl_blocks
import torch.distributed as dist
from modules import  MemoryMailbox
local_rank = dist.get_rank()
world_size = dist.get_world_size()
ngpus = world_size - 1
state_buf = torch.empty(1, dtype=torch.int32, device='cpu')

def eval(mode, df, train_edge_end, val_edge_end, mailbox: MemoryMailbox, batch_size, sampler: ParallelSampler|None, neg_link_sampler, gnn_param, memory_param, cfg_batch_size, sample_param, pbar, local_rank):
    match mode:
        case 'train':
            eval_df = df[:train_edge_end]
        case 'val':
            eval_df = df[train_edge_end:val_edge_end]
        case 'test':
            eval_df = df[val_edge_end:]
    
    iteration_total = max(len(eval_df) // cfg_batch_size // ngpus, 1) * ngpus # total iterations
    batch_size = math.ceil(len(eval_df) / iteration_total)
    
    
    names = ['mfgs', 'root', 'ts', 'eid', 'block']
    multi_lists_mfgs, multi_lists_root, multi_lists_ts, multi_lists_eid, multi_lists_block = [[] for _ in range(5)]
    ap_tot = []
    auc_tot = []

    for _, rows in eval_df.groupby(eval_df.index // batch_size):
        ap_tot, auc_tot = prepare_and_scatter_microbatch(rows, mailbox, sampler, neg_link_sampler, gnn_param, memory_param, ngpus, sample_param,
                                   multi_lists_mfgs, multi_lists_root, multi_lists_ts, multi_lists_eid, multi_lists_block, local_rank, mode='eval', ap_tot=ap_tot, auc_tot=auc_tot)
        pbar.update(1)
    ap = float(torch.tensor(ap_tot).mean())
    auc = float(torch.tensor(auc_tot).mean())
    return ap, auc, batch_size


def scatter_send(lists, rank):
    lists.append(None)
    dist.scatter_object_list([None], lists, src=rank)

def prepare_and_scatter_microbatch(rows, mailbox: MemoryMailbox, sampler: ParallelSampler|None, neg_link_sampler, gnn_param, memory_param, num_gpus, sample_param,
                                   multi_lists_mfgs, multi_lists_root, multi_lists_ts, multi_lists_eid, multi_lists_block, local_rank, mode='eval', time_sample=None, ap_tot=None, auc_tot=None):
    root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
    ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
    # sampler
    if sampler is not None:
        if sample_param.no_neg:
            pos_root_end = root_nodes.shape[0] * 2 // 3
            sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
        else:
            sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
        if time_sample is not None:
            time_sample += ret[0].sample_time()
            
    if gnn_param.arch != 'identity':
        mfgs = to_dgl_blocks(ret, sample_param.history, cuda=False)
    else:
        mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
        
    multi_lists_mfgs.append(mfgs)
    multi_lists_root.append(root_nodes)
    multi_lists_ts.append(ts)
    multi_lists_eid.append(rows['Unnamed: 0'].values)
    
    if mailbox is not None and memory_param.deliver_to == 'neighbors':
        multi_lists_block.append(to_dgl_blocks(ret, sample_param.history, reverse=True, cuda=False)[0][0])
        
    if len(multi_lists_mfgs) == num_gpus:
        state_buf.fill_(1 if mode == 'eval' else 0)
        dist.broadcast(state_buf, src=local_rank)
        scatter_send(multi_lists_mfgs, local_rank)
        if mailbox is not None:
            scatter_send(multi_lists_root, local_rank)
            scatter_send(multi_lists_ts, local_rank)
            scatter_send(multi_lists_eid, local_rank)
            if memory_param.deliver_to == 'neighbors':
                scatter_send(multi_lists_block, local_rank)
        if mode =='eval':
            gathered_ap = [None] * (num_gpus + 1)
            gathered_auc = [None] * (num_gpus + 1)
            dist.gather_object(float(0), gathered_ap, dst=local_rank)
            dist.gather_object(float(0), gathered_auc, dst=local_rank)
            ap_tot += gathered_ap[:-1]
            auc_tot += gathered_auc[:-1]
        multi_lists_mfgs, multi_lists_root, multi_lists_ts, multi_lists_eid, multi_lists_block = [[] for _ in range(5)]

    if mode =='eval':
        return ap_tot, auc_tot
    else:
        return time_sample
            
def do_reorder(reorder, batch_size, train_edge_end, group_indexes):
    # random chunk shceduling
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), batch_size // reorder)
    group_idx = np.tile(group_idx, train_edge_end // batch_size + 1)[:train_edge_end]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, reorder):
        additional_idx = np.zeros(batch_size// reorder * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])
    return group_indexes
            
def run_epoch(e, sampler: ParallelSampler|None, mailbox: MemoryMailbox|None, df, train_edge_end, val_edge_end, gnn_param, memory_param, cfg_batch_size, sample_param, train_param, neg_link_sampler, best_ap, best_e, tap, tauc, local_rank):        
    print('Epoch {:d}:'.format(e))
    time_sample = 0
    time_tot = 0
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
    # training
    train_param.batch_size = cfg_batch_size
    iteration_total = train_edge_end // train_param.batch_size // ngpus * ngpus
    # eg: train_edge_end = 12345, cfg_batch_size = 1000, ngpus = 4
    # need 12 iterations
    # train_param.batch_size = 1029
    train_param.batch_size = math.ceil(train_edge_end / iteration_total)
    names = ['mfgs', 'root', 'ts', 'eid', 'block']
    multi_lists_mfgs, multi_lists_root, multi_lists_ts, multi_lists_eid, multi_lists_block = [[] for _ in range(5)]
    group_indexes = []
    
    group_indexes.append(np.array(df[:train_edge_end].index // train_param.batch_size))
    
    if train_param.reorder is not None:
        # random chunk shceduling
        group_indexes = do_reorder(train_param.reorder, train_param.batch_size, train_edge_end, group_indexes)
            
    with tqdm(total=iteration_total + max((val_edge_end - train_edge_end) // train_param.batch_size // ngpus, 1) * ngpus) as pbar:
        for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
            t_tot_s = time.time()

            time_sample = prepare_and_scatter_microbatch(rows, mailbox, sampler, neg_link_sampler, gnn_param, memory_param, ngpus, sample_param,
                                   multi_lists_mfgs, multi_lists_root, multi_lists_ts, multi_lists_eid, multi_lists_block,  local_rank, mode='train', time_sample=time_sample)
            time_tot += time.time() - t_tot_s
            pbar.update(1)

        print('Training time:',time_tot)
        state_buf.fill_(5)
        dist.broadcast(state_buf, src=local_rank)
        
        gathered_loss = [None] * world_size
        dist.gather_object(float(0), gathered_loss, dst=ngpus)
        total_loss = np.sum(np.array(gathered_loss) * train_param.batch_size)
        ap, auc, train_param.batch_size = eval('val', df, train_edge_end, val_edge_end, mailbox, train_param.batch_size, sampler, neg_link_sampler, gnn_param, memory_param, cfg_batch_size, sample_param, pbar, local_rank)
        if ap > best_ap:
            best_e = e
            best_ap = ap
            
            state_buf.fill_(2)
            dist.broadcast(state_buf, src=local_rank)
            # for memory based models, testing after validation is faster
            tap, tauc, train_param.batch_size = eval('test', df, train_edge_end, val_edge_end, mailbox, train_param.batch_size, sampler, neg_link_sampler, gnn_param, memory_param, cfg_batch_size, sample_param, pbar,local_rank)
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
    print('\ttotal time:{:.2f}s sample time:{:.2f}s'.format(time_tot, time_sample))
    return best_ap, best_e, tap, tauc

# host.py
def run_host(sample_param, memory_param, gnn_param, train_param, g, df, mailbox):
    cfg_batch_size = train_param.batch_size
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
    sampler = None
    if not (sample_param.no_sample):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                  sample_param.num_thread, 1, sample_param.layer, sample_param.neighbor,
                                  sample_param.strategy=='recent', sample_param.prop_time,
                                  sample_param.history, float(sample_param.duration))
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)


    best_ap = best_e = tap = tauc = 0
    for e in range(train_param.epoch):
        best_ap, best_e, tap, tauc = run_epoch(e, sampler, mailbox, df, train_edge_end, val_edge_end, gnn_param, memory_param,
                                      cfg_batch_size, sample_param, train_param, neg_link_sampler,
                                      best_ap, best_e, tap, tauc, local_rank)

    print('Best model at epoch {}.'.format(best_e))
    print('\ttest ap:{:4f}  test auc:{:4f}'.format(tap, tauc))

    # let all process exit
    state_buf.fill_(-1)
    dist.broadcast(state_buf, src=local_rank)
