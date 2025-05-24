
from dataclasses import dataclass, field
import torch
import random
import math
import numpy as np
import time
from tqdm import tqdm
from sampler_core import ParallelSampler, TemporalGraphBlock
from sampler import NegLinkSampler
from utils import  parse_config, to_dgl_blocks, node_to_dgl_blocks
import torch.distributed as dist
from modules import  MemoryMailbox
local_rank = dist.get_rank()
world_size = dist.get_world_size()
ngpus = world_size - 1
state_buf = torch.empty(1, dtype=torch.int32, device='cpu')


    
def eval(eval_df,  memory_mailbox: MemoryMailbox, sampler: ParallelSampler|None, neg_link_sampler, gnn_param, memory_param, cfg_batch_size, sample_param, pbar, local_rank):
    _, batch_size = plan_gpu_batches(len(eval_df), cfg_batch_size, ngpus)

    ap_tot = []
    auc_tot = []
    batches_assignment = batches_assign(eval_df.index, batch_size)
    n_gpus_n_lists = NgpusNlists()
    for _, rows in eval_df.groupby(batches_assignment):
        ap_tot, auc_tot = prepare_and_scatter_microbatch(rows, memory_mailbox, sampler, neg_link_sampler, gnn_param, memory_param, ngpus, sample_param,
                                   n_gpus_n_lists, local_rank, mode='eval', ap_tot=ap_tot, auc_tot=auc_tot)
        pbar.update(1)
    ap = float(torch.tensor(ap_tot).mean())
    auc = float(torch.tensor(auc_tot).mean())
    return ap, auc


@dataclass
class NgpusNlists:
    gnn_mfgs: list = field(default_factory=list)
    roots: list = field(default_factory=list)
    ts: list = field(default_factory=list)
    eid: list = field(default_factory=list)
    mail_deliver_block: list = field(default_factory=list)
    
def scatter_send(lists, src=local_rank):
    lists.append(None)
    dist.scatter_object_list([None], lists, src=src)
def sampler_wrapper(no_neg, root_nodes, time_sample, sampler, ts):
    if no_neg:
        pos_root_end = root_nodes.shape[0] * 2 // 3
        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
    else:
        sampler.sample(root_nodes, ts)
    ret = sampler.get_ret()
    if time_sample is not None:
        time_sample += ret[0].sample_time()
    return ret, time_sample

from sampler import SampleParams
def prepare_and_scatter_microbatch(rows, memory_mailbox: MemoryMailbox, sampler: ParallelSampler|None, neg_link_sampler, gnn_param, memory_param, num_gpus, sample_param : SampleParams,
                                   n_gpus_n_lists: NgpusNlists, local_rank, mode='eval', time_sample=None, ap_tot=None, auc_tot=None):
    root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
    ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
    # sampler

    ret, time_sample = sampler_wrapper(sample_param.no_neg, root_nodes, time_sample, sampler, ts)
    if gnn_param.arch != 'identity':
        # eg. if history = 2, size of ret = 6, then grouped_mfgs = [[block1,block2],[block3,block4],[block5,block6]]
        grouped_mfgs = to_dgl_blocks(ret, sample_param.history, cuda=False)
    else:
        # eg. grouped_mfgs = [[block]]
        grouped_mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
        
    n_gpus_n_lists.gnn_mfgs.append(grouped_mfgs)
    n_gpus_n_lists.roots.append(root_nodes)
    n_gpus_n_lists.ts.append(ts)
    n_gpus_n_lists.eid.append(rows['Unnamed: 0'].values)
    
    if memory_mailbox is not None and memory_param.deliver_to == 'neighbors':
        mail_deliver_block = to_dgl_blocks(ret, sample_param.history, reverse=True, cuda=False)[0][0]
        n_gpus_n_lists.mail_deliver_block.append(mail_deliver_block)
        
    if len(n_gpus_n_lists.gnn_mfgs) == num_gpus:
        broadcast_state(mode)
        scatter_send(n_gpus_n_lists.gnn_mfgs)
        if memory_mailbox is not None:
            scatter_send(n_gpus_n_lists.roots)
            scatter_send(n_gpus_n_lists.ts)
            scatter_send(n_gpus_n_lists.eid)
            if memory_param.deliver_to == 'neighbors':
                scatter_send(n_gpus_n_lists.mail_deliver_block)
        if mode =='eval':
            gathered_ap = [None] * (num_gpus + 1)
            gathered_auc = [None] * (num_gpus + 1)
            dist.gather_object(float(0), gathered_ap, dst=local_rank)
            dist.gather_object(float(0), gathered_auc, dst=local_rank)
            ap_tot += gathered_ap[:-1]
            auc_tot += gathered_auc[:-1]
        n_gpus_n_lists.gnn_mfgs.clear()
        n_gpus_n_lists.roots.clear()
        n_gpus_n_lists.ts.clear()
        n_gpus_n_lists.eid.clear()
        n_gpus_n_lists.mail_deliver_block.clear()

    if mode =='eval':
        return ap_tot, auc_tot
    else:
        return time_sample

def do_reorder(reorder, batch_size, num_samples, group_indexes):
    # random chunk shceduling
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), batch_size // reorder)
    group_idx = np.tile(group_idx, num_samples // batch_size + 1)[:num_samples]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, reorder):
        additional_idx = np.zeros(batch_size// reorder * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])
    return group_indexes

def broadcast_state(state, src=local_rank):
    map = {
        'exit': -1,
        'train': 0,
        'eval': 1,
        'save': 2,
        'load': 3,
        'pass': 4,
        'loss_reduce': 5
    }
    state_buf.fill_(map[state])
    dist.broadcast(state_buf, src=src)

def plan_gpu_batches(num_samples, target_batch_size, ngpus=1):
    # This is to ensure the total number of training iterations is an integer multiple of the number of GPUs
    # eg: num_samples = 13345, target_batch_size = 1000, ngpus = 4
    approx_iterations = num_samples // target_batch_size # approximate number of iterations, 13
    nbatch_per_gpu = max(approx_iterations // ngpus, 1) # The approximate number of batches that each GPU needs to process, 3
    iteration_total = nbatch_per_gpu * ngpus # total batches, 12 
    adjusted_batch_size  = math.ceil(num_samples / iteration_total) # 1113
    return iteration_total, adjusted_batch_size

def batches_assign(index, batch_size):
    return index // batch_size
    
    
def run_epoch(e, sampler: ParallelSampler|None, memory_mailbox: MemoryMailbox|None, df, train_edge_end, val_edge_end, gnn_param, memory_param, cfg_batch_size, sample_param, train_param, neg_link_sampler, best_ap, best_e, tap, tauc, local_rank): 

    train_df = df[:train_edge_end]
    val_df = df[train_edge_end:val_edge_end]
    test_df = df[val_edge_end:]
    print('Epoch {:d}:'.format(e))
    time_sample = 0
    time_tot = 0
    if sampler is not None:
        sampler.reset()
    if memory_mailbox is not None:
        memory_mailbox.reset()
    # training
    train_param.batch_size = cfg_batch_size
    train_iteration_total, train_param.batch_size = plan_gpu_batches(len(train_df), cfg_batch_size, ngpus=ngpus)
    val_iteration_total, _ = plan_gpu_batches(len(val_df), train_param.batch_size, ngpus=ngpus) # don't care about val batch size

    n_gpus_n_lists = NgpusNlists()
    group_indexes = []
    batches_assignment = batches_assign(train_df.index, train_param.batch_size)
    group_indexes.append(batches_assignment)
    
    if train_param.reorder is not None:
        # random chunk shceduling
        group_indexes = do_reorder(train_param.reorder, train_param.batch_size, len(train_df), group_indexes)
            
    with tqdm(total=train_iteration_total + val_iteration_total) as pbar:
        # train_df
        for _, rows in train_df.groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
            t_tot_s = time.time()
            time_sample = prepare_and_scatter_microbatch(rows, memory_mailbox, sampler, neg_link_sampler, gnn_param, memory_param, ngpus, sample_param,
                                   n_gpus_n_lists,  local_rank, mode='train', time_sample=time_sample)
            time_tot += time.time() - t_tot_s
            pbar.update(1)
        print('Training time:',time_tot)
        broadcast_state('loss_reduce')
        gathered_loss = [None] * world_size
        dist.gather_object(float(0), gathered_loss, dst=local_rank)
        total_loss = np.sum(np.array(gathered_loss) * train_param.batch_size)
        
        # val_df
        ap, auc  = eval(val_df,   memory_mailbox,  sampler, neg_link_sampler, gnn_param, memory_param, cfg_batch_size, sample_param, pbar, local_rank)
        if ap > best_ap:
            best_e = e
            best_ap = ap
            
            # test_df, for memory based models, testing after validation is faster
            broadcast_state('save')
            tap, tauc = eval(test_df,  memory_mailbox,  sampler, neg_link_sampler, gnn_param, memory_param, cfg_batch_size, sample_param, pbar,local_rank)
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
    print('\ttotal time:{:.2f}s sample time:{:.2f}s'.format(time_tot, time_sample))
    return best_ap, best_e, tap, tauc

# host.py
def run_host(sample_param, memory_param, gnn_param, train_param, g, df, memory_mailbox):
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
        best_ap, best_e, tap, tauc = run_epoch(e, sampler, memory_mailbox, df, train_edge_end, val_edge_end, gnn_param, memory_param,
                                      cfg_batch_size, sample_param, train_param, neg_link_sampler,
                                      best_ap, best_e, tap, tauc, local_rank)

    print('Best model at epoch {}.'.format(best_e))
    print('\ttest ap:{:4f}  test auc:{:4f}'.format(tap, tauc))

    # let all process exit
    broadcast_state('exit')
