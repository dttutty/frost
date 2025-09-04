
from dataclasses import dataclass, field
import torch
import random
import math
import numpy as np
import time
from tqdm import tqdm
from sampler_core import ParallelSampler, TemporalGraphBlock
from sampler import NegLinkSampler, SampleParams, NegLinkInductiveSampler
from utils import  to_dgl_blocks, node_to_dgl_blocks
import torch.distributed as dist
from memorys import  MemoryMailbox
local_rank = dist.get_rank()
world_size = dist.get_world_size()
ngpus = world_size - 1
state_buf = torch.empty(1, dtype=torch.int32, device='cpu')

from worker import TransferData

from contextlib import contextmanager


    
def eval(eval_df,  memory_mailbox: MemoryMailbox, sampler: ParallelSampler|None, neg_link_sampler, gnn_param, memory_param, cfg_batch_size, sample_param, pbar, local_rank):
    _, batch_size = plan_gpu_batches(len(eval_df), cfg_batch_size, ngpus)

    ap_tot = []
    auc_tot = []
    batches_assignment = batches_assign(eval_df.index, batch_size)
    
    batch_to_send = []
    for _, rows in eval_df.groupby(batches_assignment):
        ap_tot, auc_tot = sample_and_scatter_batch(rows, memory_mailbox, sampler, neg_link_sampler, gnn_param, memory_param, ngpus, sample_param,
                                   batch_to_send, local_rank, mode='eval', ap_tot=ap_tot, auc_tot=auc_tot)
        pbar.update(1)
    ap = float(torch.tensor(ap_tot).mean())
    auc = float(torch.tensor(auc_tot).mean())
    return ap, auc

def scatter_send(lists, src=local_rank):
    lists.append(None)
    dist.scatter_object_list([None], lists, src=src)
    
from utils import sampler_wrapper, dgl_blocks_to_tensor_dicts, tensor_dicts_to_dgl_blocks

def sample_and_dispatch_batch(edges, memory_mailbox: MemoryMailbox, sampler: ParallelSampler|None, neg_link_sampler, gnn_param, memory_param, num_gpus, sample_param : SampleParams,
                                   batch_to_send, local_rank, mode='eval', total_sample_time=None, ap_tot=None, auc_tot=None):
    """Samples data and dispatches it to workers as tensor dictionaries."""
    root_nodes = np.concatenate([edges.src.values, edges.dst.values, neg_link_sampler.sample(edges.src.values, edges.time.values)]).astype(np.int32)
    ts = np.concatenate([edges.time.values, edges.time.values, edges.time.values]).astype(np.float32)
    if sampler is not None:
        ret, _ = sampler_wrapper(sample_param.no_neg, root_nodes, sampler, ts, total_sample_time)
    
    if gnn_param.arch != 'identity':
        grouped_mfgs = to_dgl_blocks(ret, sample_param.history, cuda=False)
    else:
        grouped_mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)

    # Deconstruct DGL blocks into tensor representations
    # This is the KEY change for performance
    mfgs_tensors = [dgl_blocks_to_tensor_dicts(mfg_list) for mfg_list in grouped_mfgs]

    # --- 【新增】处理 memory_mailbox 需要的特殊 block ---
    special_block_tensors = None
    if memory_mailbox is not None and memory_param.deliver_to == 'neighbors':
        # 创建这个特殊的、反转的 block
        special_block = to_dgl_blocks(ret, sample_param.history, reverse=True, cuda=False)[0][0]
        # 使用同样的工具函数将其转换为张量字典
        # 注意用列表包裹和解包，因为工具函数接收的是列表
        special_block_tensors = dgl_blocks_to_tensor_dicts([special_block])[0]
    
    
    payload = {
        'mode': mode, # <--- NEW: 告诉 worker 当前的模式
        'mfgs_tensors': mfgs_tensors,
        'special_block_tensors': special_block_tensors, # <--- NEW: 加入特殊 block 的张量字典
        'root': torch.from_numpy(root_nodes),
        'ts': torch.from_numpy(ts),
        'eid': torch.from_numpy(edges['Unnamed: 0'].values)
    }
    batch_to_send.append(payload)
    if len(batch_to_send) == num_gpus:
        broadcast_state(mode)
        for i, payload in enumerate(batch_to_send):
            dist.send_object(payload, dst=i)
        if mode == 'eval':
            ap_tot += gather_receive(world_size, dst=local_rank)
            auc_tot += gather_receive(world_size, dst=local_rank)
        batch_to_send.clear()

    if mode =='eval':
        return ap_tot, auc_tot
    else:
        return total_sample_time
    
import pickle
def sample_and_scatter_batch(edges, memory_mailbox: MemoryMailbox, sampler: ParallelSampler|None, neg_link_sampler, gnn_param, memory_param, num_gpus, sample_param : SampleParams,
                                   batch_to_send, local_rank, mode='eval', total_sample_time=None, ap_tot=None, auc_tot=None):
    root_nodes = np.concatenate([edges.src.values, edges.dst.values, neg_link_sampler.sample(edges.src.values, edges.time.values)]).astype(np.int32)
    ts = np.concatenate([edges.time.values, edges.time.values, edges.time.values]).astype(np.float32)
    
    # example
    # sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
    #                         sample_param.num_thread, 1, sample_param.layer, sample_param.neighbor,
    #                         sample_param.strategy=='recent', sample_param.prop_time,
    #                         sample_param.history, float(sample_param.duration))

    # print('Sample nodes:', root_nodes.shape[0], 'edges:', edges.shape[0], 'ts:', ts.shape[0])
    if sampler is not None:
        ret, total_sample_time = sampler_wrapper(sample_param.no_neg, root_nodes,  sampler, ts, total_sample_time)
    # print('Sampled blocks:', len(ret), 'total sample time:', total_sample_time)
    if gnn_param.arch != 'identity':
        # eg. if sample_param.history = 3, sample_param.layer = 2, ret = [block1,block2, block3,block4, block5,block6]
        # then grouped_mfgs = [[block1,block2],[block3,block4],[block5,block6]]
        grouped_mfgs = to_dgl_blocks(ret, sample_param.history, cuda=False)
    else:
        # eg. grouped_mfgs = [[block]]
        grouped_mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
    
    data_to_send = TransferData(
        mfgs = grouped_mfgs,
        root = root_nodes,
        ts = ts,
        eid = edges['Unnamed: 0'].values,
        block = None
    )
    if memory_mailbox is not None and memory_param.deliver_to == 'neighbors':
        # only deliver to neighbors in history[1]
        data_to_send.block = to_dgl_blocks(ret, sample_param.history, reverse=True, cuda=False)[0][0]
    batch_to_send.append(data_to_send)


    
    if len(batch_to_send) == num_gpus:
        broadcast_state(mode)
        scatter_send(batch_to_send)
        # # scatter_send(batch_to_send)
        # # =================================================================
        # # === 阶段 1A: 发起【所有】的数据发送 ===
        # # =================================================================
        # print("Host: Phase 1A - Initiating all data sends...")
        # send_requests = []
        # for rank, data_payload in enumerate(batch_to_send):
        #     pickled_payload = pickle.dumps(data_payload)
        #     tensor = torch.ByteTensor(list(pickled_payload))
        #     size_tensor = torch.LongTensor([len(tensor)])
            
        #     dist.isend(tensor=size_tensor, dst=rank)
        #     req = dist.isend(tensor=tensor, dst=rank)
        #     send_requests.append(req)

        # # =================================================================
        # # === 阶段 1B: 等待【所有】的数据发送完成 ===
        # # !!! 必须在发起任何接收之前，先等待发送完成 !!!
        # # 这是确保 Host 自身任务清理干净的关键同步点。
        # # =================================================================
        # print("Host: Phase 1B - Waiting for data sends to complete...")
        # for req in send_requests:
        #     req.wait()
        # print("Host: All data has been sent successfully.")


        # # =================================================================
        # # === 阶段 2A: 发起【所有】的 ACK 接收 ===
        # # =================================================================
        # print("Host: Phase 2A - Initiating non-blocking receives for all ACKs...")
        # recv_requests = []
        # ack_buffers = []
        # for rank in range(len(batch_to_send)):
        #     ack_tensor = torch.tensor([0], dtype=torch.int32, device='cpu')
        #     req = dist.irecv(tensor=ack_tensor, src=rank)
        #     recv_requests.append(req)
        #     ack_buffers.append(ack_tensor)

        # # =================================================================
        # # === 阶段 2B: 等待【所有】的 ACK 接收完成 ===
        # # =================================================================
        # print("Host: Phase 2B - Waiting for all ACKs to be received...")
        # for i, req in enumerate(recv_requests):
        #     req.wait()

        # print("Host: All workers have acknowledged. Proceeding to next step.")

        if mode =='eval':
            ap_tot += gather_receive(world_size, dst=local_rank)
            auc_tot += gather_receive(world_size, dst=local_rank)
        batch_to_send.clear()

    if mode =='eval':
        return ap_tot, auc_tot
    else:
        return total_sample_time

def gather_receive(world_size, dst=local_rank):
    gathered = [None] * world_size
    dist.gather_object(float(0), gathered, dst=dst)
    return gathered[:-1]  

def generate_reordered_indexes_1(reorder, batch_size, num_samples, batches_assignment_0):
    # train_param.batch_size = 8, reorder = 3
    # batches_assignment_0 = [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3]
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    # group_idx = [0, 1, 2, -1, 0, 1, -2, -1, 0]
    chunk_size = batch_size // reorder
    # chunk_size = 8 // 3 = 2
    group_idx = np.repeat(np.array(group_idx), chunk_size)
    # group_idx = [ 0, 0, 1, 1, 2, 2 -1 -1, 0, 0, 1, 1 -2 -2 -1 -1, 0, 0]
    # len(group_idx) = 18
    tiles = num_samples // batch_size + 1
    # tiles = 30 // 8 + 1 = 4
    group_idx = np.tile(group_idx, tiles)[:num_samples]
    # after tile: group_idx = [ 
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0, 
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0]
    # totally 18 * 4 = 72 
    # after slice: group_idx = [    
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0, 
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1]
    # totally 30
    return_val = batches_assignment_0 + group_idx
    # return_val = [
    # 0, 0, 1, 1, 2, 2, -1, -1, 1, 1, 2, 2, -1, -1,  0,  0, 2, 2,
    # 2, 2, 3, 3, 4, 4,  2,  2, 3, 3, 4, 4]
    # totally 6 kinds of indexes -1, 0, 1, 2, 3, 4
    return return_val
    
def generate_reordered_indexes_2(reorder, batch_size, batches_assignment_0):
    # random chunk shceduling
    group_indexes = []
    chunk_size = batch_size // reorder
    # chunk_size = 8 // 3 = 2
    for i in range(1, reorder):
        additional_idx = np.zeros(chunk_size * i) - 1
        group_indexes.append(np.concatenate([additional_idx, batches_assignment_0])[:batches_assignment_0.shape[0]])
    # [-1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  2.  2.  2.  2.  2.  2.  2.  2.  3.  3.  3.  3.]
    # totally 5 kinds of indexes -1, 0, 1, 2, 3
    # [-1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  2.  2.  2.  2.  2.  2.  2.  2.  3.  3.]
    # totally 5 kinds of indexes -1, 0, 1, 2, 3
    return group_indexes

def batches_assign(index, batch_size):
    return index // batch_size
    
    
def broadcast_state(state, src=local_rank):
    map = {
        'exit': -1,
        'train': 0,
        'eval': 1,
        'save': 2,
        'load': 3,
        'pass': 4,
        'loss_reduce': 5,
        'sync': 6,  # <-- add this
        'cleanup': 7,  # <-- add this
        'abort': 8,
        
        'abort_ok': 9,   # 新增：无错退出
    }
    state_buf.fill_(map[state])
    dist.broadcast(state_buf, src=src)

def plan_gpu_batches(num_samples, target_batch_size, ngpus=1):
    target_batch_size = int(target_batch_size)          # <--- add
    # This is to ensure the total number of training iterations is an integer multiple of the number of GPUs
    # eg: num_samples = 13345, target_batch_size = 1000, ngpus = 4
    approx_iterations = num_samples // target_batch_size # approximate number of iterations, 13
    nbatch_per_gpu = max(approx_iterations // ngpus, 1) # The approximate number of batches that each GPU needs to process, 3
    iteration_total = nbatch_per_gpu * ngpus # total batches, 12 
    adjusted_batch_size  = math.ceil(num_samples / iteration_total) # 1113
    return iteration_total, adjusted_batch_size


    
def run_epoch(e, sampler: ParallelSampler|None, memory_mailbox: MemoryMailbox|None, df, train_edge_end, val_edge_end, gnn_param, memory_param, cfg_batch_size, sample_param, train_param, train_neg_link_sampler, neg_link_sampler_for_eval, best_ap, best_e, tap, tauc, local_rank): 

    train_df = df[:train_edge_end]
    val_df = df[train_edge_end:val_edge_end]
    test_df = df[val_edge_end:]
    now_ts = time.time()
    local_time = time.localtime(now_ts)
    ms = int((now_ts - int(now_ts)) * 1000)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time) + f".{ms:03d}"
    print(f"{time_str} Epoch {e}:")
    total_sample_time = 0
    time_tot = 0
    if sampler is not None:
        sampler.reset()
    if memory_mailbox is not None:
        memory_mailbox.reset()
    # training
    train_param.batch_size = cfg_batch_size
    train_iteration_total, train_param.batch_size = plan_gpu_batches(len(train_df), cfg_batch_size, ngpus=ngpus)
    val_iteration_total, _ = plan_gpu_batches(len(val_df), train_param.batch_size, ngpus=ngpus) # don't care about val batch size

    batch_to_send = []
    batches_assignment_0 = batches_assign(train_df.index, train_param.batch_size)
    # example:
    # train_df.index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
    # train_param.batch_size = 8, reorder = 3
    # batches_assignment_0 = [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3]
    group_indexes  = [batches_assignment_0]
    
    if train_param.reorder is not None:
        # random chunk shceduling
        group_indexes.append(generate_reordered_indexes_1(train_param.reorder, train_param.batch_size, len(train_df), batches_assignment_0))
        group_indexes.extend(generate_reordered_indexes_2(train_param.reorder, train_param.batch_size, batches_assignment_0))
            
    with tqdm(total=train_iteration_total + val_iteration_total) as pbar:
        # train_df
        batches_assignment = group_indexes[random.randint(0, len(group_indexes) - 1)]
        batches_assignment = batches_assignment_0
        for _, edges in train_df.groupby(batches_assignment):
            t_tot_s = time.time()
            total_sample_time = sample_and_scatter_batch(edges, memory_mailbox, sampler, train_neg_link_sampler, gnn_param, memory_param, ngpus, sample_param,
                                   batch_to_send, local_rank, mode='train', total_sample_time=total_sample_time)
            time_tot += time.time() - t_tot_s
            pbar.update(1)
        print('Training time:',time_tot)
        broadcast_state('loss_reduce')
        gathered_loss = gather_receive(world_size, dst=local_rank)
        total_loss = np.sum(np.array(gathered_loss) * train_param.batch_size)
        
        # val_df
        ap, auc  = eval(val_df,   memory_mailbox,  sampler, neg_link_sampler_for_eval, gnn_param, memory_param, cfg_batch_size, sample_param, pbar, local_rank)
        if ap > best_ap:
            best_e = e
            best_ap = ap
            # test_df, for memory based models, testing after validation is faster
            broadcast_state('save')
            tap, tauc = eval(test_df,  memory_mailbox,  sampler, neg_link_sampler_for_eval, gnn_param, memory_param, cfg_batch_size, sample_param, pbar,local_rank)
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
    print('\ttotal time:{:.2f}s sample time:{:.2f}s'.format(time_tot, total_sample_time))
    return best_ap, best_e, tap, tauc

# host.py
def run_host(deduplication, sample_param, memory_param, gnn_param, train_param, g, df, memory_mailbox):
    cfg_batch_size = train_param.batch_size
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
    sampler = None
    if not (sample_param.no_sample):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                  sample_param.num_thread, 1, sample_param.layer, sample_param.neighbor,
                                  sample_param.strategy=='recent', sample_param.prop_time,
                                  sample_param.history, float(sample_param.duration))
    from sampler import create_sampler
    eval_neg_link_sampler = create_sampler(g)
    train_neg_link_sampler = NegLinkSampler(g)


    best_ap = best_e = tap = tauc = 0
    
    for e in range(train_param.epoch):
        best_ap, best_e, tap, tauc = run_epoch(e, sampler, memory_mailbox, df, train_edge_end, val_edge_end, gnn_param, memory_param,
                                      cfg_batch_size, sample_param, train_param, train_neg_link_sampler, eval_neg_link_sampler,
                                      best_ap, best_e, tap, tauc, local_rank)

    
    now_ts = time.time()
    local_time = time.localtime(now_ts)
    ms = int((now_ts - int(now_ts)) * 1000)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time) + f".{ms:03d}"
    print('{} Best model at epoch {}.'.format(time_str, best_e))
    print('\ttest ap:{:4f}  test auc:{:4f}'.format(tap, tauc))
    broadcast_state('abort')

    # broadcast_state('cleanup')   # map['cleanup'] = 7

    # # let all process exit
    # broadcast_state('pass')
    # scatter_send([None] * ngpus)     # all ranks will call scatter
    
    # broadcast_state('sync')  
    # dist.barrier() 
    # # Now safely exit
    # broadcast_state('exit')
    # dist.barrier() 
    
    # try:
    #     if sampler is not None and hasattr(sampler, "close"):
    #         sampler.close()
    #     if memory_mailbox is not None and hasattr(memory_mailbox, "finalize"):
    #         memory_mailbox.finalize()
    # except Exception:
    #     pass
    
    # dist.destroy_process_group()

