import argparse
import os
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES').split(',')
device_count = len(cuda_visible_devices)

if LOCAL_RANK != 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices[LOCAL_RANK]
import torch.distributed as dist
import datetime
import torch

from typing import Optional
parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--seed', type=int, default=0, help='random seed to use')
parser.add_argument('--omp_num_threads', type=int, default=8)
args=parser.parse_args()
dist.init_process_group(backend='gloo', timeout=datetime.timedelta(0, 3600))

local_rank = dist.get_rank()
world_size = dist.get_world_size()
assert world_size == device_count + 1, "world size should be num_gpus + 1"
nccl_group = dist.new_group(ranks=list(range(world_size - 1)), backend='nccl')

os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)
os.environ['MKL_NUM_THREADS'] = str(args.omp_num_threads)
import time

import random
import numpy as np
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from modules import  MemoryMailbox
from utils import load_feat, load_graph, parse_config
from worker import run_worker
from host import run_host
import torch.distributed as dist
from functools import wraps
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

def init_shared_array(name, tensor_on_rank0=None):
    """
    Rank 0: create + copy; others: barrier then get.
    Returns the shared tensor or None.
    """
    arr_for_current_rank: Optional[torch.Tensor] = None 
    if local_rank == 0:
        if tensor_on_rank0 is not None:
            current_shape = tuple(tensor_on_rank0.shape)
            current_dtype = tensor_on_rank0.dtype 
            
            arr_for_current_rank = create_shared_mem_array(name, current_shape, dtype=current_dtype)
            arr_for_current_rank.copy_(tensor_on_rank0)
            meta_payload = {'shape': current_shape, 'dtype': current_dtype}
        else:
            meta_payload = {'shape': None, 'dtype': None}
    else:
        meta_payload = {}
    meta_list_to_broadcast = [meta_payload if local_rank == 0 else None]
    dist.broadcast_object_list(meta_list_to_broadcast, src=0)
    received_meta = meta_list_to_broadcast[0]
    final_shape = received_meta['shape']
    final_dtype = received_meta['dtype']
    
    if final_shape is None or final_dtype is None:
        return None

    if local_rank != 0:
        arr_for_current_rank = get_shared_mem_array(name, final_shape, dtype=final_dtype)
    
    return arr_for_current_rank

# inside main init
if local_rank == 0:
    node_np, edge_np = load_feat(args.data)
else:
    node_np = edge_np = None
# create/get shared arrays
edge_feats = init_shared_array('edge_feats', tensor_on_rank0=edge_np)
node_feats = init_shared_array('node_feats', tensor_on_rank0=node_np)

def rank_exec_and_broadcast(func=None, *, needs_barrier=False, src_rank=0):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            payload = [None]  # 用于 broadcast_object_list

            if local_rank == src_rank:
                result_on_src_rank = fn(*args, **kwargs)
                payload[0] = result_on_src_rank
            if needs_barrier:
                dist.barrier()
            dist.broadcast_object_list(payload, src=src_rank)
            return payload[0]
        return wrapper
    if func is None:
        return decorator
    else:
        return decorator(func)

def _get_numpy_array_info(arr):
    if arr is None:
        return 0, 0, None
    shape = arr.shape
    dim0 = shape[0] if len(shape) > 0 else 0
    dim1 = shape[1] if len(shape) > 1 else 0
    return dim0, dim1, arr.dtype

@rank_exec_and_broadcast
def calculate_dim_feats():
    num_nodes, node_feat_dim, node_dtype = _get_numpy_array_info(node_np)
    num_edges, edge_feat_dim, edge_dtype = _get_numpy_array_info(edge_np)
    return [
        num_nodes, node_feat_dim, node_dtype,
        num_edges, edge_feat_dim, edge_dtype,
    ]
    
dim_feats_list = calculate_dim_feats()
node_feat_dim = dim_feats_list[1]
edge_feat_dim = dim_feats_list[4]

@rank_exec_and_broadcast
def prepare_path_saver(data_name):
    if not os.path.isdir('models'):
        os.mkdir('models')
    return 'models/{}_{}.pkl'.format(data_name, time.time()) # time.time() 在 rank 0 执行
path_saver = prepare_path_saver(args.data) # args.data 对于所有进程是相同的

if local_rank == world_size - 1:
    g, df = load_graph(args.data)
@rank_exec_and_broadcast(needs_barrier=True, src_rank=world_size - 1)
def load_graph_and_get_num_nodes():
    return g['indptr'].shape[0] - 1
num_nodes = load_graph_and_get_num_nodes()

sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

def init_shared(name, shape, dtype):
    """rank0 create, others get, barrier 之后返回张量句柄"""
    if local_rank == 0:
        arr = create_shared_mem_array(name, torch.Size(shape), dtype=dtype)
    dist.barrier()
    if local_rank != 0:
        arr = get_shared_mem_array(name, torch.Size(shape), dtype=dtype)
    return arr

memory_mailbox = None
if memory_param.type != 'none':
    # name → (shape list, dtype)
    shared_cfg = {
        'node_memory':      ([num_nodes, memory_param.dim_out], torch.float32),
        'node_memory_ts':   ([num_nodes], torch.float32),
        'mails':            ([num_nodes, memory_param.mailbox_size, 2 * memory_param.dim_out + edge_feat_dim], torch.float32),
        'mail_ts':          ([num_nodes, memory_param.mailbox_size], torch.float32),
        'next_mail_pos':    ([num_nodes], torch.long),
        'update_mail_pos':  ([num_nodes], torch.int32),
    }
    shared_arrays = {}
    for name, (shape, dtype) in shared_cfg.items():
        shared_arrays[name] = init_shared(name, shape, dtype)
    if local_rank == 0:
        for arr in shared_arrays.values():
            arr.zero_()
    
    memory_mailbox = MemoryMailbox(memory_param, shared_cfg, edge_feat_dim, shared_arrays)


if local_rank == world_size - 1:
    run_host(sample_param, memory_param, gnn_param, train_param, g, df, memory_mailbox)
else:
    print("dist-worker local_rank",local_rank)
    run_worker(sample_param, memory_param, gnn_param, train_param, nccl_group, memory_mailbox, node_feat_dim, edge_feat_dim, path_saver, node_feats, edge_feats)
