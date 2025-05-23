import argparse
import os
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
device_count = len(cuda_visible_devices.split(','))

if LOCAL_RANK != 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(LOCAL_RANK)
import torch.distributed as dist
import datetime
import torch

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)









from typing import Optional
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


# collect dim info once on rank0, broadcast to all
if local_rank == 0:
    dim_feats = [
        node_np.shape[0] if node_np is not None else 0,
        node_np.shape[1] if node_np is not None else 0,
        node_np.dtype if node_np is not None else None,
        edge_np.shape[0] if edge_np is not None else 0,
        edge_np.shape[1] if edge_np is not None else 0,
        edge_np.dtype if edge_np is not None else None,
    ]
else:
    dim_feats = [None] * 6
dist.broadcast_object_list(dim_feats, src=0)


sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

if local_rank == 0:
    if not os.path.isdir('models'):
        os.mkdir('models')
    path_saver = ['models/{}_{}.pkl'.format(args.data, time.time())]
else:
    path_saver = [None]
dist.broadcast_object_list(path_saver, src=0)
path_saver = path_saver[0]

if local_rank == world_size - 1:
    g, df = load_graph(args.data)
    num_nodes = [g['indptr'].shape[0] - 1]
else:
    num_nodes = [None]
dist.barrier()
dist.broadcast_object_list(num_nodes, src=world_size - 1)
num_nodes = num_nodes[0]

def init_shared(name, shape, dtype):
    """rank0 create, others get, barrier 之后返回张量句柄"""
    if local_rank == 0:
        arr = create_shared_mem_array(name, torch.Size(shape), dtype=dtype)
    dist.barrier()
    if local_rank != 0:
        arr = get_shared_mem_array(name, torch.Size(shape), dtype=dtype)
    return arr

mailbox = None
if memory_param.type != 'none':
    # name → (shape list, dtype)
    shared_cfg = {
        'node_memory':      ([num_nodes, memory_param.dim_out], torch.float32),
        'node_memory_ts':   ([num_nodes], torch.float32),
        'mails':            ([num_nodes, memory_param.mailbox_size, 2 * memory_param.dim_out + dim_feats[4]], torch.float32),
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
    
    mailbox = MemoryMailbox(memory_param, shared_cfg, dim_feats[4], shared_arrays)


if local_rank == world_size - 1:
    run_host(sample_param, memory_param, gnn_param, train_param, g, df, mailbox)
else:
    print("dist-worker local_rank",local_rank)
    run_worker(sample_param, memory_param, gnn_param, train_param, nccl_group, mailbox, dim_feats, path_saver, node_feats, edge_feats)
