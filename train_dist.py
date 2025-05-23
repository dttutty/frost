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



def _str_to_dtype(dtype_str):
    # "torch.float32" → "float32"
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str.split(".", 1)[1]
    return getattr(torch, dtype_str)

def init_shared(name, shape, dtype):
    """rank0 create, others get, barrier 之后返回张量句柄"""
    if local_rank == 0:
        arr = create_shared_mem_array(name, torch.Size(shape), dtype=dtype)
    dist.barrier()
    if local_rank != 0:
        arr = get_shared_mem_array(name, torch.Size(shape), dtype=dtype)
    return arr


def init_shared_array(name, tensor):
    """
    Rank 0: create + copy; others: barrier then get.
    Returns the shared tensor or None.
    """
    arr = None
    if local_rank == 0 and tensor is not None:
        meta = [tuple(tensor.shape), str(tensor.dtype)]
        arr = create_shared_mem_array(name, tensor.shape, dtype=tensor.dtype)
        arr.copy_(tensor)
    else:
        meta = [None, None]
    dist.broadcast_object_list(meta, src=0)
    shape, dtype_str = meta
    if shape is None:
        return None
    dtype = _str_to_dtype(dtype_str)
    if local_rank != 0:
        arr = get_shared_mem_array(name, shape, dtype=dtype)
    return arr

# inside main init
if local_rank == 0:
    node_np, edge_np = load_feat(args.data)
else:
    node_np = edge_np = None
# create/get shared arrays
edge_feats = init_shared_array('edge_feats', edge_np)
node_feats = init_shared_array('node_feats', node_np)


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
    arrays = {}
    for name, (shape, dtype) in shared_cfg.items():
        arrays[name] = init_shared(name, shape, dtype)
    if local_rank == 0:
        for arr in arrays.values():
            arr.zero_()
    
    mailbox = MemoryMailbox(memory_param, num_nodes, dim_feats[4], arrays['node_memory'], arrays['node_memory_ts'], arrays['mails'], arrays['mail_ts'], arrays['next_mail_pos'], arrays['update_mail_pos'])


if local_rank == world_size - 1:
    run_host(sample_param, memory_param, gnn_param, train_param, g, df, mailbox)
else:
    print("dist-worker local_rank",local_rank)
    run_worker(sample_param, memory_param, gnn_param, train_param, nccl_group, mailbox, dim_feats, path_saver, node_feats, edge_feats)
