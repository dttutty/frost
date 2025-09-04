import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
from profiler import Timer, profiler


# In utils.py
import dgl
import torch

def dgl_blocks_to_tensor_dicts(blocks):
    """Converts a list of DGL blocks to a list of serializable tensor dictionaries."""
    tensor_dicts = []
    for block in blocks:
        # Get the graph structure in CSC format, which is efficient
        csc_mat = block.adj_tensors('csc')
        tensor_dicts.append({
            'num_src_nodes': block.number_of_src_nodes(),
            'num_dst_nodes': block.number_of_dst_nodes(),
            'csc_indptr': csc_mat[0],
            'csc_indices': csc_mat[1],
            'edge_ids': csc_mat[2]
        })
    return tensor_dicts

def tensor_dicts_to_dgl_blocks(tensor_dicts, device='cpu'):
    """Reconstructs a list of DGL blocks from a list of tensor dictionaries."""
    blocks = []
    for d in tensor_dicts:
        # Create a graph from the CSC representation
        g = dgl.create_block(
            (d['csc_indptr'], d['csc_indices']),
            num_src_nodes=d['num_src_nodes'],
            num_dst_nodes=d['num_dst_nodes'],
            device=device
        )
        # The edge IDs are stored in the 'eid' feature of the new block
        g.edata[dgl.EID] = d['edge_ids']
        blocks.append(g)
    return blocks


def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if 'ID' in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata['ID'].long())
    else:
        eids = None
    return nids, eids


def load_feat(d, rand_de=128, rand_dn=128):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC': # a bug fixed by zhaolei @sqp17
            node_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

from sampler import SampleParams
from memorys import MemoryParams
from modules import GNNParams, TrainParams
def parse_config(f, fashion='new'):
    conf = yaml.safe_load(open(f, 'r'))
    if fashion == 'new':
        sample_param = SampleParams(**conf['sampling'][0])
        memory_param = MemoryParams(**conf['memory'][0])
        gnn_param = GNNParams(**conf['gnn'][0])
        train_param = TrainParams(**conf['train'][0])
        print(f"Parsed config: {sample_param}, {memory_param}, {gnn_param}, {train_param}")
    else:
        sample_param = conf['sampling'][0]
        memory_param = conf['memory'][0]
        gnn_param = conf['gnn'][0]
        train_param = conf['train'][0]
    
    # print(f"Parsed config: {sample_param}, {memory_param}, {gnn_param}, {train_param}")
    return sample_param, memory_param, gnn_param, train_param


def get_shape_and_dtype(meta_payload):
    final_shape, final_dtype = meta_payload['shape'], meta_payload['dtype']
    return final_shape, final_dtype


def get_node_edge_dimensions(node_np, edge_np):
    def _get_numpy_array_info(arr):
        if arr is None:
            return 0, 0, None
        shape = arr.shape
        dim0 = shape[0] if len(shape) > 0 else 0
        dim1 = shape[1] if len(shape) > 1 else 0
        return dim0, dim1, arr.dtype
    num_nodes_w_feat, node_feat_dim, node_dtype = _get_numpy_array_info(node_np)
    num_edges_w_feat, edge_feat_dim, edge_dtype = _get_numpy_array_info(edge_np)
    return num_nodes_w_feat, node_feat_dim, node_dtype, num_edges_w_feat, edge_feat_dim, edge_dtype


def get_model_save_path(data_name, config_base_name):
    if not os.path.isdir('models'):
        os.mkdir('models')
    return 'models/{}_{}_{}.pkl'.format(data_name, config_base_name, time.time()) # time.time() 在 rank 0 执行

def get_node_count_from_graph(g):
    return g['indptr'].shape[0] - 1
from sampler_core import ParallelSampler

def sampler_wrapper(no_neg, root_nodes, sampler: ParallelSampler|None, ts, total_sample_time=None):
    if no_neg:
        pos_root_end = root_nodes.shape[0] * 2 // 3
        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
    else:
        sampler.sample(root_nodes, ts)
    ret = sampler.get_ret()
    if total_sample_time is not None:
        total_sample_time += ret[0].sample_time()
    return ret, total_sample_time

from typing import List
from dgl.heterograph import DGLBlock
def to_dgl_blocks(ret, history, reverse=False, cuda=True) -> List[List[DGLBlock]]:
    # eg. if sample_param.history = 3, sample_param.layer = 2, ret = [Ly1His1, Ly1His2, Ly1His3, Ly2His1, Ly2His2, Ly2His3]
    # then grouped_mfgs = [[Ly1His1, Ly1His2], [Ly1His3, Ly2His1, [Ly2His2, Ly2His3]]
    mfgs = list()
    # 6 tgbs -> mfgs length 6 [1,2,3,4,5,6]
    # if history = 2, grouped_mfgs = [[1,2],[3,4],[5,6]]
    for tgb in ret:
        tgb_nodes = torch.from_numpy(tgb.nodes())
        tgb_dts = torch.from_numpy(tgb.dts())
        tgb_ts = torch.from_numpy(tgb.ts())
        tgb_eid = torch.from_numpy(tgb.eid())
        if not reverse:
            edges = (tgb.col(), tgb.row())
            num_src, num_dst = tgb.dim_in(), tgb.dim_out()
            dt_slice_start = num_dst
            id_ts_target = 'srcdata'
        else:
            edges = (tgb.row(), tgb.col())
            num_src, num_dst = tgb.dim_out(), tgb.dim_in()
            dt_slice_start = num_src
            id_ts_target = 'dstdata'
        block = dgl.create_block(edges, num_src_nodes=num_src, num_dst_nodes=num_dst)
        getattr(block, id_ts_target)['ID'] = tgb_nodes
        getattr(block, id_ts_target)['ts'] = tgb_ts
        
        # Assign edge data
        block.edata['dt'] = tgb_dts[dt_slice_start:]
        block.edata['ID'] = tgb_eid
        
        mfgs.append(block.to('cuda:0') if cuda == True else block)
    # [[Ly1His1, Ly1His2], [Ly1His3, Ly2His1, [Ly2His2, Ly2His3]]
    grouped_mfgs = list(map(list, zip(*[iter(mfgs)] * history)))
    grouped_mfgs.reverse()
    return grouped_mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda) -> List[List[DGLBlock]]:
    mfgs = list()
    block = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    block.srcdata['ID'] = torch.from_numpy(root_nodes)
    block.srcdata['ts'] = torch.from_numpy(ts)
    if cuda == True:
        mfgs.insert(0, [block.to('cuda:0')])
    else:
        mfgs.insert(0, [block])
    return mfgs

def mfgs_to_cuda(mfgs):
    new_mfgs = []
    for mfg in mfgs:
        new_mfg = [block.to('cuda:0') for block in mfg]
        new_mfgs.append(new_mfg)
    return new_mfgs
import torch
import dgl
from typing import List, Optional



def prepare_mfgs(deduplication, mfgs_on_same_device_as_feat, mfgs_on_gpu, node_feats, edge_feats, deduplicate_mfg_src=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):

    time_stats = {}
    count_stats = {}
    # mfgs = [[block1, block2], [block3, block4]] ,
    # 也就是layer 0 有 block1 和 block2, 
    # layer 1 有 block3 和 block4
    
    # 信息传递方向 layer 0 -> layer 1
    # layer 0: src->dst, layer 1: src->dst
    
    # 所有layer 1的dst节点才是我们需要的节点
    # DGLBlock的设计时src节点包含了dst节点
    # 所以layer0的src必然包含了所有节点
    
    mfgs_on_gpu_0 = mfgs_on_gpu[0] #[block1, block2]
    mfgs_on_same_device_as_feat_0 = mfgs_on_same_device_as_feat[0] #[block1, block2]
    # 第1部分: 可选 - 合并第一个MFG中相同的源节点
    # print(f"[Batch {batch} Combine first MFGs: {deduplicate_mfg_src}")
    if deduplicate_mfg_src:
        for i, block in enumerate(mfgs_on_gpu_0):
            # 如果src节点数等于dst节点数,说明dst就包含了所有src节点
            # src没有多余节点,dst节点和src节点没有重复节点,也就不需要合并了
            if block.num_src_nodes() > block.num_dst_nodes():
                num_dst = block.num_dst_nodes()
                
                # 下面是src节点的时间戳和ID
                ts = block.srcdata['ts'][num_dst:]
                nid = block.srcdata['ID'][num_dst:].float()
                # 201,85 202,88 203,89 201,85
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                # 201,85 202,88 203,89
                # 0 1 2 0
                uts = unts[:, 0] # 85,88,89
                unid = unts[:, 1]# 201,202,203
                
                # import pdb; pdb.set_trace()
                # use same device as block
                src = idx + num_dst #其中idx是list # 2+0, 2+1, 2+2, 2+0
                dst = block.edges()[1] # 0 1
                new_block = dgl.create_block((src, dst), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                new_block.srcdata['ts'] = torch.cat([block.srcdata['ts'][:num_dst], uts], dim=0)
                new_block.srcdata['ID'] = torch.cat([block.srcdata['ID'][:num_dst], unid], dim=0)
                new_block.edata['dt'] = block.edata['dt']
                new_block.edata['ID'] = block.edata['ID']
                mfgs_on_gpu_0[i] = new_block

    # add node features to all blocks inside 1st mfgs
    # 只需要处理mfgs_0就行了,因为mfgs_0的src必然包含全部节点
    

    # create a set to store unique node IDs
    node_feats_on_gpu = None
    node_uniq = None
    edge_feats_on_gpu = None
    edge_uniq = None
    if deduplication: 
        with Timer() as t:
        # create a set to store unique node IDs
            cat_src = torch.cat([b.srcdata['ID'] for b in mfgs_on_same_device_as_feat_0], dim=0)
            node_uniq, inv = torch.unique(cat_src, sorted=False, return_inverse=True)  # 和feat相同的设备上做
            inv = inv.to('cuda', non_blocking=True)
        time_stats['0_compute_uniq_node_id'] = t.last_duration
        with Timer() as t:
            offset = 0
            for b in mfgs_on_gpu_0:
                n = b.num_src_nodes()
                b.srcdata['pos'] = inv[offset:offset+n]
                offset += n
        time_stats['2_transfer_node_mapping'] = t.last_duration
        if node_feats is not None:
            with Timer() as t:
                if pinned:
                    torch.index_select(node_feats, 0, node_uniq, out=nfeat_buffs[:node_uniq.shape[0]])
                    node_feats_on_gpu = nfeat_buffs[:node_uniq.shape[0]].cuda(non_blocking=True)
                else:
                    node_feats_on_gpu = node_feats[node_uniq].to('cuda', non_blocking=True)
            time_stats['1_transfer_node_feat'] = t.last_duration

            
            
            count_stats['unique_node_id_count'] = node_uniq.shape[0]
            count_stats['total_node_id_count'] = cat_src.shape[0]


        if edge_feats is not None:
            with Timer() as t:
                edges = [
                    b.edata['ID']
                    for mfg in mfgs_on_same_device_as_feat
                    for b in mfg
                    if 'ID' in b.edata
                ]
                cat_edge = torch.cat(edges, dim=0) if edges else torch.tensor([], dtype=torch.long)
                edge_uniq, inv = torch.unique(cat_edge, sorted=False, return_inverse=True)  # 和feat相同的设备上做
                inv = inv.to('cuda', non_blocking=True)
            time_stats['0_compute_uniq_edge_id'] = t.last_duration
            
            with Timer() as t:   
                if pinned:
                    torch.index_select(edge_feats, 0, edge_uniq, out=efeat_buffs[:edge_uniq.shape[0]])
                    edge_feats_on_gpu = efeat_buffs[:edge_uniq.shape[0]].cuda(non_blocking=True)
                else:
                    edge_feats_on_gpu = edge_feats[edge_uniq].to('cuda', non_blocking=True)  # GPU 上做
            time_stats['1_transfer_edge_feat'] = t.last_duration
            with Timer() as t:
                offset = 0
                for mfg in mfgs_on_gpu:
                    for b in mfg:
                        n = b.num_edges()
                        b.edata['pos'] = inv[offset:offset+n]
                        offset += n
            time_stats['2_transfer_edge_mapping'] = t.last_duration
            
            
            count_stats['unique_edge_id_count'] = edge_uniq.shape[0]
            count_stats['total_edge_id_count'] = cat_edge.shape[0]
            



    else:

        i = 0
        if node_feats is not None:
            for block in mfgs_on_gpu_0:
                with Timer() as t:   
                    if pinned:
                        idx = block.srcdata['ID'].cpu().long()
                        torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                        block.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                    else:
                        idx = block.srcdata['ID'].long()
                        srch = node_feats[idx.to(node_feats.device)].float()
                        block.srcdata['h'] = srch.cuda()
                    
                    i += 1
                time_stats['1_transfer_node_feat'] = t.last_duration

        
        
        i = 0
        if edge_feats is not None:
            for mfg in mfgs_on_gpu:
                for block in mfg:
                    if block.num_src_nodes() > block.num_dst_nodes():
                        with Timer() as t:   
                            if pinned:
                                idx = block.edata['ID'].cpu().long()
                                
                                # total_edge_id_count += idx.shape[0]
                                # unique_edge_id_count += torch.unique(idx).shape[0]
                                torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                                block.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                            else:
                                idx = block.edata['ID'].long()
                                srch = edge_feats[idx.to(edge_feats.device)].float()
                                block.edata['f'] = srch.cuda()

                            i += 1
                        time_stats['1_transfer_edge_feat'] = t.last_duration

    if profiler is not None:
        profiler.increase_time(time_stats)
        profiler.increase_count(count_stats)
    return mfgs_on_gpu, node_feats_on_gpu, edge_feats_on_gpu, node_uniq, edge_uniq
