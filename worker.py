
from dataclasses import dataclass
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from modules import GeneralModel, GNNParams
from utils import   mfgs_to_cuda, prepare_mfgs
import torch.distributed as dist
from memorys import  MemoryMailbox, MemoryParams
from data_pipeline_thread import DataPipelineThread
from dgl.heterograph import DGLBlock
from dataclasses import dataclass
from typing import List, Optional
import torch
from numpy.typing import NDArray
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from memorys import GRUMemoryUpdater, RNNMemoryUpdater, TransformerMemoryUpdater
# from viztracer import get_tracer
local_rank = dist.get_rank()
world_size = dist.get_world_size()
host_rank = world_size - 1
import time
import threading

def scatter_receive(src=host_rank):
    my = [None]
    dist.scatter_object_list(my, [None] * world_size, src=src)
    return my[0]

@dataclass
class TransferData:
    mfgs: Optional[List[List[DGLBlock]]]  = None
    root: Optional[NDArray[np.int32]] = None
    ts: Optional[NDArray[np.float32]] = None
    eid: Optional[List] = None
    block: Optional[List[List[DGLBlock]]]  = None

def irecv_object(src=host_rank):
    """Initiates a non-blocking receive of an object and returns the request handle."""
    # 1. 先同步接收大小（这一步很快，可以接受）
    size_tensor = torch.LongTensor([0])
    dist.recv(tensor=size_tensor, src=src)
    
    # 2. 根据大小创建 buffer，并发起非阻塞接收
    buffer = torch.ByteTensor(size=[size_tensor.item()])
    recv_request = dist.irecv(tensor=buffer, src=src)
    # 3. 返回请求句柄和 buffer
    return recv_request, buffer



def run_training_step(stream,
                      prev_thread: DataPipelineThread,
                      mailbox: MemoryMailbox, 
                      memory_param: MemoryParams, 
                      model: DDP, 
                      optimizer, 
                      creterion, 
                      node_feats, 
                      edge_feats, 
                      pinned_nfeat_buffs, 
                      pinned_efeat_buffs, 
                      nccl_group, 
                      tot_loss,
                      deduplicate_mfg_src):
    received_data: TransferData = scatter_receive()
    # data_ready_event = threading.Event()
    curr_thread = DataPipelineThread(received_data, host_rank, stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox, memory_param, deduplicate_mfg_src)
    curr_thread.start()
    if prev_thread is not None:
        prev_thread, tot_loss = train_from_prefetch(prev_thread, mailbox, memory_param, model, optimizer, creterion, edge_feats, nccl_group, tot_loss, curr_thread)
    else:
        prev_thread = curr_thread
    # data_ready_event.wait() 
    return prev_thread, tot_loss
        
        
        

def train_from_prefetch(prev_thread: DataPipelineThread, memory_mailbox: MemoryMailbox, memory_param: MemoryParams, model: DDP|GeneralModel, optimizer, creterion, edge_feats, nccl_group, tot_loss, curr_thread):
    memory_updater = model.module.memory_updater
    prev_th_hold_notmoved_data = prev_thread.received_data
    prev_thread.join()
    # with torch.cuda.stream(prev_thread.stream):
    # with get_tracer().log_event("Train: model forward and backward"):
    model.train()
    optimizer.zero_grad()
    pred_pos, pred_neg = model(prev_thread.async_mfgs)
    loss = creterion(pred_pos, torch.ones_like(pred_pos))
    loss += creterion(pred_neg, torch.zeros_like(pred_neg))
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        tot_loss += float(loss)
    if memory_mailbox is not None:
        with torch.no_grad():
            mem_edge_feats = edge_feats[prev_th_hold_notmoved_data.eid] if edge_feats is not None else None
            memory_mailbox.update_mailbox(memory_updater.last_updated_nid, memory_updater.last_updated_memory, prev_th_hold_notmoved_data.root, prev_th_hold_notmoved_data.ts, mem_edge_feats, prev_th_hold_notmoved_data.block)
            memory_mailbox.update_memory(memory_updater.last_updated_nid, memory_updater.last_updated_memory, prev_th_hold_notmoved_data.root, memory_updater.last_updated_ts)
            if memory_param.deliver_to == 'neighbors':
                dist.barrier(group=nccl_group)
                if local_rank == 0:
                    memory_mailbox.update_next_mail_pos()
    prev_thread.clear_tensors()
    prev_thread = curr_thread
    return prev_thread, tot_loss


def run_evaluation_step(prev_thread: DataPipelineThread, memory_mailbox: MemoryMailbox,  memory_param, model: DDP|GeneralModel, optimizer, creterion, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, node_feats, tot_loss, deduplicate_mfg_src):
    memory_updater: GRUMemoryUpdater|RNNMemoryUpdater|TransformerMemoryUpdater = model.module.memory_updater
    if prev_thread is not None:
        # finish last training mini-batch
        prev_thread, tot_loss = train_from_prefetch( prev_thread, memory_mailbox,  memory_param, model, optimizer, creterion, edge_feats, nccl_group, tot_loss, None)
        
    received_data : TransferData = scatter_receive()
    mfgs = mfgs_to_cuda(received_data.mfgs)
    _, node_feats_on_gpu, edge_feats_on_gpu = prepare_mfgs(mfgs, node_feats, edge_feats, deduplicate_mfg_src=deduplicate_mfg_src, pinned=True, nfeat_buffs=pinned_nfeat_buffs, efeat_buffs=pinned_efeat_buffs)
    try:

        model.eval()
        with torch.no_grad():
            if memory_mailbox is not None:
                memory_mailbox.prep_input_mails(mfgs[0])
            # with get_tracer().log_event("Eval: model forward"):
            #     print(f"Eval: model forward, local_rank={local_rank}")
            pred_pos, pred_neg = model(mfgs, node_feats_on_gpu, edge_feats_on_gpu)
            if memory_mailbox is not None:
                
                mem_edge_feats = edge_feats[received_data.eid] if edge_feats is not None else None

                memory_mailbox.update_mailbox(memory_updater.last_updated_nid, memory_updater.last_updated_memory, received_data.root, received_data.ts, mem_edge_feats, received_data.block)
                memory_mailbox.update_memory(memory_updater.last_updated_nid, memory_updater.last_updated_memory, received_data.root, memory_updater.last_updated_ts)
                
                if memory_param.deliver_to == 'neighbors':
                    dist.barrier(group=nccl_group)
                    if local_rank == 0:
                        memory_mailbox.update_next_mail_pos()
                        
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            ap = average_precision_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            dist.gather_object(float(ap), None, dst=host_rank)
            dist.gather_object(float(auc), None, dst=host_rank)
    except Exception as e:
        print(f"!!!!!!!! Rank {local_rank} caught an exception in run_evaluation_step !!!!!!!!")
        print(e)
        # 抛出异常或进行其他处理，但关键是打印出来
        raise e

    return prev_thread, tot_loss


def alc_pin_nfeat_efeat_buf(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if sample_param.neighbor is not None and edge_feats is not None:
        edge_space = 0
        for i in sample_param.neighbor:
            limit = limit * (i + 1)
            edge_space += limit * sample_param.history
        pinned_efeat_buffs = torch.zeros((edge_space, edge_feats.shape[1]), pin_memory=True)
    # if history = 3, neighbor = [10, 10]
    
    # pinned_efeat_buffs[5].torch.shape [3.3b*11*11, edge_feats.shape[1]]
    # pinned_efeat_buffs[4].torch.shape [3.3b*11*11, edge_feats.shape[1]]
    # pinned_efeat_buffs[3].torch.shape [3.3b*11*11, edge_feats.shape[1]]
    
    # pinned_efeat_buffs[2].torch.shape [3.3b*11, edge_feats.shape[1]]
    # pinned_efeat_buffs[1].torch.shape [3.3b*11, edge_feats.shape[1]]
    # pinned_efeat_buffs[0].torch.shape [3.3b*11, edge_feats.shape[1]]
    if node_feats is not None:
        pinned_nfeat_buffs = torch.zeros((limit * sample_param.history, node_feats.shape[1]), pin_memory=True))
    # if history = 3
    # pinned_nfeat_buffs[2].torch.shape [3.3b*11*11, node_feats.shape[1]]
    # pinned_nfeat_buffs[1].torch.shape [3.3b*11*11, node_feats.shape[1]]
    # pinned_nfeat_buffs[0].torch.shape [3.3b*11*11, node_feats.shape[1]]
    return pinned_nfeat_buffs, pinned_efeat_buffs

# 128 --->GPU0 tm10-tm20
# 128 --->GPU1 tm20-tm28
# 128 --->GPU3 tm28-tm32
# 128 --->GPU1 tm32-tm48

# GPU worker process
def run_worker(sample_param, memory_param, gnn_param, train_param, nccl_group, memory_mailbox: MemoryMailbox, node_feat_dim, edge_feat_dim, path_saver, node_feats, edge_feats):
    # stream position 1 
    stream = torch.cuda.Stream()
    model = GeneralModel(node_feat_dim, edge_feat_dim, sample_param, memory_param, gnn_param, train_param, combined=train_param.combine_neighs).cuda() #try by sqp17
    find_unused_parameters = True if sample_param.history > 1 else False
    
    # ddp model
    model = DDP(model, device_ids=[local_rank], process_group=nccl_group, output_device=local_rank, find_unused_parameters=find_unused_parameters)
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param.lr)
    
    # buffer for features and mem_mailbox
    pinned_nfeat_buffs, pinned_efeat_buffs = alc_pin_nfeat_efeat_buf(sample_param, train_param.batch_size, node_feats, edge_feats)
    if memory_mailbox is not None:
        memory_mailbox.alc_pin_mem_mailbox_buf(sample_param, train_param.batch_size)
    
    tot_loss = 0
    prev_thread = None
    while True:
        state_buf = torch.empty(1, dtype=torch.int32, device='cpu')
        dist.broadcast(state_buf, src=host_rank)
        match state_buf.item():
            case -1:# exit
                break
            case 0:# train
                # stream position 2
                # stream = torch.cuda.Stream()
                prev_thread, tot_loss = run_training_step(stream, prev_thread, memory_mailbox,  memory_param, model, optimizer, creterion, node_feats, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, tot_loss, train_param.deduplicate_mfg_src)
            case 1:# eval
                prev_thread, tot_loss = run_evaluation_step(prev_thread, memory_mailbox, memory_param, model, optimizer, creterion, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, node_feats, tot_loss, train_param.deduplicate_mfg_src)
            case 2:# save model
                if local_rank == 0:
                    torch.save(model.state_dict(), path_saver)
            case 3:# load model
                model.load_state_dict(torch.load(path_saver, map_location=torch.device('cuda:0')))
            case 4:# pass
                pass
            case 5:# reduce loss
                dist.gather_object(float(tot_loss), None, dst=host_rank)
                tot_loss = 0
            case _:
                raise ValueError(f"Unhandled state: {int(state_buf.item())}")
            
