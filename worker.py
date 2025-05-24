
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from modules import GeneralModel
from utils import   mfgs_to_cuda, prepare_input
import torch.distributed as dist
from modules import  MemoryMailbox, GNNParams, GeneralModel, MemoryParams
from data_pipeline_thread import DataPipelineThread
local_rank = dist.get_rank()
world_size = dist.get_world_size()
host_rank = world_size - 1
from torch.nn.parallel import DistributedDataParallel as DDP
        

def scatter_receive(src=host_rank):
    my = [None]
    dist.scatter_object_list(my, [None] * world_size, src=src)
    return my[0]




def run_training_step(stream, prev_thread: DataPipelineThread, mailbox: MemoryMailbox,  memory_param: MemoryParams, model: DDP, optimizer, creterion, node_feats, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, tot_loss):
    
    names = ['mfgs', 'root', 'ts', 'eid', 'block']
    my_lists = {name: None for name in names}
    
    my_lists['mfgs'] = scatter_receive()
    if mailbox is not None:
        my_lists['root'] = scatter_receive()
        my_lists['ts'] = scatter_receive()
        my_lists['eid'] = scatter_receive()
        if memory_param.deliver_to == 'neighbors':
            my_lists['block'] = scatter_receive()

    curr_thread = DataPipelineThread(my_lists, stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox, memory_param)
    curr_thread.start()
    if prev_thread is not None:
        prev_thread, tot_loss = train_from_prefetch(prev_thread, mailbox, memory_param, model, optimizer, creterion, edge_feats, nccl_group, tot_loss, curr_thread)
    else:
        prev_thread = curr_thread
    return prev_thread, tot_loss
        
        

def train_from_prefetch(prev_thread: DataPipelineThread, memory_mailbox: MemoryMailbox, memory_param: MemoryParams, model: DDP, optimizer, creterion, edge_feats, nccl_group, tot_loss, curr_thread):
    memory_updater = model.module.memory_updater
    prev_thread.join()
    # with torch.cuda.stream(prev_thread.get_stream()):
    mfgs = prev_thread.get_mfgs()
    model.train()
    optimizer.zero_grad()
    pred_pos, pred_neg = model(mfgs)
    loss = creterion(pred_pos, torch.ones_like(pred_pos))
    loss += creterion(pred_neg, torch.zeros_like(pred_neg))
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        tot_loss += float(loss)
    if memory_mailbox is not None:
        with torch.no_grad():
            eid = prev_thread.get_eid()
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            root_nodes = prev_thread.get_root()
            ts = prev_thread.get_ts()
            block = prev_thread.get_block()
            memory_mailbox.update_mailbox(memory_updater.last_updated_nid, memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
            memory_mailbox.update_memory(memory_updater.last_updated_nid, memory_updater.last_updated_memory, root_nodes, memory_updater.last_updated_ts)
            if memory_param.deliver_to == 'neighbors':
                dist.barrier(group=nccl_group)
                if local_rank == 0:
                    memory_mailbox.update_next_mail_pos()
    prev_thread.clear_tensors()
    prev_thread = curr_thread
    return prev_thread, tot_loss


    
def run_evaluation_step(prev_thread: DataPipelineThread, memory_mailbox: MemoryMailbox,  memory_param, model: DDP, optimizer, creterion, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, node_feats, tot_loss):
    memory_updater = model.module.memory_updater
    if prev_thread is not None:
        # finish last training mini-batch
        prev_thread, tot_loss = train_from_prefetch( prev_thread, memory_mailbox,  memory_param, model, optimizer, creterion, edge_feats, nccl_group, tot_loss, None)
        
    my_mfgs = scatter_receive()
    mfgs = mfgs_to_cuda(my_mfgs)
    prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs, efeat_buffs=pinned_efeat_buffs)
    
    model.eval()
    with torch.no_grad():
        if memory_mailbox is not None:
            memory_mailbox.prep_input_mails(mfgs[0])
        pred_pos, pred_neg = model(mfgs)
        if memory_mailbox is not None:
            
            root_nodes = scatter_receive()
            ts = scatter_receive()
            eid = scatter_receive()
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            
            if memory_param.deliver_to == 'neighbors':
                block = scatter_receive()
            memory_mailbox.update_mailbox(memory_updater.last_updated_nid, memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
            memory_mailbox.update_memory(memory_updater.last_updated_nid, memory_updater.last_updated_memory, root_nodes, memory_updater.last_updated_ts)
            
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
    return prev_thread, tot_loss

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if sample_param.neighbor is not None:
        for i in sample_param.neighbor:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param.history):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param.history):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs


# GPU worker process
def run_worker(sample_param, memory_param, gnn_param, train_param, nccl_group, memory_mailbox: MemoryMailbox, node_feat_dim, edge_feat_dim, path_saver, node_feats, edge_feats):
    stream = torch.cuda.Stream()
    model = GeneralModel(node_feat_dim, edge_feat_dim, sample_param, memory_param, gnn_param, train_param).cuda()
    find_unused_parameters = True if sample_param.history > 1 else False
    
    # ddp model
    model = DDP(model, device_ids=[local_rank], process_group=nccl_group, output_device=local_rank, find_unused_parameters=find_unused_parameters)
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param.lr)
    
    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(sample_param, train_param.batch_size, node_feats, edge_feats)
    if memory_mailbox is not None:
        memory_mailbox.allocate_pinned_memory_buffers(sample_param, train_param.batch_size)
    
    tot_loss = 0
    prev_thread = None
    while True:
        state_buf = torch.empty(1, dtype=torch.int32, device='cpu')
        dist.broadcast(state_buf, src=host_rank)
        match state_buf.item():
            case -1:# exit
                break
            case 0:# train
                prev_thread, tot_loss = run_training_step(stream, prev_thread, memory_mailbox,  memory_param, model, optimizer, creterion, node_feats, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, tot_loss)
            case 1:# eval
                prev_thread, tot_loss = run_evaluation_step(prev_thread, memory_mailbox, memory_param, model, optimizer, creterion, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, node_feats, tot_loss)
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
