import threading

import torch
import torch.distributed as dist
from gnn.TGN import TGN
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import blocks_to_cuda, prepare_blocks
from worker.h2d import H2D, Meta


local_rank = dist.get_rank()
world_size = dist.get_world_size()
host_rank = world_size - 1
device = torch.device(f"cuda:{local_rank}")


def scatter_receive(src=host_rank):
    my = [None]
    dist.scatter_object_list(my, [None] * world_size, src=src)
    return my[0]


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


def run_training_step(
    dedup,
    stream,
    prev_cpu_to_gpu_thread: H2D,
    mailbox: MemoryMailbox,
    memory_param: MemoryParams,
    model: DDP,
    optimizer,
    creterion,
    node_feats,
    edge_feats,
    pinned_nfeat_buffs,
    pinned_efeat_buffs,
    tot_loss,
    deduplicate_block_src,
    prev_mailbox_mem_updater,
    nccl_group,
):
    # received_data: TransferData = scatter_receive()
    data_ready_event = threading.Event()
    curr_cpu_to_gpu_thread = H2D(
        dedup,
        data_ready_event,
        host_rank,
        stream,
        pinned_nfeat_buffs,
        pinned_efeat_buffs,
        node_feats,
        edge_feats,
        mailbox,
        memory_param,
        deduplicate_block_src,
        prev_mailbox_mem_updater,
    )
    curr_cpu_to_gpu_thread.start()
    if prev_cpu_to_gpu_thread is not None:
        prev_cpu_to_gpu_thread, tot_loss, prev_mailbox_mem_updater = train_from_fetch(
            dedup,
            prev_cpu_to_gpu_thread,
            mailbox,
            model,
            optimizer,
            creterion,
            edge_feats,
            tot_loss,
            curr_cpu_to_gpu_thread,
            memory_param,
            nccl_group,
        )
    else:
        prev_cpu_to_gpu_thread = curr_cpu_to_gpu_thread
    data_ready_event.wait()
    return prev_cpu_to_gpu_thread, tot_loss, prev_mailbox_mem_updater


def train_from_fetch(
    is_dedup,
    cpu_to_gpu_thread: H2D,
    memory_mailbox: MemoryMailbox,
    model: DDP | TGN,
    optimizer,
    creterion,
    edge_feats,
    tot_loss,
    curr_cpu_to_gpu_thread,
    memory_param,
    nccl_group,
):
    memory_updater = model.module.memory_updater
    prev_th_hold_notmoved_data = cpu_to_gpu_thread.received_data
    cpu_to_gpu_thread.join()

    model.train()
    optimizer.zero_grad()
    # print cpu_to_gpu_thread.blockss_on_gpu
    # print(f"--- [Rank {local_rank}] Checking devices before model call ---")
    # if cpu_to_gpu_thread.blockss_on_gpu:
    #     # DGLBlock的device属性可以这么拿
    #     print(f"  - blockss_on_gpu[0].device: {cpu_to_gpu_thread.blockss_on_gpu[0][0].device}")
    # if cpu_to_gpu_thread.node_feats_on_gpu is not None:
    #     print(f"  - node_feats_on_gpu.device: {cpu_to_gpu_thread.node_feats_on_gpu.device}")
    # if cpu_to_gpu_thread.edge_feats_on_gpu is not None:
    #     print(f"  - edge_feats_on_gpu.device: {cpu_to_gpu_thread.edge_feats_on_gpu.device}")
    # if cpu_to_gpu_thread.deduplicated_mem is not None:
    #     print(f"  - deduplicated_mem.device: {cpu_to_gpu_thread.deduplicated_mem.device}")
    # if cpu_to_gpu_thread.deduplicated_mem_input is not None:
    #     print(f"  - deduplicated_mem_input.device: {cpu_to_gpu_thread.deduplicated_mem_input.device}")
    # print(f"----------------------------------------------------------")
    pred_pos, pred_neg = model(
        is_dedup,
        cpu_to_gpu_thread.blockss_on_gpu,
        cpu_to_gpu_thread.node_feats_on_gpu,
        cpu_to_gpu_thread.edge_feats_on_gpu,
        cpu_to_gpu_thread.deduplicated_mem,
        cpu_to_gpu_thread.deduplicated_mem_input,
    )
    cpu_to_gpu_thread.cleanup()
    cpu_to_gpu_thread = None
    loss = creterion(pred_pos, torch.ones_like(pred_pos))
    loss += creterion(pred_neg, torch.zeros_like(pred_neg))
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        tot_loss += loss.item()

    prev_cpu_to_gpu_thread = curr_cpu_to_gpu_thread
    return prev_cpu_to_gpu_thread, tot_loss


def run_evaluation_step(
    dedup,
    prev_cup_to_gpu_thread: H2D,
    model: DDP | TGN,
    optimizer,
    creterion,
    edge_feats,
    pinned_nfeat_buffs,
    pinned_efeat_buffs,
    node_feats,
    tot_loss,
    deduplicate_block_src,
    prev_mailbox_mem_updater,
    nccl_group,
):
    if isinstance(prev_cup_to_gpu_thread, H2D):
        prev_cup_to_gpu_thread, _ = train_from_fetch(
            dedup,
            prev_cup_to_gpu_thread,
            model,
            optimizer,
            creterion,
            edge_feats,
            tot_loss,
            None,
            nccl_group,
        )

        if prev_cup_to_gpu_thread is not None:
            prev_cup_to_gpu_thread.cleanup()
        prev_cup_to_gpu_thread = None
    prev_th_hold_notmoved_data: Meta = scatter_receive()
    blocks_on_cpu = prev_th_hold_notmoved_data.gnn_mfgs_raw = (gnn_mfgs_raw,)
    if not blocks_on_cpu or len(blocks_on_cpu) == 0:
        print(
            f"Rank {local_rank}: Received empty batch during evaluation, skipping step."
        )
        # 该函数需要返回三个值，对于跳过的情况，返回 None, tot_loss, None 是合适的
        return None, tot_loss, None
    blocks_on_gpu = blocks_to_cuda(blocks_on_cpu)
    _, node_feats_on_gpu, edge_feats_on_gpu, node_uniq, edge_uniq = prepare_blocks(
        dedup,
        blocks_on_cpu,
        blocks_on_gpu,
        node_feats,
        edge_feats,
        deduplicate_block_src=deduplicate_block_src,
        pinned=True,
        nfeat_buffs=pinned_nfeat_buffs,
        efeat_buffs=pinned_efeat_buffs,
    )

    model.eval()
    with torch.no_grad():
        pred_pos, pred_neg = model(
            dedup, blocks_on_gpu, node_feats_on_gpu, edge_feats_on_gpu
        )

        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0
        )
        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        dist.gather_object(float(ap), None, dst=host_rank)
        dist.gather_object(float(auc), None, dst=host_rank)
    return None, tot_loss


def alc_pin_nfeat_efeat_buf_w_dedup(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = None
    pinned_efeat_buffs = None
    limit = int(batch_size * 3.3)
    if sample_param.neighbor is not None and edge_feats is not None:
        edge_space = 0
        for i in sample_param.neighbor:
            limit = limit * (i + 1)
            edge_space += limit * sample_param.history
        pinned_efeat_buffs = torch.zeros(
            (edge_space, edge_feats.shape[1]), pin_memory=True
        )

    if node_feats is not None:
        pinned_nfeat_buffs = torch.zeros(
            (limit * sample_param.history, node_feats.shape[1]), pin_memory=True
        )

    return pinned_nfeat_buffs, pinned_efeat_buffs


def alc_pin_nfeat_efeat_buf_no_dedup(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if sample_param.neighbor is not None:
        for i in sample_param.neighbor:
            limit *= i + 1

            if edge_feats is not None:
                for _ in range(sample_param.history):
                    pinned_efeat_buffs.insert(
                        0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True)
                    )
    # if history = 3, neighbor = [10, 10]

    # pinned_efeat_buffs[5].torch.shape [3.3b*11*11, edge_feats.shape[1]]
    # pinned_efeat_buffs[4].torch.shape [3.3b*11*11, edge_feats.shape[1]]
    # pinned_efeat_buffs[3].torch.shape [3.3b*11*11, edge_feats.shape[1]]

    # pinned_efeat_buffs[2].torch.shape [3.3b*11, edge_feats.shape[1]]
    # pinned_efeat_buffs[1].torch.shape [3.3b*11, edge_feats.shape[1]]
    # pinned_efeat_buffs[0].torch.shape [3.3b*11, edge_feats.shape[1]]
    if node_feats is not None:
        for _ in range(sample_param.history):
            pinned_nfeat_buffs.insert(
                0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True)
            )
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
def run_worker(
    dedup,
    sample_param,
    memory_param,
    gnn_param,
    train_param,
    nccl_group,
    memory_mailbox: MemoryMailbox,
    node_feat_dim,
    edge_feat_dim,
    path_saver,
    node_feats,
    edge_feats,
):
    # stream position 1
    stream = torch.cuda.Stream()
    model = TGN(
        node_feat_dim,
        edge_feat_dim,
        sample_param,
        memory_param,
        gnn_param,
        train_param,
        combined=train_param.merge_src,
    ).cuda()  # try by sqp17
    find_unused_parameters = True if sample_param.history > 1 else False

    # ddp model
    model = DDP(
        model,
        device_ids=[0],
        process_group=nccl_group,
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param.lr)

    # buffer for features and mem_mailbox
    if dedup:
        pinned_nfeat_buffs, pinned_efeat_buffs = alc_pin_nfeat_efeat_buf_w_dedup(
            sample_param, train_param.batch_size, node_feats, edge_feats
        )
    else:
        pinned_nfeat_buffs, pinned_efeat_buffs = alc_pin_nfeat_efeat_buf_no_dedup(
            sample_param, train_param.batch_size, node_feats, edge_feats
        )

    tot_loss = 0
    prev_cup_to_gpu_thread = None
    prev_mailbox_mem_updater = None
    while True:
        state_buf = torch.empty(1, dtype=torch.int32, device="cpu")
        dist.broadcast(state_buf, src=host_rank)
        match state_buf.item():
            case -1:  # exit
                break
            case 0:  # train
                # stream position 2
                # stream = torch.cuda.Stream()
                prev_cup_to_gpu_thread, tot_loss, prev_mailbox_mem_updater = (
                    run_training_step(
                        dedup,
                        stream,
                        prev_cup_to_gpu_thread,
                        memory_mailbox,
                        memory_param,
                        model,
                        optimizer,
                        creterion,
                        node_feats,
                        edge_feats,
                        pinned_nfeat_buffs,
                        pinned_efeat_buffs,
                        tot_loss,
                        train_param.deduplicate_block_src,
                        prev_mailbox_mem_updater,
                        nccl_group,
                    )
                )
            case 1:  # eval
                prev_cup_to_gpu_thread, tot_loss, prev_mailbox_mem_updater = (
                    run_evaluation_step(
                        dedup,
                        prev_cup_to_gpu_thread,
                        memory_mailbox,
                        memory_param,
                        model,
                        optimizer,
                        creterion,
                        edge_feats,
                        pinned_nfeat_buffs,
                        pinned_efeat_buffs,
                        node_feats,
                        tot_loss,
                        train_param.deduplicate_block_src,
                        prev_mailbox_mem_updater,
                        nccl_group,
                    )
                )
            case 2:  # save model
                if local_rank == 0:
                    torch.save(model.state_dict(), path_saver)
            case 3:  # load model
                model.load_state_dict(
                    torch.load(path_saver, map_location=torch.device("cuda:0"))
                )
            case 4:  # pass
                pass
            case 5:  # reduce loss
                dist.gather_object(float(tot_loss), None, dst=host_rank)
                tot_loss = 0
            case _:
                raise ValueError(f"Unhandled state: {int(state_buf.item())}")
