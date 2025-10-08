from dataclasses import dataclass
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from modules import GeneralModel, GNNParams
from utils import mfgs_to_cuda, prepare_mfgs
import torch.distributed as dist
from memorys import MemoryMailbox, MemoryParams
from pipeline.data_mover import DataMover, TransferData
from pipeline.mailbox_mem_updater import MailboxMemUpdater
from dgl.heterograph import DGLBlock
import threading
from dataclasses import dataclass
from typing import List, Optional
import torch
from numpy.typing import NDArray
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from memorys import GRUMemoryUpdater, RNNMemoryUpdater, TransformerMemoryUpdater
from torch.profiler import profile, record_function, ProfilerActivity
import pickle
import os, debugpy, time


from viztracer import get_tracer

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
    deduplication,
    stream,
    prev_data_mover: DataMover | None,
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
    deduplicate_mfg_src,
    prev_mailbox_mem_updater: MailboxMemUpdater,
    nccl_group,
):
    # received_data: TransferData = scatter_receive()
    data_ready_event = threading.Event()
    curr_data_mover = DataMover(
        deduplication,
        data_ready_event,
        host_rank,
        stream,
        pinned_nfeat_buffs,
        pinned_efeat_buffs,
        node_feats,
        edge_feats,
        mailbox,
        memory_param,
        deduplicate_mfg_src,
        prev_mailbox_mem_updater,
        prev_node_uniq=(
            prev_data_mover.node_uniq
            if prev_data_mover and hasattr(prev_data_mover, "node_uniq")
            else None
        ),
        prev_edge_uniq=(
            prev_data_mover.edge_uniq
            if prev_data_mover and hasattr(prev_data_mover, "edge_uniq")
            else None
        ),
        prev_node_feats_on_gpu=(
            prev_data_mover.node_feats_on_gpu
            if prev_data_mover and hasattr(prev_data_mover, "node_feats_on_gpu")
            else None
        ),
        prev_edge_feats_on_gpu=(
            prev_data_mover.edge_feats_on_gpu
            if prev_data_mover and hasattr(prev_data_mover, "edge_feats_on_gpu")
            else None
        ),
    )
    curr_data_mover.start()
    if prev_data_mover is not None:
        prev_data_mover, tot_loss, prev_mailbox_mem_updater = train_from_prefetch(
            deduplication,
            prev_data_mover,
            mailbox,
            model,
            optimizer,
            creterion,
            edge_feats,
            tot_loss,
            curr_data_mover,
            memory_param,
            nccl_group,
        )
    else:
        prev_data_mover = curr_data_mover
    data_ready_event.wait()
    return prev_data_mover, tot_loss, prev_mailbox_mem_updater


def train_from_prefetch(
    is_deduplication,
    prev_data_mover: DataMover,
    memory_mailbox: MemoryMailbox,
    model: DDP | GeneralModel,
    optimizer,
    creterion,
    edge_feats,
    tot_loss,
    curr_data_mover,
    memory_param,
    nccl_group,
):
    if memory_param.type != "none":
        memory_updater = model.module.memory_updater
    prev_th_hold_notmoved_data = prev_data_mover.received_data
    prev_data_mover.join()

    model.train()
    optimizer.zero_grad()
    torch.cuda.current_stream().wait_stream(prev_data_mover.stream)

    pred_pos, pred_neg = model(
        is_deduplication,
        prev_data_mover.mfgss_on_gpu,
        prev_data_mover.node_feats_on_gpu,
        prev_data_mover.edge_feats_on_gpu,
        prev_data_mover.deduplicated_mem,
        prev_data_mover.deduplicated_mem_input,
    )

    loss = creterion(pred_pos, torch.ones_like(pred_pos))
    loss += creterion(pred_neg, torch.zeros_like(pred_neg))
    loss.backward()
    optimizer.step()
    prev_data_mover.join()
    prev_data_mover.cleanup()
    prev_data_mover = None
    with torch.no_grad():
        tot_loss += loss.item()
    mailbox_mem_updater = None
    if memory_mailbox is not None:
        with torch.no_grad():
            mem_edge_feats = (
                edge_feats[prev_th_hold_notmoved_data.eid]
                if edge_feats is not None
                else None
            )
            last_updated_nid = memory_updater.last_updated_nid.clone()
            last_updated_memory = memory_updater.last_updated_memory.clone()
            last_updated_ts = memory_updater.last_updated_ts.clone()
            #    def __init__(self, deduplication, memory_mailbox: MemoryMailbox, nid, memory, root_nodes, ts, mem_edge_feats, block):

            mailbox_mem_updater = MailboxMemUpdater(
                deduplication=is_deduplication,
                memory_mailbox=memory_mailbox,
                nid=last_updated_nid,
                memory=last_updated_memory,
                root_nodes=prev_th_hold_notmoved_data.root,
                ts=last_updated_ts,  # Use the timestamp from the updater
                mem_edge_feats=mem_edge_feats,
                block=prev_th_hold_notmoved_data.block,
            )

            mailbox_mem_updater.start()
            # if memory_param.deliver_to == 'neighbors':
            #     mailbox_memory_update_thread.join()
            #     dist.barrier(group=nccl_group)
            #     if local_rank == 0:
            #         memory_mailbox.update_next_mail_pos()

    return curr_data_mover, tot_loss, mailbox_mem_updater


def run_evaluation_step(
    is_dedup,
    prev_data_mover: DataMover | None,
    memory_mailbox: MemoryMailbox,
    memory_param,
    model: DDP | GeneralModel,
    optimizer,
    creterion,
    edge_feats,
    pinned_nfeat_buffs,
    pinned_efeat_buffs,
    node_feats,
    tot_loss,
    deduplicate_mfg_src,
    prev_mailbox_mem_updater,
    nccl_group,
):
    if memory_param.type != "none":
        memory_updater: (
            GRUMemoryUpdater | RNNMemoryUpdater | TransformerMemoryUpdater
        ) = model.module.memory_updater
    if isinstance(prev_data_mover, DataMover):

        prev_data_mover, _, prev_mailbox_mem_updater = train_from_prefetch(
            is_dedup,
            prev_data_mover,
            memory_mailbox,
            model,
            optimizer,
            creterion,
            edge_feats,
            tot_loss,
            None,
            memory_param,
            nccl_group,
        )

    # if prev_data_mover is not None:
    #     prev_data_mover.cleanup()
    prev_th_hold_notmoved_data: TransferData = scatter_receive()
    mfgs_on_cpu = prev_th_hold_notmoved_data.mfgs
    if not mfgs_on_cpu or len(mfgs_on_cpu) == 0:
        print(
            f"Rank {local_rank}: Received empty batch during evaluation, skipping step."
        )
        # 该函数需要返回三个值，对于跳过的情况，返回 None, tot_loss, None 是合适的
        return None, tot_loss, None
    mfgs_on_gpu = mfgs_to_cuda(mfgs_on_cpu)
    _, node_feats_on_gpu, edge_feats_on_gpu, node_uniq, edge_uniq = prepare_mfgs(
        is_dedup,
        mfgs_on_cpu,
        mfgs_on_gpu,
        node_feats,
        edge_feats,
        deduplicate_mfg_src=deduplicate_mfg_src,
        pinned=True,
        nfeat_buffs=pinned_nfeat_buffs,
        efeat_buffs=pinned_efeat_buffs,
        prev_node_uniq=prev_data_mover.node_uniq if prev_data_mover else None,
        prev_edge_uniq=prev_data_mover.edge_uniq if prev_data_mover else None,
        prev_node_feats_on_gpu=(
            prev_data_mover.node_feats_on_gpu if prev_data_mover else None
        ),
        prev_edge_feats_on_gpu=(
            prev_data_mover.edge_feats_on_gpu if prev_data_mover else None
        ),
    )
    print(node_uniq)
    if prev_data_mover is not None:
        prev_data_mover.join()
    if prev_data_mover is not None:
        prev_data_mover.cleanup()
    prev_data_mover = None
    model.eval()
    with torch.no_grad():
        dedup_mem = None
        dedup_mem_input = None
        curr_mailbox_memory_update_thread = None
        if prev_mailbox_mem_updater is not None:
            prev_mailbox_mem_updater.join()
            if memory_param.deliver_to == "neighbors":
                dist.barrier(group=nccl_group)
                if local_rank == 0:
                    memory_mailbox.update_next_mail_pos()
            prev_mailbox_mem_updater.cleanup()
            prev_mailbox_mem_updater = None
        if memory_mailbox is not None:
            dedup_mem, dedup_mem_input = memory_mailbox.prep_input_mails(
                is_dedup, mfgs_on_gpu[0], node_uniq
            )

        pred_pos, pred_neg = model(
            is_dedup,
            mfgs_on_gpu,
            node_feats_on_gpu,
            edge_feats_on_gpu,
            dedup_mem,
            dedup_mem_input,
        )
        if memory_mailbox is not None:

            mem_edge_feats = (
                edge_feats[prev_th_hold_notmoved_data.eid]
                if edge_feats is not None
                else None
            )

            last_updated_nid = memory_updater.last_updated_nid.clone()
            last_updated_memory = memory_updater.last_updated_memory.clone()
            last_updated_ts = memory_updater.last_updated_ts.clone()
            curr_mailbox_memory_update_thread = MailboxMemUpdater(
                deduplication=is_dedup,
                memory_mailbox=memory_mailbox,
                nid=last_updated_nid,
                memory=last_updated_memory,
                root_nodes=prev_th_hold_notmoved_data.root,
                ts=last_updated_ts,  # Use the timestamp from the updater
                mem_edge_feats=mem_edge_feats,
                block=prev_th_hold_notmoved_data.block,
            )
            curr_mailbox_memory_update_thread.start()
            # if memory_param.deliver_to == 'neighbors':
            #     dist.barrier(group=nccl_group)
            #     if local_rank == 0:
            #         memory_mailbox.update_next_mail_pos()

        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0
        )
        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        dist.gather_object(float(ap), None, dst=host_rank)
        dist.gather_object(float(auc), None, dst=host_rank)
    return None, tot_loss, curr_mailbox_memory_update_thread


def alc_pin_nfeat_efeat_buf_w_deduplication(
    sample_param, batch_size, node_feats, edge_feats
):
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


def alc_pin_nfeat_efeat_buf_no_deduplication(
    sample_param, batch_size, node_feats, edge_feats
):
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
from enum import Enum


class WorkerState(Enum):
    EXIT = -1
    TRAIN = 0
    EVAL = 1
    SAVE_MODEL = 2
    LOAD_MODEL = 3
    PASS = 4
    REDUCE_LOSS = 5
    BARRIER = 6
    CLEANUP = 7
    ABORT_NOW = 8
    ABORT_OK = 9


# GPU worker process
def run_worker(
    is_dedup,
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
    model = GeneralModel(
        node_feat_dim,
        edge_feat_dim,
        sample_param,
        memory_param,
        gnn_param,
        train_param,
        combined=train_param.combine_neighs,
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
    if is_dedup:
        pinned_nfeat_buffs, pinned_efeat_buffs = (
            alc_pin_nfeat_efeat_buf_w_deduplication(
                sample_param, train_param.batch_size, node_feats, edge_feats
            )
        )
    else:
        pinned_nfeat_buffs, pinned_efeat_buffs = (
            alc_pin_nfeat_efeat_buf_no_deduplication(
                sample_param, train_param.batch_size, node_feats, edge_feats
            )
        )
    if memory_mailbox is not None:
        memory_mailbox.alc_pin_mem_mailbox_buf(
            is_dedup, sample_param, train_param.batch_size
        )

    tot_loss = 0

    prev_data_mover = None
    prev_mailbox_mem_updater = None
    # try:
    while True:
        state_buf = torch.empty(1, dtype=torch.int32, device="cpu")
        dist.broadcast(state_buf, src=host_rank)
        print(f"Rank {local_rank}: Received command {state_buf.item()}.")
        print("state_buf:", state_buf.item())
        print("WorkerState.EXIT:", WorkerState.TRAIN)
        match WorkerState(state_buf.item()):
            case WorkerState.EXIT:  # exit
                print(f"Rank {local_rank}: Received EXIT command, breaking the loop.")
                break
            case WorkerState.TRAIN:  # train
                print(f"Rank {local_rank}: Starting TRAIN step.")
                # stream position 2
                # stream = torch.cuda.Stream()
                prev_data_mover, tot_loss, prev_mailbox_mem_updater = run_training_step(
                    is_dedup,
                    stream,
                    prev_data_mover,
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
                    train_param.deduplicate_mfg_src,
                    prev_mailbox_mem_updater,
                    nccl_group,
                )
            case WorkerState.EVAL:  # eval
                print(f"Rank {local_rank}: Starting EVAL step.")
                prev_data_mover, tot_loss, prev_mailbox_mem_updater = (
                    run_evaluation_step(
                        is_dedup,
                        prev_data_mover,
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
                        train_param.deduplicate_mfg_src,
                        prev_mailbox_mem_updater,
                        nccl_group,
                    )
                )
            case WorkerState.SAVE_MODEL:  # save model
                print(f"Rank {local_rank}: Saving model to {path_saver}.")
                if local_rank == 0:
                    torch.save(model.state_dict(), path_saver)
            case WorkerState.LOAD_MODEL:  # load model
                print(f"Rank {local_rank}: Loading model from {path_saver}.")
                model.load_state_dict(
                    torch.load(path_saver, map_location=torch.device("cuda:0"))
                )
            case WorkerState.PASS:  # pass
                print(f"Rank {local_rank}: PASS command received, doing nothing.")
                recv = [None]
                dist.scatter_object_list(recv, None, src=host_rank)
                continue
            case WorkerState.REDUCE_LOSS:  # reduce loss
                print(f"Rank {local_rank}: Reducing loss to host.")
                dist.gather_object(float(tot_loss), None, dst=host_rank)
                tot_loss = 0
            case WorkerState.BARRIER:  # sync barrier
                print(f"Rank {local_rank}: Entering barrier.")
                dist.barrier()
                continue
            case WorkerState.CLEANUP:  # cleanup: join threads, flush CUDA, free buffers
                print(f"Rank {local_rank}: Starting CLEANUP process.")
                try:
                    stream.synchronize()
                except Exception as e:
                    raise e
                # join the in-flight prefetch thread if any
                if prev_data_mover is not None:
                    prev_data_mover.join()
                    try:
                        prev_data_mover.cleanup()
                    except Exception as e:
                        raise e

                    if prev_data_mover is not None:
                        prev_data_mover.join()
                    if prev_data_mover is not None:
                        prev_data_mover.cleanup()
                    prev_data_mover = None

                if prev_mailbox_mem_updater is not None:
                    prev_mailbox_mem_updater.join()
                    try:
                        prev_mailbox_mem_updater.cleanup()
                    except Exception as e:
                        raise e
                    prev_mailbox_mem_updater = None

                # flush mailbox pinned buffers if you have an API; otherwise skip
                try:
                    if memory_mailbox is not None and hasattr(
                        memory_mailbox, "finalize"
                    ):
                        memory_mailbox.finalize()
                except Exception as e:
                    raise e

                # make sure CUDA stream and device are drained
                try:
                    torch.cuda.synchronize()
                except Exception as e:
                    raise e
                continue
            case WorkerState.ABORT_NOW:  # abort now
                print(f"Rank {local_rank}: Aborting immediately as per host command.")
                import sys, os

                sys.exit(130)  # or: os._exit(130)
            case WorkerState.ABORT_OK:  # abort_ok
                print(f"Rank {local_rank}: Aborting gracefully as per host command.")
                import sys, os, time

                # 尽量把还在排队的本地活清一下（如果你没有前面的 'cleanup'）
                try:
                    torch.cuda.synchronize()
                except:
                    pass
                sys.stdout.flush()
                sys.stderr.flush()
                time.sleep(0.02)
                os._exit(0)
            case _:
                pass
    # finally:
    #     # Graceful shutdown logic should be here
    #     print(f"Rank {local_rank}: Exiting worker loop, performing final cleanup.")
    #     # ... del model, optimizer ...
    #     torch.cuda.synchronize()
    #     # ... destroy process groups ...

    # graceful shutdown (after loop break)
    try:
        # ensure model/optim are freed so communicators can tear down
        del optimizer
        # DDP wrapper keeps refs; drop it so NCCL communicators can close
        model = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 进入与 host 对齐的最终 barrier（默认组）
    dist.barrier()

    # 如果你创建过自定义进程组 nccl_group，单独销毁它
    try:
        if nccl_group is not None:
            dist.destroy_process_group(nccl_group)
    except Exception:
        pass

    dist.destroy_process_group()  # 默认组
