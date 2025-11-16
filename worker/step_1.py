from typing import Literal, Optional

import torch
from torch import Tensor, distributed as dist
from torch_scatter import scatter

from schemas.datatypes import DataTsNid


MAILBOX_SIZE = 10


def query_gpu(target, last_idx_dense, uniq, ts):
    # uniq 是升序的
    j = torch.bucketize(torch.tensor([target], device=uniq.device), uniq, right=False)
    if j.item() >= uniq.numel() or uniq[j] != target:
        return None
    i = last_idx_dense[j]
    return int(i.item()), int(ts[i].item()), None  # mem 若在CPU列表可再取


def lastest_1(mem_td_nid, desired_edge):  # based_on_latest_mem
    pass


# def mean_1(mem_td_nid, all_edge):  # based_on_mean_mem
#     pass


def write_mail_back_to_cpu(shared_mem, data_td_nid, device="cpu"):
    # cpu上面的mailbox特点:有所有的mail
    nid = data_td_nid.nid

    data_dim = data_td_nid.data.dim()
    if data_dim == 2:
        pos = shared_mem.next_mail_position[nid]
        shared_mem.mailbox[nid, pos] = data_td_nid.data[nid]
        shared_mem.mailbox_ts[nid] = data_td_nid.ts[nid]
        shared_mem.next_mail_position.index_add_(0, nid, nid)

    elif data_dim == 3:
        raise NotImplementedError("Only support 1 mail per batch now")


def fill_mailbox_with_gpu(
    mem_td_nid,
    root_nodes,
    ts,
    n_src,
    mail_delivery_block,
    mail_strategy,
    env,
    what_is_need_for_next_batch,
    edge_feats=None,
):
    src_mail, dst_mail = _concat_src_dst_memory(mem_td_nid, edge_feats=None)
    mail, mail_ts, unique_node_ids, inverse_indices = _prepare_mail_and_node_ids(
        root_nodes, ts, n_src, mail_delivery_block, src_mail, dst_mail
    )
    mail, mail_ts = _combine_mail_mean_last(
        mail, mail_ts, inverse_indices, unique_node_ids, mail_strategy
    )

    mail_ts_nid = MailTsNid(nid=unique_node_ids, ts=mail_ts, mail=mail)

    mail_ts_nid = scatter_by_remainder(mem_td_nid, mail_strategy, env)
    mail_ts_nid = combine_mail_within_gpu(mail_ts_nid, mail_strategy)

    next_mail_td_nid = scatter_and_gather_again(
        mem_td_nid,
        mail_strategy,
        env,
        what_is_need_for_next_batch,
        write_mail_back_to_cpu,
    )
    return next_mail_td_nid


def _concat_src_dst_memory(mem_td_nid, edge_feats=None):
    # 这里mem是按照n_src, n_dst 排序好的
    # all_edges必须按照时间排序
    # pass  # 为100个边生成mail
    memory = mem_td_nid.mem
    n_src = len(memory) / 2
    # 要求edge_feats 必须和memory对应
    # mem必须前一半全是src，后一半全是dst
    s_mem = memory[:n_src]  # e.g. Tensor[[11,12],[21,22]] 10000*100
    d_mem = memory[n_src : n_src * 2]  # e.g. Tensor[[31,32],[41,42]] 10000*100

    if edge_feats is not None:
        src_mail = torch.cat([s_mem, d_mem, edge_feats], dim=1)  # t+时刻的mem 10000*300
        dst_mail = torch.cat([d_mem, s_mem, edge_feats], dim=1)
    else:
        src_mail = torch.cat([s_mem, d_mem], dim=1)
        dst_mail = torch.cat([d_mem, s_mem], dim=1)
    return src_mail, dst_mail  # src_mail dim [n, 2*mem_dim + edge_dim]


def _prepare_mail_and_node_ids(
    root_nodes: Tensor, ts: Tensor, n_src, mail_delivery_block, src_mail, dst_mail
):
    mail = None

    mail_ts = ts[: n_src * 2]
    # TGN/JODIE
    if mail_delivery_block is None:
        src = root_nodes[:n_src]  # e.g [1,2]
        dst = root_nodes[n_src : n_src * 2].cuda()  # e.g [3,4]
        # mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
        mail = torch.cat([src_mail, dst_mail], dim=0)
        # 2 [n_ture_edges, mail_dim] -> [n_ture_edges, 2*mail_dim] -> [2*n_ture_edges, mail_dim], mail = tensor([[1, 2], [5, 6], [3, 4], [7, 8]])
        # after cat:
        # Tensor[[11,12, 31,32, 51,52,53, 31,32, 11,12, 51,52,53], [21,22, 41,42, 61,62,63, 41,42, 21,22, 61,62,63]]
        # after reshape:
        # Tensor[[11,12, 31,32, 51,52,53], [31,32, 11,12, 51,52,53], [21,22, 41,42, 61,62,63], [41,42, 21,22, 61,62,63]]
        nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
        # after unsqueeze:
        # src = tensor([[1], [2]]), dst = tensor([[3], [4]])
        # after cat:
        # nid = tensor([[1, 3], [2, 4]])
        # after reshape:
        # nid = tensor([1, 3, 2, 4])

        # find unique nid to update mailbox
        unique_node_ids, inverse_indices = torch.unique(nid, return_inverse=True)
    # APAN
    elif mail_delivery_block is not None:
        mail = torch.cat([src_mail, dst_mail], dim=0)
        # Tensor[[11,12, 31,32, 51,52,53], [21,22, 41,42, 61,62,63], [31,32, 11,12, 51,52,53], [41,42, 21,22, 61,62,63]]
        recipients_nid = mail_delivery_block.dstdata["ID"]
        unique_node_ids, inverse_indices = torch.unique(
            recipients_nid, return_inverse=True
        )
        src_of_edge = mail_delivery_block.edges()[0]
        mail = torch.cat([mail, mail[src_of_edge.long()]], dim=0)
        mail_ts = torch.cat([mail_ts, mail_ts[src_of_edge.long()]], dim=0)
    else:
        raise NotImplementedError
    return (
        mail,
        mail_ts,
        unique_node_ids,
        inverse_indices,
    )  # mail dim [2n, 2(2*mem_dim + edge_dim)]


def _combine_mail_mean_last(
    mail,
    mail_ts,
    inverse_indices,
    unique_node_ids,
    mail_combine: Literal["last", "mean"] = "last",
):
    if mail_combine == "mean":
        mail = scatter(mail, inverse_indices, reduce="mean", dim=0)
        mail_ts = scatter(mail_ts, inverse_indices, reduce="mean")
    elif mail_combine == "last":
        # find unique nid to update mailbox
        # another e.g. nid [2, 3, 2, 5, 3]
        # unique_node_ids [2, 3, 5]
        # inverse_indices [0, 1, 0, 2, 1]
        perm = torch.arange(
            inverse_indices.size(0),
            dtype=inverse_indices.dtype,
            device=inverse_indices.device,
        )
        # e.g. perm [0, 1, 2, 3, 4]
        perm = inverse_indices.new_empty(unique_node_ids.size(0)).scatter_(
            0, inverse_indices, perm
        )
        # e.g.
        # after new_empty: [0, 0, 0]
        # after scatter: [2, 4, 3]
        # nid = nid[perm] # size 3 [2, 3, 5], useless
        mail = mail[perm]
        mail_ts = mail_ts[perm]
    else:
        raise NotImplementedError
    return mail, mail_ts


def send_mail_to_neighbor(mfgs):
    pass


def top100_hits(all_edges: torch.Tensor, nid: int):
    # 找到等于 nid 的位置
    rows, cols = (all_edges == nid).nonzero(
        as_tuple=True
    )  # rows ∈ {0,1}, cols 是列索引

    if cols.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=all_edges.device)

    # 按列索引从大到小排序
    order = torch.argsort(cols, descending=True)
    rows = rows[order]
    cols = cols[order]

    # 取前100
    k = min(100, cols.numel())
    coords = torch.stack([rows[:k], cols[:k]], dim=1)  # 形状 [k, 2]，每行是 (row, idx)
    return coords


def combine_within_gpu(
    data_ts: DataTsNid, strategy: Optional[Literal["latest", "mean"]]
):  #
    nid = data_ts.nid
    # nid ts data长度是一一对应的
    data_dim = data_ts.data.dim()
    if data_dim == 2:
        assert strategy in ["latest", "mean"]
    elif data_dim == 3:
        assert strategy in ["all"]

    uniq, inv, counts = torch.unique(
        nid, sorted=False, return_inverse=True, return_counts=True
    )
    # inv 跟nid一样长,代表nid中每个元素在uniq中的位置
    # eg nid = tensor([2, 3, 2, 5, 3])
    #    uniq = tensor([2, 3, 5])
    #    inv = tensor([0, 1, 0, 2, 1])
    data_ts.nid = uniq
    data_ts.ct = counts
    pos = torch.arange(nid.size(0), device=nid.device)
    # 生成一个pos数组 0 1 2 3 ... len(nid)-1
    last_idx = torch.full((uniq.size(0),), -1, dtype=torch.long, device=nid.device)
    # last_idx_dense长度和uniq一样长，存放每个uniq对应的最后一个nid的位置
    last_idx.scatter_reduce(0, inv, pos, reduce="amax", include_self=True)
    # 对inv里面每个值,找到最大的位置,放到last_idx_dense对应位置上
    data_ts.ts = data_ts.ts[last_idx]

    data_dim = data_ts.data.dim()
    if data_dim == 2:  # this is a mailbox
        if strategy == "latest":
            data_ts.data = data_ts.data[last_idx]
            data_ts.ts = data_ts.ts[last_idx]
        elif strategy == "mean":
            mean_data = torch.full(
                (uniq.size(0),), 0, dtype=torch.float32, device=nid.device
            )
            if data_ts.ct is None:  # this is 1st step mean
                mean_data.scatter_reduce(
                    0, inv, data_ts.data, reduce="mean", include_self=False
                )
            else:
                mean_data.scatter_reduce(
                    0,
                    inv,
                    data_ts.data * data_ts.ct,
                    reduce="sum",
                    include_self=False,
                )
                mean_data /= data_ts.ct.sum()
            data_ts.data = mean_data

            mean_ts = torch.full(
                (uniq.size(0),), 0, dtype=torch.float32, device=nid.device
            )
            if data_ts.ct is None:  # this is 1st step mean
                mean_ts.scatter_reduce(
                    0, inv, data_ts.data, reduce="mean", include_self=False
                )
            else:
                mean_ts.scatter_reduce(
                    0,
                    inv,
                    data_ts.data * data_ts.ct,
                    reduce="sum",
                    include_self=False,
                )
                mean_ts /= data_ts.ct.sum()
            data_ts.ts = mean_ts

    elif data_dim == 3:  # this is a mailbox
        raise NotImplementedError("not implemented yet")

    return data_ts


def combine_mail_within_gpu(mem_td_nid: MailTsNid, strategy: Literal["latest", "mean"]):
    nid = mem_td_nid.nid
    # nid ts data长度是一一对应的

    uniq, inv, counts = torch.unique(
        nid, sorted=False, return_inverse=True, return_counts=True
    )
    # inv 跟nid一样长,代表nid中每个元素在uniq中的位置
    # eg nid = tensor([2, 3, 2, 5, 3])
    #    uniq = tensor([2, 3, 5])
    #    inv = tensor([0, 1, 0, 2, 1])
    mem_td_nid.nid = uniq
    mem_td_nid.ct = counts
    pos = torch.arange(nid.size(0), device=nid.device)
    # 生成一个pos数组 0 1 2 3 ... len(nid)-1
    last_idx = torch.full((uniq.size(0),), -1, dtype=torch.long, device=nid.device)
    # last_idx_dense长度和uniq一样长，存放每个uniq对应的最后一个nid的位置
    last_idx.scatter_reduce(0, inv, pos, reduce="amax", include_self=True)
    # 对inv里面每个值,找到最大的位置,放到last_idx_dense对应位置上
    mem_td_nid.ts = mem_td_nid.ts[last_idx]

    if strategy == "latest":
        mem_td_nid.mails = mem_td_nid.mails[last_idx]
    elif strategy == "mean":
        mean_data = torch.full(
            (uniq.size(0),), 0, dtype=torch.float32, device=nid.device
        )
        if mem_td_nid.ct is None:  # this is 1st step mean
            mean_data.scatter_reduce(
                0, inv, mem_td_nid.mails, reduce="mean", include_self=False
            )
        else:
            mean_data.scatter_reduce(
                0,
                inv,
                mem_td_nid.mails * mem_td_nid.ct,
                reduce="sum",
                include_self=False,
            )
            mean_data /= mem_td_nid.ct.sum()
        mem_td_nid.mem = mean_data

    return mem_td_nid


def combine_across_rank(
    mem_td_nid,
    strategy,
    env,
    what_is_need_for_next_batch,
    shared_mem,
    mem_q,
):
    mem_td_nid = scatter_by_remainder(mem_td_nid, strategy, env)
    mem_td_nid = combine_within_gpu(mem_td_nid, strategy)
    next_mem_td_nid = scatter_and_gather_again(
        mem_td_nid, strategy, env, what_is_need_for_next_batch, shared_mem
    )
    mem_q.put(next_mem_td_nid)


def scatter_by_remainder(mem_td_nid, strategy, env):
    device = mem_td_nid.nid.device
    node_ids = mem_td_nid.nid
    feats = mem_td_nid.data
    feats_ts = mem_td_nid.ts
    counts = mem_td_nid.ct

    D = feats.size(1)

    world_size = env.world_size
    owners = mem_td_nid.nid % world_size
    buckets_ids, buckets_x, buckets_ts, buckets_ct = [], [], [], []
    send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    for dst in range(world_size):
        m = owners == dst
        ids_dst = node_ids[m]
        buckets_ids.append(node_ids[m])
        buckets_x.append(feats[m])
        buckets_ts.append(feats_ts[m])
        if strategy == "mean":
            buckets_ct.append(counts[m])
        send_counts[dst] = ids_dst.numel()
        # 计算要送给每个gpu多少数据

    total_send = int(send_counts.sum().item())

    send_ids = (
        torch.cat(buckets_ids, dim=0)
        if total_send > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )
    send_x = (
        torch.cat(buckets_x, dim=0)
        if total_send > 0
        else torch.empty(0, feats.size(1), dtype=feats.dtype, device=device)
    )
    send_ts = (
        torch.cat(buckets_ts, dim=0)
        if total_send > 0
        else torch.empty(0, dtype=feats_ts.dtype, device=device)
    )
    if strategy == "mean":
        send_ct = (
            torch.cat(buckets_ts, dim=0)
            if total_send > 0
            else torch.empty(0, dtype=counts.dtype, device=device)
        )

    all_counts = torch.empty(world_size, world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(all_counts, send_counts)
    # 计算得到每个gpu要接受多少数据
    recv_counts = all_counts[:, env.local_rank].contiguous()
    total_recv = int(recv_counts.sum().item())

    recv_ids = torch.empty(total_recv, dtype=torch.long, device=device)
    recv_x = torch.empty(total_recv, D, dtype=feats.dtype, device=device)
    recv_ts = torch.empty(total_recv, dtype=feats_ts.dtype, device=device)
    recv_ct = torch.empty(total_recv, dtype=counts.dtype, device=device)

    in_splits = send_counts.tolist()
    out_splits = recv_counts.tolist()

    dist.all_to_all_single(
        recv_ids, send_ids, output_split_sizes=out_splits, input_split_sizes=in_splits
    )
    dist.all_to_all_single(
        recv_x, send_x, output_split_sizes=out_splits, input_split_sizes=in_splits
    )
    dist.all_to_all_single(
        recv_ts, send_ts, output_split_sizes=out_splits, input_split_sizes=in_splits
    )
    if strategy == "mean":
        dist.all_to_all_single(
            recv_ct, send_ct, output_split_sizes=out_splits, input_split_sizes=in_splits
        )

    mem_td_nid = DataTsNid(
        nid=recv_ids, mem=recv_x, ts=recv_ts, ct=recv_ct if strategy == "mean" else None
    )

    combine_within_gpu(mem_td_nid, strategy)
    return mem_td_nid


from typing import List, Set


class P2pExecuter:
    def __init__(
        self,
        env,
        shared_mem,
        *,
        mail_strategy="latest",
        mem_strategy="latest",
        has_mailbox="false",
    ):
        pass

    def exe(self, plans, data):
        (
            memplan,
            mem2disk,
            mem_in_split,
            mem_out_split,
            mailplan,
            mail_in_split,
            mail_out_split,
            mail2disk,
            featplan,
            feat_in_split,
            feat_out_split,
        ) = (
            plans.memplan,
            plans.mem2disk,
            plans.mem_in_split,
            plans.mem_out_split,
            plans.mailplan,
            plans.mail_in_split,
            plans.mail_out_split,
            plans.mail2disk,
            plans.featplan,
            plans.feat_in_split,
            plans.feat_out_split,
        )

        (mem_td_nid, mail_td_nid, mailbox_td_nid, feat_nid) = (
            data.mem_td_nid,
            data.mail_td_nid,
            data.mailbox_td_nid,
            data.feat_nid,
        )
        
        if self.deliver_to == "self":
            td.mail_next_batch_root_nodes = td.next_batch_root_nodes
        if mailbox_size == 1:
            assert self.deliver_to == "self", (
                "mailbox_size<=world_size only support deliver_to=self, otherwise slots is not enough"
            )
            if not merge_src:
                hook_mgr.register("b_inrank", combine_mail_within_gpu, mail_strategy)
                # each interaction has its own memory, so each interaction has its own mail
            hook_mgr.register("b_crrank", fill_mailbox_acorss_gpu, mail_strategy)
        elif mailbox_size > world_size:  # this gonna keep as much mail as possible
            assert not merge_src, (
                "you are wasting mailbox slots, please make sure mailbox_size <= world_size when merge_src==True"
            )
            if self.deliver_to == "self":
                hook_mgr.register(
                    "b_inrank",
                    fill_mailbox_with_gpu,
                )
                hook_mgr.register(
                    "b_crrank",
                    fill_mailbox_acorss_gpu,
                )
            elif self.deliver_to == "neighbors":
                hook_mgr.register(
                    "b_inrank",
                    mail_dilivery,
                )
                hook_mgr.register(
                    "b_crrank",
                    fill_mailbox_acorss_gpu,
                )

        if (
            self.mem_strategy == "latest"
        ):  # only keep the memory based on the latest interaction
            hook_mgr.emit("b_inrank")
            mem_td_nid = combine_within_gpu(mem_td_nid, mem_strategy)
            execute_p2p_and_writeback(
                mem_td_nid,
                self.mem_strategy,
                self.env,
                td.next_batch_root_nodes,
                self.shared_mem,
                self.mem_q,
            )  # (keep the latest mem of node[local_rank % word_size])
            hook_mgr.emit("a_crrank")
            # this is the fastest way, because it's no extra communication needed
        elif self.mem_strategy == "mean":  # average all the memory of all interactions
            hook_mgr.emit("b_inrank")
            if not self.merge_src:
                mem_td_nid = combine_within_gpu(
                    mem_td_nid, "mean"
                )  # need count the number of occurrence per node
            hook_mgr.emit("b_crrank")
            # mem_td_nid = scatter_by_remainder(mem_td_nid, strategy, env)
            mem_td_nid = combine_within_gpu(mem_td_nid, mem_strategy)
            next_mem_td_nid = execute_p2p_and_writeback(
                mem_td_nid,
                self.mem_strategy,
                self.env,
                td.next_batch_root_nodes,
                self.shared_mem,
            )
            mem_q.put(next_mem_td_nid)
            # (keep the mean mem of node[local_rank % word_size])
            hook_mgr.emit("a_crrank")
        elif self.mem_strategy in ["faster_gpu_win", "slower_gpu_win"]:
            raise NotImplementedError(
                f"mem_strategy {self.mem_strategy} not implemented yet"
            )
        
        nfeat_handler = execute_p2p_and_writeback(
            feat_nid,
            my_send_plan,
            None,
            my_input_splits,
            my_output_splits,
            None,
            env,
            shared_mem
            "feat"
        )

        # execute_p2p_and_writeback(
        #     mem_td_nid,
        #     self.mem_strategy,
        #     self.env,
        #     td.next_batch_root_nodes,
        #     self.shared_mem,
        #     self.mem_q,
        # )  # (keep the latest mem of node[local_rank % word_size])
        # memhandler = self.write_node_memory(self.dedup, mem_td_nid, n_src)

        boxhandler = self.write_mail_to_mailbox(
            mem_td_nid, mu_efeats, td.block, neg_samples
        )  # this include update mailbox, mailbox_ts, and next_mail_pos

        # # 2) 若按邻居投递，需要配合 gate 推进 next_mail_pos
        # if self.deliver_to == "neighbors" and self.gate is not None:
        #     if td.bid is not None:
        #         self.gate.mark_mm_updated(
        #             td.bid
        #         )  # 标记该 batch 中有 microbatch 完成了 mailbox 写入
        #         self.gate.wait_can_update_next_mail_pos(task.bid)
        #     # 仅让 rank0 推进写指针，避免并发写
        #     if self.local_rank == 0:
        #         self.mm.update_next_mail_pos()

        # 3) 通知 microbatch 完成（无论成功或异常）
        if self.gate is not None and mid is not None:
            self.gate.mark_finished(mid)

        self.mmq.put(
            {
                boxhandler: boxhandler,
                memhandler: memhandler,
            }
        )




def execute_p2p_and_writeback(
    mem_td_nid,
    my_send_plan: List[Set[int]],  # P2P 发送目标
    my_write_back_set: Set[int],  # 写回 CPU 目标
    my_input_splits: List[int],  # <--- 预先算好的 send_counts
    my_output_splits: List[int],  # <--- 预先算好的 recv_counts
    strategy,
    env,
    shared_mem,
    data_td_nid,
    type: Literal["mem", "mail", "mailbox", "feat"],
):
    """
    这一步之前,必须确保data_td_nid里面的数据全部是unique的,而且各个gpu持有的数据必须是unique的
    如果是mem,需要确保已经是保留最新的(latest),平均的(mean,带ct)
    如果是mail,需要确保只保留更新过的节点(pos_src,pos_dst).....另外保留最新latest,平均的(mean,带ct),或者全部(dim变大)
    如果是mailbox,需要所有节点的mailbox
    如果是feat,需要所有节点的feat
    """
    n_gpus = env.n_gpus
    device = "cuda:0"

    node_ids = data_td_nid.nid
    feats_ts = data_td_nid.ts
    feats = data_td_nid.data

    D = feats.shape[1]

    buckets_ids, buckets_x, buckets_ts, buckets_ct = [], [], [], []
    send_counts = torch.zeros(n_gpus, dtype=torch.long, device=device)

    # ---------------------------------------------------------
    # 1. 执行写回 (Write-Back)
    # ---------------------------------------------------------

    if type in ["mem", "mailbox"] and my_write_back_set:
        # 使用预先算好的 "my_write_back_set"
        # 我们可以用 torch.isin 高效地创建 mask
        write_back_tensor = torch.tensor(
            list(my_write_back_set), dtype=torch.long, device=device
        )
        write_back_mask = torch.isin(node_ids, write_back_tensor)

        # (假设 write_back_fn 的逻辑移到这里)
        remaining_nid = node_ids[write_back_mask]
        shared_mem.node_memory[remaining_nid] = feats[write_back_mask]
        shared_mem.node_memory_ts[remaining_nid] = feats_ts[write_back_mask]

    # ---------------------------------------------------------
    # 2. 准备 P2P 发送 (Scatter)
    # ---------------------------------------------------------
    # "my_send_plan" 是 List[Set[int]]
    # my_send_plan[dst_gpu_id] = {要发送给 dst_gpu_id 的 nids}

    for dst_gpu_id in range(n_gpus):
        nids_to_send_set = my_send_plan[dst_gpu_id]

        if not nids_to_send_set:
            # 这个目标GPU不需要数据
            buckets_ids.append(torch.empty(0, dtype=node_ids.dtype, device=device))
            buckets_x.append(torch.empty(0, D, dtype=feats.dtype, device=device))
            buckets_ts.append(torch.empty(0, dtype=feats_ts.dtype, device=device))
            # ... (如果 strategy == "mean", 添加 buckets_ct)
            send_counts[dst_gpu_id] = 0
            continue

        # 使用 torch.isin 高效地找到这些 nid 对应的本地数据
        nids_to_send_tensor = torch.tensor(
            list(nids_to_send_set), dtype=torch.long, device=device
        )
        send_mask_for_dst = torch.isin(node_ids, nids_to_send_tensor)

        # 打包数据
        buckets_ids.append(node_ids[send_mask_for_dst])
        buckets_x.append(feats[send_mask_for_dst])
        buckets_ts.append(feats_ts[send_mask_for_dst])
        if strategy == "mean":
            buckets_ct.append(counts[send_mask_for_dst])

    send_counts = torch.tensor(my_input_splits, dtype=torch.long, device=device)

    # ---------------------------------------------------------
    # 3. 执行 All-to-All (Gather)
    # ---------------------------------------------------------
    # 这一部分 (从 total_send = ... 开始到最后)
    # 几乎保持不变，因为它就是 "Execution" 步骤。

    total_send = int(send_counts.sum().item())

    send_ids = (
        torch.cat(buckets_ids, dim=0)
        # ... (和原来一样)
    )
    # ... (send_x, send_ts, send_ct ... 和原来一样)

    # all_counts = torch.empty(n_gpus, n_gpus, dtype=torch.long, device=device)
    # dist.all_gather_into_tensor(all_counts, send_counts)
    # ... (recv_counts, recv_ids, recv_x, ... 和原来一样)

    # ... (dist.all_to_all_single ... 和原来一样)

    # ... (return DataTsNid ... 和原来一样)

    send_ids = (
        torch.cat(buckets_ids, dim=0)
        if total_send > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )
    send_x = (
        torch.cat(buckets_x, dim=0)
        if total_send > 0
        else torch.empty(0, feats.size(1), dtype=feats.dtype, device=device)
    )
    if type in ["mem", "mailbox", "mail"]:
        send_ts = (
            torch.cat(buckets_ts, dim=0)
            if total_send > 0
            else torch.empty(0, dtype=feats_ts.dtype, device=device)
        )
    if type in ["mem", "mail"] and strategy == "mean":
        send_ct = (
            torch.cat(buckets_ts, dim=0)
            if total_send > 0
            else torch.empty(0, dtype=counts.dtype, device=device)
        )

    # all_counts = torch.empty(world_size, world_size, dtype=torch.long, device=device)
    # dist.all_gather_into_tensor(all_counts, send_counts)
    # 计算得到每个gpu要接受多少数据
    # recv_counts = all_counts[:, env.local_rank].contiguous()
    recv_counts = my_output_splits
    total_recv = int(my_output_splits.sum().item())

    recv_ids = torch.empty(total_recv, dtype=torch.long, device=device)
    recv_x = torch.empty(total_recv, D, dtype=feats.dtype, device=device)
    recv_ts = torch.empty(total_recv, dtype=feats_ts.dtype, device=device)
    recv_ct = torch.empty(total_recv, dtype=counts.dtype, device=device)

    in_splits = send_counts.tolist()
    out_splits = recv_counts.tolist()

    dist.all_to_all_single(
        recv_ids, send_ids, output_split_sizes=out_splits, input_split_sizes=in_splits
    )
    dist.all_to_all_single(
        recv_x, send_x, output_split_sizes=out_splits, input_split_sizes=in_splits
    )
    if type in ["mem", "mailbox", "mail"]:  # "mem", "mailbox", "mail"才有ts信息
        dist.all_to_all_single(
            recv_ts, send_ts, output_split_sizes=out_splits, input_split_sizes=in_splits
        )
    if type in ["mem", "mail"] and strategy == "mean":  # ["mem", "mail"] 才有mean的可能
        dist.all_to_all_single(
            recv_ct, send_ct, output_split_sizes=out_splits, input_split_sizes=in_splits
        )

    mem_td_nid = DataTsNid(
        nid=recv_ids, mem=recv_x, ts=recv_ts, ct=recv_ct if strategy == "mean" else None
    )

    return mem_td_nid


def scatter_and_gather_again(
    mem_td_nid,
    strategy,
    env,
    what_is_need_for_next_batch: list,
    shared_mem,
    write_back_fn,
    data_td_nid,
):
    def write_mail_back_to_gpu(mailbox, data_td_nid, device="cpu"):
        # cpu上面的mailbox特点:只有一部分节点的mail
        nid = data_td_nid.nid

        data_dim = data_td_nid.data.dim()
        if data_dim == 2:
            shared_mem.mailbox[nid] = data_td_nid.data[nid]
            shared_mem.next_mail_position[nid] += 1
        elif data_dim == 3:
            raise NotImplementedError("Only support 1 mail per batch now")

    # 需要mem_td_nid和what_is_need_for_next_batch里面的node都是unique
    device = mem_td_nid.nid.device
    node_ids = mem_td_nid.nid
    feats = mem_td_nid.data
    feats_ts = mem_td_nid.ts
    counts = mem_td_nid.ct

    D = feats.size(1)

    world_size = env.world_size
    owners = mem_td_nid.nid % world_size
    buckets_ids, buckets_x, buckets_ts, buckets_ct = [], [], [], []
    m_list = torch.zeros((world_size, node_ids.shape(0)))
    send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    for dst, tensor_2 in iter(what_is_need_for_next_batch):
        m = node_ids.unsqueeze(0) == tensor_2.unsqueeze(1)  # mask
        m_list[dst, m] = 1
        ids_dst = node_ids[m]
        buckets_ids.append(node_ids[m])
        buckets_x.append(feats[m])
        buckets_ts.append(feats_ts[m])
        if strategy == "mean":
            buckets_ct.append(counts[m])
        send_counts[dst] = ids_dst.numel()
        # 计算要送给每个gpu多少数据
    for_compute_remaining = torch.any(m_list, dim=0).to(torch.int)
    remaining = ~for_compute_remaining

    remaining_nid = buckets_ids[remaining]
    shared_mem.node_memory[remaining_nid] = buckets_x[remaining]
    shared_mem.node_memory_ts[remaining_nid] = buckets_ts[remaining]

    total_send = int(send_counts.sum().item())

    send_ids = (
        torch.cat(buckets_ids, dim=0)
        if total_send > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )
    send_x = (
        torch.cat(buckets_x, dim=0)
        if total_send > 0
        else torch.empty(0, feats.size(1), dtype=feats.dtype, device=device)
    )
    send_ts = (
        torch.cat(buckets_ts, dim=0)
        if total_send > 0
        else torch.empty(0, dtype=feats_ts.dtype, device=device)
    )
    if strategy == "mean":
        send_ct = (
            torch.cat(buckets_ts, dim=0)
            if total_send > 0
            else torch.empty(0, dtype=counts.dtype, device=device)
        )

    all_counts = torch.empty(world_size, world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(all_counts, send_counts)
    # 计算得到每个gpu要接受多少数据
    recv_counts = all_counts[:, env.local_rank].contiguous()
    total_recv = int(recv_counts.sum().item())

    recv_ids = torch.empty(total_recv, dtype=torch.long, device=device)
    recv_x = torch.empty(total_recv, D, dtype=feats.dtype, device=device)
    recv_ts = torch.empty(total_recv, dtype=feats_ts.dtype, device=device)
    recv_ct = torch.empty(total_recv, dtype=counts.dtype, device=device)

    in_splits = send_counts.tolist()
    out_splits = recv_counts.tolist()

    dist.all_to_all_single(
        recv_ids, send_ids, output_split_sizes=out_splits, input_split_sizes=in_splits
    )
    dist.all_to_all_single(
        recv_x, send_x, output_split_sizes=out_splits, input_split_sizes=in_splits
    )
    dist.all_to_all_single(
        recv_ts, send_ts, output_split_sizes=out_splits, input_split_sizes=in_splits
    )
    if strategy == "mean":
        dist.all_to_all_single(
            recv_ct, send_ct, output_split_sizes=out_splits, input_split_sizes=in_splits
        )

    mem_td_nid = DataTsNid(
        nid=recv_ids, mem=recv_x, ts=recv_ts, ct=recv_ct if strategy == "mean" else None
    )

    return DataTsNid


def scatter_mailbox():
    pass


def generate_mail_vectorized(
    nid: Tensor,  # [N], 唯一 node id
    new_mem: Tensor,  # [N, Dm]
    new_mem_ts: Tensor,  # [N]
    pe: Tensor,  # [E, 2] (src, dst)
    pe_ts: Tensor,  # [E]
    pe_feat: Tensor,  # [E, De]
    bs: int | None = None,  # 若需要可用于 reshape，默认不分批
):
    n_src = len(nid) / 2
    s_mem = new_mem[:n_src]
    d_mem = new_mem[n_src : n_src * 2]
    """
    生成正样本邮件（对每条正边 (src,dst) 生成两封: src<-dst 与 dst<-src）。
    返回:
        mail_node_ids: LongTensor [2E]               # 每封邮件对应的“收件人” node id（与你原先 append 的第1项一致）
        mail_payloads: Tensor     [2E, Dm+1+De]      # [对应 new_mem, 对端 ts, 边特征] 拼接后的负载
        mail_ts:       Tensor     [2E]               # 每封邮件的时间戳（取该边的 ts）
    说明:
        - 若某条边的 src 或 dst 不在 mem_id 中，会被自动过滤掉。
        - 采用排序+二分映射 id→行号，适用于离散/大 id 的通用情况。
    """
    device = nid.device
    E = pe.size(0)
    Dm = new_mem.size(1)
    De = pe_feat.size(1)
    assert new_mem.dim() == 2 and new_mem_ts.dim() == 1
    assert pe.shape == (E, 2)
    assert pe_ts.shape == (E,)
    assert pe_feat.shape == (E, De)

    # ---- 1) 建立 id -> 行号 的映射（通用：排序 + 二分）----
    sorted_idx = torch.argsort(nid)
    mem_sorted = nid[sorted_idx]  # [N]

    def map_ids_to_index(ids: Tensor):
        # 二分定位
        loc = torch.searchsorted(mem_sorted, ids)
        in_range = loc < mem_sorted.numel()
        hit = torch.zeros_like(in_range, dtype=torch.bool)
        hit[in_range] = mem_sorted[loc[in_range]] == ids[in_range]
        idx = torch.full_like(loc, -1, dtype=torch.long, device=device)
        idx[hit] = sorted_idx[loc[hit]]
        return idx, hit

    src_ids = pe[:, 0]
    dst_ids = pe[:, 1]
    src_idx, src_hit = map_ids_to_index(src_ids)
    dst_idx, dst_hit = map_ids_to_index(dst_ids)
    edge_hit = src_hit & dst_hit  # 两端都命中的边才有效

    if not torch.any(edge_hit):
        # 没有有效邮件，返回空张量
        return (
            torch.empty((0,), dtype=src_ids.dtype, device=device),
            torch.empty((0, Dm + 1 + De), dtype=new_mem.dtype, device=device),
            torch.empty((0,), dtype=pe_ts.dtype, device=device),
        )

    # 过滤无效边
    src_ids = src_ids[edge_hit]
    dst_ids = dst_ids[edge_hit]
    src_idx = src_idx[edge_hit]
    dst_idx = dst_idx[edge_hit]
    edge_ts = pe_ts[edge_hit]  # [E’]
    edge_feat = pe_feat[edge_hit]  # [E’, De]

    # ---- 2) 组装两类邮件负载 ----
    # comb_mail1 = [ new_mem[src], new_mem_ts[dst], edge_feat ]
    comb1 = torch.cat(
        (new_mem[src_idx], new_mem_ts[dst_idx].unsqueeze(1), edge_feat), dim=1
    )  # [E’, Dm+1+De]

    # comb_mail2 = [ new_mem[dst], new_mem_ts[src], edge_feat ]
    comb2 = torch.cat(
        (new_mem[dst_idx], new_mem_ts[src_idx].unsqueeze(1), edge_feat), dim=1
    )  # [E’, Dm+1+De]

    # ---- 3) 展平为 2E’ 封邮件（与原先 append 的顺序一致：先 src 再 dst）----
    mail_node_ids = torch.cat([src_ids, dst_ids], dim=0)  # [2E’]
    mail_payloads = torch.cat([comb1, comb2], dim=0)  # [2E’, Dm+1+De]
    mail_ts = torch.cat([edge_ts, edge_ts], dim=0)  # [2E’]

    # 可选：按 batch_size 切块 reshape（如果你下游想做分批发送）
    if bs is not None and bs > 0:
        # 向上取整批数
        B = (mail_node_ids.numel() + bs - 1) // bs
        pad_len = B * bs - mail_node_ids.numel()
        if pad_len > 0:
            # 做简单 padding（id/ts 用 -1 或 0；payload 用 0），并由下游用 mask 忽略
            mail_node_ids = torch.cat(
                [mail_node_ids, mail_node_ids.new_full((pad_len,), -1)]
            )
            mail_ts = torch.cat([mail_ts, mail_ts.new_zeros((pad_len,))])
            mail_payloads = torch.cat(
                [
                    mail_payloads,
                    mail_payloads.new_zeros((pad_len, mail_payloads.size(1))),
                ],
                dim=0,
            )
        mail_node_ids = mail_node_ids.view(B, bs)
        mail_ts = mail_ts.view(B, bs)
        mail_payloads = mail_payloads.view(B, bs, -1)

    return mail_node_ids, mail_payloads, mail_ts


@torch.no_grad()
def update_mailbox(
    mail_node_ids: Tensor,  # [M], 每封邮件的收件人节点索引(0..n_nodes-1)
    mail_payloads: Tensor,  # [M, D]
    mail_ts: Tensor,  # [M]
    mailbox_payloads: Tensor,  # [n_nodes, K, D]  现有信箱负载（可为空占位，会被覆盖）
    mailbox_ts: Tensor,  # [n_nodes, K]     现有信箱时间戳（可为空占位，会被覆盖）
    mode: str = "latest",  # 当 K==1 时可为 "latest" 或 "mean"；K>1 时忽略
):
    """
    将邮件写入信箱：
      - K>1: 为每个节点按 ts 降序保留最新的 K 封；
      - K==1:
          - mode="latest": 只保留每节点 ts 最大的一封；
          - mode="mean":   聚合同一节点的邮件 payload 求均值（ts 取最大或均值均可，这里取最大）。

    返回：
      new_mailbox_payloads: [n_nodes, K, D]
      new_mailbox_ts:       [n_nodes, K]
      valid_counts:         [n_nodes]  每个节点实际写入的封数（<=K；K==1 时为0/1）
    """
    assert mail_node_ids.dim() == 1
    assert mail_payloads.dim() == 2
    assert mail_ts.dim() == 1
    assert mail_payloads.size(0) == mail_node_ids.size(0) == mail_ts.size(0)

    n_nodes, K, D = (
        mailbox_payloads.size(0),
        mailbox_payloads.size(1),
        mailbox_payloads.size(2),
    )
    device = mail_payloads.device
    M = mail_node_ids.numel()

    # 过滤越界/空邮件
    mask_ok = (mail_node_ids >= 0) & (mail_node_ids < n_nodes)
    if not torch.any(mask_ok):
        # 没有可写的邮件，直接返回原信箱
        return (
            mailbox_payloads,
            mailbox_ts,
            torch.zeros(n_nodes, device=device, dtype=torch.long),
        )

    node = mail_node_ids[mask_ok]
    payload = mail_payloads[mask_ok]
    ts = mail_ts[mask_ok]

    if K == 1:
        if mode == "mean":
            # --- 每节点求均值 ---
            # sum：用 index_add_ 实现按节点聚合
            sums = torch.zeros((n_nodes, D), device=device, dtype=payload.dtype)
            sums.index_add_(0, node, payload)  # 对第0维(节点)做加和
            # 计数
            counts = torch.bincount(node, minlength=n_nodes).clamp_min_(
                1
            )  # 避免除0，用1占位
            means = sums / counts.unsqueeze(1)

            # ts 聚合，这里选“最大 ts”
            max_ts = torch.full(
                (n_nodes,), ts.new_tensor(float("-inf")), device=device, dtype=ts.dtype
            )
            # 为了获得每节点最大值，用排序或分组最大：
            # 排序后取最后一个（stable 排序确保最后一个真的是最大 ts）
            sort_ts_idx = torch.argsort(ts)  # 升序
            node_sorted = node[sort_ts_idx]
            ts_sorted = ts[sort_ts_idx]
            # 最后一次出现的就是最大 ts
            # unique_consecutive 返回每段首索引，因此我们用一个小技巧反向取最后索引
            # 反向视图：
            rev_nodes = torch.flip(node_sorted, dims=[0])
            rev_ts = torch.flip(ts_sorted, dims=[0])
            uniq_nodes, idx_first = torch.unique_consecutive(
                rev_nodes,
                return_inverse=False,
                return_counts=False,
                return_indices=True,
            )
            # idx_first 给出 rev 中每段第一个（原序最后一个），把它映射回正序索引
            last_pos = rev_nodes.size(0) - 1 - idx_first
            # 为所有节点初始化为 -inf
            max_ts[:] = float("-inf")
            # 只更新出现过的节点
            max_ts[uniq_nodes] = ts_sorted[last_pos]

            # 写入 K=1 的信箱
            new_payloads = means.unsqueeze(1)  # [n_nodes, 1, D]
            new_ts = max_ts.unsqueeze(1)  # [n_nodes, 1]
            valid = (counts > 0).long()  # [n_nodes]
            return new_payloads, new_ts, valid

        else:
            # --- 每节点取最新一封（ts 最大）---
            # 先按 ts 降序，再按 node 稳定分组，取每组第一条
            # 步骤：先按 ts 降序排序；再 stable=True 按 node 升序排序，这样同一节点内的 ts 仍降序且相邻
            idx_ts_desc = torch.argsort(ts, descending=True, stable=True)
            node_d = node[idx_ts_desc]
            pay_d = payload[idx_ts_desc]
            ts_d = ts[idx_ts_desc]

            idx_node = torch.argsort(
                node_d, stable=True
            )  # 按节点分组（稳定，不破坏组内 ts 降序）
            node_g = node_d[idx_node]
            pay_g = pay_d[idx_node]
            ts_g = ts_d[idx_node]

            # 取每个节点的组内第一条（即最新）
            uniq_nodes, first_idx = torch.unique_consecutive(
                node_g, return_inverse=False, return_counts=False, return_indices=True
            )
            newest_payload = pay_g[first_idx]  # [num_nodes_hit, D]
            newest_ts = ts_g[first_idx]  # [num_nodes_hit]

            # 写回到 K=1 的信箱位置 0
            new_payloads = mailbox_payloads.clone()
            new_ts = mailbox_ts.clone()
            new_payloads[uniq_nodes, 0, :] = newest_payload
            new_ts[uniq_nodes, 0] = newest_ts

            valid = torch.zeros((n_nodes,), dtype=torch.long, device=device)
            valid[uniq_nodes] = 1
            return new_payloads, new_ts, valid

    # --------------------
    # K > 1: 为每个节点保留最新的 K 封
    # --------------------
    # 关键：构造 “按节点升序 + 组内 ts 降序” 的全局排序，然后每组取前 K 条。
    idx_ts_desc = torch.argsort(ts, descending=True, stable=True)
    node_d = node[idx_ts_desc]
    pay_d = payload[idx_ts_desc]
    ts_d = ts[idx_ts_desc]

    idx_node = torch.argsort(node_d, stable=True)  # 先把相同节点放在一起，保持 ts 降序
    node_g = node_d[idx_node]
    pay_g = pay_d[idx_node]
    ts_g = ts_d[idx_node]

    # 计算每个节点有多少封
    counts = torch.bincount(node_g, minlength=n_nodes)  # [n_nodes]
    take = torch.minimum(counts, torch.full_like(counts, K))  # 实际写入数（<=K）

    # 构造每组的偏移量（前缀和）
    offsets = torch.cumsum(counts, dim=0)  # [n_nodes]
    starts = torch.roll(offsets, 1)
    starts[0] = 0  # 每个节点组在 node_g/pay_g/ts_g 中的起始下标

    # 预分配输出
    new_payloads = mailbox_payloads.new_zeros((n_nodes, K, D))
    new_ts = mailbox_ts.new_full((n_nodes, K), float("-inf"))

    # 我们需要一次性索引把每个节点“前 take[i] 条”放入 [i, :take[i], :]
    # 先为所有节点生成列索引（0..K-1），再用掩码过滤掉超过 take 的部分
    col_idx = (
        torch.arange(K, device=device).unsqueeze(0).expand(n_nodes, K)
    )  # [n_nodes, K]
    take_exp = take.unsqueeze(1).expand_as(col_idx)  # [n_nodes, K]
    mask_place = col_idx < take_exp  # True 表示需要填的位置

    # 目标索引
    row_idx_flat = torch.nonzero(mask_place, as_tuple=False)[
        :, 0
    ]  # [T] 每个需要填的位置的节点索引
    col_idx_flat = torch.nonzero(mask_place, as_tuple=False)[
        :, 1
    ]  # [T] 每个需要填的位置的列索引(0..K-1)

    # 源索引：对于节点 i，第 j 个条目对应全局源下标 = starts[i] + j
    src_flat = starts[row_idx_flat] + col_idx_flat  # [T]

    # 从分组后的 pay_g/ts_g 中取数据
    new_payloads[row_idx_flat, col_idx_flat, :] = pay_g[src_flat]
    new_ts[row_idx_flat, col_idx_flat] = ts_g[src_flat]

    return new_payloads, new_ts, take


def send_to_neighbor(new_payloads, new_payloads_id, new_ts, take, edges):
    """
    new_payloads: for each interactived interaction between a pair of node(A and B for example), there is a mail generated on both nodes(A and B)
    new_payloads_id: this is same length with new_payloads, for each new_payloads_id[i], there was a new_payloads[i] generate on it
    new_ts: also corrsponeding with new_payloads
    edges: a tensor of size [2, n_edges], this forms a mail deliver network, each src deilivery its mails to the dst nodes

    return:
    mail_box: a tensor of size (n_unique_dst, mailbox_size, mail_dim)
    mail_box_corresponding_node_id: a tensor of size (n_unique_dst)
    mail_ts: a tensor of size (n_unique_dst, mailbox_size)
    """
    pass


def send_to_self(new_payloads, new_payloads_id, new_ts, take):
    """
    return:
    mail_box: a tensor of size (n_unique_dst, mailbox_size, mail_dim): just new_payloads
    mail_box_corresponding_node_id: a tensor of size (n_unique_dst): just new_payloads_id
    mail_ts: a tensor of size (n_unique_dst, mailbox_size): just new_ts
    """
    pass
