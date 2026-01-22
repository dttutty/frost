import threading
from queue import Queue
from typing import List, TypeVar

import torch
from numpy.typing import NDArray
from torch_scatter import scatter_sum

from schemas.datatypes.cfg import Config
from schemas.datatypes.data_to_sync import BoxTsNid, MailTsNid, MemTsNid
from schemas.datatypes.env import Env
from schemas.datatypes.misc import Shared
from schemas.datatypes.task import (
    BoxTask,
    GatherElement,
    MemTask,
    P2PTask,
    SyncTask,
)


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")


def compute_latest_indices(nid: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """
    通用逻辑：计算去重后应当保留的索引。
    策略：按 nid 分组，保留 ts 最大的那一行。
    返回：indices_to_keep
    """
    if nid.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=nid.device)

    # 1. 双重排序 (Sort by TS desc -> Stable sort by NID)
    perm_ts = torch.argsort(ts, descending=True)
    # stable=True 保证相同 NID 内部，TS 大的排在前面
    perm_nid = torch.argsort(nid[perm_ts], stable=True)

    final_perm = perm_ts[perm_nid]
    sorted_nid = nid[final_perm]

    # 2. 构造掩码 (Keep first of each group)
    mask = torch.empty(sorted_nid.shape[0], dtype=torch.bool, device=nid.device)
    mask[0] = True
    mask[1:] = sorted_nid[1:] != sorted_nid[:-1]

    # 3. 返回保留的索引
    return final_perm[mask]


def keep_latest(data: Mem_or_Mail):
    assert data.ts is not None, "TS must be provided for keep_latest"
    indices = compute_latest_indices(data.nid, data.ts)

    # 2. 切片更新自己特有的字段
    if indices.numel() < data.nid.numel():  # 只有确实有重复才更新，微小优化
        data.nid = data.nid[indices]
        data.data = data.data[indices]  # MemTsNid 特有的字段
        data.ts = data.ts[indices]

        if data.ct is not None:
            data.ct = data.ct[indices]  # MailTsNid 特有的字段


def keep_mean(data: Mem_or_Mail):
    """
    聚合策略：对于重复的 nid，计算 data 和 ts 的加权均值。
    同时：
    1. 维护 ct (计数)：如果是首次合并，默认原权重为1；如果有旧权重，则累加。
    2. 维护 ts：计算加权平均时间 (Weighted Mean Time)。
    实现：原地修改。
    """
    assert data.ts is not None, "TS must be provided for keep_mean"

    if data.nid.numel() == 0:
        return

    # 1. ID 映射 (Mapping)
    unique_nids, compact_idx = torch.unique(data.nid, return_inverse=True)
    M = unique_nids.size(0)

    # 2. 处理权重 (Weights)
    if data.ct is None:
        weights = torch.ones_like(data.ts, dtype=torch.float32)
    else:
        weights = data.ct.float()

    # 3. 聚合计数 (Sum Weights) -> 分母
    # [M]
    new_ct = scatter_sum(weights, compact_idx, dim=0, dim_size=M)

    # 4. 聚合数据 (Weighted Sum Data) -> 分子 1
    if data.ct is not None:
        weighted_data = data.data * weights.unsqueeze(-1)
    else:
        weighted_data = data.data

    sum_data = scatter_sum(weighted_data, compact_idx, dim=0, dim_size=M)

    # 5. 聚合时间 (Weighted Sum TS) -> 分子 2  <--- 修改重点在这里
    # 时间也要乘权重 (例如：5条消息在 t=10, 1条消息在 t=20，均值应偏向 10)
    weighted_ts = data.ts * weights
    sum_ts = scatter_sum(weighted_ts, compact_idx, dim=0, dim_size=M)

    # 6. 计算均值 (Divide)
    # 注意: new_ct 至少为 1，除法安全
    new_data = sum_data / new_ct.unsqueeze(-1)
    new_ts = sum_ts / new_ct  # [M] / [M] -> [M]

    # 7. 更新自身
    data.nid = unique_nids
    data.data = new_data
    data.ts = new_ts
    data.ct = new_ct.long()


def merge_with_priority(
    high_priority_data: Mem_or_Mail, low_priority_data: Mem_or_Mail
) -> Mem_or_Mail:
    """
    合并 low_priority_data 和 high_priority_data。
    规则：优先保留 high_priority_data 的数据。如果 nid 在 high_priority_data 中存在，则忽略 high_priority_data 中的对应行。
    """
    # 1. 判空处理：如果 B 是空的，直接返回 A
    if low_priority_data.nid.numel() == 0:
        return type(high_priority_data)(
            id=high_priority_data.nid,
            data=high_priority_data.data,
            ts=high_priority_data.ts,
        )
    # 如果 A 是空的，直接返回 B
    if high_priority_data.nid.numel() == 0:
        return type(low_priority_data)(
            id=low_priority_data.nid,
            data=low_priority_data.data,
            ts=low_priority_data.ts,
        )

    # -------------------------------------------------------
    # 核心逻辑: 使用 torch.isin 找出 B 中冲突的节点
    # -------------------------------------------------------
    # mask 为 True 的位置表示：B 中的这个 nid 也在 A 中出现了
    # (注意：self.nid 不需要去重，isin 会自动处理)
    mask_in_high = torch.isin(low_priority_data.nid, high_priority_data.nid)

    # 我们只保留那些 mask 为 False (即不在 A 中) 的数据
    mask_keep_B = ~mask_in_high

    # 2. 对 B 进行过滤
    b_nid_filtered = low_priority_data.nid[mask_keep_B]
    b_mem_filtered = low_priority_data.data[mask_keep_B]
    b_ts_filtered = low_priority_data.ts[mask_keep_B]

    new_ct = None
    if high_priority_data.ct is not None and low_priority_data.ct is not None:
        b_ct_filtered = low_priority_data.ct[mask_keep_B]
        new_ct = torch.cat([high_priority_data.ct, b_ct_filtered], dim=0)

    return type(high_priority_data)(
        id=torch.cat([high_priority_data.nid, b_nid_filtered], dim=0),
        data=torch.cat([high_priority_data.data, b_mem_filtered], dim=0),
        ts=torch.cat([high_priority_data.ts, b_ts_filtered], dim=0),
        ct=new_ct,
    )


class DataSyncer(threading.Thread):
    def __init__(
        self,
        cfg: Config,
        env: Env,
        shared: Shared,
        h2d_sem: threading.Semaphore,
        sync_q: Queue[SyncTask],
        stop_ev: threading.Event,
        mem_p2p_q: Queue,
        box_p2p_q: Queue,
    ):
        super().__init__(daemon=True, name="data_syncer")
        env = env
        self.n_gpu = env.n_gpu
        self.shared = shared
        self.sync_q = sync_q
        # 将原 GNN init 中关于 mailbox 和 deliver 的配置移到这里
        assert cfg.m is not None, "Mailbox config (cfg.m) must be provided"
        self.deliver_to = cfg.m.deliver_to
        self.mail_combine = cfg.m.mail_combine
        self.boxsize = cfg.m.boxsize
        self.merge_src = cfg.t.merge_src
        self.h2d_sem = h2d_sem
        self.stream = torch.cuda.Stream()
        self.mem_stream = torch.cuda.Stream()
        self.box_stream = torch.cuda.Stream()
        self.stop_ev = stop_ev
        self.mem_p2p_q = mem_p2p_q
        self.box_p2p_q = box_p2p_q
        self.mem_task_q: Queue[MemTask] = Queue()
        self.box_task_q: Queue[BoxTask] = Queue()
        self.box_post_task_q: Queue[BoxTask] = Queue()
        self.mem_handler = threading.Thread(target=self.mem_handler, daemon=True)
        self.mem_handler.start()
        self.box_handler = threading.Thread(target=self.box_handler, daemon=True)
        self.box_handler.start()

    @torch.no_grad()
    def mem_handler(self):
        def write_back_mem(writeback, mem_ts_nid: MemTsNid):
            # write back mem
            if mem_ts_nid is not None and writeback is not None:
                mask = torch.isin(mem_ts_nid.nid, writeback)
                write_back_mem = MemTsNid(
                    nid=mem_ts_nid.nid[mask],
                    data=mem_ts_nid.data[mask],
                    ts=mem_ts_nid.ts[mask],
                )
            assert self.shared is not None
            id = write_back_mem.nid.cpu().long()
            self.shared.mem[id] = write_back_mem.data.cpu()
            self.shared.mem_ts[id] = write_back_mem.ts.cpu()

        with torch.cuda.stream(self.mem_stream):
            while True:
                task: MemTask = self.mem_task_q.get(block=True)
                (
                    root_new_mem,
                    old_mem_ts_nid,
                    plan,
                    is_last,
                ) = (
                    task.root_new_mem,
                    task.old_mem_ts_nid,
                    task.plan,
                    task.is_last,
                )

                keep_latest(root_new_mem)

                mem_ts_nid = merge_with_priority(root_new_mem, old_mem_ts_nid)
                if is_last is not None:
                    self.p2p(
                        mem_ts_nid,
                        plan.p2p,
                        plan.rec_cts,
                        plan.send_cts,
                        "latest",
                    )
                else:
                    self.mem_p2p_q.put(
                        None
                    )  # skip p2p(), directly put None to p2p_q as sentinel

                write_back_mem(
                    plan.writeback,
                    mem_ts_nid,
                )
                self.h2d_sem.release()

    @torch.no_grad()
    def box_handler(self):
        def write_back_box(writeback, box_ts_nid: BoxTsNid):
            if box_ts_nid is not None and writeback is not None:
                return
            mask = torch.isin(box_ts_nid.nid, writeback)
            write_back_box = BoxTsNid(
                nid=box_ts_nid.nid[mask],
                data=box_ts_nid.data[mask],
                ts=box_ts_nid.ts[mask],
            )

            id = write_back_box.nid.cpu().long()

            self.shared.box[id] = write_back_box.data.cpu()
            self.shared.box_ts[id] = write_back_box.ts.cpu()

            if write_back_box.next_pos is not None:
                assert self.shared.box_next_pos is not None
                self.shared.box_next_pos[id] = write_back_box.next_pos.cpu()

        with torch.cuda.stream(self.stream):
            while True:
                task: BoxTask = self.box_task_q.get(block=True)
                (
                    root_new_mem_orig,
                    is_last,
                    macro_bid,
                    box,
                    pos_efeat,
                    mail_delivery_tensor,
                    plan,
                ) = (
                    task.root_new_mem_orig,
                    task.is_last,
                    task.macro_bid,
                    task.old_box,
                    task.pos_efeat,
                    task.mail_delivery_tensor,
                    task.plan,
                )

                # handle mail or box sync
                # generate new mails from root_new_mem
                print("DataSyncer: generating new mails", macro_bid)
                new_mail = generate_interaction_mails(
                    root_new_mem_orig,
                    pos_efeat,
                )
                mail_combine = self.mail_combine
                print("DataSyncer: start p2p for mail/box", macro_bid)
                if self.boxsize == 1:
                    # handle mail sync
                    # if deliver_to is 'self', only deliver to self
                    assert self.deliver_to == "self"
                    old_mail = box.to_mail()

                    if mail_combine == "last":
                        keep_latest(new_mail)
                    else:
                        assert mail_combine == "mean"
                        keep_mean(new_mail)
                    new_mail = merge_with_priority(new_mail, old_mail)
                    print(
                        "DataSyncer: merged new mail with old mail",
                        macro_bid,
                    )

                    if is_last is not None:
                        self.p2p(
                            new_mail,
                            plan.p2p,
                            plan.rec_cts,
                            plan.send_cts,
                            self.mail_combine,
                        )
                    else:
                        pass  # this case is handler by mem_handler, mem_handler will put None to p2p_q

                    box_ts_nid = new_mail.to_box()
                else:
                    # handle box sync
                    # if deliver_to is 'neighbors', deliver to neighbors using deliver()
                    assert self.boxsize > 1
                    assert self.deliver_to == "neighbors"
                    assert mail_delivery_tensor is not None
                    assert not self.merge_src, (
                        "you are wasting mailbox slots, please make sure boxsize <= n_gpus when merge_src==True"
                    )
                    delivered_mails = new_mail.deliver(
                        mail_delivery_tensor, include_self=True
                    )
                    new_mail = delivered_mails
                    mail_combine = self.mail_combine

                    if mail_combine == "last":
                        keep_latest(new_mail)
                        box_ts_nid = box.merge_1_mail(new_mail)
                    elif mail_combine == "mean":
                        keep_mean(new_mail)
                        box_ts_nid = box.merge_1_mail(new_mail)
                    elif mail_combine == "all":
                        box_ts_nid = box.merge_n_mail(new_mail)
                    if is_last is not None:
                        self.p2p(
                            box_ts_nid,
                            plan.p2p,
                            plan.rec_cts,
                            plan.send_cts,
                            "latest",
                        )
                    else:
                        pass  # this case is handler by mem_handler, mem_handler will put None to p2p_q

                print("DataSyncer: finished p2p processing", macro_bid)

                write_back_box(
                    plan.writeback,
                    box_ts_nid,
                )
                print("DataSyncer: finished write back", macro_bid)
                self.h2d_sem.release()

    @torch.no_grad()
    def logic(self):
        task: SyncTask = self.sync_q.get(block=True)
        self.work_handles = []
        (
            macro_bid,
            plan_set,
            root_new_mem,
            x,
            mail_delivery_tensor,
            is_last,
            ready_event,
        ) = (
            task.macro_bid,
            task.plan_set,
            task.root_new_mem,
            task.x,
            task.mail_delivery_tensor,
            task.is_last,
            task.ready_event,
        )

        torch.cuda.current_stream().wait_event(ready_event)
        print(
            "DataSyncer: ready_event received, processing DyX",
            macro_bid,
        )

        root_new_mem_orig = root_new_mem.clone()  # 这里的ts是交互发生的时间

        assert x.mem is not None
        assert x.box is not None
        # handle mem sync
        old_mem_ts_nid = x.mem
        self.mem_task_q.put(
            MemTask(
                root_new_mem,
                old_mem_ts_nid,
                plan_set.mem,
                is_last,
            )
        )
        assert x.pos_efeat is not None
        self.box_task_q.put(
            BoxTask(
                root_new_mem_orig,
                x.pos_efeat,
                macro_bid,
                x.box,
                plan_set.box,
                is_last,
                mail_delivery_tensor,
            )
        )

    @torch.no_grad()
    def run(self):
        with torch.cuda.stream(self.stream):
            while True:
                self.logic()

    def p2p(
        self,
        data_td_nid: AllType,
        p2p: List[torch.Tensor],
        rec_cts: NDArray,
        send_cts: NDArray,
        strategy,
    ):
        """
        这一步之前,必须确保data_td_nid里面的数据全部是unique的
        如果是mem,需要确保已经是保留最新的(latest),或者平均的(mean,带ct)
        如果是mail,需要确保只保留更新过的节点(pos_src,pos_dst).....另外保留最新latest,平均的(mean,带ct),或者全部(dim变大)
        如果是mailbox,需要所有节点的mailbox
        如果是feat,需要所有节点的feat
        """
        device = "cuda"
        requests = {}

        nid = data_td_nid.nid
        data = data_td_nid.data
        ts = data_td_nid.ts
        has_ct = strategy == "mean"
        if has_ct:
            ct = data_td_nid.ct
        x_dim = data.shape[1]

        buckets_ids, buckets_x, buckets_ts, buckets_ct = [], [], [], []

        for dst_gpu_id in range(self.n_gpu):
            nids_to_send_tensor = p2p[dst_gpu_id]
            send_mask_for_dst = torch.isin(nid, nids_to_send_tensor)

            curr_ids = nid[send_mask_for_dst]
            buckets_ids.append(nid[send_mask_for_dst])
            buckets_x.append(data[send_mask_for_dst])
            buckets_ts.append(ts[send_mask_for_dst])
            if has_ct:  # "mail"才有ct
                buckets_ct.append(ct[send_mask_for_dst])

            # # =========================================================
            # # CHECK 1: 本地一致性检查 (Local Consistency Check)
            # # 验证: 实际打包的 Tensor 长度 == send_cts 中承诺的长度
            # # 原因: 防止 Planner 计算有误，或者 torch.isin 过滤掉了意外缺失的节点
            # # =========================================================
            # actual_pack_len = curr_ids.shape[0]
            # expected_send_len = send_cts[dst_gpu_id]

            # if actual_pack_len != expected_send_len:
            #     raise RuntimeError(
            #         f"[Rank {dist.get_rank()}] Local Packing Mismatch for Dst {dst_gpu_id}!\n"
            #         f"  - Planner (send_cts) expected: {expected_send_len}\n"
            #         f"  - Packer (torch.isin) found:   {actual_pack_len}\n"
            #         f"  - Hint: Check if Planner included NIDs not present in local storage."
            #     )

        # # =========================================================
        # # CHECK 2: 全局一致性检查 (Global Consistency Check)
        # # 验证: 我这边填写的 rec_cts == 别的 GPU 填写的 send_cts
        # # 原因: 防止死锁。这是所有通信错误的根源。
        # # 开销: 极小 (仅交换 N 个整数)，建议在 Debug 模式开启，生产环境可视情况关闭
        # # =========================================================
        # if False:  # Set to False to disable in production for micro-optimization
        #     # 1. 准备数据: 我打算发给别人的数量
        #     my_send_counts = torch.tensor(send_cts, dtype=torch.long, device=device)
        #     # 2. 准备接收缓冲: 别人实际上打算发给我的数量
        #     actual_others_send_counts = torch.empty_like(my_send_counts)

        #     # 3. 交换情报 (All-to-All 交换整数)
        #     # 逻辑: Rank i 发送 my_send_counts[j] 给 Rank j
        #     #       Rank i 接收 actual_others_send_counts[j] 来自 Rank j
        #     dist.all_to_all_single(
        #         actual_others_send_counts, my_send_counts, group=self.nccl_sync
        #     )

        #     # 4. 比对: 我预期的 rec_cts 必须等于别人承诺的 actual_others_send_counts
        #     my_expected_rec_cts = torch.tensor(rec_cts, dtype=torch.long, device=device)

        #     if not torch.equal(my_expected_rec_cts, actual_others_send_counts):
        #         raise RuntimeError(
        #             f"[Rank {dist.get_rank()}] Global Protocol Mismatch!\n"
        #             f"  - I expect to receive (rec_cts): {my_expected_rec_cts.tolist()}\n"
        #             f"  - Others plan to send (actual):  {actual_others_send_counts.tolist()}\n"
        #             f"  - Mismatch will cause NCCL Hang."
        #         )

        send_nid = torch.cat(buckets_ids, dim=0)
        send_x = torch.cat(buckets_x, dim=0)
        send_ts = torch.cat(buckets_ts, dim=0)
        if has_ct:
            send_ct = torch.cat(buckets_ct, dim=0)

        total_recv = sum(rec_cts)

        recv_nid = torch.empty(total_recv, dtype=nid.dtype, device=device)
        recv_data = torch.empty(total_recv, x_dim, dtype=data.dtype, device=device)
        recv_ts = torch.empty(total_recv, dtype=ts.dtype, device=device)
        if has_ct:  # "mail"才有ct
            recv_ct = torch.empty(total_recv, dtype=ct.dtype, device=device)

        in_splits = send_cts
        out_splits = rec_cts
        p2p_nid = GatherElement(recv_nid, send_nid, out_splits, in_splits)
        p2p_data = GatherElement(recv_data, send_x, out_splits, in_splits)
        p2p_ts = GatherElement(recv_ts, send_ts, out_splits, in_splits)

        if has_ct:  # "mail"才有ct
            p2p_ct = GatherElement(recv_ct, send_ct, out_splits, in_splits)
        match data_td_nid:
            case MemTsNid():
                p2p_type = "mem"
                self.mem_p2p_q.put(
                    P2PTask(
                        nid=p2p_nid,
                        data=p2p_data,
                        ts=p2p_ts,
                        type=p2p_type,
                    )
                )
            case MailTsNid():
                p2p_type = "mail"
                self.box_p2p_q.put(
                    P2PTask(
                        nid=p2p_nid,
                        data=p2p_data,
                        ts=p2p_ts,
                        type=p2p_type,
                    )
                )
            case BoxTsNid():
                p2p_type = "box"
                self.box_p2p_q.put(
                    P2PTask(
                        nid=p2p_nid,
                        data=p2p_data,
                        ts=p2p_ts,
                        type=p2p_type,
                    )
                )
            case _:
                raise ValueError(f"Unknown data_td_nid type: {type(data_td_nid)}")

        # work_handles = self.result_queue.get()
        # for work in work_handles:
        #     work.wait()

        # match data_td_nid:
        #     case MemTsNid():
        #         return MemTsNid(id=recv_nid, data=recv_data, ts=recv_ts)
        #     case MailTsNid():
        #         return MailTsNid(
        #             id=recv_nid,
        #             data=recv_data,
        #             ts=recv_ts,
        #             ct=recv_ct if strategy == "mean" else None,
        #         )
        #     case BoxTsNid():
        #         return BoxTsNid(
        #             id=recv_nid,
        #             data=recv_data,
        #             ts=recv_ts,
        #             ct=recv_ct if strategy == "mean" else None,
        #         )
        #     case _:
        #         raise ValueError(f"Unknown data_td_nid type: {type(data_td_nid)}")


def generate_interaction_mails(
    root_new_mem_orig: MemTsNid, edge_feats=None
) -> MailTsNid:
    """
    Args:
        root_new_mem_orig: MemTsNid 对象 (包含 2N 个节点)
                           前 N 个是 Src，后 N 个是 Dst。
                           mem 形状: [2N, mem_dim]
        edge_feats:        边的特征，形状 [N, edge_dim]
                           对应着 Src[i] -> Dst[i] 的边
    Returns:
        all_mails:         形状 [2N, 2 * mem_dim + edge_dim]
                           对应 root_new_mem_orig 中每个节点的 mail
    """
    # 1. 获取 N 的大小
    total_nodes = root_new_mem_orig.nid.shape[0]
    n = total_nodes // 2

    # 2. 切分 Src 和 Dst 的 Memory
    # root_new_mem_orig.mem 的结构是 [Src_Mem ... | Dst_Mem ...]
    src_mem = root_new_mem_orig.data[:n]  # Self Memory for Src Nodes
    dst_mem = root_new_mem_orig.data[n:]  # Other Memory for Src Nodes (Self for Dst)

    # 3. 生成给 Src 节点的 Mail
    # 逻辑: [自己, 对方, 边] -> [Src, Dst, Edge]
    if edge_feats is not None:
        mail_for_src = torch.cat([src_mem, dst_mem, edge_feats], dim=1)

        # 4. 生成给 Dst 节点的 Mail
        # 逻辑: [自己, 对方, 边] -> [Dst, Src, Edge]
        # 注意：通常这里要交换 src/dst mem 的位置，保证模型看到的第一部分永远是"自己"
        mail_for_dst = torch.cat([dst_mem, src_mem, edge_feats], dim=1)
    else:
        mail_for_src = torch.cat([src_mem, dst_mem], dim=1)

        # 4. 生成给 Dst 节点的 Mail
        # 逻辑: [自己, 对方, 边] -> [Dst, Src, Edge]
        # 注意：通常这里要交换 src/dst mem 的位置，保证模型看到的第一部分永远是"自己"
        mail_for_dst = torch.cat([dst_mem, src_mem], dim=1)
        # 5. 拼接回 2N 长度，保持顺序一致 (前N个是Src的mail，后N个是Dst的mail)
    all_mails = torch.cat([mail_for_src, mail_for_dst], dim=0)

    mail_ts_nid = MailTsNid(
        nid=root_new_mem_orig.nid,
        data=all_mails,
        ts=root_new_mem_orig.ts,  # 这里ts的len正好=2N
    )
    return mail_ts_nid
