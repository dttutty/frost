# worker/trainer.py

from typing import Callable, Dict, List

import torch
from torch_scatter import scatter_max

from schemas.datatypes import MemTsNid
from worker.h2d import Meta


Hook = Callable[..., None]


class HookManager:
    def __init__(self):
        self._hooks: Dict[str, List[Hook]] = {
            "b_inrank": [],
            "b_crrank": [],
            "a_crrank": [],
        }

    def register(self, name: str, func: Hook):
        if name not in self._hooks:
            raise KeyError(f"Unknown hook point: {name}")
        self._hooks[name].append(func)

    def emit(self, name: str, *args, **kwargs):
        for func in list(self._hooks.get(name, [])):
            func(*args, **kwargs)


hook_mgr = HookManager()


class MMFN:
    def __init__(
        self,
        next_mail_q,
        next_mem_q,
        mm,
        mu,
        dedup,
        deliver_to,
        shared_mem,
        gate,
        env,
        mem_strategy,
        mail_strategy,
        boxsize,
        world_size,
        neg_samples,
        merge_src,
    ) -> None:
        self.mail_q = next_mail_q
        self.mem_q = next_mem_q
        self.mm = mm
        self.mu = mu
        self.dedup = dedup
        self.deliver_to = deliver_to
        self.gate = gate
        self.shared_mem = shared_mem
        self.env = env
        self.mem_strategy = mem_strategy
        self.mail_strategy = mail_strategy
        self.boxsize = boxsize
        self.env = env
        self.world_size = world_size
        self.neg_samples = neg_samples
        self.merge_src = merge_src
        pass

    @torch.no_grad()
    def process(
        self,
        td: Meta,
        blocks,
    ):
        # 需要更新mem的节点:发生interaction的root节点
        # 不需要更新mem的节点,其他节点
        dst = td.next_all_nodes  # 如果是latest,必须已经计算好scatter计划
        mu_efeats = self._efeat[td.pos_edge_id] if self._efeat is not None else None
        mid = td.macro_bid
        # block = td.block
        last_updated_nid = self.mu.last_updated_nid.clone()
        last_updated_memory = self.mu.last_updated_memory.clone()
        last_updated_ts = self.mu.last_updated_ts.clone()

        # if self.mm is not None:
        # 可选将写入张量转为 CPU，视你的 MemoryMailbox 实现而定

        # this batch = 16
        # tcpStore.set('b/mid16/gpu_mem_ok', 1) # 18 training is partily good to go
        # tcpStore.set('b/mid16/write_to_cpu_ok', 1) # 20 data mover is blocked
        # write memory and mail not gonna be used in next train step in cpu

        # 1) 先写 mailbox（mail + memory）
        #    注意：根据你的 MemoryMailbox 依赖关系，若需要先写 memory 再写 mail，可对调顺序
        n_src = td.roots.shape[0] // (self.neg_samples + 2)
        pos_root_nid = last_updated_nid.long()[: n_src * 2]
        pos_root_new_mem = last_updated_memory[: n_src * 2]
        pos_root_new_mem_ts = last_updated_ts[: n_src * 2]

        mem_td_nid = MemTsNid(
            id=pos_root_nid,
            ts=pos_root_new_mem_ts,
            data=pos_root_new_mem,  # 如果是latest, 需要改成所有节点的mem(pos_root用新mem,)
        )

        if self.deliver_to == "self":
            td.mail_next_batch_root_nodes = td.next_batch_root_nodes
        if boxsize == 1:
            assert self.deliver_to == "self", (
                "boxsize<=world_size only support deliver_to=self, otherwise slots is not enough"
            )
            if not merge_src:
                hook_mgr.register("b_inrank", combine_mail_within_gpu, mail_strategy)
                # each interaction has its own memory, so each interaction has its own mail
            hook_mgr.register("b_crrank", fill_mailbox_acorss_gpu, mail_strategy)
        elif boxsize > world_size:  # this gonna keep as much mail as possible
            assert not merge_src, (
                "you are wasting mailbox slots, please make sure boxsize <= world_size when merge_src==True"
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
            # if not merge_src:
            #     mem_td_nid = combine_within_gpu(
            #         mem_td_nid, "latest"
            #     )  # each gpu keep the latest mem of node[local_rank % word_size]
            # next_mem_td_nid = direct_scatter_and_save_to_cpu(
            #     mem_td_nid,
            #     self.mem_strategy,
            #     self.env,
            #     td.next_batch_root_nodes,
            #     self.shared_mem,
            # )
            # hook_mgr.emit("b_crrank")
            send_latest_across_rank(
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
            mem_td_nid = scatter_by_remainder(mem_td_nid, strategy, env)
            mem_td_nid = combine_within_gpu(mem_td_nid, strategy)
            next_mem_td_nid = scatter_and_gather_again_and_save_to_cpu(
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

        memhandler = self.write_node_memory(self.dedup, mem_td_nid, n_src)

        boxhandler = self.write_mail_to_mailbox(
            mem_td_nid, mu_efeats, td.mdb_raw_data, neg_samples
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

    def update_next_mail_pos(self):
        mm: MemoryMailbox = self.mm
        if mm.mail_pos_update_pending_flag is not None:
            nid = torch.where(mm.mail_pos_update_pending_flag == 1)[0]
            mm.next_mail_pos[nid] = (
                mm.next_mail_pos[nid] + 1
            ) % mm.memory_param.boxsize
            mm.mail_pos_update_pending_flag.fill_(0)

    @torch.no_grad()
    def write_node_memory(self, dedup, mem_td_nid, n_src):
        pos_root_nid, pos_root_new_mem, pos_root_new_mem_ts = (
            mem_td_nid.nid,
            mem_td_nid.data,
            mem_td_nid.ts,
        )
        mm = self.mm
        if pos_root_nid is None:
            return

        idx = torch.arange(pos_root_nid.size(0), device=pos_root_nid.device)

        if dedup:
            _, last_occurrence_idx = scatter_max(idx, pos_root_nid)
            uniq_nid = torch.unique(pos_root_nid)
            selected_idx = last_occurrence_idx[uniq_nid]
            mm.node_memory[uniq_nid] = pos_root_new_mem[selected_idx].to(mm.device)
            mm.node_memory_ts[uniq_nid] = pos_root_new_mem_ts[selected_idx].to(
                mm.device
            )
        else:
            mm.node_memory[pos_root_nid.long()] = (
                pos_root_new_mem  # 得改,应该只有一部分需要保存
            )
            mm.node_memory_ts[pos_root_nid.long()] = (
                pos_root_new_mem_ts  # 得改,应该只有一部分需要保存
            )

        send_to_neighbor()
        send_to_self()
        route_to_owner()
        aggregate_owner_mean()
        aggregate_owner_latest()
        build_owner_mailbox()
        split_data_to_tensors_send_to_self()
        split_data_to_tensors_send_to_neigh()
        split_data_to_tensors()
        build_plan_on_rank0()
        scatter_plan()
        exchange_features_with_all_to_all()

        compute_what_gonna_be_used_in_next_batch()
        need_fetch_from_cpu, can_fetch_from_other_gpu = compute_what_need_to_be_fetch()

    @torch.no_grad()
    def write_mail_to_mailbox(
        self,
        mem_td_nid: MemTsNid,
        edge_feats,
        mail_delivery_block,
        neg_samples,
    ):
        pos_root_nid, pos_root_new_mem, pos_root_new_mem_ts = (
            mem_td_nid.id,
            mem_td_nid.mem,
            mem_td_nid.ts,
        )
        mm: MemoryMailbox = self.mm
        # print(f"[Mailbox Update]Batch: {batch}")
        # before update mailbox, we need to prepare the memory
        # e.g. number_edges(5,6) = num_src(1,2) = num_src(3,4) = 2, mem_dim = 2 edge_dim = 3
        n_src = root_nodes.shape[0] // (neg_samples + 2)

        # move data to device
        # memory = memory.to(mm.device)
        if edge_feats is not None:
            edge_feats = edge_feats.to(mm.device)
        if mail_delivery_block is not None:
            mail_delivery_block = mail_delivery_block.to(mm.device)

        src_mail, dst_mail = _concat_src_dst_memory(pos_root_new_mem, n_src, edge_feats)
        mail, mail_ts, unique_node_ids, inverse_indices = _prepare_mail_and_node_ids(
            root_nodes,
            pos_root_new_mem_ts,
            n_src,
            mail_delivery_block,
            src_mail,
            dst_mail,
        )
        mail, mail_ts = _combine_mail_mean_last(
            mail, mail_ts, inverse_indices, unique_node_ids
        )

        # now update the mailbox
        # every nodes can have as many mail positions(slots) as they want, typically 1 or 10
        idx = unique_node_ids.long()

        mm.mails[idx, mm.next_mail_pos[idx]] = mail
        mm.mail_ts[idx, mm.next_mail_pos[idx]] = mail_ts
        if mm.memory_param.boxsize > 1:
            if mm.mail_pos_update_pending_flag is not None:
                mm.mail_pos_update_pending_flag[idx] = 1
            else:
                pos_root_nid = idx
                mm.next_mail_pos[pos_root_nid] = (
                    mm.next_mail_pos[pos_root_nid] + 1
                ) % mm.memory_param.boxsize

        generate_mail_vectorized(
            pos_root_nid, new_mem, new_mem_ts, pe, pe_ts, pe_feat, bs
        )
        update_mailbox()
        send_to_neighbor()
        send_to_self()
        route_to_owner()
        aggregate_owner_mean()
        aggregate_owner_latest()
        build_owner_mailbox()
        split_data_to_tensors_send_to_self()
        split_data_to_tensors_send_to_neigh()
        split_data_to_tensors()
        build_plan_on_rank0()
        scatter_plan()
        exchange_features_with_all_to_all()
