import torch
import dgl
from layers import TimeEncode
from torch_scatter import scatter
from dataclasses import dataclass, field
from typing import Optional, Literal
from profiler import profiler, Timer
from torch_scatter import scatter_max
@dataclass
class MemoryParams:
    type: Optional[Literal["node"]] = "node"
    dim_time: Optional[int] = None # time dim
    deliver_to: Optional[Literal["self", "neighbors"]] = "self"
    mail_combine: Optional[Literal["mean", "last"]] = "last"
    memory_update: Optional[Literal["gru", "rnn", "transformer"]] = "gru"
    mailbox_size: Optional[int] = None
    combine_node_feature: bool = False
    attention_head: Optional[int] = None
    dim_out: Optional[int] = None # memory dim


# shared_cfg = {
#     'node_memory':      ([num_nodes, memory_param.dim_out], torch.float32),
#     'node_memory_ts':   ([num_nodes], torch.float32),
#     'mails':            ([num_nodes, memory_param.mailbox_size, 2 * memory_param.dim_out + edge_feat_dim], torch.float32),
#     'mail_ts':          ([num_nodes, memory_param.mailbox_size], torch.float32),
#     'next_mail_pos':    ([num_nodes], torch.long),
#     'mail_pos_update_pending_flag':  ([num_nodes], torch.int8),
# }
class MemoryMailbox():
    def __init__(self, 
                 memory_param: MemoryParams, 
                 shared_cfg, 
                 dim_edge_feat, 
                 shared_arrays = None):
        self.shared_cfg = shared_cfg
        self.memory_param = memory_param
        self.dim_edge_feat = dim_edge_feat
        
        if memory_param.type != 'node':
            raise NotImplementedError

        for config_key, (shape_definition, dtype_definition) in self.shared_cfg.items():
            if shared_arrays is None or shared_arrays[config_key] is None:
                if config_key == 'mail_pos_update_pending_flag':
                    setattr(self, config_key, None) # not torch.zeros(shape_definition, dtype=dtype_definition)
                else:
                    setattr(self, config_key, torch.zeros(shape_definition, dtype=dtype_definition))
            else:
                assert shared_arrays[config_key] is not None, f"Shared array {config_key} is None"
                setattr(self, config_key, shared_arrays[config_key])
                

        self.device = torch.device('cpu')
        
    def reset(self):
        for config_key, (shape_definition, dtype_definition) in self.shared_cfg.items():
            if config_key == 'mail_pos_update_pending_flag':
                continue
            shared_value = getattr(self, config_key)
            shared_value.zero_()

    def move_to_gpu(self):
        # only used in train.py and train_node.py
        for config_key, (shape_definition, dtype_definition) in self.shared_cfg.items():
            shared_value = getattr(self, config_key)
            if config_key == 'mail_pos_update_pending_flag':
                continue
            if shared_value is not None:
                setattr(self, config_key, shared_value.cuda())
        
        self.device = torch.device('cuda:0')


    
    def alc_pin_mem_mailbox_buf(self, deduplication, sample_param, batch_size):
        if not deduplication:
            limit_of_n_nodes = int(batch_size * 3.3)
            if sample_param.neighbor is not None:
                for i in sample_param.neighbor:
                    limit_of_n_nodes *= i + 1
            # e.g. neighbor = [10, 10] 
            # limit_of_n_nodes = 3.3 * batch_size * (10 + 1) * (10 + 1) = 3.3 * batch_size * 11 * 11
            # limit_of_n_nodes, i.e. the length of pinned_buffer[history_i], need to be larger than size of blocks[history_i].srcdata['ID']
            self.pinned_node_memory_buffs = list()
            self.pinned_node_memory_ts_buffs = list()
            self.pinned_mailbox_buffs = list()
            self.pinned_mailbox_ts_buffs = list()
            
            mem_dim = self.node_memory.shape[1]
            mailbox_size = self.mails.shape[1]
            mail_dim = self.mails.shape[2]
            
            for _ in range(sample_param.history):
                # if history = 2, we need 2 buffers for each type
                self.pinned_node_memory_buffs.append(torch.zeros([limit_of_n_nodes, mem_dim], pin_memory=True))
                self.pinned_node_memory_ts_buffs.append(torch.zeros([limit_of_n_nodes], pin_memory=True))
                self.pinned_mailbox_buffs.append(torch.zeros([limit_of_n_nodes, mailbox_size, mail_dim], pin_memory=True))
                self.pinned_mailbox_ts_buffs.append(torch.zeros([limit_of_n_nodes, mailbox_size], pin_memory=True))
        else:
            limit_of_n_nodes = int(batch_size * 3.3)
            if sample_param.neighbor is not None:
                for i in sample_param.neighbor:
                    limit_of_n_nodes *= i + 1
            # e.g. neighbor = [10, 10] 
            # limit_of_n_nodes = 3.3 * batch_size * (10 + 1) * (10 + 1) = 3.3 * batch_size * 11 * 11
            # limit_of_n_nodes, i.e. the length of pinned_buffer[history_i], need to be larger than size of blocks[history_i].srcdata['ID']
            self.pinned_node_memory_buffs_size = 0
            self.pinned_node_memory_ts_buffs = list()
            self.pinned_mailbox_buffs_size = 0
            self.pinned_mailbox_ts_buffs = list()
            
            mem_dim = self.node_memory.shape[1]
            mailbox_size = self.mails.shape[1]
            mail_dim = self.mails.shape[2]
            
            for _ in range(sample_param.history):
                # if history = 2, we need 2 buffers for each type
                self.pinned_node_memory_buffs_size += limit_of_n_nodes
                self.pinned_node_memory_ts_buffs.append(torch.zeros([limit_of_n_nodes], pin_memory=True))
                self.pinned_mailbox_buffs_size += limit_of_n_nodes
                self.pinned_mailbox_ts_buffs.append(torch.zeros([limit_of_n_nodes, mailbox_size], pin_memory=True))
            self.pinned_node_memory_buffs =  torch.zeros([self.pinned_node_memory_buffs_size, mem_dim], pin_memory=True)
            self.pinned_mailbox_buffs = torch.zeros([self.pinned_mailbox_buffs_size, mailbox_size, mail_dim], pin_memory=True)


    def prep_input_mails(self, deduplication, mfg, node_uniq, use_pinned_buffers=False):
        # example: memory_mailbox.prep_input_mails(mfgs[0])
        # history = 2
        # mfgs = [[block1,block2],[block3,block4],[block5,block6]]
        # receive mfg = [block1,block2] if 2 history
        data_map = [
            {'name': 'mem','src': self.node_memory, 'pinned': 'pinned_node_memory_buffs'},
            {'name': 'mem_ts','src': self.node_memory_ts, 'pinned': 'pinned_node_memory_ts_buffs'},
            {'name': 'mem_input','src': self.mails, 'pinned': 'pinned_mailbox_buffs', },
            {'name': 'mail_ts','src': self.mail_ts, 'pinned': 'pinned_mailbox_ts_buffs'},
        ]
        deduplicated_mem = None
        deduplicated_mem_input = None
        for hist_id, hist_block in enumerate(mfg):
            idx = hist_block.srcdata['ID'].cpu().long()
            n_nodes = idx.shape[0]
            for item in data_map:
                name = item['name']
                if not deduplication or name not in ['mem', 'mem_input']:
                    src_tensor = item['src']
                    pinned_list =  getattr(self, item['pinned'], None)
                    can_use_pinned = use_pinned_buffers and hist_id < len(pinned_list)
                    with Timer() as t:
                        if can_use_pinned:
                            # --- Use Pinned Memory Buffers ---
                            pinned_buffer = pinned_list[hist_id]
                            # Select data into the pinned buffer (CPU operation).
                            # we can see that the length of pinned_buffer[history_i] need to be larger than size of block.srcdata['ID']
                            torch.index_select(src_tensor, 0, idx, out=pinned_buffer[:n_nodes])
                            # Copy the selected data from pinned buffer to GPU (non-blocking).
                            gpu_data = pinned_buffer[:n_nodes].cuda(non_blocking=True)
                        else:
                            # --- Standard Method (No Pinned Buffers) ---
                            # Select data and directly copy to GPU.
                            gpu_data = src_tensor[idx].cuda()
                            # Reshape mailbox data if needed.
                        if name == 'mem_input':
                            gpu_data = gpu_data.reshape(n_nodes, -1)

                        # Assign the prepared data to the DGL block's source nodes.
                        hist_block.srcdata[name] = gpu_data
                    
                    if name == 'mem':
                        profiler.increase_time({'3_transfer_node_mem': t.last_duration})
                    elif name == 'mem_input':
                        profiler.increase_time({'4_transfer_node_mail': t.last_duration})
        if deduplication:
            src_tensor = self.node_memory
            pinned_buffer =  getattr(self, 'pinned_node_memory_buffs', None)
            with Timer() as t:
                if use_pinned_buffers:
                    torch.index_select(src_tensor, 0, node_uniq, out=pinned_buffer[:node_uniq.shape[0]])
                    deduplicated_mem = pinned_buffer[:node_uniq.shape[0]].cuda(non_blocking=True)
                else:
                    deduplicated_mem = src_tensor[node_uniq].to('cuda', non_blocking=True)
            profiler.increase_time({'3_transfer_node_mem': t.last_duration})
            
            src_tensor = self.mails
            pinned_buffer =  getattr(self, 'pinned_mailbox_buffs', None)
            with Timer() as t:
                if use_pinned_buffers:
                    torch.index_select(src_tensor, 0, node_uniq, out=pinned_buffer[:node_uniq.shape[0]])
                    deduplicated_mem_input = pinned_buffer[:node_uniq.shape[0]].cuda(non_blocking=True)
                else:
                    deduplicated_mem_input = src_tensor[node_uniq].to('cuda', non_blocking=True)
                deduplicated_mem_input = deduplicated_mem_input.reshape(node_uniq.shape[0], -1)
            profiler.increase_time({'4_transfer_node_mail': t.last_duration})


        return deduplicated_mem, deduplicated_mem_input





    @torch.no_grad()
    def update_memory(self, deduplication, nid, memory, root_nodes, ts, neg_samples=1):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        nid = nid[:num_true_src_dst].to(self.device)
        
        if deduplication:
        
            nid = nid.long()[:num_true_src_dst]   # shape: [N]
            memory = memory[:num_true_src_dst]    # shape: [N, D]
            ts = ts[:num_true_src_dst]            # shape: [N]
            idx = torch.arange(nid.size(0), device=nid.device)
            

            with Timer() as t:
                _, last_occurrence_idx = scatter_max(idx, nid)
                uniq_nid = torch.unique(nid)
                selected_idx = last_occurrence_idx[uniq_nid]
            profiler.increase_time({'6_compute_uniq_root_node_id_for_mem_update': t.last_duration})

            with Timer() as t:
                self.node_memory[uniq_nid] = memory[selected_idx].to(self.device)
            profiler.increase_time({'7_transfer_back_root_node_mem': t.last_duration})
            self.node_memory_ts[uniq_nid] = ts[selected_idx].to(self.device)
            
            
            profiler.increase_count({'unique_root_node_count': uniq_nid.shape[0],'total_root_node_count': nid.shape[0]})
            
        else:
            with Timer() as t:
                memory = memory[:num_true_src_dst].to(self.device)
                self.node_memory[nid.long()] = memory
                ts = ts[:num_true_src_dst].to(self.device)
                self.node_memory_ts[nid.long()] = ts
            profiler.increase_time({'7_transfer_back_root_node_mem': t.last_duration})



    @torch.no_grad()
    def update_mailbox(self, nid, memory: torch.Tensor, root_nodes, ts, edge_feats, mail_delivery_block, neg_samples=1):
        # print(f"[Mailbox Update]Batch: {batch}")
        # before update mailbox, we need to prepare the memory 
        # e.g. number_edges(5,6) = num_src(1,2) = num_src(3,4) = 2, mem_dim = 2 edge_dim = 3
        n_src = root_nodes.shape[0] // (neg_samples + 2)
        
        # move data to device
        memory = memory.to(self.device)
        if edge_feats is not None:
            edge_feats = edge_feats.to(self.device)
        if mail_delivery_block is not None:
            mail_delivery_block = mail_delivery_block.to(self.device)
            
        src_mail, dst_mail = self._concat_src_dst_memory(memory, n_src, edge_feats)
        mail, mail_ts, unique_node_ids, inverse_indices = self._prepare_mail_and_node_ids(root_nodes, ts, n_src, mail_delivery_block, src_mail, dst_mail)
        mail, mail_ts = self._combine_mail_mean_last(mail, mail_ts, inverse_indices, unique_node_ids)

        # now update the mailbox
        # every nodes can have as many mail positions(slots) as they want, typically 1 or 10
        idx = unique_node_ids.long()
        

        
        
        self.mails[idx, self.next_mail_pos[idx]] = mail
        self.mail_ts[idx, self.next_mail_pos[idx]] = mail_ts
        if self.memory_param.mailbox_size > 1:
            if self.mail_pos_update_pending_flag is not None:
                self.mail_pos_update_pending_flag[idx] = 1
            else:
                nid = idx
                self.next_mail_pos[nid] = (self.next_mail_pos[nid] + 1) % self.memory_param.mailbox_size
                
                


    def update_next_mail_pos(self):
        if self.mail_pos_update_pending_flag is not None:
            nid = torch.where(self.mail_pos_update_pending_flag == 1)[0]
            self.next_mail_pos[nid] = (self.next_mail_pos[nid] + 1) % self.memory_param.mailbox_size
            self.mail_pos_update_pending_flag.fill_(0)
            
    def _concat_src_dst_memory(self, memory, n_src, edge_feats=None):
        src_node_mem = memory[:n_src] # e.g. Tensor[[11,12],[21,22]]
        dst_node_mem = memory[n_src:n_src * 2] # e.g. Tensor[[31,32],[41,42]]
        
        src_mem_entries = [src_node_mem, dst_node_mem] # e.g. [Tensor[[11,12],[21,22]], Tensor[[31,32],[41,42]]]
        dst_mem_entries = [dst_node_mem, src_node_mem] # e.g. [Tensor[[31,32],[41,42]], Tensor[[11,12],[21,22]]]
        if self.dim_edge_feat > 0:
            src_mem_entries.append(edge_feats) # # e.g. [Tensor[[11,12],[21,22]], Tensor[[31,32],[41,42]], Tensor[[51,52,53],[61,62,63]]] 
            dst_mem_entries.append(edge_feats) # e.g. [Tensor[[31,32],[41,42]], Tensor[[11,12],[21,22]], Tensor[[51,52,53],[61,62,63]]] 
        src_mail = torch.cat(src_mem_entries, dim=1) # e.g. Tensor[[11,12, 31,32, 51,52,53], [21,22, 41,42, 61,62,63]]
        dst_mail = torch.cat(dst_mem_entries, dim=1) # e.g. Tensor[[31,32, 11,12, 51,52,53], [41,42, 21,22, 61,62,63]]
        return src_mail, dst_mail # src_mail dim [n, 2*mem_dim + edge_dim]
        
    def _prepare_mail_and_node_ids(self, root_nodes, ts, n_src, mail_delivery_block, src_mail, dst_mail):
        mail = None
        mail_ts = None
        # TGN/JODIE
        if self.memory_param.deliver_to == 'self':
            src = torch.from_numpy(root_nodes[:n_src]).to(self.device) # e.g [1,2]
            dst = torch.from_numpy(root_nodes[n_src:n_src * 2]).to(self.device) # e.g [3,4]
            mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1]) 
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
            mail_ts = torch.from_numpy(ts[:n_src * 2]).to(self.device)
            if mail_ts.dtype == torch.float64:
                import pdb; pdb.set_trace()
            # find unique nid to update mailbox
            unique_node_ids, inverse_indices = torch.unique(nid, return_inverse=True)
        # APAN
        elif self.memory_param.deliver_to == 'neighbors':
            mail = torch.cat([src_mail, dst_mail], dim=0) 
            # Tensor[[11,12, 31,32, 51,52,53], [21,22, 41,42, 61,62,63], [31,32, 11,12, 51,52,53], [41,42, 21,22, 61,62,63]]
            recipients_nid = mail_delivery_block.dstdata['ID']
            unique_node_ids, inverse_indices = torch.unique(recipients_nid, return_inverse=True)
            src_of_edge = mail_delivery_block.edges()[0]
            mail = torch.cat([mail, mail[src_of_edge.long()]], dim=0)
            mail_ts = torch.from_numpy(ts[:n_src * 2]).to(self.device)
            mail_ts = torch.cat([mail_ts, mail_ts[src_of_edge.long()]], dim=0)
        else:
            raise NotImplementedError
        return mail, mail_ts, unique_node_ids, inverse_indices # mail dim [2n, 2(2*mem_dim + edge_dim)]

    def _combine_mail_mean_last(self, mail, mail_ts,  inverse_indices, unique_node_ids):
        if self.memory_param.mail_combine == 'mean':
            mail = scatter(mail, inverse_indices, reduce='mean', dim=0)
            mail_ts = scatter(mail_ts, inverse_indices, reduce='mean')
        elif self.memory_param.mail_combine == 'last':
            # find unique nid to update mailbox
            # another e.g. nid [2, 3, 2, 5, 3]
            # unique_node_ids [2, 3, 5]
            # inverse_indices [0, 1, 0, 2, 1]
            perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
            # e.g. perm [0, 1, 2, 3, 4]
            perm = inverse_indices.new_empty(unique_node_ids.size(0)).scatter_(0, inverse_indices, perm)
            # e.g.
            # after new_empty: [0, 0, 0]
            # after scatter: [2, 4, 3]
            # nid = nid[perm] # size 3 [2, 3, 5], useless
            mail = mail[perm] 
            mail_ts = mail_ts[perm]
        else:
            raise NotImplementedError
        return mail, mail_ts
    


class BasicMemoryUpdater(torch.nn.Module):
    def __init__(self, memory_param: MemoryParams, edge_feat_dim, node_feat_dim):
        super().__init__()
        
        self.dim_hid = memory_param.dim_out
        self.dim_time = memory_param.dim_time
        self.rnn_input_size = 2 * self.dim_hid + edge_feat_dim + self.dim_time
        self.combine_node_feature = memory_param.combine_node_feature
        
        self.dim_node_feat = node_feat_dim
        
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if self.dim_time > 0:
            self.time_enc = TimeEncode(self.dim_time)
        if self.combine_node_feature:
            if self.dim_node_feat > 0 and self.dim_node_feat != self.dim_hid:
                self.node_feat_map = torch.nn.Linear(self.dim_node_feat, self.dim_hid)
                
    def forward(self, mfg):
        for block in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(block.srcdata['ts'] - block.srcdata['mem_ts'])
                block.srcdata['mem_input'] = torch.cat([block.srcdata['mem_input'], time_feat], dim=1)
                
            updated_memory = self.updater(block.srcdata['mem_input'], block.srcdata['mem'])
            
            self.last_updated_ts = block.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = block.srcdata['ID'].detach().clone() # too much nodes for memory updater and mailbox updater
            
            if self.combine_node_feature:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        block.srcdata['h'] += updated_memory
                    else:
                        block.srcdata['h'] = updated_memory + self.node_feat_map(block.srcdata['h'])
                else:
                    block.srcdata['h'] = updated_memory
        
        
class GRUMemoryUpdater(BasicMemoryUpdater):
    def __init__(self, memory_param: MemoryParams, edge_feat_dim,  node_feat_dim):
        super().__init__(memory_param, edge_feat_dim,  node_feat_dim)
        self.updater = torch.nn.GRUCell(self.rnn_input_size, self.dim_hid)

class RNNMemoryUpdater(BasicMemoryUpdater):
    def __init__(self, memory_param: MemoryParams, edge_feat_dim,  node_feat_dim):
        super().__init__(memory_param, edge_feat_dim,  node_feat_dim)
        self.updater = torch.nn.RNNCell(self.rnn_input_size, self.dim_hid)

import torch.nn.functional as F

class TransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, edge_feat_dim, train_param):
        super(TransformerMemoryUpdater, self).__init__()
        dim_in = 2 * memory_param.dim_out + edge_feat_dim
        dim_out = memory_param.dim_out
        self.memory_param = memory_param
        self.dim_time = memory_param.dim_time
        self.att_h = memory_param.attention_head
        if self.dim_time > 0:
            self.time_enc = TimeEncode(self.dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + self.dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + self.dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(train_param.dropout)
        self.att_dropout = torch.nn.Dropout(train_param.att_dropout)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, mfg):
        for block in mfg:
            n_src = block.num_src_nodes()
            mailbox_size = self.memory_param.mailbox_size
            Q = self.w_q(block.srcdata['mem']).reshape((n_src, self.att_h, -1))
            mails = block.srcdata['mem_input'].reshape((n_src, mailbox_size, -1))
            # mem_input dim (2*mem_dim + edge_dim)
            if self.dim_time > 0:
                time_feat = self.time_enc(block.srcdata['ts'][:, None] - block.srcdata['mail_ts']).reshape((n_src, mailbox_size, -1))
                mails = torch.cat([mails, time_feat], dim=2)
            K = self.w_k(mails).reshape((n_src, mailbox_size, self.att_h, -1))
            V = self.w_v(mails).reshape((n_src, mailbox_size, self.att_h, -1))
            att = self.att_act((Q[:,None,:,:]*K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:,:,:,None]*V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += block.srcdata['mem']
            rst = self.layer_norm(rst)
            rst = self.mlp(rst)
            rst = self.dropout(rst)
            rst = torch.nn.functional.relu(rst)
            block.srcdata['h'] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = block.srcdata['ID'].detach().clone()
            self.last_updated_ts = block.srcdata['ts'].detach().clone()

class SwiGLU(torch.nn.Module):
    """
    Implements the SwiGLU activation module.
    It splits the input into two halves, applies SiLU to the first half (gate),
    and multiplies it with the second half.
    """
    def forward(self, x):
        # x shape: [batch_size, ..., 2 * hidden_dim]
        gate, up_proj = x.chunk(2, dim=-1)
        return F.silu(gate) * up_proj
    
class GluTransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, edge_feat_dim, train_param):
        super(GluTransformerMemoryUpdater, self).__init__()
        dim_in = 2 * memory_param.dim_out + edge_feat_dim
        dim_out = memory_param.dim_out
        self.memory_param = memory_param
        self.dim_time = memory_param.dim_time
        self.att_h = memory_param.attention_head
        if self.dim_time > 0:
            self.time_enc = TimeEncode(self.dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + self.dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + self.dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        # self.mlp = torch.nn.Linear(dim_out, dim_out)
        
        hidden_dim = dim_out * 4  # 通常 FFN 的中间层维度会扩大4倍
        self.ffn_gate_proj = torch.nn.Linear(dim_out, hidden_dim * 2, bias=False) # SwiGLU 通常将维度扩大两倍后切分
        self.swiglu = SwiGLU()
        self.ffn_down_proj = torch.nn.Linear(hidden_dim, dim_out, bias=False)
        
        
        self.dropout = torch.nn.Dropout(train_param.dropout)
        self.att_dropout = torch.nn.Dropout(train_param.att_dropout)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, mfg):
        for block in mfg:
            n_src = block.num_src_nodes()
            mailbox_size = self.memory_param.mailbox_size
            Q = self.w_q(block.srcdata['mem']).reshape((n_src, self.att_h, -1))
            mails = block.srcdata['mem_input'].reshape((n_src, mailbox_size, -1))
            # mem_input dim (2*mem_dim + edge_dim)
            if self.dim_time > 0:
                time_feat = self.time_enc(block.srcdata['ts'][:, None] - block.srcdata['mail_ts']).reshape((n_src, mailbox_size, -1))
                mails = torch.cat([mails, time_feat], dim=2)
            K = self.w_k(mails).reshape((n_src, mailbox_size, self.att_h, -1))
            V = self.w_v(mails).reshape((n_src, mailbox_size, self.att_h, -1))
            att = self.att_act((Q[:,None,:,:]*K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:,:,:,None]*V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += block.srcdata['mem']
            rst = self.layer_norm(rst)
            
            residual = rst

            # 使用新的 FFN
            x = self.ffn_gate_proj(rst)
            x = self.swiglu(x)
            x = self.ffn_down_proj(x)
            x = self.dropout(x)
            
            rst = residual + x
            
            # rst = self.mlp(rst)
            # rst = self.dropout(rst)
            # rst = torch.nn.functional.relu(rst)
            block.srcdata['h'] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = block.srcdata['ID'].detach().clone()
            self.last_updated_ts = block.srcdata['ts'].detach().clone()
