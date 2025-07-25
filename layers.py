import torch
import dgl
import math
import numpy as np

class TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        seq = np.linspace(0, 9, dim, dtype=np.float32)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** seq)).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output

class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


from torch import nn

class TransfomerAttentionLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False):
        super(TransfomerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.combined = combined
        
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        
        self.att_dropout = torch.nn.Dropout(att_dropout)
        


        if combined:
            if dim_node_feat > 0:
                self.w_q_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:
                self.w_k_e = torch.nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)

    def forward(self, b):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
        dst_h = b.srcdata['h'][:b.num_dst_nodes()] # DST位于srcdata['h']的前半部分
        src_h = b.srcdata['h'][b.num_dst_nodes():] # SRC位于后半部分
        dst_indices = b.edges()[1]
        src_indices = b.edges()[0] - b.num_dst_nodes()
        # e.g.: 0, 1, 2, 3, 4
        # dst: 0, 1
        # src: 2, 3, 4
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            K = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            V = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            if self.dim_node_feat > 0:
                Q += self.w_q_n(dst_h)[dst_indices]
                K += self.w_k_n(src_h)[src_indices]
                V += self.w_v_n(src_h)[src_indices]
            if self.dim_edge_feat > 0:
                K += self.w_k_e(b.edata['f'])
                V += self.w_v_e(b.edata['f'])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[dst_indices]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
        else:
            q_input = []
            kv_input = []
            if self.dim_node_feat > 0:
                # if the src_h size_0 not equal to edge number, it will cause error
                q_input.append(dst_h)
                kv_input.append(src_h)
            if self.dim_edge_feat > 0:
                kv_input.append(b.edata['f'])
            if self.dim_time > 0:
                q_input.append(zero_time_feat)
                kv_input.append(time_feat)
            Q = self.w_q(torch.cat(q_input, dim=1))[dst_indices] if len(q_input) > 0 else torch.ones((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            K = self.w_k(torch.cat(kv_input, dim=1))
            V = self.w_v(torch.cat(kv_input, dim=1))
        Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
        K = torch.reshape(K, (K.shape[0], self.num_head, -1))
        V = torch.reshape(V, (V.shape[0], self.num_head, -1))
        att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
        att = self.att_dropout(att)
        V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
        if self.combined:
            b.edata['v'] = V
            b.update_all(dgl.function.copy_e('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            dst_dummy = torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0'))
            b.srcdata['v'] = torch.cat([dst_dummy, V], dim=0)
            b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        
        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], dst_h], dim=1)
        else:
            rst = b.dstdata['h']
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)

class IdentityNormLayer(torch.nn.Module):

    def __init__(self, dim_out):
        super(IdentityNormLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        return self.norm(b.srcdata['h'])

class JODIETimeEmbedding(torch.nn.Module):

    def __init__(self, dim_out):
        super(JODIETimeEmbedding, self).__init__()
        self.dim_out = dim_out

        class NormalLinear(torch.nn.Linear):
        # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.time_emb = NormalLinear(1, dim_out)
    
    def forward(self, h, mem_ts, ts):
        time_diff = (ts - mem_ts) / (ts + 1)
        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst
            