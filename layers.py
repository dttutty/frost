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
        
        # feat_dims = {
        #     'n': dim_node_feat,
        #     'e': dim_edge_feat,
        #     't': dim_time
        # }
        # self.proj = nn.ModuleDict()
        # if combined:
        #     for f, d in feat_dims.items():
        #         if d <= 0:
        #             continue
        #         # k、v 对所有有维度的特征都要
        #         self.proj[f"k_{f}"] = nn.Linear(d, dim_out)  # 我们统一用 kv_f 做 k 和 v 投影
        #         self.proj[f"k_{f}"] = nn.Linear(d, dim_out)  # 我们统一用 kv_f 做 k 和 v 投影
        #         # 只对 'n' 和 't' 做 q 投影
        #         if f in ('n', 't'):
        #             self.proj[f"q_{f}"] = nn.Linear(d, dim_out)
        # else:
        #     dim_q  = dim_node_feat + dim_time
        #     if dim_q > 0:
        #         self.proj['q'] = nn.Linear(dim_q, dim_out)
                
        #     dim_kv = dim_node_feat + dim_edge_feat + dim_time
        #     self.proj['k'] = nn.Linear(dim_kv, dim_out)
        #     self.proj['v'] = nn.Linear(dim_kv, dim_out)
            
        # self.proj['out'] = nn.Linear(dim_node_feat + dim_out, dim_out)

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
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            K = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            V = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            if self.dim_node_feat > 0:
                Q += self.w_q_n(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K += self.w_k_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                V += self.w_v_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
            if self.dim_edge_feat > 0:
                K += self.w_k_e(b.edata['f'])
                V += self.w_v_e(b.edata['f'])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[b.edges()[1]]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.edata['v'] = V
            b.update_all(dgl.function.copy_edge('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            if self.dim_time == 0 and self.dim_node_feat == 0:
                Q = torch.ones((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
                K = self.w_k(b.edata['f'])
                V = self.w_v(b.edata['f'])
            elif self.dim_time == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(b.srcdata['h'][b.num_dst_nodes():])
                V = self.w_v(b.srcdata['h'][b.num_dst_nodes():])
            elif self.dim_time == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
            elif self.dim_node_feat == 0:
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
            elif self.dim_edge_feat == 0:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
            else:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
            b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
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
            