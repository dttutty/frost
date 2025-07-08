from typing import Optional, Literal
import torch
from memorys import GRUMemoryUpdater, RNNMemoryUpdater, TransformerMemoryUpdater, GluTransformerMemoryUpdater
from layers import *
from dataclasses import dataclass, field
from memorys import MemoryParams
from sampler import SampleParams

@dataclass
class GNNParams:
    arch: Literal["identity", "transformer_attention"] = "identity"
    dim_out: Optional[int] = 100
    layer: Optional[int] = 1
    dim_time: Optional[int] = 100
    att_head: Optional[int] = 2
    time_transform: Optional[str] = None
    combine: Optional[Literal["rnn"]] = None

@dataclass
class TrainParams:
    epoch: Optional[int] = 10
    batch_size: Optional[int] = 600
    reorder: Optional[int] = 8
    lr: Optional[float] = 0.0001
    dropout: Optional[float] = 0.1
    att_dropout: Optional[float] = 0.1
    all_on_gpu: Optional[bool] = False
    combine_neighs: Optional[bool] = False
    deduplicate_mfg_src: bool = False # add by sqp17 

class GeneralModel(torch.nn.Module):

    def __init__(self, 
                 node_feat_dim, 
                 edge_feat_dim, 
                 sample_param: SampleParams, 
                 memory_param: MemoryParams, 
                 gnn_param: GNNParams, 
                 train_param: TrainParams, 
                 combined=False):
        super(GeneralModel, self).__init__()
        self.combined = combined    
        self.dim_node_input = node_feat_dim
        
        self.sample_param = sample_param
        self.memory_param = memory_param
        if gnn_param.dim_out is None:
            gnn_param.dim_out = memory_param.dim_out
            
        self.gnn_param = gnn_param
        self.train_param = train_param
        if memory_param.type == 'node':
            match memory_param.memory_update:
                case 'gru':
                    self.memory_updater = GRUMemoryUpdater(memory_param, edge_feat_dim, node_feat_dim)
                case 'rnn':
                    self.memory_updater = RNNMemoryUpdater(memory_param, edge_feat_dim, node_feat_dim)
                case 'transformer':
                    self.memory_updater = TransformerMemoryUpdater(memory_param, edge_feat_dim, train_param)
                case 'glu_transformer':
                    self.memory_updater = GluTransformerMemoryUpdater(memory_param, edge_feat_dim, train_param)
                case _:
                    raise NotImplementedError
            self.dim_node_input = memory_param.dim_out
        
        self.layers = torch.nn.ModuleDict()
        if gnn_param.arch == 'transformer_attention':
            for h in range(sample_param.history):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, edge_feat_dim, gnn_param.dim_time, gnn_param.att_head, train_param.dropout, train_param.att_dropout, gnn_param.dim_out, combined=combined)
            for l in range(1, gnn_param.layer):
                for h in range(sample_param.history):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param.dim_out, edge_feat_dim, gnn_param.dim_time, gnn_param.att_head, train_param.dropout, train_param.att_dropout, gnn_param.dim_out, combined=False)
        elif gnn_param.arch == 'identity':
            self.gnn_param.layer = 1
            for h in range(sample_param.history):
                self.layers['l0h' + str(h)] = IdentityNormLayer(self.dim_node_input)
                if  gnn_param.time_transform == 'JODIE':
                    self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(gnn_param.dim_out)
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(gnn_param.dim_out)
        
        if gnn_param.combine == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param.dim_out, gnn_param.dim_out, num_layers=1, bidirectional=False)
            # input: (sequence length, batch size, input feature dimension)
            # output: (sequence length, batch size, hidden dimension)
    
    


    def get_emb(self, mfgs):
        # memory part
        if self.memory_param.type == 'node':
            # in place update memory of mfgs[0]
            self.memory_updater(mfgs[0]) 
        hist_out = list()
        
        # gnn part
        # for each layer and history, apply the corresponding layer
        for l in range(self.gnn_param.layer):
            for h in range(self.sample_param.history):
                mfgs_l_h = mfgs[l][h]
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs_l_h)
                if self.gnn_param.time_transform == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs_l_h.srcdata['mem_ts'], mfgs_l_h.srcdata['ts'])
                is_last_gnn_layer = (l == self.gnn_param.layer - 1)
                if is_last_gnn_layer:
                    hist_out.append(rst)
                else:
                    mfgs[l + 1][h].srcdata['h'] = rst
        # guess if history is 1, out size is 1
        # if history > 1, out size is history, out is a list, out[0] shape is (num_nodes, dim_out)
        if self.sample_param.history == 1:
            hist_out = hist_out[0]
        else:
            hist_out = torch.stack(hist_out, dim=0) # (history, num_nodes, dim_out)
            all_outputs, h_n = self.combiner(hist_out)  # all_outputs shape: (history, num_nodes, dim_out), (1, num_nodes, dim_out)
            hist_out = all_outputs[-1, :, :]
        # predictor header
        return hist_out
    
    def forward(self, mfgs, neg_samples=1):
        hist_out = self.get_emb(mfgs)
        return self.edge_predictor(hist_out, neg_samples=neg_samples)


class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
