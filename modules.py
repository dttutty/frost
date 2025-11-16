from typing import Any, Callable, Optional

import torch
from dgl.heterograph import DGLBlock

from layers import (
    EdgePredictor,
    IdentityNormLayer,
    JODIETimeEmbedding,
    TransfomerAttentionLayer,
)
from memorys import (
    GluTransformerMemoryUpdater,
    GRUMemoryUpdater,
    RNNMemoryUpdater,
    TransformerMemoryUpdater,
)
from profiler import Timer, profiler
from schemas.datatypes import Config
from utils import Config


class GeneralModel(torch.nn.Module):
    def __init__(
        self,
        ndim,
        edim,
        cfg: Config,
        comps,
        comms,
        mmfn: Optional[Callable[..., Any]] = None,
    ):
        super(GeneralModel, self).__init__()
        (scfg, mcfg, gcfg, tcfg) = (
            cfg.s,
            cfg.m,
            cfg.g,
            cfg.t,
        )
        self.train_param = cfg.t

        self.combined = tcfg.merge_src if tcfg.merge_src else False
        self.dim_node_input = ndim
        self.dedup = tcfg.dedup

        self.scfg = cfg.s
        self.mcfg = cfg.m
        self.gcfg = cfg.g
        self.comp_stream = comps
        self.comm_stream = comms
        self.mmfn = mmfn
        if self.gcfg.dim_out is None:
            self.gcfg.dim_out = self.mcfg.dim_out

        self.gcfg = gcfg
        if mcfg.type == "node":
            match mcfg.memory_update:
                case "gru":
                    self.memory_updater = GRUMemoryUpdater(mcfg, edim, ndim)
                case "rnn":
                    self.memory_updater = RNNMemoryUpdater(mcfg, edim, ndim)
                case "transformer":
                    self.memory_updater = TransformerMemoryUpdater(mcfg, edim, tcfg)
                case "glu_transformer":
                    self.memory_updater = GluTransformerMemoryUpdater(mcfg, edim, tcfg)
                case _:
                    raise NotImplementedError
            self.dim_node_input = mcfg.dim_out

        self.layers = torch.nn.ModuleDict()
        if gcfg.arch == "transformer_attention":
            for h in range(scfg.history):
                self.layers["l0h" + str(h)] = TransfomerAttentionLayer(
                    self.dim_node_input,
                    edim,
                    gcfg.dim_time,
                    gcfg.att_head,
                    tcfg.dropout,
                    tcfg.att_dropout,
                    gcfg.dim_out,
                    combined=self.combined,
                )
            for l in range(1, gcfg.layer):
                for h in range(scfg.history):
                    self.layers["l" + str(l) + "h" + str(h)] = TransfomerAttentionLayer(
                        gcfg.dim_out,
                        edim,
                        gcfg.dim_time,
                        gcfg.att_head,
                        tcfg.dropout,
                        tcfg.att_dropout,
                        gcfg.dim_out,
                        combined=False,
                    )
        elif gcfg.arch == "identity":
            self.gcfg.layer = 1
            for h in range(scfg.history):
                self.layers["l0h" + str(h)] = IdentityNormLayer(self.dim_node_input)
                if gcfg.time_transform == "JODIE":
                    self.layers["l0h" + str(h) + "t"] = JODIETimeEmbedding(gcfg.dim_out)
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(gcfg.dim_out)

        if gcfg.combine == "rnn":
            self.combiner = torch.nn.RNN(
                gcfg.dim_out, gcfg.dim_out, num_layers=1, bidirectional=False
            )
            # input: (sequence length, batch size, input feature dimension)
            # output: (sequence length, batch size, hidden dimension)

    def get_emb(self, mfgss: list[list[DGLBlock]]):
        # memory part
        if self.mcfg.type == "node":
            # in place update memory of mfgs[0]
            self.memory_updater(mfgss[0])

        def detach_mfg(mfg: DGLBlock):
            mfg = mfg.clone()  # 浅拷贝结构（边、节点引用仍然共享）
            for k, v in mfg.srcdata.items():
                if torch.is_tensor(v):
                    mfg.srcdata[k] = v.detach()
            for k, v in mfg.dstdata.items():
                if torch.is_tensor(v):
                    mfg.dstdata[k] = v.detach()
            return mfg

        mfgs_detached = [detach_mfg(b) for b in mfgss[0]]
        with self.comm_stream:
            self.mmfn.init(mfgs_detached)
            self.mmfn.process(mfgs_detached)
        with self.comp_stream:
            hist_out = list()

            # gnn part
            # for each layer and history, apply the corresponding layer
            for l in range(self.gcfg.layer):
                for h in range(self.scfg.history):
                    mfgs_l_h = mfgss[l][h]
                    rst = self.layers["l" + str(l) + "h" + str(h)](mfgs_l_h)
                    if self.gcfg.time_transform == "JODIE":
                        rst = self.layers["l0h" + str(h) + "t"](
                            rst, mfgs_l_h.srcdata["mem_ts"], mfgs_l_h.srcdata["ts"]
                        )
                    is_last_gnn_layer = l == self.gcfg.layer - 1
                    if is_last_gnn_layer:
                        hist_out.append(rst)
                    else:
                        mfgss[l + 1][h].srcdata["h"] = rst
            # guess if history is 1, out size is 1
            # if history > 1, out size is history, out is a list, out[0] shape is (num_nodes, dim_out)
            if self.scfg.history == 1:
                hist_out = hist_out[0]
            else:
                hist_out = torch.stack(hist_out, dim=0)  # (history, num_nodes, dim_out)
                all_outputs, h_n = self.combiner(
                    hist_out
                )  # all_outputs shape: (history, num_nodes, dim_out), (1, num_nodes, dim_out)
                hist_out = all_outputs[-1, :, :]
            # predictor header
            return hist_out

    def forward(
        self,
        gpudata,
        neg_samples=1,
    ):
        mfgs = gpudata.mfgs
        node_feats_on_gpu = gpudata.nfeat
        edge_feats_on_gpu = gpudata.efeat
        dedup_mem = gpudata.gpumem
        dedup_mem_input = gpudata.gpumail

        import torch.distributed as dist

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # print(f"rank={rank}, world_size={world_size}")
        # print("GeneralModel forward")
        if self.dedup:
            with Timer() as t:
                if node_feats_on_gpu is not None:
                    for block in mfgs[0]:
                        # assert block.srcdata['pos'] is within the range of node_feats_on_gpu
                        # print(f"1.block.srcdata['pos']: {block.srcdata['pos']}")
                        assert (
                            torch.max(block.srcdata["pos"]) < node_feats_on_gpu.shape[0]
                        ), (
                            f"block.srcdata['pos'] max {torch.max(block.srcdata['pos'])} >= node_feats_on_gpu.shape[0] {node_feats_on_gpu.shape[0]}"
                        )
                        block.srcdata["h"] = node_feats_on_gpu[block.srcdata["pos"]]
                    del node_feats_on_gpu
                if dedup_mem is not None:
                    for block in mfgs[0]:
                        # print(f"2.block.srcdata['pos']: {block.srcdata['pos']}")
                        assert torch.max(block.srcdata["pos"]) < dedup_mem.shape[0], (
                            f"block.srcdata['pos'] max {torch.max(block.srcdata['pos'])} >= deduplicated_mem.shape[0] {dedup_mem.shape[0]}"
                        )
                        block.srcdata["mem"] = dedup_mem[block.srcdata["pos"]]
                    del dedup_mem
                if dedup_mem_input is not None:
                    for block in mfgs[0]:
                        # print(f"3.block.srcdata['pos']: {block.srcdata['pos']}")
                        assert (
                            torch.max(block.srcdata["pos"]) < dedup_mem_input.shape[0]
                        ), (
                            f"block.srcdata['pos'] max {torch.max(block.srcdata['pos'])} >= deduplicated_mem_input.shape[0] {dedup_mem_input.shape[0]}"
                        )
                        block.srcdata["mem_input"] = dedup_mem_input[
                            block.srcdata["pos"]
                        ]
                    del dedup_mem_input
                if edge_feats_on_gpu is not None:
                    for blocks in mfgs:
                        for block in blocks:
                            # print(f"4.block.edata['pos']: {block.edata['pos']}")
                            # assert torch.max(block.edata['pos']) < edge_feats_on_gpu.shape[0], f"block.edata['pos'] max {torch.max(block.edata['pos'])} >= edge_feats_on_gpu.shape[0] {edge_feats_on_gpu.shape[0]}"
                            block.edata["f"] = edge_feats_on_gpu[block.edata["pos"]]
                    del edge_feats_on_gpu
            profiler.increase_time({"5_mapping_back": t.last_duration})
        # print("6.after dedup")
        hist_out = self.get_emb(mfgs)
        # print("7.fter get_emb")
        return_value = self.edge_predictor(hist_out, neg_samples=neg_samples)
        # print("8.after edge_predictor")
        return return_value


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
