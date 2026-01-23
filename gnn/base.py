# worker/trainer.py


import logging

import torch
from memorys import (
    BasicMemoryUpdater,
    GluTransformerMemoryUpdater,
    TransformerMemoryUpdater,
)

from data_syncer.data_syncer import DataSyncer
from gnn.components import (
    EdgePredictor,
    IdentityNormLayer,
    TransfomerAttentionLayer,
)
from schemas.datatypes.cfg import Config
from schemas.datatypes.data_to_sync import MemTsNid
from schemas.datatypes.misc import Meta
from schemas.datatypes.plan import PlanSet


logger = logging.getLogger(__name__)


class BaseModel(torch.nn.Module):
    def __init__(self, cfg: Config, data_syncer: DataSyncer):
        super().__init__()
        self.cfg = cfg
        ndim = cfg.t.ndim
        self.ndim = ndim
        edim = cfg.t.edim
        self.edim = edim
        self.data_syncer = data_syncer

        self.neg_samples = cfg.s.neg_samples

        self.merge_src = cfg.t.merge_src
        self.d_mem = cfg.m.d_mem if cfg.m is not None else None

        if cfg.g is not None and cfg.m is not None:
            assert cfg.g.d_hid == cfg.m.d_mem
        if cfg.m is not None:
            self.deliver_to = cfg.m.deliver_to
            self.mail_combine = cfg.m.mail_combine
            self.boxsize = cfg.m.boxsize
            self.d_mem = cfg.m.d_mem

            if ndim > 0 and ndim != cfg.m.d_mem:
                self.n_feat_map = torch.nn.Linear(ndim, cfg.m.d_mem)

            if cfg.g is not None:
                cfg.g.d_hid = cfg.m.d_mem

            self.memory_updater = self.initialize_memory_updater(cfg)

        self.layers = torch.nn.ModuleDict()

        self.history = (
            cfg.s.history if cfg.s else 1
        )  # define history = 1 if no sampling
        if cfg.g.arch == "attn":
            for layer in range(0, cfg.g.layer):
                for hist in range(self.history):
                    d_in = cfg.g.d_in if layer == 0 else cfg.g.d_hid
                    combined = self.merge_src if layer == 0 else False
                    self.layers[f"l{layer}h{hist}"] = TransfomerAttentionLayer(
                        d_in,
                        edim,
                        cfg.g.d_time,
                        cfg.g.att_h,
                        cfg.t.dropout,
                        cfg.g.d_hid,
                        combined=combined,
                    )
        elif cfg.g.arch == "identity":
            cfg.g.layer = 1
            for hist in range(self.history):
                self.layers["l0h" + str(hist)] = IdentityNormLayer(cfg.g.d_in)
                if cfg.g.time_transform == "JODIE":
                    self.layers[f"l0h{hist}"] = JODIETimeEmbedding(cfg.g.d_hid)
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(cfg.g.d_hid)

        if cfg.g.combine == "rnn":
            self.combiner = torch.nn.RNN(
                cfg.g.d_hid, cfg.g.d_hid, num_layers=1, bidirectional=False
            )
            # input: (sequence length, batch size, input feature dimension)
            # output: (sequence length, batch size, hidden dimension)

    def fill_new_mem(self, blockss_00, upd_mem, incremental=None):
        # if self.combine_node_feature:
        if self.ndim > 0 and self.ndim == self.d_mem:
            blockss_00.srcdata["h"] += upd_mem
        elif self.ndim > 0 and self.ndim != self.d_mem:
            blockss_00.srcdata["h"] = upd_mem + self.n_feat_map(blockss_00.srcdata["h"])
        else:
            # node feature = combine updated memory with node feature
            blockss_00.srcdata["h"] = upd_mem

    def forward(
        self,
        meta: Meta,
        x: DyX,
        plan_set: PlanSet,
        neg_samples=1,
        is_last=False,
    ):
        blockss = meta.gnn_mfgs_raw = (gnn_mfgs_raw,)
        # sort feat by nid
        # memory partg
        root_new_mem = None
        n_root_src_n_pos_dst = meta.pos_edge.shape[0] * 2
        if self.cfg.m is not None:
            assert len(blockss) == 1 and len(blockss[0]) == 1, (
                "Memory updater only supports one history window and one layer"
            )
            ts, upd_mem, nid = self.memory_updater(blockss)
            root_new_mem = MemTsNid(
                ts=ts[:n_root_src_n_pos_dst],  # 新交互的ts
                data=upd_mem[:n_root_src_n_pos_dst],
                nid=nid[:n_root_src_n_pos_dst],  # 新交互的nid
            )
            self.fill_new_mem(
                blockss[0][0], upd_mem, self.n_feat_map(blockss[0][0].srcdata["h"])
            )

        hist_out = list()

        # gnn part
        # for each layer and history, apply the corresponding layer
        for layer in range(self.cfg.g.layer):
            for hist in range(self.history):
                blocks_l_h = blockss[layer][hist]  # 0-1-2
                rst = self.layers["l" + str(layer) + "h" + str(hist)](blocks_l_h)
                if self.cfg.g.time_transform == "JODIE":
                    rst = self.layers["l0h" + str(hist) + "t"](
                        rst, blocks_l_h.srcdata["mem_ts"], blocks_l_h.srcdata["ts"]
                    )
                is_last_gnn_layer = layer == self.cfg.g.layer - 1
                if is_last_gnn_layer:
                    hist_out.append(rst)
                else:
                    blockss[layer + 1][hist].srcdata["h"] = rst
        # guess if history is 1, out size is 1
        # if history > 1, out size is history, out is a list, out[0] shape is (num_nodes, d_hid)
        if self.history == 1:
            hist_out = hist_out[0]
        else:
            hist_out = torch.stack(hist_out, dim=0)  # (history, num_nodes, d_hid)
            all_outputs, h_n = self.combiner(
                hist_out
            )  # all_outputs shape: (history, num_nodes, d_hid), (1, num_nodes, d_hid)
            hist_out = all_outputs[-1, :, :]
        # predictor header

        # print("7.fter get_emb")
        return_value = self.edge_predictor(hist_out, neg_samples=neg_samples)
        # print("8.after edge_predictor")

        print("exit forward, meta.macro_bid:", meta.macro_bid)

        self.data_syncer.run(
            plan_set,
            root_new_mem,
            x,
            meta.mdt,
            is_last=is_last,
        )

        return return_value

    def initialize_memory_updater(self, cfg: Config):
        assert cfg.m is not None

        match cfg.m.cell:
            case "gru" | "rnn":
                mem_updater_class = (
                    torch.nn.GRUCell if cfg.m.cell == "gru" else torch.nn.RNNCell
                )
                memory_updater = BasicMemoryUpdater(
                    cfg.m,
                    cfg.t.ndim,
                    cfg.t.edim,
                    mem_updater_class,
                )
            case "transformer":
                assert cfg.m.att_h is not None
                assert cfg.m.boxsize is not None
                memory_updater = TransformerMemoryUpdater(
                    cfg.m,
                    cfg.t.ndim,
                    cfg.t.edim,
                    cfg.t.dropout,
                )
            case "glu_transformer":
                assert cfg.m.att_h is not None
                assert cfg.m.boxsize is not None
                memory_updater = GluTransformerMemoryUpdater(
                    cfg.m,
                    cfg.t.ndim,
                    cfg.t.edim,
                    cfg.t.dropout,
                )
            case _:
                raise NotImplementedError
        return memory_updater


class NodeClassificationModel(torch.nn.Module):
    def __init__(self, dim_in, d_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, d_hid)
        self.fc2 = torch.nn.Linear(d_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
