# worker/trainer.py

from queue import Queue
from typing import cast

import torch
from dgl.heterograph import DGLBlock

from layers import (
    EdgePredictor,
    IdentityNormLayer,
    JODIETimeEmbedding,
    TransfomerAttentionLayer,
)
from memorys import (
    BasicMemoryUpdater,
    GluTransformerMemoryUpdater,
    TransformerMemoryUpdater,
)
from schemas.datatypes import Config, MemTsNid, Meta, PlanSet, SyncTask, X


class BaseModel(torch.nn.Module):
    def __init__(self, cfg: Config, sync_q: Queue[SyncTask]):
        assert cfg.m is not None
        super().__init__()
        self.cfg = cfg
        ndim = cfg.t.ndim
        self.ndim = ndim
        edim = cfg.t.edim
        self.edim = edim
        self.sync_q = sync_q
        self.stream = torch.cuda.Stream()

        self.neg_samples = cfg.s.neg_samples

        self.merge_src = cfg.t.merge_src
        self.d_mem = cfg.m.d_mem

        assert cfg.g.d_hid == cfg.m.d_mem
        self.deliver_to = cfg.m.deliver_to
        self.mail_combine = cfg.m.mail_combine
        self.boxsize = cfg.m.boxsize
        self.d_mem = cfg.m.d_mem

        if ndim > 0 and ndim != cfg.m.d_mem:
            self.n_feat_map = torch.nn.Linear(ndim, cfg.m.d_mem)

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

    def fill_block(
        self,
        block: DGLBlock,
        x: X,
    ):
        # print(f"fill_blocks block.srcdata['pos']: {block.srcdata['pos']}")
        assert x.is_sorted  # make sure x.nid is sorted
        _ID: torch.Tensor = cast(torch.Tensor, block.srcdata["ID"])
        # step 1: find index of each x.nid in _ID
        permute_nidx = torch.searchsorted(
            x.nid, _ID
        )  # need to make sure _ID is subset of x.nid
        if x.mem is not None:
            block.srcdata["mem"] = x.mem.data[permute_nidx]  # 这里放memory
            block.srcdata["mem_ts"] = x.mem.ts[permute_nidx]
        if x.box is not None:
            block.srcdata["box"] = x.box.data[permute_nidx]
            block.srcdata["box_ts"] = x.box.ts[permute_nidx]
        if x.nfeat is not None:
            block.srcdata["h"] = x.nfeat.data[permute_nidx]  # 这里放feat
        if x.efeat is not None:
            _ID: torch.Tensor = cast(torch.Tensor, block.edata["ID"])
            permute_eidx = torch.searchsorted(
                x.efeat.id, _ID
            )  # need to make sure _ID is subset of x.eid
            block.edata["feat"] = x.efeat.data[permute_eidx]

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
        x: X,
        plan_set: PlanSet,
        neg_samples=1,
        is_last=False,
    ):
        with torch.cuda.stream(self.stream):
            print("enter forward, meta.macro_bid:", meta.macro_bid)
            blockss = meta.blockss
            self.fill_block(meta.blockss[0][0], x)
            # sort feat by nid
            # memory partg
            root_new_mem = None
            n_root_src_n_pos_dst = meta.pos_edge.shape[0] * 2
            assert len(blockss) == 1 and len(blockss[0]) == 1, (
                "Memory updater only supports one history window and one layer"
            )
            ts, upd_mem, nid = self.memory_updater(blockss)

            root_new_mem = MemTsNid(
                ts=ts[:n_root_src_n_pos_dst],  # 新交互的ts
                data=upd_mem[:n_root_src_n_pos_dst],
                id=nid[:n_root_src_n_pos_dst],  # 新交互的nid
            )
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())
            self.sync_q.put(
                SyncTask(
                    meta=meta,
                    plan_set=plan_set,
                    root_new_mem=root_new_mem,
                    x=x,
                    mail_delivery_tensor=meta.mdt,
                    is_last=is_last,
                    ready_event=ready_event,  # type: ignore
                )
            )

            self.fill_new_mem(
                blockss[0][0], upd_mem, self.n_feat_map(blockss[0][0].srcdata["h"])
            )
            for i, blocks in enumerate(blockss):
                for j, block in enumerate(blocks):
                    if i != 0 or j != 0:
                        self.fill_block(block, x)

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

            print("GNN: end of forward", meta.macro_bid)

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
