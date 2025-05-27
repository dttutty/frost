import torch
import threading
from utils import mfgs_to_cuda, prepare_mfgs, get_ids
from modules import MemoryMailbox


class DataPipelineThread(threading.Thread):
    def __init__(self, received_data , stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox: MemoryMailbox, memory_param):
        super(DataPipelineThread, self).__init__()
        self.received_data = received_data
        self.raw_mfgs = received_data.mfgs
        self.stream = stream
        self.async_mfgs = None
        self.pinned_nfeat_buffs = pinned_nfeat_buffs
        self.pinned_efeat_buffs = pinned_efeat_buffs
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.mailbox = mailbox
        self.memory_param = memory_param

    def run(self):
        with torch.cuda.stream(self.stream):
            # print(local_rank, 'start thread')
            nids, eids = get_ids(self.raw_mfgs, self.node_feats, self.edge_feats)
            gnn_mfgs = mfgs_to_cuda(self.raw_mfgs)
            prepare_mfgs(gnn_mfgs, self.node_feats, self.edge_feats, pinned=True, nfeat_buffs=self.pinned_nfeat_buffs, efeat_buffs=self.pinned_efeat_buffs, nids=nids, eids=eids)
            if self.mailbox is not None:
                self.mailbox.prep_input_mails(gnn_mfgs[0], use_pinned_buffers=True)
            self.async_mfgs = gnn_mfgs

    def clear_tensors(self):
        self.async_mfgs.clear()
        # self.raw_mfgs.clear()
