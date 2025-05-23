import torch
import threading
from utils import mfgs_to_cuda, prepare_input, get_ids
from modules import MemoryMailbox

class DataPipelineThread(threading.Thread):
    def __init__(self, my_lists , stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox: MemoryMailbox, memory_param):
        super(DataPipelineThread, self).__init__()
        keys = ['mfgs', 'root', 'ts', 'eid', 'block']
        self.my_mfgs, self.my_root, self.my_ts, self.my_eid, self.my_block  = [my_lists[k] for k in keys]
        self.stream = stream
        self.mfgs = None
        self.root = None
        self.ts = None
        self.eid = None
        self.block = None
        self.pinned_nfeat_buffs = pinned_nfeat_buffs
        self.pinned_efeat_buffs = pinned_efeat_buffs
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.mailbox = mailbox
        self.memory_param = memory_param

    def run(self):
        with torch.cuda.stream(self.stream):
            # print(local_rank, 'start thread')
            nids, eids = get_ids(self.my_mfgs[0], self.node_feats, self.edge_feats)
            mfgs = mfgs_to_cuda(self.my_mfgs[0])
            prepare_input(mfgs, self.node_feats, self.edge_feats, pinned=True, nfeat_buffs=self.pinned_nfeat_buffs, efeat_buffs=self.pinned_efeat_buffs, nids=nids, eids=eids)
            if self.mailbox is not None:
                self.mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=True)
                if self.memory_param.deliver_to == 'neighbors':
                    self.block = self.my_block[0]
            self.mfgs = mfgs
            self.root = self.my_root[0]
            self.ts = self.my_ts[0]
            self.eid = self.my_eid[0]
            # print(local_rank, 'finished')

    def get_stream(self):
        return self.stream

    def get_mfgs(self):
        return self.mfgs
    
    def get_root(self):
        return self.root
    
    def get_ts(self):
        return self.ts

    def get_eid(self):
        return self.eid

    def get_block(self):
        return self.block
