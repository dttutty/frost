import torch
import threading
from utils import mfgs_to_cuda, prepare_mfgs, get_ids
from memorys import MemoryMailbox
import torch.distributed as dist
local_rank = dist.get_rank()
world_size = dist.get_world_size()
host_rank = world_size - 1
def scatter_receive(src=host_rank):
    my = [None]
    dist.scatter_object_list(my, [None] * world_size, src=src)
    return my[0]

from dataclasses import dataclass
from typing import List, Optional
from dgl.heterograph import DGLBlock
from numpy.typing import NDArray
import numpy as np
@dataclass
class TransferData:
    mfgs: Optional[List[List[DGLBlock]]]  = None
    root: Optional[NDArray[np.int32]] = None
    ts: Optional[NDArray[np.float32]] = None
    eid: Optional[List] = None
    block: Optional[List[List[DGLBlock]]]  = None
    
class MoveDataToGPU(threading.Thread):
    def __init__(self, deduplication, data_received_event: threading.Event, host_rank, stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox: MemoryMailbox, memory_param, deduplicate_mfg_src, prev_mailbox_memory_update_thread):
        
        
        super(MoveDataToGPU, self).__init__()
        self.host_rank = host_rank
        self.data_received_event = data_received_event
        
        self.stream = stream
        self.async_mfgs = None
        self.pinned_nfeat_buffs = pinned_nfeat_buffs
        self.pinned_efeat_buffs = pinned_efeat_buffs
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.mailbox = mailbox
        self.memory_param = memory_param
        self.deduplicate_mfg_src = deduplicate_mfg_src
        self.prev_mailbox_memory_update_thread = prev_mailbox_memory_update_thread
        self.deduplication = deduplication

    def run(self):
        self.received_data : TransferData = scatter_receive()
        self.data_received_event.set()
        self.mfgss_on_cpu = self.received_data.mfgs
        
        with torch.cuda.stream(self.stream):
            nids, eids = get_ids(self.mfgss_on_cpu, self.node_feats, self.edge_feats)
            mfgss_on_gpu = mfgs_to_cuda(self.mfgss_on_cpu)
            mfgss_on_gpu, node_feats_on_gpu, edge_feats_on_gpu, node_uniq, edge_uniq = prepare_mfgs(self.deduplication, self.mfgss_on_cpu, mfgss_on_gpu, \
                self.node_feats, self.edge_feats, deduplicate_mfg_src=self.deduplicate_mfg_src, pinned=True, nfeat_buffs=self.pinned_nfeat_buffs, efeat_buffs=self.pinned_efeat_buffs, nids=nids, eids=eids)
            if self.prev_mailbox_memory_update_thread is not None:
                self.prev_mailbox_memory_update_thread.join()
                self.prev_mailbox_memory_update_thread.cleanup()
                self.prev_mailbox_memory_update_thread = None
            if self.mailbox is not None:
                deduplicated_mem,  deduplicated_mem_input = self.mailbox.prep_input_mails(self.deduplication, mfgss_on_gpu[0], node_uniq, use_pinned_buffers=True)
                    # def prep_input_mails(self, deduplication, mfg, node_uniq, use_pinned_buffers=False):

            self.async_mfgs = mfgss_on_gpu
            self.node_feats_on_gpu = node_feats_on_gpu
            self.edge_feats_on_gpu = edge_feats_on_gpu
            self.node_uniq = node_uniq
            self.edge_uniq = edge_uniq
            self.deduplicated_mem = deduplicated_mem if self.mailbox is not None else None
            self.deduplicated_mem_input = deduplicated_mem_input if self.mailbox is not None else None

    def cleanup(self):
        # clear data on GPU side
        self.async_mfgs = None
        self.node_feats_on_gpu = None
        self.edge_feats_on_gpu = None
        self.node_uniq = None
        self.edge_uniq = None
        self.deduplicated_mem = None
        self.deduplicated_mem_input = None

        # clear data on CPU side
        self.received_data = None
        self.mfgss_on_cpu = None
