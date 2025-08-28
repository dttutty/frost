import torch
import threading
from utils import mfgs_to_cuda, prepare_mfgs, get_ids
from memorys import MemoryMailbox
import pickle
import torch.distributed as dist
local_rank = dist.get_rank()
world_size = dist.get_world_size()
host_rank = world_size - 1
def scatter_receive(src=host_rank):
    my = [None]
    dist.scatter_object_list(my, [None] * world_size, src=src)
    return my[0]

import  torch.distributed as dist
class DataPipelineThread(threading.Thread):
    def __init__(self, deduplication, data_received_event: threading.Event, host_rank, stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox: MemoryMailbox, memory_param, deduplicate_mfg_src, prev_mailbox_memory_update_thread):
        
        
        super(DataPipelineThread, self).__init__()
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
        self.received_data = scatter_receive()
        self.data_received_event.set()
        self.raw_mfgs = self.received_data.mfgs
        
        with torch.cuda.stream(self.stream):
            nids, eids = get_ids(self.raw_mfgs, self.node_feats, self.edge_feats)
            gnn_mfgs = mfgs_to_cuda(self.raw_mfgs)
            prepare_mfgs(self.deduplication, gnn_mfgs, self.node_feats, self.edge_feats, deduplicate_mfg_src=self.deduplicate_mfg_src, pinned=True, nfeat_buffs=self.pinned_nfeat_buffs, efeat_buffs=self.pinned_efeat_buffs, nids=nids, eids=eids)
            self.prev_mailbox_memory_update_thread.join() if self.prev_mailbox_memory_update_thread is not None else None
            if self.mailbox is not None:
                self.mailbox.prep_input_mails(gnn_mfgs[0], use_pinned_buffers=True)
            self.async_mfgs = gnn_mfgs

    def clear_tensors(self):
        self.async_mfgs.clear()
        # self.raw_mfgs.clear()


class MailboxUpdateThread(threading.Thread):
    def __init__(self, deduplication, memory_mailbox: MemoryMailbox, nid, memory, root_nodes, ts, mem_edge_feats, block):
        super().__init__()
        # --- Store all the necessary data for the update ---
        self.deduplication = deduplication
        self.memory_mailbox = memory_mailbox
        self.nid = nid
        self.memory = memory
        self.root_nodes = root_nodes
        self.ts = ts
        self.mem_edge_feats = mem_edge_feats
        self.block = block

    def run(self):
        """
        The main logic of the thread.
        This part can be executed in parallel with other operations.
        """
        # These two operations are safe to run in a separate thread.
        # They primarily involve tensor manipulations that PyTorch can handle concurrently.
        self.memory_mailbox.update_mailbox(self.nid, self.memory, self.root_nodes, self.ts, self.mem_edge_feats, self.block)
        self.memory_mailbox.update_memory(self.deduplication, self.nid, self.memory, self.root_nodes, self.ts)

        