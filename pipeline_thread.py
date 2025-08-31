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
            _, node_feats_on_gpu, edge_feats_on_gpu, node_uniq, edge_uniq = prepare_mfgs(self.deduplication, self.raw_mfgs, gnn_mfgs, self.node_feats, self.edge_feats, deduplicate_mfg_src=self.deduplicate_mfg_src, pinned=True, nfeat_buffs=self.pinned_nfeat_buffs, efeat_buffs=self.pinned_efeat_buffs, nids=nids, eids=eids)
            if self.prev_mailbox_memory_update_thread is not None:
                self.prev_mailbox_memory_update_thread.join()
                self.prev_mailbox_memory_update_thread.cleanup()
                self.prev_mailbox_memory_update_thread = None
            if self.mailbox is not None:
                deduplicated_mem,  deduplicated_mem_input = self.mailbox.prep_input_mails(self.deduplication, gnn_mfgs[0], node_uniq, use_pinned_buffers=True)
                    # def prep_input_mails(self, deduplication, mfg, node_uniq, use_pinned_buffers=False):

            self.async_mfgs = gnn_mfgs
            self.node_feats_on_gpu = node_feats_on_gpu
            self.edge_feats_on_gpu = edge_feats_on_gpu
            self.node_uniq = node_uniq
            self.edge_uniq = edge_uniq
            self.deduplicated_mem = deduplicated_mem if self.mailbox is not None else None
            self.deduplicated_mem_input = deduplicated_mem_input if self.mailbox is not None else None

    def cleanup(self):
        self.async_mfgs = None
        self.node_feats_on_gpu = None
        self.edge_feats_on_gpu = None
        self.node_uniq = None
        self.edge_uniq = None
        self.deduplicated_mem = None
        self.deduplicated_mem_input = None

        # --- 同时清理在CPU上的大型原始数据 ---
        # 这些对象也可能占用大量RAM
        self.received_data = None
        self.raw_mfgs = None

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

    def cleanup(self):
        """
        Explicitly clear all tensor references held by this thread object.
        """
        self.nid = None
        self.memory = None
        self.root_nodes = None
        self.ts = None
        self.mem_edge_feats = None
        self.block = None
        # 其他可能持有的张量也在这里设为 None
