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
    mfgs: Optional[List[List[DGLBlock]]] = None
    root: Optional[NDArray[np.int32]] = None
    ts: Optional[NDArray[np.float32]] = None
    eid: Optional[List] = None
    block: Optional[List[List[DGLBlock]]] = None


class DataMover(threading.Thread):
    def __init__(
        self,
        deduplication,
        data_received_event: threading.Event,
        host_rank,
        stream,
        pinned_nfeat_buffs,
        pinned_efeat_buffs,
        node_feats,
        edge_feats,
        mailbox: MemoryMailbox,
        memory_param,
        deduplicate_mfg_src,
        prev_mailbox_mem_updater,
        prev_node_uniq=None,
        prev_edge_uniq=None,
        prev_node_feats_on_gpu=None,
        prev_edge_feats_on_gpu=None,
    ):

        super(DataMover, self).__init__()
        self.host_rank = host_rank
        self.data_received_event = data_received_event

        self.stream = stream
        self.mfgss_on_gpu = None
        self.pinned_nfeat_buffs = pinned_nfeat_buffs
        self.pinned_efeat_buffs = pinned_efeat_buffs
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.mailbox = mailbox
        self.memory_param = memory_param
        self.deduplicate_mfg_src = deduplicate_mfg_src
        self.prev_mailbox_mem_updater = prev_mailbox_mem_updater
        self.deduplication = deduplication
        self.prev_node_uniq = prev_node_uniq
        self.prev_edge_uniq = prev_edge_uniq
        self.prev_node_feats_on_gpu = prev_node_feats_on_gpu
        self.prev_edge_feats_on_gpu = prev_edge_feats_on_gpu

    def run(self):
        self.received_data: TransferData = scatter_receive()
        self.data_received_event.set()
        mfgss_on_cpu = self.received_data.mfgs

        with torch.cuda.stream(self.stream):
            nids, eids = get_ids(mfgss_on_cpu, self.node_feats, self.edge_feats)
            mfgss_on_gpu = mfgs_to_cuda(mfgss_on_cpu)

            (
                self.mfgss_on_gpu,
                self.node_feats_on_gpu,
                self.edge_feats_on_gpu,
                self.node_uniq,
                self.edge_uniq,
            ) = prepare_mfgs(
                self.deduplication,
                mfgss_on_cpu,
                mfgss_on_gpu,
                self.node_feats,
                self.edge_feats,
                deduplicate_mfg_src=self.deduplicate_mfg_src,
                pinned=True,
                nfeat_buffs=self.pinned_nfeat_buffs,
                efeat_buffs=self.pinned_efeat_buffs,
                prev_node_uniq=self.prev_node_uniq,
                prev_edge_uniq=self.prev_edge_uniq,
                prev_node_feats_on_gpu=self.prev_node_feats_on_gpu,
                prev_edge_feats_on_gpu=self.prev_edge_feats_on_gpu,
            )
            self.prev_datamove_thread = None
            if self.prev_mailbox_mem_updater is not None:
                self.prev_mailbox_mem_updater.join()
                self.prev_mailbox_mem_updater.cleanup()
                self.prev_mailbox_mem_updater = None
            if self.mailbox is not None:
                deduplicated_mem, deduplicated_mem_input = (
                    self.mailbox.prep_input_mails(
                        self.deduplication,
                        mfgss_on_gpu[0],
                        self.node_uniq,
                        use_pinned_buffers=True,
                    )
                )

            self.deduplicated_mem = (
                deduplicated_mem if self.mailbox is not None else None
            )
            self.deduplicated_mem_input = (
                deduplicated_mem_input if self.mailbox is not None else None
            )

    def cleanup(self):
        # finish all work enqueued on this stream
        if getattr(self, "stream", None) is not None:
            self.stream.synchronize()

        # clear data on GPU side
        self.mfgss_on_gpu = None
        self.node_feats_on_gpu = None
        self.edge_feats_on_gpu = None
        self.node_uniq = None
        self.edge_uniq = None
        self.deduplicated_mem = None
        self.deduplicated_mem_input = None

        self.mfgss_on_gpu = None
        self.pinned_nfeat_buffs = None
        self.pinned_efeat_buffs = None
        self.node_feats = None
        self.edge_feats = None
        self.mailbox = None
        self.memory_param = None
        self.deduplicate_mfg_src = None
        self.prev_mailbox_mem_updater = None
        self.deduplication = None
        self.prev_node_uniq = None
        self.prev_edge_uniq = None
        self.prev_node_feats_on_gpu = None
        self.prev_edge_feats_on_gpu = None

        # clear data on CPU side
        self.received_data = None
        self.mfgss_on_cpu = None

        # optional: drop the stream itself if this object won't be reused
        # if getattr(self, "stream", None) is not None:
        #     s = self.stream
        #     self.stream = None
        #     del s
