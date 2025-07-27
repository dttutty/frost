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
    def __init__(self, received_data, host_rank, stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox: MemoryMailbox, memory_param, deduplicate_mfg_src):
        
        
        super(DataPipelineThread, self).__init__()
        self.received_data = received_data
        self.host_rank = host_rank
        # self.data_received_event = data_received_event
        
        self.mfgs_on_cpu = self.received_data.mfgs
        # self.received_data = received_data
        self.stream = stream
        self.async_mfgs = None
        self.pinned_nfeat_buffs = pinned_nfeat_buffs
        self.pinned_efeat_buffs = pinned_efeat_buffs
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.mailbox = mailbox
        self.memory_param = memory_param
        self.deduplicate_mfg_src = deduplicate_mfg_src

    def run(self):
        # # 1. 等待主数据接收完成
        # self.recv_request.wait()
        
        # # 2. 发送 ACK，并【等待】它完成
        # ack_tensor = torch.tensor([1], dtype=torch.int32, device='cpu') 
        
        # # isend 返回一个请求对象
        # ack_request = dist.isend(ack_tensor, dst=self.host_rank)
        
        # # !!! 这是最关键的修正：等待 ACK 发送操作完成 !!!
        # ack_request.wait()
        
        # # 3. 现在可以安全地继续执行或结束线程了
        # print(f"Rank {torch.distributed.get_rank()} received data, starting processing")
        # self.received_data = scatter_receive()
        # self.data_received_event.set()

        # self.received_data = pickle.loads(bytes(self.buffer.tolist()))
        
        with torch.cuda.stream(self.stream):
            # print(local_rank, 'start thread')
            nids, eids = get_ids(self.mfgs_on_cpu, self.node_feats, self.edge_feats)
            mfgs_on_gpu = mfgs_to_cuda(self.mfgs_on_cpu)
            mfgs, node_feats_on_gpu, edge_feats_on_gpu = prepare_mfgs(self.mfgs_on_cpu, mfgs_on_gpu, self.node_feats, self.edge_feats, deduplicate_mfg_src=self.deduplicate_mfg_src, pinned=True, nfeat_buffs=self.pinned_nfeat_buffs, efeat_buffs=self.pinned_efeat_buffs, nids=nids, eids=eids)
            if self.mailbox is not None:
                self.mailbox.prep_input_mails(mfgs_on_gpu[0], use_pinned_buffers=True)
            self.async_mfgs = mfgs_on_gpu

    def clear_tensors(self):
        self.async_mfgs.clear()
        # self.raw_mfgs.clear()
