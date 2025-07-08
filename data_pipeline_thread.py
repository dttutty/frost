import torch
import threading
from utils import mfgs_to_cuda, prepare_mfgs, get_ids
from memorys import MemoryMailbox
import pickle


import  torch.distributed as dist
class DataPipelineThread(threading.Thread):
    def __init__(self, host_rank, recv_request, buffer, stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox: MemoryMailbox, memory_param, deduplicate_mfg_src):
        super(DataPipelineThread, self).__init__()
        self.recv_request = recv_request
        self.buffer = buffer
        self.received_data = None
        self.host_rank = host_rank
        
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
        # 1. 等待主数据接收完成
        self.recv_request.wait()
        
        # 2. 发送 ACK，并【等待】它完成
        ack_tensor = torch.tensor([1], dtype=torch.int32, device='cpu') 
        
        # isend 返回一个请求对象
        ack_request = dist.isend(ack_tensor, dst=self.host_rank)
        
        # !!! 这是最关键的修正：等待 ACK 发送操作完成 !!!
        ack_request.wait()
        
        # 3. 现在可以安全地继续执行或结束线程了
        print(f"Rank {torch.distributed.get_rank()} received data, starting processing")
        

        self.received_data = pickle.loads(bytes(self.buffer.tolist()))
        self.raw_mfgs = self.received_data.mfgs
        
        with torch.cuda.stream(self.stream):
            # print(local_rank, 'start thread')
            nids, eids = get_ids(self.raw_mfgs, self.node_feats, self.edge_feats)
            gnn_mfgs = mfgs_to_cuda(self.raw_mfgs)
            prepare_mfgs(gnn_mfgs, self.node_feats, self.edge_feats, deduplicate_mfg_src=self.deduplicate_mfg_src, pinned=True, nfeat_buffs=self.pinned_nfeat_buffs, efeat_buffs=self.pinned_efeat_buffs, nids=nids, eids=eids)
            if self.mailbox is not None:
                self.mailbox.prep_input_mails(gnn_mfgs[0], use_pinned_buffers=True)
            self.async_mfgs = gnn_mfgs

    def clear_tensors(self):
        self.async_mfgs.clear()
        # self.raw_mfgs.clear()
