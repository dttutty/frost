import threading
from memorys import MemoryMailbox


class MailboxMemUpdater(threading.Thread):
    # def __init__(
    #     self,
    #     deduplication: bool,
    #     memory_mailbox: MemoryMailbox,
    #     nid,
    #     memory,
    #     root_nodes,
    #     ts,
    #     mem_edge_feats,
    #     block,
    # ):
    #     super().__init__()
    #     # --- Store all the necessary data for the update ---
    #     self.deduplication = deduplication
    #     self.memory_mailbox = memory_mailbox
    #     self.nid = nid
    #     self.memory = memory
    #     self.root_nodes = root_nodes
    #     self.ts = ts
    #     self.mem_edge_feats = mem_edge_feats
    #     self.block = block

    def run(self):
        """
        The main logic of the thread.
        This part can be executed in parallel with other operations.
        """
        # These two operations are safe to run in a separate thread.
        # They primarily involve tensor manipulations that PyTorch can handle concurrently.
        self.memory_mailbox.update_mailbox(
            self.nid,
            self.memory,
            self.root_nodes,
            self.ts,
            self.mem_edge_feats,
            self.block,
        )
        self.memory_mailbox.update_memory(
            self.deduplication, self.nid, self.memory, self.root_nodes, self.ts
        )

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

        self.deduplication = None
        self.memory_mailbox = None


from queue import Queue
from dataclasses import dataclass
import threading
import torch


@dataclass
class MailTask:
    deduplication: bool
    nid: torch.Tensor  # CPU
    memory: torch.Tensor  # CPU
    root_nodes: torch.Tensor  # CPU
    ts: torch.Tensor  # CPU
    mem_edge_feats: torch.Tensor  # CPU
    block: object  # CPU


STOP = object()


# class MailboxWorker(threading.Thread):
#     def __init__(self, q, memory_mailbox: MemoryMailbox):
#         super().__init__(daemon=False)
#         self.q, self.memory_mailbox = q, memory_mailbox

#     def run(self):
#         while True:
#             task = self.q.get()
#             if task is STOP:
#                 break
#             with torch.no_grad():
#                 self.memory_mailbox.update_mailbox(
#                     task.mid,  # microbatch_id
#                     task.nid,
#                     task.memory,
#                     task.root_nodes,
#                     task.ts,
#                     task.mem_edge_feats,
#                     task.block,
#                 )
#                 self.memory_mailbox.update_memory(
#                     task.deduplication, task.nid, task.memory, task.root_nodes, task.ts
#                 )
#             self.q.task_done()


# mail_q = Queue(maxsize=4)  # 有界队列 → 背压，防止堆积
# mail_worker = MailboxWorker(mail_q, memory_mailbox)
# mail_worker.start()

# # 训练循环里：
# # cpu_nid, cpu_mem, cpu_ts = last_updated_*.detach().cpu()
# mail_q.put(
#     MailTask(
#         deduplication,
#         nid,
#         memory,
#         root_nodes,
#         ts,
#         mem_edge_feats,
#         block,
#     )
# )

# # 训练结束：
# mail_q.put(STOP)
# mail_q.join()
# mail_worker.join()
