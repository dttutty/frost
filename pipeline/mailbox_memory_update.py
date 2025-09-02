import threading
from memorys import MemoryMailbox


class update_mailbox_memory(threading.Thread):
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
