from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from frost.shm._core import create_shared_mem_array as csm
from frost.shm._core import get_shared_mem_array as gsm


@dataclass(kw_only=True)
class MemBoxStore:
    """一个专门存放所有共享内存张量的数据类"""

    mem: torch.Tensor
    mem_ts: torch.Tensor
    box: torch.Tensor
    box_ts: torch.Tensor

    def reset(self) -> None:
        """将所有共享内存张量清零"""
        self.mem.zero_()
        self.mem_ts.zero_()
        self.box.zero_()
        self.box_ts.zero_()


class MemBoxStoreManager:
    def __init__(
        self,
        env,
        run_id: str,
        n_nodes: int,
        boxsize: int,  # from hydra config
        d_mem: int,  # from hydra config
        edim: int,  # from hydra config
        mdtype: torch.dtype,
    ):
        self.env = env
        self.run_id = run_id
        self.mdtype = mdtype
        self.n_nodes = n_nodes
        self.boxsize = boxsize
        self.d_mem = d_mem
        self.d_mail = 2 * d_mem + edim

        # Initialize internal state
        self.state: Optional[MemBoxStore] = None

    def prepare(self):
        """
        Host creates shared memory, workers attach.
        Populates internal state.
        """
        n_nodes = self.n_nodes
        boxsize = self.boxsize
        d_mem = self.d_mem
        d_mail = self.d_mail

        mail_shape = [n_nodes, boxsize, d_mail]

        # Initialize local variables to avoid UnboundLocalError
        mem = mem_ts = box = box_ts = None

        if self.env.is_host:
            mem = csm(f"{self.run_id}_mem", [n_nodes, d_mem], self.mdtype)
            mem_ts = csm(f"{self.run_id}_mem_ts", [n_nodes], self.mdtype)
            box = csm(f"{self.run_id}_mail", mail_shape, self.mdtype)
            box_ts = csm(f"{self.run_id}_mail_ts", [n_nodes, boxsize], self.mdtype)

            # Initialize memory states to zero
            mem.zero_()
            mem_ts.zero_()
            box.zero_()
            box_ts.zero_()

        dist.barrier()

        if not self.env.is_host:
            mem = gsm(f"{self.run_id}_mem", [n_nodes, d_mem], self.mdtype)
            mem_ts = gsm(f"{self.run_id}_mem_ts", [n_nodes], self.mdtype)
            box = gsm(f"{self.run_id}_mail", mail_shape, self.mdtype)
            box_ts = gsm(f"{self.run_id}_mail_ts", [n_nodes, boxsize], self.mdtype)

        # Store the dataclass as an instance attribute
        assert mem is not None
        assert mem_ts is not None
        assert box is not None
        assert box_ts is not None
        self.state = MemBoxStore(mem=mem, mem_ts=mem_ts, box=box, box_ts=box_ts)

        dist.barrier()

        # Return self for method chaining, matching DatasetManager
        return self.state
