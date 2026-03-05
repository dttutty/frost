import logging
import time

import torch
import torch.distributed as dist

# Assuming you have this env datatype defined
from frost.datatypes.env import Env

from ._core import create_shared_mem_array, get_shared_mem_array

logger = logging.getLogger(__name__)


def cal_max_buf_size(
    n_src: int, fanout: list[int] | None, n_neg: int, sample_4_neg: bool = True
) -> int:
    total = 0
    if not sample_4_neg:  # save space by not sampling neighbors for negative nodes
        if fanout and len(fanout) > 1:
            logger.warning("sample_4_neg is False, fanout layer cannot be multi-hop.")

        # Initial seed nodes
        total += n_neg
        current_layer_nodes = n_src * 2

    current_layer_nodes = n_src * (2 + n_neg)
    total += current_layer_nodes

    # Multiply progressively for multi-hop
    if fanout is not None:
        for ratio in fanout:
            current_layer_nodes *= ratio
            total += current_layer_nodes

    return total


class RingBuf:
    def __init__(
        self,
        env: Env,
        run_id: str,
        max_batch_size: int,  # from hydra config
        fanout: list[int] | None,  # from hydra config
        n_neg: int,  # from hydra config
        sample_4_neg: bool,  # from hydra config
        max_size: int,  # from hydra config
        ndtype: torch.dtype,
        edtype: torch.dtype,
    ) -> None:
        self.env = env
        self.n_gpu = env.n_gpu
        self.max_size = max_size
        self.create = env.is_host
        self.namespace = run_id
        self.gpu_id = env.local_rank if not env.is_host else None
        self.ndtype = ndtype
        self.edtype = edtype
        self.sample_4_neg = sample_4_neg

        total = cal_max_buf_size(max_batch_size, fanout, n_neg, sample_4_neg)

        self.schema = {
            "NID": ((total,), ndtype),
            "TS": ((total,), torch.float32),
            "EID": ((total,), edtype),
        }

        prefix = f"{self.namespace}_" if self.namespace else ""
        head_name = f"{prefix}TGL_SHM_HEAD"
        tail_name = f"{prefix}TGL_SHM_TAIL"

        # Initialize structure
        self.buffers: list[list[dict[str, torch.Tensor]]] = [
            [{} for _ in range(self.max_size)] for _ in range(self.n_gpu)
        ]

        if self.create:
            # Host logic: create all SHM pointers and buffers
            self.head = create_shared_mem_array(head_name, (self.n_gpu,), torch.int64)
            self.tail = create_shared_mem_array(tail_name, (self.n_gpu,), torch.int64)
            self.head.zero_()
            self.tail.zero_()

            for g_idx in range(self.n_gpu):
                for buf_idx in range(self.max_size):
                    for key, (shape, dtype) in self.schema.items():
                        name = f"{prefix}TGL_SHM_g{g_idx}_b{buf_idx}_{key}"
                        self.buffers[g_idx][buf_idx][key] = create_shared_mem_array(
                            name, shape, dtype
                        )

            # Sync once after all SHM allocations are done
            dist.barrier()

        else:
            # Worker logic: wait for Host, then map all SHM pointers and buffers
            dist.barrier()

            self.head = get_shared_mem_array(head_name, (self.n_gpu,), torch.int64)
            self.tail = get_shared_mem_array(tail_name, (self.n_gpu,), torch.int64)

            for g_idx in range(self.n_gpu):
                for buf_idx in range(self.max_size):
                    for key, (shape, dtype) in self.schema.items():
                        name = f"{prefix}TGL_SHM_g{g_idx}_b{buf_idx}_{key}"
                        self.buffers[g_idx][buf_idx][key] = get_shared_mem_array(
                            name, shape, dtype
                        )

        # Final sync to ensure all workers have mapped memory
        dist.barrier()

    def acquire_slot(
        self, gpu_id: int, block: bool = True, timeout: float = -1.0
    ) -> dict[str, torch.Tensor] | None:
        assert self.gpu_id is None, "GPU ID should not be set for the host."
        """Wait for an empty slot and return its tensor references."""
        start_time = time.time()

        # Wait until there is an empty slot
        while self.head[gpu_id] - self.tail[gpu_id] >= self.max_size:
            if not block:
                return None
            if timeout > 0 and (time.time() - start_time) > timeout:
                return None

            # Sleep to prevent high CPU usage
            time.sleep(0.001)

        current_head = int(self.head[gpu_id].item())
        buf_idx = current_head % self.max_size
        return self.buffers[gpu_id][buf_idx]

    def commit_slot(self, gpu_id: int):
        """Mark the slot as ready for the consumer."""
        self.head[gpu_id] += 1

    def get(
        self,
        lengths: int,
        block: bool = True,
        timeout: float = -1.0,
    ) -> dict[str, torch.Tensor] | None:
        assert self.gpu_id is not None, "GPU ID should be set for workers."
        start_time = time.time()
        gpu_id = self.gpu_id
        assert gpu_id is not None, "GPU ID should be set for workers."

        # Wait until data is available
        while self.head[gpu_id].item() <= self.tail[gpu_id].item():
            if not block:
                return None
            if timeout > 0 and (time.time() - start_time) > timeout:
                return None
            time.sleep(0.001)

        current_tail = int(self.tail[gpu_id].item())
        buf_idx = current_tail % self.max_size
        target_slot = self.buffers[gpu_id][buf_idx]

        # Hardcode extraction for maximum speed
        return {
            "NID": target_slot["NID"][:lengths],
            "TS": target_slot["TS"][:lengths],
            "EID": target_slot["EID"][:lengths],
        }

    def release(self):
        """Mark the buffer slot as consumed."""
        assert self.gpu_id is not None, "GPU ID should be set for workers."
        self.tail[self.gpu_id] += 1
