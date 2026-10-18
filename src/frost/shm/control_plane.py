"""Shared-memory ring for host-to-worker control messages."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from frost.datatypes.env import Env
from frost.runtime.control_plane import ControlMessageKind

from ._core import create_shared_mem_array, get_shared_mem_array


def control_payload_capacity(
    *, n_gpus: int, node_capacity: int, mask_capacity: int
) -> int:
    """Conservative encoded-plan capacity for one worker batch."""
    if n_gpus < 1 or node_capacity < 1 or mask_capacity < 1:
        raise ValueError("Control-plane capacity inputs must be positive.")
    # Each of memory and mailbox may contain n_gpus P2P node lists plus fetch
    # and writeback lists. Static features add cache/fetch, while the negative
    # mask is bounded independently by the sampled graph layout capacity.
    scalar_fields = 21 + 6 * n_gpus
    return (
        scalar_fields
        + (2 * n_gpus + 6) * node_capacity
        + mask_capacity
    )


class SharedControlPlane:
    """Per-worker SPSC control rings backed by POSIX shared memory."""

    def __init__(
        self,
        *,
        env: Env,
        run_id: str,
        payload_capacity: int,
        max_size: int = 2,
        poll_interval_s: float = 0.00005,
    ) -> None:
        if payload_capacity < 1:
            raise ValueError("payload_capacity must be positive.")
        if max_size < 1:
            raise ValueError("max_size must be positive.")
        self.env = env
        self.n_gpu = env.n_gpu
        self.create = env.is_host
        self.gpu_id = None if env.is_host else env.local_rank
        self.payload_capacity = payload_capacity
        self.max_size = max_size
        self.poll_interval_s = poll_interval_s
        prefix = f"{run_id}_" if run_id else ""
        head_name = f"{prefix}CONTROL_HEAD"
        tail_name = f"{prefix}CONTROL_TAIL"
        self.headers: list[list[torch.Tensor]] = [
            [torch.empty(0) for _ in range(max_size)] for _ in range(self.n_gpu)
        ]
        self.payloads: list[list[torch.Tensor]] = [
            [torch.empty(0) for _ in range(max_size)] for _ in range(self.n_gpu)
        ]

        if self.create:
            self.head = create_shared_mem_array(
                head_name, (self.n_gpu,), torch.int64
            )
            self.tail = create_shared_mem_array(
                tail_name, (self.n_gpu,), torch.int64
            )
            self.head.zero_()
            self.tail.zero_()
            for gpu_id in range(self.n_gpu):
                for slot in range(max_size):
                    self.headers[gpu_id][slot] = create_shared_mem_array(
                        f"{prefix}CONTROL_gpu{gpu_id}_s{slot}_HEADER",
                        (2,),
                        torch.int64,
                    )
                    self.payloads[gpu_id][slot] = create_shared_mem_array(
                        f"{prefix}CONTROL_gpu{gpu_id}_s{slot}_PAYLOAD",
                        (payload_capacity,),
                        torch.int64,
                    )
            dist.barrier()
        else:
            dist.barrier()
            self.head = get_shared_mem_array(
                head_name, (self.n_gpu,), torch.int64
            )
            self.tail = get_shared_mem_array(
                tail_name, (self.n_gpu,), torch.int64
            )
            for gpu_id in range(self.n_gpu):
                for slot in range(max_size):
                    self.headers[gpu_id][slot] = get_shared_mem_array(
                        f"{prefix}CONTROL_gpu{gpu_id}_s{slot}_HEADER",
                        (2,),
                        torch.int64,
                    )
                    self.payloads[gpu_id][slot] = get_shared_mem_array(
                        f"{prefix}CONTROL_gpu{gpu_id}_s{slot}_PAYLOAD",
                        (payload_capacity,),
                        torch.int64,
                    )
        dist.barrier()

    def publish(
        self,
        gpu_id: int,
        kind: ControlMessageKind,
        payload: torch.Tensor | None = None,
    ) -> None:
        """Publish one message from the host into a worker's control ring."""
        if not self.create or self.gpu_id is not None:
            raise RuntimeError("Only the host may publish control messages.")
        if not 0 <= gpu_id < self.n_gpu:
            raise IndexError(f"Invalid worker GPU ID {gpu_id}.")
        length = 0 if payload is None else payload.numel()
        if length > self.payload_capacity:
            raise ValueError(
                f"Control payload has {length} values; capacity is "
                f"{self.payload_capacity}."
            )
        while int(self.head[gpu_id].item()) - int(self.tail[gpu_id].item()) >= self.max_size:
            time.sleep(self.poll_interval_s)
        head = int(self.head[gpu_id].item())
        slot = head % self.max_size
        if payload is not None and length:
            self.payloads[gpu_id][slot][:length].copy_(payload)
        header = self.headers[gpu_id][slot]
        header[0] = int(kind)
        header[1] = length
        self.head[gpu_id] += 1

    def receive(self) -> tuple[ControlMessageKind, torch.Tensor]:
        """Wait for and return the next worker-local message payload view."""
        if self.create or self.gpu_id is None:
            raise RuntimeError("Only workers may receive control messages.")
        gpu_id = self.gpu_id
        while int(self.head[gpu_id].item()) <= int(self.tail[gpu_id].item()):
            time.sleep(self.poll_interval_s)
        tail = int(self.tail[gpu_id].item())
        slot = tail % self.max_size
        header = self.headers[gpu_id][slot]
        kind = ControlMessageKind(int(header[0].item()))
        length = int(header[1].item())
        if not 0 <= length <= self.payload_capacity:
            raise RuntimeError(
                f"Corrupt control payload length {length}; capacity is "
                f"{self.payload_capacity}."
            )
        return kind, self.payloads[gpu_id][slot][:length]

    def release(self) -> None:
        """Release the current worker-local slot after decoding it."""
        if self.create or self.gpu_id is None:
            raise RuntimeError("Only workers may release control messages.")
        self.tail[self.gpu_id] += 1
