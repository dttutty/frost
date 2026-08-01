"""Receiver thread for the worker-side pipeline.

Receives tensor-encoded ``Meta`` control messages from the host and forwards
them into the ``received_q`` for downstream H2D processing.
Epoch-end results are gathered back to the host via ``dist.gather_object``.
"""

import json
import logging
import os
import threading
import time
from queue import Empty
from queue import Queue as Q
import torch
import torch.distributed as dist

from frost.runtime.worker.types import ReceivedQ
from frost.datatypes.env import Env
from frost.datatypes.graph import FlatGraph
from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.runtime._thread_base import ThreadBase, put_shutdown_sentinel
from frost.runtime.control_plane import (
    ControlMessageKind,
    decode_meta,
)
from frost.shm.ring_buf import RingBuf
from frost.shm.control_plane import SharedControlPlane

logger = logging.getLogger(__name__)


def _timing_enabled() -> bool:
    return os.environ.get("FROST_PIPELINE_TIMING_BENCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class Receiver(ThreadBase):
    """Background thread that receives batches scattered from the host process.

    On each iteration it receives an encoded tensor payload from its shared
    control ring:

    * ``Meta`` object → assembles a ``FlatGraph`` from the shared ring buffer
      and forwards the annotated ``Meta`` to ``received_q``.
    * ``"end_of_epoch"`` → forwards the sentinel downstream, then blocks to
      gather the epoch ``Result`` and sends it back to the host.
    * ``None`` → forwards the shutdown sentinel and exits.

    Args:
        env: Distributed environment (rank/device info).
        received_q: Output queue consumed by the H2D thread.
        result_q: Single-element queue that carries the epoch ``Result`` from
            the main loop back to this thread for the gather call.
        stop_ev: Shared stop event; set by the main process on shutdown.
        ring_buf: Shared-memory ring buffer used to retrieve raw graph data.
    """

    def __init__(
        self,
        env: Env,
        received_q: ReceivedQ,
        result_q: Q[Result],
        stop_ev: threading.Event,
        ring_buf: RingBuf,
        control_plane: SharedControlPlane,
    ) -> None:
        super().__init__(name="receiver", stop_ev=stop_ev)
        self.env: Env = env
        self.received_q = received_q
        self.stop_ev = stop_ev
        self.result_q = result_q
        self._gather_seq = 0
        self._recv_seq = 0
        self.ring_buf = ring_buf
        self.control_plane = control_plane
        self._timing_enabled = _timing_enabled()
        self._scatter_wait_s_curve: list[float] = []
        self._control_decode_s_curve: list[float] = []
        self._ring_buf_wait_s_curve: list[float] = []

    @torch.no_grad()
    def logic(self) -> None:
        """Main loop: receive messages from host and route them into the pipeline.

        Runs until a ``None`` sentinel is received from the host or ``stop_ev``
        is set.
        """
        batch_count = 0

        while not self.stop_ev.is_set():
            recv_start = time.perf_counter()
            kind, shared_payload = self.control_plane.receive()
            try:
                # Decoded plans outlive the control-ring slot.  Copy the encoded
                # block once so TensorSet views remain worker-local after the host
                # reuses the shared slot, and release producer backpressure before
                # the Python object reconstruction begins.
                payload = shared_payload.clone()
            finally:
                self.control_plane.release()
            recv_end = time.perf_counter()
            decode_start = recv_end
            msg: Meta | str | None
            if kind is ControlMessageKind.BATCH:
                msg = decode_meta(payload, n_gpus=self.env.n_gpu)
            elif kind is ControlMessageKind.END_OF_EPOCH:
                msg = "end_of_epoch"
            else:
                msg = None
            decode_end = time.perf_counter()
            self._recv_seq += 1

            if msg is None:
                put_shutdown_sentinel(self.received_q, None)
                break
            elif msg == "end_of_epoch":
                self.received_q.put("end_of_epoch", block=True)
                while not self.stop_ev.is_set():
                    try:
                        result = self.result_q.get(timeout=0.1)
                        break
                    except Empty:
                        continue
                else:
                    break

                self._gather_seq += 1
                # Send result back via collective gather
                dist.gather_object(result, dst=self.env.host_rank)

            else:
                if not isinstance(msg, Meta):
                    raise TypeError(f"Unexpected control message {msg!r}.")
                meta = msg
                batch_count += 1
                ring_buf_start = time.perf_counter()
                buf = self.ring_buf.get(meta.buf_len, block=True)
                ring_buf_end = time.perf_counter()
                if buf is None:
                    raise RuntimeError(
                        f"Ring buffer returned None for meta with batch_id {meta.mini_batch_idx}"
                    )
                buf_len = meta.buf_len
                cpu_nid = buf["NID"][:buf_len].clone()
                meta.flat_graph = FlatGraph(
                    nid=cpu_nid,
                    eid=buf["EID"][:buf_len].clone(),
                    ts=buf["TS"][:buf_len].clone(),
                )
                # Neighborhood-mail routing needs the sampled CPU layout.
                meta.flat_nid_cpu = cpu_nid.clone()
                buf = None
                self.ring_buf.release()

                if self._timing_enabled:
                    self._scatter_wait_s_curve.append(recv_end - recv_start)
                    self._control_decode_s_curve.append(decode_end - decode_start)
                    self._ring_buf_wait_s_curve.append(ring_buf_end - ring_buf_start)
                self.received_q.put(meta, block=True)

        if self._timing_enabled and self._scatter_wait_s_curve:
            timing_payload = {
                "system": "frost",
                "rank": self.env.local_rank,
                "batch_count": batch_count,
                "scatter_wait_s_curve": [
                    round(v, 6) for v in self._scatter_wait_s_curve
                ],
                "control_decode_s_curve": [
                    round(v, 6) for v in self._control_decode_s_curve
                ],
                "ring_buf_wait_s_curve": [
                    round(v, 6) for v in self._ring_buf_wait_s_curve
                ],
            }
            print(
                "APPENDIX_RECEIVER_TIMING_CURVE "
                f"{json.dumps(timing_payload, sort_keys=True)}"
            )
