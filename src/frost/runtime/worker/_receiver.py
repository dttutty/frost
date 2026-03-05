import logging
import pickle
import threading
from queue import Queue as Q
from typing import Union

import torch
import torch.distributed as dist

from frost.aliases.worker import ReceivedQ
from frost.datatypes.env import Env
from frost.datatypes.graph import FlatGraph
from frost.datatypes.meta import Meta
from frost.datatypes.result import Result
from frost.runtime._thread_base import ThreadBase
from frost.runtime._wire import ControlWire, unpack_wire_message, wire_to_meta
from frost.shm.ring_buf import RingBuf

logger = logging.getLogger(__name__)


class Receiver(ThreadBase):
    def __init__(
        self,
        env: Env,
        received_q: ReceivedQ,
        result_q: Q[Result],
        stop_ev: threading.Event,
        ring_buf: RingBuf,
    ) -> None:
        super().__init__(name="receiver", stop_ev=stop_ev)
        self.env: Env = env
        self.received_q = received_q
        self.stop_ev = stop_ev
        self.result_q = result_q
        self._gather_seq = 0
        self._recv_seq = 0
        self.ring_buf = ring_buf

    def _send_result(self, result: Result) -> None:
        payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        nbytes = torch.tensor([len(payload)], dtype=torch.long)
        dist.send(nbytes, dst=self.env.host_rank)
        if nbytes.item() > 0:
            data = torch.tensor(list(payload), dtype=torch.uint8)
            dist.send(data, dst=self.env.host_rank)

    @torch.no_grad()
    def logic(self) -> None:
        batch_count = 0

        while not self.stop_ev.is_set():
            # self.manifest_buf = [None] * MANIFEST_SIZE
            nbytes = torch.empty(1, dtype=torch.long)
            dist.recv(nbytes, src=self.env.host_rank)
            payload = bytes()
            if nbytes.item() > 0:
                payload_tensor = torch.empty(int(nbytes.item()), dtype=torch.uint8)
                dist.recv(payload_tensor, src=self.env.host_rank)
                payload = bytes(payload_tensor.tolist())

            msg: ControlWire = unpack_wire_message(payload)
            if msg["kind"] == "meta":
                meta: Union[dict, str, None] = msg["meta"]
            elif msg["kind"] == "end_of_epoch":
                meta = "end_of_epoch"
            else:
                meta = None
            self._recv_seq += 1

            if meta is None:
                self.received_q.put(None, block=True)
                break
            elif meta == "end_of_epoch":
                self.received_q.put("end_of_epoch", block=True)
                result = self.result_q.get(block=True)

                self._gather_seq += 1
                self._send_result(result)

            else:
                assert isinstance(meta, dict)
                meta = wire_to_meta(meta)
                assert isinstance(meta, Meta)
                batch_count += 1
                buf = self.ring_buf.get(meta.buf_len, block=True)
                if buf is None:
                    raise RuntimeError(
                        f"Ring buffer returned None for meta with batch_id {meta.batch_idx}"
                    )
                buf_len = meta.buf_len
                # --- Strict sanity checks ---
                nid_slice = buf["NID"][:buf_len]
                eid_slice = buf["EID"][:buf_len]

                n_src = int(meta.n_src)
                if n_src < 0:
                    raise ValueError(f"Invalid meta.n_src={n_src}")

                need = 3 * n_src
                if need > nid_slice.numel():
                    raise ValueError(
                        f"Invalid meta.n_src={n_src}: need 3*n_src={need} <= buf_len={nid_slice.numel()} for NID checks."
                    )
                if need > eid_slice.numel():
                    raise ValueError(
                        f"Invalid meta.n_src={n_src}: need 3*n_src={need} <= buf_len={eid_slice.numel()} for EID checks."
                    )

                # 1) nid[:3*n_src] 不允许出现 -1
                nid_head = nid_slice[:need]
                if (nid_head == -1).any():
                    bad_pos = torch.nonzero(nid_head == -1, as_tuple=False).squeeze(-1)
                    preview_pos = bad_pos[:20].tolist()
                    raise ValueError(
                        f"Invalid flat_graph.nid: found -1 in nid[:3*n_src] "
                        f"(n_src={n_src}, range=[0, {need})). First {len(preview_pos)} positions: {preview_pos}"
                    )

                # 2) eid[:n_src] 不允许出现 -1
                eid_head = eid_slice[:n_src]
                if (eid_head == -1).any():
                    bad_pos = torch.nonzero(eid_head == -1, as_tuple=False).squeeze(-1)
                    preview_pos = bad_pos[:20].tolist()
                    raise ValueError(
                        f"Invalid flat_graph.eid: found -1 in eid[:n_src] (n_src={n_src}). "
                        f"First {len(preview_pos)} positions: {preview_pos}"
                    )

                # 3) eid[n_src:3*n_src] 必须全是 -1
                eid_mid = eid_slice[n_src : 3 * n_src]
                if eid_mid.numel() > 0 and not (eid_mid == -1).all():
                    bad_pos = torch.nonzero(eid_mid != -1, as_tuple=False).squeeze(-1)
                    preview_pos = bad_pos[:20].tolist()
                    bad_vals = eid_mid[bad_pos[:20]].tolist()
                    raise ValueError(
                        f"Invalid flat_graph.eid: expected all -1 in eid[n_src:3*n_src] "
                        f"(n_src={n_src}, range=[{n_src}, {3 * n_src})). "
                        f"First {len(preview_pos)} offending positions (local): {preview_pos}, values: {bad_vals}"
                    )
                # --- end checks ---
                meta.flat_graph = FlatGraph(
                    nid=buf["NID"][:buf_len].clone().contiguous(),
                    eid=buf["EID"][:buf_len].clone().contiguous(),
                    ts=buf["TS"][:buf_len].clone().contiguous(),
                )
                buf = None
                self.ring_buf.release()
                self.received_q.put(meta, block=True)
