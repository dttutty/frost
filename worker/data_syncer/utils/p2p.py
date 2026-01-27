import logging
from queue import Queue
from typing import Literal, TypeVar

import torch
from numpy.typing import NDArray

from schemas.datatypes.data_to_sync import BoxTsNid, MailTsNid, MemTsNid
from schemas.datatypes.task import GatherElement, P2PTask


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


class P2PHandler:
    def __init__(
        self,
        n_gpu: int,
        mem_p2p_q: Queue[P2PTask | None],
        box_p2p_q: Queue[P2PTask | None],
    ) -> None:
        super().__init__()
        self.n_gpu = n_gpu
        self.mem_p2p_q = mem_p2p_q
        self.box_p2p_q = box_p2p_q

    def p2p(
        self,
        data_td_nid: AllType,
        p2p: list[torch.Tensor],
        rec_cts: NDArray,
        send_cts: NDArray,
        strategy: Literal["latest", "mean", "all"],
        target_q: Queue[P2PTask | None],
    ) -> None:
        """
        这一步之前,必须确保data_td_nid里面的数据全部是unique的
        如果是mem,需要确保已经是保留最新的(latest),或者平均的(mean,带ct)
        如果是mail,需要确保只保留更新过的节点(pos_src,pos_dst).....另外保留最新latest,平均的(mean,带ct),或者全部(dim变大)
        如果是mailbox,需要所有节点的mailbox
        如果是feat,需要所有节点的feat
        """
        device = "cuda"

        nid = data_td_nid.nid
        data = data_td_nid.data
        ts = data_td_nid.ts

        x_dim = data.shape[1]

        buckets_ids, buckets_x, buckets_ts = [], [], []

        for dst_gpu_id in range(self.n_gpu):
            nids_to_send_tensor = p2p[dst_gpu_id]
            send_mask_for_dst = torch.isin(nid, nids_to_send_tensor)

            curr_ids = nid[send_mask_for_dst]
            buckets_ids.append(nid[send_mask_for_dst])
            buckets_x.append(data[send_mask_for_dst])
            buckets_ts.append(ts[send_mask_for_dst])

        send_nid = torch.cat(buckets_ids, dim=0)
        send_x = torch.cat(buckets_x, dim=0)
        send_ts = torch.cat(buckets_ts, dim=0)

        total_recv = sum(rec_cts)

        recv_nid = torch.empty(total_recv, dtype=nid.dtype, device=device)
        recv_data = torch.empty(total_recv, x_dim, dtype=data.dtype, device=device)
        recv_ts = torch.empty(total_recv, dtype=ts.dtype, device=device)

        in_splits = send_cts
        out_splits = rec_cts
        p2p_nid = GatherElement(recv_nid, send_nid, out_splits, in_splits)
        p2p_data = GatherElement(recv_data, send_x, out_splits, in_splits)
        p2p_ts = GatherElement(recv_ts, send_ts, out_splits, in_splits)

        match data_td_nid:
            case MemTsNid():
                p2p_type = "mem"
                target_q.put(
                    P2PTask(nid=p2p_nid, data=p2p_data, ts=p2p_ts, type=p2p_type)
                )
            case MailTsNid():
                p2p_type = "mail"
                target_q.put(
                    P2PTask(nid=p2p_nid, data=p2p_data, ts=p2p_ts, type=p2p_type)
                )
            case BoxTsNid():
                p2p_type = "box"
                target_q.put(
                    P2PTask(nid=p2p_nid, data=p2p_data, ts=p2p_ts, type=p2p_type)
                )
            case _:
                raise ValueError(f"Unknown data_td_nid type: {type(data_td_nid)}")

    def skip(self, target_q: Queue[P2PTask | None]) -> None:
        target_q.put("skip")
