import logging
from typing import TypeVar

import numpy
import torch

from frost.datatypes.mem_mail_box.box import BoxTsNid
from frost.datatypes.mem_mail_box.mail import MailTsNid
from frost.datatypes.mem_mail_box.mem import MemTsNid
from frost.datatypes.plan import DyPlan
from frost.datatypes.tasks import P2PTask

AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")
logger = logging.getLogger(__name__)


def make_p2p_task(
    data_td_nid: AllType,
    plan: DyPlan,
    device: torch.device,
) -> P2PTask:
    """
    这一步之前,必须确保data_td_nid里面的数据全部是unique的
    如果是mem,需要确保已经是保留最新的(last),或者平均的(mean,带ct)
    如果是mail,需要确保只保留更新过的节点(pos_src,pos_dst).....另外保留最新last,平均的(mean,带ct),或者全部(dim变大)
    如果是mailbox,需要所有节点的mailbox
    如果是feat,需要所有节点的feat
    """
    p2p = plan.p2p
    rec_cts = plan.rec_cts
    send_cts = plan.send_cts

    nid = data_td_nid.nid
    data = data_td_nid.data
    ts = data_td_nid.ts

    x_dim = data.shape[1]

    buckets_ids, buckets_data, buckets_ts = [], [], []

    sizes = numpy.zeros(len(p2p), dtype=int)
    for dst_gpu_id, _ in enumerate(p2p):
        nids_to_send_tensor = p2p[dst_gpu_id].data
        send_mask_for_dst = torch.isin(nid, nids_to_send_tensor)
        matched_count = send_mask_for_dst.sum().item()
        expected_count = nids_to_send_tensor.numel()
        assert matched_count == expected_count, (
            f"make_p2p_task: GPU {dst_gpu_id} expected to send {expected_count} nodes, "
            f"but found {matched_count} in data (total {nid.numel()})."
        )
        buckets_ids.append(nid[send_mask_for_dst])
        buckets_data.append(data[send_mask_for_dst])
        buckets_ts.append(ts[send_mask_for_dst])
        sizes[dst_gpu_id] = send_mask_for_dst.sum().item()
    send_nid = torch.cat(buckets_ids, dim=0)
    send_data = torch.cat(buckets_data, dim=0)
    send_ts = torch.cat(buckets_ts, dim=0)

    total_recv = sum(rec_cts)
    recv_nid = torch.empty(total_recv, dtype=nid.dtype, device=device)
    recv_data = torch.empty(total_recv, x_dim, dtype=data.dtype, device=device)
    recv_ts = torch.empty(total_recv, dtype=ts.dtype, device=device)

    assert numpy.array_equal(sizes, send_cts)

    return P2PTask(
        nid2send=send_nid,
        nid2recv=recv_nid,
        data2send=send_data,
        data2recv=recv_data,
        ts2send=send_ts,
        ts2recv=recv_ts,
        rec_cts=rec_cts,
        send_cts=send_cts,
    )
