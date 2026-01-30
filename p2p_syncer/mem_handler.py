import logging
import threading
from queue import Queue as Q
from typing import Literal, TypeVar

import torch

from buffer import Buffers, DyBuf
from schemas.datatypes.mem_mail_box._data_ops import merge_with_priority
from schemas.datatypes.mem_mail_box.box import BoxTsNid
from schemas.datatypes.mem_mail_box.mail import MailTsNid
from schemas.datatypes.mem_mail_box.mem import MemTsNid
from schemas.datatypes.task import MemTask, P2PTask
from thread_base_with_stream import StreamThreadBase

from .utils.data_ops import make_p2p_task


AllType = TypeVar("AllType", MemTsNid, MailTsNid, BoxTsNid)
Mem_or_Mail = TypeVar("Mem_or_Mail", "MemTsNid", "MailTsNid")

logger = logging.getLogger(__name__)


class MemHandler(StreamThreadBase):
    def __init__(
        self,
        buf: Buffers,
        shared_mem: torch.Tensor,
        mem_task_q: Q[tuple[MemTask, torch.cuda.Event] | None],
        mem_p2p_q: Q[P2PTask | Literal["nothing_to_p2p"]],
        shared_mem_ts: torch.Tensor,
        h2d_sem: threading.Semaphore,
        stop_ev: threading.Event,
        device: torch.device,
    ) -> None:
        super().__init__(name="mem_handler", stop_ev=stop_ev)
        logger.debug("MemHandler: Initializing")
        self.buf = buf
        self.device = device
        self.shared_mem = shared_mem
        self.shared_mem_ts = shared_mem_ts
        self.h2d_sem = h2d_sem
        self.mem_task_q = mem_task_q
        self.mem_p2p_q = mem_p2p_q
        self.stop_ev = stop_ev
        logger.debug(
            f"MemHandler: shared_mem shape={shared_mem.shape}, shared_mem_ts shape={shared_mem_ts.shape}"
        )
        logger.debug(
            f"MemHandler: mem_d2h buffer={'enabled' if buf.mem_d2h is not None else 'disabled'}"
        )
        logger.debug("MemHandler: Initialized")

    def _write_back_via_pinned(
        self,
        valid_nid_gpu: torch.Tensor,
        valid_data_gpu: torch.Tensor,
        valid_ts_gpu: torch.Tensor,
        pinned_buf: DyBuf,
    ) -> None:
        """
        利用 Pinned Memory 进行 D2H 回写优化
        """
        n = valid_nid_gpu.numel()
        if n == 0:
            logger.debug("MemHandler: No valid entries to write back, returning early")
            return
        logger.debug(
            f"MemHandler: _write_back_via_pinned called, n={n}, buffer capacity={pinned_buf.data.shape[0]}"
        )

        # 1. 检查 Buffer 容量
        if n > pinned_buf.data.shape[0]:
            logger.debug(
                f"MemHandler: Buffer insufficient (need {n}, have {pinned_buf.data.shape[0]}), using fallback path"
            )
            # Fallback 到普通模式
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_mem[valid_nid] = valid_data_gpu.cpu()
            self.shared_mem_ts[valid_nid] = valid_ts_gpu.cpu()
            logger.debug("MemHandler: Fallback write-back complete")
            return

        logger.debug("MemHandler: Using pinned buffer for D2H transfer")
        # 2. 获取 Buffer 视图
        buf_data = pinned_buf.data[:n]
        buf_nid = pinned_buf.nid[:n]
        buf_ts = pinned_buf.ts[:n]

        # 3. Async Copy: GPU -> Pinned CPU
        logger.debug("MemHandler: Starting async D2H transfer")
        valid_data_gpu.to(buf_data, non_blocking=False)
        # TS 和 NID 在 GPU 上通常是 1D，Buffer 是 2D (N,1)
        valid_ts_gpu.to(buf_ts, non_blocking=False)
        # NID 需要转成 float/long 对应类型，通常 Buffer 初始化时类型定了，这里简单 copy
        # 注意：shared_mem 索引需要 Long，但 buffer 可能是 Float，这里仅做传输中介
        valid_nid_gpu.to(dtype=buf_nid.dtype).to(buf_nid, non_blocking=False)

        # 4. Synchronize: 等待传输完成
        # 因为下一步马上要写入 shared_mem (Pageable)，必须确保数据已在 Pinned 中
        logger.debug("MemHandler: Synchronizing CUDA stream")
        torch.cuda.current_stream().synchronize()

        # 5. Copy: Pinned CPU -> Shared Memory (CPU)
        # 从 buffer 读取回来转成 long 作为索引
        logger.debug("MemHandler: Copying to shared memory")
        idx = buf_nid.view(-1).long()
        self.shared_mem[idx] = buf_data
        self.shared_mem_ts[idx] = buf_ts.view(-1)
        logger.debug(
            f"MemHandler: Write-back complete via pinned buffer, wrote {n} entries"
        )

    def write_back_mem(
        self, writeback: torch.Tensor | None, mem_ts_nid: MemTsNid
    ) -> None:
        logger.debug(
            f"MemHandler: write_back_mem called, writeback={'provided' if writeback is not None else 'none'}, mem_ts_nid={'provided' if mem_ts_nid is not None else 'none'}"
        )

        if mem_ts_nid is None or writeback is None:
            logger.debug(
                "MemHandler: No write-back needed (mem_ts_nid or writeback is None), releasing h2d_sem"
            )

            self.h2d_sem.release()
            return

        logger.debug(
            f"MemHandler: writeback size={writeback.numel()}, mem_ts_nid size={mem_ts_nid.nid.numel()}"
        )

        # 优化: 确保 writeback 在 GPU 上以加速 mask 计算
        if writeback.device.type != "cuda":
            logger.debug("MemHandler: Moving writeback tensor to GPU")
            writeback = writeback.cuda()

        # 在 GPU 上做 mask
        logger.debug("MemHandler: Filtering valid entries on GPU")
        mask = torch.isin(mem_ts_nid.nid, writeback)
        num_valid = mask.sum().item()
        logger.debug(f"MemHandler: Found {num_valid} valid entries to write back")

        # 获取需要写回的数据 (仍在 GPU)
        valid_nid_gpu = mem_ts_nid.nid[mask]
        valid_data_gpu = mem_ts_nid.data[mask]
        valid_ts_gpu = mem_ts_nid.ts[mask]

        if self.buf.mem_d2h is not None:
            logger.debug("MemHandler: Using pinned memory path for write-back")
            self._write_back_via_pinned(
                valid_nid_gpu, valid_data_gpu, valid_ts_gpu, self.buf.mem_d2h
            )
        else:
            logger.debug("MemHandler: Using regular (blocking) path for write-back")
            # 原始逻辑 Fallback
            valid_nid = valid_nid_gpu.cpu().long()
            self.shared_mem[valid_nid] = valid_data_gpu.cpu()
            self.shared_mem_ts[valid_nid] = valid_ts_gpu.cpu()
            logger.debug(
                f"MemHandler: Regular write-back complete, wrote {num_valid} entries"
            )

        logger.debug("MemHandler: Releasing h2d_sem")
        self.h2d_sem.release()

    def logic(self) -> None:
        batch_count = 0
        logger.debug("MemHandler: Starting logic loop")

        while self.stop_ev.is_set() is False:
            logger.debug(
                f"MemHandler: Waiting for mem_task from queue (queue size: {self.mem_task_q.qsize()})"
            )
            mem_task_tuple = self.mem_task_q.get(block=True)

            if mem_task_tuple is None:
                logger.debug("MemHandler: received sentinel, exiting logic loop")
                logger.debug(
                    f"MemHandler: Processed {batch_count} memory tasks before exit"
                )
                break

            batch_count += 1
            mem_task, ready_event = mem_task_tuple
            logger.debug(
                f"MemHandler: Batch #{batch_count}, macro_bid={mem_task.macro_bid}, waiting for ready_event"
            )
            torch.cuda.current_stream().wait_event(ready_event)

            mem_plan = mem_task.mem_plan
            logger.debug(
                f"MemHandler: Batch #{batch_count}, keeping last from root_new_mem, size={mem_task.root_new_mem.nid.numel()}"
            )
            root_new_mem = mem_task.root_new_mem.keep_last()
            logger.debug(
                f"MemHandler: Batch #{batch_count}, after keep_last, size={root_new_mem.nid.numel()}"
            )

            logger.debug(
                f"MemHandler: Batch #{batch_count}, merging root_new_mem with old_mem, old_mem size={mem_task.old_mem_ts_nid.nid.numel()}"
            )
            mem_ts_nid = merge_with_priority(root_new_mem, mem_task.old_mem_ts_nid)
            logger.debug(
                f"MemHandler: Batch #{batch_count}, after merge, mem_ts_nid size={mem_ts_nid.nid.numel()}"
            )

            if mem_task.is_last is not None:
                logger.debug(f"MemHandler: Batch #{batch_count}, creating P2P task")
                p2p_task = make_p2p_task(
                    data_td_nid=mem_ts_nid,
                    plan=mem_plan,
                    strategy="last",
                    device=self.device,
                )
                logger.debug(
                    f"MemHandler: Batch #{batch_count}, putting p2p_task to queue"
                )
                self.mem_p2p_q.put(p2p_task)
            else:
                logger.debug(
                    f"MemHandler: Batch #{batch_count}, no p2p needed, putting 'nothing_to_p2p'"
                )
                self.mem_p2p_q.put("nothing_to_p2p")
                # skip p2p(), directly put None to p2p_q as sentinel

            logger.debug(
                f"MemHandler: Batch #{batch_count}, starting write_back_mem, writeback={'provided' if mem_plan.writeback else 'none'}"
            )
            self.write_back_mem(mem_plan.writeback.data, mem_ts_nid)
            logger.debug(f"MemHandler: Batch #{batch_count} complete")

        logger.debug(
            f"MemHandler: Exiting logic loop, processed {batch_count} memory tasks"
        )
