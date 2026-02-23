import logging

import torch
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array


logger = logging.getLogger(__name__)


class DoubleBufferedSHM:
    def __init__(
        self,
        worker_rank: int,
        create: bool,
        layout: dict[str, tuple[tuple[int, ...], torch.dtype]],
        max_size: int = 10,
        namespace: str = "",
    ) -> None:
        """
        limits: dict, e.g. {'nodes': 500000, 'edges': 1000000}
        """
        self.max_size = max_size
        self.buffers: list[dict[str, torch.Tensor]] = [
            {} for _ in range(max_size)
        ]  # Buffer 0, 1, 2, 3, 4, 5
        self.create = create
        self.worker_rank = worker_rank
        self.namespace = namespace

        self.schema = layout

        # 定义 Schema：Key -> (Shape, Dtype)

        for buf_idx in range(max_size):
            for key, (shape, dtype) in self.schema.items():
                # 构造唯一的共享内存名称
                prefix = f"{namespace}_" if namespace else ""
                name = f"{prefix}TGL_SHM_w{worker_rank}_b{buf_idx}_{key}"

                try:
                    if create:
                        # 创建模式：Host端
                        self.buffers[buf_idx][key] = create_shared_mem_array(
                            name, shape, dtype
                        )
                    else:
                        # 读取模式：Worker端
                        self.buffers[buf_idx][key] = get_shared_mem_array(
                            name, shape, dtype
                        )
                except Exception as e:
                    logger.error(
                        f"SHM Init Error: Failed to {'create' if create else 'open'} {name}. {e}"
                    )
                    # 如果是创建失败，提示清理
                    if create:
                        if namespace:
                            logger.error("Try running: rm /dev/shm/%sTGL_SHM_*", prefix)
                        else:
                            logger.error("Try running: rm /dev/shm/TGL_SHM_*")
                    raise e

    def write(self, batch_idx: int, key: str, tensor: torch.Tensor | None) -> int:
        """写入指定 Buffer (0 或 1)"""
        if tensor is None:
            return 0

        buf_idx = batch_idx % self.max_size

        # 1. Key 检查
        if key not in self.buffers[buf_idx]:
            # 这是一个严重的配置错误，必须抛出
            raise KeyError(
                f"Key '{key}' not found in SHM layout. Please check schemas/shm_layout.py."
            )

        target_buf = self.buffers[buf_idx][key]

        # 2. Device 检查 (确保是 CPU)
        if tensor.device.type != "cpu":
            t_cpu = tensor.detach().cpu()
        else:
            t_cpu = tensor

        # 3. 展平检查 (应对之前 N2R_DIMS 的维度问题)
        # 如果 Buffer 是 1D 但输入是 2D，且总元素数一致，尝试 view(-1)
        if target_buf.dim() == 1 and t_cpu.dim() > 1:
            t_cpu = t_cpu.view(-1)

        length = t_cpu.shape[0]

        # 4. 容量检查
        if length > target_buf.shape[0]:
            error_msg = (
                f"[SHM Error] Buffer Overflow for key '{key}'! "
                f"Batch: {batch_idx}. "
                f"Input Data: {length}, Buffer Capacity: {target_buf.shape[0]}. "
                f"Please increase limits in 'schemas/shm_layout.py'."
            )
            # 使用 logger 而不是 print，确保被记录
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 5. [Fix] 类型检查与自动转换
        # 如果类型不匹配，PyTorch 有时会报错，有时会发生奇怪的行为。我们显式处理。
        if t_cpu.dtype != target_buf.dtype:
            try:
                # 尝试转换
                t_cpu = t_cpu.to(target_buf.dtype)
            except Exception as e:
                msg = (
                    f"[SHM Type Error] Key '{key}': Cannot cast {t_cpu.dtype} to {target_buf.dtype}. "
                    f"Error: {e}"
                )
                logger.error(msg)
                raise TypeError(msg)

        # 6. 执行写入
        try:
            target_buf[:length] = t_cpu
        except Exception as e:
            # 捕获可能的底层拷贝错误（如 shape mismatch 漏网之鱼）
            logger.error(
                f"[SHM Copy Error] Key '{key}': Failed to copy data. "
                f"Input shape: {t_cpu.shape}, Target slice shape: {target_buf[:length].shape}. "
                f"Error: {e}"
            )
            raise e

        return length

    def read(self, batch_idx: int, key: str, length: int) -> torch.Tensor:
        """从指定 Buffer 读取"""
        if length == 0:
            return torch.empty(0)

        buf_idx = batch_idx % self.max_size

        if key not in self.buffers[buf_idx]:
            raise KeyError(f"Key '{key}' not found in SHM layout.")

        # 获取 slice
        # 注意：shared_mem_array 实际上是 torch.Tensor，支持切片
        target_buf = self.buffers[buf_idx][key]
        return target_buf[:length]
