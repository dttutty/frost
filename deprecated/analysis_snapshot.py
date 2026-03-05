import logging
import pickle


logger = logging.getLogger(__name__)


with open("memory_snapshot.pkl", "rb") as f:
    snapshot = pickle.load(f)

# 使用正确的键 'allocated_size' 来进行排序
try:
    large_tensors = sorted(
        snapshot, key=lambda x: x.get("allocated_size", 0), reverse=True
    )
except Exception as e:
    logger.debug(f"排序时发生错误: {e}")
    exit()

logger.debug("--- 内存中最大的10个张量 ---")
for i, block in enumerate(large_tensors[:10]):
    # 使用 'allocated_size' 来获取并打印大小
    size_in_mb = block.get("allocated_size", 0) / 1024**2
    logger.debug(f"#{i + 1}: {size_in_mb:.2f} MiB")

    # 打印创建这个张量的代码位置，使用正确的键 'frames'
    logger.debug("  创建位置 (Traceback):")
    # 使用 .get('frames', []) 来安全地获取堆栈信息
    for frame in block.get("frames", []):
        logger.debug(
            f"    File '{frame.get('filename', 'N/A')}', line {frame.get('line', 'N/A')}, in {frame.get('name', 'N/A')}"
        )
    logger.debug("-" * 20)
