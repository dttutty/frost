import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


# 你的文件路径
src_path = "/home/sqp17/Projects/lei_pipe/DATA/WIKI/edge_features.pt"
# 自动生成目标路径: .../edge_features.npy
dst_path = src_path.replace(".pt", ".npy")

logger.info("正在加载: %s ...", src_path)

# 1. 加载 PyTorch Tensor
# map_location='cpu' 至关重要：
# 如果原文件是在 GPU 上保存的，这可以防止它加载时占用你的显存，
# 同时也防止因没有 GPU 环境而报错。
try:
    tensor_data = torch.load(src_path, map_location="cpu")
except Exception as e:
    logger.info("加载失败: %s", e)
    exit(1)

# 2. 检查数据类型
if not isinstance(tensor_data, torch.Tensor):
    logger.info("警告: 该文件包含的是 %s，而不是纯 Tensor。", type(tensor_data))
    # 如果是字典，你可能需要手动提取其中的 Tensor
    # tensor_data = tensor_data['feat']

logger.info("数据形状: %s", tensor_data.shape)
logger.info("数据类型: %s", tensor_data.dtype)

# 3. 转换为 NumPy 并保存
logger.info("正在转换为 NumPy 并保存至: %s ...", dst_path)
try:
    # 如果 Tensor 需要梯度，先 detach
    if tensor_data.requires_grad:
        tensor_data = tensor_data.detach()

    np_arr = tensor_data.numpy()
    np.save(dst_path, np_arr)
    logger.info("✅ 转换成功！")

    # 4. 验证测试（不加载数据，只读头信息）
    logger.info("%s", "-" * 30)
    logger.info("验证生成的 .npy 文件:")
    check_arr = np.load(dst_path, mmap_mode="r")
    logger.info("Npy Shape: %s", check_arr.shape)
    logger.info("Npy Dtype: %s", check_arr.dtype)

except Exception as e:
    logger.info("❌ 保存失败: %s", e)
