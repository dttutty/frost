import numpy as np
import torch


# 你的文件路径
src_path = "~/Projects/lei_pipe/DATA/WIKI/edge_features.pt"
# 自动生成目标路径: .../edge_features.npy
dst_path = src_path.replace(".pt", ".npy")

print(f"正在加载: {src_path} ...")

# 1. 加载 PyTorch Tensor
# map_location='cpu' 至关重要：
# 如果原文件是在 GPU 上保存的，这可以防止它加载时占用你的显存，
# 同时也防止因没有 GPU 环境而报错。
try:
    tensor_data = torch.load(src_path, map_location="cpu")
except Exception as e:
    print(f"加载失败: {e}")
    exit(1)

# 2. 检查数据类型
if not isinstance(tensor_data, torch.Tensor):
    print(f"警告: 该文件包含的是 {type(tensor_data)}，而不是纯 Tensor。")
    # 如果是字典，你可能需要手动提取其中的 Tensor
    # tensor_data = tensor_data['feat']

print(f"数据形状: {tensor_data.shape}")
print(f"数据类型: {tensor_data.dtype}")

# 3. 转换为 NumPy 并保存
print(f"正在转换为 NumPy 并保存至: {dst_path} ...")
try:
    # 如果 Tensor 需要梯度，先 detach
    if tensor_data.requires_grad:
        tensor_data = tensor_data.detach()

    np_arr = tensor_data.numpy()
    np.save(dst_path, np_arr)
    print("✅ 转换成功！")

    # 4. 验证测试（不加载数据，只读头信息）
    print("-" * 30)
    print("验证生成的 .npy 文件:")
    check_arr = np.load(dst_path, mmap_mode="r")
    print(f"Npy Shape: {check_arr.shape}")
    print(f"Npy Dtype: {check_arr.dtype}")

except Exception as e:
    print(f"❌ 保存失败: {e}")
