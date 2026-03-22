import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


# Path to the source .pt file
src_path = "~/Projects/lei_pipe/DATA/WIKI/edge_features.pt"
# Auto-generate the output path: .../edge_features.npy
dst_path = src_path.replace(".pt", ".npy")

logger.info("Loading: %s ...", src_path)

# 1. Load PyTorch Tensor
# map_location='cpu' is critical:
# If the file was saved on GPU, this prevents it from consuming GPU memory on load,
# and also prevents errors in environments without a GPU.
try:
    tensor_data = torch.load(src_path, map_location="cpu")
except Exception as e:
    logger.info("Load failed: %s", e)
    exit(1)

# 2. Check data type
if not isinstance(tensor_data, torch.Tensor):
    logger.info("Warning: file contains %s, not a plain Tensor.", type(tensor_data))
    # If it is a dict, manually extract the Tensor, e.g.:
    # tensor_data = tensor_data['feat']

logger.info("Data shape: %s", tensor_data.shape)
logger.info("Data dtype: %s", tensor_data.dtype)

# 3. Convert to NumPy and save
logger.info("Converting to NumPy and saving to: %s ...", dst_path)
try:
    # Detach first if the tensor requires grad
    if tensor_data.requires_grad:
        tensor_data = tensor_data.detach()

    np_arr = tensor_data.numpy()
    np.save(dst_path, np_arr)
    logger.info("Conversion successful.")

    # 4. Verification (only reads array header, does not load data)
    logger.info("%s", "-" * 30)
    logger.info("Verifying generated .npy file:")
    check_arr = np.load(dst_path, mmap_mode="r")
    logger.info("Npy Shape: %s", check_arr.shape)
    logger.info("Npy Dtype: %s", check_arr.dtype)

except Exception as e:
    logger.info("Save failed: %s", e)
