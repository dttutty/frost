import os

import numpy as np
import torch
import yaml

# Get absolute path of current script and define default config path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "precision.yaml")


def load_precision_config(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # dtype mapping table
    type_map = {
        "int32": (np.int32, torch.int32),
        "int64": (np.int64, torch.int64),
        "float32": (np.float32, torch.float32),
        "float64": (np.float64, torch.float64),
    }

    idx_np, idx_th = type_map[cfg["graph"]["index"]]
    time_np, time_th = type_map[cfg["graph"]["time"]]
    feat_np, feat_th = type_map[cfg["feature"]["feat_dtype"]]

    return {
        "idx_np": idx_np,
        "idx_th": idx_th,
        "time_np": time_np,
        "time_th": time_th,
        "feat_np": feat_np,
        "feat_th": feat_th,
    }
