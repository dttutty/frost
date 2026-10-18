import logging
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

# In utils.py

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def broadcast_data(data: Any = None, src: int = 0) -> Any:  # noqa: ANN401
    payload = [data]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]
