# Adapted from lxd99/TPNet and lxd99/TGB_TPNet; see NOTICE.md and LICENSE.
import os
import random

import numpy as np
import torch
from torch import nn


def set_thread(num_thread):
    torch.set_num_threads(num_thread)


def set_random_seed(seed: int = 0, deterministic_alg=False):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic_alg)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_alg:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    optimizer: torch.optim.Optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer
