# batch_utils.py
import logging
import math

import numpy as np


logger = logging.getLogger(__name__)


def generate_reordered_indexes_1(
    reorder, batch_size, num_samples, batches_assignment_0
):
    # train_param.batch_size = 8, reorder = 3
    # batches_assignment_0 = [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3]
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    # group_idx = [0, 1, 2, -1, 0, 1, -2, -1, 0]
    chunk_size = batch_size // reorder
    # chunk_size = 8 // 3 = 2
    group_idx = np.repeat(np.array(group_idx), chunk_size)
    # group_idx = [ 0, 0, 1, 1, 2, 2 -1 -1, 0, 0, 1, 1 -2 -2 -1 -1, 0, 0]
    # len(group_idx) = 18
    tiles = num_samples // batch_size + 1
    # tiles = 30 // 8 + 1 = 4
    group_idx = np.tile(group_idx, tiles)[:num_samples]
    # after tile: group_idx = [
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0,
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0]
    # totally 18 * 4 = 72
    # after slice: group_idx = [
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0,
    # 0, 0, 1, 1, 2, 2, -1, -1, 0, 0, 1, 1]
    # totally 30
    return_val = batches_assignment_0 + group_idx
    # return_val = [
    # 0, 0, 1, 1, 2, 2, -1, -1, 1, 1, 2, 2, -1, -1,  0,  0, 2, 2,
    # 2, 2, 3, 3, 4, 4,  2,  2, 3, 3, 4, 4]
    # totally 6 kinds of indexes -1, 0, 1, 2, 3, 4
    return return_val


def generate_reordered_indexes_2(reorder, batch_size, batches_assignment_0):
    # random chunk shceduling
    group_indexes = []
    chunk_size = batch_size // reorder
    # chunk_size = 8 // 3 = 2
    for i in range(1, reorder):
        additional_idx = np.zeros(chunk_size * i) - 1
        group_indexes.append(
            np.concatenate([additional_idx, batches_assignment_0])[
                : batches_assignment_0.shape[0]
            ]
        )
    # [-1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  2.  2.  2.  2.  2.  2.  2.  2.  3.  3.  3.  3.]
    # totally 5 kinds of indexes -1, 0, 1, 2, 3
    # [-1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  2.  2.  2.  2.  2.  2.  2.  2.  3.  3.]
    # totally 5 kinds of indexes -1, 0, 1, 2, 3
    return group_indexes


def batches_assign(index, batch_size):
    return index // batch_size


def plan_gpu_batches(num_samples, target_batch_size, ngpus=1):
    """
    this is to ensure iterations number is divisible by number of gpus
    """
    target_batch_size = int(target_batch_size)  # <--- add
    # This is to ensure the total number of training iterations is an integer multiple of the number of GPUs
    # eg: num_samples = 13345, target_batch_size = 1000, ngpus = 4
    approx_iterations = (
        num_samples // target_batch_size
    )  # approximate number of iterations, 13
    nbatch_per_gpu = max(
        approx_iterations // ngpus, 1
    )  # The approximate number of batches that each GPU needs to process, 3
    iteration_total = nbatch_per_gpu * ngpus  # total batches, 12
    adjusted_batch_size = math.ceil(num_samples / iteration_total)  # 1113
    return iteration_total, adjusted_batch_size
