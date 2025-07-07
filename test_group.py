import numpy as np
from typing import List
from dgl.heterograph import DGLBlock
from dgl import DGLGraph
import numpy as np
import pandas as pd

def generate_reordered_indexes(reorder, batch_size, num_samples, group_indexes):
    # random chunk shceduling
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    print("Group index:", group_idx)
    group_idx = np.repeat(np.array(group_idx), batch_size // reorder)
    print("Group index repeated:", group_idx)
    group_idx = np.tile(group_idx, num_samples // batch_size + 1)[:num_samples]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, reorder):
        additional_idx = np.zeros(batch_size// reorder * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])
    return group_indexes

def batches_assign(index, batch_size):
    return index // batch_size


reorder = 3
batch_size = 8
train = pd.Series(np.arange(30), name='value')
# print("Train data:", train)
batches_assignment = batches_assign(train, batch_size)
group_indexes = [batches_assignment]
group_indexes = generate_reordered_indexes(reorder, batch_size, len(train), group_indexes)
for i, group in enumerate(group_indexes):
    print(f"Group {i}:", group)
for _, rows in train.groupby(group_indexes[1]):
    print("Rows in group:")
    print(rows.values)
