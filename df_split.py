import math

import pandas as pd


def make_balance_plan(
    df: pd.DataFrame, org_batch_size: int, n_gpu: int
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """
    Goal:
      - Keep full macro-batches of size (org_batch_size * n_gpu) as-is.
      - For the final (incomplete) part, ensure the last step has >= 1 sample per GPU.
        If leftover < n_gpu, pad by repeating the last row until leftover == n_gpu.

    Returns:
      - df_out: possibly padded dataframe
      - step_id: list[int] of length len(df_out), mapping each row to a step_id in [0, normal_macro_batch_number)
                  (used to partition rows per step across GPUs)
      - gpu_id: list[int] of length len(df_out), mapping each row to a gpu_id in [0, n_gpu)
                  (used to partition rows per step across GPUs)
    """
    assert n_gpu > 0
    assert org_batch_size > 0
    assert len(df) >= org_batch_size * n_gpu, (
        "The length of the dataframe should be >= org_batch_size * n_gpu"
    )

    every_batch_max_num = org_batch_size * n_gpu
    length_of_df = len(df)

    # number of complete macro-batches
    normal_macro_batch_number = length_of_df // every_batch_max_num
    done_df = normal_macro_batch_number * every_batch_max_num
    leftover = length_of_df - done_df

    df_out = df

    # If leftover exists but fewer than n_gpu, pad to exactly n_gpu
    if 0 < leftover < n_gpu:
        lack = n_gpu - leftover
        last_row = df.iloc[[-1]]
        df_out = pd.concat([df] + [last_row] * lack, ignore_index=True)
        length_of_df = len(df_out)

    # Build row->gpu assignment per "step":
    # within each step, GPU 0 gets first chunk, GPU 1 next chunk, ...
    # chunk size per GPU = ceil(step_len / n_gpu), where step_len is either
    # full macro-batch (org_batch_size*n_gpu) or the last leftover (>= n_gpu after padding)
    step_id: list[int] = []
    gpu_id: list[int] = []

    # full macro-batches: each step has exactly org_batch_size items per GPU
    for s in range(normal_macro_batch_number):
        for g in range(n_gpu):
            step_id.extend([s] * org_batch_size)
            gpu_id.extend([g] * org_batch_size)

    # last step (if any leftover after done_df)
    last_len = length_of_df - done_df
    if last_len > 0:
        s = normal_macro_batch_number  # last step index
        per_gpu = int(math.ceil(last_len / n_gpu))

        for g in range(n_gpu):
            step_id.extend([s] * per_gpu)
            gpu_id.extend([g] * per_gpu)

        # truncate to exact length
        step_id = step_id[:length_of_df]
        gpu_id = gpu_id[:length_of_df]

    assert len(step_id) == length_of_df, (len(step_id), length_of_df)
    assert len(gpu_id) == length_of_df, (len(gpu_id), length_of_df)

    return df_out, step_id, gpu_id
