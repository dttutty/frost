# filename: run_training.py
# This is the entrypoint script.
# Its only job is to set up the environment and then run the main application.
# It should have minimal imports.

import logging
import os
import sys


logger = logging.getLogger(__name__)


# ---------- Threads & NCCL safety BEFORE importing torch ----------
# 放在最前，确保底层库读取到
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def bootstrap_visible_devices(host_is_last: bool = True) -> None:
    """
    Sets the CUDA_VISIBLE_DEVICES environment variable for each process.
    This MUST be called before importing torch.
    """
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Get the list of GPUs visible to all processes at launch time.
    # Note: torchrun will narrow this down to a single device ID (e.g., "0" or "1")
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_gpus_at_start = [x for x in raw.split(",") if x] if raw else []
    num_visible_gpus_at_start = len(visible_gpus_at_start)

    logger.debug(
        f"[Bootstrap] Rank {rank}/{world_size} (Local Rank {local_rank}) "
        f"sees initial CUDA_VISIBLE_DEVICES='{raw}'"
    )

    # Your strategy: the last rank is the host and should not see any GPUs.
    if host_is_last and world_size > 1 and rank == world_size - 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Host sees no GPUs
        logger.debug(
            f"[Bootstrap] Rank {rank} is the host. CUDA_VISIBLE_DEVICES set to ''."
        )
    else:
        # Each worker process is pinned to a single GPU using its LOCAL_RANK.
        if visible_gpus_at_start:
            # This logic correctly pins each process to one GPU.
            # It is idempotent: if torchrun has already set it to "0",
            # local_rank=0 % num_visible=1 results in index 0, keeping "0".
            target_gpu = visible_gpus_at_start[local_rank % num_visible_gpus_at_start]
            os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu
            logger.debug(
                f"[Bootstrap] Rank {rank} is a worker. "
                f"CUDA_VISIBLE_DEVICES narrowed to '{target_gpu}'."
            )
        # If raw was empty (e.g., CPU mode), it remains empty.
        else:
            logger.debug(
                f"[Bootstrap] Rank {rank} sees no initial GPUs. Not changing environment."
            )


if __name__ == "__main__":
    # --- STEP 1: Bootstrap the environment FIRST ---
    # This is the critical step.
    bootstrap_visible_devices(host_is_last=True)

    # --- STEP 2: NOW, import and run the main application logic ---
    # isort/ruff are happy because the import is at the top-level of this logic block.
    # The main_app.py file can have all its imports at the top without issues.
    logger.debug("-" * 50)
    logger.debug("Bootstrap complete. Importing main application (and torch)...")

    # We add the current directory to sys.path to ensure 'train' can be imported.
    # In a real project, this would be handled by packaging.
    sys.path.insert(0, os.path.dirname(__file__))

    import train_dist as train

    # --- STEP 3: Execute the main function from the application ---
    train.main()
