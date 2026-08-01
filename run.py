import os

import hydra
from omegaconf import DictConfig


def _apply_stable_mode_env(cfg: DictConfig) -> dict[str, str]:
    if not bool(cfg.get("stable_mode", False)):
        return {}

    applied: dict[str, str] = {}

    if "CUDA_DEVICE_MAX_CONNECTIONS" not in os.environ:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        applied["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        applied["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FROST_STABLE_MODE"] = "1"
    return applied


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    from frost.launch.affinity import configure_process_affinity

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
    )
    affinity = configure_process_affinity(
        cfg.get("numa_policy", "auto"),
        n_gpu=max(0, local_world_size - 1),
        local_rank=local_rank,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        apply_auto_worker=False,
    )
    normalized_numa_policy = str(cfg.get("numa_policy", "auto")).strip().lower()
    if local_rank == local_world_size - 1 or (
        local_rank == 0 and normalized_numa_policy != "auto"
    ):
        print(affinity.describe())

    applied = _apply_stable_mode_env(cfg)
    if applied:
        print(
            "Stable mode enabled with env overrides: "
            + ", ".join(f"{key}={value}" for key, value in sorted(applied.items()))
        )
    from frost.launcher import launch

    launch(cfg)


if __name__ == "__main__":
    main()
