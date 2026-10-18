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
    from frost.data.paths import static_features_present_on_disk
    from frost.launch.affinity import (
        configure_process_affinity,
        configure_process_threads,
        resolve_numa_runtime_plan,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
    )
    n_gpu = max(0, local_world_size - 1)
    numa_policy = cfg.get("numa_policy", "auto")
    has_static_features = static_features_present_on_disk(
        str(cfg.dataset),
        ignore_node_features=bool(cfg.get("ignore_nfeat_on_disk", False)),
        ignore_edge_features=bool(cfg.get("ignore_efeat_on_disk", False)),
    )
    numa_runtime = resolve_numa_runtime_plan(
        numa_policy,
        n_gpu=n_gpu,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        has_static_features=has_static_features,
    )
    affinity = configure_process_affinity(
        numa_policy,
        n_gpu=n_gpu,
        local_rank=local_rank,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        apply_auto_worker=numa_runtime.localize_workers,
    )
    threading = configure_process_threads(
        cfg.get("cpu_threads", "auto"),
        n_gpu=n_gpu,
        local_rank=local_rank,
        numa_policy=numa_policy,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    if local_rank == 0 or local_rank == local_world_size - 1:
        print(affinity.describe())
    print(f"CPU threading rank={local_rank}: {threading.describe()}")

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
