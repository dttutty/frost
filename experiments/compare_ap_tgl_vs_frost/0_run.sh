#!/usr/bin/env bash

set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$SCRIPT_SOURCE")"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RUNNER="$REPO_ROOT/scripts/run_on_gpu_pairs.py"
CONFIG_PATH="$SCRIPT_DIR/0_run.yaml"

usage() {
    cat <<'EOF'
Usage:
  bash experiments/compare_ap_tgl_vs_frost/0_run.sh [GPU_IDS]
  uv run python scripts/run_on_gpu_pairs.py --script experiments/compare_ap_tgl_vs_frost/0_run.sh [--gpus 0,1,2,3]

This script defines FROST compare_ap_tgl_vs_frost jobs.
When called normally, it delegates scheduling to scripts/run_on_gpu_pairs.py.
When called with --emit-jobs, it prints job definitions for the scheduler.
Experiment settings are loaded from experiments/compare_ap_tgl_vs_frost/0_run.yaml.
Each emitted job uses 2 visible GPUs with torchrun --nproc_per_node=3.
EOF
}

resolve_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "$PYTHON_BIN"
    elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        printf '%s\n' "$REPO_ROOT/.venv/bin/python"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        printf '%s\n' "${VIRTUAL_ENV}/bin/python"
    elif command -v python >/dev/null 2>&1; then
        command -v python
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        echo "Python interpreter not found." >&2
        exit 1
    fi
}

validate_python_bin() {
    local python_bin="$1"
    if ! "$python_bin" - <<'PY' >/dev/null 2>&1
import frost  # noqa: F401
import torch  # noqa: F401
import yaml  # noqa: F401
PY
    then
        cat >&2 <<EOF
Selected Python does not have the required FROST runtime packages:
  $python_bin

Use the project uv environment first, for example:
  uv sync --locked
  uv run bash experiments/compare_ap_tgl_vs_frost/0_run.sh
EOF
        exit 1
    fi
}

EMIT_JOBS=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emit-jobs)
            EMIT_JOBS=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$EMIT_JOBS" -eq 0 ]]; then
    PYTHON_BIN="$(resolve_python_bin)"
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$PYTHON_BIN")"
    fi
    validate_python_bin "$PYTHON_BIN"

    runner_args=("$RUNNER" --script "$SCRIPT_PATH")
    if [[ "${#FORWARD_ARGS[@]}" -gt 0 ]]; then
        runner_args+=(--gpus "${FORWARD_ARGS[0]}")
        if [[ "${#FORWARD_ARGS[@]}" -gt 1 ]]; then
            runner_args+=(-- "${FORWARD_ARGS[@]:1}")
        fi
    fi

    exec "$PYTHON_BIN" "${runner_args[@]}"
fi

PYTHON_BIN="$(resolve_python_bin)"
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi
validate_python_bin "$PYTHON_BIN"

OMP_THREADS="${OMP_NUM_THREADS:-8}"
MKL_THREADS="${MKL_NUM_THREADS:-$OMP_THREADS}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29750}"
MASTER_PORT_STRIDE="${MASTER_PORT_STRIDE:-256}"
MASTER_PORT_SLOT="${MASTER_PORT_SLOT:-}"
TASK_NUM_GPUS=2
LOG_ROOT="${LOG_DIR:-$SCRIPT_DIR/logs}"
RUN_LOG_DIR="$LOG_ROOT"

if [[ -z "$MASTER_PORT_SLOT" ]]; then
    # Spread concurrent scheduler instances onto disjoint port ranges.
    # 139 slots * 256 ports fits under the TCP max port when BASE_MASTER_PORT=29750.
    MASTER_PORT_SLOT=$(( $$ % 139 ))
fi
MASTER_PORT_OFFSET=$(( MASTER_PORT_SLOT * MASTER_PORT_STRIDE ))

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

mkdir -p "$RUN_LOG_DIR"

load_experiment_rows() {
    "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f) or {}

experiments = conf.get("experiments") or []
mem_dims = conf.get("mem_dims") or []
seeds_list = conf.get("seeds")
if seeds_list is not None:
    if not isinstance(seeds_list, list) or not seeds_list:
        raise RuntimeError("0_run.yaml seeds must be a non-empty list")
    seeds_list = [int(s) for s in seeds_list]
else:
    repeats = int(conf.get("repeats", 1))
    seed_base = int(conf.get("seed_base", 0))
    seeds_list = list(range(seed_base, seed_base + repeats))
n_seeds = len(seeds_list)
nproc_per_node = int(conf.get("nproc_per_node", 3))
target_epoch = int(conf.get("target_epoch", 100))
batch_size = int(conf.get("batch_size", 600))
batch_sizes = conf.get("batch_sizes")

if batch_sizes is None:
    batch_sizes = [batch_size]
else:
    if not isinstance(batch_sizes, list) or not batch_sizes:
        raise RuntimeError("0_run.yaml batch_sizes must be a non-empty list")
    batch_sizes = [int(size) for size in batch_sizes]

if not experiments:
    raise RuntimeError("0_run.yaml must define at least one experiment")
if not mem_dims:
    raise RuntimeError("0_run.yaml must define at least one mem_dim")

for exp in experiments:
    name = str(exp["name"])
    raw_datasets = exp.get("dataset")
    if raw_datasets is None:
        raise RuntimeError(f"experiment {name!r} must define dataset")
    if isinstance(raw_datasets, list):
        if not raw_datasets:
            raise RuntimeError(f"experiment {name!r} dataset list must be non-empty")
        datasets = [str(dataset) for dataset in raw_datasets]
    else:
        datasets = [str(raw_datasets)]

    base_overrides = exp.get("overrides") or []
    if not base_overrides:
        raise RuntimeError(f"experiment {name!r} must define overrides")

    for dataset in datasets:
        overrides = []
        dataset_override_seen = False
        for item in base_overrides:
            item = str(item)
            if item.startswith("dataset=") or item.startswith("dataset_name="):
                overrides.append(f"dataset={dataset}")
                dataset_override_seen = True
            else:
                overrides.append(item)
        if not dataset_override_seen:
            overrides.append(f"dataset={dataset}")

        override_str = " ".join(overrides)
        for batch_size in batch_sizes:
            for mem_dim in mem_dims:
                mem_dim = int(mem_dim)
                for seed_idx, seed in enumerate(seeds_list, start=1):
                    print(
                        "\t".join(
                            [
                                name,
                                dataset,
                                override_str,
                                str(mem_dim),
                                str(seed_idx),
                                str(n_seeds),
                                str(seed),
                                str(nproc_per_node),
                                str(target_epoch),
                                str(batch_size),
                            ]
                        )
                    )
PY
}

mapfile -t experiment_rows < <(load_experiment_rows)

if [[ "${#experiment_rows[@]}" -eq 0 ]]; then
    echo "No experiments generated from $CONFIG_PATH" >&2
    exit 1
fi

MAX_MASTER_PORT=$((BASE_MASTER_PORT + MASTER_PORT_OFFSET + ${#experiment_rows[@]} - 1))
if (( MAX_MASTER_PORT > 65535 )); then
    cat >&2 <<EOF
Master port range exceeds 65535:
  base=$BASE_MASTER_PORT
  slot=$MASTER_PORT_SLOT
  stride=$MASTER_PORT_STRIDE
  jobs=${#experiment_rows[@]}
  max_port=$MAX_MASTER_PORT

Adjust BASE_MASTER_PORT, MASTER_PORT_SLOT, or MASTER_PORT_STRIDE.
EOF
    exit 1
fi

first_row="${experiment_rows[0]}"
IFS=$'\t' read -r _ _ _ _ _ REPEATS _ NPROC_PER_NODE TARGET_EPOCH _ <<< "$first_row"
TORCH_WORKERS=$((NPROC_PER_NODE - 1))

declare -A seen_batch_sizes=()
batch_sizes_display=()
declare -A seen_datasets=()
datasets_display=()
for row in "${experiment_rows[@]}"; do
    IFS=$'\t' read -r _ row_dataset _ _ _ _ _ _ _ row_batch_size <<< "$row"
    if [[ -z "${seen_batch_sizes[$row_batch_size]+x}" ]]; then
        seen_batch_sizes["$row_batch_size"]=1
        batch_sizes_display+=("$row_batch_size")
    fi
    if [[ -z "${seen_datasets[$row_dataset]+x}" ]]; then
        seen_datasets["$row_dataset"]=1
        datasets_display+=("$row_dataset")
    fi
done

if [[ "$TORCH_WORKERS" -ne "$TASK_NUM_GPUS" ]]; then
    echo "Expected nproc_per_node - 1 to equal $TASK_NUM_GPUS, got: $NPROC_PER_NODE" >&2
    exit 1
fi

echo "Logs: $RUN_LOG_DIR" >&2
echo "Python: $PYTHON_BIN" >&2
echo "Config: $CONFIG_PATH" >&2
echo "Visible GPUs per job: $TASK_NUM_GPUS" >&2
echo "torchrun --nproc_per_node: $NPROC_PER_NODE" >&2
echo "Master port base: $BASE_MASTER_PORT" >&2
echo "Master port slot: $MASTER_PORT_SLOT" >&2
echo "Master port stride: $MASTER_PORT_STRIDE" >&2
echo "Master port offset: $MASTER_PORT_OFFSET" >&2
echo "Datasets: ${datasets_display[*]}" >&2
echo "Batch sizes: ${batch_sizes_display[*]}" >&2
echo "Epochs: $TARGET_EPOCH" >&2
echo "Repeats: $REPEATS" >&2
echo "OMP threads: $OMP_THREADS" >&2
echo "MKL threads: $MKL_THREADS" >&2
echo "Total runs: ${#experiment_rows[@]}" >&2

run_idx=0

for row in "${experiment_rows[@]}"; do
    IFS=$'\t' read -r name dataset override_str mem_dim repeat_idx repeats seed nproc_per_node target_epoch batch_size <<< "$row"

    overrides=()
    if [[ -n "$override_str" ]]; then
        read -r -a overrides <<< "$override_str"
    fi

    master_port=$((BASE_MASTER_PORT + MASTER_PORT_OFFSET + run_idx))
    log_file="$RUN_LOG_DIR/${name}__${dataset}__memdim${mem_dim}__ep${target_epoch}__bs${batch_size}__nproc${nproc_per_node}__seed${seed}__rep${repeat_idx}.log"
    desc="${name}/${dataset}/memdim${mem_dim}/ep${target_epoch}/bs${batch_size}/seed${seed}/rep${repeat_idx}"

    cmd=(
        env
        "OMP_NUM_THREADS=$OMP_THREADS"
        "MKL_NUM_THREADS=$MKL_THREADS"
        "TOKENIZERS_PARALLELISM=false"
        "PYTHONUNBUFFERED=1"
        "$PYTHON_BIN" -u -m torch.distributed.run
        --nproc_per_node "$nproc_per_node"
        --master_addr "$MASTER_ADDR"
        --master_port "$master_port"
        "$REPO_ROOT/run.py"
        "${overrides[@]}"
        "mem_updater.d_mem=$mem_dim"
        "epoch=$target_epoch"
        "batch_size=$batch_size"
        "seed=$seed"
    )

    echo "============================================================" >&2
    echo "[run $((run_idx + 1))/${#experiment_rows[@]}] name=${name} dataset=${dataset} mem_dim=${mem_dim} epoch=${target_epoch} batch_size=${batch_size} seed=${seed} repeat=${repeat_idx}/${repeats}" >&2
    echo "log: ${log_file}" >&2
    echo "master_port: ${master_port}" >&2
    echo "============================================================" >&2

    printf '%s\t%s\t' "$desc" "$log_file"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "" >&2

    run_idx=$((run_idx + 1))
done
