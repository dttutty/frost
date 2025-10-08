#!/usr/bin/env bash
#
# run_single_dataset.sh — Runs models sequentially for a given dataset and GPU pair.
# Each model is run 3 times.
# Usage:
#   Make it executable: chmod +x run_single_dataset.sh
#   Run for one dataset: ./run_single_dataset.sh <dataset_name> <gpu_pair> [model_list] [log_dir]
#   Example: ./run_single_dataset.sh MOOC "0,1" "APAN,DySAT,JODIE" "custom_logs"
#
set -euo pipefail

# --- Argument validation ---
if [[ $# -lt 2 ]]; then
    echo "Error: Invalid arguments."
    echo "Usage: $0 <dataset_name> <gpu_pair> [model_list] [log_dir]"
    echo "Example: $0 MOOC \"0,1\" \"APAN,DySAT,JODIE\" \"custom_logs\""
    exit 1
fi

DATASET=$1
GPU_PAIR=$2

# --- Parse user-specified models or use defaults ---
if [[ $# -ge 3 ]]; then
    # Expect comma-separated list
    IFS=',' read -ra models <<< "$3"
else
    models=("APAN" "DySAT" "JODIE" "TGAT" "TGN")
fi

# --- Set log directory (use default 'logs' if not specified) ---
# Use the fourth argument as the log directory, or default to "logs"
LOG_DIR=${4:-"logs"}

# --- Dynamic process count based on GPU pair ---
IFS=',' read -ra GPUS <<< "$GPU_PAIR"
NPROC_PER_NODE=3 # Use the number of specified GPUs

# --- Configuration (adjust as needed) ---
MASTER_ADDR="127.0.0.1"
PORT_MIN=20000
PORT_MAX=45535
CONFIG_DIR="/home/sqp17/Projects/tgl/config/dist"

# Create the log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

echo "Starting job for dataset: ${DATASET} on GPUs: ${GPU_PAIR}"
echo "Using models: ${models[*]}"
echo "Each model will be run 3 times."
echo "Logs will be saved in the ${LOG_DIR}/ directory."

time_start=$(date +'%s')

# Loop through all models for the given dataset
for model in "${models[@]}"; do
    # Define the log file for the 3 runs of this model
    log_file="${LOG_DIR}/${DATASET}_${model}_3runs.log"

    # Clear previous log file for this model before starting the 3 new runs
    echo "Clearing previous logs and starting 3 runs for model: ${model}. New logs will be in: ${log_file}"
    > "${log_file}"

    # Loop 3 times for the current model
    for run in {1..3}; do
        # Generate a new random port for each run to avoid conflicts
        MASTER_PORT=$(( RANDOM % (PORT_MAX-PORT_MIN+1) + PORT_MIN ))
        ts=$(date +'%Y%m%d_%H%M%S')

        echo "--- [${DATASET}] Running model: ${model} (Run ${run}/3) at ${ts} ---" | tee -a "${log_file}"

        CUDA_VISIBLE_DEVICES=${GPU_PAIR} TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun \
            --nproc_per_node=${NPROC_PER_NODE} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            train_dist.py \
            --data "${DATASET}" \
            --config "${CONFIG_DIR}/${model}.yml" \
            2>&1 | tee -a "${log_file}"

        echo "--- [${DATASET}] Model ${model} (Run ${run}/3) finished. ---" | tee -a "${log_file}"
        
        # Optional: Add a delay if not the last run
        if [[ ${run} -lt 3 ]]; then
            echo "Sleeping for 5 seconds before the next run..."
            sleep 5
        fi
    done

    echo "✅ All 3 runs for model '${model}' have completed." | tee -a "${log_file}"
done

elapsed=$(( $(date +'%s') - time_start ))
echo "✅ All models for dataset '${DATASET}' have completed in ${elapsed}s."
