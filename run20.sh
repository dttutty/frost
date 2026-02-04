#!/bin/bash

# ==========================================
# Configuration
# ==========================================
# Number of times to run the experiment
MAX_RUNS=20

# Directory to save logs
LOG_DIR="experiment_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# File to store the extracted metrics for calculation
METRICS_FILE="${LOG_DIR}/all_metrics.txt"

# Path to python interpreter
PYTHON_EXEC="/home/sqp17/miniforge3/envs/simple_py310/bin/python"

echo "=================================================="
echo "Starting ${MAX_RUNS} runs..."
echo "Logs will be saved to: ${LOG_DIR}"
echo "=================================================="

# ==========================================
# 1. Environment Setup (From your snippet)
# ==========================================
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=OFF
export LOG_LEVEL=INFO
export NUMBA_DEBUG=0
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export NCCL_DEBUG=WARN
export TORCH_CUDNN_BENCHMARK=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:2
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# export PYTHONOPTIMIZE=1

# ==========================================
# 2. Training Loop
# ==========================================
for ((i=1; i<=MAX_RUNS; i++)); do
    echo "[Run $i / $MAX_RUNS] Started at $(date '+%H:%M:%S')..."
    
    CURRENT_LOG="${LOG_DIR}/run_${i}.log"

    # Execute the command
    # Redirecting both stdout and stderr to the log file
    $PYTHON_EXEC -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_addr=127.0.0.1 \
        --master_port=$((29505 + i)) \
        train_dist.py \
        --dataset LASTFM \
        --config /home/sqp17/Projects/tgl/config/dist/TGN.yml \
        --rnd_edim 128 \
        --rnd_ndim 128 \
        > "$CURRENT_LOG" 2>&1

    # Check if run was successful
    if [ $? -eq 0 ]; then
        echo "[Run $i / $MAX_RUNS] Finished successfully."
    else
        echo "[Run $i / $MAX_RUNS] Failed! Check $CURRENT_LOG"
    fi

    # ==========================================
    # 3. Metric Extraction (Auto-Parsing)
    # ==========================================
    # TODO: ADJUST THESE PATTERNS BASED ON YOUR ACTUAL LOG OUTPUT
    # Example assumption: Log contains "Test AP: 0.9543" and "Test AUC: 0.9678"
    # We use grep to find the line, and awk to grab the number.
    
    # Extract AP (Change 'Test AP:' to your actual print statement)
    # awk '{print $NF}' assumes the number is the last part of the line.
    val_ap=$(grep "Test AP" "$CURRENT_LOG" | tail -n 1 | awk -F'[:= ]+' '{for(j=1;j<=NF;j++) if($j ~ /[0-9]+\.[0-9]+/) print $j}')
    
    # Extract AUC (Change 'Test AUC:' to your actual print statement)
    val_auc=$(grep "Test AUC" "$CURRENT_LOG" | tail -n 1 | awk -F'[:= ]+' '{for(j=1;j<=NF;j++) if($j ~ /[0-9]+\.[0-9]+/) print $j}')

    # Save to metrics file if values were found
    if [ ! -z "$val_ap" ] && [ ! -z "$val_auc" ]; then
        echo "$val_ap $val_auc" >> "$METRICS_FILE"
        echo "   -> Parsed: AP=$val_ap, AUC=$val_auc"
    else
        echo "   -> WARNING: Could not parse AP/AUC from log. Check grep pattern."
    fi

    # Optional: small sleep to let GPU memory clear
    sleep 2
done

# ==========================================
# 4. Calculate Average
# ==========================================
echo "=================================================="
echo "Calculation Results:"
echo "=================================================="

if [ -f "$METRICS_FILE" ]; then
    awk '{ sum_ap += $1; sum_auc += $2; count++ } END { 
        if (count > 0) {
            printf "Total Runs Parsed: %d\n", count;
            printf "Average AP:  %.5f\n", sum_ap / count;
            printf "Average AUC: %.5f\n", sum_auc / count;
        } else {
            print "No metrics found to calculate.";
        }
    }' "$METRICS_FILE"
else
    echo "Metrics file not created. No successful runs or parsing failed."
fi