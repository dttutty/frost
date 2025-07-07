#!/bin/bash

# --- Configuration ---
# 请确保这里的路径是正确的
TORCHRUN_BIN="/home/sqp17/miniconda3/envs/simple_py310/bin/torchrun"
CONFIG_DIR="/home/sqp17/Projects/tgl/config"
DATASET="WIKI"

# !! 重要: 定义一个包含所有可用GPU ID的列表 !!
# 这里我们排除了 0, 1, 2
AVAILABLE_GPUS=(0 1 4 5 6)

# 计算可用GPU的数量
NUM_AVAILABLE_GPUS=${#AVAILABLE_GPUS[@]}

# --- Pre-run setup ---
# 根据您的要求，创建包含数据集名称的日志子目录
LOG_DIR="logs/${DATASET}"
mkdir -p "$LOG_DIR"

# --- Job Launching ---
# 随机生成一个起始端口号 (范围: 20000-59999)
base_port=$(( RANDOM % 40000 + 20000 ))
job_id=0

echo "Using random base port: $base_port"
echo "Available GPUs for jobs: ${AVAILABLE_GPUS[*]}"

# 从 8 开始，每次翻倍，直到 8192
value=8
while [ $value -le 8192 ]; do
    # --- 新的GPU分配逻辑 ---
    # 1. 计算在可用GPU列表中的索引
    index=$(( job_id % NUM_AVAILABLE_GPUS ))
    # 2. 从列表中获取实际的GPU ID
    gpu_id=${AVAILABLE_GPUS[$index]}
    
    # 构造新的日志文件完整路径
    log_file="${LOG_DIR}/TGN_${value}.log"

    echo "Launching config TGN_${value}.yml on GPU $gpu_id (port $base_port)..."
    echo "Log file will be saved to: $log_file"

    # !!! --- 重要提醒 --- !!!
    # 下面的 --nproc_per_node=2 与 CUDA_VISIBLE_DEVICES=$gpu_id (单GPU) 的设置不匹配。
    # 强烈建议您将其改回 --nproc_per_node=1 来匹配单GPU的分配策略。
    CUDA_VISIBLE_DEVICES=$gpu_id TORCH_DISTRIBUTED_DEBUG=DETAIL \
    $TORCHRUN_BIN \
    --nproc_per_node=2 \
    --master_addr=127.0.0.1 \
    --master_port=$base_port \
    train_dist.py \
    --omp_num_threads=4 \
    --data $DATASET \
    --config "${CONFIG_DIR}/TGN_${value}.yml" \
    > "$log_file" 2>&1 &

    # 为下一个任务准备 (端口+1, 确保本次任务内部不冲突)
    value=$(( value * 2 ))
    base_port=$(( base_port + 1 ))
    job_id=$(( job_id + 1 ))
done

echo "All jobs launched in background. Check logs in ${LOG_DIR}/ for output."
echo "Use 'jobs' or 'ps -u $USER -f | grep train_dist.py' to see running processes."
