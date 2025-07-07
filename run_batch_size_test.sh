#!/bin/bash

# --- 1. 预检：检查是否提供了配置前缀 ---
# 脚本现在需要一个命令行参数来确定要运行哪种模型配置
if [ -z "$1" ]; then
    echo "错误: 请提供配置前缀作为参数。"
    echo "用法: $0 <CONFIG_PREFIX>"
    echo "例如: $0 TGN"
    echo "      $0 TGAT"
    exit 1
fi

# --- 2. 配置区 (Configuration) ---
# 从命令行参数获取配置前缀
CONFIG_PREFIX=$1

# 使用您指定的新Batch Size列表
BATCH_SIZES=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000)
# BATCH_SIZES=(500 )

# 请确保这里的路径是正确的
PYTHON_BIN="/home/sqp17/miniconda3/envs/simple_py310/bin/python"
PROGRAM_PATH="/home/sqp17/Projects/tgl/train.py"
CONFIG_DIR="/home/sqp17/Projects/tgl/config"
DATASET="MOOC"

# 定义所有可用的GPU ID
AVAILABLE_GPUS=( 2 3 4 5 6 7)

# 计算可用GPU的数量
NUM_AVAILABLE_GPUS=${#AVAILABLE_GPUS[@]}

# --- 3. 运行前设置 (Pre-run setup) ---
# 创建日志子目录
LOG_DIR="logs/${DATASET}"
mkdir -p "$LOG_DIR"

# --- 4. 任务启动 (Job Launching) ---
job_id=0

echo "Configuration prefix set to: ${CONFIG_PREFIX}"
echo "Will run for batch sizes: ${BATCH_SIZES[*]}"
echo "Available GPUs for jobs: ${AVAILABLE_GPUS[*]}"
echo "--------------------------------------------------"

# 遍历 BATCH_SIZES 列表中的每一个值
for value in "${BATCH_SIZES[@]}"; do
    # --- GPU分配逻辑 ---
    # 1. 计算在可用GPU列表中的索引
    index=$(( job_id % NUM_AVAILABLE_GPUS ))
    # 2. 从列表中获取实际的GPU ID
    gpu_id=${AVAILABLE_GPUS[$index]}

    # 使用变量构造配置文件名 (e.g., TGN_500)
    config_file_name="${CONFIG_PREFIX}_${value}"
    
    # 构造包含数据集和配置信息的完整模型名
    full_model_name="${DATASET}_${config_file_name}_SINGLE_GPU"
    
    # 构造日志文件名
    log_file="${LOG_DIR}/${config_file_name}.log"
    
    # 构造配置文件完整路径
    config_file_path="${CONFIG_DIR}/${config_file_name}.yml"

    # 检查配置文件是否存在
    if [ ! -f "$config_file_path" ]; then
        echo "警告: 配置文件不存在，跳过任务: $config_file_path"
    else
        echo "Launching config ${config_file_name}.yml on GPU $gpu_id..."
        echo "Log file will be saved to: $log_file"
        echo "Model name for this job will be: ${full_model_name}"

        # 启动Python训练任务
        nohup $PYTHON_BIN -u $PROGRAM_PATH \
          --data $DATASET \
          --config "$config_file_path" \
          --model_name "${full_model_name}" \
          --gpu "${gpu_id}" \
          2>&1 | tee "$log_file" &
        
        # 在启动一个任务后，可以稍微等待一下，避免瞬间启动过多进程导致系统负载过高
        sleep 2
    fi
    
    # 为下一个任务准备
    job_id=$(( job_id + 1 ))
done

echo "--------------------------------------------------"
echo "All valid jobs launched in background. Check logs in ${LOG_DIR}/ for output."
echo "Use 'jobs' or 'ps -u $USER -f | grep train.py' to see running processes."
