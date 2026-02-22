#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate simple_py310
# 最终版自动消融实验运行脚本
# 说明：
# 1. 即使某个命令运行失败，也不会中断整个脚本的执行。
# 2. 所有实验的日志（标准输出和错误输出）都将保存在 log_20250806 目录中。
# 3. 每个实验配置都将运行 dedup=True 和 dedup=False 两个版本。

# 定义并创建日志目录
LOG_DIR="log_202508063"
echo "INFO: Creating log directory at $LOG_DIR"
mkdir -p $LOG_DIR


# # --- 0. 预热文件系统 (Preheat OS) ---
# LOG_FILE_TRUE="$LOG_DIR/0_baseline_dedup_true.log"
# echo "======> [0. BASELINE] Running Baseline, dedup=True. Log: $LOG_FILE_TRUE"
# python -u /home/sqp17/Projects/lei_pipe/train.py \
#     --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer 2 \
#     --dedup 2>&1 | tee "$LOG_FILE_TRUE";

# --- 0. 运行基线配置 (Baseline Configuration) ---
# LOG_FILE_TRUE="$LOG_DIR/0_baseline_dedup_true.log"
# echo "======> [0. BASELINE] Running Baseline, dedup=True. Log: $LOG_FILE_TRUE"
# python -u /home/sqp17/Projects/lei_pipe/train.py \
#     --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer 2 \
#     --dedup 2>&1 | tee "$LOG_FILE_TRUE";

# LOG_FILE_FALSE="$LOG_DIR/0_baseline_dedup_false.log"
# echo "======> [0. BASELINE] Running Baseline, dedup=False. Log: $LOG_FILE_FALSE"
# python -u /home/sqp17/Projects/lei_pipe/train.py \
#     --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer 2 \
#     2>&1 | tee "$LOG_FILE_FALSE";


# # --- 1. 测试 rand_edge_features 参数 ---
# for val in 256 384; do
#     LOG_FILE_TRUE="$LOG_DIR/1_rand_edge_features_${val}_dedup_true.log"
#     echo "======> [1. Test rand_edge_features] Testing ${val}, dedup=True. Log: $LOG_FILE_TRUE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features $val --rand_node_features $val --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer 2 --dedup 2>&1 | tee "$LOG_FILE_TRUE";

#     LOG_FILE_FALSE="$LOG_DIR/1_rand_edge_features_${val}_dedup_false.log"
#     echo "======> [1. Test rand_edge_features] Testing ${val}, dedup=False. Log: $LOG_FILE_FALSE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features $val --rand_node_features $val --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer 2 2>&1 | tee "$LOG_FILE_FALSE";
# done


# # --- 2. 测试 mem_dim 参数 ---
# for val in 200 300 400 500; do
#     LOG_FILE_TRUE="$LOG_DIR/2_mem_dim_${val}_dedup_true.log"
#     echo "======> [2. Test mem_dim] Testing ${val}, dedup=True. Log: $LOG_FILE_TRUE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128  --mem_dim $val --batch_size 600 --n_neighbor 10 --n_layer 2 --dedup 2>&1 | tee "$LOG_FILE_TRUE";

#     LOG_FILE_FALSE="$LOG_DIR/2_mem_dim_${val}_dedup_false.log"
#     echo "======> [2. Test mem_dim] Testing ${val}, dedup=False. Log: $LOG_FILE_FALSE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim $val --batch_size 600 --n_neighbor 10 --n_layer 2 2>&1 | tee "$LOG_FILE_FALSE";
# done


# # --- 3. 测试 batch_size 参数 ---
# for val in 300 900 1200 1500; do
#     LOG_FILE_TRUE="$LOG_DIR/3_batch_size_${val}_dedup_true.log"
#     echo "======> [3. Test batch_size] Testing ${val}, dedup=True. Log: $LOG_FILE_TRUE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size $val --n_neighbor 10 --n_layer 2 --dedup 2>&1 | tee "$LOG_FILE_TRUE";

#     LOG_FILE_FALSE="$LOG_DIR/3_batch_size_${val}_dedup_false.log"
#     echo "======> [3. Test batch_size] Testing ${val}, dedup=False. Log: $LOG_FILE_FALSE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size $val --n_neighbor 10 --n_layer 2 2>&1 | tee "$LOG_FILE_FALSE";
# done


# # --- 4. 测试 neighbor number (n_neighbor) 参数 ---
# for val in 20 30 40 50; do
#     LOG_FILE_TRUE="$LOG_DIR/4_n_neighbor_${val}_dedup_true.log"
#     echo "======> [4. Test n_neighbor] Testing ${val}, dedup=True. Log: $LOG_FILE_TRUE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor $val --n_layer 2 --dedup 2>&1 | tee "$LOG_FILE_TRUE";
    
#     LOG_FILE_FALSE="$LOG_DIR/4_n_neighbor_${val}_dedup_false.log"
#     echo "======> [4. Test n_neighbor] Testing ${val}, dedup=False. Log: $LOG_FILE_FALSE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor $val --n_layer 2 2>&1 | tee "$LOG_FILE_FALSE";
# done


# --- 5. 测试 layers (n_layer) 参数 ---
# for val in 1 3; do
#     LOG_FILE_TRUE="$LOG_DIR/5_n_layer_${val}_dedup_true.log"
#     echo "======> [5. Test n_layer] Testing ${val}, dedup=True. Log: $LOG_FILE_TRUE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer $val --dedup 2>&1 | tee "$LOG_FILE_TRUE";
    
#     LOG_FILE_FALSE="$LOG_DIR/5_n_layer_${val}_dedup_false.log"
#     echo "======> [5. Test n_layer] Testing ${val}, dedup=False. Log: $LOG_FILE_FALSE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer $val 2>&1 | tee "$LOG_FILE_FALSE";
# done

# (请确保 LOG_DIR 变量已经提前设置好, 例如: LOG_DIR="./logs")

# --- 6. 测试 n_history (默认值: 3) ---
# 测试值: 2, 4
# 对每个值，分别测试 dedup 为 True 和 False 的情况
for val in 2 4; do
    # Case 1: dedup = True
    LOG_FILE_TRUE="$LOG_DIR/6_n_history_${val}_dedup_true.log"
    echo "======> [6. Test n_history] Testing ${val}, dedup=True. Log: $LOG_FILE_TRUE"
    python -u /home/sqp17/Projects/lei_pipe/train.py \
        --data LASTFM \
        --config config/DySAT.yml \
        --gpu 6 \
        --rand_edge_features 128 \
        --rand_node_features 128 \
        --mem_dim 100 \
        --batch_size 600 \
        --n_neighbor 10 \
        --n_history $val \
        --dedup 2>&1 | tee "$LOG_FILE_TRUE";
    
    # Case 2: dedup = False
    LOG_FILE_FALSE="$LOG_DIR/6_n_history_${val}_dedup_false.log"
    echo "======> [6. Test n_history] Testing ${val}, dedup=False. Log: $LOG_FILE_FALSE"
    python -u /home/sqp17/Projects/lei_pipe/train.py \
        --data LASTFM \
        --config config/DySAT.yml \
        --gpu 6 \
        --rand_edge_features 128 \
        --rand_node_features 128 \
        --mem_dim 100 \
        --batch_size 600 \
        --n_neighbor 10 \
        --n_history $val 2>&1 | tee "$LOG_FILE_FALSE";
done

# --- 7. 测试 history_duration (默认值: 10000) ---
# 测试值: 20000, 30000
# 对每个值，分别测试 dedup 为 True 和 False 的情况
for val in 20000 30000; do
    # Case 1: dedup = True
    LOG_FILE_TRUE="$LOG_DIR/7_history_duration_${val}_dedup_true.log"
    echo "======> [7. Test history_duration] Testing ${val}, dedup=True. Log: $LOG_FILE_TRUE"
    python -u /home/sqp17/Projects/lei_pipe/train.py \
        --data LASTFM \
        --config config/DySAT.yml \
        --gpu 6 \
        --rand_edge_features 128 \
        --rand_node_features 128 \
        --mem_dim 100 \
        --batch_size 600 \
        --n_neighbor 10 \
        --history_duration $val \
        --dedup 2>&1 | tee "$LOG_FILE_TRUE";
        
    # Case 2: dedup = False
    LOG_FILE_FALSE="$LOG_DIR/7_history_duration_${val}_dedup_false.log"
    echo "======> [7. Test history_duration] Testing ${val}, dedup=False. Log: $LOG_FILE_FALSE"
    python -u /home/sqp17/Projects/lei_pipe/train.py \
        --data LASTFM \
        --config config/DySAT.yml \
        --gpu 6 \
        --rand_edge_features 128 \
        --rand_node_features 128 \
        --mem_dim 100 \
        --batch_size 600 \
        --n_neighbor 10 \
        --history_duration $val 2>&1 | tee "$LOG_FILE_FALSE";
done

# # --- 6. 测试 config 参数 ---
# for val in APAN TGAT DySAT JODIE; do
#     LOG_FILE_TRUE="$LOG_DIR/6_config_${val}_dedup_true.log"
#     echo "======> [6. Test config] Testing ${val}, dedup=True. Log: $LOG_FILE_TRUE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/${val}.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer 2 --dedup 2>&1 | tee "$LOG_FILE_TRUE";

#     LOG_FILE_FALSE="$LOG_DIR/6_config_${val}_dedup_false.log"
#     echo "======> [6. Test config] Testing ${val}, dedup=False. Log: $LOG_FILE_FALSE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data LASTFM --config config/${val}.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer 2 2>&1 | tee "$LOG_FILE_FALSE";
# done


# # --- 7. 测试 dataset 参数 ---
# for val in WIKI MOOC REDDIT; do
#     LOG_FILE_TRUE="$LOG_DIR/7_dataset_${val}_dedup_true.log"
#     echo "======> [7. Test dataset] Testing ${val}, dedup=True. Log: $LOG_FILE_TRUE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data $val --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer 2 --dedup 2>&1 | tee "$LOG_FILE_TRUE";

#     LOG_FILE_FALSE="$LOG_DIR/7_dataset_${val}_dedup_false.log"
#     echo "======> [7. Test dataset] Testing ${val}, dedup=False. Log: $LOG_FILE_FALSE"
#     python -u /home/sqp17/Projects/lei_pipe/train.py --data $val --config config/DySAT.yml --gpu 6 --rand_edge_features 128  --rand_node_features 128 --mem_dim 100 --batch_size 600 --n_neighbor 10 --n_layer 2 2>&1 | tee "$LOG_FILE_FALSE";
# done


echo "======> All experiments finished. Check logs in '$LOG_DIR' directory."
