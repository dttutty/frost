#!/bin/bash

# 可用 GPU 数量
max_gpus=8

# 所有 dataset 和 config 组合（去掉 DySAT）
datasets=("MOOC" "WIKI" "REDDIT" "LASTFM")
configs=("APAN" "JODIE" "TGAT" "TGN")

# 日志目录
log_dir="tgl_batch_plan_logs"
mkdir -p "$log_dir"

# 任务编号，用于分配 GPU
task_id=0

# 遍历所有组合
for dataset in "${datasets[@]}"; do
    for config in "${configs[@]}"; do
        gpu_id=$(( task_id % max_gpus ))
        log_file="${log_dir}/${dataset}_${config}.log"

        echo "Launching: dataset=$dataset config=$config on GPU $gpu_id"

        python3 ~/Projects/lei_pipe/train.py \
            --data "$dataset" \
            --config "config/${config}.yml" \
            --gpu "$gpu_id" \
            > "$log_file" 2>&1 &

        ((task_id++))
    done
done

# 等待所有后台任务完成
wait

echo "All tasks completed."
