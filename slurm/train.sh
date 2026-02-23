# 1. 设置优化环境变量 (由 launch.json 净化而来)
export CUDA_VISIBLE_DEVICES=2,3
export TORCH_DISTRIBUTED_DEBUG=OFF          # 关闭分布式调试
export LOG_LEVEL=WARNING                       # 减少日志输出 IO
export NUMBA_DEBUG=0
export CUDA_LAUNCH_BLOCKING=0               # 关键：允许 GPU 异步执行 (极快)
export TORCH_USE_CUDA_DSA=0                 # 关闭设备端断言
export NCCL_DEBUG=WARN                   # 仅显示版本，减少 NCCL 日志
export TORCH_CUDNN_BENCHMARK=1              # 开启 CuDNN 自动寻优 (加速卷积)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:2
export OMP_NUM_THREADS=8                    # 防止 CPU 过度调度 (根据你的 CPU 核数调整)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONOPTIMIZE=1
# 2. 执行命令 (使用 python -O 启动)
# 请手动替换 YOUR_DATASET 和 YOUR_CONFIG_NAME
/home/sqp17/miniconda3/envs/simple_py310/bin/python -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_addr=127.0.0.1 \
    --master_port=29506 \
    train_dist.py \
    --dataset LASTFM \
    --config /home/sqp17/Projects/lei_pipe/config/dist/TGN.yml \
    --rnd_edim 128 \
    --rnd_ndim 128

# # 1. 设置优化环境变量 (由 launch.json 净化而来)
# export CUDA_VISIBLE_DEVICES=0,1
# export TORCH_DISTRIBUTED_DEBUG=OFF          # 关闭分布式调试
# export LOG_LEVEL=INFO                       # 减少日志输出 IO
# export NUMBA_DEBUG=0
# export CUDA_LAUNCH_BLOCKING=0               # 关键：允许 GPU 异步执行 (极快)
# export TORCH_USE_CUDA_DSA=0                 # 关闭设备端断言
# export NCCL_DEBUG=WARN                   # 仅显示版本，减少 NCCL 日志
# export TORCH_CUDNN_BENCHMARK=1              # 开启 CuDNN 自动寻优 (加速卷积)
# # export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

# export PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:2
# export OMP_NUM_THREADS=8                    # 防止 CPU 过度调度 (根据你的 CPU 核数调整)
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_BLOCKING_WAIT=0
# export MKL_NUM_THREADS=8
# export TOKENIZERS_PARALLELISM=false

# /home/sqp17/miniconda3/envs/simple_py310/bin/python \
#     -m viztracer \
#     --min_duration 200us \
#     --output_file distributed_trace.json \
#     -m torch.distributed.run \
#     --nproc_per_node=3 \
#     --master_addr=127.0.0.1 \
#     --master_port=29505 \
#     train_dist.py \
#     --dataset LASTFM \
#     --config /home/sqp17/Projects/lei_pipe/config/dist/TGN.yml \
#     --rnd_edim 128 \
#     --rnd_ndim 128
