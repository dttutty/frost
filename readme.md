# 显卡选择 (保持不变)
export CUDA_VISIBLE_DEVICES=0,1

# 关闭调试阻塞，开启异步执行（性能提升最明显）
export CUDA_LAUNCH_BLOCKING=0

# 降低日志开销
export LOG_LEVEL=WARNING
export HYDRA_FULL_ERROR=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=VERSION

# 分布式训练与通信优化
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# 内存管理优化 (减少显存分配碎片带来的停顿)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# CPU 并行加速 (根据你的 CPU 核心数调整，建议设为 4, 8 或 16)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 清除之前可能设置的调试堆栈追踪 (防止干扰)
unset TORCH_SHOW_CPP_STACKTRACES
unset TORCH_DISABLE_ADDR2LINE

Memory Based Model:
TGN:
torchrun --nproc_per_node=3 run.py analyzer=tgn dataset=LASTFM encoder=tgn mem_box_store=base mem_updater=tgn ngh_sampler=1_hop mail_combine=last

APAN:
torchrun --nproc_per_node=3 run.py analyzer=apan dataset=LASTFM encoder=apan mem_box_store=base mem_updater=apan ngh_sampler=1_hop ngh_sampler.sample_4_neg=False postman=apan mail_combine=last

JODIE:
torchrun --nproc_per_node=3 run.py analyzer=jodie dataset=LASTFM encoder=jodie mem_box_store=base mem_updater=jodie neg_sampler=0_hop mail_combine=last


No Memory Model:
TGAT:
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=tagt ngh_sampler=2_hop

DyGFormer:
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=dygformer ngh_sampler=1_hop

GraphMixer:
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=graphmixer ngh_sampler=1_hop
