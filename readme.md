# Lei Pipe / FROST 速查手册

这个 README 主要用于**快速启动训练与评估**，并整理了常用环境变量与不同模型的推荐命令。

---

## 1) 运行环境变量（建议）

> 目标：减少调试开销、提升吞吐、降低显存碎片。

```bash
# GPU 选择
export CUDA_VISIBLE_DEVICES=0,1

# 异步执行（关键性能项）
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0

# 显存管理：减少碎片化重分配
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# cuDNN 自动调优
export TORCH_CUDNN_BENCHMARK=1

# NCCL 通信
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# 关闭调试开销
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=0
export LOG_LEVEL=WARNING

# CPU 并行（通常设为物理核数的一半）
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONOPTIMIZE=2

# 清除常见调试变量
unset TORCH_SHOW_CPP_STACKTRACES
unset TORCH_DISABLE_ADDR2LINE


Memory Based Model:
TGN:
torchrun --nproc_per_node=3 run.py analyzer=tgn dataset=LASTFM encoder=tgn mem_box_store=base mem_updater=tgn ngh_sampler=1_hop mail_combine=last

APAN:
torchrun --nproc_per_node=3 run.py analyzer=apan dataset=LASTFM encoder=apan mem_box_store=base mem_updater=apan ngh_sampler=1_hop ngh_sampler.sample_4_neg=False postman=apan mail_combine=last

JODIE:
torchrun --nproc_per_node=3 run.py analyzer=jodie dataset=LASTFM encoder=jodie mem_box_store=base mem_updater=jodie ngh_sampler=0_hop mail_combine=last


No Memory Model:
TGAT:
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=tgat ngh_sampler=2_hop

DyGFormer:
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=dygformer ngh_sampler=1_hop fanout=[32]

GraphMixer:
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=graph_mixer ngh_sampler=1_hop fanout=[20]

num_tokens 应与 fanout[0] 保持一致（如 fanout=[20] num_tokens=20）。若 fanout > num_tokens，会自动截断；若 fanout < num_tokens，会自动补零

TCL:
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=tcl ngh_sampler=1_hop fanout=[20]
num_depths = fanout[0] + 1，应与 fanout 保持一致（如 fanout=[20] → num_depths=21）


EdgeBank:
# 三种记忆模式
python eval_edgebank.py --dataset LASTFM
python eval_edgebank.py --dataset LASTFM --memory_mode time_window_memory
python eval_edgebank.py --dataset LASTFM --memory_mode repeat_threshold

# Inductive split
python eval_edgebank.py --dataset LASTFM --split int_roll

# 调整负采样数量
python eval_edgebank.py --dataset LASTFM --n_neg 1
```

---

## 7) 常见排查建议

- 先确认 `CUDA_VISIBLE_DEVICES` 与 `--nproc_per_node` 匹配。  
- 多卡异常优先查看 NCCL 相关变量是否被外部覆盖。  
- 吞吐下降时先检查是否误开启调试选项（如阻塞执行、详细错误栈）。
