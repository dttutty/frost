# FROST

<p align="center">
  <img src="images/logo.svg" alt="FROST logo" width="180">
</p>
<p align="center">
  <strong>FROST: Forward-Looking Routing for Exact-Memory Multi-GPU TGNN Training</strong>
</p>
<p align="center">
  Research artifact repository for the FROST paper submission.
</p>
<p align="center">
  <a href="https://github.com/dttutty/frost/actions/workflows/manual-ruff.yml"><img src="https://github.com/dttutty/frost/actions/workflows/manual-ruff.yml/badge.svg" alt="ruff status"></a>
  <a href="https://github.com/dttutty/frost/actions/workflows/manual-mypy.yml"><img src="https://github.com/dttutty/frost/actions/workflows/manual-mypy.yml/badge.svg" alt="mypy status"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-2ea44f" alt="version 0.1.0">
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-3776AB" alt="python >=3.12">
</p>

## Overview

This repository accompanies the paper submission **FROST: Forward-Looking Routing for Exact-Memory Multi-GPU TGNN Training**.

FROST is a distributed training system for temporal graph neural networks (TGNNs) with exact memory semantics. The system uses a dedicated host process to perform sampling, access analysis, and forward-looking planning, while GPU worker processes execute model training. The implementation supports both memory-based TGNNs and non-memory baselines through a unified Hydra configuration interface, shared-memory data movement, and a C++/nanobind sequence sampler.

If you use this repository, its code, or its design ideas in research or derived systems, please cite the paper.

## Citation

```bibtex
@misc{zhao_frost,
  title  = {FROST: Forward-Looking Routing for Exact-Memory Multi-GPU TGNN Training},
  author = {Lei Zhao and Chul-Ho Lee},
  note   = {Paper submission accompanying this repository}
}
```

## What Is in This Repository?

- Distributed multi-process TGNN training via `torchrun`
- Hydra-based experiment configuration in [`configs/`](configs/)
- Core runtime, samplers, memory system, and model implementations in [`src/frost/`](src/frost/)
- EdgeBank baseline evaluation in [`eval_edgebank.py`](eval_edgebank.py)
- A one-command environment setup script in [`create_env.sh`](create_env.sh)

# Device selection
export CUDA_VISIBLE_DEVICES=0,1

# Async CUDA execution
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0

# Reduce allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# cuDNN autotuning
export TORCH_CUDNN_BENCHMARK=1

# NCCL communication
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# Reduce debug overhead
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=0
export LOG_LEVEL=WARNING

# CPU threading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONOPTIMIZE=2

# Clear common debug flags
unset TORCH_SHOW_CPP_STACKTRACES
unset TORCH_DISABLE_ADDR2LINE
```

### Runtime Model

FROST uses `torchrun`, but one rank is reserved for the host/planner process.

- `nproc_per_node = n_worker_gpus + 1`
- worker ranks train on GPUs
- the last rank acts as the host, orchestrating sampling, planning, and dispatch

Example:

- `CUDA_VISIBLE_DEVICES=0,1`
- `torchrun --nproc_per_node=3 ...`
- meaning: `2` GPU workers plus `1` host rank

### Canonical First Run

The following command is the canonical starting point for reproducing a memory-based FROST experiment on LASTFM with TGN:

```bash
torchrun --nproc_per_node=3 run.py \
  analyzer=tgn \
  dataset=LASTFM \
  encoder=tgn \
  mem_box_store=base \
  mem_updater=tgn \
  ngh_sampler=1_hop \
  mail_combine=last \
  compile=false \
  tf32=false
```

Practical notes:

- Use `compile=false` first when debugging runtime issues.
- Use `tf32=false` when you want stricter numerical comparisons during validation.

## Configuration Guide

The main experiment entry point is [`configs/config.yaml`](configs/config.yaml). FROST uses Hydra config groups to compose datasets, samplers, models, memory components, and runtime helpers.

### Defaults at a Glance

| Key | Default |
| --- | --- |
| `dataset` | `LASTFM` |
| `batch_size` | `4000` |
| `fanout` | `[10]` |
| `n_neg` | `1` |
| `final_emb_dim` | `128` |
| `d_time` | `128` |
| `compile` | `true` |
| `tf32` | `true` |

### 1. Root Experiment Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `epoch` | `1` | Number of training epochs. |
| `batch_size` | `4000` | Per-worker local batch size. The host groups `batch_size * n_gpu` samples into a macro-batch before dispatch. |
| `patience` | `50` | Early stopping patience in epochs. |
| `dropout` | `0.2` | Shared dropout hyperparameter forwarded into several encoders and memory modules. |
| `seed` | `null` | Shared random seed for sampling and reproducibility. `null` leaves runs unfixed. |
| `d_time` | `128` | Time encoding dimension. |
| `final_emb_dim` | `128` | Output embedding dimension used by downstream prediction and several encoders. |
| `dataset` | `LASTFM` | Dataset name under `DATA/`. |
| `ndim` | `null` | Node feature override. `null`: load from disk if present; `0`: disable node features; `>0`: generate random node features of that size. |
| `edim` | `null` | Edge feature override with the same semantics as `ndim`. |
| `mail_combine` | `null` | Mailbox aggregation mode on memory-model paths. Supported values in the current implementation are `last` and `mean`. |
| `optimizer` | `adam` | Worker optimizer. Supported values are `adam` and `sgd`. |
| `gpu_topo` | `[0, 1]` | Flat or nested GPU topology description used by the planner to build an inter-GPU distance matrix. |
| `fanout` | `[10]` | Per-hop neighbor counts used by samplers and some encoders. |
| `lr` | `0.0001` | Learning rate. |
| `n_neg` | `1` | Single source of truth for the number of negative samples per positive edge. It propagates through Hydra into the negative sampler and related components. |
| `handle_stale` | `false` | Allows planner fallback to stale leftovers when next-batch needs cannot be fully satisfied. |
| `compile` | `true` | Enables `torch.compile` for the assembled worker model. |
| `tf32` | `true` | Enables TF32 matmul and cuDNN behavior globally. |

### 2. Hydra Config Groups

| Group | Role | Common Choices in This Repo | User-Facing Status |
| --- | --- | --- | --- |
| `analyzer` | Host-side access counting and mail planning logic | `no_mem`, `tgn`, `jodie`, `dyrep`, `apan` | User-facing; choose per model family |
| `dataset_mgr` | Loads topology and features into shared memory | `base` | Mostly runtime plumbing |
| `encoder` | TGNN encoder architecture | `tgn`, `jodie`, `apan`, `dyrep`, `tgat`, `dygformer`, `graph_mixer`, `tcl` | User-facing |
| `loss_fn` | Training loss | `bce` | Mostly fixed in current experiments |
| `mem_box_store` | Shared memory/mailbox store | `base` or `null` | User-facing only for memory-based models |
| `mem_updater` | Memory update module | `tgn`, `jodie`, `dyrep`, `apan` or `null` | User-facing only for memory-based models |
| `neg_sampler` | Negative destination sampling | `base` | Mostly fixed, but now follows root `n_neg` |
| `ngh_sampler` | Temporal neighborhood sampling | `0_hop`, `1_hop`, `2_hop` | User-facing |
| `postman` | Neighbor-directed mail routing | `apan` or `null` | User-facing only for APAN |
| `precision` | Precision config placeholder | `base` | Present in config tree, but not part of the main experiment path |
| `predictor` | Downstream prediction head | `link` | Mostly fixed in current code |
| `ring_buf` | Shared-memory ring buffer sizing and layout | `base` | Mostly runtime plumbing |

### 3. Parameter Flow and Coupling

Several configuration values are intentionally shared across subsystems.

- `n_neg` flows from the root config into `neg_sampler`, analyzers, neighborhood samplers, the link predictor, and APAN postman-related components.
- `fanout` is consumed by the samplers and directly affects TGAT, GraphMixer, TCL, and DyGFormer behavior.
- `d_mem`, `boxsize`, and `edim` determine mailbox dimensionality through `d_mail = 2 * d_mem + edim`.
- `mem_box_store` should be enabled only for memory-based models. Non-memory models should leave `mem_box_store=null` and `mem_updater=null`.
- `mail_combine` matters only when mailbox/message aggregation is active on memory-model paths.

## Model-Specific Configuration Notes

### Memory-Based Models

#### TGN

TGN is a 1-hop memory-based model with GRU-style memory updates and an enabled mailbox store.

```bash
torchrun --nproc_per_node=3 run.py \
  analyzer=tgn \
  dataset=LASTFM \
  encoder=tgn \
  mem_box_store=base \
  mem_updater=tgn \
  ngh_sampler=1_hop \
  mail_combine=last \
  compile=false \
  tf32=false
```

#### JODIE

JODIE is a 0-hop memory-based model using the RNN-style memory updater.

```bash
torchrun --nproc_per_node=3 run.py \
  analyzer=jodie \
  dataset=LASTFM \
  encoder=jodie \
  mem_box_store=base \
  mem_updater=jodie \
  ngh_sampler=0_hop \
  mail_combine=last
```

#### DyRep

DyRep currently uses the JODIE-style analyzer path and an approximation for message construction inside FROST.

```bash
torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM \
  encoder=dyrep \
  analyzer=dyrep \
  mem_box_store=base \
  mem_updater=dyrep \
  ngh_sampler=0_hop \
  mail_combine=last
```

In the original DyRep formulation, the message contains `[self_mem, dst_graph_attn_emb, edge_feat]`. In FROST, mail is constructed before encoder execution, so the current approximation is `[self_mem, other_mem, edge_feat]`.

#### APAN

APAN requires `postman=apan`, `ngh_sampler=1_hop`, and `ngh_sampler.sample_4_neg=False`. It uses a transformer-style memory updater and a larger mailbox (`boxsize=10`).

```bash
torchrun --nproc_per_node=3 run.py \
  analyzer=apan \
  dataset=LASTFM \
  encoder=apan \
  mem_box_store=base \
  mem_updater=apan \
  ngh_sampler=1_hop \
  ngh_sampler.sample_4_neg=False \
  postman=apan \
  mail_combine=last \
  compile=false
```

### Non-Memory Models

#### TGAT

TGAT is a non-memory baseline that uses 2-hop temporal neighborhood sampling.

```bash
torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM \
  encoder=tgat \
  ngh_sampler=2_hop
```

#### GraphMixer

GraphMixer uses 1-hop neighborhoods. `num_tokens` should track `fanout[0]`; if `fanout[0] > num_tokens`, the sequence is truncated, and if `fanout[0] < num_tokens`, the remaining positions are zero-padded.

```bash
torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM \
  encoder=graph_mixer \
  ngh_sampler=1_hop \
  fanout=[20]
```

#### TCL

TCL uses 1-hop neighborhoods. `num_depths` should be `fanout[0] + 1`; mismatches lead to truncation or padding in the constructed sequence.

```bash
torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM \
  encoder=tcl \
  ngh_sampler=1_hop \
  fanout=[20]
```

#### DyGFormer

DyGFormer uses 1-hop neighborhoods. `max_input_sequence_length` truncates the root-plus-neighbor sequence after patch alignment, so it effectively caps the usable temporal context.

```bash
torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM \
  encoder=dygformer \
  ngh_sampler=1_hop \
  fanout=[32] \
  dropout=0.1
```

## EdgeBank Baseline

EdgeBank is included as a non-parametric baseline evaluator.

```bash
# Unlimited memory
python eval_edgebank.py --dataset LASTFM

# Time-window memory
python eval_edgebank.py --dataset LASTFM --memory_mode time_window_memory

# Repeat-threshold memory
python eval_edgebank.py --dataset LASTFM --memory_mode repeat_threshold

# Inductive split
python eval_edgebank.py --dataset LASTFM --split int_roll

# More negative samples
python eval_edgebank.py --dataset LASTFM --n_neg 5
```

## Notes and Limitations

- `precision` exists in [`configs/`](configs/) but is not meaningfully wired into the main launcher path used for the normal experiment workflow.
- `ring_buf.max_size` is pipeline depth per worker, not dataset size, queue length in steps, or total batch count.
- `gpu_topo` affects the planner's GPU distance matrix, not `CUDA_VISIBLE_DEVICES` or PyTorch rank assignment.
- `batch_size` is local per worker. It should not be interpreted as a global batch size.
- This README documents the supported experiment path for the current repository state. It does not attempt to expose every internal extension point.

## Repository Layout

- [`run.py`](run.py): Hydra entry point for distributed experiments
- [`configs/`](configs/): experiment configuration and Hydra config groups
- [`src/frost/`](src/frost/): runtime, samplers, memory system, and model implementations
- [`eval_edgebank.py`](eval_edgebank.py): standalone EdgeBank baseline evaluator
- [`DATA/`](DATA/): dataset directory expected by the training and evaluation pipeline
- [`tests/`](tests/): test suite for validation and regression checks

## Quick Command Reference

This section keeps the original command-oriented workflow for day-to-day use.

### Recommended Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0,1

# Async execution
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0

# Reduce allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# cuDNN autotuning
export TORCH_CUDNN_BENCHMARK=1

# NCCL communication
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# Reduce debug overhead
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=0
export LOG_LEVEL=WARNING

# CPU threading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONOPTIMIZE=2

# Clear common debug flags
unset TORCH_SHOW_CPP_STACKTRACES
unset TORCH_DISABLE_ADDR2LINE
```

### Quick Training Commands

Memory-based models:

```bash
# TGN
torchrun --nproc_per_node=3 run.py analyzer=tgn dataset=LASTFM encoder=tgn mem_box_store=base mem_updater=tgn ngh_sampler=1_hop mail_combine=last compile=false tf32=false

# APAN
torchrun --nproc_per_node=3 run.py analyzer=apan dataset=LASTFM encoder=apan mem_box_store=base mem_updater=apan ngh_sampler=1_hop ngh_sampler.sample_4_neg=False postman=apan mail_combine=last compile=false

# JODIE
torchrun --nproc_per_node=3 run.py analyzer=jodie dataset=LASTFM encoder=jodie mem_box_store=base mem_updater=jodie ngh_sampler=0_hop mail_combine=last

# DyRep
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=dyrep analyzer=dyrep mem_box_store=base mem_updater=dyrep ngh_sampler=0_hop mail_combine=last
```

Non-memory models:

```bash
# TGAT
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=tgat ngh_sampler=2_hop

# DyGFormer
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=dygformer ngh_sampler=1_hop fanout=[32] dropout=0.1

# GraphMixer
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=graph_mixer ngh_sampler=1_hop fanout=[20]

# TCL
torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=tcl ngh_sampler=1_hop fanout=[20]
```

EdgeBank:

```bash
python eval_edgebank.py --dataset LASTFM
python eval_edgebank.py --dataset LASTFM --memory_mode time_window_memory
python eval_edgebank.py --dataset LASTFM --memory_mode repeat_threshold
python eval_edgebank.py --dataset LASTFM --split int_roll
python eval_edgebank.py --dataset LASTFM --n_neg 5
```

## Troubleshooting

- If `CUDA_VISIBLE_DEVICES=0,1`, `--nproc_per_node` should usually be `3`, not `2`.
- If distributed training fails early, check whether external NCCL variables are overriding the intended runtime configuration.
- If throughput drops unexpectedly, first confirm that blocking CUDA or verbose debug settings were not enabled.
- For per-rank failures, inspect files such as [`rank_0.log`](rank_0.log).
