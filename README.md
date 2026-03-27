# FROST

<p align="center">
  <strong>FROST: Forward-Looking Routing for Exact-Memory Multi-GPU TGNN Training</strong>
</p>
<p align="center">
  Research artifact repository for the FROST paper submission.
</p>
<p align="center">
  <a href="https://github.com/dttutty/FROST/actions/workflows/manual-ruff.yml"><img src="https://github.com/dttutty/FROST/actions/workflows/manual-ruff.yml/badge.svg" alt="ruff status"></a>
  <a href="https://github.com/dttutty/FROST/actions/workflows/manual-mypy.yml"><img src="https://github.com/dttutty/FROST/actions/workflows/manual-mypy.yml/badge.svg" alt="mypy status"></a>
  <a href="https://github.com/dttutty/FROST/actions/workflows/manual-test.yml"><img src="https://github.com/dttutty/FROST/actions/workflows/manual-test.yml/badge.svg" alt="test status"></a>
  <a href="https://huggingface.co/datasets/dttutty/frost_dataset"><img src="https://img.shields.io/badge/Hugging%20Face-dataset%20repo-FFD21E?logo=huggingface&logoColor=000" alt="Hugging Face dataset repo"></a>
</p>

## Overview

FROST is a distributed training system for temporal graph neural networks (TGNNs) with exact memory semantics. The system uses a dedicated host process to perform sampling, access analysis, and forward-looking planning, while GPU worker processes execute model training. The implementation supports both memory-based TGNNs and non-memory baselines through a unified Hydra configuration interface, shared-memory data movement, and a C++/nanobind sequence sampler.

FROST is implemented directly on top of PyTorch and PyTorch Distributed. It does not depend on [DGL](https://github.com/dmlc/dgl), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter), or [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse). (That means fewer dependency headaches.)

If you use this repository, its code, or its design ideas in research or derived systems, please cite the paper.

## Citation

```bibtex
@misc{zhao_frost,
  title  = {FROST: Forward-Looking Routing for Exact-Memory Multi-GPU TGNN Training},
  author = {Lei Zhao and Chul-Ho Lee},
  note   = {Paper submission accompanying this repository}
}
```

## Quick Start

### 🛠️ Environment Setup

```bash
uv python install 3.12
uv sync --python 3.12
```

`uv` will honor the project-level Python pin in [`.python-version`](.python-version) and create a local `.venv`. You can either activate it with `source .venv/bin/activate` or prefix commands with `uv run`.

### 🤗 Download Datasets

FROST's curated datasets are hosted on the Hugging Face dataset repository [`dttutty/frost_dataset`](https://huggingface.co/datasets/dttutty/frost_dataset). The recommended workflow is to mirror that repository directly into [`DATA/`](DATA/) so the runtime sees the expected `DATA/<dataset>/...` layout.

```bash
# Download the full curated bundle into DATA/
uvx --from huggingface_hub hf download \
  dttutty/frost_dataset \
  --repo-type dataset \
  --local-dir DATA

# Or download only one dataset directory
uvx --from huggingface_hub hf download \
  dttutty/frost_dataset \
  --repo-type dataset \
  --include "lastfm/*" \
  --local-dir DATA
```

`hf download --local-dir DATA` preserves the repository structure under `DATA/` and writes incremental-download metadata under `DATA/.cache/huggingface/` (gitignored in this repo).

The Hugging Face mirror is the only supported dataset acquisition path in this repository.

For local runtime loading, `DatasetManager` expects `edges.csv` together with `full_graph_with_reverse_edges.npz`, plus optional top-level `node_features.npy` / `edge_features.npy` files or their `.pt` equivalents for local custom datasets.

### 🚀 Runtime Model

FROST uses `torchrun`, with one rank reserved for the host/planner process:

- `nproc_per_node = n_worker_gpus + 1`
- Worker ranks train on GPUs; the last rank acts as host

Example with 2 GPUs:

### ⚙️ Configuration

## Repository Layout

| Path | Description |
| --- | --- |
| [`run.py`](run.py) | Hydra entry point for distributed experiments |
| [`configs/`](configs/) | Experiment configuration and Hydra config groups |
| [`src/frost/`](src/frost/) | Runtime, samplers, memory system, and model implementations |
| [`eval_edgebank.py`](eval_edgebank.py) | Standalone EdgeBank baseline evaluator |
| [`DATA/`](DATA/) | Curated dataset bundle mirrored from Hugging Face |
| [`tests/`](tests/) | Test suite |
| [`scripts/`](scripts/) | Utility scripts (training and dataset helpers) |

## Notes and Limitations

- `ring_buf_size` is pipeline depth per worker, not dataset size or total batch count.
- The planner auto-detects visible-GPU topology from `CUDA_VISIBLE_DEVICES` and `nvidia-smi topo -m`, ranking peers as NVLink, same-NUMA, or cross-NUMA when possible.
- `batch_size` is local per worker, not a global batch size.
- Per-rank logs are written to the Hydra output directory (`outputs/<date>/<time>/rank_*.log`).

## Troubleshooting

- If `CUDA_VISIBLE_DEVICES=0,1`, `--nproc_per_node` should be `3`, not `2`.
- If distributed training fails early, check whether external NCCL variables are overriding the intended runtime configuration.
- If throughput drops unexpectedly, confirm that `CUDA_LAUNCH_BLOCKING` and verbose debug settings are not enabled.
