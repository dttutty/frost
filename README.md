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
  <a href="https://github.com/dttutty/FROST/actions/workflows/manual-ruff.yml"><img src="https://github.com/dttutty/FROST/actions/workflows/manual-ruff.yml/badge.svg" alt="ruff status"></a>
  <a href="https://github.com/dttutty/FROST/actions/workflows/manual-mypy.yml"><img src="https://github.com/dttutty/FROST/actions/workflows/manual-mypy.yml/badge.svg" alt="mypy status"></a>
  <a href="https://github.com/dttutty/FROST/actions/workflows/manual-test.yml"><img src="https://github.com/dttutty/FROST/actions/workflows/manual-test.yml/badge.svg" alt="test status"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-2ea44f" alt="version 0.1.0">
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-3776AB" alt="python >=3.12">
</p>

## Overview

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

## Quick Start

### Environment Setup

```bash
uv python install 3.12
uv sync --python 3.12
```

`uv` will honor the project-level Python pin in [`.python-version`](.python-version) and create a local `.venv`. You can either activate it with `source .venv/bin/activate` or prefix commands with `uv run`.

### Download Datasets

```bash
bash scripts/download_data.sh
```

Each dataset is expected under `DATA/<DATASET>/`. Required: `ext_full.npz`, `edges.csv`. Optional: `node_features.pt`, `edge_features.pt`.

### Runtime Model

FROST uses `torchrun`, with one rank reserved for the host/planner process:

- `nproc_per_node = n_worker_gpus + 1`
- Worker ranks train on GPUs; the last rank acts as host

Example with 2 GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
uv run torchrun --nproc_per_node=3 run.py \
  analyzer=tgn dataset=LASTFM encoder=tgn \
  mem_box_store=base mem_updater=tgn \
  ngh_sampler=1_hop mail_combine=last
```

### Configuration

See [`configs/README.md`](configs/README.md) for the full configuration reference, all model commands, and parameter coupling details.

## Repository Layout

| Path | Description |
| --- | --- |
| [`run.py`](run.py) | Hydra entry point for distributed experiments |
| [`configs/`](configs/) | Experiment configuration and Hydra config groups |
| [`src/frost/`](src/frost/) | Runtime, samplers, memory system, and model implementations |
| [`eval_edgebank.py`](eval_edgebank.py) | Standalone EdgeBank baseline evaluator |
| [`DATA/`](DATA/) | Dataset directory |
| [`tests/`](tests/) | Test suite |
| [`scripts/`](scripts/) | Utility scripts (data download, training, etc.) |

## Notes and Limitations

- `ring_buf.max_size` is pipeline depth per worker, not dataset size or total batch count.
- `gpu_topo` affects the planner's GPU distance matrix, not `CUDA_VISIBLE_DEVICES` or PyTorch rank assignment.
- `batch_size` is local per worker, not a global batch size.
- Per-rank logs are written to the Hydra output directory (`outputs/<date>/<time>/rank_*.log`).

## Troubleshooting

- If `CUDA_VISIBLE_DEVICES=0,1`, `--nproc_per_node` should be `3`, not `2`.
- If distributed training fails early, check whether external NCCL variables are overriding the intended runtime configuration.
- If throughput drops unexpectedly, confirm that `CUDA_LAUNCH_BLOCKING` and verbose debug settings are not enabled.
