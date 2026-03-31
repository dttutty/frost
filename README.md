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

FROST is a distributed training system for temporal graph neural networks (TGNNs) with exact memory semantics. A dedicated host rank performs sampling, access analysis, and forward-looking planning, while GPU worker ranks execute model training.

The implementation is built directly on top of PyTorch and PyTorch Distributed. It does not depend on DGL, PyG, `pytorch_scatter`, or `pytorch_sparse`.

## Citation

```bibtex
@misc{zhao_frost,
  title  = {FROST: Forward-Looking Routing for Exact-Memory Multi-GPU TGNN Training},
  author = {Lei Zhao and Chul-Ho Lee},
  note   = {Paper submission accompanying this repository}
}
```

## Quick Start

### Environment

```bash
uv python install 3.12
uv sync --python 3.12
```

`uv` will honor [`.python-version`](.python-version), create a local `.venv`, and let you run commands via `uv run`.

### Datasets

The supported dataset source is the Hugging Face dataset repository [`dttutty/frost_dataset`](https://huggingface.co/datasets/dttutty/frost_dataset). Mirror it into [`DATA/`](DATA/) so runtime paths match `DATA/<dataset>/...`.

```bash
uvx --from huggingface_hub hf download \
  dttutty/frost_dataset \
  --repo-type dataset \
  --local-dir DATA
```

To fetch a single dataset:

```bash
uvx --from huggingface_hub hf download \
  dttutty/frost_dataset \
  --repo-type dataset \
  --include "LASTFM/*" \
  --local-dir DATA
```

### Runtime Shape

FROST uses `torchrun` with one extra rank reserved for the host/planner process:

- `nproc_per_node = n_worker_gpus + 1`
- Worker ranks use GPUs; the last rank is the host

Example with 2 GPUs:

## Common Entry Points

List the shared presets:

Launch a preset:

Inspect the equivalent `torchrun` command without launching:

Run a manual experiment:

```bash
uv run torchrun --nproc_per_node=3 run.py <hydra overrides...>
```

## Documentation

## Repository Layout

| Path | Description |
| --- | --- |
| [`run.py`](run.py) | Hydra entry point for distributed experiments |
| [`configs/`](configs/) | Experiment configuration and Hydra config groups |
| [`src/frost/`](src/frost/) | Runtime, samplers, memory system, and model implementations |
| [`scripts/`](scripts/) | Utility scripts for presets, sweeps, and dataset helpers |
| [`tests/`](tests/) | Test suite |
| [`DATA/`](DATA/) | Curated dataset bundle mirrored from Hugging Face |
| [`eval_edgebank.py`](eval_edgebank.py) | Standalone EdgeBank baseline evaluator |

## Notes

- `batch_size` is local per worker, not global.
- `ring_buf_size` is pipeline depth per worker, not dataset size or total batch count.
- Per-rank logs are written under Hydra outputs such as `outputs/<date>/<time>/rank_*.log`.
- GPU topology is inferred from `CUDA_VISIBLE_DEVICES` and `nvidia-smi topo -m` when available.

## Troubleshooting

- If `CUDA_VISIBLE_DEVICES=0,1`, then `--nproc_per_node` should be `3`, not `2`.
- If distributed training fails very early, check whether external NCCL environment variables are conflicting with the intended runtime setup.
- If throughput drops unexpectedly, confirm that `CUDA_LAUNCH_BLOCKING` or verbose debug flags are not still enabled.
- If CAWN is much slower than TGN, TGAT, or GraphMixer, reduce `batch_size` first, then `encoder.fanout`.
