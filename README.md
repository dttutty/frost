# FROST: Accelerating Temporal Graph Neural Network Training via a Unified Multi-GPU Framework

<p align="center">
  Research artifact repository for the FROST paper submission.
</p>

## Overview

FROST is a distributed training system for temporal graph neural networks (TGNNs) with exact memory semantics. A dedicated host rank performs sampling, access analysis, and forward-looking planning, while GPU worker ranks execute model training.

The implementation is built directly on top of PyTorch and PyTorch Distributed. It does not depend on DGL, PyG, `pytorch_scatter`, or `pytorch_sparse`.

## Quick Start

### Environment

```bash
uv python install 3.12
uv sync --python 3.12
```

`uv` will honor [`.python-version`](.python-version), create a local `.venv`, and let you run commands via `uv run`.

### Datasets

Mirror the prepared dataset bundle into the locally ignored `DATA/` path so runtime paths match `DATA/<dataset>/...`. `DATA/` may be a directory or a symbolic link to shared storage. If you host the datasets on Hugging Face, set the repository id via an environment variable before downloading.

```bash
export HF_DATASET_REPO=<your-dataset-repo>
uvx --from huggingface_hub hf download \
  "$HF_DATASET_REPO" \
  --repo-type dataset \
  --local-dir DATA
```

To fetch a single dataset:

```bash
export HF_DATASET_REPO=<your-dataset-repo>
uvx --from huggingface_hub hf download \
  "$HF_DATASET_REPO" \
  --repo-type dataset \
  --include "LASTFM/*" \
  --local-dir DATA
```

For `tgbl-wiki-v2`, keep the official TGB download outside the repository and
convert only the FROST runtime files into `DATA/`:

```bash
# Expected official files: ~/datasets/tgb/raw/tgbl_wiki/{ml_tgbl-wiki.pkl,...}
uv run python scripts/prepare_tgbl_wiki.py \
  --raw-dir ~/datasets/tgb/raw/tgbl_wiki \
  --output-dir DATA/TGBL_WIKI
```

`DATA/` can itself be a symlink such as `DATA -> ~/datasets/frost_dataset`.
The converter links the large official negative-sample files instead of copying
them. See the TGB command under [configuration](configs/README.md#tgb-link-prediction).

### Runtime Shape

FROST uses `torchrun` with one extra rank reserved for the host/planner process:

- `nproc_per_node = n_worker_gpus + 1`
- Worker ranks use GPUs; the last rank is the host

Example with 2 GPUs:

For throughput benchmarking, we recommend explicitly setting
`OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8` before launching either FROST
or the TGL baselines so CPU-side sampling and data preparation use a
consistent thread budget.

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
| [`scripts/`](scripts/) | Thin CLI wrappers plus legacy experiment entry points being migrated |
| [`tests/`](tests/) | Test suite |
| `artifacts/` | Ignored benchmark outputs; override with `FROST_ARTIFACT_ROOT` |
| `DATA/` | Locally ignored dataset bundle or link to shared storage |
| [`eval_edgebank.py`](eval_edgebank.py) | Standalone EdgeBank baseline evaluator |

## Notes

- `macro_batch_size` is the total batch size across all GPUs. The launcher auto-computes `per_gpu_batch_size = macro_batch_size // n_gpu`.
- `ring_buf_size` is pipeline depth per worker, not dataset size or total batch count.
- Per-rank logs are written under Hydra outputs such as `outputs/<date>/<time>/rank_*.log`.
- GPU topology is inferred from `CUDA_VISIBLE_DEVICES` and `nvidia-smi topo -m` when available.

## Troubleshooting

- If `CUDA_VISIBLE_DEVICES=0,1`, then `--nproc_per_node` should be `3`, not `2`.
- If distributed training fails very early, check whether external NCCL environment variables are conflicting with the intended runtime setup.
- If throughput drops unexpectedly, confirm that `CUDA_LAUNCH_BLOCKING` or verbose debug flags are not still enabled.
- If CAWN is much slower than TGN, TGAT, or GraphMixer, reduce `macro_batch_size` first, then `encoder.fanout`.
