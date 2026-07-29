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

Copy [`.env.example`](.env.example) to the locally ignored `.env.local` and
configure the two machine-local dataset roots:

```bash
cp .env.example .env.local
# FROST_DATA_ROOT=/absolute/path/to/frost_dataset
# TGB_DATA_ROOT=/absolute/path/to/tgb
```

FROST loads `.env.local` automatically. `FROST_DATA_ROOT` contains materialized
artifact datasets such as `LASTFM/` and `WIKIPEDIA/`. `TGB_DATA_ROOT` contains
official TGB downloads beneath `raw/`; these remain separate from the published
artifact bundle. A repository-local `DATA` directory or symlink is not required.
Source `.env.local` explicitly before using its variables in shell-only commands:

```bash
set -a
source .env.local
set +a
```

If you host the materialized datasets on Hugging Face, download them directly
to the configured artifact root:

```bash
export HF_DATASET_REPO=<your-dataset-repo>
uvx --from huggingface_hub hf download \
  "$HF_DATASET_REPO" \
  --repo-type dataset \
  --local-dir /absolute/path/to/frost_dataset
```

To fetch a single dataset:

```bash
export HF_DATASET_REPO=<your-dataset-repo>
uvx --from huggingface_hub hf download \
  "$HF_DATASET_REPO" \
  --repo-type dataset \
  --include "LASTFM/*" \
  --local-dir /absolute/path/to/frost_dataset
```

Download the complete
[official dynamic link property prediction catalog](https://tgb.complexdatalab.com/docs/linkprop/)
into `TGB_DATA_ROOT` with:

```bash
# Debian/Ubuntu: sudo apt install aria2
uv run frost-data fetch-tgb --list
uv run frost-data fetch-tgb --all
```

The pinned catalog contains `tgbl-wiki-v2`, `tgbl-review-v2`, `tgbl-coin`,
`tgbl-comment`, and `tgbl-flight`. The five official archives total about
4.52 GiB compressed and require substantially more space after extraction.
FROST delegates retries, segmented transfers, resume support, progress, and MD5
verification to `aria2c`, then extracts beneath `$TGB_DATA_ROOT/raw/`.
Re-running the command skips complete datasets. Use `--download-only` to retain
verified archives without extracting.

The current FROST runtime adapter supports `TGBL_WIKI`; it can consume a freshly
extracted official v2 CSV directly. The host rank constructs FROST's edge frame,
temporal CSR, padded edge features, and node-role features in memory at launch.
No materialized FROST `TGBL_WIKI/` directory or export step is needed. The other
four datasets are downloaded for reproducible local availability but still
require runtime adapters before training in FROST.

Dataset transformation code is maintained in FROST rather than in the published
artifact bundle. Use `frost-data` to build or validate the common on-disk format:

```bash
# Build the temporal CSR consumed by DatasetManager into a scratch artifact path.
uv run frost-data build-graph "$FROST_DATA_ROOT/LASTFM/edges.csv" \
  --bidirectional \
  --output artifacts/data/LASTFM/full_graph_with_reverse_edges.npz

# TGL's GDELT and MAG protocols keep only the published edge direction.
uv run frost-data build-graph "$FROST_DATA_ROOT/GDELT/edges.csv" \
  --directed \
  --output artifacts/data/GDELT/full_graph_no_reverse_edges.npz

# Generate node_role.npy after verifying that src/dst are disjoint partitions.
uv run frost-data node-roles "$FROST_DATA_ROOT/LASTFM/edges.csv" \
  --output artifacts/data/LASTFM/node_role.npy

# Keep tied timestamps in one train/validation/test split.
uv run frost-data split "$FROST_DATA_ROOT/LASTFM/edges.csv" \
  --strategy timestamp --output artifacts/data/LASTFM/edges.csv

# Build fixed evaluation negatives from the observed destination partition.
uv run frost-data negatives "$FROST_DATA_ROOT/LASTFM/edges.csv" \
  --output artifacts/data/LASTFM/neg_pre_sampling_1_result.csv

# Verify schema, temporal splits, graph, sidecars, and negative candidates.
uv run frost-data validate "$FROST_DATA_ROOT/LASTFM"
```

Generated outputs are not overwritten unless `--force` is supplied. Run
`uv run frost-data --help` for explicit output-path and direction options.
`frost-data import-dgb` performs the complete conversion from an extracted DGB
Zenodo `ml_*` directory and records every protocol choice in `manifest.json`.
`frost-data catalog "$FROST_DATA_ROOT"` derives release statistics and the exact
evaluation batch-size ceiling from those manifests, so those figures do not
need to be maintained in multiple READMEs.

Negative-sampling names are not interchangeable across frameworks. See
[Negative-sampling protocols](docs/negative_sampling.md) for FROST's exact
random, fixed, historical, inductive, and TGB semantics and their differences
from DyGLib.

`graph_direction=auto` is the runtime default. It enforces TGL's official graph
semantics for maintained datasets: GDELT and MAG are directed, while WIKI,
WIKIPEDIA, REDDIT, MOOC, and LASTFM include reverse adjacencies. For a custom
dataset, `auto` accepts either representation after inferring it from the graph;
set `graph_direction=directed` or `bidirectional` to enforce one explicitly.

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

Run a declared seed or model/dataset sweep sequentially:

The comparison writes raw logs, `throughput.tsv`, `convergence.tsv`,
`convergence.png`, and `summary.json` under `artifacts/comparisons/`.

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
| [`scripts/`](scripts/) | Specialized experiment and measurement frontends |
| [`tests/`](tests/) | Test suite |
| `artifacts/` | Ignored benchmark outputs; override with `FROST_ARTIFACT_ROOT` |
| `.env.local` | Ignored machine-local dataset root configuration |
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
- If CAWN is much slower than TGN, TGAT, or GraphMixer, reduce `macro_batch_size` first, then `model.neighborhood.fanout`.
