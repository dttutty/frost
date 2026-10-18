# FROST

FROST is a distributed temporal graph neural network training system with exact
memory semantics. A dedicated host rank performs sampling and access planning;
GPU worker ranks train the model with PyTorch Distributed.

The runtime does not depend on DGL, PyG, `pytorch_scatter`, or
`pytorch_sparse`.

## Quick start

```bash
uv python install 3.12
uv sync --python 3.12
cp .env.example .env.local
```

Configure machine-local dataset roots in the ignored `.env.local`:

```dotenv
FROST_DATA_ROOT=/absolute/path/to/frost_dataset
TGB_DATA_ROOT=/absolute/path/to/tgb
```

`FROST_DATA_ROOT` contains materialized datasets such as `LASTFM/` and
`WIKIPEDIA/`. Official TGB downloads stay separately under
`TGB_DATA_ROOT/raw/`. A repository-local `DATA` path is neither required nor
supported. Shell exports override `.env.local`.

Run TGN with two GPU workers and one host rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run frost-train tgn --nproc-per-node=3 \
  dataset=LASTFM
```

In general, `nproc_per_node = number of GPU workers + 1`.

For one GPU, the same entry can select the dedicated resident backend. It keeps
the full static feature tables and dynamic memory/mailbox tensors on the GPU,
skips DDP, feature staging, dynamic access planning, and P2P state transport,
while retaining the host sampler rank:

```bash
CUDA_VISIBLE_DEVICES=0 uv run frost-train tgn --nproc-per-node=2 \
  dataset=WIKIPEDIA runtime.single_gpu_resident=on
```

The default is `off`. `auto` enables the backend only for one worker when its
persistent tensors fit `runtime.single_gpu_resident_memory_fraction` of device
memory; `on` reports an error instead of silently falling back.

## Datasets

Download the complete official TGB link-prediction catalog with `aria2c`:

```bash
uv run frost-data fetch-tgb --list
uv run frost-data fetch-tgb --all
```

The pinned catalog contains `tgbl-wiki-v2`, `tgbl-review-v2`, `tgbl-coin`,
`tgbl-comment`, and `tgbl-flight`. Downloads resume and verified, complete
datasets are skipped. `--download-only` retains archives without extracting.

FROST currently runs `TGBL_WIKI` directly from the official CSV files. Its edge
frame, temporal CSR, and padded edge features are created in memory; the
official dataset provides no node features. No exported `TGBL_WIKI/` artifact
directory is needed. The other four datasets are downloadable but still need
runtime adapters.

Materialized FROST datasets can be downloaded from a configured Hugging Face
dataset repository:

```bash
uvx --from huggingface_hub hf download "$HF_DATASET_REPO" \
  --repo-type dataset --local-dir "$FROST_DATA_ROOT"
```

`frost-data` also builds and validates the common artifact format:

```bash
uv run frost-data build-graph "$FROST_DATA_ROOT/LASTFM/edges.csv" \
  --bidirectional --output artifacts/data/LASTFM/full_graph_with_reverse_edges.npz
uv run frost-data validate "$FROST_DATA_ROOT/LASTFM"
uv run frost-data catalog "$FROST_DATA_ROOT"
```

Generated files are not overwritten without `--force`. Run
`uv run frost-data --help` for graph, split, negative-sampling, DGB import, and
catalog commands. Negative-sampling semantics are documented in
[docs/negative_sampling.md](docs/negative_sampling.md).

## Training and benchmarks

The maintained entry points share model recipes and YAML experiment specs:

```bash
uv run frost-train --list
uv run frost-train tcl
uv run frost-train tcl --dry-run

uv run frost-sweep --list
uv run frost-sweep memory_seeds --gpus 0,1

uv run frost-suite --list
uv run frost-suite tgn_memdim_2gpu --dry-run
```

TPNet stays behind the same command entry point. For an official TGB dataset,
`--report-mrr` evaluates the final epoch on rank 0 with Frost's own official
negative sampler and tie-aware MRR implementation:

```bash
uv run frost-data fetch-tgb tgbl-wiki
CUDA_VISIBLE_DEVICES=0,1 uv run --group tgb frost-train tpnet \
  --nproc-per-node=2 --dataset_name tgbl-wiki --batch_size 400 \
  --use_random_projection --report-mrr \
  --metrics-json results/tgbl-wiki-tpnet.json
```

TPNet's `--batch_size` is global, so 400 gives each GPU 200 events with two
GPU processes; use 800 with four processes to keep the same per-GPU batch.
Wiki and review use an evaluation batch of 20 by default; override it with
`--eval-batch-size` if needed. The JSON result contains `val_mrr` and
`test_mrr` and identifies the evaluated checkpoint as `final_epoch`.

Use Hydra directly for a manual experiment:

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
uv run torchrun --nproc_per_node=3 run.py \
  model=cawn dataset=WIKIPEDIA macro_batch_size=1200
```

The canonical `frost-train` entry point distinguishes automatic values from
explicit thread environment variables before torchrun starts. Direct torchrun
commands should therefore set OMP/MKL explicitly when thread count matters.

The FROST/PipeTGL comparison has a dedicated frontend:

```bash
third_party/PipeTGL/scripts/bootstrap_gnnflow.sh
uv run frost-compare-pipetgl --dataset WIKIPEDIA --epochs 20 \
  --measure-start-epoch 1
uv run frost-compare-pipetgl --dataset TGBL_WIKI --epochs 20
```

The comparison defaults to global batch sizes 600/1200/2000/4000 for
WIKIPEDIA/MOOC/REDDIT/LASTFM respectively. PipeTGL's parity adapter also
supports directed MAG/GDELT runs at global batch size 32000, including bounded
train-only throughput windows. Pass `--batch-size` to override the global value
explicitly.
TGBL-WIKI uses the official TGB split, edge features, one-to-many negatives,
and tie-aware MRR from `TGB_DATA_ROOT`.
Ordinary convergence runs process the complete train, validation, and test
splits in that order every epoch. Best-epoch selection uses validation AP/MRR
only; the reported test metric comes from that same epoch.

Benchmark outputs go under ignored `artifacts/` by default. Set
`FROST_ARTIFACT_ROOT` to use another location. FROST defaults to topology-aware
automatic OMP/MKL sizing; maintained comparison suites resolve the same
automatic worker budget for external runtimes. Explicit `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` values remain authoritative.

## Documentation

- [Configuration reference](configs/README.md)
- [Model/runtime boundary](docs/model_runtime_boundary.md)
- [Negative-sampling protocols](docs/negative_sampling.md)
- [Benchmark architecture](benchmarks/README.md)
- [Historical experiment index](experiments/README.md)

Model components and defaults live in `configs/model/`; model capabilities live
in `src/frost/models/recipes.py`; maintained launch protocols live in
`benchmarks/specs/`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `run.py` | Hydra distributed-training entry point |
| `configs/` | Model and runtime configuration |
| `src/frost/` | Runtime, models, samplers, and data tooling |
| `src/frost_bench/` | Shared benchmark runners and backend adapters |
| `benchmarks/specs/` | Presets, parity settings, sweeps, and throughput matrices |
| `tests/` | Test suite |
| `third_party/`, `third_party_papers/` | Preserved git submodules |

## Common pitfalls

- `macro_batch_size` is global across workers.
- `ring_buf_size` is pipeline depth, not dataset size.
- GDELT and MAG use directed temporal graphs; maintained interaction datasets
  use reverse adjacency. `frost-data build-graph` requires an explicit choice.
- If two GPUs are visible, `torchrun` needs three processes.
- Unexpectedly low throughput often comes from `CUDA_LAUNCH_BLOCKING`, verbose
  debug logging, or mismatched CPU thread settings.
