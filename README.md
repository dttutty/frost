# FROST: anonymous revision artifact

This branch is the double-anonymous evaluation artifact for the paper
submission. It contains the FROST implementation, tests, experiment
specifications, and public-data preparation tools. Author metadata, private
infrastructure notes, paper sources, and external baseline source trees are not
part of this branch.

Reviewers should start with the step-by-step
[artifact evaluation guide](docs/artifact_evaluation.md).

FROST is a distributed temporal graph neural network training system with exact
memory semantics. A dedicated host rank performs sampling and access planning;
GPU worker ranks train the model with PyTorch Distributed.

The runtime does not depend on DGL, PyG, `pytorch_scatter`, or
`pytorch_sparse`.

## Quick start

```bash
uv python install 3.12
uv sync --locked --python 3.12
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

In general, `nproc_per_node = number of GPU workers + 1`.

For one GPU, the same entry can select the dedicated resident backend. It keeps
the full static feature tables and dynamic memory/mailbox tensors on the GPU,
skips DDP, feature staging, dynamic access planning, and P2P state transport,
while retaining the host sampler rank:

The default is `off`. `auto` enables the backend only for one worker when its
persistent tensors fit `runtime.single_gpu_resident_memory_fraction` of device
memory; `on` reports an error instead of silently falling back.

## Datasets

This artifact does not use or link to an author-owned dataset mirror. The
datasets used by the maintained examples can be reconstructed from the public
DGB Zenodo release, while TGB datasets are fetched from the official TGB
object store.

For a verified LASTFM example, including archive checksum, extraction,
conversion, validation, and training commands, follow
[Dataset setup](docs/artifact_evaluation.md#dataset-setup). The same guide lists
the official archives used for WIKIPEDIA, MOOC, and REDDIT.

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

## Documentation

## Repository layout

| Path | Purpose |
| --- | --- |
| `run.py` | Hydra distributed-training entry point |
| `configs/` | Model and runtime configuration |
| `src/frost/` | Runtime, models, samplers, and data tooling |
| `tests/` | Test suite |
| `docs/artifact_evaluation.md` | Reviewer setup and evaluation procedure |

## Common pitfalls

- `macro_batch_size` is global across workers.
- `ring_buf_size` is pipeline depth, not dataset size.
- GDELT and MAG use directed temporal graphs; maintained interaction datasets
  use reverse adjacency. `frost-data build-graph` requires an explicit choice.
- If two GPUs are visible, `torchrun` needs three processes.
- Unexpectedly low throughput often comes from `CUDA_LAUNCH_BLOCKING`, verbose
  debug logging, or mismatched CPU thread settings.
