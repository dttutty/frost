# FROST

The runtime does not depend on DGL, PyG, `pytorch_scatter`, or
`pytorch_sparse`.

## Quick start

```bash
uv python install 3.12
uv sync --locked --python 3.12
cp .env.example .env.local
```

Configure machine-local dataset roots in `.env.local`:

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

Download the prepared datasets from
[`dttutty/frost_dataset`](https://huggingface.co/datasets/dttutty/frost_dataset).
Set the download directory to the same `FROST_DATA_ROOT` configured in `.env.local`:

```bash
export FROST_DATA_ROOT=/absolute/path/to/frost_dataset
uvx --from huggingface_hub hf download dttutty/frost_dataset \
  --repo-type dataset --local-dir "$FROST_DATA_ROOT"
```

To download only LASTFM, add `--include "LASTFM/*"`. The bundle contains the
prepared edge tables and graph caches; no raw-data conversion is needed.
Dataset contents and any dataset-specific extra downloads are documented in
the [dataset repository](https://huggingface.co/datasets/dttutty/frost_dataset).

The default `neg_sampler=rnd_dst_then_random` samples from all nodes at runtime
for training, validation, and testing. It does not use the bundle's fixed
negative-sample files. See [negative-sampling protocols](docs/negative_sampling.md)
for explicit alternatives.

Official TGB benchmarks use separate downloads under `TGB_DATA_ROOT/raw/`;
see [dataset setup](docs/artifact_evaluation.md#dataset-setup).

## Documentation

## Repository layout

| Path | Purpose |
| --- | --- |
| `run.py` | Hydra distributed-training entry point |
| `configs/` | Model and runtime configuration |
| `src/frost/` | Runtime, models, samplers, and data tooling |
| `tests/` | Test suite |
| `docs/artifact_evaluation.md` | Installation, dataset setup, and evaluation workflows |
