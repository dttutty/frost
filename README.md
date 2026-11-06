# FROST

## Quick start

FROST runs on Linux x86-64 with Python 3.12. Training requires an NVIDIA GPU
and a compatible driver; the locked environment uses the CUDA 12.8 PyTorch
wheel. CPU tests and command inspection do not require a GPU.

```bash
uv python install 3.12
uv sync --locked --python 3.12
cp .env.example .env.local
```

Set machine-local data roots in `.env.local` (shell exports take precedence):

```dotenv
FROST_DATA_ROOT=/absolute/path/to/frost_dataset
TGB_DATA_ROOT=/absolute/path/to/tgb
```

After downloading a dataset below, run TGN with two GPU workers and one host:

For native FROST, `nproc_per_node = GPU workers + 1`. With one worker, the
resident backend keeps static features and dynamic state on its GPU while
retaining the host sampler and skipping DDP, feature staging and P2P transport:

The resident backend defaults to `off`. `auto` enables it only for one worker
when persistent tensors fit `runtime.single_gpu_resident_memory_fraction`
(default 0.7) of device memory; `on` reports an error if it cannot be enabled.

## Datasets

Download the prepared [dataset bundle](https://huggingface.co/datasets/dttutty/frost_dataset)
to the same root configured in `.env.local`:

```bash
export FROST_DATA_ROOT=/absolute/path/to/frost_dataset
uvx --from huggingface_hub hf download dttutty/frost_dataset \
  --repo-type dataset --include "LASTFM/*" --local-dir "$FROST_DATA_ROOT"
```

Remove `--include "LASTFM/*"` for the full bundle. Prepared edge tables and
graph caches need no raw-data conversion. The dataset repository documents
its catalog and any extra downloads. A repository-local `DATA` path is not used.
The default sampler generates negatives at runtime; protocol choices and
comparison rules are in the [configuration guide](configs/README.md#negative-sampling).

Official TGB downloads require `aria2c` and use a separate raw-data root:

```bash
export TGB_DATA_ROOT=/absolute/path/to/tgb
uv run frost-data fetch-tgb tgbl-wiki
```

## Training and checks

## Documentation and repository layout

| Path | Purpose |
| --- | --- |
| [configs/](configs/README.md) | Model/runtime parameters, negative-sampling protocols and TGB configuration |
| [Model/runtime boundary](docs/model_runtime_boundary.md) | Model contracts and runtime responsibilities |
| `run.py` | Hydra distributed-training entry point |
| `src/frost/` | Runtime, models, samplers and data tooling |
| `tests/` | Test suite |

Model components live in `configs/model/`; capabilities are declared in
`src/frost/models/recipes.py`. See [AGENTS.md](AGENTS.md) for contributor workflow
and GPU resource rules.
