# FROST

FROST is a Python library for temporal graph neural networks, with distributed
training, sequence-based neighborhood sampling, and explicit model/runtime
contracts. The package provides models, samplers, graph data structures, data
preparation, and host/worker runtime components under `frost`.

## Install

FROST targets Linux x86-64 and Python 3.12. GPU training requires an NVIDIA
GPU and a compatible driver. The locked environment uses CUDA 12.8 PyTorch;
CPU tests do not require a GPU.

```bash
uv python install 3.12
uv sync --locked --python 3.12
cp .env.example .env.local
```

The build compiles the native sequence sampler. After changing its C++ source,
rebuild the editable package with `uv pip install -e .`.

## Use the library

```python
from frost.models.recipes import RECIPE_TYPES
from frost.sampling.ngh_sampler import NeighborhoodSampler
from frost.datatypes.sampled_layout import SampledLayoutView
```

Models declare a `ModelRecipe` and `ModelContract`; the launcher compiles them
into an `ExecutionPlan`. See the [model/runtime boundary](docs/model_runtime_boundary.md)
for component ownership and how to extend the library.

`run.py` and `configs/` provide a native training example. Set the prepared
dataset roots in the ignored `.env.local`:

```dotenv
FROST_DATA_ROOT=/absolute/path/to/frost_dataset
TGB_DATA_ROOT=/absolute/path/to/tgb
```

Prepared datasets are available in the
[FROST dataset bundle](https://huggingface.co/datasets/dttutty/frost_dataset).
Shell exports take precedence over `.env.local`. Do not restore a generic
`DATA` symlink; use the configured dataset roots.

After [GPU resource checks](#gpu-experiment-preflight), run two GPU workers
plus one host rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=3 run.py \
  model=tgn dataset=LASTFM
```

For one GPU worker, use two processes. The optional resident backend keeps
static features and dynamic state on that worker's GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc_per_node=2 run.py \
  model=tgn dataset=WIKIPEDIA runtime.single_gpu_resident=on
```

Inspect configuration without starting training:

```bash
uv run python run.py --cfg job model=tgn
uv run frost-data --help
```

The [configuration guide](configs/README.md) describes model parameters,
negative-sampling protocols, runtime settings, and official TGB data/MRR support.

## Development

Prefer the repository's uv-managed environment: `uv run ...` for commands and
`uv sync --locked` for dependency synchronization. Preserve unrelated user
changes and existing Git submodules unless explicitly asked to change them.
Commit coherent changes after their relevant checks pass, then push to the
current branch's upstream. Use HTTPS for Git remotes and cloning.

```bash
uv run ruff check .
uv run mypy
uv run pytest tests/ -q
```

| Path | Purpose |
| --- | --- |
| `src/frost/` | Library implementation and native sequence sampler |
| [configs/](configs/README.md) | Model and runtime configuration examples |
| `run.py` | Native distributed-training example |
| `tests/` | Library and runtime regression tests |
| [docs/model_runtime_boundary.md](docs/model_runtime_boundary.md) | Architecture and extension rules |

## Design boundary

Read [docs/model_runtime_boundary.md](docs/model_runtime_boundary.md) before
changing model/runtime interfaces.

- `models/` owns differentiable computation.
- A `ModelRecipe` declares components and a semantic `ModelContract`.
- Launch compiles the contract into an `ExecutionPlan`.
- Host runtime owns sampling, access planning, batching, and scatter.
- Worker runtime owns transfer, dynamic-state publication, optimization,
  metrics, checkpoints, and telemetry.
- Runtime code must not branch on concrete model names.

Encoders return `EncoderResult`; representation adapters produce explicit
`EdgeRepresentations`; heads score those pairs. Dynamic-state publication stays
in `DynamicStateCoordinator`, and updaters cache state on `flat_graph.detached_*`.

## Important invariants

- `model.embedding_dim` is the shared embedding/memory/head width.
- `model.neighborhood.fanout` is the only fanout source.
- `d_mail = 2 * d_mem + edim`.
- Memory models support at most one hop; stateless models may use multiple hops.
- Use `SampledLayoutView` and contract node sets instead of re-deriving flat
  graph offsets.
- Neighborhood sampling uses `NoSampler` or `NeighborhoodSampler` through one
  protocol; negative samplers implement `NegSamplerProto`.
- Do not duplicate TGB negative sampling or benchmark matrices.

The public `dynamic_state_ablation` key is compatibility surface. Resolve it in
the launcher; runtime components consume only the policy enums under
`runtime.dynamic_state_policy`. Do not add hidden debug branches to the
baseline path.

When adding a model, add its encoder, recipe/contract, `configs/model` entry,
and contract tests. Do not add model-specific analyzers, predictor switches,
or pipeline branches.

## GPU experiment preflight

