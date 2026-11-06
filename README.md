# FROST

A Python library for temporal graph neural networks: models, graph data,
neighborhood sampling and distributed training. Experiment orchestration and
paper reproduction belong in [frost_bench](https://github.com/dttutty/frost_bench),
not in the library package or its dependencies.

## Install

Linux x86-64, Python 3.12; the lockfile uses CUDA 12.8 PyTorch. GPU training needs
an NVIDIA GPU and compatible driver; CPU tests do not require a GPU.

```bash
uv sync --locked --python 3.12
cp .env.example .env.local
```

Set `FROST_DATA_ROOT` and `TGB_DATA_ROOT` in ignored `.env.local`; shell exports
win. Use these paths, not a generic `DATA` symlink. Prepared data is available
in the [dataset bundle](https://huggingface.co/datasets/dttutty/frost_dataset);
`uv run frost-data --help` lists data operations.

## Use the library

```python
from frost.models.recipes import RECIPE_TYPES
from frost.sampling.ngh_sampler import NeighborhoodSampler
from frost.datatypes.sampled_layout import SampledLayoutView
```

`run.py` and `configs/` provide native training examples. After
[GPU preflight](configs/README.md#gpu-experiment-preflight), run two GPU workers
plus one host rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=3 run.py \
  model=tgn dataset=LASTFM
```

Use `uv run python run.py --cfg job model=tgn` to inspect configuration without
training. The [configuration guide](configs/README.md) covers parameters,
single-GPU execution, negative sampling and TGB evaluation.

## Architecture and extension

`ModelRecipe` declares components and a semantic `ModelContract`; the launcher
compiles it into an immutable `ExecutionPlan` before runtime construction.

- Models own differentiable computation: encoders return `EncoderResult`,
  representation adapters produce `EdgeRepresentations`, and heads score pairs.
  Queues, CUDA events, P2P and shared-memory operations belong to runtime.
- Host runtime owns sampling, access planning, batching and scatter; workers
  own transfer, optimization, metrics, checkpoints and telemetry.
- `DynamicStateCoordinator` publishes updater state through a forward hook,
  allowing overlap with encoding. Updaters cache state on `flat_graph.detached_*`.
- Use contract node sets and `SampledLayoutView`, not handwritten graph offsets.
  Sampling uses `RootExpansion`; feature access and mailbox delivery use
  `AccessPlanAnalyzer` and `NeighborhoodMailboxRouter`.
- Add an encoder, recipe/contract, model config and contract tests for a new model.
  Extend contract vocabulary only when needed; do not add model-name branches,
  model-specific analyzers or predictor switches to runtime.

See [parameter and sampling constraints](configs/README.md#parameter-flow-and-coupling).

## Development

```bash
uv run ruff check .
uv run mypy
uv run pytest tests/ -q
```

Use `uv run` and `uv sync --locked`; rebuild C++ sampler changes with
`uv pip install -e .`. Preserve unrelated changes, submodules and local experiment
material. Commit checked, scoped changes and push to the current upstream;
use HTTPS remotes and do not rewrite shared history. Keep credentials and data
out of Git. Library code is in `src/frost/`, regression tests in `tests/`.
