# AGENTS.md

Guidance for coding agents working in FROST.

## Working conventions

- Use the repository's uv environment: prefer `uv run ...` and `uv sync --locked`.
- Preserve user changes in a dirty worktree and keep commits scoped.
- Commit a coherent change after its tests pass when a commit is useful.
- Preserve all git submodules unless the user explicitly requests otherwise.
- Never clone over the git protocol; use HTTPS URLs.

## Build, test, and run

```bash
uv sync --locked

# Rebuild the editable package after changing the C++ sequence sampler.
uv pip install -e .

uv run ruff check .
uv run mypy
uv run pytest tests/ -q

# Two GPU workers plus one host rank.
uv run torchrun --nproc_per_node=3 run.py model=tgn dataset=LASTFM
```

Supported model recipes are `tgn`, `jodie`, `apan`, `tgat`, `dygformer`,
`graph_mixer`, `tcl`, and `cawn`. EdgeBank uses
`uv run python eval_edgebank.py --dataset LASTFM`.

## Architectural boundary

FROST is a distributed temporal link-prediction runtime. GPU workers occupy
ranks `0..n_gpus-1`; the host is the last rank.

`run.py` calls `frost.launcher.launch()`. The launcher resolves a
`ModelRecipe`, compiles its semantic `ModelContract` into an `ExecutionPlan`,
constructs shared storage, and dispatches either `Host` or `Worker`.

Keep these ownership boundaries:

- `src/frost/models/`: differentiable computation only. Encoders return
  `EncoderResult`; representation adapters produce `EdgeRepresentations`; heads
  score those structured pairs.
- `src/frost/models/recipes.py`: a model's components and declared capabilities.
- `src/frost/models/contracts.py` and `src/frost/launch/plan.py`: semantic model
  requirements and their runtime-ready plan.
- `src/frost/runtime/host/`: batching, access planning, neighborhood sampling,
  and scatter.
- `src/frost/runtime/worker/`: transfer, dynamic-state assembly/publication,
  optimization, metrics, checkpoints, and telemetry.
- `src/frost/launch/storage.py`: dataset attachment, ring-buffer sizing, and
  shared-memory allocation.

Do not branch on model names in the runtime. Add a contract capability only
when it describes a real runtime requirement shared independently of a concrete
model.

Host and worker constructors take grouped `Resources` and `Options` dataclasses
from their `_config.py` modules. Keep benchmark observation in
`WorkerBenchmarkObserver`, not in the training loop.

## Core data flow

```text
Host: sample -> resolve contract-declared access -> plan -> scatter
Worker: receive -> H2D -> assemble dynamic state -> model -> publish state
```

`FlatGraph` stores roots as `[src | positive dst | negative dst...]`, followed
by neighbor tiers. `RootExpansion.ALL_ROOTS` samples every root;
`RootExpansion.POSITIVE_ROOTS` samples only positive source/destination roots.
Use `SampledLayoutView` and contract node sets instead of manually re-deriving
flat offsets.

## Model and runtime invariants

- `model.embedding_dim` is the single embedding/memory/head width. JODIE and
  APAN require their updater memory width to match it.
- `model.neighborhood.fanout` is the single fanout source. APAN declares
  positive-root, one-hop sampling in its contract.
- `d_mail = 2 * d_mem + edim`.
- Encoders never encode predictor layout assumptions. The representation
  adapter handles node-aligned versus pair-aligned outputs, and heads consume
  `EdgeRepresentations`.
- Dynamic-state publication belongs to `DynamicStateCoordinator`, not model
  modules. Updaters write cached values to `flat_graph.detached_nid/ts/mem`.
- Neighborhood-mail planning must match router writes exactly: padding `-1` is
  filtered and only positive neighbors are mailbox updates.
- Memory models support at most one hop. Stateless models may use multiple
  hops.

## Sampling

Neighborhood sampling has one protocol and two implementations:
`NoSampler` for zero hop and `NeighborhoodSampler` for one or more hops. The C++
sampler supports `uniform`, `recent`, and `time_interval_aware`; the time scaling
factor belongs only to `ngh_sampler` config.

Negative samplers implement `NegSamplerProto`. Training/evaluation selection is
configuration, while the runtime consumes only that protocol. Do not duplicate
TGB negative-sampling logic; use the official evaluation artifacts through the
registered TGB sampler.

## Experimental dynamic-state policies

The public Hydra key remains `dynamic_state_ablation` for benchmark
compatibility. Resolve it once in the launcher with
`runtime.experimental.dynamic_state.resolve_dynamic_state_policy()`. Runtime
components consume only the resolved planner, transport, and writeback enums;
they must not interpret preset names or add hidden debug branches.

Direct-CPU policies are experimental and currently limited to supported
one-hop interaction-mail memory models with mailbox size one. Baseline behavior
uses the pipelined planner and asynchronous P2P transport.

## Data

Machine-local roots belong in the gitignored `.env.local`:

```dotenv
FROST_DATA_ROOT=/path/to/materialized/frost/datasets
TGB_DATA_ROOT=/path/to/official/tgb
```

Shell exports override `.env.local`. Never restore a generic `DATA` pointer.
Materialized FROST datasets resolve under `FROST_DATA_ROOT`; official TGB files
resolve under `TGB_DATA_ROOT/raw`. Use `uv run frost-data` for downloading,
building, and validating data.

A materialized dataset contains `edges.csv`, exactly one supported temporal CSR,
and optional `node_features.pt` / `edge_features.pt`. `graph_direction=auto`
treats GDELT and MAG as directed and the remaining maintained datasets as
bidirectional. Graph building requires an explicit direction.

## Adding a model

1. Add an encoder returning `EncoderResult` under `src/frost/models/encoder/`.
2. Add a complete recipe and `ModelContract` in `models/recipes.py`.
3. Add one composition root under `configs/model/`.
4. Register the recipe type and add contract/config tests.
5. Run with `run.py model=<name>`.

Do not add model-specific analyzers, predictor-layout switches, or pipeline
branches. Extend semantic types only when the new capability cannot be expressed
by the existing contract.
