# AGENTS.md

Repository guidance for coding agents.

## Workflow

- Prefer the uv-managed environment: `uv run ...` and `uv sync --locked`.
- Preserve unrelated user changes and keep commits scoped.
- Commit coherent changes after their relevant checks pass.
- Preserve all git submodules unless explicitly asked to change them.
- Never clone over the git protocol; use HTTPS.

```bash
uv sync --locked
uv pip install -e .       # after changing the C++ sequence sampler
uv run ruff check .
uv run mypy
uv run pytest tests/ -q
```

Two GPU workers require one additional host rank:

```bash
uv run torchrun --nproc_per_node=3 run.py model=tgn dataset=LASTFM
```

## GPU experiment preflight

- Before choosing or launching on a GPU, follow the decision sequence in
  `/home/sqp17/Projects/compute-resource-ops`: read its `AGENTS.md`,
  `RESOURCE_POOL.md`, and the matching runbook; run `uv run resource-status
  --json` from that repository; then recheck the exact device immediately
  before launch.
- Treat GPU availability as ownership state, not a momentary utilization
  sample. Inspect compute processes and active relay/job records as well as
  memory and utilization; a GPU can read 0% and nearly 0 MiB between stages of
  a live host-heavy benchmark. Never schedule from an old relay result.
- Before a remote run, verify the intended commit and recursive submodule
  commits, then require `uv sync --locked --group tgb --check` to make no
  changes in the canonical checkout. Frost uses the repository `.venv`;
  baselines such as TGL, PipeTGL, MSPipe, and DyGLib use their own environments
  and must not be launched through the Frost interpreter.
- Detached or staged code snapshots do not contain ignored virtual
  environments or native build products. Rebuild them or attach explicitly
  verified, ABI-compatible artifacts, then smoke-import
  `frost.seq_sampler._core`; for TGL also smoke-import `dgl` and
  `sampler_core`, before starting a sweep. Use a user-owned uv cache. Set
  `UV_NO_SYNC=1` only after the selected environment and imports have passed
  these checks.
- Start a new remote matrix with one representative smoke case. On a
  distributed failure, preserve and inspect the earliest rank traceback;
  later Gloo/NCCL peer-closed and elastic child-failure messages are commonly
  secondary symptoms.
- Follow host-specific restrictions from the compute runbooks, including
  `NCCL_P2P_DISABLE=1` for multi-GPU runs on Sun. Long jobs must record the code
  commit, data roots, visible GPU mapping, full command, output/log directory,
  metrics, and exit status.

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
`runtime.experimental`. Do not add hidden debug branches to the baseline path.

## Data and benchmarks

Dataset roots belong in ignored `.env.local` as `FROST_DATA_ROOT` and
`TGB_DATA_ROOT`; never restore a generic `DATA` symlink. Use `uv run frost-data`
for dataset operations.

Benchmark defaults and matrices live in `benchmarks/specs/*.yaml`. Use
`frost-train`, `frost-suite`, and `frost-sweep`; keep execution and parsing in
the common `frost_bench` runners. Each job consumes all visible GPU workers plus
one host rank.

When adding a model, add its encoder, recipe/contract, `configs/model` entry, and
contract tests. Do not add model-specific analyzers, predictor switches, or
pipeline branches.
