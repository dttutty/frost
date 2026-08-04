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

