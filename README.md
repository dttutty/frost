# FROST

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.9.0](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](uv.lock)
[![Linux x86-64](https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black)](#install)

A Python library for temporal graph neural networks: models, graph data,
neighborhood sampling and distributed training. Experiment orchestration and
paper reproduction belong in [frost_bench](https://github.com/dttutty/frost_bench),
not in the library package or its dependencies.

## Install

GPU training requires an NVIDIA GPU with a compatible driver.
CPU tests do not require a GPU.

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
[GPU preflight](#gpu-experiment-preflight), run two GPU workers
plus one host rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=3 run.py \
  model=tgn dataset=LASTFM
```

## Configuration

Hydra composes [shared defaults](configs/config.yaml), [model constructor
settings](configs/model/) and the capabilities declared by
[`ModelRecipe`](src/frost/models/recipes.py). Override values with `key=value`;
inspect the resolved configuration without training:

```bash
uv run python run.py --cfg job model=tgn
```

Library configs are native training examples. Paper presets and baseline parity
settings belong in [frost_bench](https://github.com/dttutty/frost_bench).

| Setting | Default | Meaning |
| --- | --- | --- |
| `model` / `dataset` | `tgn` / `LASTFM` | Model recipe and dataset; prepared datasets use `FROST_DATA_ROOT`, registered TGB datasets use `TGB_DATA_ROOT/raw`. |
| `epoch` / `patience` | `100` / `100` | Maximum epochs and early-stopping patience. |
| `early_stop_tolerance` | `0.0` | Minimum relative validation improvement; use `1e-6` for the official TGB protocol. |
| `macro_batch_size` | `4000` | Total training batch across GPUs; per-worker batch is `macro_batch_size // n_gpu`. |
| `eval_macro_batch_size` | `null` | Defaults to the training batch; reduce for one-vs-many evaluation. |
| `optimizer` / `lr` | `adam` / `0.0001` | Optimizer (`adam` or `sgd`) and learning rate. |
| `dropout` / `seed` / `d_time` | `0.2` / `42` / `128` | Shared dropout, random seed and time-encoding width. |
| `n_neg` / `eval_metric` | `1` / `ap` | Training negatives per positive and metric (`ap` or `mrr`); the metric also selects checkpoints and drives early stopping. |
| `ring_buf_size` | `64` | Shared-memory ring-buffer slots per worker. |
| `ignore_nfeat_on_disk` | `false` | Load `node_features.npy` when present. |
| `numa_policy` / `cpu_threads` | `auto` / `auto` | CPU and shared-memory placement and OMP/MKL thread budgets. |
| `if_carry_over_route` | `false` | Permit planner fallback to stale leftovers. |
| `compile` / `compute.tf32` / `compute.amp` | `false` | Enable `torch.compile`, TF32 or CUDA mixed precision with gradient scaling. |

For one GPU worker, use two processes. The optional resident backend keeps
static features and dynamic state on that worker's GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc_per_node=2 run.py \
  model=tgn dataset=WIKIPEDIA runtime.single_gpu_resident=on
```

### Parameter flow and coupling

- `model=<name>` selects an encoder, optional state updater, link head and
  semantic contract. `model.embedding_dim` controls embedding, memory and head
  widths. Stateless recipes use `model.state=null`; memory recipes declare the
  updater, mailbox size, aggregation (`last` or `mean`) and routing there.
- `model.neighborhood.fanout` owns fanouts for every model, including APAN:
  `[]` means 0-hop, one entry means 1-hop, and more entries mean multi-hop.
  Memory models support at most one hop; stateless models may use multiple hops.
- `ngh_sampler` selects `recent`, `uniform` or `time_interval_aware`; the latter
  uses `ngh_sampler.time_scaling_factor`. Negative sampling, `n_neg`, fanouts,
  layout, batch size and ring-buffer slots jointly determine capacity.
- TGN's `model.encoder.attention_layout=auto` switches from dense to packed
  attention when dense fanout slots times projected neighbor width reach
  `auto_pack_min_projected_elements`. This shape-based decision avoids per-batch
  CUDA mask synchronization; explicit `dense` and `packed` remain available.
- Keep model-specific fields aligned: GraphMixer's `num_tokens` must fit its
  sampled history; TCL's `num_depths` follows `fanout[0] + 1`; CAWN cost grows
  with the product of fanouts; DyGFormer's sequence must fit its maximum length
  and patch size. Current values live in the model configs.
- GPU topology comes from `CUDA_VISIBLE_DEVICES` and `nvidia-smi topo -m`,
  distinguishing NVLink, same-NUMA and cross-NUMA peers. `numa_policy=off`
  disables placement; a node ID binds all local ranks to that NUMA node.
  `cpu_threads=auto` divides GPU-local physical cores among workers, caps each
  at 32 threads and resolves the host separately. Use `inherit` or a positive
  integer to override; explicit `OMP_NUM_THREADS` / `MKL_NUM_THREADS` win.
- Edge width `edim` comes from features or the registered raw dataset;
  mailbox width is `d_mail = 2 * d_mem + edim`.

## Negative sampling

FROST scores destination corruption: each positive `(src, dst, time)` produces
`[src | positive dst | negative dst × n_neg]`, with the positive source reused
for every score. Independent negative sources require changes to sampling,
layouts, encoders, predictors and analyzers.

| `neg_sampler` | Training destinations | Evaluation destinations |
| --- | --- | --- |
| `rnd_dst_then_random` (default) | Uniform over all node IDs `[0, num_nodes)` | Same full range, sampled at runtime with replacement and no positive-collision filtering. |
| `rnd_dst_then_fixed` | Observed destination partition | Fixed `eid,neg_dst` artifact; requires `n_neg=1`. |
| `rnd_dst_then_fixed_all` | Observed destination partition | Fixed artifact covering the full contiguous node range, excluding the positive destination. |
| `rnd_dst_then_hist` | Observed destination partition | Prefer the current source's destinations strictly before the batch; exclude its current positives; fall back to the observed destination partition. |
| `rnd_dst_then_post_train` | Observed destination partition | Prefer the current source's destinations after the training cutoff and strictly before the batch; exclude current positives; same fallback. |
| `rnd_dst_then_tgb` / `rnd_dst_then_tgb_wiki` | Random destinations | Official TGB validation/test candidates and required width. |

Historical and post-training sampling use replacement, so negatives may repeat.
Fixed evaluation reads `$FROST_DATA_ROOT/<dataset>/neg_pre_sampling_1_result.csv`;
`rnd_dst_then_fixed_all` uses `neg_pre_sampling_1_all_nodes_seed0.csv` instead.
AP requires one evaluation negative per positive; runtime-random evaluation
advances its RNG. For one-vs-many MRR, set `neg_sampler.eval.n_neg`.
Dataset manifests describe fixed artifacts; FROST owns runtime protocol selection.

For comparisons, match and report the temporal split (including tied timestamps),
exact protocol, candidate set, positive exclusions, replacement rule, negative
count, seed or artifact checksum, and metric reduction/tie policy. All models
must consume the same splits and negative queries.

Upstream DyGLib snapshot `3aacc36b` used full negative `(src, dst)` pairs for
historical/inductive evaluation: remove current-batch edges from global history,
and additionally remove training-history edges for inductive sampling; fall
back to collision-checked random pairs. These metrics are not directly comparable
to FROST's per-source destination protocols. DyGLib's random path instead reuses
the positive source, matching FROST's task when its candidate partition matches,
though RNG implementations can produce different samples. The maintained fork
uses fixed evaluation negatives and global AP; see its
[protocol](https://github.com/dttutty/DyGLib/blob/74cbefd0babd9de9a61a7f1c10eedc32bfc82ff8/PROTOCOL.md).
FROST implementations: [historical](src/frost/neg_sampler/historical.py),
[post-training](src/frost/neg_sampler/inductive.py),
[random](src/frost/neg_sampler/random.py).

## TGB link prediction

Fetch the official tgbl-wiki-v2 files into `$TGB_DATA_ROOT/raw/tgbl_wiki`, then
run two GPU workers and one host after [GPU preflight](#gpu-experiment-preflight):

```bash
uv run frost-data fetch-tgb tgbl-wiki
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=3 run.py model=tgn \
  dataset=TGBL_WIKI \
  neg_sampler=rnd_dst_then_tgb_wiki eval_metric=mrr \
  macro_batch_size=4096 eval_macro_batch_size=128
```

Training uses one random negative per positive. Wiki evaluation reserves 999
negatives and masks the four wiki-v2 queries with only 998; it uses neither
ragged tensors nor random replacements. For generic TGB, set `eval_metric=mrr`
and `neg_sampler.eval.n_neg=100` for `tgbl-review`, or 20 for `tgbl-coin`,
`tgbl-comment` and `tgbl-flight`; every query must match that fixed capacity.
MRR averages optimistic/pessimistic tie ranks. The separate, smaller evaluation
batch bounds ring-buffer and GPU memory use.

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

Storage and transport use `int64` IDs/timestamps and `float32` features/state;
these are runtime invariants. Compute relative times before casting to model
floating point. Link prediction uses binary cross-entropy with logits; changing
the objective requires a task-level contract. Neighborhood samplers share one
protocol; negative samplers implement `NegSamplerProto`. Reuse TGB samplers
and benchmark matrices. The launcher resolves `dynamic_state_ablation` into
`runtime.dynamic_state_policy` enums; keep hidden debug branches out of baseline
execution. See [parameter constraints](#parameter-flow-and-coupling).

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

## GPU experiment preflight

1. Follow the [compute-resource-ops procedures](/home/sqp17/Projects/compute-resource-ops/README.md)
   and its agent rules, resource pool and target runbook. Run `uv run resource-status --json`
   there, then recheck the selected GPU's processes, ownership, memory, utilization
   and health immediately before launch. A momentarily idle GPU is not unowned.
   Follow host restrictions, including `NCCL_P2P_DISABLE=1` on Sun.
2. Verify code and recursive submodule commits and require
   `uv sync --locked --check` to report no changes. Use the FROST environment
   for FROST; baselines keep separate environments. For staged snapshots, rebuild
   or attach verified ABI-compatible native products and smoke-import
   `frost.seq_sampler._core` (TGL also needs `dgl` and `sampler_core`). Use a
   user-owned uv cache; set `UV_NO_SYNC=1` only after these checks pass.
3. Start a new remote matrix with one smoke case. On distributed failures, inspect
   the earliest rank traceback before secondary Gloo/NCCL or elastic errors.
   Long jobs must record commits, data roots, GPU mapping, full command, output/log
   directory, metrics and exit status.
