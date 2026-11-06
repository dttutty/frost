# Configuration Guide

FROST uses Hydra config groups to compose datasets, models, memory components, and runtime helpers. The main entry point is [`config.yaml`](config.yaml).

## Defaults at a Glance

| Key | Default |
| --- | --- |
| `dataset` | `LASTFM` |
| `macro_batch_size` | `4000` |
| `eval_macro_batch_size` | `null` |
| `model` | `tgn` |
| `n_neg` | `1` |
| `eval_metric` | `ap` |
| `d_time` | `128` |
| `compile` | `false` |
| `compute.tf32` | `false` |
| `compute.amp` | `false` |

## Root Experiment Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `epoch` | `100` | Number of training epochs. |
| `macro_batch_size` | `4000` | Total batch size across all GPUs. The launcher auto-computes `per_gpu_batch_size = macro_batch_size // n_gpu`. |
| `eval_macro_batch_size` | `null` | Total validation/test batch size across GPUs. Defaults to `macro_batch_size`; use a smaller value for TGB one-vs-many evaluation. |
| `patience` | `100` | Early stopping patience in epochs. |
| `early_stop_tolerance` | `0.0` | Minimum relative validation-metric improvement required to reset early stopping. Use `1e-6` for the official TGB protocol. |
| `dropout` | `0.2` | Shared dropout hyperparameter forwarded into several encoders and memory modules. |
| `seed` | `42` | Shared random seed for sampling and reproducibility. |
| `d_time` | `128` | Time encoding dimension. |
| `dataset` | `LASTFM` | Runtime dataset name. Artifact datasets resolve under `FROST_DATA_ROOT`; registered TGB datasets resolve from official files under `TGB_DATA_ROOT/raw`. |
| `optimizer` | `adam` | Worker optimizer. Supported values: `adam` and `sgd`. |
| `lr` | `0.0001` | Learning rate. |
| `n_neg` | `1` | Number of negative samples per positive edge. Propagates into the negative sampler and related components. |
| `eval_metric` | `ap` | Evaluation, checkpoint-selection, logging, and early-stopping metric. Supported values are `ap` and `mrr`; AP remains the default. |
| `ring_buf_size` | `64` | Number of shared-memory ring-buffer slots per worker GPU. |
| `ignore_nfeat_on_disk` | `false` | Load `node_features.npy` by default. |
| `numa_policy` | `auto` | Topology-aware CPU and shared-memory placement. Use `off` to disable it or a NUMA node ID to bind every local rank there. |
| `cpu_threads` | `auto` | OMP/MKL thread policy. Automatic workers receive at most 32 threads from their GPU-local physical-core budget; the host is resolved separately. Use `inherit` or a positive integer to override. |
| `if_carry_over_route` | `false` | Allows planner fallback to stale leftovers when next-batch needs cannot be fully satisfied. |
| `compile` | `false` | Enables `torch.compile` for the assembled worker model. |
| `compute.tf32` | `false` | Enables TF32 matmul and cuDNN behavior globally. |
| `compute.amp` | `false` | Enables CUDA automatic mixed precision and gradient scaling. |

## Hydra Config Groups

| Group | Role | Common Choices | User-Facing? |
| --- | --- | --- | --- |
| `model` | Complete model recipe and semantic contract | `tgn`, `jodie`, `apan`, `tgat`, `dygformer`, `graph_mixer`, `tcl`, `cawn` | Yes |
| `neg_sampler` | Negative destination sampling | `rnd_dst_then_random` (default), `rnd_dst_then_fixed`, `rnd_dst_then_hist`, `rnd_dst_then_post_train`, `rnd_dst_then_tgb`, `rnd_dst_then_tgb_wiki` | Yes |
| `ngh_sampler` | Runtime neighborhood-sampling implementation | `recent`, `uniform`, `time_interval_aware` | Yes |

Storage and transport dtypes are fixed runtime invariants: IDs and timestamps
use `int64`, while features and dynamic-state payloads use `float32`. They are
not a Hydra config group. Relative-time features are computed before conversion
to model-side floating point so large absolute timestamps retain low-order bits.
The temporal link-prediction objective is likewise fixed to binary
cross-entropy with logits; changing the objective requires a task-level contract,
not a different `torch.nn` module target.

## Parameter Flow and Coupling

`n_neg` flows through negative and neighborhood sampling into structured edge
representations. See [negative sampling](#negative-sampling) for protocols,
candidate sets, evaluation widths and comparison rules.

- `model=<name>` selects the encoder, optional state updater, link head, and a semantic contract. The launcher compiles that contract into sampling, layout, feature-access, and dynamic-state plans.
- `model.embedding_dim` is the single embedding/memory/head width. Recipe interpolation maps it to each concrete constructor.
- `model.neighborhood.fanout` is the single fanout source for every model, including APAN. An empty list selects 0-hop, a one-element list selects 1-hop, and a longer list selects multi-hop.
- TGN's `model.encoder.attention_layout=auto` keeps dense attention for small
  layouts and switches to packed attention when `dense fanout slots × projected
  neighbor width` reaches `auto_pack_min_projected_elements`. Explicit `dense`
  and `packed` values remain available for ablations. The heuristic uses tensor
  shapes and model dimensions only, so it does not synchronize a CUDA mask back
  to the host on every batch.
- `model.state` is `null` for stateless models. Memory recipes declare their updater, mailbox size, combine rule, and routing semantics under this node.
- `macro_batch_size`, `n_neg`, `model.neighborhood.fanout`, the model layout contract, and `ring_buf_size` determine ring-buffer capacity and pipeline depth.
- The runtime neighborhood sampler is configured under `ngh_sampler`. In particular, `ngh_sampler.strategy` selects `uniform`, `recent`, or `time_interval_aware`, and `ngh_sampler.time_scaling_factor` is consumed by the `time_interval_aware` sampler.
- GPU placement is auto-detected from `CUDA_VISIBLE_DEVICES` plus `nvidia-smi topo -m`. The planner distinguishes three levels when available: NVLink peers, same-NUMA peers, and cross-NUMA peers.
- `cpu_threads=auto` divides each NUMA node's physical cores among the GPU workers attached to that node, caps each worker at 32 threads, and resolves the host independently. Explicit `OMP_NUM_THREADS` or `MKL_NUM_THREADS` values take precedence per variable.
- `edim` is inferred from the materialized edge-feature file or the registered raw dataset source and combines with `d_mem` / `boxsize` to determine mailbox dimensionality: `d_mail = 2 * d_mem + edim`.
- `model.state.mail_combine` controls mailbox aggregation for memory models. Supported values are `last` and `mean`.

## Presets and Manual Overrides

Configuration has one owner per concern:

| Source | Owns |
| --- | --- |
| `configs/model/` | Model component constructor defaults |
| `src/frost/models/recipes.py` | Model capabilities and runtime contract |
| `configs/config.yaml` | Shared experiment and runtime defaults |

When tuning manually, keep coupled fields aligned: GraphMixer's `num_tokens`
cannot exceed its sampled history; TCL's `num_depths` follows `fanout[0] + 1`;
CAWN cost grows with the product of its fanouts; DyGFormer's sampled sequence
must remain compatible with its maximum length and patch size. Current values
remain in configuration rather than prose.

## TGB Link Prediction

With the official tgbl-wiki-v2 files under
`$TGB_DATA_ROOT/raw/tgbl_wiki`, a two-GPU TGN run is:

```bash
uv run frost-data fetch-tgb tgbl-wiki
```

Training still uses one random negative per positive. Only validation and test
use the official 998/999-candidate sets. The smaller evaluation batch is
independent of the training batch and bounds ring-buffer and GPU memory use.

## Negative sampling

Negative-sampling results are comparable only when the split, candidate set,
corruption rule, number of negatives, replacement rule, random seed, and metric
are the same. FROST therefore names its protocols by their actual behavior
rather than treating `historical` or `inductive` as universal definitions.

### FROST's destination-corruption layout

For a positive event `(src, dst, time)`, FROST's common link-prediction layout
is:

```text
[src | positive dst | negative dst x n_neg]
```

The predictor repeats the positive source and scores `(src, negative dst)`.
Consequently, the current runtime can express destination corruption but cannot
represent an independently sampled `negative src`. Supporting full-pair
negative edges would require coordinated changes to the sampling buffer, graph
layout, encoders, predictors, and analyzers—not just another sampler class.

### Shipped protocols

| Config | Training | Evaluation | Interpretation |
| --- | --- | --- | --- |
| `rnd_dst_then_random` (default) | Uniform random destination from all node IDs in `[0, num_nodes)` | Runtime random sampling from the same full node range | Sampling with replacement; no positive-destination collision filtering |
| `rnd_dst_then_fixed` | Random destination from the observed destination partition | Fixed `eid,neg_dst` artifact | Reproducible fixed random destination corruption; currently `n_neg=1` |
| `rnd_dst_then_fixed_all` | Random destination from the observed destination partition | Fixed artifact covering the complete contiguous node range, excluding the positive destination | Fixed all-node destination corruption |
| `rnd_dst_then_hist` | Random destination from the observed destination partition | Prefer destinations previously connected to the current source, strictly before the batch; exclude that source's current positive destinations; otherwise fall back to the observed destination partition | Per-source historical destination corruption |
| `rnd_dst_then_post_train` | Random destination from the observed destination partition | Prefer destinations connected to the current source after the training cutoff and strictly before the batch; exclude current positives; otherwise fall back to the observed destination partition | Per-source post-training destination corruption |
| `rnd_dst_then_tgb` / `rnd_dst_then_tgb_wiki` | Random destination | Official TGB validation/test candidates | TGB protocol; use its required metric and candidate width |

Historical and post-training destination candidates are sampled with replacement,
so a positive edge may receive the same negative destination more than once
when `n_neg > 1`.

### Evaluation files and widths

`rnd_dst_then_fixed` reads `$FROST_DATA_ROOT/<dataset>/neg_pre_sampling_1_result.csv`
with one `eid,neg_dst` negative per edge and requires `n_neg=1`.
`rnd_dst_then_fixed_all` reads `neg_pre_sampling_1_all_nodes_seed0.csv` instead.
AP requires one evaluation negative per positive. Runtime-random evaluation
draws fresh samples as its RNG advances. For matched one-vs-many MRR
comparisons, override `neg_sampler.eval.n_neg`.

Generic TGB uses one random training negative and official evaluation pickles;
set `eval_metric=mrr` and `neg_sampler.eval.n_neg=100` for `tgbl-review`, or 20
for `tgbl-coin`, `tgbl-comment` and `tgbl-flight`. Its runtime capacity is fixed,
so every query must match that width. Wiki uses capacity 999 and masks the four
wiki-v2 queries with 998 negatives; it introduces neither ragged tensors nor
random replacements. MRR averages optimistic/pessimistic tie ranks and drives
checkpoint selection, logging and early stopping.

### Historical difference from upstream DyGLib

The original DyGLib snapshot (`3aacc36b`) gave `historical` and `inductive`
different semantics:

- **Historical:** form the global set of historical `(src, dst)` edges, remove
  edges in the current batch, and sample complete negative pairs. If that set
  is too small, fill the remainder with collision-checked random pairs.
- **Inductive:** form the global historical edge set, remove edges observed by
  the training cutoff and edges in the current batch, and sample complete
  negative pairs, again with random-pair fallback when needed.

These are full-pair edge-classification protocols: both `negative src` and
`negative dst` may differ from the positive event. FROST's corresponding
protocols condition on the current source and rank alternative destinations.
Neither task definition is inherently correct or incorrect, but their metrics
must not be compared as if they used the same negatives.

DyGLib's random link-prediction path is the close case: its training and
evaluation call sites discard the random sampler's source output and combine
the current batch source with a sampled destination. With FROST configured for
the observed destination partition, the task semantics match. The concrete
sample sequence can still differ because DyGLib and FROST use different RNG
implementations.

The relevant implementations are:

- FROST: [`historical.py`](../src/frost/neg_sampler/historical.py),
  [`inductive.py`](../src/frost/neg_sampler/inductive.py), and
  [`random.py`](../src/frost/neg_sampler/random.py)

### Reporting and comparison

For a result intended for comparison, report at least:

1. The temporal split policy, including how tied timestamps are handled.
2. The exact protocol name from the table above—not merely `historical` or
   `inductive`.
3. The candidate node/edge set and whether current positives are excluded.
4. Whether sampling uses replacement and the number of negatives per positive.
5. The seed, or the fixed-negative artifact and its checksum.
6. The metric and its reduction/tie policy.

All models within one experiment should consume the same split and negative
queries. FROST historical/inductive results should be labeled with their
per-source destination-corruption names and should not be presented as direct
DyGLib historical/inductive comparisons.

Published `frost_dataset` manifests describe the fixed negative artifact only.
Runtime sampler selection and the definitions above are owned by FROST.
