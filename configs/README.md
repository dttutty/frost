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

The canonical definitions and cross-framework comparison rules live in
[`docs/negative_sampling.md`](../docs/negative_sampling.md).

- `n_neg` flows from root config into negative and neighborhood sampling; scoring receives already structured edge representations.
- `neg_sampler=rnd_dst_then_fixed` uses fresh destination-partition random negatives for training and reads one fixed `eid,neg_dst` evaluation negative per edge from `$FROST_DATA_ROOT/<dataset>/neg_pre_sampling_1_result.csv`; it currently requires `n_neg=1`.
- `neg_sampler=rnd_dst_then_fixed_all` keeps the same destination-partition
  training sampler but evaluates with the fixed
  `neg_pre_sampling_1_all_nodes_seed0.csv` artifact. Its candidates span the
  complete contiguous node range, and the positive destination is excluded.
- `neg_sampler=rnd_dst_then_hist` uses destination-partition random negatives during training and per-source historical destination corruption during eval. It is not DyGLib's full-pair historical protocol, which may replace both endpoints.
- `neg_sampler=rnd_dst_then_post_train` uses destination-partition random negatives during training and per-source post-training destination corruption during eval. It is likewise a destination-only protocol.
- `neg_sampler=rnd_dst_then_tgb` uses one random training negative and official TGB validation/test pickle files. Run with `eval_metric=mrr`; set `neg_sampler.eval.n_neg=100` for `tgbl-review`, or `20` for `tgbl-coin`, `tgbl-comment`, and `tgbl-flight`.
- The runtime keeps a fixed negative capacity. Generic `neg_sampler=rnd_dst_then_tgb` requires every query to have exactly that width. `neg_sampler=rnd_dst_then_tgb_wiki` uses capacity 999 and masks the four official wiki-v2 queries that contain 998 negatives; it does not introduce ragged tensors or random replacements.
- AP mode remains the default and currently requires one evaluation negative per positive edge. MRR is opt-in and uses TGB's optimistic/pessimistic average tie rank for checkpoint selection, logging, and early stopping.
- `neg_sampler=rnd_dst_then_random` samples uniformly from the full node range `[0, num_nodes)` at runtime for training, validation, and testing. Evaluation draws fresh samples as its RNG advances; override `neg_sampler.eval.n_neg` for matched one-vs-many MRR comparisons.
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
| `benchmarks/specs/presets.yaml` | Maintained launch and batch-size defaults |
| `benchmarks/specs/dyglib_parity.yaml` | Reviewed DyGLib parity overrides |
| `configs/config.yaml` | Shared experiment and runtime defaults |

Use `uv run frost-train <model> --dry-run` to inspect the composed command and
`uv run frost-train <model> key=value` to override it. Use Hydra directly when
benchmark parity is not intended.

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

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run frost-train tgn --nproc-per-node=3 \
  dataset=TGBL_WIKI \
  neg_sampler=rnd_dst_then_tgb_wiki eval_metric=mrr \
  macro_batch_size=4096 eval_macro_batch_size=128
```

Training still uses one random negative per positive. Only validation and test
use the official 998/999-candidate sets. The smaller evaluation batch is
independent of the training batch and bounds ring-buffer and GPU memory use.
