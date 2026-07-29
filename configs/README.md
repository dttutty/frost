# Configuration Guide

FROST uses Hydra config groups to compose datasets, models, memory components, and runtime helpers. The main entry point is [`config.yaml`](config.yaml).

## Defaults at a Glance

| Key | Default |
| --- | --- |
| `dataset` | `LASTFM` |
| `macro_batch_size` | `4000` |
| `eval_macro_batch_size` | `null` |
| `access_analyzer` | omitted |
| `n_neg` | `1` |
| `eval_metric` | `ap` |
| `d_time` | `128` |
| `compile` | `false` |
| `tf32` | `false` |

## Root Experiment Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `epoch` | `100` | Number of training epochs. |
| `macro_batch_size` | `4000` | Total batch size across all GPUs. The launcher auto-computes `per_gpu_batch_size = macro_batch_size // n_gpu`. |
| `eval_macro_batch_size` | `null` | Total validation/test batch size across GPUs. Defaults to `macro_batch_size`; use a smaller value for TGB one-vs-many evaluation. |
| `patience` | `100` | Early stopping patience in epochs. |
| `dropout` | `0.2` | Shared dropout hyperparameter forwarded into several encoders and memory modules. |
| `seed` | `42` | Shared random seed for sampling and reproducibility. |
| `d_time` | `128` | Time encoding dimension. |
| `dataset` | `LASTFM` | Runtime dataset name. Artifact datasets resolve under `FROST_DATA_ROOT`; registered TGB datasets resolve from official files under `TGB_DATA_ROOT/raw`. |
| `access_analyzer` | omitted | Non-memory runs fall back to `StaticAnalyzer` in `launcher.py`. Memory runs must select `memory_analyzer` (TGN/JODIE/DyRep) or `apan_memory_analyzer` (APAN). |
| `optimizer` | `adam` | Worker optimizer. Supported values: `adam` and `sgd`. |
| `lr` | `0.0001` | Learning rate. |
| `n_neg` | `1` | Number of negative samples per positive edge. Propagates into the negative sampler and related components. |
| `eval_metric` | `ap` | Evaluation, checkpoint-selection, logging, and early-stopping metric. Supported values are `ap` and `mrr`; AP remains the default. |
| `ring_buf_size` | `64` | Number of shared-memory ring-buffer slots per worker GPU. |
| `if_carry_over_route` | `false` | Allows planner fallback to stale leftovers when next-batch needs cannot be fully satisfied. |
| `compile` | `false` | Enables `torch.compile` for the assembled worker model. |
| `tf32` | `false` | Enables TF32 matmul and cuDNN behavior globally. |

## Hydra Config Groups

| Group | Role | Common Choices | User-Facing? |
| --- | --- | --- | --- |
| `access_analyzer` | Host-side access counting and mail planning | `memory_analyzer`, `apan_memory_analyzer` | Memory models only |
| `encoder` | TGNN encoder architecture | `tgn`, `jodie`, `apan`, `dyrep`, `tgat`, `dygformer`, `graph_mixer`, `tcl` | Yes |
| `loss_fn` | Training loss | `bce` | Fixed |
| `mem_updater` | Memory update module | `gru`, `rnn`, `apan` or `null` | Memory models only |
| `neg_sampler` | Negative destination sampling | `rnd_dst_then_fixed`, `rnd_dst_then_hist`, `rnd_dst_then_post_train`, `rnd_dst_then_tgb`, `rnd_dst_then_tgb_wiki` | Yes |
| `predictor` | Downstream prediction head | `link` | Fixed |

## Parameter Flow and Coupling

The canonical definitions and cross-framework comparison rules live in
[`docs/negative_sampling.md`](../docs/negative_sampling.md).

- `n_neg` flows from root config into `neg_sampler`, access analyzers, neighborhood samplers, the link predictor, and APAN postman components.
- `neg_sampler=rnd_dst_then_fixed` uses fresh destination-partition random negatives for training and reads one fixed `eid,neg_dst` evaluation negative per edge from `$FROST_DATA_ROOT/<dataset>/neg_pre_sampling_1_result.csv`; it currently requires `n_neg=1`.
- `neg_sampler=rnd_dst_then_hist` uses destination-partition random negatives during training and per-source historical destination corruption during eval. It is not DyGLib's full-pair historical protocol, which may replace both endpoints.
- `neg_sampler=rnd_dst_then_post_train` uses destination-partition random negatives during training and per-source post-training destination corruption during eval. It is likewise a destination-only protocol.
- `neg_sampler=rnd_dst_then_tgb` uses one random training negative and official TGB validation/test pickle files. Run with `eval_metric=mrr`; set `neg_sampler.eval.n_neg=100` for `tgbl-review`, or `20` for `tgbl-coin`, `tgbl-comment`, and `tgbl-flight`.
- The runtime keeps a fixed negative capacity. Generic `neg_sampler=rnd_dst_then_tgb` requires every query to have exactly that width. `neg_sampler=rnd_dst_then_tgb_wiki` uses capacity 999 and masks the four official wiki-v2 queries that contain 998 negatives; it does not introduce ragged tensors or random replacements.
- AP mode remains the default and currently requires one evaluation negative per positive edge. MRR is opt-in and uses TGB's optimistic/pessimistic average tie rank for checkpoint selection, logging, and early stopping.
- Omitting `access_analyzer` selects `StaticAnalyzer` only on non-memory runs. Memory runs must still provide an access-analyzer override.
- `fanout` is derived, not stored at the root: APAN uses `mem_updater.mail_recipients`, while every other model uses `encoder.fanout`. `null` selects 0-hop, a 1-element list selects 1-hop, and a longer list selects multi-hop.
- `encoder.out_dim` is the shared embedding size used by the predictor and, for memory models, the default source of `mem_updater.d_mem`.
- `encoder.out_dim` is a config-level alias, not a direct constructor argument for every encoder. It maps to `d_hid` in TGN/TGAT, `d_out` in GraphMixer/TCL/DyGFormer/DyRep, and `d_mem` in JODIE/APAN. `launcher.py` resolves this alias before instantiating the concrete model classes.
- `macro_batch_size`, `n_neg`, the derived `fanout`, `mem_updater.mail_recipients`, and `ring_buf_size` together determine the ring-buffer slot layout and pipeline depth.
- The runtime neighborhood sampler is configured under `ngh_sampler`. In particular, `ngh_sampler.strategy` selects `uniform`, `recent`, or `time_interval_aware`, and `ngh_sampler.time_scaling_factor` is consumed by the `time_interval_aware` sampler.
- GPU placement is auto-detected from `CUDA_VISIBLE_DEVICES` plus `nvidia-smi topo -m`. The planner distinguishes three levels when available: NVLink peers, same-NUMA peers, and cross-NUMA peers.
- `edim` is inferred from the materialized edge-feature file or the registered raw dataset source and combines with `d_mem` / `boxsize` to determine mailbox dimensionality: `d_mail = 2 * d_mem + edim`.
- Setting `mem_updater` also enables the shared memory/mailbox store. Non-memory models should leave `mem_updater=null`.
- `mem_updater.mail_recipients>0` enables APAN-style neighbor-directed mail routing. The shipped `apan` updater sets `10`; the other memory updaters must keep it at `0`.
- `mem_updater.mail_combine` controls mailbox aggregation for memory models. Supported values are `last` and `mean`; the shipped updaters default to `last`.

## TGB Link Prediction

With the official tgbl-wiki-v2 files under
`$TGB_DATA_ROOT/raw/tgbl_wiki`, a two-GPU TGN run is:

```bash
uv run frost-data fetch-tgb tgbl-wiki
```

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=3 run.py \
  access_analyzer=memory_analyzer dataset=TGBL_WIKI encoder=tgn \
  mem_updater=gru neg_sampler=rnd_dst_then_tgb_wiki eval_metric=mrr \
  macro_batch_size=4096 eval_macro_batch_size=128
```

Training still uses one random negative per positive. Only validation and test
use the official 998/999-candidate sets. The smaller evaluation batch is
independent of the training batch and bounds ring-buffer and GPU memory use.
