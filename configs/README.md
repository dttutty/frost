# Configuration Guide

FROST uses Hydra config groups to compose datasets, models, memory components, and runtime helpers. The main entry point is [`config.yaml`](config.yaml).

## Defaults at a Glance

| Key | Default |
| --- | --- |
| `dataset` | `LASTFM` |
| `macro_batch_size` | `4000` |
| `access_analyzer` | omitted |
| `n_neg` | `1` |
| `d_time` | `128` |
| `compile` | `false` |
| `tf32` | `false` |

## Root Experiment Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `epoch` | `100` | Number of training epochs. |
| `macro_batch_size` | `4000` | Total batch size across all GPUs. The launcher auto-computes `per_gpu_batch_size = macro_batch_size // n_gpu`. |
| `patience` | `100` | Early stopping patience in epochs. |
| `dropout` | `0.2` | Shared dropout hyperparameter forwarded into several encoders and memory modules. |
| `seed` | `42` | Shared random seed for sampling and reproducibility. |
| `d_time` | `128` | Time encoding dimension. |
| `dataset` | `LASTFM` | Dataset name under `DATA/`. |
| `access_analyzer` | omitted | Non-memory runs fall back to `StaticAnalyzer` in `launcher.py`. Memory runs must select `memory_analyzer` (TGN/JODIE/DyRep) or `apan_memory_analyzer` (APAN). |
| `optimizer` | `adam` | Worker optimizer. Supported values: `adam` and `sgd`. |
| `lr` | `0.0001` | Learning rate. |
| `n_neg` | `1` | Number of negative samples per positive edge. Propagates into the negative sampler and related components. |
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
| `neg_sampler` | Negative destination sampling | `precomputed`, `rnd_then_rnd`, `rnd_then_hist`, `rnd_then_ind` | Yes |
| `predictor` | Downstream prediction head | `link` | Fixed |

## Parameter Flow and Coupling

- `n_neg` flows from root config into `neg_sampler`, access analyzers, neighborhood samplers, the link predictor, and APAN postman components.
- `neg_sampler=precomputed` reads one precomputed negative destination per edge from `DATA/<dataset>/neg_pre_sampling_1_result.csv` (configurable via `neg_sampler.neg_file`) and currently requires `n_neg=1`.
- `neg_sampler=rnd_then_rnd` uses `RandomNegLinkSampler` for both train and eval; override `neg_sampler.train.candidate_mode` and `neg_sampler.eval.candidate_mode` independently to switch between `all` and `dst_partition`.
- `neg_sampler=rnd_then_hist` uses random negatives during training and `HistoricalNegLinkSampler` during eval.
- `neg_sampler=rnd_then_ind` uses random negatives during training and `InductiveNegLinkSampler` during eval.
- Omitting `access_analyzer` selects `StaticAnalyzer` only on non-memory runs. Memory runs must still provide an access-analyzer override.
- `fanout` is derived, not stored at the root: APAN uses `mem_updater.mail_recipients`, while every other model uses `encoder.fanout`. `null` selects 0-hop, a 1-element list selects 1-hop, and a longer list selects multi-hop.
- `encoder.out_dim` is the shared embedding size used by the predictor and, for memory models, the default source of `mem_updater.d_mem`.
- `encoder.out_dim` is a config-level alias, not a direct constructor argument for every encoder. It maps to `d_hid` in TGN/TGAT, `d_out` in GraphMixer/TCL/DyGFormer/DyRep, and `d_mem` in JODIE/APAN. `launcher.py` resolves this alias before instantiating the concrete model classes.
- `macro_batch_size`, `n_neg`, the derived `fanout`, `mem_updater.mail_recipients`, and `ring_buf_size` together determine the ring-buffer slot layout and pipeline depth.
- The runtime neighborhood sampler is configured under `ngh_sampler`. In particular, `ngh_sampler.strategy` selects `uniform`, `recent`, or `time_interval_aware`, and `ngh_sampler.time_scaling_factor` is consumed by the `time_interval_aware` sampler.
- GPU placement is auto-detected from `CUDA_VISIBLE_DEVICES` plus `nvidia-smi topo -m`. The planner distinguishes three levels when available: NVLink peers, same-NUMA peers, and cross-NUMA peers.
- `edim` is inferred from the edge-feature file under `DATA/<dataset>/` (typically `edge_features.npy`) and combines with `d_mem` / `boxsize` to determine mailbox dimensionality: `d_mail = 2 * d_mem + edim`.
- Setting `mem_updater` also enables the shared memory/mailbox store. Non-memory models should leave `mem_updater=null`.
- `mem_updater.mail_recipients>0` enables APAN-style neighbor-directed mail routing. The shipped `apan` updater sets `10`; the other memory updaters must keep it at `0`.
- `mem_updater.mail_combine` controls mailbox aggregation for memory models. Supported values are `last` and `mean`; the shipped updaters default to `last`.

