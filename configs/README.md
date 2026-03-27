# Configuration Guide

FROST uses Hydra config groups to compose datasets, models, memory components, and runtime helpers. The main entry point is [`config.yaml`](config.yaml).

## Defaults at a Glance

| Key | Default |
| --- | --- |
| `dataset` | `LASTFM` |
| `batch_size` | `4000` |
| `access_analyzer` | omitted |
| `n_neg` | `1` |
| `d_time` | `128` |
| `compile` | `false` |
| `tf32` | `false` |

## Root Experiment Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `epoch` | `100` | Number of training epochs. |
| `batch_size` | `4000` | Per-worker local batch size. The host groups `batch_size * n_gpu` samples into a macro-batch before dispatch. |
| `patience` | `100` | Early stopping patience in epochs. |
| `dropout` | `0.2` | Shared dropout hyperparameter forwarded into several encoders and memory modules. |
| `seed` | `42` | Shared random seed for sampling and reproducibility. |
| `d_time` | `128` | Time encoding dimension. |
| `dataset` | `LASTFM` | Dataset name under `DATA/`. |
| `access_analyzer` | omitted | Non-memory runs fall back to `NoMemAnalyzer` in `launcher.py`. Memory runs must still select a concrete access analyzer config such as `tgn`, `jodie`, `dyrep`, or `apan`. |
| `optimizer` | `adam` | Worker optimizer. Supported values: `adam` and `sgd`. |
| `lr` | `0.0001` | Learning rate. |
| `n_neg` | `1` | Number of negative samples per positive edge. Propagates into the negative sampler and related components. |
| `ring_buf_size` | `32` | Number of shared-memory ring-buffer slots per worker GPU. |
| `handle_stale` | `false` | Allows planner fallback to stale leftovers when next-batch needs cannot be fully satisfied. |
| `compile` | `false` | Enables `torch.compile` for the assembled worker model. |
| `tf32` | `false` | Enables TF32 matmul and cuDNN behavior globally. |

## Hydra Config Groups

| Group | Role | Common Choices | User-Facing? |
| --- | --- | --- | --- |
| `access_analyzer` | Host-side access counting and mail planning | `tgn`, `jodie`, `dyrep`, `apan` | Memory models only |
| `encoder` | TGNN encoder architecture | `tgn`, `jodie`, `apan`, `dyrep`, `tgat`, `dygformer`, `graph_mixer`, `tcl` | Yes |
| `loss_fn` | Training loss | `bce` | Fixed |
| `mem_updater` | Memory update module | `tgn`, `jodie`, `dyrep`, `apan` or `null` | Memory models only |
| `neg_sampler` | Negative destination sampling | `base` | Fixed |
| `predictor` | Downstream prediction head | `link` | Fixed |

## Parameter Flow and Coupling

- `n_neg` flows from root config into `neg_sampler`, access analyzers, neighborhood samplers, the link predictor, and APAN postman components.
- Omitting `access_analyzer` selects `NoMemAnalyzer` only on non-memory runs. Memory runs must still provide an access-analyzer override.
- `fanout` is derived, not stored at the root: APAN uses `mem_updater.mail_recipients`, while every other model uses `encoder.fanout`. `null` selects 0-hop, a 1-element list selects 1-hop, and a longer list selects multi-hop.
- `encoder.out_dim` is the shared embedding size used by the predictor and, for memory models, the default source of `mem_updater.d_mem`.
- `encoder.out_dim` is a config-level alias, not a direct constructor argument for every encoder. It maps to `d_hid` in TGN/TGAT, `d_out` in GraphMixer/TCL/DyGFormer/DyRep, and `d_mem` in JODIE/APAN. `launcher.py` resolves this alias before instantiating the concrete model classes.
- `batch_size`, `n_neg`, the derived `fanout`, `mem_updater.mail_recipients`, and `ring_buf_size` together determine the ring-buffer slot layout and pipeline depth.
- The neighborhood sampling strategy is also derived: APAN uses `mem_updater.postman_delivery_strategy`, while every other model uses `encoder.ngh_sampler_strategy`.
- GPU placement is auto-detected from `CUDA_VISIBLE_DEVICES` plus `nvidia-smi topo -m`. The planner distinguishes three levels when available: NVLink peers, same-NUMA peers, and cross-NUMA peers.
- `edim` is inferred from the edge-feature file under `DATA/<dataset>/` (typically `edge_features.npy`) and combines with `d_mem` / `boxsize` to determine mailbox dimensionality: `d_mail = 2 * d_mem + edim`.
- Setting `mem_updater` also enables the shared memory/mailbox store. Non-memory models should leave `mem_updater=null`.
- `mem_updater.mail_recipients>0` enables APAN-style neighbor-directed mail routing. The shipped `apan` updater sets `10`; the other memory updaters must keep it at `0`.
- `mem_updater.mail_combine` controls mailbox aggregation for memory models. Supported values are `last` and `mean`; the shipped updaters default to `last`.

## Model-Specific Commands

The common training entry point is [`run.py`](../run.py):

```bash
uv run torchrun --nproc_per_node=3 run.py <hydra overrides...>
```

### 🧠 Memory-Based Models

**TGN** — 1-hop, GRU memory, mailbox:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  access_analyzer=tgn encoder=tgn mem_updater=tgn dataset=LASTFM
```

**JODIE** — 0-hop, RNN memory:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  access_analyzer=jodie encoder=jodie mem_updater=jodie dataset=LASTFM
```

**DyRep** — 0-hop, JODIE-style analyzer, approximated messages (`[self_mem, other_mem, edge_feat]`):

```bash
uv run torchrun --nproc_per_node=3 run.py \
  access_analyzer=dyrep encoder=dyrep mem_updater=dyrep dataset=LASTFM
```

**APAN** — 1-hop (src+dst only), transformer memory, `boxsize=10`:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  access_analyzer=apan encoder=apan mem_updater=apan dataset=LASTFM
```

### 📘 Non-Memory Models

**TGAT** — 2-hop:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  encoder=tgat dataset=LASTFM
```

**GraphMixer** — 1-hop, `num_tokens` should track `fanout[0]`:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  encoder=graph_mixer encoder.fanout=[20] dataset=LASTFM
```

**TCL** — 1-hop, `num_depths = fanout[0] + 1`:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  encoder=tcl encoder.fanout=[20] dataset=LASTFM
```

**DyGFormer** — 1-hop:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  encoder=dygformer encoder.fanout=[32] dropout=0.1 dataset=LASTFM
```

