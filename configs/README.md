# Configuration Guide

FROST uses Hydra config groups to compose datasets, samplers, models, memory components, and runtime helpers. The main entry point is [`config.yaml`](config.yaml).

## Defaults at a Glance

| Key | Default |
| --- | --- |
| `dataset` | `LASTFM` |
| `batch_size` | `4000` |
| `fanout` | `[10]` |
| `n_neg` | `1` |
| `final_emb_dim` | `128` |
| `d_time` | `128` |
| `compile` | `true` |
| `tf32` | `true` |

## Root Experiment Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `epoch` | `1` | Number of training epochs. |
| `batch_size` | `4000` | Per-worker local batch size. The host groups `batch_size * n_gpu` samples into a macro-batch before dispatch. |
| `patience` | `50` | Early stopping patience in epochs. |
| `dropout` | `0.2` | Shared dropout hyperparameter forwarded into several encoders and memory modules. |
| `seed` | `null` | Shared random seed for sampling and reproducibility. `null` leaves runs unfixed. |
| `d_time` | `128` | Time encoding dimension. |
| `final_emb_dim` | `128` | Output embedding dimension used by downstream prediction and several encoders. |
| `dataset` | `LASTFM` | Dataset name under `DATA/`. |
| `ndim` | `null` | Node feature override. `null`: load from disk if present; `0`: disable node features; `>0`: generate random node features of that size. |
| `edim` | `null` | Edge feature override with the same semantics as `ndim`. |
| `mail_combine` | `null` | Mailbox aggregation mode on memory-model paths. Supported values: `last` and `mean`. |
| `optimizer` | `adam` | Worker optimizer. Supported values: `adam` and `sgd`. |
| `gpu_topo` | `[0, 1]` | Flat or nested GPU topology description used by the planner to build an inter-GPU distance matrix. |
| `fanout` | `[10]` | Per-hop neighbor counts used by samplers and some encoders. |
| `lr` | `0.0001` | Learning rate. |
| `n_neg` | `1` | Number of negative samples per positive edge. Propagates into the negative sampler and related components. |
| `handle_stale` | `false` | Allows planner fallback to stale leftovers when next-batch needs cannot be fully satisfied. |
| `compile` | `true` | Enables `torch.compile` for the assembled worker model. |
| `tf32` | `true` | Enables TF32 matmul and cuDNN behavior globally. |

## Hydra Config Groups

| Group | Role | Common Choices | User-Facing? |
| --- | --- | --- | --- |
| `analyzer` | Host-side access counting and mail planning | `no_mem`, `tgn`, `jodie`, `dyrep`, `apan` | Yes |
| `dataset_mgr` | Loads topology and features into shared memory | `base` | Plumbing |
| `encoder` | TGNN encoder architecture | `tgn`, `jodie`, `apan`, `dyrep`, `tgat`, `dygformer`, `graph_mixer`, `tcl` | Yes |
| `loss_fn` | Training loss | `bce` | Fixed |
| `mem_box_store` | Shared memory/mailbox store | `base` or `null` | Memory models only |
| `mem_updater` | Memory update module | `tgn`, `jodie`, `dyrep`, `apan` or `null` | Memory models only |
| `neg_sampler` | Negative destination sampling | `base` | Fixed |
| `ngh_sampler` | Temporal neighborhood sampling | `0_hop`, `1_hop`, `2_hop` | Yes |
| `postman` | Neighbor-directed mail routing | `apan` or `null` | APAN only |
| `predictor` | Downstream prediction head | `link` | Fixed |
| `ring_buf` | Shared-memory ring buffer sizing and layout | `base` | Plumbing |

## Parameter Flow and Coupling

- `n_neg` flows from root config into `neg_sampler`, analyzers, neighborhood samplers, the link predictor, and APAN postman components.
- `fanout` is consumed by samplers and directly affects TGAT, GraphMixer, TCL, and DyGFormer behavior.
- `d_mem`, `boxsize`, and `edim` determine mailbox dimensionality: `d_mail = 2 * d_mem + edim`.
- `mem_box_store` should be enabled only for memory-based models. Non-memory models should leave `mem_box_store=null` and `mem_updater=null`.
- `mail_combine` matters only when mailbox/message aggregation is active.

## Model-Specific Commands

### Memory-Based Models

**TGN** — 1-hop, GRU memory, mailbox:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  analyzer=tgn dataset=LASTFM encoder=tgn \
  mem_box_store=base mem_updater=tgn \
  ngh_sampler=1_hop mail_combine=last
```

**JODIE** — 0-hop, RNN memory:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  analyzer=jodie dataset=LASTFM encoder=jodie \
  mem_box_store=base mem_updater=jodie \
  ngh_sampler=0_hop mail_combine=last
```

**DyRep** — 0-hop, JODIE-style analyzer, approximated messages (`[self_mem, other_mem, edge_feat]`):

```bash
uv run torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM encoder=dyrep analyzer=dyrep \
  mem_box_store=base mem_updater=dyrep \
  ngh_sampler=0_hop mail_combine=last
```

**APAN** — 1-hop (src+dst only), transformer memory, `boxsize=10`:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  analyzer=apan dataset=LASTFM encoder=apan \
  mem_box_store=base mem_updater=apan \
  ngh_sampler=1_hop ngh_sampler.sample_4_neg=False \
  postman=apan mail_combine=last
```

### Non-Memory Models

**TGAT** — 2-hop:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM encoder=tgat ngh_sampler=2_hop
```

**GraphMixer** — 1-hop, `num_tokens` should track `fanout[0]`:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM encoder=graph_mixer \
  ngh_sampler=1_hop fanout=[20]
```

**TCL** — 1-hop, `num_depths = fanout[0] + 1`:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM encoder=tcl \
  ngh_sampler=1_hop fanout=[20]
```

**DyGFormer** — 1-hop:

```bash
uv run torchrun --nproc_per_node=3 run.py \
  dataset=LASTFM encoder=dygformer \
  ngh_sampler=1_hop fanout=[32] dropout=0.1
```

