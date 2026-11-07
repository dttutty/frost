# 🧠 Model and training guide

[← Score candidate items: installation, CSV format, features and queries](../README.md)

Use this guide to choose models, change training parameters, select sampling
methods or run multi-GPU training. FROST trains temporal graph neural networks
for recommendation systems, using interaction histories to score candidate items.

## 🧭 Choose a training entry point

| Entry point | Models | Configuration / input |
| --- | --- | --- |
| `frost train` | TGN, TGAT | Starter YAML and CLI flags; your event CSV; CPU or one GPU |
| `torchrun … -m frost` | Eight native recipes listed below | Hydra overrides; prepared graph artifacts; one or more GPU workers plus a host rank |
| `torchrun … -m frost.tpnet.train` | TPNet | Dedicated trainer flags; classic processed data or TGB; one process per GPU |

The CSV trainer uses [configs/predict.yaml](../configs/predict.yaml). The native
trainer uses [configs/config.yaml](../configs/config.yaml) and [model recipes](../configs/model/).
Their hyperparameters, sampling and runtime behavior are configured separately;
the native support matrix below does not expand the CSV CLI's model choices.
Install the package using the [prediction README](../README.md#-install).

## ⚙️ Tune the CSV trainer

`frost train events.csv` uses [configs/predict.yaml](../configs/predict.yaml):

| Setting | Value |
| --- | --- |
| Model / memory | TGN / GRU, last-message aggregation |
| Neighbor sampling | Recent, 1 hop, 10 neighbors |
| Embedding / memory / time dimensions | 128 / 128 / 128 |
| Attention heads / dropout | 2 / 0.2 |
| Epochs / batch size | 10 / 200 |
| Optimizer / learning rate | Adam / 0.001 |
| Negatives per positive | 1 random destination |
| Seed / device / CPU threads | 42 / CPU / 4 |

```bash
frost train --show-config
frost train events.csv --epochs 20 --lr 0.0005
frost train --show-config > settings.yaml
frost train events.csv --config settings.yaml --epochs 20
frost train events.csv --model tgat --output tgat.pt
```

Precedence: CLI flags → your YAML → bundled defaults. Unknown YAML keys and
invalid values are rejected before training. These are starter settings, not
paper experiment settings; the advanced trainer uses [configs/config.yaml](../configs/config.yaml).

The CSV trainer uses learned node embeddings and bidirectional historical
neighborhoods. Its random destination negatives exclude the source and all
simultaneous positive destinations. Both CSV models use one sampling hop;
`--model tgat` selects the stateless TGAT encoder. These settings differ from
some native recipes in the support matrix.

TGN resets memory each epoch and publishes positive-event messages after each
chronological batch. Equal-time events share one frozen memory snapshot; large
groups use bounded compute chunks. Saving replays history with the final weights
and persists both memory and pending messages. Predictions read this state without
updating it, so test rows and query order do not change subsequent answers.
Resolved settings are stored in the checkpoint. TGAT remains selectable, and
checkpoints from 0.2.0 can still be loaded.

## 📋 Support matrix

### 🗃️ Datasets

| Data source | Supported datasets |
| --- | --- |
| [🤗 frost_dataset](https://huggingface.co/datasets/dttutty/frost_dataset) | Wikipedia (`WIKI`), Reddit, MOOC, LastFM, MAG, GDELT |
| TGB link prediction | `TGBL_WIKI` (v2), `TGBL_REVIEW` (v2), `TGBL_COIN` (v2), `TGBL_COMMENT` (v1), `TGBL_FLIGHT` (v2) |
| Your data | [Event CSVs with optional timestamps and node features](../README.md#-score-candidate-items) via `frost train` |

For the advanced trainer, use `dataset=<name>` matching the prepared dataset directory, or a TGB identifier
above. `frost-data` provides graph construction, temporal splitting, feature
conversion and TGB downloads. Graph direction can be automatic, directed or
bidirectional. These are loader capabilities, not a claim that every
model–dataset combination was evaluated in the paper.

### 🕒 Temporal neighbor sampling

For native recipes, positive examples are observed interactions; the options
below select historical neighbors for encoding or mailbox delivery. Select with `ngh_sampler=<method>`.

| Method | Behavior |
| --- | --- |
| [`recent`](../configs/ngh_sampler/recent.yaml) | Most recent eligible neighbors |
| [`uniform`](../configs/ngh_sampler/uniform.yaml) | Uniform sampling from eligible neighbors |
| [`time_interval_aware`](../configs/ngh_sampler/time_interval_aware.yaml) | Time-aware weighted sampling, controlled by `ngh_sampler.time_scaling_factor` |

Sampling depth and per-hop fanout are set by `model.neighborhood.fanout`, subject
to the model's supported depth below. JODIE uses no neighbor sampling.

### 🎲 Negative sampling

Native recipes keep the source fixed and sample negative destinations.
Select a train/evaluation pair with `neg_sampler=<configuration>`.

| Configuration | Training | Validation / test |
| --- | --- | --- |
| [`rnd_dst_then_random`](../configs/neg_sampler/rnd_dst_then_random.yaml) | Random, all nodes | Random, all nodes |
| [`rnd_dst_then_fixed`](../configs/neg_sampler/rnd_dst_then_fixed.yaml) | Random, destination partition | Precomputed destination negatives |
| [`rnd_dst_then_fixed_all`](../configs/neg_sampler/rnd_dst_then_fixed_all.yaml) | Random, destination partition | Precomputed all-node negatives |
| [`rnd_dst_then_hist`](../configs/neg_sampler/rnd_dst_then_hist.yaml) | Random, destination partition | Previously observed destinations for each source |
| [`rnd_dst_then_post_train`](../configs/neg_sampler/rnd_dst_then_post_train.yaml) | Random, destination partition | Destinations observed for each source after the training cutoff |
| [`rnd_dst_then_tgb`](../configs/neg_sampler/rnd_dst_then_tgb.yaml) | Random, destination partition | Automatic official layout: Wiki 999 (masked), Review 100, Coin/Comment/Flight 20 |

`rnd_dst_then_tgb` requires `dataset=TGBL_WIKI`, `TGBL_COIN`, `TGBL_COMMENT`,
`TGBL_FLIGHT` or `TGBL_REVIEW` and `eval_metric=mrr`. Candidate capacity and Wiki
padding masks are selected automatically; no `neg_sampler.eval.n_neg` override
is needed. Unsupported datasets and conflicting candidate settings are rejected
before training setup. `rnd_dst_then_tgb_wiki` remains a compatibility alias.

Historical and post-training sampling use only prior events, exclude current
positive destinations and fall back to random destination candidates when needed.
The random default and precomputed CSV recipes use one negative per positive;
TGB recipes use one during training and multiple during evaluation. Evaluation
supports AP/AUC and `eval_metric=mrr` for TGB candidate ranking.

### 🧠 Models

All eight native recipes support single- and multi-GPU training. Values below
are **defaults in this library**; click a model for its configuration.

| Model | Default encoder | Sampling hops (fanout) | Encoder neighbor aggregation | Neighbor mailbox delivery | Node memory / updater |
| --- | --- | --- | --- | --- | --- |
| [TGN](../configs/model/tgn.yaml) | Temporal graph attention | 1 (`[10]`) | Yes, 1 hop | No, 0 hops | Yes / GRU |
| [JODIE](../configs/model/jodie.yaml) | Memory + time projection | 0 (`[]`) | No, 0 hops | No, 0 hops | Yes / GRU |
| [APAN](../configs/model/apan.yaml) | LayerNorm of updated memory | 1 (`[10]`) | No, 0 hops | Yes, 1 hop | Yes / Transformer |
| [TGAT](../configs/model/tgat.yaml) | Stacked temporal graph attention | 2 (`[20, 20]`) | Yes, 2 hops | No, 0 hops | No |
| [CAWN](../configs/model/cawn.yaml) | Anonymous walks: BiLSTM + attention | 2 (`[20, 20]`) | Yes, 2-hop walks | No, 0 hops | No |
| [TCL](../configs/model/tcl.yaml) | Transformer self-/cross-attention | 1 (`[20]`) | Yes, 1-hop histories | No, 0 hops | No |
| [GraphMixer](../configs/model/graph_mixer.yaml) | MLP-Mixer + neighbor aggregation | 1 (`[2000]`) | Yes, 1-hop histories | No, 0 hops | No |
| [DyGFormer](../configs/model/dygformer.yaml) | Transformer over patched temporal sequences | 1 (`[32]`) | Yes, 1-hop histories | No, 0 hops | No |

Encoder aggregation describes the sampled graph receptive field, not neural-layer
count. Mailbox delivery means sending interaction messages to sampled neighbors;
TGN and JODIE still update the interaction endpoints. APAN samples neighbors for
1-hop mailbox delivery and integrates received messages in its memory updater.
TGAT and CAWN accept 1–8 sampling hops; CAWN's `model.encoder.walk_length` must
match the hop count. The other recipes use the fixed depths shown above.

**TPNet (dedicated trainer).** [TPNet](../src/frost/tpnet/) is included with its own
single-/multi-GPU trainer. It combines an MLP-Mixer encoder with temporal-walk
random projections. Its globally decayed projection state uses a dedicated update
path rather than the native memory/mailbox routing and pipeline; launch it with
`-m frost.tpnet.train` instead of `model=tpnet`. See the [source notice](../src/frost/tpnet/NOTICE.md)
for upstream versions and licensing.

## ⚡ Advanced multi-GPU training

Set `FROST_DATA_ROOT` and `TGB_DATA_ROOT` in your environment or `.env.local`.
Use [prepared datasets](https://huggingface.co/datasets/dttutty/frost_dataset)
or `frost-data --help` to construct graph artifacts. Run two GPU workers and one host rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=3 -m frost \
  model=tgn dataset=LASTFM
```

For one GPU worker, select one GPU and use `--nproc_per_node=2`.
Override parameters with `key=value`; see [defaults](../configs/config.yaml)
and [model configs](../configs/model/). For example:

```bash
python -m frost --cfg job model=tgn
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=3 -m frost \
  model=tgn dataset=LASTFM epoch=20 lr=0.0005 ngh_sampler=uniform
```

`epoch` and `lr` here are Hydra settings; the CSV trainer uses `--epochs` and
`--lr`. The native multi-GPU trainer consumes prepared graph artifacts, not the
raw CSV interface shown in the prediction README.

For TPNet, use one process per GPU (no host rank), after downloading the TGB data:

```bash
frost-data fetch-tgb tgbl-wiki
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m frost.tpnet.train --dataset_name tgbl-wiki --batch_size 400 \
  --use_random_projection --report-mrr
```

TPNet's `--batch_size` is global. For classic datasets, pass
`--dataset_name wikipedia --processed-data-root /path/to/processed_data --report-ap`
in place of the TGB dataset and MRR options. Classic data uses
`<dataset>/ml_<dataset>.csv`, `ml_<dataset>.npy` and `ml_<dataset>_node.npy`.

## 🛠️ Development

From a checkout, install with `uv sync --locked --python 3.12`.
Building from source requires a C++17 compiler, CMake and OpenMP;
`python -m pip install .` is also supported.

```bash
uv run ruff check .
uv run mypy
uv run pytest tests/ -q
```

The coverage badge measures Python lines exercised by CPU tests; C++ and GPU
execution are not measured. GitHub Actions updates it after successful main-branch
tests and retains the full report as an artifact.

Code is in [src/frost](../src/frost/), tests in [tests](../tests/).
Rebuild C++ sampler changes with `uv pip install -e .`.

For citation and licensing, see the [prediction README](../README.md#-citation).
