# FROST

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.9.0](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](uv.lock)
[![Linux x86-64](https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black)](#-install)

A multi-GPU framework for Graph Neural Network training.

## 📋 Support matrix

### 🗃️ Datasets

| Data source | Supported datasets |
| --- | --- |
| Prepared temporal graphs | Wikipedia (`WIKI`), Reddit, MOOC, LastFM, MAG, GDELT |
| TGB link prediction | `TGBL_WIKI` (v2), `TGBL_REVIEW` (v2), `TGBL_COIN` (v2), `TGBL_COMMENT` (v1), `TGBL_FLIGHT` (v2) |
| Custom data | Temporal edge tables and graph artifacts in the [FROST data format](src/frost/data/preparation.py) |

Use `dataset=<name>` matching the prepared dataset directory, or a TGB identifier
above. `frost-data` provides graph construction, temporal splitting, feature
conversion and TGB downloads. Graph direction can be automatic, directed or
bidirectional. These are loader capabilities, not a claim that every
model–dataset combination was evaluated in the paper.

### 🕒 Temporal neighbor sampling

Positive examples are observed interactions; the options below select historical
neighbors for encoding or mailbox delivery. Select with `ngh_sampler=<method>`.

| Method | Behavior |
| --- | --- |
| [`recent`](configs/ngh_sampler/recent.yaml) (default) | Most recent eligible neighbors |
| [`uniform`](configs/ngh_sampler/uniform.yaml) | Uniform sampling from eligible neighbors |
| [`time_interval_aware`](configs/ngh_sampler/time_interval_aware.yaml) | Time-aware weighted sampling, controlled by `ngh_sampler.time_scaling_factor` |

Sampling depth and per-hop fanout are set by `model.neighborhood.fanout`, subject
to the model's supported depth below. JODIE uses no neighbor sampling.

### 🎲 Negative sampling

All built-in methods keep the source fixed and sample negative destinations.
Select a train/evaluation pair with `neg_sampler=<configuration>`.

| Configuration | Training | Validation / test |
| --- | --- | --- |
| [`rnd_dst_then_random`](configs/neg_sampler/rnd_dst_then_random.yaml) (default) | Random, all nodes | Random, all nodes |
| [`rnd_dst_then_fixed`](configs/neg_sampler/rnd_dst_then_fixed.yaml) | Random, destination partition | Precomputed destination negatives |
| [`rnd_dst_then_fixed_all`](configs/neg_sampler/rnd_dst_then_fixed_all.yaml) | Random, destination partition | Precomputed all-node negatives |
| [`rnd_dst_then_hist`](configs/neg_sampler/rnd_dst_then_hist.yaml) | Random, destination partition | Previously observed destinations for each source |
| [`rnd_dst_then_post_train`](configs/neg_sampler/rnd_dst_then_post_train.yaml) | Random, destination partition | Destinations observed for each source after the training cutoff |
| [`rnd_dst_then_tgb`](configs/neg_sampler/rnd_dst_then_tgb.yaml) | Random, destination partition | Official TGB candidates (20 by default; set `neg_sampler.eval.n_neg=100` for Review) |
| [`rnd_dst_then_tgb_wiki`](configs/neg_sampler/rnd_dst_then_tgb_wiki.yaml) | Random, destination partition | Official Wiki candidates, padded to 999 with validity masks |

Historical and post-training sampling use only prior events, exclude current
positive destinations and fall back to random destination candidates when needed.
The random default and precomputed CSV recipes use one negative per positive;
TGB recipes use one during training and multiple during evaluation. Evaluation
supports AP/AUC (default) and `eval_metric=mrr` for TGB candidate ranking.

### 🧠 Models

All eight native recipes support single- and multi-GPU training. Values below
are **defaults in this library**; click a model for its configuration.

| Model | Default encoder | Sampling hops (fanout) | Encoder neighbor aggregation | Neighbor mailbox delivery | Node memory / updater |
| --- | --- | --- | --- | --- | --- |
| [TGN](configs/model/tgn.yaml) | Temporal graph attention | 1 (`[10]`) | Yes, 1 hop | No, 0 hops | Yes / GRU |
| [JODIE](configs/model/jodie.yaml) | Memory + time projection | 0 (`[]`) | No, 0 hops | No, 0 hops | Yes / GRU |
| [APAN](configs/model/apan.yaml) | LayerNorm of updated memory | 1 (`[10]`) | No, 0 hops | Yes, 1 hop | Yes / Transformer |
| [TGAT](configs/model/tgat.yaml) | Stacked temporal graph attention | 2 (`[20, 20]`) | Yes, 2 hops | No, 0 hops | No |
| [CAWN](configs/model/cawn.yaml) | Anonymous walks: BiLSTM + attention | 2 (`[20, 20]`) | Yes, 2-hop walks | No, 0 hops | No |
| [TCL](configs/model/tcl.yaml) | Transformer self-/cross-attention | 1 (`[20]`) | Yes, 1-hop histories | No, 0 hops | No |
| [GraphMixer](configs/model/graph_mixer.yaml) | MLP-Mixer + neighbor aggregation | 1 (`[2000]`) | Yes, 1-hop histories | No, 0 hops | No |
| [DyGFormer](configs/model/dygformer.yaml) | Transformer over patched temporal sequences | 1 (`[32]`) | Yes, 1-hop histories | No, 0 hops | No |

Encoder aggregation describes the sampled graph receptive field, not neural-layer
count. Mailbox delivery means sending interaction messages to sampled neighbors;
TGN and JODIE still update the interaction endpoints. APAN samples neighbors for
1-hop mailbox delivery and integrates received messages in its memory updater.
TGAT and CAWN accept 1–8 sampling hops; CAWN's `model.encoder.walk_length` must
match the hop count. The other recipes use the fixed depths shown above.

**TPNet (separate integration).** The paper evaluates TPNet through a separate
multi-GPU adapter. Its globally decayed random-projection state requires a special
update path that does not use FROST's native dynamic-state routing and pipeline.
This library has no built-in `model=tpnet` recipe.

## 📦 Install

GPU training requires an NVIDIA GPU with a compatible driver.

```bash
uv sync --locked --python 3.12
cp .env.example .env.local
```

Set `FROST_DATA_ROOT` and `TGB_DATA_ROOT` in `.env.local`.
Download the [prepared datasets](https://huggingface.co/datasets/dttutty/frost_dataset)
or use `uv run frost-data --help` for data operations.

## 🚀 Usage

Run two GPU workers and one host rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=3 run.py \
  model=tgn dataset=LASTFM
```

For one GPU worker, select one GPU and use `--nproc_per_node=2`.
Override parameters with `key=value`; see [defaults](configs/config.yaml)
and [model configs](configs/model/). Preview the resolved configuration with
`uv run python run.py --cfg job model=tgn`.

## 🛠️ Development

```bash
uv run ruff check .
uv run mypy
uv run pytest tests/ -q
```

Code is in [src/frost](src/frost/), tests in [tests](tests/).
Rebuild C++ sampler changes with `uv pip install -e .`.

## 📚 Citation

If you use FROST, please cite [our paper](https://doi.org/10.1145/3856345):

```bibtex
@article{zhao2026frost,
  author    = {Zhao, Lei and Nke, Joseph and Kim, Jongryool and Lee, Chul-Ho},
  title     = {{FROST}: Accelerating Temporal Graph Neural Network Training via a Unified {Multi-GPU} Framework},
  journal   = {Proceedings of the ACM on Management of Data},
  volume    = {4},
  number    = {6},
  articleno = {350},
  year      = {2026},
  month     = dec,
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3856345},
  url       = {https://doi.org/10.1145/3856345}
}
```

Citation metadata: [CITATION.cff](CITATION.cff).
The paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
