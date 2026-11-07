# FROST

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.9.0](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](uv.lock)
[![Linux x86-64](https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black)](#-install)
[![CPU coverage](https://img.shields.io/badge/CPU%20coverage-69.6%25-yellowgreen)](https://github.com/dttutty/frost_dev/actions/workflows/coverage.yml)

A multi-GPU framework for Temporal (Dynamic) Graph Neural Network training.

## 📋 Support matrix

### 🗃️ Datasets

| Data source | Supported datasets |
| --- | --- |
| [🤗 frost_dataset](https://huggingface.co/datasets/dttutty/frost_dataset) | Wikipedia (`WIKI`), Reddit, MOOC, LastFM, MAG, GDELT |
| TGB link prediction | `TGBL_WIKI` (v2), `TGBL_REVIEW` (v2), `TGBL_COIN` (v2), `TGBL_COMMENT` (v1), `TGBL_FLIGHT` (v2) |
| Custom data | Temporal edge tables and graph artifacts in the [FROST data format](src/frost/data/preparation.py) |

Use `dataset=<name>` matching the prepared dataset directory, or a TGB identifier
above. `frost-data` provides graph construction, temporal splitting, feature
conversion and TGB downloads. Graph direction can be automatic, directed or
bidirectional. These are loader capabilities, not a claim that every
model–dataset combination was evaluated in the paper.

### 🕒 Temporal neighbor sampling

For native recipes, positive examples are observed interactions; the options
below select historical neighbors for encoding or mailbox delivery. Select with `ngh_sampler=<method>`.

| Method | Behavior |
| --- | --- |
| [`recent`](configs/ngh_sampler/recent.yaml) | Most recent eligible neighbors |
| [`uniform`](configs/ngh_sampler/uniform.yaml) | Uniform sampling from eligible neighbors |
| [`time_interval_aware`](configs/ngh_sampler/time_interval_aware.yaml) | Time-aware weighted sampling, controlled by `ngh_sampler.time_scaling_factor` |

Sampling depth and per-hop fanout are set by `model.neighborhood.fanout`, subject
to the model's supported depth below. JODIE uses no neighbor sampling.

### 🎲 Negative sampling

Native recipes keep the source fixed and sample negative destinations.
Select a train/evaluation pair with `neg_sampler=<configuration>`.

| Configuration | Training | Validation / test |
| --- | --- | --- |
| [`rnd_dst_then_random`](configs/neg_sampler/rnd_dst_then_random.yaml) | Random, all nodes | Random, all nodes |
| [`rnd_dst_then_fixed`](configs/neg_sampler/rnd_dst_then_fixed.yaml) | Random, destination partition | Precomputed destination negatives |
| [`rnd_dst_then_fixed_all`](configs/neg_sampler/rnd_dst_then_fixed_all.yaml) | Random, destination partition | Precomputed all-node negatives |
| [`rnd_dst_then_hist`](configs/neg_sampler/rnd_dst_then_hist.yaml) | Random, destination partition | Previously observed destinations for each source |
| [`rnd_dst_then_post_train`](configs/neg_sampler/rnd_dst_then_post_train.yaml) | Random, destination partition | Destinations observed for each source after the training cutoff |
| [`rnd_dst_then_tgb`](configs/neg_sampler/rnd_dst_then_tgb.yaml) | Random, destination partition | Automatic official layout: Wiki 999 (masked), Review 100, Coin/Comment/Flight 20 |

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

**TPNet (dedicated trainer).** [TPNet](src/frost/tpnet/) is included with its own
single-/multi-GPU trainer. It combines an MLP-Mixer encoder with temporal-walk
random projections. Its globally decayed projection state uses a dedicated update
path rather than the native memory/mailbox routing and pipeline; launch it with
`-m frost.tpnet.train` instead of `model=tpnet`. See the [source notice](src/frost/tpnet/NOTICE.md)
for upstream versions and licensing.

## 📦 Install

The release wheel targets Linux x86-64 (glibc 2.31+), Python 3.12 and
PyTorch 2.9.0 with CUDA 12.8. GPU training requires a compatible NVIDIA driver.
Download the wheel from [GitHub Releases](https://github.com/dttutty/frost_dev/releases)
(repository access required), then install it in your Python environment:

```bash
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ./frost-0.1.0-cp312-cp312-manylinux_2_31_x86_64.whl
```

Set `FROST_DATA_ROOT` and `TGB_DATA_ROOT` as environment variables or in a
`.env.local` file in your working directory. Download the
[prepared datasets](https://huggingface.co/datasets/dttutty/frost_dataset)
or use `frost-data --help` for data operations.

For development from a checkout: `uv sync --locked --python 3.12`, then
`cp .env.example .env.local` and edit the dataset paths.

## 🚀 Usage

Run two GPU workers and one host rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=3 -m frost \
  model=tgn dataset=LASTFM
```

For one GPU worker, select one GPU and use `--nproc_per_node=2`.
Override parameters with `key=value`; see [defaults](configs/config.yaml)
and [model configs](configs/model/). Preview the resolved configuration with
`python -m frost --cfg job model=tgn`.

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

```bash
uv run ruff check .
uv run mypy
uv run pytest tests/ -q
```

The coverage badge measures Python lines exercised by CPU tests; C++ and GPU
execution are not measured. GitHub Actions updates it after successful main-branch
tests and retains the full report as an artifact.

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
