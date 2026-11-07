# FROST

![Forward-looking Routing and Orchestrated Synchronization for TGNN](assets/frost-banner.png)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.9.0](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](uv.lock)
[![Linux x86-64](https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black)](#-install)
[![CPU coverage](https://img.shields.io/badge/CPU%20coverage-69.6%25-yellowgreen)](https://github.com/dttutty/frost_dev/actions/workflows/coverage.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Use GNNs to predict interactions in your data.

Train from an event CSV, save a model, and rank candidate interactions. FROST also
provides a multi-GPU framework for Temporal (Dynamic) Graph Neural Network training.

## 📦 Install

Python 3.12 and Linux x86-64 are the packaged target. Install the **0.2.0 or newer**
wheel from [private GitHub Releases](https://github.com/dttutty/frost_dev/releases)
(repository access required):

```bash
python -m pip install ./frost_gnn-0.2.0-cp312-cp312-manylinux_2_31_x86_64.whl
```

When upgrading from the FROST 0.1.0 wheel, first remove the old distribution
with `python -m pip uninstall frost` to avoid overlapping package files.

The distribution is named `frost_gnn` (Python imports remain `frost`). It is **not
published on PyPI**: bare `pip install frost_gnn` is not available yet. From a
checkout, `python -m pip install .` builds it locally with a C++17 compiler,
CMake and OpenMP. For a CPU-only PyTorch installation, install
`torch==2.9.0` from `https://download.pytorch.org/whl/cpu` first.

## 🚀 Predict your data

An `events.csv` needs source and destination IDs; time is optional:

```csv
src,dst,time
42,121,1
7,12312,2
42,31231,3
7,12331,4
```

Train and score candidates in one command:

```bash
frost train events.csv --src 42 --candidates 121,12312,31231,12331
```

This saves `frost-model.pt` and prints CSV columns `src,dst,time,score` in candidate
order. Training progress goes to stderr. Each score is in [0, 1]; it is a
**ranking score, not a calibrated probability** or a probability over a specified
future time window. The four rows above illustrate the format; useful predictions
require representative training data and evaluation on held-out events.

Reuse the checkpoint without retraining, choose a query time, or score a test CSV:

```bash
frost predict frost-model.pt --src 42 --candidates 121,12312 --time 10
frost predict frost-model.pt --test test.csv --predictions scores.csv
frost train events.csv --output model.pt --test test.csv --predictions scores.csv
```

- IDs are strings; leading zeros are preserved. Source and destination IDs share
  one namespace. Prefix IDs if two distinct node types reuse the same numbers.
- Column aliases `src_id,dst_id,timestamp` and `u,i,ts` are recognized. Other names
  can be selected with `--src-col`, `--dst-col`, `--time-col` (also applied to test CSVs).
- Numeric times must use a consistent unit; ISO-8601 dates are converted to UTC
  seconds. Events are stably sorted by time. Without times, row order becomes
  time 1, 2, …; this represents sequence order, not measured elapsed time.
- Queries use the latest training timestamp unless `--time` or test-row times
  are supplied. `--time` overrides every test row. A test CSV without times uses
  the training cutoff for every row; it never updates the saved graph.
- Training samples strictly earlier neighbors; prediction uses all observed
  training events up to the query time. Earlier-than-training-cutoff queries are
  rejected: for backtesting, train a separate model only on earlier events.
  Predictions at the cutoff describe the fitted history, not held-out accuracy.
- The simple trainer uses one-hop TGAT with learned node embeddings, bidirectional
  historical neighborhoods, and a directional link head. It samples one random
  destination negative per positive, excluding the source and simultaneous
  positives. Nodes must have appeared in training; unknown IDs are rejected.
- Runs on CPU with four threads; select a GPU explicitly with `--device cuda:0`.
  Tune `--epochs`, `--batch-size`, `--fanout`, `--hidden`, `--lr`, `--seed` and
  `--threads`. Existing output files are protected; choose new paths for new runs.
  See `frost train --help`. The multi-GPU recipes below use the advanced entry point.

### 🧩 Node features

```bash
frost train events.csv --node-features nodes.npy --output featured-model.pt
```

A numeric NPY matrix has shape `[nodes, feature_dim]`; row `n` corresponds to
nonnegative integer node ID `n`. For string IDs or compactly stored sparse IDs,
pass `--feature-ids ids.json`, a JSON list in NPY row order. An adjacent
`nodes.npy.ids.json` is detected automatically. Every training node needs a row;
features and the ID mapping are saved in the model checkpoint.

For tabular attributes, use the included converter:

```bash
# nodes.csv: n_id,age,category,... (one row per node)
frost features nodes.csv --id-col n_id --output nodes.npy
```

Numeric columns are standardized, with missing values filled by the observed
mean. Categorical values are hashed into 64 columns (`--categorical-dim` adjusts
this); collisions are possible. The converter writes the NPY, ID list and a schema
recording column types and numeric statistics. Use attributes available at the
training cutoff. Text and images need semantic embeddings from a suitable encoder;
this converter treats nonnumeric strings as categories, not language meaning.

## 📋 Support matrix

### 🗃️ Datasets

| Data source | Supported datasets |
| --- | --- |
| [🤗 frost_dataset](https://huggingface.co/datasets/dttutty/frost_dataset) | Wikipedia (`WIKI`), Reddit, MOOC, LastFM, MAG, GDELT |
| TGB link prediction | `TGBL_WIKI` (v2), `TGBL_REVIEW` (v2), `TGBL_COIN` (v2), `TGBL_COMMENT` (v1), `TGBL_FLIGHT` (v2) |
| Your data | `src,dst` event CSVs with optional timestamps and node features via `frost train` |

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

## ⚡ Advanced multi-GPU training

Set `FROST_DATA_ROOT` and `TGB_DATA_ROOT` in your environment or `.env.local`.
Use [prepared datasets](https://huggingface.co/datasets/dttutty/frost_dataset)
or `frost-data --help` to construct graph artifacts. Run two GPU workers and one host rank:

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

From a checkout, install with `uv sync --locked --python 3.12`.

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
Code is licensed under [MIT](LICENSE); TPNet retains its [upstream notices](src/frost/tpnet/NOTICE.md).
The paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
