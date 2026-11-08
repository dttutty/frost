# FROST (**F**orward-looking **R**outing and **O**rchestrated **S**ynchronization for **T**GNN)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.9.0](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](uv.lock)
[![Linux x86-64](https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black)](#-install)
[![CPU coverage](https://img.shields.io/badge/CPU%20coverage-70.8%25-yellowgreen)](https://github.com/dttutty/frost/actions/workflows/coverage.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Temporal(Dynamic) GNN Training framework for recommendation systems, link prediction
and self-supervised embedding learning.

## 📦 Install

Python 3.12 · Linux x86-64 (glibc 2.31+)

```bash
python -m pip install frost-gnn
```

## 🚀 Score candidate items

Create `events.csv`: `src` is the user ID, `dst` is the item ID, and `time` is optional.

```csv
src,dst,time
42,46638,1
7,70286,2
42,38083,3
7,95034,4
```

Train TGN and score candidate items for user `42`:

```bash
frost train events.csv --src 42 --candidates 46638,70286,38083,95034
```

Saves `frost-model.pt` and prints `src,dst,time,score`. Scores are in [0, 1]; each is
**a ranking score, not a calibrated probability**. Query nodes must have appeared
in training. Query time is the latest training time unless you pass `--time`.

Reuse the model at a chosen time:

```bash
frost predict frost-model.pt --src 42 --candidates 46638,70286 --time 10
```

Training uses CPU; add `--device cuda:0` to use a GPU.

## 🧩 Node features

Pass a numeric NPY matrix, or convert tabular attributes (`n_id,age,category,...`):

```bash
frost features nodes.csv --id-col n_id --output nodes.npy
frost train events.csv --node-features nodes.npy --output featured-model.pt
```

NPY row `n` maps to node ID `n`, or use `--feature-ids ids.json` for other IDs.
The converter writes this mapping automatically.

## 📖 Guides

- [CSV prediction](docs/prediction.md): data rules, test CSVs, feature mapping and installation options.
- [Model training](docs/model_training.md): configuration, support matrix and multi-GPU training.

## 📚 Citation

Please cite [our paper](https://doi.org/10.1145/3856345). Metadata: [CITATION.cff](CITATION.cff).

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
