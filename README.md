# FROST

![Forward-looking Routing and Orchestrated Synchronization for TGNN](assets/frost-banner.png)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.9.0](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](uv.lock)
[![Linux x86-64](https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black)](#-install)
[![CPU coverage](https://img.shields.io/badge/CPU%20coverage-70.8%25-yellowgreen)](https://github.com/dttutty/frost_dev/actions/workflows/coverage.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Use GNNs to predict interactions in your data.

Train from an event CSV, save a model, and score candidate interactions.

- **Predict your data:** follow this README; a TGN starter configuration is included.
- **Tune or develop models:** use the [model and training guide](README.models.md)
  for hyperparameters, model/sampler support and multi-GPU training.

## 📦 Install

Python 3.12 and Linux x86-64 are the packaged target. Install the **0.3.0 or newer**
wheel from [private GitHub Releases](https://github.com/dttutty/frost_dev/releases)
(repository access required):

```bash
python -m pip install ./frost_gnn-0.3.0-cp312-cp312-manylinux_2_31_x86_64.whl
```

When upgrading from the FROST 0.1.0 wheel, first remove the old distribution
with `python -m pip uninstall frost` to avoid overlapping package files.

The distribution is named `frost_gnn` (Python imports remain `frost`). It is **not
published on PyPI**: bare `pip install frost_gnn` is not available yet.
For a CPU-only PyTorch installation, install
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

Train with the bundled TGN configuration and score candidates in one command:

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

## 🗂️ Data and query rules

- Use `src,dst` with optional `time`. Aliases `src_id,dst_id,timestamp` and `u,i,ts`
  are recognized; override names with `--src-col`, `--dst-col`, `--time-col`.
  These column settings also apply to test CSVs.
- IDs are strings, including leading zeros. Source and destination share one
  namespace; prefix IDs when different node types reuse the same numbers.
  Query nodes must have appeared in training.
- Times can be numeric (use a consistent unit) or ISO-8601 dates (converted to UTC
  seconds). Events are sorted by time. Without timestamps, rows become time
  1, 2, …, representing sequence order rather than measured elapsed time.
- Queries use the latest training time unless a test row supplies a time or you
  pass `--time`. That flag overrides every test row. Test rows never update the
  saved graph or memory, so query order does not change subsequent answers.
- Queries before the training cutoff are rejected. To backtest, train only on
  earlier events; predictions at the cutoff are not held-out accuracy.
- Existing output files are protected. Use `--output another-model.pt` or a new
  `--predictions` path for another run.

The starter runs TGN on CPU with four threads. Select a GPU with `--device cuda:0`.
Use `frost train --show-config` to inspect settings, or see the
[configuration guide](README.models.md#-tune-the-csv-trainer) to adjust them.
For all command options, run `frost train --help` or `frost predict --help`.

## 🧩 Node features

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
