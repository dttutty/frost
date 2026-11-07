# 🔎 CSV prediction guide

[← Quick start](../README.md)

## 📦 Installation options

Install from [PyPI](https://pypi.org/project/frost-gnn/) or download a wheel from
[GitHub Releases](https://github.com/dttutty/frost/releases).
The distribution is `frost-gnn`; the Python import and CLI command are `frost`.
The CSV trainer runs on CPU unless you select a GPU. For CPU-only PyTorch,
install `torch==2.9.0` from `https://download.pytorch.org/whl/cpu` first.
CUDA execution requires a compatible NVIDIA driver.

When upgrading from the FROST 0.1.0 wheel, first run
`python -m pip uninstall frost` to avoid overlapping package files.

## 🗂️ Data and query rules

- Use `src,dst` with optional `time`. Aliases `src_id,dst_id,timestamp` and `u,i,ts`
  are recognized; override names with `--src-col`, `--dst-col`, `--time-col`.
  These column settings also apply to test CSVs.
- IDs are strings, including leading zeros. Source and destination share one
  namespace; use prefixes such as `user:42` and `item:42` when users and items
  reuse the same numbers. Query users and items must have appeared in training.
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
[configuration guide](model_training.md#-tune-the-csv-trainer) to adjust them.
For all command options, run `frost train --help` or `frost predict --help`.

## 🔮 Queries and outputs

Reuse the checkpoint without retraining, choose a query time, or score a test CSV:

```bash
frost predict frost-model.pt --src 42 --candidates 46638,70286 --time 10
frost predict frost-model.pt --test test.csv --predictions scores.csv
frost train events.csv --output model.pt --test test.csv --predictions scores.csv
```

The quick-start training command saves `frost-model.pt` and prints CSV columns
`src,dst,time,score` in candidate order. Training progress goes to stderr. Each score is in [0, 1]; it is a
**ranking score, not a calibrated probability** or a probability over a specified
future time window. The quick-start CSV illustrates the format; useful predictions
require representative training data and evaluation on held-out events.

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
