# FROST

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.9.0](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](uv.lock)
[![Linux x86-64](https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black)](#install)

A Python library for temporal graph neural networks and multi-GPU training.

## Install

GPU training requires an NVIDIA GPU with a compatible driver.

```bash
uv sync --locked --python 3.12
cp .env.example .env.local
```

Set `FROST_DATA_ROOT` and `TGB_DATA_ROOT` in `.env.local`.
Prepared data is available in the
[dataset bundle](https://huggingface.co/datasets/dttutty/frost_dataset);
`uv run frost-data --help` lists data operations.

## Use the library

Inspect the configuration without training:

```bash
uv run python run.py --cfg job model=tgn
```

After [GPU preflight](#gpu-experiment-preflight), run two GPU workers and one
host rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=3 run.py \
  model=tgn dataset=LASTFM
```

For one GPU worker, select one GPU and use `--nproc_per_node=2`.
`runtime.single_gpu_resident=on` optionally keeps features and dynamic state
on that worker's GPU.

## Configuration

Use Hydra `key=value` overrides. See [shared defaults](configs/config.yaml)
and [model configs](configs/model/) for available settings.
`macro_batch_size` is the total batch across GPUs; `eval_macro_batch_size`
controls evaluation separately and defaults to the training batch size.

`model.embedding_dim` sets embedding/memory width, and
`model.neighborhood.fanout` sets sampled neighbors per hop. Memory models
support at most one hop. Keep model-specific sequence and token settings
consistent with fanouts; capabilities are declared by
[model recipes](src/frost/models/recipes.py).

## Negative sampling

FROST corrupts destinations while keeping the positive source fixed. The default
`rnd_dst_then_random` samples all node IDs with replacement, without filtering
positive-destination collisions. AP uses one evaluation negative per positive.

Fixed, historical, post-training and TGB options are in
[negative-sampler configs](configs/neg_sampler/) and their
[implementations](src/frost/neg_sampler/). For comparisons, match splits,
candidate sets, negative counts, sampling rules, seeds and metrics.

## TGB link prediction

Fetch tgbl-wiki-v2 into `$TGB_DATA_ROOT/raw/tgbl_wiki`, then train with official
evaluation candidates and MRR:

```bash
uv run frost-data fetch-tgb tgbl-wiki
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=3 run.py model=tgn \
  dataset=TGBL_WIKI \
  neg_sampler=rnd_dst_then_tgb_wiki eval_metric=mrr \
  macro_batch_size=4096 eval_macro_batch_size=128
```

Training uses one random negative; wiki evaluation handles the official
998/999-negative queries. For generic TGB, set `neg_sampler.eval.n_neg=100`
for `tgbl-review`, or 20 for `tgbl-coin`, `tgbl-comment` and `tgbl-flight`.
Use a smaller evaluation batch to limit memory use.

## Development

```bash
uv run ruff check .
uv run mypy
uv run pytest tests/ -q
```

CPU tests do not require a GPU. Rebuild C++ sampler changes with
`uv pip install -e .`. Library code is in [src/frost](src/frost/), with
regression tests in [tests](tests/). Models own computation and contracts;
queues, transfers and synchronization belong to runtime.

## GPU experiment preflight

Check GPU availability and health, follow your cluster's scheduling policy,
and verify dependencies with `uv sync --locked --check`. Start with a short
smoke run; on distributed failures, inspect the earliest rank traceback.
Record the code revision, data, GPU mapping, command, logs and exit status.

## Citation

If you use FROST in your research, please cite:

Lei Zhao, Joseph Nke, Jongryool Kim, and Chul-Ho Lee.
[**FROST: Accelerating Temporal Graph Neural Network Training via a Unified Multi-GPU Framework**](https://doi.org/10.1145/3856345).
*Proc. ACM Manag. Data*, Vol. 4, No. 6 (SIGMOD), Article 350, December 2026.

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

The paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
