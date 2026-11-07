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
Download the [prepared datasets](https://huggingface.co/datasets/dttutty/frost_dataset)
or use `uv run frost-data --help` for data operations.

## Usage

Run two GPU workers and one host rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=3 run.py \
  model=tgn dataset=LASTFM
```

For one GPU worker, select one GPU and use `--nproc_per_node=2`.
Override parameters with `key=value`; see [defaults](configs/config.yaml)
and [model configs](configs/model/). Preview the resolved configuration with
`uv run python run.py --cfg job model=tgn`.

## Development

```bash
uv run ruff check .
uv run mypy
uv run pytest tests/ -q
```

Code is in [src/frost](src/frost/), tests in [tests](tests/).
Rebuild C++ sampler changes with `uv pip install -e .`.

## Citation

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
