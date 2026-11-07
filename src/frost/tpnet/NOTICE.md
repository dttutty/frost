# TPNet source provenance

This package contains FROST's TPNet implementation and dedicated multi-GPU
trainer, launched with `torchrun -m frost.tpnet.train`. It is independent of
external repository checkouts and uses its own state-update path.

Sources pinned for this extraction:

- [lxd99/TPNet](https://github.com/lxd99/TPNet/tree/7dd0cf49695a581c5c541baeda47c1b5e98e8748),
  commit `7dd0cf49695a581c5c541baeda47c1b5e98e8748`.
  Copyright (c) 2024 Xiaodong Lu; its MIT license is reproduced in `LICENSE`.
- [lxd99/TGB_TPNet](https://github.com/lxd99/TGB_TPNet/tree/86d84e26eda708443fb74ea48c4846edd294a296),
  commit `86d84e26eda708443fb74ea48c4846edd294a296`.
  This repository identifies itself as an adaptation of TPNet. It has no
  separate LICENSE file at that revision; the included MIT text is from TPNet.

## Extracted code

- `model.py`: `models/TPNet.py`, using the TGB variant's explicit output width
  and optional embedding path. Omitting `output_dim` selects the classic
  variant's node-feature width. RP reset defaults, parameter names, initialization
  order, batched state updates, and upstream padding behavior are preserved.
- `layers.py`: only `TimeEncoder` and `LinkPredictor_v1` from `models/modules.py`;
  these definitions are identical in both source revisions.
- `sampling.py`: shared `NeighborSampler` and `get_neighbor_sampler` from
  `utils/utils.py`; classic negative sampling plus the TGB variant's overrides.
  Even unused random draws are retained to preserve RNG progression.
- `data.py`: classic `utils/DataLoader.py`, including the common event container
  and index loader. The data root is explicit. Sorted population sampling
  replaces the Python 3.12 compatibility monkeypatch used by the old wrapper.
- `config.py`: TPNet options and best-config overrides extracted from both
  `utils/load_configs.py` files. Options specific to other models are omitted.
- `training.py`: seed, thread, and optimizer helpers from `utils/utils.py`.

Imports, formatting, type annotations, and invalid-input checks were adapted
for this package. Unused model implementations, official TGB loader imports,
numba utilities, result logging, and upstream training scripts are not included.
TGB data adaptation, official negatives, MRR, DDP synchronization, and evaluation
are implemented in `train.py`.

CPU regression fixtures in `tests/fixtures/tpnet/` were produced by the pinned
upstream checkouts. They cover logits, gradient/parameter summaries, complete
RP state, reset/restore, temporal neighbors, and negative sampling. During the
extraction, full gradients and parameters were also compared after each update.
GPU DDP training is not covered by those CPU fixtures.
