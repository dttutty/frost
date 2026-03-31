import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from frost.neg_sampler._utils import resolve_dataset_dir

logger = logging.getLogger(__name__)


class TGBEvalNegSampler:
    """TGB evaluation sampler loading precomputed `{dataset}_[val|test]_ns.pkl` files.
    
    TGB link property prediction benchmark generates 100 negative examples for each
    positive test/val edge. This sampler loads the precomputed dictionary and returns
    the pre-designated negative destinations via exact (src, dst, timestamp) matching.
    """

    def __init__(
        self,
        n_nodes: int,
        n_neg: int,
        dataset: str,
        data_root: str | Path = "DATA",
    ) -> None:
        self.n_nodes = n_nodes
        self.n_neg = n_neg
        self.dataset = dataset

        dataset_dir = resolve_dataset_dir(self.dataset, data_root=data_root)

        # Load val and test dictionaries into a unified evaluation set
        # Schema: (src, dst, ts) -> numpy.ndarray of neg_dsts
        self.evaluation_dict: dict[tuple[int, int, int | float], np.ndarray] = {}
        
        # Datasets often have prefixes or lowercase names in TGB e.g. tgbl-wiki
        base_names = [dataset, dataset.lower()]
        if not dataset.startswith("tgbl-"):
            base_names.append(f"tgbl-{dataset.lower()}")
            
        for split in ["val", "test"]:
            file_loaded = False
            for bname in base_names:
                pkl_path = dataset_dir / f"{bname}_{split}_ns.pkl"
                if pkl_path.exists():
                    logger.info(f"TGBEvalNegSampler: Loading {pkl_path}")
                    with open(pkl_path, "rb") as f:
                        eval_set = pickle.load(f)
                        self.evaluation_dict.update(eval_set)
                    logger.info(f"Loaded {len(eval_set)} {split} query records.")
                    file_loaded = True
                    break
            
            if not file_loaded:
                logger.warning(
                    f"TGB eval file for '{split}' not found in {dataset_dir} for names {base_names}. "
                    "TGB evaluation will fallback to random if queries miss."
                )

        self._batch_src: np.ndarray | None = None
        self._batch_dst: np.ndarray | None = None
        self._batch_ts: np.ndarray | None = None

    def set_batch_context(
        self,
        *,
        batch_src: np.ndarray,
        batch_dst: np.ndarray,
        batch_start_time: float,
        batch_end_time: float,
        batch_ts: np.ndarray | None = None,
    ) -> None:
        """Capture batch specifics for exact Dict lookup."""
        self._batch_src = batch_src
        self._batch_dst = batch_dst
        if batch_ts is None:
            raise ValueError(
                "TGBEvalNegSampler requires batch_ts array to lookup exact timestamps. "
                "Ensure host engine is updated to provide it."
            )
        self._batch_ts = batch_ts

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        n_src = src_ids.numel()
        if n_src == 0:
            return torch.empty(0, dtype=torch.long)

        if self._batch_src is None or self._batch_ts is None or len(self._batch_src) != n_src:
            raise RuntimeError(
                "TGBEvalNegSampler: set_batch_context must be called with exact arrays "
                "before sample."
            )

        result = torch.empty((n_src, self.n_neg), dtype=torch.long)
        misses = 0

        for i in range(n_src):
            s = int(self._batch_src[i])
            d = int(self._batch_dst[i])
            t = self._batch_ts[i].item()

            # The timestamp in TGB pickle might be float or int depending on generation.
            # Try float first, then int.
            key_f = (s, d, float(t))
            key_i = (s, d, int(t))
            
            neg_arr = None
            if key_i in self.evaluation_dict:
                neg_arr = self.evaluation_dict[key_i]
            elif key_f in self.evaluation_dict:
                neg_arr = self.evaluation_dict[key_f]
                
            if neg_arr is not None:
                available = len(neg_arr)
                if available < self.n_neg:
                    # Pad if the precomputed set has fewer negatives than requested
                    padded = np.pad(neg_arr, (0, self.n_neg - available), mode="edge")
                    result[i] = torch.from_numpy(padded)
                else:
                    result[i] = torch.from_numpy(neg_arr[: self.n_neg])
            else:
                misses += 1
                # Fallback: random fallback
                result[i] = torch.randint(0, self.n_nodes, (self.n_neg,))

        if misses > 0:
            logger.warning(
                f"TGBEvalNegSampler: {misses}/{n_src} queries missed. "
                f"Key (src, dst, ts) not found in .pkl! Used random fallback. Example key: (s={s}, d={d}, t={t})"
            )

        return result.view(-1)
