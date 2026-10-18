import logging
import pickle
import re
from pathlib import Path

import numpy as np
import torch

from frost.data.paths import (
    is_raw_tgb_dataset,
    resolve_artifact_dataset_dir,
    resolve_tgb_raw_dataset_dir,
)

logger = logging.getLogger(__name__)


class TGBEvalNegSampler:
    """Load official TGB negatives into a fixed-capacity runtime layout.

    Strict mode requires every query to have exactly ``n_neg`` destinations.
    ``allow_variable_width`` permits shorter queries by padding with ``-1`` and
    exposing a validity mask; queries wider than the configured capacity still
    fail. Missing benchmark queries never fall back to random negatives.
    """

    def __init__(
        self,
        n_nodes: int,
        n_neg: int,
        dataset: str,
        data_root: str | Path | None = None,
        seed: int | None = None,
        allow_variable_width: bool = False,
    ) -> None:
        del seed  # Accepted so nested train/eval sampler configs share one interface.
        if n_neg <= 0:
            raise ValueError(f"n_neg must be positive, got {n_neg}")
        self.n_nodes = n_nodes
        self.n_neg = n_neg
        self.dataset = dataset
        self.allow_variable_width = allow_variable_width

        dataset_dir = (
            resolve_tgb_raw_dataset_dir(self.dataset)
            if data_root is None and is_raw_tgb_dataset(self.dataset)
            else resolve_artifact_dataset_dir(self.dataset, data_root=data_root)
        )

        # Load val and test dictionaries into a unified evaluation set
        # Schema: (src, dst, ts) -> numpy.ndarray of neg_dsts
        self.evaluation_dict: dict[tuple[int, int, int | float], np.ndarray] = {}

        base_names = self._candidate_base_names(dataset)
        requested_version = self._requested_version(dataset)

        for split in ["val", "test"]:
            pkl_path = self._resolve_negative_file(
                dataset_dir,
                base_names,
                split,
                requested_version=requested_version,
            )
            logger.info("TGBEvalNegSampler: loading %s", pkl_path)
            with pkl_path.open("rb") as f:
                eval_set = pickle.load(f)
            if not isinstance(eval_set, dict):
                raise TypeError(
                    f"Expected a dictionary in {pkl_path}, got {type(eval_set)!r}."
                )
            self.evaluation_dict.update(eval_set)
            logger.info("Loaded %d %s query records.", len(eval_set), split)

        self._batch_src: np.ndarray | None = None
        self._batch_dst: np.ndarray | None = None
        self._batch_ts: np.ndarray | None = None
        self._last_negative_mask: torch.Tensor | None = None

    def init_from_df(self, df: object) -> None:
        """Official negative dictionaries are initialized from pickle files."""
        del df

    def reset(self) -> None:
        """Clear per-batch lookup state between evaluation splits."""
        self._batch_src = None
        self._batch_dst = None
        self._batch_ts = None
        self._last_negative_mask = None

    @staticmethod
    def _candidate_base_names(dataset: str) -> list[str]:
        normalized = dataset.strip().lower().replace("_", "-")
        normalized = re.sub(r"-v\d+$", "", normalized)
        candidates = [normalized]
        if not normalized.startswith("tgbl-"):
            candidates.append(f"tgbl-{normalized}")
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _requested_version(dataset: str) -> int | None:
        normalized = dataset.strip().lower().replace("_", "-")
        match = re.search(r"-v(\d+)$", normalized)
        return int(match.group(1)) if match else None

    @staticmethod
    def _version_number(path: Path) -> int:
        match = re.search(r"_v(\d+)\.pkl$", path.name)
        return int(match.group(1)) if match else -1

    @classmethod
    def _resolve_negative_file(
        cls,
        dataset_dir: Path,
        base_names: list[str],
        split: str,
        requested_version: int | None = None,
    ) -> Path:
        checked: list[Path] = []
        for base_name in base_names:
            if requested_version is not None:
                exact = dataset_dir / f"{base_name}_{split}_ns_v{requested_version}.pkl"
                checked.append(exact)
                if exact.is_file():
                    return exact
                continue
            versioned = sorted(
                dataset_dir.glob(f"{base_name}_{split}_ns_v*.pkl"),
                key=cls._version_number,
                reverse=True,
            )
            unversioned = dataset_dir / f"{base_name}_{split}_ns.pkl"
            checked.extend(versioned)
            checked.append(unversioned)
            for path in [*versioned, unversioned]:
                if path.is_file():
                    return path
        raise FileNotFoundError(
            f"Could not find official TGB negatives for split={split!r} in "
            f"{dataset_dir}. Checked: {[str(path) for path in checked]}"
        )

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
        self._last_negative_mask = None
        if batch_ts is None:
            raise ValueError(
                "TGBEvalNegSampler requires batch_ts array to lookup exact timestamps. "
                "Ensure host engine is updated to provide it."
            )
        self._batch_ts = batch_ts

    def sample(self, src_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        n_src = src_ids.numel()
        if edge_ids.numel() != n_src:
            raise ValueError(
                "src_ids and edge_ids must have the same length, got "
                f"{n_src} and {edge_ids.numel()}."
            )
        if n_src == 0:
            self._last_negative_mask = torch.empty(0, dtype=torch.bool)
            return torch.empty(0, dtype=torch.long)

        if (
            self._batch_src is None
            or self._batch_dst is None
            or self._batch_ts is None
            or len(self._batch_src) != n_src
            or len(self._batch_dst) != n_src
            or len(self._batch_ts) != n_src
        ):
            raise RuntimeError(
                "TGBEvalNegSampler: set_batch_context must be called with exact arrays "
                "before sample."
            )

        result = torch.full((n_src, self.n_neg), -1, dtype=torch.long)
        valid_mask = torch.zeros((n_src, self.n_neg), dtype=torch.bool)

        for i in range(n_src):
            s = int(self._batch_src[i])
            d = int(self._batch_dst[i])
            t = self._batch_ts[i].item()

            # The timestamp in TGB pickle might be float or int depending on generation.
            # Try the integer representation first, then the float representation.
            key_f = (s, d, float(t))
            key_i = (s, d, int(t))

            neg_arr = None
            if key_i in self.evaluation_dict:
                neg_arr = self.evaluation_dict[key_i]
            elif key_f in self.evaluation_dict:
                neg_arr = self.evaluation_dict[key_f]

            if neg_arr is not None:
                neg_arr = np.asarray(neg_arr, dtype=np.int64).reshape(-1)
                available = int(neg_arr.shape[0])
                if not self.allow_variable_width and available != self.n_neg:
                    raise ValueError(
                        "FROST currently supports fixed-width TGB negatives only: "
                        f"configured n_neg={self.n_neg}, but query {(s, d, t)} has "
                        f"{available}. Variable-width candidate sets are not supported."
                    )
                if available > self.n_neg:
                    raise ValueError(
                        f"TGB query {(s, d, t)} has {available} negatives, which "
                        f"exceeds configured capacity n_neg={self.n_neg}."
                    )
                result[i, :available] = torch.from_numpy(neg_arr)
                valid_mask[i, :available] = True
            else:
                raise KeyError(
                    "Official TGB negatives are missing for query "
                    f"(src={s}, dst={d}, timestamp={t}). Refusing to use random "
                    "fallback because it would invalidate benchmark MRR."
                )

        # Row-major layout: all negatives for edge 0, then edge 1, etc.
        self._last_negative_mask = valid_mask.reshape(-1)
        return result.reshape(-1)

    def get_last_negative_mask(self) -> torch.Tensor | None:
        """Return the validity mask produced by the immediately preceding sample."""
        if self._last_negative_mask is None:
            raise RuntimeError(
                "sample() must be called before reading its negative mask."
            )
        return self._last_negative_mask
