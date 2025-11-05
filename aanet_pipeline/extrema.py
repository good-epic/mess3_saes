from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


def _import_utils():
    try:
        from AAnet.AAnet_torch import utils as aanet_utils  # type: ignore
        return aanet_utils
    except ImportError:
        import sys

        module_root = Path(__file__).resolve().parents[1] / "AAnet"
        if module_root.exists() and str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
        from AAnet_torch import utils as aanet_utils  # type: ignore

        return aanet_utils


aanet_utils = _import_utils()


@dataclass(frozen=True)
class ExtremaConfig:
    enabled: bool = True
    knn: int = 10
    subsample: bool = True
    max_points: Optional[int] = None
    random_seed: int = 0


def _select_points(data: np.ndarray, max_points: Optional[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or data.shape[0] <= max_points:
        indices = np.arange(data.shape[0], dtype=np.int64)
        return data, indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(data.shape[0], size=max_points, replace=False)
    return data[selected], selected


def compute_diffusion_extrema(
    data: np.ndarray,
    *,
    max_k: int,
    config: ExtremaConfig,
) -> Optional[torch.Tensor]:
    if not config.enabled or data.shape[0] == 0 or max_k <= 0:
        return None
    trimmed, index_map = _select_points(data, config.max_points, config.random_seed)
    if trimmed.shape[0] < max_k:
        return None
    extrema_indices = aanet_utils.get_laplacian_extrema(
        trimmed,
        n_extrema=max_k,
        knn=config.knn,
        subsample=config.subsample,
    )
    extrema_indices = np.asarray(extrema_indices, dtype=np.int64)
    selected = trimmed[extrema_indices]
    return torch.tensor(selected, dtype=torch.float32)
