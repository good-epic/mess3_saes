from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA


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


@dataclass(frozen=True, kw_only=True)
class ExtremaConfig:
    enabled: bool = True
    knn: int = 10
    subsample: bool = True
    max_points: Optional[int] = None
    pca_components: Optional[float | int] = None
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
    # Apply PCA if requested
    data_for_graph = trimmed
    if config.pca_components is not None:
        try:
            # Handle float vs int for PCA
            n_comp = config.pca_components
            if isinstance(n_comp, float) and n_comp >= 1.0:
                n_comp = int(n_comp)
            
            # Only run PCA if n_components < n_features
            if isinstance(n_comp, (int, float)) and (isinstance(n_comp, float) or n_comp < trimmed.shape[1]):
                print(f"Applying PCA (n={n_comp}) to {trimmed.shape[0]} points...")
                pca = PCA(n_components=n_comp, random_state=config.random_seed)
                data_for_graph = pca.fit_transform(trimmed)
                print(f"PCA reduced dim from {trimmed.shape[1]} to {data_for_graph.shape[1]}")
        except Exception as e:
            print(f"PCA failed: {e}. Using original data.")
            data_for_graph = trimmed

    extrema_indices = aanet_utils.get_laplacian_extrema(
        data_for_graph,
        n_extrema=max_k,
        knn=config.knn,
        subsample=False, # We already subsampled in _select_points
    )
    extrema_indices = np.asarray(extrema_indices, dtype=np.int64)
    selected = trimmed[extrema_indices]
    return torch.tensor(selected, dtype=torch.float32)
