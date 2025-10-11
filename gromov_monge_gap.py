"""Gromov-Monge Gap computation for K-simplex structure scoring.

This module provides functions to:
1. Project SAE decoder directions onto K-simplices
2. Compute the Gromov-Wasserstein distance for geometry preservation
3. Calculate the Gromov-Monge Gap (GMG) as a measure of simplex structure quality

The GMG quantifies how well a set of directions represents a K-simplex by comparing
the distortion induced by a projection to the theoretical minimum distortion.
"""

from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    import ot
    from ot.gromov import entropic_gromov_wasserstein
    from ot.bregman import sinkhorn_log
except ImportError:
    raise ImportError(
        "POT library required for GMG computation. Install with: pip install POT"
    )


def cosine_cost_matrix(
    vectors: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Compute pairwise cosine distance cost matrix.
    
    Args:
        vectors: Array of shape (n, d) containing n vectors of dimension d
        normalize: If True, normalize vectors to unit length first
        
    Returns:
        Cost matrix of shape (n, n) where C[i,j] = 1 - cos_sim(v_i, v_j)
    """
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        vectors = vectors / norms
    
    # Cosine similarity matrix
    cos_sim = vectors @ vectors.T
    
    # Clip to valid range to handle numerical errors
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Convert to distance (cost)
    return 1.0 - cos_sim


def euclidean_cost_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distance cost matrix.
    
    Args:
        vectors: Array of shape (n, d)
        
    Returns:
        Cost matrix of shape (n, n) where C[i,j] = ||v_i - v_j||^2
    """
    return ot.dist(vectors, metric='sqeuclidean')


def project_to_simplex_barycentric(
    vectors: np.ndarray,
    K: int,
    method: Literal['pca', 'lda'] = 'pca',
    labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """Project vectors to K-simplex via dimensionality reduction + normalization.
    
    This is a simple, closed-form projection that:
    1. Projects to K dimensions using PCA or LDA
    2. Shifts to ensure all coordinates are positive
    3. Normalizes so coordinates sum to 1 (barycentric coordinates)
    
    Args:
        vectors: Array of shape (n, d) containing n vectors
        K: Simplex dimension (K+1 vertices)
        method: 'pca' for PCA projection, 'lda' for LDA (requires labels)
        labels: Cluster labels for LDA, shape (n,)
        
    Returns:
        Projected vectors of shape (n, K+1) representing barycentric coordinates
    """
    n = len(vectors)
    n_components = min(K + 1, vectors.shape[1], n - 1)
    
    # Dimensionality reduction
    if method == 'lda' and labels is not None:
        n_classes = len(np.unique(labels))
        n_components = min(n_components, n_classes - 1)
        reducer = LinearDiscriminantAnalysis(n_components=n_components)
        projected = reducer.fit_transform(vectors, labels)
    else:
        reducer = PCA(n_components=n_components)
        projected = reducer.fit_transform(vectors)
    
    # Pad to K+1 dimensions if needed
    if projected.shape[1] < K + 1:
        padding = np.zeros((n, K + 1 - projected.shape[1]))
        projected = np.hstack([projected, padding])
    elif projected.shape[1] > K + 1:
        projected = projected[:, :K + 1]
    
    # Shift to ensure all coordinates are positive
    projected = projected - projected.min(axis=0, keepdims=True)
    projected = projected + 1e-8  # Small epsilon for numerical stability
    
    # Normalize to barycentric coordinates (sum to 1)
    projected = projected / projected.sum(axis=1, keepdims=True)
    
    return projected


def project_to_simplex_sinkhorn(
    vectors: np.ndarray,
    K: int,
    n_simplex_samples: int = 1000,
    reg: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """Project vectors to K-simplex via entropic optimal transport.
    
    Uses Sinkhorn algorithm to find optimal transport from vectors to
    uniformly sampled points on the K-simplex, then projects each vector
    to its transported barycenter.
    
    Args:
        vectors: Array of shape (n, d)
        K: Simplex dimension
        n_simplex_samples: Number of uniformly sampled simplex points
        reg: Entropic regularization parameter
        seed: Random seed for simplex sampling
        
    Returns:
        Projected vectors of shape (n, K+1)
    """
    n = len(vectors)
    rng = np.random.RandomState(seed)
    
    # Sample uniform points on K-simplex using Dirichlet distribution
    simplex_samples = rng.dirichlet(np.ones(K + 1), size=n_simplex_samples)
    
    # Source and target distributions (uniform)
    a = np.ones(n) / n
    b = np.ones(n_simplex_samples) / n_simplex_samples
    
    # Cost matrix (squared Euclidean distance in original space)
    # We use a simple heuristic: distance to simplex via PCA preprocessing
    pca = PCA(n_components=min(K + 1, vectors.shape[1]))
    vectors_lowdim = pca.fit_transform(vectors)
    
    # Pad if needed
    if vectors_lowdim.shape[1] < K + 1:
        padding = np.zeros((n, K + 1 - vectors_lowdim.shape[1]))
        vectors_lowdim = np.hstack([vectors_lowdim, padding])
    else:
        vectors_lowdim = vectors_lowdim[:, :K + 1]
    
    # Normalize to simplex
    vectors_lowdim = vectors_lowdim - vectors_lowdim.min()
    vectors_lowdim = vectors_lowdim / (vectors_lowdim.sum(axis=1, keepdims=True) + 1e-8)
    
    M = ot.dist(vectors_lowdim, simplex_samples, metric='sqeuclidean')
    
    # Solve entropic OT with log-stabilized Sinkhorn
    coupling = sinkhorn_log(a, b, M, reg=reg)
    
    # Project: weighted average of simplex points
    row_sums = coupling.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    projected = (coupling @ simplex_samples) / row_sums
    
    return projected


def compute_gromov_wasserstein_distance(
    C_source: np.ndarray,
    C_target: np.ndarray,
    p: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    epsilon: float = 0.1,
    solver: Literal['PPA', 'PGD'] = 'PPA',
    max_iter: int = 100,
    tol: float = 1e-9,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """Compute entropic Gromov-Wasserstein distance between two metric spaces.
    
    Args:
        C_source: Source cost matrix of shape (n, n)
        C_target: Target cost matrix of shape (n, n)
        p: Source distribution of shape (n,). If None, uniform distribution
        q: Target distribution of shape (n,). If None, uniform distribution
        epsilon: Entropic regularization parameter
        solver: 'PPA' (Proximal Point) or 'PGD' (Projected Gradient Descent)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        verbose: Print iteration information
        
    Returns:
        Tuple of (gw_distance, log_dict) where log_dict contains:
            - 'gw_dist': The GW distance
            - 'T': The optimal coupling (if log=True in implementation)
    """
    n = len(C_source)
    
    if p is None:
        p = np.ones(n) / n
    if q is None:
        q = np.ones(n) / n
    
    # Ensure proper normalization
    p = p / p.sum()
    q = q / q.sum()
    
    # Call POT's entropic GW solver
    gw_result = entropic_gromov_wasserstein(
        C_source,
        C_target,
        p,
        q,
        loss_fun='square_loss',
        epsilon=epsilon,
        solver=solver,
        max_iter=max_iter,
        tol_rel=tol,
        tol_abs=tol,
        log=True,
        verbose=verbose
    )
    
    # Result is (coupling_matrix, log_dict)
    T_optimal, log_dict = gw_result
    
    return log_dict['gw_dist'], log_dict


def compute_map_distortion(
    C_source: np.ndarray,
    C_target: np.ndarray
) -> float:
    """Compute distortion induced by a map between two metric spaces.
    
    The distortion measures the average squared difference between
    pairwise distances in source and target spaces.
    
    Args:
        C_source: Source cost matrix of shape (n, n)
        C_target: Target cost matrix of shape (n, n)
        
    Returns:
        Distortion value (lower = better preservation)
    """
    return np.mean((C_source - C_target) ** 2)


def compute_gromov_monge_gap(
    vectors: np.ndarray,
    K: int,
    cost_fn: Literal['cosine', 'euclidean'] = 'cosine',
    projection_method: Literal['barycentric', 'sinkhorn'] = 'barycentric',
    epsilon: float = 0.1,
    solver: Literal['PPA', 'PGD'] = 'PPA',
    normalize_vectors: bool = True,
    verbose: bool = False,
    **projection_kwargs
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Compute Gromov-Monge Gap for K-simplex structure scoring.
    
    The GMG measures how well a set of vectors represents a K-simplex by:
    1. Projecting vectors to a K-simplex
    2. Computing the distortion of this projection
    3. Finding the optimal (minimal) distortion via Gromov-Wasserstein
    4. Computing the gap: GMG = distortion - optimal_distortion
    
    A low GMG (near 0) indicates the projection is optimal, and low absolute
    distortion indicates good simplex structure.
    
    Args:
        vectors: Array of shape (n, d) containing n vectors
        K: Simplex dimension (K-simplex has K+1 vertices)
        cost_fn: Distance metric - 'cosine' or 'euclidean'
        projection_method: 'barycentric' (fast, closed-form) or 'sinkhorn' (OT-based)
        epsilon: Entropic regularization for both Sinkhorn and GW
        solver: GW solver - 'PPA' (more stable) or 'PGD' (faster)
        normalize_vectors: Whether to L2-normalize vectors before computing costs
        verbose: Print progress information
        **projection_kwargs: Additional arguments for projection function
            (e.g., 'method', 'labels', 'n_simplex_samples', 'seed')
        
    Returns:
        Dictionary containing:
            - 'K': Simplex dimension
            - 'gmg': Gromov-Monge Gap
            - 'distortion': Map distortion
            - 'optimal_distortion': Minimal achievable distortion
            - 'projected': Projected vectors on simplex (n, K+1)
            - 'C_source': Source cost matrix (n, n)
            - 'C_target': Target cost matrix (n, n)
    """
    n = len(vectors)
    
    if n < K + 1:
        raise ValueError(f"Need at least {K+1} vectors for a {K}-simplex, got {n}")
    
    # Project to K-simplex
    if projection_method == 'barycentric':
        projected = project_to_simplex_barycentric(vectors, K, **projection_kwargs)
    elif projection_method == 'sinkhorn':
        projected = project_to_simplex_sinkhorn(
            vectors, K, reg=epsilon, **projection_kwargs
        )
    else:
        raise ValueError(f"Unknown projection_method: {projection_method}")
    
    # Compute pairwise cost matrices
    if cost_fn == 'cosine':
        C_source = cosine_cost_matrix(vectors, normalize=normalize_vectors)
        C_target = cosine_cost_matrix(projected, normalize=False)  # Already normalized
    elif cost_fn == 'euclidean':
        C_source = euclidean_cost_matrix(vectors)
        C_target = euclidean_cost_matrix(projected)
    else:
        raise ValueError(f"Unknown cost_fn: {cost_fn}")
    
    # Compute map distortion
    distortion = compute_map_distortion(C_source, C_target)
    
    # Compute optimal GW distortion
    optimal_distortion, gw_log = compute_gromov_wasserstein_distance(
        C_source,
        C_target,
        epsilon=epsilon,
        solver=solver,
        verbose=verbose
    )
    
    # Compute GMG
    gmg = distortion - optimal_distortion
    
    if verbose:
        print(f"K={K}: GMG={gmg:.6f}, distortion={distortion:.6f}, "
              f"optimal={optimal_distortion:.6f}")
    
    return {
        'K': K,
        'gmg': gmg,
        'distortion': distortion,
        'optimal_distortion': optimal_distortion,
        'projected': projected,
        'C_source': C_source,
        'C_target': C_target,
        'n_vectors': n
    }


def scan_simplex_dimensions(
    vectors: np.ndarray,
    K_range: Optional[Tuple[int, int]] = None,
    **gmg_kwargs
) -> Dict[int, Dict]:
    """Scan over multiple K values to find best-fitting simplex dimension.
    
    Args:
        vectors: Array of shape (n, d)
        K_range: Tuple of (K_min, K_max). If None, uses (1, min(n-1, 10))
        **gmg_kwargs: Additional arguments passed to compute_gromov_monge_gap
        
    Returns:
        Dictionary mapping K -> result dict from compute_gromov_monge_gap
    """
    n = len(vectors)
    
    if K_range is None:
        K_max = min(n - 1, 10)
        K_range = (1, K_max)
    
    K_min, K_max = K_range
    
    results = {}
    for K in range(K_min, K_max + 1):
        if K >= n:
            break
        try:
            result = compute_gromov_monge_gap(vectors, K, **gmg_kwargs)
            results[K] = result
        except Exception as e:
            print(f"Warning: Failed to compute GMG for K={K}: {e}")
            continue
    
    return results


def find_best_simplex_dimension(
    vectors: np.ndarray,
    K_range: Optional[Tuple[int, int]] = None,
    metric: Literal['distortion', 'gmg'] = 'distortion',
    **gmg_kwargs
) -> Tuple[int, Dict]:
    """Find the K that best describes the simplex structure.
    
    Args:
        vectors: Array of shape (n, d)
        K_range: Tuple of (K_min, K_max)
        metric: 'distortion' (recommended) or 'gmg' for scoring
        **gmg_kwargs: Additional arguments for compute_gromov_monge_gap
        
    Returns:
        Tuple of (best_K, best_result_dict)
    """
    results = scan_simplex_dimensions(vectors, K_range, **gmg_kwargs)
    
    if not results:
        raise ValueError("No valid K values produced results")
    
    # Find K with minimum distortion/GMG
    best_K = min(results.keys(), key=lambda k: results[k][metric])
    
    return best_K, results[best_K]


# Convenience function matching your codebase style
def score_cluster_simplex_structure(
    cluster_directions: np.ndarray,
    K_values: Optional[list] = None,
    **kwargs
) -> Dict[int, Dict]:
    """Score a cluster of SAE directions for simplex structure across K values.
    
    This is a convenience wrapper matching the style of your codebase.
    
    Args:
        cluster_directions: SAE decoder directions, shape (n_features, d_model)
        K_values: List of K values to test. If None, uses range(1, 11)
        **kwargs: Passed to compute_gromov_monge_gap
        
    Returns:
        Dictionary mapping K -> scoring results
    """
    if K_values is None:
        K_values = list(range(1, min(len(cluster_directions), 11)))
    
    K_range = (min(K_values), max(K_values))
    
    return scan_simplex_dimensions(
        cluster_directions,
        K_range=K_range,
        **kwargs
    )



# Exmaple code in the style of how you'd use the above:
#
# from gromov_monge_gap import score_cluster_simplex_structure, find_best_simplex_dimension

# # In your clustering loop:
# for cluster_id, cluster_mask in enumerate(cluster_assignments):
#     cluster_dirs = decoder_directions[cluster_mask]
    
#     # Score across K values
#     results = score_cluster_simplex_structure(
#         cluster_dirs,
#         K_values=list(range(2, 10)),
#         cost_fn='cosine',
#         epsilon=0.1
#     )
    
#     # Find best K
#     best_K = min(results.keys(), key=lambda k: results[k]['distortion'])
#     print(f"Cluster {cluster_id}: best K={best_K}, "
#           f"distortion={results[best_K]['distortion']:.4f}")
