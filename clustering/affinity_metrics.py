"""
Co-occurrence affinity metrics for SAE feature clustering.

This module implements streaming collection of co-occurrence statistics
and computation of various affinity metrics for spectral clustering.

Supported metrics:
- Geometry-based: cosine, euclidean (computed from decoder directions)
- Co-occurrence-based: jaccard, dice, overlap, phi, mutual_info

References:
- "The Geometry of Concepts" (Li et al., 2024) - Appendix A.1
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Callable, Any
import numpy as np
import torch
from tqdm import tqdm

# Type alias for supported metrics
AffinityMetric = Literal[
    "cosine", "euclidean",  # Geometry-based
    "jaccard", "dice", "overlap", "phi", "mutual_info"  # Co-occurrence-based
]

# Metrics that require co-occurrence statistics
COOCCURRENCE_METRICS = {"jaccard", "dice", "overlap", "phi", "mutual_info"}

# Metrics that use decoder geometry
GEOMETRY_METRICS = {"cosine", "euclidean"}


@dataclass
class CooccurrenceStats:
    """
    Accumulated co-occurrence statistics from streaming data.

    Stores sufficient statistics to compute all co-occurrence affinity metrics:
    - N11[i,j]: Number of samples where both features i and j fired
    - N1[i]: Number of samples where feature i fired (marginal)
    - n_samples: Total number of samples (token positions) observed

    Memory: O(n_features²) for N11 matrix
    """
    N11: np.ndarray      # (n_features, n_features) co-occurrence counts
    N1: np.ndarray       # (n_features,) marginal counts
    n_samples: int       # Total samples observed

    @property
    def n_features(self) -> int:
        return len(self.N1)

    @property
    def sparsity(self) -> float:
        """Average feature activation rate."""
        if self.n_samples == 0:
            return 0.0
        return self.N1.sum() / (self.n_samples * self.n_features)

    def to_affinity(self, method: str, eps: float = 1e-12) -> np.ndarray:
        """Compute affinity matrix from accumulated stats."""
        return cooccurrence_affinity_from_stats(self, method, eps)

    def save(self, path: str) -> None:
        """Save stats to disk for reuse with different metrics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            N11=self.N11,
            N1=self.N1,
            n_samples=np.array([self.n_samples])
        )
        print(f"Saved co-occurrence stats to {path}")

    @classmethod
    def load(cls, path: str) -> 'CooccurrenceStats':
        """Load stats from disk."""
        data = np.load(path)
        return cls(
            N11=data['N11'],
            N1=data['N1'],
            n_samples=int(data['n_samples'][0])
        )

    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"CooccurrenceStats(\n"
            f"  n_features={self.n_features:,},\n"
            f"  n_samples={self.n_samples:,},\n"
            f"  avg_sparsity={self.sparsity:.4%},\n"
            f"  avg_observations_per_feature={self.N1.mean():.1f}\n"
            f")"
        )


def collect_cooccurrence_stats(
    model: Any,
    sae: Any,
    sampler: Any,
    hook_name: str,
    sae_encode_fn: Callable,
    *,
    n_batches: int = 1000,
    batch_size: int = 32,
    seq_len: int = 256,
    activation_threshold: float = 1e-6,
    device: str = "cuda",
    show_progress: bool = True,
    skip_special_tokens: bool = True,
) -> CooccurrenceStats:
    """
    Stream through data accumulating token-level co-occurrence statistics.

    For each token position, records which features fired and updates
    pairwise co-occurrence counts. This captures which features tend to
    fire together at the same token positions.

    Args:
        model: Transformer model (HookedTransformer)
        sae: Sparse autoencoder
        sampler: Data sampler with sample_tokens_batch() method
        hook_name: Name of hook point for activations
        sae_encode_fn: Function to encode activations through SAE
        n_batches: Number of batches to process
        batch_size: Sequences per batch
        seq_len: Tokens per sequence
        activation_threshold: Feature counts as "firing" if |activation| > threshold
        device: Device for computation
        show_progress: Show progress bar
        skip_special_tokens: Skip position 0 (BOS token)

    Returns:
        CooccurrenceStats with accumulated statistics

    Example:
        With n_batches=1000, batch_size=32, seq_len=256:
        Total = 8.2M tokens processed
        For 1% sparse features: ~82k observations per feature on average
    """
    dict_size = sae.W_dec.shape[0]

    # Accumulators - dense is faster than sparse for random access patterns
    N11 = np.zeros((dict_size, dict_size), dtype=np.int64)
    N1 = np.zeros(dict_size, dtype=np.int64)
    n_samples = 0

    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Collecting co-occurrence stats")

    for batch_idx in iterator:
        # Sample batch using the sampler
        tokens = sampler.sample_tokens_batch(batch_size, seq_len, device)

        with torch.no_grad():
            # Run model to get activations
            _, cache = model.run_with_cache(
                tokens,
                return_type=None,
                names_filter=[hook_name]
            )
            acts = cache[hook_name]  # (batch, seq, d_model)
            acts_flat = acts.reshape(-1, acts.shape[-1])

            # Encode through SAE
            feature_acts, _, _ = sae_encode_fn(sae, acts_flat)  # (batch*seq, n_features)

            # Reshape back to (batch, seq, n_features)
            feature_acts = feature_acts.reshape(batch_size, seq_len, -1)

            # Binary: which features fired?
            fired = (feature_acts.abs() > activation_threshold)  # (batch, seq, n_features)

        # Process each sequence
        for b in range(batch_size):
            start_pos = 1 if skip_special_tokens else 0  # Skip BOS token
            for t in range(start_pos, seq_len):
                # Get active features at this token position
                fired_t = fired[b, t, :].cpu().numpy().astype(np.int64)

                # Update marginal counts
                N1 += fired_t

                # Update co-occurrence counts
                # N11[i,j] += fired_t[i] * fired_t[j] for all i,j
                # This is outer product: fired_t[:, None] * fired_t[None, :]
                active_indices = np.where(fired_t)[0]
                if len(active_indices) > 0:
                    # Efficient update using outer product on active indices only
                    N11[np.ix_(active_indices, active_indices)] += 1

                n_samples += 1

        # Memory cleanup
        del cache, acts, acts_flat, feature_acts, fired

        if show_progress and (batch_idx + 1) % 100 == 0:
            avg_active = N1.sum() / n_samples if n_samples > 0 else 0
            iterator.set_postfix({
                'tokens': f'{n_samples:,}',
                'avg_active': f'{avg_active:.1f}'
            })

    return CooccurrenceStats(N11=N11, N1=N1, n_samples=n_samples)


def collect_cooccurrence_stats_from_activations(
    feature_acts: np.ndarray,
    activation_threshold: float = 1e-6,
    show_progress: bool = True,
) -> CooccurrenceStats:
    """
    Compute co-occurrence statistics from a pre-collected activation matrix.

    This is useful when you already have activations in memory (e.g., from
    the existing activity collection pipeline).

    Args:
        feature_acts: (n_samples, n_features) activation matrix
        activation_threshold: Feature counts as firing if |activation| > threshold
        show_progress: Show progress bar

    Returns:
        CooccurrenceStats
    """
    n_samples, n_features = feature_acts.shape

    # Binary activation matrix
    fired = (np.abs(feature_acts) > activation_threshold).astype(np.int64)

    # Marginal counts
    N1 = fired.sum(axis=0)

    # Co-occurrence matrix: N11 = fired.T @ fired
    # This counts for each (i,j) how many samples had both i and j firing
    if show_progress:
        print(f"Computing co-occurrence matrix for {n_features:,} features...")
    N11 = fired.T @ fired

    return CooccurrenceStats(N11=N11, N1=N1, n_samples=n_samples)


def cooccurrence_affinity_from_stats(
    stats: CooccurrenceStats,
    method: str,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute affinity matrix from pre-accumulated co-occurrence statistics.

    This is fast - just arithmetic on the accumulated counts.

    Args:
        stats: Pre-computed co-occurrence statistics
        method: One of 'jaccard', 'dice', 'overlap', 'phi', 'mutual_info'
        eps: Small constant for numerical stability

    Returns:
        (n_features, n_features) symmetric affinity matrix
    """
    N11 = stats.N11.astype(np.float64)
    N1 = stats.N1.astype(np.float64)
    N = float(stats.n_samples)

    # Broadcast for pairwise operations
    N1_i = N1[:, None]  # (n, 1)
    N1_j = N1[None, :]  # (1, n)

    if method == "jaccard":
        # Jaccard similarity: J = |A ∩ B| / |A ∪ B| = n_ij / (n_i + n_j - n_ij)
        denom = N1_i + N1_j - N11 + eps
        S = N11 / denom

    elif method == "dice":
        # Dice coefficient: DSC = 2|A ∩ B| / (|A| + |B|) = 2*n_ij / (n_i + n_j)
        denom = N1_i + N1_j + eps
        S = 2.0 * N11 / denom

    elif method == "overlap":
        # Overlap coefficient: |A ∩ B| / min(|A|, |B|) = n_ij / min(n_i, n_j)
        min_N1 = np.minimum(N1_i, N1_j) + eps
        S = N11 / min_N1

    elif method == "phi":
        # Phi coefficient (Pearson correlation for binary variables)
        # φ = (n11*n00 - n10*n01) / sqrt(n1• * n0• * n•1 * n•0)
        N10 = N1_i - N11  # i fires, j doesn't
        N01 = N1_j - N11  # j fires, i doesn't
        N00 = N - N1_i - N1_j + N11  # neither fires

        numer = N11 * N00 - N10 * N01
        denom = np.sqrt(N1_i * (N - N1_i) * N1_j * (N - N1_j)) + eps
        S = numer / denom

    elif method == "mutual_info":
        # Normalized mutual information
        S = _normalized_mutual_info_from_counts(N11, N1, N, eps)

    else:
        raise ValueError(f"Unknown co-occurrence method: {method}")

    # Clean up numerical issues
    S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(S, 1.0)

    # Clip to valid range
    if method == "phi":
        np.clip(S, -1.0, 1.0, out=S)
    else:
        np.clip(S, 0.0, 1.0, out=S)

    return S


def _normalized_mutual_info_from_counts(
    N11: np.ndarray,
    N1: np.ndarray,
    N: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute normalized mutual information matrix from co-occurrence counts.

    MI(X,Y) = Σ P(x,y) * log(P(x,y) / (P(x)*P(y)))

    Normalized by sqrt(H(X)*H(Y)) to get values in [0, 1].

    Args:
        N11: (n, n) co-occurrence count matrix
        N1: (n,) marginal counts
        N: Total sample count
        eps: Numerical stability constant

    Returns:
        (n, n) normalized mutual information matrix
    """
    # Convert counts to probabilities
    P11 = N11 / (N + eps)  # P(X=1, Y=1)
    P1_i = N1[:, None] / (N + eps)  # P(X=1)
    P1_j = N1[None, :] / (N + eps)  # P(Y=1)

    # Other joint probabilities
    P10 = P1_i - P11  # P(X=1, Y=0)
    P01 = P1_j - P11  # P(X=0, Y=1)
    P00 = 1.0 - P1_i - P1_j + P11  # P(X=0, Y=0)

    # Clip to valid probability range
    P11 = np.clip(P11, eps, 1.0 - eps)
    P10 = np.clip(P10, eps, 1.0 - eps)
    P01 = np.clip(P01, eps, 1.0 - eps)
    P00 = np.clip(P00, eps, 1.0 - eps)
    P1_i = np.clip(P1_i, eps, 1.0 - eps)
    P1_j = np.clip(P1_j, eps, 1.0 - eps)

    # Binary entropy: H(X) = -p*log(p) - (1-p)*log(1-p)
    def binary_entropy(p):
        p = np.clip(p, eps, 1.0 - eps)
        return -p * np.log(p + eps) - (1.0 - p) * np.log(1.0 - p + eps)

    H_i = binary_entropy(P1_i.flatten())  # (n,)
    H_j = binary_entropy(P1_j.flatten())  # (n,)

    # MI term: p(x,y) * log(p(x,y) / (p(x)*p(y)))
    def mi_term(pxy, px, py):
        return pxy * np.log((pxy + eps) / ((px * py) + eps))

    # MI = sum over all 4 joint states
    MI = (
        mi_term(P11, P1_i, P1_j) +
        mi_term(P10, P1_i, 1.0 - P1_j) +
        mi_term(P01, 1.0 - P1_i, P1_j) +
        mi_term(P00, 1.0 - P1_i, 1.0 - P1_j)
    )

    # Normalize: NMI = MI / sqrt(H(X) * H(Y))
    H_prod = np.sqrt(np.outer(H_i, H_j)) + eps
    NMI = MI / H_prod

    return NMI


def build_affinity_matrix(
    decoder_directions: Optional[np.ndarray] = None,
    cooccurrence_stats: Optional[CooccurrenceStats] = None,
    method: AffinityMetric = "cosine",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build affinity matrix for spectral clustering.

    Unified interface for both geometry-based and co-occurrence-based metrics.

    Args:
        decoder_directions: (n_features, d_model) decoder weight matrix
            Required for geometry-based methods (cosine, euclidean)
        cooccurrence_stats: Pre-computed co-occurrence statistics
            Required for co-occurrence methods (jaccard, dice, overlap, phi, mutual_info)
        method: Affinity metric to use
        eps: Numerical stability constant

    Returns:
        (n_features, n_features) symmetric affinity matrix
    """
    if method in GEOMETRY_METRICS:
        if decoder_directions is None:
            raise ValueError(f"Geometry-based method '{method}' requires decoder_directions")
        return _geometry_affinity(decoder_directions, method)

    elif method in COOCCURRENCE_METRICS:
        if cooccurrence_stats is None:
            raise ValueError(f"Co-occurrence method '{method}' requires cooccurrence_stats")
        return cooccurrence_affinity_from_stats(cooccurrence_stats, method, eps)

    else:
        raise ValueError(f"Unknown affinity method: {method}")


def _geometry_affinity(data: np.ndarray, method: str) -> np.ndarray:
    """Compute geometry-based affinity from decoder directions."""
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

    if method == "cosine":
        return cosine_similarity(data)
    elif method == "euclidean":
        D = euclidean_distances(data)
        return 1.0 / (1.0 + D)
    else:
        raise ValueError(f"Unknown geometry method: {method}")


def get_metric_type(method: str) -> Literal["geometry", "cooccurrence"]:
    """Return whether a metric is geometry-based or co-occurrence-based."""
    if method in GEOMETRY_METRICS:
        return "geometry"
    elif method in COOCCURRENCE_METRICS:
        return "cooccurrence"
    else:
        raise ValueError(f"Unknown metric: {method}")
