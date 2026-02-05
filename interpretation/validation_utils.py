#!/usr/bin/env python3
"""
Utility functions for belief state validation experiments.

Includes:
- KL divergence computation
- Reference distribution computation
- RGB coloring for simplex visualization
- UMAP embedding of distributions
- Statistical summary functions
"""

import matplotlib
matplotlib.use('Agg')  # Headless backend - must be before pyplot import

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DistributionStats:
    """Statistics for a distribution of values."""
    min: float
    max: float
    p1: float   # 1st percentile
    p25: float  # 25th percentile (Q1)
    p50: float  # 50th percentile (median)
    p75: float  # 75th percentile (Q3)
    p99: float  # 99th percentile
    mean: float
    variance: float
    n_values: int
    raw_values: np.ndarray

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'min': self.min,
            'max': self.max,
            'p1': self.p1,
            'p25': self.p25,
            'p50': self.p50,
            'p75': self.p75,
            'p99': self.p99,
            'mean': self.mean,
            'variance': self.variance,
            'n_values': self.n_values,
            # Don't include raw_values in dict - save separately if needed
        }

    @classmethod
    def from_values(cls, values: np.ndarray) -> 'DistributionStats':
        """Compute stats from array of values."""
        values = np.asarray(values).flatten()
        return cls(
            min=float(np.min(values)),
            max=float(np.max(values)),
            p1=float(np.percentile(values, 1)),
            p25=float(np.percentile(values, 25)),
            p50=float(np.percentile(values, 50)),
            p75=float(np.percentile(values, 75)),
            p99=float(np.percentile(values, 99)),
            mean=float(np.mean(values)),
            variance=float(np.var(values)),
            n_values=len(values),
            raw_values=values,
        )


def kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Compute KL divergence KL(P || Q).

    Args:
        p: Distribution P, shape (..., vocab_size)
        q: Distribution Q, shape (..., vocab_size)
        epsilon: Small value for numerical stability

    Returns:
        KL divergence, shape (...)
    """
    p = p + epsilon
    q = q + epsilon
    # Normalize to ensure they sum to 1
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    return (p * (p.log() - q.log())).sum(dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence JS(P || Q).

    This is a symmetric version of KL divergence.

    Args:
        p: Distribution P, shape (..., vocab_size)
        q: Distribution Q, shape (..., vocab_size)
        epsilon: Small value for numerical stability

    Returns:
        JS divergence, shape (...)
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, epsilon) + 0.5 * kl_divergence(q, m, epsilon)


def compute_pairwise_kl(
    distributions: torch.Tensor,
    max_pairs: Optional[int] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Compute KL divergence for all pairs (or random sample of pairs).

    Args:
        distributions: Shape (n_samples, vocab_size)
        max_pairs: If set, randomly sample this many pairs
        seed: Random seed for pair sampling

    Returns:
        Array of KL divergence values
    """
    n = distributions.shape[0]

    if max_pairs is None or max_pairs >= n * (n - 1) // 2:
        # Compute all pairs
        kl_values = []
        for i in range(n):
            for j in range(i + 1, n):
                kl = kl_divergence(distributions[i], distributions[j])
                kl_values.append(kl.item())
        return np.array(kl_values)
    else:
        # Random sample of pairs
        rng = np.random.RandomState(seed)
        kl_values = []
        pairs_sampled = set()

        while len(kl_values) < max_pairs:
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j and (min(i, j), max(i, j)) not in pairs_sampled:
                pairs_sampled.add((min(i, j), max(i, j)))
                kl = kl_divergence(distributions[i], distributions[j])
                kl_values.append(kl.item())

        return np.array(kl_values)


def barycentric_to_rgb(coords: np.ndarray) -> np.ndarray:
    """
    Convert barycentric coordinates to RGB colors (for k=3 simplex).

    Vertex 0 -> Red   (1, 0, 0)
    Vertex 1 -> Green (0, 1, 0)
    Vertex 2 -> Blue  (0, 0, 1)
    Center   -> Gray  (~0.33, ~0.33, ~0.33)

    Args:
        coords: Shape (n_samples, 3) barycentric coordinates summing to 1

    Returns:
        Shape (n_samples, 3) RGB values in [0, 1]
    """
    coords = np.asarray(coords)
    if coords.shape[1] != 3:
        raise ValueError(f"Expected 3 coordinates for RGB mapping, got {coords.shape[1]}")

    # Direct mapping - barycentric coords ARE the RGB values
    # Clip to [0, 1] in case of numerical issues
    return np.clip(coords, 0, 1)


def kl_to_rgb(
    kl_to_vertices: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Convert KL divergences to RGB colors (for k=3 simplex).

    Lower KL to a vertex -> stronger color for that vertex.

    Args:
        kl_to_vertices: Shape (n_samples, 3) KL divergence to each vertex's reference
        temperature: Softmax temperature (lower = more peaked colors)

    Returns:
        Shape (n_samples, 3) RGB values in [0, 1]
    """
    kl_to_vertices = np.asarray(kl_to_vertices)
    if kl_to_vertices.shape[1] != 3:
        raise ValueError(f"Expected 3 KL values for RGB mapping, got {kl_to_vertices.shape[1]}")

    # Convert KL to "similarity" (negative KL, so lower KL = higher similarity)
    # Use softmax to normalize to sum to 1
    similarity = -kl_to_vertices / temperature
    # Softmax
    exp_sim = np.exp(similarity - similarity.max(axis=1, keepdims=True))  # Subtract max for stability
    rgb = exp_sim / exp_sim.sum(axis=1, keepdims=True)

    return rgb


def select_top_k_vertices(
    samples_by_vertex: Dict[int, list],
    k: int = 3
) -> List[int]:
    """
    Select the k vertices with the most samples.

    Args:
        samples_by_vertex: Dict mapping vertex_id -> list of samples
        k: Number of vertices to select

    Returns:
        List of vertex IDs, sorted by sample count (descending)
    """
    counts = {v: len(samples) for v, samples in samples_by_vertex.items()}
    sorted_vertices = sorted(counts.keys(), key=lambda v: counts[v], reverse=True)
    return sorted_vertices[:k]


def embed_distributions_umap(
    distributions: np.ndarray,
    n_tokens: int = 1000,
    token_selection: str = 'variance',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42,
) -> np.ndarray:
    """
    Embed probability distributions into 2D via UMAP.

    Args:
        distributions: Shape (n_samples, vocab_size) probability distributions
        n_tokens: Number of tokens to keep for embedding
        token_selection: How to select tokens ('variance', 'mean_prob', 'full')
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric for UMAP
        random_state: Random seed

    Returns:
        Shape (n_samples, 2) UMAP embedding
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP not installed. Run: pip install umap-learn")

    distributions = np.asarray(distributions)

    # Select tokens to keep
    if token_selection == 'full' or n_tokens >= distributions.shape[1]:
        reduced = distributions
    elif token_selection == 'variance':
        variances = distributions.var(axis=0)
        top_tokens = np.argsort(variances)[-n_tokens:]
        reduced = distributions[:, top_tokens]
    elif token_selection == 'mean_prob':
        means = distributions.mean(axis=0)
        top_tokens = np.argsort(means)[-n_tokens:]
        reduced = distributions[:, top_tokens]
    else:
        raise ValueError(f"Unknown token_selection: {token_selection}")

    # Run UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(reduced)

    return embedding


def plot_simplex_scatter(
    coords_2d: np.ndarray,
    colors: np.ndarray,
    title: str,
    save_path: str,
    figsize: Tuple[int, int] = (10, 10),
    alpha: float = 0.6,
    s: float = 10,
):
    """
    Create scatter plot with RGB coloring.

    Args:
        coords_2d: Shape (n_samples, 2) coordinates
        colors: Shape (n_samples, 3) RGB colors in [0, 1]
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        alpha: Point transparency
        s: Point size
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, alpha=alpha, s=s)
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_kl_distribution_grid(
    intra_vertex_kls: Dict[int, np.ndarray],
    random_pair_kls: np.ndarray,
    title: str,
    save_path: str,
    figsize: Tuple[int, int] = (12, 10),
    bins: int = 50,
):
    """
    Create 2x2 grid of KL distribution histograms.

    Args:
        intra_vertex_kls: Dict mapping vertex_id -> array of intra-vertex KL values
        random_pair_kls: Array of random pair KL values
        title: Overall title
        save_path: Path to save figure
        figsize: Figure size
        bins: Number of histogram bins
    """
    import matplotlib.pyplot as plt

    # Get vertex IDs (sorted)
    vertex_ids = sorted(intra_vertex_kls.keys())[:3]  # Use top 3 for 2x2 grid

    # Compute global x-axis limits
    all_values = np.concatenate([random_pair_kls] + [intra_vertex_kls[v] for v in vertex_ids])
    x_min, x_max = np.percentile(all_values, [0.5, 99.5])

    n_panels = len(vertex_ids) + 1  # vertices + random pairs
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Plot intra-vertex distributions
    colors = ['red', 'green', 'blue']
    for i, vertex_id in enumerate(vertex_ids):
        ax = axes[i]
        values = intra_vertex_kls[vertex_id]
        ax.hist(values, bins=bins, range=(x_min, x_max), color=colors[i % len(colors)], alpha=0.7, edgecolor='black')
        ax.set_title(f'Vertex {vertex_id} pairs (n={len(values)})')
        ax.set_xlabel('KL Divergence')
        ax.set_ylabel('Count')
        ax.set_xlim(x_min, x_max)

        # Add stats annotation
        stats = DistributionStats.from_values(values)
        ax.axvline(stats.p50, color='black', linestyle='--', linewidth=2, label=f'Median: {stats.p50:.4f}')
        ax.legend()

    # Plot random pairs in the next available slot
    ax = axes[len(vertex_ids)]
    ax.hist(random_pair_kls, bins=bins, range=(x_min, x_max), color='gray', alpha=0.7, edgecolor='black')
    ax.set_title(f'Random pairs (n={len(random_pair_kls)})')
    ax.set_xlabel('KL Divergence')
    ax.set_ylabel('Count')
    ax.set_xlim(x_min, x_max)

    stats = DistributionStats.from_values(random_pair_kls)
    ax.axvline(stats.p50, color='black', linestyle='--', linewidth=2, label=f'Median: {stats.p50:.4f}')
    ax.legend()

    # Hide unused axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
