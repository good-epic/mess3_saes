#!/usr/bin/env python3
"""
Validate belief state simplices through distributional analysis.

Modes:
- coherence: Check that intra-vertex samples are similar (prerequisite)
- baseline: RGB visualization comparing coord-based vs KL-based coloring
- nexttoken: UMAP embedding of next-token distributions
- all: Run coherence + baseline in sequence (recommended, shares cached distributions)
- steering: Intervene by adding steering vectors (TODO)
- patching: Patch cluster activations between states (TODO)

Usage:
    # Run all in one invocation (recommended)
    python validate_belief_states.py \
        --mode all \
        --prepared_samples_dir outputs/interpretations/prepared_samples_512 \
        --uniform_samples_path outputs/validation_samples/validation_samples_512_261.jsonl \
        --n_clusters 512 --cluster_id 261 \
        --output_dir outputs/validation/512_261 \
        --model_name gemma-2-9b \
        --device cuda
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Add project root and script directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from validation_utils import (
    kl_divergence,
    compute_pairwise_kl,
    barycentric_to_rgb,
    kl_to_rgb,
    select_top_k_vertices,
    embed_distributions_umap,
    plot_simplex_scatter,
    plot_kl_distribution_grid,
    DistributionStats,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate belief state simplices")

    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                       choices=['coherence', 'baseline', 'nexttoken', 'steering', 'patching', 'all'],
                       help="Validation mode to run")

    # Cluster specification
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--cluster_id", type=int, required=True)

    # Input paths
    parser.add_argument("--prepared_samples_dir", type=str, required=True,
                       help="Directory with prepared vertex samples (from prepare_vertex_samples.py)")
    parser.add_argument("--uniform_samples_path", type=str, default=None,
                       help="Path to uniform samples JSONL (from collect_validation_samples.py)")
    parser.add_argument("--aanet_checkpoint", type=str, default=None,
                       help="Path to AANet checkpoint (required for steering/patching)")
    parser.add_argument("--cluster_manifest", type=str, default=None,
                       help="Path to manifest.json with cluster info")

    # Model & SAE
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_16k/average_l0_68")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="/workspace/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None)

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save validation results")

    # Coherence check params
    parser.add_argument("--max_pairs_per_vertex", type=int, default=500,
                       help="Max pairs to sample per vertex for intra-vertex KL")
    parser.add_argument("--random_pairs", type=int, default=5000,
                       help="Number of random pairs from uniform samples")

    # UMAP params
    parser.add_argument("--distribution_tokens", type=int, default=1000,
                       help="Number of tokens to include in distribution embedding")
    parser.add_argument("--token_selection", type=str, default="variance",
                       choices=['variance', 'mean_prob', 'full'],
                       help="How to select tokens for UMAP")
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)

    # Steering params (for future)
    parser.add_argument("--source_vertex", type=int, default=0)
    parser.add_argument("--target_vertex", type=int, default=1)
    parser.add_argument("--steering_strengths", type=str, default="0.1,0.25,0.5,1.0,2.0")
    parser.add_argument("--steering_mode", type=str, default="interpolate",
                       choices=['add', 'interpolate', 'replace'])

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples_for_reference", type=int, default=100,
                       help="Number of near-vertex samples to use for reference distribution")

    args = parser.parse_args()
    return args


def load_prepared_samples(samples_dir: str, n_clusters: int, cluster_id: int) -> dict:
    """Load prepared samples for a cluster."""
    sample_path = Path(samples_dir) / f"cluster_{n_clusters}_{cluster_id}.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Prepared samples not found: {sample_path}")

    with open(sample_path) as f:
        return json.load(f)


def load_uniform_samples(samples_path: str, max_samples: Optional[int] = None) -> List[dict]:
    """Load uniform samples from JSONL file."""
    samples = []
    with open(samples_path) as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def get_next_token_distribution(
    model,
    tokenizer,
    text: str,
    token_idx: int,
    device: str
) -> torch.Tensor:
    """
    Get the next-token probability distribution at a specific position.

    Uses model.to_tokens(prepend_bos=True) so token_idx values from
    collect_validation_samples.py (which are adjusted for BOS) are correct.

    Args:
        model: HookedTransformer model
        tokenizer: Tokenizer
        text: Input text
        token_idx: Position to get distribution for (relative to BOS-prefixed sequence)
        device: Device

    Returns:
        Probability distribution over vocabulary, shape (vocab_size,)
    """
    # Use model.to_tokens to match original tokenization (always prepends BOS)
    tokens = model.to_tokens(text, prepend_bos=True).to(device)

    if token_idx >= tokens.shape[1]:
        token_idx = tokens.shape[1] - 1

    with torch.no_grad():
        logits = model(tokens)

    target_logits = logits[0, token_idx, :]
    probs = F.softmax(target_logits, dim=-1)

    return probs


class DistributionCache:
    """
    Cache for computed next-token distributions.

    Computes distributions once and shares them between coherence and baseline modes.
    Saves/loads from disk so distributions survive across invocations.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self._vertex_distributions = {}  # vertex_id -> tensor (n_samples, vocab_size)
        self._uniform_distributions = None  # tensor (n_samples, vocab_size)
        self._uniform_coords = None  # numpy array (n_samples, k)
        self._reference_distributions = {}  # vertex_id -> tensor (vocab_size,)

    def _vertex_dist_path(self, vertex_id: int) -> Path:
        return self.cache_dir / f"vertex_{vertex_id}_distributions.pt"

    def _uniform_dist_path(self) -> Path:
        return self.cache_dir / "uniform_distributions.pt"

    def _uniform_coords_path(self) -> Path:
        return self.cache_dir / "uniform_coords.npy"

    def _reference_dist_path(self) -> Path:
        return self.cache_dir / "reference_distributions.pt"

    def has_vertex_distributions(self, vertex_id: int) -> bool:
        return vertex_id in self._vertex_distributions or self._vertex_dist_path(vertex_id).exists()

    def has_uniform_distributions(self) -> bool:
        return self._uniform_distributions is not None or self._uniform_dist_path().exists()

    def has_reference_distributions(self) -> bool:
        return len(self._reference_distributions) > 0 or self._reference_dist_path().exists()

    def get_vertex_distributions(self, vertex_id: int) -> Optional[torch.Tensor]:
        if vertex_id in self._vertex_distributions:
            return self._vertex_distributions[vertex_id]
        path = self._vertex_dist_path(vertex_id)
        if path.exists():
            self._vertex_distributions[vertex_id] = torch.load(path, weights_only=True)
            return self._vertex_distributions[vertex_id]
        return None

    def set_vertex_distributions(self, vertex_id: int, dists: torch.Tensor):
        self._vertex_distributions[vertex_id] = dists
        torch.save(dists, self._vertex_dist_path(vertex_id))

    def get_uniform_distributions(self) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
        if self._uniform_distributions is not None:
            return self._uniform_distributions, self._uniform_coords
        if self._uniform_dist_path().exists() and self._uniform_coords_path().exists():
            self._uniform_distributions = torch.load(self._uniform_dist_path(), weights_only=True)
            self._uniform_coords = np.load(self._uniform_coords_path())
            return self._uniform_distributions, self._uniform_coords
        return None

    def set_uniform_distributions(self, dists: torch.Tensor, coords: np.ndarray):
        self._uniform_distributions = dists
        self._uniform_coords = coords
        torch.save(dists, self._uniform_dist_path())
        np.save(self._uniform_coords_path(), coords)

    def get_reference_distributions(self) -> Optional[Dict[int, torch.Tensor]]:
        if len(self._reference_distributions) > 0:
            return self._reference_distributions
        path = self._reference_dist_path()
        if path.exists():
            self._reference_distributions = torch.load(path, weights_only=True)
            return self._reference_distributions
        return None

    def set_reference_distributions(self, ref_dists: Dict[int, torch.Tensor]):
        self._reference_distributions = ref_dists
        torch.save(ref_dists, self._reference_dist_path())


def compute_distributions_for_samples(
    model,
    tokenizer,
    samples: List[dict],
    device: str,
    desc: str = "Computing distributions",
) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute next-token distributions for a list of samples.

    Returns:
        distributions: tensor (n_valid, vocab_size) on CPU
        valid_indices: list of indices into the original samples list
    """
    distributions = []
    valid_indices = []

    for i, sample in enumerate(tqdm(samples, desc=desc)):
        if 'trigger_token_indices' in sample:
            token_idx = sample['trigger_token_indices'][0]
        elif 'trigger_token_idx' in sample:
            token_idx = sample['trigger_token_idx']
        else:
            continue

        try:
            # Reconstruct the exact token sequence the model originally saw.
            # chunk_token_ids: raw tokens from the original chunk (with or without BOS as-is).
            # This avoids re-tokenization round-trip issues entirely.
            if 'chunk_token_ids' in sample:
                tokens = torch.tensor([sample['chunk_token_ids']], device=device)
            else:
                # Fallback for legacy data without chunk_token_ids.
                # Re-tokenize with BOS and hope for the best.
                tokens = model.to_tokens(sample['full_text'], prepend_bos=True).to(device)

            if token_idx >= tokens.shape[1]:
                continue

            # Verify token ID if available
            expected_token_id = None
            if 'trigger_token_ids' in sample:
                expected_token_id = sample['trigger_token_ids'][0]
            elif 'token_id' in sample:
                expected_token_id = sample['token_id']

            if expected_token_id is not None and tokens[0, token_idx].item() != expected_token_id:
                print(f"  WARNING: Token ID mismatch at sample {i}: "
                      f"expected {expected_token_id}, got {tokens[0, token_idx].item()} "
                      f"at index {token_idx}. Skipping.")
                continue

            # Get next-token distribution at the trigger position
            with torch.no_grad():
                logits = model(tokens)
            dist = torch.softmax(logits[0, token_idx, :].float(), dim=-1)
            distributions.append(dist.cpu())
            valid_indices.append(i)
        except Exception:
            continue

    if distributions:
        return torch.stack(distributions), valid_indices
    return torch.empty(0), []


def ensure_vertex_distributions(
    cache: DistributionCache,
    model,
    tokenizer,
    samples_by_vertex: Dict[int, List[dict]],
    vertex_ids: List[int],
    max_samples: int,
    device: str,
):
    """Compute and cache vertex distributions if not already cached."""
    for vertex_id in vertex_ids:
        if cache.has_vertex_distributions(vertex_id):
            dists = cache.get_vertex_distributions(vertex_id)
            print(f"  Vertex {vertex_id}: loaded {dists.shape[0]} cached distributions")
            continue

        samples = samples_by_vertex.get(vertex_id, [])
        if len(samples) < 2:
            print(f"  Skipping vertex {vertex_id}: only {len(samples)} samples")
            continue

        samples_to_use = samples[:max_samples]
        dists, _ = compute_distributions_for_samples(
            model, tokenizer, samples_to_use, device,
            desc=f"Vertex {vertex_id} distributions"
        )

        if dists.shape[0] > 0:
            cache.set_vertex_distributions(vertex_id, dists)
            print(f"  Vertex {vertex_id}: computed and cached {dists.shape[0]} distributions")


def ensure_uniform_distributions(
    cache: DistributionCache,
    model,
    tokenizer,
    uniform_samples_path: str,
    device: str,
):
    """Compute and cache uniform distributions if not already cached."""
    if cache.has_uniform_distributions():
        dists, coords = cache.get_uniform_distributions()
        print(f"  Loaded {dists.shape[0]} cached uniform distributions")
        return

    print("  Loading uniform samples...")
    uniform_samples = load_uniform_samples(uniform_samples_path)
    print(f"  Loaded {len(uniform_samples)} uniform samples")

    dists, valid_indices = compute_distributions_for_samples(
        model, tokenizer, uniform_samples, device,
        desc="Uniform distributions"
    )

    # Extract coords for valid samples
    coords = np.array([
        uniform_samples[i]['barycentric_coords']
        for i in valid_indices
    ])

    if dists.shape[0] > 0:
        cache.set_uniform_distributions(dists, coords)
        print(f"  Computed and cached {dists.shape[0]} uniform distributions")


def ensure_reference_distributions(
    cache: DistributionCache,
    vertex_ids: List[int],
    samples_per_vertex: int,
):
    """Compute reference distributions (vertex averages) from cached vertex distributions."""
    if cache.has_reference_distributions():
        ref_dists = cache.get_reference_distributions()
        print(f"  Loaded cached reference distributions for vertices: {list(ref_dists.keys())}")
        return

    ref_dists = {}
    for vertex_id in vertex_ids:
        dists = cache.get_vertex_distributions(vertex_id)
        if dists is None or dists.shape[0] == 0:
            print(f"  WARNING: No distributions for vertex {vertex_id}, skipping reference")
            continue
        # Average in probability space, using up to samples_per_vertex
        ref_dists[vertex_id] = dists[:samples_per_vertex].mean(dim=0)
        print(f"  Vertex {vertex_id}: reference from {min(dists.shape[0], samples_per_vertex)} samples")

    cache.set_reference_distributions(ref_dists)


def run_coherence_check(args, model, tokenizer, cache: DistributionCache):
    """
    Run coherence check: verify intra-vertex KL << random-pair KL.
    """
    print("\n" + "="*80)
    print("COHERENCE CHECK")
    print("="*80)

    output_dir = Path(args.output_dir) / "coherence"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prepared samples
    print("\nLoading prepared samples...")
    prepared = load_prepared_samples(args.prepared_samples_dir, args.n_clusters, args.cluster_id)
    k = prepared['k']
    samples_by_vertex = {int(v): samples for v, samples in prepared['vertices'].items()}

    print(f"  k={k}")
    for v, samples in samples_by_vertex.items():
        print(f"  Vertex {v}: {len(samples)} samples")

    # Select top 3 vertices
    if k > 3:
        top_vertices = select_top_k_vertices(samples_by_vertex, k=3)
        print(f"\n  Using top 3 vertices for visualization: {top_vertices}")
    else:
        top_vertices = list(range(k))

    # Compute vertex distributions (cached)
    print("\nComputing vertex distributions...")
    ensure_vertex_distributions(
        cache, model, tokenizer, samples_by_vertex, top_vertices,
        max_samples=args.max_pairs_per_vertex * 2,
        device=args.device,
    )

    # Compute intra-vertex KL divergences
    print("\nComputing intra-vertex KL divergences...")
    intra_vertex_kls = {}

    for vertex_id in top_vertices:
        dists = cache.get_vertex_distributions(vertex_id)
        if dists is None or dists.shape[0] < 2:
            continue

        kl_values = compute_pairwise_kl(dists, max_pairs=args.max_pairs_per_vertex, seed=args.seed)
        intra_vertex_kls[vertex_id] = kl_values
        stats = DistributionStats.from_values(kl_values)
        print(f"  Vertex {vertex_id}: median KL = {stats.p50:.4f}, mean = {stats.mean:.4f}, n_pairs = {len(kl_values)}")

    # Compute uniform distributions (cached)
    random_pair_kls = np.array([])
    if args.uniform_samples_path:
        print("\nComputing uniform distributions...")
        ensure_uniform_distributions(cache, model, tokenizer, args.uniform_samples_path, args.device)

        result = cache.get_uniform_distributions()
        if result is not None:
            uniform_dists, _ = result

            print("\nComputing random-pair KL divergences...")
            random_pair_kls = compute_pairwise_kl(uniform_dists, max_pairs=args.random_pairs, seed=args.seed)
            stats = DistributionStats.from_values(random_pair_kls)
            print(f"  Random pairs: median KL = {stats.p50:.4f}, mean = {stats.mean:.4f}, n_pairs = {len(random_pair_kls)}")
    else:
        print("\nWARNING: No uniform samples provided, skipping random-pair comparison")

    # Save statistics
    print("\nSaving statistics...")
    stats_results = {
        'intra_vertex': {},
        'random_pairs': None,
    }

    for vertex_id, kl_values in intra_vertex_kls.items():
        stats = DistributionStats.from_values(kl_values)
        stats_results['intra_vertex'][vertex_id] = stats.to_dict()
        np.save(output_dir / f"intra_vertex_{vertex_id}_kl_values.npy", kl_values)

    if len(random_pair_kls) > 0:
        stats = DistributionStats.from_values(random_pair_kls)
        stats_results['random_pairs'] = stats.to_dict()
        np.save(output_dir / f"random_pair_kl_values.npy", random_pair_kls)

    with open(output_dir / "coherence_stats.json", 'w') as f:
        json.dump(stats_results, f, indent=2)

    # Generate visualization
    if len(intra_vertex_kls) >= 1 and len(random_pair_kls) > 0:
        print("\nGenerating KL distribution grid...")
        plot_kl_distribution_grid(
            intra_vertex_kls,
            random_pair_kls,
            title=f"KL Divergence Distributions - Cluster {args.n_clusters}_{args.cluster_id}",
            save_path=str(output_dir / "kl_distribution_grid.png"),
        )
        print(f"  Saved: kl_distribution_grid.png")

    # Coherence assessment
    print("\n" + "-"*80)
    print("COHERENCE ASSESSMENT")
    print("-"*80)

    if len(random_pair_kls) > 0 and len(intra_vertex_kls) > 0:
        random_median = np.median(random_pair_kls)
        intra_medians = [np.median(v) for v in intra_vertex_kls.values()]
        avg_intra_median = np.mean(intra_medians)

        ratio = random_median / avg_intra_median if avg_intra_median > 0 else float('inf')

        print(f"Average intra-vertex median KL: {avg_intra_median:.4f}")
        print(f"Random-pair median KL: {random_median:.4f}")
        print(f"Ratio (random / intra): {ratio:.2f}x")

        if ratio > 2:
            print("\nPASS: Intra-vertex samples are significantly more similar than random pairs")
            print("  The simplex appears to capture meaningful structure.")
        elif ratio > 1.5:
            print("\nMARGINAL: Some signal, but not strongly conclusive")
            print("  Consider examining the distributions more carefully.")
        else:
            print("\nFAIL: Intra-vertex samples are not much more similar than random pairs")
            print("  The simplex may not capture meaningful belief state structure.")

    print(f"\nResults saved to: {output_dir}")
    return stats_results


def run_baseline_visualization(args, model, tokenizer, cache: DistributionCache):
    """
    Run baseline visualization: RGB coloring by coords vs by KL.
    """
    print("\n" + "="*80)
    print("BASELINE VISUALIZATION")
    print("="*80)

    output_dir = Path(args.output_dir) / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prepared samples
    print("\nLoading prepared samples...")
    prepared = load_prepared_samples(args.prepared_samples_dir, args.n_clusters, args.cluster_id)
    k = prepared['k']
    samples_by_vertex = {int(v): samples for v, samples in prepared['vertices'].items()}

    # Select top 3 vertices
    if k > 3:
        top_vertices = select_top_k_vertices(samples_by_vertex, k=3)
        print(f"  Using top 3 vertices: {top_vertices}")
    else:
        top_vertices = list(range(k))

    # Ensure vertex distributions are computed (may already be cached from coherence)
    print("\nEnsuring vertex distributions are available...")
    ensure_vertex_distributions(
        cache, model, tokenizer, samples_by_vertex, top_vertices,
        max_samples=args.samples_for_reference,
        device=args.device,
    )

    # Compute reference distributions (cached)
    print("\nComputing reference distributions...")
    ensure_reference_distributions(cache, top_vertices, args.samples_for_reference)

    reference_dists = cache.get_reference_distributions()
    if not reference_dists:
        print("ERROR: Could not compute reference distributions")
        return

    # Save reference distributions as npz for external use
    ref_dists_np = {f"vertex_{v}": d.numpy() for v, d in reference_dists.items()}
    np.savez(output_dir / "reference_distributions.npz", **ref_dists_np)

    # Ensure uniform distributions are available (may be cached from coherence)
    if args.uniform_samples_path:
        print("\nEnsuring uniform distributions are available...")
        ensure_uniform_distributions(cache, model, tokenizer, args.uniform_samples_path, args.device)

    result = cache.get_uniform_distributions()
    if result is None:
        print("ERROR: No uniform distributions available for baseline visualization")
        return

    all_distributions, all_coords = result
    all_distributions_np = all_distributions.numpy()

    print(f"  Using {all_distributions.shape[0]} samples for visualization")

    # Compute KL to each reference for all uniform samples
    # Use top_vertices ordering to match coord-based RGB mapping
    print("\nComputing KL to reference distributions...")
    all_kl_to_vertices = []

    for i in tqdm(range(all_distributions.shape[0]), desc="KL to references"):
        sample_dist = all_distributions[i]
        kl_to_vertices = []
        for v in top_vertices:
            kl = kl_divergence(sample_dist, reference_dists[v])
            kl_to_vertices.append(kl.item())
        all_kl_to_vertices.append(kl_to_vertices)

    all_kl_to_vertices = np.array(all_kl_to_vertices)

    # For visualization, use only the top 3 vertices' coordinates
    if k > 3:
        # Remap coordinates to top-3 vertex space
        vis_coords = all_coords[:, top_vertices]
        # Renormalize to sum to 1
        vis_coords = vis_coords / vis_coords.sum(axis=1, keepdims=True)
    else:
        vis_coords = all_coords

    # Create RGB visualizations (requires 3 coordinates)
    if vis_coords.shape[1] == 3:
        print("\nGenerating RGB visualizations...")

        rgb_coords = barycentric_to_rgb(vis_coords)
        rgb_kl = kl_to_rgb(all_kl_to_vertices)

        # UMAP embedding for 2D layout
        print("  Running UMAP on distributions...")
        embedding_2d = embed_distributions_umap(
            all_distributions_np,
            n_tokens=args.distribution_tokens,
            token_selection=args.token_selection,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.seed,
        )

        # Plot with coord-based colors
        plot_simplex_scatter(
            embedding_2d, rgb_coords,
            title=f"UMAP - Colored by Simplex Coordinates\nCluster {args.n_clusters}_{args.cluster_id}",
            save_path=str(output_dir / "umap_by_coords.png"),
        )
        print("  Saved: umap_by_coords.png")

        # Plot with KL-based colors
        plot_simplex_scatter(
            embedding_2d, rgb_kl,
            title=f"UMAP - Colored by KL to References\nCluster {args.n_clusters}_{args.cluster_id}",
            save_path=str(output_dir / "umap_by_kl.png"),
        )
        print("  Saved: umap_by_kl.png")

        # Compute correlation
        correlations = {}
        for i, color in enumerate(['R', 'G', 'B']):
            corr = np.corrcoef(rgb_coords[:, i], rgb_kl[:, i])[0, 1]
            correlations[color] = float(corr)
            print(f"  Correlation ({color} channel): {corr:.3f}")

        # Save UMAP embedding
        np.savez(
            output_dir / "umap_embedding.npz",
            embedding=embedding_2d,
            rgb_coords=rgb_coords,
            rgb_kl=rgb_kl,
        )
    else:
        correlations = None

    # Save full data
    print("\nSaving data...")
    np.savez(
        output_dir / "baseline_data.npz",
        coords=all_coords,
        kl_to_vertices=all_kl_to_vertices,
        distributions=all_distributions_np,
    )

    # Save summary
    summary = {
        "n_samples": int(all_distributions.shape[0]),
        "k": k,
        "top_vertices": top_vertices,
        "reference_vertices": sorted(reference_dists.keys()),
        "correlations": correlations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_dir / "baseline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return summary


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("BELIEF STATE VALIDATION")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Cluster: {args.n_clusters}_{args.cluster_id}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Load model (needed for all modes)
    print("\nLoading model...")
    from transformer_lens import HookedTransformer
    from huggingface_hub import login

    if args.hf_token:
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    model_kwargs = {}
    if args.cache_dir:
        model_kwargs['cache_dir'] = args.cache_dir

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        center_unembed=False,
        center_writing_weights=False,
        dtype="bfloat16",
        **model_kwargs
    )
    model.eval()
    tokenizer = model.tokenizer

    # Create distribution cache (shared between modes)
    cache = DistributionCache(output_dir / "dist_cache")

    # Run requested mode(s)
    if args.mode == 'coherence' or args.mode == 'all':
        run_coherence_check(args, model, tokenizer, cache)

    if args.mode == 'baseline' or args.mode == 'all':
        run_baseline_visualization(args, model, tokenizer, cache)

    if args.mode == 'nexttoken':
        print("\nNext-token analysis mode - running baseline visualization with UMAP")
        run_baseline_visualization(args, model, tokenizer, cache)

    if args.mode == 'steering':
        print("\nSteering mode not yet implemented")

    if args.mode == 'patching':
        print("\nPatching mode not yet implemented")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
