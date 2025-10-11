"""Metric evaluation system for clustering results."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np

# Optional import for GMG metrics
try:
    from gromov_monge_gap import compute_gromov_monge_gap, score_cluster_simplex_structure
    HAS_GMG = True
except ImportError:
    HAS_GMG = False
    compute_gromov_monge_gap = None
    score_cluster_simplex_structure = None


class MetricEvaluator(ABC):
    """Abstract base class for metric evaluators."""

    @abstractmethod
    def evaluate(self, result: 'ClusteringResult', **kwargs) -> Dict[str, float]:
        """Evaluate metrics on a clustering result.

        Args:
            result: ClusteringResult to evaluate
            **kwargs: Additional context (e.g., decoder directions, beliefs)

        Returns:
            Dict mapping metric name -> value
        """
        pass


class GMGMetricEvaluator(MetricEvaluator):
    """Evaluates Gromov-Monge Gap and distortion per cluster."""

    def __init__(
        self,
        cost_fn: str = "cosine",
        projection_method: str = "barycentric",
        epsilon: float = 0.1,
        normalize_vectors: bool = True,
    ):
        if not HAS_GMG:
            raise ImportError(
                "GMG metrics require the POT library. Install with: pip install POT"
            )
        self.cost_fn = cost_fn
        self.projection_method = projection_method
        self.epsilon = epsilon
        self.normalize_vectors = normalize_vectors

    def evaluate(
        self,
        result: 'ClusteringResult',
        decoder_dirs: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate GMG metrics.

        Args:
            result: ClusteringResult
            decoder_dirs: Full decoder directions (n_latents, d_model)
            **kwargs: Ignored

        Returns:
            Dict with per-cluster and aggregate GMG metrics
        """
        if decoder_dirs is None:
            raise ValueError("GMGMetricEvaluator requires decoder_dirs")

        metrics = {}
        cluster_gmgs = []
        cluster_distortions = []
        cluster_optimal_distortions = []

        for cluster_id in range(result.n_clusters):
            cluster_mask = result.cluster_labels == cluster_id
            cluster_dirs = decoder_dirs[cluster_mask]

            if len(cluster_dirs) < 2:
                # Skip trivial clusters
                continue

            # Determine K based on cluster size
            # Try K = n_latents - 1 (maximum simplex dimension)
            K = min(len(cluster_dirs) - 1, 10)  # Cap at 10 for computational reasons

            if K < 1:
                continue

            try:
                gmg_result = compute_gromov_monge_gap(
                    cluster_dirs,
                    K=K,
                    cost_fn=self.cost_fn,
                    projection_method=self.projection_method,
                    epsilon=self.epsilon,
                    normalize_vectors=self.normalize_vectors,
                    verbose=False,
                )

                metrics[f"cluster_{cluster_id}_gmg"] = gmg_result["gmg"]
                metrics[f"cluster_{cluster_id}_distortion"] = gmg_result["distortion"]
                metrics[f"cluster_{cluster_id}_optimal_distortion"] = gmg_result["optimal_distortion"]
                metrics[f"cluster_{cluster_id}_K"] = K

                cluster_gmgs.append(gmg_result["gmg"])
                cluster_distortions.append(gmg_result["distortion"])
                cluster_optimal_distortions.append(gmg_result["optimal_distortion"])

            except Exception as e:
                print(f"Warning: GMG computation failed for cluster {cluster_id}: {e}")
                continue

        # Aggregate metrics
        if cluster_gmgs:
            metrics["mean_gmg"] = float(np.mean(cluster_gmgs))
            metrics["total_distortion"] = float(np.sum(cluster_distortions))
            metrics["mean_distortion"] = float(np.mean(cluster_distortions))
            metrics["mean_optimal_distortion"] = float(np.mean(cluster_optimal_distortions))
        else:
            metrics["mean_gmg"] = float('inf')
            metrics["total_distortion"] = float('inf')

        return metrics


class BeliefR2Evaluator(MetricEvaluator):
    """Evaluates R² for belief prediction."""

    def evaluate(self, result: 'ClusteringResult', **kwargs) -> Dict[str, float]:
        """Evaluate belief R² metrics.

        Args:
            result: ClusteringResult (should have belief_r2_summary)
            **kwargs: Ignored

        Returns:
            Dict with R² metrics
        """
        metrics = {}

        if result.belief_r2_summary is None:
            return metrics

        all_r2_values = []
        for cluster_id, cluster_r2 in result.belief_r2_summary.items():
            for comp_name, comp_r2 in cluster_r2.items():
                mean_r2 = comp_r2.get("mean_r2", 0.0)
                metrics[f"cluster_{cluster_id}_{comp_name}_r2"] = mean_r2
                all_r2_values.append(mean_r2)

        if all_r2_values:
            metrics["mean_belief_r2"] = float(np.mean(all_r2_values))
            metrics["min_belief_r2"] = float(np.min(all_r2_values))

        return metrics


class CompositeMetricEvaluator(MetricEvaluator):
    """Combines multiple metric evaluators."""

    def __init__(self, evaluators: List[MetricEvaluator]):
        self.evaluators = evaluators

    def evaluate(self, result: 'ClusteringResult', **kwargs) -> Dict[str, float]:
        """Evaluate all metrics.

        Args:
            result: ClusteringResult
            **kwargs: Passed to all evaluators

        Returns:
            Combined dict of all metrics
        """
        all_metrics = {}
        for evaluator in self.evaluators:
            metrics = evaluator.evaluate(result, **kwargs)
            all_metrics.update(metrics)
        return all_metrics


def create_default_evaluators() -> List[MetricEvaluator]:
    """Create default set of metric evaluators."""
    evaluators = []

    # GMG metrics now computed only during geometry refinement phase
    # to avoid redundant GW computation and Sinkhorn convergence warnings
    # if HAS_GMG:
    #     evaluators.append(
    #         GMGMetricEvaluator(
    #             cost_fn="cosine",
    #             projection_method="barycentric",
    #             epsilon=0.1,
    #             normalize_vectors=True,
    #         )
    #     )

    evaluators.append(BeliefR2Evaluator())

    return evaluators


def evaluate_clustering_metrics(
    result: 'ClusteringResult',
    decoder_dirs: Optional[np.ndarray] = None,
    evaluators: Optional[List[MetricEvaluator]] = None,
) -> Dict[str, float]:
    """Convenience function to evaluate all metrics.

    Args:
        result: ClusteringResult to evaluate
        decoder_dirs: Decoder directions for GMG
        evaluators: List of evaluators (defaults to GMG + R²)

    Returns:
        Dict of all metrics
    """
    if evaluators is None:
        evaluators = create_default_evaluators()

    composite = CompositeMetricEvaluator(evaluators)
    return composite.evaluate(result, decoder_dirs=decoder_dirs)
