"""Main clustering pipeline orchestrator."""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from BatchTopK.sae import TopKSAE, VanillaSAE
from multipartite_utils import collect_latent_activity_data
from mess3_gmg_analysis_utils import sae_encode_features
from subspace_clustering_utils import normalize_and_deduplicate

from .config import ClusteringConfig
from .results import ClusteringResult
from .strategies import create_clustering_strategy, ClusteringStrategyResult
from .belief_seeding import BeliefSeeder, BeliefSeedingResult
from .analysis import ClusterAnalyzer
from .epdf_generator import EPDFGenerator


class SiteClusteringPipeline:
    """Main orchestrator for clustering a single site."""

    def __init__(
        self,
        sae_folder: str,
        site_hook_map: Dict[str, str],
    ):
        self.sae_folder = sae_folder
        self.site_hook_map = site_hook_map

    def run(
        self,
        config: ClusteringConfig,
        model,
        cache: Dict,
        data_source,
        site_dir: str,
        component_beliefs_flat: Optional[Dict[str, np.ndarray]] = None,
        component_order: Optional[List[str]] = None,
        component_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        site_idx: int = 0,
        device: str = "cpu",
    ) -> ClusteringResult:
        """Run complete clustering pipeline for a site.

        Args:
            config: Clustering configuration
            model: Transformer model
            cache: Activation cache from model
            data_source: Data source for sampling
            site_dir: Output directory for site-specific files
            component_beliefs_flat: Component beliefs (for seeding/R²)
            component_order: Ordered component names
            component_metadata: Component metadata (for EPDFs)
            site_idx: Site index (for random seeding)
            device: Device for computation

        Returns:
            ClusteringResult with all outputs
        """
        site = config.site
        selected_k = config.selected_k

        # Load SAE (supports both TopK and Vanilla)
        if config.sae_type == "top_k":
            sae_filename = f"{site}_top_k_k{int(config.sae_param)}.pt"
            sae_class = TopKSAE
        else:  # vanilla
            sae_filename = f"{site}_vanilla_lambda_{config.sae_param}.pt"
            sae_class = VanillaSAE

        sae_path = os.path.join(self.sae_folder, sae_filename)
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE checkpoint not found at {sae_path}")

        ckpt = torch.load(sae_path, map_location=device, weights_only=False)
        ckpt["cfg"]["device"] = "cuda" if device.startswith("cuda") else "cpu"
        sae_cfg = dict(ckpt["cfg"])
        sae_cfg["device"] = ckpt["cfg"]["device"]
        sae = sae_class(sae_cfg).to(device)
        sae.load_state_dict(ckpt["state_dict"])
        sae.eval()

        decoder_dirs = sae.W_dec.detach().cpu().numpy()

        # Collect activity statistics
        hook_name = self.site_hook_map[site]
        activity_stats = collect_latent_activity_data(
            model,
            sae,
            data_source,
            hook_name,
            batch_size=config.sampling_config.sample_sequences,
            sample_len=config.sampling_config.sample_seq_len or model.cfg.n_ctx,
            target_len=model.cfg.n_ctx,
            n_batches=config.sampling_config.activation_batches,
            seed=config.seed + site_idx * (config.sampling_config.activation_batches + 1),
            device=device,
            activation_eps=config.sampling_config.latent_activation_eps,
            collect_matrix=(config.method == "spectral" and config.spectral_params.sim_metric == "phi"),
        )

        activity_rates = activity_stats["activity_rates"]
        mean_abs_activation = activity_stats["mean_abs_activation"]
        latent_activity_matrix = activity_stats["latent_matrix"]
        total_activity_samples = int(activity_stats["total_samples"])

        # Prepare activations
        acts = cache[hook_name].detach()
        acts_flat = acts.reshape(-1, acts.shape[-1]).to(device)
        subsample_idx = None

        if config.sampling_config.max_activations and acts_flat.shape[0] > config.sampling_config.max_activations:
            idx = torch.randperm(acts_flat.shape[0], device=acts_flat.device)[:config.sampling_config.max_activations]
            acts_flat = acts_flat[idx].contiguous()
            subsample_idx = idx.cpu().numpy()
        else:
            acts_flat = acts_flat.contiguous()

        # Encode features
        with torch.no_grad():
            feature_acts, x_mean, x_std = sae_encode_features(sae, acts_flat)
        feature_acts = feature_acts.detach()

        # Filter active latents
        active_mask = activity_rates >= config.sampling_config.latent_activity_threshold
        active_indices = np.nonzero(active_mask)[0]
        inactive_indices = np.where(~active_mask)[0]

        print(
            f"{site}: {active_indices.size}/{decoder_dirs.shape[0]} latents pass activity threshold "
            f"{config.sampling_config.latent_activity_threshold:.2%}"
        )

        cluster_labels_full = np.full(decoder_dirs.shape[0], -1, dtype=int)

        # Handle no active latents case
        if active_indices.size == 0:
            return self._create_empty_result(
                config, site, selected_k, cluster_labels_full,
                activity_rates, mean_abs_activation, total_activity_samples,
                active_indices, inactive_indices,
            )

        decoder_active = decoder_dirs[active_indices]
        n_active_before_dedup = len(decoder_active)

        # Normalize and deduplicate upfront (before any other processing)
        # Determine protected positions for deduplication (belief-seeded indices if applicable)
        protected_positions = None
        if config.belief_seeding.enabled and config.belief_seeding.protect_seed_duplicates:
            # Note: We can't protect belief seeds yet since we haven't computed them
            # Belief seeding will happen after dedup, so protection isn't applicable here
            pass

        decoder_normalized, kept_indices_in_active = normalize_and_deduplicate(
            decoder_active,
            cosine_threshold=config.subspace_params.cosine_dedup_threshold,
            protected_indices=protected_positions,
        )

        # Update active_indices to refer to kept (deduplicated) indices
        active_indices = active_indices[kept_indices_in_active]
        decoder_active = decoder_normalized  # Use deduplicated, normalized decoder

        print(f"{site}: after deduplication, kept {len(decoder_active)}/{n_active_before_dedup} active latents")

        # Belief-aligned seeding
        belief_seeder = BeliefSeeder(config.belief_seeding)
        belief_seed_result: Optional[BeliefSeedingResult] = None

        if component_beliefs_flat and component_order:
            # Subsample beliefs if needed
            component_beliefs_for_seeding = {}
            for comp_name in component_order:
                comp_flat = component_beliefs_flat[comp_name]
                if subsample_idx is not None:
                    comp_flat = comp_flat[subsample_idx]
                component_beliefs_for_seeding[comp_name] = comp_flat

            belief_seed_result = belief_seeder.compute_seed_clusters(
                acts_flat,
                feature_acts,
                decoder_active,
                active_indices,
                component_beliefs_for_seeding,
                component_order,
                site,
                site_idx,
            )

        # Run clustering
        strategy = create_clustering_strategy(config)

        # Handle trivial case: single active latent
        if decoder_active.shape[0] == 1:
            # Create soft weights for single point
            soft_weights_trivial = np.array([[1.0]])  # (1, 1) - single point, single cluster
            strategy_result = ClusteringStrategyResult(
                cluster_labels=np.array([0], dtype=int),
                n_clusters=1,
                diagnostics={"clustering_method": config.method},
                soft_weights=soft_weights_trivial,
            )
        else:
            strategy_result = strategy.cluster(
                decoder_active,
                active_indices,
                config,
                site,
                site_dir,
                latent_activity_matrix=latent_activity_matrix,
                belief_seed_clusters=belief_seed_result.seed_clusters_active if belief_seed_result and belief_seed_result.succeeded else None,
                component_order=component_order,
            )

        # Map labels to full set
        cluster_labels_full[active_indices] = strategy_result.cluster_labels

        # Report seed retention
        if belief_seed_result and belief_seed_result.succeeded:
            belief_seeder.report_seed_retention(
                belief_seed_result.seed_clusters_global,
                cluster_labels_full,
                site,
            )

        print(f"{site}: clustered {decoder_active.shape[0]} active latents into {strategy_result.n_clusters} groups")

        # Cluster analysis
        analyzer = ClusterAnalyzer(config.analysis_config)
        encoded_cache = (feature_acts, x_mean, x_std)

        cluster_recons, cluster_stats = analyzer.collect_reconstructions(
            acts_flat,
            sae,
            cluster_labels_full,
            config.sampling_config.cluster_activation_threshold,
            encoded_cache=encoded_cache,
        )

        feature_np_for_r2 = feature_acts.detach().cpu().numpy()
        del feature_acts, encoded_cache, x_mean, x_std

        # Add activity rates to stats
        per_site_cluster_rates = analyzer.add_activity_rates_to_stats(
            cluster_stats, activity_rates, mean_abs_activation
        )

        # PCA analysis (if enabled and stats available)
        pca_results = None
        if cluster_stats:
            pca_results = analyzer.fit_pca_and_project(
                cluster_recons,
                cluster_stats,
                sae,
                config.sampling_config.min_cluster_samples,
            )

        # Belief R² scoring (hard + soft assignments)
        belief_r2_summary = None
        belief_r2_summary_soft = None
        component_assignment = None
        component_assignment_soft = None

        if component_beliefs_flat and component_order:
            # Re-subsample beliefs if needed
            component_beliefs_for_scoring = {}
            for comp_name in component_order:
                comp_flat = component_beliefs_flat[comp_name]
                if subsample_idx is not None:
                    comp_flat = comp_flat[subsample_idx]
                component_beliefs_for_scoring[comp_name] = comp_flat

            # Compute hard R²
            belief_r2_summary = analyzer.compute_belief_r2(
                acts_flat,
                feature_np_for_r2,
                decoder_dirs,
                cluster_labels_full,
                strategy_result.n_clusters,
                component_beliefs_for_scoring,
                component_order,
                config.belief_seeding.ridge_alpha,
                site,
                soft_assignments=None,
                assignment_name="hard",
            )

            # Compute optimal assignment for hard R²
            if belief_r2_summary:
                component_assignment = analyzer.compute_optimal_component_assignment(
                    belief_r2_summary,
                    component_order,
                    strategy_result.n_clusters,
                )

            # Compute soft R² if soft assignments exist
            if strategy_result.soft_weights is not None:
                # Filter to active latents for soft R²
                # soft_weights has shape (n_active_latents, n_clusters)
                # Need to pass filtered data matching this shape
                active_feature_acts = feature_np_for_r2[:, active_indices]
                active_decoder_dirs = decoder_dirs[active_indices]
                active_cluster_labels = cluster_labels_full[active_indices]

                belief_r2_summary_soft = analyzer.compute_belief_r2(
                    acts_flat,
                    active_feature_acts,
                    active_decoder_dirs,
                    active_cluster_labels,
                    strategy_result.n_clusters,
                    component_beliefs_for_scoring,
                    component_order,
                    config.belief_seeding.ridge_alpha,
                    site,
                    soft_assignments=strategy_result.soft_weights,
                    assignment_name="soft",
                )

                # Compute optimal assignment for soft R²
                if belief_r2_summary_soft:
                    component_assignment_soft = analyzer.compute_optimal_component_assignment(
                        belief_r2_summary_soft,
                        component_order,
                        strategy_result.n_clusters,
                    )

        # Activation coherence metrics
        coherence_metrics = None
        coherence_metrics_soft = None
        if feature_np_for_r2 is not None and cluster_labels_full is not None:
            # Hard coherence
            coherence_metrics = analyzer.compute_activation_coherence_metrics(
                feature_np_for_r2,
                cluster_labels_full,
                strategy_result.n_clusters,
                soft_assignments=None,
            )

            # Soft coherence (if soft assignments exist)
            if strategy_result.soft_weights is not None:
                # Filter to active latents only for soft coherence
                # soft_weights shape: (n_active_latents, n_clusters)
                # feature_np_for_r2 shape: (n_samples, n_latents_total)
                # We need to filter feature_np_for_r2 to only active latents
                active_feature_acts = feature_np_for_r2[:, active_indices]
                active_cluster_labels = cluster_labels_full[active_indices]

                coherence_metrics_soft = analyzer.compute_activation_coherence_metrics(
                    active_feature_acts,
                    active_cluster_labels,
                    strategy_result.n_clusters,
                    soft_assignments=strategy_result.soft_weights,
                )

        # Build result
        result = ClusteringResult(
            config=config,
            site=site,
            selected_k=selected_k,
            cluster_labels=cluster_labels_full,
            n_clusters=strategy_result.n_clusters,
            active_indices=active_indices,
            inactive_indices=inactive_indices,
            activity_rates=activity_rates,
            mean_abs_activation=mean_abs_activation,
            total_activity_samples=total_activity_samples,
            cluster_stats=cluster_stats,
            belief_seed_metadata=belief_seed_result.metadata if belief_seed_result and belief_seed_result.succeeded else None,
            belief_r2_summary=belief_r2_summary,
            belief_r2_summary_soft=belief_r2_summary_soft,
            component_assignment=component_assignment,
            component_assignment_soft=component_assignment_soft,
            coherence_metrics=coherence_metrics,
            coherence_metrics_soft=coherence_metrics_soft,
            subspace_diagnostics=strategy_result.diagnostics,
            pca_results=pca_results,
        )

        # Store strategy result for geometry refinement (soft assignments)
        result._strategy_result = strategy_result

        # Store PCA plotting data for later (when we have site_dir)
        if pca_results:
            result._pca_plot_data = {
                'seed': config.seed,
            }

        # EPDF generation (if enabled and has component data)
        epdf_generator = EPDFGenerator(config.epdf_config)
        if component_metadata and component_beliefs_flat and component_order:
            component_beliefs_for_epdf = {}
            for comp_name in component_order:
                comp_flat = component_beliefs_flat[comp_name]
                if subsample_idx is not None:
                    comp_flat = comp_flat[subsample_idx]
                component_beliefs_for_epdf[comp_name] = comp_flat

            # Note: EPDFs will be generated when save_to_directory is called
            # Store the necessary data in result for later
            result._epdf_data = {
                'sae': sae,
                'acts_flat': acts_flat,
                'component_beliefs_flat': component_beliefs_for_epdf,
                'component_metadata': component_metadata,
                'component_order': component_order,
                'decoder_normalized': strategy_result.decoder_normalized,
                'normalized_to_full_idx': strategy_result.normalized_to_full_idx,
                'decoder_dirs': decoder_dirs,
            }

        return result

    def _create_empty_result(
        self,
        config: ClusteringConfig,
        site: str,
        selected_k: int,
        cluster_labels_full: np.ndarray,
        activity_rates: np.ndarray,
        mean_abs_activation: np.ndarray,
        total_activity_samples: int,
        active_indices: np.ndarray,
        inactive_indices: np.ndarray,
    ) -> ClusteringResult:
        """Create result for case with no active latents."""
        return ClusteringResult(
            config=config,
            site=site,
            selected_k=selected_k,
            cluster_labels=cluster_labels_full,
            n_clusters=0,
            active_indices=active_indices,
            inactive_indices=inactive_indices,
            activity_rates=activity_rates,
            mean_abs_activation=mean_abs_activation,
            total_activity_samples=total_activity_samples,
            cluster_stats={},
            belief_seed_metadata=None if not config.belief_seeding.enabled else {"applied": False},
        )
