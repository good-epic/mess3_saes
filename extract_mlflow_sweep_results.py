#!/usr/bin/env python3
"""Extract hyperparameter sweep results from MLflow.

This script downloads cluster_summary.json artifacts from all runs in the
MLflow experiment and extracts key metrics into a tabular format.

Output: CSV file with one row per layer per run (3 rows per run).
"""

import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


# Configuration
EXPERIMENT_NAME = "/Shared/mp_clustering_sweep"
LAYERS = ["layer_0", "layer_1", "layer_2"]
GEOMETRIES = [f"simplex_{i}" for i in range(1, 9)] + ["circle"]


def get_component_geometry(component_name: str) -> str:
    """Get the expected geometry for a component.

    Args:
        component_name: Component name like "mess3", "mess3_1", "tom_quantum"

    Returns:
        Geometry name like "simplex_2" or "circle"
    """
    if component_name.startswith("mess3"):
        return "simplex_2"
    elif component_name.startswith("tom_quantum"):
        return "circle"
    else:
        return None  # Unknown component


def extract_best_geometries(geometry_refinement: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Extract best geometry and score for each cluster.

    Returns:
        Tuple of (best_geometries_json, best_geometry_scores_json)
        - best_geometries: {cluster_id → geometry_name}
        - best_geometry_scores: {cluster_id → optimal_distance}
    """
    if geometry_refinement is None or not geometry_refinement.get("enabled", False):
        return None, None

    clusters = geometry_refinement.get("clusters", {})
    if not clusters:
        return None, None

    best_geoms = {}
    best_scores = {}

    for cluster_id_str, cluster_data in clusters.items():
        best_geoms[cluster_id_str] = cluster_data.get("best_geometry", None)
        best_scores[cluster_id_str] = cluster_data.get("best_optimal_distance", None)

    return (
        json.dumps(best_geoms) if best_geoms else None,
        json.dumps(best_scores) if best_scores else None
    )


def extract_all_cluster_geometry_fits(geometry_refinement: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract geometry fit data for all clusters as a nested map.

    Args:
        geometry_refinement: The geometry_refinement dict from cluster_summary

    Returns:
        JSON string of structure:
        {
            "0": {
                "best_geometry": "simplex_3",
                "best_distance": 0.456,
                "simplex_1": 0.789,
                "simplex_2": 0.654,
                ...
                "circle": 0.543
            },
            "1": { ... },
            ...
        }
        or None if no geometry refinement
    """
    if geometry_refinement is None or not geometry_refinement.get("enabled", False):
        return None

    clusters = geometry_refinement.get("clusters", {})
    if not clusters:
        return None

    all_dists = {}

    for cluster_id_str, cluster_data in clusters.items():
        cluster_map = {}

        # Best geometry info
        cluster_map["best_geometry"] = cluster_data.get("best_geometry", None)
        cluster_map["best_distance"] = cluster_data.get("best_optimal_distance", None)

        # All geometry distances
        all_fits = cluster_data.get("all_geometry_fits", {})
        for geom_name in GEOMETRIES:
            if geom_name in all_fits:
                optimal_dist = all_fits[geom_name].get("optimal_distance", None)
                cluster_map[geom_name] = optimal_dist
            else:
                cluster_map[geom_name] = None

        all_dists[cluster_id_str] = cluster_map

    return json.dumps(all_dists) if all_dists else None


def extract_component_assignment(assignment_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract component assignment data as maps.

    Args:
        assignment_data: The component_assignment dict (soft or refined)

    Returns:
        Dict with:
            assignments: JSON map of component -> cluster_id (or "noise" -> [list of cluster ids])
            assignment_scores: JSON map of component -> R² score
    """
    result = {
        "assignments": None,
        "assignment_scores": None,
    }

    if assignment_data is None:
        return result

    # Build assignments map
    assignments_map = {}

    # Add component assignments
    assignments = assignment_data.get("assignments", {})
    for component, cluster_id in assignments.items():
        assignments_map[component] = cluster_id

    # Add noise clusters as a list (even if single element)
    noise_clusters = assignment_data.get("noise_clusters", [])
    if noise_clusters:
        assignments_map["noise"] = noise_clusters
    else:
        assignments_map["noise"] = None

    result["assignments"] = json.dumps(assignments_map) if assignments_map else None

    # Build assignment scores map
    assignment_scores = assignment_data.get("assignment_scores", {})
    result["assignment_scores"] = json.dumps(assignment_scores) if assignment_scores else None

    return result


def compute_n_geo_matches(assignments_json: Optional[str], best_geometries_json: Optional[str]) -> Optional[int]:
    """Compute number of components whose assigned cluster matches their expected geometry.

    Args:
        assignments_json: JSON map of component → cluster_id
        best_geometries_json: JSON map of cluster_id → geometry_name

    Returns:
        Integer 0-5 representing number of matches, or None if data missing
    """
    if not assignments_json or not best_geometries_json:
        return None

    assignments = json.loads(assignments_json)
    best_geoms = json.loads(best_geometries_json)

    matches = 0
    for component, cluster_id in assignments.items():
        if component == "noise":
            continue  # Skip noise

        expected_geom = get_component_geometry(component)
        if expected_geom is None:
            continue

        cluster_id_str = str(cluster_id)
        if cluster_id_str in best_geoms:
            if best_geoms[cluster_id_str] == expected_geom:
                matches += 1

    return matches


def extract_belief_cluster_r2(
    belief_cluster_r2_data: Optional[Dict[str, Any]],
    assignments_json: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Extract belief cluster R² metrics.

    Args:
        belief_cluster_r2_data: The belief_cluster_r2_<type> dict
        assignments_json: The <type>_assignments JSON string

    Returns:
        Tuple of (assigned_r2_json, all_r2_json)
        - assigned_r2: {cluster_id → mean_r2} for assigned component only
        - all_r2: {cluster_id → {component → mean_r2}} for all components
    """
    if belief_cluster_r2_data is None:
        return None, None

    # Build all_r2 (all components for all clusters)
    all_r2 = {}
    for cluster_id_str, cluster_data in belief_cluster_r2_data.items():
        component_r2s = {}
        for component, component_data in cluster_data.items():
            if isinstance(component_data, dict) and "mean_r2" in component_data:
                component_r2s[component] = component_data["mean_r2"]
        if component_r2s:
            all_r2[cluster_id_str] = component_r2s

    # Build assigned_r2 (only assigned component per cluster)
    assigned_r2 = {}
    if assignments_json:
        assignments = json.loads(assignments_json)
        for component, cluster_id in assignments.items():
            if component == "noise":
                continue
            cluster_id_str = str(cluster_id)
            if cluster_id_str in belief_cluster_r2_data:
                cluster_data = belief_cluster_r2_data[cluster_id_str]
                if component in cluster_data and isinstance(cluster_data[component], dict):
                    assigned_r2[cluster_id_str] = cluster_data[component].get("mean_r2", None)

    return (
        json.dumps(assigned_r2) if assigned_r2 else None,
        json.dumps(all_r2) if all_r2 else None
    )


def extract_layer_metrics(cluster_summary: Dict[str, Any], layer: str) -> Dict[str, Any]:
    """Extract all metrics for a single layer from cluster_summary.json.

    Returns:
        Dictionary with layer-specific metrics as maps
    """
    metrics = {"layer": layer}

    # Cluster quality metrics
    principal_angles = cluster_summary.get("principal_angles_deg", None)
    metrics["principal_angles_deg"] = json.dumps(principal_angles) if principal_angles is not None else None

    min_principal_angles = cluster_summary.get("min_principal_angles_deg", None)
    metrics["min_principal_angles_deg"] = json.dumps(min_principal_angles) if min_principal_angles is not None else None

    metrics["overall_min_principal_angle_deg"] = cluster_summary.get("overall_min_principal_angle_deg", None)

    within_projection_energy = cluster_summary.get("within_projection_energy", None)
    metrics["within_projection_energy"] = json.dumps(within_projection_energy) if within_projection_energy is not None else None

    metrics["between_projection_energy"] = cluster_summary.get("between_projection_energy", None)
    metrics["energy_contrast_ratio"] = cluster_summary.get("energy_contrast_ratio", None)

    coherence_metrics_hard = cluster_summary.get("coherence_metrics_hard", None)
    metrics["coherence_metrics_hard"] = json.dumps(coherence_metrics_hard) if coherence_metrics_hard is not None else None

    # Component assignments (hard, soft, refined)
    for assignment_type in ["hard", "soft", "refined"]:
        assignment_data = cluster_summary.get(f"component_assignment_{assignment_type}", None)
        extracted = extract_component_assignment(assignment_data)
        metrics[f"{assignment_type}_assignments"] = extracted["assignments"]
        metrics[f"{assignment_type}_assignment_scores"] = extracted["assignment_scores"]

    # Geometry refinement data
    geometry_refinement = cluster_summary.get("geometry_refinement", None)

    # Best geometries and scores
    best_geoms, best_scores = extract_best_geometries(geometry_refinement)
    metrics["best_geometries"] = best_geoms
    metrics["best_geometry_scores"] = best_scores

    # All geometry scores (renamed from all_dists)
    metrics["all_geo_scores"] = extract_all_cluster_geometry_fits(geometry_refinement)

    # n_geo_matches for each assignment type
    for assignment_type in ["hard", "soft", "refined"]:
        metrics[f"n_geo_matches_{assignment_type}"] = compute_n_geo_matches(
            metrics[f"{assignment_type}_assignments"],
            best_geoms
        )

    # Belief cluster R² for each assignment type
    for assignment_type in ["hard", "soft", "refined"]:
        belief_data = cluster_summary.get(f"belief_cluster_r2_{assignment_type}", None)
        assigned_r2, all_r2 = extract_belief_cluster_r2(
            belief_data,
            metrics[f"{assignment_type}_assignments"]
        )
        metrics[f"assigned_belief_cluster_r2_{assignment_type}"] = assigned_r2
        metrics[f"all_belief_cluster_r2_{assignment_type}"] = all_r2

    # Cluster ranks from k-subspaces
    cluster_ranks = cluster_summary.get("cluster_ranks", None)
    metrics["cluster_ranks_k_subspaces"] = json.dumps(cluster_ranks) if cluster_ranks else None

    return metrics


def download_cluster_summary(client: MlflowClient, run_id: str, layer: str) -> Optional[Dict[str, Any]]:
    """Download and parse cluster_summary.json for a specific layer.

    Args:
        client: MLflow client
        run_id: Run ID
        layer: Layer name (e.g., 'layer_0')

    Returns:
        Parsed JSON as dict, or None if artifact not found
    """
    artifact_path = f"{layer}/cluster_summary.json"

    try:
        # Download artifact to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
            with open(local_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not download {artifact_path} for run {run_id}: {e}")
        return None


def extract_run_params(run) -> Dict[str, Any]:
    """Extract configuration parameters from an MLflow run.

    Returns:
        Dictionary of run parameters with fit_mess3_gmg_ prefix removed and layer-specific params filtered out
    """
    params = {}

    # Basic run info (no duplicates)
    params["run_id"] = run.info.run_id
    params["run_name"] = run.info.run_name
    params["start_time"] = datetime.fromtimestamp(run.info.start_time / 1000).isoformat()
    params["status"] = run.info.status

    # Parameters to skip - these are layer-specific and extracted from artifacts
    skip_prefixes = [
        "layer_0_", "layer_1_", "layer_2_", "embeddings_",
        "layer_0/", "layer_1/", "layer_2/", "embeddings/"
    ]

    # Extract all logged parameters
    for key, value in run.data.params.items():
        # Skip layer-specific parameters
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue

        # Remove fit_mess3_gmg_ prefix if present
        clean_key = key.replace("fit_mess3_gmg_", "")

        # Skip if it's a duplicate of basic run info
        if clean_key in ["run_id", "run_name"]:
            continue

        # Try to convert to appropriate type
        try:
            # Try int
            params[clean_key] = int(value)
        except ValueError:
            try:
                # Try float
                params[clean_key] = float(value)
            except ValueError:
                # Keep as string
                params[clean_key] = value

    return params


def main():
    """Main execution function."""
    print("MLflow Hyperparameter Sweep Results Extractor")
    print("=" * 60)

    # Set up MLflow client
    print(f"\nConnecting to MLflow experiment: {EXPERIMENT_NAME}")

    # Check for Databricks environment variables
    if "DATABRICKS_HOST" not in os.environ or "DATABRICKS_TOKEN" not in os.environ:
        print("\nError: DATABRICKS_HOST and DATABRICKS_TOKEN environment variables must be set.")
        print("Set them with:")
        print("  export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'")
        print("  export DATABRICKS_TOKEN='your-token'")
        return

    # Set tracking URI to Databricks
    print(f"Setting tracking URI to 'databricks'...")
    print(f"Databricks host: {os.environ.get('DATABRICKS_HOST')}")
    mlflow.set_tracking_uri("databricks")

    # Initialize client
    try:
        client = MlflowClient()

        # First, list all experiments to help find the right one
        # Use ViewType.ALL to see all experiments (including archived/deleted)
        print("\nSearching for experiments (including all view types)...")
        all_experiments = client.search_experiments(
            view_type=ViewType.ALL,
            max_results=1000
        )

        print(f"\nFound {len(all_experiments)} experiments:")
        matching_experiments = []
        for exp in all_experiments:
            exp_name = exp.name
            lifecycle = exp.lifecycle_stage
            print(f"  - {exp_name} (ID: {exp.experiment_id}, lifecycle: {lifecycle})")
            # Look for experiments containing 'clustering' or 'sweep'
            if 'clustering' in exp_name.lower() or 'sweep' in exp_name.lower():
                matching_experiments.append(exp)

        # Try to find the experiment
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            print(f"\nError: Experiment '{EXPERIMENT_NAME}' not found.")

            if matching_experiments:
                print(f"\nFound {len(matching_experiments)} experiments that might match:")
                for exp in matching_experiments:
                    print(f"  - {exp.name}")
                print(f"\nUpdate EXPERIMENT_NAME in the script to one of the above.")
            else:
                print("\nNo experiments found matching 'clustering' or 'sweep'.")
                print("Update EXPERIMENT_NAME in the script to the correct experiment name.")
            return

        experiment_id = experiment.experiment_id
        print(f"Found experiment ID: {experiment_id}")

    except Exception as e:
        print(f"\nError connecting to MLflow: {e}")
        import traceback
        traceback.print_exc()
        return

    # Search for all runs in the experiment with pagination
    print("\nSearching for runs (handling pagination)...")
    page_token = None
    runs = []
    page_num = 0

    while True:
        page_num += 1
        print(f"  Fetching page {page_num}...", end=" ")

        result = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"],
            max_results=1000,  # Max per page
            page_token=page_token
        )

        runs.extend(result)
        print(f"got {len(result)} runs (total: {len(runs)})")

        # Check if there are more pages
        # The token attribute will be None or empty string when no more pages
        next_token = getattr(result, 'token', None)
        if not next_token:
            print(f"  No more pages (result.token = {repr(next_token)})")
            break

        page_token = next_token
        print(f"  More pages available, continuing...")

    print(f"\nCompleted: Found {len(runs)} total runs across {page_num} page(s)")

    if len(runs) == 0:
        print("No runs found. Exiting.")
        return

    # Process each run
    all_rows = []

    for i, run in enumerate(runs, 1):
        run_id = run.info.run_id
        run_name = run.info.run_name or run_id[:8]

        print(f"\n[{i}/{len(runs)}] Processing run: {run_name}")
        print(f"  Run ID: {run_id}")

        # Extract run parameters (common across all layers)
        run_params = extract_run_params(run)

        # Process each layer
        for layer in LAYERS:
            print(f"  Downloading {layer}/cluster_summary.json...")
            cluster_summary = download_cluster_summary(client, run_id, layer)

            if cluster_summary is None:
                print(f"    Skipping {layer} (artifact not found)")
                continue

            # Extract metrics for this layer
            layer_metrics = extract_layer_metrics(cluster_summary, layer)

            # Combine run params and layer metrics
            row = {**run_params, **layer_metrics}
            all_rows.append(row)

            print(f"    Extracted metrics for {layer}")

    # Convert to DataFrame
    print(f"\n\nCreating DataFrame with {len(all_rows)} rows...")
    df = pd.DataFrame(all_rows)

    # Identify and remove duplicate columns
    # Keep only one version of duplicated info
    all_columns = list(df.columns)
    seen = set()
    cols_to_keep = []

    for col in all_columns:
        # Check for duplicates by comparing actual column values
        if col not in seen:
            cols_to_keep.append(col)
            seen.add(col)

    df = df[cols_to_keep]

    # Sort columns for better readability
    # Order: run info, layer, config params, then metrics
    run_info_cols = ["run_id", "run_name", "start_time", "status"]
    layer_col = ["layer"]

    # Metric columns in logical order
    metric_cols = [
        "energy_contrast_ratio",
        "coherence_metrics_hard",
        # Assignments
        "hard_assignments", "hard_assignment_scores",
        "soft_assignments", "soft_assignment_scores",
        "refined_assignments", "refined_assignment_scores",
        # Geometry info
        "best_geometries", "best_geometry_scores", "all_geo_scores",
        # Computed metrics
        "n_geo_matches_hard", "n_geo_matches_soft", "n_geo_matches_refined",
        # Belief R²
        "assigned_belief_cluster_r2_hard", "all_belief_cluster_r2_hard",
        "assigned_belief_cluster_r2_soft", "all_belief_cluster_r2_soft",
        "assigned_belief_cluster_r2_refined", "all_belief_cluster_r2_refined",
        # Subspace info
        "cluster_ranks_k_subspaces"
    ]

    # Config params are everything else
    config_cols = [col for col in df.columns
                   if col not in run_info_cols + layer_col + metric_cols]

    # Build final column order
    ordered_cols = run_info_cols + layer_col + config_cols + metric_cols
    # Make sure we only use columns that actually exist
    ordered_cols = [col for col in ordered_cols if col in df.columns]
    df = df[ordered_cols]

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"mlflow_sweep_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Total runs: {len(runs)}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nColumn structure:")
    print(f"  Run info: {run_info_cols}")
    print(f"  Layer: {layer_col}")
    print(f"  Config params: {len(config_cols)} columns")
    print(f"  Metrics: {metric_cols}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
