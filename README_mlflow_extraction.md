# MLflow Hyperparameter Sweep Results Extraction

This document explains how to extract results from the clustering hyperparameter sweep stored in MLflow.

## Script: `extract_mlflow_sweep_results.py`

Downloads cluster geometry metrics from all runs in the MLflow experiment and saves them to a CSV file.

### Setup

1. **Install required packages** (if not already installed):
```bash
pip install mlflow pandas
```

2. **Set Databricks credentials**:
```bash
export DATABRICKS_HOST='https://your-workspace.databricks.com'
export DATABRICKS_TOKEN='your-databricks-token'
```

To get your Databricks token:
- Go to your Databricks workspace
- Click on your profile icon (top right)
- Settings → Developer → Access Tokens
- Generate New Token

### Usage

```bash
cd /home/mattylev/projects/simplex/SAEs/mess3_sae
python extract_mlflow_sweep_results.py
```

The script will:
1. Connect to MLflow experiment `/Shared/mp_clustering_sweep`
2. Search for all runs
3. Download `layer_0/cluster_summary.json`, `layer_1/cluster_summary.json`, and `layer_2/cluster_summary.json` for each run
4. Extract metrics and save to `mlflow_sweep_results_{timestamp}.csv`

### Output Format

**One row per layer per run** (3 rows per run):

#### Configuration Columns (repeated for all 3 rows per run)
- `run_id`: MLflow run ID
- `run_name`: Run name (encodes hyperparameters)
- `start_time`: When run started
- `status`: Run status (FINISHED, FAILED, etc.)
- Run parameters: `sae_type`, `force_k`, `force_lambda`, `clustering_method`, `cosine_dedup_threshold`, `latent_activity_threshold`, `sim_metric`, `geo_per_point_threshold`, etc.

#### Layer-Specific Columns
- `layer`: Layer index (0, 1, or 2)
- `energy_contrast_ratio`: Within/between cluster energy ratio

#### Component Assignment (Soft)
- `soft_assignments_{component}`: Cluster ID assigned to each component
- `soft_assignment_scores_{component}`: R² score for each component
- `soft_noise_clusters`: JSON list of noise cluster IDs

#### Component Assignment (Refined - after geometry filtering)
- `refined_assignments_{component}`: Cluster ID after geometry refinement
- `refined_assignment_scores_{component}`: R² score after refinement
- `refined_noise_clusters`: JSON list of noise cluster IDs after refinement

#### Geometry Fitting (per cluster)
For each cluster N:
- `cluster_N_best_geometry`: Best-fitting geometry name (e.g., "simplex_3", "circle")
- `cluster_N_best_optimal_distance`: Gromov-Wasserstein distance for best fit
- `cluster_N_simplex_1_dist` through `cluster_N_simplex_8_dist`: GW distance to each simplex geometry
- `cluster_N_circle_dist`: GW distance to circle geometry

### Testing Locally

Test the extraction logic on a local file:
```bash
python test_extraction_logic.py
```

This verifies the extraction functions work correctly before running on the full MLflow dataset.

## Notes

- The script handles missing artifacts gracefully (e.g., if a layer wasn't processed)
- If `component_assignment_refined` doesn't exist (geometry refinement disabled), those columns will be empty
- Cluster IDs are 0-indexed
- The number of clusters per layer may vary depending on the clustering method and parameters

## Example Analysis Queries

Once you have the CSV, you can analyze it with pandas:

```python
import pandas as pd

df = pd.read_csv("mlflow_sweep_results_20241017_120000.csv")

# Filter to layer 0 only
layer0 = df[df['layer'] == 0]

# Find runs with best geometry fit for cluster 0
best_fits = layer0.nsmallest(10, 'cluster_0_best_optimal_distance')

# Compare soft vs refined assignment scores
comparison = df[['run_name', 'layer',
                 'soft_assignment_scores_mess3',
                 'refined_assignment_scores_mess3']]

# Group by hyperparameter
by_sae_type = df.groupby('sae_type')['energy_contrast_ratio'].mean()
```
