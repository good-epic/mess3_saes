# Complete AANet Analysis Pipeline Guide

## Overview

This pipeline analyzes SAE latent clusters using Archetypal Analysis Networks (AANets) across three stages:

1. **Stage 1**: Train AANets for all clusters/k values → Generate analysis plots → Save models
2. **Stage 2**: Manual review of plots → Automated cluster selection
3. **Stage 3**: Load/retrain selected clusters → Collect vertex samples for interpretation

---

## Stage 1: Initial AANet Training & Analysis

### Script: `run_scripts/run_real_data_analysis.sh`

**What it does:**
- Clusters SAE latents using KMeans (n_clusters = 128, 256, 512, 768)
- Trains AANets for each cluster across k=2 to k=8
- Calculates elbow quality metrics (recon/arch elbow k, strength, monotonicity, etc.)
- **Saves AANet models** (~1.6 MB each, ~8 GB total)
- **Auto-generates 5 analysis plots per n_clusters**

**Outputs to:** `/workspace/outputs/real_data_analysis_canonical/clusters_{n}/`

| File | Description |
|------|-------------|
| `consolidated_metrics_n{n}.csv` | Complete metrics CSV (no "corrected" version needed!) |
| `aanet_cluster_{cid}_k{k}.pt` | Trained AANet model weights |
| `training_curves_n{n}.jsonl` | Full training history |
| **`plots/pca_vs_geometric_rank_n{n}.png`** | **NEW**: PCA rank vs Geometric rank scatter |
| `plots/decoder_rank_dist_n{n}.png` | Geometric rank distribution |
| `plots/elbow_strength_dists_n{n}.png` | Recon & Arch elbow strength histograms |
| **`plots/recon_vs_arch_unzoomed_n{n}.png`** | **KEY**: Full elbow strength scatter |
| **`plots/recon_vs_arch_zoomed_n{n}.png`** | **KEY**: Quality-filtered elbow strength scatter |

**Key Features:**
- ✅ Early stopping with loss smoothing (saves ~2/3 of training time!)
- ✅ LR scheduling with ReduceLROnPlateau
- ✅ FineWeb dataset (pre-randomized, better than Pile)
- ✅ Automatic model saving
- ✅ Automatic plot generation

**Run time:** ~8-12 hours for 4 n_clusters values (depends on early stopping)

---

## Stage 2: Manual Review & Selection

### Script: `scratch/analyze_aanet_results.py` (Jupyter notebook)

**What it does:**
- Loads `consolidated_metrics_n{n}.csv` from Stage 1
- Interactive exploration of elbow plots
- **Cluster selection happens AUTOMATICALLY in Stage 3!**

**Purpose:**
- Review auto-generated plots from Stage 1
- Optionally adjust selection thresholds in code before Stage 3
- No manual cluster ID list needed - Stage 3 auto-selects!

**Selection Criteria (Hardcoded in `refit_selected_clusters.py`):**
```python
# Quality filters
- n_latents >= 2
- recon_is_monotonic == True
- arch_is_monotonic == True
- recon_pct_decrease >= 20%
- arch_pct_decrease >= 20%
- |k_differential| <= 1

# Categories (priority: A > D > B > C)
A: Both elbow strengths > mean + 1σ (strong both axes)
B: Recon strength > mean + 3σ (reconstruction outliers)
C: Arch strength > mean + 3σ (archetypal outliers)
D: k_differential == 0 AND both > mean (perfect agreement)
```

**To adjust:** Edit `select_promising_clusters()` function before running Stage 3

---

## Stage 3: Refit Selected + Collect Vertex Samples

### Script: `run_scripts/run_refit_selected_clusters.sh`

**What it does:**
- Loads CSV from Stage 1
- **Automatically selects** clusters using criteria above
- **Two modes:**
  - **MODE 1 (Default)**: Retrains AANets at elbow k → Collects vertex samples
  - **MODE 2 (Skip Training)**: Loads Stage 1 models → Only collects vertex samples

**MODE 1: Normal Workflow (Retrain + Collect)**

```bash
python -u real_data_tests/refit_selected_clusters.py \
    --n_clusters_list "512,768" \
    --csv_dir "/workspace/outputs/real_data_analysis_canonical" \
    --save_dir "/workspace/outputs/selected_clusters_canonical" \
    # ... (AANet training params) ...
    --collect_vertex_samples \
    --samples_per_vertex 1000
```

**When to use:** Default mode. Always works. Retrains for freshness.

**MODE 2: Skip Training (Load Stage 1 Models)**

```bash
python -u real_data_tests/refit_selected_clusters.py \
    --skip_training \
    --stage1_models_dir "/workspace/outputs/real_data_analysis_canonical" \
    --vertex_skip_docs 300000 \
    # ... (fewer params needed) ...
    --collect_vertex_samples
```

**When to use:** When Stage 1 models are good enough. Saves ~2-3 hours.

**Outputs to:** `/workspace/outputs/selected_clusters_canonical/n{n}/`

| File | Description |
|------|-------------|
| `cluster_{cid}_k{k}_category{cat}.pt` | AANet model (retrained or from Stage 1) |
| `cluster_{cid}_k{k}_category{cat}_vertex_samples.jsonl` | **Text examples near each vertex** |
| `cluster_{cid}_k{k}_category{cat}_vertex_stats.json` | Collection statistics |
| `cluster_{cid}_k{k}_category{cat}_metadata.json` | Cluster info + params |
| `cluster_{cid}_k{k}_category{cat}_curves.json` | Training curves (if retrained) |

**Run time:**
- MODE 1: ~4-6 hours (retraining + vertex collection)
- MODE 2: ~2-3 hours (vertex collection only)

---

## Model File Sizes

**Individual AANet:**
- Parameters: ~413K
- Size: ~1.6 MB (float32)

**Full Stage 1 Output:**
- 768 clusters × 7 k values = 5,376 models
- Total: **~8.6 GB** ✅ Very manageable!

**Recommendation:** Always save Stage 1 models. Storage is cheap, retraining is not!

---

## Complete Workflow Example

```bash
# Stage 1: Train everything, save models, generate plots
./run_scripts/run_real_data_analysis.sh
# Outputs: /workspace/outputs/real_data_analysis_canonical/clusters_{128,256,512,768}/
#   - consolidated_metrics_n{n}.csv  (complete metrics, no correction needed!)
#   - aanet_cluster_*_k*.pt          (all trained models)
#   - plots/*.png                     (5 auto-generated analysis plots)

# Stage 2: Review plots
ls /workspace/outputs/real_data_analysis_canonical/clusters_*/plots/
# Key plots to review:
#   - recon_vs_arch_zoomed_n*.png    (shows quality-filtered clusters)
#   - pca_vs_geometric_rank_n*.png   (validates clustering)

# (Optional) Run analyze_aanet_results.py for interactive exploration
# (Optional) Adjust selection thresholds in refit_selected_clusters.py code

# Stage 3: Collect vertex samples (auto-selects clusters)
./run_scripts/run_refit_selected_clusters.sh
# Outputs: /workspace/outputs/selected_clusters_canonical/n{n}/
#   - cluster_*_vertex_samples.jsonl  (text examples for interpretation!)
```

---

## Key Updates from Previous Version

### ✅ **No More "Corrected" CSV**
- Stage 1 now outputs complete metrics directly
- `consolidated_metrics_n{n}.csv` has everything
- No separate correction step needed!

### ✅ **Auto-Generated Plots**
- 5 plots generated automatically after each n_clusters
- No need to run separate analysis script first
- Includes requested PCA vs Geometric rank scatter!

### ✅ **Model Saving**
- All AANet models saved in Stage 1
- Can skip retraining in Stage 3
- Models are tiny (~1.6 MB each)

### ✅ **Automated Selection**
- No manual cluster ID list needed
- Script auto-selects based on metrics
- Adjust thresholds in code if needed

### ✅ **Column Name Consistency**
- All scripts use same column names
- `aanet_recon_loss_elbow_k`, `aanet_recon_loss_elbow_strength`
- `aanet_archetypal_loss_elbow_k`, `aanet_archetypal_loss_elbow_strength`

### ✅ **Early Stopping**
- Implemented in both Stage 1 and Stage 3
- Loss smoothing with interval-based checking
- ~2/3 time savings!

### ✅ **FineWeb Dataset**
- Replaced Pile with FineWeb sample-10BT
- Pre-randomized, better quality
- Simplified data sampling code

---

## Troubleshooting

**"CSV not found" in Stage 3:**
- Check that Stage 1 completed successfully
- Verify `--csv_dir` points to Stage 1 output directory
- Ensure file is named `consolidated_metrics_n{n}.csv` (not `_corrected.csv`)

**"Model not found" with `--skip_training`:**
- Check that `--stage1_models_dir` points to Stage 1 output root
- Models should be at: `{stage1_models_dir}/clusters_{n}/aanet_cluster_{cid}_k{k}.pt`
- If missing, remove `--skip_training` to retrain

**"No clusters selected":**
- Check that elbow metrics were calculated in Stage 1
- CSV should have columns: `aanet_recon_loss_elbow_k`, etc.
- Try relaxing selection thresholds in `select_promising_clusters()` function

**Training too slow:**
- Early stopping should kick in around step 200-300
- If running full 3000 steps, check early stopping params
- Recommended: `--aanet_early_stop_patience 250 --aanet_loss_smoothing_window 20`

---

## File Naming Convention

**Stage 1 outputs:**
- `consolidated_metrics_n{n}.csv`
- `aanet_cluster_{cluster_id}_k{k}.pt`
- `training_curves_n{n}.jsonl`

**Stage 3 outputs:**
- `cluster_{cluster_id}_k{k}_category{A/B/C/D}.pt`
- `cluster_{cluster_id}_k{k}_category{A/B/C/D}_vertex_samples.jsonl`

---

## CSV Column Reference

**Basic Info:**
- `n_clusters_total`, `cluster_id`, `n_latents`, `latent_indices`, `aanet_k`, `aanet_status`

**Training Losses:**
- `aanet_loss`, `aanet_recon_loss`, `aanet_archetypal_loss`, `aanet_extrema_loss`

**Clustering Metrics:**
- `decoder_dir_rank` (geometric rank from clustering)
- `activation_pca_rank` (PCA rank from activations)
- `cluster_recon_error`

**Elbow Metrics:**
- `aanet_recon_loss_elbow_k`, `aanet_recon_loss_elbow_strength`
- `aanet_archetypal_loss_elbow_k`, `aanet_archetypal_loss_elbow_strength`
- `k_differential` (|recon_k - arch_k|)

**Quality Metrics:**
- `recon_is_monotonic`, `arch_is_monotonic`
- `recon_pct_decrease`, `arch_pct_decrease`

**SAE Info:**
- `sae_id`, `average_l0`

---

## Questions?

All three scripts are now fully synchronized with consistent naming and automatic workflows!
