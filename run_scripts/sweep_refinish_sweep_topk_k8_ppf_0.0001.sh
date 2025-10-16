#!/usr/bin/bash

# Auto-generated on 2025-10-16T22:20:11.148730Z to resume sweep_topk_k8_ppf_0.0001.sh
# START_INDEX indicates how many combinations to skip (0-based).

SAE_TYPE="topk"
SAE_VALUE="8"
PER_POINT_THRESHOLD="0.0001"
START_INDEX=46

# Parameter arrays for sweep (must match original order)
CLUSTER_CONFIGS=("auto_0.83_2.0" "auto_0.83_2.5" "auto_0.9_2.0" "auto_0.9_2.5" "manual_6" "manual_9")
DEDUP_THRESHOLDS=(0.96 0.99)
LATENT_ACTS=(0.01 0.001)
SIM_METRICS=("cosine" "euclidean")

TOTAL_RUNS=48

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

RUN_INDEX=0

echo "Resuming sweep: Top K k=$SAE_VALUE, per-point filtering=$PER_POINT_THRESHOLD"
echo "Total runs: $TOTAL_RUNS (skipping the first $START_INDEX combinations)"
echo "=================================================="

for cluster_config in "${CLUSTER_CONFIGS[@]}"; do
  for dedup in "${DEDUP_THRESHOLDS[@]}"; do
    for latent_act in "${LATENT_ACTS[@]}"; do
      for sim_metric in "${SIM_METRICS[@]}"; do
        if (( RUN_INDEX < START_INDEX )); then
          RUN_INDEX=$((RUN_INDEX + 1))
          continue
        fi

        CURRENT_RUN=$((RUN_INDEX + 1))
        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: cluster=$cluster_config, dedup=$dedup, latent=$latent_act, sim=$sim_metric"

        if [[ $cluster_config == auto_* ]]; then
          IFS='_' read -r _ var_thresh gap_thresh <<< "$cluster_config"
          CLUSTER_ARGS="--subspace_variance_threshold $var_thresh --subspace_gap_threshold $gap_thresh"
          CLUSTER_STR="auto_v${var_thresh}_g${gap_thresh}"
        else
          IFS='_' read -r _ n_clusters <<< "$cluster_config"
          CLUSTER_ARGS="--subspace_n_clusters $n_clusters"
          CLUSTER_STR="manual_n${n_clusters}"
        fi

        RUN_NAME="topk_k${SAE_VALUE}_${CLUSTER_STR}_dedup${dedup}_lat${latent_act}_${sim_metric}_ppf${PER_POINT_THRESHOLD}"
        SAE_ARGS="--sae_type top_k --force_k $SAE_VALUE"

        START_TIME=$(date +%s)
        ${PYTHON_BIN} -u "${SCRIPT_DIR}/../fit_mess3_gmg.py" \
          $SAE_ARGS \
          $CLUSTER_ARGS \
          --sae_folder "/workspace/outputs/saes/multipartite_003e" \
          --metrics_summary "/workspace/outputs/saes/multipartite_003e/metrics_summary.json" \
          --output_dir "/workspace/outputs/reports/multipartite_003e" \
          --model_ckpt "/workspace/outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt" \
          --device "cpu" \
          --seed 43 \
          --process_config "${SCRIPT_DIR}/../process_configs.json" \
          --process_config_name "3xmess3_2xtquant_003" \
          --d_model 128 \
          --n_heads 4 \
          --n_layers 3 \
          --n_ctx 16 \
          --d_head 32 \
          --act_fn "relu" \
          --sim_metric "$sim_metric" \
          --clustering_method "k_subspaces" \
          --min_clusters 6 \
          --max_clusters 12 \
          --cosine_dedup_threshold $dedup \
          --sample_sequences 1024 \
          --max_activations 50000 \
          --cluster_activation_threshold 1e-6 \
          --center_decoder_rows \
          --latent_activity_threshold $latent_act \
          --latent_activation_eps 1e-6 \
          --activation_batches 8 \
          --refine_with_geometries \
          --geo_include_circle \
          --geo_filter_metrics gw_full \
          --geo_sinkhorn_max_iter 5000 \
          --geo_sinkhorn_epsilon 0.2 \
          --geo_threshold_mode raw \
          --geo_per_point_threshold $PER_POINT_THRESHOLD \
          --log_to_mlflow \
          --mlflow_experiment "/Shared/mp_clustering_sweep" \
          --mlflow_run_name_base "$RUN_NAME"

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "Run completed in $((DURATION / 60)) minutes and $((DURATION % 60)) seconds."
        echo "---"

        RUN_INDEX=$((RUN_INDEX + 1))
      done
    done
  done
done

echo ""
echo "=================================================="
echo "Resume sweep finished."
