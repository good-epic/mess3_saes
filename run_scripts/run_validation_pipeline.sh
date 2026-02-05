#!/bin/bash

# Run validation pipeline for all clusters with synthesis results
#
# This script:
# 1. Finds all clusters with synthesis results in haiku_512 and haiku_768
# 2. For each cluster, runs:
#    - Step 1: Collect uniform validation samples (loads model + SAE + AANet)
#    - Step 2: Run coherence + baseline in single invocation (loads model once, caches distributions)
#
# Usage:
#   ./run_scripts/run_validation_pipeline.sh

set -e  # Exit on error

export PYTHONPATH=.

# Configuration
SYNTHESIS_DIRS=(
    "outputs/interpretations/haiku_512/synthesis"
    "outputs/interpretations/haiku_768/synthesis"
)
PREPARED_SAMPLES_BASE="outputs/interpretations"
AANET_BASE="outputs/real_data_analysis_canonical"
MANIFEST_BASE="outputs/selected_clusters_canonical"
OUTPUT_BASE="outputs/validation"
UNIFORM_SAMPLES_BASE="outputs/validation_samples"

# Validation sample collection parameters
TARGET_SAMPLES=10000
SKIP_DOCS=400000  # Skip docs used for training

# Model parameters
MODEL_NAME="gemma-2-9b"
SAE_RELEASE="gemma-scope-9b-pt-res"
SAE_ID="layer_20/width_16k/average_l0_68"
DEVICE="cuda"
CACHE_DIR="/workspace/hf_cache"

echo "============================================================"
echo "VALIDATION PIPELINE"
echo "============================================================"
echo "Looking for synthesis results in:"
for dir in "${SYNTHESIS_DIRS[@]}"; do
    echo "  - $dir"
done
echo ""

# Collect all clusters to process
declare -a CLUSTERS_TO_PROCESS

for synth_dir in "${SYNTHESIS_DIRS[@]}"; do
    if [[ -d "$synth_dir" ]]; then
        for synth_file in "$synth_dir"/*_synthesis.json; do
            if [[ -f "$synth_file" ]]; then
                # Extract cluster key from filename (e.g., 512_261_synthesis.json -> 512_261)
                filename=$(basename "$synth_file")
                cluster_key="${filename%_synthesis.json}"
                CLUSTERS_TO_PROCESS+=("$cluster_key")
            fi
        done
    fi
done

echo "Found ${#CLUSTERS_TO_PROCESS[@]} clusters with synthesis results:"
for cluster in "${CLUSTERS_TO_PROCESS[@]}"; do
    echo "  - $cluster"
done
echo ""

# Process each cluster
for cluster_key in "${CLUSTERS_TO_PROCESS[@]}"; do
    echo ""
    echo "============================================================"
    echo "PROCESSING: $cluster_key"
    echo "============================================================"

    # Parse n_clusters and cluster_id
    N_CLUSTERS="${cluster_key%_*}"
    CLUSTER_ID="${cluster_key#*_}"

    echo "  n_clusters: $N_CLUSTERS"
    echo "  cluster_id: $CLUSTER_ID"

    # Get k value from prepared samples
    PREPARED_SAMPLES_DIR="${PREPARED_SAMPLES_BASE}/prepared_samples_${N_CLUSTERS}"
    PREPARED_FILE="${PREPARED_SAMPLES_DIR}/cluster_${cluster_key}.json"

    if [[ ! -f "$PREPARED_FILE" ]]; then
        echo "  WARNING: Prepared samples not found: $PREPARED_FILE"
        echo "  Skipping cluster..."
        continue
    fi

    K=$(python3 -c "import json; d=json.load(open('$PREPARED_FILE')); print(d['k'])")
    echo "  k: $K"

    # Find AANet checkpoint
    AANET_CHECKPOINT="${AANET_BASE}/clusters_${N_CLUSTERS}/aanet_cluster_${CLUSTER_ID}_k${K}.pt"

    if [[ ! -f "$AANET_CHECKPOINT" ]]; then
        echo "  WARNING: AANet checkpoint not found: $AANET_CHECKPOINT"
        echo "  Skipping cluster..."
        continue
    fi

    echo "  AANet checkpoint: $AANET_CHECKPOINT"

    # Output paths
    OUTPUT_DIR="${OUTPUT_BASE}/${cluster_key}"
    UNIFORM_SAMPLES_PATH="${UNIFORM_SAMPLES_BASE}/validation_samples_${cluster_key}.jsonl"
    MANIFEST_PATH="${MANIFEST_BASE}/manifest_${N_CLUSTERS}.json"

    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$UNIFORM_SAMPLES_BASE"

    # ============================================================
    # STEP 1: Collect uniform validation samples (if not already done)
    # ============================================================
    if [[ -f "$UNIFORM_SAMPLES_PATH" ]]; then
        echo ""
        echo "  Step 1: Uniform samples already exist, skipping collection"
        echo "          $UNIFORM_SAMPLES_PATH"
    else
        echo ""
        echo "  Step 1: Collecting uniform validation samples..."
        echo "          Target: $TARGET_SAMPLES samples"
        echo "          Skip docs: $SKIP_DOCS"

        python -u interpretation/collect_validation_samples.py \
            --n_clusters "$N_CLUSTERS" \
            --cluster_id "$CLUSTER_ID" \
            --aanet_checkpoint "$AANET_CHECKPOINT" \
            --cluster_manifest "$MANIFEST_PATH" \
            --output_dir "$UNIFORM_SAMPLES_BASE" \
            --target_samples "$TARGET_SAMPLES" \
            --skip_docs "$SKIP_DOCS" \
            --model_name "$MODEL_NAME" \
            --sae_release "$SAE_RELEASE" \
            --sae_id "$SAE_ID" \
            --device "$DEVICE" \
            --cache_dir "$CACHE_DIR" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"}

        echo "  Step 1 complete!"
    fi

    # ============================================================
    # STEP 2: Run coherence check + baseline visualization
    #         Single invocation: loads model once, caches distributions
    # ============================================================
    COHERENCE_OUTPUT="${OUTPUT_DIR}/coherence"
    BASELINE_OUTPUT="${OUTPUT_DIR}/baseline"

    if [[ -f "${COHERENCE_OUTPUT}/coherence_stats.json" ]] && [[ -f "${BASELINE_OUTPUT}/baseline_summary.json" ]]; then
        echo ""
        echo "  Step 2: Coherence + baseline already complete, skipping"
        echo "          ${COHERENCE_OUTPUT}/coherence_stats.json"
        echo "          ${BASELINE_OUTPUT}/baseline_summary.json"
    else
        echo ""
        echo "  Step 2: Running coherence check + baseline visualization..."

        python -u interpretation/validate_belief_states.py \
            --mode all \
            --n_clusters "$N_CLUSTERS" \
            --cluster_id "$CLUSTER_ID" \
            --prepared_samples_dir "$PREPARED_SAMPLES_DIR" \
            --uniform_samples_path "$UNIFORM_SAMPLES_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL_NAME" \
            --sae_release "$SAE_RELEASE" \
            --sae_id "$SAE_ID" \
            --device "$DEVICE" \
            --cache_dir "$CACHE_DIR" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"}

        echo "  Step 2 complete!"
    fi

    echo ""
    echo "  ✓ Cluster $cluster_key complete!"
    echo "    Results: $OUTPUT_DIR"

done

echo ""
echo "============================================================"
echo "VALIDATION PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Processed ${#CLUSTERS_TO_PROCESS[@]} clusters"
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "To view results:"
echo "  ls -la $OUTPUT_BASE/*/coherence/"
echo "  ls -la $OUTPUT_BASE/*/baseline/"
echo ""
