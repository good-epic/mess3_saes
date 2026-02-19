#!/bin/bash

# Top Predicted Tokens by Vertex
# Qualitative validation: checks whether model predictions at trigger positions
# categorically differ between vertices in a way matching the interpretation.
# Requires GPU for model forward passes (no SAE needed).
# Run from project root: ./run_scripts/run_top_tokens.sh

export PYTHONPATH=.

python -u validation/top_tokens_by_vertex.py \
    --prepared_samples_dir "outputs/interpretations/prepared_samples_current_no_whitespace" \
    --output_dir "outputs/validation/top_tokens" \
    --clusters 512_464,512_504,512_292,768_484 \
    --model_name gemma-2-9b \
    --device cuda \
    --top_k 20 \
    --max_samples_per_vertex 200 \
    --batch_size 16

echo ""
echo "============================================================"
echo "Top Predicted Tokens by Vertex — Complete!"
echo "============================================================"
echo "Results saved to: outputs/validation/top_tokens/"
echo "============================================================"
