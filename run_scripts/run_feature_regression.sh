#!/bin/bash

# Feature Regression Validation
# Tests whether NLP features of trigger words can predict vertex assignment.
# Run from project root: ./run_scripts/run_feature_regression.sh
#
# Requires: spacy with en_core_web_sm model
# No GPU needed — runs on CPU.

export PYTHONPATH=.

# Install spacy if needed
pip install spacy 2>/dev/null
python -m spacy download en_core_web_sm 2>/dev/null

# Run validation on all 4 priority clusters
python -u validation/feature_regression.py \
    --prepared_samples_dir "outputs/interpretations/prepared_samples_current" \
    --output_dir "outputs/validation/feature_regression" \
    --n_folds 5

echo ""
echo "============================================================"
echo "Feature Regression Validation Complete!"
echo "============================================================"
echo "Results saved to: outputs/validation/feature_regression/"
echo "============================================================"
