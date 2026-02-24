#!/bin/bash

# Feature regression (1c) on broad_2 clusters
# Run from project root: ./run_scripts/run_feature_regression_broad_2.sh

export PYTHONPATH=.

NEW_CLUSTERS="512_181,768_140,512_67,768_596"

echo "Running feature regression on broad_2 clusters (no whitespace)..."
python -u validation/feature_regression.py \
    --prepared_samples_dir outputs/interpretations/prepared_samples_broad_2_no_whitespace \
    --output_dir outputs/validation/feature_regression_broad_2_no_whitespace \
    --clusters "${NEW_CLUSTERS}"

echo ""
echo "Running feature regression on broad_2 clusters (with whitespace)..."
python -u validation/feature_regression.py \
    --prepared_samples_dir outputs/interpretations/prepared_samples_broad_2 \
    --output_dir outputs/validation/feature_regression_broad_2 \
    --clusters "${NEW_CLUSTERS}"

echo ""
echo "Done. Results in outputs/validation/feature_regression_broad_2*/"
