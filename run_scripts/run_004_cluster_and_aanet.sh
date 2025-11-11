#!/usr/bin/bash
set -e

START_TIME=$(date +%s)

#./run_fit_mess3_gmg.sh 2>&1
./run_aanet_multipartite.sh 2>&1

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Runtime: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds."
