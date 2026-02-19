run_scripts/run_null_baselines.sh > /workspace/outputs/validation/null_baselines/null_baselines.log 2>&1


# Run 1: no whitespace
python -u validation/top_tokens_by_vertex.py \
    --prepared_samples_dir "outputs/interpretations/prepared_samples_current_no_whitespace" \
    --output_dir "outputs/validation/top_tokens_no_whitespace" \
    --clusters 512_464,512_504,512_292,768_484 \
    --model_name gemma-2-9b --device cuda --top_k 20 --max_samples_per_vertex 200 \
    --batch_size 16 > /workspace/outputs/validation/top_tokens_no_whitespace/top_tokens_no_whitespace.log 2>&1

# Run 2: with whitespace
python -u validation/top_tokens_by_vertex.py \
    --prepared_samples_dir "outputs/interpretations/prepared_samples_current" \
    --output_dir "outputs/validation/top_tokens_with_whitespace" \
    --clusters 512_464,512_504,512_292,768_484 \
    --model_name gemma-2-9b --device cuda --top_k 20 --max_samples_per_vertex 200 \
    --batch_size 16 > /workspace/outputs/validation/top_tokens_with_whitespace/top_tokens_with_whitespace.log 2>&1