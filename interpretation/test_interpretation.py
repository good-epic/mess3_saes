#!/usr/bin/env python3
"""
Minimal test script for interpretation pipeline.
Runs a few API calls so you can manually review prompts and responses.

Usage:
    python test_interpretation.py --cluster cluster_256_245
    python test_interpretation.py --cluster 256_245
    python test_interpretation.py  # auto-selects first available cluster
"""

import argparse
import json
import os
import random
from pathlib import Path
from anthropic import Anthropic

def load_cluster_samples(prepared_samples_dir, cluster_key):
    """Load prepared samples for a specific cluster."""
    sample_file = Path(prepared_samples_dir) / f"{cluster_key}.json"
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_file}")

    with open(sample_file, 'r') as f:
        return json.load(f)

def sample_examples(vertex_samples, n_samples, seed=42):
    """Sample n examples from vertex samples."""
    random.seed(seed)
    if len(vertex_samples) <= n_samples:
        return vertex_samples
    return random.sample(vertex_samples, n_samples)

def format_samples_for_prompt(samples):
    """Format samples for prompt."""
    formatted = []
    for i, sample in enumerate(samples, 1):
        text = sample['full_text']
        trigger_indices = sample.get('trigger_token_indices', [sample.get('trigger_token_index')])
        trigger_words = sample.get('trigger_words', [sample.get('trigger_word')])

        # Show trigger positions
        if isinstance(trigger_indices, list):
            triggers = ", ".join([f"'{w}' at pos {idx}" for w, idx in zip(trigger_words, trigger_indices)])
        else:
            triggers = f"'{trigger_words}' at pos {trigger_indices}"

        formatted.append(f"Example {i}:\nTrigger token(s): {triggers}\nText: {text}\n")

    return "\n".join(formatted)

def format_path_a_prompt(template_path, cluster_data, sampled_examples):
    """Format Path A prompt (all vertices together)."""
    with open(template_path, 'r') as f:
        template = f.read()

    # Build vertex sections
    vertex_sections = []
    for vertex_id in sorted(sampled_examples.keys()):
        samples = sampled_examples[vertex_id]
        formatted = format_samples_for_prompt(samples)
        vertex_sections.append(f"VERTEX {vertex_id} SAMPLES:\n{formatted}")

    all_vertices_text = "\n\n".join(vertex_sections)

    # Add cluster metadata
    k = cluster_data['k']
    n_latents = cluster_data['n_latents']
    category = cluster_data['category']

    prompt = template.replace("SAMPLES BELOW:", f"SAMPLES BELOW:\n\nCluster info: k={k}, n_latents={n_latents}, category={category}\n\n{all_vertices_text}")

    return prompt

def format_path_b_vertex_prompt(template_path, vertex_id, samples):
    """Format Path B vertex prompt (single vertex)."""
    with open(template_path, 'r') as f:
        template = f.read()

    formatted = format_samples_for_prompt(samples)
    prompt = template.replace("SAMPLES BELOW:", f"SAMPLES BELOW:\n\nVERTEX {vertex_id} SAMPLES:\n{formatted}")

    return prompt

def call_claude_api(prompt, api_key, model="sonnet"):
    """Call Claude API."""
    client = Anthropic(api_key=api_key)

    model_map = {
        "sonnet": "claude-sonnet-4-20250514",
        "opus": "claude-opus-4-20250514",
        "haiku": "claude-3-5-haiku-20241022"
    }

    response = client.messages.create(
        model=model_map[model],
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

def parse_json_response(response_text):
    """Try to parse JSON from response."""
    try:
        # Try direct parse
        return json.loads(response_text), None
    except json.JSONDecodeError:
        # Try to find JSON in markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1)), None
            except json.JSONDecodeError as e:
                return None, f"JSON in code block invalid: {e}"
        return None, f"No valid JSON found in response"

def main():
    parser = argparse.ArgumentParser(description="Minimal test for interpretation pipeline")
    parser.add_argument('--cluster', type=str, default=None,
                        help='Cluster to test (e.g., "cluster_256_245" or "256_245")')
    parser.add_argument('--samples_per_vertex', type=int, default=10,
                        help='Number of samples per vertex (default: 10)')
    parser.add_argument('--model', type=str, default='sonnet',
                        choices=['sonnet', 'haiku', 'opus'],
                        help='Model to use (default: sonnet)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_iterations', type=int, default=1,
                        help='Number of iterations (each makes 3 API calls) (default: 1)')
    args = parser.parse_args()

    # Configuration
    PREPARED_SAMPLES_DIR = "outputs/interpretations/prepared_samples"
    OUTPUT_DIR = "outputs/interpretations/test_run"
    PATH_A_TEMPLATE = "interpretation/prompt_templates/detailed_all_vertices.txt"
    PATH_B_VERTEX_TEMPLATE = "interpretation/prompt_templates/detailed_one_vertex.txt"
    SAMPLES_PER_VERTEX = args.samples_per_vertex
    MODEL = args.model
    SEED = args.seed

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        return

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MINIMAL INTERPRETATION TEST")
    print("="*80)
    print(f"Samples per vertex: {SAMPLES_PER_VERTEX}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Find available clusters
    prepared_dir = Path(PREPARED_SAMPLES_DIR)
    if not prepared_dir.exists():
        print(f"ERROR: Prepared samples directory not found: {prepared_dir}")
        print("Run prepare_vertex_samples.py first!")
        return

    available_clusters = sorted([f.stem for f in prepared_dir.glob("cluster_*.json")])
    if not available_clusters:
        print(f"ERROR: No prepared cluster files found in {prepared_dir}")
        return

    print(f"Available clusters: {available_clusters}")

    # Select cluster(s) to test
    if args.cluster:
        # Normalize cluster name (allow "256_245" or "cluster_256_245")
        cluster_key = args.cluster
        if not cluster_key.startswith("cluster_"):
            cluster_key = f"cluster_{cluster_key}"

        if cluster_key not in available_clusters:
            print(f"ERROR: Cluster '{cluster_key}' not found")
            print(f"Available: {available_clusters}")
            return

        test_clusters = [cluster_key]
    else:
        # Auto-select first available cluster
        test_clusters = [available_clusters[0]]

    print(f"Testing with clusters: {test_clusters}")
    print()

    # Process each test cluster
    for cluster_key in test_clusters:
        print("="*80)
        print(f"Testing: {cluster_key}")
        print("="*80)

        # Load cluster data
        cluster_data = load_cluster_samples(PREPARED_SAMPLES_DIR, cluster_key)
        k = cluster_data['k']
        vertices_data = cluster_data['vertices']

        # Run multiple iterations
        for iteration in range(args.num_iterations):
            iteration_seed = SEED + iteration  # Different seed each iteration

            if args.num_iterations > 1:
                print(f"\n  --- Iteration {iteration + 1}/{args.num_iterations} (seed={iteration_seed}) ---")

            # Sample examples for each vertex (handle string keys and missing vertices)
            sampled_examples = {}
            missing_vertices = []
            for vertex_id in range(k):
                # Try both string and int keys
                vertex_key = str(vertex_id)
                if vertex_key in vertices_data:
                    vertex_entry = vertices_data[vertex_key]
                    # Handle both formats: list of samples or dict with 'samples' key
                    if isinstance(vertex_entry, list):
                        vertex_samples = vertex_entry
                    elif isinstance(vertex_entry, dict) and 'samples' in vertex_entry:
                        vertex_samples = vertex_entry['samples']
                    else:
                        vertex_samples = []

                    if len(vertex_samples) > 0:
                        sampled = sample_examples(vertex_samples, SAMPLES_PER_VERTEX, iteration_seed)
                        sampled_examples[vertex_id] = sampled
                        print(f"    Vertex {vertex_id}: {len(sampled)} samples")
                    else:
                        missing_vertices.append(vertex_id)
                        print(f"    Vertex {vertex_id}: ⚠ WARNING - 0 samples available, skipping")
                elif vertex_id in vertices_data:
                    # Int key exists
                    vertex_entry = vertices_data[vertex_id]
                    if isinstance(vertex_entry, list):
                        vertex_samples = vertex_entry
                    elif isinstance(vertex_entry, dict) and 'samples' in vertex_entry:
                        vertex_samples = vertex_entry['samples']
                    else:
                        vertex_samples = []

                    if len(vertex_samples) > 0:
                        sampled = sample_examples(vertex_samples, SAMPLES_PER_VERTEX, iteration_seed)
                        sampled_examples[vertex_id] = sampled
                        print(f"    Vertex {vertex_id}: {len(sampled)} samples")
                    else:
                        missing_vertices.append(vertex_id)
                        print(f"    Vertex {vertex_id}: ⚠ WARNING - 0 samples available, skipping")
                else:
                    missing_vertices.append(vertex_id)
                    print(f"    Vertex {vertex_id}: ⚠ WARNING - vertex not found in data, skipping")

            if missing_vertices:
                print(f"\n    ⚠ WARNING: {len(missing_vertices)}/{k} vertices missing: {missing_vertices}")
                print(f"    Proceeding with {len(sampled_examples)} available vertices...")

            if len(sampled_examples) == 0:
                print(f"    ERROR: No vertices have samples, skipping iteration")
                continue

            print()

            # Create cluster/iteration output directory
            if args.num_iterations > 1:
                cluster_output = output_dir / cluster_key / f"iteration_{iteration:03d}"
            else:
                cluster_output = output_dir / cluster_key
            cluster_output.mkdir(parents=True, exist_ok=True)

            # Save sampled examples
            with open(cluster_output / "samples_used.json", 'w') as f:
                json.dump(sampled_examples, f, indent=2)

            # ===== PATH A: Holistic Analysis =====
            print("    Running Path A (all vertices together)...")
            path_a_dir = cluster_output / "path_a"
            path_a_dir.mkdir(exist_ok=True)

            path_a_prompt = format_path_a_prompt(PATH_A_TEMPLATE, cluster_data, sampled_examples)

            with open(path_a_dir / "prompt.txt", 'w') as f:
                f.write(path_a_prompt)

            print(f"      Calling Claude API ({MODEL})...")
            path_a_response = call_claude_api(path_a_prompt, api_key, MODEL)

            with open(path_a_dir / "response.txt", 'w') as f:
                f.write(path_a_response)

            path_a_parsed, path_a_error = parse_json_response(path_a_response)
            if path_a_error:
                print(f"      ⚠ JSON parsing failed: {path_a_error}")
                with open(path_a_dir / "parse_error.txt", 'w') as f:
                    f.write(path_a_error)
            else:
                print(f"      ✓ Parsed successfully")
                with open(path_a_dir / "parsed.json", 'w') as f:
                    json.dump(path_a_parsed, f, indent=2)

                # Print summary
                print(f"      State space: {path_a_parsed.get('state_space_hypothesis', 'N/A')}")
                print(f"      Confidence: {path_a_parsed.get('confidence', 'N/A')}")

            print()

            # ===== PATH B: Per-Vertex Analysis =====
            print("    Running Path B (per-vertex)...")
            path_b_dir = cluster_output / "path_b"
            path_b_dir.mkdir(exist_ok=True)

            # Only test first 2 available vertices to save API calls
            available_vertex_ids = sorted(sampled_examples.keys())
            test_vertices = available_vertex_ids[:min(2, len(available_vertex_ids))]

            for vertex_id in test_vertices:
                print(f"      Vertex {vertex_id}...")
                vertex_dir = path_b_dir / f"vertex_{vertex_id}"
                vertex_dir.mkdir(exist_ok=True)

                vertex_prompt = format_path_b_vertex_prompt(
                    PATH_B_VERTEX_TEMPLATE,
                    vertex_id,
                    sampled_examples[vertex_id]
                )

                with open(vertex_dir / "prompt.txt", 'w') as f:
                    f.write(vertex_prompt)

                print(f"        Calling Claude API ({MODEL})...")
                vertex_response = call_claude_api(vertex_prompt, api_key, MODEL)

                with open(vertex_dir / "response.txt", 'w') as f:
                    f.write(vertex_response)

                vertex_parsed, vertex_error = parse_json_response(vertex_response)
                if vertex_error:
                    print(f"        ⚠ JSON parsing failed: {vertex_error}")
                    with open(vertex_dir / "parse_error.txt", 'w') as f:
                        f.write(vertex_error)
                else:
                    print(f"        ✓ Parsed successfully")
                    with open(vertex_dir / "parsed.json", 'w') as f:
                        json.dump(vertex_parsed, f, indent=2)

                    # Print proposals
                    candidates = vertex_parsed.get('vertex_label_candidates', [])
                    if candidates:
                        print(f"        Proposal 1: {candidates[0]}")
                        if len(candidates) > 1:
                            print(f"        Proposal 2: {candidates[1]}")

            print()

    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"Review outputs in: {OUTPUT_DIR}")
    print()
    print("To inspect:")
    print(f"  - Prompts: {OUTPUT_DIR}/cluster_*/path_*/prompt.txt")
    print(f"  - Responses: {OUTPUT_DIR}/cluster_*/path_*/response.txt")
    print(f"  - Parsed JSON: {OUTPUT_DIR}/cluster_*/path_*/parsed.json")
    print()

if __name__ == "__main__":
    main()
