#!/usr/bin/env python3
"""
Interpret clusters using dual-path analysis with Claude.

PATH A (Holistic): All vertices together → state space + labels
PATH B (Compositional): Each vertex → proposals → synthesis → state space + labels

Usage:
    python interpret_clusters_dual_path.py \
        --prepared_samples_dir outputs/interpretations/prepared_samples \
        --path_a_template prompt_templates/detailed_all_vertices.txt \
        --path_b_vertex_template prompt_templates/detailed_one_vertex.txt \
        --path_b_synthesis_template prompt_templates/detailed_synthesis.txt \
        --output_dir outputs/interpretations/iterations \
        --analysis_mode both \
        --model sonnet \
        --samples_per_vertex 20 \
        --num_iterations 5 \
        --seed 42
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
import anthropic


def load_prompt_template(template_path):
    """Load the prompt template from file."""
    with open(template_path, 'r') as f:
        template = f.read()
    return template


def load_prepared_samples(samples_dir):
    """Load all prepared sample files."""
    samples_dir = Path(samples_dir)
    cluster_samples = {}

    for sample_file in sorted(samples_dir.glob("cluster_*.json")):
        with open(sample_file, 'r') as f:
            data = json.load(f)

        cluster_key = f"{data['n_clusters']}_{data['cluster_id']}"
        cluster_samples[cluster_key] = data

    return cluster_samples


def sample_vertex_examples(vertex_samples, n_samples, rng):
    """Randomly sample n examples from vertex samples."""
    if len(vertex_samples) <= n_samples:
        return vertex_samples[:]
    else:
        return rng.sample(vertex_samples, n_samples)


def format_prompt_all_vertices(template, sampled_examples):
    """Format prompt with all vertices' samples (Path A)."""
    samples_text = ""
    for vertex_id in sorted(sampled_examples.keys()):
        samples = sampled_examples[vertex_id]
        texts = [s['full_text'] for s in samples]
        samples_text += f"Vertex {vertex_id}: {json.dumps(texts, indent=2)}\n\n"

    return template.strip() + "\n\n" + samples_text.strip()


def format_prompt_one_vertex(template, vertex_id, vertex_samples):
    """Format prompt with one vertex's samples (Path B step 1)."""
    texts = [s['full_text'] for s in vertex_samples]
    samples_text = f"Vertex {vertex_id}: {json.dumps(texts, indent=2)}"

    return template.strip() + "\n\n" + samples_text.strip()


def format_prompt_synthesis(template, vertex_proposals):
    """Format prompt with all vertex proposals for synthesis (Path B step 2)."""
    proposals_text = "VERTEX PROPOSALS FROM INDEPENDENT ANALYSIS:\n\n"

    for vertex_id in sorted(vertex_proposals.keys()):
        proposals = vertex_proposals[vertex_id]
        proposals_text += f"Vertex {vertex_id}:\n"
        proposals_text += json.dumps(proposals, indent=2) + "\n\n"

    return template.strip() + "\n\n" + proposals_text.strip()


def call_claude_api(prompt, model, api_key):
    """Call Claude API and return response."""
    client = anthropic.Anthropic(api_key=api_key)

    model_map = {
        'sonnet': 'claude-sonnet-4-5-20250929',
        'haiku': 'claude-3-5-haiku-20241022',
        'opus': 'claude-opus-4-5-20251101',
    }
    model_id = model_map.get(model, model)

    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    return response


def parse_json_response(response, expected_fields):
    """Parse Claude response as JSON, handling markdown code blocks."""
    text_content = response.content[0].text.strip()

    # Remove markdown code fences if present
    if text_content.startswith('```'):
        lines = text_content.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines[-1].startswith('```'):
            lines = lines[:-1]
        text_content = '\n'.join(lines).strip()

    try:
        parsed = json.loads(text_content)

        # Validate required fields
        for field in expected_fields:
            if field not in parsed:
                return None, f"Missing required field: {field}"

        return parsed, None

    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"


def save_api_call_results(output_dir, prompt, response, parsed, error):
    """Save prompt, raw response, and parsed output."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full prompt
    with open(output_dir / "prompt_full.txt", 'w') as f:
        f.write(prompt)

    # Save raw API response
    response_data = {
        'id': response.id,
        'model': response.model,
        'role': response.role,
        'content': [{'type': c.type, 'text': c.text} for c in response.content],
        'stop_reason': response.stop_reason,
        'usage': {
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens
        }
    }
    with open(output_dir / "api_response_raw.json", 'w') as f:
        json.dump(response_data, f, indent=2)

    # Save parsed output or error
    if error:
        result = {
            'error': error,
            'raw_text': response.content[0].text,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    else:
        result = {
            **parsed,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'model': response.model,
            'tokens_used': {
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens
            }
        }

    return result


def process_path_a(cluster_key, sampled_examples, template, model, api_key, output_dir, iteration):
    """Process Path A: Holistic analysis with all vertices."""
    print(f"    PATH A: Holistic analysis...")

    path_dir = output_dir / f"iteration_{iteration:03d}" / cluster_key / "path_a"

    # Format prompt
    prompt = format_prompt_all_vertices(template, sampled_examples)

    # Call API
    try:
        response = call_claude_api(prompt, model, api_key)
        parsed, error = parse_json_response(response,
            ['state_space_hypothesis', 'vertex_labels', 'confidence', 'reasoning'])

        if error:
            print(f"      ERROR parsing response: {error}")
        else:
            print(f"      Success: {parsed['state_space_hypothesis'][:60]}...")
            print(f"      Confidence: {parsed['confidence']}")

    except Exception as e:
        print(f"      ERROR calling API: {e}")
        class DummyResponse:
            id = "error"
            model = model
            role = "assistant"
            content = [type('obj', (object,), {'type': 'text', 'text': str(e)})]
            stop_reason = "error"
            usage = type('obj', (object,), {'input_tokens': 0, 'output_tokens': 0})
        response = DummyResponse()
        parsed = None
        error = str(e)

    # Save results
    result = save_api_call_results(path_dir, prompt, response, parsed, error)

    with open(path_dir / "interpretation.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result, error


def process_path_b_vertex(vertex_id, vertex_samples, template, model, api_key, output_dir):
    """Process one vertex for Path B step 1."""
    print(f"      Vertex {vertex_id}...")

    vertex_dir = output_dir / f"vertex_{vertex_id}"

    # Save vertex samples
    samples_data = [
        {
            'sample_id': s.get('sample_id', f'sample_{i}'),
            'trigger_word': s['trigger_word'],
            'full_text': s['full_text']
        }
        for i, s in enumerate(vertex_samples)
    ]
    with open(vertex_dir / "samples_sent.json", 'w') as f:
        json.dump(samples_data, f, indent=2)

    # Format prompt
    prompt = format_prompt_one_vertex(template, vertex_id, vertex_samples)

    # Call API
    try:
        response = call_claude_api(prompt, model, api_key)
        parsed, error = parse_json_response(response,
            ['vertex_label_candidates', 'confidence', 'reasoning'])

        if error:
            print(f"        ERROR: {error}")
        else:
            print(f"        Candidates: {parsed['vertex_label_candidates']}")

    except Exception as e:
        print(f"        ERROR: {e}")
        class DummyResponse:
            id = "error"
            model = model
            role = "assistant"
            content = [type('obj', (object,), {'type': 'text', 'text': str(e)})]
            stop_reason = "error"
            usage = type('obj', (object,), {'input_tokens': 0, 'output_tokens': 0})
        response = DummyResponse()
        parsed = None
        error = str(e)

    # Save results
    result = save_api_call_results(vertex_dir, prompt, response, parsed, error)

    with open(vertex_dir / "proposals.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result, error


def process_path_b_synthesis(vertex_proposals, template, model, api_key, output_dir):
    """Process Path B step 2: Synthesize vertex proposals."""
    print(f"      Synthesis...")

    synth_dir = output_dir / "synthesis"

    # Save input proposals
    with open(synth_dir / "input_proposals.json", 'w') as f:
        json.dump(vertex_proposals, f, indent=2)

    # Format prompt
    prompt = format_prompt_synthesis(template, vertex_proposals)

    # Call API
    try:
        response = call_claude_api(prompt, model, api_key)
        parsed, error = parse_json_response(response,
            ['state_space_hypothesis', 'vertex_labels', 'confidence', 'reasoning'])

        if error:
            print(f"        ERROR: {error}")
        else:
            print(f"        Success: {parsed['state_space_hypothesis'][:60]}...")
            print(f"        Confidence: {parsed['confidence']}")

    except Exception as e:
        print(f"        ERROR: {e}")
        class DummyResponse:
            id = "error"
            model = model
            role = "assistant"
            content = [type('obj', (object,), {'type': 'text', 'text': str(e)})]
            stop_reason = "error"
            usage = type('obj', (object,), {'input_tokens': 0, 'output_tokens': 0})
        response = DummyResponse()
        parsed = None
        error = str(e)

    # Save results
    result = save_api_call_results(synth_dir, prompt, response, parsed, error)

    with open(synth_dir / "interpretation.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result, error


def process_path_b(cluster_key, cluster_data, sampled_examples, vertex_template, synthesis_template,
                   model, api_key, output_dir, iteration):
    """Process Path B: Compositional analysis (per-vertex → synthesis)."""
    print(f"    PATH B: Compositional analysis...")

    path_dir = output_dir / f"iteration_{iteration:03d}" / cluster_key / "path_b"

    # Step 1: Process each vertex independently
    print(f"      Step 1: Per-vertex analysis...")
    vertex_proposals = {}

    for vertex_id, vertex_samples in sampled_examples.items():
        result, error = process_path_b_vertex(
            vertex_id, vertex_samples, vertex_template, model, api_key, path_dir
        )

        if not error:
            vertex_proposals[vertex_id] = result

    # Step 2: Synthesis (only if we have proposals for all vertices)
    print(f"      Step 2: Synthesis...")
    k = cluster_data['k']

    if len(vertex_proposals) == k:
        synth_result, synth_error = process_path_b_synthesis(
            vertex_proposals, synthesis_template, model, api_key, path_dir
        )
        return synth_result, vertex_proposals, synth_error
    else:
        print(f"        SKIPPING: Not all vertices have proposals ({len(vertex_proposals)}/{k})")
        return None, vertex_proposals, "Incomplete vertex proposals"


def update_aggregated_summary_path_a(output_dir, cluster_key, cluster_data, iteration, result, error):
    """Update Path A aggregated summary."""
    summary_dir = Path(output_dir) / "aggregated_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{cluster_key}_path_a.txt"

    # Load or create summary
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            lines = f.readlines()
    else:
        lines = []
        lines.append(f"CLUSTER {cluster_key} - PATH A (HOLISTIC ANALYSIS)\n")
        lines.append("=" * 80 + "\n")
        lines.append(f"k={cluster_data['k']}, N_latents={cluster_data['n_latents']}, Category={cluster_data['category']}\n")
        lines.append("=" * 80 + "\n\n")

        # Add sections for each vertex
        for v_id in range(cluster_data['k']):
            lines.append(f"VERTEX {v_id} - All interpretations across iterations:\n")
            lines.append("-" * 80 + "\n\n")

        lines.append("STATE SPACE - All interpretations across iterations:\n")
        lines.append("-" * 80 + "\n\n")

    # Append new iteration
    if error:
        new_line = f"  Iteration {iteration}: ERROR - {error}\n"
    else:
        # We'll insert at appropriate places - rebuild
        lines_str = ''.join(lines)

        # Add to each vertex section
        for v_id in range(cluster_data['k']):
            marker = f"VERTEX {v_id} - All interpretations across iterations:\n"
            if marker in lines_str and v_id < len(result.get('vertex_labels', [])):
                label = result['vertex_labels'][v_id]
                insert_line = f"  Iteration {iteration}: \"{label}\"\n"
                # Find where to insert (after the dashed line)
                idx = lines_str.index(marker) + len(marker) + 81  # +81 for dashed line
                lines_str = lines_str[:idx] + insert_line + lines_str[idx:]

        # Add to state space section
        if 'state_space_hypothesis' in result:
            marker = "STATE SPACE - All interpretations across iterations:\n"
            if marker in lines_str:
                insert_line = f"  Iteration {iteration}: \"{result['state_space_hypothesis']}\"\n"
                idx = lines_str.index(marker) + len(marker) + 81
                lines_str = lines_str[:idx] + insert_line + lines_str[idx:]

        lines = [lines_str]

    with open(summary_path, 'w') as f:
        f.writelines(lines)

    return summary_path


def update_aggregated_summary_path_b(output_dir, cluster_key, cluster_data, iteration,
                                     vertex_proposals, synth_result, error):
    """Update Path B aggregated summary."""
    summary_dir = Path(output_dir) / "aggregated_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{cluster_key}_path_b.txt"

    # This is more complex - need to show both per-vertex proposals and final synthesis
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            content = f.read()
    else:
        content = f"CLUSTER {cluster_key} - PATH B (COMPOSITIONAL ANALYSIS)\n"
        content += "=" * 80 + "\n"
        content += f"k={cluster_data['k']}, N_latents={cluster_data['n_latents']}, Category={cluster_data['category']}\n"
        content += "=" * 80 + "\n\n"

    # Append new iteration results
    content += f"\nITERATION {iteration}:\n"
    content += "-" * 80 + "\n"

    # Per-vertex proposals
    for v_id in sorted(vertex_proposals.keys()):
        proposal = vertex_proposals[v_id]
        if 'error' not in proposal:
            content += f"  Vertex {v_id} proposals:\n"
            for i, (label, conf) in enumerate(zip(proposal['vertex_label_candidates'], proposal['confidence'])):
                content += f"    Candidate {i+1}: \"{label}\" (confidence: {conf})\n"
        else:
            content += f"  Vertex {v_id}: ERROR - {proposal['error']}\n"

    content += "\n"

    # Synthesis result
    if error:
        content += f"  Synthesis: ERROR - {error}\n"
    elif synth_result:
        content += f"  Synthesis:\n"
        content += f"    State space: \"{synth_result['state_space_hypothesis']}\"\n"
        content += f"    Vertex labels: {synth_result['vertex_labels']}\n"
        content += f"    Confidence: {synth_result['confidence']}\n"

    content += "\n"

    with open(summary_path, 'w') as f:
        f.write(content)

    return summary_path


def update_comparison_summary(output_dir, cluster_key, cluster_data, iteration, path_a_result, path_b_result):
    """Update comparison summary showing Path A vs Path B."""
    summary_dir = Path(output_dir) / "aggregated_summaries"
    summary_path = summary_dir / f"{cluster_key}_comparison.txt"

    if summary_path.exists():
        with open(summary_path, 'r') as f:
            content = f.read()
    else:
        content = f"CLUSTER {cluster_key} - PATH A vs PATH B COMPARISON\n"
        content += "=" * 80 + "\n"
        content += f"k={cluster_data['k']}, N_latents={cluster_data['n_latents']}, Category={cluster_data['category']}\n"
        content += "=" * 80 + "\n\n"

    content += f"\nITERATION {iteration}:\n"
    content += "-" * 80 + "\n"

    # Path A
    if path_a_result and 'error' not in path_a_result:
        content += "Path A (Holistic):\n"
        content += f"  State space: \"{path_a_result['state_space_hypothesis']}\"\n"
        for i, label in enumerate(path_a_result['vertex_labels']):
            content += f"  Vertex {i}: \"{label}\"\n"
        content += f"  Confidence: {path_a_result['confidence']}\n\n"
    else:
        content += "Path A: ERROR\n\n"

    # Path B
    if path_b_result and 'error' not in path_b_result:
        content += "Path B (Compositional → Synthesis):\n"
        content += f"  State space: \"{path_b_result['state_space_hypothesis']}\"\n"
        for i, label in enumerate(path_b_result['vertex_labels']):
            content += f"  Vertex {i}: \"{label}\"\n"
        content += f"  Confidence: {path_b_result['confidence']}\n\n"
    else:
        content += "Path B: ERROR\n\n"

    # Simple agreement check
    if (path_a_result and path_b_result and
        'error' not in path_a_result and 'error' not in path_b_result):
        # Check if labels are similar (exact match or close)
        agrees = path_a_result['vertex_labels'] == path_b_result['vertex_labels']
        content += f"Agreement: {'✓ Exact match' if agrees else '✗ Different labels'}\n"

    content += "\n"

    with open(summary_path, 'w') as f:
        f.write(content)

    return summary_path


def process_cluster_iteration(cluster_key, cluster_data, iteration, templates, samples_per_vertex,
                              model, api_key, output_dir, rng, analysis_mode):
    """Process one iteration for one cluster."""
    print(f"\n  Iteration {iteration}:")

    iter_dir = Path(output_dir) / f"iteration_{iteration:03d}" / cluster_key

    # Sample examples (shared across both paths)
    sampled_examples = {}
    for vertex_id_str, vertex_samples in cluster_data['vertices'].items():
        vertex_id = int(vertex_id_str)
        sampled = sample_vertex_examples(vertex_samples, samples_per_vertex, rng)
        sampled_examples[vertex_id] = sampled
        print(f"    Vertex {vertex_id}: sampled {len(sampled)} examples")

    # Save shared samples
    samples_data = {
        'iteration': iteration,
        'cluster_key': cluster_key,
        'samples_per_vertex': samples_per_vertex,
        'vertex_samples': {
            v_id: [
                {
                    'sample_id': s.get('sample_id', f'sample_{i}'),
                    'trigger_word': s['trigger_word'],
                    'full_text': s['full_text'],
                    'distance_to_vertex': s['distance_to_vertex']
                }
                for i, s in enumerate(samples)
            ]
            for v_id, samples in sampled_examples.items()
        },
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    iter_dir.mkdir(parents=True, exist_ok=True)
    with open(iter_dir / "samples_sent.json", 'w') as f:
        json.dump(samples_data, f, indent=2)

    # Process paths
    path_a_result = None
    path_b_result = None
    vertex_proposals = None

    if analysis_mode in ['path_a', 'both']:
        path_a_result, path_a_error = process_path_a(
            cluster_key, sampled_examples, templates['path_a'],
            model, api_key, Path(output_dir), iteration
        )

        # Update Path A summary
        if path_a_result:
            update_aggregated_summary_path_a(
                output_dir, cluster_key, cluster_data, iteration,
                path_a_result, path_a_error if path_a_error else None
            )

    if analysis_mode in ['path_b', 'both']:
        path_b_result, vertex_proposals, path_b_error = process_path_b(
            cluster_key, cluster_data, sampled_examples,
            templates['path_b_vertex'], templates['path_b_synthesis'],
            model, api_key, Path(output_dir), iteration
        )

        # Update Path B summary
        if vertex_proposals:
            update_aggregated_summary_path_b(
                output_dir, cluster_key, cluster_data, iteration,
                vertex_proposals, path_b_result, path_b_error if path_b_error else None
            )

    # Update comparison summary (if both paths ran)
    if analysis_mode == 'both' and path_a_result and path_b_result:
        update_comparison_summary(
            output_dir, cluster_key, cluster_data, iteration,
            path_a_result, path_b_result
        )


def main():
    parser = argparse.ArgumentParser(description="Dual-path cluster interpretation")
    parser.add_argument('--prepared_samples_dir', type=str, required=True)
    parser.add_argument('--path_a_template', type=str, required=True,
                        help='Prompt template for Path A (all vertices together)')
    parser.add_argument('--path_b_vertex_template', type=str, required=True,
                        help='Prompt template for Path B step 1 (one vertex)')
    parser.add_argument('--path_b_synthesis_template', type=str, required=True,
                        help='Prompt template for Path B step 2 (synthesis)')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--analysis_mode', type=str, choices=['path_a', 'path_b', 'both'], default='both')
    parser.add_argument('--model', type=str, choices=['sonnet', 'haiku', 'opus'], default='sonnet')
    parser.add_argument('--samples_per_vertex', type=int, required=True)
    parser.add_argument('--num_iterations', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clusters_to_process', type=str, default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--api_key', type=str, default=None)

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Must provide --api_key or set ANTHROPIC_API_KEY")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # Load templates
    print("\nLoading prompt templates...")
    templates = {
        'path_a': load_prompt_template(args.path_a_template),
        'path_b_vertex': load_prompt_template(args.path_b_vertex_template),
        'path_b_synthesis': load_prompt_template(args.path_b_synthesis_template),
    }
    print(f"  Path A: {args.path_a_template}")
    print(f"  Path B (vertex): {args.path_b_vertex_template}")
    print(f"  Path B (synthesis): {args.path_b_synthesis_template}")

    # Load prepared samples
    print("\nLoading prepared samples...")
    all_cluster_samples = load_prepared_samples(args.prepared_samples_dir)
    print(f"  Loaded {len(all_cluster_samples)} clusters")

    # Determine clusters to process
    if args.clusters_to_process == 'all':
        clusters_to_process = list(all_cluster_samples.keys())
    else:
        clusters_to_process = args.clusters_to_process.split(',')

    print(f"\nProcessing {len(clusters_to_process)} clusters in {args.analysis_mode} mode")

    # Process each cluster
    for cluster_key in clusters_to_process:
        cluster_data = all_cluster_samples[cluster_key]

        print("\n" + "="*80)
        print(f"CLUSTER {cluster_key} (k={cluster_data['k']})")
        print("="*80)

        cluster_seed = args.seed + hash(cluster_key) % (2**31)
        rng = random.Random(cluster_seed)

        # Determine starting iteration
        start_iteration = 0
        if args.resume:
            for i in range(args.num_iterations):
                iter_dir = Path(args.output_dir) / f"iteration_{i:03d}" / cluster_key
                # Check if both paths complete (if mode=both)
                path_a_done = (iter_dir / "path_a" / "interpretation.json").exists()
                path_b_done = (iter_dir / "path_b" / "synthesis" / "interpretation.json").exists()

                if args.analysis_mode == 'both' and not (path_a_done and path_b_done):
                    start_iteration = i
                    break
                elif args.analysis_mode == 'path_a' and not path_a_done:
                    start_iteration = i
                    break
                elif args.analysis_mode == 'path_b' and not path_b_done:
                    start_iteration = i
                    break
            else:
                start_iteration = args.num_iterations

            if start_iteration > 0:
                print(f"  Resuming from iteration {start_iteration}")

        # Run iterations
        for iteration in range(start_iteration, args.num_iterations):
            process_cluster_iteration(
                cluster_key, cluster_data, iteration, templates,
                args.samples_per_vertex, args.model, api_key,
                args.output_dir, rng, args.analysis_mode
            )

    print("\n" + "="*80)
    print("INTERPRETATION COMPLETE")
    print("="*80)
    print(f"Mode: {args.analysis_mode}")
    print(f"Summaries: {args.output_dir}/aggregated_summaries/")


if __name__ == '__main__':
    main()
