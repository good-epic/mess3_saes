#!/usr/bin/env python3
"""
Interpret clusters using dual-path analysis with Claude.

For each cluster and each iteration:
1. Sample N examples per vertex
2. Format prompt with samples
3. Call Claude API
4. Parse structured JSON response
5. Save everything (samples, prompt, response, interpretation)
6. Update aggregated summary

Usage:
    python interpret_clusters.py \
        --prepared_samples_dir outputs/interpretations/prepared_samples \
        --prompt_template prompts/holistic_v1.txt \
        --output_dir outputs/interpretations/iterations \
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
from datetime import datetime, timezone
import anthropic


def load_prompt_template(template_path):
    """Load the prompt template from file."""
    print(f"Loading prompt template from {template_path}")
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
        print(f"  Loaded {cluster_key}: k={data['k']}, {data['total_samples_available']} samples")

    print(f"Total clusters loaded: {len(cluster_samples)}")
    return cluster_samples


def sample_vertex_examples(vertex_samples, n_samples, rng):
    """Randomly sample n examples from vertex samples."""
    if len(vertex_samples) <= n_samples:
        # Use all available
        return vertex_samples[:]
    else:
        # Random sample without replacement
        return rng.sample(vertex_samples, n_samples)


def format_prompt_with_samples(template, cluster_data, sampled_examples):
    """Format the prompt template with sampled examples."""
    # Build the samples section
    samples_text = ""
    for vertex_id in sorted(sampled_examples.keys()):
        samples = sampled_examples[vertex_id]
        # Extract just the full_text for each sample
        texts = [s['full_text'] for s in samples]
        # Format as Python list
        samples_text += f"Vertex {vertex_id}: {json.dumps(texts, indent=2)}\n\n"

    # Combine template with samples
    full_prompt = template.strip() + "\n\n" + samples_text.strip()

    return full_prompt


def call_claude_api(prompt, model, api_key):
    """Call Claude API and return response."""
    client = anthropic.Anthropic(api_key=api_key)

    # Map model names
    model_map = {
        'sonnet': 'claude-sonnet-4-5-20250929',
        'haiku': 'claude-haiku-4-5-20251001',
        'opus': 'claude-opus-4-5-20251101',
    }
    model_id = model_map.get(model, model)

    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response


def parse_interpretation(response):
    """Parse the Claude API response and extract structured interpretation."""
    # Get the text content
    text_content = response.content[0].text

    # Try to parse as JSON
    try:
        # Sometimes models wrap JSON in markdown code blocks
        text_content = text_content.strip()
        if text_content.startswith('```'):
            # Remove code fences
            lines = text_content.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines[-1].startswith('```'):
                lines = lines[:-1]
            text_content = '\n'.join(lines)

        interpretation = json.loads(text_content)

        # Validate required fields
        required = ['state_space_hypothesis', 'vertex_labels', 'confidence', 'reasoning']
        for field in required:
            if field not in interpretation:
                return None, f"Missing required field: {field}"

        return interpretation, None

    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"


def save_iteration_results(output_dir, cluster_key, iteration, sampled_examples,
                           full_prompt, response, interpretation, error=None):
    """Save all results from one iteration."""
    # Create directory
    iter_dir = Path(output_dir) / f"iteration_{iteration:03d}" / cluster_key
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Save sampled examples
    samples_path = iter_dir / "samples_sent.json"
    samples_data = {
        'iteration': iteration,
        'cluster_key': cluster_key,
        'samples_per_vertex': len(next(iter(sampled_examples.values()))),
        'vertex_samples': {
            str(v_id): [
                {
                    'sample_id': s.get('sample_id', f'sample_{i}'),
                    'trigger_words': s.get('trigger_words', []),
                    'full_text': s['full_text'],
                    'distance_to_vertex': s.get('distances_to_vertex', [s.get('distance_to_vertex')])[0] if s.get('distances_to_vertex') else s.get('distance_to_vertex')
                }
                for i, s in enumerate(samples)
            ]
            for v_id, samples in sampled_examples.items()
        },
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    with open(samples_path, 'w') as f:
        json.dump(samples_data, f, indent=2)

    # Save full prompt
    prompt_path = iter_dir / "prompt_full.txt"
    with open(prompt_path, 'w') as f:
        f.write(full_prompt)

    # Save raw API response
    response_path = iter_dir / "api_response_raw.json"
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
    with open(response_path, 'w') as f:
        json.dump(response_data, f, indent=2)

    # Save interpretation (or error)
    interp_path = iter_dir / "interpretation.json"
    if error:
        interp_data = {
            'error': error,
            'raw_text': response.content[0].text,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    else:
        interp_data = {
            **interpretation,
            'iteration': iteration,
            'model': response.model,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tokens_used': {
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens
            }
        }
    with open(interp_path, 'w') as f:
        json.dump(interp_data, f, indent=2)

    return iter_dir


def update_aggregated_summary(output_dir, cluster_key, cluster_data, iteration, interpretation, error=None):
    """Update the human-readable aggregated summary for this cluster."""
    summary_dir = Path(output_dir) / "aggregated_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{cluster_key}_summary.txt"

    # Load existing summary or create new
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = f.read()
    else:
        # Create header
        summary = f"CLUSTER {cluster_key} (k={cluster_data['k']})\n"
        summary += "=" * 80 + "\n"
        summary += f"N_latents: {cluster_data['n_latents']}\n"
        summary += f"Category: {cluster_data['category']}\n"
        summary += f"Total samples available: {cluster_data['total_samples_available']}\n"
        summary += "=" * 80 + "\n\n"

        # Initialize sections for each vertex
        for vertex_id in range(cluster_data['k']):
            summary += f"VERTEX {vertex_id} - All interpretations across iterations:\n"
            summary += "-" * 80 + "\n\n"

        summary += f"STATE SPACE - All interpretations across iterations:\n"
        summary += "-" * 80 + "\n\n"

    # Append new iteration results
    if error:
        # Add error notice
        summary += f"  Iteration {iteration}: ERROR - {error}\n"
    else:
        # Find the right place to insert for each vertex
        k = cluster_data['k']

        # We'll rebuild the summary to insert in the right spots
        lines = summary.split('\n')
        new_lines = []

        current_vertex = None
        for line in lines:
            new_lines.append(line)
            # Check if we're at a vertex section
            if line.startswith(f"VERTEX ") and " - All interpretations" in line:
                vertex_id = int(line.split()[1])
                current_vertex = vertex_id
            elif line.startswith("STATE SPACE - All interpretations"):
                current_vertex = 'state_space'
            elif current_vertex is not None and line.strip() == "":
                # End of section, add our new iteration
                if isinstance(current_vertex, int) and current_vertex < len(interpretation['vertex_labels']):
                    new_lines.insert(-1, f"  Iteration {iteration}: \"{interpretation['vertex_labels'][current_vertex]}\"")
                elif current_vertex == 'state_space':
                    new_lines.insert(-1, f"  Iteration {iteration}: \"{interpretation['state_space_hypothesis']}\"")
                current_vertex = None

        summary = '\n'.join(new_lines)

    # Write updated summary
    with open(summary_path, 'w') as f:
        f.write(summary)

    return summary_path


def collect_iteration_results(output_dir, cluster_key, num_iterations):
    """Collect all iteration results for a cluster."""
    results = []
    for iteration in range(num_iterations):
        iter_dir = Path(output_dir) / f"iteration_{iteration:03d}" / cluster_key
        interp_path = iter_dir / "interpretation.json"

        if interp_path.exists():
            with open(interp_path, 'r') as f:
                data = json.load(f)
                if 'error' not in data:
                    results.append({
                        'iteration': iteration,
                        'state_space_hypothesis': data.get('state_space_hypothesis'),
                        'vertex_labels': data.get('vertex_labels', []),
                        'confidence': data.get('confidence'),
                        'reasoning': data.get('reasoning', '')
                    })
    return results


def format_synthesis_prompt(cluster_key, cluster_data, iteration_results):
    """Format a prompt asking the model to synthesize all iteration results."""
    k = cluster_data['k']
    n_latents = cluster_data['n_latents']

    prompt = f"""You previously analyzed a cluster of SAE latents multiple times, each time with different randomly sampled examples near each vertex of a {k}-vertex simplex.

CLUSTER INFO:
- Cluster: {cluster_key}
- Number of vertices (k): {k}
- Number of latents in cluster: {n_latents}

Below are the interpretations from {len(iteration_results)} independent analyses. Each iteration saw different examples but was analyzing the same underlying structure.

"""

    for result in iteration_results:
        prompt += f"ITERATION {result['iteration']}:\n"
        prompt += f"  State space hypothesis: {result['state_space_hypothesis']}\n"
        prompt += f"  Vertex labels: {result['vertex_labels']}\n"
        prompt += f"  Confidence: {result['confidence']}\n"
        prompt += f"  Reasoning: {result['reasoning'][:500]}{'...' if len(result['reasoning']) > 500 else ''}\n\n"

    prompt += """YOUR TASK:
Synthesize these multiple interpretations into a single, consolidated interpretation. Consider:
1. Which interpretations are consistent across iterations?
2. Where there is disagreement, what is the most likely explanation?
3. What is your overall confidence given the consistency (or lack thereof) across iterations?

Note that we are trying to find simplices being leveraged by the model to represent the current belief state across a state space. The examples provided are ones where
one token has activated near a vertex of the simplex. We are hoping that the model is tracking its current belief state in this simplex, and want to use the near-vertex
examples to understand what pure state is represented by each vertex. It should lower our confidence if similar concepts are inferred between iterations but are assigned
to different vertices.

OUTPUT FORMAT (JSON only, no markdown):
{
  "consolidated_state_space": "Your best guess for what belief state or decision space this simplex represents",
  "consolidated_vertex_labels": ["Label for vertex 0", "Label for vertex 1", ...],
  "confidence": "high|medium|low",
  "consistency_assessment": "How consistent were the iterations? What patterns did you notice?",
  "reasoning": "Brief explanation of how you arrived at this consolidated interpretation"
}"""

    return prompt


def run_synthesis(output_dir, cluster_key, cluster_data, num_iterations, model, api_key):
    """Run the final synthesis step for a cluster."""
    print(f"\n  Running synthesis across {num_iterations} iterations...")

    # Collect all iteration results
    iteration_results = collect_iteration_results(output_dir, cluster_key, num_iterations)

    if len(iteration_results) < 2:
        print(f"    Skipping synthesis: only {len(iteration_results)} successful iterations (need at least 2)")
        return None

    print(f"    Found {len(iteration_results)} successful iterations")

    # Format synthesis prompt
    synthesis_prompt = format_synthesis_prompt(cluster_key, cluster_data, iteration_results)
    print(f"    Synthesis prompt length: {len(synthesis_prompt)} chars")

    # Call API
    print(f"    Calling Claude API ({model}) for synthesis...")
    try:
        response = call_claude_api(synthesis_prompt, model, api_key)
        print(f"    Response received: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

        # Parse response
        text_content = response.content[0].text.strip()
        if text_content.startswith('```'):
            lines = text_content.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines[-1].startswith('```'):
                lines = lines[:-1]
            text_content = '\n'.join(lines)

        synthesis = json.loads(text_content)
        print(f"    Synthesis complete!")
        print(f"      Consolidated state space: {synthesis.get('consolidated_state_space', 'N/A')[:80]}...")
        print(f"      Confidence: {synthesis.get('confidence', 'N/A')}")

    except Exception as e:
        print(f"    ERROR in synthesis: {e}")
        synthesis = {'error': str(e)}

    # Save synthesis results
    synthesis_dir = Path(output_dir) / "synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)

    synthesis_path = synthesis_dir / f"{cluster_key}_synthesis.json"
    synthesis_data = {
        'cluster_key': cluster_key,
        'k': cluster_data['k'],
        'n_latents': cluster_data['n_latents'],
        'num_iterations_used': len(iteration_results),
        'iteration_summaries': iteration_results,
        'synthesis': synthesis,
        'model': model,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

    with open(synthesis_path, 'w') as f:
        json.dump(synthesis_data, f, indent=2)

    print(f"    Saved synthesis to {synthesis_path}")

    # Also save the prompt for reference
    prompt_path = synthesis_dir / f"{cluster_key}_synthesis_prompt.txt"
    with open(prompt_path, 'w') as f:
        f.write(synthesis_prompt)

    return synthesis


def process_cluster_iteration(cluster_key, cluster_data, iteration, template,
                              samples_per_vertex, model, api_key, output_dir, rng):
    """Process one iteration for one cluster."""
    print(f"\n  Iteration {iteration}:")

    # Sample examples for each vertex
    sampled_examples = {}
    for vertex_id_str, vertex_samples in cluster_data['vertices'].items():
        vertex_id = int(vertex_id_str)
        sampled = sample_vertex_examples(vertex_samples, samples_per_vertex, rng)
        sampled_examples[vertex_id] = sampled
        print(f"    Vertex {vertex_id}: sampled {len(sampled)} examples")

    # Format prompt
    full_prompt = format_prompt_with_samples(template, cluster_data, sampled_examples)
    print(f"    Prompt length: {len(full_prompt)} chars")

    # Call API
    print(f"    Calling Claude API ({model})...")
    try:
        response = call_claude_api(full_prompt, model, api_key)
        print(f"    Response received: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

        # Parse interpretation
        interpretation, error = parse_interpretation(response)
        if error:
            print(f"    ERROR parsing response: {error}")
        else:
            print(f"    Successfully parsed interpretation")
            print(f"      State space: {interpretation['state_space_hypothesis'][:80]}...")
            print(f"      Confidence: {interpretation['confidence']}")

    except Exception as e:
        print(f"    ERROR calling API: {e}")
        # Create dummy response for error handling
        class DummyResponse:
            id = "error"
            model = model
            role = "assistant"
            content = [type('obj', (object,), {'type': 'text', 'text': str(e)})]
            stop_reason = "error"
            usage = type('obj', (object,), {'input_tokens': 0, 'output_tokens': 0})
        response = DummyResponse()
        interpretation = None
        error = str(e)

    # Save results
    save_iteration_results(output_dir, cluster_key, iteration, sampled_examples,
                          full_prompt, response, interpretation, error)

    # Update aggregated summary
    if interpretation:
        summary_path = update_aggregated_summary(output_dir, cluster_key, cluster_data,
                                                iteration, interpretation, error)
        print(f"    Updated summary: {summary_path}")

    return interpretation, error


def main():
    parser = argparse.ArgumentParser(description="Interpret clusters using Claude API")
    parser.add_argument('--prepared_samples_dir', type=str, required=True,
                        help='Directory containing prepared sample files')
    parser.add_argument('--prompt_template', type=str, required=True,
                        help='Path to prompt template file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save iteration results')
    parser.add_argument('--model', type=str, choices=['sonnet', 'haiku', 'opus'], default='sonnet',
                        help='Claude model to use')
    parser.add_argument('--samples_per_vertex', type=int, required=True,
                        help='Number of samples to use per vertex per iteration')
    parser.add_argument('--num_iterations', type=int, required=True,
                        help='Number of iterations to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--clusters_to_process', type=str, default='all',
                        help='Which clusters to process: "all" or comma-separated like "128_5,256_35"')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last completed iteration')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Must provide --api_key or set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    random.seed(args.seed)

    # Load prompt template
    template = load_prompt_template(args.prompt_template)

    # Load prepared samples
    print("\nLoading prepared samples...")
    all_cluster_samples = load_prepared_samples(args.prepared_samples_dir)

    # Determine which clusters to process
    if args.clusters_to_process == 'all':
        clusters_to_process = list(all_cluster_samples.keys())
    else:
        clusters_to_process = args.clusters_to_process.split(',')
        # Validate
        for cluster_key in clusters_to_process:
            if cluster_key not in all_cluster_samples:
                print(f"ERROR: Cluster {cluster_key} not found in prepared samples")
                sys.exit(1)

    print(f"\nProcessing {len(clusters_to_process)} clusters")

    # Process each cluster
    for cluster_key in clusters_to_process:
        cluster_data = all_cluster_samples[cluster_key]

        print("\n" + "="*80)
        print(f"CLUSTER {cluster_key} (k={cluster_data['k']})")
        print("="*80)

        # Create RNG for this cluster (for reproducibility)
        cluster_seed = args.seed + hash(cluster_key) % (2**31)
        rng = random.Random(cluster_seed)

        # Determine starting iteration (for resume)
        start_iteration = 0
        if args.resume:
            # Check which iterations are already complete
            for i in range(args.num_iterations):
                iter_dir = Path(args.output_dir) / f"iteration_{i:03d}" / cluster_key
                interp_path = iter_dir / "interpretation.json"
                if not interp_path.exists():
                    start_iteration = i
                    break
            else:
                start_iteration = args.num_iterations

            if start_iteration > 0:
                print(f"  Resuming from iteration {start_iteration} (iterations 0-{start_iteration-1} already complete)")

        # Run iterations
        for iteration in range(start_iteration, args.num_iterations):
            process_cluster_iteration(
                cluster_key, cluster_data, iteration, template,
                args.samples_per_vertex, args.model, api_key, args.output_dir, rng
            )

        # Run synthesis if we have multiple iterations
        if args.num_iterations > 1:
            run_synthesis(
                args.output_dir, cluster_key, cluster_data,
                args.num_iterations, args.model, api_key
            )

    # Final summary
    print("\n" + "="*80)
    print("INTERPRETATION COMPLETE")
    print("="*80)
    print(f"Processed {len(clusters_to_process)} clusters")
    print(f"Iterations per cluster: {args.num_iterations}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nAggregated summaries: {args.output_dir}/aggregated_summaries/")
    if args.num_iterations > 1:
        print(f"Synthesis results: {args.output_dir}/synthesis/")


if __name__ == '__main__':
    main()
