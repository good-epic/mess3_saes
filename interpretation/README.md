# Cluster Interpretation Pipeline

Tools for interpreting AANet simplex clusters using dual-path LLM analysis.

## Overview

This pipeline analyzes vertex samples from trained AANet models using two independent approaches:
- **Path A (Holistic)**: Analyze all vertices together to identify the state space
- **Path B (Compositional)**: Analyze each vertex independently, then synthesize

Comparing both paths helps validate interpretations and identify high-confidence vs uncertain clusters.

## Workflow

### 1. Prepare Samples

Load vertex samples from completed AANet runs and organize for interpretation.

```bash
python prepare_vertex_samples.py \
  --manifest /workspace/outputs/selected_clusters_canonical/manifest.json \
  --output_dir outputs/interpretations/prepared_samples \
  --max_samples_per_vertex 1000
```

**Outputs:**
- `prepared_samples/cluster_{n}_{id}.json` - One file per cluster with all vertex samples

### 2. Run Dual-Path Interpretations

For each cluster and iteration:
- Sample N examples per vertex (shared across both paths)
- **Path A**: Send all vertices together → get state space + labels
- **Path B**:
  - Send each vertex independently → get 2 label proposals per vertex
  - Send all proposals together → synthesize into state space + labels
- Save everything
- Update aggregated summaries

```bash
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
```

**Required Arguments:**
- `--prepared_samples_dir`: Directory with prepared sample files
- `--path_a_template`: Prompt for holistic analysis (all vertices together)
- `--path_b_vertex_template`: Prompt for per-vertex analysis
- `--path_b_synthesis_template`: Prompt for synthesizing vertex proposals
- `--output_dir`: Where to save results
- `--samples_per_vertex`: How many examples per vertex per iteration
- `--num_iterations`: How many iterations to run

**Optional Arguments:**
- `--analysis_mode`: Which paths to run - choices: `path_a`, `path_b`, `both` (default: `both`)
- `--model`: Claude model - choices: `sonnet`, `haiku`, `opus` (default: `sonnet`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--clusters_to_process`: "all" or comma-separated like "128_5,256_35" (default: `all`)
- `--resume`: Resume from last completed iteration
- `--api_key`: Anthropic API key (or set ANTHROPIC_API_KEY env var)

**Outputs:**

Per-iteration detailed results:
```
iterations/
├── iteration_000/
│   ├── cluster_128_5/
│   │   ├── samples_sent.json          # Samples used (shared across paths)
│   │   ├── path_a/
│   │   │   ├── prompt_full.txt
│   │   │   ├── api_response_raw.json
│   │   │   └── interpretation.json    # {state_space_hypothesis, vertex_labels[], ...}
│   │   └── path_b/
│   │       ├── vertex_0/
│   │       │   ├── samples_sent.json
│   │       │   ├── prompt_full.txt
│   │       │   ├── api_response_raw.json
│   │       │   └── proposals.json     # {vertex_label_candidates[], confidence[], ...}
│   │       ├── vertex_1/...
│   │       ├── vertex_2/...
│   │       └── synthesis/
│   │           ├── input_proposals.json
│   │           ├── prompt_full.txt
│   │           ├── api_response_raw.json
│   │           └── interpretation.json  # Final synthesized interpretation
│   └── cluster_256_35/...
└── iteration_001/...
```

Aggregated human-readable summaries:
```
iterations/aggregated_summaries/
├── cluster_128_5_path_a.txt          # Path A results across all iterations
├── cluster_128_5_path_b.txt          # Path B results (proposals + synthesis)
└── cluster_128_5_comparison.txt      # Side-by-side Path A vs Path B
```

**Example Path A summary (`cluster_128_5_path_a.txt`):**
```
CLUSTER 128_5 - PATH A (HOLISTIC ANALYSIS)
================================================================================
k=3, N_latents=124, Category=D
================================================================================

VERTEX 0 - All interpretations across iterations:
--------------------------------------------------------------------------------
  Iteration 0: "Past tense verbs and completed actions"
  Iteration 1: "Historical events and past references"
  Iteration 2: "Past tense verbs and completed actions"
  ...

VERTEX 1 - All interpretations across iterations:
--------------------------------------------------------------------------------
  Iteration 0: "Present tense and ongoing actions"
  ...

STATE SPACE - All interpretations across iterations:
--------------------------------------------------------------------------------
  Iteration 0: "Temporal/tense belief state for verb forms"
  Iteration 1: "Temporal reference distribution"
  ...
```

**Example Path B summary (`cluster_128_5_path_b.txt`):**
```
CLUSTER 128_5 - PATH B (COMPOSITIONAL ANALYSIS)
================================================================================
k=3, N_latents=124, Category=D
================================================================================

ITERATION 0:
--------------------------------------------------------------------------------
  Vertex 0 proposals:
    Candidate 1: "Past tense verbs" (confidence: high)
    Candidate 2: "Completed actions" (confidence: medium)
  Vertex 1 proposals:
    Candidate 1: "Present tense" (confidence: high)
    Candidate 2: "Ongoing actions" (confidence: high)
  Vertex 2 proposals:
    Candidate 1: "Future tense" (confidence: high)
    Candidate 2: "Planned actions" (confidence: medium)

  Synthesis:
    State space: "Temporal/tense belief state"
    Vertex labels: ["Past tense/completed actions", "Present tense/ongoing", "Future tense/planned"]
    Confidence: high

ITERATION 1:
--------------------------------------------------------------------------------
...
```

**Example comparison summary (`cluster_128_5_comparison.txt`):**
```
CLUSTER 128_5 - PATH A vs PATH B COMPARISON
================================================================================

ITERATION 0:
--------------------------------------------------------------------------------
Path A (Holistic):
  State space: "Temporal/tense belief state for verb forms"
  Vertex 0: "Past tense verbs and completed actions"
  Vertex 1: "Present tense and ongoing actions"
  Vertex 2: "Future tense and planned actions"
  Confidence: high

Path B (Compositional → Synthesis):
  State space: "Temporal/tense belief state"
  Vertex 0: "Past tense/completed actions"
  Vertex 1: "Present tense/ongoing"
  Vertex 2: "Future tense/planned"
  Confidence: high

Agreement: ✓ Exact match

ITERATION 1:
--------------------------------------------------------------------------------
...
```

## Prompt Templates

Three templates required for dual-path analysis:

1. **detailed_all_vertices.txt** (Path A)
   - Input: Samples from all k vertices
   - Output: `{state_space_hypothesis, vertex_labels[], confidence, reasoning}`

2. **detailed_one_vertex.txt** (Path B step 1)
   - Input: Samples from one vertex
   - Output: `{vertex_label_candidates[2], confidence[2], reasoning[2]}`

3. **detailed_synthesis.txt** (Path B step 2)
   - Input: All vertex proposals from step 1
   - Output: `{state_space_hypothesis, vertex_labels[], confidence, reasoning}`

All templates are in `prompt_templates/`.

## Analysis Modes

### `--analysis_mode both` (recommended)
- Runs both Path A and Path B
- Enables comparison to validate interpretations
- Higher cost but better confidence

### `--analysis_mode path_a`
- Only holistic analysis
- Faster, cheaper
- Good for quick initial exploration

### `--analysis_mode path_b`
- Only compositional analysis
- Useful for debugging vertex-level patterns
- Can run independently of Path A

## Cost Estimation

For 19 clusters (k=3 to k=5) using Sonnet:

| Mode | Samples/Vertex | Iterations | Cost per Iteration | Total Cost |
|------|----------------|------------|-------------------|------------|
| `path_a` | 20 | 5 | ~$0.54 | ~$2.70 |
| `path_b` | 20 | 5 | ~$1.21 | ~$6.05 |
| `both` | 20 | 5 | ~$1.75 | ~$8.75 |
| `both` | 20 | 10 | ~$1.75 | ~$17.50 |
| `both` | 20 | 30 | ~$1.75 | ~$52.50 |

Path B is more expensive because it makes k+1 API calls per cluster (k vertices + synthesis) vs 1 for Path A.

## Resumption

Use `--resume` flag to continue from last completed iteration. The script checks for:
- Path A: `path_a/interpretation.json` exists
- Path B: `path_b/synthesis/interpretation.json` exists
- Both: Both files exist

## Interpreting Results

### High Confidence Clusters
- Both paths converge to similar interpretations
- Consistent labels across iterations
- Clear semantic patterns in samples

### Low Confidence Clusters
- Paths disagree significantly
- Labels vary across iterations
- May need manual review or different sampling

### Red Flags
- Synthesis has low confidence and notes "proposals don't combine well"
- Path A and Path B give completely different state spaces
- Many JSON parsing errors (prompt may need adjustment)

## Notes

- All parameters are CLI arguments for reproducibility
- Random seed ensures reproducible sampling within cluster
- Samples drawn without replacement within iteration
- Same samples may appear across iterations
- JSON parsing failures saved with errors for manual review
- Paths run sequentially (Path A → Path B) to avoid rate limits
