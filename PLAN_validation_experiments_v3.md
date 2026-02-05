# Implementation Plan v3: Belief State Validation Experiments

## Overview

Create a script `interpretation/validate_belief_states.py` that validates whether identified simplices represent meaningful belief states.

**Core approach**:
- KL divergence as primary metric
- Validate that near-vertex samples are actually coherent before using averages
- RGB visualization for k=3 (use top-3 vertices for k>3)
- New FineWeb sampling run needed for uniform simplex coverage

---

## Approach 0: Baseline Validation & Visualization

### Step 0a: Intra-Vertex Coherence Check

**Goal**: Before using average distributions as references, verify that samples near the same vertex actually have similar next-token distributions.

#### Process

```
For each vertex v:
    1. Collect all samples near vertex v (already have these)
    2. For each sample, get next-token probability distribution
    3. Compute pairwise KL divergence between all pairs (or sample 500 pairs if many samples)
    4. Store the distribution of KL values

For uniform/random samples:
    1. Run new FineWeb sampling to get points across the simplex
    2. For each sample, get next-token distribution
    3. Sample 5000 random pairs
    4. Compute KL divergence for each pair
    5. Store the distribution of KL values
```

#### Key Comparison

If vertices are meaningful:
- Intra-vertex KL divergences should be **small** (samples near same vertex are similar)
- Random-pair KL divergences should be **larger** (samples from different regions differ)

#### Statistics to Compute

For each distribution, compute and store:
```python
stats = {
    'min': ...,
    'max': ...,
    'p1': ...,    # 1st percentile
    'p25': ...,   # 25th percentile (Q1)
    'p50': ...,   # 50th percentile (median)
    'p75': ...,   # 75th percentile (Q3)
    'p99': ...,   # 99th percentile
    'mean': ...,
    'variance': ...,
    'n_pairs': ...,  # How many pairs computed
    'raw_values': [...],  # Keep all values for visualization
}
```

#### Visualization: KL Distribution Grid

For k=3, create a 2x2 grid of histograms:
```
+------------------+------------------+
|  Vertex 0 pairs  |  Vertex 1 pairs  |
|  (intra-V0 KL)   |  (intra-V1 KL)   |
+------------------+------------------+
|  Vertex 2 pairs  |  Random pairs    |
|  (intra-V2 KL)   |  (uniform KL)    |
+------------------+------------------+
```

**Critical**: Same x-axis and y-axis scales across all panels for visual comparison.

For k>3, use top-3 vertices by sample count, plus random pairs (still 2x2 grid).

#### CLI Arguments for Coherence Check

```
--mode coherence
--max_pairs_per_vertex 500  # Sample this many pairs if more available
--random_pairs 5000  # How many random pairs from uniform sample
--uniform_samples_path PATH  # Path to uniform samples (from separate collection run)
```

---

### Step 0b: Reference Distribution Computation

**Only proceed if coherence check passes** (intra-vertex KL << random-pair KL).

```python
def compute_reference_distribution(samples_near_vertex, model, tokenizer):
    """
    Compute average next-token distribution for samples near a vertex.
    """
    distributions = []
    for sample in samples_near_vertex:
        logits = model(sample.input_ids)
        probs = F.softmax(logits[:, -1, :], dim=-1)  # Next-token probs
        distributions.append(probs)

    # Average (in probability space, not log space)
    reference = torch.stack(distributions).mean(dim=0)
    return reference
```

---

### Step 0c: RGB Gradient Visualization

#### Process

```
1. Use near-vertex samples + uniform samples (combined)
2. For each sample:
   a. Get simplex coordinates (barycentric) from AANet
   b. Get next-token distribution
   c. Compute KL to each vertex's reference distribution
3. Create two scatter plots:
   a. Coordinate-based coloring: RGB from barycentric coords
   b. KL-based coloring: RGB from inverse KL (normalized)
```

#### Coordinate-to-RGB Mapping (k=3)

```python
def barycentric_to_rgb(coords):
    """
    coords: (n_samples, 3) barycentric coordinates summing to 1
    returns: (n_samples, 3) RGB values in [0, 1]

    Vertex 0 -> Red   (1, 0, 0)
    Vertex 1 -> Green (0, 1, 0)
    Vertex 2 -> Blue  (0, 0, 1)
    Center   -> Gray  (0.33, 0.33, 0.33)
    """
    return coords  # Direct mapping works!
```

#### KL-to-RGB Mapping

```python
def kl_to_rgb(kl_to_each_vertex):
    """
    kl_to_each_vertex: (n_samples, 3) KL divergence to each vertex's reference
    returns: (n_samples, 3) RGB values

    Low KL to vertex 0 -> more Red
    Low KL to vertex 1 -> more Green
    Low KL to vertex 2 -> more Blue
    """
    # Convert KL to "similarity" (inverse, with softmax for normalization)
    similarity = 1.0 / (kl_to_each_vertex + epsilon)
    rgb = F.softmax(similarity, dim=-1)  # Normalize to sum to 1
    return rgb
```

#### For k > 3

Select the 3 vertices with the most near-vertex samples. Ignore other vertices for RGB visualization. (Most k>3 clusters have at least one vertex with few/no samples anyway.)

```python
def select_top_3_vertices(samples_by_vertex):
    """Select 3 vertices with most samples."""
    counts = {v: len(samples) for v, samples in samples_by_vertex.items()}
    top_3 = sorted(counts.keys(), key=lambda v: counts[v], reverse=True)[:3]
    return top_3
```

---

## Approach 1: Uniform Simplex Sampling (New Data Collection)

**This is a prerequisite** - need to collect samples across the full simplex, not just near vertices.

### Process

```
1. Stream through FineWeb (same as vertex collection, but different filtering)
2. For each token where cluster is active:
   a. Encode through AANet to get simplex coordinates
   b. Keep the sample (no distance threshold - we want the full distribution)
3. Save samples with their simplex coordinates
```

### Sampling Strategy

- **Don't filter by distance to vertex** - we want uniform coverage
- Optionally stratify: ensure we get samples from all regions of the simplex
- Target: ~5000-10000 samples for good coverage

### CLI for Collection Script

This might be a separate script or a mode in the existing refit script:

```
--mode collect_uniform
--target_samples 10000
--skip_docs 300000  # Skip same docs used for vertex collection (avoid overlap)
--save_interval 1000
```

### Output Format

```json
{
    "sample_id": 12345,
    "full_text": "...",
    "trigger_words": ["the", "patient"],
    "trigger_word_indices": [45, 46],
    "simplex_coords": [0.4, 0.35, 0.25],
    "distances_to_vertices": [0.52, 0.48, 0.61]
}
```

---

## Approach 2: Next-Token Distribution Analysis

### UMAP Visualization

**On token selection for UMAP input**:

Variance-based selection is one option but may not be standard. Alternatives:
1. **Top-k by mean probability**: Tokens that are frequently predicted
2. **Top-k by variance**: Tokens with high variability (your suggestion)
3. **Union of top-k per sample**: Ensures all "important" tokens included
4. **Log-probability space**: Use log-probs directly, clip very small values

Recommend: Start with top-1000 by variance, but make it configurable. Can experiment.

```
--distribution_tokens 1000  # How many tokens to include
--token_selection variance  # variance, mean_prob, or full
```

### UMAP Embedding

```python
def embed_distributions(distributions, n_tokens=1000, selection='variance'):
    """
    Embed next-token distributions into 2D via UMAP.

    distributions: (n_samples, vocab_size) probability distributions
    """
    # Select tokens
    if selection == 'variance':
        variances = distributions.var(dim=0)
        top_tokens = variances.argsort(descending=True)[:n_tokens]
    elif selection == 'mean_prob':
        means = distributions.mean(dim=0)
        top_tokens = means.argsort(descending=True)[:n_tokens]

    # Subset to selected tokens
    reduced = distributions[:, top_tokens]

    # Run UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding = reducer.fit_transform(reduced.numpy())

    return embedding
```

### Visualization

Plot UMAP embedding colored by:
1. Simplex coordinates (RGB)
2. Nearest vertex (categorical colors)
3. Distance from nearest vertex (gradient)

---

## Approach 3: Steering

### Steering Vector Computation

```python
def compute_steering_vector(aanet, sae, source_vertex, target_vertex, cluster_indices):
    """
    Compute steering vector in residual space.

    direction = target_vertex_embedding - source_vertex_embedding
    (projected through AANet decoder then SAE decoder)
    """
    k = aanet.k

    # One-hot simplex coords
    source_coords = torch.zeros(k)
    source_coords[source_vertex] = 1.0
    target_coords = torch.zeros(k)
    target_coords[target_vertex] = 1.0

    # Decode through AANet to cluster latent space
    source_latents = aanet.decode(source_coords)
    target_latents = aanet.decode(target_coords)
    direction_latents = target_latents - source_latents

    # Project to full SAE latent space (zeros elsewhere)
    full_direction = torch.zeros(sae.num_latents)
    full_direction[cluster_indices] = direction_latents

    # Decode through SAE to residual space
    steering_vector = sae.decode(full_direction)

    return steering_vector
```

### Measurements

1. **KL to reference distributions**:
   - KL(original || ref_v) for each vertex v
   - KL(steered || ref_v) for each vertex v
   - Did we get closer to target, further from source?

2. **Top-K probability shifts**:
   - Tokens with largest |P(steered) - P(original)|

3. **Strength sweep**:
   - Test strengths: 0.1, 0.25, 0.5, 1.0, 2.0
   - Plot KL vs strength curves

### CLI Arguments

```
--mode steering
--source_vertex 0
--target_vertex 1
--steering_strengths 0.1,0.25,0.5,1.0,2.0
--steering_mode interpolate  # add, interpolate, replace
--steer_positions last  # last, all, or specific indices
--num_samples 50
--top_k_shifts 30
```

---

## Approach 4: Activation Patching

### Process

```python
def run_patching_experiment(source_sample, target_sample, cluster_indices, model, sae, layer=20):
    """
    Patch cluster activations from target into source.
    """
    # Get residuals at layer 20
    source_residual = run_to_layer(model, source_sample, layer)
    target_residual = run_to_layer(model, target_sample, layer)

    # Encode through SAE
    source_latents = sae.encode(source_residual)
    target_latents = sae.encode(target_residual)

    # Patch: replace cluster latents
    patched_latents = source_latents.clone()
    patched_latents[:, cluster_indices] = target_latents[:, cluster_indices]

    # Decode and continue
    patched_residual = sae.decode(patched_latents)

    # Get final logits for all three
    source_logits = run_from_layer(model, source_residual, layer)
    target_logits = run_from_layer(model, target_residual, layer)
    patched_logits = run_from_layer(model, patched_residual, layer)

    return source_logits, target_logits, patched_logits
```

### Measurements

1. **KL analysis**:
   - KL(patched || source): How much did we change?
   - KL(patched || target): How close to target?
   - KL(source || target): Baseline distance
   - Interpolation ratio: KL(patched || source) / KL(source || target)

2. **Random cluster control**:
   - Patch a random cluster's latents instead
   - Should have less specific effect

### CLI Arguments

```
--mode patching
--source_vertex 0
--target_vertex 1
--num_pairs 50
--patch_positions last
--include_random_control
```

---

## Output Structure

```
outputs/validation/{cluster_key}/
├── coherence/
│   ├── intra_vertex_kl_stats.json      # Stats for each vertex
│   ├── random_pair_kl_stats.json       # Stats for random pairs
│   ├── kl_distribution_grid.png        # 2x2 histogram visualization
│   ├── raw_kl_values.npz               # All KL values for further analysis
│   └── coherence_summary.txt
├── baseline/
│   ├── reference_distributions.pt
│   ├── rgb_by_coords.png
│   ├── rgb_by_kl.png
│   ├── coord_kl_correlation.json
│   └── summary.txt
├── nexttoken/
│   ├── umap_by_coords.png
│   ├── umap_by_vertex.png
│   ├── umap_embedding.npz
│   └── summary.txt
├── steering/
│   ├── strength_sweep.png
│   ├── kl_analysis.json
│   ├── top_shifts.json
│   └── summary.txt
└── patching/
    ├── kl_analysis.json
    ├── interpolation_scores.json
    ├── random_control.json
    └── summary.txt
```

---

## Implementation Order

### Phase 0: Data Collection
1. Implement uniform simplex sampling (modification to refit script)
2. Collect ~10k samples across simplex for each cluster of interest

### Phase 1: Coherence Check (Critical!)
1. Implement KL computation between sample pairs
2. Compute intra-vertex distributions
3. Compute random-pair distribution
4. Generate 2x2 histogram grid
5. **Decision point**: If intra-vertex KL is not clearly smaller than random-pair KL, the simplex may not be meaningful

### Phase 2: Baseline Visualization
1. Compute reference distributions (averages)
2. RGB visualization by coords
3. RGB visualization by KL
4. Correlation analysis

### Phase 3: UMAP Analysis
1. Implement distribution embedding
2. Generate visualizations with different colorings

### Phase 4: Steering
1. Implement steering vector computation
2. Implement forward pass hooking
3. KL measurements
4. Strength sweep

### Phase 5: Patching
1. Implement activation caching
2. Implement patching
3. KL analysis
4. Random control

---

## CLI Structure

```
python interpretation/validate_belief_states.py \
    --mode {coherence, baseline, nexttoken, steering, patching, all} \

    # Model/SAE specification
    --model_name gemma-2-9b \
    --sae_release gemma-scope-9b-pt-res \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device cuda \
    --cache_dir /workspace/hf_cache \
    --hf_token $HF_TOKEN \

    # Cluster specification
    --n_clusters 512 \
    --cluster_id 261 \
    --aanet_checkpoint PATH \
    --prepared_samples_dir PATH \
    --uniform_samples_path PATH \

    # Coherence check params
    --max_pairs_per_vertex 500 \
    --random_pairs 5000 \

    # UMAP params
    --distribution_tokens 1000 \
    --token_selection variance \
    --umap_n_neighbors 15 \
    --umap_min_dist 0.1 \

    # Steering params
    --source_vertex 0 \
    --target_vertex 1 \
    --steering_strengths 0.1,0.25,0.5,1.0,2.0 \
    --steering_mode interpolate \
    --steer_positions last \

    # Patching params
    --num_pairs 50 \
    --include_random_control \

    # Output
    --output_dir outputs/validation
```

---

## Summary of Key Changes from v2

1. **Added coherence check** (Step 0a): Validate that intra-vertex samples are similar before using averages
2. **KL distribution comparison**: 2x2 histogram grid with same scales
3. **Five-number summary + mean/variance** for all distributions
4. **Explicit uniform sampling requirement**: Need new FineWeb run for random simplex coverage
5. **k>3 handling**: Use top-3 vertices by sample count
6. **Token selection for UMAP**: Configurable, default to variance-based
