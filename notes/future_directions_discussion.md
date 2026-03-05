# Future Directions Discussion
*Captured from research conversation, 2026-03-04*

## Context

This discussion arose while current steering experiments were computing, prompted by the question:
"If we only end up with so-so steering results and don't end up with super strong evidence of the model leveraging a belief state in a simplex, do you think there is any other way of looking at our results that gives interesting, publishable interpretability insights?"

---

## What We Have That's Publishable Even With Mixed Steering Results

### 1. The null cluster calibration finding
512_138 (null cluster) produced high-confidence, fully consistent grammatical person labels across 20 iterations — and failed causal testing. This is a concrete demonstration that LLM interpretation consistency is not sufficient for validation, and that causal steering is a necessary check. Useful contribution to the autointerp methodology literature, where people are actively debating how to validate feature interpretations.

### 2. The method pipeline is novel
Geometric clustering of decoder vectors → AANet → near-vertex sampling → iterated interpretation → causal validation is a coherent, reusable methodology for characterizing SAE feature subspaces. Even with mixed results, the pipeline itself is a contribution.

### 3. 512_17 is a real result
One high-confidence cluster with meaningful causal evidence (text register: instructional vs promotional) is publishable on its own terms. The V0→V1 steering (instructional→promotional) produced coherent qualitative shifts in the predicted direction at moderate steering strengths.

### 4. Honest aggregate finding has value
"We surveyed N clusters; most apparent structure doesn't survive causal testing; the exception tends to be semantic register rather than syntactic patterns" is informative about what kinds of structure SAEs actually encode in this way.

### 5. Reframing question
Whether the "belief state / simplex" framing survives at all, or whether a more modest reframe toward "activation archetypes" or "operational modes of feature clusters" is more appropriate. The simplex geometry is real — the question is whether calling it a belief state is justified.

---

## The Theoretical Tension: Is a Simplex Efficient?

**The efficiency argument for simplices:**
A simplex is efficient if the thing being tracked is genuinely categorical and mutually exclusive — if the model needs to maintain "am I in state A, B, or C" where only one can be true at once, a (k-1)-dimensional simplex is the minimum possible representation.

**The efficiency argument against:**
If the "states" aren't mutually exclusive, or if the same information feeds into many downstream computations in different combinations, superposition wins — much higher effective dimensionality per unit of residual space. The SAE latents presumably are of use in other computations beyond just tracking this belief state, which is consistent with the latter view.

**The transformer mechanics problem:**
The residual stream at each token position is independent. There's no direct mechanism that copies a subspace from position t-1 to position t. Information can be carried forward via attention heads reading value vectors from earlier positions, but this requires a head that consistently attends to the previous token and has a value matrix that approximately preserves the relevant subspace — fragile and indirect compared to an RNN hidden state.

The more natural transformer mechanism for persistent belief states might be re-computation at each token from context via attention over the full past sequence — each token attends back to relevant evidence and recomputes the current belief, rather than carrying it forward. This would predict simplex coordinates that jump around less smoothly than a true carried-forward hidden state, more like a function of recent context.

**An alternative framing:**
The belief state might not be "these latents are all on simultaneously" but rather "the projection of the residual onto this subspace encodes the current belief," with individual latents contributing transiently when the belief updates. Consistent with what the AANet is actually measuring — the empirical picture from SAE literature shows most latents fire sparsely and locally, not persistently over long spans.

---

## Specialized Data as a More Powerful Test

### The core idea
Rather than using diverse web text (Pile/FineWeb) and hoping to find simplices, identify state spaces you *know* the model must track, engineer the input distribution to maximize contrast between states, and validate simplex coordinates against ground truth labels — without needing LLM interpretation at all.

**Requirements for a good dataset:**
1. Easy automatic labeling — label directly recoverable from document structure
2. Sharp, locally-detectable transitions — state changes at an identifiable point
3. Mutual exclusivity — genuinely "one at a time" states so the simplex interpretation is natural
4. Heavy training data representation — model needs dedicated representations

### Code as the best candidate
Code scores well on all four criteria:
- Labels recoverable from syntax (docstring, function signature, function body, class definition, import block, comment)
- Transitions are sharp and syntactically explicit
- States are mutually exclusive
- Heavily represented in training data, syntactically strict

Could validate by checking whether barycentric coordinates correlate with ground truth labels, and whether steering toward a "function body" vertex makes completions look more like function bodies.

### Legal opinions
Colleague has a dataset of Supreme Court opinions labeled by constitutional principles at the opinion level. Issues:
- Opinion-level labels are too coarse — an opinion may invoke multiple principles across different paragraphs
- Constitutional principle is more like a slowly-varying background context than an actively-updated belief state
- Model might recognize "First Amendment opinion" from the first few paragraphs and maintain it weakly, rather than tracking it dynamically

More interesting version: track *structure of legal argument* (claim, precedent citation, application, holding) which is locally detectable and shifts in a meaningful way.

**Practical resources for US legal structure annotation:**
- CourtListener / RECAP: massive corpus of US opinions without structural labels, but consistent formatting of judicial opinions might allow automatic LLM-based tagging
- ECtHR dataset: more consistently structured sections (facts, law, operative provisions)
- LegalBench / LexGLUE: consolidated legal NLP benchmarks worth checking for existing annotation

Key issue: even with good annotation, code is safer because the structural regularities are syntactically enforced whereas legal argument structure is softer.

### The probe comparison problem
Linear probes at each layer tell you "this information is linearly decodable here" but not how it's encoded or whether it's being actively used. The pipeline, even with mixed results, is trying to say something stronger — a specific low-dimensional subspace of the residual, spanned by a geometrically coherent cluster of SAE latents, is the mechanism. That's a much more specific claim.

The genuinely novel claim the pipeline could make: not just "the model knows what code context it's in" but "it encodes this using a simplex-structured subspace of a small number of SAE latents, and the simplex coordinates causally determine downstream behavior." Probes can't make that claim.

### Sparse indicator learning idea
Learn which latents are jointly predictive of a ground truth state label with a sparsity penalty forcing selection to a small subset (L1-penalized classifier over latent activations). Key connection: if the same small set of latents that predict the label also happen to be geometrically clustered in decoder space, that's convergent evidence the model is using a coherent subspace for this computation. Finding the cluster from the functional direction rather than the geometric direction and checking if they agree.

---

## Co-occurrence Clustering as Alternative to Geometric Clustering

### Background
Previous attempt at co-occurrence-based clustering produced giant cluster problem: Phi, Jaccard, MI all created 10,000-14,000 latent clusters (50-70% of all features) with hundreds of singletons. AMI avoided giant clusters but produced essentially random clustering (ARI vs geometry ≈ 0).

### Is geometric clustering necessary for the simplex hypothesis?
In high dimensions, random vectors are nearly orthogonal, so the geometric constraint from decoder-direction clustering may not be as strong as it seems. More precisely: k-subspace clustering finds groups where all vectors approximately lie in a k-dimensional plane — a real constraint, not just pairwise similarity. But the AANet fitting is agnostic to decoder direction — it finds whatever low-dimensional structure exists in activation patterns regardless. So co-occurrence clusters with genuinely simplex-structured activations would still be found.

### Graph backbone extraction (Serrano et al. 2009 disparity filter)
Specifically designed for networks with heavy-tailed degree distributions. For each node, retains only edges that are statistically surprising relative to that node's own connectivity profile. A hub latent with 5000 edges might retain only 30 — the ones where its co-occurrence share is disproportionately high.

**Mathematical behavior with hub latents (corrected analysis):**
- alpha_ij = (1 - p_ij)^(k_i - 1), where p_ij = w_ij / strength_i
- For hub i with k_i = 1600 and uniform weight distribution: p_ij ≈ 1/1600, so (1 - 1/1600)^1599 ≈ e^{-1} ≈ 0.37 — NOT significant at alpha=0.05 from hub's perspective
- For rare latent j with k_j = 5 where hub constitutes large fraction of j's budget: p_ji could be substantial, giving genuinely small alpha — significant from j's perspective
- Under OR logic: hub contributes nothing (all alphas ≈ 0.37), but rare latent's strong connection to hub survives if p_ji is large enough

**Known failure mode from literature (Coscia & Neffke 2017):**
> "In the Disparity Filter, links connecting peripheral nodes to hubs are kept, because periphery-hub connections always seem strong from the peripheral node's perspective, even though the strong attraction of the hub makes it likely that such edges form randomly."

The DF "only ever looks at one node at a time" — it cannot account for the fact that a peripheral-hub connection may be expected simply because the hub is very active.

**Recommended alternative: Noise Corrected (NC) backbone**
Uses a joint null model: expected weight of edge (i,j) = s_i * s_j / W. Hub connections are only retained if stronger than expected given the hub's overall activity. Specifically designed to fix the one-endpoint problem of DF. Tends to retain periphery-periphery edges and drop hub-periphery edges that are only "surprising" from the peripheral node's local perspective.

### Implementation notes
Raw co-occurrence statistics already saved at `outputs/cooc_comparison/cooc_stats.npz` (324 MB):
- N11: dense 16384×16384 co-occurrence count matrix
- N1: marginal activation counts per latent
- n_samples: total sample count

No re-running of data collection needed. A vectorized scipy sparse implementation of NC backbone would be ~100 lines, CPU-feasible.

---

## Summary of Potential Next Steps (Priority Order)

1. **Finish current steering experiments** — get results for all 13 real priority clusters before deciding on pivots
2. **If 512_17 holds up** — make the strongest possible case from that one cluster with sequence-level analysis
3. **Null vs real cluster comparison** — use the calibration finding (null cluster passed interpretation but failed steering) as a methodological contribution in its own right
4. **Code corpus experiment** — if a pivot is needed and compute budget allows, this is the cleanest path to a labeled-belief-state validation
5. **Co-occurrence + NC backbone** — worth a quick experiment since raw stats are already saved; low additional compute cost
