#!/usr/bin/env python3
"""
Phase 3b: Causal Steering Auto-Interpretation via vLLM.

For each cluster, classifies steered and unsteered continuations into vertex
categories using Qwen-72B. Uses consolidated vertex labels as grounding, plus
exemplar text windows from prepared_samples.

Sampling (paired design): for each vertex v, n_samples examples are drawn
once. Those same examples provide all 20 continuations per example:
  document  — original post-trigger text from the source document
  baseline  — unsteered (greedy) continuation
  steered   — (k-1) target vertices × 3 types × 3 scales = 18 per example

Using the same examples across all conditions ensures example-level difficulty
is shared, controlling for variation when n_samples is small (e.g. 30).

Score function (all case types):
  1.0 if predicted label == expected vertex
  0.5 if "Between" label that includes expected vertex
  0.0 otherwise (including Unclear)

Degenerate continuations are excluded from mean/std and counted separately.
Unclear responses score 0, stay in the denominator, and are counted separately.

Usage:
    python validation/causal_steering_autointerp.py \\
        --steering_dir /workspace/outputs/validation/causal_steering \\
        --synthesis_dir outputs/interpretations/sonnet_broad_2/synthesis \\
        --prepared_samples_dir outputs/interpretations/prepared_samples_broad_2 \\
        --output_dir /workspace/outputs/validation/causal_steering_autointerp \\
        --clusters 512_17,512_181,768_140,768_596 \\
        --model_name Qwen/Qwen2.5-72B-Instruct-AWQ \\
        --quantization awq_marlin \\
        --cache_dir /workspace/hf_cache \\
        --hf_token $HF_TOKEN
"""

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
# Synthesis loading
# =============================================================================

def load_synthesis(synthesis_dir, cluster_key):
    """Load synthesis JSON. Returns (hypothesis_str, vertex_labels_list, k) or None."""
    path = Path(synthesis_dir) / f"{cluster_key}_synthesis.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    synth = d["synthesis"]
    return (
        synth["consolidated_hypothesis"],
        synth["consolidated_vertex_labels"],
        d["k"],
    )


# =============================================================================
# Exemplar window builder
# =============================================================================

def build_exemplar_window(chunk_token_ids, trigger_idx, tokenizer, n_tokens=80):
    """Decode a ~80-token window centered on trigger, marking it >>>TOKEN<<<."""
    half = n_tokens // 2
    start = max(0, trigger_idx - half)
    end = min(len(chunk_token_ids), trigger_idx + half)
    tokens = chunk_token_ids[start:end]
    local_trigger = trigger_idx - start

    parts = []
    for i, tok_id in enumerate(tokens):
        try:
            decoded = tokenizer.decode([int(tok_id)])
        except Exception:
            decoded = " "
        if i == local_trigger:
            parts.append(f">>>{decoded.strip()}<<<")
        else:
            parts.append(decoded)
    return "".join(parts).strip()


def load_exemplars(prepared_samples_dir, cluster_key, k, n_exemplars, tokenizer, rng):
    """Load n_exemplars text windows per vertex from prepared_samples.

    Returns dict {vertex_id (int): [window_str, ...]}.
    """
    path = Path(prepared_samples_dir) / f"cluster_{cluster_key}.json"
    if not path.exists():
        print(f"  WARNING: prepared_samples not found: {path}")
        return {}

    with open(path) as f:
        data = json.load(f)

    exemplars = {}
    for v_str, samples in data["vertices"].items():
        v = int(v_str)
        pool = [s for s in samples if s.get("trigger_token_indices") and s.get("chunk_token_ids")]
        selected = rng.sample(pool, min(n_exemplars, len(pool)))
        windows = []
        for s in selected:
            trigger_idx = s["trigger_token_indices"][0]
            window = build_exemplar_window(s["chunk_token_ids"], trigger_idx, tokenizer)
            if window:
                windows.append(window)
        exemplars[v] = windows
    return exemplars


# =============================================================================
# Load steering results → build test cases
# =============================================================================

STEERING_TYPES = ("type1", "type2", "type3")


def get_steered_continuation(steered_continuations, scale, steering_type):
    """Retrieve steered continuation from the nested {steering_type: {scale: text}} dict."""
    type_dict = steered_continuations.get(steering_type, {})
    for key in [str(float(scale)), str(int(scale)) if scale == int(scale) else str(scale),
                f"{scale:g}", str(scale)]:
        if key in type_dict:
            return type_dict[key]
    return ""


def _parse_ex_idx(record):
    """Parse example index from record_id: '{cluster_key}_ex{NNNN}_V{src}toV{tgt}'."""
    try:
        return int(record["record_id"].split("_ex")[1].split("_")[0])
    except (IndexError, ValueError):
        return id(record)


def load_test_cases(steering_dir, cluster_key, k, n_samples, scales, rng):
    """Load all continuations for n_samples examples per vertex (paired design).

    For each source vertex v, samples n_samples examples once. Those same
    examples provide all 20 continuations:
      - document:  original post-trigger text from the source document
      - baseline:  unsteered greedy continuation
      - steered:   (k-1) target vertices × len(scales) × len(STEERING_TYPES)

    This ensures example-level difficulty is shared across all conditions.
    """
    results_path = Path(steering_dir) / f"cluster_{cluster_key}" / "steering_results.jsonl"
    if not results_path.exists():
        print(f"  WARNING: {results_path} not found")
        return []

    # Index records by source vertex and by (src, tgt) direction
    by_source    = defaultdict(dict)   # {src: {ex_idx: record}}
    by_direction = defaultdict(dict)   # {(src, tgt): {ex_idx: record}}

    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            src    = record["source_vertex"]
            tgt    = record["target_vertex"]
            ex_idx = _parse_ex_idx(record)

            if ex_idx not in by_source[src]:
                by_source[src][ex_idx] = record
            by_direction[(src, tgt)][ex_idx] = record

    cases = []

    for src in range(k):
        pool = list(by_source.get(src, {}).values())
        if not pool:
            print(f"  WARNING: no examples for V{src} in {cluster_key}")
            continue
        rng.shuffle(pool)
        selected = pool[:n_samples]

        for record in selected:
            ex_idx = _parse_ex_idx(record)
            common = {
                "cluster_key":         cluster_key,
                "source_vertex":       src,
                "ex_idx":              ex_idx,
                "record_id":           record["record_id"],
                "trigger_token_index": record["trigger_token_index"],
                "trigger_word":        record.get("trigger_word", ""),
                "pre_trigger_text":    record.get("pre_trigger_text",
                                                   record.get("original_text", "")),
            }

            # Original post-trigger text from source document
            cases.append({
                **common,
                "test_type":       "document",
                "continuation":    record.get("document_continuation", ""),
                "expected_vertex": src,
            })

            # Unsteered greedy continuation
            cases.append({
                **common,
                "test_type":       "baseline",
                "continuation":    record["unsteered_continuation"],
                "expected_vertex": src,
            })

            # All steered continuations — same examples for every (tgt, scale, stype)
            for tgt in range(k):
                if tgt == src:
                    continue
                dir_record = by_direction.get((src, tgt), {}).get(ex_idx)
                if dir_record is None:
                    print(f"  WARNING: missing steered record {cluster_key} "
                          f"ex{ex_idx:04d} V{src}→V{tgt}")
                    continue
                for scale in scales:
                    for stype in STEERING_TYPES:
                        cont = get_steered_continuation(
                            dir_record.get("steered_continuations", {}), scale, stype
                        )
                        cases.append({
                            **common,
                            "test_type":       "steered",
                            "target_vertex":   tgt,
                            "scale":           scale,
                            "steering_type":   stype,
                            "continuation":    cont,
                            "expected_vertex": tgt,
                        })

    return cases


# =============================================================================
# Classification options
# =============================================================================

def build_options(k):
    """Return the ordered list of valid classification labels for a k-vertex cluster."""
    opts = [f"V{i}" for i in range(k)]
    if k == 3:
        opts += ["Between V0-V1", "Between V1-V2", "Between V0-V2"]
    # For k=4 we skip Between options (6 combinations is too many)
    opts.append("Degenerate")
    opts.append("Unclear")
    return opts


# =============================================================================
# Prompt builder
# =============================================================================

SYSTEM_PROMPT = (
    "You are a precise text classifier helping to analyze neural network features. "
    "Follow the instructions exactly and respond only in the requested format."
)


def build_prompt(case, hypothesis, vertex_labels, exemplars, k):
    """Build the full classification prompt for one test case."""
    opts = build_options(k)
    opts_str = " / ".join(opts)

    vertex_block = "\n".join(f"  V{i}: {label}" for i, label in enumerate(vertex_labels))

    exemplar_lines = []
    for v in range(k):
        vex = exemplars.get(v, [])
        if vex:
            ex_str = "\n    ".join(f'"{w}"' for w in vex)
            exemplar_lines.append(f"[V{v} EXAMPLES]\n    {ex_str}")
        else:
            exemplar_lines.append(f"[V{v} EXAMPLES]\n    (none available)")
    exemplar_block = "\n\n".join(exemplar_lines)

    pre_text = case.get("pre_trigger_text") or ""
    words = pre_text.split()
    context_snippet = " ".join(words[-50:]) if len(words) > 50 else pre_text

    trigger_word = case.get("trigger_word") or "(unknown)"
    trigger_idx  = case.get("trigger_token_index", "?")
    continuation = case["continuation"] or "(empty)"

    return f"""We are studying a language model feature cluster. A frontier AI model analyzed \
thousands of activating text examples and identified {k} distinct activation patterns:

{vertex_block}

Here are example texts where the cluster fires strongly at each vertex.
The active trigger token is marked >>>TOKEN<<<:

{exemplar_block}

---
Now classify the following text.

The trigger token is "{trigger_word}" (position {trigger_idx}/256 in the context window).
The original context up to and including the trigger is shown first; the model-generated
CONTINUATION — which may have been modified to emphasize a particular vertex — follows
after [>>>CONT<<<]:

CONTEXT: ...{context_snippet}
[>>>CONT<<<]
CONTINUATION: {continuation}

Which vertex does the CONTINUATION most likely represent?
Options: {opts_str}

Choose "Degenerate" if the continuation is repetitive, incoherent, or stuck in a loop \
(e.g. the same word or phrase repeated many times). This is a known failure mode of the \
model being studied; label it Degenerate rather than guessing a vertex.
Choose "Unclear" only if the continuation is coherent but does not fit any vertex description well.

Respond with ONLY the option label (e.g., "V0" or "Between V0-V1" or "Degenerate")."""


# =============================================================================
# vLLM helpers
# =============================================================================

def apply_chat_template(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_PROMPT}\n\n### User\n{user_prompt}\n\n### Assistant\n"


def batch_generate(llm, tokenizer, prompts, max_tokens=20, batch_size=256):
    from vllm import SamplingParams
    formatted = [apply_chat_template(tokenizer, p) for p in prompts]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    responses = []
    for i in range(0, len(formatted), batch_size):
        chunk = formatted[i : i + batch_size]
        outputs = llm.generate(chunk, params)
        responses.extend(o.outputs[0].text.strip() for o in outputs)
    return responses


# =============================================================================
# Response parsing
# =============================================================================

def parse_label(response, k):
    """Parse LLM response into one of the canonical option labels."""
    text = response.strip()
    opts = build_options(k)

    # Exact match (case-insensitive)
    for opt in opts:
        if text.lower() == opt.lower():
            return opt

    # Prefix match
    for opt in opts:
        if text.lower().startswith(opt.lower()):
            return opt

    # Fuzzy: "Between V{i}-V{j}" or "Between V{i} and V{j}"
    between_m = re.search(r'[Bb]etween\s+V(\d)\s*[-–and ]+\s*V(\d)', text)
    if between_m:
        va, vb = sorted([int(between_m.group(1)), int(between_m.group(2))])
        candidate = f"Between V{va}-V{vb}"
        if candidate in opts:
            return candidate

    # Fuzzy: single vertex mention
    v_matches = re.findall(r'\bV(\d)\b', text)
    if v_matches:
        v = int(v_matches[0])
        if v < k:
            return f"V{v}"

    return "Unclear"


# =============================================================================
# Scoring and metrics
# =============================================================================

def label_shift_score(label, expected_vertex):
    """Score toward expected vertex: 1.0 exact, 0.5 if Between includes it, else 0."""
    tgt = f"V{expected_vertex}"
    if label == tgt:
        return 1.0
    if "Between" in label:
        vs = re.findall(r'V(\d)', label)
        if str(expected_vertex) in vs:
            return 0.5
    return 0.0


def compute_group_stats(case_list, expected_vertex):
    """Compute stats for a group of cases sharing the same expected vertex.

    Degenerate: excluded from mean/std, counted in n_degenerate.
    Unclear:    scores 0 (included in mean/std), counted in n_unclear.
    """
    scores       = []
    n_degenerate = 0
    n_unclear    = 0
    for c in case_list:
        label = c["predicted_label"]
        if label == "Degenerate":
            n_degenerate += 1
        else:
            if label == "Unclear":
                n_unclear += 1
            scores.append(label_shift_score(label, expected_vertex))
    return {
        "mean":         float(np.mean(scores))  if scores else None,
        "std":          float(np.std(scores))   if scores else None,
        "n_valid":      len(scores),
        "n_degenerate": n_degenerate,
        "n_unclear":    n_unclear,
    }


def _find_best_combo(steered_cases, k, scales):
    """Find the (steering_type, scale) with best average score across all directions.

    Eligibility: total_valid / (total_valid + total_degenerate) >= 2/3 globally
    (across all directions combined for that combo).

    Returns a dict with the best combo and per-direction stats, or None.
    """
    combo_scores = defaultdict(list)
    combo_degen  = defaultdict(int)

    for c in steered_cases:
        key = (c["steering_type"], c["scale"])
        if c["predicted_label"] == "Degenerate":
            combo_degen[key] += 1
        else:
            combo_scores[key].append(
                label_shift_score(c["predicted_label"], c["target_vertex"])
            )

    best_key  = None
    best_mean = -1.0

    for key in combo_scores:
        n_valid = len(combo_scores[key])
        n_degen = combo_degen.get(key, 0)
        if n_valid + n_degen == 0:
            continue
        if n_valid / (n_valid + n_degen) < 2 / 3:
            continue
        mean = float(np.mean(combo_scores[key]))
        if mean > best_mean:
            best_mean = mean
            best_key  = key

    if best_key is None:
        return None

    stype, scale = best_key

    per_direction = {}
    for src in range(k):
        for tgt in range(k):
            if tgt == src:
                continue
            group = [
                c for c in steered_cases
                if c["source_vertex"] == src
                and c["target_vertex"] == tgt
                and c["steering_type"] == stype
                and c["scale"] == scale
            ]
            per_direction[f"V{src}toV{tgt}"] = compute_group_stats(group, tgt)

    return {
        "steering_type":          stype,
        "scale":                  scale,
        "mean_across_directions": best_mean,
        "per_direction":          per_direction,
    }


def compute_metrics(cases, k, scales):
    """Full metrics: per-vertex detailed breakdown and best (type, scale) summary.

    Returns:
        vertex_results: {
            "V{src}": {
                "document": group_stats,
                "baseline": group_stats,
                "steered": {
                    "V{src}toV{tgt}_{stype}_s{scale}": group_stats, ...
                }
            }, ...
        }
        best_combo: {
            "steering_type", "scale", "mean_across_directions",
            "per_direction": {"V{src}toV{tgt}": group_stats, ...}
        } or None
    """
    steered_cases = [c for c in cases if c["test_type"] == "steered"]

    vertex_results = {}
    for src in range(k):
        src_cases = [c for c in cases if c["source_vertex"] == src]

        steered_combos = {}
        for tgt in range(k):
            if tgt == src:
                continue
            for scale in scales:
                for stype in STEERING_TYPES:
                    group = [
                        c for c in src_cases
                        if c["test_type"] == "steered"
                        and c.get("target_vertex") == tgt
                        and c["scale"] == scale
                        and c["steering_type"] == stype
                    ]
                    combo_key = f"V{src}toV{tgt}_{stype}_s{scale:g}"
                    steered_combos[combo_key] = {
                        "source_vertex": src,
                        "target_vertex": tgt,
                        "scale":         scale,
                        "steering_type": stype,
                        **compute_group_stats(group, tgt),
                    }

        vertex_results[f"V{src}"] = {
            "document": compute_group_stats(
                [c for c in src_cases if c["test_type"] == "document"], src
            ),
            "baseline": compute_group_stats(
                [c for c in src_cases if c["test_type"] == "baseline"], src
            ),
            "steered": steered_combos,
        }

    return {
        "vertex_results": vertex_results,
        "best_combo":     _find_best_combo(steered_cases, k, scales),
    }


# =============================================================================
# Console reporting
# =============================================================================

def _fmt(stats):
    """Format a group_stats dict as a compact one-liner."""
    if stats is None or stats.get("mean") is None:
        return "N/A"
    return (f"mean={stats['mean']:.3f} ±{stats['std']:.3f}  "
            f"n={stats['n_valid']}  degen={stats['n_degenerate']}  "
            f"unclear={stats['n_unclear']}")


def print_cluster_results(cluster_key, k, vertex_labels, metrics, scales):
    print(f"\n{'='*68}")
    print(f"Cluster {cluster_key}  (k={k})")

    vr = metrics["vertex_results"]

    # --- Per-vertex detailed table ---
    for src in range(k):
        label_short = vertex_labels[src][:55] if len(vertex_labels[src]) > 55 \
                      else vertex_labels[src]
        print(f"\n  [V{src}] {label_short}")
        print(f"    document : {_fmt(vr[f'V{src}']['document'])}")
        print(f"    baseline : {_fmt(vr[f'V{src}']['baseline'])}")
        for tgt in range(k):
            if tgt == src:
                continue
            print(f"    V{src}→V{tgt} steered:")
            for stype in STEERING_TYPES:
                parts = []
                for scale in scales:
                    key = f"V{src}toV{tgt}_{stype}_s{scale:g}"
                    s = vr[f"V{src}"]["steered"].get(key, {})
                    m = f"{s['mean']:.3f}" if s.get("mean") is not None else "N/A"
                    parts.append(
                        f"s={scale:g}: {m} "
                        f"(n={s.get('n_valid','?')} "
                        f"degen={s.get('n_degenerate','?')} "
                        f"unc={s.get('n_unclear','?')})"
                    )
                print(f"      [{stype}] " + "  |  ".join(parts))

    # --- Best combo summary ---
    bc = metrics.get("best_combo")
    print(f"\n  {'─'*60}")
    print(f"  Best (type, scale): ", end="")
    if bc is None:
        print("none qualified (all combos failed 2/3 non-degenerate threshold)")
        return

    print(f"{bc['steering_type']}  s={bc['scale']:g}  "
          f"→ mean across directions = {bc['mean_across_directions']:.3f}")

    dirs = [f"V{src}toV{tgt}" for src in range(k) for tgt in range(k) if tgt != src]
    for d in dirs:
        s = bc["per_direction"].get(d, {})
        print(f"    {d}: {_fmt(s)}")

    print(f"  Document / unsteered per vertex:")
    for src in range(k):
        print(f"    V{src}: doc={_fmt(vr[f'V{src}']['document'])}  "
              f"unsteer={_fmt(vr[f'V{src}']['baseline'])}")


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(args, llm, vllm_tokenizer, gemma_tokenizer):
    cluster_keys = [c.strip() for c in args.clusters.split(",") if c.strip()]
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for cluster_key in cluster_keys:
        print(f"\n{'='*60}")
        print(f"Processing cluster {cluster_key}...")

        synth = load_synthesis(args.synthesis_dir, cluster_key)
        if synth is None:
            print(f"  WARNING: no synthesis for {cluster_key}, skipping")
            continue
        hypothesis, vertex_labels, k = synth

        exemplars = load_exemplars(
            args.prepared_samples_dir, cluster_key, k,
            args.n_exemplars, gemma_tokenizer, rng,
        )

        cases = load_test_cases(
            args.steering_dir, cluster_key, k,
            args.n_samples, args.scales, rng,
        )

        n_doc  = sum(1 for c in cases if c["test_type"] == "document")
        n_base = sum(1 for c in cases if c["test_type"] == "baseline")
        n_ste  = sum(1 for c in cases if c["test_type"] == "steered")
        print(f"  {n_doc} doc + {n_base} baseline + {n_ste} steered = {len(cases)} total")

        if not cases:
            print("  No cases — skipping.")
            continue

        prompts = [
            build_prompt(case, hypothesis, vertex_labels, exemplars, k)
            for case in cases
        ]

        print(f"  Running {len(prompts)} prompts "
              f"(batch_size={args.vllm_batch_size})...")
        t0 = time.perf_counter()
        responses = batch_generate(
            llm, vllm_tokenizer, prompts,
            max_tokens=args.max_response_tokens,
            batch_size=args.vllm_batch_size,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s  ({elapsed / len(prompts):.3f}s/prompt)")

        for case, raw_response in zip(cases, responses):
            case["raw_response"]    = raw_response
            case["predicted_label"] = parse_label(raw_response, k)

        metrics = compute_metrics(cases, k, args.scales)
        print_cluster_results(cluster_key, k, vertex_labels, metrics, args.scales)

        # Per-cluster JSON — omit bulky pre_trigger_text from case list
        slim_cases = [
            {k2: v for k2, v in c.items() if k2 != "pre_trigger_text"}
            for c in cases
        ]
        cluster_out = {
            "cluster_key":   cluster_key,
            "k":             k,
            "hypothesis":    hypothesis,
            "vertex_labels": vertex_labels,
            "metrics":       metrics,
            "cases":         slim_cases,
        }
        out_path = output_dir / f"cluster_{cluster_key}_autointerp.json"
        with open(out_path, "w") as f:
            json.dump(cluster_out, f, indent=2)
        print(f"  → {out_path}")

        all_results[cluster_key] = {
            "k":             k,
            "vertex_labels": vertex_labels,
            "metrics":       metrics,
            "n_cases":       len(cases),
        }

        # Write combined results after each cluster for crash recovery
        combined_path = output_dir / "all_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nAll results → {output_dir / 'all_results.json'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3b: Causal steering auto-interpretation via vLLM"
    )
    parser.add_argument("--steering_dir",         required=True,
                        help="Dir with cluster_{key}/steering_results.jsonl")
    parser.add_argument("--synthesis_dir",        required=True,
                        help="Dir with {cluster_key}_synthesis.json files")
    parser.add_argument("--prepared_samples_dir", required=True,
                        help="Dir with cluster_{key}.json prepared samples")
    parser.add_argument("--output_dir",           required=True)
    parser.add_argument("--clusters",             required=True,
                        help="Comma-separated cluster keys, e.g. 512_17,768_140")

    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 5.0, 20.0],
                        help="Steering scales to evaluate")
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Examples per vertex; same pool used for document, "
                             "baseline, and all steered conditions (paired design)")
    parser.add_argument("--n_exemplars", type=int, default=5,
                        help="Exemplar windows per vertex shown in each prompt")

    parser.add_argument("--model_name",             default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    parser.add_argument("--quantization",           default="awq_marlin")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len",          type=int,   default=8192)
    parser.add_argument("--max_response_tokens",    type=int,   default=20,
                        help="Max tokens for the classification response")
    parser.add_argument("--vllm_batch_size",        type=int,   default=256,
                        help="Max prompts per llm.generate() call (prevents OOM)")

    parser.add_argument("--gemma_tokenizer", default="google/gemma-2-9b",
                        help="Tokenizer for decoding prepared_samples token IDs "
                             "(must match the SAE base model, NOT the classifier LLM)")
    parser.add_argument("--cache_dir", default="/workspace/hf_cache")
    parser.add_argument("--hf_token",  default=None)
    parser.add_argument("--seed",      type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    print(f"Loading Gemma tokenizer ({args.gemma_tokenizer})...")
    from transformers import AutoTokenizer
    from huggingface_hub import login
    if args.hf_token:
        login(token=args.hf_token)
    gemma_tokenizer = AutoTokenizer.from_pretrained(
        args.gemma_tokenizer, cache_dir=args.cache_dir, token=args.hf_token
    )

    print(f"\nLoading {args.model_name} via vLLM...")
    from vllm import LLM
    llm_kwargs = dict(
        model=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        download_dir=args.cache_dir,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)
    vllm_tokenizer = llm.get_tokenizer()
    print("Model loaded.")

    run_pipeline(args, llm, vllm_tokenizer, gemma_tokenizer)

    import gc, os
    del llm
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
