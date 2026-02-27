#!/usr/bin/env python3
"""
Phase 3b: Causal Steering Auto-Interpretation via vLLM.

For each cluster, classifies steered and unsteered continuations into vertex
categories using Qwen-72B. Uses the existing frontier-model synthesis
(consolidated_vertex_labels + consolidated_hypothesis) as grounding, plus
exemplar text windows from prepared_samples.

Two test types:
  baseline  — unsteered continuations from natural vertex examples.
              Expected label: source vertex. Measures classification ceiling.
  steered   — steered continuations at each scale for each src→tgt direction.
              Expected label: target vertex. Measures steering shift.

All prompts are collected into one flat list with full metadata, run in a
single vLLM batch, then scored.

Usage:
    python validation/causal_steering_autointerp.py \\
        --steering_dir /workspace/outputs/validation/causal_steering \\
        --synthesis_dir outputs/interpretations/sonnet_broad_2_no_whitespace/synthesis \\
        --prepared_samples_dir outputs/interpretations/prepared_samples_broad_2_no_whitespace \\
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
import sys
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
    """Decode a ~80-token window centered on trigger, marking it >>>TOKEN<<<.

    ~80 tokens ≈ 64 words for typical English text.
    """
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


def load_test_cases(steering_dir, cluster_key, n_baseline, n_steered, scales, rng):
    """Load baseline and steered test cases from steering_results.jsonl.

    Baseline: one case per unique example (deduplicated by ex_idx), using
              unsteered_continuation. Cap n_baseline per source vertex.
    Steered:  n_steered cases per (src→tgt) direction, same examples reused
              for all scales (enables proper dose-response comparison).

    Returns list of case dicts, each carrying all metadata needed for scoring.
    """
    results_path = Path(steering_dir) / f"cluster_{cluster_key}" / "steering_results.jsonl"
    if not results_path.exists():
        print(f"  WARNING: {results_path} not found")
        return []

    # Read all records
    by_source = defaultdict(dict)    # {src_vertex: {ex_idx: record}}
    by_direction = defaultdict(list)  # {(src, tgt): [records]}

    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            src = record["source_vertex"]
            tgt = record["target_vertex"]

            # Parse ex_idx from record_id: "{cluster_key}_ex{NNNN}_V{src}toV{tgt}"
            try:
                ex_idx = int(record["record_id"].split("_ex")[1].split("_")[0])
            except (IndexError, ValueError):
                ex_idx = id(record)  # fallback unique key

            # Deduplicate for baseline (one entry per example, not per target)
            if ex_idx not in by_source[src]:
                by_source[src][ex_idx] = record

            by_direction[(src, tgt)].append(record)

    cases = []

    # --- Baseline cases ---
    for src in sorted(by_source):
        pool = list(by_source[src].values())
        rng.shuffle(pool)
        for record in pool[:n_baseline]:
            cases.append({
                "test_type": "baseline",
                "cluster_key": cluster_key,
                "source_vertex": src,
                "target_vertex": src,
                "scale": None,
                "record_id": record["record_id"],
                "continuation": record["unsteered_continuation"],
                "expected_label": f"V{src}",
                "trigger_token_index": record["trigger_token_index"],
                "trigger_word": record.get("trigger_word", ""),
                "pre_trigger_text": record.get("pre_trigger_text", record.get("original_text", "")),
            })

    # --- Steered cases (same examples across all scales and steering types) ---
    for (src, tgt) in sorted(by_direction):
        pool = list(by_direction[(src, tgt)])
        rng.shuffle(pool)
        selected = pool[:n_steered]
        for record in selected:
            for scale in scales:
                for steering_type in STEERING_TYPES:
                    continuation = get_steered_continuation(
                        record.get("steered_continuations", {}), scale, steering_type
                    )
                    cases.append({
                        "test_type": "steered",
                        "cluster_key": cluster_key,
                        "source_vertex": src,
                        "target_vertex": tgt,
                        "scale": scale,
                        "steering_type": steering_type,
                        "record_id": record["record_id"],
                        "continuation": continuation,
                        "expected_label": f"V{tgt}",
                        "trigger_token_index": record["trigger_token_index"],
                        "trigger_word": record.get("trigger_word", ""),
                        "pre_trigger_text": record.get("pre_trigger_text", record.get("original_text", "")),
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

    # Vertex descriptions block
    vertex_block = "\n".join(f"  V{i}: {label}" for i, label in enumerate(vertex_labels))

    # Exemplar block — one line per example, grouped by vertex
    exemplar_lines = []
    for v in range(k):
        vex = exemplars.get(v, [])
        if vex:
            ex_str = "\n    ".join(f'"{w}"' for w in vex)
            exemplar_lines.append(f"[V{v} EXAMPLES]\n    {ex_str}")
        else:
            exemplar_lines.append(f"[V{v} EXAMPLES]\n    (none available)")
    exemplar_block = "\n\n".join(exemplar_lines)

    # Context: last ~50 words of pre_trigger_text
    pre_text = case.get("pre_trigger_text") or ""
    words = pre_text.split()
    context_snippet = " ".join(words[-50:]) if len(words) > 50 else pre_text

    trigger_word = case.get("trigger_word") or "(unknown)"
    trigger_idx = case.get("trigger_token_index", "?")
    continuation = case["continuation"] or "(empty)"

    return f"""We are studying a language model feature cluster. A frontier AI model analyzed thousands of activating text examples and produced the following interpretation:

STATE SPACE: {hypothesis}

The cluster has {k} vertex types:
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

Respond with ONLY the option label (e.g., "V0" or "Between V0-V1")."""


# =============================================================================
# vLLM helpers (from meta_sae pattern)
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


def batch_generate(llm, tokenizer, prompts, max_tokens=20):
    from vllm import SamplingParams
    formatted = [apply_chat_template(tokenizer, p) for p in prompts]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(formatted, params)
    return [o.outputs[0].text.strip() for o in outputs]


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

    # Fuzzy: look for "Between V{i}-V{j}" or "Between V{i} and V{j}"
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
# Scoring
# =============================================================================

def label_shift_score(label, target_vertex):
    """Continuous shift score: 1.0 if label == V_tgt, 0.5 if Between and includes V_tgt, else 0."""
    tgt = f"V{target_vertex}"
    if label == tgt:
        return 1.0
    if "Between" in label:
        vs = re.findall(r'V(\d)', label)
        if str(target_vertex) in vs:
            return 0.5
    return 0.0


def compute_metrics(cases, k):
    """Return per-cluster metrics dict from a list of cases with predicted_label."""
    baseline = [c for c in cases if c["test_type"] == "baseline"]
    steered  = [c for c in cases if c["test_type"] == "steered"]

    # Baseline: per-vertex accuracy
    by_vertex = defaultdict(list)
    for c in baseline:
        correct = int(c["predicted_label"] == c["expected_label"])
        by_vertex[c["source_vertex"]].append(correct)

    baseline_acc = {
        f"V{v}": float(np.mean(scores)) for v, scores in sorted(by_vertex.items())
    }
    if baseline_acc:
        baseline_acc["macro"] = float(np.mean(list(baseline_acc.values())))

    # Steered: shift rate per (direction, scale, steering_type)
    by_dir_scale = defaultdict(list)
    for c in steered:
        key = (c["source_vertex"], c["target_vertex"], c["scale"],
               c.get("steering_type", "type1"))
        by_dir_scale[key].append(label_shift_score(c["predicted_label"], c["target_vertex"]))

    shift_rates = {}
    for (src, tgt, scale, stype), scores in sorted(by_dir_scale.items()):
        label = f"V{src}toV{tgt}_s{scale:g}_{stype}"
        shift_rates[label] = {
            "source_vertex": src,
            "target_vertex": tgt,
            "scale": scale,
            "steering_type": stype,
            "mean_shift": float(np.mean(scores)),
            "n": len(scores),
            "label_distribution": _label_dist(
                [c["predicted_label"] for c in steered
                 if c["source_vertex"] == src
                 and c["target_vertex"] == tgt
                 and c["scale"] == scale
                 and c.get("steering_type", "type1") == stype],
                k,
            ),
        }

    return {"baseline_accuracy": baseline_acc, "shift_rates": shift_rates}


def _label_dist(labels, k):
    """Count occurrences of each canonical label."""
    opts = build_options(k)
    counts = {o: 0 for o in opts}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(args, llm, vllm_tokenizer, gemma_tokenizer):
    cluster_keys = [c.strip() for c in args.clusters.split(",") if c.strip()]
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load all cluster data; build flat (case, prompt) list
    # ------------------------------------------------------------------
    print("\nLoading cluster data and building prompts...")

    cluster_meta = {}   # cluster_key → (hypothesis, vertex_labels, k)
    all_cases   = []    # flat list of metadata dicts (one per prompt)
    all_prompts = []    # flat list of prompt strings (parallel to all_cases)

    for cluster_key in cluster_keys:
        synth = load_synthesis(args.synthesis_dir, cluster_key)
        if synth is None:
            print(f"  WARNING: no synthesis for {cluster_key}, skipping")
            continue
        hypothesis, vertex_labels, k = synth
        cluster_meta[cluster_key] = synth

        exemplars = load_exemplars(
            args.prepared_samples_dir, cluster_key, k,
            args.n_exemplars, gemma_tokenizer, rng,
        )

        cases = load_test_cases(
            args.steering_dir, cluster_key,
            args.n_baseline, args.n_steered, args.scales, rng,
        )

        n_base    = sum(1 for c in cases if c["test_type"] == "baseline")
        n_steered = sum(1 for c in cases if c["test_type"] == "steered")
        print(f"  {cluster_key} k={k}: {n_base} baseline, {n_steered} steered")

        for case in cases:
            prompt = build_prompt(case, hypothesis, vertex_labels, exemplars, k)
            all_cases.append(case)
            all_prompts.append(prompt)

    print(f"\nTotal prompts: {len(all_prompts)}")
    if not all_prompts:
        print("No prompts to run — exiting.")
        return

    # ------------------------------------------------------------------
    # Single vLLM batch
    # ------------------------------------------------------------------
    print(f"Running batch through {args.model_name}...")
    t0 = time.perf_counter()
    responses = batch_generate(llm, vllm_tokenizer, all_prompts, max_tokens=args.max_response_tokens)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s  ({elapsed / len(all_prompts):.3f}s/prompt)")

    # Attach parsed labels back to cases
    for case, raw_response in zip(all_cases, responses):
        _, _, k = cluster_meta[case["cluster_key"]]
        case["raw_response"]    = raw_response
        case["predicted_label"] = parse_label(raw_response, k)

    # ------------------------------------------------------------------
    # Score + save
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for cluster_key in cluster_keys:
        if cluster_key not in cluster_meta:
            continue
        hypothesis, vertex_labels, k = cluster_meta[cluster_key]
        c_cases = [c for c in all_cases if c["cluster_key"] == cluster_key]
        metrics  = compute_metrics(c_cases, k)

        # Console summary
        print(f"\n{'='*55}")
        print(f"Cluster {cluster_key}  (k={k})")
        print(f"  Baseline accuracy: {metrics['baseline_accuracy']}")
        # Print shift rates grouped by direction for readability
        for stype in STEERING_TYPES:
            type_rows = {k2: v for k2, v in metrics["shift_rates"].items()
                         if v.get("steering_type") == stype}
            if type_rows:
                print(f"  [{stype}]")
                for sr_key, sr in sorted(type_rows.items()):
                    print(f"    V{sr['source_vertex']}→V{sr['target_vertex']} "
                          f"s={sr['scale']:g}: mean_shift={sr['mean_shift']:.3f}  n={sr['n']}")

        # Save per-cluster file (omit bulky text fields from the case list)
        slim_cases = []
        for c in c_cases:
            slim = {k2: v for k2, v in c.items() if k2 not in ("pre_trigger_text",)}
            slim_cases.append(slim)

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
            "k":         k,
            "metrics":   metrics,
            "n_baseline": sum(1 for c in c_cases if c["test_type"] == "baseline"),
            "n_steered":  sum(1 for c in c_cases if c["test_type"] == "steered"),
        }

    combined_path = output_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results → {combined_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3b: Causal steering auto-interpretation via vLLM"
    )
    parser.add_argument("--steering_dir", required=True,
                        help="Dir with causal_steering outputs "
                             "(cluster_{key}/steering_results.jsonl)")
    parser.add_argument("--synthesis_dir", required=True,
                        help="Dir containing {cluster_key}_synthesis.json files")
    parser.add_argument("--prepared_samples_dir", required=True,
                        help="Dir with prepared vertex samples (for exemplar windows)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clusters", required=True,
                        help="Comma-separated cluster keys, e.g. 512_17,768_140")

    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 5.0, 20.0],
                        help="Steering scales to evaluate")
    parser.add_argument("--n_baseline", type=int, default=30,
                        help="Baseline (unsteered) examples per source vertex")
    parser.add_argument("--n_steered", type=int, default=30,
                        help="Steered examples per (src→tgt) direction "
                             "(same examples reused across all scales)")
    parser.add_argument("--n_exemplars", type=int, default=5,
                        help="Exemplar text windows per vertex shown in each prompt")

    parser.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    parser.add_argument("--quantization", default="awq_marlin")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_response_tokens", type=int, default=20,
                        help="Max tokens for the classification response")

    parser.add_argument("--gemma_tokenizer", default="google/gemma-2-9b",
                        help="Tokenizer used to decode prepared_samples token IDs "
                             "(must match the SAE base model, NOT the classifier LLM)")
    parser.add_argument("--cache_dir", default="/workspace/hf_cache")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    # Load Gemma tokenizer (cheap — just for decoding prepared_samples token IDs)
    print(f"Loading Gemma tokenizer ({args.gemma_tokenizer})...")
    from transformers import AutoTokenizer
    from huggingface_hub import login
    if args.hf_token:
        login(token=args.hf_token)
    gemma_tokenizer = AutoTokenizer.from_pretrained(
        args.gemma_tokenizer, cache_dir=args.cache_dir, token=args.hf_token
    )

    # Load vLLM classifier
    print(f"\nLoading {args.model_name} via vLLM...")
    from vllm import LLM
    llm_kwargs = dict(
        model=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
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
