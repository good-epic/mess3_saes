#!/usr/bin/env python3
"""
Feature regression validation for AANet simplex interpretations.

Tests whether measurable NLP features of trigger words can predict vertex
assignment. If the LLM interpretation is correct (e.g., "common nouns vs
proper nouns"), a POS tagger should classify vertex membership accurately.

Requires: pip install spacy && python -m spacy download en_core_web_sm

Usage:
    python validation/feature_regression.py \
        --prepared_samples_dir outputs/interpretations/prepared_samples_current \
        --output_dir outputs/validation/feature_regression

    # Single cluster:
    python validation/feature_regression.py \
        --prepared_samples_dir outputs/interpretations/prepared_samples_current \
        --output_dir outputs/validation/feature_regression \
        --clusters 512_464
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# =============================================================================
# Constants
# =============================================================================

FUNCTION_WORDS = {
    # Articles
    "a", "an", "the",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about",
    "into", "through", "during", "before", "after", "above", "below", "between",
    "under", "over", "out", "off", "down", "across", "behind", "along", "around",
    "near", "against", "upon", "within", "without", "toward", "towards",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "for", "because", "although",
    "while", "if", "when", "than", "that", "whether", "as",
    # Auxiliary verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    # Pronouns
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what",
    # Determiners / particles
    "not", "no", "some", "any", "each", "every", "all", "both", "few",
    "more", "most", "other", "such",
    # Other function words
    "there", "here", "then", "now", "very", "also", "just", "only", "even",
}

INSTRUCTION_VOCAB = {
    "must", "should", "shall", "required", "install", "click", "select",
    "enter", "type", "press", "open", "close", "create", "delete", "add",
    "remove", "set", "configure", "run", "execute", "use", "apply",
    "ensure", "verify", "check", "make", "note", "follow", "step",
    "download", "upload", "submit", "save", "copy", "paste", "navigate",
    "enable", "disable", "start", "stop", "restart",
}

DELIBERATIVE_VOCAB = {
    "study", "review", "assess", "investigate", "determine", "consider",
    "evaluate", "examine", "analyze", "explore", "research", "observe",
    "conclude", "suggest", "indicate", "demonstrate", "reveal", "find",
    "found", "report", "noted", "proposed", "hypothesize", "tested",
    "measured", "compared", "discussed", "considered", "addressed",
    "identified", "recommended", "concluded", "estimated", "calculated",
}


# =============================================================================
# Data Loading
# =============================================================================

def load_vertex_samples(prepared_path):
    """Load prepared vertex samples from JSON file.

    Returns dict mapping vertex_id (int) -> list of sample dicts.
    """
    with open(prepared_path) as f:
        data = json.load(f)

    samples_by_vertex = {}
    for vertex_str, samples in data["vertices"].items():
        vertex_id = int(vertex_str)
        samples_by_vertex[vertex_id] = samples

    return samples_by_vertex, data


def flatten_samples(samples_by_vertex):
    """Flatten vertex-grouped samples into a flat list with vertex labels.

    Each sample may have multiple trigger words; we create one row per trigger word.
    """
    rows = []
    for vertex_id, samples in samples_by_vertex.items():
        for sample in samples:
            trigger_words = sample.get("trigger_words", [])
            trigger_word_indices = sample.get("trigger_word_indices", [])
            full_text = sample.get("full_text", "")

            if not trigger_words:
                continue

            for tw, twi in zip(trigger_words, trigger_word_indices):
                rows.append({
                    "vertex_id": vertex_id,
                    "trigger_word": tw,
                    "trigger_word_idx": twi,
                    "full_text": full_text,
                })

    return rows


# =============================================================================
# Context Helpers
# =============================================================================

def get_trigger_context(full_text, trigger_word_idx, window=10):
    """Get surrounding words around the trigger word."""
    words = full_text.split()
    start = max(0, trigger_word_idx - window)
    end = min(len(words), trigger_word_idx + window + 1)
    return " ".join(words[start:end])


def has_camel_case(text):
    """Check if text contains camelCase patterns."""
    import re
    return bool(re.search(r'[a-z][A-Z]', text))


# =============================================================================
# Feature Extraction: Common
# =============================================================================

def extract_common_features(trigger_word, context, nlp):
    """Extract features common across all clusters."""
    tw = trigger_word.strip()
    tw_lower = tw.lower()

    # Basic word properties
    features = {
        "word_length": len(tw),
        "starts_uppercase": tw[0].isupper() if tw else False,
        "is_all_caps": tw.isupper() and len(tw) > 1,
        "is_all_alpha": tw.isalpha(),
        "has_digits": any(c.isdigit() for c in tw),
        "is_numeric": tw.replace(".", "").replace(",", "").replace("-", "").isdigit() if tw else False,
        "contains_punctuation": any(not c.isalnum() and c != " " for c in tw),
        "is_function_word": tw_lower in FUNCTION_WORDS,
    }

    # spacy features on the trigger word in context
    if nlp is not None and context:
        doc = nlp(context)
        # Find the trigger word token in the spacy parse
        trigger_token = None
        for token in doc:
            if token.text == tw or token.text.strip() == tw:
                trigger_token = token
                break

        if trigger_token is None:
            # Fallback: try matching lowercase
            for token in doc:
                if token.text.lower() == tw_lower:
                    trigger_token = token
                    break

        if trigger_token is None:
            # Last resort: parse just the trigger word
            tw_doc = nlp(tw)
            if len(tw_doc) > 0:
                trigger_token = tw_doc[0]

        if trigger_token is not None:
            features["pos_tag"] = trigger_token.pos_
            features["pos_fine"] = trigger_token.tag_
            features["is_proper_noun"] = trigger_token.pos_ == "PROPN"
            features["is_noun"] = trigger_token.pos_ == "NOUN"
            features["is_verb"] = trigger_token.pos_ == "VERB"
            features["is_adj"] = trigger_token.pos_ == "ADJ"
            features["is_adp"] = trigger_token.pos_ == "ADP"  # adposition (preposition)
            features["is_det"] = trigger_token.pos_ == "DET"
            features["is_named_entity"] = trigger_token.ent_type_ != ""
            features["ner_type"] = trigger_token.ent_type_ if trigger_token.ent_type_ else "NONE"
        else:
            features.update({
                "pos_tag": "UNK", "pos_fine": "UNK",
                "is_proper_noun": False, "is_noun": False, "is_verb": False,
                "is_adj": False, "is_adp": False, "is_det": False,
                "is_named_entity": False, "ner_type": "UNK",
            })

    return features


# =============================================================================
# Cluster-Specific Feature Extraction
# =============================================================================

def extract_features_512_464(rows, nlp):
    """512_464: Common nouns vs proper nouns. V0 vs V1."""
    records = []
    for row in rows:
        tw = row["trigger_word"]
        ctx = get_trigger_context(row["full_text"], row["trigger_word_idx"])
        feats = extract_common_features(tw, ctx, nlp)
        feats["vertex_id"] = row["vertex_id"]
        records.append(feats)

    df = pd.DataFrame(records)
    feature_cols = [
        "is_proper_noun", "is_named_entity", "starts_uppercase",
        "is_all_alpha", "word_length", "is_all_caps",
    ]
    primary_feature = "is_proper_noun"
    return df, feature_cols, primary_feature


def extract_features_768_484(rows, nlp):
    """768_484: Structured data vs narrative prose. V0 vs V2."""
    records = []
    for row in rows:
        tw = row["trigger_word"]
        ctx = get_trigger_context(row["full_text"], row["trigger_word_idx"])
        feats = extract_common_features(tw, ctx, nlp)

        # Context-level features
        context_words = ctx.split()
        if context_words:
            feats["context_digit_ratio"] = sum(
                any(c.isdigit() for c in w) for w in context_words
            ) / len(context_words)
            feats["context_punct_density"] = sum(
                sum(1 for c in w if not c.isalnum()) for w in context_words
            ) / len(context_words)
            feats["context_avg_word_length"] = np.mean([len(w) for w in context_words])
        else:
            feats["context_digit_ratio"] = 0
            feats["context_punct_density"] = 0
            feats["context_avg_word_length"] = 0

        feats["vertex_id"] = row["vertex_id"]
        records.append(feats)

    df = pd.DataFrame(records)
    feature_cols = [
        "has_digits", "is_numeric", "contains_punctuation", "word_length",
        "is_function_word", "context_digit_ratio", "context_punct_density",
        "context_avg_word_length",
    ]
    primary_feature = "has_digits"
    return df, feature_cols, primary_feature


def extract_features_512_504(rows, nlp):
    """512_504: Function words vs content nouns vs prepositions. Multi-class."""
    records = []
    for row in rows:
        tw = row["trigger_word"]
        ctx = get_trigger_context(row["full_text"], row["trigger_word_idx"])
        feats = extract_common_features(tw, ctx, nlp)

        # Coarse POS mapping
        pos = feats.get("pos_tag", "UNK")
        if pos in ("DET", "AUX", "CCONJ", "SCONJ", "PART", "PRON"):
            feats["pos_coarse"] = "function_word"
        elif pos in ("NOUN", "PROPN"):
            feats["pos_coarse"] = "noun"
        elif pos == "VERB":
            feats["pos_coarse"] = "verb"
        elif pos == "ADP":
            feats["pos_coarse"] = "preposition"
        elif pos == "ADJ":
            feats["pos_coarse"] = "adjective"
        elif pos == "ADV":
            feats["pos_coarse"] = "adverb"
        else:
            feats["pos_coarse"] = "other"

        feats["vertex_id"] = row["vertex_id"]
        records.append(feats)

    df = pd.DataFrame(records)
    feature_cols = [
        "is_function_word", "is_noun", "is_verb", "is_adp", "is_det",
        "is_proper_noun", "is_adj", "word_length",
    ]
    primary_feature = "pos_coarse"
    return df, feature_cols, primary_feature


def extract_features_512_292(rows, nlp):
    """512_292: Procedural instructions vs deliberative processes. V1 vs V2."""
    records = []
    for row in rows:
        tw = row["trigger_word"]
        tw_lower = tw.strip().lower()
        ctx = get_trigger_context(row["full_text"], row["trigger_word_idx"])
        feats = extract_common_features(tw, ctx, nlp)

        feats["in_instruction_vocab"] = tw_lower in INSTRUCTION_VOCAB
        feats["in_deliberative_vocab"] = tw_lower in DELIBERATIVE_VOCAB

        # Context features
        ctx_lower = ctx.lower()
        feats["context_has_code_markers"] = (
            "`" in ctx or has_camel_case(ctx) or "==" in ctx or
            "//" in ctx or "/*" in ctx or "def " in ctx_lower or
            "function " in ctx_lower or "class " in ctx_lower
        )

        # Check if trigger word starts a sentence (possible imperative)
        words = row["full_text"].split()
        idx = row["trigger_word_idx"]
        feats["starts_sentence"] = False
        if idx == 0:
            feats["starts_sentence"] = True
        elif idx > 0 and idx < len(words):
            prev_word = words[idx - 1]
            if prev_word.endswith((".","!","?",":")):
                feats["starts_sentence"] = True

        feats["vertex_id"] = row["vertex_id"]
        records.append(feats)

    df = pd.DataFrame(records)
    feature_cols = [
        "is_verb", "is_noun", "in_instruction_vocab", "in_deliberative_vocab",
        "context_has_code_markers", "starts_sentence", "is_function_word",
        "word_length",
    ]
    primary_feature = "in_instruction_vocab"
    return df, feature_cols, primary_feature


# =============================================================================
# Classification
# =============================================================================

def run_classification(df, feature_cols, primary_feature, cluster_key, n_folds=5, max_class_ratio=5):
    """Run classification to predict vertex_id from features.

    Args:
        df: DataFrame with features and vertex_id column
        feature_cols: list of numeric feature column names
        primary_feature: name of the single best feature for baseline test
        cluster_key: for labeling output
        n_folds: stratified CV folds
        max_class_ratio: if class imbalance exceeds this, subsample majority
    """
    results = {"cluster_key": cluster_key}

    # Get labels
    labels = df["vertex_id"].values
    classes = sorted(df["vertex_id"].unique())
    class_counts = Counter(labels)
    results["class_counts"] = {str(k): v for k, v in sorted(class_counts.items())}
    results["n_classes"] = len(classes)

    print(f"\n  Samples per vertex: {dict(sorted(class_counts.items()))}")

    # Handle class imbalance via subsampling
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())

    if max_count / min_count > max_class_ratio:
        print(f"  Subsampling majority class (ratio {max_count/min_count:.1f}x > {max_class_ratio}x)")
        target_per_class = min_count * max_class_ratio
        balanced_indices = []
        rng = np.random.RandomState(42)
        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            if len(cls_indices) > target_per_class:
                cls_indices = rng.choice(cls_indices, target_per_class, replace=False)
            balanced_indices.extend(cls_indices)
        df = df.iloc[balanced_indices].reset_index(drop=True)
        labels = df["vertex_id"].values
        class_counts = Counter(labels)
        print(f"  After subsampling: {dict(sorted(class_counts.items()))}")

    # Check minimum samples for CV
    min_class = min(class_counts.values())
    if min_class < n_folds:
        print(f"  WARNING: Smallest class has {min_class} samples, reducing to {min_class}-fold CV")
        n_folds = max(2, min_class)

    # Prepare feature matrix
    X = df[feature_cols].values.astype(float)
    y = labels

    # Handle any NaN
    X = np.nan_to_num(X, nan=0.0)

    # --- 1. Single-feature baseline ---
    if primary_feature in df.columns:
        if primary_feature in feature_cols:
            # Numeric feature — use as single predictor
            X_single = df[[primary_feature]].values.astype(float)
            X_single = np.nan_to_num(X_single, nan=0.0)
        else:
            # Categorical feature — one-hot encode
            le = LabelEncoder()
            encoded = le.fit_transform(df[primary_feature].fillna("UNK"))
            X_single = encoded.reshape(-1, 1).astype(float)

        try:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            preds_single = cross_val_predict(
                LogisticRegression(max_iter=1000, class_weight="balanced"),
                X_single, y, cv=skf
            )
            single_acc = accuracy_score(y, preds_single)
            single_bal_acc = balanced_accuracy_score(y, preds_single)
            results["single_feature"] = {
                "feature": primary_feature,
                "accuracy": single_acc,
                "balanced_accuracy": single_bal_acc,
            }
            print(f"  Single-feature baseline ({primary_feature}): "
                  f"acc={single_acc:.3f}, balanced_acc={single_bal_acc:.3f}")
        except Exception as e:
            print(f"  Single-feature baseline failed: {e}")
            results["single_feature"] = {"error": str(e)}

    # --- 2. Logistic regression with all features ---
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        preds_lr = cross_val_predict(lr, X, y, cv=skf)
        lr_acc = accuracy_score(y, preds_lr)
        lr_bal_acc = balanced_accuracy_score(y, preds_lr)

        # Confusion matrix
        cm = confusion_matrix(y, preds_lr, labels=classes)

        # Feature importances from full-data fit
        lr.fit(X, y)
        if len(classes) == 2:
            importances = np.abs(lr.coef_[0])
        else:
            importances = np.mean(np.abs(lr.coef_), axis=0)

        importance_ranking = sorted(
            zip(feature_cols, importances), key=lambda x: -x[1]
        )

        results["logistic_regression"] = {
            "accuracy": lr_acc,
            "balanced_accuracy": lr_bal_acc,
            "confusion_matrix": cm.tolist(),
            "confusion_labels": [str(c) for c in classes],
            "feature_importances": {name: float(imp) for name, imp in importance_ranking},
        }

        print(f"  Logistic regression (all features): "
              f"acc={lr_acc:.3f}, balanced_acc={lr_bal_acc:.3f}")
        print(f"  Feature importances:")
        for name, imp in importance_ranking[:5]:
            print(f"    {name:<30s} {imp:.4f}")
        print(f"  Confusion matrix (rows=actual, cols=predicted):")
        print(f"    Classes: {classes}")
        for i, row in enumerate(cm):
            print(f"    V{classes[i]}: {row}")

    except Exception as e:
        print(f"  Logistic regression failed: {e}")
        results["logistic_regression"] = {"error": str(e)}

    # --- 3. Random forest (for comparison) ---
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        )
        preds_rf = cross_val_predict(rf, X, y, cv=skf)
        rf_acc = accuracy_score(y, preds_rf)
        rf_bal_acc = balanced_accuracy_score(y, preds_rf)

        rf.fit(X, y)
        rf_importances = sorted(
            zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1]
        )

        results["random_forest"] = {
            "accuracy": rf_acc,
            "balanced_accuracy": rf_bal_acc,
            "feature_importances": {name: float(imp) for name, imp in rf_importances},
        }

        print(f"  Random forest: acc={rf_acc:.3f}, balanced_acc={rf_bal_acc:.3f}")

    except Exception as e:
        print(f"  Random forest failed: {e}")
        results["random_forest"] = {"error": str(e)}

    # --- 4. Majority-class baseline ---
    majority_class = max(class_counts, key=class_counts.get)
    majority_acc = class_counts[majority_class] / sum(class_counts.values())
    results["majority_baseline"] = majority_acc
    print(f"  Majority-class baseline: acc={majority_acc:.3f}")

    # --- 5. POS tag distribution per vertex (descriptive) ---
    if "pos_tag" in df.columns:
        pos_by_vertex = {}
        for vertex in classes:
            vertex_df = df[df["vertex_id"] == vertex]
            pos_counts = vertex_df["pos_tag"].value_counts().to_dict()
            pos_by_vertex[str(vertex)] = pos_counts
        results["pos_distribution"] = pos_by_vertex

        print(f"  POS tag distribution by vertex:")
        for vertex in classes:
            top_pos = sorted(pos_by_vertex[str(vertex)].items(), key=lambda x: -x[1])[:5]
            top_str = ", ".join(f"{pos}={count}" for pos, count in top_pos)
            print(f"    V{vertex}: {top_str}")

    return results


# =============================================================================
# Contingency Tables
# =============================================================================

def build_contingency_feature(df, cluster_key):
    """Create a human-readable categorical feature for contingency tables.

    Returns (series, feature_name) or (None, None) if not applicable.
    """
    if cluster_key == "512_464":
        if "pos_tag" in df.columns:
            def map_pos(pos):
                if pos == "PROPN":
                    return "PROPN (proper noun)"
                elif pos == "NOUN":
                    return "NOUN (common noun)"
                else:
                    return f"other ({pos})"
            return df["pos_tag"].map(map_pos), "POS category"
        elif "is_proper_noun" in df.columns:
            return df["is_proper_noun"].map(
                {True: "proper noun", False: "not proper noun"}
            ), "is_proper_noun"

    elif cluster_key == "512_504":
        if "pos_coarse" in df.columns:
            return df["pos_coarse"], "POS coarse"

    elif cluster_key == "512_292":
        if "in_instruction_vocab" in df.columns and "in_deliberative_vocab" in df.columns:
            def map_vocab(row):
                if row["in_instruction_vocab"]:
                    return "instruction vocab"
                elif row["in_deliberative_vocab"]:
                    return "deliberative vocab"
                else:
                    return "other"
            return df.apply(map_vocab, axis=1), "vocab category"

    elif cluster_key == "768_484":
        if "has_digits" in df.columns:
            return df["has_digits"].map(
                {True: "has digits", False: "no digits"}
            ), "digit content"

    return None, None


def print_contingency_tables(df, cat_series, feature_name):
    """Print contingency tables between a categorical feature and vertex_id.

    Prints both orientations:
      1. Feature → vertex  (row % within each feature category)
      2. Vertex  → feature (col % within each vertex)

    Returns a dict suitable for JSON serialization.
    """
    import pandas as pd

    vertices = sorted(df["vertex_id"].unique())
    ct = pd.crosstab(cat_series, df["vertex_id"])
    # Ensure all vertices present
    for v in vertices:
        if v not in ct.columns:
            ct[v] = 0
    ct = ct[vertices]
    row_totals = ct.sum(axis=1)
    col_totals = ct.sum(axis=0)
    grand_total = ct.values.sum()

    col_w = 8  # width per vertex column

    def header_line():
        return "  " + f"{'':28}" + "".join(f"V{v:>{col_w - 1}}" for v in vertices) + f"{'n':>{col_w}}"

    def divider():
        return "  " + "-" * (28 + col_w * (len(vertices) + 1))

    # ---- 1. Feature → vertex (row %) ----------------------------------------
    print(f"\n  [{feature_name}] → vertex  (row %: of all samples with this feature, what fraction go to each vertex?)")
    print(header_line())
    print(divider())
    feat_to_vertex = {}
    for cat in ct.index:
        n = int(row_totals[cat])
        row_pcts = {v: 100.0 * ct.loc[cat, v] / n if n > 0 else 0.0 for v in vertices}
        feat_to_vertex[str(cat)] = {"pcts": row_pcts, "n": n}
        cat_display = str(cat)[:26]
        pct_str = "".join(f"{row_pcts[v]:>{col_w}.1f}%" for v in vertices)
        print(f"  {cat_display:<28}{pct_str}{n:>{col_w}}")
    print(divider())
    total_pcts = "".join(f"{100.0 * col_totals[v] / grand_total:>{col_w}.1f}%" for v in vertices)
    print(f"  {'(marginal)':28}{total_pcts}{grand_total:>{col_w}}")

    # ---- 2. Vertex → feature (col %) ----------------------------------------
    print(f"\n  vertex → [{feature_name}]  (col %: of all samples in this vertex, what fraction have each feature?)")
    print(header_line())
    print(divider())
    vertex_to_feat = {str(v): {} for v in vertices}
    for cat in ct.index:
        col_pcts = {v: 100.0 * ct.loc[cat, v] / col_totals[v] if col_totals[v] > 0 else 0.0
                    for v in vertices}
        for v in vertices:
            vertex_to_feat[str(v)][str(cat)] = col_pcts[v]
        cat_display = str(cat)[:26]
        pct_str = "".join(f"{col_pcts[v]:>{col_w}.1f}%" for v in vertices)
        print(f"  {cat_display:<28}{pct_str}")
    print(divider())
    n_str = "".join(f"{int(col_totals[v]):>{col_w}}" for v in vertices)
    print(f"  {'n (vertex total)':28}{n_str}")

    return {
        "feature_name": feature_name,
        "feature_to_vertex": feat_to_vertex,
        "vertex_to_feature": vertex_to_feat,
        "counts": {
            str(cat): {str(v): int(ct.loc[cat, v]) for v in vertices}
            for cat in ct.index
        },
    }


# =============================================================================
# Main
# =============================================================================

CLUSTER_EXTRACTORS = {
    "512_464": ("Common nouns vs proper nouns", extract_features_512_464),
    "768_484": ("Structured data vs narrative prose", extract_features_768_484),
    "512_504": ("Function words vs content nouns vs prepositions", extract_features_512_504),
    "512_292": ("Procedural instructions vs deliberative processes", extract_features_512_292),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Feature regression validation")
    parser.add_argument("--prepared_samples_dir", type=str, required=True,
                        help="Directory with prepared sample JSON files")
    parser.add_argument("--output_dir", type=str, default="outputs/validation/feature_regression")
    parser.add_argument("--clusters", type=str, default="all",
                        help="Comma-separated cluster keys (e.g., '512_464,768_484') or 'all'")
    parser.add_argument("--n_folds", type=int, default=5, help="CV folds")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load spacy
    print("Loading spacy model...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("  Loaded en_core_web_sm")
    except (ImportError, OSError) as e:
        print(f"  WARNING: spacy not available ({e})")
        print("  Install with: pip install spacy && python -m spacy download en_core_web_sm")
        print("  Running without NLP features (reduced accuracy expected)")
        nlp = None

    # Determine which clusters to process
    if args.clusters == "all":
        clusters = list(CLUSTER_EXTRACTORS.keys())
    else:
        clusters = [c.strip() for c in args.clusters.split(",")]

    print("\n" + "=" * 80)
    print("FEATURE REGRESSION VALIDATION")
    print("=" * 80)

    all_results = {}

    for cluster_key in clusters:
        if cluster_key not in CLUSTER_EXTRACTORS:
            print(f"\nSkipping {cluster_key}: no feature extractor defined")
            continue

        description, extractor_fn = CLUSTER_EXTRACTORS[cluster_key]

        print(f"\n{'=' * 80}")
        print(f"CLUSTER {cluster_key}: {description}")
        print(f"{'=' * 80}")

        # Load samples
        sample_path = Path(args.prepared_samples_dir) / f"cluster_{cluster_key}.json"
        if not sample_path.exists():
            print(f"  ERROR: Sample file not found: {sample_path}")
            continue

        samples_by_vertex, metadata = load_vertex_samples(sample_path)
        print(f"  Loaded from {sample_path}")
        print(f"  k={metadata['k']}, n_latents={metadata['n_latents']}")

        # Flatten samples (one row per trigger word)
        rows = flatten_samples(samples_by_vertex)
        if not rows:
            print(f"  ERROR: No samples found")
            continue

        # Filter whitespace trigger words
        rows = [r for r in rows if r["trigger_word"].strip()]
        print(f"  Total trigger word instances: {len(rows)}")

        # Extract features
        print(f"  Extracting features...")
        df, feature_cols, primary_feature = extractor_fn(rows, nlp)

        if len(df) < 10:
            print(f"  ERROR: Too few samples ({len(df)})")
            continue

        # Run classification
        results = run_classification(
            df, feature_cols, primary_feature, cluster_key, n_folds=args.n_folds
        )

        # Add metadata
        results["description"] = description
        results["k"] = metadata["k"]
        results["n_latents"] = metadata["n_latents"]
        results["latent_indices"] = metadata.get("latent_indices", [])
        results["feature_cols"] = feature_cols
        results["primary_feature"] = primary_feature

        # Contingency tables
        cat_series, feature_name = build_contingency_feature(df, cluster_key)
        if cat_series is not None:
            print(f"\n  --- Contingency tables ---")
            results["contingency_tables"] = print_contingency_tables(
                df, cat_series, feature_name
            )

        all_results[cluster_key] = results

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Cluster':<12} {'Description':<45} {'Majority':>9} {'Single':>8} {'LR':>8} {'RF':>8} {'Verdict'}")
    print("-" * 110)

    for cluster_key, results in all_results.items():
        majority = results.get("majority_baseline", 0)

        single_acc = "-"
        if "single_feature" in results and "balanced_accuracy" in results["single_feature"]:
            single_acc = f"{results['single_feature']['balanced_accuracy']:.3f}"

        lr_acc = "-"
        if "logistic_regression" in results and "balanced_accuracy" in results["logistic_regression"]:
            lr_acc = f"{results['logistic_regression']['balanced_accuracy']:.3f}"

        rf_acc = "-"
        if "random_forest" in results and "balanced_accuracy" in results["random_forest"]:
            rf_acc = f"{results['random_forest']['balanced_accuracy']:.3f}"

        # Verdict
        best_acc = 0
        for key in ["logistic_regression", "random_forest"]:
            if key in results and "balanced_accuracy" in results[key]:
                best_acc = max(best_acc, results[key]["balanced_accuracy"])

        if best_acc > 0.85:
            verdict = "STRONG"
        elif best_acc > 0.70:
            verdict = "MODERATE"
        elif best_acc > majority + 0.1:
            verdict = "WEAK"
        else:
            verdict = "NONE"

        print(f"{cluster_key:<12} {results['description']:<45} {majority:>9.3f} "
              f"{single_acc:>8} {lr_acc:>8} {rf_acc:>8} {verdict:>8}")

    # Save results
    results_path = output_dir / "feature_regression_results.json"

    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
