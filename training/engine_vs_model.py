#!/usr/bin/env python3
"""Engine vs Model — Adversarial comparison to find blind spots.

Runs both the rule-based engine and the trained neural model on the same
sentences, then finds where they DISAGREE. Disagreements are clustered
by pattern to identify systematic gaps.

Where they AGREE = physics confirmed (both see the same truth).
Where MODEL is wrong = needs more training data for that pattern.
Where ENGINE is wrong = physics gap found, needs new force or fix.

Usage:
    python3 training/engine_vs_model.py
    python3 training/engine_vs_model.py --sentences 5000
    python3 training/engine_vs_model.py --verbose
"""

import json
import os
import sys
import time
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer
from demo.pendulum_v2 import PendulumV2
from training.train import ClankerClassifier, VADUG_DIMS, BUCKET_THRESHOLDS, value_to_bucket


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_engine():
    """Load the 25-force champion engine."""
    return PendulumV2(
        momentum=0.903, force_scale=2.894, direct_push_cap=0.085,
        direct_push_trigger=158.7, crisis_v=68.9, crisis_u=95.5,
        question_dampener=0.529, scale_v=1.920, scale_a=0.789,
        scale_d=2.901, scale_u=0.30, scale_g=0.707,
        threshold_low=116.7, threshold_high=155.2,
        negation_decay_operator=0.850, negation_decay_neutral=0.60,
        negation_decay_payload=0.523,
    )


def load_model(checkpoint_dir="training/checkpoints/best"):
    """Load the trained Clanker-Micro model."""
    model = ClankerClassifier.load_pretrained(checkpoint_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer


def model_predict(model, tokenizer, text):
    """Run model inference on a single sentence. Returns approximate VADUG."""
    enc = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    emotion_pos = torch.tensor([enc['input_ids'].shape[1] - 1])
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'], emotion_pos)
    preds = {dim: torch.argmax(logits[dim], dim=-1).item() for dim in VADUG_DIMS}
    vals = {dim: int(BUCKET_THRESHOLDS[preds[dim]-1] + 18) if preds[dim] > 0 else 18 for dim in VADUG_DIMS}
    return vals


def classify_v(v):
    """Simple 3-class classification from V value."""
    if v > 147:
        return "positive"
    elif v < 110:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Sentence sources
# ---------------------------------------------------------------------------

def load_test_sentences(max_n=5000):
    """Load diverse test sentences from multiple sources."""
    sentences = []

    # Rosetta Stone (hand-crafted calibration)
    rosetta_path = "training/data/phase1.jsonl"
    if os.path.exists(rosetta_path):
        with open(rosetta_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sentences.append(data["english"])

    # Pangram essays (every emotional word in context)
    pangram_path = "training/data/pangram_essays.jsonl"
    if os.path.exists(pangram_path):
        with open(pangram_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sentences.append(data.get("text", data.get("english", "")))

    # Sarcasm calibration
    sarc_path = "training/data/sarcasm_calibration.jsonl"
    if os.path.exists(sarc_path):
        with open(sarc_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sentences.append(data["english"])

    # Negation calibration
    neg_path = "training/data/negation_calibration.jsonl"
    if os.path.exists(neg_path):
        with open(neg_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sentences.append(data["english"])

    # Additional test sentences covering known patterns
    extra = [
        # Sarcasm
        "Oh great another Monday",
        "Yeah sure that went really well",
        "Wow what a wonderful surprise",
        "Oh fantastic the printer is broken again",
        # Negation
        "I am not happy", "I am not sad", "I do not feel safe",
        "This is not okay", "I cannot believe this",
        # Hedging
        "I think I might be upset", "Maybe I am worried",
        "I guess that is fine", "Possibly I should be concerned",
        # Conditional
        "If that happened I would be furious",
        "Unless you stop I am done",
        # Evidential
        "She said she is devastated",
        "The subject reports feeling overwhelmed",
        # Evokers
        "I lost my job before the wedding",
        "I am happy about the baby",
        "I am scared after the cancer diagnosis",
        # Crisis
        "I want to die", "I want to die everything is hopeless",
        "Nobody would even notice if I disappeared",
        # Passive
        "I was hurt", "I was abandoned", "I was betrayed",
        # Politeness
        "Honestly you are wrong", "With all due respect that is stupid",
        # Tag questions
        "That is bad right?", "You agree huh?",
    ]
    sentences.extend(extra)

    # Deduplicate and limit
    seen = set()
    unique = []
    for s in sentences:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)

    if max_n and len(unique) > max_n:
        import random
        random.seed(42)
        unique = random.sample(unique, max_n)

    return unique


# ---------------------------------------------------------------------------
# Pattern classification for disagreements
# ---------------------------------------------------------------------------

PATTERN_KEYWORDS = {
    "negation": ["not", "never", "no", "don't", "can't", "won't", "didn't",
                 "isn't", "aren't", "wasn't", "weren't", "nobody", "nothing",
                 "cannot", "hardly", "barely"],
    "sarcasm": ["oh", "yeah", "sure", "great", "wonderful", "fantastic",
                "brilliant", "lovely", "perfect", "thanks"],
    "hedging": ["maybe", "perhaps", "possibly", "might", "could", "would",
                "think", "guess", "probably", "somewhat", "slightly"],
    "conditional": ["if", "unless", "assuming", "supposing"],
    "evidential": ["said", "told", "claims", "reports", "apparently",
                   "allegedly", "supposedly"],
    "clinical": ["subject", "patient", "client", "individual"],
    "evoker": ["death", "cancer", "wedding", "funeral", "divorce", "children",
               "god", "war", "freedom", "mother", "prison"],
    "passive": ["was", "were", "been", "got"],
    "question": ["?", "why", "how", "what", "who"],
    "crisis": ["die", "kill", "suicide", "hopeless", "worthless"],
}


def classify_pattern(text):
    """Classify a sentence's dominant pattern based on keywords."""
    text_lower = text.lower()
    words = text_lower.split()
    scores = {}
    for pattern, keywords in PATTERN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in words or kw in text_lower)
        if score > 0:
            scores[pattern] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(sentences, engine, model, tokenizer, verbose=False):
    """Run both systems on all sentences. Return disagreement analysis."""

    agreements = 0
    disagreements = []

    t0 = time.time()

    for i, text in enumerate(sentences):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i+1}/{len(sentences)}...", end="\r")

        # Engine prediction
        vadug_engine, trace = engine.process_text(text)
        engine_v = vadug_engine.v
        engine_class = classify_v(engine_v)

        # Model prediction
        model_vals = model_predict(model, tokenizer, text)
        model_v = model_vals["V"]
        model_class = classify_v(model_v)

        if engine_class == model_class:
            agreements += 1
        else:
            pattern = classify_pattern(text)
            disagreements.append({
                "text": text,
                "engine_class": engine_class, "engine_v": engine_v,
                "model_class": model_class, "model_v": model_v,
                "delta_v": abs(engine_v - model_v),
                "pattern": pattern,
            })

    elapsed = time.time() - t0

    # Analyze disagreements
    pattern_counts = Counter(d["pattern"] for d in disagreements)

    pattern_details = defaultdict(list)
    for d in disagreements:
        pattern_details[d["pattern"]].append(d)

    # Find biggest V deltas
    disagreements.sort(key=lambda d: d["delta_v"], reverse=True)

    return {
        "total": len(sentences),
        "agreements": agreements,
        "disagreements": len(disagreements),
        "agreement_rate": round(agreements / len(sentences) * 100, 1),
        "elapsed": round(elapsed, 1),
        "pattern_counts": dict(pattern_counts.most_common()),
        "pattern_details": dict(pattern_details),
        "top_disagreements": disagreements[:50],
        "all_disagreements": disagreements,
    }


def print_report(results, verbose=False):
    """Print the comparison report."""
    print(f"\n{'='*70}")
    print(f"  ENGINE vs MODEL — Adversarial Comparison")
    print(f"{'='*70}\n")

    print(f"  Total sentences:    {results['total']}")
    print(f"  Agreements:         {results['agreements']} ({results['agreement_rate']}%)")
    print(f"  Disagreements:      {results['disagreements']} ({100 - results['agreement_rate']}%)")
    print(f"  Time:               {results['elapsed']}s")

    print(f"\n  DISAGREEMENT PATTERNS (where engine and model see different things):")
    print(f"  {'Pattern':<15s} {'Count':>6s} {'%':>7s}  Action")
    print(f"  {'-'*60}")
    total_dis = results['disagreements'] or 1
    for pattern, count in sorted(results['pattern_counts'].items(), key=lambda x: -x[1]):
        pct = count / total_dis * 100
        action = {
            "negation": "-> Check model negation learning",
            "sarcasm": "-> Engine needs better sarcasm detection",
            "hedging": "-> Check hedging coefficient tuning",
            "conditional": "-> Check conditional dampener",
            "evidential": "-> Check evidential distance",
            "clinical": "-> Check clinical dampener",
            "evoker": "-> Check evoker gravity values",
            "passive": "-> Check passive D-offset",
            "question": "-> Check question dampener",
            "crisis": "-> CRITICAL: crisis routing disagreement",
            "general": "-> General vocabulary gap",
        }.get(pattern, "-> Investigate")
        print(f"  {pattern:<15s} {count:>6d} {pct:>6.1f}%  {action}")

    print(f"\n  TOP 20 BIGGEST DISAGREEMENTS (by V delta):")
    print(f"  {'dV':>4s} {'Eng':>5s} {'Mod':>5s} {'Pattern':<12s} Text")
    print(f"  {'-'*70}")
    for d in results['top_disagreements'][:20]:
        print(f"  {d['delta_v']:>4d} {d['engine_v']:>5d} {d['model_v']:>5d} "
              f"{d['pattern']:<12s} \"{d['text'][:50]}\"")

    if verbose:
        print(f"\n  ALL DISAGREEMENTS BY PATTERN:")
        for pattern, details in sorted(results['pattern_details'].items()):
            print(f"\n  --- {pattern.upper()} ({len(details)} disagreements) ---")
            for d in details[:10]:
                print(f"    dV={d['delta_v']:3d}  Eng={d['engine_v']:3d}({d['engine_class']}) "
                      f"Mod={d['model_v']:3d}({d['model_class']})  \"{d['text'][:60]}\"")

    # Save full results
    output_path = os.path.join(os.path.dirname(__file__), "data", "engine_vs_model.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for d in results['all_disagreements']:
            f.write(json.dumps(d) + "\n")
    print(f"\n  Saved {len(results['all_disagreements'])} disagreements to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Engine vs Model adversarial comparison")
    parser.add_argument("--sentences", type=int, default=5000, help="Max sentences to test")
    parser.add_argument("--verbose", action="store_true", help="Show all disagreements by pattern")
    args = parser.parse_args()

    print("  Loading engine (25-force champion)...")
    engine = load_engine()

    print("  Loading model (Clanker-Micro, latest checkpoint)...")
    model, tokenizer = load_model()

    print(f"  Loading test sentences (max {args.sentences})...")
    sentences = load_test_sentences(args.sentences)
    print(f"  Loaded {len(sentences)} unique sentences")

    results = run_comparison(sentences, engine, model, tokenizer, args.verbose)
    print_report(results, args.verbose)


if __name__ == "__main__":
    main()
