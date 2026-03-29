#!/usr/bin/env python3
"""
Per-Dataset Threshold Tuning Sweep

Runs a bracket tournament sweep SEPARATELY on each dataset (SST-2, GoEmotions,
TweetEval) to find per-dataset optimal thresholds for neutral_low, neutral_high,
and momentum.

Sweep ranges:
  neutral_low:  95 to 125 (step 5)   → 7 values
  neutral_high: 130 to 160 (step 5)  → 7 values
  momentum:     0.70 to 0.90 (step 0.05) → 5 values
  Total: 7 × 7 × 5 = 245 configs per dataset

Usage:
    python3 benchmarks/per_dataset_tune.py
"""

import itertools
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.pendulum import SequentialPendulum

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets")
    sys.exit(1)


# ── GoEmotions label mapping (same as academic_benchmark.py) ──────────────────
GOEMO = {
    0: ("admiration", "positive"), 1: ("amusement", "positive"),
    2: ("anger", "negative"), 3: ("annoyance", "negative"),
    4: ("approval", "positive"), 5: ("caring", "positive"),
    6: ("confusion", "neutral"), 7: ("curiosity", "neutral"),
    8: ("desire", "positive"), 9: ("disappointment", "negative"),
    10: ("disapproval", "negative"), 11: ("disgust", "negative"),
    12: ("embarrassment", "negative"), 13: ("excitement", "positive"),
    14: ("fear", "negative"), 15: ("gratitude", "positive"),
    16: ("grief", "negative"), 17: ("joy", "positive"),
    18: ("love", "positive"), 19: ("nervousness", "negative"),
    20: ("optimism", "positive"), 21: ("pride", "positive"),
    22: ("realization", "neutral"), 23: ("relief", "positive"),
    24: ("remorse", "negative"), 25: ("sadness", "negative"),
    26: ("surprise", "neutral"), 27: ("neutral", "neutral"),
}

# ── TweetEval emotion label mapping ──────────────────────────────────────────
# 0=anger, 1=joy, 2=optimism, 3=sadness
TWEET_SENT = {0: "negative", 1: "positive", 2: "positive", 3: "negative"}


# ── Dataset definitions ───────────────────────────────────────────────────────

DATASETS = {
    "sst2": {
        "loader": lambda: load_dataset("stanfordnlp/sst2", split="validation"),
        "get_text": lambda r: r["sentence"],
        "get_label": lambda r: "positive" if r["label"] == 1 else "negative",
    },
    "goemotions": {
        "loader": lambda: load_dataset(
            "google-research-datasets/go_emotions", split="test"
        ).select(range(1000)),
        "get_text": lambda r: r["text"],
        "get_label": lambda r: (
            GOEMO.get(r["labels"][0], ("unk", "neutral"))[1] if r["labels"] else None
        ),
    },
    "tweeteval": {
        "loader": lambda: load_dataset(
            "cardiffnlp/tweet_eval", "emotion", split="test"
        ),
        "get_text": lambda r: r["text"],
        "get_label": lambda r: TWEET_SENT[r["label"]],
    },
}


def run_clanker_with_config(text, momentum):
    """Run the pendulum on text with a given momentum value. Returns final V."""
    p = SequentialPendulum()
    p.momentum = momentum
    words = re.findall(r"[a-z']+", text.lower())
    for i, w in enumerate(words):
        p.process_word(w, words, i)
    return int(p.v)


def classify(v, neutral_low, neutral_high):
    """Classify a valence score into positive/negative/neutral."""
    if v > neutral_high:
        return "positive"
    elif v < neutral_low:
        return "negative"
    else:
        return "neutral"


def evaluate_config(samples, neutral_low, neutral_high, momentum):
    """Run all samples with given config, return accuracy."""
    correct = 0
    total = 0
    for text, truth in samples:
        v = run_clanker_with_config(text, momentum)
        pred = classify(v, neutral_low, neutral_high)
        if pred == truth:
            correct += 1
        total += 1
    return correct / total * 100 if total > 0 else 0.0


def load_samples(ds_name, ds_config):
    """Load and pre-process samples as (text, label) tuples."""
    print(f"  Loading {ds_name}...")
    ds = ds_config["loader"]()
    get_text = ds_config["get_text"]
    get_label = ds_config["get_label"]
    samples = []
    for row in ds:
        label = get_label(row)
        if label is None:
            continue
        samples.append((get_text(row), label))
    print(f"  Loaded {len(samples)} samples for {ds_name}")
    return samples


def sweep_dataset(ds_name, samples):
    """Sweep all 245 configs for a dataset. Returns (best_config, best_acc)."""
    neutral_lows = list(range(95, 126, 5))    # 95, 100, 105, 110, 115, 120, 125
    neutral_highs = list(range(130, 161, 5))   # 130, 135, 140, 145, 150, 155, 160
    momentums = [round(0.70 + i * 0.05, 2) for i in range(5)]  # 0.70 to 0.90

    total_configs = len(neutral_lows) * len(neutral_highs) * len(momentums)
    print(f"  Sweeping {total_configs} configs for {ds_name} ({len(samples)} samples)...")

    best_acc = 0.0
    best_config = None
    done = 0
    t0 = time.perf_counter()

    for nl, nh, mom in itertools.product(neutral_lows, neutral_highs, momentums):
        acc = evaluate_config(samples, nl, nh, mom)
        done += 1
        if acc > best_acc:
            best_acc = acc
            best_config = {"neutral_low": nl, "neutral_high": nh, "momentum": mom}
        if done % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {done}/{total_configs} configs tested ({elapsed:.1f}s)...", end="\r")

    elapsed = time.perf_counter() - t0
    print(f"  {ds_name}: swept {total_configs} configs in {elapsed:.1f}s")
    return best_config, best_acc


def main():
    print(f"\n{'#' * 70}")
    print(f"  PER-DATASET THRESHOLD TUNING SWEEP")
    print(f"{'#' * 70}\n")

    results = {}

    for ds_name, ds_config in DATASETS.items():
        samples = load_samples(ds_name, ds_config)
        best_config, best_acc = sweep_dataset(ds_name, samples)
        results[ds_name] = {
            "neutral_low": best_config["neutral_low"],
            "neutral_high": best_config["neutral_high"],
            "momentum": best_config["momentum"],
            "accuracy": round(best_acc, 1),
            "samples": len(samples),
        }

    # Print summary
    print(f"\n{'=' * 70}")
    print("PER-DATASET OPTIMAL THRESHOLDS:")
    for ds_name, r in results.items():
        nl, nh, mom, acc = r["neutral_low"], r["neutral_high"], r["momentum"], r["accuracy"]
        print(f"  {ds_name:<14} neutral=[{nl}, {nh}], momentum={mom:.2f} -> {acc:.1f}%")
    print(f"{'=' * 70}\n")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "per_dataset_thresholds.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
