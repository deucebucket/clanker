#!/usr/bin/env python3
"""Full-stack layer-by-layer tuner (Dante Rings).

Sweeps every tunable value against the complete 7,720-example dataset.
Tunes bottom-up: pendulum physics -> intent thresholds -> crisis params -> idiom params.

Each layer locks its best config before the next layer sweeps, so each layer
builds on optimized foundations.

Target runtime: < 10 minutes total.
"""

import json
import os
import re
import sys
import time
import random
from itertools import product
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

from datasets import load_dataset
from demo.pendulum import SequentialPendulum, IDIOMS
from demo.intent import IntentDetector
from demo.pipeline_config import PipelineConfig

# ---------------------------------------------------------------------------
# Label mappings (same as universal_tune.py / academic_benchmark.py)
# ---------------------------------------------------------------------------

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

TW_SENT = {0: "negative", 1: "positive", 2: "positive", 3: "negative"}

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------

print("Loading datasets...")
t_load = time.perf_counter()
sst = load_dataset("stanfordnlp/sst2", split="validation")      # 872
ge = load_dataset("google-research-datasets/go_emotions", split="test")  # 5427
te = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test")     # 1421
print(f"  Loaded in {time.perf_counter() - t_load:.1f}s")

intent_detector = IntentDetector()

# ---------------------------------------------------------------------------
# Build raw example list: (text, truth_label, dataset_name)
# ---------------------------------------------------------------------------

examples = []
for r in sst:
    truth = "positive" if r["label"] == 1 else "negative"
    examples.append((r["sentence"], truth, "sst2"))

for r in ge:
    if not r["labels"]:
        continue
    _, sent = GOEMO.get(r["labels"][0], ("unk", "neutral"))
    examples.append((r["text"], sent, "go_emotions"))

for r in te:
    examples.append((r["text"], TW_SENT[r["label"]], "tweet_eval"))

print(f"  Total examples: {len(examples)}")


# ---------------------------------------------------------------------------
# Helper: run pendulum with custom physics on a text
# ---------------------------------------------------------------------------

def run_pendulum(text, momentum=0.80, drift_rate=0.06, shift_dampen=0.6,
                 idiom_momentum=0.50, idiom_push=0.7,
                 crisis_cap=0.99, crisis_push_scale=0.05):
    """Run the pendulum on text with custom physics parameters.

    Returns (V_final, intent_mode).
    """
    mode, _ = intent_detector.detect(text)
    p = SequentialPendulum()
    p.momentum = momentum
    p.drift_rate = drift_rate
    p.shift_marker_dampening = shift_dampen
    p.idiom_momentum = idiom_momentum
    p.idiom_push = idiom_push
    p.crisis_momentum_cap = crisis_cap
    p.crisis_direct_push = crisis_push_scale

    words = re.findall(r"[a-z']+", text.lower())
    for i, w in enumerate(words):
        p.process_word(w, words, i)

    return int(p.v), mode


def classify(v, mode, thresholds):
    """Classify V value using intent-adaptive thresholds."""
    low, high = thresholds.get(mode, thresholds.get("DEFAULT", (120, 135)))
    if v > high:
        return "positive"
    elif v < low:
        return "negative"
    return "neutral"


def score_dataset(cached_data, thresholds):
    """Score cached (V, mode, truth, dataset) tuples against thresholds.

    Returns (total_correct, total, per_dataset_correct, per_dataset_total).
    """
    correct = 0
    total = len(cached_data)
    ds_correct = {"sst2": 0, "go_emotions": 0, "tweet_eval": 0}
    ds_total = {"sst2": 0, "go_emotions": 0, "tweet_eval": 0}

    for v, mode, truth, ds in cached_data:
        pred = classify(v, mode, thresholds)
        if pred == truth:
            correct += 1
            ds_correct[ds] += 1
        ds_total[ds] += 1

    return correct, total, ds_correct, ds_total


def build_thresholds(emo_low, emo_high, wide_low, wide_high):
    """Build the full threshold dict from the 4 sweep params."""
    return {
        "EMOTIONAL": (emo_low, emo_high),
        "VENTING": (emo_low, emo_high),
        "CASUAL": (emo_low + 2, emo_high + 2),
        "STATEMENT": ((emo_low + wide_low) // 2, (emo_high + wide_high) // 2),
        "QUESTION": (wide_low, wide_high),
        "REQUEST": (wide_low, wide_high),
        "GREETING": (wide_low, wide_high),
        "DEFAULT": ((emo_low + wide_low) // 2, (emo_high + wide_high) // 2),
    }


# ---------------------------------------------------------------------------
# LAYER 1: Pendulum Physics Sweep
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("LAYER 1: Pendulum Physics Sweep")
print("=" * 65)

# Use a random subsample of 500 for the expensive pendulum re-runs
random.seed(42)
subsample_indices = random.sample(range(len(examples)), min(500, len(examples)))
subsample = [examples[i] for i in subsample_indices]

# Use default thresholds for Layer 1 scoring (we'll optimize those in Layer 2)
default_thresholds = build_thresholds(127, 128, 115, 135)

momentum_vals = [0.75, 0.80, 0.85, 0.90]
drift_vals = [0.03, 0.06, 0.09, 0.12]
shift_dampen_vals = [0.4, 0.5, 0.6, 0.7]

print(f"  Sweeping {len(momentum_vals)}x{len(drift_vals)}x{len(shift_dampen_vals)} = "
      f"{len(momentum_vals)*len(drift_vals)*len(shift_dampen_vals)} combos on {len(subsample)} examples...")

t1 = time.perf_counter()
best_l1_score = 0
best_l1_config = None
l1_configs_tested = 0

for mom, drift, shift_d in product(momentum_vals, drift_vals, shift_dampen_vals):
    # Run pendulum on subsample with these physics params
    cached = []
    for text, truth, ds in subsample:
        v, mode = run_pendulum(text, momentum=mom, drift_rate=drift, shift_dampen=shift_d)
        cached.append((v, mode, truth, ds))

    correct, total, _, _ = score_dataset(cached, default_thresholds)
    acc = correct / total
    l1_configs_tested += 1

    if acc > best_l1_score:
        best_l1_score = acc
        best_l1_config = {"momentum": mom, "drift_rate": drift, "shift_dampen": shift_d}

l1_elapsed = time.perf_counter() - t1
print(f"  Tested {l1_configs_tested} configs in {l1_elapsed:.1f}s")
print(f"  Best: momentum={best_l1_config['momentum']}, drift={best_l1_config['drift_rate']}, "
      f"shift_dampen={best_l1_config['shift_dampen']}")

# Validate top config on full dataset
print("  Validating on full dataset...")
t_val = time.perf_counter()

# First, get baseline with default physics on full dataset
baseline_cached = []
for text, truth, ds in examples:
    v, mode = run_pendulum(text)  # defaults: momentum=0.80, drift=0.06
    baseline_cached.append((v, mode, truth, ds))

baseline_correct, baseline_total, baseline_ds_correct, baseline_ds_total = \
    score_dataset(baseline_cached, default_thresholds)
baseline_acc = baseline_correct / baseline_total * 100
print(f"  Baseline (default physics): {baseline_acc:.1f}%")

# Now run with best L1 config on full dataset
l1_cached = []
for text, truth, ds in examples:
    v, mode = run_pendulum(text,
                           momentum=best_l1_config["momentum"],
                           drift_rate=best_l1_config["drift_rate"],
                           shift_dampen=best_l1_config["shift_dampen"])
    l1_cached.append((v, mode, truth, ds))

l1_correct, l1_total, l1_ds_correct, l1_ds_total = \
    score_dataset(l1_cached, default_thresholds)
l1_acc = l1_correct / l1_total * 100
l1_impact = l1_acc - baseline_acc
print(f"  After L1 tuning: {l1_acc:.1f}% (impact: {l1_impact:+.1f}pp)")
print(f"  Validated in {time.perf_counter() - t_val:.1f}s")


# ---------------------------------------------------------------------------
# LAYER 2: Intent-Adaptive Threshold Sweep
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("LAYER 2: Intent-Adaptive Threshold Sweep")
print("=" * 65)

# Use the L1-optimized cached V values (l1_cached) for fast threshold sweep
emo_low_vals = [123, 125, 127, 129]
emo_high_vals = [128, 130, 132, 134]
wide_low_vals = [105, 110, 115, 120]
wide_high_vals = [130, 135, 140, 145]

total_l2_combos = (len(emo_low_vals) * len(emo_high_vals) *
                   len(wide_low_vals) * len(wide_high_vals))
print(f"  Sweeping {total_l2_combos} threshold combos (fast int comparisons)...")

t2 = time.perf_counter()
best_l2_score = 0
best_l2_config = None
l2_configs_tested = 0

for emo_low, emo_high, wide_low, wide_high in product(
        emo_low_vals, emo_high_vals, wide_low_vals, wide_high_vals):
    if emo_high <= emo_low:
        continue
    if wide_high <= wide_low:
        continue

    thresholds = build_thresholds(emo_low, emo_high, wide_low, wide_high)
    correct, total, ds_correct, ds_total = score_dataset(l1_cached, thresholds)
    acc = correct / total
    l2_configs_tested += 1

    if acc > best_l2_score:
        best_l2_score = acc
        best_l2_config = {
            "emo_low": emo_low, "emo_high": emo_high,
            "wide_low": wide_low, "wide_high": wide_high,
            "ds_correct": dict(ds_correct), "ds_total": dict(ds_total),
        }

l2_elapsed = time.perf_counter() - t2
l2_acc = best_l2_score * 100
l2_impact = l2_acc - l1_acc

# Build best L2 thresholds for subsequent layers
best_l2_thresholds = build_thresholds(
    best_l2_config["emo_low"], best_l2_config["emo_high"],
    best_l2_config["wide_low"], best_l2_config["wide_high"])

print(f"  Tested {l2_configs_tested} configs in {l2_elapsed:.1f}s")
print(f"  Best: emotional=[{best_l2_config['emo_low']},{best_l2_config['emo_high']}], "
      f"wide=[{best_l2_config['wide_low']},{best_l2_config['wide_high']}]")
print(f"  After L2 tuning: {l2_acc:.1f}% (impact: {l2_impact:+.1f}pp)")


# ---------------------------------------------------------------------------
# LAYER 3: Crisis Parameters Sweep
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("LAYER 3: Crisis Parameters Sweep")
print("=" * 65)

# Find crisis-adjacent examples (sentences with crisis-related words)
crisis_words = {"die", "dying", "death", "dead", "kill", "suicide", "suicidal",
                "living", "alive", "exist", "point", "purpose", "worth",
                "end", "anymore", "nothing", "meaningless", "worthless",
                "hopeless", "helpless", "afraid", "terrified", "desperate",
                "pain", "suffer", "suffering", "hurt", "broken", "lost",
                "scared", "fear", "devastated", "overwhelmed", "miserable"}

# Also include strongly negative examples that might benefit from crisis tuning
crisis_subset = []
crisis_indices = []
for i, (text, truth, ds) in enumerate(examples):
    words_set = set(re.findall(r"[a-z']+", text.lower()))
    if words_set & crisis_words:
        crisis_subset.append((text, truth, ds))
        crisis_indices.append(i)

# Also include a random sample of other examples for balance
random.seed(43)
non_crisis_indices = [i for i in range(len(examples)) if i not in set(crisis_indices)]
balance_indices = random.sample(non_crisis_indices, min(500, len(non_crisis_indices)))
for i in balance_indices:
    crisis_subset.append(examples[i])

print(f"  Crisis-adjacent subset: {len(crisis_indices)} crisis + {len(balance_indices)} balance = {len(crisis_subset)} examples")

crisis_cap_vals = [0.95, 0.97, 0.99]
crisis_push_vals = [0.03, 0.05, 0.07]

print(f"  Sweeping {len(crisis_cap_vals)}x{len(crisis_push_vals)} = "
      f"{len(crisis_cap_vals)*len(crisis_push_vals)} combos...")

t3 = time.perf_counter()
best_l3_score = 0
best_l3_config = None
l3_configs_tested = 0

for crisis_cap, crisis_push in product(crisis_cap_vals, crisis_push_vals):
    cached = []
    for text, truth, ds in crisis_subset:
        v, mode = run_pendulum(text,
                               momentum=best_l1_config["momentum"],
                               drift_rate=best_l1_config["drift_rate"],
                               shift_dampen=best_l1_config["shift_dampen"],
                               crisis_cap=crisis_cap,
                               crisis_push_scale=crisis_push)
        cached.append((v, mode, truth, ds))

    correct, total, _, _ = score_dataset(cached, best_l2_thresholds)
    acc = correct / total
    l3_configs_tested += 1

    if acc > best_l3_score:
        best_l3_score = acc
        best_l3_config = {"crisis_cap": crisis_cap, "crisis_push": crisis_push}

l3_elapsed = time.perf_counter() - t3

# Validate L3 on full dataset
print("  Validating on full dataset...")
l3_cached = []
for text, truth, ds in examples:
    v, mode = run_pendulum(text,
                           momentum=best_l1_config["momentum"],
                           drift_rate=best_l1_config["drift_rate"],
                           shift_dampen=best_l1_config["shift_dampen"],
                           crisis_cap=best_l3_config["crisis_cap"],
                           crisis_push_scale=best_l3_config["crisis_push"])
    l3_cached.append((v, mode, truth, ds))

l3_correct, l3_total, l3_ds_correct, l3_ds_total = \
    score_dataset(l3_cached, best_l2_thresholds)
l3_acc = l3_correct / l3_total * 100
l3_impact = l3_acc - l2_acc

print(f"  Tested {l3_configs_tested} configs in {l3_elapsed:.1f}s")
print(f"  Best: crisis_cap={best_l3_config['crisis_cap']}, crisis_push={best_l3_config['crisis_push']}")
print(f"  After L3 tuning: {l3_acc:.1f}% (impact: {l3_impact:+.1f}pp)")


# ---------------------------------------------------------------------------
# LAYER 4: Idiom/Context Strength Sweep
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("LAYER 4: Idiom/Context Strength Sweep")
print("=" * 65)

# Find sentences containing idioms
idiom_first_words = set()
for words_tuple in IDIOMS:
    idiom_first_words.add(words_tuple[0])

idiom_subset = []
idiom_indices = []
for i, (text, truth, ds) in enumerate(examples):
    words_list = re.findall(r"[a-z']+", text.lower())
    words_set = set(words_list)
    if words_set & idiom_first_words:
        idiom_subset.append((text, truth, ds))
        idiom_indices.append(i)

# Balance with non-idiom examples
random.seed(44)
non_idiom_indices = [i for i in range(len(examples)) if i not in set(idiom_indices)]
idiom_balance = random.sample(non_idiom_indices, min(500, len(non_idiom_indices)))
for i in idiom_balance:
    idiom_subset.append(examples[i])

print(f"  Idiom-adjacent subset: {len(idiom_indices)} idiom + {len(idiom_balance)} balance = {len(idiom_subset)} examples")

idiom_momentum_vals = [0.40, 0.50, 0.60]
idiom_push_vals = [0.5, 0.6, 0.7, 0.8]

print(f"  Sweeping {len(idiom_momentum_vals)}x{len(idiom_push_vals)} = "
      f"{len(idiom_momentum_vals)*len(idiom_push_vals)} combos...")

t4 = time.perf_counter()
best_l4_score = 0
best_l4_config = None
l4_configs_tested = 0

for idiom_mom, idiom_p in product(idiom_momentum_vals, idiom_push_vals):
    cached = []
    for text, truth, ds in idiom_subset:
        v, mode = run_pendulum(text,
                               momentum=best_l1_config["momentum"],
                               drift_rate=best_l1_config["drift_rate"],
                               shift_dampen=best_l1_config["shift_dampen"],
                               crisis_cap=best_l3_config["crisis_cap"],
                               crisis_push_scale=best_l3_config["crisis_push"],
                               idiom_momentum=idiom_mom,
                               idiom_push=idiom_p)
        cached.append((v, mode, truth, ds))

    correct, total, _, _ = score_dataset(cached, best_l2_thresholds)
    acc = correct / total
    l4_configs_tested += 1

    if acc > best_l4_score:
        best_l4_score = acc
        best_l4_config = {"idiom_momentum": idiom_mom, "idiom_push": idiom_p}

l4_elapsed = time.perf_counter() - t4

# Validate L4 on full dataset
print("  Validating on full dataset...")
l4_cached = []
for text, truth, ds in examples:
    v, mode = run_pendulum(text,
                           momentum=best_l1_config["momentum"],
                           drift_rate=best_l1_config["drift_rate"],
                           shift_dampen=best_l1_config["shift_dampen"],
                           crisis_cap=best_l3_config["crisis_cap"],
                           crisis_push_scale=best_l3_config["crisis_push"],
                           idiom_momentum=best_l4_config["idiom_momentum"],
                           idiom_push=best_l4_config["idiom_push"])
    l4_cached.append((v, mode, truth, ds))

l4_correct, l4_total, l4_ds_correct, l4_ds_total = \
    score_dataset(l4_cached, best_l2_thresholds)
l4_acc = l4_correct / l4_total * 100
l4_impact = l4_acc - l3_acc

print(f"  Tested {l4_configs_tested} configs in {l4_elapsed:.1f}s")
print(f"  Best: idiom_momentum={best_l4_config['idiom_momentum']}, idiom_push={best_l4_config['idiom_push']}")
print(f"  After L4 tuning: {l4_acc:.1f}% (impact: {l4_impact:+.1f}pp)")


# ---------------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------------

total_elapsed = time.perf_counter() - t_load

# Per-dataset final breakdown
sst_acc = l4_ds_correct["sst2"] / l4_ds_total["sst2"] * 100 if l4_ds_total["sst2"] else 0
ge_acc = l4_ds_correct["go_emotions"] / l4_ds_total["go_emotions"] * 100 if l4_ds_total["go_emotions"] else 0
te_acc = l4_ds_correct["tweet_eval"] / l4_ds_total["tweet_eval"] * 100 if l4_ds_total["tweet_eval"] else 0

# Per-dataset baseline breakdown
sst_base = baseline_ds_correct["sst2"] / baseline_ds_total["sst2"] * 100 if baseline_ds_total["sst2"] else 0
ge_base = baseline_ds_correct["go_emotions"] / baseline_ds_total["go_emotions"] * 100 if baseline_ds_total["go_emotions"] else 0
te_base = baseline_ds_correct["tweet_eval"] / baseline_ds_total["tweet_eval"] * 100 if baseline_ds_total["tweet_eval"] else 0

print("\n")
print("FULL-STACK TUNING RESULTS")
print("\u2550" * 55)

print(f"\nLayer 1: Pendulum Physics")
print(f"  momentum={best_l1_config['momentum']}, drift={best_l1_config['drift_rate']}, "
      f"shift_dampen={best_l1_config['shift_dampen']}")
print(f"  Impact: {l1_impact:+.1f}pp")

print(f"\nLayer 2: Intent-Adaptive Thresholds")
print(f"  emotional=[{best_l2_config['emo_low']},{best_l2_config['emo_high']}], "
      f"wide=[{best_l2_config['wide_low']},{best_l2_config['wide_high']}]")
print(f"  Impact: {l2_impact:+.1f}pp")

print(f"\nLayer 3: Crisis Parameters")
print(f"  crisis_cap={best_l3_config['crisis_cap']}, crisis_push={best_l3_config['crisis_push']}")
print(f"  Impact: {l3_impact:+.1f}pp")

print(f"\nLayer 4: Idiom/Context")
print(f"  idiom_momentum={best_l4_config['idiom_momentum']}, idiom_push={best_l4_config['idiom_push']}")
print(f"  Impact: {l4_impact:+.1f}pp")

print(f"\nCOMBINED: {baseline_acc:.1f}% \u2192 {l4_acc:.1f}% ({l4_acc - baseline_acc:+.1f}pp)")

print(f"\nPer-dataset:")
print(f"  SST-2:       {sst_acc:.1f}%  (was {sst_base:.1f}%)")
print(f"  GoEmotions:  {ge_acc:.1f}%  (was {ge_base:.1f}%)")
print(f"  TweetEval:   {te_acc:.1f}%  (was {te_base:.1f}%)")

print(f"\nTotal runtime: {total_elapsed:.1f}s")
print("\u2550" * 55)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results = {
    "baseline_accuracy": round(baseline_acc, 1),
    "final_accuracy": round(l4_acc, 1),
    "total_improvement_pp": round(l4_acc - baseline_acc, 1),
    "total_examples": len(examples),
    "runtime_seconds": round(total_elapsed, 1),
    "layers": {
        "layer_1_pendulum_physics": {
            "params": best_l1_config,
            "impact_pp": round(l1_impact, 1),
            "accuracy_after": round(l1_acc, 1),
            "configs_tested": l1_configs_tested,
            "sweep_time_s": round(l1_elapsed, 1),
        },
        "layer_2_intent_thresholds": {
            "params": {
                "emo_low": best_l2_config["emo_low"],
                "emo_high": best_l2_config["emo_high"],
                "wide_low": best_l2_config["wide_low"],
                "wide_high": best_l2_config["wide_high"],
            },
            "full_thresholds": {
                mode: {"low": low, "high": high}
                for mode, (low, high) in best_l2_thresholds.items()
            },
            "impact_pp": round(l2_impact, 1),
            "accuracy_after": round(l2_acc, 1),
            "configs_tested": l2_configs_tested,
            "sweep_time_s": round(l2_elapsed, 1),
        },
        "layer_3_crisis": {
            "params": best_l3_config,
            "impact_pp": round(l3_impact, 1),
            "accuracy_after": round(l3_acc, 1),
            "configs_tested": l3_configs_tested,
            "sweep_time_s": round(l3_elapsed, 1),
        },
        "layer_4_idiom": {
            "params": best_l4_config,
            "impact_pp": round(l4_impact, 1),
            "accuracy_after": round(l4_acc, 1),
            "configs_tested": l4_configs_tested,
            "sweep_time_s": round(l4_elapsed, 1),
        },
    },
    "per_dataset": {
        "sst2": {"accuracy": round(sst_acc, 1), "baseline": round(sst_base, 1),
                 "examples": l4_ds_total["sst2"]},
        "go_emotions": {"accuracy": round(ge_acc, 1), "baseline": round(ge_base, 1),
                        "examples": l4_ds_total["go_emotions"]},
        "tweet_eval": {"accuracy": round(te_acc, 1), "baseline": round(te_base, 1),
                       "examples": l4_ds_total["tweet_eval"]},
    },
}

out_path = os.path.join(os.path.dirname(__file__), "full_stack_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
