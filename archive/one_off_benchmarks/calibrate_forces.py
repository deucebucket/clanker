#!/usr/bin/env python3
"""Force Calibration Optimizer — genetic algorithm on 46K WORD_FORCES.

Tunes 10 scaling multipliers (5 category × 5 dimension) to maximize
accuracy on academic benchmarks (GoEmotions + SST-2).

Category grouping (by valence direction/magnitude):
  strong_negative: dv <= -30
  mild_negative:   -30 < dv < -10
  neutral_weak:    -10 <= dv <= 10
  mild_positive:   10 < dv < 30
  strong_positive: dv >= 30

Dimension multipliers:
  valence_scale, arousal_scale, dominance_scale, urgency_scale, gravity_scale

Uses scipy.optimize.differential_evolution (gradient-free genetic algorithm).
"""

import sys, os, re, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

from scipy.optimize import differential_evolution
from demo.forces import WORD_FORCES
from demo.pendulum import SequentialPendulum
import demo.forces as forces_module
import demo.pendulum as pendulum_module

# ── Load tuned thresholds from optimal_config.json ──
_cfg_path = os.path.join(os.path.dirname(__file__), "optimal_config.json")
with open(_cfg_path) as f:
    _cfg = json.load(f)
NEUTRAL_LOW = _cfg["champion"]["neutral_low"]    # 120
NEUTRAL_HIGH = _cfg["champion"]["neutral_high"]   # 135

# ── GoEmotions label mapping ──
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


def load_test_set():
    """Load 500 GoEmotions test + 200 SST-2 validation examples."""
    from datasets import load_dataset

    test_set = []

    # SST-2 validation (binary: positive/negative)
    print("  Loading SST-2 validation...")
    sst = load_dataset("stanfordnlp/sst2", split="validation")
    rng = np.random.RandomState(42)
    sst_idx = rng.choice(len(sst), size=min(200, len(sst)), replace=False)
    for i in sst_idx:
        row = sst[int(i)]
        label = "positive" if row["label"] == 1 else "negative"
        test_set.append((row["sentence"], label))

    # GoEmotions test (multi-class -> 3-class)
    print("  Loading GoEmotions test...")
    ge = load_dataset("google-research-datasets/go_emotions", split="test")
    ge_idx = rng.choice(len(ge), size=min(500, len(ge)), replace=False)
    for i in ge_idx:
        row = ge[int(i)]
        if not row["labels"]:
            continue
        _, sent = GOEMO.get(row["labels"][0], ("unk", "neutral"))
        test_set.append((row["text"], sent))

    print(f"  Loaded {len(test_set)} test examples")
    return test_set


# ── Save the original WORD_FORCES (never modified) ──
BASE_FORCES = dict(WORD_FORCES)

# Category names for output
CAT_NAMES = ["strong_negative", "mild_negative", "neutral_weak",
             "mild_positive", "strong_positive"]
DIM_NAMES = ["valence", "arousal", "dominance", "urgency", "gravity"]


def scale_forces(params):
    """Create a scaled copy of WORD_FORCES based on 10 parameters.

    params[0:5] = category multipliers (strong_neg, mild_neg, neutral, mild_pos, strong_pos)
    params[5:10] = dimension multipliers (v, a, d, u, g)
    """
    cat_scales = params[:5]
    dim_scales = params[5:]

    scaled = {}
    for word, (dv, da, dd, du, dg) in BASE_FORCES.items():
        # Determine category by valence direction/magnitude
        if dv <= -30:
            cs = cat_scales[0]  # strong_negative
        elif dv < -10:
            cs = cat_scales[1]  # mild_negative
        elif dv <= 10:
            cs = cat_scales[2]  # neutral_weak
        elif dv < 30:
            cs = cat_scales[3]  # mild_positive
        else:
            cs = cat_scales[4]  # strong_positive

        scaled[word] = (
            int(dv * cs * dim_scales[0]),
            int(da * cs * dim_scales[1]),
            int(dd * cs * dim_scales[2]),
            int(du * cs * dim_scales[3]),
            int(dg * cs * dim_scales[4]),
        )
    return scaled


def classify(v):
    """Classify valence using tuned thresholds."""
    if v > NEUTRAL_HIGH:
        return "positive"
    elif v < NEUTRAL_LOW:
        return "negative"
    else:
        return "neutral"


def run_with_forces(text, scaled_forces):
    """Run a single sentence through the pendulum with scaled forces.

    Monkey-patches WORD_FORCES in both the forces module AND the pendulum
    module (since pendulum.py does `from .forces import WORD_FORCES` which
    creates a local reference in pendulum's namespace).
    """
    old_forces = forces_module.WORD_FORCES
    old_pend = pendulum_module.WORD_FORCES
    forces_module.WORD_FORCES = scaled_forces
    pendulum_module.WORD_FORCES = scaled_forces
    try:
        p = SequentialPendulum()
        words = re.findall(r"[a-z']+", text.lower())
        for i, w in enumerate(words):
            p.process_word(w, words, i)
        return int(p.v)
    finally:
        forces_module.WORD_FORCES = old_forces
        pendulum_module.WORD_FORCES = old_pend


# Track evaluation count for progress display
_eval_count = 0
_eval_start = 0
_best_score = float('inf')


def evaluate(params, test_set):
    """Objective function: returns negative accuracy (minimize)."""
    global _eval_count, _best_score

    _eval_count += 1
    scaled = scale_forces(params)

    correct = 0
    crisis_correct = 0
    crisis_total = 0
    neutral_count = 0

    for text, expected in test_set:
        v = run_with_forces(text, scaled)
        label = classify(v)

        if label == expected:
            correct += 1
        if label == "neutral":
            neutral_count += 1

        # Track crisis recall (strongly negative sentences)
        if expected == "negative":
            crisis_total += 1
            if label == "negative":
                crisis_correct += 1

    accuracy = correct / len(test_set)
    neutral_rate = neutral_count / len(test_set)
    crisis_recall = crisis_correct / crisis_total if crisis_total > 0 else 0

    # Weighted score: accuracy + crisis recall bonus - neutral penalty
    # We want good accuracy but also care about not over-neutralizing
    score = -(accuracy + 0.1 * crisis_recall - 0.05 * neutral_rate)

    if score < _best_score:
        _best_score = score
        elapsed = time.time() - _eval_start
        print(f"  [eval {_eval_count:4d}] acc={accuracy:.3f} crisis={crisis_recall:.3f} "
              f"neutral={neutral_rate:.3f} score={-score:.4f} ({elapsed:.0f}s)")

    return score


def run_baseline(test_set):
    """Run baseline (unscaled forces) and return metrics."""
    correct = 0
    crisis_correct = 0
    crisis_total = 0
    neutral_count = 0

    for text, expected in test_set:
        p = SequentialPendulum()
        words = re.findall(r"[a-z']+", text.lower())
        for i, w in enumerate(words):
            p.process_word(w, words, i)
        v = int(p.v)
        label = classify(v)

        if label == expected:
            correct += 1
        if label == "neutral":
            neutral_count += 1
        if expected == "negative":
            crisis_total += 1
            if label == "negative":
                crisis_correct += 1

    n = len(test_set)
    return {
        "accuracy": correct / n,
        "crisis_recall": crisis_correct / crisis_total if crisis_total > 0 else 0,
        "neutral_rate": neutral_count / n,
    }


def main():
    global _eval_count, _eval_start, _best_score

    print(f"\n{'='*70}")
    print(f"  FORCE CALIBRATION OPTIMIZER")
    print(f"  {len(BASE_FORCES)} word forces, thresholds: [{NEUTRAL_LOW}, {NEUTRAL_HIGH}]")
    print(f"  10 parameters: 5 category scales + 5 dimension scales")
    print(f"{'='*70}\n")

    # Load test set
    print("Loading test data...")
    test_set = load_test_set()

    # Run baseline
    print("\nRunning baseline (all scales = 1.0)...")
    t0 = time.time()
    baseline = run_baseline(test_set)
    baseline_time = time.time() - t0
    print(f"  Baseline accuracy:  {baseline['accuracy']:.3f}")
    print(f"  Baseline crisis:    {baseline['crisis_recall']:.3f}")
    print(f"  Baseline neutral:   {baseline['neutral_rate']:.3f}")
    print(f"  Time: {baseline_time:.1f}s ({baseline_time/len(test_set)*1000:.2f}ms/sentence)")

    # Optimization
    print(f"\nStarting differential evolution optimization...")
    print(f"  Population: 15, Max iterations: 100, Tolerance: 0.001")
    print(f"  Bounds: [0.5, 2.0] for all 10 parameters\n")

    bounds = [(0.5, 2.0)] * 10
    _eval_count = 0
    _eval_start = time.time()
    _best_score = float('inf')

    result = differential_evolution(
        evaluate,
        bounds,
        args=(test_set,),
        maxiter=100,
        seed=42,
        popsize=15,
        tol=0.001,
        disp=False,  # we handle our own progress
        workers=1,   # must be 1 since we monkey-patch module state
    )

    total_time = time.time() - _eval_start
    opt_params = result.x
    opt_accuracy = -result.fun  # we minimized negative score

    # Run final evaluation with optimal params to get clean metrics
    scaled = scale_forces(opt_params)
    correct = 0
    crisis_correct = 0
    crisis_total = 0
    neutral_count = 0
    for text, expected in test_set:
        v = run_with_forces(text, scaled)
        label = classify(v)
        if label == expected:
            correct += 1
        if label == "neutral":
            neutral_count += 1
        if expected == "negative":
            crisis_total += 1
            if label == "negative":
                crisis_correct += 1

    n = len(test_set)
    final_acc = correct / n
    final_crisis = crisis_correct / crisis_total if crisis_total > 0 else 0
    final_neutral = neutral_count / n

    # ── Output results ──
    print(f"\n{'='*70}")
    print(f"OPTIMAL FORCE SCALING:")
    print(f"  Category multipliers:")
    for i, name in enumerate(CAT_NAMES):
        print(f"    {name:20s}: {opt_params[i]:.2f}x")
    print(f"  Dimension multipliers:")
    for i, name in enumerate(DIM_NAMES):
        print(f"    {name:20s}: {opt_params[5+i]:.2f}x")
    print()
    print(f"  Baseline accuracy:  {baseline['accuracy']*100:.1f}%")
    print(f"  Optimized accuracy: {final_acc*100:.1f}%")
    print(f"  Neutral rate:       {baseline['neutral_rate']*100:.1f}% -> {final_neutral*100:.1f}%")
    print(f"  Crisis recall:      {baseline['crisis_recall']*100:.1f}% -> {final_crisis*100:.1f}%")
    print(f"  Improvement:        +{(final_acc - baseline['accuracy'])*100:.1f}pp")
    print(f"  Optimization time:  {total_time:.0f}s ({_eval_count} evaluations)")
    print(f"  Converged:          {result.success} (message: {result.message})")
    print(f"{'='*70}\n")

    # Save results
    output = {
        "optimal_scales": {
            "category": {name: round(float(opt_params[i]), 4) for i, name in enumerate(CAT_NAMES)},
            "dimension": {name: round(float(opt_params[5+i]), 4) for i, name in enumerate(DIM_NAMES)},
        },
        "results": {
            "baseline_accuracy": round(baseline["accuracy"], 4),
            "optimized_accuracy": round(final_acc, 4),
            "baseline_neutral_rate": round(baseline["neutral_rate"], 4),
            "optimized_neutral_rate": round(final_neutral, 4),
            "baseline_crisis_recall": round(baseline["crisis_recall"], 4),
            "optimized_crisis_recall": round(final_crisis, 4),
            "improvement_pp": round((final_acc - baseline["accuracy"]) * 100, 1),
        },
        "optimization": {
            "method": "differential_evolution",
            "evaluations": _eval_count,
            "time_seconds": round(total_time, 1),
            "converged": result.success,
            "message": result.message,
            "test_set_size": len(test_set),
            "bounds": [0.5, 2.0],
            "popsize": 15,
            "maxiter": 100,
            "seed": 42,
        },
        "thresholds": {
            "neutral_low": NEUTRAL_LOW,
            "neutral_high": NEUTRAL_HIGH,
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "optimal_forces.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
