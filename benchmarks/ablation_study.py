#!/usr/bin/env python3
"""Ablation Study — Kill-switch test for every tunable force in PendulumV2.

Loads the 27-param champion config, runs the academic benchmark (SST-2,
GoEmotions, TweetEval) as baseline, then systematically disables ONE force
at a time and re-runs. Reports delta vs baseline to identify bloat.

Forces with delta <= 0 are BLOAT (not helping the composite score).

Usage:
    python3 benchmarks/ablation_study.py
    python3 benchmarks/ablation_study.py --workers 4   # fewer workers
"""

import sys, os, json, time, argparse
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Load champion config
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gpu_optimal_v2.json")

with open(CONFIG_PATH) as f:
    _raw = json.load(f)
CHAMPION_CONFIG = _raw["champion_config"]

# ---------------------------------------------------------------------------
# Kill-switch overrides: param_name -> disabled_value
# Setting a multiplier to 1.0 = identity (no effect).
# Setting an offset to 0 = no shift.
# Negation has 3 params — kill all three together.
# ---------------------------------------------------------------------------

KILL_SWITCHES = {
    "negation": {
        "negation_decay_operator": 1.0,
        "negation_decay_neutral": 1.0,
        "negation_decay_payload": 1.0,
    },
    "evoker_decay": {"evoker_decay": 1.0},
    "conditional_dampener": {"conditional_dampener": 1.0},
    "clinical_dampener": {"clinical_dampener": 1.0},
    "passive_d_offset": {"passive_d_offset": 0.0},
    "comparative_amplifier": {"comparative_amplifier": 1.0},
    "superlative_amplifier": {"superlative_amplifier": 1.0},
    "filler_d_offset": {"filler_d_offset": 0.0},
    "litotes_dampener": {"litotes_dampener": 1.0},
    "weaponized_d_boost": {"weaponized_d_boost": 0.0},
    "tag_d_offset": {"tag_d_offset": 0.0},
    "question_dampener": {"question_dampener": 1.0},
}

# ---------------------------------------------------------------------------
# GoEmotions label map (same as academic_benchmark.py)
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
# Dataset loading (done once in main process, passed as lists to workers)
# ---------------------------------------------------------------------------

def load_all_datasets():
    """Load and flatten datasets into simple (text, truth, mode) lists."""
    examples = []

    # SST-2 validation (872 examples)
    print("  Loading SST-2...")
    sst = load_dataset("stanfordnlp/sst2", split="validation")
    for row in sst:
        truth = "positive" if row["label"] == 1 else "negative"
        examples.append((row["sentence"], truth, "binary"))

    # GoEmotions test (first 1000)
    print("  Loading GoEmotions...")
    ge = load_dataset("google-research-datasets/go_emotions", split="test")
    ge = ge.select(range(min(1000, len(ge))))
    for row in ge:
        if not row["labels"]:
            continue
        _, sent = GOEMO.get(row["labels"][0], ("unk", "neutral"))
        examples.append((row["text"], sent, "three_way"))

    # TweetEval emotion test (first 1000)
    print("  Loading TweetEval...")
    te = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test")
    te = te.select(range(min(1000, len(te))))
    for row in te:
        truth = TW_SENT[row["label"]]
        examples.append((row["text"], truth, "binary"))

    return examples


# ---------------------------------------------------------------------------
# Scoring worker — runs one config against all examples
# ---------------------------------------------------------------------------

def score_config(config_dict, examples):
    """Score a single config against all examples. Returns per-dataset accuracy."""
    from demo.pendulum_v2 import PendulumV2

    engine = PendulumV2(**config_dict)

    # Track per-dataset results.
    # Examples are ordered: SST-2 (binary) -> GoEmotions (three_way) -> TweetEval (binary).
    # We use a simple state machine based on mode transitions.
    counts = {
        "sst2_correct": 0, "sst2_total": 0,
        "goemo_correct": 0, "goemo_total": 0,
        "tweet_correct": 0, "tweet_total": 0,
    }

    sst_phase_done = False

    for text, truth, mode in examples:
        vadug, _ = engine.process_text(text)
        label = engine.classify(vadug.v, mode=mode)

        if mode == "binary" and not sst_phase_done:
            counts["sst2_total"] += 1
            if label == truth:
                counts["sst2_correct"] += 1
        elif mode == "three_way":
            sst_phase_done = True
            counts["goemo_total"] += 1
            if label == truth:
                counts["goemo_correct"] += 1
        elif mode == "binary" and sst_phase_done:
            counts["tweet_total"] += 1
            if label == truth:
                counts["tweet_correct"] += 1

    sst_acc = counts["sst2_correct"] / counts["sst2_total"] * 100 if counts["sst2_total"] else 0
    ge_acc = counts["goemo_correct"] / counts["goemo_total"] * 100 if counts["goemo_total"] else 0
    tw_acc = counts["tweet_correct"] / counts["tweet_total"] * 100 if counts["tweet_total"] else 0
    composite = (sst_acc + ge_acc + tw_acc) / 3.0

    return {
        "sst2": round(sst_acc, 2),
        "goemotions": round(ge_acc, 2),
        "tweeteval": round(tw_acc, 2),
        "composite": round(composite, 2),
    }


def _worker(args):
    """Multiprocessing wrapper — unpacks (name, config, examples)."""
    name, config, examples = args
    result = score_config(config, examples)
    return name, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation study -- kill-switch test for PendulumV2 forces")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    workers = min(args.workers, cpu_count())

    print(f"\n{'#' * 80}")
    print(f"  ABLATION STUDY -- Kill-Switch Test for PendulumV2 Forces")
    print(f"  Champion config: {CONFIG_PATH}")
    print(f"  Workers: {workers}")
    print(f"{'#' * 80}\n")

    # Load datasets once
    examples = load_all_datasets()
    print(f"  Total examples: {len(examples)}\n")

    # Build all jobs: (name, config_dict, examples)
    jobs = []

    # Baseline (full champion config)
    jobs.append(("BASELINE", dict(CHAMPION_CONFIG), examples))

    # One job per kill switch
    for force_name, overrides in KILL_SWITCHES.items():
        config = dict(CHAMPION_CONFIG)
        config.update(overrides)
        jobs.append((force_name, config, examples))

    # Run all in parallel
    t0 = time.perf_counter()
    print(f"  Running {len(jobs)} configurations ({workers} workers)...")

    with Pool(workers) as pool:
        results = pool.map(_worker, jobs)

    elapsed = time.perf_counter() - t0

    # Collect results
    result_map = {}
    for name, scores in results:
        result_map[name] = scores

    baseline = result_map["BASELINE"]

    # Build ablation table sorted by delta (worst drop first = most important)
    ablations = []
    for force_name in KILL_SWITCHES:
        scores = result_map[force_name]
        delta = round(scores["composite"] - baseline["composite"], 2)
        ablations.append({
            "force": force_name,
            "composite": scores["composite"],
            "sst2": scores["sst2"],
            "goemotions": scores["goemotions"],
            "tweeteval": scores["tweeteval"],
            "delta": delta,
        })

    ablations.sort(key=lambda x: x["delta"])

    # Print results
    print(f"\n{'#' * 80}")
    print(f"  BASELINE: {baseline['composite']:.2f}  "
          f"(SST-2: {baseline['sst2']:.2f}  "
          f"GoEmo: {baseline['goemotions']:.2f}  "
          f"Tweet: {baseline['tweeteval']:.2f})")
    print(f"{'#' * 80}\n")

    hdr = f"  {'Force':<28} {'Composite':>10} {'Delta':>8} {'SST-2':>8} {'GoEmo':>8} {'Tweet':>8}  {'Verdict'}"
    print(hdr)
    print(f"  {'-' * 90}")

    bloat_count = 0
    for a in ablations:
        is_bloat = a["delta"] >= 0
        if is_bloat:
            bloat_count += 1
        verdict = "BLOAT" if is_bloat else ""
        sign = "+" if a["delta"] >= 0 else ""
        print(f"  {a['force']:<28} {a['composite']:>10.2f} {sign}{a['delta']:>7.2f} "
              f"{a['sst2']:>8.2f} {a['goemotions']:>8.2f} {a['tweeteval']:>8.2f}  {verdict}")

    print(f"  {'-' * 90}")
    print(f"\n  {bloat_count} / {len(ablations)} forces are BLOAT (delta >= 0 when disabled)")
    print(f"  Time: {elapsed:.1f}s\n")

    # Save JSON results
    output = {
        "baseline": baseline,
        "ablations": ablations,
        "bloat_count": bloat_count,
        "total_forces": len(ablations),
        "time_seconds": round(elapsed, 1),
        "champion_config": CHAMPION_CONFIG,
    }
    out_path = os.path.join(os.path.dirname(__file__), "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_path}\n")


if __name__ == "__main__":
    main()
