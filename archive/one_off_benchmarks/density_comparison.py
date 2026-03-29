#!/usr/bin/env python3
"""Benchmark: Raw V vs Density V comparison on academic datasets.

Runs the same sentences through the Clanker pipeline twice:
1. Raw V — accumulated forces as-is
2. Density V — length-normalized via compute_density()

Compares accuracy and per-length-bucket performance to see if density
reduces the gap between short and long sentence accuracy.
"""

import time, re, sys, os, json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

from demo.simulator import SequentialPendulum, VADUG, WORD_FORCES
from demo.classifier import AdaptiveClassifier
from demo.intent import IntentDetector
from demo.multipath import MultiPathProcessor

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets"); sys.exit(1)

try:
    from sklearn.metrics import f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

_classifier = AdaptiveClassifier()
_intent_detector = IntentDetector()


GOEMO = {0:("admiration","positive"),1:("amusement","positive"),2:("anger","negative"),
         3:("annoyance","negative"),4:("approval","positive"),5:("caring","positive"),
         6:("confusion","neutral"),7:("curiosity","neutral"),8:("desire","positive"),
         9:("disappointment","negative"),10:("disapproval","negative"),11:("disgust","negative"),
         12:("embarrassment","negative"),13:("excitement","positive"),14:("fear","negative"),
         15:("gratitude","positive"),16:("grief","negative"),17:("joy","positive"),
         18:("love","positive"),19:("nervousness","negative"),20:("optimism","positive"),
         21:("pride","positive"),22:("realization","neutral"),23:("relief","positive"),
         24:("remorse","negative"),25:("sadness","negative"),26:("surprise","neutral"),
         27:("neutral","neutral")}


def run_both(text):
    """Run a sentence through the pipeline, return raw and density predictions."""
    intent_mode, _ = _intent_detector.detect(text)

    # Use multi-path processing (same as academic_benchmark.py)
    mp = MultiPathProcessor()
    v, a, d, u, g, path, consistency, history = mp.process(text)

    # Raw classification
    if history and _classifier._entropy.should_override_neutral(history, v, intent_mode):
        raw_label = "neutral"
    else:
        raw_label = _classifier.classify(v, intent_mode)

    # Density classification: re-run a fresh pendulum to get compute_density()
    words = re.findall(r"[a-z']+", text.lower())
    p = SequentialPendulum()
    for i, w in enumerate(words):
        p.process_word(w, words, i)
    density = p.compute_density()
    density_v = density['v']
    density_label = _classifier.classify_density(density_v, intent_mode, p.history)

    n_words = len(words)
    return raw_label, density_label, v, density_v, n_words


def bench(name, ds, get_text, get_truth, max_n=None):
    """Run benchmark on a dataset, comparing raw vs density."""
    if max_n:
        ds = ds.select(range(min(max_n, len(ds))))

    raw_correct = 0
    density_correct = 0
    total = 0

    # Per-length-bucket stats
    buckets = {
        "1-5": {"raw_c": 0, "den_c": 0, "n": 0},
        "6-10": {"raw_c": 0, "den_c": 0, "n": 0},
        "11-20": {"raw_c": 0, "den_c": 0, "n": 0},
        "21+": {"raw_c": 0, "den_c": 0, "n": 0},
    }

    raw_preds, den_preds, truths = [], [], []

    for i, row in enumerate(ds):
        text = get_text(row)
        truth = get_truth(row)
        if truth is None:
            continue

        raw_label, density_label, raw_v, density_v, n_words = run_both(text)

        if raw_label == truth:
            raw_correct += 1
        if density_label == truth:
            density_correct += 1
        total += 1

        raw_preds.append(raw_label)
        den_preds.append(density_label)
        truths.append(truth)

        # Bucket
        if n_words <= 5:
            b = "1-5"
        elif n_words <= 10:
            b = "6-10"
        elif n_words <= 20:
            b = "11-20"
        else:
            b = "21+"
        buckets[b]["n"] += 1
        if raw_label == truth:
            buckets[b]["raw_c"] += 1
        if density_label == truth:
            buckets[b]["den_c"] += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(ds)}...", end="\r")

    raw_acc = raw_correct / total * 100 if total else 0
    den_acc = density_correct / total * 100 if total else 0

    raw_f1 = f1_score(truths, raw_preds, average="macro", zero_division=0,
                      labels=["positive", "negative", "neutral"]) * 100 if HAS_SKLEARN else 0
    den_f1 = f1_score(truths, den_preds, average="macro", zero_division=0,
                      labels=["positive", "negative", "neutral"]) * 100 if HAS_SKLEARN else 0

    print(f"\n  {name} ({total} examples)")
    print(f"  {'Method':<12} {'Acc':>8} {'F1':>8}")
    print(f"  {'-'*30}")
    print(f"  {'Raw V':<12} {raw_acc:>7.1f}% {raw_f1:>7.1f}%")
    print(f"  {'Density V':<12} {den_acc:>7.1f}% {den_f1:>7.1f}%")
    diff = den_acc - raw_acc
    print(f"  Delta: {diff:+.1f}pp")

    print(f"\n  Per-length breakdown:")
    print(f"  {'Bucket':<10} {'N':>6} {'Raw Acc':>10} {'Den Acc':>10} {'Delta':>8}")
    print(f"  {'-'*48}")
    for b_name, b_data in buckets.items():
        if b_data["n"] == 0:
            continue
        r = b_data["raw_c"] / b_data["n"] * 100
        d = b_data["den_c"] / b_data["n"] * 100
        print(f"  {b_name:<10} {b_data['n']:>6} {r:>9.1f}% {d:>9.1f}% {d-r:>+7.1f}")

    return {
        "dataset": name, "total": total,
        "raw_acc": round(raw_acc, 1), "density_acc": round(den_acc, 1),
        "raw_f1": round(raw_f1, 1), "density_f1": round(den_f1, 1),
        "delta": round(diff, 1),
    }


def main():
    print(f"\n{'#'*70}")
    print(f"  DENSITY COMPARISON — Raw V vs Density V")
    print(f"  {len(WORD_FORCES)} word forces")
    print(f"{'#'*70}")

    all_results = []

    # SST-2
    sst = load_dataset("stanfordnlp/sst2", split="validation")
    all_results.append(bench("SST-2", sst,
                             lambda r: r["sentence"],
                             lambda r: "positive" if r["label"] == 1 else "negative",
                             max_n=1000))

    # GoEmotions
    ge = load_dataset("google-research-datasets/go_emotions", split="test")
    def ge_truth(r):
        if not r["labels"]:
            return None
        _, sent = GOEMO.get(r["labels"][0], ("unk", "neutral"))
        return sent
    all_results.append(bench("GoEmotions", ge, lambda r: r["text"], ge_truth, max_n=1000))

    # TweetEval
    te = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test")
    tw_sent = {0: "negative", 1: "positive", 2: "positive", 3: "negative"}
    all_results.append(bench("TweetEval", te,
                             lambda r: r["text"],
                             lambda r: tw_sent[r["label"]],
                             max_n=1000))

    # Summary
    print(f"\n{'#'*70}")
    print(f"  {'Dataset':<20} {'Raw Acc':>10} {'Density Acc':>12} {'Delta':>8}")
    print(f"  {'-'*54}")
    for r in all_results:
        print(f"  {r['dataset']:<20} {r['raw_acc']:>9.1f}% {r['density_acc']:>11.1f}% {r['delta']:>+7.1f}")
    print(f"{'#'*70}\n")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "density_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
