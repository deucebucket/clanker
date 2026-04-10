"""V9 Baseline Benchmark — measure V9 accuracy on verified sentences.

Records the starting point. Every future V9 change must improve or maintain this.

Usage:
  python3 benchmarks/v9_baseline.py
  python3 benchmarks/v9_baseline.py --verbose
  python3 benchmarks/v9_baseline.py --category grief
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v9.pipeline import compute_vadug


def load_verified():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "datasets", "verified_sentences.json")
    with open(path) as f:
        data = json.load(f)
    return data


def classify(v):
    if v >= 145:
        return "pos"
    elif v < 110:
        return "neg"
    return "neutral"


def run_baseline(verbose=False, category=None):
    data = load_verified()
    sentences = data["sentences"]

    if category:
        sentences = [s for s in sentences if s.get("source", "") == category]

    print(f"V9 Baseline Benchmark")
    print(f"Engine: V9 Equation Decomposition")
    print(f"Sentences: {len(sentences)}")
    print("=" * 70)

    correct = 0
    wrong = []
    by_label = {"pos": [0, 0], "neg": [0, 0], "neutral": [0, 0]}

    for s in sentences:
        text = s["text"]
        human = s["human_label"]
        result, trace = compute_vadug(text)
        v9_label = classify(result.v)

        is_correct = (v9_label == human)
        if is_correct:
            correct += 1
        else:
            wrong.append({
                "text": text,
                "human": human,
                "v9_label": v9_label,
                "v9_v": result.v,
                "nucleus": trace["equation"]["nucleus"],
                "nucleus_root": trace["equation"]["nucleus_root"],
            })

        by_label[human][0] += 1
        if is_correct:
            by_label[human][1] += 1

    accuracy = correct / len(sentences) * 100 if sentences else 0

    print(f"\nOVERALL: {correct}/{len(sentences)} ({accuracy:.1f}%)")
    print(f"\nBy label:")
    for label, (total, ok) in by_label.items():
        pct = ok / total * 100 if total else 0
        print(f"  {label:>8}: {ok}/{total} ({pct:.1f}%)")

    if verbose or len(wrong) <= 20:
        print(f"\nMISSES ({len(wrong)}):")
        for w in wrong:
            print(f"  \"{w['text'][:60]}\"")
            print(f"    Human={w['human']} V9={w['v9_label']} (V={w['v9_v']}) "
                  f"Nucleus={w['nucleus']} ({w['nucleus_root']})")

    print(f"\n{'=' * 70}")
    print(f"V9 BASELINE: {accuracy:.1f}%")
    print(f"V8 BASELINE: see benchmarks/full_barrage.py for comparison")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()
    run_baseline(verbose=args.verbose, category=args.category)
