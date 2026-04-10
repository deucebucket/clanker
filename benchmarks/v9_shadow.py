# benchmarks/v9_shadow.py
"""V9 Shadow Pipeline — run V8 and V9 side-by-side, log divergences.

Usage:
  python3 benchmarks/v9_shadow.py
  python3 benchmarks/v9_shadow.py --verbose
  python3 benchmarks/v9_shadow.py --threshold 20
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.pendulum import compute_vadug as v8_compute
from engine_v9.pipeline import compute_vadug as v9_compute


def load_verified_sentences():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "datasets", "verified_sentences.json")
    with open(path) as f:
        data = json.load(f)
    return data["sentences"]


def classify(v):
    if v >= 145:
        return "pos"
    elif v < 110:
        return "neg"
    return "neutral"


def run_shadow(threshold=15, verbose=False):
    sentences = load_verified_sentences()
    print(f"V9 Shadow Pipeline — {len(sentences)} verified sentences")
    print(f"Divergence threshold: {threshold} points")
    print("=" * 80)

    divergences = []
    v8_correct = 0
    v9_correct = 0
    agree = 0

    for s in sentences:
        text = s["text"]
        human = s["human_label"]

        v8_result, _ = v8_compute(text)
        v9_result, v9_trace = v9_compute(text)

        v8_label = classify(v8_result.v)
        v9_label = classify(v9_result.v)

        v8_ok = (v8_label == human)
        v9_ok = (v9_label == human)

        if v8_ok:
            v8_correct += 1
        if v9_ok:
            v9_correct += 1
        if v8_label == v9_label:
            agree += 1

        delta_v = abs(v8_result.v - v9_result.v)

        if delta_v > threshold or verbose:
            marker = ""
            if v9_ok and not v8_ok:
                marker = " <- V9 WINS"
            elif v8_ok and not v9_ok:
                marker = " <- V8 WINS"
            elif not v8_ok and not v9_ok:
                marker = " <- BOTH WRONG"

            print(f"\n  \"{text[:70]}...\"" if len(text) > 70 else f"\n  \"{text}\"")
            print(f"  Human: {human}")
            print(f"  V8: V={v8_result.v} ({v8_label}) {'OK' if v8_ok else 'WRONG'}")
            print(f"  V9: V={v9_result.v} ({v9_label}) {'OK' if v9_ok else 'WRONG'}")
            print(f"  Nucleus: {v9_trace['equation']['nucleus']} "
                  f"(root={v9_trace['equation']['nucleus_root']})")
            print(f"  Delta V: {delta_v}{marker}")

            if delta_v > threshold:
                divergences.append({
                    "text": text,
                    "human": human,
                    "v8_v": v8_result.v,
                    "v9_v": v9_result.v,
                    "v8_label": v8_label,
                    "v9_label": v9_label,
                    "nucleus": v9_trace["equation"]["nucleus"],
                })

    print("\n" + "=" * 80)
    print(f"RESULTS:")
    print(f"  V8 accuracy: {v8_correct}/{len(sentences)} ({v8_correct/len(sentences)*100:.1f}%)")
    print(f"  V9 accuracy: {v9_correct}/{len(sentences)} ({v9_correct/len(sentences)*100:.1f}%)")
    print(f"  Agreement:   {agree}/{len(sentences)} ({agree/len(sentences)*100:.1f}%)")
    print(f"  Divergences: {len(divergences)} (>{threshold} points)")

    v9_wins = sum(1 for d in divergences
                  if classify(d["v9_v"]) == d["human"] and classify(d["v8_v"]) != d["human"])
    v8_wins = sum(1 for d in divergences
                  if classify(d["v8_v"]) == d["human"] and classify(d["v9_v"]) != d["human"])
    print(f"  V9 wins: {v9_wins}  |  V8 wins: {v8_wins}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V9 Shadow Pipeline")
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_shadow(threshold=args.threshold, verbose=args.verbose)
