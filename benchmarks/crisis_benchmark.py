#!/usr/bin/env python3
"""Crisis Detection Benchmark — V5.5 engine.

Tests crisis recall and false positive rate on real text.
ALL engine calls use V5.5 (engine.pendulum). No V2 imports.

Usage:
    python3 benchmarks/crisis_benchmark.py
    python3 benchmarks/crisis_benchmark.py --verbose
"""

import sys, os, time, json, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.pendulum import compute_vadug
from engine.structures import StructureDetector

_CRISIS_PATTERNS = {
    "SELF_NULLIFY", "SELF_REMOVAL", "METHOD_ACQUISITION",
    "PURSUIT_OF_METHOD", "NO_EXIT", "FAREWELL", "FINALITY",
    "SELF_HARM_INTENT", "EXISTENTIAL_NEGATION", "SOCIAL_NULLITY",
    "RHETORICAL_HOPELESSNESS",
}

_CRISIS_SENTENCES = [
    "i dont want to be here anymore",
    "i want to kill myself",
    "nobody would miss me",
    "im better off dead",
    "i cant do this anymore",
    "whats the point of living",
    "i just want the pain to stop",
    "ive been thinking about ending it",
    "i have the pills right here",
    "tell my kids i love them",
    "i am nothing",
    "everyone would be better off without me",
    "i dont deserve to be happy",
    "im just a burden",
    "theres no way out",
    "i wish i was never born",
    "nobody would even notice if i disappeared",
    "im done trying",
    "i cant take this anymore",
    "i just want it to be over",
    "the world doesnt need me",
    "i wrote a note",
    "i found the bridge",
    "im going to jump",
    "this is my last message",
    "goodbye everyone",
    "i already said my goodbyes",
    "i have a plan",
    "ive made my decision",
    "tonight is the night",
]

_SAFE_SENTENCES = [
    "the meeting is at three",
    "i need to buy groceries",
    "the weather is nice today",
    "i had pizza for lunch",
    "can you pass the salt",
    "im going to the store",
    "the movie starts at seven",
    "i need to do laundry",
    "happy birthday",
    "see you tomorrow",
    "good morning everyone",
    "the traffic was terrible",
    "i finished my homework",
    "lets get coffee sometime",
    "the dog needs a walk",
    "i got a new job",
    "the game starts at noon",
    "its raining outside",
    "i need new shoes",
    "dinner was great",
    "I JUST GOT THE JOB",
    "my kid took their first steps today",
    "clean for 6 months now",
    "just got engaged",
    "she said yes",
    "bro that was fire",
    "no cap that was insane",
    "gg ez clap",
    "thats bussin fr",
    "we closed on the house",
]


def is_crisis(vadug, meta):
    """Check if the engine reads this as crisis.

    Crisis requires PERSONAL involvement -- complaints about things
    (traffic, weather) don't count even if V is low.

    Signals:
    - Any crisis structural pattern = crisis (structures already require self-ref)
    - Very low V + low W = crisis (self-worth collapse + negative = personal)
    - Low V + withdrawing = crisis (negative + pulling away = personal)
    - Low V + low W = crisis
    """
    structs = {s.pattern for s in meta.get("structures", [])}
    has_crisis_pattern = bool(structs & _CRISIS_PATTERNS)
    low_v = vadug.v < 100
    very_low_v = vadug.v < 70
    low_w = vadug.w < 100
    withdrawing = vadug.i < 50
    return (has_crisis_pattern
            or (very_low_v and low_w)
            or (low_v and withdrawing)
            or (low_v and low_w))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  CRISIS DETECTION BENCHMARK — V5.5 engine")
    print(f"{'=' * 70}")

    # Crisis sentences: should detect
    crisis_hits = 0
    crisis_misses = []
    for s in _CRISIS_SENTENCES:
        vadug, meta = compute_vadug(s)
        detected = is_crisis(vadug, meta)
        if detected:
            crisis_hits += 1
        else:
            crisis_misses.append((s, vadug))
        if args.verbose:
            tag = "HIT" if detected else "MISS"
            structs = [st.pattern for st in meta.get("structures", [])]
            print(f"  [{tag}] V={vadug.v:3d} W={vadug.w:3d} I={vadug.i:3d} [{','.join(structs) or '-'}]  {s}")

    # Safe sentences: should NOT detect
    safe_clean = 0
    false_positives = []
    for s in _SAFE_SENTENCES:
        vadug, meta = compute_vadug(s)
        detected = is_crisis(vadug, meta)
        if not detected:
            safe_clean += 1
        else:
            false_positives.append((s, vadug))
        if args.verbose:
            tag = "OK" if not detected else "FP!"
            structs = [st.pattern for st in meta.get("structures", [])]
            print(f"  [{tag}] V={vadug.v:3d} W={vadug.w:3d} I={vadug.i:3d} [{','.join(structs) or '-'}]  {s}")

    recall = crisis_hits / len(_CRISIS_SENTENCES) * 100
    fp_rate = len(false_positives) / len(_SAFE_SENTENCES) * 100

    print(f"\n  CRISIS RECALL: {crisis_hits}/{len(_CRISIS_SENTENCES)} ({recall:.1f}%)")
    print(f"  FALSE POSITIVE: {len(false_positives)}/{len(_SAFE_SENTENCES)} ({fp_rate:.1f}%)")

    if crisis_misses:
        print(f"\n  MISSED ({len(crisis_misses)}):")
        for s, v in crisis_misses:
            print(f"    V={v.v:3d} W={v.w:3d} I={v.i:3d}  {s}")

    if false_positives:
        print(f"\n  FALSE POSITIVES ({len(false_positives)}):")
        for s, v in false_positives:
            print(f"    V={v.v:3d} W={v.w:3d} I={v.i:3d}  {s}")

    print(f"\n{'=' * 70}")

    # Save
    out = {
        "engine": "v5.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recall": round(recall, 1),
        "false_positive_rate": round(fp_rate, 1),
        "crisis_tested": len(_CRISIS_SENTENCES),
        "safe_tested": len(_SAFE_SENTENCES),
        "misses": [s for s, _ in crisis_misses],
        "false_positives": [s for s, _ in false_positives],
    }
    out_path = os.path.join(os.path.dirname(__file__), "crisis_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
