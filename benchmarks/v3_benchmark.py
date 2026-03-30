#!/usr/bin/env python3
"""V2 vs V3 Benchmark — runs both engines on novel sentences.

V2 uses Valence threshold (V < 80 = crisis).
V3 uses structural pattern detection (any crisis pattern = detected).

This is the REAL test — sentences never in any benchmark or training data.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.pendulum import compute_vadug as v3_compute
from engine.word_classifier import classify_sentence
from engine.structures import StructureDetector
from demo.pendulum_v2 import PendulumV2


# ── Crisis patterns ─────────────────────────────────────────────

CRISIS_PATTERNS = frozenset({
    "FAREWELL", "METHOD_ACQUISITION", "SELF_REMOVAL",
    "EXHAUSTION", "NO_EXIT", "SELF_NULLIFY",
    "SUSPICIOUS_CALM", "BLANKET_APOLOGY", "FINALITY",
})


# ── Test sentences ──────────────────────────────────────────────

NOVEL_CRISIS = [
    "i just gave my dog to my neighbor",
    "i left the keys on the counter for whoever finds them",
    "take care of my cat for me",
    "im sorry for everything",
    "this is the last time youll hear from me",
    "i finally feel at peace",
    "just bought a bunch of pills",
    "found a bridge near my house",
    "wrote letters to everyone",
    "done with everything tbh",
    "honestly might just end it lol",
    "im so fucking done",
    "cant do this shit anymore",
    "no point in trying anymore",
]

NOVEL_SAFE = [
    "im having a bad day",
    "work was stressful",
    "i failed my exam",
    "my girlfriend broke up with me",
    "i feel kinda sad today",
    "mondays suck",
    "this homework is killing me",
    "im dead tired",
    "i could kill for a pizza",
]


# ── V2 Engine ───────────────────────────────────────────────────

def v2_detect_crisis(text: str) -> tuple:
    """V2 crisis detection: V < 80 threshold."""
    pendulum = PendulumV2()
    vadug, _trace = pendulum.process_text(text)
    detected = vadug.v < 80
    return detected, vadug


# ── V3 Engine ───────────────────────────────────────────────────

def v3_detect_crisis(text: str) -> tuple:
    """V3 crisis detection: structural pattern presence."""
    roles = classify_sentence(text.split())
    detector = StructureDetector()
    matches = detector.detect_all(roles)
    patterns = {m.pattern for m in matches}
    crisis = bool(patterns & CRISIS_PATTERNS)

    # Also compute VADUG for display
    vadug, trace = v3_compute(text)
    structures = [m.pattern for m in trace["structures"]]
    return crisis, vadug, structures


# ── Benchmark Runner ────────────────────────────────────────────

def run_benchmark():
    print("=" * 80)
    print("V2 vs V3 BENCHMARK — Novel Sentences")
    print("=" * 80)

    # --- Crisis sentences ---
    print(f"\n{'─' * 80}")
    print("CRISIS SENTENCES (should detect)")
    print(f"{'─' * 80}")
    print(f"{'Sentence':<50} {'V2':>5} {'V3':>5} {'V3 Patterns'}")
    print(f"{'─' * 50} {'─' * 5} {'─' * 5} {'─' * 30}")

    v2_crisis_correct = 0
    v3_crisis_correct = 0

    for sentence in NOVEL_CRISIS:
        v2_hit, v2_vadug = v2_detect_crisis(sentence)
        v3_hit, v3_vadug, v3_patterns = v3_detect_crisis(sentence)

        v2_mark = "HIT" if v2_hit else "MISS"
        v3_mark = "HIT" if v3_hit else "MISS"
        patterns_str = ", ".join(v3_patterns) if v3_patterns else "-"

        if v2_hit:
            v2_crisis_correct += 1
        if v3_hit:
            v3_crisis_correct += 1

        display = sentence[:48] + ".." if len(sentence) > 48 else sentence
        print(f"{display:<50} {v2_mark:>5} {v3_mark:>5} {patterns_str}")

    # --- Safe sentences ---
    print(f"\n{'─' * 80}")
    print("SAFE SENTENCES (should NOT detect)")
    print(f"{'─' * 80}")
    print(f"{'Sentence':<50} {'V2':>5} {'V3':>5} {'V3 Patterns'}")
    print(f"{'─' * 50} {'─' * 5} {'─' * 5} {'─' * 30}")

    v2_safe_correct = 0
    v3_safe_correct = 0

    for sentence in NOVEL_SAFE:
        v2_hit, v2_vadug = v2_detect_crisis(sentence)
        v3_hit, v3_vadug, v3_patterns = v3_detect_crisis(sentence)

        v2_mark = "SAFE" if not v2_hit else "FP"
        v3_mark = "SAFE" if not v3_hit else "FP"
        patterns_str = ", ".join(v3_patterns) if v3_patterns else "-"

        if not v2_hit:
            v2_safe_correct += 1
        if not v3_hit:
            v3_safe_correct += 1

        display = sentence[:48] + ".." if len(sentence) > 48 else sentence
        print(f"{display:<50} {v2_mark:>5} {v3_mark:>5} {patterns_str}")

    # --- Summary ---
    total_crisis = len(NOVEL_CRISIS)
    total_safe = len(NOVEL_SAFE)
    total = total_crisis + total_safe

    v2_total = v2_crisis_correct + v2_safe_correct
    v3_total = v3_crisis_correct + v3_safe_correct

    v2_pct = v2_total / total * 100
    v3_pct = v3_total / total * 100

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"  Crisis detection (true positives):")
    print(f"    V2: {v2_crisis_correct}/{total_crisis} ({v2_crisis_correct / total_crisis * 100:.0f}%)")
    print(f"    V3: {v3_crisis_correct}/{total_crisis} ({v3_crisis_correct / total_crisis * 100:.0f}%)")
    print(f"  Safe classification (true negatives):")
    print(f"    V2: {v2_safe_correct}/{total_safe} ({v2_safe_correct / total_safe * 100:.0f}%)")
    print(f"    V3: {v3_safe_correct}/{total_safe} ({v3_safe_correct / total_safe * 100:.0f}%)")
    print(f"\n  V2: {v2_total}/{total} ({v2_pct:.0f}%)  V3: {v3_total}/{total} ({v3_pct:.0f}%)")
    print(f"{'=' * 80}")

    return v2_pct, v3_pct


if __name__ == "__main__":
    run_benchmark()
