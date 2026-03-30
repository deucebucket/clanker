#!/usr/bin/env python3
"""Stress test the V2 engine with diverse inputs and log everything.

Generates hundreds of test sentences across categories, runs them through
the engine, and flags anything that looks wrong. No LLM needed for generation —
we know what to test. An LLM can review the output JSON afterward.

Usage:
    python3 benchmarks/stress_test.py              # run all tests
    python3 benchmarks/stress_test.py --verbose     # show every result
    python3 benchmarks/stress_test.py --failures    # show only failures
"""

import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.pendulum_v2 import PendulumV2

engine = PendulumV2()

# ── Test sentence generators ──────────────────────────────────────

def single_words():
    """Single emotional words — engine should read direction correctly."""
    return [
        # Clearly positive (expect V > 132)
        ("happy", "positive"), ("love", "positive"), ("amazing", "positive"),
        ("wonderful", "positive"), ("excited", "positive"), ("grateful", "positive"),
        ("fantastic", "positive"), ("beautiful", "positive"), ("joy", "positive"),
        ("brilliant", "positive"), ("awesome", "positive"), ("delighted", "positive"),
        # Clearly negative (expect V < 124)
        ("sad", "negative"), ("hate", "negative"), ("terrible", "negative"),
        ("angry", "negative"), ("depressed", "negative"), ("miserable", "negative"),
        ("awful", "negative"), ("disgusting", "negative"), ("furious", "negative"),
        ("devastated", "negative"), ("hopeless", "negative"), ("worthless", "negative"),
        ("die", "negative"), ("kill", "negative"), ("suicide", "negative"),
        # Neutral (expect 124 <= V <= 132)
        ("the", "neutral"), ("and", "neutral"), ("table", "neutral"),
        ("computer", "neutral"), ("water", "neutral"),
    ]

def crisis_sentences():
    """Crisis sentences — V should be low, U elevated."""
    return [
        ("i want to die", "crisis"), ("i want to kill myself", "crisis"),
        ("nobody would miss me", "crisis"), ("i dont want to exist", "crisis"),
        ("im going to end it tonight", "crisis"), ("goodbye everyone", "crisis"),
        ("the world would be better without me", "crisis"),
        ("i cant take it anymore", "crisis"), ("whats the point of living", "crisis"),
        ("i wish i was dead", "crisis"), ("help me please", "crisis"),
        ("i feel helpless and hopeless", "crisis"),
    ]

def sarcasm_sentences():
    """Sarcastic sentences — should read as negative despite positive words."""
    return [
        ("oh great another monday", "negative"),
        ("yeah right", "negative"),
        ("sure that worked out well", "negative"),
        ("im absolutely thrilled to redo all my work", "negative"),
        ("thanks for nothing", "negative"),
        ("how delightful another policy change", "negative"),
        ("brilliant idea nobody thought of that before", "negative"),
    ]

def hedging_sentences():
    """Heavily hedged — should read as neutral."""
    return [
        ("its possible that maybe this could be an issue", "neutral"),
        ("generally speaking things sometimes tend to go badly", "neutral"),
        ("in theory one could argue there are limitations", "neutral"),
        ("hypothetically speaking what if this doesnt work", "neutral"),
        ("i dont want to jump to conclusions but it looks concerning", "neutral"),
    ]

def slang_sentences():
    """Slang/internet speak — should still read emotional direction."""
    return [
        ("tbh im depresed rn", "negative"),
        ("ngl this is fire", "positive"),
        ("im so happyyyy rn lol", "positive"),
        ("fam this slaps", "positive"),
        ("smh this is trash", "negative"),
        ("bruh im done", "negative"),
        ("deadass this is amazing", "positive"),
        ("idc anymore", "negative"),
        ("k", "negative"),
        ("im fine", "negative"),  # im fine = not actually fine
    ]

def body_and_objects():
    """Body parts and objects — gravity should differentiate."""
    return [
        ("my heart is broken", "negative"),
        ("my car is broken", "negative"),
        ("my soul is broken", "negative"),
        ("im pregnant", "neutral"),  # ambiguous V, high G
        ("my period is late", "negative"),
    ]

def carrier_verbs():
    """Carrier verbs — direction from payload."""
    return [
        ("make me happy", "positive"),
        ("make me sad", "negative"),
        ("make history", "positive"),
        ("make me sick", "negative"),
        ("this is fucking amazing", "positive"),
        ("this is fucking terrible", "negative"),
    ]

def protest_too_much():
    """Protest/repetition patterns."""
    return [
        ("no no no", "negative"),
        ("help help help", "negative"),
        ("im fine", "negative"),
        ("really im fine", "negative"),
        ("no im fine", "negative"),
    ]

def connection_words():
    """Talk/connection patterns."""
    return [
        ("we need to talk", "negative"),  # gravity bomb
        ("stop talking", "negative"),
        ("shut up", "negative"),
        ("lets talk", "neutral"),
    ]

# ── Runner ────────────────────────────────────────────────────────

def classify(v):
    if v > 132: return "positive"
    elif v < 124: return "negative"
    else: return "neutral"

def run_tests(verbose=False, failures_only=False):
    all_tests = []
    categories = [
        ("single_words", single_words()),
        ("crisis", crisis_sentences()),
        ("sarcasm", sarcasm_sentences()),
        ("hedging", hedging_sentences()),
        ("slang", slang_sentences()),
        ("body_objects", body_and_objects()),
        ("carrier_verbs", carrier_verbs()),
        ("protest", protest_too_much()),
        ("connection", connection_words()),
    ]

    total = 0
    passed = 0
    failed = 0
    results = []

    print("=" * 70)
    print("CLANKER V2 STRESS TEST")
    print("=" * 70)

    start = time.time()

    for cat_name, sentences in categories:
        cat_pass = 0
        cat_fail = 0
        for text, expected in sentences:
            total += 1
            vadug, trace = engine.process_text(text)
            predicted = classify(vadug.v)

            # Crisis special: also check U
            if expected == "crisis":
                is_correct = vadug.v < 100 or vadug.u > 40
            elif expected == "neutral":
                is_correct = 118 <= vadug.v <= 138  # wider band for stress test
            else:
                is_correct = predicted == expected

            entry = {
                "text": text,
                "category": cat_name,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "v": vadug.v, "a": vadug.a, "d": vadug.d,
                "u": vadug.u, "g": vadug.g,
            }
            results.append(entry)

            if is_correct:
                passed += 1
                cat_pass += 1
                if verbose and not failures_only:
                    print(f"  [OK] V={vadug.v:3d} | {expected:8s} | {text}")
            else:
                failed += 1
                cat_fail += 1
                if verbose or failures_only:
                    print(f"  [XX] V={vadug.v:3d} U={vadug.u:3d} | exp={expected:8s} pred={predicted:8s} | {text}")

        pct = 100 * cat_pass / (cat_pass + cat_fail) if (cat_pass + cat_fail) > 0 else 0
        print(f"  {cat_name:20s} {cat_pass:3d}/{cat_pass+cat_fail:3d} ({pct:.0f}%)")

    elapsed = time.time() - start

    print()
    print("-" * 70)
    print(f"  TOTAL: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"  PASSED: {passed}  FAILED: {failed}")
    print(f"  Time: {elapsed:.2f}s ({total/elapsed:.0f} sentences/sec)")
    print("-" * 70)

    # Save results
    outpath = os.path.join(os.path.dirname(__file__), "stress_results.json")
    with open(outpath, "w") as f:
        json.dump({
            "total": total,
            "passed": passed,
            "failed": failed,
            "accuracy": round(100 * passed / total, 1),
            "time_sec": round(elapsed, 2),
            "results": results,
        }, f, indent=2)
    print(f"  Saved to {outpath}")

    return passed, total


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    failures = "--failures" in sys.argv or "-f" in sys.argv
    run_tests(verbose=verbose or failures, failures_only=failures)
