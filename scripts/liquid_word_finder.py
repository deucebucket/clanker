#!/usr/bin/env python3
"""Liquid Word Finder — discovers words whose polarity flips in context.

Pairs high-|V| words from the vocabulary with opposing-polarity context words,
scores each pair through the engine, and identifies LIQUID words (context can
flip their V across the 128 neutral threshold).

Usage:
    python3 scripts/liquid_word_finder.py
    python3 scripts/liquid_word_finder.py --threshold 40   # lower V threshold
    python3 scripts/liquid_word_finder.py --top 50         # show top 50 liquid words
"""

import argparse
import sys
import os
import time

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.forces_curated import EMOTIONAL_VOCABULARY
from engine.pendulum import compute_vadug

# ── Context word pools ──────────────────────────────────────────

POSITIVE_CONTEXT = [
    "joy", "love", "happy", "excited", "amazing", "beautiful", "wonderful",
    "great", "best", "winning", "success", "celebrate", "proud", "goal",
    "achievement", "baby", "wedding", "birthday", "christmas", "promotion",
]

NEGATIVE_CONTEXT = [
    "death", "pain", "hurt", "sad", "angry", "terrible", "horrible",
    "worst", "losing", "failure", "funeral", "abuse", "violence", "war",
    "crime", "accident",
]

NEUTRAL_THRESHOLD = 128  # V center point


def score_pair(word: str, context: str) -> dict:
    """Score two phrase patterns and return the best result."""
    patterns = [
        f"{context} {word}",
        f"{word} of {context}",
    ]
    best = None
    for phrase in patterns:
        vadug, trace = compute_vadug(phrase)
        if best is None or abs(vadug.v - NEUTRAL_THRESHOLD) > abs(best["v"] - NEUTRAL_THRESHOLD):
            best = {
                "phrase": phrase,
                "v": vadug.v,
                "a": vadug.a,
                "d": vadug.d,
                "u": vadug.u,
                "g": vadug.g,
            }
    return best


def find_liquid_words(v_threshold: int = 50) -> tuple:
    """Find liquid and solid words.

    Returns (liquid_words, solid_words) where each entry is a dict with
    word info and scoring results.
    """
    # Step 1: Filter to high-|V| words
    high_v_words = {}
    for word, forces in EMOTIONAL_VOCABULARY.items():
        dv = forces[0]
        if abs(dv) >= v_threshold:
            high_v_words[word] = dv

    print(f"Vocabulary size: {len(EMOTIONAL_VOCABULARY)}")
    print(f"Words with |dV| >= {v_threshold}: {len(high_v_words)}")

    neg_words = {w: dv for w, dv in high_v_words.items() if dv < 0}
    pos_words = {w: dv for w, dv in high_v_words.items() if dv > 0}
    print(f"  Negative words (dV < 0): {len(neg_words)}")
    print(f"  Positive words (dV > 0): {len(pos_words)}")

    total_pairs = len(neg_words) * len(POSITIVE_CONTEXT) + len(pos_words) * len(NEGATIVE_CONTEXT)
    print(f"Total pairs to score: {total_pairs}")
    print()

    liquid = []
    solid = []
    scored = 0
    t0 = time.time()

    # Step 2: Score negative words against positive context
    for word, dv in neg_words.items():
        best_flip = None
        best_swing = 0

        for ctx in POSITIVE_CONTEXT:
            result = score_pair(word, ctx)
            scored += 1

            # Word is negative (dv < 0), so base V < 128.
            # A flip means the pair scored V > 128.
            swing = result["v"] - NEUTRAL_THRESHOLD  # positive = flipped
            if swing > best_swing:
                best_swing = swing
                best_flip = result

        if best_flip and best_swing > 0:
            # Compute solo V for reference
            solo_vadug, _ = compute_vadug(word)
            liquid.append({
                "word": word,
                "dv": dv,
                "solo_v": solo_vadug.v,
                "best_phrase": best_flip["phrase"],
                "flipped_v": best_flip["v"],
                "swing": best_swing,
                "full_vadug": best_flip,
            })
        else:
            solo_vadug, _ = compute_vadug(word)
            solid.append({
                "word": word,
                "dv": dv,
                "solo_v": solo_vadug.v,
            })

        if scored % 500 == 0:
            elapsed = time.time() - t0
            rate = scored / elapsed if elapsed > 0 else 0
            print(f"  ... scored {scored}/{total_pairs} pairs ({rate:.0f} pairs/sec)")

    # Step 3: Score positive words against negative context
    for word, dv in pos_words.items():
        best_flip = None
        best_swing = 0

        for ctx in NEGATIVE_CONTEXT:
            result = score_pair(word, ctx)
            scored += 1

            # Word is positive (dv > 0), so base V > 128.
            # A flip means the pair scored V < 128.
            swing = NEUTRAL_THRESHOLD - result["v"]  # positive = flipped
            if swing > best_swing:
                best_swing = swing
                best_flip = result

        if best_flip and best_swing > 0:
            solo_vadug, _ = compute_vadug(word)
            liquid.append({
                "word": word,
                "dv": dv,
                "solo_v": solo_vadug.v,
                "best_phrase": best_flip["phrase"],
                "flipped_v": best_flip["v"],
                "swing": best_swing,
                "full_vadug": best_flip,
            })
        else:
            solo_vadug, _ = compute_vadug(word)
            solid.append({
                "word": word,
                "dv": dv,
                "solo_v": solo_vadug.v,
            })

        if scored % 500 == 0:
            elapsed = time.time() - t0
            rate = scored / elapsed if elapsed > 0 else 0
            print(f"  ... scored {scored}/{total_pairs} pairs ({rate:.0f} pairs/sec)")

    elapsed = time.time() - t0
    print(f"\nScored {scored} pairs in {elapsed:.1f}s ({scored/elapsed:.0f} pairs/sec)")

    # Sort liquid by swing (easiest to flip first)
    liquid.sort(key=lambda x: x["swing"], reverse=True)
    solid.sort(key=lambda x: abs(x["dv"]), reverse=True)

    return liquid, solid


def print_results(liquid: list, solid: list, top_n: int = 100):
    """Print formatted results."""

    print("\n" + "=" * 80)
    print(f"  LIQUID WORDS — context flips V across 128 threshold ({len(liquid)} found)")
    print("=" * 80)
    print(f"{'Word':<20} {'dV':>5} {'Solo V':>7} {'Flipped V':>10} {'Swing':>6}  Best Phrase")
    print("-" * 80)

    for entry in liquid[:top_n]:
        direction = "NEG->POS" if entry["dv"] < 0 else "POS->NEG"
        print(
            f"{entry['word']:<20} {entry['dv']:>+5} {entry['solo_v']:>7} "
            f"{entry['flipped_v']:>10} {entry['swing']:>+6}  "
            f"{entry['best_phrase']}  [{direction}]"
        )

    if len(liquid) > top_n:
        print(f"\n  ... and {len(liquid) - top_n} more liquid words")

    print("\n" + "=" * 80)
    print(f"  SOLID WORDS — context cannot flip V ({len(solid)} found)")
    print("=" * 80)
    print(f"{'Word':<20} {'dV':>5} {'Solo V':>7}  Interpretation")
    print("-" * 80)

    for entry in solid[:top_n]:
        interp = "strongly negative" if entry["dv"] < -80 else \
                 "negative" if entry["dv"] < 0 else \
                 "strongly positive" if entry["dv"] > 80 else "positive"
        print(f"{entry['word']:<20} {entry['dv']:>+5} {entry['solo_v']:>7}  {interp} — correctly pinned")

    if len(solid) > top_n:
        print(f"\n  ... and {len(solid) - top_n} more solid words")

    # Summary stats
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    total = len(liquid) + len(solid)
    liq_pct = len(liquid) / total * 100 if total else 0
    sol_pct = len(solid) / total * 100 if total else 0
    print(f"  Total high-|V| words analyzed: {total}")
    print(f"  LIQUID (context-dependent):    {len(liquid)} ({liq_pct:.1f}%)")
    print(f"  SOLID  (context-resistant):    {len(solid)} ({sol_pct:.1f}%)")

    if liquid:
        avg_swing = sum(e["swing"] for e in liquid) / len(liquid)
        max_swing = liquid[0]["swing"]
        print(f"  Average flip swing:            {avg_swing:.1f}")
        print(f"  Max flip swing:                {max_swing} ({liquid[0]['word']} -> \"{liquid[0]['best_phrase']}\")")

    # Top 10 most liquid (biggest swings)
    if liquid:
        print(f"\n  TOP 10 MOST LIQUID WORDS:")
        for i, entry in enumerate(liquid[:10], 1):
            print(f"    {i:>2}. {entry['word']:<18} swing={entry['swing']:>+4}  \"{entry['best_phrase']}\"")

    # Breakdown by original polarity
    liq_neg = [e for e in liquid if e["dv"] < 0]
    liq_pos = [e for e in liquid if e["dv"] > 0]
    sol_neg = [e for e in solid if e["dv"] < 0]
    sol_pos = [e for e in solid if e["dv"] > 0]

    print(f"\n  POLARITY BREAKDOWN:")
    print(f"    Negative words: {len(liq_neg)} liquid / {len(sol_neg)} solid "
          f"({len(liq_neg)/(len(liq_neg)+len(sol_neg))*100:.0f}% liquid)" if (liq_neg or sol_neg) else "")
    print(f"    Positive words: {len(liq_pos)} liquid / {len(sol_pos)} solid "
          f"({len(liq_pos)/(len(liq_pos)+len(sol_pos))*100:.0f}% liquid)" if (liq_pos or sol_pos) else "")


def main():
    parser = argparse.ArgumentParser(description="Find liquid words in Clanker vocabulary")
    parser.add_argument("--threshold", type=int, default=50,
                        help="Minimum |dV| to consider (default: 50)")
    parser.add_argument("--top", type=int, default=100,
                        help="Show top N results per category (default: 100)")
    args = parser.parse_args()

    print("LIQUID WORD FINDER — Clanker Engine")
    print(f"Finding words where context flips polarity (|dV| >= {args.threshold})")
    print()

    liquid, solid = find_liquid_words(v_threshold=args.threshold)
    print_results(liquid, solid, top_n=args.top)


if __name__ == "__main__":
    main()
