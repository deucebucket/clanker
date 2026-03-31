#!/usr/bin/env python3
"""Bayesian Vocabulary Analyzer -- corpus-based word frequency vs current V values.

For every word in the Clanker vocabulary, counts how often it appears in
positive vs negative sentences in the EmpatheticDialogues corpus, then
computes a Bayesian-weighted suggested dV based on actual usage frequency.

Usage:
    python3 scripts/bayesian_vocab.py
    python3 scripts/bayesian_vocab.py --max-sentences 50000
    python3 scripts/bayesian_vocab.py --min-count 10
"""

import sys
import os
import json
import random
import argparse
import re
from collections import defaultdict
from pathlib import Path

# Engine imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.vocabulary import VOCABULARY
from engine.pendulum import compute_vadug

# ── Config ─────────────────────────────────────────────────────────

CORPUS_PATH = Path(__file__).resolve().parent.parent / "training" / "data" / "empathetic_dialogues.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent / "bayesian_results.jsonl"
V_NEUTRAL = 128  # center point
MAX_SWING = 127  # max distance from center
DEFAULT_MAX_SENTENCES = 30000
DEFAULT_MIN_COUNT = 5  # minimum appearances to be statistically meaningful
PRIOR_STRENGTH = 10  # Bayesian prior pseudo-count (pulls toward 0.5 with few observations)


def load_corpus(path: Path, max_sentences: int) -> list:
    """Load sentences from JSONL corpus. Uses pre-scored VADUG if available."""
    sentences = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("english", "")
            if not text or len(text.split()) < 3:
                continue
            # Use pre-scored V if available, otherwise mark for engine scoring
            vadug = row.get("vadug")
            if vadug and len(vadug) >= 1:
                v_score = vadug[0]
            else:
                v_score = None
            sentences.append({"text": text, "v": v_score})

    # Sample if too large
    if len(sentences) > max_sentences:
        random.seed(42)
        sentences = random.sample(sentences, max_sentences)

    return sentences


def score_missing(sentences: list) -> list:
    """Score any sentences that don't have pre-computed V values."""
    needs_scoring = [s for s in sentences if s["v"] is None]
    if needs_scoring:
        print(f"  Scoring {len(needs_scoring)} sentences through engine...")
        for s in needs_scoring:
            vadug, _ = compute_vadug(s["text"])
            s["v"] = vadug.v
    return sentences


def tokenize(text: str) -> list:
    """Simple word tokenization matching engine behavior."""
    # Lowercase, strip punctuation from edges
    words = text.lower().split()
    cleaned = []
    for w in words:
        w = re.sub(r"^[^a-z]+", "", w)
        w = re.sub(r"[^a-z]+$", "", w)
        if w:
            cleaned.append(w)
    return cleaned


def analyze(sentences: list, min_count: int) -> dict:
    """Count positive/negative occurrences for each vocabulary word."""

    # Build vocab lookup (lowercase)
    vocab_words = {}
    for word, forces in VOCABULARY.items():
        vocab_words[word.lower()] = forces

    # Count per word
    word_stats = defaultdict(lambda: {
        "pos_count": 0, "neg_count": 0,
        "pos_examples": [], "neg_examples": []
    })

    n_pos = 0
    n_neg = 0

    for sent in sentences:
        v = sent["v"]
        is_positive = v > V_NEUTRAL
        if is_positive:
            n_pos += 1
        else:
            n_neg += 1

        words_in_sent = set(tokenize(sent["text"]))
        for w in words_in_sent:
            if w not in vocab_words:
                continue
            stats = word_stats[w]
            if is_positive:
                stats["pos_count"] += 1
                if len(stats["pos_examples"]) < 3:
                    stats["pos_examples"].append(sent["text"])
            else:
                stats["neg_count"] += 1
                if len(stats["neg_examples"]) < 3:
                    stats["neg_examples"].append(sent["text"])

    # Compute metrics
    results = []
    for word, stats in word_stats.items():
        total = stats["pos_count"] + stats["neg_count"]
        if total < min_count:
            continue

        forces = vocab_words[word]
        current_dv = forces[0]  # dV from vocabulary

        # Bayesian smoothed pos_ratio (with prior pulling toward 0.5)
        pos_ratio = (stats["pos_count"] + PRIOR_STRENGTH * 0.5) / (total + PRIOR_STRENGTH)
        neg_ratio = 1.0 - pos_ratio

        # Suggested dV: preserve intensity magnitude, adjust direction by corpus evidence
        intensity = abs(current_dv)
        suggested_dv = round((pos_ratio - 0.5) * 2 * intensity)

        # Alternative: frequency-proportional suggestion (not capped by current intensity)
        # Uses the full swing range scaled by how skewed the distribution is
        freq_suggested_dv = round((pos_ratio - 0.5) * 2 * MAX_SWING)

        # Mismatch: current dV says positive but corpus says negative, or vice versa
        direction_match = (current_dv > 0 and pos_ratio > 0.5) or \
                         (current_dv < 0 and pos_ratio < 0.5) or \
                         (current_dv == 0)
        mismatch = abs(suggested_dv - current_dv)

        results.append({
            "word": word,
            "current_dv": current_dv,
            "pos_count": stats["pos_count"],
            "neg_count": stats["neg_count"],
            "total": total,
            "pos_ratio": round(pos_ratio, 4),
            "neg_ratio": round(neg_ratio, 4),
            "suggested_dv": suggested_dv,
            "freq_suggested_dv": freq_suggested_dv,
            "mismatch": mismatch,
            "direction_agrees": direction_match,
            "pos_example": stats["pos_examples"][0] if stats["pos_examples"] else "",
            "neg_example": stats["neg_examples"][0] if stats["neg_examples"] else "",
        })

    return {
        "results": sorted(results, key=lambda r: r["mismatch"], reverse=True),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_sentences": len(sentences),
        "n_vocab": len(vocab_words),
        "n_found": len(results),
        "n_below_threshold": len(word_stats) - len(results),
    }


def print_report(data: dict):
    """Print human-readable report."""
    results = data["results"]
    print()
    print("=" * 80)
    print("BAYESIAN VOCABULARY ANALYSIS")
    print("=" * 80)
    print()
    print(f"  Corpus sentences:     {data['n_sentences']:,}")
    print(f"  Positive (V > 128):   {data['n_pos']:,}")
    print(f"  Negative (V <= 128):  {data['n_neg']:,}")
    print(f"  Pos/Neg ratio:        {data['n_pos']/max(data['n_neg'],1):.2f}")
    print(f"  Vocabulary size:      {data['n_vocab']:,}")
    print(f"  Words found:          {data['n_found']:,}")
    print(f"  Below threshold:      {data['n_below_threshold']:,}")
    print()

    # Direction disagreements
    disagree = [r for r in results if not r["direction_agrees"]]
    agree = [r for r in results if r["direction_agrees"]]

    print(f"  Direction AGREES:     {len(agree):,} words")
    print(f"  Direction DISAGREES:  {len(disagree):,} words")
    print()

    # Top mismatches
    print("-" * 80)
    print("TOP 40 MISMATCHES (biggest gap between current dV and corpus-suggested dV)")
    print("-" * 80)
    print(f"{'Word':<20} {'currDV':>7} {'sugDV':>7} {'gap':>5} {'posR':>6} {'pos#':>5} {'neg#':>5} {'dir':>5}")
    print("-" * 80)
    for r in results[:40]:
        flag = " OK" if r["direction_agrees"] else " XX"
        print(f"{r['word']:<20} {r['current_dv']:>7} {r['suggested_dv']:>7} {r['mismatch']:>5} "
              f"{r['pos_ratio']:>6.3f} {r['pos_count']:>5} {r['neg_count']:>5} {flag}")
    print()

    # Direction disagreements detail
    if disagree:
        print("-" * 80)
        print(f"DIRECTION DISAGREEMENTS ({len(disagree)} words where corpus says opposite sign)")
        print("-" * 80)
        print(f"{'Word':<20} {'currDV':>7} {'sugDV':>7} {'posR':>6} {'total':>6}")
        print("-" * 80)
        for r in sorted(disagree, key=lambda x: x["total"], reverse=True)[:50]:
            print(f"{r['word']:<20} {r['current_dv']:>7} {r['suggested_dv']:>7} "
                  f"{r['pos_ratio']:>6.3f} {r['total']:>6}")
        print()

    # High-confidence words (lots of data, small Bayesian correction)
    high_conf = [r for r in results if r["total"] >= 100]
    if high_conf:
        print("-" * 80)
        print(f"HIGH-CONFIDENCE WORDS ({len(high_conf)} with 100+ occurrences)")
        print("-" * 80)
        print(f"{'Word':<20} {'currDV':>7} {'sugDV':>7} {'posR':>6} {'total':>6} {'dir':>5}")
        print("-" * 80)
        for r in sorted(high_conf, key=lambda x: x["mismatch"], reverse=True)[:30]:
            flag = " OK" if r["direction_agrees"] else " XX"
            print(f"{r['word']:<20} {r['current_dv']:>7} {r['suggested_dv']:>7} "
                  f"{r['pos_ratio']:>6.3f} {r['total']:>6} {flag}")
        print()

    # Neutral words that corpus shows are NOT neutral
    neutral_words = [r for r in results if r["current_dv"] == 0 and abs(r["pos_ratio"] - 0.5) > 0.1]
    if neutral_words:
        print("-" * 80)
        print(f"NEUTRAL WORDS WITH CORPUS BIAS ({len(neutral_words)} words with dV=0 but pos_ratio off-center)")
        print("-" * 80)
        for r in sorted(neutral_words, key=lambda x: abs(x["pos_ratio"] - 0.5), reverse=True)[:20]:
            direction = "POS" if r["pos_ratio"] > 0.5 else "NEG"
            print(f"  {r['word']:<20} pos_ratio={r['pos_ratio']:.3f} ({direction})  "
                  f"freq_suggested_dv={r['freq_suggested_dv']}  n={r['total']}")
        print()


def save_results(data: dict, path: Path):
    """Save results to JSONL."""
    with open(path, "w") as f:
        # Header line with summary
        summary = {
            "type": "summary",
            "n_sentences": data["n_sentences"],
            "n_pos": data["n_pos"],
            "n_neg": data["n_neg"],
            "n_vocab": data["n_vocab"],
            "n_found": data["n_found"],
            "prior_strength": PRIOR_STRENGTH,
        }
        f.write(json.dumps(summary) + "\n")

        for r in data["results"]:
            f.write(json.dumps(r) + "\n")

    print(f"Results saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Bayesian vocabulary analyzer")
    parser.add_argument("--max-sentences", type=int, default=DEFAULT_MAX_SENTENCES,
                        help=f"Max sentences to process (default: {DEFAULT_MAX_SENTENCES})")
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                        help=f"Min word occurrences to include (default: {DEFAULT_MIN_COUNT})")
    parser.add_argument("--all", action="store_true",
                        help="Use all sentences (no sampling)")
    parser.add_argument("--corpus", type=str, default=str(CORPUS_PATH),
                        help="Path to corpus JSONL")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        sys.exit(1)

    max_sent = 999_999_999 if args.all else args.max_sentences

    print(f"Loading corpus from {corpus_path}...")
    sentences = load_corpus(corpus_path, max_sent)
    print(f"  Loaded {len(sentences):,} sentences")

    sentences = score_missing(sentences)

    print(f"Analyzing vocabulary ({len(VOCABULARY):,} words)...")
    data = analyze(sentences, args.min_count)

    print_report(data)
    save_results(data, OUTPUT_PATH)


if __name__ == "__main__":
    main()
