#!/usr/bin/env python3
"""
Idiom Discovery System — finds idioms MATHEMATICALLY from residuals.

Concept: if bigram/trigram patterns predict V=70 but the actual sentence V=140,
that 70-point residual flags an idiom candidate (or sarcasm, or missing rule).

Processes 99.5K EmpatheticDialogues sentences to find systematic deviations
between n-gram-predicted emotion and actual engine-scored emotion.
"""

import json
import sys
import re
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Project paths
PROJECT_ROOT = '/var/home/deucebucket/ai-drive/clanker-lang'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'demo'))

from demo.pendulum import IDIOMS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, 'training/data/empathetic_dialogues.jsonl')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'training/data/discovered_idioms.jsonl')
MIN_NGRAM_COUNT = 5          # minimum occurrences to form a pattern
DEVIATION_THRESHOLD = 20     # V-point deviation to flag as candidate
TOP_N = 50                   # top candidates to print
MIN_MEAN_DEVIATION = 15      # minimum mean deviation to show in top results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into words."""
    text = text.lower()
    text = re.sub(r"[^\w'\-]", ' ', text)  # keep apostrophes and hyphens
    return [w for w in text.split() if w]


def extract_ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract all n-grams from a word list."""
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def get_existing_idiom_keys() -> set:
    """Get all existing idiom phrases as frozensets of lowercased words."""
    keys = set()
    for idiom_tuple in IDIOMS.keys():
        keys.add(tuple(w.lower() for w in idiom_tuple))
    return keys


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data() -> List[dict]:
    """Load EmpatheticDialogues JSONL."""
    print(f"Loading data from {DATA_PATH}...")
    data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Need english text and vadug
            if 'english' in rec and 'vadug' in rec and len(rec['vadug']) >= 1:
                data.append(rec)
    print(f"  Loaded {len(data):,} sentences")
    return data


# ---------------------------------------------------------------------------
# Step 2: Build n-gram emotional patterns (expected V per n-gram)
# ---------------------------------------------------------------------------
@dataclass
class NgramStats:
    v_sum: float = 0.0
    count: int = 0
    sentences: list = field(default_factory=list)  # indices

    @property
    def mean_v(self) -> float:
        return self.v_sum / self.count if self.count > 0 else 128.0


def build_ngram_patterns(data: List[dict]) -> Dict[Tuple[str, ...], NgramStats]:
    """
    For every bigram and trigram that appears 5+ times,
    compute the average V of sentences containing it.
    """
    print("Building bigram/trigram emotional patterns...")
    stats: Dict[Tuple[str, ...], NgramStats] = defaultdict(NgramStats)

    for idx, rec in enumerate(data):
        v = rec['vadug'][0]  # Valence is first element
        words = tokenize(rec['english'])

        # Track which n-grams we've already counted for this sentence
        # (avoid double-counting if same bigram appears twice in one sentence)
        seen = set()

        for n in (2, 3):
            for ngram in extract_ngrams(words, n):
                if ngram not in seen:
                    seen.add(ngram)
                    s = stats[ngram]
                    s.v_sum += v
                    s.count += 1
                    s.sentences.append(idx)

    # Filter to MIN_NGRAM_COUNT
    filtered = {k: v for k, v in stats.items() if v.count >= MIN_NGRAM_COUNT}
    n_bigrams = sum(1 for k in filtered if len(k) == 2)
    n_trigrams = sum(1 for k in filtered if len(k) == 3)
    print(f"  {n_bigrams:,} bigrams and {n_trigrams:,} trigrams with {MIN_NGRAM_COUNT}+ occurrences")
    return filtered


# ---------------------------------------------------------------------------
# Step 3: Find deviations — per-sentence residuals
# ---------------------------------------------------------------------------
@dataclass
class DeviationRecord:
    ngram: Tuple[str, ...]
    sentence_idx: int
    predicted_v: float  # from n-gram average
    actual_v: float     # from vadug
    deviation: float    # actual - predicted


def find_deviations(
    data: List[dict],
    patterns: Dict[Tuple[str, ...], NgramStats]
) -> List[DeviationRecord]:
    """
    For each sentence, find n-grams whose average V deviates from
    this sentence's actual V by more than DEVIATION_THRESHOLD.
    """
    print(f"Finding deviations (threshold: {DEVIATION_THRESHOLD} V-points)...")
    deviations = []

    for idx, rec in enumerate(data):
        actual_v = rec['vadug'][0]
        words = tokenize(rec['english'])
        seen = set()

        for n in (2, 3):
            for ngram in extract_ngrams(words, n):
                if ngram in seen:
                    continue
                seen.add(ngram)

                if ngram not in patterns:
                    continue

                predicted_v = patterns[ngram].mean_v
                dev = actual_v - predicted_v

                if abs(dev) > DEVIATION_THRESHOLD:
                    deviations.append(DeviationRecord(
                        ngram=ngram,
                        sentence_idx=idx,
                        predicted_v=predicted_v,
                        actual_v=actual_v,
                        deviation=dev,
                    ))

    print(f"  Found {len(deviations):,} individual deviations")
    return deviations


# ---------------------------------------------------------------------------
# Step 4: Cluster deviations by n-gram
# ---------------------------------------------------------------------------
@dataclass
class IdiomCandidate:
    phrase: str
    ngram: Tuple[str, ...]
    expected_v: float
    actual_v: float        # mean actual V of deviating sentences
    deviation: float       # mean deviation
    count: int             # number of deviating sentences
    consistency: float     # fraction of deviations in same direction
    score: float           # |deviation| * count (ranking metric)
    in_engine: bool
    example_sentences: List[str] = field(default_factory=list)


def cluster_deviations(
    data: List[dict],
    deviations: List[DeviationRecord],
    existing_idioms: set,
) -> List[IdiomCandidate]:
    """
    Group deviations by n-gram. If the same phrase consistently deviates
    in the same direction, it's a confirmed idiom candidate.
    """
    print("Clustering deviations by n-gram...")

    # Group by ngram
    by_ngram: Dict[Tuple[str, ...], List[DeviationRecord]] = defaultdict(list)
    for dr in deviations:
        by_ngram[dr.ngram].append(dr)

    candidates = []
    for ngram, records in by_ngram.items():
        if len(records) < 3:  # need at least 3 deviating occurrences
            continue

        devs = [r.deviation for r in records]
        mean_dev = sum(devs) / len(devs)
        actual_vs = [r.actual_v for r in records]
        mean_actual = sum(actual_vs) / len(actual_vs)
        predicted_v = records[0].predicted_v  # same for all (it's the ngram average)

        # Consistency: what fraction deviate in the same direction as the mean?
        if mean_dev > 0:
            same_dir = sum(1 for d in devs if d > 0)
        else:
            same_dir = sum(1 for d in devs if d < 0)
        consistency = same_dir / len(devs)

        # Only keep if consistent (>60% same direction)
        if consistency < 0.6:
            continue

        phrase = ' '.join(ngram)
        in_engine = ngram in existing_idioms

        # Collect example sentences
        examples = []
        for r in records[:5]:
            examples.append(data[r.sentence_idx]['english'])

        # Score: deviation^2 * sqrt(count) — rewards large deviations much more
        score = (mean_dev ** 2) * (len(records) ** 0.5)

        candidates.append(IdiomCandidate(
            phrase=phrase,
            ngram=ngram,
            expected_v=round(predicted_v, 1),
            actual_v=round(mean_actual, 1),
            deviation=round(mean_dev, 1),
            count=len(records),
            consistency=round(consistency, 3),
            score=round(score, 1),
            in_engine=in_engine,
            example_sentences=examples,
        ))

    # Sort by score (|deviation| * count)
    candidates.sort(key=lambda c: c.score, reverse=True)
    print(f"  {len(candidates):,} idiom candidates after clustering")
    return candidates


# ---------------------------------------------------------------------------
# Step 5: Suggest force tuples
# ---------------------------------------------------------------------------
def suggest_force_tuple(candidate: IdiomCandidate) -> str:
    """
    Suggest a VADUG force tuple based on the deviation direction and magnitude.
    This is a rough heuristic — the V delta comes from the residual.
    """
    dv = round(candidate.deviation * 0.5)  # scale down for force delta
    # Arousal: large deviations imply something emotionally significant
    da = round(abs(candidate.deviation) * 0.15)
    # Dominance: positive deviations often correlate with agency
    dd = round(candidate.deviation * 0.2)
    # Urgency: larger deviations = more salient
    du = round(abs(candidate.deviation) * 0.1)
    # Gravity: follows valence direction
    dg = round(candidate.deviation * 0.25)

    return f"({dv:+d}, {da:+d}, {dd:+d}, {du:+d}, {dg:+d})"


# ---------------------------------------------------------------------------
# Step 6: Output
# ---------------------------------------------------------------------------
def print_one(i: int, c: IdiomCandidate):
    """Print a single candidate."""
    status = "KNOWN" if c.in_engine else "NEW"
    force = suggest_force_tuple(c)
    direction = "HIGHER" if c.deviation > 0 else "LOWER"

    print(f"  {i:2d}. [{status:5s}] \"{c.phrase}\"")
    print(f"      Expected V: {c.expected_v:5.1f}  |  Actual V: {c.actual_v:5.1f}  |  "
          f"Deviation: {c.deviation:+.1f} ({direction})")
    print(f"      Occurrences: {c.count}  |  Consistency: {c.consistency:.0%}  |  "
          f"Score: {c.score:.0f}")
    print(f"      Suggested force: {force}")
    if c.example_sentences:
        ex = c.example_sentences[0][:100]
        if len(c.example_sentences[0]) > 100:
            ex += "..."
        print(f"      Example: \"{ex}\"")
    print()


def print_results(candidates: List[IdiomCandidate]):
    """Print top N candidates, split into real idiom discoveries vs noise."""

    # Filter: only show candidates with meaningful deviation
    strong = [c for c in candidates if abs(c.deviation) >= MIN_MEAN_DEVIATION]
    known_in_top = [c for c in candidates if c.in_engine]

    # --- Section 1: Known idioms found ---
    if known_in_top:
        print(f"\n{'='*90}")
        print(f"  ENGINE IDIOMS CONFIRMED BY DATA (already in pendulum.py)")
        print(f"{'='*90}\n")
        for i, c in enumerate(known_in_top, 1):
            print_one(i, c)

    # --- Section 2: Strong new discoveries ---
    new_strong = [c for c in strong if not c.in_engine]
    print(f"\n{'='*90}")
    print(f"  TOP {TOP_N} NEW IDIOM CANDIDATES (deviation >= {MIN_MEAN_DEVIATION}, scored by dev^2 * sqrt(count))")
    print(f"{'='*90}\n")

    new_count = 0
    for i, c in enumerate(new_strong[:TOP_N], 1):
        print_one(i, c)
        new_count += 1

    print(f"  Showing {new_count} of {len(new_strong)} strong candidates "
          f"(deviation >= {MIN_MEAN_DEVIATION})")

    # --- Section 3: Breakdown by deviation direction ---
    negative = [c for c in new_strong if c.deviation < 0]
    positive = [c for c in new_strong if c.deviation > 0]

    print(f"\n{'='*90}")
    print(f"  TOP 20 NEGATIVE DEVIATIONS (phrases sadder/angrier than patterns predict)")
    print(f"{'='*90}\n")
    neg_sorted = sorted(negative, key=lambda c: c.deviation)
    for i, c in enumerate(neg_sorted[:20], 1):
        print_one(i, c)

    print(f"\n{'='*90}")
    print(f"  TOP 20 POSITIVE DEVIATIONS (phrases happier/calmer than patterns predict)")
    print(f"{'='*90}\n")
    pos_sorted = sorted(positive, key=lambda c: -c.deviation)
    for i, c in enumerate(pos_sorted[:20], 1):
        print_one(i, c)


def save_results(candidates: List[IdiomCandidate]):
    """Save all candidates to JSONL."""
    print(f"Saving {len(candidates):,} candidates to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        for c in candidates:
            record = {
                "phrase": c.phrase,
                "ngram_len": len(c.ngram),
                "expected_v": c.expected_v,
                "actual_v": c.actual_v,
                "deviation": c.deviation,
                "count": c.count,
                "consistency": c.consistency,
                "score": c.score,
                "in_engine": c.in_engine,
                "suggested_force": suggest_force_tuple(c),
                "examples": c.example_sentences[:3],
            }
            f.write(json.dumps(record) + '\n')
    print(f"  Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*90)
    print("  IDIOM DISCOVERY SYSTEM — Mathematical Residual Analysis")
    print("  Finding idioms from the GAP between predicted and actual emotion")
    print("="*90 + "\n")

    # Step 1: Load
    data = load_data()

    # Step 2: Build patterns
    patterns = build_ngram_patterns(data)

    # Step 3: Find deviations
    deviations = find_deviations(data, patterns)

    # Step 4: Cluster
    existing_idioms = get_existing_idiom_keys()
    print(f"  Loaded {len(existing_idioms)} existing idioms from engine")
    candidates = cluster_deviations(data, deviations, existing_idioms)

    # Step 5 & 6: Output
    print_results(candidates)
    save_results(candidates)

    # Summary stats
    total_new = sum(1 for c in candidates if not c.in_engine)
    total_known = sum(1 for c in candidates if c.in_engine)
    high_confidence = sum(1 for c in candidates if c.consistency >= 0.8 and c.count >= 10)
    print(f"\n  FINAL STATS:")
    print(f"    Total candidates:     {len(candidates):,}")
    print(f"    New discoveries:      {total_new:,}")
    print(f"    Already in engine:    {total_known:,}")
    print(f"    High confidence:      {high_confidence:,} (consistency >= 80%, count >= 10)")
    print(f"    Data processed:       {len(data):,} sentences")
    print()


if __name__ == '__main__':
    main()
