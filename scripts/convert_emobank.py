#!/usr/bin/env python3
"""Convert EmoBank CSV to Clanker-Lang VADUG training data.

EmoBank has human-rated V, A, D scores (1.0-5.0 scale) for 10,063 sentences.
We convert their V/A/D to 0-255 scale, then generate U and G from our engine.
Cross-check: compare human V vs engine V, flag disagreements >= 40 points.
"""

import csv
import json
import sys
import os

# Ensure imports work
sys.path.insert(0, '/var/home/deucebucket/ai-drive/clanker-lang')
sys.path.insert(0, '/var/home/deucebucket/ai-drive/clanker-lang/demo')

from demo.pendulum import SequentialPendulum


def emobank_to_255(score):
    """Convert EmoBank 1.0-5.0 scale to 0-255."""
    return int((score - 1.0) / 4.0 * 255)


def main():
    csv_path = '/var/home/deucebucket/ai-drive/clanker-lang/benchmarks/emobank.csv'
    out_path = '/var/home/deucebucket/ai-drive/clanker-lang/training/data/emobank_vadug.jsonl'

    # Stats
    total = 0
    skipped = 0
    written = 0
    agree_count = 0
    disagree_count = 0
    v_diffs = []

    # V-axis distribution bands (0-255 mapped to 7 bands)
    band_labels = [
        "0-36 (very negative)",
        "37-72 (negative)",
        "73-109 (slightly negative)",
        "110-145 (neutral)",
        "146-182 (slightly positive)",
        "183-218 (positive)",
        "219-255 (very positive)",
    ]
    band_counts = [0] * 7

    records = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from EmoBank CSV")
    print(f"Columns: {rows[0].keys() if rows else 'EMPTY'}")
    print()

    for i, row in enumerate(rows):
        total += 1
        text = row.get('text', '').strip()

        # Skip empty or too short
        if len(text) < 3:
            skipped += 1
            continue

        # Parse human V, A, D scores
        try:
            human_v_raw = float(row['V'])
            human_a_raw = float(row['A'])
            human_d_raw = float(row['D'])
        except (ValueError, KeyError):
            skipped += 1
            continue

        # Convert to 0-255
        human_v = emobank_to_255(human_v_raw)
        human_a = emobank_to_255(human_a_raw)
        human_d = emobank_to_255(human_d_raw)

        # Run through our engine for U and G (and to cross-check V)
        engine = SequentialPendulum()
        vadug, trace = engine.process_text(text)
        engine_v = vadug.v
        engine_u = vadug.u
        engine_g = vadug.g

        # Cross-check V
        v_diff = abs(human_v - engine_v)
        v_diffs.append(v_diff)

        if v_diff < 40:
            agree_count += 1
            needs_review = False
        else:
            disagree_count += 1
            needs_review = True

        # Determine V band
        band_idx = min(human_v // 37, 6)
        band_counts[band_idx] += 1

        # Build features string
        features = f"human_vad, engine_ug, v_diff={v_diff}"
        if needs_review:
            features += ", needs_review"

        record = {
            "english": text,
            "vadug": [human_v, human_a, human_d, engine_u, engine_g],
            "source": "emobank_human",
            "features": features,
        }
        records.append(record)
        written += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(rows)} sentences... ({written} written, {skipped} skipped)")

    # Write JSONL
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Print summary
    print()
    print("=" * 60)
    print("EMOBANK -> VADUG CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total rows:    {total}")
    print(f"Skipped:       {skipped}")
    print(f"Written:       {written}")
    print(f"Output:        {out_path}")
    print()

    # Agreement stats
    print("--- V-AXIS AGREEMENT (human vs engine) ---")
    print(f"Agree (diff < 40):    {agree_count} ({100*agree_count/written:.1f}%)")
    print(f"Disagree (diff >= 40): {disagree_count} ({100*disagree_count/written:.1f}%)")
    if v_diffs:
        avg_diff = sum(v_diffs) / len(v_diffs)
        print(f"Mean V diff:          {avg_diff:.1f}")
        print(f"Max V diff:           {max(v_diffs)}")
        print(f"Min V diff:           {min(v_diffs)}")
    print()

    # V-axis distribution
    print("--- V-AXIS DISTRIBUTION (human scores, 0-255) ---")
    for label, count in zip(band_labels, band_counts):
        bar = "#" * (count // 20)
        print(f"  {label:30s}: {count:5d}  {bar}")
    print()

    # Needs-review sample
    review_samples = [r for r in records if "needs_review" in r["features"]]
    if review_samples:
        print(f"--- SAMPLE DISAGREEMENTS ({min(5, len(review_samples))} of {len(review_samples)}) ---")
        for r in review_samples[:5]:
            print(f"  [{r['vadug'][0]:3d}] {r['english'][:80]}")
            print(f"        features: {r['features']}")
        print()


if __name__ == '__main__':
    main()
