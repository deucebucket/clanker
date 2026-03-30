#!/usr/bin/env python3
"""Generate engine traces for ALL available training sentences.

Loads from every JSONL source file, deduplicates, runs through PendulumV2,
and saves traced examples to training/data/all_traces.jsonl.
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from demo.pendulum_v2 import PendulumV2

engine = PendulumV2()

# Source files: (filename, field_name_options)
SOURCES = [
    ("phase1.jsonl", ["english"]),
    ("phase1_expanded.jsonl", ["english"]),
    ("emobank_significant.jsonl", ["english", "text"]),
    ("emobank_vadug.jsonl", ["english", "text"]),
    ("balance_neutral.jsonl", ["english"]),
    ("balance_positive.jsonl", ["english"]),
    ("balance_pos_high.jsonl", ["english"]),
    ("balance_pos_mild.jsonl", ["english"]),
    ("error_corrections.jsonl", ["english", "text"]),
    ("engine_errors.jsonl", ["text"]),
]


def load_sentences():
    """Load and deduplicate sentences from all source files."""
    seen = set()
    sentences = []
    source_map = {}  # sentence -> source filename

    data_dir = os.path.join(PROJECT_ROOT, "training", "data")

    for filename, fields in SOURCES:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP (not found): {filename}")
            continue

        count = 0
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Try each field name option
                text = None
                for field in fields:
                    if field in d and d[field]:
                        text = d[field]
                        break

                if not text or not isinstance(text, str):
                    continue

                text = text.strip()
                if not text or text in seen:
                    continue

                seen.add(text)
                sentences.append(text)
                source_map[text] = filename
                count += 1

        print(f"  {filename}: {count} unique sentences")

    return sentences, source_map


def main():
    print("=" * 60)
    print("Generating ALL traces from training data sources")
    print("=" * 60)

    print("\nLoading sentences...")
    sentences, source_map = load_sentences()
    print(f"\nTotal unique sentences: {len(sentences)}")

    print("\nRunning engine traces...")
    start = time.time()
    output = []
    errors = 0

    for i, sent in enumerate(sentences):
        try:
            vadug, trace = engine.process_text(sent)
            output.append({
                "english": sent,
                "vadug": [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g],
                "trace": trace,
                "source": source_map.get(sent, "unknown"),
            })
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR on '{sent[:60]}...': {e}")

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(sentences)} ({rate:.0f} sent/s)...")

    elapsed = time.time() - start
    print(f"\nGenerated {len(output)} traced examples in {elapsed:.1f}s")
    if errors:
        print(f"  ({errors} errors skipped)")

    # Save
    outpath = os.path.join(PROJECT_ROOT, "training", "data", "all_traces.jsonl")
    with open(outpath, "w") as f:
        for d in output:
            f.write(json.dumps(d) + "\n")

    size_mb = os.path.getsize(outpath) / 1e6
    print(f"\nSaved to {outpath}")
    print(f"  Lines: {len(output)}")
    print(f"  Size:  {size_mb:.1f}MB")


if __name__ == "__main__":
    main()
