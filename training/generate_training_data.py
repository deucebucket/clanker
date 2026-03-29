#!/usr/bin/env python3
"""Generate training data from the Rosetta Stone + engine-processed corpora.

Phase 1: Rosetta Stone → perfect calibration data
Phase 2: Books + academic → large-scale engine-labeled data
Output: training/data/phase1.jsonl, training/data/phase2.jsonl
"""

import sys
import os
import re
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))


def generate_phase1():
    """Phase 1: Rosetta Stone — hand-scored perfect calibration data."""
    from benchmarks.rosetta_stone import ROSETTA_STONE

    os.makedirs("training/data", exist_ok=True)

    examples = []
    for text, tv, ta, td, tu, tg, features in ROSETTA_STONE:
        example = {
            "english": text,
            "vadug": [tv, ta, td, tu, tg],
            "source": "rosetta_stone",
            "features": features,
        }
        examples.append(example)

    with open("training/data/phase1.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Phase 1: {len(examples)} Rosetta Stone examples → training/data/phase1.jsonl")
    return len(examples)


def generate_phase2():
    """Phase 2: Engine-processed corpora — large-scale training data."""
    from demo.simulator import SequentialPendulum
    from demo.intent import IntentDetector

    intent = IntentDetector()
    os.makedirs("training/data", exist_ok=True)

    # Check if tuning data exists (already processed)
    tuning_path = "benchmarks/tuning_data.jsonl"
    if os.path.exists(tuning_path):
        examples = []
        with open(tuning_path) as f:
            for line in f:
                row = json.loads(line)
                example = {
                    "english": row["text"],
                    "vadug": [row["v"], row["a"], row["d"], row["u"], row["g"]],
                    "source": row.get("source", "unknown"),
                    "vader_label": row.get("vader_label", "unknown"),
                }
                examples.append(example)

        with open("training/data/phase2.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        print(f"Phase 2: {len(examples)} engine-processed examples → training/data/phase2.jsonl")
        return len(examples)
    else:
        print("Phase 2: No tuning data found. Run benchmarks/find_cracks.py first.")
        return 0


if __name__ == "__main__":
    print("Clanker Training Data Generator")
    print("=" * 50)
    n1 = generate_phase1()
    n2 = generate_phase2()
    print(f"\nTotal: {n1 + n2} training examples")
    print(f"Phase 1 (Rosetta Stone): {n1} perfect examples")
    print(f"Phase 2 (engine-labeled): {n2} examples")
