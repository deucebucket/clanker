#!/usr/bin/env python3
"""Generate zone-labeled training data.

Reads all source JSONL files, runs each text through PendulumV2 to get
engine VADUG, then classifies into emotional zones via ZoneClassifier.

Output: training/data/zone_labeled.jsonl
Format: {"english": text, "vadug": [v,a,d,u,g], "zone": "GRIEF", "source": original_source}
"""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from demo.pendulum_v2 import PendulumV2
from demo.zones import ZoneClassifier

DATA_DIR = os.path.join(PROJECT_ROOT, "training", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "zone_labeled.jsonl")

# Source files to process
SOURCE_FILES = [
    ("phase1.jsonl", "rosetta_stone"),
    ("negation_pairs.jsonl", "negation_pairs"),
    ("sarcasm_calibration.jsonl", "sarcasm_calibration"),
    ("negation_calibration.jsonl", "negation_calibration"),
]


def main():
    engine = PendulumV2()
    classifier = ZoneClassifier()

    total = 0
    zone_counts = {}

    with open(OUTPUT_PATH, "w") as out:
        for filename, default_source in SOURCE_FILES:
            filepath = os.path.join(DATA_DIR, filename)
            if not os.path.exists(filepath):
                print(f"  SKIP {filename} — not found")
                continue

            file_count = 0
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)

                    text = data.get("english", data.get("text", ""))
                    if not text:
                        continue

                    # Run through engine
                    vadug, _ = engine.process_text(text)
                    vadug_list = [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g]

                    # Classify zone
                    result = classifier.classify(vadug)
                    zone = result.zone

                    # Use original source if available
                    source = data.get("source", default_source)

                    record = {
                        "english": text,
                        "vadug": vadug_list,
                        "zone": zone,
                        "source": source,
                    }
                    out.write(json.dumps(record) + "\n")

                    zone_counts[zone] = zone_counts.get(zone, 0) + 1
                    file_count += 1
                    total += 1

            print(f"  {filename}: {file_count} examples")

    print(f"\nTotal: {total} zone-labeled examples → {OUTPUT_PATH}")
    print(f"\nZone distribution:")
    for zone, count in sorted(zone_counts.items(), key=lambda x: -x[1]):
        print(f"  {zone:15s}: {count:6d} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    main()
