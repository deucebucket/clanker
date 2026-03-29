#!/usr/bin/env python3
"""
Convert EmpatheticDialogues dataset to Clanker-Lang A+B=C triplet training data.

Downloads from Facebook Research, groups by conversation, scores through
the Clanker engine, and outputs JSONL with VADUG coordinates.
"""

import csv
import json
import os
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict

# Add clanker-lang to path
ROOT = '/var/home/deucebucket/ai-drive/clanker-lang'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'demo'))

from demo.pendulum import SequentialPendulum

# --- Config ---
DATA_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
CACHE_DIR = "/tmp/empathetic_dialogues"
CSV_DIR = os.path.join(CACHE_DIR, "empatheticdialogues")
OUTPUT_FILE = os.path.join(ROOT, "training/data/empathetic_dialogues.jsonl")

# --- Emotion -> expected VADUG direction (for validation) ---
# V: 0=negative, 128=neutral, 255=positive
EMOTION_EXPECTATIONS = {
    # Positive emotions - V should be > 128
    "joyful": {"v_high": True},
    "excited": {"v_high": True},
    "proud": {"v_high": True},
    "grateful": {"v_high": True},
    "confident": {"v_high": True},
    "hopeful": {"v_high": True},
    "impressed": {"v_high": True},
    "content": {"v_high": True},
    "caring": {"v_high": True},
    "trusting": {"v_high": True},
    "faithful": {"v_high": True},
    "prepared": {"v_high": True},
    "anticipating": {"v_high": True},
    # Negative emotions - V should be < 128
    "angry": {"v_high": False},
    "furious": {"v_high": False},
    "sad": {"v_high": False},
    "afraid": {"v_high": False},
    "terrified": {"v_high": False},
    "anxious": {"v_high": False},
    "disgusted": {"v_high": False},
    "disappointed": {"v_high": False},
    "devastated": {"v_high": False},
    "embarrassed": {"v_high": False},
    "guilty": {"v_high": False},
    "ashamed": {"v_high": False},
    "jealous": {"v_high": False},
    "annoyed": {"v_high": False},
    "lonely": {"v_high": False},
    "apprehensive": {"v_high": False},
    # Ambiguous - could go either way
    "surprised": {"v_high": None},
    "sentimental": {"v_high": None},
    "nostalgic": {"v_high": None},
}


def download_data():
    """Download and extract the dataset if not cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tarpath = os.path.join(CACHE_DIR, "data.tar.gz")

    if not os.path.exists(os.path.join(CSV_DIR, "train.csv")):
        if not os.path.exists(tarpath):
            print("Downloading EmpatheticDialogues...", flush=True)
            urllib.request.urlretrieve(DATA_URL, tarpath)
            print("Downloaded.", flush=True)

        print("Extracting...", flush=True)
        with tarfile.open(tarpath, "r:gz") as tar:
            tar.extractall(CACHE_DIR)
        print("Extracted.", flush=True)
    else:
        print("Data already cached.", flush=True)


def clean_text(text):
    """Clean up the _comma_ and other encoding artifacts."""
    text = text.replace("_comma_", ",")
    text = text.strip()
    return text


def load_conversations(csv_path):
    """Load CSV and group utterances by conversation."""
    conversations = defaultdict(lambda: {"emotion": None, "prompt": None, "turns": []})

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row["conv_id"]
            conv = conversations[conv_id]
            conv["emotion"] = row["context"]
            conv["prompt"] = clean_text(row["prompt"])
            conv["turns"].append({
                "idx": int(row["utterance_idx"]),
                "speaker_idx": int(row["speaker_idx"]),
                "utterance": clean_text(row["utterance"]),
            })

    # Sort turns within each conversation
    for conv in conversations.values():
        conv["turns"].sort(key=lambda t: t["idx"])

    return conversations


def vadug_to_list(vadug):
    """Convert VADUG object to [V, A, D, U, G] list."""
    return [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g]


def v_band(v):
    """Map V value (0-255) to 7 bands for stats."""
    if v < 37:
        return "0-36 (very negative)"
    elif v < 73:
        return "37-72 (negative)"
    elif v < 110:
        return "73-109 (slightly negative)"
    elif v < 146:
        return "110-145 (neutral)"
    elif v < 183:
        return "146-182 (slightly positive)"
    elif v < 219:
        return "183-218 (positive)"
    else:
        return "219-255 (very positive)"


def main():
    download_data()

    # Load all splits
    all_examples = []
    flagged_count = 0
    total_scored = 0
    emotion_counts = Counter()
    v_band_counts = Counter()

    for split_name, csv_file in [("train", "train.csv"), ("valid", "valid.csv"), ("test", "test.csv")]:
        csv_path = os.path.join(CSV_DIR, csv_file)
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping.", flush=True)
            continue

        print(f"\nProcessing {split_name}...", flush=True)
        conversations = load_conversations(csv_path)
        print(f"  {len(conversations)} conversations loaded", flush=True)

        conv_count = 0
        for conv_id, conv in conversations.items():
            emotion = conv["emotion"]
            emotion_counts[emotion] += 1

            # Separate speaker (odd speaker_idx) and listener (even speaker_idx) turns
            # In this dataset, the first speaker tells the story, the second responds
            # speaker_idx alternates but the FIRST turn's speaker is the "storyteller"
            turns = conv["turns"]
            if not turns:
                continue

            first_speaker = turns[0]["speaker_idx"]

            for turn in turns:
                text = turn["utterance"]
                if not text or len(text.strip()) < 3:
                    continue

                # Fresh engine per utterance — history accumulates and slows down
                engine = SequentialPendulum()

                # Score through engine
                try:
                    vadug, _ = engine.process_text(text)
                except Exception as e:
                    continue

                vadug_list = vadug_to_list(vadug)
                total_scored += 1

                # Determine role
                is_speaker = (turn["speaker_idx"] == first_speaker)
                role = "speaker" if is_speaker else "listener"
                source_type = "stimulus" if is_speaker else "response"

                # Validate against emotion label
                flagged = False
                expected = EMOTION_EXPECTATIONS.get(emotion, {})
                v_expected = expected.get("v_high")
                if v_expected is not None:
                    v_val = vadug.v
                    # Only flag strong disagreements (engine says clearly opposite)
                    if v_expected and v_val < 100:
                        flagged = True
                    elif not v_expected and v_val > 156:
                        flagged = True

                if flagged:
                    flagged_count += 1

                # Track V distribution
                v_band_counts[v_band(vadug.v)] += 1

                # Build training example
                example = {
                    "english": text,
                    "vadug": vadug_list,
                    "source": f"empathetic_dialogues_{source_type}",
                    "features": f"emotion:{emotion}, role:{role}",
                    "split": split_name,
                    "conv_id": conv_id,
                }
                if flagged:
                    example["flag"] = "engine_label_disagreement"

                all_examples.append(example)

            conv_count += 1
            if conv_count % 2000 == 0:
                print(f"  Processed {conv_count}/{len(conversations)} conversations...", flush=True)

        print(f"  {split_name} complete: {conv_count} conversations", flush=True)

    # Write output
    print(f"\nWriting {len(all_examples)} examples to {OUTPUT_FILE}...", flush=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")
    print("Done.", flush=True)

    # --- Stats ---
    print("\n" + "=" * 60, flush=True)
    print("STATS", flush=True)
    print("=" * 60, flush=True)
    print(f"Total examples: {len(all_examples)}", flush=True)
    print(f"Total scored through engine: {total_scored}", flush=True)
    print(f"Flagged (engine vs label disagreement): {flagged_count} ({100*flagged_count/max(1,total_scored):.1f}%)", flush=True)

    stimulus_count = sum(1 for e in all_examples if "stimulus" in e["source"])
    response_count = sum(1 for e in all_examples if "response" in e["source"])
    print(f"Stimulus examples: {stimulus_count}", flush=True)
    print(f"Response examples: {response_count}", flush=True)

    print(f"\nEmotion distribution ({len(emotion_counts)} emotions):", flush=True)
    for emo, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"  {emo:20s}: {count:6d}", flush=True)

    print(f"\nValence (V) distribution across 7 bands:", flush=True)
    for band in sorted(v_band_counts.keys()):
        count = v_band_counts[band]
        pct = 100 * count / max(1, total_scored)
        bar = "#" * int(pct)
        print(f"  {band:35s}: {count:6d} ({pct:5.1f}%) {bar}", flush=True)


if __name__ == "__main__":
    main()
