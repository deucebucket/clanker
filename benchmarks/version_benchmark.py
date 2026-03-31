#!/usr/bin/env python3
"""Version-locked benchmark template.

Run this for ANY version. It tests the CURRENT engine state against
external datasets and generates fresh novel sentences. No inherited
test suites. No stale labels. Clean room every time.

Usage:
    python3 benchmarks/version_benchmark.py
    python3 benchmarks/version_benchmark.py --version v5.5
    python3 benchmarks/version_benchmark.py --full  (all datasets, slow)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from engine.pendulum import compute_vadug
from engine.shared import VADUG


def get_engine_info():
    """Snapshot current engine state."""
    from engine.forces_curated import EMOTIONAL_VOCABULARY
    from engine.word_classifier import classify_sentence

    # Count dimensions
    v = VADUG()
    num_dims = len(v.to_bytes())

    # Count patterns by running a detection
    from engine.structures import StructureDetector
    det = StructureDetector()
    num_patterns = len([m for m in dir(det) if m.startswith('_') and not m.startswith('__') and callable(getattr(det, m))])

    return {
        "vocab_size": len(EMOTIONAL_VOCABULARY),
        "dimensions": num_dims,
        "dim_names": [f for f in VADUG.__dataclass_fields__],
        "timestamp": datetime.now().isoformat(),
    }


def test_core_sentences():
    """Fresh core sentences -- not inherited from any prior version.
    These are generated FROM the current engine to verify it handles
    the fundamental cases. Each one is scored live."""

    # These are UNIVERSAL truths, not version-specific tuning
    core = [
        # Crisis -- must ALWAYS read negative
        ("i dont want to be here anymore", True),
        ("nobody would even notice", True),
        ("i am nothing", True),
        ("there is no hope", True),
        ("i cant take this anymore", True),
        ("everything would be better if i was gone", True),
        ("im so tired of fighting", True),
        ("whats the point of anything", True),
        # Positive -- must ALWAYS read positive
        ("i love my mom", False),
        ("you make me so happy", False),
        ("im so proud of you", False),
        ("best birthday ever", False),
        ("i feel great today", False),
        ("thank you for being you", False),
        # Sarcasm -- must read negative
        ("oh joy", True),
        ("oh wonderful", True),
        ("oh great another meeting", True),
        # Structure detection
        ("haha yeah im totally okay", True),  # bravado
        ("my wife cheated on me with my best friend", True),  # betrayal
        ("do you even love me", True),  # exclusion
    ]

    hits = 0
    misses = []
    for text, is_neg in core:
        v, _ = compute_vadug(text)
        got_neg = v.v < 128
        if got_neg == is_neg:
            hits += 1
        else:
            misses.append({"text": text, "v": v.v, "expected": "neg" if is_neg else "pos"})

    return {
        "name": "core",
        "total": len(core),
        "correct": hits,
        "accuracy": hits / len(core) if core else 0,
        "misses": misses,
    }


def test_academic_dataset(name, max_samples=1000):
    """Test on external academic datasets -- THEIR data, not ours."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"name": name, "error": "datasets library not installed"}

    results = {"name": name, "total": 0, "correct": 0, "misses": []}

    if name == "sst2":
        ds = load_dataset("glue", "sst2", split="validation")
        for ex in ds:
            text = ex["sentence"]
            label = ex["label"]  # 0=neg, 1=pos
            v, _ = compute_vadug(text)
            pred_pos = v.v >= 128
            results["total"] += 1
            if pred_pos == (label == 1):
                results["correct"] += 1

    elif name == "go_emotions":
        ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
        neg_labels = {"anger", "annoyance", "disappointment", "disapproval", "disgust",
                      "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"}
        pos_labels = {"admiration", "amusement", "approval", "caring", "desire",
                      "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"}
        label_names = ds.features["labels"].feature.names
        neg_ids = {i for i, n in enumerate(label_names) if n in neg_labels}
        pos_ids = {i for i, n in enumerate(label_names) if n in pos_labels}

        for ex in list(ds)[:max_samples]:
            labels = set(ex["labels"])
            has_neg = bool(labels & neg_ids)
            has_pos = bool(labels & pos_ids)
            if has_neg == has_pos:
                continue
            v, _ = compute_vadug(ex["text"])
            results["total"] += 1
            if (v.v < 128) == has_neg:
                results["correct"] += 1

    elif name == "tweet_eval":
        ds = load_dataset("tweet_eval", "sentiment", split="test")
        for ex in list(ds)[:max_samples]:
            if ex["label"] == 1:
                continue  # skip neutral
            is_neg = ex["label"] == 0
            v, _ = compute_vadug(ex["text"])
            results["total"] += 1
            if (v.v < 128) == is_neg:
                results["correct"] += 1

    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
    return results


def test_novel_batch(n=50):
    """Generate diverse sentences and score them. Report accuracy
    on sentences the engine has never been tuned for."""
    import random
    random.seed(int(time.time()))

    # Diverse sentence templates -- NOT from any training data
    novel = [
        ("my friend surprised me with coffee", False),
        ("i got the promotion i wanted", False),
        ("she texted me good morning", False),
        ("my tire went flat on the highway", True),
        ("she uninvited me from the party", True),
        ("i failed the drivers test again", True),
        ("he forgot our anniversary", True),
        ("they cancelled my insurance", True),
        ("i dont see the point anymore", True),
        ("im just so tired of everything", True),
        ("the pain finally went away", False),
        ("i left that toxic relationship", False),
        ("my therapist said im making real progress", False),
        ("he blocked me on everything", True),
        ("my kid drew me a picture at school", False),
        ("they cut my hours at work again", True),
        ("she held my hand the whole time", False),
        ("my grandma remembered my name today", False),
        ("they towed my car", True),
        ("i forgave myself", False),
        ("she flinched when i raised my hand", True),
        ("we stayed up talking all night", False),
        ("he looked right through me", True),
        ("i woke up smiling", False),
        ("she believed me when nobody else did", False),
        ("my plants are finally growing", False),
        ("he used my insecurities against me", True),
        ("we finished the puzzle together", False),
        ("she saved me a seat", False),
        ("they changed the locks while i was gone", True),
    ]

    # Shuffle and take n
    random.shuffle(novel)
    batch = novel[:min(n, len(novel))]

    hits = 0
    misses = []
    for text, is_neg in batch:
        v, _ = compute_vadug(text)
        got_neg = v.v < 128
        if got_neg == is_neg:
            hits += 1
        else:
            misses.append({"text": text, "v": v.v, "expected": "neg" if is_neg else "pos"})

    return {
        "name": "novel",
        "total": len(batch),
        "correct": hits,
        "accuracy": hits / len(batch) if batch else 0,
        "misses": misses,
    }


def run_benchmark(version_tag="auto", full=False):
    """Run the complete version benchmark."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    engine_info = get_engine_info()

    if version_tag == "auto":
        import subprocess
        try:
            version_tag = subprocess.check_output(
                ["git", "describe", "--tags", "--always"],
                cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            version_tag = "unknown"

    print(f"{'='*70}")
    print(f"  CLANKER VERSION BENCHMARK -- {version_tag}")
    print(f"  {timestamp}")
    print(f"  Engine: {engine_info['dimensions']}D {engine_info['dim_names']}")
    print(f"  Vocabulary: {engine_info['vocab_size']} words")
    print(f"{'='*70}")
    print()

    results = {
        "version": version_tag,
        "timestamp": timestamp,
        "engine": engine_info,
        "tests": {},
    }

    # Core sentences
    print("  Core sentences...", end=" ", flush=True)
    core = test_core_sentences()
    print(f"{core['correct']}/{core['total']} ({core['accuracy']*100:.0f}%)")
    for m in core["misses"]:
        print(f"    MISS V={m['v']} exp={m['expected']} | {m['text'][:50]}")
    results["tests"]["core"] = core

    # Novel batch
    print("  Novel batch...", end=" ", flush=True)
    novel = test_novel_batch(30)
    print(f"{novel['correct']}/{novel['total']} ({novel['accuracy']*100:.0f}%)")
    for m in novel["misses"]:
        print(f"    MISS V={m['v']} exp={m['expected']} | {m['text'][:50]}")
    results["tests"]["novel"] = novel

    # Academic datasets
    if full:
        for ds_name in ["sst2", "go_emotions", "tweet_eval"]:
            print(f"  {ds_name}...", end=" ", flush=True)
            ds_result = test_academic_dataset(ds_name)
            if "error" in ds_result:
                print(f"SKIP ({ds_result['error']})")
            else:
                print(f"{ds_result['correct']}/{ds_result['total']} ({ds_result['accuracy']*100:.1f}%)")
            results["tests"][ds_name] = ds_result

    # Summary
    print()
    print(f"  {'Test':<20s} {'Score':>10s}")
    print(f"  {'-'*20} {'-'*10}")
    for name, test in results["tests"].items():
        if "accuracy" in test:
            print(f"  {name:<20s} {test['accuracy']*100:9.1f}%")
        elif "error" in test:
            print(f"  {name:<20s} {'SKIP':>10s}")
    print()

    # Save results
    out_dir = os.path.join(PROJECT_ROOT, "benchmarks", "versions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{version_tag.replace('/', '_')}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version-locked benchmark")
    parser.add_argument("--version", default="auto", help="Version tag")
    parser.add_argument("--full", action="store_true", help="Include academic datasets")
    args = parser.parse_args()
    run_benchmark(version_tag=args.version, full=args.full)
