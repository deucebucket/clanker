#!/usr/bin/env python3
"""Layer Ablation Test: disable each layer one at a time and measure accuracy impact.

Runs a subset of GoEmotions through the Clanker pipeline with each layer toggled
off individually, reporting which layers help, hurt, or are neutral.

Also runs key combinations to find the optimal layer stack.
"""

import re
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.pipeline_config import PipelineConfig
from demo.simulator import SequentialPendulum, WORD_FORCES
from demo.fuzzy import fuzzy_match
from demo.intent import IntentDetector
from demo.nonsense import NonsenseDetector

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets")
    sys.exit(1)

# GoEmotions label mapping (id -> (emotion, sentiment))
GOEMO = {
    0: ("admiration", "positive"), 1: ("amusement", "positive"),
    2: ("anger", "negative"), 3: ("annoyance", "negative"),
    4: ("approval", "positive"), 5: ("caring", "positive"),
    6: ("confusion", "neutral"), 7: ("curiosity", "neutral"),
    8: ("desire", "positive"), 9: ("disappointment", "negative"),
    10: ("disapproval", "negative"), 11: ("disgust", "negative"),
    12: ("embarrassment", "negative"), 13: ("excitement", "positive"),
    14: ("fear", "negative"), 15: ("gratitude", "positive"),
    16: ("grief", "negative"), 17: ("joy", "positive"),
    18: ("love", "positive"), 19: ("nervousness", "negative"),
    20: ("optimism", "positive"), 21: ("pride", "positive"),
    22: ("realization", "neutral"), 23: ("relief", "positive"),
    24: ("remorse", "negative"), 25: ("sadness", "negative"),
    26: ("surprise", "neutral"), 27: ("neutral", "neutral"),
}


def run_clanker(text, config: PipelineConfig):
    """Run text through the Clanker pipeline, respecting config toggles."""
    words = re.findall(r"[a-z']+", text.lower())

    # Input preprocessing: fuzzy matching
    if config.fuzzy_matching:
        resolved = []
        for w in words:
            if w not in WORD_FORCES:
                match = fuzzy_match(w)
                resolved.append(match if match else w)
            else:
                resolved.append(w)
        words = resolved

    # Input preprocessing: nonsense filter
    if config.nonsense_filter:
        det = NonsenseDetector()
        is_nonsense, _reason = det.detect(text)
        if is_nonsense:
            return "neutral", 128

    # Input preprocessing: intent detection (recorded, does not change score)
    if config.intent_detection:
        _intent = IntentDetector()
        _intent.detect(text)

    # Core pendulum — pass config so internal layers can be toggled
    p = SequentialPendulum(config=config)
    for i, w in enumerate(words):
        p.process_word(w, words, i)

    v = int(p.v)
    label = "positive" if v > 145 else "negative" if v < 110 else "neutral"
    return label, v


def run_with_config(config: PipelineConfig, ds) -> float:
    """Run full dataset with given config, return accuracy percentage."""
    correct = 0
    total = 0
    for row in ds:
        if not row["labels"]:
            continue
        _emo, truth = GOEMO.get(row["labels"][0], ("unk", "neutral"))
        label, _v = run_clanker(row["text"], config)
        if label == truth:
            correct += 1
        total += 1
    return (correct / total * 100) if total > 0 else 0.0


# All testable layers — both external and pendulum-internal
TESTABLE_LAYERS = [
    "fuzzy_matching",
    "intent_detection",
    "nonsense_filter",
    "idiom_detection",
    "context_modifiers",
    "crisis_momentum_lock",
    "morpheme_decomposition",
    "recency_weighting",
    "shift_markers",
    "tonal_analysis",
    "emotional_chunking",
    "contrast_detection",
]


def make_combo_config(enabled_layers: dict) -> PipelineConfig:
    """Create a PipelineConfig with only the specified layers enabled.

    Starts from minimal (all off) and enables the layers in the dict.
    """
    config = PipelineConfig.minimal()
    config.pendulum = True  # pendulum is always on
    for layer, on in enabled_layers.items():
        setattr(config, layer, on)
    return config


# Key combinations to test
COMBINATIONS = [
    (
        "minimal (pendulum only)",
        {},
    ),
    (
        "+idioms",
        {"idiom_detection": True},
    ),
    (
        "+idioms +context",
        {"idiom_detection": True, "context_modifiers": True},
    ),
    (
        "+idioms +context +recency +shift",
        {
            "idiom_detection": True, "context_modifiers": True,
            "recency_weighting": True, "shift_markers": True,
        },
    ),
    (
        "+idioms +context +recency +shift +morpheme",
        {
            "idiom_detection": True, "context_modifiers": True,
            "recency_weighting": True, "shift_markers": True,
            "morpheme_decomposition": True,
        },
    ),
    (
        "+idioms +context +recency +shift +morpheme +crisis",
        {
            "idiom_detection": True, "context_modifiers": True,
            "recency_weighting": True, "shift_markers": True,
            "morpheme_decomposition": True, "crisis_momentum_lock": True,
        },
    ),
    (
        "full -tonal",
        "__full_minus_tonal__",  # special marker
    ),
    (
        "full -fuzzy",
        "__full_minus_fuzzy__",  # special marker
    ),
    (
        "full (all on)",
        "__full__",  # special marker
    ),
]


def ablation_test(n_examples: int = 300):
    """Full layer ablation: individual + combinatorial."""
    print(f"Loading GoEmotions test set ({n_examples} examples)...")
    ds = load_dataset(
        "google-research-datasets/go_emotions", split="test"
    ).select(range(n_examples))

    # ── Baseline ──────────────────────────────────────────────
    print("Running baseline (all layers enabled)...")
    t0 = time.perf_counter()
    baseline_config = PipelineConfig.full()
    baseline_acc = run_with_config(baseline_config, ds)
    baseline_ms = (time.perf_counter() - t0) * 1000

    # Header
    width = 60
    print(f"\nLAYER ABLATION RESULTS ({n_examples} GoEmotions examples)")
    print("\u2550" * width)

    # ── Individual ablation (disable one at a time) ───────────
    print(f"\nINDIVIDUAL (disable one at a time):")
    print(f"  {'Baseline (all on):':<36} {baseline_acc:>5.1f}%  ({baseline_ms:.0f}ms)")

    individual_results = {}
    best_single_name = None
    best_single_acc = baseline_acc

    for layer in TESTABLE_LAYERS:
        config = PipelineConfig.full()
        setattr(config, layer, False)
        t0 = time.perf_counter()
        acc = run_with_config(config, ds)
        elapsed = (time.perf_counter() - t0) * 1000
        impact = acc - baseline_acc

        if impact > 0.5:
            marker = "\u2191 HELPS (currently hurting!)"
        elif impact < -0.5:
            marker = "\u2193 HURTS"
        else:
            marker = "= NEUTRAL"

        print(f"  -{layer + ':':<35} {acc:>5.1f}% ({impact:>+5.1f}pp) {marker}")
        individual_results[layer] = (acc, impact)

        if acc > best_single_acc:
            best_single_acc = acc
            best_single_name = layer

    if best_single_name:
        print(f"\n  Best single removal: -{best_single_name} ({best_single_acc:.1f}%)")

    # ── Combination testing ───────────────────────────────────
    print(f"\nCOMBINATIONS:")

    best_combo_name = None
    best_combo_acc = 0.0

    for name, layers in COMBINATIONS:
        if layers == "__full__":
            config = PipelineConfig.full()
        elif layers == "__full_minus_tonal__":
            config = PipelineConfig.full()
            config.tonal_analysis = False
        elif layers == "__full_minus_fuzzy__":
            config = PipelineConfig.full()
            config.fuzzy_matching = False
        else:
            config = make_combo_config(layers)

        t0 = time.perf_counter()
        acc = run_with_config(config, ds)
        elapsed = (time.perf_counter() - t0) * 1000

        best_marker = ""
        if acc > best_combo_acc:
            best_combo_acc = acc
            best_combo_name = name

        print(f"  {name + ':':<48} {acc:>5.1f}%  ({elapsed:.0f}ms)")

    print(f"\n  Best combination: {best_combo_name} ({best_combo_acc:.1f}%)")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * width}")
    print("SUMMARY:")
    print(f"  Baseline (all layers):  {baseline_acc:.1f}%")
    if best_single_name:
        print(f"  Best -1 removal:        {best_single_acc:.1f}% (remove {best_single_name})")
    print(f"  Best combination:       {best_combo_acc:.1f}% ({best_combo_name})")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Layer ablation test")
    parser.add_argument("-n", "--examples", type=int, default=300,
                        help="Number of GoEmotions examples (default: 300)")
    args = parser.parse_args()
    ablation_test(args.examples)
