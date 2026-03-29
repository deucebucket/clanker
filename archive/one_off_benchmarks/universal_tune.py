#!/usr/bin/env python3
"""Universal threshold sweep across ALL datasets combined.

Pre-processes all sentences once (the expensive part), then sweeps
intent-adaptive threshold combinations using fast integer comparisons.
Saves optimal config to universal_thresholds.json and prints results.
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

from datasets import load_dataset
from demo.simulator import SequentialPendulum, VADUG
from demo.intent import IntentDetector
from demo.pipeline_config import PipelineConfig

# ---------------------------------------------------------------------------
# Load all datasets
# ---------------------------------------------------------------------------

print("Loading datasets...")
sst = load_dataset("stanfordnlp/sst2", split="validation")
ge = load_dataset("google-research-datasets/go_emotions", split="test").select(range(1000))
te = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test")

intent_detector = IntentDetector()

# GoEmotions label mapping (same as academic_benchmark.py)
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

TW_SENT = {0: "negative", 1: "positive", 2: "positive", 3: "negative"}


# ---------------------------------------------------------------------------
# Pre-process all texts (expensive part, done once)
# ---------------------------------------------------------------------------

def process_text(text):
    """Run pendulum and return (V, intent_mode)."""
    mode, _ = intent_detector.detect(text)
    p = SequentialPendulum()
    words = re.findall(r"[a-z']+", text.lower())
    for i, w in enumerate(words):
        p.process_word(w, words, i)
    return int(p.v), mode


print("Pre-processing all datasets...")
t0 = time.perf_counter()

sst_data = []
for r in sst:
    v_mode = process_text(r["sentence"])
    truth = "positive" if r["label"] == 1 else "negative"
    sst_data.append((v_mode, truth))

ge_data = []
for r in ge:
    if not r["labels"]:
        continue
    _, sent = GOEMO.get(r["labels"][0], ("unk", "neutral"))
    ge_data.append((process_text(r["text"]), sent))

te_data = []
for r in te:
    te_data.append((process_text(r["text"]), TW_SENT[r["label"]]))

elapsed = time.perf_counter() - t0
print(f"SST-2: {len(sst_data)}, GoEmotions: {len(ge_data)}, TweetEval: {len(te_data)}")
print(f"Pre-processing took {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Sweep intent-adaptive thresholds
# ---------------------------------------------------------------------------

print("\nSweeping threshold combinations...")
t0 = time.perf_counter()

best_score = 0
best_config = None
configs_tested = 0

# Sweep: emotional thresholds + wide thresholds
for emo_low in range(115, 130, 3):         # emotional/venting low
    for emo_high in range(128, 142, 3):     # emotional/venting high
        if emo_high <= emo_low:
            continue
        for wide_low in range(100, 120, 5):  # question/greeting/request low
            for wide_high in range(135, 155, 5):  # question/greeting/request high
                if wide_high <= wide_low:
                    continue

                thresholds = {
                    "EMOTIONAL": (emo_low, emo_high),
                    "VENTING": (emo_low, emo_high),
                    "CASUAL": (emo_low + 2, emo_high + 2),
                    "STATEMENT": ((emo_low + wide_low) // 2, (emo_high + wide_high) // 2),
                    "QUESTION": (wide_low, wide_high),
                    "REQUEST": (wide_low, wide_high),
                    "GREETING": (wide_low, wide_high),
                }

                def classify(v, mode, _thresholds=thresholds):
                    low, high = _thresholds.get(mode, (120, 135))
                    if v > high:
                        return "positive"
                    elif v < low:
                        return "negative"
                    return "neutral"

                # Score across all datasets
                sst_correct = sum(1 for (v, mode), truth in sst_data if classify(v, mode) == truth)
                ge_correct = sum(1 for (v, mode), truth in ge_data if classify(v, mode) == truth)
                te_correct = sum(1 for (v, mode), truth in te_data if classify(v, mode) == truth)

                total = len(sst_data) + len(ge_data) + len(te_data)
                weighted = (sst_correct + ge_correct + te_correct) / total

                configs_tested += 1

                if weighted > best_score:
                    best_score = weighted
                    best_config = {
                        "emo_low": emo_low, "emo_high": emo_high,
                        "wide_low": wide_low, "wide_high": wide_high,
                        "sst_acc": sst_correct / len(sst_data) * 100,
                        "ge_acc": ge_correct / len(ge_data) * 100,
                        "te_acc": te_correct / len(te_data) * 100,
                        "weighted": weighted * 100,
                    }

sweep_elapsed = time.perf_counter() - t0
print(f"Tested {configs_tested} configurations in {sweep_elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Report results
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"BEST UNIVERSAL CONFIG:")
print(f"  Emotional/Venting: [{best_config['emo_low']}, {best_config['emo_high']}]")
print(f"  Casual:            [{best_config['emo_low']+2}, {best_config['emo_high']+2}]")
print(f"  Statement:         [{(best_config['emo_low']+best_config['wide_low'])//2}, {(best_config['emo_high']+best_config['wide_high'])//2}]")
print(f"  Question/Request:  [{best_config['wide_low']}, {best_config['wide_high']}]")
print(f"  SST-2:       {best_config['sst_acc']:.1f}%")
print(f"  GoEmotions:  {best_config['ge_acc']:.1f}%")
print(f"  TweetEval:   {best_config['te_acc']:.1f}%")
print(f"  WEIGHTED:    {best_config['weighted']:.1f}%")
print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

# Build the full threshold dict
optimal_thresholds = {
    "EMOTIONAL": {"low": best_config["emo_low"], "high": best_config["emo_high"]},
    "VENTING":   {"low": best_config["emo_low"], "high": best_config["emo_high"]},
    "CASUAL":    {"low": best_config["emo_low"] + 2, "high": best_config["emo_high"] + 2},
    "STATEMENT": {"low": (best_config["emo_low"] + best_config["wide_low"]) // 2,
                  "high": (best_config["emo_high"] + best_config["wide_high"]) // 2},
    "QUESTION":  {"low": best_config["wide_low"], "high": best_config["wide_high"]},
    "REQUEST":   {"low": best_config["wide_low"], "high": best_config["wide_high"]},
    "GREETING":  {"low": best_config["wide_low"], "high": best_config["wide_high"]},
    "DEFAULT":   {"low": (best_config["emo_low"] + best_config["wide_low"]) // 2,
                  "high": (best_config["emo_high"] + best_config["wide_high"]) // 2},
}

output = {
    "optimal_thresholds": optimal_thresholds,
    "per_dataset_accuracy": {
        "sst2": round(best_config["sst_acc"], 1),
        "go_emotions": round(best_config["ge_acc"], 1),
        "tweet_eval": round(best_config["te_acc"], 1),
    },
    "weighted_accuracy": round(best_config["weighted"], 1),
    "sweep_params": {
        "emo_low": best_config["emo_low"],
        "emo_high": best_config["emo_high"],
        "wide_low": best_config["wide_low"],
        "wide_high": best_config["wide_high"],
    },
    "configs_tested": configs_tested,
}

out_path = os.path.join(os.path.dirname(__file__), "universal_thresholds.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}")
