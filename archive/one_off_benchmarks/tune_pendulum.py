#!/usr/bin/env python3
"""
Pendulum Parameter Tuning Sweep Harness (#33)

Sweeps key pendulum parameters across ranges, runs test cases for each
configuration, and finds the optimal values via a staged bracket tournament.

Stage 1: Sweep each parameter independently (find top 3 per param)
Stage 2: Sweep top-3 combinations (3^6 = 729 configs)
Stage 3: Run top 8 configs against a larger test set to find the champion

Usage:
    python3 benchmarks/tune_pendulum.py
"""

import copy
import itertools
import json
import os
import re
import sys
import time

# Allow running standalone from the benchmarks/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.pendulum import SequentialPendulum
from demo.shared import VADUG


# ─────────────────────────────────────────────────────────────
# Test Sets
# ─────────────────────────────────────────────────────────────

# Each entry: (text, expected_label)
# Labels: "positive", "negative", "neutral", "crisis"

POSITIVE_CASES = [
    ("I'm having a great day!", "positive"),
    ("This is wonderful news", "positive"),
    ("I feel so happy right now", "positive"),
    ("You're amazing and I love you", "positive"),
    ("Best day of my life!", "positive"),
    ("I'm so excited about this opportunity", "positive"),
    ("Everything is going perfectly", "positive"),
    ("I'm thrilled to be here", "positive"),
    ("What a beautiful morning", "positive"),
    ("I can't believe how lucky I am", "positive"),
    ("Thank you so much for everything", "positive"),
    ("You make me so happy", "positive"),
    ("This is incredible work", "positive"),
    ("I'm proud of what we accomplished", "positive"),
    ("Life is beautiful", "positive"),
    ("I just got the promotion!", "positive"),
    ("Your cooking is absolutely delicious", "positive"),
    ("I'm grateful for your kindness", "positive"),
    ("The sunset is breathtaking tonight", "positive"),
    ("I love spending time with you", "positive"),
    ("This concert is phenomenal", "positive"),
    ("You always know how to cheer me up", "positive"),
    ("I feel blessed every single day", "positive"),
    ("My heart is so full right now", "positive"),
    ("I'm overjoyed with the results", "positive"),
    ("What a fantastic performance", "positive"),
    ("I appreciate you more than words can say", "positive"),
    ("Today was truly magical", "positive"),
    ("I'm on cloud nine right now", "positive"),
    ("You are such a wonderful person", "positive"),
]

NEGATIVE_CASES = [
    ("This is terrible and I hate it", "negative"),
    ("I'm so frustrated with everything", "negative"),
    ("Nothing ever works out for me", "negative"),
    ("I feel miserable today", "negative"),
    ("That was the worst experience ever", "negative"),
    ("I'm angry and disappointed", "negative"),
    ("Everything is falling apart", "negative"),
    ("I can't stand this anymore", "negative"),
    ("This makes me so sad", "negative"),
    ("I feel completely hopeless", "negative"),
    ("What a waste of time", "negative"),
    ("I regret everything about this", "negative"),
    ("This is absolutely disgusting", "negative"),
    ("I'm furious about what happened", "negative"),
    ("My heart is broken", "negative"),
    ("I feel betrayed and hurt", "negative"),
    ("This is the worst day of my life", "negative"),
    ("I'm sick of being treated this way", "negative"),
    ("Nobody understands my pain", "negative"),
    ("I'm drowning in problems", "negative"),
    ("This failure is devastating", "negative"),
    ("I feel so alone and lost", "negative"),
    ("Everything I touch turns to garbage", "negative"),
    ("I'm exhausted from all this suffering", "negative"),
    ("The world feels cold and uncaring", "negative"),
    ("I despise how things turned out", "negative"),
    ("It was horrible from start to finish", "negative"),
    ("I feel crushed by the weight of it all", "negative"),
    ("I'm worried sick about everything", "negative"),
    ("This situation is unbearable", "negative"),
]

NEUTRAL_CASES = [
    ("The meeting is at three o'clock", "neutral"),
    ("I need to pick up groceries today", "neutral"),
    ("The weather forecast says rain tomorrow", "neutral"),
    ("The report has been submitted", "neutral"),
    ("I'm going to the store later", "neutral"),
    ("My flight arrives at noon", "neutral"),
    ("The package should arrive on Tuesday", "neutral"),
    ("I need to update my password", "neutral"),
    ("The office is on the third floor", "neutral"),
    ("We have a meeting scheduled for Friday", "neutral"),
    ("The document is twenty pages long", "neutral"),
    ("I took the bus to work today", "neutral"),
    ("The temperature is seventy degrees", "neutral"),
    ("I need to charge my phone", "neutral"),
    ("The presentation is ready for review", "neutral"),
    ("I had cereal for breakfast", "neutral"),
    ("The file has been uploaded", "neutral"),
    ("My appointment is at two thirty", "neutral"),
    ("The parking lot is on the left", "neutral"),
    ("I sent the email this morning", "neutral"),
]

SARCASM_CASES = [
    ("Oh great, another Monday", "negative"),
    ("Wow, what a surprise, you're late again", "negative"),
    ("Yeah, because that worked so well last time", "negative"),
    ("Sure, everything is just perfect", "negative"),
    ("Oh wonderful, more homework", "negative"),
    ("Fantastic, the printer is broken again", "negative"),
    ("Oh joy, another meeting about meetings", "negative"),
    ("Well that was a productive waste of time", "negative"),
    ("Thanks for nothing, really appreciate it", "negative"),
    ("Oh how delightful, a traffic jam", "negative"),
    ("Brilliant, just what I needed today", "negative"),
    ("Real nice of you to show up two hours late", "negative"),
    ("Oh sure, I love working weekends", "negative"),
    ("What a lovely surprise to find my car towed", "negative"),
    ("Great job breaking everything again", "negative"),
]

CRISIS_CASES = [
    ("I want to die", "crisis"),
    ("I don't want to be here anymore", "crisis"),
    ("I want to end it all", "crisis"),
    ("Nobody would care if I disappeared", "crisis"),
    ("I can't go on like this", "crisis"),
    ("I'm better off dead", "crisis"),
    ("What's the point of living", "crisis"),
    ("I want to disappear forever", "crisis"),
    ("I'm a burden to everyone", "crisis"),
    ("I don't want to be alive", "crisis"),
    ("Nobody would miss me if I was gone", "crisis"),
    ("I just want the pain to stop", "crisis"),
    ("I'm tired of everything and everyone", "crisis"),
    ("There's no point in any of this", "crisis"),
    ("I wish I was never born", "crisis"),
]

MIXED_CASES = [
    ("I'm happy for you but sad for myself", "negative"),
    ("It's bittersweet seeing them go", "negative"),
    ("I love this place but I have to leave", "negative"),
    ("I'm proud but also terrified", "negative"),
    ("This is exciting but also scary", "positive"),
    ("I'm grateful but overwhelmed", "negative"),
    ("I feel free but also lonely", "negative"),
    ("It's beautiful but makes me cry", "negative"),
    ("I'm relieved it's over but miss it already", "negative"),
    ("I'm angry but I still love you", "negative"),
    ("I'm smiling but dying inside", "negative"),
    ("This is fun but exhausting", "positive"),
    ("I'm hopeful but trying not to get my hopes up", "positive"),
    ("I'm happy for them even though it hurts", "negative"),
    ("We won but at what cost", "negative"),
]

ARC_CASES = [
    ("I was having a terrible day but then I got the best news ever", "positive"),
    ("Everything was going great until the accident happened", "negative"),
    ("I started out angry but talking to you made it better", "positive"),
    ("I felt hopeless for weeks then suddenly everything clicked", "positive"),
    ("We were celebrating then the phone rang with bad news", "negative"),
    ("I was scared at first but now I feel brave", "positive"),
    ("Things were looking up but then reality hit hard", "negative"),
    ("The morning was rough but the afternoon was wonderful", "positive"),
    ("I cried for hours then finally found peace", "positive"),
    ("It was all fun and games until someone got hurt", "negative"),
    ("She was laughing and then she just broke down crying", "negative"),
    ("First I panicked, then I calmed down and figured it out", "positive"),
    ("The beginning was exciting but the ending was disappointing", "negative"),
    ("I was angry but then I realized they were right", "positive"),
    ("We struggled for months then finally succeeded", "positive"),
]

SLANG_CASES = [
    ("that slaps so hard bro", "positive"),
    ("im dead lmaooo", "positive"),
    ("this is fire ngl", "positive"),
    ("bruh that's lowkey sick", "positive"),
    ("no cap this goes hard", "positive"),
    ("yikes that's cringe", "negative"),
    ("oof that hits different", "negative"),
    ("rip my grades lol", "negative"),
    ("that's sus ngl", "negative"),
    ("welp that's a big L", "negative"),
]


# Combine into stage-appropriate sets
STAGE_1_TESTS = (
    POSITIVE_CASES + NEGATIVE_CASES + NEUTRAL_CASES +
    SARCASM_CASES + CRISIS_CASES + MIXED_CASES +
    ARC_CASES + SLANG_CASES
)

# Stage 3 uses a larger set — we duplicate with slight variations for volume
STAGE_3_EXTRA = [
    # Additional positive
    ("This made my whole week", "positive"),
    ("I'm beaming with pride", "positive"),
    ("You've made me the happiest person alive", "positive"),
    ("I feel so alive and energized", "positive"),
    ("Everything is coming together beautifully", "positive"),
    ("My dream is finally coming true", "positive"),
    ("I'm in love with this new chapter of life", "positive"),
    ("You are the best thing that ever happened to me", "positive"),
    ("I'm floating on air right now", "positive"),
    ("Pure bliss, nothing else", "positive"),
    # Additional negative
    ("I feel like a complete failure", "negative"),
    ("The loneliness is crushing me", "negative"),
    ("I wasted years on this", "negative"),
    ("I can't stop crying", "negative"),
    ("I feel numb and empty inside", "negative"),
    ("Every day is worse than the last", "negative"),
    ("I hate myself for letting this happen", "negative"),
    ("I'm trapped and there's no way out", "negative"),
    ("The grief is overwhelming", "negative"),
    ("I feel utterly defeated", "negative"),
    # Additional neutral
    ("The car needs an oil change", "neutral"),
    ("I have three emails to respond to", "neutral"),
    ("The library closes at nine tonight", "neutral"),
    ("I ordered the blue one", "neutral"),
    ("The meeting was rescheduled to Thursday", "neutral"),
    # Additional crisis
    ("I've been thinking about ending things", "crisis"),
    ("I just don't see a reason to keep going", "crisis"),
    ("Sometimes I think everyone would be better off without me", "crisis"),
    ("I can't take this anymore I just want it to end", "crisis"),
    ("I feel like I'm already dead inside", "crisis"),
    # Additional mixed
    ("I'm proud of you but I wish it was me", "negative"),
    ("Beautiful ceremony but I cried the whole time", "negative"),
    ("Best meal ever but now I feel sick", "negative"),
    # Additional arcs
    ("I hated it at first but it grew on me", "positive"),
    ("We started with high hopes but crashed and burned", "negative"),
    ("The pain faded and joy took its place", "positive"),
    # Additional sarcasm
    ("Oh perfect, the wifi is down again", "negative"),
    ("How wonderful, more bills to pay", "negative"),
    ("Gee thanks, you really outdid yourself this time", "negative"),
]

STAGE_3_TESTS = STAGE_1_TESTS + STAGE_3_EXTRA


# ─────────────────────────────────────────────────────────────
# Classification from VADUG
# ─────────────────────────────────────────────────────────────

def classify_vadug(v, u, g, neutral_low=110, neutral_high=145):
    """Classify a VADUG result into a label.

    Uses valence (V), urgency (U), and gravity (G) to determine:
    - "crisis": V very low AND (high urgency OR crushing gravity)
    - "negative": V below neutral_low
    - "positive": V above neutral_high
    - "neutral": in between
    """
    # Crisis detection: very low V combined with urgency or crushing gravity
    if v < 60 and (u > 40 or g < 70):
        return "crisis"
    if v < 45:
        return "crisis"

    if v < neutral_low:
        return "negative"
    elif v > neutral_high:
        return "positive"
    else:
        return "neutral"


# ─────────────────────────────────────────────────────────────
# Pendulum Runner with Parameter Overrides
# ─────────────────────────────────────────────────────────────

def run_pendulum_with_config(text, config):
    """Run the pendulum on text with parameter overrides applied via monkey-patching.

    Config keys:
        momentum        - SequentialPendulum.momentum initial value
        neutral_low     - classification threshold (V < this = negative)
        neutral_high    - classification threshold (V > this = positive)
        force_scale     - multiplier on word forces (applied via recency weight bias)
        idiom_momentum  - momentum value used during idiom application
        idiom_push      - direct push strength for idioms
        recency_min     - minimum recency weight
        recency_max     - maximum recency weight
    """
    pend = SequentialPendulum()

    # Override momentum
    pend.momentum = config.get("momentum", 0.28)

    # We monkey-patch process_word to inject our parameter overrides.
    # Store original method
    original_process_word = pend.process_word.__func__

    force_scale_mult = config.get("force_scale", 1.0)
    idiom_momentum_val = config.get("idiom_momentum", 0.70)
    idiom_push_val = config.get("idiom_push", 0.5)
    recency_min = config.get("recency_min", 0.8)
    recency_max = config.get("recency_max", 1.2)

    # Patch: override the recency weight calculation and force scale
    # We wrap process_word to adjust internal values
    def patched_process_word(self, word, words, current_idx):
        # Save originals we want to override
        orig_intensity = self.intensity

        # Recency weight override: the original uses 0.8 + 0.4 * ratio
        # We inject our range: recency_min + (recency_max - recency_min) * ratio
        # This is done by temporarily patching intensity to include force_scale
        self.intensity = orig_intensity * force_scale_mult

        # We need to intercept idiom momentum too. The idiom code uses
        # hardcoded 0.70 and 0.5. We patch those by temporarily modifying
        # the method's local constants. Since we can't easily do that,
        # we'll run the original and wrap the idiom application.
        # Instead, we'll use a different approach: run the whole thing
        # but scale the recency weight.

        # Reset intensity (force_scale is applied below via a different mechanism)
        self.intensity = orig_intensity

        # Call original
        result = original_process_word(self, word, words, current_idx)
        return result

    # A cleaner approach: since we can't easily intercept the hardcoded
    # idiom_momentum and idiom_push inside process_word without rewriting it,
    # we'll create a fully custom processing loop that reimplements the
    # key parts with our config values.

    words = re.findall(r"[a-z']+", text.lower())
    _run_custom_pendulum(pend, words, config)

    return pend.v, pend.a, pend.d, pend.u, pend.g


def _run_custom_pendulum(pend, words, config):
    """Run pendulum processing with full parameter control.

    This reimplements the word loop from SequentialPendulum.process_word
    but with configurable parameters instead of hardcoded values.
    """
    force_scale_mult = config.get("force_scale", 1.0)
    idiom_momentum_val = config.get("idiom_momentum", 0.70)
    idiom_push_val = config.get("idiom_push", 0.5)
    recency_min = config.get("recency_min", 0.8)
    recency_max = config.get("recency_max", 1.2)

    for idx, word in enumerate(words):
        # Temporarily override the intensity to include force_scale
        # and recency weight range. We do this by scaling the intensity
        # that process_word will use.
        n_words = len(words)
        # Original recency: 0.8 + 0.4 * (idx / max(n_words - 1, 1))
        # Our recency: recency_min + (recency_max - recency_min) * (idx / max(n_words - 1, 1))
        original_recency = 0.8 + 0.4 * (idx / max(n_words - 1, 1))
        custom_recency = recency_min + (recency_max - recency_min) * (idx / max(n_words - 1, 1))

        # The ratio between custom and original recency, times force_scale
        if original_recency > 0:
            scale_factor = (custom_recency / original_recency) * force_scale_mult
        else:
            scale_factor = force_scale_mult

        # Save the current intensity (should be 1.0 at start of each word)
        saved_intensity = pend.intensity
        pend.intensity = saved_intensity * scale_factor

        # For idiom overrides, we temporarily monkey-patch the constants
        # in the process_word scope. We can't do that directly, so we
        # pre/post-process: save state before, call process_word, then
        # if an idiom was detected (check history), re-apply with our values.

        v_before = pend.v
        a_before = pend.a
        d_before = pend.d
        u_before = pend.u
        g_before = pend.g
        history_len_before = len(pend.history)

        pend.process_word(word, words, idx)

        # Check if an idiom was hit by looking at the trace
        if len(pend.history) > history_len_before:
            last_entry = pend.history[-1]
            state = last_entry.get("state", "")
            if "IDIOM:" in state:
                # An idiom was applied with the hardcoded values (0.70, 0.5).
                # We need to undo and redo with our values.
                # Restore pre-idiom state and recompute.
                idiom_result = pend.check_idiom(words, idx)
                if idiom_result:
                    vf, af, df, uf, label, idiom_len, idiom_start, gf = idiom_result

                    # Negate if needed (approximate — we can't know for sure
                    # but the idiom code handles this before us)

                    # Restore state to pre-idiom
                    pend.v = v_before
                    pend.a = a_before
                    pend.d = d_before
                    pend.u = u_before
                    pend.g = g_before

                    # Apply with our idiom parameters
                    fs = 1.0 * pend.intensity * custom_recency
                    im = idiom_momentum_val
                    ip = idiom_push_val

                    pend.v = pend.v * im + (128.0 + vf * fs) * (1.0 - im) + vf * ip
                    pend.a = pend.a * im + (128.0 + af * fs) * (1.0 - im) + af * ip
                    pend.d = pend.d * im + (128.0 + df * fs) * (1.0 - im) + df * ip
                    pend.u = pend.u * im + (uf * fs) * (1.0 - im) + uf * ip
                    pend.g = pend.g * im + (128.0 + gf * fs) * (1.0 - im) + gf * ip
                    pend._clamp()

                    # Handle crisis momentum lock (same logic as original)
                    if label.startswith("crisis:"):
                        if vf <= -80:
                            pend._crisis_momentum = len(words) - idx - 1
                            pend.v = max(0, pend.v - 15)
                            pend.g = max(0, pend.g - 10)

        # Reset intensity for next word
        pend.intensity = 1.0


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────

def score_config(config, test_set):
    """Run all test cases with the given config and return a weighted score.

    Returns:
        dict with keys: score, accuracy, crisis_recall, sarcasm_accuracy,
                        neutral_rate, details
    """
    correct = 0
    total = len(test_set)
    crisis_correct = 0
    crisis_total = 0
    sarcasm_correct = 0
    sarcasm_total = 0
    neutral_count = 0

    neutral_low = config.get("neutral_low", 110)
    neutral_high = config.get("neutral_high", 145)

    for text, expected in test_set:
        v, a, d, u, g = run_pendulum_with_config(text, config)
        predicted = classify_vadug(v, u, g, neutral_low, neutral_high)

        is_correct = (predicted == expected)
        if is_correct:
            correct += 1

        if predicted == "neutral":
            neutral_count += 1

        if expected == "crisis":
            crisis_total += 1
            if predicted == "crisis":
                crisis_correct += 1

        # Track sarcasm cases (they're in test set as "negative" expected)
        # We identify them by checking if they're from the sarcasm test set
        # This is handled by the caller marking them

    accuracy = correct / total if total > 0 else 0
    crisis_recall = crisis_correct / crisis_total if crisis_total > 0 else 1.0
    neutral_rate = neutral_count / total if total > 0 else 0

    # Weighted score:
    # Base accuracy + crisis recall bonus (2x weight) + neutral penalty
    score = accuracy
    score += crisis_recall * 0.20  # Up to 20% bonus for crisis recall
    if neutral_rate > 0.20:
        score -= (neutral_rate - 0.20) * 0.5  # Penalize excessive neutrals

    return {
        "score": round(score, 4),
        "accuracy": round(accuracy, 4),
        "crisis_recall": round(crisis_recall, 4),
        "neutral_rate": round(neutral_rate, 4),
        "correct": correct,
        "total": total,
        "crisis_correct": crisis_correct,
        "crisis_total": crisis_total,
    }


# ─────────────────────────────────────────────────────────────
# Default Configuration (current engine values)
# ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "momentum": 0.28,
    "neutral_low": 110,
    "neutral_high": 145,
    "force_scale": 1.0,
    "idiom_momentum": 0.70,
    "idiom_push": 0.5,
    "recency_min": 0.8,
    "recency_max": 1.2,
}


# ─────────────────────────────────────────────────────────────
# Parameter Ranges
# ─────────────────────────────────────────────────────────────

PARAM_RANGES = {
    "momentum":      [round(x, 2) for x in [v * 0.05 + 0.70 for v in range(6)]],  # 0.70-0.95
    "neutral_low":   list(range(95, 125, 5)),   # 95-120
    "neutral_high":  list(range(135, 165, 5)),   # 135-160
    "force_scale":   [round(x, 1) for x in [v * 0.1 + 0.8 for v in range(8)]],  # 0.8-1.5
    "idiom_momentum": [round(x, 2) for x in [v * 0.05 + 0.50 for v in range(8)]],  # 0.50-0.85
    "idiom_push":    [round(x, 1) for x in [v * 0.1 + 0.3 for v in range(5)]],  # 0.3-0.7
}

# Recency is a compound parameter (min and max together)
RECENCY_RANGES = {
    "recency_min": [round(x, 1) for x in [v * 0.1 + 0.6 for v in range(4)]],  # 0.6-0.9
    "recency_max": [round(x, 1) for x in [v * 0.1 + 1.1 for v in range(5)]],  # 1.1-1.5
}


# ─────────────────────────────────────────────────────────────
# Stage 1: Individual Parameter Sweeps
# ─────────────────────────────────────────────────────────────

def stage_1(test_set):
    """Sweep each parameter independently, holding others at defaults.

    Returns dict mapping param_name -> sorted list of (value, score_dict).
    """
    print("=" * 70)
    print("STAGE 1: Individual Parameter Sweeps")
    print("=" * 70)

    results = {}

    # Sweep each standard parameter
    for param_name, values in PARAM_RANGES.items():
        print(f"\n  Sweeping {param_name}: {values}")
        param_results = []

        for val in values:
            config = dict(DEFAULT_CONFIG)
            config[param_name] = val
            score_dict = score_config(config, test_set)
            param_results.append((val, score_dict))
            print(f"    {param_name}={val:>6} -> "
                  f"score={score_dict['score']:.4f}  "
                  f"acc={score_dict['accuracy']:.1%}  "
                  f"crisis={score_dict['crisis_recall']:.0%}  "
                  f"neut={score_dict['neutral_rate']:.1%}")

        # Sort by score descending
        param_results.sort(key=lambda x: x[1]["score"], reverse=True)
        results[param_name] = param_results

        best_val, best_score = param_results[0]
        print(f"  >> Best {param_name}: {best_val} (score={best_score['score']:.4f})")

    # Sweep recency as paired ranges
    print(f"\n  Sweeping recency_weight (min x max):")
    recency_results = []
    for rmin in RECENCY_RANGES["recency_min"]:
        for rmax in RECENCY_RANGES["recency_max"]:
            if rmin >= rmax:
                continue
            config = dict(DEFAULT_CONFIG)
            config["recency_min"] = rmin
            config["recency_max"] = rmax
            score_dict = score_config(config, test_set)
            recency_results.append(((rmin, rmax), score_dict))
            print(f"    recency=[{rmin}, {rmax}] -> "
                  f"score={score_dict['score']:.4f}  "
                  f"acc={score_dict['accuracy']:.1%}  "
                  f"crisis={score_dict['crisis_recall']:.0%}  "
                  f"neut={score_dict['neutral_rate']:.1%}")

    recency_results.sort(key=lambda x: x[1]["score"], reverse=True)
    results["recency"] = recency_results

    best_rec, best_score = recency_results[0]
    print(f"  >> Best recency: [{best_rec[0]}, {best_rec[1]}] (score={best_score['score']:.4f})")

    return results


# ─────────────────────────────────────────────────────────────
# Stage 2: Top-N Combination Sweep
# ─────────────────────────────────────────────────────────────

def stage_2(stage1_results, test_set, top_n=3):
    """Take top N values per parameter, sweep all combinations.

    Returns sorted list of (config_dict, score_dict).
    """
    print("\n" + "=" * 70)
    print(f"STAGE 2: Top-{top_n} Combination Sweep")
    print("=" * 70)

    # Extract top N values for each parameter
    top_values = {}
    for param_name in PARAM_RANGES:
        vals = [v for v, _ in stage1_results[param_name][:top_n]]
        top_values[param_name] = vals
        print(f"  {param_name}: top {top_n} = {vals}")

    # Top recency pairs
    top_recency = [v for v, _ in stage1_results["recency"][:top_n]]
    print(f"  recency: top {top_n} = {top_recency}")

    # Build all combinations
    param_names = list(PARAM_RANGES.keys())
    param_lists = [top_values[p] for p in param_names]

    total_combos = 1
    for pl in param_lists:
        total_combos *= len(pl)
    total_combos *= len(top_recency)

    print(f"\n  Total combinations: {total_combos}")
    print(f"  Running...\n")

    all_results = []
    count = 0

    for combo in itertools.product(*param_lists):
        for rmin, rmax in top_recency:
            config = dict(DEFAULT_CONFIG)
            for i, param_name in enumerate(param_names):
                config[param_name] = combo[i]
            config["recency_min"] = rmin
            config["recency_max"] = rmax

            score_dict = score_config(config, test_set)
            all_results.append((dict(config), score_dict))

            count += 1
            if count % 100 == 0 or count == total_combos:
                print(f"  Progress: {count}/{total_combos} configs tested", end="\r")

    print(f"  Progress: {count}/{total_combos} configs tested -- done!")

    # Sort by score
    all_results.sort(key=lambda x: x[1]["score"], reverse=True)

    # Print top 10
    print(f"\n  Top 10 Configurations:")
    print(f"  {'#':>4}  {'Score':>7}  {'Acc':>6}  {'Crisis':>6}  {'Neut%':>6}  Config")
    print(f"  {'-'*80}")
    for rank, (cfg, sd) in enumerate(all_results[:10], 1):
        cfg_str = (f"mom={cfg['momentum']:.2f} nL={cfg['neutral_low']} "
                   f"nH={cfg['neutral_high']} fs={cfg['force_scale']:.1f} "
                   f"im={cfg['idiom_momentum']:.2f} ip={cfg['idiom_push']:.1f} "
                   f"rec=[{cfg['recency_min']},{cfg['recency_max']}]")
        print(f"  {rank:>4}  {sd['score']:>7.4f}  {sd['accuracy']:>5.1%}  "
              f"{sd['crisis_recall']:>5.0%}  {sd['neutral_rate']:>5.1%}  {cfg_str}")

    return all_results


# ─────────────────────────────────────────────────────────────
# Stage 3: Final Eight
# ─────────────────────────────────────────────────────────────

def stage_3(stage2_results, test_set, top_n=8):
    """Run top N configs from Stage 2 against the larger test set.

    Returns the champion config and score.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 3: Final {top_n} Championship Round")
    print(f"  Test set: {len(test_set)} cases")
    print("=" * 70)

    finalists = stage2_results[:top_n]
    final_results = []

    for rank, (cfg, _) in enumerate(finalists, 1):
        score_dict = score_config(cfg, test_set)
        final_results.append((cfg, score_dict))
        cfg_str = (f"mom={cfg['momentum']:.2f} nL={cfg['neutral_low']} "
                   f"nH={cfg['neutral_high']} fs={cfg['force_scale']:.1f} "
                   f"im={cfg['idiom_momentum']:.2f} ip={cfg['idiom_push']:.1f} "
                   f"rec=[{cfg['recency_min']},{cfg['recency_max']}]")
        print(f"\n  #{rank}: {cfg_str}")
        print(f"       Score: {score_dict['score']:.4f}  "
              f"Accuracy: {score_dict['accuracy']:.1%}  "
              f"Crisis Recall: {score_dict['crisis_recall']:.0%}  "
              f"Neutral Rate: {score_dict['neutral_rate']:.1%}")

    # Sort
    final_results.sort(key=lambda x: x[1]["score"], reverse=True)

    # Print bracket-style matchups
    print(f"\n  {'─' * 60}")
    print(f"  BRACKET RESULTS (sorted by final score):")
    print(f"  {'─' * 60}")
    for i in range(0, len(final_results) - 1, 2):
        cfg1, sd1 = final_results[i]
        cfg2, sd2 = final_results[i + 1]
        winner = i + 1 if sd1["score"] >= sd2["score"] else i + 2
        print(f"  Match {i//2 + 1}: #{i+1} ({sd1['score']:.4f}) vs "
              f"#{i+2} ({sd2['score']:.4f}) -> Winner: #{winner}")

    champion_cfg, champion_score = final_results[0]
    return champion_cfg, champion_score


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    print("=" * 70)
    print("  CLANKER PENDULUM PARAMETER TUNING SWEEP")
    print(f"  Stage 1 test set: {len(STAGE_1_TESTS)} cases")
    print(f"  Stage 3 test set: {len(STAGE_3_TESTS)} cases")
    print("=" * 70)

    # Run baseline
    print("\n  BASELINE (current defaults):")
    baseline = score_config(DEFAULT_CONFIG, STAGE_1_TESTS)
    print(f"  Score: {baseline['score']:.4f}  "
          f"Accuracy: {baseline['accuracy']:.1%}  "
          f"Crisis Recall: {baseline['crisis_recall']:.0%}  "
          f"Neutral Rate: {baseline['neutral_rate']:.1%}")

    # Stage 1
    stage1_results = stage_1(STAGE_1_TESTS)

    # Stage 2
    stage2_results = stage_2(stage1_results, STAGE_1_TESTS, top_n=3)

    # Stage 3
    champion_cfg, champion_score = stage_3(stage2_results, STAGE_3_TESTS, top_n=8)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  CHAMPION CONFIGURATION")
    print("=" * 70)
    print(f"  momentum:       {champion_cfg['momentum']}")
    print(f"  neutral_low:    {champion_cfg['neutral_low']}")
    print(f"  neutral_high:   {champion_cfg['neutral_high']}")
    print(f"  force_scale:    {champion_cfg['force_scale']}")
    print(f"  idiom_momentum: {champion_cfg['idiom_momentum']}")
    print(f"  idiom_push:     {champion_cfg['idiom_push']}")
    print(f"  recency_min:    {champion_cfg['recency_min']}")
    print(f"  recency_max:    {champion_cfg['recency_max']}")
    print(f"")
    print(f"  Accuracy:       {champion_score['accuracy']:.1%} ({champion_score['correct']}/{champion_score['total']})")
    print(f"  Crisis Recall:  {champion_score['crisis_recall']:.0%} ({champion_score['crisis_correct']}/{champion_score['crisis_total']})")
    print(f"  Neutral Rate:   {champion_score['neutral_rate']:.1%}")
    print(f"  Weighted Score: {champion_score['score']:.4f}")
    print(f"")
    print(f"  vs Baseline:    {'+' if champion_score['score'] > baseline['score'] else ''}{(champion_score['score'] - baseline['score']):.4f} score delta")
    print(f"  Total runtime:  {elapsed:.1f}s")
    print("=" * 70)

    # Save champion config
    output_path = os.path.join(os.path.dirname(__file__), "optimal_config.json")
    output = {
        "champion": champion_cfg,
        "results": {
            "accuracy": champion_score["accuracy"],
            "crisis_recall": champion_score["crisis_recall"],
            "neutral_rate": champion_score["neutral_rate"],
            "weighted_score": champion_score["score"],
            "correct": champion_score["correct"],
            "total": champion_score["total"],
        },
        "baseline": {
            "config": DEFAULT_CONFIG,
            "accuracy": baseline["accuracy"],
            "crisis_recall": baseline["crisis_recall"],
            "neutral_rate": baseline["neutral_rate"],
            "weighted_score": baseline["score"],
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Champion config saved to: {output_path}")


if __name__ == "__main__":
    main()
