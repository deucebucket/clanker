#!/usr/bin/env python3
"""
Multi-Model VADUG Force Synthesis Pipeline

Generates complete 5D VADUG force tuples for 50K+ English words by running
multiple emotion/sentiment models as sensors and synthesizing their outputs.

Models used:
  A) NRC VAD Lexicon (pre-generated) -- gold-standard V/A/D
  B) GoEmotions classifier (SamLowe/roberta-base-go_emotions) -- U/G via 27 emotions
  C) VADER sentiment -- valence cross-check
  D) TextBlob -- valence cross-check

Output:
  benchmarks/synthesized_forces.py  -- Python dict of force tuples
  benchmarks/synthesis_report.txt   -- Human-readable analysis

Usage:
    python benchmarks/synthesize_forces.py
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_FORCES = SCRIPT_DIR / "synthesized_forces.py"
OUTPUT_REPORT = SCRIPT_DIR / "synthesis_report.txt"
CHECKPOINT_FILE = SCRIPT_DIR / "synthesis_checkpoint.json"

# ---------------------------------------------------------------------------
# GoEmotions to U/G mappings
# ---------------------------------------------------------------------------
URGENCY_MAP = {
    "anger": 0.7, "annoyance": 0.4, "fear": 0.9, "nervousness": 0.8,
    "surprise": 0.5, "excitement": 0.4, "grief": 0.6, "sadness": 0.2,
    "joy": 0.1, "love": 0.1, "gratitude": 0.1, "amusement": 0.1,
    "curiosity": 0.3, "confusion": 0.4, "desire": 0.5, "disgust": 0.5,
    "disappointment": 0.3, "disapproval": 0.4, "embarrassment": 0.3,
    "optimism": 0.2, "pride": 0.1, "realization": 0.2, "relief": 0.0,
    "remorse": 0.4, "caring": 0.2, "admiration": 0.1, "approval": 0.1,
    "neutral": 0.0,
}

GRAVITY_MAP = {
    "anger": 0.4, "annoyance": 0.1, "fear": -0.3, "nervousness": -0.1,
    "surprise": 0.3, "excitement": 0.7, "grief": -0.9, "sadness": -0.6,
    "joy": 0.8, "love": 0.6, "gratitude": 0.4, "amusement": 0.5,
    "curiosity": 0.2, "confusion": -0.1, "desire": 0.3, "disgust": 0.2,
    "disappointment": -0.4, "disapproval": 0.0, "embarrassment": -0.3,
    "optimism": 0.5, "pride": 0.6, "realization": 0.0, "relief": 0.4,
    "remorse": -0.5, "caring": 0.3, "admiration": 0.5, "approval": 0.2,
    "neutral": 0.0,
}

# Arousal proxy from GoEmotions (high-energy emotions)
AROUSAL_MAP = {
    "anger": 0.8, "annoyance": 0.4, "fear": 0.8, "nervousness": 0.6,
    "surprise": 0.7, "excitement": 0.9, "grief": 0.4, "sadness": 0.2,
    "joy": 0.7, "love": 0.6, "gratitude": 0.4, "amusement": 0.6,
    "curiosity": 0.5, "confusion": 0.3, "desire": 0.6, "disgust": 0.6,
    "disappointment": 0.3, "disapproval": 0.4, "embarrassment": 0.4,
    "optimism": 0.4, "pride": 0.5, "realization": 0.3, "relief": 0.2,
    "remorse": 0.3, "caring": 0.3, "admiration": 0.4, "approval": 0.3,
    "neutral": 0.1,
}

# Positive / negative emotion sets for GoEmotions valence
POS_EMOTIONS = {
    "joy", "love", "excitement", "gratitude", "admiration", "amusement",
    "optimism", "pride", "relief", "caring", "approval",
}
NEG_EMOTIONS = {
    "anger", "sadness", "fear", "disgust", "grief", "annoyance",
    "disappointment", "disapproval", "embarrassment", "nervousness", "remorse",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_avg(sources):
    """Weighted average from list of (name, value, weight) tuples."""
    total_w = sum(w for _, _, w in sources)
    if total_w == 0:
        return 0.5
    return sum(v * w for _, v, w in sources) / total_w


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Step 1: Build word list from datasets
# ---------------------------------------------------------------------------

def build_word_list():
    """Combine words from GoEmotions, SST-2, and TweetEval."""
    from datasets import load_dataset

    print("=== Step 1: Building word list from datasets ===")
    all_words = Counter()

    # GoEmotions (58K Reddit comments)
    print("  Loading GoEmotions...", end=" ", flush=True)
    ge = load_dataset("google-research-datasets/go_emotions", split="train")
    for row in ge:
        words = re.findall(r"[a-z']+", row["text"].lower())
        all_words.update(words)
    print(f"done ({len(ge)} rows)")

    # SST-2 (67K movie reviews)
    print("  Loading SST-2...", end=" ", flush=True)
    sst = load_dataset("stanfordnlp/sst2", split="train")
    for row in sst:
        words = re.findall(r"[a-z']+", row["sentence"].lower())
        all_words.update(words)
    print(f"done ({len(sst)} rows)")

    # TweetEval emotion
    print("  Loading TweetEval...", end=" ", flush=True)
    te = load_dataset("cardiffnlp/tweet_eval", "emotion", split="train")
    for row in te:
        words = re.findall(r"[a-z']+", row["text"].lower())
        all_words.update(words)
    print(f"done ({len(te)} rows)")

    # Filter: 3+ occurrences, length > 2, alphabetic only
    word_list = [
        w for w, c in all_words.most_common()
        if c >= 3 and len(w) > 2 and w.isalpha()
    ]
    print(f"  Total unique words (freq>=3, len>2, alpha): {len(word_list)}")
    return word_list


# ---------------------------------------------------------------------------
# Step 2: Load existing forces
# ---------------------------------------------------------------------------

def load_existing_forces():
    """Load WORD_FORCES from simulator and NRC-generated forces."""
    existing = {}

    # Load from simulator.py WORD_FORCES
    sim_path = PROJECT_ROOT / "demo" / "simulator.py"
    if sim_path.exists():
        try:
            content = sim_path.read_text()
            # Find WORD_FORCES = { ... } block
            start = content.find("WORD_FORCES = {")
            if start >= 0:
                # Find matching closing brace
                depth = 0
                for j in range(start + len("WORD_FORCES = {") - 1, len(content)):
                    if content[j] == '{':
                        depth += 1
                    elif content[j] == '}':
                        depth -= 1
                        if depth == 0:
                            block = content[start:j + 1]
                            sim_globals = {}
                            exec(block, sim_globals)  # noqa: S102
                            existing.update(sim_globals["WORD_FORCES"])
                            print(f"  Loaded {len(sim_globals['WORD_FORCES'])} words from simulator.py")
                            break
        except Exception as e:
            print(f"  Warning: could not load simulator WORD_FORCES: {e}")

    # Load NRC-generated forces
    gen_path = SCRIPT_DIR / "generated_forces.py"
    if gen_path.exists():
        gen_globals = {}
        try:
            exec(gen_path.read_text(), gen_globals)  # noqa: S102
            if "GENERATED_FORCES" in gen_globals:
                existing.update(gen_globals["GENERATED_FORCES"])
                print(f"  Loaded {len(gen_globals['GENERATED_FORCES'])} words from generated_forces.py")
        except Exception as e:
            print(f"  Warning: could not load generated_forces: {e}")

    return existing


def load_nrc_forces():
    """Load NRC VAD data as a dict of word -> (V, A, D) in 0..1 scale."""
    nrc = {}

    # Try to load raw lexicon files directly
    for lexicon_name in ["nrc_vad_lexicon_v2.txt", "nrc_vad_lexicon_v1.txt"]:
        lex_path = SCRIPT_DIR / lexicon_name
        if lex_path.exists():
            print(f"  Loading raw NRC VAD from {lexicon_name}...")
            with open(lex_path, "r") as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        word = parts[0].lower()
                        try:
                            v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
                            # Normalize to 0..1 if v2 (which is -1..1)
                            if "v2" in lexicon_name:
                                v = (v + 1) / 2
                                a = (a + 1) / 2
                                d = (d + 1) / 2
                            nrc[word] = (v, a, d)
                        except ValueError:
                            continue
            print(f"  Loaded {len(nrc)} NRC VAD entries")
            return nrc

    # Fallback: reverse-engineer from generated forces (approximate)
    gen_path = SCRIPT_DIR / "generated_forces.py"
    if gen_path.exists():
        print("  No raw NRC lexicon found, reverse-engineering from generated_forces.py...")
        gen_globals = {}
        try:
            exec(gen_path.read_text(), gen_globals)  # noqa: S102
            gf = gen_globals.get("GENERATED_FORCES", {})
            for word, (dv, da, dd, du, dg) in gf.items():
                # Reverse the scaling: dv = (v - 0.5) * 110 -> v = dv/110 + 0.5
                v = dv / 110.0 + 0.5
                a = da / 90.0 + 0.5
                d = dd / 80.0 + 0.5
                nrc[word] = (clamp(v, 0, 1), clamp(a, 0, 1), clamp(d, 0, 1))
            print(f"  Reverse-engineered {len(nrc)} NRC entries from generated forces")
        except Exception as e:
            print(f"  Warning: could not reverse-engineer NRC forces: {e}")

    return nrc


# ---------------------------------------------------------------------------
# Step 3: Model runners
# ---------------------------------------------------------------------------

def run_goemo_batch(words, pipeline_fn, batch_size=100):
    """Run GoEmotions classifier on words in batches.
    Returns dict of word -> {emotion: prob}."""
    results = {}
    prompts = []
    word_keys = []

    for w in words:
        prompts.append(f"I feel {w}")
        word_keys.append(w)

    total = len(prompts)
    for i in range(0, total, batch_size):
        batch = prompts[i:i + batch_size]
        batch_words = word_keys[i:i + batch_size]
        try:
            outputs = pipeline_fn(batch)
            for w, out in zip(batch_words, outputs):
                probs = {item["label"]: item["score"] for item in out}
                results[w] = probs
        except Exception as e:
            print(f"    GoEmo batch error at {i}: {e}")
            # Fill with empty for this batch
            for w in batch_words:
                results[w] = {}

    return results


def run_vader_batch(words):
    """Run VADER on all words. Returns dict of word -> compound score."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    results = {}
    for w in words:
        results[w] = vader.polarity_scores(w)["compound"]
    return results


def run_textblob_batch(words):
    """Run TextBlob on all words. Returns dict of word -> polarity."""
    from textblob import TextBlob
    results = {}
    for w in words:
        results[w] = TextBlob(w).sentiment.polarity
    return results


# ---------------------------------------------------------------------------
# Step 4: Synthesize
# ---------------------------------------------------------------------------

def synthesize(word, nrc_vad, goemo_probs, vader_compound, textblob_pol):
    """
    Combine model outputs into a single VADUG force tuple.

    Args:
        nrc_vad: (V, A, D) tuple in 0..1 scale, or None
        goemo_probs: dict of {emotion: probability}, or None/empty
        vader_compound: float -1..1, or None
        textblob_pol: float -1..1, or None

    Returns:
        (dv, da, dd, du, dg) integer force tuple
    """
    # --- Valence (dv) ---
    v_sources = []
    if nrc_vad is not None:
        v_sources.append(("nrc", nrc_vad[0], 0.50))
    if vader_compound is not None:
        v_sources.append(("vader", (vader_compound + 1) / 2, 0.25))
    if textblob_pol is not None:
        v_sources.append(("textblob", (textblob_pol + 1) / 2, 0.15))
    if goemo_probs:
        pos = sum(goemo_probs.get(e, 0) for e in POS_EMOTIONS)
        neg = sum(goemo_probs.get(e, 0) for e in NEG_EMOTIONS)
        v_sources.append(("goemo", clamp((pos - neg + 1) / 2, 0, 1), 0.10))

    v_norm = weighted_avg(v_sources) if v_sources else 0.5
    dv = v_norm * 110 - 55  # map 0..1 to -55..+55

    # --- Arousal (da) ---
    a_sources = []
    if nrc_vad is not None:
        a_sources.append(("nrc", nrc_vad[1], 0.60))
    if goemo_probs:
        goemo_arousal = sum(goemo_probs.get(e, 0) * AROUSAL_MAP.get(e, 0.1)
                           for e in goemo_probs)
        a_sources.append(("goemo", clamp(goemo_arousal, 0, 1), 0.25))
    # VADER/TextBlob lack arousal; use absolute compound as weak proxy
    if vader_compound is not None:
        a_sources.append(("vader_abs", abs(vader_compound) * 0.5 + 0.25, 0.15))

    a_norm = weighted_avg(a_sources) if a_sources else 0.5
    da = a_norm * 90 - 45  # map 0..1 to -45..+45

    # --- Dominance (dd) ---
    d_sources = []
    if nrc_vad is not None:
        d_sources.append(("nrc", nrc_vad[2], 0.70))
    if goemo_probs:
        # Dominant emotions vs submissive
        dom_emos = {"anger", "pride", "admiration", "excitement", "disgust", "disapproval"}
        sub_emos = {"fear", "nervousness", "sadness", "grief", "embarrassment", "remorse"}
        dom_score = sum(goemo_probs.get(e, 0) for e in dom_emos)
        sub_score = sum(goemo_probs.get(e, 0) for e in sub_emos)
        d_sources.append(("goemo", clamp((dom_score - sub_score + 1) / 2, 0, 1), 0.30))

    d_norm = weighted_avg(d_sources) if d_sources else 0.5
    dd = d_norm * 80 - 40  # map 0..1 to -40..+40

    # --- Urgency (du) ---
    if goemo_probs:
        du = sum(goemo_probs.get(e, 0) * URGENCY_MAP.get(e, 0)
                 for e in goemo_probs) * 60
    elif nrc_vad is not None:
        # Fallback: urgency from arousal * (1 - valence) -- high arousal + negative = urgent
        du = (nrc_vad[1] * (1 - nrc_vad[0])) * 30
    else:
        du = 0

    # --- Gravity (dg) ---
    if goemo_probs:
        dg = sum(goemo_probs.get(e, 0) * GRAVITY_MAP.get(e, 0)
                 for e in goemo_probs) * 40
    elif nrc_vad is not None:
        # Fallback: gravity from valence + dominance
        dg = ((nrc_vad[0] - 0.5) * 20 + (nrc_vad[2] - 0.5) * 15)
    else:
        dg = 0

    return (
        round(clamp(dv, -55, 55)),
        round(clamp(da, -45, 45)),
        round(clamp(dd, -40, 40)),
        round(clamp(du, 0, 60)),
        round(clamp(dg, -40, 40)),
    )


# ---------------------------------------------------------------------------
# Step 5: Output
# ---------------------------------------------------------------------------

def write_forces_file(results, path):
    """Write synthesized forces as a Python dict file."""
    print(f"\nWriting {len(results)} entries to {path.name}...")
    with open(path, "w") as f:
        f.write("# Auto-generated by synthesize_forces.py\n")
        f.write("# Multi-model VADUG synthesis: NRC VAD + GoEmotions + VADER + TextBlob\n")
        f.write(f"# {len(results)} words\n")
        f.write("#\n")
        f.write("# Format: \"word\": (dv, da, dd, du, dg)\n")
        f.write("# dv=valence(-55..+55), da=arousal(-45..+45), dd=dominance(-40..+40),\n")
        f.write("# du=urgency(0..60), dg=gravity(-40..+40)\n")
        f.write("#\n")
        f.write("# To merge into simulator.py:\n")
        f.write("#   from benchmarks.synthesized_forces import SYNTHESIZED_FORCES\n")
        f.write("#   WORD_FORCES.update(SYNTHESIZED_FORCES)\n\n")
        f.write("SYNTHESIZED_FORCES = {\n")
        for word in sorted(results.keys()):
            forces = results[word]
            f.write(f'    "{word}": {forces},\n')
        f.write("}\n")


def write_report(results, model_data, path):
    """Write human-readable synthesis report."""
    print(f"Writing report to {path.name}...")

    # Analyze
    valences = {w: f[0] for w, f in results.items()}
    sorted_by_v = sorted(valences.items(), key=lambda x: x[1])

    # Distribution buckets
    strong_neg = sum(1 for v in valences.values() if v <= -30)
    mild_neg = sum(1 for v in valences.values() if -30 < v <= -10)
    neutral = sum(1 for v in valences.values() if -10 < v < 10)
    mild_pos = sum(1 for v in valences.values() if 10 <= v < 30)
    strong_pos = sum(1 for v in valences.values() if v >= 30)

    # Model agreement: how often models agree on polarity direction
    agreements = 0
    disagreements = 0
    disagree_words = []
    vader_data = model_data.get("vader", {})
    textblob_data = model_data.get("textblob", {})
    nrc_data = model_data.get("nrc", {})

    for w in results:
        polarities = []
        if w in vader_data and vader_data[w] != 0:
            polarities.append(("vader", 1 if vader_data[w] > 0 else -1))
        if w in textblob_data and textblob_data[w] != 0:
            polarities.append(("textblob", 1 if textblob_data[w] > 0 else -1))
        if w in nrc_data:
            nrc_v = nrc_data[w][0]
            if abs(nrc_v - 0.5) > 0.05:
                polarities.append(("nrc", 1 if nrc_v > 0.5 else -1))

        if len(polarities) >= 2:
            signs = set(s for _, s in polarities)
            if len(signs) == 1:
                agreements += 1
            else:
                disagreements += 1
                disagree_words.append((w, polarities, results[w]))

    # Sort disagreements by magnitude of final valence (most impactful first)
    disagree_words.sort(key=lambda x: abs(x[2][0]), reverse=True)

    with open(path, "w") as f:
        f.write("VADUG Force Synthesis Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total words processed: {len(results)}\n")
        f.write("Models used: NRC VAD, GoEmotions, VADER, TextBlob\n\n")

        f.write("Model Coverage\n")
        f.write("-" * 40 + "\n")
        f.write(f"  NRC VAD:    {len(nrc_data)} words\n")
        f.write(f"  GoEmotions: {len(model_data.get('goemo', {}))} words\n")
        f.write(f"  VADER:      {len(vader_data)} words\n")
        f.write(f"  TextBlob:   {len(textblob_data)} words\n\n")

        f.write("Model Agreement on Polarity\n")
        f.write("-" * 40 + "\n")
        total_compared = agreements + disagreements
        if total_compared > 0:
            f.write(f"  Agree:    {agreements} ({100 * agreements / total_compared:.1f}%)\n")
            f.write(f"  Disagree: {disagreements} ({100 * disagreements / total_compared:.1f}%)\n\n")

        f.write("Valence Distribution\n")
        f.write("-" * 40 + "\n")
        total = len(results)
        f.write(f"  Strong negative (dv <= -30): {strong_neg:6d} ({100 * strong_neg / total:.1f}%)\n")
        f.write(f"  Mild negative   (-30 < dv <= -10): {mild_neg:6d} ({100 * mild_neg / total:.1f}%)\n")
        f.write(f"  Neutral         (-10 < dv < 10):   {neutral:6d} ({100 * neutral / total:.1f}%)\n")
        f.write(f"  Mild positive   (10 <= dv < 30):   {mild_pos:6d} ({100 * mild_pos / total:.1f}%)\n")
        f.write(f"  Strong positive (dv >= 30):        {strong_pos:6d} ({100 * strong_pos / total:.1f}%)\n\n")

        f.write("Top 30 Most Positive Words\n")
        f.write("-" * 40 + "\n")
        for w, v in sorted_by_v[-30:][::-1]:
            tup = results[w]
            f.write(f"  {w:25s} dv={tup[0]:+4d} da={tup[1]:+4d} dd={tup[2]:+4d} du={tup[3]:3d} dg={tup[4]:+4d}\n")
        f.write("\n")

        f.write("Top 30 Most Negative Words\n")
        f.write("-" * 40 + "\n")
        for w, v in sorted_by_v[:30]:
            tup = results[w]
            f.write(f"  {w:25s} dv={tup[0]:+4d} da={tup[1]:+4d} dd={tup[2]:+4d} du={tup[3]:3d} dg={tup[4]:+4d}\n")
        f.write("\n")

        f.write("Top 30 Words Where Models Disagree Most\n")
        f.write("-" * 60 + "\n")
        for w, pols, tup in disagree_words[:30]:
            pol_str = ", ".join(f"{name}={'pos' if s > 0 else 'neg'}" for name, s in pols)
            f.write(f"  {w:25s} [{pol_str}] -> dv={tup[0]:+4d}\n")
        f.write("\n")

    print("Report written.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # --- Step 1: Word list ---
    word_list = build_word_list()

    # --- Step 2: Existing forces ---
    print("\n=== Step 2: Loading existing forces ===")
    existing = load_existing_forces()
    nrc_data = load_nrc_forces()
    print(f"  Existing force entries: {len(existing)}")
    print(f"  NRC VAD entries: {len(nrc_data)}")

    # Words to process: everything in word_list (we re-synthesize with more models)
    all_target_words = word_list
    print(f"\n  Target words for synthesis: {len(all_target_words)}")

    # --- Step 3: Load checkpoint if available ---
    goemo_results = {}
    vader_results = {}
    textblob_results = {}
    checkpoint_words_done = set()

    if CHECKPOINT_FILE.exists():
        print("\n  Loading checkpoint...")
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                checkpoint = json.load(f)
            checkpoint_words_done = set(checkpoint.get("words_done", []))
            print(f"  Checkpoint has {len(checkpoint_words_done)} words already processed")
        except Exception as e:
            print(f"  Checkpoint load failed: {e}")

    words_to_process = [w for w in all_target_words if w not in checkpoint_words_done]
    print(f"  Words remaining to process: {len(words_to_process)}")

    # --- Step 3a: VADER (fast, CPU) ---
    print("\n=== Step 3a: Running VADER ===")
    t1 = time.time()
    vader_results = run_vader_batch(all_target_words)
    print(f"  VADER done: {len(vader_results)} words in {time.time() - t1:.1f}s")

    # --- Step 3b: TextBlob (fast, CPU) ---
    print("\n=== Step 3b: Running TextBlob ===")
    t1 = time.time()
    textblob_results = run_textblob_batch(all_target_words)
    print(f"  TextBlob done: {len(textblob_results)} words in {time.time() - t1:.1f}s")

    # --- Step 3c: GoEmotions (GPU, batched) ---
    print("\n=== Step 3c: Running GoEmotions classifier ===")
    print("  Loading model SamLowe/roberta-base-go_emotions...")
    t1 = time.time()

    from transformers import pipeline as hf_pipeline
    go_emo = hf_pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        truncation=True,
        device=0,  # GPU
    )
    print(f"  Model loaded in {time.time() - t1:.1f}s")

    # Process in chunks of 5000 for checkpointing, batches of 100 for GPU
    CHUNK_SIZE = 5000
    BATCH_SIZE = 100
    goemo_results = {}

    for chunk_start in range(0, len(words_to_process), CHUNK_SIZE):
        chunk = words_to_process[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_goemo = run_goemo_batch(chunk, go_emo, batch_size=BATCH_SIZE)
        goemo_results.update(chunk_goemo)

        # Progress
        done = chunk_start + len(chunk)
        elapsed = time.time() - t1
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(words_to_process) - done) / rate if rate > 0 else 0
        print(f"  GoEmo progress: {done}/{len(words_to_process)} "
              f"({100 * done / len(words_to_process):.1f}%) "
              f"[{rate:.0f} words/s, ETA {eta:.0f}s]")

        # Save checkpoint
        checkpoint_words_done.update(chunk)
        try:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"words_done": list(checkpoint_words_done)}, f)
        except Exception:
            pass

    print(f"  GoEmotions done: {len(goemo_results)} words in {time.time() - t1:.1f}s")

    # Free GPU memory
    del go_emo
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # --- Step 4: Synthesize ---
    print("\n=== Step 4: Synthesizing VADUG forces ===")
    t1 = time.time()
    synthesized = {}

    for i, word in enumerate(all_target_words):
        nrc_vad = nrc_data.get(word)
        goemo = goemo_results.get(word) or {}
        vader_c = vader_results.get(word)
        tb_pol = textblob_results.get(word)

        forces = synthesize(word, nrc_vad, goemo, vader_c, tb_pol)
        synthesized[word] = forces

        if (i + 1) % 10000 == 0:
            print(f"  Synthesized {i + 1}/{len(all_target_words)}")

    print(f"  Synthesis done: {len(synthesized)} words in {time.time() - t1:.1f}s")

    # --- Step 5: Output ---
    print("\n=== Step 5: Writing output ===")
    model_data = {
        "nrc": nrc_data,
        "goemo": goemo_results,
        "vader": vader_results,
        "textblob": textblob_results,
    }

    write_forces_file(synthesized, OUTPUT_FORCES)
    write_report(synthesized, model_data, OUTPUT_REPORT)

    # Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("  Checkpoint file removed.")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! {len(synthesized)} words synthesized in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"Output: {OUTPUT_FORCES.name}, {OUTPUT_REPORT.name}")


if __name__ == "__main__":
    main()
