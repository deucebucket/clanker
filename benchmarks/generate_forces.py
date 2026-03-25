#!/usr/bin/env python3
"""
Generate WORD_FORCES entries from the NRC VAD Lexicon.

Uses the NRC VAD Lexicon (Saif Mohammad) to automatically produce
Clanker-lang force tuples (dv, da, dd, du, dg) from academically-measured
Valence, Arousal, and Dominance scores.

Supports both:
  - NRC VAD v1  (0-1 scale, ~20K words, neutral=0.5)
  - NRC VAD v2.1 (bipolar -1 to +1, ~44K unigrams, neutral=0)

Usage:
    python benchmarks/generate_forces.py [--version 1|2]

Downloads lexicon data automatically if not present locally.
"""

import csv
import os
import sys
import urllib.request
import zipfile
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Lexicon file paths (relative to benchmarks/)
V1_LEXICON = SCRIPT_DIR / "nrc_vad_lexicon_v1.txt"
V2_LEXICON = SCRIPT_DIR / "nrc_vad_lexicon_v2.txt"

# Download URLs
V1_ZIP_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip"
V2_ZIP_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"

# Output files
PREVIEW_FILE = SCRIPT_DIR / "nrc_forces_preview.txt"
GENERATED_FILE = SCRIPT_DIR / "generated_forces.py"

# Force scaling factors
# V1: 0-1 scale, neutral=0.5 → subtract 0.5 then scale
# V2: -1 to +1 scale, neutral=0 → scale directly (but halve the multiplier)
V1_SCALE = {
    "valence": 110,     # (val - 0.5) * 110 → range -55 to +55
    "arousal": 90,      # (aro - 0.5) * 90  → range -45 to +45
    "dominance": 80,    # (dom - 0.5) * 80  → range -40 to +40
    "urgency_aro": 60,  # arousal component of urgency
    "gravity_val": 40,  # valence component of gravity
    "gravity_dom": 30,  # dominance component of gravity
}

V2_SCALE = {
    "valence": 55,      # val * 55 → range -55 to +55
    "arousal": 45,       # aro * 45 → range -45 to +45
    "dominance": 40,    # dom * 40 → range -40 to +40
    "urgency_aro": 30,  # arousal component of urgency
    "gravity_val": 20,  # valence component of gravity
    "gravity_dom": 15,  # dominance component of gravity
}

# Skip words where all forces are within this threshold (truly neutral)
NEUTRAL_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_lexicon(version: int) -> Path:
    """Download and extract the NRC VAD lexicon if not already present."""
    if version == 1:
        target = V1_LEXICON
        url = V1_ZIP_URL
        inner_path = "NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt"
    else:
        target = V2_LEXICON
        url = V2_ZIP_URL
        inner_path = "NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"

    if target.exists():
        print(f"  Lexicon v{version} already present: {target}")
        return target

    print(f"  Downloading NRC VAD Lexicon v{version} from {url}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "nrc_vad.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(inner_path) as src, open(target, 'wb') as dst:
                dst.write(src.read())
    print(f"  Saved to {target}")
    return target


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------

def load_v1(path: Path) -> dict:
    """Load NRC VAD v1 (0-1 scale, no header, tab-separated: word val aro dom)."""
    lexicon = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            word = parts[0].lower().strip()
            try:
                v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            if word and word.isalpha():
                lexicon[word] = (v, a, d)
    return lexicon


def load_v2(path: Path) -> dict:
    """Load NRC VAD v2.1 (bipolar -1 to +1, header row, tab-separated)."""
    lexicon = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word = row['term'].lower().strip()
            try:
                v = float(row['valence'])
                a = float(row['arousal'])
                d = float(row['dominance'])
            except (ValueError, KeyError):
                continue
            # Only single words (no spaces, hyphens ok but skip for now)
            if word and word.isalpha():
                lexicon[word] = (v, a, d)
    return lexicon


# ---------------------------------------------------------------------------
# Force computation
# ---------------------------------------------------------------------------

def compute_forces_v1(v: float, a: float, d: float) -> tuple:
    """Convert NRC VAD v1 scores (0-1) to Clanker force tuple (dv, da, dd, du, dg)."""
    S = V1_SCALE
    dv = (v - 0.5) * S["valence"]
    da = (a - 0.5) * S["arousal"]
    dd = (d - 0.5) * S["dominance"]
    # Urgency: high arousal + low valence = urgent
    du = max(0, (a - 0.5) * S["urgency_aro"] * (1 - v))
    # Gravity: positive+dominant = rising, negative+submissive = sinking
    dg = (v - 0.5) * S["gravity_val"] + (d - 0.5) * S["gravity_dom"]
    return (round(dv), round(da), round(dd), round(du), round(dg))


def compute_forces_v2(v: float, a: float, d: float) -> tuple:
    """Convert NRC VAD v2.1 scores (-1 to +1) to Clanker force tuple (dv, da, dd, du, dg)."""
    S = V2_SCALE
    dv = v * S["valence"]
    da = a * S["arousal"]
    dd = d * S["dominance"]
    # Urgency: high arousal + negative valence = urgent
    # v is -1 to +1, so (1-v)/2 normalizes to 0-1 where 1=most negative
    du = max(0, a * S["urgency_aro"] * (1 - v) / 2) if a > 0 else 0
    # Gravity: positive+dominant = rising, negative+submissive = sinking
    dg = v * S["gravity_val"] + d * S["gravity_dom"]
    return (round(dv), round(da), round(dd), round(du), round(dg))


def is_neutral(forces: tuple) -> bool:
    """Check if all forces are within neutral threshold."""
    return all(abs(f) <= NEUTRAL_THRESHOLD for f in forces)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_existing_word_forces() -> dict:
    """Parse WORD_FORCES from simulator.py source (avoids import issues).

    Uses ast.literal_eval (safe — only parses Python literals, no code execution).
    """
    import ast
    import re

    sim_path = PROJECT_ROOT / 'demo' / 'simulator.py'
    source = sim_path.read_text(encoding='utf-8')

    match = re.search(r'^WORD_FORCES\s*=\s*\{', source, re.MULTILINE)
    if not match:
        print("  WARNING: Could not find WORD_FORCES in simulator.py")
        return {}

    # Find matching closing brace
    depth = 0
    end = match.end() - 1
    for i in range(match.end() - 1, len(source)):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
        if depth == 0:
            end = i
            break

    dict_str = source[match.start():end + 1]
    dict_str = dict_str[dict_str.index('{'):]
    return ast.literal_eval(dict_str)


def generate(version: int = 2):
    """Main generation pipeline."""
    print(f"\n{'='*70}")
    print(f"  NRC VAD Lexicon → Clanker WORD_FORCES Generator")
    print(f"  Using NRC VAD v{'2.1' if version == 2 else '1'}")
    print(f"{'='*70}\n")

    # Step 1: Ensure lexicon is available
    print("[1/5] Loading lexicon...")
    path = download_lexicon(version)
    if version == 1:
        lexicon = load_v1(path)
        compute_fn = compute_forces_v1
    else:
        lexicon = load_v2(path)
        compute_fn = compute_forces_v2
    print(f"  Loaded {len(lexicon)} words from NRC VAD v{'2.1' if version == 2 else '1'}")

    # Step 2: Load existing WORD_FORCES
    print("\n[2/5] Loading existing WORD_FORCES...")
    existing = load_existing_word_forces()
    print(f"  Existing WORD_FORCES: {len(existing)} entries")

    # Step 3: Compute forces for new words
    print("\n[3/5] Computing forces...")
    new_forces = {}
    skipped_neutral = 0
    skipped_existing = 0

    for word, (v, a, d) in lexicon.items():
        if word in existing:
            skipped_existing += 1
            continue
        forces = compute_fn(v, a, d)
        if is_neutral(forces):
            skipped_neutral += 1
            continue
        new_forces[word] = {
            "nrc": (v, a, d),
            "forces": forces,
        }

    print(f"  NRC words total: {len(lexicon)}")
    print(f"  Already in WORD_FORCES (overlap): {skipped_existing}")
    print(f"  Skipped (neutral): {skipped_neutral}")
    print(f"  NEW words to add: {len(new_forces)}")

    # Step 4: Categorize
    strong_pos = []   # dv >= 25
    mild_pos = []     # 4 <= dv < 25
    mild_neg = []     # -25 < dv <= -4
    strong_neg = []   # dv <= -25

    for word, data in new_forces.items():
        dv = data["forces"][0]
        if dv >= 25:
            strong_pos.append((word, data))
        elif dv >= 4:
            mild_pos.append((word, data))
        elif dv <= -25:
            strong_neg.append((word, data))
        elif dv <= -4:
            mild_neg.append((word, data))
        # else: near-zero valence but non-neutral in other dims

    print(f"\n  Distribution:")
    print(f"    Strong positive (dv >= 25):  {len(strong_pos)}")
    print(f"    Mild positive   (4-24):      {len(mild_pos)}")
    print(f"    Mild negative   (-4 to -24): {len(mild_neg)}")
    print(f"    Strong negative (dv <= -25): {len(strong_neg)}")
    other = len(new_forces) - len(strong_pos) - len(mild_pos) - len(mild_neg) - len(strong_neg)
    print(f"    Other (low valence, high other): {other}")

    # Step 5: Sanity check — top 20 strongest
    sorted_by_val = sorted(new_forces.items(), key=lambda x: x[1]["forces"][0])

    print(f"\n  Top 20 STRONGEST NEGATIVE new words:")
    print(f"  {'Word':<25} {'dv':>5} {'da':>5} {'dd':>5} {'du':>5} {'dg':>5}  (NRC V/A/D)")
    for word, data in sorted_by_val[:20]:
        f = data["forces"]
        n = data["nrc"]
        print(f"  {word:<25} {f[0]:>5} {f[1]:>5} {f[2]:>5} {f[3]:>5} {f[4]:>5}  ({n[0]:+.3f}/{n[1]:+.3f}/{n[2]:+.3f})")

    print(f"\n  Top 20 STRONGEST POSITIVE new words:")
    for word, data in sorted_by_val[-20:]:
        f = data["forces"]
        n = data["nrc"]
        print(f"  {word:<25} {f[0]:>5} {f[1]:>5} {f[2]:>5} {f[3]:>5} {f[4]:>5}  ({n[0]:+.3f}/{n[1]:+.3f}/{n[2]:+.3f})")

    # Step 6: Write outputs
    print(f"\n[4/5] Writing preview to {PREVIEW_FILE}...")
    write_preview(new_forces, lexicon, existing, version)

    print(f"\n[5/5] Writing generated_forces.py to {GENERATED_FILE}...")
    write_generated(new_forces)

    print(f"\nDone! Generated {len(new_forces)} new WORD_FORCES entries.")
    print(f"  Preview:   {PREVIEW_FILE}")
    print(f"  Python:    {GENERATED_FILE}")


def write_preview(new_forces: dict, lexicon: dict, existing: dict, version: int):
    """Write human-readable preview file."""
    # Sort by absolute valence (strongest emotions first)
    sorted_words = sorted(
        new_forces.items(),
        key=lambda x: abs(x[1]["forces"][0]),
        reverse=True
    )

    with open(PREVIEW_FILE, 'w') as f:
        f.write(f"# NRC VAD Lexicon v{'2.1' if version == 2 else '1'} → Clanker WORD_FORCES Preview\n")
        f.write(f"# Generated by benchmarks/generate_forces.py\n")
        f.write(f"#\n")
        f.write(f"# NRC words total:           {len(lexicon)}\n")
        f.write(f"# Already in WORD_FORCES:    {len(existing)} (preserved, not overwritten)\n")
        overlap = sum(1 for w in lexicon if w in existing)
        f.write(f"# Overlap (NRC ∩ existing):  {overlap}\n")
        f.write(f"# New words generated:       {len(new_forces)}\n")
        f.write(f"#\n")
        f.write(f"# Sorted by |dv| (strongest emotions first)\n")
        f.write(f"#\n")
        f.write(f"# {'Word':<25} {'V':>7} {'A':>7} {'D':>7}  │ {'dv':>5} {'da':>5} {'dd':>5} {'du':>5} {'dg':>5}\n")
        f.write(f"# {'─'*25} {'─'*7} {'─'*7} {'─'*7}  │ {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}\n")

        for word, data in sorted_words:
            n = data["nrc"]
            forces = data["forces"]
            f.write(f"  {word:<25} {n[0]:>7.3f} {n[1]:>7.3f} {n[2]:>7.3f}  │ {forces[0]:>5} {forces[1]:>5} {forces[2]:>5} {forces[3]:>5} {forces[4]:>5}\n")


def write_generated(new_forces: dict):
    """Write importable Python dict."""
    # Sort alphabetically for stable output
    sorted_words = sorted(new_forces.items(), key=lambda x: x[0])

    with open(GENERATED_FILE, 'w') as f:
        f.write('# Auto-generated from NRC VAD Lexicon\n')
        f.write(f'# {len(new_forces)} new words not in existing WORD_FORCES\n')
        f.write('#\n')
        f.write('# Format: "word": (dv, da, dd, du, dg)\n')
        f.write('# dv=valence, da=arousal, dd=dominance, du=urgency, dg=gravity\n')
        f.write('#\n')
        f.write('# To merge into simulator.py:\n')
        f.write('#   from benchmarks.generated_forces import GENERATED_FORCES\n')
        f.write('#   WORD_FORCES.update(GENERATED_FORCES)\n')
        f.write('\n')
        f.write('GENERATED_FORCES = {\n')

        for word, data in sorted_words:
            forces = data["forces"]
            f.write(f'    "{word}": ({forces[0]}, {forces[1]}, {forces[2]}, {forces[3]}, {forces[4]}),\n')

        f.write('}\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Clanker WORD_FORCES from NRC VAD Lexicon")
    parser.add_argument("--version", type=int, default=2, choices=[1, 2],
                        help="NRC VAD version: 1 (v1, ~20K words, 0-1 scale) or 2 (v2.1, ~44K words, bipolar)")
    args = parser.parse_args()
    generate(version=args.version)
