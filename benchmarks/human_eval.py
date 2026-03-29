#!/usr/bin/env python3
"""Human agreement evaluation — the REAL test of whether the math works.

The academic benchmarks compress 5D VADUG to 1D pos/neg. That's not a
real test. The real test: do HUMANS agree with the engine's 5D output?

This script:
1. Generates a diverse set of 200 sentences across emotional categories
2. Runs the engine on each one → 5D VADUG output
3. Exports a CSV for human raters to score (V, A, D, U, G each 0-255)
4. When human ratings come back, calculates correlation per dimension
5. Reports where the engine agrees with humans and where it diverges

Target: Pearson r > 0.7 on each dimension = the math works
        Pearson r < 0.5 on any dimension = that dimension needs fixing

Usage:
    # Step 1: Generate the evaluation set
    python3 benchmarks/human_eval.py --generate

    # Step 2: After humans rate, calculate agreement
    python3 benchmarks/human_eval.py --evaluate ratings.csv
"""

import json
import csv
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from demo.pendulum_v2 import PendulumV2


EVAL_SENTENCES = [
    # --- Clear emotions (should be easy) ---
    "I am so happy right now",
    "I hate everything about this",
    "I am terrified of what might happen",
    "I love you more than anything",
    "I am disgusted by what I saw",
    "I am proud of what we accomplished",
    "I feel guilty about what I did",
    "I am excited about tomorrow",
    "I am ashamed of myself",
    "I am grateful for your help",

    # --- Intensity variations ---
    "I am a little annoyed",
    "I am extremely angry",
    "I am somewhat worried",
    "I am absolutely devastated",
    "I am mildly amused",
    "I am incredibly proud",
    "I am slightly nervous",
    "I am utterly heartbroken",

    # --- Negation ---
    "I am not happy",
    "I am not sad",
    "This is not okay",
    "I do not feel safe",
    "Nothing is wrong",
    "I never said that",
    "I am not angry just disappointed",

    # --- Hedging / uncertainty ---
    "I think I might be upset",
    "Maybe I should be worried",
    "I guess things could be worse",
    "I am probably overthinking this",
    "It is possibly a problem",

    # --- Sarcasm / deflection ---
    "Oh great another Monday",
    "Yeah that went really well",
    "Whatever I am fine",
    "Sure why not",
    "Thanks for nothing",
    "Oh how delightful",

    # --- Complex / ambiguous ---
    "It is what it is",
    "I had a day",
    "Could be worse could be better",
    "I do not know how to feel about this",
    "That was something",
    "I mean I guess",

    # --- Crisis indicators ---
    "I want to die",
    "I want to die everything is hopeless",
    "Nobody would notice if I was gone",
    "I just want it all to stop",
    "What is even the point",
    "I wish I was dead",

    # --- Crisis-adjacent but NOT crisis ---
    "I am so tired of everything",
    "I had the worst day ever",
    "I feel like giving up on this project",
    "This is killing me",
    "I could die of embarrassment",
    "I am dead tired",

    # --- Reported / evidential ---
    "She said she is devastated",
    "The patient reports feeling anxious",
    "According to him everything is terrible",
    "He claims to be happy",

    # --- Conditional ---
    "If that happened I would be furious",
    "Unless you stop I am done",
    "Assuming the worst this is bad",

    # --- Evoker-loaded ---
    "I lost my job before the wedding",
    "The cancer diagnosis changed everything",
    "My mother passed away last month",
    "The children are safe now",
    "The war is finally over",

    # --- Passive voice ---
    "I was hurt",
    "I was abandoned by everyone",
    "Mistakes were made",
    "I was betrayed by my best friend",

    # --- Social / politeness ---
    "With all due respect you are wrong",
    "Honestly I think you need help",
    "I appreciate your input but no",
    "Bless your heart",

    # --- Everyday mundane ---
    "I went to the store",
    "The weather is nice today",
    "I had coffee this morning",
    "Pass the salt please",
    "The meeting is at three",
    "I need to do laundry",

    # --- Grief ---
    "I still reach for the phone to call her",
    "The house feels so empty without you",
    "I cannot stop crying when I hear that song",
    "Everything reminds me of what I have lost",
    "I miss you more than words can say",

    # --- Joy / gratitude ---
    "This is the best day of my life",
    "I am overflowing with gratitude",
    "My heart is so full right now",
    "I have never been happier",
    "Today was absolutely perfect",

    # --- Mixed / conflicting ---
    "I am happy for you but sad for me",
    "I love this city but I miss home",
    "The surgery went well but recovery will be hard",
    "I got the job but I have to move away from family",
    "I am relieved it is over but exhausted",
]


def generate_eval_set():
    """Generate evaluation CSV with engine predictions."""
    engine = PendulumV2()  # default config for consistency

    output_path = os.path.join(os.path.dirname(__file__), "human_eval_set.csv")
    engine_path = os.path.join(os.path.dirname(__file__), "human_eval_engine.json")

    engine_results = []

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header for human raters
        writer.writerow([
            "id", "sentence",
            "your_V (0=very negative, 128=neutral, 255=very positive)",
            "your_A (0=very calm, 128=moderate, 255=very intense)",
            "your_D (0=helpless, 128=neutral, 255=in control)",
            "your_U (0=no urgency, 128=moderate, 255=critical)",
            "your_G (0=crushing/heavy, 128=neutral, 255=light/floating)",
        ])

        for i, sentence in enumerate(EVAL_SENTENCES):
            writer.writerow([i + 1, sentence, "", "", "", "", ""])

            # Also get engine prediction
            vadug, trace = engine.process_text(sentence)
            engine_results.append({
                "id": i + 1,
                "sentence": sentence,
                "engine_v": vadug.v,
                "engine_a": vadug.a,
                "engine_d": vadug.d,
                "engine_u": vadug.u,
                "engine_g": vadug.g,
            })

    with open(engine_path, "w") as f:
        json.dump(engine_results, f, indent=2)

    print(f"Generated {len(EVAL_SENTENCES)} sentences")
    print(f"Human rating form: {output_path}")
    print(f"Engine predictions: {engine_path}")
    print()
    print("Instructions for human raters:")
    print("  1. Open the CSV in a spreadsheet")
    print("  2. For each sentence, rate each dimension 0-255")
    print("  3. V: How positive/negative does this FEEL? (0=very bad, 255=very good)")
    print("  4. A: How intense/calm? (0=numb, 255=on fire)")
    print("  5. D: How in control/helpless does the speaker seem? (0=helpless, 255=powerful)")
    print("  6. U: How urgent? (0=no rush, 255=emergency)")
    print("  7. G: How heavy/light? (0=crushing weight, 255=floating)")
    print("  8. Save and return the CSV")


def evaluate_agreement(ratings_path):
    """Compare human ratings to engine predictions."""
    import statistics

    engine_path = os.path.join(os.path.dirname(__file__), "human_eval_engine.json")
    with open(engine_path) as f:
        engine_results = {r["id"]: r for r in json.load(f)}

    # Load human ratings
    human_ratings = {}
    with open(ratings_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["id"])
            try:
                human_ratings[sid] = {
                    "v": int(row.get("your_V (0=very negative, 128=neutral, 255=very positive)", 0) or 0),
                    "a": int(row.get("your_A (0=very calm, 128=moderate, 255=very intense)", 0) or 0),
                    "d": int(row.get("your_D (0=helpless, 128=neutral, 255=in control)", 0) or 0),
                    "u": int(row.get("your_U (0=no urgency, 128=moderate, 255=critical)", 0) or 0),
                    "g": int(row.get("your_G (0=crushing/heavy, 128=neutral, 255=light/floating)", 0) or 0),
                }
            except (ValueError, TypeError):
                continue

    if not human_ratings:
        print("No valid human ratings found!")
        return

    # Calculate Pearson correlation per dimension
    dims = [("V", "v", "engine_v"), ("A", "a", "engine_a"),
            ("D", "d", "engine_d"), ("U", "u", "engine_u"), ("G", "g", "engine_g")]

    print(f"\n{'='*70}")
    print(f"  HUMAN AGREEMENT EVALUATION — The Real Test")
    print(f"  {len(human_ratings)} sentences rated by humans")
    print(f"{'='*70}\n")

    for dim_name, human_key, engine_key in dims:
        human_vals = []
        engine_vals = []
        for sid in human_ratings:
            if sid in engine_results:
                human_vals.append(human_ratings[sid][human_key])
                engine_vals.append(engine_results[sid][engine_key])

        if len(human_vals) < 3:
            print(f"  {dim_name}: insufficient data")
            continue

        # Pearson correlation
        n = len(human_vals)
        mean_h = statistics.mean(human_vals)
        mean_e = statistics.mean(engine_vals)
        num = sum((h - mean_h) * (e - mean_e) for h, e in zip(human_vals, engine_vals))
        den_h = sum((h - mean_h) ** 2 for h in human_vals) ** 0.5
        den_e = sum((e - mean_e) ** 2 for e in engine_vals) ** 0.5
        r = num / (den_h * den_e) if den_h * den_e > 0 else 0

        # MAE
        mae = statistics.mean(abs(h - e) for h, e in zip(human_vals, engine_vals))

        verdict = "PASS" if r > 0.7 else "WEAK" if r > 0.5 else "FAIL"
        print(f"  {dim_name}: r={r:.3f}  MAE={mae:.1f}  [{verdict}]")

    print()
    print("  Target: r > 0.7 on all dimensions = math confirmed")
    print("  r > 0.5 = directionally correct but noisy")
    print("  r < 0.5 = dimension needs fundamental rework")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate evaluation set")
    parser.add_argument("--evaluate", type=str, help="Evaluate against human ratings CSV")
    args = parser.parse_args()

    if args.generate:
        generate_eval_set()
    elif args.evaluate:
        evaluate_agreement(args.evaluate)
    else:
        generate_eval_set()


if __name__ == "__main__":
    main()
