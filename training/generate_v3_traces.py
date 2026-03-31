#!/usr/bin/env python3
"""Generate V3 engine traces for training.

Runs every available sentence through the V3 engine (compute_vadug),
captures word roles, structures, and VADUG coordinates.

Output: training/data/v3_traces.jsonl
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from engine.pendulum import compute_vadug

# Source files to pull sentences from
SOURCES = [
    ("phase1.jsonl", ["english"]),
    ("phase1_expanded.jsonl", ["english"]),
    ("emobank_significant.jsonl", ["english", "text"]),
    ("emobank_vadug.jsonl", ["english", "text"]),
    ("balance_neutral.jsonl", ["english"]),
    ("balance_positive.jsonl", ["english"]),
    ("balance_pos_high.jsonl", ["english"]),
    ("balance_pos_mild.jsonl", ["english"]),
    ("error_corrections.jsonl", ["english", "text"]),
    ("engine_errors.jsonl", ["text"]),
    ("negation_calibration.jsonl", ["english"]),
    ("negation_pairs.jsonl", ["english"]),
    ("gap_analysis.jsonl", ["english", "text"]),
    ("math_flashcards.jsonl", ["english"]),
    ("emobank_neutral_sample.jsonl", ["english", "text"]),
    ("empathetic_dialogues.jsonl", ["english", "text"]),
    ("pangram_essays.jsonl", ["english"]),
]

# Additional novel sentences for new patterns
NOVEL_SENTENCES = [
    # BETRAYAL
    "my wife cheated on me with my best friend",
    "she betrayed my trust",
    "he lied to me about everything",
    "my best friend stabbed me in the back",
    "my ex is dating my friend",
    "she cheated on me",
    "he deceived everyone",
    "my boyfriend lied to me about everything",
    # BRAVADO
    "haha yeah im totally okay",
    "lol im fine",
    "im totally fine",
    "haha dont worry about me",
    "lol whatever its fine",
    "haha yeah im good",
    "lmao im not even mad",
    # VICTIMIZATION
    "boyfriend hit me",
    "she left me",
    "he ignored me for a week",
    "they kicked me out",
    "she told everyone my secret",
    "he ghosted me after 3 years",
    "they laughed at me",
    "my mom threw away my stuff",
    "he deleted all my photos",
    "she blocked me on everything",
    "she replaced me",
    # CALLING_OUT
    "why do you do that",
    "why are you like that",
    "why would you say that",
    "how could you do this",
    "why do you always do that",
    # DIRECTED_POSITIVE (passive aggression)
    "good for you",
    "must be nice",
    "i hope youre happy",
    "glad someone is having fun",
    "nice of you to finally show up",
    # MINIMIZER
    "it was just a joke",
    "youre too sensitive",
    "youre overreacting",
    "youre being dramatic",
    # Resignation
    "whatever",
    "k",
    "cool",
    "sure",
    "idc",
    "nevermind",
    # Chopper
    "i love you but i cant do this",
    "i want to be happy but i cant",
    "i was fine but now everything hurts",
    "it was hard but we made it",
    # Guilt tripping
    "after everything i did for you",
    "nobody cares about me anyway",
    "nothing i do is ever good enough",
    "i sacrificed everything for you",
    "fine just forget about me",
    # Deflection
    "i dont wanna talk about it",
    "drop it",
    "forget i said anything",
    "im over it",
    "i said im fine",
    # Vulnerability
    "i dont know what to do anymore",
    "i just feel so lost",
    "everything feels heavy",
    "i feel like im drowning",
    "nobody understands",
    "im scared of whats happening to me",
    "i keep messing everything up",
    # Positive (must stay positive)
    "you make me so happy",
    "im so proud of you",
    "you did amazing",
    "i love spending time with you",
    "thank you for being you",
    "best birthday ever",
    "I love my mom",
    "my baby took her first steps",
    "we are getting married",
    "i feel so lucky",
]


def load_sentences():
    """Load and deduplicate sentences from all source files."""
    seen = set()
    sentences = []
    data_dir = os.path.join(PROJECT_ROOT, "training", "data")

    for filename, fields in SOURCES:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP (not found): {filename}")
            continue

        count = 0
        with open(filepath) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = None
                for field in fields:
                    if field in d:
                        text = d[field]
                        break
                if text and text.strip() and text.strip().lower() not in seen:
                    seen.add(text.strip().lower())
                    sentences.append(text.strip())
                    count += 1
        if count > 0:
            print(f"  Loaded {count} from {filename}")

    # Add novel sentences
    novel_count = 0
    for s in NOVEL_SENTENCES:
        if s.lower() not in seen:
            seen.add(s.lower())
            sentences.append(s)
            novel_count += 1
    print(f"  Loaded {novel_count} novel sentences for new patterns")

    return sentences


def trace_sentence(text):
    """Run one sentence through V3 engine and capture trace."""
    vadug, trace_data = compute_vadug(text)
    word_roles = []
    for entry in trace_data["trace"]:
        word_roles.append({
            "word": entry["word"],
            "role": entry["role"],
        })
    structures = []
    for sm in trace_data["structures"]:
        structures.append({
            "pattern": sm.pattern,
            "confidence": round(sm.confidence, 3),
        })
    return {
        "english": text,
        "vadug": [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g, vadug.w, vadug.i],
        "word_roles": word_roles,
        "structures": structures,
    }


def main():
    print("Loading sentences...")
    sentences = load_sentences()
    print(f"Total unique sentences: {len(sentences)}")

    out_path = os.path.join(PROJECT_ROOT, "training", "data", "v3_traces.jsonl")
    print(f"\nGenerating V3 traces -> {out_path}")

    t0 = time.time()
    count = 0
    pattern_counts = {}

    with open(out_path, "w") as f:
        for i, text in enumerate(sentences):
            try:
                trace = trace_sentence(text)
                f.write(json.dumps(trace) + "\n")
                count += 1
                for s in trace["structures"]:
                    p = s["pattern"]
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
            except Exception as e:
                print(f"  ERROR on '{text[:50]}': {e}")
                continue

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  {i+1}/{len(sentences)} ({rate:.0f}/sec)")

    elapsed = time.time() - t0
    print(f"\nDone: {count} traces in {elapsed:.1f}s ({count/elapsed:.0f}/sec)")
    print(f"\nPattern distribution:")
    for p, c in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {p:25s} {c:6d}")


if __name__ == "__main__":
    main()
