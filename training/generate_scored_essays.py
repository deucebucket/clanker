#!/usr/bin/env python3
"""Generate scored essay training data — emotional pangrams.

For each emotional category, generates sentences that use EVERY word in the
vocabulary. Each sentence is scored by the V2 engine with full VADUG traces.

The output is training data: the model learns the MATH, not just labels.

Each training example includes:
  - text: the sentence
  - vadug: engine's V/A/D/U/G scores
  - trace: per-word breakdown (word, role, force applied, coefficient, state after)
  - category: which emotional band this sentence exercises
  - syntax_type: what sentence structure was used

Output: training/data/scored_essays.jsonl

Usage:
    python3 training/generate_scored_essays.py
"""

import json
import os
import sys
import random
import re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.pendulum_v2 import PendulumV2
from demo.forces_curated import EMOTIONAL_VOCABULARY
from demo.context_operators import CONTEXT_OPERATORS

# ---------------------------------------------------------------------------
# Sentence templates — each exercises a different syntax/equation type
# ---------------------------------------------------------------------------

# Templates use {word} as the payload placeholder
# Multiple templates per syntax type to get variety

SYNTAX_TEMPLATES = {
    "direct_statement": [
        "I am {word}.",
        "I feel {word}.",
        "I am really {word} right now.",
        "I feel so {word} about this.",
        "I am extremely {word} today.",
        "I feel deeply {word}.",
        "I am genuinely {word}.",
        "I feel absolutely {word} about everything.",
    ],
    "negated_statement": [
        "I am not {word}.",
        "I don't feel {word} at all.",
        "I never feel {word}.",
        "I can't say I feel {word}.",
        "Nobody would call this {word}.",
        "I wouldn't say I'm {word}.",
        "I'm not even remotely {word}.",
    ],
    "question": [
        "Do you feel {word}?",
        "Are you {word} about this?",
        "Why am I so {word}?",
        "How can you be so {word}?",
        "Don't you feel {word}?",
        "Isn't this {word}?",
    ],
    "other_person": [
        "She seems really {word}.",
        "He is {word} about the situation.",
        "They were {word} when I told them.",
        "She looked {word} after hearing the news.",
        "He became extremely {word}.",
    ],
    "past_tense": [
        "I was {word} yesterday.",
        "I felt {word} when it happened.",
        "I was really {word} about the whole thing.",
        "I had been feeling {word} for weeks.",
        "I was so {word} that I couldn't speak.",
    ],
    "hypothetical": [
        "I would feel {word} if that happened.",
        "I might be {word} about this.",
        "I could feel {word} in that situation.",
        "Maybe I should feel {word}.",
        "Perhaps this would make me {word}.",
    ],
    "hedged": [
        "I'm somewhat {word}, I think.",
        "I'm kind of {word}, maybe.",
        "Generally I feel {word} about these things.",
        "Sometimes I feel a little {word}.",
        "I'm slightly {word}, if I'm being honest.",
    ],
    "intensified": [
        "I am absolutely {word}.",
        "I am completely {word} beyond belief.",
        "I am incredibly {word} right now.",
        "I am truly deeply {word}.",
        "I am totally {word} and I mean it.",
    ],
    "comparative": [
        "I feel more {word} than ever before.",
        "This is the most {word} I have ever been.",
        "I am far more {word} than yesterday.",
        "Nothing has ever made me this {word}.",
    ],
    "compound": [
        "I feel {word} and I don't know what to do.",
        "I am {word} but I'm trying to stay strong.",
        "I feel {word} yet somehow also relieved.",
        "I am {word} and it's getting worse.",
        "I feel {word} because of what happened.",
    ],
    "possessive": [
        "My {word} feelings are overwhelming.",
        "My heart is {word}.",
        "This {word} feeling won't go away.",
    ],
    "article_distanced": [
        "A {word} situation unfolded.",
        "The {word} atmosphere was palpable.",
        "It was a truly {word} experience.",
        "The {word} news spread quickly.",
    ],
}


def categorize_word(word, force):
    """Assign a word to an emotional category based on its force vector."""
    dv, da, dd, du, dg = force
    if dv < -80 and da > 80:
        return "rage"
    elif dv < -80 and da < 40:
        return "despair"
    elif dv < -50:
        return "sadness"
    elif dv < -15:
        return "mild_negative"
    elif dv > 80:
        return "ecstasy"
    elif dv > 50:
        return "joy"
    elif dv > 15:
        return "mild_positive"
    else:
        return "near_neutral"


def word_fits_template(word, template):
    """Check if a word grammatically fits a template.

    Simple heuristic — adjective/emotion words fit most templates.
    Nouns need different handling.
    """
    # Most templates work with adjectives and emotion words
    # Skip words that are clearly nouns in "I feel {word}" type sentences
    noun_indicators = {"abuse", "agony", "bliss", "chaos", "comfort", "crisis",
                       "cruelty", "despair", "disgust", "distress", "empathy",
                       "envy", "fury", "grief", "guilt", "hatred", "horror",
                       "injustice", "jealousy", "joy", "love", "mercy",
                       "misery", "panic", "passion", "pity", "pride",
                       "rage", "regret", "relief", "resentment", "revenge",
                       "shame", "sorrow", "spite", "sympathy", "terror",
                       "trauma", "triumph", "trust", "wrath"}

    # For "I feel" templates, nouns need "filled with" instead
    if word in noun_indicators:
        if "I feel {word}" in template or "I am {word}" in template:
            return False
    return True


def generate_sentence(word, force, syntax_type):
    """Generate a scored sentence using a word in a specific syntax type."""
    templates = SYNTAX_TEMPLATES.get(syntax_type, SYNTAX_TEMPLATES["direct_statement"])

    # Try templates until one fits
    for template in templates:
        if word_fits_template(word, template):
            sentence = template.format(word=word)
            return sentence

    # Fallback: simple carrier sentence
    return f"The feeling of {word} is strong."


def main():
    engine = PendulumV2()
    output_path = os.path.join(os.path.dirname(__file__), "data", "scored_essays.jsonl")

    # Group words by category
    categories = defaultdict(list)
    for word, force in EMOTIONAL_VOCABULARY.items():
        cat = categorize_word(word, force)
        categories[cat].append((word, force))

    syntax_types = list(SYNTAX_TEMPLATES.keys())
    total = 0
    per_cat = defaultdict(int)

    print(f"\n{'='*70}")
    print(f"  GENERATING SCORED ESSAY TRAINING DATA")
    print(f"  Vocabulary: {len(EMOTIONAL_VOCABULARY)} words")
    print(f"  Syntax types: {len(syntax_types)}")
    print(f"  Categories: {len(categories)}")
    print(f"{'='*70}\n")

    with open(output_path, "w") as f:
        for cat, words in sorted(categories.items()):
            print(f"  {cat}: {len(words)} words...", end=" ", flush=True)
            cat_count = 0

            for word, force in words:
                # Generate one sentence per syntax type for this word
                for syntax_type in syntax_types:
                    sentence = generate_sentence(word, force, syntax_type)

                    # Score with engine
                    vadug, trace = engine.process_text(sentence)

                    # Build training example
                    example = {
                        "text": sentence,
                        "v": vadug.v,
                        "a": vadug.a,
                        "d": vadug.d,
                        "u": vadug.u,
                        "g": vadug.g,
                        "category": cat,
                        "syntax_type": syntax_type,
                        "payload_word": word,
                        "payload_force": list(force),
                        "trace": [
                            {
                                "word": t["word"],
                                "role": t["role"],
                                "v": t["v"],
                                "note": t["note"],
                            }
                            for t in trace
                        ],
                    }

                    f.write(json.dumps(example) + "\n")
                    total += 1
                    cat_count += 1

            per_cat[cat] = cat_count
            print(f"{cat_count} examples")

    print(f"\n  {'='*70}")
    print(f"  TOTAL: {total} scored training examples")
    print(f"  Output: {output_path}")
    print(f"  {'='*70}")

    # Summary
    print(f"\n  Per category:")
    for cat, count in sorted(per_cat.items()):
        print(f"    {cat:<20}: {count:>6} examples")

    # Verify: spot check
    print(f"\n  Spot check (first 5 examples from each category):")
    with open(output_path) as f:
        seen = defaultdict(int)
        for line in f:
            ex = json.loads(line)
            cat = ex["category"]
            if seen[cat] < 2:
                print(f"    [{cat}] V={ex['v']:>3} \"{ex['text'][:65]}\"")
                seen[cat] += 1


if __name__ == "__main__":
    main()
