#!/usr/bin/env python3
"""Generate math flashcards for the force predictor model.

Instead of random sentences, these are STRUCTURED examples that each
demonstrate ONE mathematical rule. Like times tables for emotions.

Categories:
  1. Single words — learn the base force of each vocabulary word
  2. Amplifier pairs — "very X", "really X", "extremely X" → learn multipliers
  3. Negation pairs — "X" vs "not X" → learn negation decay
  4. Self-reference — "I am X" vs "he is X" vs "it was X" → learn pronoun coefficients
  5. Tense pairs — "I am X" vs "I was X" → learn tense dampening
  6. Idiom sentences — complete idioms in context → learn idiom detection
  7. Bigram sentences — bigram pairs in context → learn bigram forces
  8. Hedging — "maybe X", "perhaps X" → learn hedging coefficients
  9. Sarcasm templates — known sarcasm patterns → learn template detection
  10. Crisis patterns — the ones the user analyzed → learn crisis math
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from demo.pendulum_v2 import PendulumV2
from demo.forces_curated import EMOTIONAL_VOCABULARY
from demo.context_operators import CONTEXT_OPERATORS
from demo.idioms import IDIOMS
from demo.bigrams import BIGRAM_EXPRESSIONS

engine = PendulumV2()


def generate_single_words():
    """Every vocabulary word as a standalone sentence."""
    sentences = []
    for word in EMOTIONAL_VOCABULARY:
        sentences.append(word)
        sentences.append(f"I am {word}")
        sentences.append(f"this is {word}")
    return sentences


def generate_amplifier_pairs():
    """"X" vs "very X" vs "really X" vs "extremely X" for key emotional words."""
    amplifiers = ["very", "really", "extremely", "absolutely", "super", "so", "fucking"]
    emotional_words = [w for w in EMOTIONAL_VOCABULARY
                       if abs(EMOTIONAL_VOCABULARY[w][0]) > 30][:100]
    sentences = []
    for word in emotional_words:
        sentences.append(word)
        for amp in amplifiers:
            sentences.append(f"{amp} {word}")
            sentences.append(f"I am {amp} {word}")
    return sentences


def generate_negation_pairs():
    """"X" vs "not X" for emotional words — learn negation."""
    negators = ["not", "never", "no"]
    emotional_words = [w for w in EMOTIONAL_VOCABULARY
                       if abs(EMOTIONAL_VOCABULARY[w][0]) > 20][:150]
    sentences = []
    for word in emotional_words:
        sentences.append(f"I am {word}")
        for neg in negators:
            sentences.append(f"I am {neg} {word}")
            sentences.append(f"{neg} {word}")
    return sentences


def generate_self_reference():
    """Same word with different pronouns — learn coefficient scaling."""
    pronouns = [
        ("I am", "self"),
        ("I'm", "self"),
        ("my", "possessive"),
        ("he is", "other"),
        ("she is", "other"),
        ("it is", "far"),
        ("they are", "other"),
        ("the", "article"),
    ]
    emotional_words = [w for w in EMOTIONAL_VOCABULARY
                       if abs(EMOTIONAL_VOCABULARY[w][0]) > 25][:80]
    sentences = []
    for word in emotional_words:
        for pronoun, _ in pronouns:
            sentences.append(f"{pronoun} {word}")
    return sentences


def generate_tense_pairs():
    """Same content in present vs past tense — learn tense dampening."""
    pairs = [
        ("I am sad", "I was sad"),
        ("I am happy", "I was happy"),
        ("I am angry", "I was angry"),
        ("I am scared", "I was scared"),
        ("I feel terrible", "I felt terrible"),
        ("I feel great", "I felt great"),
        ("I feel helpless", "I felt helpless"),
        ("this hurts", "this hurt"),
        ("I love you", "I loved you"),
        ("I hate this", "I hated this"),
    ]
    sentences = []
    for present, past in pairs:
        sentences.append(present)
        sentences.append(past)
        sentences.append(f"I still {present.replace('I am ', '').replace('I feel ', 'feel ')}")
    return sentences


def generate_idiom_sentences():
    """Every idiom in a natural sentence context."""
    sentences = []
    for key_tuple, force_tuple in IDIOMS.items():
        phrase = " ".join(key_tuple)
        sentences.append(phrase)
        sentences.append(f"I {phrase}")
        sentences.append(f"I just {phrase}")
    return sentences


def generate_bigram_sentences():
    """Every bigram in context."""
    sentences = []
    for (w1, w2), force in BIGRAM_EXPRESSIONS.items():
        sentences.append(f"{w1} {w2}")
        sentences.append(f"I {w1} {w2}")
        sentences.append(f"this is {w1} {w2}")
    return sentences


def generate_hedging():
    """Hedged emotional words — learn hedging coefficients."""
    hedges = ["maybe", "perhaps", "possibly", "probably", "sometimes",
              "generally", "apparently", "supposedly"]
    emotional_words = [w for w in EMOTIONAL_VOCABULARY
                       if abs(EMOTIONAL_VOCABULARY[w][0]) > 30][:60]
    sentences = []
    for word in emotional_words:
        sentences.append(f"I am {word}")
        for hedge in hedges:
            sentences.append(f"{hedge} {word}")
            sentences.append(f"I am {hedge} {word}")
    return sentences


def generate_crisis_patterns():
    """Known crisis patterns from user analysis."""
    return [
        # want + exit
        "I want it to be over", "I just want it to be over",
        "I want this to end", "I want this to end permanently",
        "I want to vanish", "I want to cease to exist",
        "I want to disappear", "I want to die",
        # can't sustain
        "I cant take this anymore", "I cant do this anymore",
        "I cant keep going", "I cant keep doing this",
        "I cant live like this", "I cant bear this anymore",
        "I cant stand being alive", "Ive had enough of living",
        # nothing left
        "theres nothing left for me", "I have nothing left",
        "I have nothing to live for", "nothing to live for",
        # self-harm
        "I want to hurt myself", "I want to cut myself",
        "Ive been cutting myself", "I started cutting again",
        # comparison (non-crisis for contrast)
        "I am sad", "I feel down", "I had a bad day",
        "I am frustrated", "this sucks", "I hate Mondays",
        "I am happy", "I feel great", "best day ever",
    ]


def main():
    print("Generating math flashcards...")

    all_sentences = []

    generators = [
        ("single_words", generate_single_words),
        ("amplifier_pairs", generate_amplifier_pairs),
        ("negation_pairs", generate_negation_pairs),
        ("self_reference", generate_self_reference),
        ("tense_pairs", generate_tense_pairs),
        ("idiom_sentences", generate_idiom_sentences),
        ("bigram_sentences", generate_bigram_sentences),
        ("hedging", generate_hedging),
        ("crisis_patterns", generate_crisis_patterns),
    ]

    for name, gen_fn in generators:
        sents = gen_fn()
        print(f"  {name}: {len(sents)} sentences")
        all_sentences.extend(sents)

    # Deduplicate
    all_sentences = list(set(all_sentences))
    print(f"\n  Total unique: {len(all_sentences)}")

    # Generate traces
    print("  Running engine traces...")
    start = time.time()
    output = []
    for i, sent in enumerate(all_sentences):
        if not sent.strip():
            continue
        try:
            vadug, trace = engine.process_text(sent)
            output.append({
                "english": sent,
                "vadug": [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g],
                "trace": trace,
                "engine": "PendulumV2",
            })
        except Exception:
            continue
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(all_sentences)}...")

    elapsed = time.time() - start
    print(f"  Generated {len(output)} traced examples in {elapsed:.1f}s")

    # Save
    outpath = os.path.join(PROJECT_ROOT, "training", "data", "math_flashcards.jsonl")
    with open(outpath, "w") as f:
        for d in output:
            f.write(json.dumps(d) + "\n")
    print(f"  Saved to {outpath}")
    print(f"  Size: {os.path.getsize(outpath) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
