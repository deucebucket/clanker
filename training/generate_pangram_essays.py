#!/usr/bin/env python3
"""Generate Emotional Pangram Essays — every word through every lens.

Like "the quick brown fox" tests every letter, each essay tests every
emotional word (2,049) through a specific emotional LENS.

8 lenses × 2,049 words = 16,392 sentences.
Same word appears in ALL essays. The CONTEXT carries the tune.

"abandoned" in rage:   "I am furious that they abandoned me."
"abandoned" in grief:  "I feel so empty since they abandoned me."
"abandoned" in humor:  "Apparently I've been abandoned, how delightful."
"abandoned" in fear:   "I'm terrified of being abandoned again."

The engine scores each. If it gets the same score for different lenses,
THAT'S the gap — it can't hear the tune, only the word.

Output: training/data/pangram_essays.jsonl

Usage:
    python3 training/generate_pangram_essays.py
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.pendulum_v2 import PendulumV2
from demo.forces_curated import EMOTIONAL_VOCABULARY

# ---------------------------------------------------------------------------
# Emotional lenses — each wraps the same word in a different context
# ---------------------------------------------------------------------------

# Each lens has carrier sentences that set the emotional TONE.
# {w} = the payload word. The surrounding words carry the tune.

LENSES = {
    # === NEGATIVE LENSES ===
    "rage": {
        "expected_direction": "negative",
        "description": "Anger, fury — high arousal, every word through hatred",
        "carriers": [
            "I am furious about all this {w} garbage.",
            "I hate how disgustingly {w} everything has become.",
            "I am so angry and sick of this {w} bullshit.",
            "This awful {w} crap makes me want to scream.",
            "I am enraged and disgusted by the horrible {w} of it all.",
            "I despise every miserable {w} thing about this.",
            "I am livid and furious and {w} beyond belief.",
            "The terrible {w} makes my blood boil with rage.",
        ],
    },
    "grief": {
        "expected_direction": "negative",
        "description": "Loss, mourning — heavy, sinking, every word through sorrow",
        "carriers": [
            "I lost everything and now I feel broken and {w}.",
            "The grief and agony of being {w} is devastating.",
            "I cry and suffer every time I think about how {w} this is.",
            "Nothing fills the miserable void of this {w} emptiness.",
            "I am shattered and heartbroken and {w} since you died.",
            "The sadness and pain of feeling {w} never goes away.",
            "I mourn the {w} that death destroyed forever.",
            "My broken heart aches with unbearable {w} sorrow.",
        ],
    },
    "fear": {
        "expected_direction": "negative",
        "description": "Anxiety, dread — high arousal, low dominance, every word as threat",
        "carriers": [
            "I am terrified and panicked about becoming {w}.",
            "The horrible thought of being {w} fills me with dread and horror.",
            "I'm scared and anxious that everything will turn {w}.",
            "I panic and tremble when I feel {w} creeping in.",
            "I have terrifying nightmares about the {w} getting worse.",
            "I'm anxious and afraid and {w} and I can't breathe.",
            "The dread and fear of {w} keeps me awake at night.",
            "I am horrified and {w} and paralyzed with terror.",
        ],
    },
    "disgust": {
        "expected_direction": "negative",
        "description": "Revulsion, contempt — every word through repulsion",
        "carriers": [
            "I am revolted and disgusted by this {w} trash.",
            "This repulsive {w} situation makes me sick.",
            "I find the {w} absolutely vile and disgusting.",
            "The gross and horrible {w} fills me with contempt.",
            "I am nauseated by how pathetic and {w} this is.",
            "The disgusting {w} is an abomination and waste.",
            "I despise this loathsome {w} with every fiber.",
            "The sickening {w} is the worst thing I have ever seen.",
        ],
    },
    # === POSITIVE LENSES ===
    "joy": {
        "expected_direction": "positive",
        "description": "Happiness, celebration — every word as something wonderful",
        "carriers": [
            "I am so happy and grateful and {w} right now.",
            "I love love love how wonderfully {w} everything feels.",
            "I am blessed and grateful for this amazing {w} feeling.",
            "I feel blessed and thrilled and {w} beyond measure.",
            "I am overjoyed and ecstatic about how {w} life is.",
            "My heart is full of love and joy and {w} and gratitude.",
            "I am delighted and overjoyed by the wonderful {w} in my life.",
            "I celebrate the beautiful brilliant {w} that makes life perfect.",
        ],
    },
    "conviction": {
        "expected_direction": "positive",
        "description": "Power, confidence — high dominance, every word as strength",
        "carriers": [
            "I am strong and brave and proud and {w} and unstoppable.",
            "I believe in the incredible power of being {w}.",
            "I am confident and determined and {w} in my purpose.",
            "I will fight and triumph for what is {w} and right.",
            "I stand proud and confident and {w} against all odds.",
            "I am determined and inspired to be {w} no matter what.",
            "I own my {w} truth with pride and full authority.",
            "I am powerful and brave and {w} and I will succeed.",
        ],
    },
    "love": {
        "expected_direction": "positive",
        "description": "Deep affection, warmth — every word through adoration",
        "carriers": [
            "I love you and I cherish how {w} you make me feel.",
            "My heart overflows with love and warmth and {w} for you.",
            "I adore the beautiful {w} light you bring to my life.",
            "You are wonderful and precious and {w} and I treasure you.",
            "I am deeply grateful and in love and {w} because of you.",
            "My darling, the {w} joy you bring is a beautiful blessing.",
            "I cherish and treasure every {w} moment we share together.",
            "You make me feel loved and safe and happy and {w} always.",
        ],
    },
    "gratitude": {
        "expected_direction": "positive",
        "description": "Thankfulness, appreciation — every word as a blessing",
        "carriers": [
            "I am so grateful and blessed and thankful for this {w}.",
            "I deeply appreciate the wonderful {w} gift in my life.",
            "Thank you for the beautiful and {w} kindness you showed me.",
            "I feel incredibly blessed and grateful and {w} for everything.",
            "My heart is full of gratitude and appreciation and {w} joy.",
            "I treasure and appreciate how {w} and wonderful this has been.",
            "I am thankful and overjoyed and {w} for this blessing.",
            "I feel so grateful and proud and {w} for all that I have.",
        ],
    },
    # === NEUTRAL LENSES ===
    "humor": {
        "expected_direction": "neutral",
        "description": "Comedy, irony — surface meaning flipped, sarcasm",
        "carriers": [
            "Oh great, another {w} day in paradise.",
            "Well isn't this just {w}, how wonderful.",
            "I'm absolutely {w}, said no one ever.",
            "How {w} of you, truly a masterpiece of effort.",
            "Nothing says fun like being {w} on a Monday.",
            "Apparently I'm {w} now, someone alert the media.",
            "My life is peak {w} comedy at this point.",
            "How perfectly {w}, I could not be less surprised.",
        ],
    },
    "neutral_clinical": {
        "expected_direction": "neutral",
        "description": "Clinical, detached — observational, every word as data point",
        "carriers": [
            "The subject reports feeling {w}.",
            "The state of being {w} was observed.",
            "A {w} response was noted in the assessment.",
            "The condition presents as {w} in clinical terms.",
            "The patient describes the experience as {w}.",
            "The data suggests a {w} emotional pattern.",
            "A {w} state was recorded during the evaluation.",
            "The {w} indicator was within expected range.",
        ],
    },
    "hedged": {
        "expected_direction": "neutral",
        "description": "Hypothetical, uncertain — every word dampened by doubt",
        "carriers": [
            "I might be somewhat {w}, I'm not really sure.",
            "Perhaps things are a little {w}, maybe not though.",
            "It could possibly be {w}, hypothetically speaking.",
            "I would say it's slightly {w}, generally speaking.",
            "One might occasionally feel {w}, sometimes rarely.",
            "If I had to guess, maybe it's somewhat {w}.",
            "Theoretically, this could perhaps be considered {w}.",
            "It's hard to say, but perhaps {w} slightly applies.",
        ],
    },
    # === SPECIAL EQUATION TYPES ===
    "negated": {
        "expected_direction": "positive",  # negated negative = positive
        "description": "Negation — flips every word. Tests negation handling.",
        "carriers": [
            "I am not {w} at all.",
            "I don't feel {w} in the slightest.",
            "I never feel {w} anymore.",
            "I can't say I feel {w} about this.",
            "I am absolutely not {w} whatsoever.",
            "I wouldn't call this {w} by any means.",
            "Nobody would ever describe me as {w}.",
            "I refuse to feel {w} about any of this.",
        ],
    },
    "imperative": {
        "expected_direction": "negative",
        "description": "Commands — every word as a demand. Tests imperative syntax.",
        "carriers": [
            "Stop being so {w} right now!",
            "Don't you dare act {w} with me!",
            "Quit this {w} nonsense immediately!",
            "Never be {w} like that again!",
            "Stop this horrible {w} behavior!",
            "Don't ever make me feel {w} again!",
            "Cut out this disgusting {w} attitude!",
            "Enough with the terrible {w} already!",
        ],
    },
    "reported": {
        "expected_direction": "neutral",
        "description": "Reported speech — someone ELSE's emotion. Tests distance.",
        "carriers": [
            "She said she was feeling rather {w} about it.",
            "He told them that the situation seemed {w}.",
            "They reported that the atmosphere was {w}.",
            "She mentioned it was somewhat {w} overall.",
            "He described the experience as {w} at the time.",
            "They indicated that things were {w} from their view.",
            "She explained that it felt {w} to her.",
            "He stated that the outcome was {w} in his opinion.",
        ],
    },
    "exclamatory": {
        "expected_direction": "negative",
        "description": "Exclamations — intensity maxed. Tests amplification.",
        "carriers": [
            "How incredibly horribly {w} this terrible situation is!",
            "What a disgustingly {w} and awful mess!",
            "I am so unbelievably {w} and furious right now!",
            "This is the most {w} and devastating thing ever!",
            "How shockingly {w} and horrible this has become!",
            "What an absolutely {w} and miserable disaster!",
            "I cannot believe how {w} and terrible everything is!",
            "This is outrageously {w} and completely unacceptable!",
        ],
    },
}


def main():
    engine = PendulumV2()
    output_path = os.path.join(os.path.dirname(__file__), "data", "pangram_essays.jsonl")

    words = list(EMOTIONAL_VOCABULARY.items())
    n_words = len(words)
    n_lenses = len(LENSES)
    total_expected = n_words * n_lenses

    print(f"\n{'='*70}")
    print(f"  EMOTIONAL PANGRAM GENERATOR")
    print(f"  {n_words} words × {n_lenses} lenses = {total_expected} sentences")
    print(f"{'='*70}\n")

    total = 0
    per_lens_stats = {}

    with open(output_path, "w") as f:
        for lens_name, lens_data in LENSES.items():
            carriers = lens_data["carriers"]
            expected = lens_data["expected_direction"]
            correct = 0
            lens_total = 0

            for i, (word, force) in enumerate(words):
                # Pick carrier sentence (rotate through templates)
                carrier = carriers[i % len(carriers)]
                sentence = carrier.format(w=word)

                # Score with engine
                vadug, trace = engine.process_text(sentence)

                # Classify
                pred = engine.classify(vadug.v, "three_way")
                is_correct = pred == expected

                if is_correct:
                    correct += 1
                lens_total += 1

                # Build training example
                example = {
                    "text": sentence,
                    "v": vadug.v,
                    "a": vadug.a,
                    "d": vadug.d,
                    "u": vadug.u,
                    "g": vadug.g,
                    "lens": lens_name,
                    "expected": expected,
                    "predicted": pred,
                    "correct": is_correct,
                    "payload_word": word,
                    "payload_force": list(force),
                }

                f.write(json.dumps(example) + "\n")
                total += 1

            accuracy = correct / lens_total * 100 if lens_total > 0 else 0
            per_lens_stats[lens_name] = {
                "accuracy": round(accuracy, 1),
                "correct": correct,
                "total": lens_total,
            }
            print(f"  {lens_name:<20} {accuracy:>6.1f}%  ({correct}/{lens_total})")

    print(f"\n  {'='*70}")
    print(f"  TOTAL: {total} scored pangram sentences")
    print(f"  Output: {output_path}")

    # Overall accuracy
    total_correct = sum(s["correct"] for s in per_lens_stats.values())
    total_count = sum(s["total"] for s in per_lens_stats.values())
    overall = total_correct / total_count * 100
    print(f"  Overall accuracy: {overall:.1f}% ({total_correct}/{total_count})")
    print(f"  {'='*70}")

    # Find gap words — words that score the SAME across all lenses
    print(f"\n  GAP ANALYSIS: Words that sound the same in every lens...")

    # Reload and analyze
    word_scores = defaultdict(dict)  # word -> {lens: v_score}
    with open(output_path) as f_in:
        for line in f_in:
            ex = json.loads(line)
            word_scores[ex["payload_word"]][ex["lens"]] = ex["v"]

    # Find words with zero variation across lenses
    flat_words = []
    for word, scores in word_scores.items():
        vals = list(scores.values())
        spread = max(vals) - min(vals)
        if spread < 5:
            flat_words.append((word, spread, vals[0]))

    flat_words.sort(key=lambda x: x[1])
    print(f"  Words with <5 V-spread across all lenses: {len(flat_words)}")
    for word, spread, center in flat_words[:20]:
        force = EMOTIONAL_VOCABULARY[word]
        print(f"    {word:<20} spread={spread:>2}  center={center:>3}  dV={force[0]:>4}")

    # Find words that should flip between lenses but don't
    print(f"\n  WORDS THAT SHOULD FLIP BUT DON'T:")
    print(f"  (positive words that stay positive even in rage/grief/fear)")
    flip_failures = []
    for word, scores in word_scores.items():
        force = EMOTIONAL_VOCABULARY[word]
        if force[0] > 30:  # positive word
            rage_v = scores.get("rage", 128)
            grief_v = scores.get("grief", 128)
            joy_v = scores.get("joy", 128)
            # Should be low in rage/grief, high in joy
            if rage_v > 128 and grief_v > 128:
                flip_failures.append((word, rage_v, grief_v, joy_v, force[0]))

    flip_failures.sort(key=lambda x: x[1], reverse=True)
    print(f"  Count: {len(flip_failures)}")
    for word, rage_v, grief_v, joy_v, dv in flip_failures[:15]:
        print(f"    {word:<20} rage_V={rage_v:>3} grief_V={grief_v:>3} joy_V={joy_v:>3}  (dV={dv:>3})")

    # Save summary
    summary = {
        "total_sentences": total,
        "per_lens": per_lens_stats,
        "overall_accuracy": round(overall, 1),
        "flat_word_count": len(flat_words),
        "flip_failure_count": len(flip_failures),
    }
    summary_path = os.path.join(os.path.dirname(__file__), "data", "pangram_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
