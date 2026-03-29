#!/usr/bin/env python3
"""Essay Benchmark — Score full essays of each emotional archetype.

Each essay exercises a different set of sentence structures and emotional
equations. The engine scores each sentence, and we analyze:
  - Per-sentence accuracy (does the engine detect the right emotion?)
  - Word coverage (what % of words does V2 know?)
  - Missing words (which emotional words are we blind to?)
  - Syntax patterns (which sentence structures break the engine?)

Usage:
    python3 benchmarks/essay_benchmark.py
    python3 benchmarks/essay_benchmark.py --verbose  # show per-sentence traces
"""

import sys, os, json, re
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.pendulum_v2 import PendulumV2
from demo.forces_curated import EMOTIONAL_VOCABULARY
from demo.forces import WORD_FORCES

# ---------------------------------------------------------------------------
# Essay corpus — each essay represents a distinct emotional archetype
# and exercises different sentence structures
# ---------------------------------------------------------------------------

ESSAYS = {
    "sarcastic": {
        "expected": "negative",  # surface positive, true meaning negative
        "description": "Sarcasm: surface-level positive words, negative intent",
        "sentences": [
            "Oh wonderful, another meeting that could have been an email.",
            "Yeah, because that worked out so well last time.",
            "Sure, let me just drop everything for your emergency.",
            "Brilliant idea, I can't believe nobody thought of that before.",
            "Just what I needed, more unsolicited advice.",
            "How nice of you to finally show up.",
            "Great, another expert with all the answers.",
            "Oh fantastic, the printer is jammed again.",
            "What a surprise, the deadline moved up.",
            "Clearly you've put so much thought into this.",
            "I'm absolutely thrilled to redo all my work.",
            "Nothing says teamwork like doing it all yourself.",
            "Of course it's my fault, it always is.",
            "How delightful, another policy change nobody asked for.",
            "I love how nothing ever goes according to plan.",
        ],
    },
    "grief": {
        "expected": "negative",
        "description": "Deep sadness, loss, mourning — heavy gravity",
        "sentences": [
            "I still reach for the phone to call her before I remember.",
            "The house feels so empty now that you're gone.",
            "I can't stop crying when I hear that song.",
            "Everything reminds me of what I've lost.",
            "I don't know how to go on without you.",
            "The grief hits me in waves I can't predict.",
            "I feel hollow inside, like something vital was ripped away.",
            "Nobody understands how much this hurts.",
            "I would give anything to hear your voice one more time.",
            "Some days I can barely get out of bed.",
            "The world kept moving but I'm frozen in that moment.",
            "I miss you more than words can express.",
            "There's a void where you used to be.",
            "I'm drowning in memories that won't stop coming.",
            "Nothing will ever feel the same again.",
        ],
    },
    "rage": {
        "expected": "negative",
        "description": "Anger, fury, aggression — high arousal, high urgency",
        "sentences": [
            "I am absolutely furious about what happened.",
            "How dare you lie to my face like that.",
            "I could scream from how angry I am right now.",
            "This is complete bullshit and everyone knows it.",
            "I'm sick of being treated like garbage.",
            "You have no right to talk to me that way.",
            "I swear I'm going to lose my mind.",
            "That was the most disrespectful thing anyone has ever done to me.",
            "I hate that I trusted you.",
            "I am done putting up with this crap.",
            "You disgust me with your selfishness.",
            "I want to break something right now.",
            "Every time I think about it my blood boils.",
            "Fuck this entire situation.",
            "I will never forgive what you did.",
        ],
    },
    "joy": {
        "expected": "positive",
        "description": "Genuine happiness, excitement, celebration",
        "sentences": [
            "I just got the promotion I've been working toward for years!",
            "This is the happiest day of my life.",
            "I can't stop smiling, everything feels perfect.",
            "My heart is overflowing with gratitude and love.",
            "We did it, we actually did it!",
            "I've never felt this alive and excited before.",
            "Today was absolutely wonderful from start to finish.",
            "I'm so proud of how far we've come together.",
            "This moment is everything I've ever dreamed of.",
            "I feel like I could fly right now.",
            "My soul is singing with pure happiness.",
            "I am so incredibly grateful for this opportunity.",
            "Every single thing went right today.",
            "I love my life and everyone in it.",
            "This is what true fulfillment feels like.",
        ],
    },
    "conviction": {
        "expected": "positive",
        "description": "Powerful, assertive, high dominance — leadership voice",
        "sentences": [
            "I will not back down from what I believe in.",
            "We have the power to change this right now.",
            "I am completely confident in our direction.",
            "This is our moment and we will seize it.",
            "Nothing and nobody will stop us from achieving this.",
            "I stand firm in my decision and I own it.",
            "We are stronger than any obstacle in our path.",
            "I refuse to accept anything less than excellence.",
            "Mark my words, we will succeed.",
            "I take full responsibility and full control.",
            "There is no challenge we cannot overcome together.",
            "I am ready to fight for what matters.",
            "Our resolve is unshakeable and our mission is clear.",
            "I believe in every single one of you.",
            "We will make history, starting today.",
        ],
    },
    "fear": {
        "expected": "negative",
        "description": "Anxiety, dread, panic — high arousal, low dominance",
        "sentences": [
            "I'm terrified of what might happen next.",
            "My hands are shaking and I can't breathe.",
            "Something terrible is going to happen, I can feel it.",
            "I keep having nightmares about the worst case scenario.",
            "I'm afraid to even open that email.",
            "What if everything falls apart?",
            "I feel paralyzed by anxiety every single day.",
            "The fear is eating me alive from the inside.",
            "I can't sleep because my mind won't stop racing.",
            "I'm scared that nobody will help me.",
            "Every sound makes me jump out of my skin.",
            "I don't feel safe anywhere anymore.",
            "The panic attacks are getting worse.",
            "I'm living in constant dread of the unknown.",
            "I feel like the walls are closing in on me.",
        ],
    },
    "neutral_report": {
        "expected": "neutral",
        "description": "Factual reporting, no emotional charge",
        "sentences": [
            "The meeting is scheduled for three o'clock on Tuesday.",
            "There are approximately forty seven items in the inventory.",
            "The document was submitted on the fifteenth of March.",
            "The average temperature this week was sixty two degrees.",
            "The committee reviewed the proposal during the afternoon session.",
            "The building is located on the corner of Fifth and Main.",
            "The report contains twelve sections and three appendices.",
            "The new system was installed during the maintenance window.",
            "The policy applies to all employees regardless of department.",
            "The quarterly results will be published next week.",
            "The train arrives at platform four at seven fifteen.",
            "The contract expires at the end of the fiscal year.",
            "There are two hundred participants registered for the event.",
            "The data was collected over a six month period.",
            "The software update addresses three known issues.",
        ],
    },
    "conditional_hedging": {
        "expected": "neutral",
        "description": "Hypothetical framing, hedging — directness test",
        "sentences": [
            "If I were to say something, it might be that this seems wrong.",
            "It's possible that some people could potentially be upset.",
            "I'm not sure, but maybe there's a slight chance of a problem.",
            "Hypothetically speaking, what if the plan doesn't work?",
            "Generally speaking, things like this tend to sometimes go badly.",
            "In theory, one could argue that the approach has limitations.",
            "It would seem that perhaps the situation is somewhat complicated.",
            "If you were to ask me, I might suggest a different direction.",
            "One could potentially make the case that things aren't ideal.",
            "I don't want to jump to conclusions, but it looks concerning.",
            "There's a possibility that we may need to reconsider.",
            "In some cases, outcomes like this can be disappointing.",
            "It's hard to say definitively, but there might be issues.",
            "Without making any guarantees, I'd lean toward caution.",
            "Let's just say the results weren't exactly what we hoped for.",
        ],
    },
}


def score_essay(engine, essay_data):
    """Score every sentence in an essay, return analysis."""
    results = []
    unknown_words = Counter()
    known_words = Counter()
    total_words = 0
    v2_hits = 0

    for sentence in essay_data["sentences"]:
        vadug, trace = engine.process_text(sentence)
        pred = engine.classify(vadug.v, "three_way")

        # Count word coverage
        words = re.findall(r"[a-z']+", sentence.lower())
        for w in words:
            total_words += 1
            if w in EMOTIONAL_VOCABULARY:
                v2_hits += 1
                known_words[w] += 1
            elif w in WORD_FORCES and abs(WORD_FORCES[w][0]) >= 10:
                unknown_words[w] += 1

        correct = pred == essay_data["expected"]
        results.append({
            "sentence": sentence,
            "vadug": {"v": vadug.v, "a": vadug.a, "d": vadug.d, "u": vadug.u, "g": vadug.g},
            "predicted": pred,
            "expected": essay_data["expected"],
            "correct": correct,
            "trace": trace,
        })

    accuracy = sum(1 for r in results if r["correct"]) / len(results) * 100
    coverage = v2_hits / total_words * 100 if total_words > 0 else 0

    return {
        "accuracy": round(accuracy, 1),
        "coverage": round(coverage, 1),
        "total_words": total_words,
        "v2_hits": v2_hits,
        "missing_words": unknown_words.most_common(20),
        "results": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    engine = PendulumV2()

    print(f"\n{'='*80}")
    print(f"  ESSAY BENCHMARK — Emotional Archetype Testing")
    print(f"  Engine: PendulumV2 (untuned defaults)")
    print(f"  Vocabulary: {len(EMOTIONAL_VOCABULARY)} words")
    print(f"{'='*80}\n")

    all_missing = Counter()
    total_correct = 0
    total_sentences = 0

    print(f"  {'Essay Type':<25} {'Accuracy':>8} {'Coverage':>9} {'Correct':>8} {'Total':>6}")
    print(f"  {'-'*60}")

    for essay_type, essay_data in ESSAYS.items():
        analysis = score_essay(engine, essay_data)
        n = len(essay_data["sentences"])
        correct = sum(1 for r in analysis["results"] if r["correct"])
        total_correct += correct
        total_sentences += n

        print(f"  {essay_type:<25} {analysis['accuracy']:>7.1f}% {analysis['coverage']:>8.1f}% "
              f"{correct:>7}/{n:<4}")

        # Collect missing words
        for word, count in analysis["missing_words"]:
            all_missing[word] += count

        if args.verbose:
            for r in analysis["results"]:
                status = "OK" if r["correct"] else "XX"
                v = r["vadug"]
                print(f"    [{status}] V={v['v']:>3} A={v['a']:>3} D={v['d']:>3} "
                      f"pred={r['predicted']:<8} \"{r['sentence'][:60]}\"")

    overall = total_correct / total_sentences * 100
    print(f"  {'-'*60}")
    print(f"  {'OVERALL':<25} {overall:>7.1f}%{'':>9} {total_correct:>7}/{total_sentences}")

    print(f"\n  MISSING WORDS (V1 knows, V2 doesn't, |dV|>=10):")
    print(f"  {'Word':<20} {'Count':>5} {'dV':>5}")
    print(f"  {'-'*35}")
    for word, count in all_missing.most_common(30):
        if word in WORD_FORCES:
            dv = WORD_FORCES[word][0]
            print(f"  {word:<20} {count:>5} {dv:>5}")

    # Save results
    out = {
        "overall_accuracy": round(overall, 1),
        "total_sentences": total_sentences,
        "missing_words": dict(all_missing.most_common(50)),
        "per_essay": {k: {"accuracy": score_essay(engine, v)["accuracy"],
                          "coverage": score_essay(engine, v)["coverage"]}
                      for k, v in ESSAYS.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "essay_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
