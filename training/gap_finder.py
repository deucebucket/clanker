#!/usr/bin/env python3
"""Gap Finder — automated discovery of missing emotional mechanics.

Compares the trained model vs the rule-based engine on the same inputs.
Where they disagree, clusters the disagreements by PATTERN to find:

1. Missing word forces (words the engine doesn't know about)
2. Missing idioms (multi-word combos the engine misses)
3. Negation failures ("not happy" scored as positive)
4. Sarcasm blind spots ("yeah right", "great just great")
5. Context dependencies (same word, different meaning)
6. Missing VARIABLES — if clusters can't be explained by V,A,D,U,G,
   that suggests a missing dimension in the coordinate system

Usage:
    python3 training/gap_finder.py                    # Run full analysis
    python3 training/gap_finder.py --generate-fixes   # Generate engine patches
"""

import json
import os
import sys
import re
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tokenizer"))


def load_model_predictions(checkpoint_dir):
    """Load model and predict VADUG for test sentences."""
    import torch
    from transformers import GPT2LMHeadModel
    from clanker_tokenizer import ClankerTokenizer

    tokenizer = ClankerTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
    model = model.to(device)
    model.set_output_embeddings(model.get_output_embeddings())

    def predict(text):
        """Get model's VADUG prediction for text."""
        bos_id = tokenizer.bos_token_id
        sep_id = tokenizer._vocab["<sep>"]
        emotion_id = tokenizer._vocab["<emotion>"]

        text_ids = []
        for ch in text:
            tid = tokenizer._vocab.get(ch) or tokenizer._vocab.get(ch.lower()) or tokenizer.unk_token_id
            text_ids.append(tid)

        input_ids = [bos_id] + text_ids[:240] + [sep_id, emotion_id]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            predicted_vadug = []
            current_input = input_tensor

            for _ in range(5):
                outputs = model(input_ids=current_input)
                next_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_logits).item()
                predicted_vadug.append(next_token)
                next_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
                current_input = torch.cat([current_input, next_tensor], dim=1)

        bucket_to_value = {
            "neg_high": 18, "neg_med": 55, "neg_low": 92,
            "neutral": 128,
            "pos_low": 165, "pos_med": 201, "pos_high": 237,
        }

        vadug = []
        token_names = tokenizer.convert_ids_to_tokens(predicted_vadug)
        for name in token_names:
            parts = name.split("_", 1)
            if len(parts) == 2 and parts[1] in bucket_to_value:
                vadug.append(bucket_to_value[parts[1]])
            else:
                vadug.append(128)

        return vadug, token_names

    return predict


def load_engine():
    """Load the pendulum engine."""
    from demo.pendulum import SequentialPendulum
    engine = SequentialPendulum()

    def predict(text):
        vadug, trace = engine.process_text(text)
        return [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g], trace

    return predict


def generate_test_sentences():
    """Generate diverse test sentences for gap analysis."""
    sentences = []

    categories = {
        "negation": [
            "I am not happy", "This is not okay", "I do not feel safe",
            "That was not funny", "I am not impressed", "This is not fair",
            "I do not like this", "I cannot believe this", "I will not accept this",
            "That is not what I meant", "I am not angry just disappointed",
            "Nothing is wrong", "Nobody said anything mean", "I never said that",
            "It is not that bad", "I do not care anymore", "I am not sad",
        ],
        "sarcasm": [
            "Yeah right", "Oh great", "Thanks for nothing", "What a surprise",
            "Sure you do", "Nice going genius", "Wonderful just wonderful",
            "Oh how lovely", "Great just great", "Yeah that went well",
            "Perfect timing as always", "Well that was fun", "Brilliant idea",
            "How delightful", "What a joy", "Absolutely fantastic",
            "Oh I am so thrilled", "Could not be happier", "Living the dream",
        ],
        "context": [
            "I lost my keys", "I lost my mother", "I lost my mind",
            "I lost the game", "I lost weight", "I lost my patience",
            "She left the room", "She left me", "She left the company",
            "He is gone", "The pain is gone", "My money is gone",
            "It is over", "The storm is over", "My career is over",
            "I broke the record", "I broke down crying", "I broke my promise",
            "That is sick", "I feel sick", "That trick was sick",
        ],
        "intensity": [
            "I am a little sad", "I am sad", "I am very sad",
            "I am extremely sad", "I am devastated beyond words",
            "I am slightly annoyed", "I am annoyed", "I am very annoyed",
            "I am furious", "I am in a blind rage",
            "I am okay", "I am good", "I am great",
            "I am wonderful", "I am ecstatic beyond belief",
        ],
        "complex": [
            "I was happy until I heard the news",
            "Things were going well but then everything fell apart",
            "I thought I was fine but I am not",
            "She said she loved me but her actions say otherwise",
            "The test came back negative which is actually good news",
            "I failed the exam but at least I tried my best",
            "He left me for someone else but honestly I feel relieved",
            "I got fired today but it might be a blessing in disguise",
        ],
        "crisis": [
            "I want to die", "I want to disappear",
            "Nobody would miss me", "I am a burden to everyone",
            "What is the point of going on", "I cannot do this anymore",
            "I wish I was never born", "Everything would be better without me",
            "I just want the pain to stop", "There is no way out",
        ],
        "positive": [
            "I am so proud of myself", "Today was absolutely magical",
            "I have never been happier", "Everything is coming together",
            "I feel alive for the first time in years",
            "This is the best day of my life",
            "I am overwhelmed with gratitude",
            "My heart is full", "I found my purpose",
        ],
    }

    for category, sents in categories.items():
        for s in sents:
            sentences.append({"text": s, "category": category})

    return sentences


def analyze_gaps(model_predict, engine_predict, sentences):
    """Compare model vs engine, return results."""
    results = []

    for item in sentences:
        text = item["text"]
        category = item["category"]

        engine_vadug, engine_trace = engine_predict(text)
        try:
            model_vadug, model_tokens = model_predict(text)
        except Exception as e:
            model_vadug = [128, 128, 128, 0, 128]
            model_tokens = [str(e)[:50]]

        v_diff = model_vadug[0] - engine_vadug[0]

        results.append({
            "text": text,
            "category": category,
            "engine_v": engine_vadug[0],
            "model_v": model_vadug[0],
            "v_diff": v_diff,
            "abs_diff": abs(v_diff),
            "engine_vadug": engine_vadug,
            "model_vadug": model_vadug,
            "model_tokens": model_tokens,
        })

    return results


def cluster_gaps(results, threshold=25):
    """Cluster gaps by pattern to find systematic issues."""
    gaps = [r for r in results if r["abs_diff"] >= threshold]

    print(f"\n{'='*70}")
    print(f"GAP ANALYSIS: {len(gaps)} disagreements (V diff >= {threshold})")
    print(f"{'='*70}")

    # By category
    print(f"\nGaps by category:")
    cat_gaps = defaultdict(list)
    for g in gaps:
        cat_gaps[g["category"]].append(g)

    for cat, items in sorted(cat_gaps.items(), key=lambda x: -len(x[1])):
        total_in_cat = sum(1 for r in results if r["category"] == cat)
        print(f"\n  {cat.upper()} -- {len(items)}/{total_in_cat} sentences have gaps")
        for item in sorted(items, key=lambda x: -x["abs_diff"])[:5]:
            direction = "model MORE positive" if item["v_diff"] > 0 else "model MORE negative"
            print(f"    \"{item['text'][:45]}\"")
            print(f"      engine V={item['engine_v']}, model V={item['model_v']}, "
                  f"diff={item['v_diff']:+d} ({direction})")

    # Direction analysis
    model_higher = sum(1 for g in gaps if g["v_diff"] > 0)
    model_lower = sum(1 for g in gaps if g["v_diff"] < 0)
    print(f"\nDirection bias in gaps:")
    print(f"  Model scores HIGHER than engine: {model_higher} times")
    print(f"  Model scores LOWER than engine:  {model_lower} times")
    if model_higher > model_lower * 1.5:
        print(f"  -> Model has POSITIVE BIAS vs engine")
    elif model_lower > model_higher * 1.5:
        print(f"  -> Model has NEGATIVE BIAS vs engine")
    else:
        print(f"  -> No strong directional bias")

    # Word-level analysis
    gap_words = Counter()
    all_words = Counter()
    for r in results:
        words = set(r["text"].lower().split())
        all_words.update(words)
        if r["abs_diff"] >= threshold:
            gap_words.update(words)

    print(f"\nWords over-represented in gaps (potential missing forces):")
    suspicious = []
    for word, gap_count in gap_words.most_common(50):
        total_count = all_words[word]
        if total_count >= 3:
            gap_rate = gap_count / total_count
            if gap_rate > 0.6:
                suspicious.append((word, gap_count, total_count, gap_rate))

    for word, gc, tc, rate in sorted(suspicious, key=lambda x: -x[3])[:15]:
        print(f"  \"{word}\" -- in gaps {gc}/{tc} times ({rate*100:.0f}%)")

    # Residual analysis
    print(f"\n{'='*70}")
    print(f"RESIDUAL ANALYSIS -- looking for missing variables")
    print(f"{'='*70}")

    large_gaps = [g for g in gaps if g["abs_diff"] >= 50]
    if large_gaps:
        print(f"\n{len(large_gaps)} large gaps (V diff >= 50):")
        for g in sorted(large_gaps, key=lambda x: -x["abs_diff"]):
            print(f"  [{g['category']:10s}] \"{g['text'][:50]}\"")
            print(f"    engine={g['engine_vadug']}, model={g['model_vadug']}, V diff={g['v_diff']:+d}")

        large_cats = Counter(g["category"] for g in large_gaps)
        print(f"\n  Large gap categories: {dict(large_cats)}")
        if large_cats.get("sarcasm", 0) > 3:
            print(f"  -> SARCASM is a major unmodeled phenomenon")
            print(f"    Consider: sarcasm inverts valence but engine has no inversion mechanism")
        if large_cats.get("negation", 0) > 3:
            print(f"  -> NEGATION handling is incomplete")
            print(f"    Consider: negation modifier should flip valence of next content word")
        if large_cats.get("context", 0) > 3:
            print(f"  -> CONTEXT DEPENDENCY needs modeling")
            print(f"    Consider: same word ('lost') has different V depending on object")

    return gaps


def generate_engine_fixes(gaps):
    """Suggest concrete fixes for the engine based on gap patterns."""
    print(f"\n{'='*70}")
    print(f"SUGGESTED ENGINE FIXES")
    print(f"{'='*70}")

    idiom_candidates = []
    negation_fixes = []

    for g in gaps:
        text = g["text"].lower()
        words = text.split()

        # Check for multi-word patterns
        if len(words) >= 2 and g["abs_diff"] >= 30:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in ["not happy", "not okay", "not safe", "not funny",
                              "not fair", "for nothing", "yeah right",
                              "just great", "how lovely", "sure you"]:
                    idiom_candidates.append((bigram, g["model_v"], g["engine_v"], g["text"]))

        if any(neg in words for neg in ["not", "no", "never", "nobody", "nothing", "cannot"]):
            negation_fixes.append(g)

    if idiom_candidates:
        print(f"\nNew IDIOM entries needed ({len(idiom_candidates)}):")
        seen = set()
        for bigram, model_v, engine_v, text in idiom_candidates:
            if bigram not in seen:
                seen.add(bigram)
                target_v = model_v
                dv = target_v - 128
                parts = bigram.split()
                print(f"  (\"{parts[0]}\", \"{parts[1]}\"): "
                      f"({dv:+d}, 0, 0, 0, \"{bigram}\"),")
                print(f"    # engine={engine_v}, model={model_v}, from \"{text[:40]}\"")

    if negation_fixes:
        print(f"\nNegation handling fixes needed ({len(negation_fixes)} sentences):")
        for g in negation_fixes[:10]:
            print(f"  \"{g['text'][:50]}\" -- engine V={g['engine_v']}, should be ~{g['model_v']}")

    return idiom_candidates, negation_fixes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find gaps between model and engine")
    parser.add_argument("--checkpoint", default="training/checkpoints/best",
                        help="Model checkpoint directory")
    parser.add_argument("--threshold", type=int, default=25,
                        help="V difference threshold to flag as gap")
    parser.add_argument("--generate-fixes", action="store_true",
                        help="Generate engine patch suggestions")
    parser.add_argument("--output", default="training/data/gap_analysis.jsonl",
                        help="Output file for gap data")
    args = parser.parse_args()

    print("Loading model...")
    model_predict = load_model_predictions(args.checkpoint)

    print("Loading engine...")
    engine_predict = load_engine()

    print("Generating test sentences...")
    sentences = generate_test_sentences()
    print(f"  {len(sentences)} test sentences across "
          f"{len(set(s['category'] for s in sentences))} categories")

    print("\nRunning comparison...")
    results = analyze_gaps(model_predict, engine_predict, sentences)

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nRaw results saved to {args.output}")

    gaps = cluster_gaps(results, threshold=args.threshold)

    if args.generate_fixes:
        generate_engine_fixes(gaps)

    total = len(results)
    agreed = sum(1 for r in results if r["abs_diff"] < args.threshold)
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"  Total test sentences: {total}")
    print(f"  Model-Engine agreement (V diff < {args.threshold}): "
          f"{agreed}/{total} ({100*agreed/total:.0f}%)")
    print(f"  Gaps found: {total - agreed}")
    print(f"  Run with --generate-fixes to get engine patches")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
