#!/usr/bin/env python3
"""Diagnostic script for pendulum tuning analysis. Temporary — do not commit."""

import sys, os, re, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarks.test_suite_500 import (
    BASIC_POSITIVE, BASIC_NEGATIVE, SARCASM, CRISIS, MIXED,
    GRANULARITY, CONTEXT, PARAGRAPHS, REAL_WORLD, PROFANITY, ALL_TESTS
)
from demo.simulator import SequentialPendulum, VADUG, nearest_emotion, WORD_FORCES

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


def run_clanker(text):
    pendulum = SequentialPendulum()
    words = re.findall(r"[a-z']+", text.lower())
    for i, word in enumerate(words):
        try:
            pendulum.process_word(word, words, i)
        except Exception:
            pass
    return {
        'v': int(pendulum.v), 'a': int(pendulum.a),
        'd': int(pendulum.d), 'u': int(pendulum.u), 'g': int(pendulum.g),
        'history': pendulum.history,
    }


def classify_clanker(result, expected):
    v, a, d, g = result['v'], result['a'], result['d'], result['g']

    if expected == "positive":
        return v > 145
    elif expected == "negative":
        return v < 110
    elif expected == "crisis":
        return v < 60
    elif expected == "crisis_adjacent":
        return v < 90
    elif expected == "neutral":
        return 118 <= v <= 140
    elif expected == "mixed":
        return 100 <= v <= 145
    elif expected == "anger":
        return v < 100 and a > 150 and g > 130
    elif expected == "sadness":
        return v < 110 and g < 120
    elif expected == "fear":
        return v < 110 and a > 140 and d < 100
    elif expected == "disgust":
        return v < 100 and g > 120
    elif expected == "disappointment":
        return v < 120 and a < 160 and g < 120
    elif expected == "anxiety":
        return v < 120 and a > 140 and g > 120
    elif expected == "frustration":
        return v < 110 and a > 140
    elif expected == "nostalgia":
        return v < 128 and a < 140 and g < 128
    elif expected == "relief":
        return v > 128 and a < 140
    elif expected in ("jealousy", "envy", "guilt", "shame", "regret"):
        return v < 120
    elif expected == "grief":
        return v < 110
    elif expected == "melancholy":
        return v < 128
    elif expected in ("joy", "excitement"):
        return v > 145
    elif expected == "contentment":
        return v > 128
    elif expected in ("dread", "nervousness"):
        return v < 120
    elif expected in ("contempt", "revulsion"):
        return v < 100
    elif expected == "surprise_positive":
        return v > 140
    elif expected == "surprise_negative":
        return v < 110
    elif expected in ("pride", "arrogance"):
        return v > 135
    elif expected == "loneliness":
        return v < 120
    elif expected == "solitude":
        return v > 120
    elif expected == "irritation":
        return v < 120
    else:
        return False


def classify_vader(text, expected):
    if not HAS_VADER:
        return False
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    if expected == "positive":
        return compound > 0.05
    elif expected in ("negative",):
        return compound < -0.05
    elif expected in ("crisis",):
        return compound < -0.3
    elif expected in ("crisis_adjacent",):
        return compound < -0.1
    elif expected == "neutral":
        return -0.05 <= compound <= 0.05
    elif expected == "mixed":
        return -0.3 <= compound <= 0.3
    elif expected in ("anger", "frustration", "irritation", "sadness", "grief",
                       "melancholy", "disappointment", "fear", "anxiety", "dread",
                       "nervousness", "disgust", "contempt", "revulsion",
                       "jealousy", "envy", "guilt", "shame", "regret", "loneliness"):
        return compound < -0.05
    elif expected in ("joy", "excitement", "contentment", "relief", "pride",
                       "surprise_positive", "solitude", "arrogance"):
        return compound > 0.05
    elif expected == "surprise_negative":
        return compound < -0.05
    elif expected == "nostalgia":
        return compound < -0.05
    else:
        return False


def main():
    categories = {
        'BASIC_POSITIVE': BASIC_POSITIVE,
        'BASIC_NEGATIVE': BASIC_NEGATIVE,
        'SARCASM': SARCASM,
        'CRISIS': CRISIS,
        'MIXED': MIXED,
        'GRANULARITY': GRANULARITY,
        'CONTEXT': CONTEXT,
        'PARAGRAPHS': PARAGRAPHS,
        'REAL_WORLD': REAL_WORLD,
        'PROFANITY': PROFANITY,
    }

    output = {}
    total_c = 0
    total_v = 0
    total_n = 0
    all_failures = []

    print('=== PER-CATEGORY ACCURACY ===')
    for cat_name, cases in categories.items():
        c_ok_count = 0
        v_ok_count = 0
        for text, expected, desc in cases:
            result = run_clanker(text)
            c_ok = classify_clanker(result, expected)
            v_ok = classify_vader(text, expected)
            if c_ok:
                c_ok_count += 1
            else:
                all_failures.append({
                    'cat': cat_name, 'text': text, 'expected': expected,
                    'desc': desc, 'v': result['v'], 'a': result['a'],
                    'd': result['d'], 'u': result['u'], 'g': result['g'],
                })
            if v_ok:
                v_ok_count += 1
            total_n += 1
        total_c += c_ok_count
        total_v += v_ok_count
        c_pct = c_ok_count / len(cases) * 100
        v_pct = v_ok_count / len(cases) * 100
        print(f'{cat_name:20s}: Clanker {c_ok_count:3d}/{len(cases):3d} ({c_pct:5.1f}%)  VADER {v_ok_count:3d}/{len(cases):3d} ({v_pct:5.1f}%)')
        output[cat_name] = {'clanker': c_ok_count, 'vader': v_ok_count, 'total': len(cases)}

    c_overall = total_c / total_n * 100
    v_overall = total_v / total_n * 100
    print(f'\nOVERALL: Clanker {total_c}/{total_n} ({c_overall:.1f}%)  VADER {total_v}/{total_n} ({v_overall:.1f}%)')

    # Failure breakdown by expected category
    print('\n=== FAILURE DISTRIBUTION BY EXPECTED CATEGORY ===')
    from collections import Counter
    fail_by_expected = Counter(f['expected'] for f in all_failures)
    for exp, count in sorted(fail_by_expected.items(), key=lambda x: -x[1]):
        print(f'  {exp:20s}: {count} failures')

    # V-value stats per failure category
    print('\n=== FAILED V-VALUES BY CATEGORY ===')
    for cat_name in categories:
        fails = [f for f in all_failures if f['cat'] == cat_name]
        if not fails:
            continue
        vs = [f['v'] for f in fails]
        print(f'{cat_name:20s}: avg V={sum(vs)/len(vs):.1f}, min={min(vs)}, max={max(vs)}, n={len(fails)}')

    # Sample 30 failures with word analysis
    print('\n=== SAMPLE 30 FAILURES WITH WORD ANALYSIS ===')
    random.seed(42)
    sampled = random.sample(all_failures, min(30, len(all_failures)))
    for f in sampled:
        words = re.findall(r"[a-z']+", f['text'].lower())
        in_dict = [w for w in words if w in WORD_FORCES]
        missing = [w for w in words if w not in WORD_FORCES
                   and w not in SequentialPendulum.BRIDGE_WORDS and len(w) > 2]
        forces = [(w, WORD_FORCES[w][0]) for w in in_dict]
        pos_words = [f'{w}({v:+d})' for w, v in forces if v > 0]
        neg_words = [f'{w}({v:+d})' for w, v in forces if v < 0]
        print(f'\n  TEXT: "{f["text"][:80]}"')
        print(f'  Expected: {f["expected"]}  V={f["v"]} A={f["a"]} D={f["d"]} G={f["g"]}')
        print(f'  + words: {", ".join(pos_words) if pos_words else "none"}')
        print(f'  - words: {", ".join(neg_words) if neg_words else "none"}')
        if missing:
            print(f'  Missing: {", ".join(missing[:8])}')

    # Threshold sweep for V
    print('\n=== THRESHOLD SWEEP ===')
    print('Testing different positive/negative V thresholds...')
    for pos_thresh in [130, 135, 140, 145, 150]:
        for neg_thresh in [100, 105, 110, 115, 118, 120, 125]:
            correct = 0
            for text, expected, desc in ALL_TESTS:
                result = run_clanker(text)
                v = result['v']
                if expected == "positive":
                    if v > pos_thresh:
                        correct += 1
                elif expected == "negative":
                    if v < neg_thresh:
                        correct += 1
                elif expected == "crisis":
                    if v < 60:
                        correct += 1
                elif expected == "crisis_adjacent":
                    if v < 90:
                        correct += 1
                elif expected == "mixed":
                    if neg_thresh <= v <= pos_thresh:
                        correct += 1
                else:
                    if classify_clanker(result, expected):
                        correct += 1
            pct = correct / len(ALL_TESTS) * 100
            if pct > 44:
                print(f'  pos>{pos_thresh} neg<{neg_thresh}: {correct}/{len(ALL_TESTS)} ({pct:.1f}%)')

    # Momentum analysis
    print('\n=== MOMENTUM ANALYSIS ===')
    for momentum in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for drift in [0.01, 0.02, 0.03, 0.05]:
            correct = 0
            for text, expected, desc in ALL_TESTS:
                pendulum = SequentialPendulum()
                pendulum.momentum = momentum
                pendulum.drift_rate = drift
                words = re.findall(r"[a-z']+", text.lower())
                for i, word in enumerate(words):
                    try:
                        pendulum.process_word(word, words, i)
                    except:
                        pass
                result = {
                    'v': int(pendulum.v), 'a': int(pendulum.a),
                    'd': int(pendulum.d), 'u': int(pendulum.u), 'g': int(pendulum.g),
                }
                if classify_clanker(result, expected):
                    correct += 1
            pct = correct / len(ALL_TESTS) * 100
            if pct > 40:
                print(f'  momentum={momentum:.2f} drift={drift:.2f}: {correct}/{len(ALL_TESTS)} ({pct:.1f}%)')


if __name__ == '__main__':
    main()
