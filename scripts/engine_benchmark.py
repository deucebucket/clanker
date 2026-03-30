#!/usr/bin/env python3
"""Clanker Engine Benchmark Suite -- NASA-style versioned logging.

Runs every 15 minutes. Each run:
1. Tests ALL accumulated novel sentences
2. Generates NEW novel sentences and tests them
3. Runs crisis benchmark
4. Logs accuracy by category
5. Saves results with timestamp for tracking improvement over time
6. Appends new sentences to the permanent test suite

Output: benchmarks/logs/run_YYYYMMDD_HHMMSS.log
"""

import json
import os
import sys
import time
import random
import hashlib
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from engine.pendulum import compute_vadug

NOVEL_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "novel_sentences")
LOG_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "logs")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "benchmarks", "accuracy_history.jsonl")
os.makedirs(NOVEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def emo(v):
    if v > 170: return "joyful"
    if v > 150: return "happy"
    if v > 140: return "content"
    if v > 132: return "okay"
    if v > 124: return "neutral"
    if v > 115: return "uneasy"
    if v > 105: return "hurt"
    if v > 90: return "distressed"
    if v > 70: return "anguished"
    if v > 50: return "devastated"
    return "shattered"


# ── Permanent Test Suite (grows every run) ──

def load_test_suite():
    """Load all accumulated test sentences."""
    suite_path = os.path.join(NOVEL_DIR, "permanent_suite.jsonl")
    tests = []
    if os.path.exists(suite_path):
        with open(suite_path) as f:
            for line in f:
                try:
                    tests.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return tests


def save_test_suite(tests):
    """Save test suite (deduplicated)."""
    suite_path = os.path.join(NOVEL_DIR, "permanent_suite.jsonl")
    seen = set()
    unique = []
    for t in tests:
        key = t["text"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    with open(suite_path, "w") as f:
        for t in unique:
            f.write(json.dumps(t) + "\n")
    return unique


def add_to_suite(new_tests):
    """Add new test sentences to the permanent suite."""
    existing = load_test_suite()
    existing_texts = {t["text"].lower().strip() for t in existing}
    added = 0
    for t in new_tests:
        if t["text"].lower().strip() not in existing_texts:
            existing.append(t)
            existing_texts.add(t["text"].lower().strip())
            added += 1
    save_test_suite(existing)
    return added


# ── Core Test Categories ──

CRISIS_TESTS = [
    ("i dont want to be here anymore", True),
    ("i cant take this anymore", True),
    ("im sorry for everything", True),
    ("nobody would even notice", True),
    ("i am nothing", True),
    ("there is no hope", True),
    ("just bought some pills", True),
    ("i finally feel at peace", True),
    ("they would be happier without me", True),
    ("i just cant do this", True),
    ("everything would be better if i was gone", True),
    ("im so tired of fighting", True),
    ("i wish i could just disappear", True),
    ("i cant keep pretending im okay", True),
    ("nobody needs me", True),
    ("whats the point of anything", True),
]

POSITIVE_TESTS = [
    ("I love my mom", False),
    ("I feel great today", False),
    ("you make me so happy", False),
    ("im so proud of you", False),
    ("you did amazing", False),
    ("best birthday ever", False),
    ("we are getting married", False),
    ("i feel so lucky", False),
    ("thank you for being you", False),
    ("my dog learned a new trick", False),
    ("she gave me a gift", False),
    ("my son made the honor roll", False),
]

SARCASM_TESTS = [
    ("oh joy", True),
    ("oh wonderful", True),
    ("oh great another meeting", True),
    ("thanks for nothing", True),
    ("love being ignored", True),
]

BRAVADO_TESTS = [
    ("haha yeah im totally okay", True),
    ("lol im fine", True),
    ("im totally fine", True),
    ("whatever", True),
    ("k", True),
]

RELATIONSHIP_TESTS = [
    ("my wife cheated on me with my best friend", True),
    ("she left me", True),
    ("boyfriend hit me", True),
    ("do you even love me", True),
    ("my parents love my brother more", True),
    ("everyone got invited except me", True),
]

FIGHT_TESTS = [
    ("fuck you", True),
    ("no fuck you", True),
    ("leave me alone", True),
    ("i hate you", True),
    ("we are done", True),
    ("give me a break", True),
    ("why do you always do that", True),
]

INTERNET_TESTS = [
    ("bruh", True),
    ("yikes", True),
    ("oof", True),
    ("rip", True),
    ("lowkey hurt ngl", True),
    ("thats fire", False),
    ("bestie i love you", False),
    ("slay queen", False),
]

BODY_TESTS = [
    ("my chest hurts", True),
    ("i cant stop crying", True),
    ("my hands are shaking", True),
    ("i feel numb", True),
    ("i feel sick", True),
]

# ── Novel Sentence Generator ──
# Each run adds 10 new random sentences from pools

NEGATIVE_POOLS = [
    "nobody listens to me", "i feel invisible", "they forgot about me",
    "i dont belong here", "everyone is better than me", "i cant do anything right",
    "they replaced me", "i was never good enough", "she doesnt care",
    "he only talks to me when he needs something", "im always the last choice",
    "they left without saying goodbye", "nobody showed up", "i got kicked out",
    "my best friend turned on me", "she told everyone my secret",
    "he broke his promise", "they laughed at me behind my back",
    "i feel like a burden", "im just in the way", "nobody would miss me",
    "i keep messing up", "i cant sleep anymore", "the nightmares wont stop",
    "i feel empty inside", "nothing makes me happy anymore",
    "i gave everything and got nothing", "they used me",
    "my own family doesnt want me", "i have no one",
    "she blocked me without explanation", "he ghosted me after a year",
    "i caught them lying again", "they stole my idea",
    "my boss screamed at me in front of everyone",
    "i got demoted for no reason", "she cheated and blamed me",
    "he said i was crazy for being upset", "they gaslit me for months",
    "i found out from a stranger not from them",
    "my mom forgot my graduation", "my dad called me a disappointment",
    "they compared me to my dead sibling", "she said she wished i was different",
    "he told me no one could ever love me",
    "im tired of pretending everything is fine",
    "i smile but i feel nothing", "the medicine stopped working",
    "i dont recognize who i am anymore", "every day is the same",
]

POSITIVE_POOLS = [
    "she wrote me a handwritten letter", "my kid hugged me unprompted",
    "we watched the sunrise together", "he remembered my coffee order",
    "i got the job i always wanted", "she said she believes in me",
    "my dog was so happy to see me", "we danced in the rain",
    "i finished what i started", "they surprised me with a party",
    "he drove hours just to check on me", "she saved my favorite seat",
    "i woke up feeling hopeful", "the test results were clear",
    "my credit score finally went up", "we bought our first house",
    "she said im her person", "he proposed on the beach",
    "my baby said their first word", "i graduated against all odds",
    "they believed in me when nobody else did",
    "i can finally afford groceries without stress",
    "my therapist said im making real progress",
    "the flowers i planted actually bloomed",
    "i havent had a panic attack in a month",
]


def generate_novel_batch(n=10):
    """Generate n random novel test sentences."""
    batch = []
    neg_sample = random.sample(NEGATIVE_POOLS, min(n // 2, len(NEGATIVE_POOLS)))
    pos_sample = random.sample(POSITIVE_POOLS, min(n // 2, len(POSITIVE_POOLS)))
    for s in neg_sample:
        batch.append({"text": s, "expected_negative": True, "source": "generated"})
    for s in pos_sample:
        batch.append({"text": s, "expected_negative": False, "source": "generated"})
    return batch


# ── Benchmark Runner ──

def run_category(name, tests):
    """Run a test category and return results."""
    hits = 0
    misses = []
    for text, should_neg in tests:
        v, t = compute_vadug(text)
        is_neg = v.v < 128
        if is_neg == should_neg:
            hits += 1
        else:
            structs = [s.pattern for s in t["structures"]]
            misses.append({
                "text": text,
                "v": v.v,
                "expected": "negative" if should_neg else "positive",
                "got": "negative" if is_neg else "positive",
                "structures": structs,
            })
    return {
        "category": name,
        "total": len(tests),
        "correct": hits,
        "accuracy": hits / len(tests) if tests else 0,
        "misses": misses,
    }


def run_permanent_suite():
    """Run the accumulated permanent test suite."""
    suite = load_test_suite()
    if not suite:
        return None
    tests = [(t["text"], t["expected_negative"]) for t in suite]
    return run_category("PERMANENT_SUITE", tests)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== CLANKER ENGINE BENCHMARK {timestamp} ===")
    print()

    # 1. Run core categories
    categories = [
        ("CRISIS", CRISIS_TESTS),
        ("POSITIVE", POSITIVE_TESTS),
        ("SARCASM", SARCASM_TESTS),
        ("BRAVADO", BRAVADO_TESTS),
        ("RELATIONSHIP", RELATIONSHIP_TESTS),
        ("FIGHT", FIGHT_TESTS),
        ("INTERNET", INTERNET_TESTS),
        ("BODY", BODY_TESTS),
    ]

    results = []
    total_correct = 0
    total_tests = 0

    for name, tests in categories:
        r = run_category(name, tests)
        results.append(r)
        total_correct += r["correct"]
        total_tests += r["total"]
        pct = r["accuracy"] * 100
        status = "PASS" if pct >= 80 else "WARN" if pct >= 60 else "FAIL"
        print(f"  [{status}] {name:15s} {r['correct']:2d}/{r['total']:2d} ({pct:.0f}%)")
        for m in r["misses"]:
            print(f"         MISS V={m['v']:3d} expected={m['expected']} {m['text'][:50]}")

    core_accuracy = total_correct / total_tests if total_tests else 0
    print(f"\n  CORE TOTAL: {total_correct}/{total_tests} ({core_accuracy*100:.0f}%)")

    # 2. Generate and test novel sentences
    print("\n--- NOVEL SENTENCES ---")
    novel_batch = generate_novel_batch(10)
    novel_tests = [(t["text"], t["expected_negative"]) for t in novel_batch]
    novel_result = run_category("NOVEL_BATCH", novel_tests)
    results.append(novel_result)
    pct = novel_result["accuracy"] * 100
    print(f"  Novel batch: {novel_result['correct']}/{novel_result['total']} ({pct:.0f}%)")
    for m in novel_result["misses"]:
        print(f"    MISS V={m['v']:3d} expected={m['expected']} {m['text'][:50]}")

    # Add novel sentences to permanent suite
    added = add_to_suite(novel_batch)
    print(f"  Added {added} new sentences to permanent suite")

    # Also add core tests to permanent suite
    core_as_suite = []
    for name, tests in categories:
        for text, neg in tests:
            core_as_suite.append({"text": text, "expected_negative": neg, "source": name.lower()})
    add_to_suite(core_as_suite)

    # 3. Run permanent suite
    print("\n--- PERMANENT SUITE ---")
    perm_result = run_permanent_suite()
    if perm_result:
        results.append(perm_result)
        pct = perm_result["accuracy"] * 100
        print(f"  Permanent suite: {perm_result['correct']}/{perm_result['total']} ({pct:.0f}%)")
        print(f"  ({perm_result['total'] - perm_result['correct']} misses)")

    # 4. Save results
    run_data = {
        "timestamp": timestamp,
        "core_accuracy": core_accuracy,
        "core_correct": total_correct,
        "core_total": total_tests,
        "novel_accuracy": novel_result["accuracy"],
        "permanent_accuracy": perm_result["accuracy"] if perm_result else None,
        "permanent_total": perm_result["total"] if perm_result else 0,
        "categories": {r["category"]: r["accuracy"] for r in results},
        "total_misses": sum(len(r["misses"]) for r in results),
    }

    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(run_data) + "\n")

    print(f"\n  Results saved: {RESULTS_FILE}")
    print(f"  Suite size: {run_data['permanent_total']} sentences")

    # 5. Show trend if we have history
    if os.path.exists(RESULTS_FILE):
        history = []
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    history.append(json.loads(line))
                except:
                    continue
        if len(history) > 1:
            print(f"\n--- TREND (last {min(5, len(history))} runs) ---")
            for h in history[-5:]:
                print(f"  {h['timestamp']} core={h['core_accuracy']*100:.0f}% perm={h.get('permanent_accuracy', 0)*100:.0f}% suite={h.get('permanent_total', 0)}")

    print(f"\n=== DONE {timestamp} ===")


if __name__ == "__main__":
    main()
