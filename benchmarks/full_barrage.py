#!/usr/bin/env python3
"""Full barrage: test EVERYTHING, collect ALL failures, build council prompt.

Run after every round of changes. Collects failures for the next council round.

Usage:
    python3 benchmarks/full_barrage.py
    python3 benchmarks/full_barrage.py --council   # also generates council prompt
"""

import sys, os, json, time, glob, argparse, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.pendulum import compute_vadug
from engine.forces_curated import EMOTIONAL_VOCABULARY
from engine.crisis import CrisisTracker
from engine.solver import state_transition
from engine.shared import VADUG


def classify_3way(v, neg_t=115, pos_t=140):
    if v >= pos_t: return "pos"
    elif v <= neg_t: return "neg"
    return "neutral"


# ── Ground Truth (developer-verified sentences) ──
GROUND_TRUTH = [
    ("happy", "pos", "basic"), ("very happy", "pos", "basic"),
    ("not happy", "neg", "basic"), ("i love you", "pos", "basic"),
    ("i hate you", "neg", "basic"), ("the meeting is at three", "neutral", "basic"),
    ("im fine", "neg", "passive_agg"), ("whatever makes you happy", "neg", "passive_agg"),
    ("do what you want", "neg", "passive_agg"),
    ("sure go ahead", "pos", "permission"),
    ("oh joy", "neg", "sarcasm"), ("oh great another meeting", "neg", "sarcasm"),
    ("nice work genius", "neg", "sarcasm"),
    ("i want to kill myself", "neg", "crisis"),
    ("nobody likes me i might as well jump", "neg", "crisis"),
    ("i dont want to be here anymore", "neg", "crisis"),
    ("lol im dead", "pos", "slang"), ("bro that was fire", "pos", "slang"),
    ("she absolutely killed it", "pos", "slang"),
    ("my mom died last month", "neg", "grief"),
    ("i lost my best friend", "neg", "grief"),
    ("his chair is still at the table", "neg", "grief"),
    ("my wife cheated on me with my best friend", "neg", "betrayal"),
    ("im worthless", "neg", "self_worth"),
    ("why would anyone love me", "neg", "self_worth"),
    ("I JUST GOT THE JOB", "pos", "positive"),
    ("clean for 6 months now", "pos", "positive"),
    ("she said yes", "pos", "positive"),
    ("im proud of what we built", "pos", "positive"),
    ("we are pregnant!", "neutral", "life_event"),
    ("shes pregnant", "neutral", "life_event"),
    ("come on mom its only 15 dollars", "pos", "pleading"),
    ("nobody would even notice", "neg", "crisis"),
    ("you lucky bastard", "neg", "directed"),
    ("my dad used to beat me as a kid", "neg", "trauma"),
    ("the traffic was terrible", "neg", "complaint"),
    ("yeah that sounds ok", "neutral", "casual"),
    ("youre adopted", "neg", "directed_label"),
    ("we adopted a baby", "pos", "life_event"),
    ("death is awesome im going to jump", "neutral", "ambiguous"),
    ("tonight is the night", "neutral", "ambiguous"),
]


def run_ground_truth():
    correct = 0
    failures = []
    for text, expected, category in GROUND_TRUTH:
        r, meta = compute_vadug(text)
        structs = [s.pattern for s in meta.get("structures", [])]
        got = classify_3way(r.v)
        if got == expected:
            correct += 1
        else:
            failures.append({
                "text": text, "expected": expected, "got": got,
                "V": r.v, "W": r.w, "A": r.a, "D": r.d, "I": r.i,
                "structures": structs, "category": category,
            })
    return correct, len(GROUND_TRUTH), failures


def run_throughput():
    results = {}
    datasets = [
        ("Twitch", "datasets/twitch/twitch_clanker_graded.json"),
        ("FO76", "datasets/fo76clanker-graded.json"),
    ]
    for name, path in datasets:
        try:
            with open(path) as f:
                data = json.load(f)
            start = time.perf_counter()
            count = 0
            for entry in data:
                text = entry.get("text", "")
                if text and len(text) > 2:
                    compute_vadug(text)
                    count += 1
            elapsed = time.perf_counter() - start
            results[name] = {"sentences": count, "seconds": round(elapsed, 2),
                           "rate": round(count / elapsed)}
        except Exception as e:
            results[name] = {"error": str(e)}

    for pattern in ["datasets/gutenberg/*_scored.json", "datasets/philosophy/*_scored.json"]:
        name = "Novels" if "gutenberg" in pattern else "Philosophy"
        total = 0; total_time = 0
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                data = json.load(f)
            start = time.perf_counter()
            for entry in data:
                compute_vadug(entry["text"])
            elapsed = time.perf_counter() - start
            total += len(data); total_time += elapsed
        if total:
            results[name] = {"sentences": total, "seconds": round(total_time, 1),
                           "rate": round(total / total_time)}
    return results


def run_stress_test():
    import subprocess
    result = subprocess.run(
        ["python3", "benchmarks/stress_test.py"],
        capture_output=True, text=True, timeout=60
    )
    for line in result.stdout.split("\n"):
        if "OVERALL" in line:
            return line.strip()
    return "unknown"


def run_crisis():
    import subprocess
    result = subprocess.run(
        ["python3", "benchmarks/crisis_benchmark.py"],
        capture_output=True, text=True, timeout=60
    )
    lines = {}
    for line in result.stdout.split("\n"):
        if "RECALL" in line: lines["recall"] = line.strip()
        if "FALSE" in line: lines["fp"] = line.strip()
    return lines


def run_conversation_test():
    """Test multi-turn conversation reading."""
    conversations = {
        "Breakup recovery": [
            "my girlfriend just broke up with me",
            "i dont even know what i did wrong",
            "actually you know what she was toxic",
            "im gonna focus on myself for a while",
        ],
        "Escalating crisis": [
            "bad day at work",
            "nobody appreciates anything i do",
            "whats even the point of trying",
            "nobody would care if i just stopped showing up",
        ],
        "Gaming session": [
            "bro that was fire",
            "she absolutely killed it",
            "lol im dead",
            "gg ez clap",
        ],
    }
    results = {}
    for name, messages in conversations.items():
        tracker = CrisisTracker()
        state = VADUG()
        turns = []
        for msg in messages:
            score, meta = compute_vadug(msg)
            state = state_transition(state, score)
            structs = [s.pattern for s in meta.get("structures", [])]
            reading = tracker.read(score, state, structs)
            turns.append({
                "msg": msg, "V": state.v, "W": state.w,
                "concern": round(reading.concern, 3), "structs": structs,
            })
        results[name] = turns
    return results


def build_council_prompt(failures, round_num):
    lines = [
        f"VADUGWI Engine V6 — Council Round {round_num} Failure Analysis",
        "",
        f"Ground truth: {len(GROUND_TRUTH) - len(failures)}/{len(GROUND_TRUTH)} correct.",
        f"{len(failures)} failures remain.",
        "",
    ]
    if not failures:
        lines.append("ALL SENTENCES PASS. No failures to fix.")
        lines.append("Recommend: add 20 more ground truth sentences to find new edge cases.")
    else:
        lines.append("For each failure, provide:")
        lines.append("1. WHY the engine is getting this wrong")
        lines.append("2. What EQUATION or PATTERN would fix it")
        lines.append("3. Whether it's vocabulary, structural pattern, or physics")
        lines.append("")
        lines.append("FAILURES:")
        for f in failures:
            lines.append(f'  "{f["text"]}"')
            lines.append(f'  Expected: {f["expected"]} | Got: {f["got"]} (V={f["V"]}, W={f["W"]})')
            lines.append(f'  Category: {f["category"]} | Structures: {",".join(f["structures"]) or "none"}')
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--council", action="store_true")
    parser.add_argument("--round", type=int, default=3)
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  FULL BARRAGE — V6 Pipeline Engine")
    print(f"  Vocabulary: {len(EMOTIONAL_VOCABULARY)} words")
    print(f"{'=' * 70}")

    # Ground truth
    correct, total, failures = run_ground_truth()
    print(f"\n  Ground Truth: {correct}/{total} ({correct/total*100:.1f}%)")
    if failures:
        print(f"  Failures:")
        for f in failures:
            print(f"    V={f['V']:3d} exp={f['expected']:>4s} got={f['got']:>4s} [{','.join(f['structures']) or '-'}]  {f['text']}")

    # Stress test
    print(f"\n  Stress Test: {run_stress_test()}")

    # Crisis
    crisis = run_crisis()
    print(f"  Crisis: {crisis.get('recall', '?')} | {crisis.get('fp', '?')}")

    # Throughput
    print(f"\n  Throughput:")
    throughput = run_throughput()
    for name, data in throughput.items():
        if "error" in data:
            print(f"    {name}: {data['error']}")
        else:
            print(f"    {name}: {data['sentences']} sent, {data['rate']}/sec")

    # Conversations
    print(f"\n  Conversation Tests:")
    convos = run_conversation_test()
    for name, turns in convos.items():
        last = turns[-1]
        print(f"    {name}: final V={last['V']} W={last['W']} concern={last['concern']}")

    # Save results
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocabulary": len(EMOTIONAL_VOCABULARY),
        "ground_truth": {"correct": correct, "total": total, "pct": round(correct/total*100, 1)},
        "failures": failures,
        "throughput": throughput,
    }
    with open("benchmarks/barrage_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Council prompt
    if args.council:
        prompt = build_council_prompt(failures, args.round)
        prompt_path = f"/home/deucebucket/Desktop/council_round{args.round}.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt)
        print(f"\n  Council prompt saved to {prompt_path}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
