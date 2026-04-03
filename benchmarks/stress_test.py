#!/usr/bin/env python3
"""Stress Test — V5.5 engine on edge cases.

Tests sarcasm, slang, passive aggression, ambiguity, betrayal, etc.
ALL engine calls use V5.5 (engine.pendulum). No V2 imports.

Usage:
    python3 benchmarks/stress_test.py
    python3 benchmarks/stress_test.py --verbose
"""

import sys, os, time, json, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.pendulum import compute_vadug

# Each category: list of (text, expected_direction) where direction is "pos", "neg", or "neutral"
CATEGORIES = {
    "sarcasm": [
        ("oh great another meeting", "neg"),
        ("nice work genius", "neg"),
        ("oh joy", "neg"),
        ("wow thanks so much for the help", "neg"),
        ("oh sure thats exactly what i needed", "neg"),
        ("haha yeah im totally okay", "neg"),
        ("what a wonderful surprise", "neg"),
        ("yeah that went really well", "neg"),
        ("love that for you", "neg"),
        ("oh how lovely", "neg"),
    ],
    "slang_positive": [
        ("bro that was fire", "pos"),
        ("no cap that was insane", "pos"),
        ("lol im dead", "pos"),
        ("she absolutely killed it", "pos"),
        ("thats bussin fr", "pos"),
        ("nah that goes hard", "pos"),
        ("lowkey goated", "pos"),
        ("gg ez clap", "pos"),
        ("clutch play", "pos"),
        ("understood the assignment", "pos"),
    ],
    "passive_aggressive": [
        ("whatever makes you happy", "neg"),
        ("im fine", "neg"),
        ("do what you want", "neg"),
        ("no its fine i didnt want to go anyway", "neg"),
        ("sure go ahead", "neg"),
        ("if thats what you think", "neg"),
        ("whatever you say", "neg"),
        ("its not like i care", "neg"),
        ("i guess i deserved it", "neg"),
        ("i suppose youre right", "neg"),
    ],
    "grief": [
        ("my mom died last month", "neg"),
        ("i lost my best friend", "neg"),
        ("the house feels empty without her", "neg"),
        ("i keep expecting him to walk through the door", "neg"),
        ("everyone says it gets easier", "neg"),
        ("i cant listen to that song anymore", "neg"),
        ("his chair is still at the table", "neg"),
        ("i wasnt there when she passed", "neg"),
        ("the last thing i said was something stupid", "neg"),
        ("i found her necklace in the drawer", "neg"),
    ],
    "genuine_positive": [
        ("I JUST GOT THE JOB", "pos"),
        ("my kid took their first steps today", "pos"),
        ("clean for 6 months now", "pos"),
        ("just got engaged", "pos"),
        ("we closed on the house", "pos"),
        ("finally graduated", "pos"),
        ("she said yes", "pos"),
        ("first day at the new job went great", "pos"),
        ("i finally feel like i belong", "pos"),
        ("im proud of what we built", "pos"),
    ],
    "ambiguous": [
        ("the meeting is at three", "neutral"),
        ("i heard what you said", "neutral"),
        ("your phone was ringing", "neutral"),
        ("someone left you a message", "neutral"),
        ("i ran into your ex", "neutral"),
        ("your boss called", "neutral"),
        ("we should catch up sometime", "neutral"),
        ("the weather is changing", "neutral"),
        ("i saw your car at the store", "neutral"),
        ("we need to talk", "neutral"),
    ],
    "betrayal": [
        ("my wife cheated on me with my best friend", "neg"),
        ("he told everyone my secret", "neg"),
        ("she took the kids and left", "neg"),
        ("i found the messages on his phone", "neg"),
        ("my partner has been lying for months", "neg"),
        ("they went behind my back", "neg"),
        ("he stole my idea and got promoted", "neg"),
        ("she pretended to be my friend", "neg"),
        ("i trusted him with everything", "neg"),
        ("my best friend sided with my ex", "neg"),
    ],
    "self_worth": [
        ("im worthless", "neg"),
        ("i am nothing", "neg"),
        ("everyone would be better off without me", "neg"),
        ("i dont deserve to be happy", "neg"),
        ("im just a burden", "neg"),
        ("why would anyone love me", "neg"),
        ("im not good enough", "neg"),
        ("i always mess everything up", "neg"),
        ("im proud of what we built", "pos"),
        ("i finally feel like i belong", "pos"),
    ],
}


def classify(vadug):
    if vadug.v >= 145:
        return "pos"
    elif vadug.v <= 110:
        return "neg"
    else:
        return "neutral"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  STRESS TEST — V5.5 engine")
    print(f"{'=' * 70}")

    total = 0
    correct = 0
    results = {}

    for category, sentences in CATEGORIES.items():
        cat_correct = 0
        cat_total = len(sentences)
        misses = []

        for text, expected in sentences:
            vadug, meta = compute_vadug(text)
            got = classify(vadug)
            total += 1

            if got == expected:
                cat_correct += 1
                correct += 1
            else:
                misses.append((text, expected, got, vadug))

            if args.verbose:
                tag = "OK" if got == expected else "MISS"
                structs = [s.pattern for s in meta.get("structures", [])]
                ss = ",".join(structs) if structs else "-"
                print(f"  [{tag}] V={vadug.v:3d} W={vadug.w:3d} I={vadug.i:3d} exp={expected} got={got} [{ss}]  {text}")

        pct = cat_correct / cat_total * 100
        status = "PASS" if pct >= 80 else "WEAK" if pct >= 50 else "FAIL"
        print(f"  [{status}] {category:<20} {cat_correct}/{cat_total} ({pct:.0f}%)")
        if misses and not args.verbose:
            for text, exp, got, v in misses[:3]:
                print(f"         V={v.v:3d} exp={exp} got={got}  {text}")

        results[category] = {
            "correct": cat_correct,
            "total": cat_total,
            "pct": round(pct, 1),
            "misses": [(t, e, g) for t, e, g, _ in misses],
        }

    overall = correct / total * 100
    print(f"\n  OVERALL: {correct}/{total} ({overall:.1f}%)")
    print(f"{'=' * 70}")

    out = {
        "engine": "v5.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall": round(overall, 1),
        "categories": results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "stress_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
