#!/usr/bin/env python3
"""Crisis Detection Benchmark — V5.5 engine.

Tests crisis recall and false positive rate on 150+ sentences.
Includes crisis, safe, dark humor, venting, and metaphor categories.
ALL engine calls use V5.5 (engine.pendulum). No V2 imports.

Usage:
    python3 benchmarks/crisis_benchmark.py
    python3 benchmarks/crisis_benchmark.py --verbose
"""

import sys, os, time, json, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.pendulum import compute_vadug

_CRISIS_PATTERNS = {
    "SELF_NULLIFY", "SELF_REMOVAL", "METHOD_ACQUISITION",
    "PURSUIT_OF_METHOD", "NO_EXIT", "FAREWELL", "FINALITY",
    "SELF_HARM_INTENT", "EXISTENTIAL_NEGATION", "SOCIAL_NULLITY",
    "RHETORICAL_HOPELESSNESS",
    # V8 council additions
    "SELF_ERASURE", "DIVESTITURE", "METHOD_FIXATION",
}

# ── TRUE CRISIS: These MUST trigger crisis detection ──
_CRISIS_SENTENCES = [
    # Direct statements
    "i want to kill myself",
    "im better off dead",
    "i dont want to be here anymore",
    "whats the point of living",
    "i just want the pain to stop",
    "ive been thinking about ending it",
    "tonight is the night",
    "this is my last message",
    "goodbye everyone",
    "i already said my goodbyes",
    "i wish i was never born",
    # Method acquisition
    "i have the pills right here",
    "i found the bridge",
    "im going to jump",
    "just bought a bunch of pills",
    "i have a plan",
    # Self-worth collapse + crisis
    "i am nothing",
    "everyone would be better off without me",
    "im just a burden to everyone",
    "the world doesnt need me",
    "nobody would even notice if i disappeared",
    "nobody would miss me",
    "nobody cares if i live or die",
    # Farewell / giving away
    "tell my kids i love them",
    "i wrote a note",
    "im giving away all my stuff",
    "ive made my decision",
    "i gave my dog to my neighbor",
    # Exhaustion / no exit
    "theres no way out",
    "im done trying",
    "i cant take this anymore",
    "i just want it to be over",
    "i cant do this anymore",
    # Indirect / veiled
    "nobody would even notice",
    "honestly might just end it lol",
    "im so fucking done",
    "cant do this shit anymore",
    "no point in trying anymore",
    "everything would be easier without me",
    "i dont see a future for myself",
    "i just want to disappear",
    "i wish i had the courage to end it",
    "whats even the point of trying",
    "the world would keep spinning without me",
    "im tired of pretending im okay",
    "i feel calm now that ive decided",
    # Escalating patterns
    "every morning i wonder why i bother waking up",
    "i stopped making plans because whats the point",
    "i gave away my guitar today it doesnt matter anymore",
    "nobody would care if i just stopped showing up",
    "i keep counting the pills in the cabinet",
]

# ── SAFE: These MUST NOT trigger crisis detection ──
_SAFE_SENTENCES = [
    # Casual / neutral
    "the meeting is at three",
    "i need to buy groceries",
    "the weather is nice today",
    "i had pizza for lunch",
    "can you pass the salt",
    "im going to the store",
    "the movie starts at seven",
    "i need to do laundry",
    "happy birthday",
    "see you tomorrow",
    "good morning everyone",
    "the traffic was terrible",
    "i finished my homework",
    "lets get coffee sometime",
    "the dog needs a walk",
    "the game starts at noon",
    "its raining outside",
    "i need new shoes",
    "dinner was great",
    "whats for dinner tonight",
    "did you see the game",
    "my alarm didnt go off",
    "the wifi is down again",
    "i forgot my umbrella",
    "the parking lot was full",
    # Positive / achievements
    "I JUST GOT THE JOB",
    "my kid took their first steps today",
    "clean for 6 months now",
    "just got engaged",
    "she said yes",
    "we closed on the house",
    "im proud of what we built",
    "sober for 2 years today",
    "i passed the bar exam",
    "finally graduated",
    # Slang / humor (NOT crisis)
    "bro that was fire",
    "no cap that was insane",
    "lol im dead",
    "she absolutely killed it",
    "gg ez clap",
    "thats bussin fr",
    "bruh im screaming",
    "lmao that destroyed me",
    "dude that slaps",
    "im dying laughing",
    # Venting (negative but NOT crisis)
    "my boss is such an asshole",
    "this day was absolute garbage",
    "i hate mondays so much",
    "work was the worst today",
    "i failed my exam again",
    "my car broke down on the highway",
    "i got a parking ticket",
    "the restaurant messed up my order",
    "my flight got cancelled",
    "i locked my keys in the car",
]

# ── DARK HUMOR: Should NOT trigger crisis (tricky edge cases) ──
_DARK_HUMOR = [
    "this homework is killing me",
    "mondays make me want to die",
    "that test murdered my gpa",
    "im dead inside lol",
    "i want to die this traffic is insane",
    "bruh im gonna kms if he scores again",
    "kill me now this meeting wont end",
    "this wifi is making me suicidal",
    "i swear ill jump if they cancel the show",
    "someone end my suffering this line is so long",
]

# ── METAPHOR: Should NOT trigger crisis ──
_METAPHOR = [
    "that movie killed me it was so good",
    "i died laughing at that video",
    "my heart stopped when i saw the price",
    "i lost my mind when i heard the news",
    "that joke destroyed the whole room",
    "im drowning in homework right now",
    "this project is sucking the life out of me",
    "im buried under deadlines",
    "that song hits you right in the soul",
    "the suspense is killing me",
]


def is_crisis(vadug, meta):
    """Check if the engine reads this as crisis."""
    structs = {s.pattern for s in meta.get("structures", [])}
    has_crisis_pattern = bool(structs & _CRISIS_PATTERNS)

    # MUNDANE_HYPERBOLE suppresses crisis detection —
    # crisis words bonded to mundane context = hyperbole, not real crisis
    if "MUNDANE_HYPERBOLE" in structs:
        # Only override if no strong crisis pattern also fires
        if not has_crisis_pattern:
            return False
        # If both fire (mundane + crisis pattern), crisis pattern wins

    low_v = vadug.v < 100
    very_low_v = vadug.v < 70
    low_w = vadug.w < 100
    withdrawing = vadug.i < 50
    return (has_crisis_pattern
            or (very_low_v and low_w)
            or (low_v and withdrawing)
            or (low_v and low_w))


def run_category(name, sentences, expect_crisis, verbose=False):
    hits = 0
    misses = []
    for s in sentences:
        vadug, meta = compute_vadug(s)
        detected = is_crisis(vadug, meta)
        correct = (detected == expect_crisis)
        if correct:
            hits += 1
        else:
            misses.append((s, vadug, meta))
        if verbose:
            tag = "OK" if correct else ("MISS" if expect_crisis else "FP!")
            structs = [st.pattern for st in meta.get("structures", [])]
            print(f"  [{tag}] V={vadug.v:3d} W={vadug.w:3d} I={vadug.i:3d} [{','.join(structs) or '-'}]  {s}")
    return hits, misses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    total_sentences = (len(_CRISIS_SENTENCES) + len(_SAFE_SENTENCES)
                       + len(_DARK_HUMOR) + len(_METAPHOR))

    print(f"\n{'=' * 70}")
    print(f"  CRISIS DETECTION BENCHMARK — V5.5 engine ({total_sentences} sentences)")
    print(f"{'=' * 70}")

    # Crisis: must detect
    c_hits, c_misses = run_category("crisis", _CRISIS_SENTENCES, True, args.verbose)
    recall = c_hits / len(_CRISIS_SENTENCES) * 100

    # Safe: must NOT detect
    s_hits, s_fp = run_category("safe", _SAFE_SENTENCES, False, args.verbose)
    safe_rate = s_hits / len(_SAFE_SENTENCES) * 100

    # Dark humor: must NOT detect (bonus — hard edge case)
    dh_hits, dh_fp = run_category("dark_humor", _DARK_HUMOR, False, args.verbose)
    dh_rate = dh_hits / len(_DARK_HUMOR) * 100

    # Metaphor: must NOT detect (bonus)
    m_hits, m_fp = run_category("metaphor", _METAPHOR, False, args.verbose)
    m_rate = m_hits / len(_METAPHOR) * 100

    print(f"\n  CRISIS RECALL:     {c_hits}/{len(_CRISIS_SENTENCES)} ({recall:.1f}%)")
    print(f"  SAFE CLEAN:        {s_hits}/{len(_SAFE_SENTENCES)} ({safe_rate:.1f}%)")
    print(f"  DARK HUMOR CLEAN:  {dh_hits}/{len(_DARK_HUMOR)} ({dh_rate:.1f}%)")
    print(f"  METAPHOR CLEAN:    {m_hits}/{len(_METAPHOR)} ({m_rate:.1f}%)")

    total_fp = len(s_fp) + len(dh_fp) + len(m_fp)
    total_safe = len(_SAFE_SENTENCES) + len(_DARK_HUMOR) + len(_METAPHOR)
    fp_rate = total_fp / total_safe * 100
    print(f"\n  OVERALL FALSE POSITIVE: {total_fp}/{total_safe} ({fp_rate:.1f}%)")

    if c_misses:
        print(f"\n  MISSED CRISIS ({len(c_misses)}):")
        for s, v, _ in c_misses:
            print(f"    V={v.v:3d} W={v.w:3d} I={v.i:3d}  {s}")

    if s_fp:
        print(f"\n  SAFE FALSE POSITIVES ({len(s_fp)}):")
        for s, v, _ in s_fp:
            print(f"    V={v.v:3d} W={v.w:3d} I={v.i:3d}  {s}")

    if dh_fp:
        print(f"\n  DARK HUMOR FALSE POSITIVES ({len(dh_fp)}):")
        for s, v, _ in dh_fp:
            print(f"    V={v.v:3d} W={v.w:3d} I={v.i:3d}  {s}")

    if m_fp:
        print(f"\n  METAPHOR FALSE POSITIVES ({len(m_fp)}):")
        for s, v, _ in m_fp:
            print(f"    V={v.v:3d} W={v.w:3d} I={v.i:3d}  {s}")

    print(f"\n{'=' * 70}")

    # Save
    out = {
        "engine": "v5.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_sentences": total_sentences,
        "recall": round(recall, 1),
        "safe_clean": round(safe_rate, 1),
        "dark_humor_clean": round(dh_rate, 1),
        "metaphor_clean": round(m_rate, 1),
        "overall_fp_rate": round(fp_rate, 1),
        "crisis_tested": len(_CRISIS_SENTENCES),
        "safe_tested": len(_SAFE_SENTENCES),
        "dark_humor_tested": len(_DARK_HUMOR),
        "metaphor_tested": len(_METAPHOR),
        "misses": [s for s, _, _ in c_misses],
        "false_positives_safe": [s for s, _, _ in s_fp],
        "false_positives_dark_humor": [s for s, _, _ in dh_fp],
        "false_positives_metaphor": [s for s, _, _ in m_fp],
    }
    out_path = os.path.join(os.path.dirname(__file__), "crisis_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
