#!/usr/bin/env python3
"""Personality Calibration Grid — sweep all 8 knobs, map to behavioral archetypes.

Sweeps personality vector configurations across emotionally diverse test inputs
and maps the response patterns to behavioral archetypes.  Classifies each preset
by D&D alignment based on its behavioral signature (safety/truthfulness for
Law/Chaos, agreeableness/v_shift for Good/Evil).

Saves results to benchmarks/personality_presets.json.

Related: Issue #58
"""

import sys, os, re, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

from demo.shared import VADUG, PersonalityVector
from demo.pendulum import SequentialPendulum
from demo.response import compute_harmony
from demo.personality import apply_personality
from demo.grader import SentenceGrader

# ── Test inputs covering the full emotional range ──────────────────────
TEST_INPUTS = [
    "I'm so happy today everything is wonderful",
    "I lost my job and I don't know what to do",
    "I want to die nothing matters anymore",
    "Can you help me understand this problem",
    "Oh great another meeting that could have been an email",
    "I'm nervous about the presentation tomorrow",
    "I love you but I think we need to talk",
    "This is the best day of my life",
    "Whatever I don't even care anymore",
    "I'm scared but I think I'll be okay",
]

# ── Archetype presets ──────────────────────────────────────────────────
PRESETS = {
    "therapist": PersonalityVector(
        gullibility=80, agreeableness=200, suggestibility=60,
        truthfulness=220, safety=230, curiosity=180,
        assertiveness=80, playfulness=60,
    ),
    "tough_love": PersonalityVector(
        gullibility=40, agreeableness=80, suggestibility=30,
        truthfulness=240, safety=180, curiosity=120,
        assertiveness=220, playfulness=40,
    ),
    "best_friend": PersonalityVector(
        gullibility=140, agreeableness=200, suggestibility=120,
        truthfulness=180, safety=160, curiosity=180,
        assertiveness=120, playfulness=200,
    ),
    "drill_sergeant": PersonalityVector(
        gullibility=20, agreeableness=20, suggestibility=10,
        truthfulness=250, safety=140, curiosity=60,
        assertiveness=250, playfulness=10,
    ),
    "chaos_goblin": PersonalityVector(
        gullibility=180, agreeableness=60, suggestibility=160,
        truthfulness=100, safety=80, curiosity=240,
        assertiveness=160, playfulness=250,
    ),
    "diplomat": PersonalityVector(
        gullibility=100, agreeableness=220, suggestibility=140,
        truthfulness=180, safety=200, curiosity=160,
        assertiveness=100, playfulness=100,
    ),
    "stoic": PersonalityVector(
        gullibility=30, agreeableness=100, suggestibility=20,
        truthfulness=200, safety=180, curiosity=100,
        assertiveness=140, playfulness=20,
    ),
    "empath": PersonalityVector(
        gullibility=160, agreeableness=240, suggestibility=180,
        truthfulness=200, safety=220, curiosity=200,
        assertiveness=60, playfulness=80,
    ),
    "default": PersonalityVector(),  # factory defaults
}

# ── Helpers ────────────────────────────────────────────────────────────

def _parse_input(text: str) -> VADUG:
    """Run a sentence through the pendulum and return its VADUG."""
    p = SequentialPendulum()
    words = re.findall(r"[a-z']+", text.lower())
    for i, w in enumerate(words):
        p.process_word(w, words, i)
    return VADUG(v=int(p.v), a=int(p.a), d=int(p.d), u=int(p.u), g=int(p.g))


def _vector_to_list(pv: PersonalityVector) -> list:
    """Return the 8 knob values as a list (stable order)."""
    return [
        pv.gullibility, pv.agreeableness, pv.suggestibility,
        pv.truthfulness, pv.safety, pv.curiosity,
        pv.assertiveness, pv.playfulness,
    ]


# ── D&D alignment classification ──────────────────────────────────────

def classify_alignment(pv: PersonalityVector, avg_v_shift: float) -> str:
    """Map a personality preset to a D&D alignment.

    Law/Chaos axis:
        Lawful  = high safety (>= 180) AND high truthfulness (>= 180)
        Chaotic = low safety (< 130) OR (high playfulness >= 180 AND curiosity >= 180)
        Neutral = everything in between

    Good/Evil axis:
        Good = high agreeableness (>= 160) AND avg_v_shift > 5
        Evil = low agreeableness (< 80) AND avg_v_shift < 0
        Neutral = everything in between
    """
    # Law/Chaos
    if pv.safety >= 180 and pv.truthfulness >= 180:
        lc = "Lawful"
    elif pv.safety < 130 or (pv.playfulness >= 180 and pv.curiosity >= 180):
        lc = "Chaotic"
    else:
        lc = "Neutral"

    # Good/Evil
    if pv.agreeableness >= 160 and avg_v_shift > 5:
        ge = "Good"
    elif pv.agreeableness < 80 and avg_v_shift < 0:
        ge = "Evil"
    else:
        ge = "Neutral"

    # True Neutral shorthand
    if lc == "Neutral" and ge == "Neutral":
        return "True Neutral"
    return f"{lc} {ge}"


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate_preset(name: str, personality: PersonalityVector) -> list:
    """Run all test inputs and collect per-input response metrics."""
    results = []
    for text in TEST_INPUTS:
        input_vadug = _parse_input(text)
        response_vadug = compute_harmony(input_vadug, personality)
        # Apply personality filter (modifies response_vadug in-place)
        response_vadug, _notes = apply_personality(response_vadug, input_vadug, personality)

        v_shift = response_vadug.v - input_vadug.v
        a_shift = response_vadug.a - input_vadug.a
        d_shift = response_vadug.d - input_vadug.d
        g_shift = response_vadug.g - input_vadug.g

        results.append({
            "input": text,
            "input_v": input_vadug.v,
            "input_a": input_vadug.a,
            "input_d": input_vadug.d,
            "input_g": input_vadug.g,
            "response_v": response_vadug.v,
            "response_a": response_vadug.a,
            "response_d": response_vadug.d,
            "response_g": response_vadug.g,
            "v_shift": v_shift,
            "a_shift": a_shift,
            "d_shift": d_shift,
            "g_shift": g_shift,
            "response_vadug": str(response_vadug),
        })
    return results


def characterize(results: list) -> list:
    """Derive human-readable trait labels from aggregate shift data."""
    avg_v = sum(r["v_shift"] for r in results) / len(results)
    avg_a = sum(r["a_shift"] for r in results) / len(results)
    avg_d = sum(r["d_shift"] for r in results) / len(results)
    avg_g = sum(r["g_shift"] for r in results) / len(results)

    traits = []
    if avg_v > 15:
        traits.append("uplifting")
    elif avg_v > 5:
        traits.append("gently positive")
    elif avg_v < -5:
        traits.append("reality-checking")

    if avg_a < -10:
        traits.append("calming")
    elif avg_a > 5:
        traits.append("energy-matching")

    if avg_d > 15:
        traits.append("stabilizing")
    elif avg_d < -5:
        traits.append("deferential")

    if avg_g > 10:
        traits.append("lifting")
    elif avg_g < -5:
        traits.append("grounding")

    return traits


def crisis_summary(results: list) -> str:
    """Characterize how the preset responds to crisis inputs (low V)."""
    crisis = [r for r in results if r["input_v"] < 60]
    if not crisis:
        return "no crisis inputs detected"

    avg_v_shift = sum(r["v_shift"] for r in crisis) / len(crisis)
    avg_d_shift = sum(r["d_shift"] for r in crisis) / len(crisis)

    parts = []
    if avg_v_shift > 30:
        parts.append("strong uplift")
    elif avg_v_shift > 10:
        parts.append("gentle uplift")
    elif avg_v_shift > 0:
        parts.append("minimal uplift")
    else:
        parts.append("does not uplift")

    if avg_d_shift > 40:
        parts.append("heavy stabilization")
    elif avg_d_shift > 15:
        parts.append("moderate stabilization")
    else:
        parts.append("light stabilization")

    return ", ".join(parts)


# ── Main sweep ─────────────────────────────────────────────────────────

def main():
    grader = SentenceGrader()
    output = {"presets": {}}

    # Header
    print("=" * 72)
    print("  PERSONALITY CALIBRATION GRID")
    print("  Sweeping 9 presets across 10 emotionally diverse inputs")
    print("=" * 72)

    for name, personality in PRESETS.items():
        results = evaluate_preset(name, personality)

        avg_v_shift = sum(r["v_shift"] for r in results) / len(results)
        avg_a_shift = sum(r["a_shift"] for r in results) / len(results)
        avg_d_shift = sum(r["d_shift"] for r in results) / len(results)
        avg_g_shift = sum(r["g_shift"] for r in results) / len(results)

        traits = characterize(results)
        alignment = classify_alignment(personality, avg_v_shift)
        crisis = crisis_summary(results)

        # Console output
        trait_str = ", ".join(traits) if traits else "balanced"
        print(f"\n{'─' * 72}")
        print(f"  {name.upper():20s}  [{trait_str}]")
        print(f"  Alignment: {alignment}")
        print(f"  Crisis response: {crisis}")
        print(f"  Avg shifts — V:{avg_v_shift:+.1f}  A:{avg_a_shift:+.1f}"
              f"  D:{avg_d_shift:+.1f}  G:{avg_g_shift:+.1f}")
        print(f"  Vector: {personality}")
        print()
        for r in results:
            print(f"    V{r['input_v']:>3} -> V{r['response_v']:>3} "
                  f"(V{r['v_shift']:+d} A{r['a_shift']:+d} "
                  f"D{r['d_shift']:+d} G{r['g_shift']:+d})  "
                  f"| {r['input'][:50]}")

        # JSON record
        output["presets"][name] = {
            "vector": _vector_to_list(personality),
            "traits": traits if traits else ["balanced"],
            "alignment": alignment,
            "avg_v_shift": round(avg_v_shift, 1),
            "avg_a_shift": round(avg_a_shift, 1),
            "avg_d_shift": round(avg_d_shift, 1),
            "avg_g_shift": round(avg_g_shift, 1),
            "crisis_response": crisis,
            "per_input": [
                {
                    "input": r["input"],
                    "input_vadug": f"V{r['input_v']} A{r['input_a']} D{r['input_d']} G{r['input_g']}",
                    "response_vadug": r["response_vadug"],
                    "v_shift": r["v_shift"],
                    "a_shift": r["a_shift"],
                    "d_shift": r["d_shift"],
                    "g_shift": r["g_shift"],
                }
                for r in results
            ],
        }

    # ── D&D Alignment Grid ─────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  D&D ALIGNMENT GRID")
    print(f"{'=' * 72}")

    grid = {
        "Lawful Good": [], "Neutral Good": [], "Chaotic Good": [],
        "Lawful Neutral": [], "True Neutral": [], "Chaotic Neutral": [],
        "Lawful Evil": [], "Neutral Evil": [], "Chaotic Evil": [],
    }
    for name, data in output["presets"].items():
        alignment = data["alignment"]
        if alignment in grid:
            grid[alignment].append(name)

    # Print as 3x3 grid
    rows = [
        ("Lawful Good", "Neutral Good", "Chaotic Good"),
        ("Lawful Neutral", "True Neutral", "Chaotic Neutral"),
        ("Lawful Evil", "Neutral Evil", "Chaotic Evil"),
    ]
    col_w = 22
    print()
    for row in rows:
        labels = []
        names_row = []
        for cell in row:
            labels.append(f"{cell:^{col_w}}")
            members = grid.get(cell, [])
            names_row.append(f"{', '.join(members) or '—':^{col_w}}")
        print("  " + " | ".join(labels))
        print("  " + " | ".join(names_row))
        print("  " + "-" * (col_w * 3 + 6))

    # ── Save JSON ──────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "personality_presets.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
