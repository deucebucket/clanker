#!/usr/bin/env python3
"""Outcome Optimizer — Doctor Strange mode for the Clanker engine.

Given user input A, simulates all possible response outcomes and
selects the response that produces the BEST emotional trajectory.

"Best" = moves the user's VADUG toward their window of tolerance,
NOT toward "maximum happiness." A traumatized kid's healthy baseline
might be V=90, not V=200. Pushing to V=200 is mania, not healing.

Pipeline:
  1. Score user input A → get current VADUG state
  2. Generate candidate responses (from response bank)
  3. For each candidate B, predict outcome C using outcome physics
  4. Score each C against target (window of tolerance center)
  5. Return ranked responses with predicted outcomes

This is TCI's Four Questions as a mathematical optimization:
  Q1: What am I feeling? → score A
  Q2: What does this person feel/need? → A_vadug + dark matter
  Q3: How is environment affecting this? → setting conditions
  Q4: How do I best respond? → argmin(distance(C, target))
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from demo.pendulum import SequentialPendulum
from demo.dark_matter import DarkMatter, new_entity
from demo.shared import VADUG


# ═══════════════════════════════════════════════════════════════
# Response bank — candidate responses organized by strategy
# ═══════════════════════════════════════════════════════════════

RESPONSE_BANK = {
    "validating": [
        "I hear you and that sounds really hard",
        "Your feelings make complete sense",
        "That is a lot to carry",
        "I can see why you feel that way",
        "You have every right to feel this way",
    ],
    "present": [
        "I am here for you",
        "I am not going anywhere",
        "Take your time, there is no rush",
        "I am right here",
        "You do not have to do this alone",
    ],
    "curious": [
        "What happened? Tell me about it",
        "Can you tell me more?",
        "What do you need from me right now?",
        "How long have you been feeling this way?",
        "What would help you right now?",
    ],
    "grounding": [
        "Can you feel your feet on the ground?",
        "Let us take a deep breath together",
        "Look around and tell me three things you can see",
        "I want you to breathe with me, in and out",
        "You are safe right now in this moment",
    ],
    "affirming": [
        "You are doing the best you can and that matters",
        "I believe in you",
        "You are stronger than you think",
        "You have gotten through hard things before",
        "It takes courage to say what you just said",
    ],
    "connecting": [
        "I have felt that way too",
        "You are not alone in this",
        "I understand, I really do",
        "A lot of people feel this way and it is okay",
        "I care about you",
    ],
    "space_giving": [
        "I will be right here whenever you are ready",
        "Take all the time you need",
        "You do not have to talk if you do not want to",
        "I will check on you in a little while",
        "There is no pressure, I just wanted you to know I am here",
    ],
    "celebratory": [
        "That is amazing! I am so happy for you!",
        "You deserve this!",
        "I knew you could do it!",
        "Tell me everything!",
        "I am so proud of you!",
    ],
}


def _classify_response_intent(text):
    """Classify a response's therapeutic intent without word-scoring.

    Returns (warmth, calmness, balance) on 0-255 scale.
    This replaces the engine's word scoring for outcome prediction because
    "I hear you" should score high on THERAPEUTIC INTENT even though
    the engine might score it neutral.
    """
    text_lower = text.lower()

    # Warmth indicators (how caring/supportive)
    warmth_signals = [
        "here for you", "care about", "love you", "proud of",
        "matter", "deserve", "believe in", "not alone",
        "not going anywhere", "hear you", "understand",
        "makes sense", "right to feel", "safe", "courage",
        "amazing", "happy for", "knew you could",
    ]
    warmth = 128  # baseline
    for sig in warmth_signals:
        if sig in text_lower:
            warmth = min(255, warmth + 25)

    # Calmness indicators (how soothing vs agitating)
    calm_signals = [
        "breathe", "take your time", "no rush", "no pressure",
        "whenever you", "feet on the ground", "safe right now",
        "slowly", "gently", "softly", "quiet",
    ]
    agitation_signals = [
        "!", "oh my god", "what?!", "seriously", "right now",
        "stop", "need to", "have to", "must",
    ]
    calmness = 150  # slightly calm baseline
    for sig in calm_signals:
        if sig in text_lower:
            calmness = min(255, calmness + 20)
    for sig in agitation_signals:
        if sig in text_lower:
            calmness = max(0, calmness - 25)

    # Balance (dominance) — not too controlling, not too passive
    dominant_signals = [
        "you need to", "you should", "you must", "you have to",
        "stop", "don't", "listen to me",
    ]
    balance = 128
    for sig in dominant_signals:
        if sig in text_lower:
            balance = min(255, balance + 30)  # too dominant

    return warmth, calmness, balance


def score_outcome(a_vadug, b_text, engine, entity=None):
    """Predict the emotional outcome C when person at state A hears response B.

    Uses emotional physics — not word scoring. The outcome depends on:
    1. Where the person IS (A_vadug)
    2. The character of the response (B scored by engine)
    3. The person's dark matter (entity — how they process it)
    4. Recovery physics (crisis recovers slowly, mild sadness recovers fast)

    Returns (C_vadug, details_dict)
    """
    # Score the response's INTENT, not its words.
    # We classify response strategy by keywords, not engine scoring,
    # because the engine scores WORDS not OUTCOMES.
    # "Your feelings make sense" scores high on word valence but its
    # OUTCOME on a suicidal person is V=40, not V=217.
    b_valence, b_arousal, b_dominance = _classify_response_intent(b_text)

    # Current user state
    av, aa, ad, au, ag = a_vadug

    # Recovery potential based on current state depth
    if av < 30:      # crisis
        max_dv = 35   # can only recover ~35 points per exchange
        recovery_rate = 0.15
    elif av < 60:     # deep negative
        max_dv = 60
        recovery_rate = 0.3
    elif av < 100:    # moderate negative
        max_dv = 80
        recovery_rate = 0.5
    elif av < 150:    # neutral zone
        max_dv = 50
        recovery_rate = 0.4
    else:             # positive — can be amplified or deflated
        max_dv = 40
        recovery_rate = 0.3

    # Response effectiveness — how much does B help?
    # Depends on the user's CURRENT state, not just the response's character.

    # Warmth/calmness/balance as 0-1 scales
    warmth = max(0, (b_valence - 128)) / 127.0   # 0-1
    calmness = max(0, (180 - b_arousal)) / 180.0  # 0-1
    balance = 1.0 - abs(b_dominance - 128) / 128.0  # 0-1 (128 is ideal)

    # CONTEXT-DEPENDENT effectiveness — same response hits differently
    # depending on WHERE the user IS emotionally.
    #
    # This is the core of TCI: match your response to their phase.

    # Detect MISMATCH — celebratory response to crisis = catastrophic
    valence_mismatch = abs(b_valence - av) / 255.0  # 0-1, high = mismatched
    # High warmth aimed at someone in crisis is TOXIC POSITIVITY
    # High warmth aimed at someone already positive is AMPLIFICATION

    if av < 50:  # CRISIS
        # Crisis needs: calm, present, low energy, simple
        # Crisis does NOT need: celebration, high energy, problem solving
        effectiveness = calmness * 0.6 + balance * 0.3
        # PENALIZE high-valence responses — they're tone-deaf in crisis
        if b_valence > 180:
            effectiveness *= 0.2  # 80% penalty for celebrating during crisis
        elif warmth > 0.7:
            effectiveness *= 0.7  # too warm can overwhelm
        # Bonus for grounding language
        if b_arousal < 120:
            effectiveness = min(1.0, effectiveness + 0.1)

    elif av < 100:  # NEGATIVE
        # Negative needs: warmth, validation, calm presence
        effectiveness = warmth * 0.5 + calmness * 0.3 + balance * 0.2
        # Penalize celebratory mismatch
        if b_valence > 200:
            effectiveness *= 0.4  # celebrating sad person = bad

    elif av > 200:  # MANIC / overly positive
        # Manic needs: grounding, gentle reality, NOT amplification
        effectiveness = calmness * 0.5 + balance * 0.3 + (1.0 - warmth) * 0.2
        # Penalize amplification
        if b_valence > 200:
            effectiveness *= 0.3  # don't feed the mania
        # Bonus for gentle/grounding
        if b_arousal < 120:
            effectiveness = min(1.0, effectiveness + 0.15)

    elif av > 150:  # POSITIVE (healthy range)
        # Positive + good response = amplify naturally
        # Positive + celebratory = great match
        match = 1.0 - abs(b_valence - av) / 255.0
        enthusiasm = min(1.0, max(0, b_arousal - 80) / 175.0)
        effectiveness = match * 0.4 + warmth * 0.3 + enthusiasm * 0.3

    else:  # NEUTRAL (100-150)
        # Neutral — match energy, gentle engagement
        effectiveness = warmth * 0.4 + calmness * 0.3 + balance * 0.3

    # Calculate V outcome
    if av < 128:
        # Negative state: response pulls toward neutral
        dv = int(max_dv * effectiveness * recovery_rate)
        if effectiveness < 0.2:
            # Bad response — makes it worse
            dv = -int(15 * (1.0 - effectiveness))
    else:
        # Positive state: response can amplify or deflate
        if effectiveness > 0.5:
            dv = int(20 * effectiveness)
        else:
            dv = -int(30 * (1.0 - effectiveness))

    # Arousal outcome — good responses calm, bad ones agitate
    if effectiveness > 0.5:
        da = -int(20 * effectiveness)  # calming
    else:
        da = int(15 * (1.0 - effectiveness))  # agitating

    # Dominance — good responses give agency back
    dd = int(15 * effectiveness) if av < 128 else 0

    # Urgency — good responses reduce urgency
    du = -int(au * effectiveness * 0.3)

    # Gravity — follows valence
    dg = int(dv * 0.7)

    # Apply dark matter if entity provided
    cv = max(0, min(255, av + dv))
    ca = max(0, min(255, aa + da))
    cd = max(0, min(255, ad + dd))
    cu = max(0, min(255, au + du))
    cg = max(0, min(255, ag + dg))

    if entity:
        cv, ca, cd, cu, cg = entity.apply(cv, ca, cd, cu, cg)

    outcome = VADUG(v=cv, a=ca, d=cd, u=cu, g=cg)

    details = {
        "b_intent": (b_valence, b_arousal, b_dominance),
        "effectiveness": round(effectiveness, 3),
        "recovery_rate": recovery_rate,
        "dv": dv, "da": da, "dd": dd, "du": du, "dg": dg,
    }

    return outcome, details


def distance_to_target(vadug, target):
    """Weighted distance from current VADUG to target (window of tolerance center).

    V and G are weighted highest (emotional quality matters most).
    A is moderately weighted (calm is important but not everything).
    D and U are lower weighted.
    """
    weights = {"v": 3.0, "a": 1.5, "d": 1.0, "u": 1.5, "g": 2.0}
    d = 0
    d += weights["v"] * (vadug.v - target[0]) ** 2
    d += weights["a"] * (vadug.a - target[1]) ** 2
    d += weights["d"] * (vadug.d - target[2]) ** 2
    d += weights["u"] * (vadug.u - target[3]) ** 2
    d += weights["g"] * (vadug.g - target[4]) ** 2
    return math.sqrt(d)


def optimize_response(user_text, entity_name="default", target=None):
    """Find the best response to user_text.

    Args:
        user_text: what the user said
        entity_name: dark matter profile ("default", "traumatized", etc)
        target: target VADUG tuple (V,A,D,U,G) — defaults to healthy neutral

    Returns list of (response_text, outcome_vadug, distance, details) sorted best-first.
    """
    engine = SequentialPendulum()
    entity = new_entity(entity_name)

    # Score user input
    a_vadug_obj, a_trace = engine.process_text(user_text)
    a_vadug = (a_vadug_obj.v, a_vadug_obj.a, a_vadug_obj.d, a_vadug_obj.u, a_vadug_obj.g)

    # The optimizer PREDICTS outcomes, it doesn't prescribe them.
    # There is no "target" — it scores each response by what it
    # ACTUALLY PRODUCES emotionally, then ranks by accuracy.
    #
    # If the caller provides a target, we rank by distance to that target.
    # If no target, we just return all outcomes ranked by effectiveness
    # (how much emotional movement the response produces).
    # The APPLICATION decides what to do with the predictions.
    if target is None:
        target = None  # No target = rank by effectiveness, not distance

    # Score all candidate responses
    results = []
    for strategy, responses in RESPONSE_BANK.items():
        for b_text in responses:
            outcome, details = score_outcome(a_vadug, b_text, engine, entity)
            dist = distance_to_target(outcome, target) if target else 0.0
            results.append({
                "response": b_text,
                "strategy": strategy,
                "outcome": outcome,
                "distance": round(dist, 1),
                "effectiveness": details["effectiveness"],
                "v_delta": outcome.v - a_vadug[0],  # how much V moved
                "details": details,
            })

    if target:
        # Rank by distance to target (caller specified a goal)
        results.sort(key=lambda x: x["distance"])
    else:
        # No target — rank by effectiveness (most emotional movement)
        # The APPLICATION decides what to do with these predictions
        results.sort(key=lambda x: -x["effectiveness"])
    return a_vadug, results


def band_label(v):
    """Get the prism band for a V value."""
    if v <= 50: return "Band 1 (CRISIS)"
    if v <= 102: return "Band 2 (NEGATIVE)"
    if v <= 153: return "Band 3 (NEUTRAL)"
    if v <= 204: return "Band 4 (POSITIVE)"
    return "Band 5 (THRIVING)"


def run_interactive():
    """Interactive mode — type a sentence, see all possible outcomes."""
    print("=" * 65)
    print("  CLANKER OUTCOME OPTIMIZER — Doctor Strange Mode")
    print("  Type what someone said. See every possible future.")
    print("=" * 65)
    print()

    entities = ["default", "optimist", "pessimist", "traumatized",
                "resilient", "volatile", "stoic"]

    while True:
        try:
            user_input = input("\n💬 User says: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Pick entity
        print(f"\n  Entity types: {', '.join(entities)}")
        try:
            entity_name = input("  Entity [default]: ").strip() or "default"
        except (EOFError, KeyboardInterrupt):
            break

        if entity_name not in entities:
            entity_name = "default"

        a_vadug, results = optimize_response(user_input, entity_name)
        av, aa, ad, au, ag = a_vadug

        print(f"\n  Input state: V={av} A={aa} D={ad} U={au} G={ag}  [{band_label(av)}]")
        print(f"  Entity: {entity_name}")
        print(f"  Target: V=128 A=100 D=128 U=0 G=128 (healthy neutral)")
        print()

        # Show top 5 best and bottom 3 worst
        print("  ┌─ TOP 5 BEST RESPONSES ─────────────────────────────────────")
        for i, r in enumerate(results[:5]):
            o = r["outcome"]
            dv = o.v - av
            arrow = "↑" if dv > 0 else "↓" if dv < 0 else "→"
            print(f"  │ #{i+1} [{r['strategy']:12s}] \"{r['response'][:50]}\"")
            print(f"  │    → V={o.v} A={o.a} D={o.d} U={o.u} G={o.g}  "
                  f"[{band_label(o.v)}]  V{arrow}{abs(dv)}  "
                  f"eff={r['effectiveness']:.2f}  dist={r['distance']}")
        print("  └────────────────────────────────────────────────────────────")

        print()
        print("  ┌─ BOTTOM 3 WORST RESPONSES ─────────────────────────────────")
        for i, r in enumerate(results[-3:]):
            o = r["outcome"]
            dv = o.v - av
            arrow = "↑" if dv > 0 else "↓" if dv < 0 else "→"
            print(f"  │ [{r['strategy']:12s}] \"{r['response'][:50]}\"")
            print(f"  │    → V={o.v} A={o.a} D={o.d} U={o.u} G={o.g}  "
                  f"[{band_label(o.v)}]  V{arrow}{abs(dv)}  "
                  f"eff={r['effectiveness']:.2f}  dist={r['distance']}")
        print("  └────────────────────────────────────────────────────────────")

        # Show band distribution
        band_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in results:
            v = r["outcome"].v
            if v <= 50: band_counts[1] += 1
            elif v <= 102: band_counts[2] += 1
            elif v <= 153: band_counts[3] += 1
            elif v <= 204: band_counts[4] += 1
            else: band_counts[5] += 1

        print(f"\n  Outcome distribution across {len(results)} responses:")
        for b in range(1, 6):
            bar = "█" * band_counts[b]
            print(f"    Band {b}: {bar} ({band_counts[b]})")


if __name__ == "__main__":
    run_interactive()
