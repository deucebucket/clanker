"""
Clanker Layer 7 Template Decoder — VADUG → Natural Language

Decodes VADUG coordinates into natural language using cross-axis blended
template selection. Each template slot considers MULTIPLE axes, not just one.

Based on initial template approach by Gemini, extended with cross-axis blending
to capture interactions between emotional dimensions (e.g., V65+G62 = "exhausting"
but V65+G180 = "frustrating" — same valence, different gravity, different word).

This is the "hat" on top of the Clanker model — it doesn't decide WHAT to say
(Layers 1-6 handle that), it decides HOW to say it in human language.
"""

import random


def decode_vadug_to_text(v: int, a: int, d: int, u: int, g: int,
                         goal: int = 0x00) -> str:
    """Decode a VADUG coordinate + goal into natural language response.

    Uses the {acknowledge} + {stabilize} + {redirect} template structure
    with cross-axis blending for word selection.
    """

    # ─── ACKNOWLEDGE ─── (V × G interaction)
    # Valence sets the polarity, Gravity sets the weight/tone
    if v < 60:
        if g < 60:
            # very negative + sinking = heavy, weighed down
            ack = random.choice([
                "That sounds really heavy.",
                "I can feel the weight of that.",
                "That's crushing. I hear you.",
            ])
        elif g > 170:
            # very negative + rising = angry, volatile
            ack = random.choice([
                "I can feel how fired up you are.",
                "That's clearly hit a nerve.",
                "I hear the frustration boiling over.",
            ])
        else:
            # very negative + grounded
            ack = random.choice([
                "That sounds really rough.",
                "I hear you. That's not easy.",
                "That's a lot to carry.",
            ])
    elif v < 90:
        if g < 70:
            # negative + sinking = tired, drained, low
            ack = random.choice([
                "That sounds exhausting.",
                "I can tell this is draining you.",
                "That's wearing on you.",
            ])
        elif g > 160:
            # negative + floating = anxious, ungrounded
            ack = random.choice([
                "I can tell you're on edge.",
                "That uncertainty is tough.",
                "I hear the worry in that.",
            ])
        else:
            # negative + grounded
            ack = random.choice([
                "That's frustrating.",
                "I understand that's disappointing.",
                "That's not what you were hoping for.",
            ])
    elif v < 140:
        # neutral zone
        if a > 170:
            ack = random.choice(["I can see this matters to you.", "Got it — this is important."])
        else:
            ack = random.choice(["I see.", "Got it.", "Understood."])
    elif v < 200:
        if g > 170:
            # positive + floating = elated, lifted
            ack = random.choice([
                "That's wonderful!",
                "Love to hear that!",
                "That's great news!",
            ])
        else:
            # positive + grounded = satisfied, content
            ack = random.choice([
                "Nice!",
                "Good to hear.",
                "That's solid.",
            ])
    else:
        # very positive
        if g > 200:
            # soaring
            ack = random.choice([
                "That's incredible!",
                "YES! That's amazing!",
                "I'm thrilled for you!",
            ])
        else:
            ack = random.choice([
                "That's awesome!",
                "Fantastic!",
                "Hell yeah!",
            ])

    # ─── STABILIZE ─── (D × A interaction)
    # Dominance sets the stability message, Arousal affects delivery intensity
    if d < 60:
        if a > 170:
            # helpless + intense = panicking, needs grounding
            stab = random.choice([
                "Take a breath. I'm right here.",
                "Let's slow down for a second. I've got you.",
                "One thing at a time. We'll get through this.",
            ])
        else:
            # helpless + calm = defeated, resigned
            stab = random.choice([
                "I'm right here with you.",
                "You don't have to figure this out alone.",
                "I've got your back on this.",
            ])
    elif d < 120:
        # moderate control
        stab = random.choice([
            "Let's work through this together.",
            "We can figure this out.",
            "Let's look at this step by step.",
        ])
    else:
        if a > 170:
            # in control + intense = assertive, energetic
            stab = random.choice([
                "Let's channel that energy.",
                "You've got this. Let's go.",
                "I'm with you. Full speed.",
            ])
        else:
            # in control + calm = confident, steady
            stab = random.choice([
                "Looking good.",
                "You're in a solid position.",
                "Steady as she goes.",
            ])

    # ─── REDIRECT ─── (G × U interaction)
    # Gravity sets the lift/ground direction, Urgency affects pace
    if g < 60:
        if u > 150:
            # sinking + urgent = need immediate lift
            stab_redirect = random.choice([
                "What's the most critical thing right now?",
                "Let's tackle the biggest blocker first.",
                "What needs to happen RIGHT NOW?",
            ])
        else:
            # sinking + not urgent = gentle exploration
            stab_redirect = random.choice([
                "What specifically is weighing on you?",
                "Let's look at this fresh.",
                "Show me where it's stuck.",
            ])
    elif g < 120:
        # grounded
        stab_redirect = random.choice([
            "What would you like to do next?",
            "How can I help from here?",
            "Where do you want to start?",
        ])
    else:
        if u > 150:
            # floating + urgent = excited rush
            stab_redirect = random.choice([
                "What's the next move?",
                "Let's ride this wave. What's first?",
                "Strike while the iron's hot — what do you need?",
            ])
        else:
            # floating + not urgent = positive momentum
            stab_redirect = random.choice([
                "Let's keep the momentum going.",
                "What's next on the list?",
                "Ready when you are.",
            ])

    # ─── GOAL OVERRIDE ─── (explicit intent overrides template when appropriate)
    GOAL_OVERRIDES = {
        0x01: "Could you tell me a bit more?",           # CLARIFY
        0x03: "Let me explain.",                          # TEACH
        0x04: "On it.",                                   # EXECUTE
        0x05: "I can't do that, but here's why.",         # REFUSE
        0x06: "I'm here. Take your time.",                # EMPATHIZE (override redirect)
    }

    if goal in GOAL_OVERRIDES:
        if goal == 0x06:
            # EMPATHIZE: keep acknowledge + stabilize, replace redirect
            stab_redirect = GOAL_OVERRIDES[goal]
        elif goal == 0x04:
            # EXECUTE: minimal acknowledge, skip stabilize
            return f"{ack} {GOAL_OVERRIDES[goal]}"
        elif goal == 0x01:
            # CLARIFY: keep acknowledge, replace redirect
            stab_redirect = GOAL_OVERRIDES[goal]

    # ─── URGENCY PREFIX ─── (U > 200 = critical, prepend urgency marker)
    urgency_prefix = ""
    if u > 200:
        urgency_prefix = random.choice([
            "I'm on this RIGHT NOW. ",
            "Dropping everything. ",
            "Priority one. ",
        ])
    elif u > 150:
        urgency_prefix = random.choice([
            "I hear the urgency. ",
            "Let's move on this. ",
        ])

    return f"{urgency_prefix}{ack} {stab} {stab_redirect}"


# ─── Test Suite ───
if __name__ == "__main__":
    tests = [
        ("Burnout (V65 A129 D160 U10 G62)",    65, 129, 160, 10,  62,  0x00),
        ("Crisis (V38 A167 D54 U65 G20)",       38, 167,  54, 65,  20,  0x06),
        ("Excited (V200 A220 D160 U20 G230)",  200, 220, 160, 20, 230,  0x00),
        ("Pure task (V130 A130 D130 U10 G128)", 130, 130, 130, 10, 128,  0x04),
        ("Angry rising (V30 A220 D200 U80 G190)", 30, 220, 200, 80, 190, 0x00),
        ("Despair sinking (V20 A80 D20 U40 G15)", 20,  80,  20, 40,  15, 0x06),
        ("Anxious floating (V70 A200 D40 U180 G170)", 70, 200, 40, 180, 170, 0x00),
        ("Content grounded (V180 A60 D150 U0 G128)", 180, 60, 150, 0, 128, 0x00),
        ("Dislike sinking (V60 A120 D130 U10 G80)", 60, 120, 130, 10, 80, 0x00),
        ("Relief lifting (V180 A60 D150 U0 G200)", 180, 60, 150, 0, 200, 0x00),
    ]

    print("CLANKER LAYER 7 DECODER — VADUG → NATURAL LANGUAGE")
    print("=" * 70)
    for label, v, a, d, u, g, goal in tests:
        print(f"\n{label}:")
        # Run 3 times to show template variation
        for i in range(3):
            print(f"  → {decode_vadug_to_text(v, a, d, u, g, goal)}")
    print(f"\n{'=' * 70}")
    print("Cross-axis blending: same V, different G = different words.")
    print("Layers 1-6 decided WHAT to say. Layer 7 just picks the words.")
