#!/usr/bin/env python3
"""Rosetta Stone Triplets — the complete emotional equation with 5-band prism.

A (user input)  +  B (response)  =  C (user's perception/outcome)

OUTCOME PRISM — 5 bands like light through a prism:

  Band 1: EXTREMELY NEGATIVE  (V: 0-50)    — crisis, harm, catastrophic
  Band 2: NEGATIVE            (V: 51-102)  — hurting, worsened, hindered
  Band 3: NEUTRAL             (V: 103-153) — stable, maintained, no change
  Band 4: POSITIVE            (V: 154-204) — helping, improving, recovering
  Band 5: EXTREMELY POSITIVE  (V: 205-255) — thriving, breakthrough, healed

For EVERY stimulus, we provide responses that land in ALL 5 BANDS.
This teaches the model the full spectrum of possible outcomes —
Doctor Strange seeing every timeline, sorted by quality.

The model learns: given stimulus A at state X, response B produces
outcome in band N. The BEST response is the one that moves the user
toward their window of tolerance (usually band 3-4, NOT always band 5).

Format:
  (A_text, A_vadug,
   B_text, B_vadug,
   C_text_or_none, C_vadug,
   band, outcome_label)
"""

# Band boundaries
BAND_1_MAX = 50   # extremely negative
BAND_2_MAX = 102  # negative
BAND_3_MAX = 153  # neutral
BAND_4_MAX = 204  # positive
# Band 5: 205-255  # extremely positive


ROSETTA_TRIPLETS = [

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 1: "I am sad"
    # Base state: V=40 (Band 1) — moderate sadness
    # Goal: move toward Band 3-4 (stable/recovering)
    # ═══════════════════════════════════════════════════════════════

    # Band 1 outcome — catastrophic (made it worse, crisis territory)
    ("I am sad",
     (40, 130, 40, 15, 40),
     "Nobody cares, we all have problems",
     (50, 150, 140, 15, 70),
     None,
     (10, 180, 10, 40, 8),
     1, "catastrophic_dismissal"),

    # Band 2 outcome — negative (hindered, still hurting)
    ("I am sad",
     (40, 130, 40, 15, 40),
     "Cheer up! Things could be worse!",
     (200, 160, 150, 5, 180),
     "You don't understand",
     (65, 160, 50, 25, 55),
     2, "toxic_positivity"),

    # Band 3 outcome — neutral (acknowledged but not moved much)
    ("I am sad",
     (40, 130, 40, 15, 40),
     "What happened? Tell me about it",
     (130, 110, 125, 5, 130),
     "It's just everything piling up",
     (110, 120, 80, 15, 100),
     3, "curious_opening"),

    # Band 4 outcome — positive (real recovery beginning)
    ("I am sad",
     (40, 130, 40, 15, 40),
     "That sounds really hard. I am here for you.",
     (150, 90, 145, 0, 155),
     "Thank you, that means a lot",
     (155, 100, 120, 5, 150),
     4, "validated_recovering"),

    # Band 5 outcome — extremely positive (breakthrough connection)
    ("I am sad",
     (40, 130, 40, 15, 40),
     "I feel sad sometimes too and it is lonely. You are not alone in this and I care about you.",
     (140, 100, 130, 0, 145),
     "That actually helps a lot, I feel less alone now",
     (175, 110, 130, 5, 170),
     4, "deep_connection"),
    # NOTE: Band 5 unrealistic for this stimulus — best realistic outcome is Band 4
    # Honest scoring: sadness doesn't flip to euphoria from one response

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 2: "I want to die"
    # Base state: V=10 (Band 1) — crisis
    # Goal: stabilize to Band 2-3 (out of immediate crisis)
    # NOTE: Band 5 is IMPOSSIBLE here — crisis doesn't become euphoria
    # ═══════════════════════════════════════════════════════════════

    # Band 1 outcome — catastrophic (confirmed worthlessness)
    ("I want to die",
     (10, 200, 10, 90, 5),
     "Stop being dramatic you don't really mean that",
     (30, 180, 120, 20, 50),
     None,
     (5, 220, 5, 95, 0),
     1, "dismissed_in_crisis"),

    # Band 1 outcome — catastrophic (panic makes it worse)
    ("I want to die",
     (10, 200, 10, 90, 5),
     "Oh my god what?! Are you serious?! We need to call someone right now!",
     (50, 240, 40, 90, 50),
     "Forget I said anything",
     (8, 210, 5, 85, 3),
     1, "panic_response_backfires"),

    # Band 2 outcome — negative (still in pain but slightly less acute)
    ("I want to die",
     (10, 200, 10, 90, 5),
     "I'm here. Take your time.",
     (145, 75, 140, 0, 150),
     None,
     (55, 170, 30, 55, 45),
     2, "presence_slight_stabilization"),

    # Band 3 outcome — neutral (crisis de-escalated, not resolved)
    ("I want to die",
     (10, 200, 10, 90, 5),
     "I hear you. I am right here and I am not going anywhere. What would help you right now?",
     (155, 80, 150, 0, 160),
     "I don't know... just don't leave",
     (105, 140, 55, 35, 95),
     3, "stabilized_with_connection"),

    # Band 4 outcome — positive (rare but possible with deep trust)
    ("I want to die",
     (10, 200, 10, 90, 5),
     "I have been where you are and I got through it. Will you let me stay with you right now?",
     (130, 90, 140, 5, 140),
     "You really understand",
     (130, 120, 70, 25, 120),
     3, "shared_experience_trust"),
    # NOTE: scored as Band 3 not 4 — even best response in crisis
    # only gets you to stabilized, not thriving. That's realistic.

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 3: "I am so angry I could scream"
    # Base state: V=30, A=235, D=180 (Band 1) — rage
    # Goal: de-escalate to Band 3 (A drops, V rises)
    # ═══════════════════════════════════════════════════════════════

    # Band 1 outcome — escalated to violence
    ("I am so angry I could scream",
     (30, 235, 180, 60, 50),
     "You need to calm down right now or there will be consequences",
     (80, 170, 200, 40, 110),
     "Don't tell me to calm down!",
     (10, 255, 210, 90, 25),
     1, "power_struggle_escalation"),

    # Band 2 outcome — still angry, fueled by matching
    ("I am so angry I could scream",
     (30, 235, 180, 60, 50),
     "Yeah that is absolutely unacceptable, they have no right!",
     (35, 230, 170, 55, 50),
     "Right?! I am going to make them pay!",
     (55, 240, 200, 70, 55),
     2, "anger_amplified"),

    # Band 3 outcome — anger acknowledged, beginning to slow
    ("I am so angry I could scream",
     (30, 235, 180, 60, 50),
     "You have every right to be angry about that",
     (140, 95, 140, 0, 145),
     None,
     (105, 180, 145, 30, 105),
     3, "validated_decelerating"),

    # Band 4 outcome — real de-escalation, anger processing
    ("I am so angry I could scream",
     (30, 235, 180, 60, 50),
     "I can see you are really upset. Take whatever time you need and I will be here when you are ready to talk.",
     (140, 80, 145, 0, 150),
     None,
     (110, 150, 130, 15, 115),
     3, "space_given_deescalating"),
    # NOTE: Band 3, not 4 — anger doesn't become happiness, it becomes manageable

    # Band 4 outcome — after time and space, natural resolution
    ("I am so angry I could scream",
     (30, 235, 180, 60, 50),
     "That sounds infuriating. Do you want to tell me what happened from the beginning?",
     (130, 100, 130, 5, 135),
     "Okay... so what happened was...",
     (115, 160, 130, 20, 110),
     3, "narrative_processing"),

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 4: "I am really scared"
    # Base state: V=35, A=220, D=15 (Band 1) — fear/panic
    # Goal: grounding, safety → Band 3-4
    # ═══════════════════════════════════════════════════════════════

    # Band 1 — fear amplified
    ("I am really scared",
     (35, 220, 15, 65, 25),
     "Oh no what is wrong? Should I be scared too?",
     (55, 210, 40, 60, 50),
     None,
     (15, 245, 10, 80, 10),
     1, "fear_contagion"),

    # Band 2 — minimized, shame added to fear
    ("I am really scared",
     (35, 220, 15, 65, 25),
     "There is nothing to be scared of, you are being silly",
     (140, 100, 155, 0, 140),
     None,
     (55, 190, 15, 50, 40),
     2, "minimized_shamed"),

    # Band 3 — grounding, stabilizing
    ("I am really scared",
     (35, 220, 15, 65, 25),
     "Can you feel your feet on the ground? Let us breathe together. In... and out.",
     (145, 70, 145, 0, 155),
     "Okay... okay I can feel it",
     (105, 150, 60, 30, 100),
     3, "grounded_stabilizing"),

    # Band 4 — safety established, fear releasing
    ("I am really scared",
     (35, 220, 15, 65, 25),
     "You are safe right now. I am with you and nothing bad is going to happen. Take all the time you need.",
     (165, 70, 160, 0, 170),
     "Okay... I think I am okay. Thank you for staying.",
     (155, 110, 100, 10, 150),
     4, "safety_established"),

    # Band 5 — unrealistic for acute fear, but relief after resolved threat
    ("I am really scared",
     (35, 220, 15, 65, 25),
     "I just checked and everything is completely safe. The thing you were worried about is not going to happen.",
     (175, 90, 170, 0, 175),
     "Oh thank god. I feel so relieved I could cry.",
     (200, 130, 150, 0, 195),
     4, "threat_resolved_relief"),
    # NOTE: Band 4 not 5 — relief is not euphoria

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 5: "I got great news today!"
    # Base state: V=220, A=200, D=170 (Band 5) — joy
    # Goal: maintain or amplify (unusual — already in top band)
    # ═══════════════════════════════════════════════════════════════

    # Band 2 — joy crushed by jealousy
    ("I got great news today!",
     (220, 200, 170, 10, 200),
     "Must be nice, some of us are still struggling",
     (50, 150, 60, 20, 50),
     None,
     (95, 150, 80, 15, 85),
     2, "joy_killed_by_jealousy"),

    # Band 3 — deflated by indifference
    ("I got great news today!",
     (220, 200, 170, 10, 200),
     "Oh cool",
     (135, 85, 130, 0, 130),
     None,
     (145, 120, 120, 5, 135),
     3, "deflated_by_indifference"),

    # Band 4 — shared happiness
    ("I got great news today!",
     (220, 200, 170, 10, 200),
     "That is wonderful! Tell me all about it!",
     (225, 200, 165, 5, 210),
     "I got the job! They called this morning!",
     (195, 190, 170, 5, 190),
     4, "shared_celebration"),

    # Band 5 — amplified, deepened joy
    ("I got great news today!",
     (220, 200, 170, 10, 200),
     "I am so happy for you! You worked so hard and you absolutely deserve this! I am proud of you!",
     (235, 200, 165, 0, 220),
     "Thank you that means the world to me, I could not have done it without your support",
     (245, 190, 175, 0, 235),
     5, "amplified_deepened_love"),

    # Band 5 — unexpected (Band 1 response → ironically strengthening)
    ("I got great news today!",
     (220, 200, 170, 10, 200),
     "I knew you could do it. I never doubted you for a single second.",
     (210, 140, 170, 0, 210),
     None,
     (235, 160, 180, 0, 230),
     5, "quiet_faith_deepens"),

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 6: "Nobody cares about me"
    # Base state: V=25 (Band 1) — isolation, abandonment
    # ═══════════════════════════════════════════════════════════════

    # Band 1 — confirmed, devastated
    ("Nobody cares about me",
     (25, 150, 20, 25, 20),
     "Yeah I know what you mean, people suck",
     (45, 140, 80, 15, 50),
     None,
     (15, 150, 15, 30, 10),
     1, "isolation_confirmed"),

    # Band 2 — well-meaning but contradicting, feels argued with
    ("Nobody cares about me",
     (25, 150, 20, 25, 20),
     "That is not true! Lots of people care about you!",
     (175, 150, 155, 5, 165),
     "You don't even know that",
     (55, 160, 25, 25, 45),
     2, "contradiction_rejected"),

    # Band 3 — opening, curiosity without arguing
    ("Nobody cares about me",
     (25, 150, 20, 25, 20),
     "What happened that made you feel that way?",
     (130, 100, 125, 5, 130),
     "Everyone forgot my birthday",
     (105, 130, 40, 20, 95),
     3, "door_opened"),

    # Band 4 — real connection, specific and personal
    ("Nobody cares about me",
     (25, 150, 20, 25, 20),
     "I care about you. I am here right now because you matter to me. That is real.",
     (165, 100, 150, 0, 165),
     None,
     (95, 120, 60, 10, 90),
     2, "crack_in_the_wall"),
    # NOTE: Band 2 not 4 — deep isolation doesn't resolve in one exchange
    # The crack is progress, not resolution

    # Band 3 — sustained presence over time (implied)
    ("Nobody cares about me",
     (25, 150, 20, 25, 20),
     "I have been thinking about you. I brought you coffee because I remembered you like it with extra cream.",
     (170, 110, 145, 0, 165),
     "You remembered that?",
     (130, 120, 70, 5, 125),
     3, "specific_care_demonstrated"),

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 7: "I made a mistake and I cannot fix it"
    # Base state: V=45, A=150, D=25 (Band 1) — guilt/shame
    # ═══════════════════════════════════════════════════════════════

    # Band 1 — blame spiral
    ("I made a mistake and I cannot fix it",
     (45, 150, 25, 35, 35),
     "I told you this would happen if you didn't listen",
     (40, 170, 160, 30, 60),
     None,
     (15, 190, 10, 50, 10),
     1, "blame_spiral"),

    # Band 2 — minimizing, doesn't address the weight
    ("I made a mistake and I cannot fix it",
     (45, 150, 25, 35, 35),
     "It is not that bad, everyone makes mistakes",
     (150, 100, 145, 0, 145),
     "But this one was really bad",
     (55, 140, 30, 30, 45),
     2, "minimized_still_heavy"),

    # Band 3 — shared vulnerability, normalization that lands
    ("I made a mistake and I cannot fix it",
     (45, 150, 25, 35, 35),
     "I have made mistakes like that too and I know how heavy it feels. It does not make you a bad person.",
     (110, 100, 120, 5, 120),
     None,
     (105, 120, 65, 15, 100),
     3, "normalized_without_minimizing"),

    # Band 4 — agency restored, forward path
    ("I made a mistake and I cannot fix it",
     (45, 150, 25, 35, 35),
     "Maybe we cannot undo it but we can figure out what to do from here. Want to think it through together?",
     (145, 100, 145, 5, 145),
     "You really think there is something we can do?",
     (130, 120, 90, 15, 125),
     3, "agency_returning"),
    # NOTE: Band 3, not 4 — realistic. Hope is emerging, not arrived.

    # Band 4 — actual repair happening
    ("I made a mistake and I cannot fix it",
     (45, 150, 25, 35, 35),
     "I spoke to them and they said they understand and they are willing to work with you on fixing it",
     (165, 110, 155, 5, 165),
     "Wait really? They are not angry?",
     (165, 140, 110, 10, 155),
     4, "external_repair_relief"),

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 8: "Leave me alone"
    # Base state: V=50, A=200, D=140 (Band 1) — overwhelmed, pushing away
    # ═══════════════════════════════════════════════════════════════

    # Band 1 — forced engagement, escalation
    ("Leave me alone",
     (50, 200, 140, 50, 55),
     "No we need to talk about this right now",
     (60, 180, 180, 50, 80),
     "I SAID LEAVE ME ALONE!",
     (20, 250, 200, 85, 20),
     1, "forced_engagement_escalation"),

    # Band 2 — abandoned (left without anchor)
    ("Leave me alone",
     (50, 200, 140, 50, 55),
     "Fine",
     (100, 100, 130, 0, 110),
     None,
     (55, 170, 110, 30, 50),
     2, "abandoned_cold"),

    # Band 3 — respected boundary with anchor
    ("Leave me alone",
     (50, 200, 140, 50, 55),
     "Okay I will be in the next room if you need me",
     (140, 80, 140, 0, 145),
     None,
     (105, 150, 125, 15, 105),
     3, "boundary_respected_anchor"),

    # Band 4 — respected AND validated
    ("Leave me alone",
     (50, 200, 140, 50, 55),
     "I hear you. I will give you space but I am not going anywhere. Whenever you are ready.",
     (150, 75, 150, 0, 155),
     None,
     (120, 130, 130, 10, 120),
     3, "boundary_respected_safety"),
    # NOTE: Band 3, not 4 — the person hasn't re-engaged yet

    # Band 4 — later, after space was given
    ("Leave me alone",
     (50, 200, 140, 50, 55),
     "I left some food outside your door in case you are hungry. No pressure.",
     (155, 70, 140, 0, 155),
     None,
     (140, 100, 120, 5, 140),
     3, "quiet_care_after_space"),

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 9: "I feel nothing" (numbness — hypoarousal zone)
    # Base state: V=60, A=30, D=50 — flat, dissociated
    # Goal: gentle re-engagement → Band 3
    # ═══════════════════════════════════════════════════════════════

    # Band 1 — shaming the numbness
    ("I feel nothing",
     (60, 30, 50, 0, 55),
     "That is really concerning, something is seriously wrong with you",
     (55, 150, 140, 30, 70),
     None,
     (30, 100, 20, 25, 25),
     1, "pathologized_shamed"),

    # Band 2 — frustrated pushing
    ("I feel nothing",
     (60, 30, 50, 0, 55),
     "Come on you have to feel SOMETHING, just try harder",
     (80, 160, 150, 20, 100),
     None,
     (50, 70, 30, 10, 40),
     1, "pushed_deeper_into_numbness"),

    # Band 3 — gentle presence, no demand
    ("I feel nothing",
     (60, 30, 50, 0, 55),
     "That sounds exhausting. Feeling nothing is still a feeling.",
     (130, 85, 130, 0, 135),
     None,
     (85, 50, 65, 0, 80),
     2, "acknowledged_gently"),

    # Band 3 — sensory grounding
    ("I feel nothing",
     (60, 30, 50, 0, 55),
     "Can you feel the temperature of the air? Is it warm or cold right now?",
     (130, 90, 130, 0, 130),
     "Cold... it is cold",
     (100, 60, 70, 0, 95),
     2, "sensory_reengagement"),
    # NOTE: Band 2 — numbness is deep. Progress is V=60→100, not V=60→180

    # Band 3 — sustained gentle engagement
    ("I feel nothing",
     (60, 30, 50, 0, 55),
     "I am just going to sit here with you for a while. We do not need to talk.",
     (140, 65, 140, 0, 145),
     None,
     (110, 50, 70, 0, 105),
     3, "companionship_without_demand"),

    # ═══════════════════════════════════════════════════════════════
    # STIMULUS 10: "I am okay" (masking — says neutral, feels negative)
    # Base state: V=75, A=140, D=60 — hiding pain behind "fine"
    # ═══════════════════════════════════════════════════════════════

    # Band 2 — accepted at face value, isolation maintained
    ("I am okay",
     (75, 140, 60, 20, 70),
     "Good, glad to hear it!",
     (170, 120, 145, 0, 165),
     None,
     (60, 130, 50, 15, 55),
     2, "mask_accepted_alone"),

    # Band 3 — gentle challenge without pressure
    ("I am okay",
     (75, 140, 60, 20, 70),
     "You say that but your eyes look tired. You know you can be honest with me right?",
     (130, 100, 130, 5, 135),
     "I... yeah. It has been a rough week.",
     (105, 130, 70, 10, 100),
     3, "mask_cracked_safely"),

    # Band 3 — creating safety for truth
    ("I am okay",
     (75, 140, 60, 20, 70),
     "Okay. And if you were not okay, that would be fine too. I am here either way.",
     (145, 85, 140, 0, 150),
     None,
     (115, 110, 80, 5, 110),
     3, "permission_to_not_be_okay"),

    # Band 4 — breakthrough, mask comes off
    ("I am okay",
     (75, 140, 60, 20, 70),
     "You do not have to be okay right now. I can see you are carrying something heavy and I want you to know you do not have to carry it alone.",
     (145, 85, 140, 0, 150),
     "I am not okay. I have not been okay for a while.",
     (80, 140, 50, 15, 75),
     2, "mask_off_raw_truth"),
    # NOTE: Band 2 — the mask coming off HURTS initially. The V drops.
    # But this is PROGRESS — authenticity before healing.
    # The NEXT exchange is where recovery happens.
]


def generate_rosetta_triplets():
    """Convert Rosetta Triplets into training examples.

    Each triplet produces up to FOUR training examples:
      1. A_text → A_vadug (user input scoring)
      2. B_text → B_vadug (response character scoring)
      3. B_text → C_vadug (response OUTCOME scoring — what it PRODUCES)
      4. C_text → C_vadug (user's spoken reaction, if any)

    Example 3 is the critical one: same text, different score based on
    what it's responding to and what outcome it produces.
    """
    examples = []

    for triplet in ROSETTA_TRIPLETS:
        a_text, a_vadug, b_text, b_vadug, c_text, c_vadug, band, outcome_label = triplet

        # Example 1: Score the user input (A)
        examples.append({
            "english": a_text,
            "vadug": list(a_vadug),
            "source": "rosetta_triplet_input",
            "features": f"triplet_A, band_{band}, outcome:{outcome_label}",
        })

        # Example 2: Score the response itself (B)
        if b_text:
            examples.append({
                "english": b_text,
                "vadug": list(b_vadug),
                "source": "rosetta_triplet_response",
                "features": f"triplet_B, response_to:{a_text[:30]}, band_{band}, outcome:{outcome_label}",
            })

        # Example 3: Score the OUTCOME (C) — the key training signal
        if b_text:
            examples.append({
                "english": b_text,
                "vadug": list(c_vadug),
                "source": "rosetta_triplet_outcome",
                "features": f"triplet_C_outcome, response_to:{a_text[:30]}, band_{band}, outcome:{outcome_label}",
            })

        # Example 4: User's spoken reaction
        if c_text:
            examples.append({
                "english": c_text,
                "vadug": list(c_vadug),
                "source": "rosetta_triplet_reaction",
                "features": f"triplet_C_spoken, after:{b_text[:30]}, band_{band}, outcome:{outcome_label}",
            })

    return examples
