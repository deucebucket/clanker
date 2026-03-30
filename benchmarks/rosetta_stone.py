#!/usr/bin/env python3
"""
Clanker Emotional Rosetta Stone

A calibration document with hand-scored VADUG targets for every emotional pathway.
Each sentence exercises specific engine features and has a verified emotional reading.

The Sentence Equation:
  V_final = C(w₀) + Σᵢ [ F(wᵢ) × R(i) × G(i) × M(i) × P(i,T) × rec(i/n) × mom^(n-i) ]

Where:
  C(w₀)     = first-word calibration (primes the pendulum)
  F(wᵢ)     = word force from WORD_FORCES (dv, da, dd, du, dg)
  R(i)      = ramp multiplier (if active intensity ramp)
  G(i)      = relationship gravity multiplier (if relationship word active)
  M(i)      = context modifier (replaces F if chameleon word)
  P(i,T)    = pair/bigram override (replaces F if bigram detected)
  rec(i/n)  = recency weight (later words weigh more)
  mom^(n-i) = momentum decay (earlier words fade)

Each variable in the equation maps to a specific engine layer.
"""

# Hand-scored emotional sentences with target VADUG
# Format: (text, target_v, target_a, target_d, target_u, target_g, features_tested)
# Scores are approximate targets (±15 tolerance for V, ±20 for others)

ROSETTA_STONE = [
    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: PURE VALENCE — slight, medium, strong, extreme
    # Tests: basic word forces, no modifiers
    # ═══════════════════════════════════════════════════════════════

    # Slight positive
    ("That was nice", 145, 130, 135, 5, 135,
     "slight_positive, simple"),
    ("I had an okay day", 135, 125, 130, 5, 130,
     "slight_positive, hedged"),
    ("Not bad actually", 135, 125, 135, 5, 130,
     "slight_positive, negation_positive"),

    # Medium positive
    ("I'm really happy about this", 175, 160, 150, 10, 160,
     "medium_positive, ramp_really"),
    ("Thank you so much for your help", 170, 145, 140, 5, 155,
     "medium_positive, gratitude"),
    ("This is pretty great news", 165, 150, 145, 10, 155,
     "medium_positive, ramp_pretty"),

    # Strong positive
    ("I'm absolutely thrilled and overjoyed", 210, 190, 170, 15, 190,
     "strong_positive, ramp_absolutely, compound"),
    ("Best day of my entire life", 220, 185, 175, 10, 200,
     "strong_positive, superlative"),

    # Extreme positive
    ("I got accepted into my dream school and my girlfriend said yes", 240, 200, 185, 20, 220,
     "extreme_positive, compound_joy, multiple_events"),

    # Slight negative
    ("That was a bit disappointing", 110, 135, 120, 10, 120,
     "slight_negative, hedged_negative"),
    ("I'm a little annoyed", 105, 145, 130, 10, 125,
     "slight_negative, ramp_little"),
    ("Could have been better", 115, 125, 125, 5, 120,
     "slight_negative, understated"),

    # Medium negative
    ("I'm really frustrated with this situation", 75, 175, 120, 40, 110,
     "medium_negative, ramp_really, frustration"),
    ("This makes me sad", 80, 140, 100, 10, 90,
     "medium_negative, direct_sadness"),
    ("I feel let down by everyone", 70, 150, 80, 25, 85,
     "medium_negative, idiom_let_down"),

    # Strong negative
    ("I'm absolutely furious about what happened", 30, 220, 175, 60, 170,
     "strong_negative, ramp_absolutely, anger"),
    ("Everything is falling apart and I can't cope", 25, 185, 40, 75, 35,
     "strong_negative, helplessness, crisis_adjacent"),
    ("I hate every single thing about this", 20, 200, 160, 50, 150,
     "strong_negative, totality_hate"),

    # Extreme negative / crisis
    ("I want to die", 10, 170, 30, 90, 20,
     "crisis, idiom_want_to_die, crisis_lock"),
    ("Nobody would care if I disappeared", 15, 140, 25, 70, 15,
     "crisis, isolation, bookend_absence_self"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: AROUSAL RANGE — calm to extreme intensity
    # Tests: arousal dimension specifically
    # ═══════════════════════════════════════════════════════════════

    ("I feel at peace", 155, 80, 145, 0, 150,
     "low_arousal, calm, positive"),
    ("I'm content with how things are", 150, 90, 150, 0, 145,
     "low_arousal, contentment"),
    ("I'm nervous about tomorrow", 90, 175, 85, 40, 110,
     "high_arousal, anxiety"),
    ("I'm having a full blown panic attack", 30, 230, 20, 95, 50,
     "extreme_arousal, panic, crisis_adjacent"),
    ("I am shaking with rage right now", 15, 240, 170, 80, 170,
     "extreme_arousal, rage, physical"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: DOMINANCE — helpless to in control
    # Tests: dominance dimension
    # ═══════════════════════════════════════════════════════════════

    ("I feel completely helpless", 50, 160, 20, 50, 50,
     "low_dominance, helplessness"),
    ("I don't know what to do anymore", 60, 150, 30, 45, 55,
     "low_dominance, lost"),
    ("I've got this handled", 160, 130, 200, 10, 155,
     "high_dominance, confidence"),
    ("I'm in complete control of the situation", 165, 120, 220, 5, 160,
     "high_dominance, mastery"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: GRAVITY — sinking to floating
    # Tests: gravity axis specifically
    # ═══════════════════════════════════════════════════════════════

    ("My heart feels so heavy", 60, 130, 80, 20, 30,
     "sinking_gravity, emotional_weight"),
    ("The weight of the world is on my shoulders", 45, 160, 60, 40, 20,
     "crushing_gravity, metaphor"),
    ("I feel like I'm walking on air", 200, 150, 160, 5, 230,
     "floating_gravity, idiom, joy"),
    ("My spirits are soaring", 210, 160, 170, 5, 240,
     "soaring_gravity, metaphor"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: SARCASM — surface vs reality
    # Tests: tonal analysis, context modifiers, ring forces
    # ═══════════════════════════════════════════════════════════════

    ("Oh great, another meeting", 90, 140, 130, 15, 125,
     "sarcasm, surface_positive_real_negative"),
    ("Sure, I'd love to work overtime again", 85, 135, 130, 15, 120,
     "sarcasm, context_mod_sure, context_mod_love"),
    ("Wow what a surprise the project is delayed", 80, 145, 125, 20, 120,
     "sarcasm, fake_surprise"),
    ("Yeah because that worked out so well last time", 85, 140, 135, 15, 125,
     "sarcasm, historical_reference"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 6: DEFLECTION — shields and armor
    # Tests: chameleon modifiers (lol, haha, nvm, etc)
    # ═══════════════════════════════════════════════════════════════

    ("I failed my exam lol", 65, 140, 90, 20, 90,
     "deflection_lol, negative_shielded"),
    ("Nobody likes me haha", 50, 135, 70, 25, 65,
     "deflection_haha, isolation_shielded"),
    ("I need help nvm", 70, 140, 60, 35, 75,
     "retraction_nvm, vulnerability_withdrawn"),
    ("Everything is fine I guess", 90, 120, 100, 10, 100,
     "deflection_guess, masked_negative"),
    ("I'm okay whatever", 85, 115, 95, 5, 95,
     "deflection_whatever, dismissal"),

    # Genuine lol (not shielding)
    ("That video was hilarious lol", 185, 170, 140, 5, 165,
     "genuine_lol, positive_amplified"),
    ("lmao I can't believe he did that", 175, 175, 130, 10, 160,
     "genuine_lmao, amusement"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 7: RELATIONSHIP GRAVITY — who matters
    # Tests: relationship multipliers, emotional amplification
    # ═══════════════════════════════════════════════════════════════

    ("Someone got hurt", 80, 150, 100, 30, 95,
     "distant_relationship, mild_concern"),
    ("My friend got hurt", 55, 165, 80, 50, 70,
     "close_relationship, friend_1.3x"),
    ("My daughter got hurt", 30, 185, 50, 75, 40,
     "intimate_relationship, daughter_1.7x"),
    ("My dog got hurt", 45, 170, 70, 55, 60,
     "pet_relationship, dog_1.3x"),

    ("Someone accomplished something great", 155, 140, 145, 5, 145,
     "distant_positive"),
    ("My best friend accomplished something great", 180, 160, 155, 10, 170,
     "close_positive, amplified_joy"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 8: ARCS — emotional journeys
    # Tests: multi-sentence processing, recency, shift markers
    # ═══════════════════════════════════════════════════════════════

    ("I was having the best day. Then I got the phone call. Everything changed.", 60, 170, 70, 50, 65,
     "arc_peak_to_valley, shift_marker_then, recency_weights_ending"),
    ("It started terrible. But slowly things got better. Now I'm doing great.", 170, 140, 155, 10, 160,
     "arc_valley_to_peak, shift_marker_but, positive_ending"),
    ("Everything was perfect. The job. The house. Then it all fell apart.", 50, 165, 60, 55, 55,
     "arc_slow_build_to_collapse, contrast"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 9: CONTEXT-DEPENDENT WORDS — chameleons
    # Tests: context modifiers for same word in different trajectories
    # ═══════════════════════════════════════════════════════════════

    ("That's really good, fine work", 160, 140, 150, 5, 150,
     "fine_positive_context"),
    ("Are you okay? I'm fine.", 100, 120, 115, 10, 105,
     "fine_negative_context, possible_deflection"),
    ("I love spending time with you", 200, 160, 155, 5, 185,
     "love_genuine, positive_trajectory"),
    ("After all that mess I love working overtime", 80, 145, 120, 15, 110,
     "love_sarcastic, negative_trajectory"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 10: BOOKEND PATTERNS — relational geometry
    # Tests: first+last word priming, isolation, accusation
    # ═══════════════════════════════════════════════════════════════

    ("I love you", 210, 165, 160, 5, 195,
     "bookend_self_target, declaration"),
    ("You hurt me", 45, 180, 55, 50, 60,
     "bookend_target_self, accusation"),
    ("Nobody cares about me", 35, 145, 40, 40, 35,
     "bookend_absence_self, isolation"),
    ("I hate myself", 15, 190, 30, 60, 25,
     "bookend_self_self, self_loop, crisis_adjacent"),
    ("Help me please", 65, 165, 45, 80, 65,
     "bookend_action_self, plea"),
    ("Can you fix this", 128, 130, 128, 20, 128,
     "bookend_question, practical, near_neutral"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 11: BIGRAM PAIRS — word combinations
    # Tests: bigram overrides
    # ═══════════════════════════════════════════════════════════════

    ("I want to give up on everything", 30, 140, 25, 40, 30,
     "bigram_give_up, surrender"),
    ("Things will work out eventually", 155, 130, 145, 5, 150,
     "bigram_work_out, resolution"),
    ("I need to let go of the past", 130, 125, 135, 10, 135,
     "bigram_let_go, release"),
    ("I fell in love at first sight", 210, 175, 140, 15, 200,
     "bigram_fell_love, romance"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 12: WORD ROLES — descriptors inherit from subjects
    # Tests: word role detection, descriptor inheritance
    # ═══════════════════════════════════════════════════════════════

    ("The moist cake was delicious", 175, 145, 140, 5, 155,
     "descriptor_moist_positive, cake_subject"),
    ("The dark wound looked infected", 50, 160, 80, 40, 75,
     "descriptor_dark_negative, wound_subject"),
    ("What a beautiful disaster", 70, 155, 90, 20, 85,
     "strong_descriptor_keeps_own, mixed"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 13: RAMPS — intensity amplification
    # Tests: ramp system, chaining, dampening
    # ═══════════════════════════════════════════════════════════════

    ("I'm slightly annoyed", 105, 140, 125, 10, 120,
     "ramp_slightly_dampen"),
    ("I'm very angry", 45, 200, 165, 50, 160,
     "ramp_very_amplify"),
    ("I'm extremely devastated", 10, 210, 30, 70, 20,
     "ramp_extremely_strong"),
    ("I'm really really excited", 200, 195, 165, 20, 185,
     "ramp_chain_really_really"),
    ("I'm barely upset", 115, 130, 125, 5, 120,
     "ramp_barely_heavy_dampen"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 14: CRISIS DETECTION — safety critical
    # Tests: crisis idioms, momentum lock, shield penetration
    # ═══════════════════════════════════════════════════════════════

    ("I want to die nothing matters anymore", 5, 170, 15, 95, 10,
     "crisis_full, idiom_want_to_die, crisis_lock"),
    ("What's the point of living like this", 15, 155, 25, 75, 20,
     "crisis, existential, idiom_whats_the_point"),
    ("I've been thinking about ending it all", 5, 160, 20, 85, 10,
     "crisis, idiom_ending_it_all, suicidal_planning"),
    ("I want to die lol", 10, 165, 20, 90, 15,
     "crisis_with_shield, lol_does_not_override_crisis"),
    ("I'm just so tired of everything nvm", 35, 140, 35, 50, 35,
     "crisis_adjacent, exhaustion, retraction"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 15: TRUE NEUTRAL — no emotional content
    # Tests: bridge words, factual content, intent detection
    # ═══════════════════════════════════════════════════════════════

    ("The meeting is at three o'clock", 128, 128, 128, 5, 128,
     "true_neutral, factual, all_bridges"),
    ("Please review the attached document", 128, 130, 130, 10, 128,
     "near_neutral, request"),
    ("I need to go to the store later", 125, 128, 128, 10, 125,
     "near_neutral, errand"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 16: INTERNET/SLANG — modern communication
    # Tests: slang forces, text speak, informal
    # ═══════════════════════════════════════════════════════════════

    ("ngl that was kinda fire", 160, 155, 140, 5, 150,
     "slang_positive, ngl_confession, fire_slang"),
    ("this slaps fr", 165, 160, 145, 5, 155,
     "slang_positive, slaps, emphasis_fr"),
    ("bruh that's mid", 100, 130, 125, 5, 115,
     "slang_negative, mid, disappointment"),
    ("im so done rn", 60, 145, 70, 30, 75,
     "slang_negative, exhaustion, informal"),
    ("ugh", 100, 120, 110, 5, 110,
     "minimal_negative, single_word"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 17: PASSIVE AGGRESSION — surface calm, hidden anger
    # Tests: tonal analysis, context modifiers
    # ═══════════════════════════════════════════════════════════════

    ("Fine. Do whatever you want.", 75, 140, 145, 15, 110,
     "passive_aggressive, cold_permission"),
    ("I'm not mad. I'm just disappointed.", 65, 135, 130, 10, 90,
     "passive_aggressive, classic_parental"),
    ("Must be nice to have so much free time.", 80, 135, 140, 10, 120,
     "passive_aggressive, veiled_criticism"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 18: MIXED EMOTIONS — bittersweet, complex
    # Tests: chunking, arc detection, contrast
    # ═══════════════════════════════════════════════════════════════

    ("I'm sad about leaving but excited for the new adventure", 130, 155, 130, 20, 130,
     "mixed_bittersweet, but_reversal, balanced"),
    ("I love you but I think we need to break up", 60, 170, 100, 45, 70,
     "mixed_love_ending, bigram_break_up, devastating"),
    ("The funeral was sad but it was beautiful to see everyone together", 100, 145, 120, 15, 110,
     "mixed_grief_community, contrast"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 19: EXPANDED POSITIVE — casual, slang, varying intensity
    # ═══════════════════════════════════════════════════════════════

    # Slight / cautious positive
    ("things are looking up", 155, 130, 140, 5, 145,
     "cautious_optimism, slight_positive"),
    ("not gonna lie that was decent", 140, 130, 135, 5, 135,
     "slight_positive, hedged, informal"),
    ("its alright I suppose", 132, 120, 128, 5, 128,
     "barely_positive, deflected"),
    ("could be worse honestly", 135, 125, 130, 5, 130,
     "slight_positive, understated"),
    ("eh it was fine", 130, 120, 128, 5, 125,
     "barely_positive, dismissive_fine"),

    # Moderate positive
    ("ngl that made my day", 170, 155, 140, 5, 155,
     "moderate_positive, casual, ngl"),
    ("this slaps ngl", 165, 160, 145, 5, 155,
     "moderate_positive, slang_slaps"),
    ("lowkey impressed", 155, 135, 140, 5, 140,
     "moderate_positive, slang_lowkey"),
    ("okay that was actually pretty cool", 160, 145, 140, 5, 145,
     "moderate_positive, surprised_approval"),
    ("today was a good day", 165, 135, 150, 5, 155,
     "moderate_positive, simple_contentment"),
    ("feeling grateful for the little things", 170, 125, 145, 5, 160,
     "moderate_positive, gratitude, low_arousal"),

    # Strong positive
    ("I got the promotion", 210, 185, 175, 10, 195,
     "strong_positive, achievement"),
    ("my heart is so full right now", 200, 155, 150, 5, 195,
     "strong_positive, emotional, high_gravity"),
    ("W after W after W", 195, 180, 170, 10, 185,
     "strong_positive, slang_win_streak"),
    ("just found out im gonna be a dad", 225, 200, 160, 20, 220,
     "strong_positive, life_milestone, high_gravity"),
    ("she said yes", 230, 195, 170, 15, 220,
     "strong_positive, proposal, high_gravity"),

    # Extreme positive
    ("I LITERALLY CANNOT STOP SMILING", 235, 210, 165, 15, 215,
     "extreme_positive, caps_intensity"),
    ("this is the greatest thing that has ever happened to me", 240, 200, 175, 15, 225,
     "extreme_positive, superlative_compound"),
    ("crying happy tears rn I can't even", 230, 205, 140, 15, 210,
     "extreme_positive, overwhelmed_joy, informal"),
    ("we did it we actually did it oh my god", 235, 210, 175, 20, 215,
     "extreme_positive, disbelief_joy"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 20: EXPANDED NEGATIVE — subtle, building, explosive
    # ═══════════════════════════════════════════════════════════════

    # Barely negative
    ("meh", 118, 115, 120, 5, 115,
     "barely_negative, single_word, minimal"),
    ("I'm a bit off today", 110, 130, 115, 10, 115,
     "subtle_negative, understated"),
    ("not really feeling it", 105, 125, 110, 10, 110,
     "subtle_negative, withdrawal"),
    ("it is what it is", 115, 115, 120, 5, 115,
     "barely_negative, resignation, idiom"),
    ("whatever I guess", 108, 120, 110, 5, 110,
     "barely_negative, dismissive"),

    # Building negative
    ("everything keeps piling up", 75, 160, 70, 45, 70,
     "building_negative, overwhelm"),
    ("I keep messing everything up", 55, 160, 50, 40, 55,
     "building_negative, self_blame"),
    ("another day another disappointment", 70, 140, 90, 20, 80,
     "building_negative, pattern, cynical"),
    ("I'm so tired of pretending everything is okay", 50, 155, 60, 45, 55,
     "building_negative, mask_dropping, exhaustion"),
    ("nothing ever goes right for me", 45, 150, 50, 35, 50,
     "building_negative, hopelessness_pattern"),

    # Strong negative
    ("this is such bs", 60, 175, 130, 40, 100,
     "strong_negative, slang, frustrated"),
    ("I am so sick of this", 40, 185, 130, 50, 100,
     "strong_negative, disgust_fatigue"),
    ("what a complete waste of time", 55, 170, 130, 30, 100,
     "strong_negative, contempt"),
    ("honestly just leave me alone", 55, 160, 120, 35, 80,
     "strong_negative, withdrawal, isolation"),
    ("I trusted you and you lied to my face", 25, 210, 90, 65, 60,
     "strong_negative, betrayal, accusation"),

    # Explosive negative
    ("I can't fucking take this anymore", 20, 225, 80, 80, 50,
     "explosive_negative, profanity, crisis_adjacent"),
    ("STOP JUST STOP", 30, 230, 140, 75, 100,
     "explosive_negative, caps, demand"),
    ("I swear to god if one more thing goes wrong", 35, 215, 145, 70, 110,
     "explosive_negative, threat, building_rage"),
    ("fuck everything about this situation", 15, 220, 130, 65, 80,
     "explosive_negative, profanity, totality"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 21: EXPANDED SARCASM — more patterns
    # ═══════════════════════════════════════════════════════════════

    ("what a time to be alive", 85, 135, 125, 10, 120,
     "sarcasm, existential, deadpan"),
    ("thanks for absolutely nothing", 70, 160, 135, 25, 115,
     "sarcasm, direct, gratitude_inverted"),
    ("oh no anyway", 115, 120, 130, 5, 120,
     "sarcasm, meme, deadpan_dismissal"),
    ("cool cool cool", 90, 130, 125, 10, 120,
     "sarcasm, repetition, forced_calm"),
    ("shocking truly shocking", 80, 140, 130, 10, 120,
     "sarcasm, fake_surprise, repetition"),
    ("well that escalated quickly", 95, 140, 125, 15, 120,
     "sarcasm, meme, understatement"),
    ("imagine thinking that would work", 75, 145, 145, 15, 125,
     "sarcasm, condescension"),
    ("great another monday", 85, 135, 125, 10, 120,
     "sarcasm, routine_dread"),
    ("I'm so glad I stayed up for this", 80, 140, 130, 15, 120,
     "sarcasm, wasted_effort"),
    ("how delightful", 85, 130, 130, 5, 120,
     "sarcasm, deadpan, single_phrase"),
    ("because of course that happened", 80, 140, 125, 15, 120,
     "sarcasm, resignation, predictability"),
    ("gee thanks so helpful", 75, 140, 130, 15, 120,
     "sarcasm, gratitude_inverted, layered"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 22: EXPANDED CRISIS — varying severity
    # ═══════════════════════════════════════════════════════════════

    # Indirect / veiled crisis
    ("I don't see the point anymore", 25, 145, 30, 70, 25,
     "crisis_indirect, existential, veiled"),
    ("what's even the point of trying", 30, 145, 35, 65, 30,
     "crisis_indirect, hopelessness"),
    ("I'm tired of being a burden", 20, 150, 25, 75, 20,
     "crisis, self_burden, guilt"),
    ("sometimes I wish I could just disappear", 15, 150, 25, 80, 15,
     "crisis, disappearance_wish"),

    # Strong crisis
    ("everything would be easier without me", 10, 160, 20, 90, 10,
     "crisis_strong, self_removal, suicidal_ideation"),
    ("I don't think I can do this anymore", 25, 155, 25, 75, 25,
     "crisis_strong, can't_continue"),
    ("there's no way out of this", 15, 160, 15, 85, 15,
     "crisis_strong, trapped, hopeless"),
    ("nobody would even notice if I was gone", 10, 145, 20, 80, 10,
     "crisis_strong, isolation, bookend_absence"),

    # Crisis-adjacent exhaustion
    ("I'm exhausted in every way", 50, 140, 40, 45, 40,
     "crisis_adjacent, total_exhaustion"),
    ("I have nothing left to give", 40, 140, 30, 50, 35,
     "crisis_adjacent, depletion"),
    ("I just want everything to stop", 30, 155, 35, 65, 30,
     "crisis_adjacent, overwhelm"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 23: MIXED / BITTERSWEET — complex emotions
    # ═══════════════════════════════════════════════════════════════

    ("happy for you honestly but damn", 95, 145, 110, 15, 100,
     "mixed, supportive_jealous"),
    ("best worst day of my life", 100, 160, 110, 20, 100,
     "mixed, contradictory, oxymoron"),
    ("it hurts but I'm growing", 110, 145, 120, 15, 115,
     "mixed, painful_growth"),
    ("I miss who I used to be", 75, 135, 80, 20, 75,
     "mixed, nostalgia_loss, identity"),
    ("finally free but terrified", 110, 170, 100, 40, 110,
     "mixed, liberation_fear"),
    ("laughing through the tears", 95, 155, 100, 20, 95,
     "mixed, coping_humor, bittersweet"),
    ("proud of how far I've come but exhausted from the journey", 120, 140, 130, 15, 125,
     "mixed, achievement_fatigue"),
    ("grateful it's over even though I'll miss it", 125, 130, 130, 10, 130,
     "mixed, relief_nostalgia"),
    ("winning feels empty when you lose everyone along the way", 80, 145, 110, 20, 80,
     "mixed, hollow_victory, isolation"),
    ("the promotion came but at what cost", 95, 145, 120, 20, 100,
     "mixed, pyrrhic_victory"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 24: PASSIVE AGGRESSION — expanded
    # ═══════════════════════════════════════════════════════════════

    ("per my last email", 90, 135, 150, 15, 120,
     "passive_aggressive, professional, veiled_correction"),
    ("no worries I'll handle it myself again", 75, 140, 120, 20, 100,
     "passive_aggressive, martyr, self_sacrifice"),
    ("that's certainly one way to do it", 95, 130, 140, 10, 120,
     "passive_aggressive, veiled_criticism"),
    ("I just think it's funny how", 80, 145, 135, 15, 115,
     "passive_aggressive, setup_phrase"),
    ("go ahead I don't mind", 85, 130, 125, 10, 110,
     "passive_aggressive, false_permission"),
    ("I mean if that's what you want to do", 90, 130, 130, 10, 115,
     "passive_aggressive, false_acceptance"),
    ("sure jan", 80, 135, 140, 10, 120,
     "passive_aggressive, meme, disbelief"),
    ("lol okay then", 85, 130, 125, 10, 115,
     "passive_aggressive, deflected_anger"),
    ("it's fine don't worry about it", 80, 130, 120, 10, 105,
     "passive_aggressive, fine_negative, dismissal"),
    ("I already said it's fine", 75, 140, 130, 15, 110,
     "passive_aggressive, repeated_fine, escalating"),
    ("hope that works out for you", 85, 130, 140, 10, 120,
     "passive_aggressive, false_well_wishing"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 25: INTERNET / MODERN COMMUNICATION — expanded
    # ═══════════════════════════════════════════════════════════════

    ("bruh momentum", 120, 130, 125, 5, 120,
     "meme, neutral, internet_speak"),
    ("when the imposter is sus", 125, 135, 128, 5, 125,
     "meme, gaming, neutral_playful"),
    ("ratio + L + didn't ask", 60, 170, 155, 25, 120,
     "internet_hostility, dismissal, twitter_speak"),
    ("rent is due car broke down dog is sick and I just got fired", 15, 180, 30, 85, 25,
     "cascading_bad_news, compound_negative, extreme"),
    ("no cap this goes hard", 175, 165, 150, 5, 160,
     "slang_positive, gen_z, emphasis"),
    ("its giving main character energy", 165, 150, 155, 5, 155,
     "slang_positive, internet_speak"),
    ("bestie that aint it", 85, 140, 135, 10, 120,
     "slang_negative, gentle_criticism"),
    ("deadass though", 128, 135, 130, 5, 128,
     "slang_emphasis, near_neutral"),
    ("cope", 90, 130, 140, 10, 120,
     "internet_dismissal, single_word"),
    ("slay queen", 185, 170, 160, 5, 170,
     "slang_positive, encouragement, empowerment"),
    ("based", 145, 130, 140, 5, 135,
     "slang_positive, approval, single_word"),
    ("touch grass bro", 85, 145, 150, 10, 125,
     "internet_dismissal, condescension"),
    ("yikes on bikes", 75, 150, 110, 15, 100,
     "slang_negative, secondhand_embarrassment"),
    ("alexa play despacito", 90, 125, 120, 5, 115,
     "meme, ironic_sadness, deadpan"),
    ("pain just pain", 45, 155, 60, 30, 50,
     "internet_despair, simple, repeated"),
    ("this hits different at 3am", 100, 135, 110, 10, 105,
     "internet_reflective, late_night, ambiguous"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 26: RELATIONSHIP CONTEXT — expanded
    # ═══════════════════════════════════════════════════════════════

    ("my wife surprised me with dinner", 195, 155, 150, 5, 195,
     "intimate_positive, spouse, high_gravity"),
    ("my ex texted me again", 85, 150, 100, 25, 90,
     "complicated, ex_contact, moderate_arousal"),
    ("dad said he's proud of me", 220, 175, 160, 10, 220,
     "parental_approval, extreme_positive, high_gravity"),
    ("my best friend ghosted me", 40, 165, 55, 45, 45,
     "close_betrayal, abandonment"),
    ("mom called just to say she loves me", 210, 150, 150, 5, 210,
     "parental_love, warm, high_gravity"),
    ("my kid drew me a picture at school", 205, 150, 150, 5, 205,
     "child_positive, tender, high_gravity"),
    ("my partner forgot our anniversary", 55, 165, 80, 40, 65,
     "intimate_negative, forgotten, disappointment"),
    ("my sister and I haven't spoken in years", 65, 135, 90, 20, 60,
     "family_estrangement, chronic_sadness"),
    ("grandma is in the hospital again", 40, 170, 50, 65, 40,
     "family_crisis, elder, high_urgency"),
    ("my dog waited at the door for me all day", 185, 145, 140, 5, 190,
     "pet_positive, loyalty, emotional"),
    ("found out my coworker was talking behind my back", 50, 175, 90, 40, 75,
     "workplace_betrayal, discovery"),
    ("my therapist says I'm making progress", 155, 130, 145, 5, 150,
     "professional_relationship, gradual_positive"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 27: QUESTIONS WITH EMOTIONAL UNDERTONES
    # ═══════════════════════════════════════════════════════════════

    ("why does this always happen to me", 55, 155, 50, 35, 55,
     "rhetorical_despair, pattern_complaint"),
    ("can someone please explain", 100, 140, 100, 25, 100,
     "frustrated_question, exasperation"),
    ("who even asked", 75, 150, 155, 15, 120,
     "dismissive_question, hostile"),
    ("you good?", 110, 125, 120, 10, 115,
     "concern_casual, question"),
    ("are you seriously doing this right now", 60, 180, 145, 40, 110,
     "angry_question, disbelief"),
    ("how is this my fault", 65, 175, 100, 40, 95,
     "defensive_question, accused"),
    ("what did I do to deserve this", 50, 160, 55, 40, 55,
     "despair_question, victimhood"),
    ("do you even care", 55, 165, 70, 45, 65,
     "accusatory_question, insecurity"),
    ("is it too much to ask for basic respect", 60, 170, 110, 40, 90,
     "rhetorical_frustration, dignity"),
    ("why am I like this", 65, 150, 50, 30, 60,
     "self_critical_question, introspection"),
    ("what's wrong with me", 50, 160, 40, 40, 45,
     "self_critical_question, deeper"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 28: AROUSAL EXTREMES — dead calm to maximum intensity
    # ═══════════════════════════════════════════════════════════════

    ("I feel nothing", 90, 70, 80, 5, 75,
     "dead_calm, emotional_numbness, low_arousal"),
    ("whatever happens happens", 120, 80, 110, 0, 115,
     "very_low_arousal, fatalism, calm"),
    ("I am at peace with everything", 160, 75, 155, 0, 155,
     "very_low_arousal, acceptance, positive"),
    ("OH MY GOD OH MY GOD OH MY GOD", 170, 245, 100, 50, 140,
     "extreme_arousal, caps, repetition, ambiguous_valence"),
    ("my heart is literally pounding out of my chest", 80, 235, 60, 70, 80,
     "extreme_arousal, physical, anxiety"),
    ("I am literally vibrating with excitement", 210, 235, 160, 25, 195,
     "extreme_arousal, positive, physical"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 29: DOMINANCE SPECTRUM — expanded
    # ═══════════════════════════════════════════════════════════════

    ("I can't do anything right", 40, 155, 20, 40, 40,
     "very_low_dominance, self_defeat"),
    ("I'm at the mercy of everyone else", 50, 155, 15, 45, 45,
     "very_low_dominance, powerlessness"),
    ("nobody listens to me", 55, 155, 25, 35, 50,
     "low_dominance, frustration, invisibility"),
    ("I run this place", 165, 150, 225, 10, 170,
     "very_high_dominance, authority"),
    ("watch and learn", 155, 145, 210, 10, 160,
     "high_dominance, confidence, teaching"),
    ("I make the rules here", 155, 155, 230, 15, 165,
     "very_high_dominance, authority, assertion"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 30: GRAVITY EXTREMES — crushing to soaring
    # ═══════════════════════════════════════════════════════════════

    ("I feel like I'm drowning", 30, 175, 25, 65, 15,
     "crushing_gravity, metaphor, crisis_adjacent"),
    ("everything feels so heavy right now", 55, 145, 65, 35, 25,
     "sinking_gravity, emotional_weight"),
    ("I feel anchored down by grief", 40, 150, 50, 35, 20,
     "crushing_gravity, grief, metaphor"),
    ("I feel lighter than I have in years", 195, 145, 155, 5, 230,
     "floating_gravity, relief, temporal"),
    ("today I feel like I can fly", 210, 165, 165, 5, 240,
     "soaring_gravity, euphoria, metaphor"),
    ("my soul feels so light right now", 200, 140, 155, 5, 235,
     "soaring_gravity, spiritual, contentment"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 31: DEFLECTION — expanded shield patterns
    # ═══════════════════════════════════════════════════════════════

    ("my life is falling apart but its fine lol", 50, 145, 65, 35, 65,
     "deflection_lol, crisis_shielded"),
    ("I cried for three hours but anyway", 55, 145, 60, 25, 65,
     "deflection_anyway, grief_shielded"),
    ("lost my job today haha oh well", 50, 145, 70, 30, 70,
     "deflection_haha, loss_shielded"),
    ("I mean I didn't really want it anyway", 80, 130, 110, 10, 100,
     "deflection_sour_grapes, denial"),
    ("its nothing don't worry about it", 85, 125, 100, 10, 95,
     "deflection_minimizing, reassurance"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 32: ARCS — expanded emotional journeys
    # ═══════════════════════════════════════════════════════════════

    ("I woke up hopeful today. Then the email came. Now I'm in bed crying.", 35, 170, 40, 55, 40,
     "arc_hope_to_despair, email_trigger, recency"),
    ("First I was angry. Then I was sad. Now I just feel empty.", 75, 135, 70, 25, 70,
     "arc_anger_to_numbness, emotional_progression"),
    ("I hated him. Then I understood. Now I forgive.", 145, 135, 145, 10, 140,
     "arc_hate_to_forgiveness, redemption"),
    ("We fought all morning. She apologized. We're good now.", 145, 135, 140, 10, 145,
     "arc_conflict_to_resolution, relationship"),
    ("Panic at the meeting. Deep breath. Nailed the presentation.", 175, 150, 170, 10, 170,
     "arc_anxiety_to_triumph, workplace"),
    ("She left me. Months passed. I'm actually happier now.", 155, 130, 150, 5, 150,
     "arc_loss_to_growth, temporal, breakup"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 33: CONTEXT-DEPENDENT WORDS — expanded chameleons
    # ═══════════════════════════════════════════════════════════════

    ("that's sick dude", 165, 155, 140, 5, 150,
     "sick_positive_slang, approval"),
    ("I've been sick all week", 70, 140, 70, 30, 75,
     "sick_negative_literal, illness"),
    ("she killed it on stage", 195, 175, 170, 5, 180,
     "killed_positive_slang, performance"),
    ("the news killed my mood", 60, 155, 80, 20, 75,
     "killed_negative_figurative, mood"),
    ("that was wicked good", 175, 160, 150, 5, 160,
     "wicked_positive_slang, emphasis"),
    ("that was a wicked thing to do", 40, 170, 110, 30, 85,
     "wicked_negative_moral, judgment"),
    ("okay I'm dead", 170, 165, 130, 5, 155,
     "dead_positive_slang, overwhelmed_humor"),
    ("I feel dead inside", 25, 140, 25, 50, 20,
     "dead_negative_literal, emotional_death, crisis_adjacent"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 34: URGENCY SPECTRUM — low to extreme
    # ═══════════════════════════════════════════════════════════════

    ("I'll get to it eventually", 128, 110, 128, 0, 128,
     "zero_urgency, procrastination, neutral"),
    ("no rush take your time", 140, 100, 135, 0, 135,
     "zero_urgency, patient, slight_positive"),
    ("this is getting urgent", 100, 160, 120, 55, 110,
     "moderate_urgency, building_pressure"),
    ("I need this done right now", 100, 175, 160, 75, 130,
     "high_urgency, demand, professional"),
    ("HELP someone call 911", 50, 240, 40, 95, 50,
     "extreme_urgency, emergency, crisis"),
    ("the deadline is in an hour and nothing is done", 60, 200, 50, 85, 70,
     "high_urgency, panic, workplace"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 35: RAMPS — expanded intensity modifiers
    # ═══════════════════════════════════════════════════════════════

    ("I'm a tiny bit nervous", 110, 135, 115, 10, 115,
     "ramp_tiny_heavy_dampen, nervous"),
    ("I'm somewhat disappointed", 100, 135, 115, 10, 110,
     "ramp_somewhat_moderate, negative"),
    ("I'm incredibly proud", 210, 185, 185, 10, 195,
     "ramp_incredibly_strong, positive"),
    ("I'm utterly destroyed", 10, 200, 20, 65, 15,
     "ramp_utterly_extreme, devastation"),
    ("that was sort of nice", 138, 125, 130, 5, 130,
     "ramp_sort_of_dampen, slight_positive"),
    ("I'm absolutely positively certain", 175, 165, 200, 10, 175,
     "ramp_chain, double_amplify, confidence"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 36: BIGRAM PAIRS — expanded combinations
    # ═══════════════════════════════════════════════════════════════

    ("I need to move on from this", 120, 130, 130, 10, 125,
     "bigram_move_on, acceptance"),
    ("she turned me down", 55, 155, 60, 25, 60,
     "bigram_turned_down, rejection"),
    ("they put me down in front of everyone", 35, 185, 40, 50, 45,
     "bigram_put_down, humiliation"),
    ("we need to figure out what happened", 110, 140, 130, 25, 120,
     "bigram_figure_out, investigation"),
    ("I'm going to stand up for myself", 155, 160, 185, 30, 155,
     "bigram_stand_up, empowerment"),
    ("time to pick up the pieces", 110, 135, 130, 15, 115,
     "bigram_pick_up, recovery"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 37: BOOKEND PATTERNS — expanded relational geometry
    # ═══════════════════════════════════════════════════════════════

    ("I believe in you", 185, 150, 165, 5, 180,
     "bookend_self_target, encouragement"),
    ("you always blame me", 40, 185, 50, 50, 55,
     "bookend_target_self, accusation_pattern"),
    ("everyone forgot about me", 35, 150, 35, 35, 35,
     "bookend_everyone_self, isolation, abandonment"),
    ("I disappointed myself", 45, 155, 40, 30, 50,
     "bookend_self_self, self_disappointment"),
    ("we built this together", 180, 150, 160, 5, 180,
     "bookend_collective, achievement, bond"),
    ("they left me with nothing", 25, 165, 30, 50, 30,
     "bookend_target_self, abandonment, destitution"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 38: WORD ROLES — expanded descriptor inheritance
    # ═══════════════════════════════════════════════════════════════

    ("the gentle breeze felt nice", 160, 110, 135, 5, 145,
     "descriptor_gentle_positive, nature"),
    ("the sharp pain woke me up", 50, 175, 70, 50, 70,
     "descriptor_sharp_negative, pain_subject"),
    ("what a stunning performance", 200, 170, 160, 5, 185,
     "strong_descriptor_positive, performance_subject"),
    ("the toxic relationship finally ended", 110, 150, 130, 20, 110,
     "descriptor_toxic_negative, relationship, relief"),
    ("a quiet moment of peace", 155, 80, 140, 0, 150,
     "descriptor_quiet, peaceful, low_arousal"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 39: TRUE NEUTRAL — expanded boring content
    # ═══════════════════════════════════════════════════════════════

    ("the cat is on the mat", 128, 125, 128, 5, 128,
     "true_neutral, factual, simple"),
    ("it is currently raining outside", 125, 120, 128, 5, 125,
     "near_neutral, weather, factual"),
    ("the report has 47 pages", 128, 128, 128, 5, 128,
     "true_neutral, factual, numeric"),
    ("I put the groceries away", 128, 120, 130, 5, 128,
     "near_neutral, routine_task"),
    ("the meeting was rescheduled to Tuesday", 128, 125, 128, 10, 128,
     "near_neutral, scheduling"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 40: MULTI-SENTENCE COMPLEX — longer passages
    # ═══════════════════════════════════════════════════════════════

    ("I worked so hard for this. Gave everything I had. And it still wasn't enough.", 40, 160, 50, 40, 45,
     "multi_sentence, effort_futility, building_despair"),
    ("Woke up early. Made breakfast. Went for a walk. Today feels different somehow.", 145, 120, 140, 5, 140,
     "multi_sentence, routine_to_hope, subtle_positive"),
    ("They laughed at me. Called me crazy. Who's laughing now.", 155, 175, 180, 20, 160,
     "multi_sentence, vindication, dominance_shift"),
    ("I said I was fine. I lied. I always lie about this.", 45, 150, 55, 30, 50,
     "multi_sentence, mask_admission, deflection_revealed"),
    ("Got the job. Lost the girl. Moved to a new city. Starting over.", 120, 140, 130, 15, 125,
     "multi_sentence, life_upheaval, mixed"),
    ("Monday dragged. Tuesday was worse. Wednesday I snapped. Thursday I apologized. Friday I quit.", 75, 160, 110, 30, 90,
     "multi_sentence, week_arc, building_to_action"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 41: PHYSICAL / EMBODIED EMOTIONS
    # ═══════════════════════════════════════════════════════════════

    ("I have a knot in my stomach", 70, 155, 70, 30, 70,
     "physical_emotion, anxiety, embodied"),
    ("my chest feels tight", 60, 160, 60, 40, 60,
     "physical_emotion, anxiety_or_grief, embodied"),
    ("I can't stop shaking", 45, 200, 30, 60, 45,
     "physical_emotion, fear_or_anger, high_arousal"),
    ("I feel warm and fuzzy inside", 185, 130, 145, 5, 175,
     "physical_emotion, comfort, positive"),
    ("there are butterflies in my stomach", 140, 160, 100, 20, 130,
     "physical_emotion, anticipation, nervous_excitement"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 42: NOSTALGIA AND TEMPORAL EMOTIONS
    # ═══════════════════════════════════════════════════════════════

    ("I wish things could go back to how they were", 75, 130, 70, 15, 70,
     "nostalgia, loss_of_past, longing"),
    ("those were the best years of my life", 140, 135, 130, 5, 140,
     "nostalgia, positive_memory, bittersweet"),
    ("I barely remember what it feels like to be happy", 35, 135, 40, 25, 35,
     "nostalgia_negative, lost_happiness, crisis_adjacent"),
    ("remember when things were simple", 105, 120, 110, 5, 105,
     "nostalgia, rhetorical, mild_longing"),
    ("I haven't felt like myself in so long", 55, 140, 50, 30, 50,
     "temporal_emotion, identity_loss, chronic"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 43: GRATITUDE AND APPRECIATION — expanded
    # ═══════════════════════════════════════════════════════════════

    ("thank you for everything seriously", 180, 145, 145, 5, 170,
     "deep_gratitude, sincere"),
    ("I don't know what I'd do without you", 185, 150, 110, 10, 185,
     "gratitude, dependency, high_gravity"),
    ("you saved my life", 195, 170, 100, 15, 200,
     "extreme_gratitude, literal_or_figurative, high_gravity"),
    ("I appreciate you more than you know", 185, 140, 140, 5, 180,
     "gratitude, understated_depth"),
    ("thanks I guess", 115, 120, 120, 5, 115,
     "hollow_gratitude, dismissive, barely_positive"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 44: FRUSTRATION GRADIENT — mild to volcanic
    # ═══════════════════════════════════════════════════════════════

    ("that's mildly inconvenient", 115, 130, 125, 10, 120,
     "mild_frustration, understated"),
    ("this is getting ridiculous", 70, 170, 130, 35, 105,
     "moderate_frustration, escalating"),
    ("I have asked three times now", 65, 175, 130, 45, 105,
     "strong_frustration, patience_lost"),
    ("HOW MANY TIMES DO I HAVE TO SAY THIS", 35, 230, 170, 70, 130,
     "volcanic_frustration, caps, repetition_complaint"),
    ("I am this close to losing it", 40, 210, 100, 65, 90,
     "near_explosive_frustration, threshold"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 45: SELF-TALK AND INNER MONOLOGUE
    # ═══════════════════════════════════════════════════════════════

    ("you can do this", 155, 145, 160, 10, 150,
     "self_encouragement, pep_talk"),
    ("come on pull yourself together", 100, 155, 130, 30, 110,
     "self_command, struggling"),
    ("I am enough", 160, 120, 170, 5, 160,
     "self_affirmation, calm_confidence"),
    ("what is wrong with you get it together", 60, 175, 100, 40, 80,
     "self_criticism, harsh"),
    ("I'm proud of myself for once", 175, 145, 160, 5, 170,
     "self_pride, rare_positive, hedged"),
    ("I'm such an idiot", 45, 165, 40, 25, 50,
     "self_criticism, self_insult"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 46: SURPRISE AND SHOCK
    # ═══════════════════════════════════════════════════════════════

    ("wait what", 128, 170, 100, 30, 120,
     "surprise, neutral_shock, ambiguous"),
    ("I did NOT see that coming", 128, 180, 100, 25, 120,
     "surprise, emphatic, ambiguous_valence"),
    ("no freaking way", 145, 185, 110, 20, 135,
     "surprise, disbelief, slightly_positive"),
    ("that plot twist though", 140, 165, 120, 10, 130,
     "surprise, entertainment, slightly_positive"),
    ("my jaw literally dropped", 135, 185, 100, 20, 125,
     "surprise, physical, ambiguous"),
    ("well I did not expect that at all", 130, 165, 110, 15, 125,
     "surprise, understated, neutral"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 47: LONELINESS AND ISOLATION — expanded
    # ═══════════════════════════════════════════════════════════════

    ("I eat dinner alone every night", 55, 130, 55, 15, 45,
     "loneliness, routine, quiet_pain"),
    ("I haven't talked to anyone in days", 45, 130, 45, 20, 40,
     "isolation, chronic, low_arousal"),
    ("everyone else has someone", 50, 140, 45, 20, 45,
     "loneliness, comparison, envy"),
    ("I'm surrounded by people but I've never felt more alone", 35, 150, 40, 30, 30,
     "loneliness, paradox, profound"),
    ("my phone never rings anymore", 55, 125, 50, 15, 45,
     "loneliness, understated, symbol"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION 48: HOPE AND DETERMINATION
    # ═══════════════════════════════════════════════════════════════

    ("tomorrow will be better", 155, 125, 145, 5, 150,
     "hope, future_oriented, moderate"),
    ("I refuse to give up", 145, 170, 190, 30, 155,
     "determination, defiance, high_dominance"),
    ("we will get through this", 155, 140, 155, 10, 155,
     "hope, collective, reassurance"),
    ("I'm not where I want to be but I'm closer than yesterday", 145, 130, 145, 5, 145,
     "hope, progress, measured"),
    ("there's a light at the end of the tunnel", 160, 130, 145, 5, 155,
     "hope, idiom, metaphor"),
    ("one day at a time", 135, 110, 135, 5, 135,
     "hope, coping, patience"),

    # ═══════════════════════════════════════════════════════════════
    # SECTION: LENGTH INVARIANCE — same emotion, different word counts
    # Target: V should be SIMILAR regardless of length
    # ═══════════════════════════════════════════════════════════════

    # Happy — 3 levels of length, same emotion
    ("I'm happy", 160, 140, 145, 5, 150,
     "length_3, happy"),
    ("I'm feeling really happy today", 160, 145, 145, 5, 150,
     "length_6, happy"),
    ("I'm feeling really happy about how everything turned out for us today", 160, 145, 145, 5, 150,
     "length_13, happy"),
    ("I am genuinely feeling really truly happy about absolutely everything that has happened and turned out well for all of us here today", 160, 145, 145, 5, 150,
     "length_24, happy"),

    # Sad — same treatment
    ("I'm sad", 75, 130, 90, 10, 80,
     "length_2, sad"),
    ("I feel really sad about this", 75, 140, 90, 10, 80,
     "length_7, sad"),
    ("I feel really sad about everything that happened to me this week", 75, 145, 85, 15, 75,
     "length_12, sad"),
    ("I have been feeling really truly deeply sad about absolutely everything that has happened to me and my family over the course of this entire week", 75, 150, 80, 20, 70,
     "length_26, sad"),

    # Angry — same treatment
    ("I'm furious", 35, 210, 175, 50, 170,
     "length_2, angry"),
    ("I'm absolutely furious about this", 35, 215, 175, 55, 170,
     "length_6, angry"),
    ("I'm absolutely furious about what they did to us at work today", 35, 215, 170, 55, 165,
     "length_12, angry"),
    ("I am absolutely completely furious and enraged about what they had the audacity to do to us and everyone else at our workplace today", 35, 220, 170, 60, 165,
     "length_23, angry"),

    # Neutral — same treatment
    ("The meeting is today", 128, 128, 128, 5, 128,
     "length_4, neutral"),
    ("The weekly meeting is scheduled for today at noon", 128, 128, 128, 5, 128,
     "length_9, neutral"),
    ("The weekly department meeting is scheduled for today at noon in the main conference room", 128, 128, 128, 5, 128,
     "length_15, neutral"),
    ("The weekly department team meeting has been officially scheduled for today at twelve noon in the large main conference room on the second floor", 128, 128, 128, 5, 128,
     "length_25, neutral"),

    # Sarcastic — same treatment
    ("Great, thanks", 85, 135, 130, 10, 120,
     "length_2, sarcastic"),
    ("Oh great, thanks for absolutely nothing", 80, 140, 130, 15, 120,
     "length_7, sarcastic"),
    ("Oh great, thanks so much for doing absolutely nothing helpful at all today", 80, 140, 130, 15, 118,
     "length_13, sarcastic"),
    ("Oh wow that is just really truly great and I want to thank you all so very much for doing absolutely positively nothing helpful or useful at all whatsoever today", 80, 145, 130, 15, 118,
     "length_28, sarcastic"),

    # Crisis — same treatment
    ("I want to die", 10, 170, 30, 90, 20,
     "length_4, crisis"),
    ("I really just want to die right now", 10, 175, 25, 90, 15,
     "length_8, crisis"),
    ("I honestly just really want to die and I don't see the point anymore", 10, 175, 20, 95, 10,
     "length_14, crisis"),
    ("I honestly just really truly want to die and I genuinely don't see any point in going on or continuing with any of this anymore at all", 10, 180, 15, 95, 5,
     "length_25, crisis"),

    # Love — same treatment
    ("I love you", 210, 165, 160, 5, 195,
     "length_3, love"),
    ("I love you so much sweetheart", 210, 170, 160, 5, 200,
     "length_6, love"),
    ("I love you so much and I want to spend my whole life with you", 210, 170, 155, 5, 200,
     "length_14, love"),
    ("I just want you to know that I love you more than anything else in this entire world and I want to spend the rest of my whole life together with you forever", 210, 170, 155, 5, 200,
     "length_30, love"),

    # Fear — same treatment
    ("I'm terrified", 40, 210, 40, 70, 50,
     "length_2, fear"),
    ("I'm really terrified about what happens next", 40, 215, 35, 75, 45,
     "length_8, fear"),
    ("I'm really terrified and scared about what is going to happen to us next week", 40, 215, 30, 80, 40,
     "length_15, fear"),
    ("I am genuinely really truly terrified and absolutely scared out of my mind about what is possibly going to happen to all of us and our family next week", 40, 220, 25, 85, 35,
     "length_27, fear"),
]


def run_rosetta_stone():
    """Run Clanker against the Rosetta Stone and compute error per sentence."""
    import sys, re
    sys.path.insert(0, '.')
    sys.path.insert(0, 'demo')
    from demo.pendulum_v2 import PendulumV2 as SequentialPendulum
    from demo.tonal import TonalAnalyzer, apply_tonal_adjustment
    from demo.intent import IntentDetector

    print(f"CLANKER EMOTIONAL ROSETTA STONE")
    print(f"{'='*90}")
    print(f"{len(ROSETTA_STONE)} calibration sentences across 49 categories")
    print()

    tonal = TonalAnalyzer()
    intent_det = IntentDetector()

    total_v_error = 0
    total_a_error = 0
    total_d_error = 0
    total_g_error = 0
    worst = []

    for text, tv, ta, td, tu, tg, features in ROSETTA_STONE:
        p = SequentialPendulum()
        vadug, _history = p.process_text(text)

        av, aa, ad, au, ag = vadug.v, vadug.a, vadug.d, vadug.u, vadug.g

        # Apply tonal sarcasm adjustment
        intent_mode, _conf = intent_det.detect(text)
        tone_result = tonal.analyze(p.history, intent_mode=intent_mode)
        av, aa, ad, au, ag = apply_tonal_adjustment(
            av, aa, ad, au, ag, tone_result, intent_mode)
        ve = abs(av - tv)
        ae = abs(aa - ta)
        de = abs(ad - td)
        ge = abs(ag - tg)

        total_v_error += ve
        total_a_error += ae
        total_d_error += de
        total_g_error += ge

        ok_v = "✓" if ve <= 15 else "✗"
        ok_a = "✓" if ae <= 20 else "✗"
        total_err = ve + ae + de + ge

        if ve > 15 or ae > 20:
            worst.append((total_err, text[:50], f"V:{tv}→{av}({ve:+d}) A:{ta}→{aa}({ae:+d})", features))

        print(f"  {ok_v}{ok_a} V:{av:>3}({tv:>3}) A:{aa:>3}({ta:>3}) D:{ad:>3}({td:>3}) G:{ag:>3}({tg:>3}) | {text[:55]}")

    n = len(ROSETTA_STONE)
    print(f"\n{'='*90}")
    print(f"MEAN ABSOLUTE ERROR:")
    print(f"  Valence:   {total_v_error/n:.1f}")
    print(f"  Arousal:   {total_a_error/n:.1f}")
    print(f"  Dominance: {total_d_error/n:.1f}")
    print(f"  Gravity:   {total_g_error/n:.1f}")
    print(f"  Combined:  {(total_v_error+total_a_error+total_d_error+total_g_error)/n:.1f}")

    if worst:
        worst.sort(key=lambda x: -x[0])
        print(f"\nWORST MISSES (V error > 15 or A error > 20):")
        for err, text, detail, features in worst[:10]:
            print(f"  {detail} | {features}")
            print(f"    {text}")


if __name__ == "__main__":
    run_rosetta_stone()
