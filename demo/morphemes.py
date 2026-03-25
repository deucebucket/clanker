"""
Clanker Morphological Emotional Decomposition

Instead of storing every word, we store:
- ~1000 root morphemes with emotional weights
- ~30 prefixes with modifier rules
- ~40 suffixes with modifier rules

This covers MILLIONS of words from ~1070 entries.
Unknown words are decomposed into known parts.
"""

# ─────────────────────────────────────────────────────────
# PREFIX MODIFIERS
# Each prefix has: (v_mod, a_mod, d_mod, u_mod, g_mod, rule)
# Rules: "negate" flips V and D, "amplify" multiplies, "reduce" dampens
# G (Gravity): physical weight of emotion. 0=crushing, 128=grounded, 255=soaring.
# ─────────────────────────────────────────────────────────

PREFIXES = {
    # Negation prefixes — flip valence and reduce dominance, slight G flip
    "un":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.3, "u": 0.05, "g": -0.15},
    "in":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.2, "u": 0.05, "g": -0.1},
    "im":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.2, "u": 0.05, "g": -0.1},
    "ir":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.2, "u": 0.05, "g": -0.1},
    "il":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.2, "u": 0.05, "g": -0.1},
    "dis":     {"rule": "negate",  "v": -1.0,  "a": 0.15, "d": -0.3, "u": 0.1,  "g": -0.15},
    "de":      {"rule": "negate",  "v": -0.8,  "a": 0.1,  "d": -0.25,"u": 0.1,  "g": -0.1},
    "non":     {"rule": "negate",  "v": -0.6,  "a": 0.0,  "d": -0.1, "u": 0.0,  "g": 0.0},

    # Intensifier prefixes
    "over":    {"rule": "amplify", "v": 0.3,   "a": 0.4,  "d": 0.1,  "u": 0.2,  "g": 0.15},
    "hyper":   {"rule": "amplify", "v": 0.2,   "a": 0.6,  "d": 0.2,  "u": 0.3,  "g": 0.2},
    "super":   {"rule": "amplify", "v": 0.3,   "a": 0.5,  "d": 0.3,  "u": 0.1,  "g": 0.2},
    "ultra":   {"rule": "amplify", "v": 0.3,   "a": 0.6,  "d": 0.3,  "u": 0.2,  "g": 0.2},
    "mega":    {"rule": "amplify", "v": 0.3,   "a": 0.5,  "d": 0.3,  "u": 0.1,  "g": 0.15},

    # Reduction prefixes — things become heavier
    "under":   {"rule": "reduce",  "v": -0.2,  "a": -0.2, "d": -0.3, "u": 0.1,  "g": -0.2},
    "sub":     {"rule": "reduce",  "v": -0.1,  "a": -0.1, "d": -0.2, "u": 0.0,  "g": -0.1},
    "mis":     {"rule": "negate",  "v": -0.7,  "a": 0.2,  "d": -0.2, "u": 0.15, "g": -0.15},

    # Directional prefixes
    "re":      {"rule": "modify",  "v": 0.1,   "a": 0.1,  "d": 0.15, "u": 0.1,  "g": 0.05},
    "pre":     {"rule": "modify",  "v": 0.05,  "a": 0.05, "d": 0.15, "u": 0.15, "g": 0.0},
    "post":    {"rule": "modify",  "v": 0.0,   "a": -0.05,"d": 0.1,  "u": -0.05,"g": 0.0},
    "anti":    {"rule": "negate",  "v": -0.8,  "a": 0.3,  "d": 0.2,  "u": 0.15, "g": 0.1},
    "counter": {"rule": "negate",  "v": -0.5,  "a": 0.2,  "d": 0.3,  "u": 0.1,  "g": 0.05},
    "out":     {"rule": "amplify", "v": 0.2,   "a": 0.3,  "d": 0.3,  "u": 0.1,  "g": 0.15},
    "fore":    {"rule": "modify",  "v": 0.05,  "a": 0.1,  "d": 0.2,  "u": 0.2,  "g": 0.05},
    "co":      {"rule": "modify",  "v": 0.15,  "a": 0.05, "d": 0.0,  "u": 0.0,  "g": 0.0},

    # Additional prefixes
    "extra":   {"rule": "amplify", "v": 0.2,   "a": 0.3,  "d": 0.2,  "u": 0.1},
    "inter":   {"rule": "modify",  "v": 0.05,  "a": 0.1,  "d": 0.0,  "u": 0.05},
    "trans":   {"rule": "modify",  "v": 0.05,  "a": 0.15, "d": 0.1,  "u": 0.1},
    "multi":   {"rule": "amplify", "v": 0.1,   "a": 0.2,  "d": 0.1,  "u": 0.1},
    "semi":    {"rule": "reduce",  "v": -0.1,  "a": -0.1, "d": -0.05,"u": 0.0},
    "pseudo":  {"rule": "reduce",  "v": -0.2,  "a": 0.0,  "d": -0.1, "u": 0.0},
    "micro":   {"rule": "reduce",  "v": -0.05, "a": -0.1, "d": -0.1, "u": 0.0},
    "macro":   {"rule": "amplify", "v": 0.1,   "a": 0.15, "d": 0.15, "u": 0.05},
    "auto":    {"rule": "modify",  "v": 0.1,   "a": 0.0,  "d": 0.2,  "u": 0.0},
}

# ─────────────────────────────────────────────────────────
# SUFFIX MODIFIERS
# G modifiers: -less makes things heavier (-20), -ful makes things lighter (+15)
# ─────────────────────────────────────────────────────────

SUFFIXES = {
    # Absence / negation — things become heavier without something
    "less":    {"rule": "negate",  "v": -1.0,  "a": -0.1, "d": -0.4, "u": 0.1,  "g": -0.2},

    # Fullness / presence — things become lighter with something
    "ful":     {"rule": "amplify", "v": 0.3,   "a": 0.1,  "d": 0.15, "u": 0.0,  "g": 0.15},
    "ous":     {"rule": "amplify", "v": 0.2,   "a": 0.15, "d": 0.1,  "u": 0.05, "g": 0.1},
    "ious":    {"rule": "amplify", "v": 0.2,   "a": 0.15, "d": 0.1,  "u": 0.05, "g": 0.1},
    "ive":     {"rule": "amplify", "v": 0.15,  "a": 0.2,  "d": 0.2,  "u": 0.05, "g": 0.05},
    "able":    {"rule": "modify",  "v": 0.1,   "a": 0.0,  "d": 0.15, "u": 0.0,  "g": 0.05},
    "ible":    {"rule": "modify",  "v": 0.1,   "a": 0.0,  "d": 0.15, "u": 0.0,  "g": 0.05},

    # State / condition
    "ness":    {"rule": "modify",  "v": 0.0,   "a": -0.05,"d": -0.05,"u": 0.0,  "g": 0.0},
    "ment":    {"rule": "modify",  "v": 0.0,   "a": 0.0,  "d": 0.05, "u": 0.05, "g": 0.0},
    "tion":    {"rule": "modify",  "v": 0.0,   "a": 0.05, "d": 0.05, "u": 0.05, "g": 0.0},
    "sion":    {"rule": "modify",  "v": 0.0,   "a": 0.05, "d": 0.05, "u": 0.05, "g": 0.0},
    "ity":     {"rule": "modify",  "v": 0.0,   "a": 0.0,  "d": 0.05, "u": 0.0,  "g": 0.0},
    "ism":     {"rule": "modify",  "v": -0.05, "a": 0.1,  "d": 0.1,  "u": 0.05, "g": 0.0},
    "ist":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.15, "u": 0.05, "g": 0.0},

    # Degree / comparison
    "er":      {"rule": "amplify", "v": 0.15,  "a": 0.1,  "d": 0.1,  "u": 0.05, "g": 0.1},
    "est":     {"rule": "amplify", "v": 0.3,   "a": 0.2,  "d": 0.15, "u": 0.1,  "g": 0.15},

    # Action / process
    "ing":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.05, "u": 0.05, "g": 0.0},
    "ed":      {"rule": "modify",  "v": -0.05, "a": -0.05,"d": 0.0,  "u": -0.05,"g": 0.0},
    "ly":      {"rule": "modify",  "v": 0.0,   "a": 0.05, "d": 0.0,  "u": 0.0,  "g": 0.0},
    "en":      {"rule": "modify",  "v": 0.05,  "a": 0.1,  "d": 0.1,  "u": 0.05, "g": 0.0},
    "ize":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.15, "u": 0.1,  "g": 0.0},
    "ify":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.1,  "u": 0.1,  "g": 0.0},
    "ate":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.1,  "u": 0.1,  "g": 0.0},

    # Person/agent
    "or":      {"rule": "modify",  "v": 0.05,  "a": 0.05, "d": 0.15, "u": 0.0,  "g": 0.0},

    # Diminutive
    "ish":     {"rule": "reduce",  "v": -0.1,  "a": -0.1, "d": -0.05,"u": 0.0,  "g": 0.0},
    "ette":    {"rule": "reduce",  "v": 0.1,   "a": -0.1, "d": -0.1, "u": 0.0,  "g": 0.05},

    # Additional suffixes
    "ward":    {"rule": "modify",  "v": 0.0,   "a": 0.05, "d": 0.05, "u": 0.05},
    "proof":   {"rule": "amplify", "v": 0.15,  "a": -0.1, "d": 0.3,  "u": -0.1},
    "dom":     {"rule": "modify",  "v": 0.0,   "a": 0.0,  "d": 0.1,  "u": 0.0},
    "hood":    {"rule": "modify",  "v": 0.05,  "a": 0.0,  "d": 0.05, "u": 0.0},
    "ship":    {"rule": "modify",  "v": 0.05,  "a": 0.05, "d": 0.05, "u": 0.0},
    "ling":    {"rule": "reduce",  "v": 0.1,   "a": -0.1, "d": -0.15,"u": 0.0},
    "esque":   {"rule": "modify",  "v": 0.1,   "a": 0.05, "d": 0.0,  "u": 0.0},
}

# ─────────────────────────────────────────────────────────
# ROOT MORPHEMES — emotional core of words
# (v_weight, a_weight, d_weight, u_weight, g_weight)
# Scaled -100 to +100 for each axis
# G (Gravity): positive = lighter/soaring, negative = heavier/sinking
#   Joy: +40 to +70 (soaring), Sadness: -30 to -50 (sinking)
#   Anger: +20 to +50 (rising/boiling), Fear: +10 to +40 (floating ungrounded)
#   Calm: -5 to +10 (grounded), Despair: -60 to -80 (crushing)
#   Disgust: +20 to +50 (repulsion rises), Neutral: 0 (grounded)
# ─────────────────────────────────────────────────────────

ROOTS = {
    # ── Positive emotions ──
    "joy":      (+70, +40, +30, 0, +55),
    "hap":      (+60, +30, +25, 0, +45),     # happy, happen
    "glad":     (+50, +25, +20, 0, +35),
    "pleas":    (+55, +20, +20, 0, +30),     # please, pleasant, pleasure
    "delight":  (+65, +35, +25, 0, +50),
    "bliss":    (+75, +20, +30, 0, +65),
    "euph":     (+80, +60, +30, 0, +70),     # euphoria, euphoric
    "grat":     (+60, +20, +25, 0, +40),     # grateful, gratitude
    "content":  (+45, -15, +25, 0, +10),
    "seren":    (+40, -30, +25, 0, +15),     # serene, serenity
    "calm":     (+30, -40, +20, 0, +5),
    "peace":    (+45, -35, +25, 0, +15),
    "comfort":  (+50, -25, +30, 0, +20),
    "warm":     (+40, -10, +20, 0, +15),
    "kind":     (+50, -5, +15, 0, +25),
    "gentle":   (+35, -20, +10, 0, +10),
    "tender":   (+40, -15, +5, 0, +10),
    "lov":      (+75, +40, +20, 0, +55),     # love, loving, lovely
    "ador":     (+70, +35, +15, 0, +50),     # adore, adorable
    "cherish":  (+65, +15, +20, 0, +40),
    "car":      (+45, +5, +20, 0, +15),      # care, caring
    "trust":    (+50, -10, +30, 0, +15),
    "faith":    (+45, +10, +25, 0, +20),
    "hop":      (+55, +20, +25, 0, +40),     # hope, hoping

    # ── Negative emotions ──
    "sad":      (-55, -15, -30, +5, -35),
    "grief":    (-70, +30, -40, +20, -50),
    "sorrow":   (-65, +10, -35, +10, -45),
    "mourn":    (-60, +15, -35, +10, -40),
    "despair":  (-75, +25, -60, +35, -75),
    "desp":     (-70, +20, -55, +30, -70),   # desperate, desperation
    "miser":    (-65, +10, -45, +15, -50),   # miserable, misery
    "suffer":   (-60, +30, -40, +25, -40),
    "pain":     (-55, +35, -30, +25, -30),
    "agon":     (-65, +50, -35, +30, -35),   # agony, agonize
    "torment":  (-60, +45, -35, +30, -30),
    "doom":     (-70, +20, -55, +30, -65),
    "dread":    (-60, +40, -50, +35, -40),
    "woe":      (-60, +15, -40, +15, -45),
    "gloom":    (-45, -20, -25, +5, -35),
    "melan":    (-40, -15, -20, +5, -30),    # melancholy, melancholic

    # ── Anger ── (anger rises/boils: positive G)
    "ang":      (-50, +55, +25, +25, +30),   # anger, angry
    "rage":     (-55, +75, +40, +35, +45),
    "fur":      (-55, +70, +40, +35, +45),   # fury, furious
    "wrath":    (-55, +65, +45, +35, +40),
    "hate":     (-65, +50, +30, +25, +35),
    "host":     (-50, +40, +25, +20, +25),   # hostile, hostility
    "resent":   (-45, +30, +15, +15, +10),
    "bitter":   (-40, +20, +10, +10, -10),
    "spite":    (-45, +35, +25, +15, +20),
    "venge":    (-50, +50, +35, +30, +30),   # revenge, vengeance
    "irrit":    (-30, +30, +10, +15, +15),   # irritate, irritable

    # ── Fear ── (fear floats ungrounded: moderate positive G)
    "fear":     (-50, +50, -50, +30, +20),
    "terr":     (-65, +70, -55, +40, +30),   # terror, terrible, terrify
    "horr":     (-60, +60, -45, +35, +25),   # horror, horrible, horrify
    "panic":    (-55, +75, -60, +50, +35),
    "anxi":     (-40, +45, -35, +30, +20),   # anxiety, anxious
    "worry":    (-30, +30, -25, +25, +10),
    "nerv":     (-25, +35, -20, +20, +15),   # nervous, nerve
    "dread":    (-55, +40, -45, +30, +15),
    "fright":   (-50, +55, -50, +35, +25),
    "scar":     (-50, +50, -45, +30, +20),   # scare, scared, scary
    "phob":     (-50, +55, -50, +30, +25),   # phobia, phobic

    # ── Surprise ──
    "surpris":  (+10, +60, -15, +20, +25),
    "amaz":     (+40, +55, +10, +10, +40),   # amaze, amazing
    "astonish": (+20, +60, -10, +15, +30),
    "shock":    (-15, +70, -30, +30, +20),
    "stun":     (-5, +65, -25, +25, +15),
    "startle":  (-10, +60, -20, +20, +20),
    "awe":      (+30, +50, -10, +10, +35),

    # ── Disgust ── (disgust rises up: positive G)
    "disgust":  (-50, +40, +15, +15, +30),
    "repuls":   (-55, +45, +20, +15, +35),
    "revolt":   (-45, +50, +25, +20, +30),
    "nause":    (-40, +35, -15, +15, +20),
    "sick":     (-35, +25, -20, +15, -10),
    "gross":    (-30, +25, +5, +5, +20),
    "vile":     (-55, +35, +15, +15, +30),

    # ── Power / agency ──
    "power":    (+20, +40, +60, +15, +20),
    "strong":   (+25, +35, +55, +10, +15),
    "might":    (+15, +30, +50, +10, +15),
    "conquer":  (+20, +45, +60, +20, +25),
    "domin":    (+5, +40, +65, +15, +20),    # dominate, dominance
    "control":  (+15, +20, +55, +10, +5),
    "command":  (+10, +30, +50, +15, +15),
    "lead":     (+20, +25, +45, +10, +15),
    "confid":   (+35, +20, +45, +5, +20),    # confident, confidence
    "brave":    (+30, +35, +50, +15, +25),
    "courag":   (+35, +40, +55, +15, +30),   # courage, courageous
    "bold":     (+25, +35, +50, +10, +20),

    # ── Weakness / submission ── (sinking)
    "weak":     (-25, -15, -50, +10, -25),
    "help":     (-10, +10, -30, +20, -10),   # helpless (with -less)
    "vulner":   (-20, +20, -45, +15, -20),   # vulnerable
    "frag":     (-15, +5, -40, +10, -15),    # fragile
    "submiss":  (-20, -10, -55, +5, -20),    # submission, submissive
    "defeat":   (-45, +15, -50, +15, -40),
    "fail":     (-40, +15, -40, +15, -30),
    "lose":     (-40, +20, -35, +15, -30),   # lose, lost, loser
    "worth":    (+30, +10, +25, 0, +15),     # worth (base positive; "worthless" = worth + less)
    "use":      (+15, +10, +20, +5, +5),     # use (base positive; "useless" = use + less)

    # ── Cognitive ──
    "think":    (+10, +15, +20, +5, +5),
    "know":     (+20, +10, +30, +5, +5),
    "learn":    (+25, +20, +20, +5, +15),
    "understand": (+25, +15, +25, +5, +10),
    "confus":   (-20, +25, -25, +15, +10),   # confused, confusion (floating ungrounded)
    "doubt":    (-15, +15, -20, +10, -5),
    "wonder":   (+15, +25, -5, +5, +20),
    "curio":    (+20, +25, +10, +5, +15),    # curious, curiosity
    "creat":    (+35, +30, +30, +5, +30),    # create, creative
    "imagin":   (+30, +25, +20, +5, +30),    # imagine, imagination
    "inspir":   (+40, +35, +25, +5, +40),    # inspire, inspiration

    # ── Social ──
    "friend":   (+50, +20, +15, 0, +25),
    "belong":   (+40, +10, +15, 0, +10),
    "alone":    (-35, -15, -20, +5, -25),
    "lonely":   (-40, -10, -25, +5, -30),
    "reject":   (-45, +30, -35, +15, -25),
    "accept":   (+40, +10, +20, 0, +15),
    "welcom":   (+45, +15, +20, 0, +20),
    "safe":     (+40, -20, +35, -10, +10),
    "secur":    (+35, -15, +35, -5, +5),     # secure, security
    "danger":   (-35, +45, -30, +40, +15),
    "threat":   (-40, +50, -25, +40, +20),

    # ── Functional / neutral ──
    "work":     (+10, +15, +20, +15, 0),
    "build":    (+20, +20, +25, +10, +5),
    "fix":      (+15, +15, +20, +20, +5),
    "break":    (-30, +30, -15, +20, -15),
    "change":   (+5, +20, +10, +15, +5),
    "grow":     (+25, +15, +15, +5, +15),
    "move":     (+10, +20, +15, +10, +5),
    "stop":     (-10, +15, +15, +15, -5),
    "start":    (+15, +20, +20, +15, +10),
    "end":      (-5, +5, +10, +10, -5),
    "wait":     (-5, -10, -10, +10, -5),
    "rush":     (-5, +40, +10, +45, +15),
    "hurry":    (-5, +35, +5, +40, +10),
    "quick":    (+5, +25, +15, +30, +10),
    "slow":     (-5, -15, -5, -10, -5),

    # ── Strong positive (expanded) ──
    "incred":   (+60, +45, +20, 0, +50),     # incredible
    "magnif":   (+55, +35, +20, 0, +45),     # magnificent (alt stem)
    "magnific": (+55, +35, +25, 0, +45),     # magnificent
    "spectac":  (+55, +45, +20, 0, +50),     # spectacular
    "phenomen": (+50, +40, +15, 0, +45),     # phenomenal
    "brillian": (+50, +35, +25, 0, +45),     # brilliant
    "gorge":    (+45, +25, +15, 0, +30),     # gorgeous
    "triumph":  (+50, +45, +50, +5, +55),    # triumphant
    "radic":    (+10, +40, +20, +15, +15),   # radical

    # ── Strong negative (expanded) ──
    "pathet":   (-45, +15, -35, +10, -30),   # pathetic
    "nightmar": (-60, +50, -40, +30, -40),   # nightmare
    "catastroph": (-65, +55, -35, +40, -35), # catastrophe
    "overwhelm": (-30, +50, -30, +25, -20),  # overwhelm
    "exhaust":  (-25, -20, -30, +10, -25),   # exhausted, exhausting
    "embarrass": (-35, +30, -30, +10, -15),  # embarrassed
    "awkward":  (-20, +20, -25, +5, -5),     # awkward
    "guilt":    (-40, +25, -25, +10, -30),   # guilty
    "shame":    (-45, +30, -35, +10, -35),   # ashamed, shame
    "betray":   (-60, +45, -35, +25, -30),   # betrayal
    "abandon":  (-55, +35, -45, +20, -40),   # abandoned
    "toxic":    (-50, +30, +10, +15, +25),   # toxic (rises)
    "manipul":  (-45, +25, +15, +15, +15),   # manipulate
    "hypocrit": (-40, +30, +10, +10, +15),   # hypocrite
    "frustrat": (-35, +40, -15, +20, +15),   # frustrated, frustration (boils up)
    "devastat": (-60, +40, -40, +25, -45),   # devastated, devastating
    "wretch":   (-60, +20, -40, +15, -50),   # wretched
    "abysm":    (-55, +15, -40, +10, -55),   # abysmal

    # ── Resilience / strength (expanded) ──
    "resili":   (+30, +25, +40, +5, +20),    # resilient
    "persever": (+25, +30, +40, +10, +15),   # persevere
    "endur":    (+15, +20, +35, +10, +5),    # endure
    "surviv":   (+15, +25, +30, +10, +10),   # survive

    # ── Confrontational ──
    "arrog":    (-30, +20, +30, +5, +20),    # arrogant (rises)
    "ignor":    (-25, +15, -10, +5, 0),      # ignorant, ignore
    "contempt": (-45, +35, +30, +10, +25),   # contempt (rises)
    "disgrac":  (-50, +30, -20, +10, -25),   # disgrace (sinks)
    "ridic":    (-25, +30, +10, +10, +15),   # ridiculous
    "absurd":   (-25, +30, +10, +10, +15),   # absurd

    # ── Physical intensity ──
    "drown":    (-40, +45, -35, +25, -50),   # drowning (sinking)
    "suffoc":   (-40, +50, -35, +30, -45),   # suffocate (crushing)
    "crush":    (-35, +40, -20, +20, -60),   # crushing (heavy)
    "shatter":  (-40, +45, -25, +25, -20),   # shattering
    "explod":   (-20, +60, +10, +35, +35),   # exploding (rises)
    "burn":     (-20, +45, +10, +25, +25),   # burning (rises)

    # ── Relational (expanded) ──
    "forgiv":   (+30, +10, +20, 0, +20),     # forgive, forgiveness
    "devot":    (+40, +20, +25, 0, +25),     # devoted, devotion
    "loyal":    (+35, +10, +25, 0, +10),     # loyal, loyalty
    "treasur":  (+40, +15, +20, 0, +25),     # treasure
    "appreci":  (+35, +15, +20, 0, +20),     # appreciate
    "admir":    (+35, +20, +20, 0, +25),     # admire
    "motiv":    (+30, +30, +25, +5, +20),    # motivate
    "encourag": (+30, +20, +20, 0, +20),     # encourage
    "comfor":   (+40, -15, +25, 0, +15),     # comfort (alt stem)
    "heal":     (+30, +10, +20, 0, +20),     # heal, healing
    "recov":    (+25, +15, +20, +5, +15),    # recover
    "overcom":  (+30, +30, +35, +5, +25),    # overcome

    # ── Latin/Greek roots ──
    "bene":     (+40, +10, +20, 0),
    "mal":      (-40, +20, -10, +10),
    "mort":     (-60, +30, -40, +20),
    "vit":      (+40, +20, +20, 0),
    "liber":    (+45, +25, +35, 0),
    "serv":     (+15, +10, -10, +10),
    "rupt":     (-30, +45, -15, +25),
    "tract":    (-10, +15, +10, +10),
    "dict":     (+5, +10, +15, +5),
    "port":     (+10, +10, +10, +10),
    "ject":     (-10, +20, +5, +15),
    "cred":     (+25, +10, +20, 0),
    "fort":     (+25, +15, +30, 0),
    "pend":     (-15, +15, -15, +15),
    "sens":     (+10, +20, -5, +5),
    "phil":     (+40, +20, +10, 0),
    "psych":    (+5, +15, +5, +5),
    "soph":     (+20, +10, +20, 0),
    "chron":    (-5, +5, 0, +15),
    "gen":      (+15, +10, +10, 0),
    "nov":      (+20, +20, +10, +5),
    "clar":     (+20, +10, +15, 0),
    "grav":     (-25, +15, -15, +15),
    "lev":      (+20, +10, +15, 0),
    "press":    (-20, +20, -15, +15),
    "tens":     (-15, +25, -10, +15),
    "sol":      (+15, -10, +15, 0),
    "vac":      (-10, -15, -10, 0),
    "cap":      (+10, +15, +20, +5),

    # ── Emotional core roots ──
    "ennui":    (-20, -25, -15, 0),
    "yearn":    (-15, +20, -15, +10),
    "pang":     (-30, +30, -15, +15),
    "blush":    (-15, +25, -20, +5),
    "thrill":   (+50, +50, +20, 0),
    "glow":     (+35, +15, +15, 0),
    "sting":    (-25, +30, -15, +15),
    "sooth":    (+30, -20, +20, 0),
    "haunt":    (-35, +25, -25, +10),
    # "dread" already defined above
    "gleam":    (+25, +15, +10, 0),
    "wilt":     (-25, -15, -20, +5),
    "bloom":    (+35, +20, +15, 0),
    "rot":      (-40, +15, -25, +10),
    "thrive":   (+40, +25, +25, 0),
    "languish": (-30, -20, -25, +10),
    "exalt":    (+45, +35, +30, 0),
    "lament":   (-40, +20, -25, +10),
    "revel":    (+40, +35, +20, 0),
    "brood":    (-25, +15, -10, +5),
    "sulk":     (-25, +10, -20, +5),
    "fret":     (-20, +25, -20, +15),
    "pine":     (-30, +15, -20, +10),
    "smolder":  (-20, +30, +10, +10),
    "flutter":  (+15, +30, -10, +5),
    "simmer":   (-20, +25, +10, +10),
    "quake":    (-25, +40, -30, +20),
    "tremble":  (-25, +35, -30, +15),
    "shudder":  (-25, +35, -25, +15),

    # ── Action/state roots ──
    "clos":     (-10, +5, +5, +5),
    "open":     (+15, +15, +10, +5),
    "bind":     (-15, +10, -15, +10),
    "lock":     (-20, +15, -20, +15),
    "trap":     (-35, +30, -35, +20),
    "shield":   (+20, +15, +30, 0),
    "guard":    (+15, +15, +25, +5),
    "nourish":  (+30, +10, +20, 0),
    "starv":    (-40, +30, -30, +25),
    "wound":    (-35, +30, -25, +20),
    "mend":     (+25, +10, +20, +5),
    "wreck":    (-40, +35, -25, +25),
    "craft":    (+25, +15, +20, +5),
    "ruin":     (-45, +30, -30, +20),
    "bless":    (+45, +15, +20, 0),
    "curse":    (-40, +30, -15, +15),
    "prais":    (+40, +20, +20, 0),
    "mock":     (-30, +25, +15, +10),
    "scorn":    (-35, +25, +20, +10),
    "honor":    (+40, +20, +30, 0),
    "empower":  (+35, +25, +40, +5),
    "oppress":  (-40, +25, -40, +15),
    "liberat":  (+45, +30, +40, +5),
    "captiv":   (-30, +20, -35, +15),
    "erupt":    (-30, +55, +15, +30),
    "nurtur":   (+35, +10, +25, 0),
    "neglect":  (-35, +10, -30, +10),
    # "cherish" already defined above
    "devast":   (-55, +40, -35, +25),
    "tortur":   (-55, +50, -40, +30),
    "sancti":   (+30, +10, +25, 0),
    "corrupt":  (-40, +25, -15, +15),
    "purif":    (+30, +15, +20, 0),
    "pollut":   (-30, +15, -15, +15),
    "deplet":   (-25, +15, -20, +15),
    "restor":   (+30, +15, +20, +5),
    "demolish": (-40, +40, -20, +25),
    "construct": (+25, +20, +25, +10),
    "flourish": (+40, +25, +25, 0),
    "perish":   (-50, +30, -40, +25),
    "vanquish": (+20, +35, +40, +10),
    "relinquish": (-20, +10, -25, +5),
    "extinguish": (-25, +20, -15, +15),
    "distinguish": (+20, +15, +20, +5),

    # ── Everyday roots ──
    "cook":     (+20, +10, +15, 0),
    "eat":      (+15, +10, +10, 0),
    "drink":    (+10, +10, +5, 0),
    "sleep":    (+10, -20, +10, 0),
    "wake":     (+5, +20, +10, +5),
    "play":     (+35, +25, +15, 0),
    "sing":     (+30, +25, +10, 0),
    "danc":     (+35, +30, +15, 0),
    "laugh":    (+45, +35, +20, 0),
    "cry":      (-35, +30, -20, +10),
    "scream":   (-20, +55, +10, +25),
    "whisper":  (+5, -20, -5, +5),
    "hug":      (+40, +15, +10, 0),
    "kiss":     (+45, +25, +10, 0),
    "slap":     (-30, +40, +15, +15),
    "punch":    (-30, +50, +25, +20),
    "kick":     (-25, +40, +20, +15),
    "push":     (-15, +25, +15, +10),
    "pull":     (-10, +20, +10, +10),
    "grab":     (-10, +25, +15, +15),
    "throw":    (-15, +35, +15, +15),
    "catch":    (+10, +20, +15, +10),
    "climb":    (+15, +25, +20, +10),
    "swim":     (+15, +20, +15, 0),
    "drive":    (+10, +15, +15, +5),
    "crash":    (-40, +45, -20, +35),
    "collid":   (-35, +45, -15, +30),
    "explor":   (+25, +25, +20, +5),
    "wander":   (+10, +10, -5, 0),
    "search":   (+5, +20, +10, +15),
    "find":     (+25, +20, +15, +5),
    "discov":   (+30, +30, +20, +5),
    "invent":   (+30, +25, +25, +5),
    "teach":    (+25, +15, +20, +5),
    "studi":    (+10, +15, +10, +10),
    "read":     (+15, +10, +10, 0),
    "write":    (+15, +15, +15, +5),
    "draw":     (+20, +15, +10, 0),
    "paint":    (+20, +15, +10, 0),
    "sculpt":   (+20, +15, +15, 0),
    "design":   (+20, +20, +20, +5),
    # "fix" already defined above
    "repair":   (+15, +15, +20, +10),
    "clean":    (+15, +10, +15, 0),
    "organiz":  (+15, +10, +20, +5),
}


def decompose_word(word: str) -> dict:
    """Decompose a word into prefix + root + suffix with emotional weights.

    Returns a dict with:
      prefix: str or None
      root: str
      suffix: str or None
      v, a, d, u, g: computed emotional values (-100 to +100 scale)
      trace: list of computation steps
    """
    word = word.lower().strip()
    trace = []
    found_prefix = None
    found_suffix = None
    found_root = None
    remainder = word

    # Try to find a direct root match first (whole word)
    if word in ROOTS:
        root_vals = ROOTS[word]
        v, a, d, u = root_vals[0], root_vals[1], root_vals[2], root_vals[3]
        g = root_vals[4] if len(root_vals) > 4 else 0
        trace.append(f"    ROOT '{word}' \u2192 V{v:+} A{a:+} D{d:+} U{u:+} G{g:+}")
        return {
            "word": word, "prefix": None, "root": word, "suffix": None,
            "v": v, "a": a, "d": d, "u": u, "g": g, "trace": trace, "found": True
        }

    # Try prefix stripping (longest match first)
    for prefix in sorted(PREFIXES.keys(), key=len, reverse=True):
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            remainder = word[len(prefix):]
            found_prefix = prefix
            trace.append(f"    PREFIX '{prefix}' detected")
            break

    # Try suffix stripping (longest match first)
    working = remainder
    for suffix in sorted(SUFFIXES.keys(), key=len, reverse=True):
        if working.endswith(suffix) and len(working) > len(suffix) + 2:
            working = working[:-len(suffix)]
            found_suffix = suffix
            trace.append(f"    SUFFIX '{suffix}' detected")
            break

    # Try to match the remaining stem against roots
    # Try exact match first, then partial match (root is a prefix of stem)
    best_root = None
    best_root_key = None
    for root_key in sorted(ROOTS.keys(), key=len, reverse=True):
        if working.startswith(root_key) or root_key.startswith(working):
            best_root = ROOTS[root_key]
            best_root_key = root_key
            found_root = root_key
            _g = best_root[4] if len(best_root) > 4 else 0
            trace.append(f"    ROOT '{root_key}' \u2192 V{best_root[0]:+} A{best_root[1]:+} D{best_root[2]:+} U{best_root[3]:+} G{_g:+}")
            break

    if best_root is None:
        # No root found — return unknown
        trace.append(f"    No morpheme match for '{word}'")
        return {
            "word": word, "prefix": found_prefix, "root": working,
            "suffix": found_suffix, "v": 0, "a": 0, "d": 0, "u": 0, "g": 0,
            "trace": trace, "found": False
        }

    # Start with root values (handle both 4-tuple and 5-tuple roots)
    v, a, d, u = best_root[0], best_root[1], best_root[2], best_root[3]
    g = best_root[4] if len(best_root) > 4 else 0

    # Apply prefix modifier
    if found_prefix and found_prefix in PREFIXES:
        p = PREFIXES[found_prefix]
        if p["rule"] == "negate":
            old_v = v
            v = int(v * p["v"])  # flip valence
            d = int(d + d * p["d"])
            a = int(a + abs(a) * p["a"])
            u = int(u + 10 * p["u"])
            g = int(g + g * p.get("g", 0))  # negate flips gravity direction
            trace.append(f"    PREFIX '{found_prefix}' NEGATES: V{old_v:+} \u2192 V{v:+}, D adjusted, G adjusted")
        elif p["rule"] == "amplify":
            v = int(v + v * p["v"])
            a = int(a + abs(a) * p["a"])
            d = int(d + d * p["d"])
            u = int(u + 10 * p["u"])
            g = int(g + g * p.get("g", 0))
            trace.append(f"    PREFIX '{found_prefix}' AMPLIFIES")
        elif p["rule"] in ("reduce", "modify"):
            v = int(v + 10 * p["v"])
            a = int(a + 10 * p["a"])
            d = int(d + 10 * p["d"])
            u = int(u + 10 * p["u"])
            g = int(g + 10 * p.get("g", 0))
            trace.append(f"    PREFIX '{found_prefix}' MODIFIES")

    # Apply suffix modifier
    if found_suffix and found_suffix in SUFFIXES:
        s = SUFFIXES[found_suffix]
        if s["rule"] == "negate":
            old_v = v
            v = int(v * s["v"])
            d = int(d + d * s["d"])
            a = int(a + abs(a) * s["a"])
            g = int(g + g * s.get("g", 0))  # -less makes things heavier
            trace.append(f"    SUFFIX '{found_suffix}' NEGATES: V{old_v:+} \u2192 V{v:+}")
        elif s["rule"] == "amplify":
            v = int(v + v * s["v"])
            a = int(a + abs(a) * s["a"])
            d = int(d + d * s["d"])
            g = int(g + g * s.get("g", 0))  # -ful makes things lighter
            trace.append(f"    SUFFIX '{found_suffix}' AMPLIFIES")
        elif s["rule"] in ("reduce", "modify"):
            v = int(v + 10 * s["v"])
            a = int(a + 10 * s["a"])
            d = int(d + 10 * s["d"])
            g = int(g + 10 * s.get("g", 0))
            trace.append(f"    SUFFIX '{found_suffix}' MODIFIES")

    # Clamp to -100..+100
    v = max(-100, min(100, v))
    a = max(-100, min(100, a))
    d = max(-100, min(100, d))
    u = max(-100, min(100, u))
    g = max(-100, min(100, g))

    composite = f"{'un' if found_prefix else ''}{found_root or working}{'less' if found_suffix else ''}"
    trace.append(f"    FINAL: V{v:+} A{a:+} D{d:+} U{u:+} G{g:+}")

    return {
        "word": word, "prefix": found_prefix, "root": found_root or working,
        "suffix": found_suffix, "v": v, "a": a, "d": d, "u": u, "g": g,
        "trace": trace, "found": True
    }


def test_decomposition():
    """Test the morphological decomposition on various words."""
    test_words = [
        "hopeless", "hopeful", "unhappy", "happiness", "desperate",
        "uncomfortable", "misunderstand", "powerless", "powerful",
        "joyful", "joyless", "fearless", "fearful",
        "worthless", "useless", "wonderful", "beautiful",
        "unbreakable", "overjoyed", "discourage", "rekindle",
    ]

    for word in test_words:
        result = decompose_word(word)
        parts = []
        if result["prefix"]:
            parts.append(f"[{result['prefix']}-]")
        parts.append(f"{result['root']}")
        if result["suffix"]:
            parts.append(f"[-{result['suffix']}]")

        print(f"\n  {word}: {' + '.join(parts)}")
        for t in result["trace"]:
            print(t)
        print(f"    \u2192 V{result['v']:+} A{result['a']:+} D{result['d']:+} U{result['u']:+} G{result['g']:+}")


if __name__ == "__main__":
    test_decomposition()
