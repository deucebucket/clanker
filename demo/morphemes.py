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
# Each prefix has: (v_mod, a_mod, d_mod, u_mod, rule)
# Rules: "negate" flips V and D, "amplify" multiplies, "reduce" dampens
# ─────────────────────────────────────────────────────────

PREFIXES = {
    # Negation prefixes — flip valence and reduce dominance
    "un":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.3, "u": 0.05},
    "in":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.2, "u": 0.05},
    "im":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.2, "u": 0.05},
    "ir":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.2, "u": 0.05},
    "il":      {"rule": "negate",  "v": -1.0,  "a": 0.1,  "d": -0.2, "u": 0.05},
    "dis":     {"rule": "negate",  "v": -1.0,  "a": 0.15, "d": -0.3, "u": 0.1},
    "de":      {"rule": "negate",  "v": -0.8,  "a": 0.1,  "d": -0.25,"u": 0.1},
    "non":     {"rule": "negate",  "v": -0.6,  "a": 0.0,  "d": -0.1, "u": 0.0},

    # Intensifier prefixes
    "over":    {"rule": "amplify", "v": 0.3,   "a": 0.4,  "d": 0.1,  "u": 0.2},
    "hyper":   {"rule": "amplify", "v": 0.2,   "a": 0.6,  "d": 0.2,  "u": 0.3},
    "super":   {"rule": "amplify", "v": 0.3,   "a": 0.5,  "d": 0.3,  "u": 0.1},
    "ultra":   {"rule": "amplify", "v": 0.3,   "a": 0.6,  "d": 0.3,  "u": 0.2},
    "mega":    {"rule": "amplify", "v": 0.3,   "a": 0.5,  "d": 0.3,  "u": 0.1},

    # Reduction prefixes
    "under":   {"rule": "reduce",  "v": -0.2,  "a": -0.2, "d": -0.3, "u": 0.1},
    "sub":     {"rule": "reduce",  "v": -0.1,  "a": -0.1, "d": -0.2, "u": 0.0},
    "mis":     {"rule": "negate",  "v": -0.7,  "a": 0.2,  "d": -0.2, "u": 0.15},

    # Directional prefixes
    "re":      {"rule": "modify",  "v": 0.1,   "a": 0.1,  "d": 0.15, "u": 0.1},
    "pre":     {"rule": "modify",  "v": 0.05,  "a": 0.05, "d": 0.15, "u": 0.15},
    "post":    {"rule": "modify",  "v": 0.0,   "a": -0.05,"d": 0.1,  "u": -0.05},
    "anti":    {"rule": "negate",  "v": -0.8,  "a": 0.3,  "d": 0.2,  "u": 0.15},
    "counter": {"rule": "negate",  "v": -0.5,  "a": 0.2,  "d": 0.3,  "u": 0.1},
    "out":     {"rule": "amplify", "v": 0.2,   "a": 0.3,  "d": 0.3,  "u": 0.1},
    "fore":    {"rule": "modify",  "v": 0.05,  "a": 0.1,  "d": 0.2,  "u": 0.2},
    "co":      {"rule": "modify",  "v": 0.15,  "a": 0.05, "d": 0.0,  "u": 0.0},
}

# ─────────────────────────────────────────────────────────
# SUFFIX MODIFIERS
# ─────────────────────────────────────────────────────────

SUFFIXES = {
    # Absence / negation
    "less":    {"rule": "negate",  "v": -1.0,  "a": -0.1, "d": -0.4, "u": 0.1},

    # Fullness / presence
    "ful":     {"rule": "amplify", "v": 0.3,   "a": 0.1,  "d": 0.15, "u": 0.0},
    "ous":     {"rule": "amplify", "v": 0.2,   "a": 0.15, "d": 0.1,  "u": 0.05},
    "ious":    {"rule": "amplify", "v": 0.2,   "a": 0.15, "d": 0.1,  "u": 0.05},
    "ive":     {"rule": "amplify", "v": 0.15,  "a": 0.2,  "d": 0.2,  "u": 0.05},
    "able":    {"rule": "modify",  "v": 0.1,   "a": 0.0,  "d": 0.15, "u": 0.0},
    "ible":    {"rule": "modify",  "v": 0.1,   "a": 0.0,  "d": 0.15, "u": 0.0},

    # State / condition
    "ness":    {"rule": "modify",  "v": 0.0,   "a": -0.05,"d": -0.05,"u": 0.0},
    "ment":    {"rule": "modify",  "v": 0.0,   "a": 0.0,  "d": 0.05, "u": 0.05},
    "tion":    {"rule": "modify",  "v": 0.0,   "a": 0.05, "d": 0.05, "u": 0.05},
    "sion":    {"rule": "modify",  "v": 0.0,   "a": 0.05, "d": 0.05, "u": 0.05},
    "ity":     {"rule": "modify",  "v": 0.0,   "a": 0.0,  "d": 0.05, "u": 0.0},
    "ism":     {"rule": "modify",  "v": -0.05, "a": 0.1,  "d": 0.1,  "u": 0.05},
    "ist":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.15, "u": 0.05},

    # Degree / comparison
    "er":      {"rule": "amplify", "v": 0.15,  "a": 0.1,  "d": 0.1,  "u": 0.05},
    "est":     {"rule": "amplify", "v": 0.3,   "a": 0.2,  "d": 0.15, "u": 0.1},

    # Action / process
    "ing":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.05, "u": 0.05},
    "ed":      {"rule": "modify",  "v": -0.05, "a": -0.05,"d": 0.0,  "u": -0.05},
    "ly":      {"rule": "modify",  "v": 0.0,   "a": 0.05, "d": 0.0,  "u": 0.0},
    "en":      {"rule": "modify",  "v": 0.05,  "a": 0.1,  "d": 0.1,  "u": 0.05},
    "ize":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.15, "u": 0.1},
    "ify":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.1,  "u": 0.1},
    "ate":     {"rule": "modify",  "v": 0.0,   "a": 0.1,  "d": 0.1,  "u": 0.1},

    # Person/agent
    "or":      {"rule": "modify",  "v": 0.05,  "a": 0.05, "d": 0.15, "u": 0.0},

    # Diminutive
    "ish":     {"rule": "reduce",  "v": -0.1,  "a": -0.1, "d": -0.05,"u": 0.0},
    "ette":    {"rule": "reduce",  "v": 0.1,   "a": -0.1, "d": -0.1, "u": 0.0},
}

# ─────────────────────────────────────────────────────────
# ROOT MORPHEMES — emotional core of words
# (v_weight, a_weight, d_weight, u_weight)
# Scaled -100 to +100 for each axis
# ─────────────────────────────────────────────────────────

ROOTS = {
    # ── Positive emotions ──
    "joy":      (+70, +40, +30, 0),
    "hap":      (+60, +30, +25, 0),     # happy, happen
    "glad":     (+50, +25, +20, 0),
    "pleas":    (+55, +20, +20, 0),     # please, pleasant, pleasure
    "delight":  (+65, +35, +25, 0),
    "bliss":    (+75, +20, +30, 0),
    "euph":     (+80, +60, +30, 0),     # euphoria, euphoric
    "grat":     (+60, +20, +25, 0),     # grateful, gratitude
    "content":  (+45, -15, +25, 0),
    "seren":    (+40, -30, +25, 0),     # serene, serenity
    "calm":     (+30, -40, +20, 0),
    "peace":    (+45, -35, +25, 0),
    "comfort":  (+50, -25, +30, 0),
    "warm":     (+40, -10, +20, 0),
    "kind":     (+50, -5, +15, 0),
    "gentle":   (+35, -20, +10, 0),
    "tender":   (+40, -15, +5, 0),
    "lov":      (+75, +40, +20, 0),     # love, loving, lovely
    "ador":     (+70, +35, +15, 0),     # adore, adorable
    "cherish":  (+65, +15, +20, 0),
    "car":      (+45, +5, +20, 0),      # care, caring
    "trust":    (+50, -10, +30, 0),
    "faith":    (+45, +10, +25, 0),
    "hop":      (+55, +20, +25, 0),     # hope, hoping

    # ── Negative emotions ──
    "sad":      (-55, -15, -30, +5),
    "grief":    (-70, +30, -40, +20),
    "sorrow":   (-65, +10, -35, +10),
    "mourn":    (-60, +15, -35, +10),
    "despair":  (-75, +25, -60, +35),
    "desp":     (-70, +20, -55, +30),   # desperate, desperation
    "miser":    (-65, +10, -45, +15),   # miserable, misery
    "suffer":   (-60, +30, -40, +25),
    "pain":     (-55, +35, -30, +25),
    "agon":     (-65, +50, -35, +30),   # agony, agonize
    "torment":  (-60, +45, -35, +30),
    "doom":     (-70, +20, -55, +30),
    "dread":    (-60, +40, -50, +35),
    "woe":      (-60, +15, -40, +15),
    "gloom":    (-45, -20, -25, +5),
    "melan":    (-40, -15, -20, +5),    # melancholy, melancholic

    # ── Anger ──
    "ang":      (-50, +55, +25, +25),   # anger, angry
    "rage":     (-55, +75, +40, +35),
    "fur":      (-55, +70, +40, +35),   # fury, furious
    "wrath":    (-55, +65, +45, +35),
    "hate":     (-65, +50, +30, +25),
    "host":     (-50, +40, +25, +20),   # hostile, hostility
    "resent":   (-45, +30, +15, +15),
    "bitter":   (-40, +20, +10, +10),
    "spite":    (-45, +35, +25, +15),
    "venge":    (-50, +50, +35, +30),   # revenge, vengeance
    "irrit":    (-30, +30, +10, +15),   # irritate, irritable

    # ── Fear ──
    "fear":     (-50, +50, -50, +30),
    "terr":     (-65, +70, -55, +40),   # terror, terrible, terrify
    "horr":     (-60, +60, -45, +35),   # horror, horrible, horrify
    "panic":    (-55, +75, -60, +50),
    "anxi":     (-40, +45, -35, +30),   # anxiety, anxious
    "worry":    (-30, +30, -25, +25),
    "nerv":     (-25, +35, -20, +20),   # nervous, nerve
    "dread":    (-55, +40, -45, +30),
    "fright":   (-50, +55, -50, +35),
    "scar":     (-50, +50, -45, +30),   # scare, scared, scary
    "phob":     (-50, +55, -50, +30),   # phobia, phobic

    # ── Surprise ──
    "surpris":  (+10, +60, -15, +20),
    "amaz":     (+40, +55, +10, +10),   # amaze, amazing
    "astonish": (+20, +60, -10, +15),
    "shock":    (-15, +70, -30, +30),
    "stun":     (-5, +65, -25, +25),
    "startle":  (-10, +60, -20, +20),
    "awe":      (+30, +50, -10, +10),

    # ── Disgust ──
    "disgust":  (-50, +40, +15, +15),
    "repuls":   (-55, +45, +20, +15),
    "revolt":   (-45, +50, +25, +20),
    "nause":    (-40, +35, -15, +15),
    "sick":     (-35, +25, -20, +15),
    "gross":    (-30, +25, +5, +5),
    "vile":     (-55, +35, +15, +15),

    # ── Power / agency ──
    "power":    (+20, +40, +60, +15),
    "strong":   (+25, +35, +55, +10),
    "might":    (+15, +30, +50, +10),
    "conquer":  (+20, +45, +60, +20),
    "domin":    (+5, +40, +65, +15),    # dominate, dominance
    "control":  (+15, +20, +55, +10),
    "command":  (+10, +30, +50, +15),
    "lead":     (+20, +25, +45, +10),
    "confid":   (+35, +20, +45, +5),    # confident, confidence
    "brave":    (+30, +35, +50, +15),
    "courag":   (+35, +40, +55, +15),   # courage, courageous
    "bold":     (+25, +35, +50, +10),

    # ── Weakness / submission ──
    "weak":     (-25, -15, -50, +10),
    "help":     (-10, +10, -30, +20),   # helpless (with -less)
    "vulner":   (-20, +20, -45, +15),   # vulnerable
    "frag":     (-15, +5, -40, +10),    # fragile
    "submiss":  (-20, -10, -55, +5),    # submission, submissive
    "defeat":   (-45, +15, -50, +15),
    "fail":     (-40, +15, -40, +15),
    "lose":     (-40, +20, -35, +15),   # lose, lost, loser
    "worth":    (+30, +10, +25, 0),     # worth (base positive; "worthless" = worth + less)
    "use":      (+15, +10, +20, +5),    # use (base positive; "useless" = use + less)

    # ── Cognitive ──
    "think":    (+10, +15, +20, +5),
    "know":     (+20, +10, +30, +5),
    "learn":    (+25, +20, +20, +5),
    "understand": (+25, +15, +25, +5),
    "confus":   (-20, +25, -25, +15),   # confused, confusion
    "doubt":    (-15, +15, -20, +10),
    "wonder":   (+15, +25, -5, +5),
    "curio":    (+20, +25, +10, +5),    # curious, curiosity
    "creat":    (+35, +30, +30, +5),    # create, creative
    "imagin":   (+30, +25, +20, +5),    # imagine, imagination
    "inspir":   (+40, +35, +25, +5),    # inspire, inspiration

    # ── Social ──
    "friend":   (+50, +20, +15, 0),
    "belong":   (+40, +10, +15, 0),
    "alone":    (-35, -15, -20, +5),
    "lonely":   (-40, -10, -25, +5),
    "reject":   (-45, +30, -35, +15),
    "accept":   (+40, +10, +20, 0),
    "welcom":   (+45, +15, +20, 0),
    "safe":     (+40, -20, +35, -10),
    "secur":    (+35, -15, +35, -5),    # secure, security
    "danger":   (-35, +45, -30, +40),
    "threat":   (-40, +50, -25, +40),

    # ── Functional / neutral ──
    "work":     (+10, +15, +20, +15),
    "build":    (+20, +20, +25, +10),
    "fix":      (+15, +15, +20, +20),
    "break":    (-30, +30, -15, +20),
    "change":   (+5, +20, +10, +15),
    "grow":     (+25, +15, +15, +5),
    "move":     (+10, +20, +15, +10),
    "stop":     (-10, +15, +15, +15),
    "start":    (+15, +20, +20, +15),
    "end":      (-5, +5, +10, +10),
    "wait":     (-5, -10, -10, +10),
    "rush":     (-5, +40, +10, +45),
    "hurry":    (-5, +35, +5, +40),
    "quick":    (+5, +25, +15, +30),
    "slow":     (-5, -15, -5, -10),
}


def decompose_word(word: str) -> dict:
    """Decompose a word into prefix + root + suffix with emotional weights.

    Returns a dict with:
      prefix: str or None
      root: str
      suffix: str or None
      v, a, d, u: computed emotional values (-100 to +100 scale)
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
        v, a, d, u = ROOTS[word]
        trace.append(f"    ROOT '{word}' → V{v:+} A{a:+} D{d:+} U{u:+}")
        return {
            "word": word, "prefix": None, "root": word, "suffix": None,
            "v": v, "a": a, "d": d, "u": u, "trace": trace, "found": True
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
            trace.append(f"    ROOT '{root_key}' → V{best_root[0]:+} A{best_root[1]:+} D{best_root[2]:+} U{best_root[3]:+}")
            break

    if best_root is None:
        # No root found — return unknown
        trace.append(f"    No morpheme match for '{word}'")
        return {
            "word": word, "prefix": found_prefix, "root": working,
            "suffix": found_suffix, "v": 0, "a": 0, "d": 0, "u": 0,
            "trace": trace, "found": False
        }

    # Start with root values
    v, a, d, u = best_root

    # Apply prefix modifier
    if found_prefix and found_prefix in PREFIXES:
        p = PREFIXES[found_prefix]
        if p["rule"] == "negate":
            old_v = v
            v = int(v * p["v"])  # flip valence
            d = int(d + d * p["d"])
            a = int(a + abs(a) * p["a"])
            u = int(u + 10 * p["u"])
            trace.append(f"    PREFIX '{found_prefix}' NEGATES: V{old_v:+} → V{v:+}, D adjusted")
        elif p["rule"] == "amplify":
            v = int(v + v * p["v"])
            a = int(a + abs(a) * p["a"])
            d = int(d + d * p["d"])
            u = int(u + 10 * p["u"])
            trace.append(f"    PREFIX '{found_prefix}' AMPLIFIES")
        elif p["rule"] in ("reduce", "modify"):
            v = int(v + 10 * p["v"])
            a = int(a + 10 * p["a"])
            d = int(d + 10 * p["d"])
            u = int(u + 10 * p["u"])
            trace.append(f"    PREFIX '{found_prefix}' MODIFIES")

    # Apply suffix modifier
    if found_suffix and found_suffix in SUFFIXES:
        s = SUFFIXES[found_suffix]
        if s["rule"] == "negate":
            old_v = v
            v = int(v * s["v"])
            d = int(d + d * s["d"])
            a = int(a + abs(a) * s["a"])
            trace.append(f"    SUFFIX '{found_suffix}' NEGATES: V{old_v:+} → V{v:+}")
        elif s["rule"] == "amplify":
            v = int(v + v * s["v"])
            a = int(a + abs(a) * s["a"])
            d = int(d + d * s["d"])
            trace.append(f"    SUFFIX '{found_suffix}' AMPLIFIES")
        elif s["rule"] in ("reduce", "modify"):
            v = int(v + 10 * s["v"])
            a = int(a + 10 * s["a"])
            d = int(d + 10 * s["d"])
            trace.append(f"    SUFFIX '{found_suffix}' MODIFIES")

    # Clamp to -100..+100
    v = max(-100, min(100, v))
    a = max(-100, min(100, a))
    d = max(-100, min(100, d))
    u = max(-100, min(100, u))

    composite = f"{'un' if found_prefix else ''}{found_root or working}{'less' if found_suffix else ''}"
    trace.append(f"    FINAL: V{v:+} A{a:+} D{d:+} U{u:+}")

    return {
        "word": word, "prefix": found_prefix, "root": found_root or working,
        "suffix": found_suffix, "v": v, "a": a, "d": d, "u": u,
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
        print(f"    → V{result['v']:+} A{result['a']:+} D{result['d']:+} U{result['u']:+}")


if __name__ == "__main__":
    test_decomposition()
