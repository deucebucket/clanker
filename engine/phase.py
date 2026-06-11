"""Phase system: matter states for emotional atoms.

Words have phases that determine how context affects them:
  SOLID  — meaning never changes. "Murder" is always negative.
  LIQUID — context can flip meaning. "Crying" = sad OR laughing hard.
  GAS    — no inherent meaning. "Jump" = whatever context says.

The SOLVENT role (REGISTER_CASUAL) changes the phase of nearby LIQUID
atoms without carrying its own charge. Like a catalyst in chemistry.

"bruh im crying" → SOLVENT(bruh) dissolves LIQUID(crying) → positive
"bruh he got murdered" → SOLVENT can't dissolve SOLID(murdered) → negative
"""

# ── Phase assignments ──
# Manual overrides for words where sense count doesn't match usage

SOLID_WORDS = frozenset({
    # These NEVER flip meaning regardless of context
    "murder", "murdered", "suicide", "rape", "raped", "torture",
    "cancer", "agony", "anguish", "abuse", "abused",
    "joy", "gratitude", "love", "ecstatic",
})

LIQUID_WORDS = frozenset({
    # These FLIP in casual/slang register
    "crying", "screaming", "dying", "dead", "died",
    "stupid", "insane", "crazy", "wild", "sick",
    "fire", "killer", "killed", "killing",
    "slaps", "hits", "bangs", "hard",
    "destroyed", "wrecked", "crushed", "smashed",
    "buried", "devoured", "annihilated",
    "broke", "broken", "shattered",
    "bleeding", "choking", "gagging",
    "shook", "nuts", "mental", "unreal",
    # Two-faced modern slang (2026-06 re-rating): doom-or-praise depending
    # on register. "im cooked" = doomed, "he cooked" (casual) = excelled.
    # "sheesh" = exasperation OR awe. "snapped" = lost temper OR performed
    # brilliantly. Stored charge is the safer (negative) reading; SOLVENT
    # dissolution flips it in casual register.
    "cooked", "sheesh", "snapped",
})

# Neutralizer demotions (2026-06-11, datasets/bias_audit.json):
# Words whose LLM-rated isolated charge contradicted sentence-level labels
# beyond corpus baseline by > 0.20 with n >= 10 sentences of evidence.
# Demoted GAS -> LIQUID: the (already attenuated) stored force is a
# *presumed* charge — context machinery (SOLVENT, negation, sarcasm
# contrast) may dissolve or flip it instead of it always applying.
# dV attenuation (x (1 - contradiction_rate)) lives in forces_curated.py.
NEUTRALIZED_LIQUID = frozenset({
    "adore", "cherish", "congratulations", "overwhelmed", "fantastic",
    "survive", "thrilled", "thank", "breaking",
    "secret", "monster", "response", "truly", "delighted",
    "enjoy", "overjoyed", "major", "project", "hope",
    # "wonderful" dropped as conflicted — see forces_curated.py neutralizer note.
})

# SOLVENT words: zero charge, dissolve liquid atoms
SOLVENT_WORDS = frozenset({
    "bruh", "bro", "dude", "fam", "bestie",
    "lol", "lmao", "lmfao", "rofl", "haha", "hahaha",
    "omg", "istg", "fr", "ngl", "tbh",
    "lowkey", "highkey", "deadass",
    "literally", "actually",
    "nocap",
})


def get_phase(word: str) -> str:
    """Get the matter state of a word.

    Returns: "SOLID", "LIQUID", or "GAS"
    """
    w = word.lower()
    if w in SOLID_WORDS:
        return "SOLID"
    if w in LIQUID_WORDS or w in NEUTRALIZED_LIQUID:
        return "LIQUID"
    # Default: GAS (context fills it)
    return "GAS"


def is_solvent(word: str) -> bool:
    """Check if a word is a SOLVENT (register-casual marker)."""
    return word.lower() in SOLVENT_WORDS
