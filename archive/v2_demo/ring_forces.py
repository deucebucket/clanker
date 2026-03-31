"""Multi-ring word forces — same word, different emotional values by tone.

Words like "great" mean opposite things depending on tonal context:
- "Great job!" (sincere) = positive  -> use default WORD_FORCES
- "Oh great, another meeting" (sarcastic) = negative  -> use SARCASM_FORCES

Three rings:
- SARCASM_FORCES: words that FLIP meaning when tone is sarcastic
- COMEDY_FORCES: death-family words that mean humor in casual context
- HYPERBOLE_FORCES: words that soften when exaggerated
"""

# Sarcasm ring: words that FLIP meaning when tone is sarcastic
SARCASM_FORCES = {
    "great": (-25, +15, -10, +5, -10),
    "wonderful": (-25, +10, -10, +5, -10),
    "fantastic": (-25, +15, -10, +5, -10),
    "amazing": (-20, +15, -5, +5, -5),
    "lovely": (-20, +10, -5, +5, -5),
    "brilliant": (-25, +15, -10, +5, -10),
    "perfect": (-25, +10, -10, +5, -10),
    "awesome": (-20, +15, -5, +5, -5),
    "excellent": (-25, +10, -10, +5, -10),
    "terrific": (-20, +15, -5, +5, -5),
    "sure": (-15, +10, +10, +5, +5),
    "thanks": (-15, +10, -5, +5, -5),
    "nice": (-15, +10, -5, +5, -5),
    "love": (-20, +15, -5, +5, -5),
    "thrilled": (-25, +15, -10, +5, -10),
    "delighted": (-20, +10, -5, +5, -5),
}

# Comedy/hyperbole ring: death-family words that mean humor in casual context
COMEDY_FORCES = {
    "death": (+15, +25, +5, 0, +10),
    "dying": (+20, +30, +10, 0, +15),
    "dead": (+25, +25, +5, 0, +15),
    "kill": (+20, +25, +10, 0, +10),
    "killing": (+25, +25, +15, 0, +10),
    "slaying": (+30, +25, +15, 0, +15),
    "murder": (+15, +20, +5, 0, +5),
    "destroyed": (+20, +25, +10, 0, +10),
    "obliterated": (+20, +25, +10, 0, +10),
}

# Hyperbole ring: words that soften when exaggerated
HYPERBOLE_FORCES = {
    "worst": (-20, +15, -5, +5, -5),
    "best": (+20, +15, +5, 0, +10),
    "ever": (+5, +10, 0, 0, +5),
    "literally": (+0, +10, +0, +5, +0),
    "never": (-15, +10, -5, +5, -5),
    "always": (+5, +10, 0, +5, 0),
}


def get_ring_force(word: str, tone: str) -> tuple | None:
    """Get force override for a word based on detected tone.

    Args:
        word: lowercase word to look up
        tone: detected tone — "sarcastic", "hyperbolic", or anything else

    Returns:
        The ring-specific force tuple (dv, da, dd, du, dg), or None to use
        default WORD_FORCES.
    """
    if tone == "sarcastic" and word in SARCASM_FORCES:
        return SARCASM_FORCES[word]
    if tone == "sarcastic" and word in COMEDY_FORCES:
        return COMEDY_FORCES[word]
    if tone == "hyperbolic" and word in HYPERBOLE_FORCES:
        return HYPERBOLE_FORCES[word]
    return None
