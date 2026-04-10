"""V9 Tokenizer — splits text into atoms, resolves compound bonds.

Compound bonds are multi-word phrases that form a single emotional atom.
"laid off" → "laidoff" (one atom with its own root charge).
Resolved BEFORE root mapping so the mapper sees molecular roots.
"""

import re

# Trigram compounds (check FIRST — longer match wins)
COMPOUND_BONDS_TRI = {
    ("got", "laid", "off"): "laidoff",
    ("get", "laid", "off"): "laidoff",
    ("got", "kicked", "out"): "kickedout",
    ("got", "locked", "out"): "lockedout",
    ("got", "ripped", "off"): "rippedoff",
    ("got", "thrown", "out"): "thrownout",
    ("got", "cut", "off"): "cutoff",
    ("got", "burned", "out"): "burnedout",
    ("got", "wiped", "out"): "wipedout",
}

# Bigram compounds
COMPOUND_BONDS = {
    ("laid", "off"): "laidoff",
    ("food", "poisoning"): "foodpoisoning",
    ("broke", "down"): "brokedown",
    ("locked", "out"): "lockedout",
    ("kicked", "out"): "kickedout",
    ("passed", "away"): "passedaway",
    ("cut", "off"): "cutoff",
    ("thrown", "out"): "thrownout",
    ("ripped", "off"): "rippedoff",
    ("wiped", "out"): "wipedout",
    ("burned", "out"): "burnedout",
    ("shut", "down"): "shutdown",
    ("backed", "out"): "backedout",
    ("dropped", "out"): "droppedout",
    ("sold", "out"): "soldout",
    ("stressed", "out"): "stressedout",
    ("freaked", "out"): "freakedout",
    ("cancer", "free"): "cancerfree",
    ("debt", "free"): "debtfree",
    ("pain", "free"): "painfree",
    ("pulled", "off"): "pulledoff",
    ("pulled", "through"): "pulledthrough",
    ("worked", "out"): "workedout",
    ("paid", "off"): "paidoff",
    ("turned", "around"): "turnedaround",
    ("broke", "up"): "brokeup",
}

# Special pairs
SPECIAL_PAIRS = {
    ("no", "one"): "nobody",
    ("no", "cap"): "nocap",
    ("come", "on"): "comeon",
}


def _clean(word):
    """Lowercase and strip punctuation."""
    return re.sub(r"[^\w']+", "", word.lower()).replace("'", "")


def tokenize(text):
    """Tokenize text into atoms with compound bond resolution.

    Returns list of cleaned, lowercased tokens with compounds fused.
    """
    if not text or not text.strip():
        return []

    raw = text.split()
    words = [_clean(w) for w in raw if _clean(w)]

    if not words:
        return []

    # Resolve compounds: try trigrams first, then bigrams, then singles
    result = []
    i = 0
    while i < len(words):
        matched = False

        # Try trigram
        if i + 2 < len(words):
            tri = (words[i], words[i + 1], words[i + 2])
            if tri in COMPOUND_BONDS_TRI:
                result.append(COMPOUND_BONDS_TRI[tri])
                i += 3
                matched = True

        # Try bigram
        if not matched and i + 1 < len(words):
            bi = (words[i], words[i + 1])
            if bi in COMPOUND_BONDS:
                result.append(COMPOUND_BONDS[bi])
                i += 2
                matched = True
            elif bi in SPECIAL_PAIRS:
                result.append(SPECIAL_PAIRS[bi])
                i += 2
                matched = True

        if not matched:
            result.append(words[i])
            i += 1

    return result
