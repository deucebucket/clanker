"""V3 Layer 2: Proximity Weighting — distance-based influence fields.

Words are like celestial bodies — they have mass (emotional weight).
Placing them near each other creates gravitational effects. "give" near
"dog" near "neighbor" creates a farewell field. "give" far from "dog"
= probably unrelated.

The proximity field computes how much each word influences every other
word based on DISTANCE. Closer = stronger. Influence decays exponentially:
  influence = PROXIMITY_DECAY ^ distance

This layer answers: "Which words are pulling on this word, and how hard?"
"""

from typing import Dict, List, Tuple

from .word_classifier import WordRole


# ── Constants ────────────────────────────────────────────────────

PROXIMITY_DECAY = 0.7  # influence drops 30% per word of distance
INFLUENCE_CUTOFF = 0.1  # ignore influence below this (~5 words away)
COEFFICIENT_CAP = 3.0   # max absolute coefficient value


# ── Role modifier strengths ──────────────────────────────────────

ROLE_MODIFIERS = {
    "AMPLIFIER": 0.4,    # boost: coeff *= (1.0 + 0.4 * influence)
    "NEGATOR": -1.6,     # flip:  coeff *= (1.0 + (-1.6) * influence)
    "SELF_REF": 0.3,     # personalize: coeff *= (1.0 + 0.3 * influence)
    "HEDGE": -0.3,       # dampen: coeff *= (1.0 + (-0.3) * influence)
}


# ── Core functions ───────────────────────────────────────────────

def compute_proximity_field(
    roles: List[WordRole],
) -> Dict[int, Dict[int, float]]:
    """Compute influence of every word on every other word.

    Returns {word_idx: {other_idx: influence_strength}} where influence
    is PROXIMITY_DECAY ^ distance. Only includes pairs with influence
    above INFLUENCE_CUTOFF.
    """
    n = len(roles)
    field: Dict[int, Dict[int, float]] = {}

    for i in range(n):
        influences: Dict[int, float] = {}
        for j in range(n):
            if i == j:
                continue
            distance = abs(i - j)
            influence = PROXIMITY_DECAY ** distance
            if influence >= INFLUENCE_CUTOFF:
                influences[j] = influence
        field[i] = influences

    return field


def find_role_pairs(
    roles: List[WordRole],
    role_a: str,
    role_b: str,
    max_distance: int = 5,
) -> List[Tuple[int, int, float]]:
    """Find all pairs of specific roles within proximity.

    Returns [(idx_a, idx_b, proximity_strength)] sorted by strength
    descending (strongest first).
    """
    indices_a = [r.position for r in roles if r.role == role_a]
    indices_b = [r.position for r in roles if r.role == role_b]

    pairs: List[Tuple[int, int, float]] = []
    for ia in indices_a:
        for ib in indices_b:
            distance = abs(ia - ib)
            if distance == 0 or distance > max_distance:
                continue
            strength = PROXIMITY_DECAY ** distance
            pairs.append((ia, ib, strength))

    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def proximity_coefficient(
    roles: List[WordRole],
    target_idx: int,
) -> float:
    """Compute combined proximity coefficient for a word.

    Nearby modifier roles (AMPLIFIER, NEGATOR, SELF_REF, HEDGE) adjust
    the coefficient multiplicatively. The result is capped to
    [-COEFFICIENT_CAP, +COEFFICIENT_CAP].

    Returns a float coefficient (1.0 = no modification).
    """
    if not roles or target_idx < 0 or target_idx >= len(roles):
        return 1.0

    coeff = 1.0
    n = len(roles)

    for i in range(n):
        if i == target_idx:
            continue

        role = roles[i].role
        if role not in ROLE_MODIFIERS:
            continue

        distance = abs(i - target_idx)
        influence = PROXIMITY_DECAY ** distance
        if influence < INFLUENCE_CUTOFF:
            continue

        modifier = ROLE_MODIFIERS[role]
        coeff *= (1.0 + modifier * influence)

    return max(-COEFFICIENT_CAP, min(COEFFICIENT_CAP, coeff))
