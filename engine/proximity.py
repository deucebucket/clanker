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

PROXIMITY_DECAY = 0.90   # champion v2: wide influence range
INFLUENCE_CUTOFF = 0.1   # ignore influence below this
COEFFICIENT_CAP = 2.63   # champion v2: tighter cap prevents runaway


# ── Role modifier strengths (champion v2, genetically tuned 2026-04-03) ──

ROLE_MODIFIERS = {
    "AMPLIFIER": 0.965,   # champion v2: amplifiers hit hard
    "NEGATOR": -2.464,    # champion v2: strong negation flip
    "SELF_REF": 0.466,    # champion v2: self-reference personalizes
    "HEDGE": -0.416,      # champion v2: hedges dampen more than v1
    "COMPRESSOR": -0.301, # champion v2: compression
    "REGISTER_CASUAL": 0.0,  # SOLVENT: phase-dependent handling in modifier block
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
        distance = abs(i - target_idx)
        influence = PROXIMITY_DECAY ** distance
        if influence < INFLUENCE_CUTOFF:
            continue

        # Operator modifiers (amplifier, negator, self-ref, hedge)
        if role in ROLE_MODIFIERS:
            modifier = ROLE_MODIFIERS[role]
            # Some words resist negation:
            # EXPLETIVES: "no fuck you" stays negative
            # DECEPTION verbs: "pretended not to" = the "not" belongs to the next verb
            #   "not pretended" doesn't make sense. The deception is real.
            _NEGATION_RESISTANT = {
                "fuck", "shit", "damn", "hell", "ass", "bitch",
                "bastard", "crap", "dick", "piss",
                "pretended", "pretending", "faked", "faking", "lied", "lying",
            }
            if role == "NEGATOR" and roles[target_idx].force:
                if roles[target_idx].word in _NEGATION_RESISTANT:
                    modifier *= 0.15  # barely any negation
            # Compressor dome doesn't touch people — only values/emotions.
            # "only" + self = neutral isolation. The rest of the sentence
            # decides if that isolation is proud or lonely.
            _PERSON_ROLES = {"SELF_REF", "OTHER_REF", "RELATION_REF"}
            if role == "COMPRESSOR" and roles[target_idx].role in _PERSON_ROLES:
                modifier = 0.0  # dome passes through people
            # SOLVENT: REGISTER_CASUAL dissolves LIQUID atoms, can't dissolve SOLID
            # "bruh im crying" → crying flips to positive
            # "bruh he got murdered" → murdered stays negative
            if role == "REGISTER_CASUAL" and roles[target_idx].force:
                from .phase import get_phase
                phase = get_phase(roles[target_idx].word)
                target_dv = roles[target_idx].force[0]
                if phase == "LIQUID" and target_dv < 0:
                    # Dissolve NEGATIVE liquid → flip to positive
                    modifier = -2.0 * influence
                elif phase == "LIQUID" and target_dv > 0:
                    # POSITIVE liquid near solvent → amplify (already positive)
                    modifier = 0.4 * influence
                elif phase == "SOLID":
                    modifier = 0.0  # can't dissolve rock
                else:  # GAS
                    modifier = 0.3 * influence
            coeff *= (1.0 + modifier * influence)

        # Star-to-star gravity: stronger emotional words pull weaker ones
        # Uses EMOTIONAL DISTANCE (skip neutral/connector words between stars)
        # "cheated on me with my best" -- emotional distance cheated->best = 1
        # Connectors/neutrals are transparent conduits, not walls
        if role == "EMOTIONAL" and roles[i].force and roles[target_idx].force:
            their_v = roles[i].force[0]
            my_v = roles[target_idx].force[0]
            if abs(their_v) > abs(my_v) * 1.5:
                # Count only EMOTIONAL words between them for distance
                lo, hi = min(i, target_idx), max(i, target_idx)
                emo_between = sum(1 for k in range(lo+1, hi) 
                                  if roles[k].role == "EMOTIONAL")
                emo_distance = max(1, emo_between + 1)
                emo_influence = PROXIMITY_DECAY ** emo_distance
                mass_ratio = abs(their_v) / max(abs(my_v), 1)
                pull = emo_influence * min(mass_ratio * 0.15, 0.8)
                if their_v < 0:
                    coeff *= (1.0 - pull)
                else:
                    coeff *= (1.0 + pull * 0.5)

    # Relationship amplification: nearby RELATION_REF amplifies negative forces
    # Wife(G=40) near cheated(-127) = betrayal hits harder because trust was higher
    # The relationship G value IS the trust level -- higher trust = bigger fall
    #
    # Determiner check: "my mother" = possessive bond (full G).
    # "the mother" = distanced/clinical (dampened G). The article severs
    # the gravitational bond between speaker and relationship.
    _ARTICLES = {"the", "a", "an"}
    target_role = roles[target_idx]
    if target_role.force and target_role.force[0] < -20:  # negative emotional word
        for i in range(n):
            if i == target_idx:
                continue
            if roles[i].role == "RELATION_REF":
                distance = abs(i - target_idx)
                influence = PROXIMITY_DECAY ** distance
                if influence < INFLUENCE_CUTOFF:
                    continue
                # Get the relationship G value from vocabulary
                from .vocabulary import VOCABULARY
                rel_g = 20  # default
                if roles[i].word in VOCABULARY:
                    rel_g = max(5, VOCABULARY[roles[i].word][4])
                # "the/a mother" = distanced, dampen G contribution
                # "my mother" = bonded, full G
                if i > 0 and roles[i - 1].word in _ARTICLES:
                    rel_g = int(rel_g * 0.3)  # 70% reduction -- article severs bond
                # Amplify: higher relationship G = bigger betrayal multiplier
                betrayal_mult = (rel_g / 20.0) * influence
                coeff *= (1.0 + betrayal_mult * 0.3)

    return max(-COEFFICIENT_CAP, min(COEFFICIENT_CAP, coeff))
