"""
V9 Composer — Charge Resolution from Equation.

Takes a decomposed Equation and resolves it to a 7D VADUGWI coordinate.

Pipeline:
  1. Start at neutral state (center for all dims; U starts at 0).
  2. Extract nucleus charge, apply operator multipliers to it.
  3. Accumulate weighted nucleus charge (full weight).
  4. Accumulate context charges (reduced weight — color, don't drive).
  5. Apply subject modifier (self-reference amplifies deviation).
  6. Tanh saturation: prevents runaway values.
  7. Clamp to [0, 255] and return VADUG.
"""

from __future__ import annotations

import math
from typing import List

from engine_v9.shared import VADUG
from engine_v9.decomposer import Equation, Atom
from engine_v9.roots import RootCategory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CENTER = 128.0

FORCE_SCALE = 2.0          # optimizer gen50 champion (bal=0.622)
EVENT_WEIGHT = 1.0         # nucleus gets full weight
CONTEXT_WEIGHT = 0.53      # optimizer gen50
SATURATION = 97.0          # optimizer gen50 — tight compression

NEGATOR_FACTOR = -0.81     # optimizer: partial negation, not full flip
INTENSIFIER_FACTOR = 2.25  # optimizer: strong amplification
HEDGE_FACTOR = 0.44        # optimizer gen50
COMPRESSOR_FACTOR = 0.7    # compression

SUBJECT_SELF_BONUS = 1.0   # no self-amplification
LENGTH_EXPONENT = 0.2      # optimizer: gentle length scaling

# Neutral starting state — U starts at 0, not 128
_NEUTRAL_STATE = [128.0, 128.0, 128.0, 0.0, 128.0, 128.0, 128.0]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_operators(charge: tuple, operators: List[Atom]) -> List[float]:
    """
    Apply operator atoms to a charge vector.

    Builds a per-element multiplier from the operator list, then applies it.
    Negation flips sign; intensifier/hedge/compressor scale magnitude.

    Args:
        charge: 7-element tuple of int deltas.
        operators: List of operator Atoms.

    Returns:
        Modified charge as a list of floats.
    """
    multiplier = 1.0
    for op in operators:
        cat = op.root.category
        if cat == RootCategory.NEGATOR:
            multiplier *= NEGATOR_FACTOR
        elif cat == RootCategory.INTENSIFIER:
            multiplier *= INTENSIFIER_FACTOR
        elif cat == RootCategory.HEDGE:
            multiplier *= HEDGE_FACTOR
        elif cat == RootCategory.COMPRESSOR:
            multiplier *= COMPRESSOR_FACTOR

    return [c * multiplier for c in charge]


def _tanh_saturate(value: float, center: float, saturation: float) -> float:
    """
    Apply tanh saturation around a center point.

    Maps deviations from center through tanh so extreme forces
    compress rather than clip abruptly.

    Args:
        value: Current state value.
        center: Neutral center point.
        saturation: Compression range (half-width at ~76% saturation).

    Returns:
        Saturated value, centered around center.
    """
    deviation = value - center
    return center + saturation * math.tanh(deviation / saturation)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose(equation: Equation) -> VADUG:
    """
    Resolve a decomposed Equation to a 7D VADUGWI coordinate.

    The sentence is an ALLOY. Every word is an element. The final VADUGWI
    is the property of the alloy — not the property of the strongest element.

    All charged atoms contribute proportional to their gravity (emotional mass).
    Heavier atoms pull the alloy's properties more, but no atom is ignored.
    Operators (negators, intensifiers) modify the atoms they bond with
    before the alloy is computed.

    Args:
        equation: A decomposed sentence (from engine_v9.decomposer.decompose).

    Returns:
        VADUG coordinate representing the emotional state of the sentence.
    """
    # Step 1: Start at neutral
    state = list(_NEUTRAL_STATE)  # [V, A, D, U, G, W, I]

    # Adaptive force scale: shorter sentences = denser signal
    word_count = max(1, len(equation.all_atoms))
    effective_fs = FORCE_SCALE / (word_count ** LENGTH_EXPONENT)

    # Step 2: Collect ALL charged atoms with their gravity weights
    # Every atom contributes to the alloy proportional to its mass.
    # The nucleus gets operator modifications applied first.
    weighted_charges = []
    total_gravity = 0.0

    # Nucleus — apply operators first, then add with its gravity
    nucleus_charge = list(_apply_operators(equation.nucleus.root.charge, equation.operators))
    nucleus_gravity = equation.nucleus.gravity
    if nucleus_gravity > 0:
        weighted_charges.append((nucleus_charge, nucleus_gravity))
        total_gravity += nucleus_gravity

    # All other charged atoms — contribute at their own gravity
    for atom in equation.context:
        charge = list(atom.root.charge)
        gravity = atom.gravity
        if gravity > 0:
            weighted_charges.append((charge, gravity))
            total_gravity += gravity

    # Subject contributes too (if it has charge)
    if equation.subject.gravity > 0 and equation.subject is not equation.nucleus:
        weighted_charges.append((list(equation.subject.root.charge), equation.subject.gravity))
        total_gravity += equation.subject.gravity

    # Step 3: Compute the alloy — gravity-weighted average of all elements
    if total_gravity > 0:
        for charge, gravity in weighted_charges:
            weight = gravity / total_gravity  # normalize so weights sum to 1
            for dim in range(7):
                state[dim] += charge[dim] * weight * effective_fs

    # Step 4: Subject modifier — self-reference amplifies deviation
    if equation.subject.root.category == RootCategory.SELF_REF:
        for dim in range(7):
            center = 0.0 if dim == 3 else CENTER
            deviation = state[dim] - center
            state[dim] = center + deviation * SUBJECT_SELF_BONUS

    # Step 6: Tanh saturation
    for dim in range(7):
        center = 0.0 if dim == 3 else CENTER  # U centers at 0
        state[dim] = _tanh_saturate(state[dim], center, SATURATION)

    # Step 7: Clamp to [0, 255] and return
    clamped = [max(0.0, min(255.0, s)) for s in state]
    return VADUG(
        v=int(round(clamped[0])),
        a=int(round(clamped[1])),
        d=int(round(clamped[2])),
        u=int(round(clamped[3])),
        g=int(round(clamped[4])),
        w=int(round(clamped[5])),
        i=int(round(clamped[6])),
    )
