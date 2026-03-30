"""Fixed Physics Layer — computes VADUG from structural analysis.

Pipeline: text -> classify words -> compute proximity -> detect structures -> apply physics -> VADUG

The physics (momentum, force application, blending) are FIXED.
The inputs come from the structural layers.
"""

from typing import List, Optional, Tuple

from .shared import VADUG, PersonalityVector
from .word_classifier import WordRole, classify_sentence, _clean
from .proximity import proximity_coefficient
from .structures import StructureDetector, StructureMatch


# ── Physics constants (fixed, never tuned per-sentence) ─────────

CENTER = 128.0
MOMENTUM = 0.82
FORCE_SCALE = 0.5
DIRECT_PUSH_CAP = 0.4
DIRECT_PUSH_TRIGGER = 80.0


# ── Main entry point ────────────────────────────────────────────

def compute_vadug(
    text: str,
    personality: Optional[PersonalityVector] = None,
) -> Tuple[VADUG, dict]:
    """Compute VADUG coordinates for a text string.

    Pipeline:
      1. Split text into words
      2. Layer 1: classify_sentence() -- structural roles
      3. Layer 2: proximity_coefficient() -- distance-based influence
      4. Layer 3: StructureDetector().detect_all() -- chess-like patterns
      5. Physics loop: momentum + force blending
      6. Structure adjustments
      7. Personality adjustments (if provided)
      8. Clamp to 0-255

    Returns (VADUG, trace_dict) where trace_dict contains:
      - trace: list of per-word entries {word, role, coeff, v, a, d, u, g}
      - structures: list of detected StructureMatch objects
      - word_count: int
    """
    words = text.split()
    if not words:
        return VADUG(), {"trace": [], "structures": [], "word_count": 0}

    # Layer 1: structural roles
    roles = classify_sentence(words)

    # Layer 2: proximity coefficients (computed per-word in the loop)

    # Layer 3: structure detection
    detector = StructureDetector()
    structures = detector.detect_all(roles)

    # ── Physics loop ────────────────────────────────────────────
    state_v = CENTER
    state_a = CENTER
    state_d = CENTER
    state_u = 0.0       # urgency starts at 0, not center
    state_g = CENTER

    trace_entries: List[dict] = []

    for i, wr in enumerate(roles):
        if wr.role != "EMOTIONAL" or wr.force is None:
            trace_entries.append({
                "word": wr.word,
                "role": wr.role,
                "coeff": 0.0,
                "v": round(state_v),
                "a": round(state_a),
                "d": round(state_d),
                "u": round(state_u),
                "g": round(state_g),
            })
            continue

        dv, da, dd, du, dg = wr.force
        coeff = proximity_coefficient(roles, i)

        # Target = center + force * coefficient * scale
        target_v = CENTER + dv * coeff * FORCE_SCALE
        target_a = CENTER + da * coeff * FORCE_SCALE
        target_d = CENTER + dd * coeff * FORCE_SCALE
        target_u = du * abs(coeff) * FORCE_SCALE  # urgency is 0-based
        target_g = CENTER + dg * coeff * FORCE_SCALE

        # Direct push for strong forces
        total_force = abs(dv) + abs(da) + abs(dd) + abs(du) + abs(dg)
        push_strength = min(1.0, total_force / DIRECT_PUSH_TRIGGER) * DIRECT_PUSH_CAP

        # Direction of push follows force sign
        push_v = push_strength * (1.0 if dv * coeff >= 0 else -1.0) * abs(dv) * FORCE_SCALE
        push_a = push_strength * (1.0 if da * coeff >= 0 else -1.0) * abs(da) * FORCE_SCALE
        push_d = push_strength * (1.0 if dd * coeff >= 0 else -1.0) * abs(dd) * FORCE_SCALE
        push_u = push_strength * abs(du) * FORCE_SCALE  # urgency only goes up
        push_g = push_strength * (1.0 if dg * coeff >= 0 else -1.0) * abs(dg) * FORCE_SCALE

        # Momentum blend + direct push
        inv_m = 1.0 - MOMENTUM
        state_v = state_v * MOMENTUM + target_v * inv_m + push_v
        state_a = state_a * MOMENTUM + target_a * inv_m + push_a
        state_d = state_d * MOMENTUM + target_d * inv_m + push_d
        state_u = state_u * MOMENTUM + target_u * inv_m + push_u
        state_g = state_g * MOMENTUM + target_g * inv_m + push_g

        trace_entries.append({
            "word": wr.word,
            "role": wr.role,
            "coeff": round(coeff, 3),
            "v": round(state_v),
            "a": round(state_a),
            "d": round(state_d),
            "u": round(state_u),
            "g": round(state_g),
        })

    # ── Structure adjustments ───────────────────────────────────
    for sm in structures:
        state_v += sm.v_weight * sm.confidence * FORCE_SCALE
        state_d += sm.d_weight * sm.confidence * FORCE_SCALE
        state_u += sm.u_weight * sm.confidence * FORCE_SCALE
        state_g += sm.g_weight * sm.confidence * FORCE_SCALE

    # ── Personality adjustments ─────────────────────────────────
    if personality is not None:
        sensitivity = personality.emotional_sensitivity
        # Scale deviation from center by sensitivity
        state_v = CENTER + (state_v - CENTER) * sensitivity
        state_a = CENTER + (state_a - CENTER) * sensitivity
        state_d = CENTER + (state_d - CENTER) * sensitivity + personality.dominance_baseline
        state_u = state_u * sensitivity
        state_g = CENTER + (state_g - CENTER) * sensitivity + personality.gravity_bias

    # ── Clamp to 0-255 ─────────────────────────────────────────
    result = VADUG(
        v=int(round(max(0, min(255, state_v)))),
        a=int(round(max(0, min(255, state_a)))),
        d=int(round(max(0, min(255, state_d)))),
        u=int(round(max(0, min(255, state_u)))),
        g=int(round(max(0, min(255, state_g)))),
    )

    trace_dict = {
        "trace": trace_entries,
        "structures": structures,
        "word_count": len(words),
    }

    return result, trace_dict
