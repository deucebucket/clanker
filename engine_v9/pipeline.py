"""V9 Pipeline — equation decomposition engine with full physics stack.

Pipeline stages:
  1. Tokenize     — split text, resolve compound bonds
  2. Classify     — assign structural roles (V8 word_classifier)
  3. Decompose    — find event nucleus, assign equation roles
  4. Compose      — resolve charges from equation → raw VADUGWI
  5. Structures   — detect 45+ chess-like patterns from role sequences
  6. Force flow   — resolve WHO does WHAT to WHOM
  7. Personality  — apply personality vector (if provided)
  8. Clamp        — final bounds enforcement

Parallel outputs (not in main pipeline):
  - Zone classification (from composed VADUGWI)
  - Crisis tracking (fed per-turn, not per-pipeline)

Public API: compute_vadug(text) → (VADUG, trace_dict)
"""

from typing import Optional, Tuple

from .shared import VADUG, PersonalityVector
from .roots import RootCategory
from .tokenizer import tokenize
from .word_classifier import classify_sentence
from .decomposer import decompose, decompose_molecules
from .composer import compose
from .bonding import bond_resolve
from .structures import StructureDetector, StructureMatch
from .force_flow import resolve_force_flow, compute_flow_modifiers, compute_intent
from .zones import ZoneClassifier


# ── Module-level singletons ──────────────────────────────────────

_structure_detector = StructureDetector()
_zone_classifier = ZoneClassifier()

CENTER = 128


def compute_vadug(
    text: str,
    personality: Optional[PersonalityVector] = None,
    perspective: str = "speaker",
) -> Tuple[VADUG, dict]:
    """Compute VADUGWI coordinates for text using equation decomposition.

    Args:
        text: Input sentence/text
        personality: Optional personality vector (applied as final modifier)
        perspective: "speaker", "listener", or "bystander"

    Returns:
        (VADUG, trace_dict) where trace contains decomposition details
    """
    # ── Stage 1: Tokenize ────────────────────────────────────────
    tokens = tokenize(text)

    # ── Stage 1.5: Molecular bonding pass ────────────────────────
    molecules = bond_resolve(text)
    bond_flags = set()
    for mol in molecules:
        bond_flags |= mol.flags

    # ── Stage 2: Classify structural roles ───────────────────────
    word_roles = classify_sentence(tokens) if tokens else []

    # ── Stage 3: Decompose into equation ─────────────────────────
    # Use molecule-aware decomposition — bonded charges give better gravity
    equation = decompose_molecules(molecules) if molecules else decompose(text)

    # ── Stage 4: Compose equation → raw VADUGWI ──────────────────
    result = compose(equation)

    # ── Stage 4.5: Sarcasm / contradiction detection ───────────────
    # Two-level check:
    # 1. Adjacent-pair bonding flags (from bond_resolve)
    # 2. Sentence-level: positive nucleus + negative context = sarcasm
    sarcasm_detected = "sarcasm" in bond_flags or "irony" in bond_flags

    if not sarcasm_detected and len(molecules) >= 2:
        # Sentence-level contradiction: nucleus charge vs context charge
        nucleus_v = equation.nucleus.root.charge[0]
        # Sum context molecule charges (everything except the nucleus word)
        context_v_sum = 0
        context_count = 0
        for mol in molecules:
            # Skip the nucleus word
            if equation.nucleus.word in mol.words:
                continue
            # Skip zero-charge molecules
            if mol.surface_charge[0] == 0:
                continue
            context_v_sum += mol.surface_charge[0]
            context_count += 1

        if context_count >= 1:
            # Only fire sentence-level sarcasm when nucleus is a STRONG positive
            # evaluation — "love", "cherish", "delighted", "adore" (dV >= 35).
            # Mild positives ("good", "fine") in negative context = not sarcasm,
            # just mixed. This prevents false positives on literary narrative.
            _SARCASM_CATEGORIES = frozenset({
                RootCategory.POSITIVE_STATE,
                RootCategory.POSITIVE_EVENT,
            })
            nuc_is_strong_pos = (
                nucleus_v >= 35
                and equation.nucleus.root.category in _SARCASM_CATEGORIES
            )

            if nuc_is_strong_pos and context_v_sum < 0:
                divergence = nucleus_v - context_v_sum
                if divergence > 45:
                    sarcasm_detected = True
                    bond_flags.add("sarcasm_sentence")

    if sarcasm_detected:
        result = VADUG(
            v=max(0, min(255, 256 - result.v)),  # mirror around center
            a=result.a,
            d=result.d,
            u=result.u,
            g=result.g,
            w=result.w,
            i=result.i,
        )

    # ── Stage 5: Detect structural patterns ──────────────────────
    structures = _structure_detector.detect_all(word_roles) if word_roles else []

    # Apply structure adjustments to VADUGWI
    v, a, d, u, g, w, i = float(result.v), float(result.a), float(result.d), \
                           float(result.u), float(result.g), float(result.w), float(result.i)

    for s in structures:
        v += getattr(s, 'v_weight', 0)
        d += getattr(s, 'd_weight', 0)
        u += getattr(s, 'u_weight', 0)
        g += getattr(s, 'g_weight', 0)
        w += getattr(s, 'w_weight', 0)

    # ── Stage 6: Force flow ──────────────────────────────────────
    force_flow = resolve_force_flow(word_roles) if word_roles else None
    flow_mods = compute_flow_modifiers(force_flow) if force_flow else {}

    # Apply force flow modifiers
    if flow_mods:
        v += flow_mods.get('v_mod', 0)
        d += flow_mods.get('d_mod', 0)
        w += flow_mods.get('w_mod', 0)

    # Compute intent from force flow
    if force_flow:
        i = compute_intent(force_flow)

    # ── Stage 7: Personality ─────────────────────────────────────
    if personality is not None:
        sensitivity = personality.emotional_sensitivity
        v = CENTER + (v - CENTER) * sensitivity
        a = CENTER + (a - CENTER) * sensitivity
        d = CENTER + (d - CENTER) * sensitivity + personality.dominance_baseline
        u = u * sensitivity
        g = CENTER + (g - CENTER) * sensitivity + personality.gravity_bias
        w = CENTER + (w - CENTER) * sensitivity

    # ── Stage 8: Clamp ───────────────────────────────────────────
    result = VADUG(
        v=max(0, min(255, int(round(v)))),
        a=max(0, min(255, int(round(a)))),
        d=max(0, min(255, int(round(d)))),
        u=max(0, min(255, int(round(u)))),
        g=max(0, min(255, int(round(g)))),
        w=max(0, min(255, int(round(w)))),
        i=max(0, min(255, int(round(i)))),
    )

    # ── Zone classification ──────────────────────────────────────
    zone_result = _zone_classifier.classify(result)

    # ── Build trace ──────────────────────────────────────────────
    trace = {
        "tokens": tokens,
        "equation": {
            "subject": equation.subject.word,
            "nucleus": equation.nucleus.word,
            "nucleus_root": equation.nucleus.root.name,
            "nucleus_gravity": equation.nucleus.gravity,
            "operators": [a.word for a in equation.operators],
            "context": [a.word for a in equation.context],
        },
        "structures": [s.pattern for s in structures],
        "bonds": [{"words": m.words, "v": m.surface_charge[0]} for m in molecules if len(m.words) > 1],
        "bond_flags": list(bond_flags),
        "force_flow": {
            "actor": force_flow.actor_role if force_flow else None,
            "target": force_flow.target_role if force_flow else None,
            "force_valence": force_flow.force_valence if force_flow else 0,
        } if force_flow else None,
        "zone": zone_result.zone,
        "zone_confidence": zone_result.confidence,
        "word_count": len(tokens),
    }

    return result, trace
