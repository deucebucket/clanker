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
from .interpret import interpret_context
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

    # ── Stage 2.5: Interpret context ─────────────────────────────
    # Discourse markers, expletive-as-intensifier, register detection,
    # counterfactual marking, hard negator inversion, SOLVENT dissolution,
    # sarcasm field. Modifies word_roles in place + returns context flags.
    ctx = interpret_context(word_roles, text)
    word_roles = ctx["roles"]
    register = ctx["register"]
    register_dampener = ctx["register_dampener"]

    # ── Stage 3: Decompose into equation ─────────────────────────
    # Use molecule-aware decomposition — bonded charges give better gravity
    equation = decompose_molecules(molecules) if molecules else decompose(text)

    # ── Stage 3.5: Bridge — apply interpret_context to equation atoms ──
    # interpret_context modified word_roles (V8-style forces).
    # The equation reads from root charges. Bridge the two:
    # For each word_role that was modified, update the matching equation atom.
    # SKIP atoms that were already bonded in molecules (bonding already handled them).
    if word_roles:
        # Build lookup: word → modified force from interpret_context
        role_forces = {}
        for wr in word_roles:
            if wr.force is not None:
                role_forces[wr.word] = wr.force

        # Words already bonded in molecules — skip these to avoid double-inversion
        bonded_words = set()
        for mol in molecules:
            if len(mol.words) > 1:
                for w in mol.words:
                    bonded_words.add(w)

        # Update equation atoms with modified forces
        from .roots import Root
        from .decomposer import _compute_gravity

        for atom in equation.all_atoms:
            # Get the single word (molecules have space-joined words)
            word = atom.word.split()[0] if ' ' in atom.word else atom.word

            # Skip if this word was already bonded in a molecule
            if word in bonded_words:
                continue

            if word in role_forces:
                modified_force = role_forces[word]
                # Extend 5-tuple to 7-tuple (add dW=0, dI=0 if needed)
                if len(modified_force) == 5:
                    charge_7d = modified_force + (0, 0)
                else:
                    charge_7d = modified_force

                # Only override if interpret_context actually CHANGED it
                root_5d = atom.root.charge[:5]
                if modified_force[:5] != root_5d:
                    new_root = Root(
                        name=atom.root.name,
                        category=atom.root.category,
                        charge=tuple(charge_7d),
                        phase=atom.root.phase,
                    )
                    atom.root = new_root
                    atom.gravity = _compute_gravity(new_root)

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
            # Sentence-level sarcasm: positive nucleus + negative context
            # Expanded: any positive root category with dV >= 14 qualifies
            _SARCASM_CATEGORIES = frozenset({
                RootCategory.POSITIVE_STATE,
                RootCategory.POSITIVE_EVENT,
                RootCategory.POSITIVE_QUALITY,
                RootCategory.SOCIAL_EVAL_POS,
            })
            nuc_is_positive = (
                nucleus_v >= 14
                and equation.nucleus.root.category in _SARCASM_CATEGORIES
            )

            if nuc_is_positive and context_v_sum < -10 and context_count >= 2:
                # Context must be MEANINGFULLY negative (sum < -10)
                # AND have at least 2 negative context atoms (not just one weak word)
                divergence = nucleus_v - context_v_sum
                if divergence > 50:
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

    # ── Stage 4.6: Attributive dampening ───────────────────────────
    # No person reference in the sentence → likely descriptive/narrative.
    # Even with alloy composition, strong emotional words in description
    # push too hard. Dampen when no person is present.
    _PERSON_ROLES = frozenset({"SELF_REF", "OTHER_REF", "RELATION_REF"})
    has_person_ref = any(wr.role in _PERSON_ROLES for wr in word_roles) if word_roles else False

    if not has_person_ref and abs(result.v - 128) > 10:
        ATTRIBUTIVE_DAMPEN = 0.5
        result = VADUG(
            v=max(0, min(255, int(128 + (result.v - 128) * ATTRIBUTIVE_DAMPEN))),
            a=result.a, d=result.d, u=result.u, g=result.g, w=result.w, i=result.i,
        )

    # ── Stage 5: Detect structural patterns ──────────────────────
    structures = _structure_detector.detect_all(word_roles) if word_roles else []

    # Apply structure adjustments to VADUGWI
    # Skip structures that net-hurt accuracy (measured on combined dataset)
    _STRUCTURE_BLACKLIST = frozenset({
        "LIFE_ACHIEVEMENT",          # net -7 (false triggers on narrative)
        "NEGATED_NEGATIVE_COMPLIMENT", # net -4
    })

    v, a, d, u, g, w, i = float(result.v), float(result.a), float(result.d), \
                           float(result.u), float(result.g), float(result.w), float(result.i)

    for s in structures:
        if s.pattern in _STRUCTURE_BLACKLIST:
            continue
        v += getattr(s, 'v_weight', 0)
        d += getattr(s, 'd_weight', 0)
        u += getattr(s, 'u_weight', 0)
        g += getattr(s, 'g_weight', 0)
        w += getattr(s, 'w_weight', 0)

    # ── Stage 5.5: Apply interpret_context flags ──────────────────
    # Register dampening: literary/expository text gets force reduced
    if register_dampener < 1.0:
        v = CENTER + (v - CENTER) * register_dampener
        a = CENTER + (a - CENTER) * register_dampener
        d = CENTER + (d - CENTER) * register_dampener
        g = CENTER + (g - CENTER) * register_dampener
        w = CENTER + (w - CENTER) * register_dampener

    # Counterfactual: "supposed to be happy" → invert positive
    if ctx["counterfactual"] and v > CENTER:
        v = CENTER - (v - CENTER) * 0.75  # invert 75% toward negative

    # Sarcasm from interpret_context (ironic onset / contrast field)
    # When detected AND V is positive, do a full inversion — not just a penalty.
    # "Another fantastic decision" with sarcasm flag should flip, not nudge.
    if ctx["sarcasm_inversion"]:
        if v > CENTER:
            v = CENTER - (v - CENTER) * 0.8  # full inversion
        else:
            v += ctx["sarcasm_penalty"]  # already negative, just nudge

    # Stage 5.8: Static friction — DROPPED
    # Tested: kills emotion detection (balanced 0.617 → 0.546).
    # Was also disabled in V8. The alloy + length scaling handle weak drift.

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

    # ── Stage 6.5: Perspective dampening ─────────────────────────
    # When the subject is OTHER_REF and the emotion is NOT directed at SELF,
    # dampen the valence — it's narrative about someone else, not the speaker.
    # But when force_flow shows OTHER → SELF, keep full charge (speaker feels it).
    _SELF_WORDS = frozenset({"i", "me", "my", "myself", "im", "ive", "<implicit>"})
    subj_word = equation.subject.word.lower()
    is_other_subject = subj_word not in _SELF_WORDS

    if is_other_subject:
        # Check force flow — if directed at SELF, keep full charge
        directed_at_self = (
            force_flow is not None
            and force_flow.target_role == "SELF_REF"
        )
        if not directed_at_self:
            # Dampen: pull V toward center by 60%
            PERSPECTIVE_DAMPEN = 1.0   # optimizer gen50: no dampening
            v = CENTER + (v - CENTER) * PERSPECTIVE_DAMPEN
            # Also dampen W (self-worth shouldn't move from others' emotions)
            w = CENTER + (w - CENTER) * PERSPECTIVE_DAMPEN

    # ── Stage 6.8: W→V coupling ────────────────────────────────────
    # Low self-worth amplifies negatives and suppresses positives.
    # At W=128 (neutral): no effect. At W=50: negatives amplified ~1.8x.
    from math import exp
    w_norm = (w - CENTER) / CENTER  # -1 (broken) to +1 (strong)
    W_BETA = 0.5    # negative amplification strength
    W_GAMMA = 0.3   # positive suppression strength
    W_CAP = 1.8     # max amplification

    displacement = v - CENTER
    if displacement < 0:
        amp = min(W_CAP, 1.0 + W_BETA * exp(-w_norm))
        v = CENTER + displacement * amp
    elif displacement > 0:
        sup = max(0.0, 1.0 - W_GAMMA * (1.0 - exp(w_norm)))
        v = CENTER + displacement * sup

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

    # ── Confidence scoring ───────────────────────────────────────
    # How confident is the engine in this reading?
    # Low confidence = engine is guessing, caller should treat with caution.
    #
    # Confidence factors:
    # 1. Nucleus gravity — weak nucleus = low signal
    # 2. V distance from thresholds — borderline = uncertain
    # 3. Number of charged atoms — fewer = less evidence
    # 4. Contradictions — sarcasm flags, mixed signals
    confidence = 1.0

    # Factor 1: nucleus gravity (max 0.3 penalty)
    max_gravity = max((a.gravity for a in equation.all_atoms), default=0)
    if max_gravity < 5:
        confidence -= 0.3  # no charged atoms at all
    elif max_gravity < 15:
        confidence -= 0.15  # weak signal

    # Factor 2: distance from classification thresholds (max 0.3 penalty)
    v_final = result.v
    dist_to_pos = abs(v_final - 145)
    dist_to_neg = abs(v_final - 110)
    min_dist = min(dist_to_pos, dist_to_neg)
    if min_dist < 5:
        confidence -= 0.3  # right on the boundary
    elif min_dist < 15:
        confidence -= 0.15  # near the boundary

    # Factor 3: charged atom count (max 0.2 penalty)
    charged_count = sum(1 for a in equation.all_atoms if a.gravity > 5)
    if charged_count == 0:
        confidence -= 0.2
    elif charged_count == 1:
        confidence -= 0.1

    # Factor 4: contradictions (max 0.2 penalty)
    if bond_flags:
        confidence -= 0.1  # sarcasm/irony detected = uncertain
    if ctx.get("sarcasm_inversion"):
        confidence -= 0.1

    confidence = max(0.0, min(1.0, confidence))

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
        "register": register,
        "counterfactual": ctx["counterfactual"],
        "sarcasm_interpret": ctx["sarcasm_inversion"],
        "force_flow": {
            "actor": force_flow.actor_role if force_flow else None,
            "target": force_flow.target_role if force_flow else None,
            "force_valence": force_flow.force_valence if force_flow else 0,
        } if force_flow else None,
        "zone": zone_result.zone,
        "zone_confidence": zone_result.confidence,
        "word_count": len(tokens),
        "confidence": round(confidence, 2),
    }

    return result, trace
