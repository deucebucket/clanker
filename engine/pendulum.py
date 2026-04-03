"""Fixed Physics Layer — computes VADUGWI from structural analysis.

Pipeline: text -> classify words -> compute proximity -> detect structures -> apply physics -> VADUGWI

The physics (momentum, force application, blending) are FIXED.
The inputs come from the structural layers.

Refactored into a modular pipeline: each stage is an independent function
that reads from and writes to a shared context dict. Stages can be chained,
bypassed, or swapped.
"""

from math import tanh
from typing import List, Optional, Tuple

from .shared import VADUG, PersonalityVector
from .word_classifier import WordRole, classify_sentence, _clean
from .proximity import proximity_coefficient
from .vocabulary import VOCABULARY
from .structures import StructureDetector, StructureMatch
from .force_flow import resolve_force_flow, compute_flow_modifiers, compute_intent


# ── Physics constants (fixed, never tuned per-sentence) ─────────

CENTER = 128.0
M_BASE = 0.557            # champion v2: genetically optimized 2026-04-03
M_AROUSAL_SCALE = 0.25    # arousal-scaled momentum: high A = sticky state
M_NEGATIVITY_BIAS = 1.15  # negative states are stickier than positive
M_POSITIVITY_EASE = 0.90  # positive transitions are easier
M_MIN = 0.30              # floor: never fully unresponsive to input
M_MAX = 0.95              # ceiling: never fully locked in state
FORCE_SCALE = 1.405       # champion v2
DIRECT_PUSH_CAP = 1.0     # champion v2: max push
DIRECT_PUSH_TRIGGER = 86.2  # champion v2
SATURATION = 120.0        # tanh saturation: smooth compression replaces hard clamp


# ── Compound resolution tables ──────────────────────────────────

_BOOKEND_COMPOUNDS = {
    "shut": "up",      # shut [anything] up = silence command
    "get": "out",      # get [the fuck] out = expulsion command
    "fuck": "off",     # fuck [right] off = rejection command
    "back": "off",     # back [the hell] off = distance command
    "piss": "off",     # piss off = rejection
}

_SPICE_WORDS = {"the", "a", "an", "fuck", "fucking", "fuckin", "damn",
                "god", "hell", "mother", "motherfuckin", "stupid",
                "bitch", "ass", "right", "just", "already", "up",
                "freakin", "freaking", "effing"}

_ONE_AS_QUANTIFIER = {"thing", "person", "place", "way", "reason", "time",
                      "day", "moment", "word", "chance", "step", "bit"}

_CONTINUATION_PAIRS = {
    ("wont", "stop"), ("wont", "quit"), ("wont", "end"),
    ("wont", "leave"), ("wont", "go"),
    ("cant", "stop"), ("cant", "quit"),
    ("dont", "stop"), ("doesnt", "stop"),
    ("never", "stop"), ("never", "stops"), ("never", "end"),
    ("never", "ends"), ("never", "quit"),
}

_UNIVERSAL_ADDRESS = {
    "everyone", "everybody", "anyone", "anybody", "all", "people",
    "folks", "y'all", "yall", "ladies", "gentlemen",
}

_SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}

_ABSENCE_MARKERS = {"without", "havent", "haven't", "hasnt", "hasn't"}
_ABSENCE_FOLLOWERS = {"had", "been", "felt", "seen", "gotten", "experienced"}
_PERSON_ROLES = {"SELF_REF", "OTHER_REF", "RELATION_REF"}

_CHOICE_VERBS = {"choose", "chose", "choosing", "decide", "pick", "picked"}


# ── Stage 1: Tokenize ──────────────────────────────────────────

def tokenize(context: dict) -> dict:
    """Split text into words, resolve compound phrases and double negations.

    Reads: context["text"]
    Writes: context["words"]
    """
    text = context["text"]
    words = text.split()
    if not words:
        context["words"] = []
        return context

    # Bookend compounds: opener ... closer with variable filling
    collapsed = list(words)
    for opener, closer in _BOOKEND_COMPOUNDS.items():
        start_idx = None
        for idx, w in enumerate(collapsed):
            if w.lower() == opener:
                start_idx = idx
            elif w.lower() == closer and start_idx is not None and idx - start_idx <= 8:
                filling = [collapsed[j].lower() for j in range(start_idx + 1, idx)]
                all_spice = all(f in _SPICE_WORDS for f in filling)
                if all_spice:
                    compound = opener + closer
                    collapsed = collapsed[:start_idx] + [compound] + collapsed[idx+1:]
                start_idx = None
                break
    words = collapsed

    # Double negation compounds and special pairs
    resolved = []
    i = 0
    while i < len(words):
        w_low = words[i].lower()
        next_low = words[i + 1].lower() if i + 1 < len(words) else ""
        pair = (w_low, next_low)

        if pair in _CONTINUATION_PAIRS:
            has_negative_before = any(
                VOCABULARY.get(words[j].lower(), (0,))[0] < -25
                for j in range(max(0, i - 3), i)
            )
            if has_negative_before:
                i += 2
            else:
                resolved.append(words[i])
                i += 1
        elif w_low == "no" and next_low == "one":
            next_after = words[i + 2].lower() if i + 2 < len(words) else ""
            if next_after in _ONE_AS_QUANTIFIER:
                resolved.append(words[i])
                i += 1
            else:
                resolved.append("nobody")
                i += 2
        elif w_low == "come" and next_low == "on":
            resolved.append("comeon")
            i += 2
        elif w_low == "killed" and next_low == "it":
            resolved.append("killedit")
            i += 2
        elif w_low == "goes" and next_low == "hard":
            resolved.append("goeshard")
            i += 2
        elif w_low == "closed" and next_low == "on":
            resolved.append("closedon")
            i += 2
        else:
            resolved.append(words[i])
            i += 1
    words = resolved

    context["words"] = words
    return context


# ── Stage 2: Classify ──────────────────────────────────────────

def classify(context: dict) -> dict:
    """Classify words into structural roles, apply perspective remapping.

    Reads: context["words"], context["perspective"], context.get("personality")
    Writes: context["roles"], context["perspective"] (may be resolved from "auto"),
            context["has_universal"]
    """
    words = context["words"]
    perspective = context.get("perspective", "speaker")
    personality = context.get("personality")

    roles = classify_sentence(words)

    # Universal address detection
    has_universal = any(w.lower() in _UNIVERSAL_ADDRESS for w in words)
    context["has_universal"] = has_universal

    # Auto-detect perspective
    if perspective == "auto":
        has_self = any(wr.role == "SELF_REF" for wr in roles)
        has_second = any(wr.word.lower() in _SECOND_PERSON for wr in roles)
        if has_self:
            perspective = "speaker"
        elif has_second:
            perspective = "listener"
        elif has_universal:
            perspective = "listener"
        else:
            perspective = "bystander"

    # Perspective remapping
    if perspective == "listener":
        for wr in roles:
            if wr.role == "SELF_REF":
                wr.role = "OTHER_REF"
            elif wr.role == "OTHER_REF":
                wr.role = "SELF_REF"
    elif perspective == "bystander":
        if has_universal:
            for wr in roles:
                if wr.role == "SELF_REF":
                    wr.role = "OTHER_REF"
        else:
            for wr in roles:
                if wr.role in ("SELF_REF", "OTHER_REF"):
                    wr.role = "NEUTRAL"

    # Bystander self-projection
    if perspective == "bystander" and personality is not None:
        bystander_w = personality.assertiveness
        if bystander_w < 80:
            _THIRD_PERSON = {"she", "he", "her", "him", "they", "them"}
            for idx, wr in enumerate(roles):
                if wr.word.lower() in _THIRD_PERSON and wr.role == "NEUTRAL":
                    for j in range(max(0, idx - 3), min(len(roles), idx + 4)):
                        if j != idx and roles[j].role == "EMOTIONAL":
                            w_clean = _clean(roles[j].word)
                            forces = VOCABULARY.get(w_clean)
                            if forces and forces[0] < -10:
                                wr.role = "SELF_REF"
                                break

    # Absence scope
    absence_scope = set()
    for i, wr in enumerate(roles):
        if wr.word in _ABSENCE_MARKERS:
            if wr.word == "without":
                for j in range(i + 1, min(i + 4, len(roles))):
                    if roles[j].role not in _PERSON_ROLES:
                        absence_scope.add(j)
            elif i + 1 < len(roles) and roles[i + 1].word in _ABSENCE_FOLLOWERS:
                for j in range(i + 2, min(i + 7, len(roles))):
                    absence_scope.add(j)

    # Forced choice cancellation
    forced_choice_scope = set()
    for i, wr in enumerate(roles):
        if wr.word == "between":
            has_choice = any(roles[j].word in _CHOICE_VERBS
                           for j in range(max(0, i - 3), i))
            if has_choice:
                for j in range(i + 1, min(i + 6, len(roles))):
                    forced_choice_scope.add(j)

    context["roles"] = roles
    context["perspective"] = perspective
    context["absence_scope"] = absence_scope
    context["forced_choice_scope"] = forced_choice_scope
    return context


# ── Stage 3: Compute Coefficients (structure detection + force flow) ─

def compute_coefficients(context: dict) -> dict:
    """Detect structures and resolve force flow.

    Reads: context["roles"]
    Writes: context["structures"], context["force_flow"], context["flow_mods"]
    """
    roles = context["roles"]

    detector = StructureDetector()
    structures = detector.detect_all(roles)

    force_flow = resolve_force_flow(roles)
    flow_mods = compute_flow_modifiers(force_flow)

    context["structures"] = structures
    context["force_flow"] = force_flow
    context["flow_mods"] = flow_mods
    return context


# ── Stage 4: Accumulate Forces ──────────────────────────────────

def accumulate_forces(context: dict) -> dict:
    """Per-word force application loop with adaptive momentum.

    Reads: context["roles"], context["absence_scope"], context["forced_choice_scope"],
           context["force_flow"], context["flow_mods"]
    Writes: context["state_v"], context["state_a"], context["state_d"],
            context["state_u"], context["state_g"], context["state_w"],
            context["trace_entries"]
    """
    roles = context["roles"]
    absence_scope = context.get("absence_scope", set())
    forced_choice_scope = context.get("forced_choice_scope", set())
    force_flow = context.get("force_flow")
    flow_mods = context.get("flow_mods", {})

    state_v = CENTER
    state_a = CENTER
    state_d = CENTER
    state_u = 0.0
    state_g = CENTER
    state_w = CENTER

    trace_entries: List[dict] = []

    for i, wr in enumerate(roles):
        if wr.role == "POSSESSION":
            vf = VOCABULARY.get(wr.word)
            if vf:
                word_force = (0, 0, 0, 0, max(5, vf[4]))
            else:
                word_force = (0, 0, 0, 0, 5)
        else:
            word_force = wr.force
            if word_force is None:
                word_force = VOCABULARY.get(wr.word)
            if word_force is None:
                from .fuzzy import fuzzy_match
                matched = fuzzy_match(wr.word)
                if matched:
                    word_force = VOCABULARY.get(matched)

        if word_force is None:
            trace_entries.append({
                "word": wr.word,
                "role": wr.role,
                "coeff": 0.0,
                "v": round(state_v),
                "a": round(state_a),
                "d": round(state_d),
                "u": round(state_u),
                "g": round(state_g),
                "w": round(state_w),
            })
            continue

        dv, da, dd, du, dg = word_force

        # Forced choice cancellation
        if i in forced_choice_scope and dv > 0:
            dv = -dv

        # Absence scope dampening
        if i in absence_scope and dv < -10:
            dv = int(dv * 0.2)
            da = int(da * 0.3)
            dd = int(dd * 0.3)

        # "Without" as pure operator
        if wr.word in ("without",) and any(j in absence_scope for j in range(i+1, min(i+4, len(roles)))):
            dv = 0
            da = 0
            dd = 0

        # Force flow direction modifiers
        if i == (force_flow.force_idx if force_flow else -1):
            dv = int(dv * flow_mods["v_mod"])
            dd = int(dd * flow_mods["d_mod"])
        coeff = proximity_coefficient(roles, i)

        # Target = center + force * coefficient * scale
        target_v = CENTER + dv * coeff * FORCE_SCALE
        target_a = CENTER + da * coeff * FORCE_SCALE
        target_d = CENTER + dd * coeff * FORCE_SCALE
        target_u = du * abs(coeff) * FORCE_SCALE
        target_g = CENTER + dg * coeff * FORCE_SCALE

        # Direct push for strong forces
        total_force = abs(dv) + abs(da) + abs(dd) + abs(du) + abs(dg)
        push_strength = min(1.0, total_force / DIRECT_PUSH_TRIGGER) * DIRECT_PUSH_CAP

        push_v = push_strength * (1.0 if dv * coeff >= 0 else -1.0) * abs(dv) * FORCE_SCALE
        push_a = push_strength * (1.0 if da * coeff >= 0 else -1.0) * abs(da) * FORCE_SCALE
        push_d = push_strength * (1.0 if dd * coeff >= 0 else -1.0) * abs(dd) * FORCE_SCALE
        push_u = push_strength * abs(du) * FORCE_SCALE
        push_g = push_strength * (1.0 if dg * coeff >= 0 else -1.0) * abs(dg) * FORCE_SCALE

        # Adaptive momentum
        m_eff = M_BASE + (state_a - CENTER) / 255.0 * M_AROUSAL_SCALE

        if state_v < CENTER and target_v > state_v:
            m_v = max(M_MIN, min(M_MAX, m_eff * M_NEGATIVITY_BIAS))
        elif state_v > CENTER and target_v < state_v:
            m_v = max(M_MIN, min(M_MAX, m_eff * M_POSITIVITY_EASE))
        else:
            m_v = max(M_MIN, min(M_MAX, m_eff))

        m_eff = max(M_MIN, min(M_MAX, m_eff))

        inv_m_v = 1.0 - m_v
        inv_m = 1.0 - m_eff
        inv_m_base = 1.0 - M_BASE
        state_v = state_v * m_v + target_v * inv_m_v + push_v
        state_a = state_a * m_eff + target_a * inv_m + push_a
        state_d = state_d * m_eff + target_d * inv_m + push_d
        state_u = state_u * M_BASE + target_u * inv_m_base + push_u
        state_g = state_g * M_BASE + target_g * inv_m_base + push_g

        # W (self-worth)
        self_ref_nearby = any(
            roles[j].role == "SELF_REF" and abs(j - i) <= 4
            for j in range(max(0, i - 4), min(len(roles), i + 5))
            if j != i
        )
        if self_ref_nearby and dv != 0:
            w_damp = 0.7
            w_flow = flow_mods["w_mod"] if force_flow and i == force_flow.force_idx else 1.0
            w_effective = dv * coeff * FORCE_SCALE * w_damp * w_flow
            target_w = CENTER + w_effective
            push_w = push_strength * (1.0 if dv * coeff >= 0 else -1.0) * abs(dv) * FORCE_SCALE * w_damp * w_flow
            state_w = state_w * m_eff + target_w * inv_m + push_w

        trace_entries.append({
            "word": wr.word,
            "role": wr.role,
            "coeff": round(coeff, 3),
            "v": round(state_v),
            "a": round(state_a),
            "d": round(state_d),
            "u": round(state_u),
            "g": round(state_g),
            "w": round(state_w),
        })

    context["state_v"] = state_v
    context["state_a"] = state_a
    context["state_d"] = state_d
    context["state_u"] = state_u
    context["state_g"] = state_g
    context["state_w"] = state_w
    context["trace_entries"] = trace_entries
    return context


# ── Stage 5: Apply Structures ──────────────────────────────────

def apply_structures(context: dict) -> dict:
    """Apply structure detection adjustments to state.

    Reads: context["state_*"], context["structures"], context["roles"]
    Writes: context["state_v"], context["state_d"], context["state_u"],
            context["state_g"], context["state_w"]
    """
    structures = context.get("structures", [])
    roles = context.get("roles", [])
    state_v = context["state_v"]
    state_a = context["state_a"]
    state_d = context["state_d"]
    state_u = context["state_u"]
    state_g = context["state_g"]
    state_w = context["state_w"]

    for sm in structures:
        if sm.pattern == "SLANG_DEATH_HUMOR":
            # Nullify death word's negative dV and add back a positive-scaled version.
            # The death word is NOT functioning as death -- it's an intensifier.
            # V_corrected = V_raw - dV("dead") + 0.7 * abs(dV("dead"))
            death_dv_total = 0
            for idx in sm.matched_indices:
                if idx < len(roles):
                    w = roles[idx].word
                    vf = VOCABULARY.get(w)
                    if vf and vf[0] < -10:  # negative death word
                        death_dv_total += vf[0]
            if death_dv_total < 0:
                # Subtract the death word's accumulated negative push and add positive version.
                # Use 1.2x on the nullification to account for momentum/push amplification
                # during accumulation, and 0.7x for the positive reinterpretation.
                correction = 1.2 * abs(death_dv_total) * FORCE_SCALE + 0.7 * abs(death_dv_total) * FORCE_SCALE
                state_v += correction * sm.confidence
            else:
                # Fallback: pull toward center
                distance = CENTER - state_v
                state_v += distance * 1.2 * sm.confidence
            state_w = max(state_w, CENTER)
        elif sm.pattern == "AMBIGUITY_HOLD":
            # Extreme V contradiction with no disambiguator: pull V toward W (neutral baseline).
            # V_final = V + (W - V) * 0.85
            state_v = state_v + (state_w - state_v) * 0.85 * sm.confidence
        elif sm.pattern == "RECOVERY_MILESTONE":
            # Recovery milestone: apply v_weight as direct positive boost
            state_v += sm.v_weight * sm.confidence * FORCE_SCALE
        elif sm.pattern in ("SARCASM_INVERSION", "BRAVADO", "DIRECTED_POSITIVE", "EXCLUDED_POSITIVE", "GRIEF_LOSS", "ATMOSPHERIC_GRIEF", "RHETORICAL_SELF_NEGATION", "REPORTED_COMFORT", "PASSIVE_RESIGNATION") and state_v > CENTER:
            excess = state_v - CENTER
            pull = sm.v_weight * sm.confidence * FORCE_SCALE * (1.0 + excess / 50.0)
            state_v += pull
            if sm.pattern == "RHETORICAL_SELF_NEGATION" and state_w > CENTER:
                w_excess = state_w - CENTER
                w_pull = sm.w_weight * sm.confidence * FORCE_SCALE * (1.0 + w_excess / 50.0)
                state_w += w_pull
                sm = StructureMatch(
                    pattern=sm.pattern, confidence=sm.confidence,
                    matched_indices=sm.matched_indices, description=sm.description,
                    v_weight=sm.v_weight, d_weight=sm.d_weight, u_weight=sm.u_weight,
                    g_weight=sm.g_weight, w_weight=0.0,
                )
        elif sm.pattern == "CHOPPER_SPLIT" and sm.matched_indices:
            chop_pos = sm.matched_indices[0]
            after_words = [wr for wr in roles if wr.position > chop_pos]
            after_v_sum = 0
            for wr in after_words:
                wf = wr.force or VOCABULARY.get(wr.word)
                if wf:
                    after_v_sum += wf[0]
            has_negator_after = any(wr.role == "NEGATOR" for wr in after_words)
            if (state_v > CENTER and (after_v_sum < 0 or (after_v_sum == 0 and has_negator_after))):
                distance = state_v - CENTER
                state_v -= distance * 1.5 * sm.confidence
            elif (state_v < CENTER and after_v_sum > 10):
                distance = CENTER - state_v
                state_v += distance * 0.4 * sm.confidence
        else:
            state_v += sm.v_weight * sm.confidence * FORCE_SCALE
        state_d += sm.d_weight * sm.confidence * FORCE_SCALE
        state_u += sm.u_weight * sm.confidence * FORCE_SCALE
        state_g += sm.g_weight * sm.confidence * FORCE_SCALE
        state_w += sm.w_weight * sm.confidence * FORCE_SCALE

    context["state_v"] = state_v
    context["state_a"] = state_a
    context["state_d"] = state_d
    context["state_u"] = state_u
    context["state_g"] = state_g
    context["state_w"] = state_w
    return context


# ── Stage 6: Apply W Coefficient ────────────────────────────────

def apply_w_coefficient(context: dict) -> dict:
    """Self-worth modulates valence: low W dampens positive, amplifies negative.

    Reads: context["state_v"], context["state_w"]
    Writes: context["state_v"]
    """
    state_v = context["state_v"]
    state_w = context["state_w"]

    w_factor = max(0.0, state_w) / CENTER
    if state_v > CENTER:
        state_v = CENTER + (state_v - CENTER) * min(w_factor, 1.0)
    elif state_v < CENTER:
        state_v = CENTER + (state_v - CENTER) * max(1.0, 2.0 - w_factor)

    context["state_v"] = state_v
    return context


# ── Stage 7: Apply Personality ──────────────────────────────────

def apply_personality(context: dict) -> dict:
    """Scale state by personality vector if provided.

    Reads: context["state_*"], context.get("personality")
    Writes: context["state_*"]
    """
    personality = context.get("personality")
    if personality is None:
        return context

    state_v = context["state_v"]
    state_a = context["state_a"]
    state_d = context["state_d"]
    state_u = context["state_u"]
    state_g = context["state_g"]
    state_w = context["state_w"]

    sensitivity = personality.emotional_sensitivity
    state_v = CENTER + (state_v - CENTER) * sensitivity
    state_a = CENTER + (state_a - CENTER) * sensitivity
    state_d = CENTER + (state_d - CENTER) * sensitivity + personality.dominance_baseline
    state_u = state_u * sensitivity
    state_g = CENTER + (state_g - CENTER) * sensitivity + personality.gravity_bias
    state_w = CENTER + (state_w - CENTER) * sensitivity

    context["state_v"] = state_v
    context["state_a"] = state_a
    context["state_d"] = state_d
    context["state_u"] = state_u
    context["state_g"] = state_g
    context["state_w"] = state_w
    return context


# ── Stage 8: Saturate and Clamp ─────────────────────────────────

def saturate_and_clamp(context: dict) -> dict:
    """Tanh saturation, intent computation, and 0-255 clamping.

    Reads: context["state_*"], context["force_flow"], context["roles"],
           context["trace_entries"], context["structures"], context["words"]
    Writes: context["vadug"], context["meta"]
    """
    state_v = context["state_v"]
    state_a = context["state_a"]
    state_d = context["state_d"]
    state_u = context["state_u"]
    state_g = context["state_g"]
    state_w = context["state_w"]
    force_flow = context.get("force_flow")
    roles = context.get("roles", [])

    # Intent computation
    state_i = compute_intent(force_flow, roles)

    # Tanh saturation
    state_v = CENTER + SATURATION * tanh((state_v - CENTER) / SATURATION)
    state_a = CENTER + SATURATION * tanh((state_a - CENTER) / SATURATION)
    state_d = CENTER + SATURATION * tanh((state_d - CENTER) / SATURATION)
    state_u = SATURATION * tanh(state_u / SATURATION)
    state_g = CENTER + SATURATION * tanh((state_g - CENTER) / SATURATION)
    state_w = CENTER + SATURATION * tanh((state_w - CENTER) / SATURATION)

    # Clamp to 0-255
    result = VADUG(
        v=int(round(max(0, min(255, state_v)))),
        a=int(round(max(0, min(255, state_a)))),
        d=int(round(max(0, min(255, state_d)))),
        u=int(round(max(0, min(255, state_u)))),
        g=int(round(max(0, min(255, state_g)))),
        w=int(round(max(0, min(255, state_w)))),
        i=state_i,
    )

    trace_dict = {
        "trace": context.get("trace_entries", []),
        "structures": context.get("structures", []),
        "force_flow": force_flow,
        "word_count": len(context.get("words", [])),
    }

    context["vadug"] = result
    context["meta"] = trace_dict
    return context


# ── Pipeline ────────────────────────────────────────────────────

class Pipeline:
    """Chainable pipeline of stage functions.

    Each stage is a function(context: dict) -> dict.
    Stages read from and write to the shared context dict.
    """

    def __init__(self, stages=None):
        self.stages = stages if stages is not None else self.default_stages()

    @staticmethod
    def default_stages():
        return [
            tokenize,
            classify,
            compute_coefficients,
            accumulate_forces,
            apply_structures,
            apply_w_coefficient,
            apply_personality,
            saturate_and_clamp,
        ]

    def run(self, text: str, perspective: str = "speaker",
            personality: Optional[PersonalityVector] = None) -> Tuple[VADUG, dict]:
        """Run the full pipeline on text.

        Returns (VADUG, trace_dict) — same interface as compute_vadug().
        """
        context = {
            "text": text,
            "perspective": perspective,
            "personality": personality,
        }

        # Early exit for empty text
        words = text.split()
        if not words:
            return VADUG(), {"trace": [], "structures": [], "word_count": 0}

        for stage in self.stages:
            context = stage(context)

        return context["vadug"], context["meta"]


# ── Main entry point (thin wrapper) ────────────────────────────

def compute_vadug(
    text: str,
    personality: Optional[PersonalityVector] = None,
    perspective: str = "speaker",
) -> Tuple[VADUG, dict]:
    """Compute VADUGWI coordinates for a text string.

    Pipeline:
      1. Split text into words
      2. Layer 1: classify_sentence() -- structural roles
      3. Layer 2: proximity_coefficient() -- distance-based influence
      4. Layer 3: StructureDetector().detect_all() -- chess-like patterns
      5. Physics loop: momentum + force blending
      6. Structure adjustments
      7. Personality adjustments (if provided)
      8. Clamp to 0-255

    perspective controls whose emotional state is being scored:
      - "speaker": default. "I" = self, "you" = other. Scores the speaker.
      - "listener": "you" = self, "I" = other. Scores the person being spoken to.
      - "bystander": no self. "I" and "you" are both other people. Scores a
        detached observer who sees the emotional content but takes no directed hits.

    Returns (VADUG, trace_dict) where trace_dict contains:
      - trace: list of per-word entries {word, role, coeff, v, a, d, u, g}
      - structures: list of detected StructureMatch objects
      - word_count: int
    """
    return Pipeline().run(text, perspective=perspective, personality=personality)
