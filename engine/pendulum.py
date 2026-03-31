"""Fixed Physics Layer — computes VADUG from structural analysis.

Pipeline: text -> classify words -> compute proximity -> detect structures -> apply physics -> VADUG

The physics (momentum, force application, blending) are FIXED.
The inputs come from the structural layers.
"""

from typing import List, Optional, Tuple

from .shared import VADUG, PersonalityVector
from .word_classifier import WordRole, classify_sentence, _clean
from .proximity import proximity_coefficient
from .vocabulary import VOCABULARY
from .structures import StructureDetector, StructureMatch
from .force_flow import resolve_force_flow, compute_flow_modifiers, compute_intent


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

    # Layer 0: compound phrase resolution
    # Certain word pairs form units of meaning that differ from individual words.
    # "no one" = "nobody" (zero humans). "won't stop" = "persists" (continues).
    # "shut ... up" = command to stop talking (everything between is amplifier spice).

    # Bookend compounds: opener ... closer with variable filling
    # "shut [the fuck] up" → collapse to "shutup" (command token)
    _BOOKEND_COMPOUNDS = {
        "shut": "up",      # shut [anything] up = silence command
        "get": "out",      # get [the fuck] out = expulsion command
        "fuck": "off",     # fuck [right] off = rejection command
        "back": "off",     # back [the hell] off = distance command
        "piss": "off",     # piss off = rejection
    }
    # First pass: detect bookend compounds and collapse
    # BUT: only collapse if the filling is amplifier spice, not real objects.
    # "shut the fuck up" = shut + spice + up → collapse to "shutup"
    # "shut the door" = shut + OBJECT → do NOT collapse (different meaning)
    _SPICE_WORDS = {"the", "a", "an", "fuck", "fucking", "fuckin", "damn",
                    "god", "hell", "mother", "motherfuckin", "stupid",
                    "bitch", "ass", "right", "just", "already", "up",
                    "freakin", "freaking", "effing"}
    collapsed = list(words)
    for opener, closer in _BOOKEND_COMPOUNDS.items():
        start_idx = None
        for idx, w in enumerate(collapsed):
            if w.lower() == opener:
                start_idx = idx
            elif w.lower() == closer and start_idx is not None and idx - start_idx <= 8:
                # Check filling: if ANY word between is NOT spice, abort
                # "shut the door" has "door" = not spice = abort
                filling = [collapsed[j].lower() for j in range(start_idx + 1, idx)]
                all_spice = all(f in _SPICE_WORDS for f in filling)
                if all_spice:
                    compound = opener + closer
                    collapsed = collapsed[:start_idx] + [compound] + collapsed[idx+1:]
                start_idx = None  # reset either way
                break
    words = collapsed

    _ONE_AS_QUANTIFIER = {"thing", "person", "place", "way", "reason", "time",
                          "day", "moment", "word", "chance", "step", "bit"}
    # Double negation compounds: negator + cessation = continuation
    _CONTINUATION_PAIRS = {
        ("wont", "stop"), ("wont", "quit"), ("wont", "end"),
        ("wont", "leave"), ("wont", "go"),
        ("cant", "stop"), ("cant", "quit"),
        ("dont", "stop"), ("doesnt", "stop"),
        ("never", "stop"), ("never", "stops"), ("never", "end"),
        ("never", "ends"), ("never", "quit"),
    }
    resolved = []
    i = 0
    while i < len(words):
        w_low = words[i].lower()
        next_low = words[i + 1].lower() if i + 1 < len(words) else ""
        pair = (w_low, next_low)

        if pair in _CONTINUATION_PAIRS:
            # Double negation = continuation. But direction depends on context:
            # "nightmares wont stop" (negative before) → drop pair, let negative persist
            # "dont stop believing" (positive after) → keep both words for natural parsing
            # Check: is there a strongly negative word before this pair?
            has_negative_before = any(
                VOCABULARY.get(words[j].lower(), (0,))[0] < -25
                for j in range(max(0, i - 3), i)
            )
            if has_negative_before:
                i += 2  # drop the compound, let the negative word dominate
            else:
                resolved.append(words[i])  # keep both words for normal parsing
                i += 1
        elif w_low == "no" and next_low == "one":
            next_after = words[i + 2].lower() if i + 2 < len(words) else ""
            if next_after in _ONE_AS_QUANTIFIER:
                resolved.append(words[i])
                i += 1
            else:
                resolved.append("nobody")
                i += 2
        else:
            resolved.append(words[i])
            i += 1
    words = resolved

    # Layer 1: structural roles
    roles = classify_sentence(words)

    # Layer 1.1: absence scope — mark words that describe events that AREN'T happening.
    # "havent had a panic attack" = panic and attack are ABSENT, dampen their force.
    # "without stress" = stress is ABSENT.
    _ABSENCE_MARKERS = {"without", "havent", "haven't", "hasnt", "hasn't"}
    _ABSENCE_FOLLOWERS = {"had", "been", "felt", "seen", "gotten", "experienced"}
    _PERSON_ROLES = {"SELF_REF", "OTHER_REF", "RELATION_REF"}
    absence_scope = set()  # word indices whose events are absent/didn't happen
    for i, wr in enumerate(roles):
        if wr.word in _ABSENCE_MARKERS:
            # "without" scopes the next 3 words -- BUT NOT people.
            # "without stress" = stress is absent (relief).
            # "without me" = I am excluded (NOT relief -- exclusion).
            if wr.word == "without":
                for j in range(i + 1, min(i + 4, len(roles))):
                    if roles[j].role not in _PERSON_ROLES:
                        absence_scope.add(j)
            # "havent had" scopes the next 5 words after "had"
            elif i + 1 < len(roles) and roles[i + 1].word in _ABSENCE_FOLLOWERS:
                for j in range(i + 2, min(i + 7, len(roles))):
                    absence_scope.add(j)

    # Layer 1.2: forced choice cancellation
    # "choose between food and medicine" = forced to lose one need for another.
    # When "between" splits two positive-G words, their positive V cancels out.
    # You can only keep one numerator. The other gets zeroed.
    # Food/needs + Medicine/needs = you cancel one to get the other = loss.
    _CHOICE_VERBS = {"choose", "chose", "choosing", "decide", "pick", "picked"}
    forced_choice_scope = set()
    for i, wr in enumerate(roles):
        if wr.word == "between":
            # Check if a choice verb precedes "between"
            has_choice = any(roles[j].word in _CHOICE_VERBS
                           for j in range(max(0, i - 3), i))
            if has_choice:
                # Mark words after "between" -- their positive V gets canceled
                # because the user is forced to lose one
                for j in range(i + 1, min(i + 6, len(roles))):
                    forced_choice_scope.add(j)

    # Layer 2: proximity coefficients (computed per-word in the loop)

    # Layer 3: structure detection
    detector = StructureDetector()
    structures = detector.detect_all(roles)

    # Layer 1.5: force flow — WHO does WHAT to WHOM
    force_flow = resolve_force_flow(roles)
    flow_mods = compute_flow_modifiers(force_flow)

    # ── Physics loop ────────────────────────────────────────────
    state_v = CENTER
    state_a = CENTER
    state_d = CENTER
    state_u = 0.0       # urgency starts at 0, not center
    state_g = CENTER
    state_w = CENTER     # self-worth starts at neutral

    trace_entries: List[dict] = []

    for i, wr in enumerate(roles):
        # Every word in vocabulary exerts force, not just EMOTIONAL role.
        # A small star (monday V=-10) still pulls. Just less than a big star (dead V=-114).
        # Dark matter words (not in vocab) have no force and skip.
        # POSSESSION words are objects -- "my remote" = thing, not emotion.
        # Strip V/A/D but KEEP gravity (G). A car matters more than a pen.
        # "he stole my car" = high G amplifies damage. "he stole my pen" = low G.
        if wr.role == "POSSESSION":
            vf = VOCABULARY.get(wr.word)
            if vf:
                word_force = (0, 0, 0, 0, max(5, vf[4]))  # zero emotions, keep G
            else:
                word_force = (0, 0, 0, 0, 5)  # unknown object = minimal G
        else:
            word_force = wr.force
            if word_force is None:
                # Check vocabulary directly - word might have force but role != EMOTIONAL
                word_force = VOCABULARY.get(wr.word)
            if word_force is None:
                # Fuzzy match: stemmer catches conjugations (hates->hate, kills->kill)
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

        # Forced choice cancellation: "choose between food and medicine"
        # Both are positive needs but you can only keep one. Cancel the V.
        # The G stays -- the weight of what you're losing is still real.
        if i in forced_choice_scope and dv > 0:
            dv = -dv  # flip positive to negative -- you're LOSING this thing

        # Absence scope: words describing events that DIDN'T happen get
        # their negative force dampened. "Havent had a PANIC ATTACK" =
        # panic and attack are absent, their V force is reduced.
        if i in absence_scope and dv < -10:
            dv = int(dv * 0.2)  # 80% reduction -- the event isn't real
            da = int(da * 0.3)
            dd = int(dd * 0.3)

        # "Without" as pure operator: when followed by words in absence scope,
        # "without" is just the removal operator, not an emotional word.
        # "Without stopping" = operator + absent action. "Without" has no own charge.
        if wr.word in ("without",) and any(j in absence_scope for j in range(i+1, min(i+4, len(roles)))):
            dv = 0
            da = 0
            dd = 0

        # Apply force flow direction modifiers
        # "He hit me" (other→self) amplifies negative V, drops D
        # "I am stupid" (self→self) amplifies W drop
        if i == (force_flow.force_idx if force_flow else -1):
            dv = int(dv * flow_mods["v_mod"])
            dd = int(dd * flow_mods["d_mod"])
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

        # W (self-worth): when SELF_REF is nearby, the word's valence
        # hits self-worth too. "I am stupid" = V drops AND W drops.
        # "I am proud" = V rises AND W rises. No SELF_REF nearby = W unchanged.
        self_ref_nearby = any(
            roles[j].role == "SELF_REF" and abs(j - i) <= 4
            for j in range(max(0, i - 4), min(len(roles), i + 5))
            if j != i
        )
        if self_ref_nearby and dv != 0:
            w_damp = 0.7  # self-worth tracks valence but slightly dampened
            # Force flow amplifies self-directed W changes
            w_flow = flow_mods["w_mod"] if force_flow and i == force_flow.force_idx else 1.0
            w_effective = dv * coeff * FORCE_SCALE * w_damp * w_flow
            target_w = CENTER + w_effective
            push_w = push_strength * (1.0 if dv * coeff >= 0 else -1.0) * abs(dv) * FORCE_SCALE * w_damp * w_flow
            state_w = state_w * MOMENTUM + target_w * inv_m + push_w

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

    # ── Structure adjustments ───────────────────────────────────
    for sm in structures:
        # Sarcasm and bravado scale with distance from neutral
        # Bigger positive = stronger correction (overcompensation = more mask)
        if sm.pattern in ("SARCASM_INVERSION", "BRAVADO", "DIRECTED_POSITIVE", "EXCLUDED_POSITIVE") and state_v > CENTER:
            excess = state_v - CENTER
            pull = sm.v_weight * sm.confidence * FORCE_SCALE * (1.0 + excess / 50.0)
            state_v += pull
        elif sm.pattern == "CHOPPER_SPLIT" and sm.matched_indices:
            # "but" means: second half overrides first half.
            # Check emotional content after the chop to determine direction.
            chop_pos = sm.matched_indices[0]
            after_words = [wr for wr in roles if wr.position > chop_pos]
            after_v_sum = 0
            for wr in after_words:
                wf = wr.force or VOCABULARY.get(wr.word)
                if wf:
                    after_v_sum += wf[0]
            has_negator_after = any(wr.role == "NEGATOR" for wr in after_words)
            # Second half contradicts first half -- pull toward opposite
            if (state_v > CENTER and (after_v_sum < 0 or (after_v_sum == 0 and has_negator_after))):
                # Positive first half + negative/negated second = override hard
                # The stronger the first half positive, the more the "but" hurts
                distance = state_v - CENTER
                state_v -= distance * 1.5 * sm.confidence
            elif (state_v < CENTER and after_v_sum > 10):
                # Negative first half + positive second = recovery
                distance = CENTER - state_v
                state_v += distance * 0.4 * sm.confidence
        else:
            state_v += sm.v_weight * sm.confidence * FORCE_SCALE
        state_d += sm.d_weight * sm.confidence * FORCE_SCALE
        state_u += sm.u_weight * sm.confidence * FORCE_SCALE
        state_g += sm.g_weight * sm.confidence * FORCE_SCALE
        state_w += sm.w_weight * sm.confidence * FORCE_SCALE

    # ── W coefficient on V ─────────────────────────────────────
    # Self-worth is the lens: low W dampens positive, amplifies negative.
    # At W=128 (neutral), w_factor=1.0 -- no change to existing behavior.
    # At W=64, positive V halved, negative V doubled.
    # At W=0, positive V zeroed, negative V maximally amplified.
    w_factor = max(0.0, state_w) / CENTER  # 0.0 to ~2.0, 1.0 at center
    if state_v > CENTER:
        state_v = CENTER + (state_v - CENTER) * min(w_factor, 1.0)
    elif state_v < CENTER:
        state_v = CENTER + (state_v - CENTER) * max(1.0, 2.0 - w_factor)

    # ── Personality adjustments ─────────────────────────────────
    if personality is not None:
        sensitivity = personality.emotional_sensitivity
        # Scale deviation from center by sensitivity
        state_v = CENTER + (state_v - CENTER) * sensitivity
        state_a = CENTER + (state_a - CENTER) * sensitivity
        state_d = CENTER + (state_d - CENTER) * sensitivity + personality.dominance_baseline
        state_u = state_u * sensitivity
        state_g = CENTER + (state_g - CENTER) * sensitivity + personality.gravity_bias
        state_w = CENTER + (state_w - CENTER) * sensitivity

    # ── Intent (I) computation ────────────────────────────────
    # Intent = WHERE is the force aimed and WHY.
    # Computed from force flow direction + structural cues.
    state_i = compute_intent(force_flow, roles)

    # ── Clamp to 0-255 ─────────────────────────────────────────
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
        "trace": trace_entries,
        "structures": structures,
        "force_flow": force_flow,
        "word_count": len(words),
    }

    return result, trace_dict
