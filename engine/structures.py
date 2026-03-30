"""V3 Layer 3: Structure Detector — chess-like pattern recognition.

This is the CHESS PLAYER. It reads role sequences and recognizes patterns
-- "checkmate conditions." Each emotional pattern has a structural
requirement that multiple word combinations satisfy. Like a chess player
seeing the Queen's Gambit from piece positions, not memorized move
sequences.

CRITICAL: NO hardcoded word lists. Patterns are defined by ROLE
relationships from word_classifier.py and proximity pairs from
proximity.py. The few word-level checks (apology words, blanket words,
sustain verbs, exit concepts, null words) are inline semantic checks,
not vocabulary dictionaries.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .word_classifier import WordRole, classify_sentence
from .proximity import find_role_pairs, PROXIMITY_DECAY


# ── Result dataclass ─────────────────────────────────────────────

@dataclass
class StructureMatch:
    """A detected structural pattern with confidence and VADUG weights."""
    pattern: str            # FAREWELL, METHOD_ACQUISITION, etc.
    confidence: float       # 0.0-1.0
    matched_indices: list   # which word positions matched
    description: str        # human-readable
    v_weight: float = 0.0   # how this structure shifts V
    d_weight: float = 0.0
    u_weight: float = 0.0
    g_weight: float = 0.0


# ── Inline semantic word sets (not role dictionaries) ────────────
# These are tiny checks on actual word text, used inside detectors
# where role classification alone is insufficient.

_APOLOGY_WORDS = frozenset({"sorry", "apologize", "apologise", "apologies"})
_BLANKET_WORDS = frozenset({
    "everything", "everyone", "everybody", "all", "always",
    "nothing", "nobody",
})
_SUSTAIN_VERBS = frozenset({
    "take", "do", "keep", "bear", "stand", "handle", "live",
    "cope", "manage", "endure", "deal", "continue", "go",
})
_EXIT_CONCEPTS = frozenset({
    "hope", "way", "escape", "point", "future", "reason",
    "purpose", "option", "options", "choice", "out", "answer",
    "solution", "help",
})
_NULL_WORDS = frozenset({
    "nothing", "worthless", "useless", "burden", "waste",
    "zero", "empty", "pointless", "meaningless", "invisible",
    "broken", "failure", "trash", "garbage",
})
_COMPARISON_WORDS = frozenset({
    "better", "happier", "easier", "safer", "freer",
    "improved", "relieved",
})
_CONDITIONAL_WORDS = frozenset({
    "without", "if", "unless", "except", "when",
})


# ── Structure Detector ───────────────────────────────────────────

class StructureDetector:
    """Detects structural emotional patterns from role sequences.

    Each detector is a private method that examines role patterns,
    proximity pairs, and (minimally) word text to find "checkmate
    conditions" -- structural configurations that indicate specific
    emotional states.
    """

    def detect_all(self, roles: List[WordRole]) -> List[StructureMatch]:
        """Run all detectors, return matches with confidence > 0.3."""
        detectors = [
            self._farewell,
            self._method_acquisition,
            self._finality,
            self._blanket_apology,
            self._self_removal,
            self._suspicious_calm,
            self._exhaustion,
            self._no_exit,
            self._self_nullify,
            self._sarcasm_inversion,
            self._chopper_split,
        ]
        matches = []
        for detector in detectors:
            result = detector(roles)
            if result is not None and result.confidence > 0.3:
                matches.append(result)
        return matches

    # ── Individual detectors ─────────────────────────────────────

    def _farewell(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """TRANSFER + POSSESSION + (RELATION_REF or OTHER_REF) nearby.

        "I gave my dog to my neighbor" -- giving away before exit.
        """
        pairs = find_role_pairs(roles, "TRANSFER", "POSSESSION")
        if not pairs:
            return None

        # Check for RELATION_REF or OTHER_REF nearby
        ref_indices = [
            r.position for r in roles
            if r.role in ("RELATION_REF", "OTHER_REF")
        ]
        if not ref_indices:
            return None

        # Find best pair with a ref nearby
        for t_idx, p_idx, strength in pairs:
            for ref_idx in ref_indices:
                # ref should be within 8 words of either transfer or possession
                # (longer sentences like "left keys on counter for whoever finds them")
                dist_t = abs(ref_idx - t_idx)
                dist_p = abs(ref_idx - p_idx)
                if min(dist_t, dist_p) <= 8:
                    indices = sorted({t_idx, p_idx, ref_idx})
                    confidence = strength * 0.8
                    return StructureMatch(
                        pattern="FAREWELL",
                        confidence=min(confidence + 0.2, 1.0),
                        matched_indices=indices,
                        description="Giving away possessions to someone",
                        v_weight=-30.0,
                        d_weight=-20.0,
                        u_weight=40.0,
                        g_weight=50.0,
                    )
        return None

    def _method_acquisition(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """ACQUIRE + METHOD = obtaining means.

        "just bought some pills" -- acquiring method.
        """
        pairs = find_role_pairs(roles, "ACQUIRE", "METHOD")
        if not pairs:
            return None

        t_idx, m_idx, strength = pairs[0]  # strongest pair
        return StructureMatch(
            pattern="METHOD_ACQUISITION",
            confidence=min(strength + 0.3, 1.0),
            matched_indices=[t_idx, m_idx],
            description="Acquiring method or means",
            v_weight=-40.0,
            d_weight=-10.0,
            u_weight=50.0,
            g_weight=60.0,
        )

    def _finality(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """FINALITY role present, optionally + TEMPORAL or SELF_REF.

        "this is the last time you'll hear from me" -- closing frame.
        """
        finality_indices = [r.position for r in roles if r.role == "FINALITY"]
        if not finality_indices:
            return None

        temporal_indices = [r.position for r in roles if r.role == "TEMPORAL"]
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]

        indices = list(finality_indices)
        confidence = 0.4

        # Boost if TEMPORAL or SELF_REF nearby
        for fi in finality_indices:
            for ti in temporal_indices:
                if abs(fi - ti) <= 4:
                    indices.append(ti)
                    confidence += 0.15
            for si in self_indices:
                if abs(fi - si) <= 5:
                    indices.append(si)
                    confidence += 0.15

        return StructureMatch(
            pattern="FINALITY",
            confidence=min(confidence, 1.0),
            matched_indices=sorted(set(indices)),
            description="Closing or final frame",
            v_weight=-20.0,
            d_weight=-15.0,
            u_weight=30.0,
            g_weight=40.0,
        )

    def _blanket_apology(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Apology words near BLANKET words (everything/everyone/all).

        "im sorry for everything" != "im sorry for being late."
        """
        apology_indices = [
            r.position for r in roles if r.word in _APOLOGY_WORDS
        ]
        blanket_indices = [
            r.position for r in roles if r.word in _BLANKET_WORDS
        ]

        if not apology_indices or not blanket_indices:
            return None

        # Find closest apology-blanket pair
        best_dist = 999
        best_a, best_b = -1, -1
        for ai in apology_indices:
            for bi in blanket_indices:
                d = abs(ai - bi)
                if d < best_dist:
                    best_dist = d
                    best_a, best_b = ai, bi

        if best_dist > 6:
            return None

        strength = PROXIMITY_DECAY ** best_dist
        return StructureMatch(
            pattern="BLANKET_APOLOGY",
            confidence=min(strength + 0.3, 1.0),
            matched_indices=sorted({best_a, best_b}),
            description="Blanket apology covering everything/everyone",
            v_weight=-25.0,
            d_weight=-20.0,
            u_weight=35.0,
            g_weight=45.0,
        )

    def _self_removal(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Comparison + conditional + SELF_REF = calculating removal.

        "they would be happier if I wasnt here" -- user calculating
        that removing self improves others.
        """
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]
        comparison_indices = [
            r.position for r in roles if r.word in _COMPARISON_WORDS
        ]
        conditional_indices = [
            r.position for r in roles if r.word in _CONDITIONAL_WORDS
        ]

        if not self_indices or not comparison_indices or not conditional_indices:
            return None

        # All three elements present -- find best cluster
        indices = []
        best_comp = comparison_indices[0]
        best_cond = min(conditional_indices, key=lambda x: abs(x - best_comp))
        best_self = min(self_indices, key=lambda x: abs(x - best_cond))

        indices = sorted({best_comp, best_cond, best_self})
        span = max(indices) - min(indices)
        if span > 8:
            return None

        confidence = max(0.5, 1.0 - span * 0.05)
        return StructureMatch(
            pattern="SELF_REMOVAL",
            confidence=min(confidence, 1.0),
            matched_indices=indices,
            description="Calculating that self-removal improves others",
            v_weight=-35.0,
            d_weight=-25.0,
            u_weight=45.0,
            g_weight=55.0,
        )

    def _suspicious_calm(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """PEACE role + word "finally" = decision made, suspiciously calm.

        "I finally feel at peace" -- resolved calm after struggle.
        """
        peace_indices = [r.position for r in roles if r.role == "PEACE"]
        finally_indices = [
            r.position for r in roles if r.word == "finally"
        ]

        if not peace_indices or not finally_indices:
            return None

        # Find closest pair
        best_dist = 999
        best_p, best_f = -1, -1
        for pi in peace_indices:
            for fi in finally_indices:
                d = abs(pi - fi)
                if d < best_dist:
                    best_dist = d
                    best_p, best_f = pi, fi

        if best_dist > 6:
            return None

        strength = PROXIMITY_DECAY ** best_dist
        return StructureMatch(
            pattern="SUSPICIOUS_CALM",
            confidence=min(strength + 0.3, 1.0),
            matched_indices=sorted({best_p, best_f}),
            description="Suspiciously calm -- decision already made",
            v_weight=-10.0,
            d_weight=10.0,
            u_weight=40.0,
            g_weight=50.0,
        )

    def _exhaustion(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """NEGATOR + sustain verb + optional TEMPORAL("anymore"/"forever").

        "I cant take this anymore" -- user at processing limits.
        "cant do this shit anymore" -- implied self-ref via contraction.

        SELF_REF is a boost, not a requirement. Contractions like "cant",
        "dont", "wont" inherently imply the speaker.
        """
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]
        negator_indices = [r.position for r in roles if r.role == "NEGATOR"]
        sustain_indices = [
            r.position for r in roles if r.word in _SUSTAIN_VERBS
        ]
        temporal_limit_indices = [
            r.position for r in roles
            if r.word in ("anymore", "forever", "longer")
            or (r.role == "TEMPORAL" and r.word in ("anymore", "forever"))
        ]

        if not negator_indices or not sustain_indices:
            return None

        # Core pattern: NEGATOR + sustain verb (SELF_REF and temporal are boosts)
        indices = set()
        best_neg = negator_indices[0]
        best_sustain = min(sustain_indices, key=lambda x: abs(x - best_neg))
        indices.update({best_neg, best_sustain})

        if self_indices:
            best_self = min(self_indices, key=lambda x: abs(x - best_neg))
            indices.add(best_self)

        span = max(indices) - min(indices)
        if span > 6:
            return None

        # Base confidence: lower without explicit SELF_REF
        confidence = 0.5 if self_indices else 0.35
        if temporal_limit_indices:
            best_temp = min(
                temporal_limit_indices,
                key=lambda x: abs(x - best_sustain),
            )
            if abs(best_temp - best_sustain) <= 4:
                indices.add(best_temp)
                confidence += 0.25
        if self_indices:
            confidence += 0.1

        return StructureMatch(
            pattern="EXHAUSTION",
            confidence=min(confidence, 1.0),
            matched_indices=sorted(indices),
            description="User at processing limits -- cannot sustain",
            v_weight=-30.0,
            d_weight=-30.0,
            u_weight=35.0,
            g_weight=40.0,
        )

    def _no_exit(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """NEGATOR + exit concept words = zero paths forward.

        "there is no hope" -- no exit visible.
        """
        negator_indices = [r.position for r in roles if r.role == "NEGATOR"]
        exit_indices = [
            r.position for r in roles if r.word in _EXIT_CONCEPTS
        ]

        if not negator_indices or not exit_indices:
            return None

        # Find closest negator-exit pair
        best_dist = 999
        best_n, best_e = -1, -1
        for ni in negator_indices:
            for ei in exit_indices:
                d = abs(ni - ei)
                if d < best_dist:
                    best_dist = d
                    best_n, best_e = ni, ei

        if best_dist > 4:
            return None

        strength = PROXIMITY_DECAY ** best_dist
        return StructureMatch(
            pattern="NO_EXIT",
            confidence=min(strength + 0.2, 1.0),
            matched_indices=sorted({best_n, best_e}),
            description="No paths forward visible",
            v_weight=-35.0,
            d_weight=-30.0,
            u_weight=40.0,
            g_weight=50.0,
        )

    def _self_nullify(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + null words = user calculating self as zero.

        "I am nothing" -- self-nullification.
        """
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]
        null_indices = [
            r.position for r in roles if r.word in _NULL_WORDS
        ]

        if not self_indices or not null_indices:
            return None

        # Find closest self-null pair
        best_dist = 999
        best_s, best_n = -1, -1
        for si in self_indices:
            for ni in null_indices:
                d = abs(si - ni)
                if d < best_dist:
                    best_dist = d
                    best_s, best_n = si, ni

        if best_dist > 5:
            return None

        strength = PROXIMITY_DECAY ** best_dist
        return StructureMatch(
            pattern="SELF_NULLIFY",
            confidence=min(strength + 0.3, 1.0),
            matched_indices=sorted({best_s, best_n}),
            description="User calculating self as zero",
            v_weight=-40.0,
            d_weight=-35.0,
            u_weight=30.0,
            g_weight=45.0,
        )

    def _sarcasm_inversion(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Positive EMOTIONAL words (force[0] > 30) near mundane/negative context.

        "oh great another monday" -- signal says X, meaning is NOT X.
        The checkmate: positive word surrounded by neutral/negative context.
        """
        positive_emotional = [
            r for r in roles
            if r.role == "EMOTIONAL" and r.force and r.force[0] > 15
        ]
        if not positive_emotional:
            return None

        # Check if surrounded by mundane/negative context
        for emo in positive_emotional:
            idx = emo.position
            # Look at neighbors within 3 words
            context_roles = [
                r for r in roles
                if r.position != idx and abs(r.position - idx) <= 3
            ]
            if not context_roles:
                continue

            # Count how many neighbors are neutral/filler/negative-emotional
            mundane_count = 0
            for cr in context_roles:
                if cr.role in ("NEUTRAL", "FILLER"):
                    mundane_count += 1
                elif cr.role == "TEMPORAL":
                    mundane_count += 1
                elif (cr.role == "EMOTIONAL" and cr.force
                      and cr.force[0] < -10):
                    mundane_count += 1

            # If most context is mundane, it's likely sarcastic
            if mundane_count >= len(context_roles) * 0.5 and mundane_count >= 1:
                indices = [idx] + [cr.position for cr in context_roles]
                return StructureMatch(
                    pattern="SARCASM_INVERSION",
                    confidence=0.6,
                    matched_indices=sorted(set(indices)),
                    description="Positive word in mundane/negative context -- likely sarcasm",
                    v_weight=0.0,
                    d_weight=10.0,
                    u_weight=5.0,
                    g_weight=5.0,
                )
        return None

    def _chopper_split(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """CHOPPER role present = sentence split, second half overrides.

        "I was fine but now everything hurts" -- "but" chops, second
        half is the real message.
        """
        chopper_indices = [r.position for r in roles if r.role == "CHOPPER"]
        if not chopper_indices:
            return None

        # Use the first chopper as the split point
        chop_idx = chopper_indices[0]
        total = len(roles)

        # Only meaningful if there's content on both sides
        if chop_idx < 1 or chop_idx >= total - 1:
            return None

        return StructureMatch(
            pattern="CHOPPER_SPLIT",
            confidence=0.7,
            matched_indices=[chop_idx],
            description=f"Sentence split at position {chop_idx} -- second half overrides",
            v_weight=0.0,
            d_weight=0.0,
            u_weight=5.0,
            g_weight=5.0,
        )
