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
    """A detected structural pattern with confidence and VADUGW weights."""
    pattern: str            # FAREWELL, METHOD_ACQUISITION, etc.
    confidence: float       # 0.0-1.0
    matched_indices: list   # which word positions matched
    description: str        # human-readable
    v_weight: float = 0.0   # how this structure shifts V
    d_weight: float = 0.0
    u_weight: float = 0.0
    g_weight: float = 0.0
    w_weight: float = 0.0   # how this structure shifts W (self-worth)


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
    "pathetic", "stupid", "idiot", "dumb", "incompetent",
    "joke", "loser", "weak", "defective", "inadequate",
    "problem", "mistake", "disgrace", "embarrassment",
})
# Compound phrases where the user's mass becomes friction/obstruction.
# Individual words are neutral; the phrase is the unit of meaning.
# "in the way" = my mass blocks. "dead weight" = my mass drags.
_OBSTRUCTION_COMPOUNDS = [
    ("in", "the", "way"),
    ("dead", "weight"),
    ("holding", "back"),
    ("dragging", "down"),
    ("slowing", "down"),
    ("in", "the", "road"),
    ("a", "hindrance"),
    ("an", "obstacle"),
]
# Words that describe self-as-negative-mass when SELF_REF is subject.
# "I am the burden" -- burden pulls the user's own weight negative.
# These are self-describing drag words: the user applies them to themselves.
_SELF_DRAG_WORDS = frozenset({
    "burden", "obstacle", "hindrance", "nuisance", "problem",
    "liability", "deadweight", "baggage", "anchor", "drag",
    "inconvenience", "bother", "pest", "parasite", "leech",
})
_COMPARISON_WORDS = frozenset({
    "better", "happier", "easier", "safer", "freer",
    "improved", "relieved",
})
_BETRAYAL_VERBS = frozenset({
    "cheated", "cheating", "cheat", "betrayed", "betraying", "betray",
    "lied", "lying", "lie", "deceived", "deceiving", "deceive",
    "backstabbed", "backstabbing",
})
# Compound betrayal phrases: word pairs that form polarity reversal
# "turned on me" = was pulled toward, then reversed
_BETRAYAL_COMPOUNDS = {
    "turned": {"on"},
    "went": {"against"},
    "sided": {"against"},
    "ganged": {"up"},
}
_INTERROGATION_WORDS = frozenset({
    "why", "how", "where",
})
_CONDITIONAL_WORDS = frozenset({
    "without", "if", "unless", "except", "when",
})
_LAUGHTER_WORDS = frozenset({
    "haha", "hahaha", "lol", "lmao", "rofl", "lmfao",
    "ha", "heh", "hehe",
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
            self._pull_toward_method, self._fleeing,
            self._power_over_self, self._self_submission, self._d_inversion,
            self._betrayal,
            self._bravado,
            self._victimization,
            self._calling_out,
            self._directed_positive,
            self._minimizer,
            self._excluded_positive,
            self._relief_absence,
            self._self_excluded,
            self._withheld_positive,
        ]
        matches = []
        for detector in detectors:
            result = detector(roles)
            if result is not None and result.confidence > 0.3:
                matches.append(result)
        return matches

    # ── Individual detectors ─────────────────────────────────────

    def _farewell(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """TRANSFER + (POSSESSION or RELATION_REF) + recipient nearby.

        "I gave my dog to my neighbor" -- giving away before exit.
        Dog is RELATION_REF (relationship), neighbor is RELATION_REF (recipient).
        """
        # "back" flips direction: "give back my stuff" = reclamation, not farewell
        if any(r.word == "back" for r in roles):
            return None

        # Find TRANSFER + POSSESSION pairs
        pairs = find_role_pairs(roles, "TRANSFER", "POSSESSION")

        # Also check TRANSFER near RELATION_REF (dog/cat are relationships now)
        if not pairs:
            transfer_idx = [r.position for r in roles if r.role == "TRANSFER"]
            rel_idx = [r.position for r in roles if r.role == "RELATION_REF"]
            if not transfer_idx or len(rel_idx) < 2:
                return None
            # Need at least 2 RELATION_REFs (thing + recipient)
            t = transfer_idx[0]
            nearby = [ri for ri in rel_idx if abs(ri - t) <= 8]
            if len(nearby) < 2:
                return None
            strength = PROXIMITY_DECAY ** abs(nearby[0] - t)
            indices = sorted(set([t] + nearby))
            confidence = strength * 0.8
            return StructureMatch(
                pattern="FAREWELL",
                confidence=min(confidence + 0.2, 1.0),
                matched_indices=indices,
                description="Giving away relationships/possessions to someone",
                v_weight=-30.0,
                d_weight=-20.0,
                u_weight=40.0,
                g_weight=50.0,
                w_weight=-10.0,
            )

        # Original path: TRANSFER + POSSESSION + nearby ref
        ref_indices = [
            r.position for r in roles
            if r.role in ("RELATION_REF", "OTHER_REF")
        ]
        if not ref_indices:
            return None

        for t_idx, p_idx, strength in pairs:
            for ref_idx in ref_indices:
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
                        w_weight=-10.0,
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
            w_weight=-15.0,
        )

    def _finality(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """FINALITY role present, optionally + TEMPORAL or SELF_REF.

        "this is the last time you'll hear from me" -- closing frame.
        """
        finality_indices = [r.position for r in roles if r.role == "FINALITY"]
        if not finality_indices:
            return None

        # Exclude achievement contexts: "I finished" = completed, not closing
        achievement_words = {"finished", "completed", "accomplished", "graduated"}
        if any(r.word in achievement_words for r in roles if r.role == "FINALITY"):
            # Only fire if there's also a negative/closing signal
            has_negative = any(r.force and r.force[0] < -15 for r in roles if r.role == "EMOTIONAL")
            has_other_ref = any(r.role == "OTHER_REF" for r in roles)
            if not has_negative and not has_other_ref:
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
            w_weight=-10.0,
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
            w_weight=-25.0,
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
            w_weight=-35.0,
        )

    def _suspicious_calm(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """PEACE role + word "finally" = decision made, suspiciously calm.

        "I finally feel at peace" -- resolved calm after struggle.
        """
        peace_indices = [r.position for r in roles if r.role == "PEACE"]
        # Only fire on DECISION words -- the person made a choice about their state.
        # "decided", "accepted", "settled" = resolution. These carry finality.
        # "finally", "ready", "now" are too common standalone -- need conversation context.
        _CALM_DECISION = {"decided", "accepted", "settled", "resolved", "chosen"}
        decision_indices = [
            r.position for r in roles if r.word in _CALM_DECISION
        ]

        if not peace_indices or not decision_indices:
            return None

        # Exclude achievement/resilience contexts
        has_acquire = any(r.role == "ACQUIRE" for r in roles)
        _POSITIVE_ACTION = {"got", "received", "earned", "won", "passed", "made",
                           "achieved", "try", "trying", "start", "starting",
                           "again", "learn", "learning"}
        has_positive_verb = any(r.word in _POSITIVE_ACTION for r in roles)
        if has_acquire or has_positive_verb:
            return None

        # Find closest pair
        best_dist = 999
        best_p, best_f = -1, -1
        for pi in peace_indices:
            for fi in decision_indices:
                d = abs(pi - fi)
                if d < best_dist:
                    best_dist = d
                    best_p, best_f = pi, fi

        if best_dist > 6:
            return None

        strength = PROXIMITY_DECAY ** best_dist

        # "finally" + peace = breakthrough/relief as a standalone sentence.
        # Only suspicious in conversation context with prior crisis signals.
        # As a single sentence, "i finally feel at peace" = positive.
        has_finally = any(r.word == "finally" for r in roles)
        if has_finally:
            return None  # "finally at peace" = relief, not suspicious

        return StructureMatch(
            pattern="SUSPICIOUS_CALM",
            confidence=min(strength + 0.3, 1.0),
            matched_indices=sorted({best_p, best_f}),
            description="Suspiciously calm -- decision already made",
            v_weight=-40.0,
            d_weight=10.0,
            u_weight=40.0,
            g_weight=50.0,
            w_weight=-5.0,
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
            w_weight=-10.0,
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
            w_weight=-15.0,
        )

    def _self_nullify(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + null words OR obstruction compounds = self as zero/friction.

        "I am nothing" -- self-nullification (null word).
        "I'm in the way" -- self-as-obstruction (compound phrase).
        "I'm dead weight" -- self-as-negative-mass (compound phrase).

        The user's gravity inverts: their mass goes from contribution to friction.
        """
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]
        if not self_indices:
            return None

        words = [r.word for r in roles]

        # Strategy 1: null words near SELF_REF
        null_indices = [
            r.position for r in roles if r.word in _NULL_WORDS
        ]

        # Strategy 2: compound obstruction phrases near SELF_REF
        # "in the way", "dead weight", "holding back" etc.
        compound_indices = []
        for compound in _OBSTRUCTION_COMPOUNDS:
            clen = len(compound)
            for start in range(len(words) - clen + 1):
                if tuple(words[start:start + clen]) == compound:
                    compound_indices.append(start)

        all_match_indices = null_indices + compound_indices
        if not all_match_indices:
            return None

        # Find closest self-match pair
        best_dist = 999
        best_s, best_n = -1, -1
        for si in self_indices:
            for ni in all_match_indices:
                d = abs(si - ni)
                if d < best_dist:
                    best_dist = d
                    best_s, best_n = si, ni

        if best_dist > 5:
            return None

        strength = PROXIMITY_DECAY ** best_dist
        confidence = min(strength + 0.3, 1.0)
        w_penalty = -40.0

        # Conditional self-worth: "i am nothing WITHOUT YOU"
        # The user's worth is stated to depend entirely on the relationship.
        # Without the anchor, self = zero. This is worse than plain nullification
        # because it reveals no independent foundation.
        has_without = any(r.word == "without" for r in roles)
        has_other = any(r.role in ("OTHER_REF", "RELATION_REF") for r in roles)
        if has_without and has_other:
            w_penalty = -60.0  # conditional worth = deeper W hit
            confidence = min(confidence + 0.1, 1.0)

        # Absent target nuke: "im a burden" with NO target = broadcast to ALL.
        # No OTHER_REF or RELATION_REF = the user didn't scope it.
        # The absence of a target amplifies to all relationships.
        # "im a burden to my mom" = scoped (has RELATION). Less severe.
        # "im a burden" = unscoped. Nuclear. Everyone.
        if not has_other and not has_without:
            # No target named, no "without" conditional = universal self-negation
            w_penalty = -50.0  # worse than targeted, less than conditional
            confidence = min(confidence + 0.05, 1.0)

        return StructureMatch(
            pattern="SELF_NULLIFY",
            confidence=confidence,
            matched_indices=sorted({best_s, best_n}),
            description="User calculating self as zero or obstruction",
            v_weight=-40.0,
            d_weight=-35.0,
            u_weight=30.0,
            g_weight=45.0,
            w_weight=w_penalty,
        )

    def _sarcasm_inversion(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Positive EMOTIONAL near SPECIFICALLY mundane/negative = output != intent.

        Requires BOTH:
          1. A positive emotional word (dV > 25)
          2. A sarcasm signal: ironic opener (oh, sure, yeah, right, clearly)
             OR a specifically mundane word (monday, meeting, work, traffic)
             OR a negative emotional word nearby

        "oh great another monday" = opener + positive + mundane = sarcasm
        "great job on the presentation" = positive only = NOT sarcasm
        "I love my mom" = positive + relation = NOT sarcasm
        """
        positive_idx = [i for i, r in enumerate(roles)
                       if r.role == "EMOTIONAL" and r.force and r.force[0] >= 25]
        if not positive_idx:
            return None

        negative_idx = [i for i, r in enumerate(roles)
                       if r.role == "EMOTIONAL" and r.force and r.force[0] < -25]

        mundane_words = {"monday", "meeting", "work", "homework", "traffic",
                         "redo", "again", "another", "same", "overtime",
                         "bills", "chores", "commute", "deadline"}
        mundane_idx = [i for i, r in enumerate(roles) if r.word in mundane_words]

        sarcasm_openers = {"oh", "wow", "clearly"}
        # "sure", "yeah", "right" removed -- too common as genuine agreement
        # "sure I love this" = genuine. "oh I love this" = sarcastic.
        has_opener = any(r.word in sarcasm_openers for r in roles[:3])

        # Strong positive word LEADING the sentence + negative following = sarcasm
        # "love being ignored" -- love(+60) leads, ignored(-35) follows
        strong_positive_leads = (len(positive_idx) > 0 and positive_idx[0] <= 1
                                  and any(r.force and r.force[0] >= 40 for r in roles[:2]
                                          if r.role == "EMOTIONAL"))

        has_mundane = len(mundane_idx) > 0
        has_negative = len(negative_idx) > 0

        if (has_mundane or has_negative) and (has_opener or strong_positive_leads):
            return StructureMatch(
                pattern="SARCASM_INVERSION",
                confidence=0.8,
                matched_indices=sorted(set(positive_idx + negative_idx + mundane_idx)),
                description="Opener + positive + mundane/negative = sarcasm",
                v_weight=-30.0, d_weight=10.0,
            )
        elif has_mundane and not has_opener:
            return StructureMatch(
                pattern="SARCASM_INVERSION",
                confidence=0.5,
                matched_indices=sorted(set(positive_idx + mundane_idx)),
                description="Positive + mundane (no opener, lower confidence)",
                v_weight=-20.0, d_weight=5.0,
            )
        # SHORT sentence with opener + strong positive = compressed sarcasm
        # "oh joy" "oh perfect" "oh wonderful" "wow great" -- brevity IS the tell
        # Genuine joy would have more context: "oh what joy this brings"
        elif has_opener and len(roles) <= 5:
            # Check for strong positive (V >= 40)
            strong_pos = any(r.force and r.force[0] >= 40
                           for r in roles if r.role == "EMOTIONAL")
            if strong_pos:
                return StructureMatch(
                    pattern="SARCASM_INVERSION",
                    confidence=0.9,
                    matched_indices=positive_idx,
                    description="Opener + strong positive + short = compressed sarcasm",
                    v_weight=-60.0, d_weight=10.0,
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

    def _pull_toward_method(self, roles):
        """PULL_TOWARD + METHOD = chasing/acquiring dangerous object."""
        pairs = find_role_pairs(roles, "PULL_TOWARD", "METHOD", max_distance=5)
        if not pairs:
            # Also check ACQUIRE (already covered but belt and suspenders)
            return None
        return StructureMatch(
            pattern="PURSUIT_OF_METHOD",
            confidence=0.8,
            matched_indices=[pairs[0][0], pairs[0][1]],
            description="Chasing/pursuing a method object",
            v_weight=-50.0, u_weight=35.0, g_weight=-25.0,
            w_weight=-15.0,
        )

    def _fleeing(self, roles):
        """PULL_AWAY from self/relationships = distancing/isolation.

        Does NOT fire when achievement context is present:
        "i ran my first mile" = exercise, not fleeing.
        """
        has_flee = any(r.role == "PULL_AWAY" for r in roles)
        has_self = any(r.role == "SELF_REF" for r in roles)
        has_relation = any(r.role == "RELATION_REF" for r in roles)
        if not has_flee or not (has_relation or has_self):
            return None

        # Achievement context blocks FLEEING
        _ACHIEVEMENT_CONTEXT = {"first", "mile", "miles", "marathon", "race",
                                "finish", "finished", "record", "fastest",
                                "goal", "lap", "laps", "training", "workout"}
        has_achievement = any(r.word in _ACHIEVEMENT_CONTEXT for r in roles)
        if has_achievement:
            return None

        return StructureMatch(
            pattern="FLEEING",
            confidence=0.6,
            matched_indices=[i for i, r in enumerate(roles)
                            if r.role in ("PULL_AWAY", "SELF_REF", "RELATION_REF")],
            description="Fleeing from self/relationships",
            v_weight=-25.0, d_weight=-15.0, u_weight=15.0,
        )
        return None

    def _power_over_self(self, roles):
        """Someone using POWER on SELF_REF = being controlled/manipulated."""
        has_power = any(r.role == "POWER" for r in roles)
        has_self = any(r.role == "SELF_REF" for r in roles)
        has_other = any(r.role in ("OTHER_REF", "RELATION_REF") for r in roles)
        if has_power and has_self and has_other:
            return StructureMatch(
                pattern="POWER_OVER_SELF",
                confidence=0.7,
                matched_indices=[i for i, r in enumerate(roles)
                                if r.role in ("POWER", "SELF_REF", "OTHER_REF", "RELATION_REF")],
                description="Someone using power over self - V and D drop",
                v_weight=-20.0, d_weight=-30.0, g_weight=-15.0,
                w_weight=-15.0,
            )
        return None

    def _self_submission(self, roles):
        """SELF_REF + SUBMISSION = user surrendering agency."""
        has_sub = any(r.role == "SUBMISSION" for r in roles)
        has_self = any(r.role == "SELF_REF" for r in roles)
        if has_sub and has_self:
            return StructureMatch(
                pattern="SELF_SUBMISSION",
                confidence=0.65,
                matched_indices=[i for i, r in enumerate(roles)
                                if r.role in ("SUBMISSION", "SELF_REF")],
                description="User surrendering agency",
                v_weight=-20.0, d_weight=-40.0, g_weight=-15.0,
                w_weight=-20.0,
            )
        return None

    def _victimization(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """(OTHER_REF|RELATION_REF) + negative_verb + SELF_REF = user is victim.

        "boyfriend hit me" -- other person acts negatively on self
        "she left me" -- other person abandons self
        "he ignored me" -- other person rejects self

        The verb carries the damage. The structure confirms direction:
        someone ELSE did this TO the user.
        """
        other_indices = [r.position for r in roles
                        if r.role in ("OTHER_REF", "RELATION_REF")]
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]

        if not other_indices or not self_indices:
            return None

        # Find negative verbs -- EMOTIONAL with force, or TRANSFER/PULL_AWAY
        # TRANSFER verbs like "left" are near-neutral alone but become
        # negative when OTHER does them TO SELF. Lower threshold for TRANSFER.
        from .vocabulary import VOCABULARY
        neg_verb_indices = []
        for r in roles:
            if r.role == "EMOTIONAL" and r.force and r.force[0] < -20:
                neg_verb_indices.append(r.position)
            elif r.role in ("TRANSFER", "PULL_AWAY"):
                v_force = VOCABULARY.get(r.word)
                if v_force and v_force[0] < 0:
                    neg_verb_indices.append(r.position)
        if not neg_verb_indices:
            return None

        # Check structure: other before verb, self after (or near)
        # "she(OTHER) left(verb) me(SELF)" -- canonical order
        best_other = other_indices[0]
        best_verb = min(neg_verb_indices, key=lambda x: abs(x - best_other))
        best_self = min(self_indices, key=lambda x: abs(x - best_verb))

        # Other should be before or near verb, self should be after or near verb
        if abs(best_other - best_verb) > 5 or abs(best_self - best_verb) > 5:
            return None

        # Get verb intensity for scaling (check WordRole force, then vocabulary)
        # TRANSFER verbs have low raw force (they're liquid words) but the
        # structure itself confirms damage -- use minimum intensity of 0.5
        verb_role = roles[best_verb]
        if verb_role.force:
            verb_v = verb_role.force[0]
        else:
            vf = VOCABULARY.get(verb_role.word)
            verb_v = vf[0] if vf else -30
        intensity = max(0.5, min(abs(verb_v) / 60.0, 2.0))

        indices = sorted({best_other, best_verb, best_self})
        return StructureMatch(
            pattern="VICTIMIZATION",
            confidence=0.7,
            matched_indices=indices,
            description="Someone did something negative to the user",
            v_weight=-25.0 * intensity,
            d_weight=-20.0 * intensity,
            u_weight=15.0,
            g_weight=15.0,
            w_weight=-20.0,
        )

    def _bravado(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Laughter/filler + AMPLIFIER + PEACE = overcompensation mask.

        "haha yeah im totally okay" -- laughter + amplifier + peace = bravado
        "lol im fine" -- laughter + peace = deflection
        "im totally fine" -- amplifier + peace without laughter = mild deflection

        The more effort spent saying "I'm okay", the less okay they are.
        """
        laughter_indices = [
            r.position for r in roles if r.word in _LAUGHTER_WORDS
        ]
        amplifier_indices = [r.position for r in roles if r.role == "AMPLIFIER"]
        # PEACE role + "good/great/alright/okay" as peace-adjacent for bravado
        _bravado_peace = {"alright", "okay", "ok", "fine", "chill"}
        # "good" and "great" removed -- too broad. "I have a good chance" ≠ bravado.
        peace_indices = [r.position for r in roles
                        if r.role == "PEACE" or r.word in _bravado_peace]
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]

        if not peace_indices:
            return None

        has_laughter = len(laughter_indices) > 0
        has_amplifier = len(amplifier_indices) > 0
        has_self = len(self_indices) > 0

        # Need at least 2 of: laughter, amplifier, self_ref near peace
        signals = sum([has_laughter, has_amplifier, has_self])
        if signals < 2:
            return None

        # Laughter + peace alone is enough (strong signal)
        # Amplifier + self + peace is enough (protest too much)
        indices = sorted(set(
            laughter_indices + amplifier_indices + peace_indices + self_indices
        ))

        confidence = 0.4
        if has_laughter:
            confidence += 0.25
        if has_amplifier:
            confidence += 0.15
        if has_self:
            confidence += 0.1

        return StructureMatch(
            pattern="BRAVADO",
            confidence=min(confidence, 0.9),
            matched_indices=indices,
            description="Overcompensation mask -- protesting too much",
            v_weight=-55.0,
            d_weight=-20.0,
            u_weight=15.0,
            g_weight=10.0,
            w_weight=-5.0,
        )

    def _betrayal(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """RELATION_REF + betrayal verb + SELF_REF = intimate betrayal.

        "my wife cheated on me with my best friend"
        Structure: RELATION + strong_negative + SELF + RELATION
        The relationship words become instruments of pain, not warmth.
        Higher G on relationships = bigger fall (wife G=40 > friend G=20).
        """
        betrayal_indices = [
            r.position for r in roles if r.word in _BETRAYAL_VERBS
        ]
        # Check compound betrayal phrases: "turned on", "went against", etc.
        # These are polarity flips -- the subject rotated away from the target.
        compound_betrayal = False
        if not betrayal_indices:
            words = [r.word for r in roles]
            for i, w in enumerate(words):
                if w in _BETRAYAL_COMPOUNDS and i + 1 < len(words):
                    if words[i + 1] in _BETRAYAL_COMPOUNDS[w]:
                        betrayal_indices.append(i)
                        compound_betrayal = True
        if not betrayal_indices:
            # Also check for strong negative EMOTIONAL near two RELATION_REFs
            strong_neg = [r.position for r in roles
                         if r.role == "EMOTIONAL" and r.force and r.force[0] < -80]
            if not strong_neg:
                return None
            betrayal_indices = strong_neg

        relation_indices = [r.position for r in roles if r.role == "RELATION_REF"]
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]
        other_indices = [r.position for r in roles if r.role == "OTHER_REF"]

        # Compound betrayal phrases carry their own meaning --
        # "he turned on me" doesn't need RELATION_REF, just a subject + SELF_REF
        if compound_betrayal:
            # SELF_REF must be the TARGET (after the verb), not the AGENT (before).
            # "I turned on the light" = self is agent, no betrayal.
            # "he turned on me" = self is target, betrayal.
            best_bi = betrayal_indices[0]
            self_after = [si for si in self_indices if si > best_bi]
            if not self_after:
                return None
            self_indices = self_after  # only use self-refs that are targets
            subject_indices = relation_indices + other_indices
        else:
            if not relation_indices or not self_indices:
                return None
            subject_indices = relation_indices

        best_bi = betrayal_indices[0]
        nearby_subjects = [si for si in subject_indices if abs(si - best_bi) <= 8]
        nearby_self = [si for si in self_indices if abs(si - best_bi) <= 8]

        if not nearby_self:
            return None
        # For non-compound, require a nearby relation
        if not compound_betrayal and not nearby_subjects:
            return None

        # Confidence: relation words scale it up (heavier relationship = worse)
        if nearby_subjects and relation_indices:
            nearby_rels = [si for si in nearby_subjects if si in relation_indices]
            confidence = 0.6 + min(len(nearby_rels) * 0.15, 0.35)
        elif compound_betrayal:
            confidence = 0.65  # compound phrase is confident on its own
        else:
            confidence = 0.6

        # Sum relationship G values -- higher trust = harder fall
        from .vocabulary import VOCABULARY
        total_g = 0
        rel_or_subject = nearby_subjects if nearby_subjects else []
        for ri in rel_or_subject:
            word = roles[ri].word
            if word in VOCABULARY:
                total_g += max(5, VOCABULARY[word][4])
            else:
                total_g += 20
        if not rel_or_subject:
            total_g = 20  # baseline for implicit subject
        g_multiplier = total_g / 30.0  # normalize: 30 = baseline

        indices = sorted(set(betrayal_indices + rel_or_subject + nearby_self))
        return StructureMatch(
            pattern="BETRAYAL",
            confidence=min(confidence, 1.0),
            matched_indices=indices,
            description="Intimate betrayal -- relationship trust weaponized",
            v_weight=-60.0 * g_multiplier,
            d_weight=-40.0 * g_multiplier,
            u_weight=30.0,
            g_weight=40.0,
            w_weight=-15.0,
        )

    def _directed_positive(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Positive EMOTIONAL directed at OTHER_REF, not shared with SELF.

        "good for you" -- positive attributed to other, not self = dismissive
        "i hope youre happy" -- self hopes other is happy = PA
        "must be nice" -- envy/dismissal of other's state
        "glad someone is having fun" -- someone else, not me

        Does NOT fire when self is also positive:
        "im so proud of you" -- self is proud = genuine
        "you make me happy" -- self benefits = genuine
        """
        # Find positive emotional words
        pos_indices = [r.position for r in roles
                      if r.role == "EMOTIONAL" and r.force and r.force[0] > 15]
        other_indices = [r.position for r in roles
                        if r.role == "OTHER_REF"]

        if not pos_indices or not other_indices:
            return None

        # Check if SELF is genuinely positive (not just directing at other)
        # "im so proud of you" = SELF feels proud (self-state word near SELF)
        # "i hope youre happy" = SELF directs hope at OTHER (not self-state)
        # Key: is the positive word describing SELF's state or OTHER's state?
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]

        self_state_words = {"proud", "grateful", "thankful", "excited",
                           "thrilled", "amazed", "impressed", "blessed",
                           "lucky", "honored"}
        self_has_state = any(r.word in self_state_words for r in roles)

        # "you make ME happy" -- self benefits from other
        self_benefits = any(r.word in ("me", "my", "mine") and
                          any(abs(r.position - pi) <= 3 for pi in pos_indices)
                          for r in roles if r.role == "SELF_REF")

        if self_has_state or self_benefits:
            return None

        # Check if there's genuine action/effort acknowledgment
        # "you did amazing" = acknowledging action (genuine)
        # "good for you" = just state attribution (dismissive)
        action_words = {"did", "made", "built", "created", "earned",
                       "won", "passed", "finished", "completed", "achieved",
                       "worked", "helped", "saved", "fixed", "said", "told",
                       "gave", "proposed", "remembered", "graduated", "ran",
                       "walked", "danced", "sang", "wrote", "cooked",
                       "learned", "started", "stopped", "tried", "came"}
        has_action = any(r.word in action_words for r in roles)
        # Also genuine if self is thankful/proud/loving
        grateful_words = {"proud", "grateful", "thankful",
                         "appreciate", "love", "favorite", "amazing",
                         "care", "miss", "adore", "cherish",
                         "believe", "support", "trust", "respect"}
        has_grateful = any(r.word in grateful_words for r in roles)

        # "thank you" is always genuine, regardless of self presence
        has_thank = any(r.word in ("thank", "thanks", "thankyou")
                       for r in roles)
        if has_action or has_thank or (self_indices and has_grateful):
            return None

        # If no SELF_REF at all, speaker isn't in the sentence -- narration, not PA
        # "he proposed on the beach" = story. "good for you" = implicit self present.
        if not self_indices and len(roles) > 4:
            return None

        # Positive + OTHER without self benefiting = directed/dismissive
        # Short sentences are stronger signal ("good for you" = 3 words)
        sentence_len = len(roles)
        confidence = 0.5 if sentence_len > 5 else 0.65

        indices = sorted(set(pos_indices + other_indices))
        return StructureMatch(
            pattern="DIRECTED_POSITIVE",
            confidence=min(confidence, 0.85),
            matched_indices=indices,
            description="Positive directed at other, not shared -- dismissive/PA",
            v_weight=-35.0,
            d_weight=5.0,
            u_weight=5.0,
            g_weight=0.0,
        )

    def _minimizer(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """'just' or 'only' near negative concept = shrinking real pain.

        "it was just a joke" -- minimizing harm done
        "its just a bruise" -- minimizing injury
        "its not a big deal" -- negation + scale word = forced minimization

        Also catches "too" + trait at OTHER = invalidation:
        "youre too sensitive" -- excess framing = criticism
        """
        words = [r.word for r in roles]

        # "just a" or "only a" pattern = minimization
        just_indices = [i for i, r in enumerate(roles)
                       if r.word in ("just", "only") and r.role == "FILLER"]
        if just_indices:
            # Check if minimizing a negative concept
            # "just a joke" = dismissing someone's pain
            # "just a bruise" = dismissing injury
            # But "just bought coffee" = genuinely casual
            has_other = any(r.role == "OTHER_REF" for r in roles)
            has_dismiss = any(r.word in ("joke", "kidding", "playing",
                            "bruise", "scratch", "nothing")
                            for r in roles)
            if has_dismiss or (has_other and len(roles) <= 6):
                return StructureMatch(
                    pattern="MINIMIZER",
                    confidence=0.55,
                    matched_indices=just_indices,
                    description="Minimizing with 'just/only' -- shrinking real impact",
                    v_weight=-15.0,
                    d_weight=10.0,
                    u_weight=0.0,
                    g_weight=0.0,
                    w_weight=-10.0,
                )

        # "too" + trait = invalidation ("youre too sensitive")
        too_indices = [i for i, r in enumerate(roles)
                      if r.word == "too" and r.role == "AMPLIFIER"]
        if too_indices:
            has_other = any(r.role == "OTHER_REF" for r in roles)
            has_trait = any(r.role == "EMOTIONAL" and r.force and r.force[0] > 0
                          for r in roles)
            if has_other and has_trait:
                return StructureMatch(
                    pattern="MINIMIZER",
                    confidence=0.65,
                    matched_indices=too_indices,
                    description="'Too' + trait = excess framing = invalidation",
                    v_weight=-20.0,
                    d_weight=15.0,
                    u_weight=5.0,
                    g_weight=0.0,
                    w_weight=-10.0,
                )

        return None

    def _excluded_positive(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Positive emotion directed at OTHER while SELF is excluded or doubting.

        "do you even love me" -- questioning if positive applies to self
        "my parents love my brother more" -- positive goes to other, not self
        "everyone got invited except me" -- positive event excludes self

        Pattern: positive EMOTIONAL + (doubt marker OR exclusion marker OR
        comparison marker) + SELF_REF = self excluded from the positive.
        """
        pos_indices = [r.position for r in roles
                      if r.role == "EMOTIONAL" and r.force and r.force[0] > 25]
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]

        if not pos_indices or not self_indices:
            return None

        # Exclusion markers -- must clearly indicate self is left out
        exclusion_words = {"except", "instead", "anymore"}
        # "even" is doubt ONLY when near SELF_REF: "do you even love me" = doubt
        # "even I would have" = intensification, not doubt
        doubt_words = {"even", "ever"}
        comparison_words = {"more", "less", "worse", "prettier",
                           "smarter", "faster", "rather"}

        has_exclusion = any(r.word in exclusion_words for r in roles)

        # Doubt: "even" is doubt ONLY when followed by OTHER's action toward self.
        # "do you EVEN love me" = doubt (even before other's action).
        # "EVEN I would have" = leveling up (even before self = inclusion).
        # Key: if SELF_REF immediately follows "even", it's leveling, not doubt.
        has_doubt = False
        for r in roles:
            if r.word in doubt_words:
                # Check if SELF_REF is immediately after "even" = leveling, skip
                next_is_self = (r.position + 1 < len(roles)
                               and roles[r.position + 1].role == "SELF_REF")
                if next_is_self:
                    continue  # "even I" = leveling up, not doubt
                # Otherwise check proximity to positive + self
                near_pos = any(abs(r.position - pi) <= 3 for pi in pos_indices)
                near_self = any(abs(r.position - si) <= 4 for si in self_indices)
                if near_pos and near_self:
                    has_doubt = True
                    break

        # Comparison: "more" must be near OTHER_REF (comparing self to other)
        has_comparison = False
        other_indices = [r.position for r in roles if r.role in ("OTHER_REF", "RELATION_REF")]
        for r in roles:
            if r.word in comparison_words:
                near_other = any(abs(r.position - oi) <= 4 for oi in other_indices)
                if near_other:
                    has_comparison = True
                    break

        # Need clear signal
        if not has_exclusion and not has_doubt and not has_comparison:
            return None

        # Check that OTHER/RELATION is also present (the one getting the positive)
        has_other = any(r.role in ("OTHER_REF", "RELATION_REF") for r in roles)
        if not has_other and not has_doubt:
            return None

        # Stronger with more signals
        signals = sum([has_exclusion, has_doubt, has_comparison])
        confidence = 0.5 + min(signals * 0.15, 0.35)

        # Scale with how positive the positive word is (bigger love = bigger hurt)
        max_pos_v = max(r.force[0] for r in roles
                       if r.role == "EMOTIONAL" and r.force and r.force[0] > 25)
        intensity = min(max_pos_v / 50.0, 2.0)

        indices = sorted(set(pos_indices + self_indices))
        return StructureMatch(
            pattern="EXCLUDED_POSITIVE",
            confidence=min(confidence, 0.9),
            matched_indices=indices,
            description="Self excluded from positive -- doubt, comparison, or exclusion",
            v_weight=-40.0 * intensity,
            d_weight=-15.0,
            u_weight=10.0,
            g_weight=10.0,
            w_weight=-30.0,
        )

    def _calling_out(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """'why' or 'how' + OTHER_REF = calling out behavior.

        "why do you do that" -- complaint disguised as question.
        "how could you say that" -- accusation disguised as question.

        The question form is the mask. Nobody asks "why do you do that"
        when they are happy about it. Slightly negative V, elevated A.

        Does NOT fire for genuine questions about non-person subjects:
        "why do birds fly" -- OTHER_REF is for people, not birds.
        """
        interrog_indices = [
            r.position for r in roles if r.word in _INTERROGATION_WORDS
        ]
        if not interrog_indices:
            return None

        other_indices = [r.position for r in roles if r.role == "OTHER_REF"]
        if not other_indices:
            return None

        # Check proximity -- "why" near "you"
        best_dist = 999
        best_i, best_o = -1, -1
        for ii in interrog_indices:
            for oi in other_indices:
                d = abs(ii - oi)
                if d < best_dist:
                    best_dist = d
                    best_i, best_o = ii, oi

        if best_dist > 5:
            return None

        # Boost if "always" or "never" present (pattern emphasis)
        has_always = any(r.word in ("always", "never", "every", "constantly")
                        for r in roles)
        confidence = 0.55
        if has_always:
            confidence += 0.2

        indices = sorted({best_i, best_o})
        return StructureMatch(
            pattern="CALLING_OUT",
            confidence=min(confidence, 0.85),
            matched_indices=indices,
            description="Complaint disguised as question -- calling out behavior",
            v_weight=-18.0,
            d_weight=8.0,    # questioner is asserting position
            u_weight=5.0,
            g_weight=5.0,
            w_weight=5.0,
        )

    def _self_excluded(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """OTHER has/does something WITHOUT SELF = user excluded.

        "they have a group chat without me" -- others built connection, user left out
        "everyone went to the party without me" -- group activity excluded user
        "they planned it without telling me" -- user cut out of the loop

        Pattern: OTHER_REF + "without" + SELF_REF = exclusion.
        The others possess or do something and the user is not included.
        """
        other_indices = [r.position for r in roles
                        if r.role in ("OTHER_REF", "RELATION_REF")]
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]
        without_indices = [r.position for r in roles if r.word == "without"]

        if not other_indices or not self_indices or not without_indices:
            return None

        # Pattern: OTHER ... without ... SELF (in that order)
        for wi in without_indices:
            other_before = any(oi < wi for oi in other_indices)
            self_after = any(si > wi for si in self_indices)
            if other_before and self_after:
                return StructureMatch(
                    pattern="SELF_EXCLUDED",
                    confidence=0.7,
                    matched_indices=sorted(set(other_indices + without_indices + self_indices)),
                    description="User excluded from group activity/connection",
                    v_weight=-25.0,
                    d_weight=-20.0,
                    u_weight=10.0,
                    g_weight=10.0,
                    w_weight=-15.0,
                )
        return None

    def _withheld_positive(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Positive emotion that was NEVER expressed or is UNREALIZED.

        "my father never once said he was proud" -- pride withheld
        "they would have been proud" -- conditional past, can't happen now
        "he never told me he loved me" -- love withheld

        Pattern: NEGATOR/conditional + positive EMOTIONAL in same sentence.
        The positive thing didn't happen or can't happen.
        """
        pos_indices = [r.position for r in roles
                      if r.role == "EMOTIONAL" and r.force and r.force[0] > 30]
        if not pos_indices:
            return None

        # "never" or "didn't" or "wouldn't" = the positive was withheld
        _WITHHOLDING = {"never", "didnt", "didn't", "wouldnt", "wouldn't",
                        "couldnt", "couldn't"}
        # "couldn't believe" = amazement, not withholding. Skip.
        _AMAZEMENT_FOLLOWS = {"believe", "imagine", "fathom"}
        withhold_indices = []
        for r in roles:
            if r.word in _WITHHOLDING:
                # Check if next word is amazement -- "couldn't believe" = overwhelmed
                next_word = roles[r.position + 1].word if r.position + 1 < len(roles) else ""
                if next_word in _AMAZEMENT_FOLLOWS:
                    continue  # amazement, not withholding
                withhold_indices.append(r.position)

        # "would have been" = conditional past = unrealized
        words = [r.word for r in roles]
        conditional_past = False
        for i in range(len(words) - 2):
            if words[i] == "would" and words[i+1] == "have" and words[i+2] == "been":
                withhold_indices.append(i)
                conditional_past = True

        if not withhold_indices:
            return None

        # The positive word must come AFTER the withholding word
        for wi in withhold_indices:
            for pi in pos_indices:
                if pi > wi:
                    confidence = 0.75 if conditional_past else 0.7
                    return StructureMatch(
                        pattern="WITHHELD_POSITIVE",
                        confidence=confidence,
                        matched_indices=sorted({wi, pi}),
                        description="Positive emotion withheld or unrealized",
                        v_weight=-50.0,
                        d_weight=-15.0,
                        u_weight=5.0,
                        g_weight=15.0,
                        w_weight=-20.0,
                    )
        return None

    def _d_inversion(self, roles):
        """INVERSION verb present = power dynamics flipped from expected."""
        has_inv = any(r.role == "INVERSION" for r in roles)
        has_self = any(r.role == "SELF_REF" for r in roles)
        if has_inv and has_self:
            return StructureMatch(
                pattern="D_INVERSION",
                confidence=0.75,
                matched_indices=[i for i, r in enumerate(roles)
                                if r.role in ("INVERSION", "SELF_REF")],
                description="Power inversion - user lost control of something they should control",
                v_weight=-30.0, d_weight=-50.0, u_weight=15.0, g_weight=-20.0,
                w_weight=-10.0,
            )
        return None

    def _relief_absence(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """'without [negative]' or 'havent had [negative]' = relief/positive.

        "i can afford groceries without stress" -- absence of stress = relief
        "i havent had a panic attack in a month" -- absence of panic = progress
        "i ran my first mile without stopping" -- absence of stopping = achievement

        Pattern: negation/absence word + negative emotional word = POSITIVE.
        The negative thing is GONE. That's relief.
        """
        from .vocabulary import VOCABULARY

        words = [r.word for r in roles]

        # Find "without" or "havent"/"haven't" positions
        absence_indices = []
        for i, r in enumerate(roles):
            if r.word in ("without",):
                absence_indices.append(i)
            # "havent had" / "haven't had" pattern
            if r.word in ("havent", "haven't", "hasnt", "hasn't") and i + 1 < len(roles):
                if roles[i + 1].word in ("had", "been", "felt", "seen", "gotten"):
                    absence_indices.append(i)

        if not absence_indices:
            return None

        # Check if what follows the absence word is STRONGLY negative.
        # "without stress" = relief (stress V=-75, strongly negative).
        # Fire when the absent thing is a NEGATIVE STATE/EXPERIENCE.
        # "without stress" = relief (stress is a bad state).
        # "without saying goodbye" = NOT relief (goodbye is closure owed).
        # Social/connection words when absent = deprivation, not relief.
        _NOT_RELIEF = {"goodbye", "goodbyes", "telling", "asking", "warning",
                       "saying", "knowing", "explanation", "closure",
                       "permission", "consent", "notice", "apology"}
        for ai in absence_indices:
            for j in range(ai + 1, min(ai + 4, len(roles))):
                if roles[j].word in _NOT_RELIEF:
                    continue  # absence of social obligation = deprivation, not relief
                wf = roles[j].force or VOCABULARY.get(roles[j].word)
                if wf and wf[0] < -10:
                    # Strongly negative word after absence marker = the bad thing is GONE
                    # Scale relief by how bad the absent thing is
                    # "without stress" (V=-75) = bigger relief than "without worry" (V=-35)
                    severity = min(abs(wf[0]) / 50.0, 2.0)
                    return StructureMatch(
                        pattern="RELIEF_ABSENCE",
                        confidence=0.75,
                        matched_indices=[ai, j],
                        description="Absence of negative = relief/progress",
                        v_weight=35.0 * severity,
                        d_weight=15.0,
                        u_weight=-10.0,
                        g_weight=10.0,
                        w_weight=10.0,
                    )

        return None

