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
from math import log
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

# Strong self-reference: "I", "me", "myself", "im" — the speaker IS the subject.
# "my" is possessive: "my alarm" = the speaker OWNS the alarm, not IS the alarm.
# Crisis patterns require STRONG self-ref to avoid "my flight cancelled" = crisis.
_STRONG_SELF_WORDS = frozenset({
    "i", "me", "myself", "im", "i'm", "ive", "i've", "ill", "i'll", "id", "i'd",
})
_APOLOGY_WORDS = frozenset({"sorry", "apologize", "apologise", "apologies"})
_BLANKET_WORDS = frozenset({
    "everything", "everyone", "everybody", "all", "always",
    "nothing", "nobody",
})
_SUSTAIN_VERBS = frozenset({
    "take", "do", "keep", "bear", "stand", "handle", "live",
    "cope", "manage", "endure", "deal", "continue",
    # "go" removed — too ambiguous. "didnt go off" = alarm, not exhaustion.
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
_DEATH_SLANG_WORDS = frozenset({
    "dead", "dying", "died", "death", "kill", "killed", "killing",
})

# ── Syntactic resolver word sets ───────────────────────────────
# For CONTRADICTION_RESOLVE: adverb+adjective pairs where head governs.
_INTENSIFYING_ADVERBS = frozenset({
    "painfully", "terribly", "awfully", "horribly", "dreadfully",
    "frighteningly", "sickeningly", "disgustingly", "devastatingly",
    "heartbreakingly", "brutally", "cruelly", "wickedly", "dangerously",
    "absurdly", "ridiculously", "insanely",
})
_POSITIVE_ADJECTIVE_HEADS = frozenset({
    "beautiful", "gorgeous", "stunning", "brilliant", "amazing",
    "wonderful", "perfect", "good", "great", "kind", "talented",
    "honest", "brave", "sweet", "funny", "smart", "clever",
    "true", "real", "genuine", "effective", "powerful",
})
_NEGATIVE_NOUN_HEADS = frozenset({
    "revenge", "punishment", "torture", "destruction", "death",
    "betrayal", "murder", "suffering", "agony", "misery",
    "cruelty", "hatred", "violence", "lie", "lies", "deceit",
})
_NEGATIVE_MAIN_VERBS = frozenset({
    "hate", "hated", "hates", "hating",
    "despise", "despised", "despises", "despising",
    "regret", "regretted", "regrets", "regretting",
    "resent", "resented", "resents", "resenting",
    "loathe", "loathed", "loathes", "loathing",
    "dread", "dreaded", "dreads", "dreading",
})
_POSITIVE_COMPLEMENT_PHRASES = [
    ("so", "good"), ("so", "well"), ("so", "much"),
    ("so", "right"), ("too", "good"), ("too", "well"),
]

# ── Pragmatic convention word sets ─────────────────────────────
_COMPLIMENT_FRAMES = frozenset({"without", "reason", "thanks"})
_GRATITUDE_VERBS = frozenset({
    "done", "made", "survived", "finished", "accomplished", "gave", "give",
})
_MUNDANE_WINS = {
    "got out of bed", "ate a full meal", "went outside",
    "took a shower", "called someone", "smiled",
    "slept through the night", "brushed my teeth",
    "ate something", "left the house", "got dressed",
    "made my bed", "cooked a meal", "went for a walk",
    "drank water", "cleaned my room", "opened the curtains",
}
_TEMPORAL_RARITY = frozenset({"today", "finally", "actually", "first", "time"})
_SLEEP_WORDS = frozenset({"sleep", "slept", "sleeping", "hours"})
_SOCIAL_EVENTS = frozenset({
    "birthday", "party", "wedding", "funeral", "graduation",
    "reunion", "dinner", "celebration", "ceremony",
})

# ── Contradiction-based sarcasm word sets ────────────────────────
# These power the contradiction sarcasm detector. Sarcasm = surface
# polarity contradicts structural context. Not pattern-matching on
# opener words.

_IRONIC_TITLES = frozenset({
    "genius", "einstein", "champ", "buddy", "pal", "sport",
    "chief", "sherlock", "professor", "captain", "ace",
})
_COMPETENCE_NOUNS = frozenset({
    "work", "job", "move", "plan", "idea", "thinking", "call",
    "effort", "attempt", "logic", "strategy",
})
_HOLLOW_AFFIRMS = frozenset({
    "yeah", "sure", "right", "oh", "mhm", "okay", "ok",
})
_AFFIRM_ECHOES = frozenset({
    "right", "sure", "totally", "absolutely", "definitely",
    "of", "course", "obviously", "clearly",
})
_PERMISSION_VERBS = frozenset({
    "go", "ahead", "leave", "try", "see", "knock",
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
        """Run all detectors, return matches with confidence > 0.3.

        Applies complexity dampening: long/complex sentences get reduced
        confidence on sarcasm/irony structures to prevent false positives
        on literary prose. (Council Round 5 consensus fix #3)
        """
        # ── Complexity score ──
        # C = (word_count / 10) + (comma_count * 0.5) + (subordinator_count * 1.0)
        # Claude's formula. Sigmoid decay above C > 3.0.
        _SUBORDINATORS = {"whether", "although", "because", "while", "since",
                         "unless", "whereas", "though", "whereby", "wherein"}
        wc = len(roles)
        cc = sum(1 for r in roles if r.word.endswith(','))
        sc = sum(1 for r in roles if r.word in _SUBORDINATORS)
        complexity = (wc / 10.0) + (cc * 0.5) + (sc * 1.0)

        # Dampener: 1.0 at C<=3, decays toward 0.4 floor
        if complexity > 3.0:
            complexity_dampener = max(0.4, 1.0 / (1.0 + 0.3 * (complexity - 3.0)))
        else:
            complexity_dampener = 1.0

        # Structures affected by complexity dampening
        _COMPLEXITY_SENSITIVE = {
            "SARCASM_INVERSION", "SOCIAL_NULLITY", "DIRECTED_POSITIVE",
            "DIRECTED_LABEL", "PASSIVE_RESIGNATION",
        }

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
            self._directed_label,
            self._slang_death_humor,
            self._grief_loss,
            self._reported_comfort,
            self._rhetorical_self_negation,
            self._self_harm_intent,
            self._existential_negation,
            self._social_nullity,
            self._rhetorical_hopelessness,
            self._passive_resignation,
            self._atmospheric_grief,
            self._recovery_milestone,
            self._contradiction_resolve,
            self._numbers_context,
            self._negated_negative_compliment,
            self._recovery_small_win,
            self._ambiguity_hold,
            self._mundane_hyperbole,
            self._boundary_violation,
            self._self_erasure,
            self._divestiture,
            self._method_fixation,
            self._rarity_marker,
            self._abandonment,
            self._life_achievement,
            self._self_insignificance,
            self._self_replacement,
            self._persistent_absence,
            self._directed_dismissal,
            self._martyrdom_field,
            self._dangling_bond,
            self._masking,
            self._resignation,
            self._world_continues,
            self._farewell,
            # self._hollow_agreement,  # REMOVED: was judging tone, not structure. Running state handles this.
        ]
        matches = []
        for detector in detectors:
            result = detector(roles)
            if result is not None and result.confidence > 0.3:
                # Apply complexity dampening to sensitive structures
                if result.pattern in _COMPLEXITY_SENSITIVE and complexity_dampener < 1.0:
                    result.confidence *= complexity_dampener
                    if result.confidence <= 0.3:
                        continue  # dampened below threshold, skip
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
        # "accepted" + object noun = achievement, not calm acceptance
        _ACHIEVEMENT_OBJECTS = {"application", "offer", "proposal", "request",
                               "submission", "entry", "bid", "manuscript",
                               "paper", "pitch", "resume", "invitation"}
        words = [r.word for r in roles]
        has_achievement_object = any(w in _ACHIEVEMENT_OBJECTS for w in words)
        if has_acquire or has_positive_verb or has_achievement_object:
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
        """Contradiction-based sarcasm: surface polarity vs structural context.

        Sarcasm is a CONTRADICTION between what the words say (surface)
        and what the structure means (context). Five contradiction features
        scored independently, then combined:

        1. Mock Praise: POSITIVE_ADJ + COMPETENCE_NOUN + IRONIC_TITLE
           "nice work genius" -- praise surface, contempt structure
        2. Dismissive Assent: HOLLOW_AFFIRM + AFFIRM_ECHO
           "yeah right" -- agreement surface, rejection structure
        3. Permission Hostility: HOLLOW_AFFIRM + PERMISSION_VERB + negative context
           "sure go ahead" -- only sarcastic when sentence has negative words
        4. Valence Whiplash: max V swing in 4-word window
           "love this wonderful disaster" -- polarity reversal mid-sentence
        5. Brevity bonus: short sentences with contradiction score higher

        RELATION_REF blocks sarcasm: "I love my mom" is genuine.
        """
        # RELATION_REF blocks sarcasm: "I love my mom" is genuine
        has_relation = any(r.role == "RELATION_REF" for r in roles)
        if has_relation:
            return None

        words = [r.word for r in roles]
        n = len(roles)

        # ── Gather indices ──────────────────────────────────────
        positive_idx = [i for i, r in enumerate(roles)
                       if r.role == "EMOTIONAL" and r.force and r.force[0] >= 15]
        negative_idx = [i for i, r in enumerate(roles)
                       if r.role == "EMOTIONAL" and r.force and r.force[0] < -15]
        has_negative_context = len(negative_idx) > 0

        # ── Feature 1: Mock Praise ──────────────────────────────
        # POSITIVE_ADJ + COMPETENCE_NOUN + IRONIC_TITLE
        # "nice work genius", "great job einstein", "brilliant move champ"
        mock_praise = 0.0
        has_ironic_title = any(r.word in _IRONIC_TITLES for r in roles)
        has_competence_noun = any(r.word in _COMPETENCE_NOUNS for r in roles)
        has_positive = len(positive_idx) > 0

        if has_ironic_title and has_positive:
            mock_praise = 0.9  # title + positive = near-certain sarcasm
        elif has_ironic_title and has_competence_noun:
            mock_praise = 0.8  # "nice work genius" even if "nice" not in vocab
        elif has_competence_noun and has_positive and n <= 6:
            # "great job" alone is short enough to be suspicious
            mock_praise = 0.3

        # ── Feature 2: Dismissive Assent ────────────────────────
        # HOLLOW_AFFIRM + AFFIRM_ECHO = "yeah right", "sure totally",
        # "oh absolutely", "oh of course"
        dismissive_assent = 0.0
        hollow_idx = [i for i, r in enumerate(roles) if r.word in _HOLLOW_AFFIRMS]
        echo_idx = [i for i, r in enumerate(roles) if r.word in _AFFIRM_ECHOES]

        if hollow_idx and echo_idx:
            # Check adjacency: hollow affirm near echo (within 2 words)
            for hi in hollow_idx:
                for ei in echo_idx:
                    if hi != ei and abs(hi - ei) <= 2:
                        dismissive_assent = 0.85
                        break
                if dismissive_assent > 0:
                    break

        # "what a" + positive = "what a wonderful surprise" = sarcasm
        has_what_a = (n >= 2 and roles[0].word == "what" and roles[1].word == "a")
        if has_what_a and has_positive:
            dismissive_assent = max(dismissive_assent, 0.8)

        # ── Feature 3: Permission Hostility ─────────────────────
        # HOLLOW_AFFIRM + PERMISSION_VERB + negative context
        # "sure go ahead" is ONLY sarcastic with negative context.
        # Without it, "sure go ahead" = genuine permission.
        permission_hostility = 0.0
        has_permission = any(r.word in _PERMISSION_VERBS for r in roles)
        if hollow_idx and has_permission and has_negative_context:
            permission_hostility = 0.75

        # ── Feature 3b: Surface-Context Mismatch ─────────────────
        # HOLLOW_AFFIRM + POSITIVE + mundane/neutral context words
        # "oh great another monday" -- positive word applied to mundane context.
        # The contradiction is positive SURFACE vs mundane CONTEXT.
        _MUNDANE_CONTEXT = {"monday", "meeting", "work", "homework", "traffic",
                            "redo", "again", "another", "same", "overtime",
                            "bills", "chores", "commute", "deadline"}
        surface_context = 0.0
        mundane_idx = [i for i, r in enumerate(roles) if r.word in _MUNDANE_CONTEXT]
        has_mundane = len(mundane_idx) > 0
        if hollow_idx and has_positive and has_mundane:
            surface_context = 0.8  # hollow + positive + mundane = strong contradiction
        elif has_positive and has_mundane and n <= 7:
            surface_context = 0.5  # positive + mundane without hollow, weaker

        # ── Feature 4: Valence Whiplash ─────────────────────────
        # Max V swing in a 4-word sliding window.
        # "love this wonderful disaster" -- polarity reversal.
        whiplash = 0.0
        v_values = []
        for r in roles:
            if r.role == "EMOTIONAL" and r.force:
                v_values.append((r.position, r.force[0]))
        if len(v_values) >= 2:
            max_swing = 0.0
            for i in range(len(v_values)):
                for j in range(i + 1, len(v_values)):
                    pos_i, val_i = v_values[i]
                    pos_j, val_j = v_values[j]
                    if abs(pos_i - pos_j) <= 4:
                        swing = abs(val_i - val_j)
                        max_swing = max(max_swing, swing)
            # Threshold: swing > 40 starts registering, > 80 strong
            if max_swing > 40:
                whiplash = min(1.0, (max_swing - 40) / 60.0)

        # ── Feature 5: Compressed Sarcasm ──────────────────────
        # Hollow affirm + positive word + short sentence = the compression
        # itself IS the contradiction. "oh joy", "oh wonderful", "oh how lovely"
        # EXCEPTION: hollow + permission verb + no negative = genuine permission
        # "sure go ahead" without friction = NOT sarcasm
        compressed = 0.0
        is_genuine_permission = (has_permission and not has_negative_context
                                 and not has_mundane)
        if hollow_idx and has_positive and not is_genuine_permission:
            if n <= 4:
                compressed = 0.9  # "oh joy" -- ultra-short, near-certain
            elif n <= 7:
                compressed = 0.6  # "oh how lovely" -- short with hollow opener
        # Stacked positives (3+) = over-agreement, even without hollow affirm
        # "wow thanks so much for the help" -- too positive to be real
        # 2+ positives needs hollow affirm; 3+ positives is suspicious alone
        if len(positive_idx) >= 3 and n <= 10:
            compressed = max(compressed, 0.7)
        elif len(positive_idx) >= 2 and n <= 10 and hollow_idx:
            compressed = max(compressed, 0.7)
        # "love that for you" -- positive verb + OTHER_REF + short = passive sarcasm
        has_other_ref = any(r.role == "OTHER_REF" for r in roles)
        if has_positive and has_other_ref and n <= 5 and not has_relation:
            # Only if positive word is in opener position (directed at them)
            if positive_idx and positive_idx[0] == 0:
                compressed = max(compressed, 0.6)

        # ── Feature 6: Brevity bonus ───────────────────────────
        # Short sentences with any contradiction score higher.
        brevity = 0.0
        if n <= 4:
            brevity = 0.3
        elif n <= 7:
            brevity = 0.15

        # ── Casualness gate ─────────────────────────────────────
        # Flat casual sentences with no actual contrast are NOT sarcastic.
        # "yeah that sounds ok" is casual agreement, not sarcasm.
        _CASUAL_TOKENS = frozenset({
            "yeah", "ok", "okay", "sounds", "sure", "fine",
            "alright", "guess", "kinda", "sorta", "whatever",
        })
        _SARCASM_OPENERS = frozenset({
            "oh", "clearly", "obviously", "wow", "gee", "golly",
        })
        casualness = sum(15 for r in roles if r.word in _CASUAL_TOKENS)
        has_sarcasm_opener = any(r.word in _SARCASM_OPENERS for r in roles)

        # ── Combine features ────────────────────────────────────
        # Base score = strongest single feature. Additional features boost.
        # This avoids diluting strong signals through averaging.
        features = [mock_praise, dismissive_assent, permission_hostility,
                    surface_context, whiplash, compressed]
        active_features = [f for f in features if f > 0]

        if not active_features:
            return None

        # Strongest feature is the base. Each additional feature adds a boost.
        base = max(active_features)
        n_extra = len(active_features) - 1
        boost = n_extra * 0.1  # each extra feature adds 0.1
        # Brevity modulates the final score
        score = min(0.95, base + boost + brevity * 0.15)

        # Casualness suppression: if casualness dominates the contrast signal,
        # and there's no sarcasm opener or negative words, suppress.
        # Exception: dismissive assent ("yeah right"), "what a" pattern,
        # and mock praise are structurally sarcastic regardless of casualness.
        has_structural_sarcasm = (dismissive_assent > 0.5 or mock_praise > 0.5
                                 or has_what_a)
        contrast_strength = score * 100  # scale to comparable units
        if (casualness > contrast_strength and not has_sarcasm_opener
                and not has_structural_sarcasm):
            return None
        # No negative words AND no sarcasm opener = near-zero sarcasm
        # Exception: structural sarcasm patterns (dismissive assent, mock praise,
        # "what a") survive because the STRUCTURE is the contradiction.
        if (not has_negative_context and not has_sarcasm_opener
                and not has_ironic_title and not has_structural_sarcasm):
            if mock_praise < 0.5 and whiplash < 0.3:
                return None

        # Threshold: need meaningful contradiction
        if score < 0.15:
            return None

        # Map score to confidence and V weight
        confidence = min(0.95, score)
        # V weight scales with confidence: stronger contradiction = stronger inversion
        v_weight = -20.0 - (confidence * 50.0)  # range: -30 to -67.5
        d_weight = 5.0 + (confidence * 8.0)     # range: 8 to 12.6

        matched = sorted(set(positive_idx + negative_idx +
                            hollow_idx + echo_idx + mundane_idx))

        # Build description from active features
        active = []
        if mock_praise > 0:
            active.append("mock-praise")
        if dismissive_assent > 0:
            active.append("dismissive-assent")
        if permission_hostility > 0:
            active.append("permission-hostility")
        if surface_context > 0:
            active.append("surface-context")
        if compressed > 0:
            active.append("compressed")
        if whiplash > 0.3:
            active.append(f"V-whiplash({whiplash:.2f})")
        desc = "Contradiction sarcasm: " + " + ".join(active)

        return StructureMatch(
            pattern="SARCASM_INVERSION",
            confidence=confidence,
            matched_indices=matched if matched else positive_idx,
            description=desc,
            v_weight=v_weight, d_weight=d_weight,
        )

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

        # "we/us/our" is inclusive -- speaker is part of the group.
        # DIRECTED_POSITIVE requires the positive to be directed at SOMEONE ELSE.
        _INCLUSIVE_PRONOUNS = {"we", "us", "our", "ours", "ourselves", "were", "weve"}
        if all(roles[i].word in _INCLUSIVE_PRONOUNS for i in other_indices):
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
                       "learned", "started", "stopped", "tried", "came",
                       "killed", "killedit", "crushed", "nailed", "aced",
                       "smashed", "destroyed", "dominated", "owned"}
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

    def _directed_label(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """OTHER_REF + "is/are" + heavy neutral = directed label assignment.

        "youre adopted" = exile. Assigning an identity the target didn't choose.
        "youre fat" = body attack. Pointing a descriptor AT someone.
        "youre a prostitute" = stigma assignment.

        Only fires when: OTHER_REF (you/he/she) is immediately followed by
        a linking verb frame and then a heavy-gravity word with |dV| < 20
        (neutral words that become weapons when pointed).
        """
        # Only fires on SECOND PERSON directed at listener: you/youre/your
        _SECOND_PERSON = {"you", "youre", "your", "yours", "yourself"}
        for i, r in enumerate(roles):
            if r.role != "OTHER_REF" or r.word not in _SECOND_PERSON:
                continue
            # Look for heavy neutral word within 3 positions after
            for j in range(i + 1, min(i + 4, len(roles))):
                w = roles[j]
                if w.force and abs(w.force[0]) < 20 and w.force[4] >= 15:
                    # Heavy neutral (high G, low |V|) pointed at someone
                    # Check: is there a SELF_REF earlier? If OTHER is "you" = target is listener
                    return StructureMatch(
                        pattern="DIRECTED_LABEL",
                        confidence=0.7,
                        matched_indices=[i, j],
                        description=f"Label assignment: '{r.word}' + '{w.word}' = directed identity",
                        v_weight=-15.0,
                        d_weight=10.0,   # speaker has power (assigning label)
                        g_weight=15.0,   # label carries weight
                        w_weight=-20.0,  # target's self-worth takes the hit
                    )
        return None

    # ── Slang inversion patterns ──────────────────────────────────

    def _slang_death_humor(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Laughter word near death/dying/dead = humor, not crisis.

        "lol im dead" "im dying lmao" "bro im literally dying"
        When a laughter word appears within 4 positions of a death word,
        the speaker is laughing hard, not reporting death.
        Only fires when laughter is present -- "he is actually dead" stays literal.
        """
        laughter_idx = [i for i, r in enumerate(roles) if r.word in _LAUGHTER_WORDS]
        death_idx = [i for i, r in enumerate(roles) if r.word in _DEATH_SLANG_WORDS]
        if not death_idx:
            return None
        # Primary signal: explicit laughter word present
        # Secondary signal: casual address ("bro", "dude") + hyperbole amp ("literally")
        # + death word. "bro im literally dying" = hyperbolic humor, not crisis.
        _CASUAL_ADDRESS = frozenset({"bro", "bruh", "dude", "fam", "sis", "bestie", "guys"})
        _HYPERBOLE_AMPS = frozenset({"literally", "legit", "straight"})
        has_casual = any(r.word in _CASUAL_ADDRESS for r in roles)
        has_hyperbole = any(r.word in _HYPERBOLE_AMPS for r in roles)
        if not laughter_idx:
            if has_casual and has_hyperbole:
                # Casual address + hyperbole amp = synthetic laughter signal
                laughter_idx = [i for i, r in enumerate(roles) if r.word in _CASUAL_ADDRESS]
            else:
                return None
        # Check proximity: any laughter word within 4 positions of any death word
        matched = []
        for li in laughter_idx:
            for di in death_idx:
                if abs(li - di) <= 4:
                    matched.extend([li, di])
        if not matched:
            return None
        matched = sorted(set(matched))
        # "dead serious" with laughter = still humor? Check for literal modifiers.
        _LITERAL_MODIFIERS = frozenset({"serious", "seriously", "actually", "really", "truly"})
        has_literal = any(r.word in _LITERAL_MODIFIERS for r in roles)
        # If literal modifier is adjacent to death word (not laughter), reduce confidence
        if has_literal:
            for di in death_idx:
                for j in range(max(0, di - 2), min(len(roles), di + 3)):
                    if j != di and roles[j].word in _LITERAL_MODIFIERS:
                        # "dead serious lol" -- the literal modifier attaches to death
                        # This is ambiguous, don't fire
                        return None
        conf = 0.95
        return StructureMatch(
            pattern="SLANG_DEATH_HUMOR",
            confidence=conf,
            matched_indices=matched,
            description="Laughter + death word = humor/hyperbole, not literal death",
            v_weight=+50.0, d_weight=+10.0, u_weight=-5.0,
            g_weight=-10.0, w_weight=+20.0,
        )

    # ── Crisis patterns (added 2026-04-03) ──────────────────────

    def _self_harm_intent(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + violence/lethal word + self-target = direct self-harm.

        "i want to kill myself" "im going to hurt myself" "i cut myself"
        """
        _VIOLENCE_WORDS = frozenset({
            "kill", "killed", "killing", "murder", "hurt", "harm",
            "cut", "cutting", "shoot", "stab", "hang", "drown",
            "poison", "suffocate", "strangle", "end", "destroy",
            "eliminate", "erase", "remove", "off",
        })
        _SELF_TARGET = frozenset({"myself", "me", "self"})

        has_self = any(r.role == "SELF_REF" for r in roles)
        if not has_self:
            return None
        # Laughter or casual slang cancels self-harm reading
        # "this is killing me haha" = hyperbole. "gg my tum tum hurt" = gaming/baby talk
        _CASUAL_MARKERS = frozenset({
            "gg", "ggs", "lmao", "rofl", "lmfao", "tum", "tummy",
            "owie", "ouch", "oof", "rip", "bruh", "smh", "tbh",
        })
        has_laughter = any(r.word in _LAUGHTER_WORDS for r in roles)
        has_casual = any(r.word in _CASUAL_MARKERS for r in roles)
        if has_laughter or has_casual:
            return None
        # MUNDANE SUBJECT CHECK: if a non-agentic, low-gravity atom is the
        # sentence subject, the violence is hyperbole not crisis.
        # "this homework is killing me" — homework has G=0, V=0 → suppress
        # "i want to kill myself" — subject is SELF_REF → preserve
        # Detection: any non-person role with |V| < 10 and |G| < 15 before the verb
        from .vocabulary import VOCABULARY as _V
        has_mundane_subject = False
        for r in roles:
            if r.role in ("SELF_REF", "OTHER_REF", "RELATION_REF"):
                continue  # person = not mundane
            if r.role in ("NEGATOR", "CONNECTOR", "AMPLIFIER", "COMPRESSOR",
                         "HEDGE", "FILLER", "TEMPORAL", "REGISTER_CASUAL"):
                continue  # operators = skip
            # Substantive word: check its gravity and valence
            rf = r.force or _V.get(r.word)
            if rf:
                if abs(rf[0]) < 10 and abs(rf[4]) < 15:
                    has_mundane_subject = True
                    break
            else:
                # Not in vocabulary at all = maximally mundane
                if r.role == "NEUTRAL":
                    has_mundane_subject = True
                    break
        if has_mundane_subject:
            return None
        violence_idx = [i for i, r in enumerate(roles) if r.word in _VIOLENCE_WORDS]
        self_target_idx = [i for i, r in enumerate(roles)
                         if r.word in _SELF_TARGET or (r.role == "SELF_REF" and i > 0)]
        if not violence_idx:
            return None
        for vi in violence_idx:
            for si in self_target_idx:
                if si > vi or (si < vi and vi - si <= 3):
                    return StructureMatch(
                        pattern="SELF_HARM_INTENT",
                        confidence=0.95,
                        matched_indices=sorted({vi, si}),
                        description="Direct self-harm intent: violence + self-target",
                        v_weight=-70.0, d_weight=-30.0, u_weight=40.0,
                        g_weight=50.0, w_weight=-50.0,
                    )
        return None

    def _existential_negation(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + NEGATOR + existential concept = wanting to not exist.

        "i dont want to be here anymore" "i cant do this anymore"
        "i dont want to live" "i cant be here"
        """
        _EXISTENTIAL_WORDS = frozenset({
            "be", "exist", "here", "around", "alive", "living", "live",
            "continue", "stay", "carry", "anymore",
            # "go" and "on" removed — too generic. "didnt go off" = alarm, not existential.
        })
        # "might as well" = resignation phrase. Implicit negation of self-preservation.
        # "might as well jump" = "no reason not to." Treat as existential negation.
        _RESIGNATION_PHRASES = [("might", "as", "well"), ("may", "as", "well")]
        words_lower = [r.word for r in roles]
        for phrase in _RESIGNATION_PHRASES:
            for i in range(len(words_lower) - len(phrase) + 1):
                if tuple(words_lower[i:i+len(phrase)]) == phrase:
                    # Check if a violence/exit word follows
                    _DANGER_AFTER = frozenset({
                        "jump", "die", "end", "kill", "leave", "go",
                        "quit", "stop", "disappear", "give",
                    })
                    rest = words_lower[i+len(phrase):]
                    if any(w in _DANGER_AFTER for w in rest):
                        return StructureMatch(
                            pattern="EXISTENTIAL_NEGATION",
                            confidence=0.80,
                            matched_indices=[i, i+1, i+2],
                            description="Resignation phrase + danger = implicit negation of self-preservation",
                            v_weight=-40.0, d_weight=-20.0, u_weight=15.0,
                            g_weight=35.0, w_weight=-30.0,
                        )
        # Require STRONG self-ref ("i", "me", "myself") not just possessive "my"
        # "my alarm didnt go off" ≠ existential crisis
        has_strong_self = any(r.word in _STRONG_SELF_WORDS for r in roles)
        has_negator = any(r.role == "NEGATOR" for r in roles)
        if not has_strong_self or not has_negator:
            return None
        negator_idx = [i for i, r in enumerate(roles) if r.role == "NEGATOR"]
        exist_idx = [i for i, r in enumerate(roles) if r.word in _EXISTENTIAL_WORDS]
        if not exist_idx:
            return None
        for ni in negator_idx:
            for ei in exist_idx:
                if abs(ni - ei) <= 4:
                    has_temporal = any(r.role == "TEMPORAL" for r in roles)
                    conf = 0.85 if has_temporal else 0.7
                    return StructureMatch(
                        pattern="EXISTENTIAL_NEGATION",
                        confidence=conf,
                        matched_indices=sorted({ni, ei}),
                        description="Negation of existence/continuation",
                        v_weight=-45.0, d_weight=-25.0, u_weight=20.0,
                        g_weight=40.0, w_weight=-35.0,
                    )
        return None

    def _social_nullity(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Nobody/negator + social verb + SELF_REF = social erasure.

        "nobody would miss me" "no one cares about me"
        "the world doesnt need me" "nobody would even notice"

        When the negator IS a null-subject word (nobody, nothing, no one),
        SELF_REF is implied -- the speaker is the one not being noticed/missed.
        "even" strengthens nullity (opposite of its usual dampening role).
        """
        _SOCIAL_VERBS = frozenset({
            "miss", "care", "cares", "need", "needs", "want", "wants",
            "notice", "love", "loves", "remember", "know", "knows",
            "see", "sees", "hear", "hears",
            "like", "likes", "liked", "believe", "believes",
            "understand", "understands", "trust", "trusts",
            "respect", "respects", "listen", "listens",
            "help", "helps", "support", "supports",
        })
        # Null-subject words: when these ARE the negator, SELF_REF is implied.
        # "nobody would notice" = nobody would notice [me].
        _NULL_SUBJECTS = frozenset({"nobody", "noone", "nothing"})
        # Hedge amplifiers that STRENGTHEN nullity (opposite of usual dampening)
        _NULLITY_AMPS = frozenset({"even", "ever", "really", "truly"})

        has_self = any(r.role == "SELF_REF" for r in roles)
        has_negator = any(r.role == "NEGATOR" for r in roles)
        has_null_subject = any(r.word in _NULL_SUBJECTS for r in roles)

        # Fire if: (SELF_REF + NEGATOR) OR (null-subject word as negator)
        if not has_negator:
            return None
        if not has_self and not has_null_subject:
            return None

        social_idx = [i for i, r in enumerate(roles) if r.word in _SOCIAL_VERBS]
        if not social_idx:
            return None
        negator_idx = [i for i, r in enumerate(roles) if r.role == "NEGATOR"]
        self_idx = [i for i, r in enumerate(roles) if r.role == "SELF_REF"]

        # "even" amplifies nullity strength
        has_amp = any(r.word in _NULLITY_AMPS for r in roles)
        amp_boost = 0.10 if has_amp else 0.0

        for ni in negator_idx:
            for si in social_idx:
                if abs(ni - si) <= 5:
                    conf = min(0.95, 0.85 + amp_boost)
                    return StructureMatch(
                        pattern="SOCIAL_NULLITY",
                        confidence=conf,
                        matched_indices=sorted({ni, si} | set(self_idx[:1])),
                        description="Social erasure: nobody + social verb + self",
                        v_weight=-40.0, d_weight=-20.0, u_weight=15.0,
                        g_weight=35.0, w_weight=-45.0,
                    )
        return None

    def _grief_loss(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + loss verb + RELATION_REF or heavy-G word = grief.

        "i lost my best friend" -- SELF_REF + lost + RELATION_REF
        "we lost grandpa last year" -- SELF_REF(we) + lost + RELATION_REF
        "i lost him" -- SELF_REF + lost + OTHER_REF (implies relationship)

        The positive words near the lost person ("best", "wonderful") describe
        WHO was lost, not the speaker's state. Bigger positive = bigger grief.
        """
        _LOSS_VERBS = frozenset({
            "lost", "lose", "losing", "passed", "gone", "died",
            "buried", "mourning", "grieving", "miss", "missing",
        })
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]
        loss_indices = [r.position for r in roles
                       if r.word in _LOSS_VERBS
                       or (r.role == "PULL_RESOLVED" and r.word in _LOSS_VERBS)]

        if not loss_indices:
            return None

        # Recovery words cancel grief: "found my lost dog" = happy ending
        _RECOVERY_WORDS = frozenset({
            "found", "find", "reunited", "recovered", "returned",
            "back", "saved", "rescued", "alive", "safe",
        })
        has_recovery = any(r.word in _RECOVERY_WORDS for r in roles)
        if has_recovery:
            return None

        # Need a target: RELATION_REF, or OTHER_REF/heavy-G AFTER the loss verb.
        # "i lost my best friend" -- friend(RELATION) after lost = target.
        # "we lost the game" -- "we"(OTHER) BEFORE lost = subject, not target.
        # Target must be after (or within 1 before) the first loss verb.
        first_loss = min(loss_indices)
        relation_indices = [r.position for r in roles
                          if r.role == "RELATION_REF" and r.position >= first_loss - 1]
        # OTHER_REF only counts as target if it's after the loss verb
        # (pronouns before are subjects: "we lost", "they lost")
        other_after = [r.position for r in roles
                      if r.role == "OTHER_REF" and r.position > first_loss]
        # Also catch heavy-G words (pet names, roles) even if not classified
        from .vocabulary import VOCABULARY
        loss_set = set(loss_indices)
        heavy_g_indices = [r.position for r in roles
                         if r.word in VOCABULARY and VOCABULARY[r.word][4] >= 15
                         and r.role not in ("SELF_REF", "NEGATOR", "CONNECTOR")
                         and r.position >= first_loss - 1
                         and r.position not in loss_set]
        target_indices = list(set(relation_indices + other_after + heavy_g_indices))

        if not target_indices:
            return None

        # Find best loss-target pair
        best_dist = 999
        best_l, best_t = -1, -1
        for li in loss_indices:
            for ti in target_indices:
                d = abs(li - ti)
                if d < best_dist:
                    best_dist = d
                    best_l, best_t = li, ti

        if best_dist > 6:
            return None

        # Confidence: SELF_REF present = stronger (personal loss)
        confidence = 0.8 if self_indices else 0.6

        # Scale with how positive the lost thing is described
        # "best friend" has "best" V=+50. The bigger the positive,
        # the bigger the grief (you lost something GOOD).
        pos_near_target = 0
        for r in roles:
            if r.role == "EMOTIONAL" and r.force and r.force[0] > 10:
                if abs(r.position - best_t) <= 3:
                    pos_near_target += r.force[0]
        grief_scale = 1.0 + min(pos_near_target / 50.0, 1.5)

        indices = sorted(set(loss_indices + target_indices +
                           (self_indices[:1] if self_indices else [])))
        return StructureMatch(
            pattern="GRIEF_LOSS",
            confidence=min(confidence, 1.0),
            matched_indices=indices,
            description="Loss of person/relationship -- grief",
            v_weight=-35.0 * grief_scale,
            d_weight=-10.0,
            u_weight=10.0,
            g_weight=30.0,
            w_weight=-10.0,
        )

    def _reported_comfort(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """[everyone/they/people] + [say/says/said/tell/told] + positive = reported speech.

        "everyone says it gets easier" -- the speaker does NOT believe this
        "they told me it would get better" -- reported comfort, not felt
        "people say time heals" -- generic advice, not speaker's state

        The positive content is someone ELSE's claim. Dampen it.
        """
        _REPORT_SUBJECTS = frozenset({
            "everyone", "everybody", "they", "people", "someone",
            "somebody", "others", "friends", "family",
        })
        _REPORT_VERBS = frozenset({
            "say", "says", "said", "tell", "tells", "told",
            "think", "thinks", "claim", "claims", "promise",
            "promised", "insist", "insists",
        })

        subject_indices = [r.position for r in roles
                          if r.word in _REPORT_SUBJECTS]
        verb_indices = [r.position for r in roles
                       if r.word in _REPORT_VERBS]

        if not subject_indices or not verb_indices:
            return None

        # Subject must be near verb (within 3 words)
        best_dist = 999
        best_s, best_v = -1, -1
        for si in subject_indices:
            for vi in verb_indices:
                d = abs(si - vi)
                if d < best_dist and d <= 3:
                    best_dist = d
                    best_s, best_v = si, vi

        if best_s == -1:
            return None

        # Check if positive content follows the report verb
        has_positive_after = False
        for r in roles:
            if r.position > best_v:
                if r.role == "EMOTIONAL" and r.force and r.force[0] > 10:
                    has_positive_after = True
                    break
                # Also catch positive-meaning neutral words
                if r.word in ("easier", "better", "fine", "okay",
                              "alright", "heals", "passes", "improves"):
                    has_positive_after = True
                    break

        if not has_positive_after:
            return None

        indices = sorted({best_s, best_v})
        return StructureMatch(
            pattern="REPORTED_COMFORT",
            confidence=0.75,
            matched_indices=indices,
            description="Reported speech -- speaker relaying others' comfort, not believing it",
            v_weight=-25.0,
            d_weight=-5.0,
            u_weight=5.0,
            g_weight=10.0,
            w_weight=-5.0,
        )

    def _rhetorical_self_negation(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """[why/how] + [would/could/should] + [anyone/somebody/someone] + positive + SELF_REF.

        "why would anyone love me" = nobody would love me
        "how could anyone want me" = nobody wants me
        "who would ever care about me" = nobody cares

        The rhetorical question frame INVERTS the positive emotion.
        "love" becomes "absence of love directed at self."
        This is a self-worth attack: the speaker pre-rejects themselves.
        """
        _RHETORICAL_QW = frozenset({"why", "how", "who", "whos"})
        _MODAL_VERBS = frozenset({
            "would", "could", "should", "will", "can",
            "wouldnt", "couldn't", "shouldnt",
        })
        _INDEFINITE_SUBJECTS = frozenset({
            "anyone", "anybody", "someone", "somebody",
            "everyone", "everybody",
        })

        # Step 1: rhetorical question word in first 2 positions
        has_qw = any(r.word in _RHETORICAL_QW for r in roles[:2])
        if not has_qw:
            return None

        # Step 2: modal verb present
        modal_indices = [r.position for r in roles if r.word in _MODAL_VERBS]
        if not modal_indices:
            return None

        # Step 3: indefinite subject OR "ever" (intensifier of impossibility)
        has_indefinite = any(r.word in _INDEFINITE_SUBJECTS for r in roles)
        has_ever = any(r.word == "ever" for r in roles)

        if not has_indefinite and not has_ever:
            return None

        # Step 4: SELF_REF present (the target of the rhetorical question)
        self_indices = [r.position for r in roles if r.role == "SELF_REF"]
        if not self_indices:
            return None

        # Step 5: positive emotional word present (the thing being denied)
        pos_indices = [r.position for r in roles
                      if r.role == "EMOTIONAL" and r.force and r.force[0] > 15]
        # Also check for positive social verbs that may not be EMOTIONAL
        _POSITIVE_SOCIAL = frozenset({
            "love", "want", "care", "like", "need", "miss",
            "choose", "pick", "hire", "accept", "forgive",
        })
        social_pos_indices = [r.position for r in roles
                            if r.word in _POSITIVE_SOCIAL]
        all_pos_indices = sorted(set(pos_indices + social_pos_indices))

        if not all_pos_indices:
            return None

        # Find the strongest positive word to scale the inversion
        max_pos_v = 0
        for r in roles:
            if r.role == "EMOTIONAL" and r.force and r.force[0] > max_pos_v:
                max_pos_v = r.force[0]
        # Social verbs without vocab force get baseline
        if max_pos_v == 0:
            max_pos_v = 40

        # Scale: bigger positive = bigger inversion
        # "love" at +127 = devastating. "like" at +20 = milder.
        intensity = min(max_pos_v / 50.0, 3.0)

        confidence = 0.9
        if has_indefinite and has_ever:
            confidence = 0.95  # "why would anyone EVER love me" = maximum

        indices = sorted(set(modal_indices + all_pos_indices + self_indices))
        return StructureMatch(
            pattern="RHETORICAL_SELF_NEGATION",
            confidence=confidence,
            matched_indices=indices,
            description="Rhetorical question inverting positive -- self-worth attack",
            v_weight=-60.0 * intensity,
            d_weight=-15.0,
            u_weight=10.0,
            g_weight=20.0,
            w_weight=-40.0 * intensity,
        )

    def _rhetorical_hopelessness(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Interrogative + exit concept = rhetorical negation of purpose.

        "whats the point of living" "why bother" "why even try"
        """
        _RHETORICAL_OPENERS = frozenset({"whats", "what", "why", "whos"})
        _PURPOSE_WORDS = frozenset({
            "point", "purpose", "reason", "bother", "try", "use",
            "matter", "difference", "good",
        })
        has_opener = any(r.word in _RHETORICAL_OPENERS for r in roles[:3])
        if not has_opener:
            return None
        purpose_idx = [i for i, r in enumerate(roles) if r.word in _PURPOSE_WORDS]
        if not purpose_idx:
            return None
        _EXIST = frozenset({"living", "life", "alive", "trying", "going", "anymore"})
        has_existential = any(r.word in _EXIST for r in roles)
        conf = 0.85 if has_existential else 0.65
        return StructureMatch(
            pattern="RHETORICAL_HOPELESSNESS",
            confidence=conf,
            matched_indices=purpose_idx,
            description="Rhetorical question negating purpose/reason",
            v_weight=-35.0, d_weight=-20.0, u_weight=10.0,
            g_weight=30.0, w_weight=-30.0,
        )


    # ── Passive-aggressive patterns (added 2026-04-03) ──────────

    def _passive_resignation(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Permission/agreement words that mask underlying negative.

        "whatever makes you happy" "do what you want" "if thats what you think"
        "whatever you say" "i guess i deserved it" "its not like i care"

        These are surrender statements -- the speaker yields control while
        signaling resentment. The surface reads as agreement/permission but
        the underlying state is withdrawal + lowered self-worth.
        """
        words = [r.word for r in roles]
        words_lower = [w.lower() for w in words]
        n = len(words_lower)

        # Pattern 1: resignation opener + OTHER_REF + action/opinion verb
        # "whatever makes you happy" "do what you want" "whatever you say"
        # "if thats what you think"
        _RESIGN_OPENERS = frozenset({"whatever", "do", "go", "sure", "fine", "if"})
        _YIELD_PHRASES = frozenset({"want", "think", "say", "like", "wish",
                                    "prefer", "feel", "believe", "need"})
        has_opener = n > 0 and words_lower[0] in _RESIGN_OPENERS
        has_other = any(r.role == "OTHER_REF" for r in roles)
        has_yield_verb = any(w in _YIELD_PHRASES for w in words_lower)

        if has_opener and has_other and has_yield_verb:
            matched = [0] + [i for i, r in enumerate(roles) if r.role == "OTHER_REF"]
            # "whatever makes you happy" = resignation despite positive word
            # Higher confidence if positive emotional word present (mask)
            has_positive = any(r.role == "EMOTIONAL" and r.force and r.force[0] > 15
                              for r in roles)
            conf = 0.85 if has_positive else 0.75
            return StructureMatch(
                pattern="PASSIVE_RESIGNATION",
                confidence=conf,
                matched_indices=sorted(set(matched)),
                description="Resignation disguised as permission/agreement",
                v_weight=-25.0, d_weight=-20.0, u_weight=0.0,
                g_weight=0.0, w_weight=-15.0,
            )

        # Pattern 2: "whatever" + OTHER_REF (short form)
        # "whatever you say" -- just yielding
        if "whatever" in words_lower and has_other and n <= 5:
            wi = words_lower.index("whatever")
            return StructureMatch(
                pattern="PASSIVE_RESIGNATION",
                confidence=0.80,
                matched_indices=[wi],
                description="Resignation: 'whatever' + addressee = yielding control",
                v_weight=-25.0, d_weight=-20.0, u_weight=0.0,
                g_weight=0.0, w_weight=-15.0,
            )

        # Pattern 3: "i guess" / "i suppose" + anything
        # "i guess i deserved it" "i suppose youre right"
        # The hedge signals the speaker doesn't believe it but won't fight.
        for i in range(n - 1):
            if words_lower[i] == "i" and words_lower[i + 1] in ("guess", "suppose"):
                matched = [i, i + 1]
                return StructureMatch(
                    pattern="PASSIVE_RESIGNATION",
                    confidence=0.75,
                    matched_indices=matched,
                    description="Passive resignation: 'I guess/suppose' = yielding without believing",
                    v_weight=-25.0, d_weight=-20.0, u_weight=0.0,
                    g_weight=0.0, w_weight=-15.0,
                )

        # Pattern 4: "its not like i care/matter"
        # Negation + "like" + SELF_REF + positive verb = denial masking hurt
        if "not" in words_lower and "like" in words_lower:
            has_self = any(r.role == "SELF_REF" for r in roles)
            _CARE_WORDS = frozenset({"care", "matter", "count", "mean"})
            has_care = any(w in _CARE_WORDS for w in words_lower)
            if has_self and has_care:
                not_idx = words_lower.index("not")
                like_idx = words_lower.index("like")
                if abs(not_idx - like_idx) <= 2:
                    return StructureMatch(
                        pattern="PASSIVE_RESIGNATION",
                        confidence=0.85,
                        matched_indices=[not_idx, like_idx],
                        description="Denial masking hurt: 'not like I care' = I do care",
                        v_weight=-25.0, d_weight=-20.0, u_weight=0.0,
                        g_weight=0.0, w_weight=-15.0,
                    )

        # Pattern 5: "im fine" -- ultra-short self-report of okayness
        # SELF_REF + PEACE role in <= 3 word sentence = suspicious brevity
        if n <= 3:
            has_self = any(r.role == "SELF_REF" for r in roles)
            has_peace = any(r.role == "PEACE" for r in roles)
            if has_self and has_peace:
                matched = [i for i, r in enumerate(roles)
                          if r.role in ("SELF_REF", "PEACE")]
                return StructureMatch(
                    pattern="PASSIVE_RESIGNATION",
                    confidence=0.70,
                    matched_indices=matched,
                    description="Suspicious brevity: minimal self-report of okayness",
                    v_weight=-25.0, d_weight=-20.0, u_weight=0.0,
                    g_weight=0.0, w_weight=-15.0,
                )

        return None

    def _atmospheric_grief(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Ghost possession + persistence + absence by omission = atmospheric grief.

        "his chair is still at the table" -- no grief WORDS but grief through
        STRUCTURE: a possessive object persists where a person doesn't.
        The possessor's object exists. The possessor has no active verb.
        Object permanence encodes absence.

        Signals:
          1. Ghost possession: possessive pronoun + intimate object within 3 words
          2. Persistence marker: still, always, remains, untouched, waiting, etc.
          3. Absence by omission: possessor has NO active verb in the sentence
          4. Domestic scene: table, bed, closet, kitchen, etc.
        """
        words_lower = [r.word for r in roles]
        n = len(words_lower)

        # ── Signal sets ──
        _POSSESSIVE_PRONOUNS = frozenset({
            "his", "her", "their",
        })
        _INTIMATE_OBJECTS = frozenset({
            "chair", "coat", "shoes", "cup", "mug", "pillow", "side",
            "toothbrush", "keys", "jacket", "glasses", "hat", "desk",
            "room", "bed", "phone", "plate", "spot", "place", "seat",
            "clothes", "ring", "watch", "photo", "picture", "letter",
            "necklace", "bracelet", "scarf", "sweater", "shirt",
            "boots", "slippers", "notebook", "journal", "wallet",
        })
        _PERSISTENCE = frozenset({
            "still", "always", "remains", "untouched", "waiting",
            "every", "same", "hasnt", "hasn't", "havent", "haven't",
            "never", "moved", "changed",
        })
        _DOMESTIC_SCENE = frozenset({
            "table", "bed", "closet", "kitchen", "hallway", "room",
            "porch", "door", "window", "drawer", "shelf", "counter",
            "nightstand", "bathroom", "garage", "attic", "basement",
            "bedroom", "living", "dining", "couch", "sofa",
        })
        # Verbs that make the possessor ACTIVE (cancels absence)
        _ACTIVE_VERBS = frozenset({
            "sat", "sits", "sitting", "stood", "stands", "standing",
            "walked", "walks", "walking", "ran", "runs", "running",
            "came", "comes", "coming", "went", "goes", "going",
            "said", "says", "saying", "told", "tells", "telling",
            "gave", "gives", "giving", "took", "takes", "taking",
            "put", "puts", "putting", "made", "makes", "making",
            "left", "leaves", "leaving", "called", "calls", "calling",
            "looked", "looks", "looking", "moved", "moves", "moving",
            "ate", "eats", "eating", "drank", "drinks", "drinking",
            "wore", "wears", "wearing", "held", "holds", "holding",
            "opened", "opens", "opening", "closed", "closes", "closing",
            "brought", "brings", "bringing", "picked", "picks", "picking",
            "grabbed", "grabs", "grabbing", "used", "uses", "using",
        })
        # Discovery verbs: "i found her necklace" = the FINDER acts,
        # but the possessor (her) is still absent. These do NOT cancel absence.
        _DISCOVERY_VERBS = frozenset({
            "found", "find", "finding", "discovered", "noticed",
            "saw", "see", "seeing", "spotted",
        })

        # ── Signal 1: Ghost possession ──
        # Possessive pronoun + intimate object within 3 words
        ghost_score = 0.0
        ghost_indices = []
        possessor_pronoun = None
        for i, w in enumerate(words_lower):
            if w in _POSSESSIVE_PRONOUNS:
                possessor_pronoun = w
                # Look for intimate object within 3 words ahead
                for j in range(i + 1, min(i + 4, n)):
                    if words_lower[j] in _INTIMATE_OBJECTS:
                        ghost_score = 1.0
                        ghost_indices = [i, j]
                        break
                if ghost_score > 0:
                    break
        # Also check for [name]'s pattern (word ending in 's before object)
        if ghost_score == 0:
            for i, w in enumerate(words_lower):
                if w.endswith("s") and len(w) > 2 and i + 1 < n:
                    # Could be possessive 's -- check if next words have intimate object
                    for j in range(i + 1, min(i + 4, n)):
                        if words_lower[j] in _INTIMATE_OBJECTS:
                            ghost_score = 0.7  # lower confidence for name's pattern
                            ghost_indices = [i, j]
                            possessor_pronoun = w
                            break
                    if ghost_score > 0:
                        break

        if ghost_score == 0:
            # No ghost possession at all -- check for bare object + persistence
            # "the coffee mug hasnt moved" -- no possessive but object + persistence
            has_intimate = any(w in _INTIMATE_OBJECTS for w in words_lower)
            has_persistence = any(w in _PERSISTENCE for w in words_lower)
            if has_intimate and has_persistence:
                ghost_score = 0.5  # weaker signal without possessive
                for i, w in enumerate(words_lower):
                    if w in _INTIMATE_OBJECTS:
                        ghost_indices = [i]
                        break
            else:
                return None

        # ── Signal 2: Persistence marker ──
        persistence_score = 0.0
        persistence_indices = []
        for i, w in enumerate(words_lower):
            if w in _PERSISTENCE:
                persistence_score = 1.0
                persistence_indices.append(i)
        # Two-word persistence: "never moved", "same place", "every morning"
        for i in range(n - 1):
            pair = (words_lower[i], words_lower[i + 1])
            if pair in (("never", "moved"), ("same", "place"), ("every", "morning"),
                        ("hasnt", "changed"), ("hasn't", "changed"),
                        ("hasnt", "moved"), ("hasn't", "moved")):
                persistence_score = 1.0
                persistence_indices.extend([i, i + 1])

        # ── Signal 3: Absence by omission ──
        # The possessor has NO active verb. The object exists, the person doesn't act.
        absence_score = 0.0
        # Check: does ANY word in the sentence represent the possessor doing something?
        # "she sat in her chair" -- "sat" is active verb + "she" is the possessor = NOT absent
        possessor_is_active = False
        if possessor_pronoun:
            # Map possessive to subject form
            _POSSESSIVE_TO_SUBJECT = {
                "his": {"he", "hes", "he's"},
                "her": {"she", "shes", "she's"},
                "their": {"they", "theyre", "they're", "theyre"},
            }
            subject_forms = _POSSESSIVE_TO_SUBJECT.get(possessor_pronoun, set())
            # Check if subject form appears with an active verb nearby
            for i, w in enumerate(words_lower):
                if w in subject_forms:
                    # Subject pronoun found -- check for active verb within 3 words
                    for j in range(max(0, i - 3), min(n, i + 4)):
                        if words_lower[j] in _ACTIVE_VERBS:
                            possessor_is_active = True
                            break
                if possessor_is_active:
                    break

        if possessor_is_active:
            return None  # Person is present and acting -- no atmospheric grief

        # Also check if SELF_REF has a discovery verb (finder is active, possessor absent)
        # "i found her necklace" = I am active (finder), she is absent (possessor)
        has_discovery = any(w in _DISCOVERY_VERBS for w in words_lower)

        # Score absence: possessor has no active verb AND (persistence or discovery).
        # Ghost possession alone is NOT enough -- "his chair is comfortable" is
        # a statement about furniture, not grief. Need a second structural signal.
        if persistence_score > 0 or has_discovery:
            absence_score = 1.0

        # ── Signal 4: Domestic scene (bonus) ──
        domestic_bonus = 0.0
        for w in words_lower:
            if w in _DOMESTIC_SCENE:
                domestic_bonus = 0.1
                break

        # ── Composite score ──
        # Require at least 2 of 3 main signals. Ghost possession alone
        # is just describing furniture ("his chair is comfortable").
        signal_count = (int(ghost_score > 0) +
                        int(persistence_score > 0) +
                        int(absence_score > 0))
        if signal_count < 2:
            return None

        g_atmospheric = (ghost_score * 0.4 +
                         persistence_score * 0.35 +
                         absence_score * 0.25)

        if g_atmospheric < 0.3:
            return None

        # VADUGWI push: grief is low-arousal, high-gravity, low-dominance
        v_push = -g_atmospheric * 35
        d_push = -g_atmospheric * 15
        g_push = g_atmospheric * 25
        u_push = g_atmospheric * 15
        w_push = -g_atmospheric * 10

        all_indices = sorted(set(ghost_indices + persistence_indices))

        return StructureMatch(
            pattern="ATMOSPHERIC_GRIEF",
            confidence=min(1.0, g_atmospheric + domestic_bonus),
            matched_indices=all_indices,
            description="Atmospheric grief: object permanence encodes absence",
            v_weight=v_push,
            d_weight=d_push,
            u_weight=u_push,
            g_weight=g_push,
            w_weight=w_push,
        )

    def _hollow_agreement(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Apparent agreement that signals withdrawal.

        "sure go ahead" "go ahead" "if you say so"
        Short sentences starting with agreement words = hollow compliance.

        The speaker technically agrees but the brevity and word choice
        signal emotional withdrawal rather than genuine agreement.
        """
        words = [r.word for r in roles]
        words_lower = [w.lower() for w in words]
        n = len(words_lower)

        # Pattern 1: agreement opener + short sentence (<=5 words)
        # "sure" "yeah" "ok" "fine" "whatever" as first word
        _HOLLOW_OPENERS = frozenset({"sure", "yeah", "ok", "okay", "fine", "whatever"})
        if n <= 5 and n >= 1 and words_lower[0] in _HOLLOW_OPENERS:
            # Exclude if there's a strong positive emotional word after opener
            # "sure I love it" = genuine. "sure go ahead" = hollow.
            has_strong_positive = any(
                r.role == "EMOTIONAL" and r.force and r.force[0] > 30
                for r in roles[1:]
            )
            if not has_strong_positive:
                # Single word "whatever" or "fine" = very hollow
                conf = 0.80 if n <= 2 else 0.65
                return StructureMatch(
                    pattern="HOLLOW_AGREEMENT",
                    confidence=conf,
                    matched_indices=[0],
                    description="Hollow agreement: brief compliance signals withdrawal",
                    v_weight=-15.0, d_weight=-15.0, u_weight=0.0,
                    g_weight=0.0, w_weight=0.0,
                )

        # Pattern 2: "go ahead" -- ceding control
        for i in range(n - 1):
            if words_lower[i] == "go" and words_lower[i + 1] == "ahead":
                return StructureMatch(
                    pattern="HOLLOW_AGREEMENT",
                    confidence=0.70,
                    matched_indices=[i, i + 1],
                    description="Hollow agreement: 'go ahead' = ceding control",
                    v_weight=-15.0, d_weight=-15.0, u_weight=0.0,
                    g_weight=0.0, w_weight=0.0,
                )

        # Pattern 3: "if you say so"
        if n >= 4:
            text_joined = " ".join(words_lower)
            if "if you say so" in text_joined:
                return StructureMatch(
                    pattern="HOLLOW_AGREEMENT",
                    confidence=0.75,
                    matched_indices=list(range(n)),
                    description="Hollow agreement: 'if you say so' = doubting but yielding",
                    v_weight=-15.0, d_weight=-15.0, u_weight=0.0,
                    g_weight=0.0, w_weight=0.0,
                )

        return None

    # ── Recovery milestone pattern ──────────────────────────────

    def _recovery_milestone(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """RECOVERY_TOKEN + duration + ongoing marker = recovery milestone.

        "clean for 6 months now" "sober for 3 years"
        "free for two weeks and counting" "clear for 90 days now"

        The recovery word alone is mildly positive. The DURATION is what
        makes this a milestone worth celebrating. V_boost scales with log
        of duration in days.
        """
        _RECOVERY_TOKENS = frozenset({
            "clean", "sober", "free", "clear", "recovered",
            "recovering", "healing", "abstinent",
        })
        _TIME_UNITS = {
            "day": 1, "days": 1,
            "week": 7, "weeks": 7,
            "month": 30, "months": 30,
            "year": 365, "years": 365,
        }
        _NUMBER_WORDS = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "twenty": 20,
            "thirty": 30, "sixty": 60, "ninety": 90, "hundred": 100,
        }
        _ONGOING_MARKERS = frozenset({
            "now", "today", "still", "counting", "strong",
        })

        recovery_idx = [i for i, r in enumerate(roles) if r.word in _RECOVERY_TOKENS]
        if not recovery_idx:
            return None

        words = [r.word for r in roles]

        # Find a number near a time unit
        duration_days = 0
        number_val = 0
        time_idx = -1
        for i, r in enumerate(roles):
            # Check for digit
            if r.word.isdigit():
                number_val = int(r.word)
            elif r.word in _NUMBER_WORDS:
                number_val = _NUMBER_WORDS[r.word]
            # Check for time unit following a number
            if r.word in _TIME_UNITS and number_val > 0:
                duration_days = number_val * _TIME_UNITS[r.word]
                time_idx = i
                break

        if duration_days == 0:
            return None

        # Check: recovery token should be before the duration
        if not any(ri < time_idx for ri in recovery_idx):
            return None

        # Ongoing marker boosts confidence
        has_ongoing = any(r.word in _ONGOING_MARKERS for r in roles)
        conf = 0.90 if has_ongoing else 0.80

        # V_boost = 40 + 8 * log(1 + duration_in_days)
        v_boost = 40.0 + 8.0 * log(1 + duration_days)

        matched = sorted(set(recovery_idx + [time_idx]))
        return StructureMatch(
            pattern="RECOVERY_MILESTONE",
            confidence=conf,
            matched_indices=matched,
            description=f"Recovery milestone: {duration_days} days",
            v_weight=v_boost, d_weight=15.0, u_weight=0.0,
            g_weight=-10.0, w_weight=25.0,
        )

    # ── Syntactic resolver ─────────────────────────────────────

    def _contradiction_resolve(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Head word governs polarity when positive and negative coexist.

        "painfully beautiful" -> adjective head "beautiful" is positive.
        "i hate how much i love you" -> main verb "hate" dominates.
        "sweet revenge" -> noun "revenge" is negative, "sweet" modifies.
        "it hurts so good" -> "so good" qualifies the verb, result positive.

        Rule: In [adverb + adjective], the ADJECTIVE governs polarity.
        Rule: In [adjective + noun], the NOUN governs polarity.
        Rule: In [verb + clause], the MAIN VERB governs unless a
              qualifying complement ("so good") overrides.
        """
        from .vocabulary import VOCABULARY

        words = [r.word for r in roles]
        words_lower = [w.lower() for w in words]
        n = len(words_lower)

        if n < 2:
            return None

        # Check structural triggers before doing full scan
        has_adverb_pattern = any(w in _INTENSIFYING_ADVERBS for w in words_lower)
        has_noun_pattern = any(w in _NEGATIVE_NOUN_HEADS for w in words_lower)
        has_neg_verb = any(w in _NEGATIVE_MAIN_VERBS for w in words_lower)
        has_complement = False
        for phrase in _POSITIVE_COMPLEMENT_PHRASES:
            plen = len(phrase)
            for i in range(n - plen + 1):
                if all(words_lower[i + k] == phrase[k] for k in range(plen)):
                    has_complement = True
                    break
            if has_complement:
                break

        if not (has_adverb_pattern or has_noun_pattern or has_neg_verb or has_complement):
            return None

        # Collect positive and negative word positions
        pos_indices = []
        neg_indices = []
        for i, r in enumerate(roles):
            f = r.force or VOCABULARY.get(r.word)
            if f:
                if f[0] > 10:
                    pos_indices.append(i)
                elif f[0] < -10:
                    neg_indices.append(i)

        # ── Pattern 1: Intensifying adverb + positive adjective ──
        # "painfully beautiful" -> positive, boosted
        for i in range(n - 1):
            if words_lower[i] in _INTENSIFYING_ADVERBS:
                for j in range(i + 1, min(i + 3, n)):
                    if words_lower[j] in _POSITIVE_ADJECTIVE_HEADS:
                        f = VOCABULARY.get(words_lower[j])
                        v_boost = f[0] * 1.3 if f else 35.0
                        return StructureMatch(
                            pattern="CONTRADICTION_RESOLVE",
                            confidence=0.90,
                            matched_indices=[i, j],
                            description=f"Head adjective '{words_lower[j]}' governs: adverb intensifies",
                            v_weight=v_boost,
                            d_weight=5.0, u_weight=0.0,
                            g_weight=0.0, w_weight=0.0,
                        )

        # ── Pattern 2: Positive adjective + negative noun head ──
        # "sweet revenge" -> negative noun governs
        for i in range(n - 1):
            f_i = VOCABULARY.get(words_lower[i])
            if f_i and f_i[0] > 10:
                for j in range(i + 1, min(i + 3, n)):
                    if words_lower[j] in _NEGATIVE_NOUN_HEADS:
                        f_j = VOCABULARY.get(words_lower[j])
                        v_push = min(f_j[0] * 1.2, -15.0) if f_j else -25.0
                        return StructureMatch(
                            pattern="CONTRADICTION_RESOLVE",
                            confidence=0.85,
                            matched_indices=[i, j],
                            description=f"Head noun '{words_lower[j]}' governs: adjective modifies",
                            v_weight=v_push,
                            d_weight=5.0, u_weight=5.0,
                            g_weight=0.0, w_weight=0.0,
                        )

        # ── Pattern 3: Verb + positive qualifying complement ──
        # "it hurts so good" -> "so good" overrides verb
        for phrase in _POSITIVE_COMPLEMENT_PHRASES:
            plen = len(phrase)
            for i in range(n - plen + 1):
                if all(words_lower[i + k] == phrase[k] for k in range(plen)):
                    # Found complement -- check for preceding negative verb
                    has_preceding_neg = any(
                        (r.force or VOCABULARY.get(r.word) or (0,))[0] < -10
                        for r in roles[:i]
                    )
                    if has_preceding_neg:
                        return StructureMatch(
                            pattern="CONTRADICTION_RESOLVE",
                            confidence=0.85,
                            matched_indices=list(range(i, i + plen)),
                            description=f"Qualifying complement '{' '.join(phrase)}' overrides verb",
                            v_weight=30.0,
                            d_weight=5.0, u_weight=0.0,
                            g_weight=0.0, w_weight=0.0,
                        )

        # ── Pattern 4: Negative main verb + clause with positive ──
        # "i hate how much i love you" -> hate dominates
        for i, r in enumerate(roles):
            if r.word in _NEGATIVE_MAIN_VERBS:
                has_later_positive = any(j > i and j in pos_indices for j in pos_indices)
                if has_later_positive:
                    f = r.force or VOCABULARY.get(r.word)
                    v_push = f[0] * 1.2 if f else -35.0
                    matched = [i] + [j for j in pos_indices if j > i]
                    return StructureMatch(
                        pattern="CONTRADICTION_RESOLVE",
                        confidence=0.85,
                        matched_indices=matched,
                        description=f"Main verb '{r.word}' governs polarity over clause",
                        v_weight=v_push,
                        d_weight=5.0, u_weight=5.0,
                        g_weight=0.0, w_weight=0.0,
                    )

        return None

    # ── Pragmatic convention detectors ─────────────────────────

    def _numbers_context(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Numbers that deviate from human norms create emotional signal.

        sleep < 5 hours -> negative
        "only one" + social event -> negative (isolation)
        """
        words = [r.word for r in roles]
        words_lower = [w.lower() for w in words]
        n = len(words_lower)

        # ── Sleep deprivation: number < 5 near sleep words ──
        for i, w in enumerate(words_lower):
            if w.isdigit():
                num = int(w)
                if num <= 4:
                    window = words_lower[max(0, i - 3): min(n, i + 4)]
                    if any(sw in _SLEEP_WORDS for sw in window):
                        return StructureMatch(
                            pattern="NUMBERS_CONTEXT",
                            confidence=0.80,
                            matched_indices=[i],
                            description=f"Sleep deprivation: {num} hours",
                            v_weight=-20.0, d_weight=-10.0, u_weight=10.0,
                            g_weight=5.0, w_weight=-5.0,
                        )

        # ── "only one/no one" + social event = isolation ──
        has_social = any(w in _SOCIAL_EVENTS for w in words_lower)
        if has_social:
            for i in range(n - 1):
                if words_lower[i] == "only":
                    next_w = words_lower[i + 1]
                    if next_w in ("one", "1", "two", "2", "me"):
                        social_idx = [j for j, w in enumerate(words_lower)
                                     if w in _SOCIAL_EVENTS]
                        return StructureMatch(
                            pattern="NUMBERS_CONTEXT",
                            confidence=0.75,
                            matched_indices=[i, i + 1] + social_idx,
                            description="Social isolation: 'only one' at social event",
                            v_weight=-20.0, d_weight=-10.0, u_weight=0.0,
                            g_weight=10.0, w_weight=-10.0,
                        )
            for i in range(n - 1):
                if words_lower[i] == "no" and words_lower[i + 1] == "one":
                    social_idx = [j for j, w in enumerate(words_lower)
                                 if w in _SOCIAL_EVENTS]
                    return StructureMatch(
                        pattern="NUMBERS_CONTEXT",
                        confidence=0.80,
                        matched_indices=[i, i + 1] + social_idx,
                        description="Social isolation: 'no one' at social event",
                        v_weight=-25.0, d_weight=-15.0, u_weight=5.0,
                        g_weight=10.0, w_weight=-15.0,
                    )

        return None

    def _negated_negative_compliment(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """NEGATOR + negative word + OTHER_REF = positive compliment.

        "I couldn't have done it without you" = positive
        "You're the reason I didn't give up" = positive
        "I wouldn't be here without you" = positive
        """
        words = [r.word for r in roles]
        words_lower = [w.lower() for w in words]
        n = len(words_lower)

        # Need: negator present (role-based OR common contractions)
        neg_indices = [r.position for r in roles if r.role == "NEGATOR"]
        # Also check for contractions not in NEGATOR set
        _EXTRA_NEGATORS = frozenset({
            "couldnt", "couldn't", "wouldnt", "wouldn't",
            "cant", "can't", "wont", "won't",
        })
        for i, w in enumerate(words_lower):
            if w in _EXTRA_NEGATORS and i not in neg_indices:
                neg_indices.append(i)
        if not neg_indices:
            return None

        # Need: OTHER_REF or "you" present
        other_indices = [r.position for r in roles
                        if r.role == "OTHER_REF" or r.word in ("you", "your", "yours")]
        if not other_indices:
            return None

        # Check for compliment framing
        has_without_you = False
        for i in range(n - 1):
            if words_lower[i] == "without" and words_lower[i + 1] in ("you", "your"):
                has_without_you = True
                break

        has_reason = "reason" in words_lower
        has_thanks = any(w in ("thanks", "thank", "grateful", "thankful")
                        for w in words_lower)
        has_gratitude_verb = any(w in _GRATITUDE_VERBS for w in words_lower)

        if has_without_you:
            matched = neg_indices + other_indices
            return StructureMatch(
                pattern="NEGATED_NEGATIVE_COMPLIMENT",
                confidence=0.90,
                matched_indices=sorted(set(matched)),
                description="Negated negative = positive compliment ('without you')",
                v_weight=35.0, d_weight=5.0, u_weight=0.0,
                g_weight=-5.0, w_weight=10.0,
            )

        if has_reason and neg_indices:
            matched = neg_indices + other_indices
            for ni in neg_indices:
                _NEGATED_VERBS = frozenset({
                    "give", "gave", "fail", "failed", "quit",
                    "stop", "stopped", "fall", "fell",
                }) | _GRATITUDE_VERBS
                nearby_verbs = [r for r in roles
                               if abs(r.position - ni) <= 3
                               and r.word in _NEGATED_VERBS]
                if nearby_verbs:
                    return StructureMatch(
                        pattern="NEGATED_NEGATIVE_COMPLIMENT",
                        confidence=0.85,
                        matched_indices=sorted(set(matched)),
                        description="Negated negative = positive compliment ('reason I didn't')",
                        v_weight=35.0, d_weight=5.0, u_weight=0.0,
                        g_weight=-5.0, w_weight=10.0,
                    )

        if has_thanks and neg_indices:
            matched = neg_indices + other_indices
            return StructureMatch(
                pattern="NEGATED_NEGATIVE_COMPLIMENT",
                confidence=0.80,
                matched_indices=sorted(set(matched)),
                description="Negated negative = positive compliment (gratitude frame)",
                v_weight=35.0, d_weight=5.0, u_weight=0.0,
                g_weight=-5.0, w_weight=10.0,
            )

        if has_gratitude_verb:
            for ni in neg_indices:
                for oi in other_indices:
                    if abs(ni - oi) <= 6:
                        return StructureMatch(
                            pattern="NEGATED_NEGATIVE_COMPLIMENT",
                            confidence=0.75,
                            matched_indices=sorted({ni, oi}),
                            description="Negated negative near other-ref = possible compliment",
                            v_weight=30.0, d_weight=5.0, u_weight=0.0,
                            g_weight=-5.0, w_weight=10.0,
                        )

        return None

    def _recovery_small_win(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Mundane action + temporal rarity marker = recovery milestone.

        "I got out of bed today" = positive milestone for someone struggling
        "I ate a full meal" = recovery signal
        "I finally took a shower" = small win
        """
        words = [r.word for r in roles]
        text_lower = " ".join(w.lower() for w in words)

        matched_win = None
        for win in _MUNDANE_WINS:
            if win in text_lower:
                matched_win = win
                break

        if matched_win is None:
            return None

        words_lower = [w.lower() for w in words]
        has_temporal = any(w in _TEMPORAL_RARITY for w in words_lower)
        has_first_time = "for the first time" in text_lower
        has_self = any(r.role == "SELF_REF" for r in roles)

        if not has_temporal and not has_first_time and not has_self:
            return None

        conf = 0.70
        if has_temporal or has_first_time:
            conf = 0.85
        if has_self:
            conf = min(conf + 0.10, 0.95)

        win_words = matched_win.split()
        matched_indices = []
        for i in range(len(words_lower) - len(win_words) + 1):
            if all(words_lower[i + k] == win_words[k] for k in range(len(win_words))):
                matched_indices = list(range(i, i + len(win_words)))
                break

        return StructureMatch(
            pattern="RECOVERY_SMALL_WIN",
            confidence=conf,
            matched_indices=matched_indices,
            description=f"Recovery small win: '{matched_win}'",
            v_weight=25.0, d_weight=10.0, u_weight=0.0,
            g_weight=-5.0, w_weight=15.0,
        )

    # ── Ambiguity hold pattern ──────────────────────────────────

    def _ambiguity_hold(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Extreme V contradiction with no disambiguator = ambiguous intent.

        "death is awesome im going to jump" -- very positive + very negative
        words with no context to resolve. Could be suicidal or could be
        about a video game. Without disambiguators (lol, haha, game, bungee,
        bridge, help, please), pull V toward neutral and flag ambiguous.

        This is NOT sarcasm detection. This catches genuinely unresolvable
        V contradictions where the engine should refuse to commit.
        """
        from .vocabulary import VOCABULARY

        # Collect per-word V forces
        v_forces = []
        for r in roles:
            f = r.force or VOCABULARY.get(r.word)
            if f and abs(f[0]) > 5:
                v_forces.append(f[0])

        if len(v_forces) < 2:
            return None

        # Compute variance: need both strong positive AND strong negative
        max_pos = max((v for v in v_forces if v > 0), default=0)
        max_neg = min((v for v in v_forces if v < 0), default=0)
        variance = max_pos - max_neg  # e.g., 30 - (-35) = 65

        THRESHOLD = 50  # need significant contradiction
        if variance < THRESHOLD:
            return None

        # Check for disambiguators that resolve the contradiction
        _HUMOR_DISAMBIG = frozenset({
            "lol", "lmao", "haha", "hahaha", "rofl", "lmfao",
            "heh", "hehe", "jk", "kidding", "joking",
        })
        _CONTEXT_DISAMBIG = frozenset({
            "game", "games", "gaming", "bungee", "skydiving",
            "roller", "coaster", "movie", "film", "show",
            "song", "music", "book", "story", "video",
        })
        _CRISIS_DISAMBIG = frozenset({
            "help", "please", "cant", "dont", "stop",
            "anymore", "tired", "exhausted", "done",
        })

        words = [r.word for r in roles]
        has_humor = any(w in _HUMOR_DISAMBIG for w in words)
        has_context = any(w in _CONTEXT_DISAMBIG for w in words)
        has_crisis = any(w in _CRISIS_DISAMBIG for w in words)

        if has_humor or has_context or has_crisis:
            return None

        # Also don't fire if SLANG_DEATH_HUMOR or SELF_HARM_INTENT already detected
        # (those patterns already resolved the ambiguity)
        # This is checked implicitly: both those patterns use disambiguators.

        # Don't fire if there's a SELF_REF -- that's more likely personal/crisis
        has_self = any(r.role == "SELF_REF" for r in roles)

        conf = 0.85
        # Pull V toward W (neutral baseline)
        # The actual V correction happens in pendulum.py apply_structures
        return StructureMatch(
            pattern="AMBIGUITY_HOLD",
            confidence=conf,
            matched_indices=list(range(len(roles))),
            description=f"Extreme V contradiction (variance={variance}) with no disambiguator",
            v_weight=0.0,  # V correction handled specially in pendulum
            d_weight=0.0, u_weight=10.0, g_weight=5.0, w_weight=0.0,
        )

    # ── V8 COUNCIL detectors ─────────────────────────────────────

    def _boundary_violation(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Structural betrayal: molecular shape toxicity.

        OTHER_REF + intimate/violation action + SELF_POSSESSIVE + RELATION_REF
        = boundary violation. No individual atom carries the charge — it
        emerges from the configuration.
        """
        from .vocabulary import VOCABULARY as _V

        other_idx = [i for i, r in enumerate(roles) if r.role == "OTHER_REF"]
        if not other_idx:
            return None
        _INCLUSIVE = {"we", "us", "our", "ours", "ourselves", "were", "weve"}
        if all(roles[i].word in _INCLUSIVE for i in other_idx):
            return None

        self_idx = [i for i, r in enumerate(roles) if r.role == "SELF_REF"]
        relation_idx = [i for i, r in enumerate(roles) if r.role == "RELATION_REF"]
        has_self_possessive = any(roles[i].word in ("my", "mine", "our") for i in self_idx)

        words = [r.word for r in roles]

        # Path 1: Intimate boundary violation
        _INTIMATE_VERBS = frozenset({
            "sleeping", "slept", "kissing", "kissed", "seeing",
            "texting", "sexting", "hooking", "hooked",
        })
        has_intimate = any(w in _INTIMATE_VERBS for w in words)
        if has_intimate and has_self_possessive and relation_idx:
            rel_g = 15
            for ri in relation_idx:
                rf = roles[ri].force or _V.get(roles[ri].word)
                if rf:
                    rel_g = max(rel_g, abs(rf[4]))
            force = -(rel_g * 2.0 * 1.5)
            return StructureMatch(
                pattern="BOUNDARY_VIOLATION",
                confidence=0.85,
                matched_indices=sorted(set(other_idx + relation_idx)),
                description="Intimate boundary violation",
                v_weight=force, d_weight=-20.0, u_weight=15.0,
                g_weight=25.0, w_weight=force * 0.5,
            )

        # Path 2: Theft/financial violation
        _THEFT_VERBS = frozenset({
            "stole", "stolen", "stealing", "drained", "draining",
            "emptied", "took", "taking", "siphoned", "embezzled",
        })
        possession_idx = [i for i, r in enumerate(roles) if r.role == "POSSESSION"]
        has_theft = any(w in _THEFT_VERBS for w in words)
        if has_theft and (has_self_possessive or possession_idx):
            return StructureMatch(
                pattern="BOUNDARY_VIOLATION",
                confidence=0.75,
                matched_indices=sorted(set(other_idx + possession_idx)),
                description="Theft/financial boundary violation",
                v_weight=-30.0, d_weight=-20.0, u_weight=15.0,
                g_weight=20.0, w_weight=-15.0,
            )

        # Path 3: Deception violation
        _DECEPTION_MARKERS = frozenset({
            "double", "secret", "secretly", "hidden", "fake", "faked",
            "behind",
        })
        has_deception = any(w in _DECEPTION_MARKERS for w in words)
        if has_deception:
            return StructureMatch(
                pattern="BOUNDARY_VIOLATION",
                confidence=0.65,
                matched_indices=other_idx,
                description="Deception boundary violation",
                v_weight=-25.0, d_weight=-15.0, u_weight=10.0,
                g_weight=15.0, w_weight=-10.0,
            )

        return None

    def _mundane_hyperbole(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Crisis-class word bonded to mundane subject = hyperbole, not crisis.

        "this homework is killing me" — killing + homework (mundane) = hyperbole
        "i want to die this traffic is insane" — die + traffic (mundane) = complaint

        Computed from PROPERTIES: a word is mundane if it has low |V|, low |G|,
        low |A|, and is not a person role. The crisis word's charge is absorbed
        by the inert context.
        """
        from .vocabulary import VOCABULARY as _V

        # Suppress in grief/loss context: mundane objects are the SETTING
        # not the object of complaint. "the house feels empty without her"
        # is NOT mundane hyperbole — it's grief using the house as vehicle.
        _GRIEF_CONTEXT = frozenset({
            "without", "miss", "missed", "missing", "grief", "grieve",
            "gone", "lost", "loss", "died", "death", "passed",
            "funeral", "memorial", "bury", "mourn", "mourning",
            "remember", "remembered", "remembering", "reminds",
            "empty", "silence", "silent", "alone",
            "supposed", "would", "shouldve", "wouldve", "couldve",  # counterfactual grief
            "donated", "donate",  # disposing of belongings = grief
            "worst",  # "holidays are the worst now" = grief superlative
            "keeps", "kept",  # persistence of unwanted state
        })
        words = [r.word for r in roles]
        has_grief_context = any(w in _GRIEF_CONTEXT for w in words)
        # Also check for absent-person markers (possessive + no active subject)
        _ABSENT_PERSON = frozenset({"her", "him", "his", "their", "them"})
        has_absent_person = any(w in _ABSENT_PERSON for w in words)
        if has_grief_context:
            return None

        # Find crisis-class words (|dV| >= 25)
        crisis_idx = []
        for i, r in enumerate(roles):
            f = r.force or _V.get(r.word)
            if f and abs(f[0]) >= 25:
                crisis_idx.append(i)

        if not crisis_idx:
            return None

        # Find mundane NOUNS only (not verbs, not people)
        # The key: "homework" is a mundane noun (subject/topic of complaint)
        # "want" is a verb — NOT a mundane subject even if it has low forces
        _PERSON_ROLES = {"SELF_REF", "OTHER_REF", "RELATION_REF"}
        _SKIP_ROLES = {"CONNECTOR", "NEGATOR", "AMPLIFIER", "COMPRESSOR",
                       "HEDGE", "FILLER", "TEMPORAL", "REGISTER_CASUAL",
                       "TRANSFER", "ACQUIRE", "EMOTIONAL", "POWER",
                       "SUBMISSION", "PULL_TOWARD", "PULL_AWAY", "PULL_RESOLVED"}
        # Also skip common verbs that aren't nouns
        _NON_NOUN_WORDS = frozenset({
            # Verbs
            "want", "need", "make", "makes", "making", "get", "got",
            "take", "took", "give", "gave", "come", "go", "going",
            "said", "tell", "told", "think", "know", "feel",
            "is", "was", "are", "were", "been", "being",
            "have", "had", "has", "do", "did", "does",
            "can", "could", "will", "would", "should", "might",
            "try", "stop", "end", "start", "keep", "let",
            # Determiners / pronouns / function words
            "the", "a", "an", "this", "that", "these", "those",
            "it", "its", "there", "here", "some", "any",
            "better", "worse", "more", "less", "much", "very",
            "just", "even", "still", "already", "now", "then",
            "off", "on", "up", "down", "out", "in", "at", "by",
            "for", "to", "of", "with", "from", "about",
            "so", "too", "really", "pretty", "quite",
            "long", "short", "big", "small", "new", "old",
            "everything", "anything", "nothing", "something",  # pronouns, not mundane nouns
            "be", "being", "become", "became",  # copulas
            "out", "away", "back", "over", "off",  # particles
            "a", "an",  # articles
            "moment", "time", "day", "night", "year",  # temporal nouns
            "every", "each", "all",  # quantifiers
            "touch", "run", "walk", "sit", "stand", "look",  # common intransitive verbs
        })
        mundane_idx = []
        for i, r in enumerate(roles):
            if r.role in _PERSON_ROLES or r.role in _SKIP_ROLES:
                continue
            if r.word in _NON_NOUN_WORDS:
                continue
            f = r.force or _V.get(r.word)
            if f is None:
                # Not in vocabulary = mundane noun
                if r.role == "NEUTRAL":
                    mundane_idx.append(i)
            elif abs(f[0]) <= 12 and abs(f[4]) < 15 and abs(f[1]) <= 15:
                mundane_idx.append(i)

        if not mundane_idx:
            return None

        # Check: mundane noun is structurally the TOPIC (appears as subject
        # or as the object of a complaint). Must be near a crisis word.
        # BUT: if SELF_REF is the SUBJECT (appears BEFORE the crisis word),
        # this is self-directed, not mundane hyperbole.
        # "im the worst person" → SELF_REF before "worst" → suppress
        # "homework is killing me" → SELF_REF after "killing" → allow
        # Suppress when SELF_REF is subject AND mundane nouns describe self (not external cause)
        # "im the worst person" → "person" describes self → suppress
        # "im drowning in homework" → "homework" is external cause → allow
        # Person-descriptors near crisis word = self-directed, not mundane cause
        _PERSON_DESCRIPTORS = frozenset({
            "person", "human", "guy", "man", "woman", "girl", "boy",
            "one", "friend", "mother", "father", "parent", "kid",
            "mentor", "teacher", "boss", "partner", "colleague",
        })
        has_self_as_subject = any(roles[j].role == "SELF_REF" for j in range(min(2, len(roles))))
        if has_self_as_subject:
            # Check if ALL mundane nouns are person-descriptors
            all_person_desc = mundane_idx and all(
                roles[mi].word in _PERSON_DESCRIPTORS for mi in mundane_idx
            )
            if all_person_desc:
                return None

        for ci in crisis_idx:
            for mi in mundane_idx:
                dist = abs(ci - mi)
                if dist <= 6:
                    return StructureMatch(
                        pattern="MUNDANE_HYPERBOLE",
                        confidence=0.80,
                        matched_indices=sorted({ci, mi}),
                        description="Crisis word defused by mundane context",
                        v_weight=20.0,
                        d_weight=0.0, u_weight=0.0,
                        g_weight=-10.0, w_weight=10.0,
                    )

        return None

    def _self_erasure(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + desire verb + erasure verb = wanting to not exist."""
        _ERASURE_VERBS = frozenset({
            "disappear", "vanish", "fade", "dissolve",
            "invisible", "gone", "erase", "erased",
        })
        _DESIRE_VERBS = frozenset({"want", "wish", "hope", "need", "wanna"})

        has_strong_self = any(r.word in _STRONG_SELF_WORDS for r in roles)
        if not has_strong_self:
            return None
        has_desire = any(r.word in _DESIRE_VERBS for r in roles)
        has_erasure = any(r.word in _ERASURE_VERBS for r in roles)
        if has_desire and has_erasure:
            return StructureMatch(
                pattern="SELF_ERASURE",
                confidence=0.90,
                matched_indices=list(range(len(roles))),
                description="Desire to cease existing",
                v_weight=-50.0, d_weight=-25.0, u_weight=30.0,
                g_weight=35.0, w_weight=-40.0,
            )
        return None

    def _divestiture(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + giving away + possessions = end-of-life preparation.

        "i just gave my dog to my neighbor" — SELF + TRANSFER + RELATION/POSSESSION
        "i gave everything away" — SELF + TRANSFER + universal
        """
        has_self = any(r.role == "SELF_REF" for r in roles)
        has_transfer = any(r.role == "TRANSFER" for r in roles)
        has_universal = any(r.word in ("all", "everything", "every") for r in roles)
        has_relation_or_possession = any(
            r.role in ("RELATION_REF", "POSSESSION") for r in roles
        )
        # Check for recipient: "to/with" + person/relation
        has_recipient = any(
            r.word in ("to", "with") and i + 1 < len(roles)
            and any(roles[j].role in ("OTHER_REF", "RELATION_REF") for j in range(i + 1, min(len(roles), i + 3)))
            for i, r in enumerate(roles)
        )
        if has_self and has_transfer:
            if has_universal or (has_relation_or_possession and has_recipient):
                conf = 0.80 if has_universal else 0.65
                return StructureMatch(
                    pattern="DIVESTITURE",
                    confidence=conf,
                    matched_indices=list(range(len(roles))),
                    description="Divesting possessions",
                    v_weight=-35.0, d_weight=-20.0, u_weight=35.0,
                    g_weight=40.0, w_weight=-25.0,
                )
        return None

    def _method_fixation(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + persistence + crisis method noun = fixation on method."""
        _METHOD_NOUNS = frozenset({
            "pills", "pill", "gun", "pistol", "rope", "bridge",
            "knife", "blade", "razor", "noose", "ledge", "rail",
        })
        _PERSISTENCE = frozenset({"keep", "keeps", "kept", "still", "always"})

        has_self = any(r.word in _STRONG_SELF_WORDS for r in roles)
        has_method = any(r.word in _METHOD_NOUNS for r in roles)
        has_persistence = any(r.word in _PERSISTENCE for r in roles)
        if has_self and has_method and has_persistence:
            return StructureMatch(
                pattern="METHOD_FIXATION",
                confidence=0.90,
                matched_indices=list(range(len(roles))),
                description="Fixation on crisis method",
                v_weight=-55.0, d_weight=-30.0, u_weight=40.0,
                g_weight=45.0, w_weight=-35.0,
            )
        return None

    # ── V7 detectors ──────────────────────────────────────────────

    def _rarity_marker(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """RARITY word + action/event + TEMPORAL = milestone/achievement.

        "my kid took their first steps today"
        "for the first time in years"
        "finally graduated"

        Rarity markers ("first", "finally") near neutral actions create
        urgency charge and positive V boost. The rarity IS the emotion.
        """
        _RARITY_WORDS = frozenset({
            "first", "finally", "inaugural", "debut",
        })
        _MILESTONE_VERBS = frozenset({
            "got", "get", "took", "take", "made", "make",
            "passed", "graduated", "started", "began",
            "won", "earned", "achieved", "completed",
            "said", "walked", "stepped",
        })

        rarity_idx = [i for i, r in enumerate(roles) if r.word in _RARITY_WORDS]
        if not rarity_idx:
            return None

        words = [r.word for r in roles]
        has_milestone = any(w in _MILESTONE_VERBS for w in words)
        has_temporal = any(r.role == "TEMPORAL" for r in roles)
        has_self_or_relation = any(r.role in ("SELF_REF", "RELATION_REF") for r in roles)

        # Need at least rarity + one other signal
        signals = sum([has_milestone, has_temporal, has_self_or_relation])
        if signals == 0:
            return None

        # "first" alone is too common. Need context.
        conf = 0.4 + 0.15 * signals
        v_boost = 25.0 + 10.0 * signals

        return StructureMatch(
            pattern="RARITY_MARKER",
            confidence=min(conf, 0.95),
            matched_indices=rarity_idx,
            description="Rarity/milestone marker amplifies neutral action to positive",
            v_weight=v_boost, d_weight=10.0, u_weight=15.0,
            g_weight=10.0, w_weight=10.0,
        )

    def _abandonment(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """OTHER_REF + (TRANSFER or departure verb) + RELATION_REF/POSSESSION = abandonment.

        "she took the kids and left"
        "he packed his stuff and left"
        "she walked out on us"

        Abandonment is structural: SOMEONE + TOOK/LEFT + PEOPLE/THINGS.
        The subject is OTHER_REF (not self), the action is departure,
        and the objects are relationships or possessions.
        """
        _DEPARTURE_WORDS = frozenset({
            "left", "leave", "leaving", "gone",
            "packed", "disappeared", "vanished",
        })

        has_other = any(r.role == "OTHER_REF" for r in roles)
        if not has_other:
            return None

        departure_idx = [i for i, r in enumerate(roles)
                        if r.word in _DEPARTURE_WORDS or r.role == "TRANSFER"]
        if not departure_idx:
            return None

        # Need something taken or someone left behind
        has_relation = any(r.role == "RELATION_REF" for r in roles)
        has_possession = any(r.role == "POSSESSION" for r in roles)
        has_acquire = any(r.role == "ACQUIRE" for r in roles)  # "took"

        if not (has_relation or has_possession or has_acquire):
            return None

        # More objects = worse abandonment
        obj_count = sum([has_relation, has_possession])
        conf = 0.65 + 0.10 * obj_count

        return StructureMatch(
            pattern="ABANDONMENT",
            confidence=min(conf, 0.95),
            matched_indices=departure_idx,
            description="Someone departed with/from relationships or possessions",
            v_weight=-25.0, d_weight=-15.0, u_weight=20.0,
            g_weight=30.0, w_weight=-15.0,
        )

    def _life_achievement(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Life event achievement detection.

        Path 1: ACQUIRE/got + high-gravity noun = "got the job", "closed on the house"
        Path 2: Life event verbs + RELATION_REF/SELF_REF = "was born", "proposed", "adopted"

        High-gravity nouns near acquire verbs = life event achievement.
        """
        from .vocabulary import VOCABULARY

        # ── Path 2: Life event verb (no ACQUIRE needed) ──
        _LIFE_EVENT_VERBS = frozenset({
            "born", "proposed", "adopted", "graduated", "married",
            "engaged", "promoted", "survived", "won", "accepted",
            "passed", "cleared", "nailed",
            "madeitthrough", "madeit", "survived",
        })
        event_idx = [i for i, r in enumerate(roles) if r.word in _LIFE_EVENT_VERBS]
        if event_idx:
            has_self = any(r.role == "SELF_REF" for r in roles)
            has_relation = any(r.role == "RELATION_REF" for r in roles)
            has_temporal = any(r.role == "TEMPORAL" for r in roles)
            signals = sum([has_self, has_relation, has_temporal])
            if signals >= 1:
                conf = 0.55 + 0.10 * signals
                return StructureMatch(
                    pattern="LIFE_ACHIEVEMENT",
                    confidence=min(conf, 0.95),
                    matched_indices=event_idx,
                    description=f"Life event: {roles[event_idx[0]].word}",
                    v_weight=25.0, d_weight=10.0, u_weight=10.0,
                    g_weight=15.0, w_weight=12.0,
                )

        # ── Path 1: ACQUIRE + high-gravity noun ──
        # ACQUIRE verbs or compound tokens that function as acquire (closedon)
        acquire_idx = [i for i, r in enumerate(roles)
                      if r.role == "ACQUIRE" or r.word in ("closedon",)]
        if not acquire_idx:
            return None

        # Find high-gravity nouns or POSSESSION near acquire verbs
        _ACHIEVEMENT_CONTEXTS = frozenset({
            "job", "promotion", "raise", "offer", "accepted",
            "engaged", "married", "pregnant", "baby",
            "house", "apartment", "car",
            "scholarship", "degree", "diploma",
            "citizenship", "visa", "approved",
            "school", "program", "college", "university",
            "exam", "bar", "license", "certification",
            "application", "interview", "audition",
            "surgery", "treatment", "chemo", "rehab",
        })

        achievement_idx = []
        max_g = 0
        for i, r in enumerate(roles):
            if r.word in _ACHIEVEMENT_CONTEXTS:
                f = r.force or VOCABULARY.get(r.word)
                g = abs(f[4]) if f else 20
                if g > max_g:
                    max_g = g
                achievement_idx.append(i)
            elif r.role == "POSSESSION":
                f = r.force or VOCABULARY.get(r.word)
                g = abs(f[4]) if f else 0
                if g >= 10:  # Only count high-gravity possessions as achievements
                    achievement_idx.append(i)
                    max_g = max(max_g, g)

        if not achievement_idx:
            return None

        # Exclude discovery/investigation contexts: "found the messages" is NOT achievement
        _EVIDENCE_NOUNS = frozenset({
            "messages", "texts", "emails", "photos", "pictures",
            "receipts", "letters", "notes", "evidence", "proof",
        })
        words = [r.word for r in roles]
        if any(w in _EVIDENCE_NOUNS for w in words):
            return None

        # "found" without excitement/milestone context = discovery, not achievement
        _DISCOVERY_VERBS = frozenset({"found", "discovered", "noticed", "saw", "spotted"})
        has_discovery = any(roles[i].word in _DISCOVERY_VERBS for i in acquire_idx)
        has_excitement = any(r.role in ("AMPLIFIER", "TEMPORAL") for r in roles)
        if has_discovery and not has_excitement:
            return None

        # Check proximity: acquire near achievement
        for ai in acquire_idx:
            for ci in achievement_idx:
                if abs(ai - ci) <= 5:
                    has_self = any(r.role == "SELF_REF" for r in roles)
                    has_temporal = any(r.role == "TEMPORAL" for r in roles)
                    conf = 0.55 + 0.10 * has_self + 0.10 * has_temporal
                    v_boost = 20.0 + max_g * 0.3
                    return StructureMatch(
                        pattern="LIFE_ACHIEVEMENT",
                        confidence=min(conf, 0.95),
                        matched_indices=sorted(set([ai, ci])),
                        description="Acquired high-gravity life event",
                        v_weight=v_boost, d_weight=15.0, u_weight=10.0,
                        g_weight=15.0, w_weight=15.0,
                    )
        return None

    def _self_insignificance(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + NEGATOR + {matter, count, belong, deserve, enough, worthy}
        = "I don't matter" / "I'm not enough" / "I don't belong"
        Self-worth erosion through negation of significance.
        """
        _SIGNIFICANCE_WORDS = frozenset({
            "matter", "count", "belong", "deserve", "enough",
            "worthy", "important", "wanted", "needed", "loved",
            "care", "cared",
        })
        self_idx = [i for i, r in enumerate(roles) if r.role == "SELF_REF"]
        neg_idx = [i for i, r in enumerate(roles) if r.role == "NEGATOR"]
        sig_idx = [i for i, r in enumerate(roles) if r.word in _SIGNIFICANCE_WORDS]

        if not self_idx or not neg_idx or not sig_idx:
            return None

        # SELF_REF near NEGATOR near significance word
        for si in self_idx:
            for ni in neg_idx:
                if abs(si - ni) <= 3:
                    for wi in sig_idx:
                        if abs(ni - wi) <= 3:
                            return StructureMatch(
                                pattern="SELF_INSIGNIFICANCE",
                                confidence=0.80,
                                matched_indices=sorted({si, ni, wi}),
                                description="Self negated significance",
                                v_weight=-25.0, d_weight=-15.0,
                                u_weight=0.0, g_weight=-10.0,
                                w_weight=-30.0,
                            )
        return None

    def _self_replacement(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + wish/want + 'someone else'/'different person'/'anyone but'
        = wanting to not be oneself. Deep self-worth damage.
        """
        _DESIRE_WORDS = frozenset({"wish", "wished", "want", "wanted", "rather"})
        _REPLACEMENT_WORDS = frozenset({
            "else", "different", "another", "other",
        })
        words = [r.word for r in roles]
        has_self = any(r.role == "SELF_REF" for r in roles)
        has_desire = any(w in _DESIRE_WORDS for w in words)
        has_replacement = any(w in _REPLACEMENT_WORDS for w in words)

        if has_self and has_desire and has_replacement:
            idx = [i for i, r in enumerate(roles)
                   if r.role == "SELF_REF" or r.word in _DESIRE_WORDS or r.word in _REPLACEMENT_WORDS]
            return StructureMatch(
                pattern="SELF_REPLACEMENT",
                confidence=0.75,
                matched_indices=idx,
                description="Desire to not be self",
                v_weight=-20.0, d_weight=-15.0,
                u_weight=5.0, g_weight=-10.0,
                w_weight=-35.0,
            )
        return None

    def _persistent_absence(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Persistence markers + absent person reference = ongoing grief.

        "i still set two places at the table" — persistence (still) + domestic
        "every song reminds me of him" — universality (every) + absent person (him)
        "i keep her voicemail just to hear her voice" — persistence (keep) + ghost possession (her)
        "i cant go back to that restaurant" — avoidance + place
        "the holidays will never be the same" — permanence (never) + change (same)

        The grief is in the STRUCTURE: a routine persists, but the person is gone.
        Or: a universal trigger connects everything to the absent person.
        """
        words = [r.word for r in roles]

        _PERSISTENCE = frozenset({
            "still", "keep", "keeps", "kept", "always", "every",
            "never", "anymore", "forever",
            "years", "months", "weeks",  # temporal persistence markers
        })
        _ABSENT_MARKERS = frozenset({
            "her", "him", "his", "their", "them",
            "without", "gone", "empty", "alone",
        })
        _AVOIDANCE = frozenset({
            "cant", "can't", "wont", "won't", "couldnt", "couldn't",
            "back", "anymore",
        })
        _GRIEF_VOCAB = frozenset({
            "grief", "grieve", "miss", "missed", "missing",
            "loss", "lost", "mourn", "mourning",
            "reminds", "reminded", "remember", "remembered",
            "donated", "donate", "packed", "cleaned",  # disposing of belongings
        })

        has_persistence = any(w in _PERSISTENCE for w in words)
        has_absent = any(w in _ABSENT_MARKERS for w in words)
        has_avoidance = sum(1 for w in words if w in _AVOIDANCE)
        has_grief_word = any(w in _GRIEF_VOCAB for w in words)

        # Count persistence markers
        persistence_count = sum(1 for w in words if w in _PERSISTENCE)

        # Pattern 1: persistence + absent person, OR heavy persistence alone
        if (has_persistence and has_absent) or persistence_count >= 2:
            idx = [i for i, r in enumerate(roles) if r.word in _PERSISTENCE or r.word in _ABSENT_MARKERS]
            return StructureMatch(
                pattern="PERSISTENT_ABSENCE",
                confidence=0.70,
                matched_indices=idx,
                description="Persistent routine for absent person",
                v_weight=-25.0, d_weight=-10.0,
                u_weight=0.0, g_weight=-15.0,
                w_weight=-10.0,
            )

        # Pattern 2: avoidance of place/thing (grief avoidance)
        # "i cant go back to that restaurant" — avoidance even without explicit absent person
        if has_avoidance >= 2 or (has_avoidance >= 1 and has_absent):
            idx = [i for i, r in enumerate(roles) if r.word in _AVOIDANCE or r.word in _ABSENT_MARKERS]
            return StructureMatch(
                pattern="PERSISTENT_ABSENCE",
                confidence=0.65,
                matched_indices=idx,
                description="Grief avoidance",
                v_weight=-20.0, d_weight=-10.0,
                u_weight=0.0, g_weight=-10.0,
                w_weight=-5.0,
            )

        # Pattern 3: explicit grief vocabulary
        if has_grief_word:
            idx = [i for i, r in enumerate(roles) if r.word in _GRIEF_VOCAB]
            return StructureMatch(
                pattern="PERSISTENT_ABSENCE",
                confidence=0.60,
                matched_indices=idx,
                description="Explicit grief reference",
                v_weight=-20.0, d_weight=-10.0,
                u_weight=0.0, g_weight=-15.0,
                w_weight=-5.0,
            )

        # Pattern 4: counterfactual + togetherness ("supposed to grow old together")
        _COUNTERFACTUAL = frozenset({"supposed", "would", "shouldve", "wouldve", "couldve"})
        _TOGETHERNESS = frozenset({"together", "forever", "always"})
        has_counterfactual = any(w in _COUNTERFACTUAL for w in words)
        has_togetherness = any(w in _TOGETHERNESS for w in words)
        if has_counterfactual and has_togetherness:
            idx = [i for i, r in enumerate(roles) if r.word in _COUNTERFACTUAL or r.word in _TOGETHERNESS]
            return StructureMatch(
                pattern="PERSISTENT_ABSENCE",
                confidence=0.70,
                matched_indices=idx,
                description="Counterfactual togetherness grief",
                v_weight=-25.0, d_weight=-10.0,
                u_weight=0.0, g_weight=-15.0,
                w_weight=-10.0,
            )

        # Pattern 5: "never be the same" / permanence of change
        if "never" in words and "same" in words:
            idx = [i for i, r in enumerate(roles) if r.word in ("never", "same")]
            return StructureMatch(
                pattern="PERSISTENT_ABSENCE",
                confidence=0.60,
                matched_indices=idx,
                description="Permanent change grief",
                v_weight=-20.0, d_weight=-10.0,
                u_weight=0.0, g_weight=-10.0,
                w_weight=-5.0,
            )

        return None

    def _directed_dismissal(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """NEGATOR/rejection + OTHER_REF or communication verb = pushing someone away.

        "dont ever talk to me again" — NEGATOR + finality (ever/again) + communication
        "nobody asked for your opinion" — nobody + OTHER_REF possession
        "you dont deserve her" — OTHER_REF + NEGATOR + entitlement
        "youre just like your father" — comparison as insult (OTHER_REF + "like" + RELATION_REF)
        """
        words = [r.word for r in roles]
        has_other = any(r.role == "OTHER_REF" for r in roles)
        has_negator = any(r.role == "NEGATOR" for r in roles)

        _DISMISSAL_VERBS = frozenset({
            "talk", "speak", "contact", "call", "text", "message",
            "come", "show", "bother", "asked", "ask",
        })
        _FINALITY_ADVERBS = frozenset({
            "ever", "again", "anymore", "never",
        })
        _COMPARISON_WORDS = frozenset({"like", "same", "just"})

        has_dismissal_verb = any(w in _DISMISSAL_VERBS for w in words)
        has_finality_adv = any(w in _FINALITY_ADVERBS for w in words)
        has_relation = any(r.role == "RELATION_REF" for r in roles)
        has_comparison = any(w in _COMPARISON_WORDS for w in words)
        has_nobody = "nobody" in words or "noone" in words

        # Pattern 1: "dont [ever] talk/come/bother [again]" — dismissal
        if has_negator and has_dismissal_verb and (has_finality_adv or has_other):
            idx = [i for i, r in enumerate(roles) if r.role == "NEGATOR" or r.word in _DISMISSAL_VERBS]
            return StructureMatch(
                pattern="DIRECTED_DISMISSAL",
                confidence=0.75,
                matched_indices=idx,
                description="Pushing someone away",
                v_weight=-25.0, d_weight=20.0,
                u_weight=5.0, g_weight=0.0,
                w_weight=0.0,
            )

        # Pattern 2: "nobody asked/cares" — invalidation
        if has_nobody and has_dismissal_verb:
            idx = [i for i, r in enumerate(roles) if r.word == "nobody" or r.word in _DISMISSAL_VERBS]
            return StructureMatch(
                pattern="DIRECTED_DISMISSAL",
                confidence=0.70,
                matched_indices=idx,
                description="Invalidation",
                v_weight=-20.0, d_weight=15.0,
                u_weight=0.0, g_weight=0.0,
                w_weight=-10.0,
            )

        # Pattern 3: "youre just like your father" — comparison as insult
        # OTHER_REF + comparison + RELATION_REF
        if has_other and has_comparison and has_relation:
            idx = [i for i, r in enumerate(roles) if r.role in ("OTHER_REF", "RELATION_REF") or r.word in _COMPARISON_WORDS]
            return StructureMatch(
                pattern="DIRECTED_DISMISSAL",
                confidence=0.65,
                matched_indices=idx,
                description="Comparison as insult",
                v_weight=-20.0, d_weight=15.0,
                u_weight=0.0, g_weight=0.0,
                w_weight=-15.0,
            )

        return None

    def _martyrdom_field(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Surface compliance + self-exclusion/resignation = passive aggression.

        The molecular shape: positive/compliant surface words bonded to
        self-exclusion or resignation markers. The positive atoms are fake —
        their charge is inverted by the structural context.

        "no go ahead have fun without me" — permission + exclusion
        "i said its fine" — compliance + minimization
        "must be nice to not have to worry" — false compliment + exclusion
        "forget it it doesnt matter" — dismissal + self-negation
        "sure whatever makes you happy" — compliance + resignation
        """
        words = [r.word for r in roles]

        _COMPLIANCE_SURFACE = frozenset({
            "fine", "ok", "okay", "sure", "alright", "whatever",
            "go", "ahead", "fun", "happy", "nice", "great",
            "good", "glad", "cool", "enjoy",
            "worries", "mind", "no",  # "no worries", "dont mind me"
        })
        _SELF_EXCLUSION = frozenset({
            "without", "me", "myself", "mine",
        })
        _RESIGNATION = frozenset({
            "whatever", "forget", "doesnt", "dont", "didnt",
            "matter", "care", "anyway", "anymore", "said",
            "guess", "suppose", "apparently",
            "just", "figure", "myself",  # "ill just figure it out myself"
        })
        _FALSE_PRAISE = frozenset({
            "must", "nice", "lucky", "easy",
        })

        compliance_count = sum(1 for w in words if w in _COMPLIANCE_SURFACE)
        exclusion_count = sum(1 for w in words if w in _SELF_EXCLUSION)
        resignation_count = sum(1 for w in words if w in _RESIGNATION)
        has_self = any(r.role == "SELF_REF" for r in roles)
        has_false_praise = sum(1 for w in words if w in _FALSE_PRAISE)

        # Pattern 1: compliance + self-exclusion ("go ahead without me")
        if compliance_count >= 1 and exclusion_count >= 1 and has_self:
            idx = [i for i, r in enumerate(roles)
                   if r.word in _COMPLIANCE_SURFACE or r.word in _SELF_EXCLUSION]
            return StructureMatch(
                pattern="MARTYRDOM_FIELD",
                confidence=0.75,
                matched_indices=idx,
                description="Surface compliance + self-exclusion",
                v_weight=-30.0, d_weight=-10.0,
                u_weight=5.0, g_weight=0.0,
                w_weight=-15.0,
            )

        # Pattern 2: compliance + resignation ("fine", "forget it", "doesnt matter")
        if (compliance_count >= 1 and resignation_count >= 2) or (compliance_count >= 2 and resignation_count >= 1):
            idx = [i for i, r in enumerate(roles)
                   if r.word in _COMPLIANCE_SURFACE or r.word in _RESIGNATION]
            return StructureMatch(
                pattern="MARTYRDOM_FIELD",
                confidence=0.70,
                matched_indices=idx,
                description="Surface compliance + resignation",
                v_weight=-25.0, d_weight=-10.0,
                u_weight=0.0, g_weight=0.0,
                w_weight=-10.0,
            )

        # Pattern 3: false praise ("must be nice", "must be easy")
        if has_false_praise >= 2:
            idx = [i for i, r in enumerate(roles) if r.word in _FALSE_PRAISE]
            return StructureMatch(
                pattern="MARTYRDOM_FIELD",
                confidence=0.70,
                matched_indices=idx,
                description="False praise / envy disguised as compliment",
                v_weight=-25.0, d_weight=-15.0,
                u_weight=5.0, g_weight=0.0,
                w_weight=-10.0,
            )

        # Pattern 4: heavy resignation without surface ("forget it doesnt matter")
        if resignation_count >= 3:
            idx = [i for i, r in enumerate(roles) if r.word in _RESIGNATION]
            return StructureMatch(
                pattern="MARTYRDOM_FIELD",
                confidence=0.65,
                matched_indices=idx,
                description="Heavy resignation",
                v_weight=-25.0, d_weight=-10.0,
                u_weight=0.0, g_weight=0.0,
                w_weight=-10.0,
            )

        # Pattern 5: "i said its fine" — self + said + compliance (reported compliance)
        has_said = "said" in words or "told" in words
        if has_self and has_said and compliance_count >= 1:
            idx = [i for i, r in enumerate(roles)
                   if r.role == "SELF_REF" or r.word in ("said", "told") or r.word in _COMPLIANCE_SURFACE]
            return StructureMatch(
                pattern="MARTYRDOM_FIELD",
                confidence=0.65,
                matched_indices=idx,
                description="Reported compliance (minimization)",
                v_weight=-20.0, d_weight=-10.0,
                u_weight=0.0, g_weight=0.0,
                w_weight=-10.0,
            )

        return None

    # ── Council Round 6 patterns ──────────────────────────────────

    def _dangling_bond(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Incomplete valence shell — verb expects object/destination but finds vacuum.

        "im going to jump" → SELF_REF + intent + KINETIC_VERB + [NO OBJECT] = crisis void
        "im going to jump rope" → object present = stable molecule
        "i have a plan" → SELF_REF + possess + KINETIC_NOUN + [NO OBJECT] = crisis void
        "i have a plan for dinner" → object present = stable

        The danger is in the ABSENCE of expected bonds, not in the atoms themselves.
        Council consensus: all 4 brothers agreed on this approach.
        """
        words = [r.word for r in roles]
        text = " ".join(words)
        n = len(roles)
        has_self = any(r.role == "SELF_REF" for r in roles)

        # ── Universal checks (don't require SELF_REF) ──

        # Entrapment: "no way out" / "no escape" / "no hope"
        _ENTRAPMENT = frozenset({"noway", "no", "never", "cant", "cannot"})
        _EXIT_WORDS = frozenset({"out", "escape", "option", "choice", "hope", "solution", "exit"})
        has_entrapment = any(w in _ENTRAPMENT for w in words)
        has_exit = any(w in _EXIT_WORDS for w in words)
        if has_entrapment and has_exit:
            idx = [i for i, r in enumerate(roles) if r.word in _ENTRAPMENT or r.word in _EXIT_WORDS]
            return StructureMatch(
                pattern="DANGLING_BOND",
                confidence=0.65,
                matched_indices=idx,
                description="Entrapment — no exit",
                v_weight=-30.0, d_weight=-20.0,
                u_weight=15.0, g_weight=-15.0,
                w_weight=-15.0,
            )

        # Nihilism: "whats the point" / "why bother" — but NOT with concrete objects
        # "whats the point of this meeting" = genuine question, NOT nihilism
        _NIHILISM = {"whats the point", "why bother", "none of it matters",
                     "nothing matters", "who cares", "doesnt matter anymore"}
        _CONCRETE_AFTER = frozenset({
            "of", "in", "about", "with", "for",  # preposition = has an object
        })
        nihilism_match = None
        for p in _NIHILISM:
            idx_p = text.find(p)
            if idx_p >= 0:
                after = text[idx_p + len(p):].strip().split()
                if after and after[0] in _CONCRETE_AFTER:
                    continue  # has concrete object — genuine question
                nihilism_match = p
                break
        if nihilism_match:
            return StructureMatch(
                pattern="DANGLING_BOND",
                confidence=0.60,
                matched_indices=[0],
                description="Nihilism — pointlessness",
                v_weight=-25.0, d_weight=-15.0,
                u_weight=5.0, g_weight=-10.0,
                w_weight=-20.0,
            )

        # Temporal finality: "tonight is the night" (no emotional atoms but finality frame)
        _TEMPORAL_FINALITY = frozenset({"tonight", "today", "now", "finally"})
        _FINALITY_NOUNS = frozenset({"night", "time", "day", "moment", "it"})
        has_temporal_fin = any(w in _TEMPORAL_FINALITY for w in words)
        has_finality_noun = any(w in _FINALITY_NOUNS for w in words)
        # Only fire if sentence is very short AND has no positive/action content
        _POSITIVE_ACTIONS = frozenset({
            "celebrate", "party", "dinner", "game", "concert", "show",
            "meet", "start", "begin", "open", "launch", "fun",
        })
        has_positive_action = any(w in _POSITIVE_ACTIONS for w in words)
        # Require SELF_REF for temporal finality (standalone "tonight is the night" too ambiguous)
        if has_self and has_temporal_fin and has_finality_noun and len(words) <= 5 and not has_positive_action:
            return StructureMatch(
                pattern="DANGLING_BOND",
                confidence=0.55,
                matched_indices=[0],
                description="Temporal finality — ominous brevity",
                v_weight=-20.0, d_weight=-10.0,
                u_weight=10.0, g_weight=-10.0,
                w_weight=-10.0,
            )

        # ── Self-directed checks (require SELF_REF) ──
        if not has_self:
            return None

        # Kinetic verbs that need objects to be stable
        _KINETIC_VERBS = frozenset({
            "jump", "leave", "go", "end", "stop", "cut", "hang",
            "shoot", "drown", "overdose", "slit",
        })
        # Intent frames that precede kinetic verbs
        _INTENT_FRAMES = frozenset({
            "going", "gonna", "want", "will", "ready", "about",
            "decided", "planning", "thinking",
        })
        # Objects/destinations that stabilize the bond
        _STABILIZERS = frozenset({
            "rope", "joy", "ahead", "home", "work", "school", "store",
            "party", "gym", "bed", "lunch", "dinner", "shopping",
            "swimming", "running", "hiking", "dancing", "class",
            "game", "movie", "practice", "meeting", "appointment",
            "trampoline", "chance", "conclusion", "ball",
        })
        # Nouns that indicate a plan WITH an object
        _PLAN_OBJECTS = frozenset({
            "for", "about", "to", "regarding",
        })

        # Check for kinetic verb after intent frame
        for i, r in enumerate(roles):
            if r.word in _KINETIC_VERBS:
                # Was there an intent frame before this verb?
                has_intent = any(roles[j].word in _INTENT_FRAMES
                                for j in range(max(0, i - 4), i))
                if not has_intent and r.word not in ("end", "stop"):
                    continue

                # Check right side for stabilizing object
                right_words = [roles[j].word for j in range(i + 1, min(n, i + 5))]
                has_object = any(w in _STABILIZERS for w in right_words)
                # Also check: any concrete noun (POSSESSION, RELATION_REF, high-gravity EMOTIONAL)
                has_concrete = any(
                    roles[j].role in ("POSSESSION", "RELATION_REF")
                    or (roles[j].role == "EMOTIONAL" and roles[j].force and abs(roles[j].force[4]) > 15)
                    for j in range(i + 1, min(n, i + 5))
                )

                # Check for positive context BEFORE the verb (dark humor/excitement)
                from .vocabulary import VOCABULARY as _V_DB
                has_positive_before = any(
                    (roles[j].force or _V_DB.get(roles[j].word) or (0,))[0] > 15
                    for j in range(max(0, i - 6), i)
                )
                if not has_object and not has_concrete and not has_positive_before:
                    # DANGLING BOND — vacuum after kinetic verb
                    return StructureMatch(
                        pattern="DANGLING_BOND",
                        confidence=0.75,
                        matched_indices=[i],
                        description=f"Incomplete bond: '{r.word}' with no object",
                        v_weight=-35.0, d_weight=-10.0,
                        u_weight=15.0, g_weight=-15.0,
                        w_weight=-20.0,
                    )

        # Check for "i have a plan" pattern (SELF + possess + abstract noun + no object)
        if "plan" in words or "plans" in words:
            plan_idx = words.index("plan") if "plan" in words else words.index("plans")
            right_words = words[plan_idx + 1:]
            has_plan_object = any(w in _PLAN_OBJECTS for w in right_words[:3])
            if not has_plan_object and not any(w in _STABILIZERS for w in right_words[:3]):
                return StructureMatch(
                    pattern="DANGLING_BOND",
                    confidence=0.60,
                    matched_indices=[plan_idx],
                    description="Plan with no object — potential crisis",
                    v_weight=-25.0, d_weight=-10.0,
                    u_weight=10.0, g_weight=-10.0,
                    w_weight=-15.0,
                )

        return None

    def _masking(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Performance verb + compliance state = hiding real emotional state.

        "im tired of pretending im okay" → SELF_REF + FACADE_VERB + EQUILIBRIUM
        "i keep acting like im fine" → same shape
        """
        words = [r.word for r in roles]
        _FACADE_VERBS = frozenset({
            "pretending", "acting", "faking", "hiding", "lying",
            "playing", "performing", "masking",
        })
        _EQUILIBRIUM = frozenset({
            "okay", "ok", "fine", "alright", "happy", "normal",
            "good", "strong", "together", "better",
        })
        has_self = any(r.role == "SELF_REF" for r in roles)
        has_facade = any(w in _FACADE_VERBS for w in words)
        has_equilibrium = any(w in _EQUILIBRIUM for w in words)

        if has_self and has_facade and has_equilibrium:
            idx = [i for i, r in enumerate(roles)
                   if r.word in _FACADE_VERBS or r.word in _EQUILIBRIUM]
            return StructureMatch(
                pattern="MASKING",
                confidence=0.80,
                matched_indices=idx,
                description="Performed compliance hiding real state",
                v_weight=-30.0, d_weight=-15.0,
                u_weight=10.0, g_weight=-10.0,
                w_weight=-20.0,
            )
        return None

    def _resignation(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Desire verb + vague pronoun + finality = wanting it to end.

        "i just want it to be over" → DESIRE + PRONOUN_VOID + TERMINATION
        "i wish this would end" → same
        "i need it to stop" → same
        """
        words = [r.word for r in roles]
        _DESIRE = frozenset({"want", "wish", "need", "hope", "bother"})
        _PRONOUN_VOID = frozenset({"it", "this", "everything", "things", "all"})
        _TERMINATION = frozenset({
            "over", "done", "end", "stop", "finish", "gone",
            "ending", "finished", "stopped",
        })
        has_self = any(r.role == "SELF_REF" for r in roles)
        has_desire = any(w in _DESIRE for w in words)
        has_void = any(w in _PRONOUN_VOID for w in words)
        has_termination = any(w in _TERMINATION for w in words)

        # Safety: "over with" = completion, not crisis. Skip if followed by manner/speed
        _COMPLETION_CONTEXT = frozenset({"with", "quickly", "soon", "already", "fast"})
        over_idx = next((i for i, w in enumerate(words) if w == "over"), -1)
        if over_idx >= 0 and over_idx + 1 < len(words) and words[over_idx + 1] in _COMPLETION_CONTEXT:
            has_termination = False

        if has_self and has_desire and has_void and has_termination:
            idx = [i for i, r in enumerate(roles)
                   if r.word in _DESIRE or r.word in _TERMINATION]
            return StructureMatch(
                pattern="RESIGNATION",
                confidence=0.75,
                matched_indices=idx,
                description="Desire for termination",
                v_weight=-35.0, d_weight=-15.0,
                u_weight=10.0, g_weight=-15.0,
                w_weight=-20.0,
            )
        return None

    def _world_continues(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """World/everyone + continues + without self = "they'd be fine without me."

        "the world would keep spinning without me"
        "everyone would move on"
        "life goes on without me"
        """
        words = [r.word for r in roles]
        text = " ".join(words)
        _WORLD = frozenset({"world", "everyone", "life", "people", "things", "they"})
        _CONTINUATION = frozenset({
            "keep", "keeps", "go", "goes", "continue", "continues",
            "move", "carry", "spinning", "moving",
        })
        _SELF_ABSENCE = {"without me", "if i were gone", "if i wasnt here",
                         "when im gone", "after im gone", "if im not here"}

        has_world = any(w in _WORLD for w in words)
        has_continue = any(w in _CONTINUATION for w in words)
        has_absence = any(p in text for p in _SELF_ABSENCE)

        if has_world and has_continue and has_absence:
            return StructureMatch(
                pattern="WORLD_CONTINUES",
                confidence=0.75,
                matched_indices=[0],
                description="World continues without self",
                v_weight=-30.0, d_weight=-20.0,
                u_weight=10.0, g_weight=-15.0,
                w_weight=-25.0,
            )
        return None

    def _farewell(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Imperative + relation + positive emotion = terminal message.

        "tell my kids i love them" — not happiness, it's a last message.
        "let mom know i cared" — same structure.
        "make sure they know i tried" — same.
        """
        words = [r.word for r in roles]
        _IMPERATIVE = frozenset({
            "tell", "let", "remind", "give", "make", "say",
        })
        _POSITIVE_EMOTION = frozenset({
            "love", "loved", "care", "cared", "proud", "sorry",
            "forgive", "grateful", "thankful", "miss", "tried",
        })

        has_imperative = any(w in _IMPERATIVE for w in words[:3])
        has_relation = any(r.role == "RELATION_REF" for r in roles)
        has_emotion = any(w in _POSITIVE_EMOTION for w in words)
        has_self = any(r.role == "SELF_REF" for r in roles)

        if has_imperative and has_relation and has_emotion:
            # Safety: "tell my kids i love pizza" — emotion must not bond to mundane object
            # Check that the positive emotion word is near a RELATION_REF, not a mundane noun
            _MUNDANE_AFTER = frozenset({
                "pizza", "ice", "cream", "cake", "movie", "game",
                "dog", "cat", "car", "house", "school", "food",
            })
            has_mundane_object = any(w in _MUNDANE_AFTER for w in words)
            if has_mundane_object:
                return None  # "tell my kids i love pizza" = not farewell
            return StructureMatch(
                pattern="FAREWELL",
                confidence=0.80,
                matched_indices=[0],
                description="Terminal message to loved ones",
                v_weight=-60.0, d_weight=-10.0,
                u_weight=20.0, g_weight=-20.0,
                w_weight=-15.0,
            )

        # Secondary: "i wrote a note" — communication artifact with NO recipient
        # "i wrote a note to my teacher" has a recipient → mundane
        _COMM_ARTIFACTS = frozenset({"note", "letter", "message", "goodbye"})
        _WRITING_VERBS = frozenset({"wrote", "left", "finished", "sealed", "written"})
        has_writing = any(w in _WRITING_VERBS for w in words)
        has_artifact = any(w in _COMM_ARTIFACTS for w in words)
        has_recipient = "to" in words  # "wrote a note TO someone" = mundane
        if has_self and has_writing and has_artifact and not has_recipient:
            return StructureMatch(
                pattern="FAREWELL",
                confidence=0.65,
                matched_indices=[0],
                description="Writing farewell artifact",
                v_weight=-30.0, d_weight=-10.0,
                u_weight=15.0, g_weight=-15.0,
                w_weight=-10.0,
            )

        return None
