"""Multi-path emotional processing — literal vs sarcasm with consistency check.

Instead of committing to ONE interpretation, run the sentence through multiple
paths and pick the one where the V result is most consistent with the words
in the sentence.

Path 1 (literal): Standard pendulum processing
Path 2 (sarcasm): If positive + negative force words coexist, try reading
                   positive words as sarcastic (flip their valence contribution)

The consistency score is the key innovation: if the literal path's V reading
agrees with the dominant word direction, use it. If not, try the sarcasm flip.
"""

import re

from .pendulum import SequentialPendulum
from .forces import WORD_FORCES
from .ring_forces import get_ring_force


class MultiPathProcessor:
    """Run multiple interpretation paths, pick most internally consistent."""

    def process(self, text):
        """Process text through literal and sarcasm paths.

        Returns: (best_v, best_a, best_d, best_u, best_g, path_name, consistency_score, history)
        """
        words = re.findall(r"[a-z']+", text.lower())
        if not words:
            return 128, 128, 128, 0, 128, "empty", 0, []

        # PATH 1: Literal (standard processing)
        p1 = SequentialPendulum()
        for i, w in enumerate(words):
            p1.process_word(w, words, i)
        v1 = int(p1.v)
        consistency1 = self._compute_consistency(v1, words)

        # PATH 2: Sarcasm-aware
        # If there are BOTH positive and negative force words,
        # try reading the positive ones as sarcastic (flip their valence)
        pos_words = [w for w in words if w in WORD_FORCES and WORD_FORCES[w][0] > 15]
        neg_words = [w for w in words if w in WORD_FORCES and WORD_FORCES[w][0] < -15]

        if pos_words and neg_words:
            # Tension exists — try sarcasm path
            p2 = SequentialPendulum()
            for i, w in enumerate(words):
                # For positive words in a mixed sentence, check sarcasm ring
                ring = get_ring_force(w, "sarcastic")
                if ring and w in pos_words:
                    # Use sarcasm force instead
                    p2.process_word(w, words, i)  # normal processing but we'll adjust after
                else:
                    p2.process_word(w, words, i)

            # Sarcasm heuristic: if positive words appear BEFORE negative words,
            # it's likely sarcastic ("Great, another problem")
            first_pos = min((words.index(w) for w in pos_words), default=999)
            first_neg = min((words.index(w) for w in neg_words), default=999)

            if first_pos < first_neg:
                # Positive before negative = likely sarcasm
                # Pull V toward negative
                v2 = int(128 - (v1 - 128) * 0.5) if v1 > 128 else v1
            else:
                v2 = int(p2.v)

            consistency2 = self._compute_consistency(v2, words)
        else:
            v2 = v1
            consistency2 = consistency1

        # Pick the path with higher consistency
        if consistency2 > consistency1 and abs(v2 - v1) > 10:
            return v2, int(p1.a), int(p1.d), int(p1.u), int(p1.g), "sarcasm", consistency2, p1.history
        else:
            return v1, int(p1.a), int(p1.d), int(p1.u), int(p1.g), "literal", consistency1, p1.history

    def _compute_consistency(self, v, words):
        """How consistent is the V reading with the words in the sentence?

        If V is positive but sentence has more negative-force words -> LOW consistency
        If V matches the dominant word direction -> HIGH consistency
        """
        pos_force = sum(WORD_FORCES[w][0] for w in words if w in WORD_FORCES and WORD_FORCES[w][0] > 5)
        neg_force = sum(abs(WORD_FORCES[w][0]) for w in words if w in WORD_FORCES and WORD_FORCES[w][0] < -5)

        total = pos_force + neg_force
        if total == 0:
            return 0.5  # no strong words, medium consistency

        # Dominant direction
        pos_ratio = pos_force / total
        neg_ratio = neg_force / total

        # V direction
        v_positive = v > 128

        # Consistency: does V agree with dominant word direction?
        if v_positive and pos_ratio > neg_ratio:
            return pos_ratio  # V positive, words mostly positive -> consistent
        elif not v_positive and neg_ratio > pos_ratio:
            return neg_ratio  # V negative, words mostly negative -> consistent
        else:
            return 0.3  # mismatch -> low consistency
