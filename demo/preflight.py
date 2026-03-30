"""Pre-flight text analysis — environmental multipliers before word processing.

Reads the SHAPE of the text before the pendulum processes individual words.
Digital prosody, syntactic velocity, and structural signals that modify
the emotional environment.

These are not word-level forces — they're environmental conditions that
amplify or dampen the entire sentence processing.

Usage:
    from demo.preflight import PreflightAnalyzer
    analyzer = PreflightAnalyzer()
    mods = analyzer.analyze("I AM SO DONE WITH THIS!!!")
    # mods = {"arousal_mult": 1.4, "urgency_mult": 1.3, "dominance_mult": 0.9, ...}
"""

import re
from dataclasses import dataclass


@dataclass
class PreflightResult:
    """Environmental multipliers derived from text structure."""
    arousal_mult: float = 1.0      # A multiplier from structure
    urgency_mult: float = 1.0      # U multiplier from structure
    dominance_mult: float = 1.0    # D multiplier from structure
    gravity_offset: float = 0.0    # G offset from structure
    valence_drag: float = 1.0      # V amplifier — ellipsis drags emotion further

    # Raw metrics (for debugging/logging)
    caps_ratio: float = 0.0        # % of uppercase letters
    punct_density: float = 0.0     # punctuation marks per word
    exclamation_count: int = 0     # number of !
    ellipsis_count: int = 0        # number of ...
    question_count: int = 0        # number of ?
    word_count: int = 0            # total words
    avg_word_length: float = 0.0   # average word length


class PreflightAnalyzer:
    """Analyze text structure before word-by-word pendulum processing.

    Returns environmental multipliers that modify the pendulum's behavior
    for the entire sentence, based on how the text LOOKS, not what it SAYS.
    """

    def analyze(self, raw_text: str) -> PreflightResult:
        """Analyze raw text (before lowercasing/cleaning) for structural signals."""
        result = PreflightResult()

        if not raw_text or not raw_text.strip():
            return result

        # --- Metric extraction ---

        # Word count
        words = raw_text.split()
        result.word_count = len(words)
        if result.word_count == 0:
            return result

        # Average word length
        alpha_words = [w for w in words if any(c.isalpha() for c in w)]
        if alpha_words:
            result.avg_word_length = sum(len(w) for w in alpha_words) / len(alpha_words)

        # Capitalization ratio (exclude punctuation-only tokens)
        alpha_chars = [c for c in raw_text if c.isalpha()]
        if alpha_chars:
            upper_count = sum(1 for c in alpha_chars if c.isupper())
            result.caps_ratio = upper_count / len(alpha_chars)

        # Punctuation density
        punct_chars = sum(1 for c in raw_text if c in '.,!?;:...!?-')
        result.punct_density = punct_chars / result.word_count

        # Specific punctuation counts
        result.exclamation_count = raw_text.count('!')
        result.ellipsis_count = raw_text.count('...')
        result.question_count = raw_text.count('?')

        # --- Multiplier calculation ---

        # ALL CAPS detection
        # If >70% uppercase and more than 3 words, it's shouting
        if result.caps_ratio > 0.7 and result.word_count >= 3:
            result.arousal_mult *= 1.4    # shouting = high arousal
            result.urgency_mult *= 1.3    # shouting = urgency
            result.dominance_mult *= 1.2  # shouting = power assertion

        # Exclamation marks
        # Each ! adds arousal, diminishing returns
        if result.exclamation_count > 0:
            exc_boost = min(0.4, result.exclamation_count * 0.1)
            result.arousal_mult *= (1.0 + exc_boost)
            if result.exclamation_count >= 3:
                result.urgency_mult *= 1.2  # !!!  = very urgent

        # Ellipsis (... patterns)
        # "..." is a drumroll — DRAGS emotion further in its current direction.
        # "im fine..." = even more negative (the dots add "but wait there's more")
        # Also signals hesitation and processing weight.
        if result.ellipsis_count > 0:
            result.dominance_mult *= 0.85   # hesitation = lower D
            result.gravity_offset -= 5.0 * result.ellipsis_count  # heavier G
            result.arousal_mult *= 0.9      # trailing off = lower A
            # V-drag: amplify distance from neutral (emotion drags further)
            result.valence_drag = 1.15 * result.ellipsis_count  # 15% amplification per ...

        # Sentence length as velocity
        # Very short (1-3 words) = high density, high urgency
        # Very long (20+ words) = low density, overwhelm
        if result.word_count <= 3:
            result.urgency_mult *= 1.2
            result.arousal_mult *= 1.1
        elif result.word_count >= 20:
            result.dominance_mult *= 0.85   # can't get to the point = low D
            result.arousal_mult *= 0.9      # meandering = low A

        # Multiple sentences in one message (rapid fire)
        sentence_count = len(re.split(r'[.!?]+', raw_text.strip()))
        if sentence_count >= 3 and result.word_count <= 15:
            # Multiple short sentences = high urgency (staccato delivery)
            result.urgency_mult *= 1.2
            result.arousal_mult *= 1.1

        return result
