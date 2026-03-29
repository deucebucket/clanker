"""Sarcasm Template Detection — structural patterns, not word matching.

Sarcasm is a SKILL, not a state. Skills have practiced templates.
The content words change but the skeleton stays the same.

6 core templates cover most English sarcasm:
  1. EXCL + POS + REPEAT     "Oh great another Monday"
  2. EXCL + FRAME + POS      "Oh how nice"
  3. AGREE + FRAME + [...]   "Yeah because that worked"
  4. POS + REPEAT + [...]    "Great another expert"
  5. SELF + INTENSE + POS + MUNDANE  "Im thrilled to redo my work"
  6. NEG + POS_FRAME + REALITY  "Nothing says X like Y"

Usage:
    from demo.sarcasm_templates import SarcasmTemplateDetector
    detector = SarcasmTemplateDetector()
    result = detector.detect(words, trace)
    if result.detected:
        print(f"Template {result.template}: {result.description}")
"""

from dataclasses import dataclass
from typing import Optional
from demo.forces_curated import EMOTIONAL_VOCABULARY


@dataclass
class SarcasmTemplateResult:
    detected: bool
    template: int           # 1-6, or 0 if none
    confidence: float       # 0.0-1.0
    description: str


# Word sets for template matching
EXCLAMATIONS = frozenset({"oh", "wow", "gee", "yay", "boy", "well", "whoa"})
FAKE_AGREES = frozenset({"yeah", "sure", "right", "clearly", "obviously",
                         "totally", "absolutely", "definitely", "certainly"})
FRAMES = frozenset({"how", "what", "because", "like", "since", "as"})
REPEATS = frozenset({"another", "more", "again", "always", "every", "yet"})
INTENSIFIERS = frozenset({"so", "very", "really", "extremely", "absolutely",
                          "totally", "completely", "incredibly", "super"})


def _is_positive(word):
    """Check if a word has positive emotional force."""
    if word in EMOTIONAL_VOCABULARY:
        return EMOTIONAL_VOCABULARY[word][0] > 15
    return False


def _is_negative(word):
    """Check if a word has negative emotional force."""
    if word in EMOTIONAL_VOCABULARY:
        return EMOTIONAL_VOCABULARY[word][0] < -15
    return False


class SarcasmTemplateDetector:
    """Detect sarcasm from structural templates, not word lists."""

    def detect(self, words: list, trace: list = None) -> SarcasmTemplateResult:
        """Check if a word sequence matches a sarcasm template.

        Args:
            words: lowercased word list from tokenizer
            trace: optional pendulum trace for additional signals
        """
        if len(words) < 2:
            return SarcasmTemplateResult(False, 0, 0.0, "")

        # Try each template
        for check in [self._template1, self._template2, self._template3,
                       self._template4, self._template5, self._template6]:
            result = check(words)
            if result.detected:
                return result

        return SarcasmTemplateResult(False, 0, 0.0, "")

    def _template1(self, words):
        """EXCL + POS_WORD + REPEAT + [anything]
        'Oh great another Monday'
        'Oh fantastic the printer is jammed again'
        """
        if len(words) < 3:
            return SarcasmTemplateResult(False, 0, 0.0, "")

        has_excl = words[0] in EXCLAMATIONS
        has_pos = any(_is_positive(w) for w in words[1:4])
        has_repeat = any(w in REPEATS for w in words)

        if has_excl and has_pos and has_repeat:
            return SarcasmTemplateResult(True, 1, 0.85,
                "exclamation + positive + repetition = false enthusiasm")

        # Variant: EXCL + POS without repeat — ONLY if there's also
        # a negative or mundane signal. Pure EXCL + POS = genuine enthusiasm.
        if has_excl and has_pos and not has_repeat:
            has_neg = any(_is_negative(w) for w in words)
            has_mundane = any(w in ("meeting", "monday", "work", "homework",
                                    "traffic", "bills", "chores") for w in words)
            if has_neg or has_mundane:
                return SarcasmTemplateResult(True, 1, 0.6,
                    "exclamation + positive + negative/mundane = sarcastic")

        return SarcasmTemplateResult(False, 0, 0.0, "")

    def _template2(self, words):
        """EXCL + FRAME + POS_WORD
        'Oh how nice'
        'Wow what a wonderful surprise'
        """
        if len(words) < 3:
            return SarcasmTemplateResult(False, 0, 0.0, "")

        has_excl = words[0] in EXCLAMATIONS
        has_frame = any(w in FRAMES for w in words[0:3])
        has_pos = any(_is_positive(w) for w in words[1:])

        if has_excl and has_frame and has_pos:
            return SarcasmTemplateResult(True, 2, 0.8,
                "exclamation + framing + positive = rhetorical sarcasm")

        # Variant: FRAME at position 0 + POS = "How nice", "What a surprise"
        if words[0] in FRAMES and has_pos:
            return SarcasmTemplateResult(True, 2, 0.7,
                "framing opener + positive = rhetorical sarcasm")

        return SarcasmTemplateResult(False, 0, 0.0, "")

    def _template3(self, words):
        """FAKE_AGREE + FRAME + [anything]
        'Yeah because that worked out so well'
        'Sure let me just drop everything'
        'Clearly youve put so much thought into this'
        """
        if len(words) < 3:
            return SarcasmTemplateResult(False, 0, 0.0, "")

        has_agree = words[0] in FAKE_AGREES
        has_frame = any(w in FRAMES for w in words[1:4])

        if has_agree and has_frame:
            return SarcasmTemplateResult(True, 3, 0.75,
                "fake agreement + framing = mock reasoning")

        # Variant: fake agree at start alone
        if has_agree and len(words) >= 4:
            return SarcasmTemplateResult(True, 3, 0.5,
                "fake agreement opener = possible sarcasm")

        return SarcasmTemplateResult(False, 0, 0.0, "")

    def _template4(self, words):
        """POS_WORD + REPEAT + [anything]
        'Great another expert'
        'Brilliant idea I cant believe'
        'Thanks a lot for nothing'
        """
        if len(words) < 3:
            return SarcasmTemplateResult(False, 0, 0.0, "")

        first_pos = _is_positive(words[0])
        has_repeat = any(w in REPEATS for w in words[1:4])
        has_neg_later = any(_is_negative(w) for w in words[2:])

        if first_pos and has_repeat:
            return SarcasmTemplateResult(True, 4, 0.7,
                "positive opener + repetition = sarcastic complaint")

        if first_pos and has_neg_later:
            return SarcasmTemplateResult(True, 4, 0.6,
                "positive opener + later negative = bait and switch")

        return SarcasmTemplateResult(False, 0, 0.0, "")

    def _template5(self, words):
        """SELF + [INTENSE] + POS_WORD + mundane/negative action
        'Im absolutely thrilled to redo all my work'
        'I love how nothing ever goes according to plan'
        """
        if len(words) < 5:
            return SarcasmTemplateResult(False, 0, 0.0, "")

        has_self = words[0] in ("i", "im", "i'm")
        has_intense = any(w in INTENSIFIERS for w in words[1:4])
        has_pos = any(_is_positive(w) for w in words[1:5])
        has_neg_context = any(_is_negative(w) for w in words[3:])
        has_nothing = "nothing" in words or "never" in words
        # Mundane-task words signal sarcastic context even without explicit negatives
        mundane_tasks = {"redo", "rewrite", "retype", "refill", "refile",
                         "overtime", "paperwork", "commute", "chores",
                         "again", "another", "same", "repeat"}
        has_mundane = any(w in mundane_tasks for w in words[3:])

        if has_self and has_pos and (has_neg_context or has_nothing or has_mundane):
            conf = 0.8 if has_intense else 0.6
            return SarcasmTemplateResult(True, 5, conf,
                "self + positive claim + negative reality = ironic self-report")

        return SarcasmTemplateResult(False, 0, 0.0, "")

    def _template6(self, words):
        """NEG_WORD + positive framing + reality
        'Nothing says teamwork like doing it all yourself'
        'Of course its my fault it always is'
        """
        if len(words) < 4:
            return SarcasmTemplateResult(False, 0, 0.0, "")

        starts_neg = _is_negative(words[0]) or words[0] in ("nothing", "nobody")
        has_pos_frame = any(_is_positive(w) for w in words[1:5])
        has_like = "like" in words

        if starts_neg and has_pos_frame:
            return SarcasmTemplateResult(True, 6, 0.7,
                "negative anchor + positive framing = bitter irony")

        # "Nothing says X like Y" — structural sarcasm even without positive vocab word
        if words[0] in ("nothing", "nobody") and "says" in words[:3] and has_like:
            return SarcasmTemplateResult(True, 6, 0.75,
                "'nothing says X like Y' = classic ironic structure")

        # "Of course" pattern
        if words[0] == "of" and len(words) > 1 and words[1] == "course":
            return SarcasmTemplateResult(True, 6, 0.65,
                "'of course' opener = resigned sarcasm")

        return SarcasmTemplateResult(False, 0, 0.0, "")
