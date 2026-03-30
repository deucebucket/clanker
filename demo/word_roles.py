"""Word role detection — descriptors inherit emotion from their subject.

Classifies words into roles (descriptor, subject, bridge) and makes
descriptors inherit emotional direction from their subject via bridge words.
"""


class WordRoleDetector:
    """Detects word roles and resolves descriptor forces from context."""

    # Bridge words — zero emotional weight, just connectors
    BRIDGES = {
        "is", "are", "was", "were", "the", "a", "an", "of", "in", "at",
        "to", "for", "by", "with", "on", "this", "that", "it", "be",
        "been", "being", "has", "had", "have", "do", "does", "did",
    }

    # Descriptor words — inherit emotion from what they describe
    DESCRIPTORS = {
        "moist", "big", "small", "old", "new", "dark", "light", "hot", "cold",
        "deep", "soft", "hard", "fast", "slow", "heavy", "quiet", "loud",
        "sharp", "smooth", "thin", "thick", "raw", "rough", "wet", "dry",
        "bitter", "sweet", "rich", "poor", "bare", "bright", "dim", "fresh",
        "stale", "clean", "dirty", "full", "empty", "wide", "narrow",
        "long", "short", "high", "low", "strong", "weak", "loose", "tight",
        "plain", "fancy", "flat", "round", "straight", "wild", "tame",
        "real", "fake", "pure", "whole", "broken", "open", "closed",
        "young", "certain", "strange", "odd", "normal", "usual", "rare",
        "common", "special", "simple", "complex", "clear", "obvious",
    }

    def resolve_force(self, word: str, word_force: tuple, words: list,
                      position: int) -> tuple:
        """If word is a descriptor, check nearby words and inherit their
        valence direction.

        Args:
            word: the current word
            word_force: the base (dv, da, dd, du, dg) force from WORD_FORCES
            words: full word list of the sentence
            position: current word's index in the list

        Returns:
            Modified force tuple (may have flipped dv sign)
        """
        if word not in self.DESCRIPTORS:
            return word_force  # not a descriptor, use base force

        if word_force is None:
            return word_force

        dv, da, dd, du, dg = word_force[:5]

        # Strong descriptors (|dv| > 30) carry their OWN emotion — don't flip them.
        # "poor girl" = sympathy (poor stays negative), NOT "girl is positive so poor becomes positive"
        # "beautiful disaster" = beautiful stays positive despite negative subject
        # Only MILD descriptors (|dv| <= 30) inherit from their subject.
        if abs(dv) > 30:
            return word_force  # descriptor has strong opinion, keep it

        # Look ahead 1-2 words for the subject being described
        for offset in [1, 2]:
            next_idx = position + offset
            if next_idx < len(words):
                next_word = words[next_idx]
                if next_word in self.BRIDGES:
                    continue  # skip bridge, keep looking
                # Found a subject — check its valence in WORD_FORCES
                from .forces_curated import EMOTIONAL_VOCABULARY as WORD_FORCES  # V2 vocab
                if next_word in WORD_FORCES:
                    subject_dv = WORD_FORCES[next_word][0]
                    if subject_dv > 10:
                        # Subject is positive — descriptor becomes positive
                        dv = abs(dv)
                    elif subject_dv < -10:
                        # Subject is negative — descriptor stays/becomes negative
                        dv = -abs(dv)
                break  # only check first non-bridge word

        # Also look behind 1-2 words (for "the cake is moist" pattern)
        for offset in [1, 2]:
            prev_idx = position - offset
            if prev_idx >= 0:
                prev_word = words[prev_idx]
                if prev_word in self.BRIDGES:
                    continue
                from .forces_curated import EMOTIONAL_VOCABULARY as WORD_FORCES  # V2 vocab
                if prev_word in WORD_FORCES:
                    subject_dv = WORD_FORCES[prev_word][0]
                    if subject_dv > 10:
                        dv = abs(dv)
                    elif subject_dv < -10:
                        dv = -abs(dv)
                break

        return (dv, da, dd, du, dg)

    def is_bridge(self, word: str) -> bool:
        """Return True if word is a bridge (connector) word."""
        return word in self.BRIDGES

    def is_descriptor(self, word: str) -> bool:
        """Return True if word is a descriptor."""
        return word in self.DESCRIPTORS
