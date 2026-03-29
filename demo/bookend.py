"""Bookend parsing -- relational geometry from first + last significant word.

The first and last *significant* words of a sentence define the relational
geometry: who is speaking, who is spoken about, and in which direction
emotional force flows.  Trailing filler (lol, nvm, ...) is stripped so the
real semantic endpoint is exposed.
"""


class BookendParser:
    """Extract first word + last significant word and derive relational direction."""

    # Tail modifiers to strip when finding the real ending
    TAIL_NOISE = {
        "lol", "lmao", "haha", "hahaha", "nvm", "jk", "tbh", "ngl",
        "fr", "smh", "bruh", "idk", "anyway", "anyways", "tho", "though",
    }

    # Pronouns and their roles
    SELF_PRONOUNS = {"i", "me", "my", "myself", "i'm", "i've", "i'll", "im"}
    TARGET_PRONOUNS = {"you", "your", "yourself", "you're", "you've"}
    OTHER_PRONOUNS = {"he", "him", "his", "she", "her", "they", "them", "their"}
    ABSENCE_WORDS = {"nobody", "nothing", "no one", "none", "nowhere"}
    TOTALITY_WORDS = {"everyone", "everything", "everybody", "always", "all"}

    def parse(self, text: str) -> dict:
        """Parse bookends and return relational geometry.

        Returns:
            {
                "first_word": str,
                "last_word": str (after stripping tail noise),
                "tail_modifier": str or None (the stripped noise),
                "first_role": "self"|"target"|"other"|"absence"|"totality"|"question"|"action"|None,
                "last_role": same options,
                "direction": "self->target"|"target->self"|"self->self"|"absence->self"|"self->void"|etc,
                "pattern": "declaration"|"question"|"accusation"|"isolation"|"plea"|etc,
            }
        """
        words = text.lower().split()
        if not words:
            return {
                "first_word": "", "last_word": "", "tail_modifier": None,
                "first_role": None, "last_role": None,
                "direction": None, "pattern": None,
            }

        first = words[0]

        # Strip tail noise to find real ending
        tail_mod = None
        real_words = list(words)
        while real_words and real_words[-1] in self.TAIL_NOISE:
            tail_mod = real_words.pop()
        last = real_words[-1] if real_words else words[-1]

        # Determine roles
        first_role = self._get_role(first)
        last_role = self._get_role(last)

        # Determine direction and pattern
        direction = (
            f"{first_role}\u2192{last_role}" if first_role and last_role else None
        )
        pattern = self._detect_pattern(first_role, last_role, first, words)

        return {
            "first_word": first,
            "last_word": last,
            "tail_modifier": tail_mod,
            "first_role": first_role,
            "last_role": last_role,
            "direction": direction,
            "pattern": pattern,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_role(self, word):
        """Map a single word to its relational role."""
        if word in self.SELF_PRONOUNS:
            return "self"
        if word in self.TARGET_PRONOUNS:
            return "target"
        if word in self.OTHER_PRONOUNS:
            return "other"
        if word in self.ABSENCE_WORDS:
            return "absence"
        if word in self.TOTALITY_WORDS:
            return "totality"
        if word in {
            "can", "could", "would", "should", "will", "do", "does", "did",
            "is", "are", "why", "how", "what", "where", "when", "who",
        }:
            return "question"
        if word in {"help", "stop", "please", "let", "make", "give", "take"}:
            return "action"
        return None

    def _detect_pattern(self, first_role, last_role, first_word, words):
        """Derive the high-level relational pattern from the two roles."""
        if first_role == "question":
            return "question"
        if first_role == "self" and last_role == "self":
            return "self_loop"          # "I hate myself"
        if first_role == "self" and last_role == "target":
            return "declaration"        # "I love you"
        if first_role == "absence" and last_role == "self":
            return "isolation"          # "Nobody loves me"
        if first_role == "self" and last_role is None:
            return "statement"          # "I went to the store"
        if first_role == "action" and last_role in ("self", "action"):
            return "plea"               # "Help me" / "Help me please"
        if first_role == "target" and last_role == "self":
            return "accusation"         # "You hurt me"
        if first_role == "totality":
            return "overwhelm"          # "Everything is my fault"
        return "general"

    def get_priming_vadug(self) -> dict:
        """Return VADUG adjustments to prime the pendulum based on the pattern.

        This will be used by the Plinko router to prime processing.
        """
        # Placeholder -- actual priming values TBD when integrated with Plinko.
        pass
