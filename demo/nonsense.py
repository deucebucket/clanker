"""Layer -1: Nonsense / gibberish input detection.

Before any emotional processing, catch keyboard smashes, repeated characters,
and other unintelligible input that would waste pipeline cycles. Sits alongside
IntentDetector in the preprocessing layer.

Resolves: #29
"""

import re


# ---------------------------------------------------------------------------
# Result constants
# ---------------------------------------------------------------------------

NONSENSE = "NONSENSE"
NON_EMOTIONAL = "NON_EMOTIONAL"
VALID = "VALID"


class NonsenseDetector:
    """Detect gibberish, keyboard smashes, and unintelligible input."""

    KEYBOARD_ROWS = [
        set("qwertyuiop"),
        set("asdfghjkl"),
        set("zxcvbnm"),
    ]

    VOWELS = set("aeiouy")

    # Known words and slang that should never be flagged as nonsense.
    # This is intentionally small — only words that look like gibberish
    # but aren't.  The full WORD_FORCES dict covers the rest.
    KNOWN_WORDS = {
        # Common slang / internet-speak
        "lol", "lmao", "bruh", "ngl", "tbh", "fr", "rn", "idk", "smh",
        "omg", "wtf", "btw", "imo", "irl", "afk", "gg", "wp", "ez",
        "nah", "yep", "yup", "hmm", "huh", "mhm", "ugh", "pfft", "shh",
        "tl;dr", "fyi", "diy", "rip", "stfu", "gtfo", "rofl", "xd",
        # Words that look like gibberish but aren't
        "xkcd", "php", "sql", "css", "html", "xml", "json", "http",
        "https", "www", "ftp", "ssh", "tcp", "gpu", "cpu", "ram", "ssd",
        "rgb", "pdf", "gif", "jpg", "png", "mp3", "mp4", "url", "api",
        # Common short words / contractions
        "the", "a", "an", "i", "me", "my", "ok", "no", "so", "do",
        "is", "am", "be", "he", "we", "or", "if", "it", "at", "by",
        "to", "in", "on", "of", "up", "oh", "ah", "um", "hi", "yo",
        "go", "us", "as",
        # Emotional words that might look odd
        "haha", "hehe", "meh", "blah", "grr", "gah", "argh", "oof",
        "yay", "woo", "aww", "ooh", "eww", "hmph",
    }

    # Patterns that indicate non-emotional content (not gibberish, just
    # not something the emotional pipeline should process).
    _RE_URL = re.compile(
        r"https?://|www\.|localhost[:\d]|ftp://|\S+\.\S+/\S+", re.IGNORECASE,
    )
    _RE_CODE = re.compile(
        r"[{}\[\]();].*[{}\[\]();]|=>|->|::|import\s|def\s|class\s|"
        r"function\s|var\s|let\s|const\s|console\.",
        re.IGNORECASE,
    )

    # Repeated-character pattern (same char 3+ times in a row).
    _RE_REPEATED = re.compile(r"(.)\1{2,}")

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def detect(self, text: str) -> tuple[bool, str]:
        """Determine if *text* is nonsense or non-emotional content.

        Returns:
            (is_nonsense, reason) — reason is empty string if input is valid.
        """
        # --- Empty / whitespace-only ---
        if not text or not text.strip():
            return (True, "empty input")

        stripped = text.strip()

        # --- All punctuation / symbols (no alphanumeric at all) ---
        if not any(c.isalnum() for c in stripped):
            return (True, "no alphanumeric characters")

        # --- URL / code fragment detection (not gibberish, just wrong lane) ---
        if self._RE_URL.search(stripped):
            return (True, "non-emotional content (URL)")
        if self._RE_CODE.search(stripped):
            return (True, "non-emotional content (code)")

        # --- Per-word scoring ---
        words = stripped.lower().split()

        # Single-character-spam: "a a a a a a"
        if len(words) >= 4 and len(set(words)) == 1 and len(words[0]) == 1:
            return (True, "single character spam")

        # Deduplicate repeated-char runs for scoring: "hahahahaha" → "haha"
        cleaned_words = [self._dedup_repeats(w) for w in words]

        scores = []
        for raw, clean in zip(words, cleaned_words):
            score = self._word_nonsense_score(raw, clean)
            scores.append(score)

        if not scores:
            return (True, "empty input")

        avg = sum(scores) / len(scores)

        if avg > 0.6:
            return (True, "gibberish detected")

        return (False, "")

    # ---------------------------------------------------------------------------
    # Scoring helpers
    # ---------------------------------------------------------------------------

    def _word_nonsense_score(self, raw: str, cleaned: str) -> float:
        """Score a single word from 0.0 (real) to 1.0 (gibberish)."""
        # Strip non-alpha for analysis but keep raw for length checks
        alpha = re.sub(r"[^a-z]", "", cleaned)
        raw_alpha = re.sub(r"[^a-z]", "", raw)

        # All same character repeated: "aaaaaaa", "bbbbb" — always nonsense
        # Must check before known-word lookup since "a" is a known word
        if len(raw_alpha) > 3 and len(set(raw_alpha)) == 1:
            return 0.95

        # Known word (raw or cleaned form) → definitely real
        if raw in self.KNOWN_WORDS or cleaned in self.KNOWN_WORDS:
            return 0.0

        # Very short words are hard to judge — give benefit of the doubt
        if len(alpha) <= 3:
            return 0.0

        # Keyboard row slam: 5+ chars from a single row
        if len(alpha) >= 5:
            for row in self.KEYBOARD_ROWS:
                if all(c in row for c in alpha):
                    return 0.9

        # No vowels in a 4+ char word
        if len(alpha) >= 4 and not (set(alpha) & self.VOWELS):
            # But only if it's not a known abbreviation
            if raw not in self.KNOWN_WORDS and cleaned not in self.KNOWN_WORDS:
                return 0.8

        # Consonant-to-vowel ratio > 4:1 for 5+ char words
        if len(alpha) >= 5:
            vowel_count = sum(1 for c in alpha if c in self.VOWELS)
            consonant_count = len(alpha) - vowel_count
            if vowel_count > 0 and consonant_count / vowel_count > 4:
                return 0.7
            if vowel_count == 0:
                return 0.8

        return 0.0

    def _dedup_repeats(self, word: str) -> str:
        """Collapse repeated characters: 'hahahahaha' → 'haha', 'aaaa' → 'a'.

        Strategy: detect repeating substrings. For "hahaha", find that "ha"
        repeats. For "aaaa", find that "a" repeats.
        """
        # First, try to find a repeating substring pattern
        for length in range(1, len(word) // 2 + 1):
            pattern = word[:length]
            if len(word) % length == 0 and word == pattern * (len(word) // length):
                # "hahaha" → keep two cycles: "haha"; "aaa" → "a"
                if length == 1:
                    return pattern
                return pattern * min(2, len(word) // length)

        # No perfect repeat — just collapse runs of 3+ same char to 2
        return self._RE_REPEATED.sub(r"\1\1", word)
