"""Layer 0: Context mode detection and intent classification.

Before text hits the pendulum, this module classifies input into a mode
that determines the processing path. Pure pattern matching + word density
analysis. No ML. Fast.

Resolves: #45, #46
"""

import re


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

GREETING = "GREETING"
QUESTION = "QUESTION"
REQUEST = "REQUEST"
VENTING = "VENTING"
CASUAL = "CASUAL"
EMOTIONAL = "EMOTIONAL"
STATEMENT = "STATEMENT"

ALL_MODES = {GREETING, QUESTION, REQUEST, VENTING, CASUAL, EMOTIONAL, STATEMENT}

# ---------------------------------------------------------------------------
# Processing paths — which pipeline stages each mode activates
# ---------------------------------------------------------------------------

PROCESSING_PATHS = {
    GREETING:  {"pendulum": False, "chunker": False, "grader": False, "response": "personality_greeting"},
    QUESTION:  {"pendulum": False, "chunker": False, "grader": False, "response": "practical"},
    REQUEST:   {"pendulum": False, "chunker": False, "grader": False, "response": "action"},
    CASUAL:    {"pendulum": True,  "chunker": False, "grader": False, "response": "match_vibe"},
    EMOTIONAL: {"pendulum": True,  "chunker": True,  "grader": True,  "response": "full_empathy"},
    VENTING:   {"pendulum": True,  "chunker": True,  "grader": True,  "response": "full_empathy"},
    STATEMENT: {"pendulum": True,  "chunker": False, "grader": False, "response": "acknowledge"},
}


class IntentDetector:
    """Layer 0: Detect what the user WANTS before emotional processing."""

    # ------------------------------------------------------------------
    # Pattern sets
    # ------------------------------------------------------------------

    GREETING_WORDS = {
        "hey", "hi", "hello", "sup", "yo", "heya", "hiya", "howdy",
    }

    GREETING_PHRASES = [
        ("good", "morning"),
        ("good", "evening"),
        ("good", "afternoon"),
        ("what's", "up"),
        ("whats", "up"),
        ("how's", "it", "going"),
        ("how", "are", "you"),
    ]

    QUESTION_STARTERS = {
        "do", "does", "did", "can", "could", "would", "should",
        "will", "is", "are", "was", "were", "have", "has",
        "why", "how", "what", "where", "when", "who", "which",
    }

    REQUEST_VERBS = {
        "help", "fix", "make", "build", "create", "find", "get",
        "show", "tell", "give", "send", "check", "update", "change",
    }

    EMOTIONAL_MARKERS = {
        "feel", "feeling", "felt", "hate", "love", "miss",
        "scared", "afraid", "angry", "sad", "happy", "depressed",
        "anxious", "worried", "furious", "heartbroken", "devastated",
        "lonely", "hopeless", "miserable", "terrified", "overwhelmed",
    }

    CRISIS_WORDS = {
        "die", "dying", "death", "dead", "kill", "suicide", "suicidal",
        "living", "alive", "exist", "point", "purpose", "worth",
        "end", "anymore", "nothing", "meaningless", "worthless",
    }

    CASUAL_MARKERS = {
        "lol", "lmao", "haha", "bruh", "ngl", "tbh", "fr", "rn",
        "idk", "smh", "omg", "wtf", "btw", "imo",
    }

    NEGATIVE_EMOTIONAL = {
        "hate", "sucks", "terrible", "horrible", "awful", "worst",
        "stupid", "dumb", "angry", "furious", "sick", "tired",
        "frustrated", "annoyed", "pissed", "done", "over",
    }

    FIRST_PERSON = {"i", "i'm", "im", "i've", "ive", "my", "me", "myself"}

    # Patterns for "I'm so [adj]" / "I feel [adj]"
    _RE_I_FEEL = re.compile(
        r"\bi(?:'m| am| feel| felt)\s+(?:so\s+)?(\w+)", re.IGNORECASE,
    )

    # Patterns for "can you [verb]" / "help me [verb]"
    _RE_CAN_YOU_VERB = re.compile(
        r"\b(?:can|could|would)\s+you\s+(\w+)", re.IGNORECASE,
    )
    _RE_HELP_ME_VERB = re.compile(
        r"\bhelp\s+me\s+(\w+)", re.IGNORECASE,
    )
    _RE_PLEASE_VERB = re.compile(
        r"\bplease\s+(\w+)", re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> tuple[str, float]:
        """Classify *text* into a mode.

        Returns:
            (mode, confidence) where mode is one of the module-level
            constants and confidence is 0.0 – 1.0.
        """
        if not text or not text.strip():
            return (STATEMENT, 0.3)

        stripped = text.strip()
        lower = stripped.lower()
        words = lower.split()

        # --- Priority 1: crisis-adjacent content always EMOTIONAL --------
        # Even if phrased as a question, crisis content must route to
        # full empathy.
        if self._is_crisis(lower, words):
            return (EMOTIONAL, 0.95)

        # --- Priority 2: strong emotional override -----------------------
        # High density of emotional markers overrides question/request.
        emo_score = self._emotional_score(lower, words)
        if emo_score >= 0.85:
            return (EMOTIONAL, min(emo_score, 0.95))

        # --- Priority 3: greetings (short, unambiguous) ------------------
        greeting_conf = self._check_greeting(lower, words)
        if greeting_conf > 0.0:
            return (GREETING, greeting_conf)

        # --- Priority 4/5: questions vs requests --------------------------
        # "can you [verb]" is a request, not a question, so check both and
        # let request win when it matches a concrete request pattern.
        question_conf = self._check_question(stripped, lower, words)
        request_conf = self._check_request(lower, words)

        if request_conf > 0.0 and request_conf >= question_conf:
            return (REQUEST, request_conf)
        if question_conf > 0.0:
            return (QUESTION, question_conf)
        if request_conf > 0.0:
            return (REQUEST, request_conf)

        # --- Priority 6: casual ------------------------------------------
        casual_conf = self._check_casual(words)
        if casual_conf > 0.0:
            return (CASUAL, casual_conf)

        # --- Priority 7: emotional (lower threshold) ---------------------
        if emo_score >= 0.5:
            return (EMOTIONAL, emo_score)

        # --- Priority 8: venting -----------------------------------------
        venting_conf = self._check_venting(lower, words)
        if venting_conf > 0.0:
            return (VENTING, venting_conf)

        # --- Default -----------------------------------------------------
        return (STATEMENT, 0.5)

    def get_processing_path(self, mode: str) -> dict:
        """Return the pipeline configuration for *mode*."""
        return PROCESSING_PATHS.get(mode, PROCESSING_PATHS[STATEMENT]).copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_crisis(self, lower: str, words: list[str]) -> bool:
        """Detect crisis-adjacent language regardless of sentence form."""
        crisis_hits = sum(1 for w in words if w in self.CRISIS_WORDS)
        emotional_hits = sum(1 for w in words if w in self.EMOTIONAL_MARKERS)

        # "what's the point of living" — crisis + existential
        if crisis_hits >= 2:
            return True
        if crisis_hits >= 1 and emotional_hits >= 1:
            return True

        # Direct mentions
        for phrase in ("kill myself", "want to die", "end it all",
                       "point of living", "worth living", "no reason to live",
                       "don't want to be alive", "cant go on", "can't go on"):
            if phrase in lower:
                return True

        return False

    def _emotional_score(self, lower: str, words: list[str]) -> float:
        """Score 0-1 for how emotional the text is."""
        if not words:
            return 0.0

        hits = sum(1 for w in words if w in self.EMOTIONAL_MARKERS)
        density = hits / len(words)

        # Boost for first-person + emotional marker
        first_person = any(w in self.FIRST_PERSON for w in words)
        match = self._RE_I_FEEL.search(lower)
        if match and first_person:
            density += 0.3

        # Boost for multiple emotional markers
        if hits >= 2:
            density += 0.15

        return min(density * 2.0, 1.0)

    def _check_greeting(self, lower: str, words: list[str]) -> float:
        """Return confidence > 0 if input is a greeting, else 0."""
        if not words:
            return 0.0

        # Single greeting word (1-4 words total)
        if len(words) <= 4 and words[0] in self.GREETING_WORDS:
            return 0.9

        # Greeting phrases
        for phrase in self.GREETING_PHRASES:
            if len(words) >= len(phrase):
                if tuple(words[:len(phrase)]) == phrase:
                    return 0.9

        return 0.0

    def _check_question(self, original: str, lower: str, words: list[str]) -> float:
        """Return confidence > 0 if input is a question, else 0."""
        if not words:
            return 0.0

        has_qmark = original.rstrip().endswith("?")
        starts_with_q = words[0] in self.QUESTION_STARTERS

        if has_qmark and starts_with_q:
            return 0.85
        if has_qmark:
            return 0.8
        if starts_with_q and len(words) >= 2:
            return 0.75

        return 0.0

    def _check_request(self, lower: str, words: list[str]) -> float:
        """Return confidence > 0 if input is a request, else 0."""
        if not words:
            return 0.0

        # "please [verb]"
        m = self._RE_PLEASE_VERB.search(lower)
        if m and m.group(1) in self.REQUEST_VERBS:
            return 0.85

        # "can you [verb]" / "help me [verb]"
        for pat in (self._RE_CAN_YOU_VERB, self._RE_HELP_ME_VERB):
            m = pat.search(lower)
            if m and m.group(1) in self.REQUEST_VERBS:
                return 0.8

        # Imperative: first word is a request verb
        if words[0] in self.REQUEST_VERBS:
            return 0.8

        # "please" anywhere + verb somewhere
        if "please" in words:
            if any(w in self.REQUEST_VERBS for w in words):
                return 0.75

        return 0.0

    def _check_casual(self, words: list[str]) -> float:
        """Return confidence > 0 if input is casual, else 0."""
        if not words:
            return 0.0

        casual_count = sum(1 for w in words if w in self.CASUAL_MARKERS)

        # Short input with casual marker
        if len(words) <= 3 and casual_count >= 1:
            return 0.7

        # >30% casual markers
        if len(words) > 0 and casual_count / len(words) > 0.3:
            return 0.7

        return 0.0

    def _check_venting(self, lower: str, words: list[str]) -> bool:
        """Return confidence > 0 if input is venting, else 0."""
        if len(words) < 5:
            return 0.0

        # No question marks
        if "?" in lower:
            return 0.0

        first_person = any(w in self.FIRST_PERSON for w in words)
        neg_hits = sum(1 for w in words if w in self.NEGATIVE_EMOTIONAL)

        if first_person and neg_hits >= 1:
            conf = 0.7
            if neg_hits >= 2:
                conf = 0.8
            if neg_hits >= 3:
                conf = 0.85
            return conf

        # No first person but lots of negativity
        if neg_hits >= 2:
            return 0.65

        return 0.0
