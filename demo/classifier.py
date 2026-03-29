"""Intent-adaptive sentiment classifier for VADUG valence scores.

Instead of fixed V thresholds for positive/negative/neutral, the thresholds
adapt based on the detected intent mode. Emotional/venting inputs use tight
thresholds (force a decision), while questions/greetings use wide thresholds
(neutral is fine).
"""


from .forces import WORD_FORCES
from .entropy import EntropyClassifier
from .multipath import MultiPathProcessor

class AdaptiveClassifier:
    """Classify VADUG into positive/negative/neutral with intent-adaptive thresholds."""

    # Per-intent threshold configurations
    # Tuned on 71K sentences (15 books + Reddit + tweets), 4000 sample sweep
    _word_forces = WORD_FORCES
    _entropy = EntropyClassifier()
    _multipath = MultiPathProcessor()

    THRESHOLDS = {
        "EMOTIONAL": {"low": 123, "high": 124},   # tight — force decision after decay retune
        "VENTING":   {"low": 123, "high": 124},   # same as emotional
        "CASUAL":    {"low": 123, "high": 124},   # lean toward reading emotion
        "STATEMENT": {"low": 114, "high": 130},   # moderate — retuned with decay lambda=0.15
        "QUESTION":  {"low": 105, "high": 150},   # wide — neutral is fine
        "REQUEST":   {"low": 105, "high": 150},   # wide
        "GREETING":  {"low": 105, "high": 150},   # wide
        "DEFAULT":   {"low": 114, "high": 130},   # fallback — retuned
    }

    def classify(self, v: int, intent_mode: str = "DEFAULT") -> str:
        """Classify valence into positive/negative/neutral."""
        thresholds = self.THRESHOLDS.get(intent_mode, self.THRESHOLDS["DEFAULT"])
        if v > thresholds["high"]:
            return "positive"
        elif v < thresholds["low"]:
            return "negative"
        return "neutral"

    def get_thresholds(self, intent_mode: str = "DEFAULT") -> dict:
        """Return the threshold dict for a given intent mode."""
        return self.THRESHOLDS.get(intent_mode, self.THRESHOLDS["DEFAULT"])

    @classmethod
    def update_thresholds(cls, new_thresholds: dict):
        """Update thresholds from a dict of {mode: {"low": int, "high": int}}."""
        for mode, vals in new_thresholds.items():
            if mode in cls.THRESHOLDS:
                cls.THRESHOLDS[mode] = vals

    def classify_with_entropy(self, v: int, intent_mode: str,
                              pendulum_history: list = None) -> str:
        """Classify with entropy-based neutral override."""
        if pendulum_history and self._entropy.should_override_neutral(
                pendulum_history, v, intent_mode):
            return "neutral"
        return self.classify(v, intent_mode)

    def classify_multipath(self, text: str, intent_mode: str) -> str:
        """Classify using multi-path processing for mixed-signal sentences."""
        v, a, d, u, g, path, consistency, history = self._multipath.process(text)

        # Use entropy check on the winning path's history
        if history and self._entropy.should_override_neutral(history, v, intent_mode):
            return "neutral"

        return self.classify(v, intent_mode)

    def classify_density(self, density_v: int, intent_mode: str,
                         pendulum_history: list = None) -> str:
        """Classify using density-normalized V instead of raw V.

        Density thresholds use the same adaptive thresholds as classify(),
        with an entropy override when history is available.
        """
        if pendulum_history and self._entropy.should_override_neutral(
                pendulum_history, density_v, intent_mode):
            return "neutral"
        return self.classify(density_v, intent_mode)

    def classify_with_density(self, v: int, intent_mode: str, words: list) -> str:
        """Classify with emotional density check — low density near center = neutral."""
        # Count emotionally significant words (|dv| > 10)
        emotional = sum(1 for w in words if w in self._word_forces and abs(self._word_forces[w][0]) > 10)
        density = emotional / max(len(words), 1)
        
        # If low density AND V near center → neutral (factual content)
        if density < 0.12 and abs(v - 128) < 15:
            return "neutral"
        
        return self.classify(v, intent_mode)

    def classify_5d(self, v, a, d, u, g, intent_mode, pendulum_history=None):
        """5D classification — blends V and G for better 3-color output.
        
        Tuned: V*0.9 + G*0.1 gives +0.5pp over V-only.
        G adds signal because sinking/floating correlates with negative/positive.
        """
        # Entropy override still uses raw V
        if pendulum_history and self._entropy.should_override_neutral(
            pendulum_history, v, intent_mode):
            return "neutral"
        
        # Blend V and G
        blended = v * 0.9 + g * 0.1
        
        # Tuned thresholds for blended score
        if blended > 128:
            return "positive"
        elif blended < 124:
            return "negative"
        return "neutral"

