"""Core data structures for the Clanker pipeline."""

from dataclasses import dataclass, field

# -------------------------------------------------------------
# VADUG: 5-byte emotional coordinate system
# V=Valence(0-255), A=Arousal(0-255), D=Dominance(0-255), U=Urgency(0-255), G=Gravity(0-255)
# 128 = neutral center for V/A/D/G, 0 = minimum for U
# G: 0=crushing/sinking, 128=grounded, 255=floating/soaring
# -------------------------------------------------------------

@dataclass
class VADUG:
    v: int = 128  # valence: 0=negative, 128=neutral, 255=positive
    a: int = 128  # arousal: 0=calm, 255=intense
    d: int = 128  # dominance: 0=helpless, 255=in control
    u: int = 0    # urgency: 0=no rush, 255=critical
    g: int = 128  # gravity: 0=crushing/sinking, 128=grounded, 255=floating/soaring

    def __post_init__(self):
        self.v = max(0, min(255, self.v))
        self.a = max(0, min(255, self.a))
        self.d = max(0, min(255, self.d))
        self.u = max(0, min(255, self.u))
        self.g = max(0, min(255, self.g))

    def to_bytes(self) -> bytes:
        return bytes([self.v, self.a, self.d, self.u, self.g])

    def __str__(self):
        return f"V{self.v} A{self.a} D{self.d} U{self.u} G{self.g}"

    def describe(self) -> str:
        parts = []
        if self.v < 60: parts.append("very negative")
        elif self.v < 90: parts.append("negative")
        elif self.v < 118: parts.append("slightly negative")
        elif self.v < 138: parts.append("neutral")
        elif self.v < 170: parts.append("slightly positive")
        elif self.v < 200: parts.append("positive")
        else: parts.append("very positive")

        if self.a < 60: parts.append("very calm")
        elif self.a < 100: parts.append("calm")
        elif self.a < 156: parts.append("moderate energy")
        elif self.a < 200: parts.append("intense")
        else: parts.append("very intense")

        if self.d < 60: parts.append("feels helpless")
        elif self.d < 100: parts.append("low control")
        elif self.d < 156: parts.append("neutral control")
        elif self.d < 200: parts.append("in control")
        else: parts.append("dominant")

        if self.u > 200: parts.append("CRITICAL urgency")
        elif self.u > 150: parts.append("high urgency")
        elif self.u > 80: parts.append("moderate urgency")
        elif self.u > 30: parts.append("low urgency")
        else: parts.append("no urgency")

        if self.g < 30: parts.append("CRUSHING weight")
        elif self.g < 70: parts.append("heavy/sinking")
        elif self.g < 110: parts.append("slightly heavy")
        elif self.g < 148: parts.append("grounded")
        elif self.g < 190: parts.append("light")
        elif self.g < 230: parts.append("soaring")
        else: parts.append("floating/weightless")

        return ", ".join(parts)


# Backward-compatible alias
VADU = VADUG


# -------------------------------------------------------------
# Metadata Header: CERT, SRC, GOAL, REL
# -------------------------------------------------------------

@dataclass
class MetadataHeader:
    vadu: VADUG = field(default_factory=VADUG)
    cert: int = 128     # certainty 0-255
    src: int = 0x04     # source: USER by default for input
    goal: int = 0x00    # intent
    rel: int = 200      # relevance

    SRC_NAMES = {
        0x00: "UNKNOWN", 0x01: "TRAINED", 0x02: "RAG",
        0x03: "INFERRED", 0x04: "USER", 0x05: "EXTERNAL", 0x06: "VERIFIED"
    }
    GOAL_NAMES = {
        0x00: "HELP", 0x01: "CLARIFY", 0x02: "WARN", 0x03: "TEACH",
        0x04: "EXECUTE", 0x05: "REFUSE", 0x06: "EMPATHIZE",
        0x07: "CONFIRM", 0x08: "EXPLORE"
    }

    def to_bytes(self) -> bytes:
        return self.vadu.to_bytes() + bytes([self.cert, self.src, self.goal, self.rel])

    def __str__(self):
        return (f"[{self.vadu}] CERT={self.cert} "
                f"SRC={self.SRC_NAMES.get(self.src, '?')} "
                f"GOAL={self.GOAL_NAMES.get(self.goal, '?')} "
                f"REL={self.rel}")


# -------------------------------------------------------------
# Personality Vector: 8 bytes defining the model's character
# -------------------------------------------------------------

@dataclass
class PersonalityVector:
    gullibility: int = 25      # 0=skeptical, 255=believes everything
    agreeableness: int = 100   # 0=contrarian, 255=total yes-man
    suggestibility: int = 30   # 0=immune, 255=easily manipulated
    truthfulness: int = 235    # 0=lies freely, 255=cannot lie
    safety: int = 200          # 0=no guardrails, 255=refuses everything
    curiosity: int = 170       # 0=incurious, 255=explores everything
    assertiveness: int = 120   # 0=passive, 255=forceful
    playfulness: int = 100     # 0=dead serious, 255=everything is a joke

    @property
    def emotional_sensitivity(self) -> float:
        """How much emotional forces affect this personality.

        High sensitivity (>1.0): forces hit harder
          - High gullibility, high suggestibility, low assertiveness
          - A scared kid feels everything more intensely

        Low sensitivity (<1.0): forces are dampened
          - Low gullibility, low suggestibility, high assertiveness
          - A grizzled veteran barely flinches

        Returns multiplier: 0.5 (stoic) to 2.0 (hypersensitive)
        """
        # Factors that INCREASE sensitivity
        amplifiers = (self.gullibility + self.suggestibility + self.agreeableness) / 3
        # Factors that DECREASE sensitivity
        dampeners = (self.assertiveness + (255 - self.suggestibility)) / 2

        # Normalize to 0-1 range, then map to 0.5-2.0 multiplier
        raw = amplifiers / (dampeners + 1)
        return max(0.5, min(2.0, 0.5 + raw * 0.8))

    @property
    def gravity_bias(self) -> float:
        """Personality's baseline gravity tendency.

        Negative bias = things feel heavier (anxious, traumatized)
        Positive bias = things feel lighter (playful, resilient)

        Returns offset: -30 to +30 applied to G dimension.
        """
        lightness = (self.playfulness + self.curiosity) / 2
        heaviness = (self.safety + (255 - self.playfulness)) / 2
        return (lightness - heaviness) / 255 * 30

    @property
    def dominance_baseline(self) -> float:
        """Personality's baseline sense of control.

        High assertiveness = starts with more agency (D shifted up)
        Low assertiveness = starts feeling helpless (D shifted down)

        Returns offset: -20 to +20 applied to D dimension.
        """
        return (self.assertiveness - 128) / 128 * 20

    def __str__(self):
        return (f"GUL={self.gullibility} AGR={self.agreeableness} "
                f"SUG={self.suggestibility} TRU={self.truthfulness} "
                f"SAF={self.safety} CUR={self.curiosity} "
                f"ASR={self.assertiveness} PLY={self.playfulness}")
