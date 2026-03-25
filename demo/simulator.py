#!/usr/bin/env python3
"""
Clanker Pipeline Simulator — Interactive Demo (v0.5.2: Sarcasm Detection)

Demonstrates the full Clanker processing pipeline:
1. VADUG Sequential Pendulum: parse English word-by-word → emotional arc
2. Metadata Header: CERT, SRC, GOAL, REL tagging
3. Harmony Response: mathematically derive response VADUG
4. Personality Filter: apply personality vector weights
5. Clanker Generation: produce Clanker opcodes with headers + byte encoding
6. Decode: translate back to English
7. Emotional Chunking: paragraph-level arc detection + per-chunk responses
8. Sentence Grader: 15-step emotional guardrail (A+ through F-)
9. Sarcasm Detection: three-signal analysis from pendulum trajectory

The sequential pendulum processes each word in context: the same word applies
different force depending on the current trajectory. "buddy" when positive =
friendly; "buddy" when tense = confrontational. Momentum, idiom detection,
anticipation patterns, and morphological fallback for unknown words.

NEW in v0.4: Paragraphs with multiple emotional beats get split at natural
boundaries (sentence endings, reversals like "but"/"however", causal links).
Each chunk gets its own pendulum run. An arc analyzer detects patterns
(valley, peak, descending, ascending, flat, mixed) and generates per-chunk
responses assembled with an arc-aware closer.

NEW in v0.5.1: Sentence-level emotional grader computes an overall grade
(A+ through F-) from chunk VADUG results. The grade acts as a GUARDRAIL —
it defines what response strategies are ALLOWED and BLOCKED. Even a playful
personality gets locked into empathy-only when the grade is F.

NEW in v0.5.2: Sarcasm detection from pendulum trajectory patterns. Three
signals: (1) Trajectory Reversal — positive spike then immediate drop,
(2) Intensity Mismatch — strong positive word in negative context,
(3) Context Contradiction — positive chunk after negative context with flat
delivery. Pure math from the pendulum, no sentiment classifier needed.

Run: python3 demo/simulator.py
"""

import re
import math
import sys
import os
from dataclasses import dataclass, field

# Import morphemes from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morphemes import decompose_word, ROOTS, PREFIXES, SUFFIXES
from decoder_templates import decode_vadug_to_text
import random


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

    def __str__(self):
        return (f"GUL={self.gullibility} AGR={self.agreeableness} "
                f"SUG={self.suggestibility} TRU={self.truthfulness} "
                f"SAF={self.safety} CUR={self.curiosity} "
                f"ASR={self.assertiveness} PLY={self.playfulness}")


# =============================================================
# STEP 1: Sequential Pendulum Engine
# =============================================================

# Word-level emotional force vectors (v_force, a_force, d_force, u_force, g_force)
# These push the pendulum from center (128,128,128,0,128)
# G (Gravity): positive = lighter/soaring, negative = heavier/sinking
#   Anger: +20 to +50 (rising, boiling)
#   Sadness: -20 to -50 (sinking, heavy)
#   Fear: +10 to +40 (floating ungrounded)
#   Joy: +30 to +60 (soaring, light)
#   Calm: -5 to +10 (grounded)
#   Disgust: +20 to +50 (repulsion rises)
#   Despair: -40 to -60 (crushing)
#   Neutral: 0 (grounded)
#   Urgency: +10 to +30 (adrenaline lift)
WORD_FORCES = {
    # ── Strong negative ──
    "hate": (-60, +40, +20, +30, +35), "terrible": (-50, +30, -10, +20, -20),
    "awful": (-50, +25, -15, +15, -25), "horrible": (-55, +35, -10, +25, -20),
    "worst": (-60, +30, -20, +25, -30), "died": (-70, +50, -40, +40, -50),
    "kill": (-60, +60, +30, +50, +30), "suicide": (-80, +60, -60, +80, -55),
    "devastated": (-65, +40, -50, +30, -45), "destroyed": (-55, +40, -30, +35, -30),
    "furious": (-50, +70, +40, +40, +40), "enraged": (-55, +75, +45, +45, +45),
    "die": (-80, +50, -60, +70, -50), "death": (-70, +40, -50, +50, -45),
    "hopeless": (-70, +20, -60, +40, -55), "worthless": (-70, +15, -60, +30, -50),
    "useless": (-50, +10, -50, +20, -35), "pointless": (-50, +5, -40, +15, -35),
    "suicidal": (-80, +60, -70, +80, -55), "depressed": (-60, -20, -50, +20, -50),
    "miserable": (-65, +10, -45, +20, -45), "suffering": (-60, +30, -40, +30, -35),
    "everything": (0, +10, 0, +10, 0), "nothing": (-30, -10, -30, +10, -25),
    "disgusting": (-55, +40, +15, +15, +30), "pathetic": (-45, +15, -35, +10, -30),
    "ruined": (-55, +30, -30, +20, -30), "devastating": (-60, +40, -40, +30, -45),
    "tragic": (-60, +35, -35, +20, -40), "nightmare": (-60, +50, -40, +30, -35),
    "catastrophe": (-65, +55, -35, +40, -30), "catastrophic": (-65, +55, -35, +40, -30),
    "wretched": (-60, +20, -40, +15, -45), "agonizing": (-60, +45, -30, +30, -25),
    "torment": (-55, +40, -30, +25, -20), "dreadful": (-55, +35, -40, +25, -30),
    "abysmal": (-55, +15, -40, +10, -50),

    # ── Moderate negative ──
    "bad": (-30, +10, -10, +10, -10), "sad": (-35, -10, -20, +5, -30),
    "angry": (-40, +40, +20, +25, +30), "mad": (-35, +35, +15, +20, +25),
    "upset": (-30, +25, -10, +15, -10), "frustrated": (-35, +35, -15, +25, +15),
    "annoyed": (-20, +20, +5, +10, +10), "disappointed": (-25, +5, -15, +5, -20),
    "worried": (-20, +25, -20, +25, +10), "anxious": (-25, +35, -25, +30, +15),
    "stressed": (-25, +30, -20, +30, +10), "tired": (-15, -25, -15, +5, -20),
    "bored": (-10, -30, -5, 0, -10), "lonely": (-30, -15, -25, +5, -30),
    "confused": (-15, +15, -20, +15, +10), "lost": (-25, +10, -30, +15, -20),
    "stuck": (-20, +15, -25, +20, -20), "broken": (-40, +10, -35, +15, -30),
    "failed": (-35, +10, -30, +15, -25), "sucks": (-30, +20, -5, +10, -10),
    "shit": (-25, +15, -5, +10, -5), "fuck": (-20, +30, +10, +15, +10),
    "damn": (-15, +20, +5, +10, +5), "crap": (-20, +10, -5, +5, -5),
    "wrong": (-20, +10, -10, +15, -5), "error": (-15, +15, -5, +20, -5),
    "bug": (-15, +15, -10, +25, -5), "crash": (-30, +30, -20, +35, -20),
    "pain": (-40, +30, -25, +20, -25), "hurt": (-35, +25, -20, +15, -20),
    "sick": (-30, +15, -25, +15, -15), "afraid": (-35, +40, -40, +25, +15),
    "scared": (-40, +45, -45, +25, +20), "fear": (-35, +40, -40, +25, +15),
    "frightened": (-40, +50, -45, +30, +20), "terrified": (-55, +60, -55, +40, +25),
    "panicked": (-50, +65, -50, +45, +30), "nervous": (-20, +30, -20, +20, +10),
    "uncomfortable": (-20, +15, -15, +10, -5), "uneasy": (-20, +20, -15, +15, +5),
    "awkward": (-15, +15, -20, +5, -5), "embarrassed": (-30, +25, -25, +10, -10),
    "ashamed": (-40, +25, -30, +10, -25), "guilty": (-35, +20, -25, +10, -25),
    "regret": (-30, +15, -20, +10, -20), "mistake": (-25, +15, -15, +15, -10),
    "problem": (-15, +15, -10, +20, -5), "issue": (-10, +10, -5, +15, -5),
    "trouble": (-20, +20, -15, +20, -10), "difficult": (-15, +15, -10, +15, -5),
    "struggle": (-25, +25, -20, +20, -15), "painful": (-40, +30, -25, +20, -25),
    "overwhelming": (-30, +40, -25, +25, -20), "exhausting": (-25, -15, -25, +10, -25),
    "draining": (-20, -15, -20, +10, -20), "tedious": (-15, -20, -10, +5, -10),
    "boring": (-10, -30, -5, 0, -10), "dull": (-10, -25, -5, 0, -10),
    "mediocre": (-10, -10, -5, 0, -5), "lame": (-15, -5, -10, +5, -10),
    "garbage": (-35, +15, -10, +10, -15), "trash": (-30, +15, -10, +10, -15),
    "waste": (-25, +10, -15, +10, -15), "meaningless": (-35, -5, -30, +10, -30),

    # ── Confrontational ── (anger rises)
    "shut": (-30, +40, +25, +20, +20), "nobody": (-25, +15, -15, +10, -10),
    "asked": (-10, +10, +5, +5, 0), "rude": (-30, +25, +10, +10, +15),
    "disrespectful": (-35, +30, +10, +10, +15), "arrogant": (-30, +20, +15, +5, +15),
    "selfish": (-30, +15, +10, +5, +10), "ignorant": (-25, +15, -5, +5, 0),
    "stupid": (-35, +25, +10, +10, +15), "idiot": (-40, +30, +15, +15, +20),
    "moron": (-40, +30, +15, +15, +20), "fool": (-30, +20, +10, +10, +10),
    "ridiculous": (-25, +25, +10, +10, +15), "absurd": (-25, +25, +10, +10, +15),
    "liar": (-40, +35, +15, +15, +20), "fake": (-30, +20, +10, +10, +10),
    "hypocrite": (-35, +25, +10, +10, +15), "manipulative": (-40, +25, +10, +15, +15),
    "toxic": (-45, +30, +10, +15, +20), "hostile": (-40, +40, +20, +20, +25),

    # ── Mild negative ──
    "not": (-10, +5, 0, +5, 0), "don't": (-10, +5, 0, +5, 0),
    "didn't": (-10, +5, 0, +5, 0), "can't": (-15, +10, -15, +10, -5),
    "won't": (-10, +10, +5, +5, 0), "never": (-15, +10, -5, +5, -5),
    "no": (-10, +5, 0, +5, 0), "stop": (-10, +15, +10, +15, 0),

    # ── Neutral / functional ──
    "help": (+10, +10, -10, +20, +5), "fix": (+5, +10, +5, +20, +5),
    "need": (-5, +10, -10, +25, -5), "want": (+5, +10, +5, +15, +5),
    "please": (+5, -5, -10, +10, 0), "think": (0, +5, +5, +5, 0),
    "know": (+5, +5, +10, +5, 0), "try": (+5, +10, -5, +10, +5),
    "maybe": (-5, -5, -10, 0, 0), "might": (-5, -5, -5, 0, 0),
    "how": (0, +5, -5, +10, 0), "what": (0, +5, -5, +10, 0),
    "why": (-5, +10, -5, +15, 0), "when": (0, +5, 0, +15, 0),
    "work": (+5, +10, +10, +15, 0), "make": (+5, +10, +10, +10, +5),
    "understand": (+10, +10, +10, +5, +5),

    # ── Mild positive ──
    "good": (+25, +10, +10, 0, +10), "nice": (+20, +5, +5, 0, +10),
    "okay": (+10, -5, +5, 0, 0), "fine": (+10, -5, +5, 0, 0),
    "sure": (+10, 0, +10, +5, 0), "yes": (+15, +5, +10, +5, +5),
    "thanks": (+20, +5, 0, 0, +10), "cool": (+20, +5, +10, 0, +10),
    "interesting": (+15, +15, +10, +5, +10), "better": (+20, +10, +10, +5, +10),

    # ── Moderate positive ── (joy soars)
    "great": (+35, +20, +15, 0, +20), "happy": (+40, +20, +15, 0, +30),
    "glad": (+30, +15, +10, 0, +20), "love": (+50, +30, +15, 0, +40),
    "like": (+20, +10, +5, 0, +10), "enjoy": (+35, +15, +10, 0, +20),
    "beautiful": (+40, +15, +10, 0, +25), "perfect": (+45, +20, +20, 0, +30),
    "wonderful": (+45, +25, +15, 0, +30), "fantastic": (+45, +30, +20, 0, +35),
    "excellent": (+40, +20, +20, 0, +25), "awesome": (+45, +35, +20, 0, +35),
    "magnificent": (+50, +35, +25, 0, +40), "spectacular": (+50, +40, +25, 0, +40),
    "phenomenal": (+50, +40, +20, 0, +40), "outstanding": (+45, +30, +25, 0, +30),
    "brilliant": (+45, +30, +25, 0, +35), "genius": (+45, +30, +30, 0, +30),
    "remarkable": (+40, +25, +20, 0, +25), "extraordinary": (+45, +35, +25, 0, +35),
    "superb": (+45, +25, +25, 0, +30), "marvelous": (+45, +30, +20, 0, +30),
    "glorious": (+50, +35, +30, 0, +40), "blessed": (+40, +15, +20, 0, +25),
    "grateful": (+35, +15, +15, 0, +20), "thankful": (+30, +10, +10, 0, +15),
    "proud": (+35, +25, +30, 0, +25), "accomplished": (+35, +20, +30, 0, +25),
    "triumphant": (+45, +40, +40, 0, +45), "victorious": (+45, +40, +40, 0, +45),
    "successful": (+40, +25, +30, 0, +25), "thriving": (+40, +25, +25, 0, +25),
    "flourishing": (+40, +20, +25, 0, +25), "radiant": (+40, +25, +15, 0, +30),
    "gorgeous": (+40, +20, +10, 0, +25), "stunning": (+40, +35, +10, 0, +25),

    # ── Strong positive ── (soaring)
    "amazing": (+50, +40, +20, 0, +40), "incredible": (+50, +40, +20, 0, +40),
    "ecstatic": (+55, +55, +20, 0, +50), "thrilled": (+50, +45, +20, 0, +45),
    "excited": (+40, +45, +15, +10, +35), "celebrate": (+50, +45, +25, 0, +45),

    # ── Social / emotional ──
    "forgive": (+25, +10, +15, 0, +15), "forget": (-5, -5, +5, 0, 0),
    "remember": (+10, +10, +5, +5, +5), "miss": (-20, +15, -10, +5, -15),
    "belong": (+30, +10, +15, 0, +10), "welcome": (+30, +10, +15, 0, +15),
    "accept": (+25, +5, +15, 0, +10), "reject": (-35, +25, -25, +10, -20),
    "abandon": (-45, +30, -35, +20, -35), "betray": (-50, +40, -30, +25, -25),
    "trust": (+30, +5, +20, 0, +10), "loyal": (+30, +10, +20, 0, +10),
    "faithful": (+30, +10, +20, 0, +10), "devoted": (+35, +15, +20, 0, +15),
    "cherish": (+40, +15, +15, 0, +20), "adore": (+45, +25, +15, 0, +30),
    "treasure": (+35, +15, +15, 0, +15), "appreciate": (+30, +10, +15, 0, +15),
    "respect": (+25, +10, +15, 0, +10), "admire": (+30, +15, +15, 0, +15),
    "inspire": (+35, +25, +20, 0, +30), "motivate": (+25, +25, +20, +5, +15),
    "encourage": (+25, +15, +15, 0, +15), "support": (+25, +10, +15, 0, +10),
    "comfort": (+30, -10, +15, 0, +10), "heal": (+25, +10, +15, 0, +15),
    "recover": (+20, +10, +15, +5, +10), "grow": (+20, +10, +15, 0, +15),
    "overcome": (+25, +25, +30, +5, +20), "survive": (+15, +20, +25, +10, +5),
    "endure": (+10, +15, +25, +5, 0), "persevere": (+20, +20, +30, +5, +5),
    "resilient": (+25, +20, +35, +5, +10), "alone": (-30, -10, -20, +5, -25),

    # ── Urgency / intensity (figurative usage) ──
    "killing": (-30, +50, +15, +35, +20), "dying": (-20, +40, -20, +30, -30),
    "screaming": (-25, +55, +10, +30, +25), "scream": (-25, +55, +10, +30, +25),
    "exploding": (-20, +55, +10, +35, +30), "burning": (-20, +45, +10, +25, +25),
    "crashing": (-30, +45, -15, +30, -25), "shattering": (-35, +40, -20, +25, -15),
    "breaking": (-30, +35, -15, +20, -15), "falling": (-25, +30, -25, +20, -40),
    "drowning": (-35, +40, -30, +25, -50), "suffocating": (-35, +45, -30, +30, -45),
    "choking": (-30, +40, -25, +30, -35), "crushing": (-30, +35, -20, +20, -55),

    # ── Passive / resigned ── (heavy, sinking)
    "whatever": (-15, -10, -15, 0, -10), "guess": (-10, -5, -10, 0, -5),
    "suppose": (-5, -5, -10, 0, -5), "anyway": (-5, 0, -5, 0, 0),
    "nevermind": (-15, -10, -15, 0, -10), "meh": (-10, -15, -10, 0, -10),
    "sigh": (-10, -10, -10, 0, -10),

    # ── Context modifiers / intensifiers ──
    "anymore": (-15, +10, -10, +5, -5), "always": (0, +10, 0, +10, 0),
    "everyone": (0, +10, +5, +5, 0), "forever": (0, +10, 0, +5, 0),
    "completely": (0, +10, 0, +5, 0), "totally": (0, +10, 0, +5, 0),
    "entirely": (0, +10, 0, +5, 0), "absolutely": (0, +15, +5, +5, 0),
    "literally": (0, +10, 0, +5, 0),

    # ── Religious / existential ──
    "holy": (+10, +30, +10, +10, +20), "believe": (+15, +10, +10, +5, +10),
    "pray": (+10, +10, -5, +10, +10), "faith": (+20, +10, +15, 0, +15),
    "miracle": (+40, +35, +10, +5, +40),
    "god": (+5, +20, +5, +10, +15), "hell": (-20, +25, +5, +10, -20),

    # ── Weather / nature (metaphorical weight) ──
    "storm": (-20, +35, -10, +20, -10), "calm": (+20, -30, +15, -5, +5),
    "thunder": (-15, +40, +5, +15, -5), "lightning": (-5, +45, +5, +15, +15),
    "darkness": (-25, +10, -15, +5, -20), "light": (+20, +10, +10, 0, +30),
    "fire": (-10, +40, +15, +20, +25), "ice": (-10, -10, +5, +5, -10),

    # ── Urgency markers ── (adrenaline lift)
    "now": (0, +10, +5, +40, +10), "immediately": (0, +15, +5, +50, +15),
    "asap": (0, +15, +5, +55, +15), "urgent": (-5, +20, -5, +60, +15),
    "emergency": (-20, +40, -20, +80, +10), "hurry": (-5, +20, -5, +45, +10),
    "quickly": (0, +10, 0, +35, +10), "soon": (0, +5, 0, +20, +5),
    "deadline": (-10, +20, -10, +45, +10), "critical": (-15, +25, -5, +55, +10),
    "important": (0, +10, +5, +30, +5),

    # ── Key derived forms (override morpheme decomposition for accuracy) ──
    "hopelessness": (-70, +20, -60, +40, -55), "hopeless": (-70, +20, -60, +40, -55),
    "hopeful": (+40, +15, +20, 0, +30), "hopefulness": (+40, +15, +20, 0, +30),
    "helpless": (-40, +15, -50, +20, -35), "helplessness": (-40, +15, -50, +20, -35),
    "fearless": (+30, +20, +40, 0, +20), "fearful": (-35, +40, -35, +20, +15),
    "joyful": (+55, +30, +25, 0, +45), "joyless": (-40, -10, -25, +5, -35),
    "powerful": (+30, +30, +50, +5, +15), "powerless": (-30, +10, -50, +15, -35),
    "wonderful": (+45, +25, +15, 0, +30), "beautiful": (+40, +15, +10, 0, +25),
    "peaceful": (+35, -25, +20, 0, +10), "graceful": (+30, -10, +20, 0, +20),
    "harmful": (-40, +25, +10, +15, +10), "harmless": (+10, -10, -5, 0, +5),
    "ungrateful": (-30, +15, -10, +10, -15),
    "unacceptable": (-35, +25, +10, +15, +15),
    "unbearable": (-50, +30, -30, +25, -40), "unbelievable": (+10, +45, +5, +10, +20),
    "underwhelming": (-15, -10, -10, +5, -10),

    # ── Social / greeting words (context-sensitive base values) ──
    "hey": (+12, +12, +5, +5, +5), "hi": (+15, +10, +5, 0, +5),
    "hello": (+15, +8, +5, 0, +5), "yo": (+10, +15, +5, +5, +5),
    "buddy": (+15, +10, +5, 0, +5), "friend": (+20, +10, +5, 0, +10),
    "pal": (+15, +10, +5, 0, +5), "dude": (+10, +10, +5, 0, +5),
    "man": (+5, +5, +5, 0, 0), "bro": (+10, +12, +5, 0, +5),
    "listen": (-5, +15, +15, +15, 0), "look": (-5, +10, +10, +10, 0),
    "actually": (-8, +10, +10, +5, 0), "well": (+5, +5, +5, 0, 0),

    # ── Pronouns (context-sensitive) ──
    "i": (0, +3, +5, 0, 0), "you": (0, +5, 0, +5, 0),
    "we": (+5, +5, +5, 0, 0), "they": (0, +3, 0, 0, 0),
    "my": (0, +3, +5, 0, 0), "your": (0, +5, 0, +5, 0),
    "me": (0, +3, -5, 0, 0),

    # ── Profanity / strong exclamations ── (anger rises)
    "bullshit": (-30, +30, +10, +15, +15),
    "bastard": (-35, +30, +15, +15, +20), "ass": (-15, +15, +5, +5, +5),
    "asshole": (-40, +35, +15, +15, +20), "jerk": (-25, +20, +10, +10, +10),
    "creep": (-25, +20, -5, +10, +5), "freak": (-15, +25, -5, +10, +10),
    "psycho": (-30, +35, +10, +20, +15), "insane": (-15, +40, +10, +15, +15),
    "crazy": (-10, +30, +5, +10, +10),

    # ── Achievement / effort ── (soaring)
    "won": (+35, +30, +30, 0, +35), "win": (+35, +30, +30, +5, +35),
    "champion": (+40, +35, +35, 0, +40), "hero": (+35, +30, +30, 0, +35),
    "legend": (+35, +25, +25, 0, +25), "master": (+30, +20, +35, 0, +20),
    "achieve": (+30, +25, +25, +5, +25), "earned": (+30, +20, +25, 0, +20),

    # ── Loss / grief ── (sinking, heavy)
    "grief": (-55, +25, -35, +15, -45), "mourn": (-50, +15, -30, +10, -40),
    "sorrow": (-50, +10, -30, +10, -40), "despair": (-65, +25, -55, +30, -60),
    "anguish": (-60, +40, -35, +25, -35), "agony": (-55, +45, -30, +25, -30),
    "heartbreak": (-55, +30, -30, +15, -40), "heartbroken": (-55, +30, -35, +15, -40),
    "devastation": (-60, +35, -40, +25, -45), "tragedy": (-55, +30, -30, +20, -40),
    "doom": (-55, +20, -45, +25, -50), "cursed": (-40, +25, -25, +15, -25),

    # ── Relationship ──
    "sorry": (-10, +5, -10, +5, -5), "apologize": (-5, +10, -15, +5, -5),
    "promise": (+15, +10, +15, +10, +10), "swear": (-5, +20, +15, +10, +5),
    "blame": (-25, +25, +10, +10, +10), "fault": (-20, +15, -5, +10, -10),
    "deserve": (+5, +15, +10, +5, +5), "owe": (-5, +10, -5, +10, -5),
    "jealous": (-25, +25, -10, +10, +10), "envy": (-20, +20, -10, +5, +5),

    # ── Physical state ──
    "exhausted": (-20, -20, -25, +10, -30), "drained": (-20, -15, -20, +10, -25),
    "numb": (-20, -25, -20, +5, -20), "shaking": (-20, +40, -25, +20, +10),
    "trembling": (-20, +35, -25, +15, +10), "crying": (-35, +30, -25, +15, -30),
    "tears": (-30, +25, -20, +10, -25), "sobbing": (-40, +35, -30, +15, -35),
    "laughing": (+35, +30, +15, 0, +35), "smiling": (+30, +15, +15, 0, +20),

    # ── Certainty / uncertainty ──
    "certain": (+15, +10, +25, +5, +5), "definite": (+15, +10, +25, +5, +5),
    "doubt": (-10, +10, -15, +10, -5), "uncertain": (-10, +10, -15, +10, -5),
    "impossible": (-25, +15, -20, +10, -15), "possible": (+10, +5, +5, +5, +5),
    "inevitable": (-10, +15, -10, +15, -10), "obvious": (+5, +10, +15, +5, +5),
    "clearly": (+5, +10, +15, +5, +5), "apparently": (-5, +5, +5, +5, 0),

    # ── Time / existential ──
    "final": (-10, +15, +10, +15, -5), "last": (-10, +10, +5, +10, -5),
    "first": (+10, +10, +10, +5, +10), "begin": (+10, +10, +10, +5, +10),
    "end": (-10, +10, +5, +10, -10), "over": (-10, +5, 0, +5, -5),
    "done": (-5, +5, +5, +5, 0), "finished": (-5, +5, +5, +5, 0),
    "enough": (-10, +15, +10, +10, 0), "ever": (0, +10, 0, +5, 0),
    "entire": (0, +10, 0, +10, 0), "whole": (0, +5, 0, +5, 0),
    "life": (+5, +10, +5, +5, +5), "world": (+5, +10, +5, +5, +5),
    "best": (+40, +25, +20, 0, +30), "works": (+5, +10, +10, +10, 0),

    # ── Life stressors / everyday crisis ── (the words real people actually use)
    "broke": (-35, +25, -25, +20, -25),      # broke down, broke, broken
    "broken": (-40, +20, -30, +15, -30),
    "rent": (-20, +20, -15, +25, -15),        # financial stress
    "bill": (-20, +15, -10, +20, -10),
    "bills": (-25, +20, -15, +25, -15),
    "debt": (-30, +25, -25, +25, -20),
    "money": (-10, +15, -5, +20, -5),
    "afford": (-15, +15, -15, +20, -10),
    "expensive": (-15, +10, -10, +15, -5),
    "landlord": (-15, +15, -10, +15, -10),
    "evict": (-45, +40, -40, +50, -35),
    "eviction": (-45, +40, -40, +50, -35),
    "fired": (-50, +45, -40, +30, -35),
    "laid": (-30, +25, -25, +20, -20),        # laid off
    "unemployed": (-40, +20, -35, +30, -30),
    "late": (-15, +20, -10, +30, -5),
    "overdue": (-20, +20, -15, +30, -10),
    "payment": (-10, +10, -5, +20, -5),
    "foreclosure": (-50, +35, -45, +40, -40),
    "bankrupt": (-50, +30, -45, +35, -40),
    "homeless": (-55, +30, -50, +40, -45),

    # ── Health / medical stressors ──
    "hospital": (-30, +30, -25, +35, -20),
    "surgery": (-30, +35, -30, +35, -25),
    "cancer": (-55, +35, -40, +40, -40),
    "diagnosis": (-30, +30, -25, +30, -20),
    "disease": (-40, +25, -30, +30, -30),
    "infection": (-25, +20, -20, +25, -15),
    "injury": (-30, +25, -25, +25, -20),
    "injured": (-30, +25, -25, +25, -20),
    "doctor": (-10, +15, -5, +20, -5),
    "emergency": (-35, +50, -25, +60, -15),
    "ambulance": (-35, +50, -30, +55, -20),
    "medication": (-15, +10, -10, +15, -5),
    "therapy": (-5, +10, -5, +10, +5),
    "chronic": (-30, +10, -25, +15, -25),

    # ── Domestic / relationship stressors ──
    "divorce": (-45, +35, -30, +25, -30),
    "separated": (-35, +25, -25, +15, -25),
    "cheating": (-50, +45, -25, +30, -20),
    "cheated": (-50, +45, -25, +30, -20),
    "affair": (-45, +40, -20, +25, -15),
    "custody": (-35, +30, -25, +35, -25),
    "argument": (-25, +35, +10, +20, +10),
    "fight": (-30, +40, +15, +25, +15),
    "fighting": (-30, +40, +15, +25, +15),
    "yelling": (-30, +45, +10, +20, +15),
    "screaming": (-30, +50, +10, +25, +20),
    "abuse": (-55, +40, -45, +35, -30),
    "abused": (-55, +40, -45, +35, -30),
    "neglect": (-40, +15, -35, +15, -30),

    # ── Work / school stressors ──
    "deadline": (-20, +30, -10, +45, -5),
    "overtime": (-20, +20, -15, +25, -10),
    "burnout": (-40, +15, -35, +15, -35),
    "overwhelmed": (-35, +35, -30, +25, -25),
    "exam": (-10, +25, -10, +30, -5),
    "test": (-5, +15, -5, +20, 0),
    "grade": (-10, +15, -10, +20, -5),
    "failed": (-40, +25, -35, +20, -30),
    "failing": (-35, +25, -30, +25, -25),
    "flunked": (-40, +25, -30, +15, -25),
    "expelled": (-50, +40, -40, +30, -35),
    "suspended": (-35, +30, -30, +25, -25),
    "boss": (-10, +15, +5, +15, 0),
    "coworker": (0, +5, 0, +5, 0),
    "promotion": (+35, +30, +25, +10, +25),

    # ── Daily friction / annoyance ──
    "traffic": (-15, +20, -10, +20, -5),
    "commute": (-10, +10, -5, +10, -5),
    "waiting": (-10, +10, -10, +15, -5),
    "delayed": (-15, +15, -10, +20, -10),
    "canceled": (-25, +20, -15, +15, -15),
    "cancelled": (-25, +20, -15, +15, -15),
    "closed": (-10, +10, -5, +10, -5),
    "broken": (-35, +20, -25, +20, -25),
    "stuck": (-25, +20, -25, +20, -15),
    "flat": (-15, +15, -10, +15, -10),        # flat tire
    "towed": (-25, +25, -20, +25, -15),
    "ticket": (-20, +20, -10, +20, -10),
    "fine": (-15, +10, -5, +15, -5),          # as in penalty (context: near "pay" or "got")
    "spilled": (-10, +15, -5, +10, -5),
    "stain": (-10, +10, -5, +5, -5),
    "leak": (-15, +15, -10, +20, -10),
    "flood": (-30, +35, -25, +35, -20),
    "mold": (-20, +15, -15, +20, -15),

    # ── Loss / death ──
    "died": (-60, +40, -45, +30, -40),
    "dead": (-55, +35, -40, +25, -40),
    "death": (-55, +35, -40, +30, -35),
    "funeral": (-45, +25, -35, +20, -35),
    "grave": (-40, +20, -35, +15, -35),
    "cemetery": (-35, +15, -30, +10, -30),
    "mourning": (-45, +25, -35, +15, -35),
    "passed": (-35, +20, -30, +15, -30),      # "passed away"
    "widow": (-40, +20, -30, +10, -35),
    "orphan": (-45, +25, -40, +15, -40),
    "miscarriage": (-55, +35, -45, +25, -45),
    "stillborn": (-60, +30, -50, +20, -50),

    # ── Physical / environmental ──
    "cold": (-10, +5, -5, +5, -5),
    "hot": (-5, +10, 0, +5, +5),
    "hungry": (-15, +15, -10, +15, -10),
    "thirsty": (-10, +10, -5, +10, -5),
    "exhausted": (-30, -20, -25, +10, -25),
    "insomnia": (-25, +20, -20, +15, -20),
    "pain": (-35, +30, -25, +25, -20),
    "ache": (-20, +15, -15, +10, -15),
    "headache": (-20, +15, -15, +15, -15),
    "nausea": (-25, +20, -15, +15, -15),

    # ── Direction words that carry emotional weight ──
    "down": (-15, +5, -10, +5, -15),          # "broke down", "feeling down"
    "up": (+10, +10, +10, +5, +10),           # "cheer up", "look up" (but "shut up" handled by idiom)
    "out": (-5, +10, 0, +10, 0),
    "away": (-15, +10, -10, +10, -10),
    "apart": (-25, +15, -20, +10, -20),
    "behind": (-15, +10, -10, +10, -10),
    "under": (-15, +10, -15, +10, -15),
    "raising": (-10, +15, -5, +20, -5),        # "raising rent"
    "rising": (-5, +15, 0, +15, +5),

    # ══════════════════════════════════════════════════════════════
    # EXPANDED VOCABULARY — real-life categories (v0.4.2)
    # ══════════════════════════════════════════════════════════════

    # ── Food / cooking (comfort, disgust, craving) ──
    "delicious": (+35, +20, +10, 0, +20),
    "tasty": (+30, +15, +10, 0, +15),
    "yummy": (+30, +15, +5, 0, +20),
    "gross": (-30, +20, +5, +5, +15),
    "starving": (-25, +25, -15, +25, -15),
    "feast": (+35, +25, +15, 0, +20),
    "cook": (+15, +10, +15, 0, +5),
    "bake": (+20, +10, +15, 0, +10),
    "recipe": (+10, +5, +10, 0, +5),
    "dinner": (+15, +5, +5, 0, +5),
    "lunch": (+10, +5, +5, 0, +5),
    "breakfast": (+15, +5, +5, 0, +5),
    "snack": (+15, +10, +5, 0, +10),
    "craving": (-5, +20, -10, +15, +5),
    "appetite": (+5, +10, -5, +10, +5),
    "flavor": (+15, +10, +5, 0, +10),
    "spicy": (+5, +20, +5, +5, +10),
    "sweet": (+25, +10, +5, 0, +15),
    "bitter": (-20, +10, +5, +5, -5),
    "sour": (-10, +10, +5, +5, 0),
    "rotten": (-40, +25, -10, +10, -20),
    "spoiled": (-25, +15, -5, +10, -10),
    "fresh": (+20, +10, +10, 0, +15),
    "organic": (+10, +5, +5, 0, +5),
    "junk": (-15, +5, -5, +5, -10),

    # ── Technology / internet ──
    "glitch": (-15, +15, -10, +20, -5),
    "lag": (-15, +15, -10, +15, -5),
    "buffer": (-10, +10, -5, +10, -5),
    "loading": (-5, +5, -5, +10, -5),
    "download": (+5, +5, +5, +5, 0),
    "upload": (+5, +5, +5, +5, 0),
    "update": (+5, +10, +5, +10, 0),
    "reboot": (-5, +10, +5, +15, 0),
    "hack": (-20, +30, +15, +25, +15),
    "hacked": (-35, +40, -25, +35, -15),
    "virus": (-30, +25, -20, +25, -15),
    "malware": (-30, +25, -20, +25, -15),
    "spam": (-15, +10, -5, +10, -5),
    "scam": (-35, +30, -20, +25, -10),
    "phishing": (-25, +20, -15, +20, -10),
    "password": (-5, +10, -5, +15, 0),
    "locked": (-20, +15, -15, +20, -10),
    "banned": (-35, +30, -30, +20, -20),
    "deleted": (-25, +20, -15, +15, -15),
    "warning": (-15, +20, -5, +25, +5),
    "alert": (-10, +25, +5, +30, +10),
    "notification": (-5, +10, 0, +15, 0),
    "ping": (0, +10, 0, +10, 0),
    "offline": (-15, +10, -10, +15, -10),
    "online": (+10, +5, +5, +5, +5),
    "connected": (+15, +10, +10, 0, +10),
    "disconnected": (-20, +15, -15, +15, -10),
    "wifi": (0, +5, 0, +5, 0),

    # ── Weather / nature (metaphorical + literal) ──
    "rain": (-10, +5, -5, +5, -10),
    "sunny": (+25, +15, +10, 0, +25),
    "cloudy": (-10, -5, -5, 0, -5),
    "fog": (-10, -5, -10, +5, -10),
    "snow": (+5, +5, 0, +5, +5),
    "wind": (-5, +15, -5, +10, +5),
    "hurricane": (-35, +50, -30, +45, -20),
    "tornado": (-35, +55, -30, +45, -15),
    "earthquake": (-35, +50, -35, +45, -30),
    "drought": (-25, +10, -20, +15, -20),
    "smoke": (-15, +15, -10, +15, -10),
    "blaze": (-20, +45, +10, +30, +25),
    "heat": (-5, +15, 0, +5, +5),
    "freeze": (-15, +15, -10, +10, -15),
    "freezing": (-15, +15, -10, +10, -15),
    "warm": (+20, +5, +10, 0, +10),
    "cool": (+15, +5, +10, 0, +10),     # overrides base "cool" but same values
    "breeze": (+15, +5, +5, 0, +10),
    "gentle": (+20, -10, +5, 0, +10),
    "harsh": (-25, +25, +10, +10, -5),
    "brutal": (-40, +40, +15, +20, -10),
    "scorching": (-15, +25, -5, +10, +10),

    # ── Travel / movement ──
    "found": (+25, +20, +15, +5, +15),
    "stranded": (-30, +25, -30, +25, -25),
    "missed": (-20, +15, -15, +15, -10),
    "caught": (-15, +20, -10, +15, -5),
    "arrived": (+20, +15, +15, 0, +10),
    "departed": (-5, +10, 0, +10, -5),
    "returned": (+15, +10, +10, 0, +5),
    "escape": (-15, +35, +20, +30, +20),
    "flee": (-25, +40, -15, +35, +15),
    "chase": (-15, +40, +15, +30, +15),
    "run": (-5, +25, +10, +20, +10),
    "walk": (+5, +5, +5, 0, +5),
    "crawl": (-15, +10, -20, +10, -20),
    "fly": (+15, +25, +15, +10, +35),
    "soar": (+35, +30, +20, 0, +40),
    "fall": (-25, +25, -25, +20, -35),
    "trip": (-10, +15, -10, +10, -10),
    "stumble": (-10, +15, -10, +10, -10),
    "wander": (+5, +5, -5, 0, +5),
    "drift": (-5, -5, -10, 0, -5),
    "home": (+20, -5, +15, 0, +10),
    "away": (-15, +10, -10, +10, -10),      # already exists but dict will keep last
    "far": (-10, +5, -5, +5, -5),
    "near": (+10, +5, +5, 0, +5),
    "journey": (+10, +15, +10, +5, +10),
    "adventure": (+25, +30, +15, +5, +25),
    "destination": (+15, +10, +10, +5, +10),

    # ── Education / learning ──
    "study": (+5, +15, +5, +15, 0),
    "learn": (+15, +15, +10, +5, +10),
    "teach": (+15, +10, +15, +5, +10),
    "graduate": (+35, +25, +25, 0, +30),
    "diploma": (+25, +15, +20, 0, +20),
    "degree": (+25, +15, +20, 0, +20),
    "scholarship": (+35, +25, +20, 0, +30),
    "tutor": (+10, +10, +10, +5, +5),
    "homework": (-10, +10, -5, +15, -5),
    "essay": (-5, +10, +5, +15, 0),
    "research": (+10, +15, +10, +10, +5),
    "discover": (+30, +25, +15, +5, +25),
    "knowledge": (+20, +10, +15, 0, +15),
    "wisdom": (+25, +5, +20, 0, +15),
    "ignorant": (-25, +15, -10, +5, -10),
    "smart": (+25, +15, +20, 0, +15),
    "dumb": (-25, +10, -15, +5, -10),
    "pass": (+20, +15, +15, +5, +15),

    # ── Music / art / creativity ──
    "ugly": (-30, +15, -10, +5, -10),
    "masterpiece": (+45, +30, +25, 0, +35),
    "art": (+20, +15, +10, 0, +15),
    "music": (+25, +15, +10, 0, +20),
    "song": (+20, +10, +10, 0, +15),
    "sing": (+25, +20, +10, 0, +20),
    "dance": (+30, +25, +15, 0, +30),
    "paint": (+15, +15, +10, 0, +10),
    "draw": (+15, +10, +10, 0, +10),
    "create": (+25, +20, +20, 0, +20),
    "destroy": (-40, +40, +20, +20, -15),
    "boring": (-10, -30, -5, 0, -10),       # exists but reinforces
    "vibrant": (+30, +25, +15, 0, +25),
    "colorful": (+20, +15, +10, 0, +15),
    "dark": (-15, +10, -10, +5, -15),
    "bright": (+25, +15, +10, 0, +20),
    "dim": (-10, -10, -5, 0, -10),
    "loud": (-5, +25, +5, +10, +10),
    "quiet": (+10, -20, +5, 0, 0),
    "silent": (-5, -25, -5, 0, -5),
    "harmony": (+30, +5, +15, 0, +20),
    "chaos": (-25, +45, -15, +25, +15),
    "rhythm": (+15, +15, +10, 0, +10),
    "flow": (+20, +10, +15, 0, +15),
    "block": (-15, +10, -15, +10, -10),

    # ── Sports / competition ──
    "lose": (-30, +25, -20, +10, -20),
    "defeat": (-30, +25, -25, +10, -20),
    "victory": (+40, +35, +35, 0, +40),
    "score": (+25, +25, +20, +5, +20),
    "goal": (+20, +20, +15, +10, +15),
    "team": (+15, +10, +10, 0, +10),
    "opponent": (-10, +20, +5, +10, +5),
    "rival": (-10, +25, +10, +10, +5),
    "compete": (+5, +25, +15, +10, +10),
    "practice": (+10, +10, +10, +5, +5),
    "train": (+10, +15, +15, +5, +5),
    "effort": (+10, +15, +10, +10, +5),
    "lazy": (-15, -20, -15, 0, -15),
    "cheat": (-35, +25, +10, +15, +10),
    "fair": (+15, +5, +15, 0, +10),
    "foul": (-25, +25, +5, +15, +10),
    "penalty": (-20, +20, -10, +15, -5),
    "referee": (0, +10, +15, +10, 0),
    "coach": (+10, +15, +15, +5, +5),
    "player": (+5, +10, +5, +5, +5),
    "game": (+10, +15, +10, +5, +10),
    "match": (+5, +15, +10, +5, +5),
    "tournament": (+10, +20, +10, +10, +10),
    "record": (+15, +15, +15, +5, +10),

    # ── Legal / justice ──
    "innocent": (+20, +15, -10, +10, +10),
    "arrest": (-35, +40, -35, +35, -15),
    "arrested": (-35, +40, -35, +35, -15),
    "jail": (-40, +25, -40, +25, -30),
    "prison": (-40, +25, -40, +25, -30),
    "sentence": (-30, +20, -25, +20, -20),
    "judge": (-10, +15, +20, +10, 0),
    "jury": (-5, +15, +10, +10, 0),
    "trial": (-20, +25, -15, +20, -10),
    "lawyer": (-10, +15, +10, +15, 0),
    "sue": (-25, +25, +15, +20, +5),
    "sued": (-30, +30, -20, +25, -10),
    "lawsuit": (-25, +25, -10, +20, -10),
    "crime": (-35, +30, -15, +20, -10),
    "criminal": (-35, +30, -15, +20, -10),
    "victim": (-40, +25, -35, +20, -25),
    "witness": (-10, +20, +5, +15, 0),
    "evidence": (+5, +15, +15, +10, +5),
    "verdict": (-10, +25, +10, +20, 0),
    "appeal": (-10, +15, -10, +20, +5),
    "parole": (+10, +15, -5, +10, +5),
    "bail": (-15, +20, -10, +25, -5),
    "justice": (+20, +20, +20, +10, +15),
    "injustice": (-40, +30, -25, +20, -15),
    "corrupt": (-40, +25, +10, +15, -10),
    "fraud": (-35, +25, -10, +20, -10),

    # ── Time / aging ──
    "young": (+15, +15, +10, 0, +15),
    "old": (-10, -5, -5, +5, -10),
    "aging": (-15, +5, -10, +5, -15),
    "elderly": (-10, -5, -10, +5, -15),
    "birthday": (+25, +20, +10, 0, +20),
    "anniversary": (+20, +15, +10, 0, +15),
    "expired": (-15, +10, -10, +15, -10),
    "rush": (-10, +25, -5, +30, +10),
    "patience": (+15, -15, +15, -5, +5),
    "impatient": (-15, +20, +5, +20, +5),
    "temporary": (-5, +5, -5, +5, 0),
    "permanent": (+5, +5, +10, +5, 0),
    "brief": (-5, +5, 0, +5, 0),
    "long": (-5, +5, 0, +5, -5),
    "short": (-5, +5, 0, +5, 0),
    "instant": (+5, +15, +5, +15, +5),
    "moment": (+5, +5, 0, +5, +5),
    "memory": (+10, +10, +5, 0, +5),
    "memories": (+10, +10, +5, 0, +5),
    "nostalgic": (+10, +10, -5, 0, -5),
    "future": (+10, +10, +5, +5, +10),
    "past": (-5, +5, -5, 0, -5),
    "present": (+10, +5, +5, 0, +5),

    # ── Parenting / family ──
    "baby": (+25, +15, -5, +10, +15),
    "child": (+20, +10, -5, +10, +10),
    "kid": (+15, +10, 0, +5, +10),
    "parent": (+10, +10, +10, +10, +5),
    "mother": (+25, +10, +10, +5, +10),
    "father": (+20, +10, +10, +5, +10),
    "mom": (+25, +10, +10, +5, +10),
    "dad": (+20, +10, +10, +5, +10),
    "son": (+20, +10, +5, +5, +10),
    "daughter": (+20, +10, +5, +5, +10),
    "brother": (+15, +10, +5, +5, +5),
    "sister": (+15, +10, +5, +5, +5),
    "family": (+25, +10, +10, +5, +10),
    "pregnant": (+10, +25, -10, +15, +5),
    "birth": (+20, +30, -5, +15, +15),
    "born": (+20, +20, 0, +5, +15),
    "raise": (+10, +10, +10, +10, +5),
    "growing": (+15, +10, +10, +5, +10),
    "teenager": (-5, +15, -5, +10, 0),
    "tantrum": (-25, +40, -10, +20, +15),
    "bedtime": (+5, -10, +5, +10, 0),
    "school": (+5, +10, +5, +10, 0),
    "protective": (+15, +15, +20, +10, +5),
    "nurture": (+25, +5, +15, 0, +10),
    "discipline": (+5, +15, +20, +10, 0),

    # ── Animals / pets ──
    "dog": (+20, +15, +10, 0, +15),
    "cat": (+15, +5, +5, 0, +10),
    "pet": (+20, +10, +10, 0, +10),
    "puppy": (+35, +25, +5, 0, +30),
    "kitten": (+35, +20, +5, 0, +30),
    "animal": (+10, +10, +5, 0, +5),
    "bird": (+10, +10, +5, 0, +15),
    "fish": (+5, +5, +5, 0, +5),
    "horse": (+10, +15, +10, 0, +10),
    "cute": (+30, +15, +5, 0, +20),
    "adorable": (+35, +20, +5, 0, +25),
    "playful": (+25, +25, +10, 0, +20),
    "aggressive": (-25, +35, +15, +20, +15),
    "wild": (-5, +30, +5, +10, +15),
    "tame": (+10, -10, +10, 0, 0),
    "bark": (-10, +20, +5, +10, +5),
    "bite": (-25, +30, +10, +20, +5),
    "scratch": (-15, +15, -5, +10, 0),
    "feed": (+10, +5, +10, +5, +5),
    "vet": (-10, +15, +5, +15, -5),
    "veterinarian": (-10, +15, +5, +15, -5),
    "adoption": (+25, +20, +15, +10, +15),
    "rescue": (+20, +30, +20, +20, +15),
    "shelter": (+10, +10, +5, +5, +5),
    "stray": (-15, +10, -10, +5, -10),

    # ── Shopping / consumerism ──
    "buy": (+10, +10, +10, +5, +5),
    "sell": (+5, +10, +10, +5, 0),
    "shop": (+15, +10, +10, 0, +10),
    "store": (+5, +5, +5, 0, 0),
    "price": (-5, +10, -5, +10, 0),
    "cheap": (+5, +5, +5, +5, -5),
    "sale": (+15, +10, +10, +5, +10),
    "discount": (+15, +10, +10, +5, +10),
    "bargain": (+15, +15, +10, +5, +10),
    "refund": (+10, +15, +10, +15, +5),
    "return": (-5, +10, +5, +10, 0),
    "exchange": (0, +10, +5, +10, 0),
    "quality": (+15, +10, +10, 0, +10),
    "brand": (+5, +5, +5, 0, +5),
    "luxury": (+20, +15, +15, 0, +20),
    "budget": (-10, +10, -5, +10, -5),
    "save": (+15, +10, +10, +5, +5),
    "spend": (-10, +10, 0, +10, -5),
    "splurge": (+10, +20, +10, +5, +10),
    "deal": (+15, +10, +10, +5, +10),
    "order": (+5, +5, +5, +5, 0),
    "delivery": (+10, +10, +5, +10, +5),
    "package": (+10, +10, +5, +5, +5),
    "shipped": (+5, +5, +5, +5, +5),

    # ── Housing / home ──
    "house": (+15, +5, +10, 0, +5),
    "apartment": (+5, +5, +5, 0, 0),
    "room": (+5, +5, +5, 0, 0),
    "bedroom": (+10, -5, +10, 0, +5),
    "kitchen": (+10, +5, +10, 0, +5),
    "bathroom": (+5, +5, +5, 0, 0),
    "clean": (+15, +10, +15, 0, +10),
    "dirty": (-15, +10, -5, +5, -10),
    "mess": (-15, +10, -10, +10, -10),
    "messy": (-15, +10, -10, +10, -10),
    "tidy": (+15, +5, +15, 0, +10),
    "organize": (+15, +10, +15, +5, +10),
    "clutter": (-10, +10, -10, +5, -5),
    "repair": (+5, +10, +10, +10, +5),
    "maintenance": (-5, +5, +5, +10, 0),
    "moving": (-10, +20, -5, +20, -5),
    "packing": (-10, +15, +5, +15, -5),
    "unpacking": (+5, +10, +5, +10, +5),
    "neighbor": (+5, +5, +5, 0, 0),
    "neighborhood": (+5, +5, +5, 0, 0),
    "safe": (+20, -10, +20, 0, +10),
    "unsafe": (-25, +25, -20, +20, -10),
    "cozy": (+25, -10, +15, 0, +15),
    "cramped": (-15, +10, -10, +5, -10),
    "spacious": (+15, +5, +10, 0, +10),

    # ── Communication ──
    "talk": (+5, +10, +5, +5, 0),
    "speak": (+5, +10, +10, +5, 0),
    "hear": (+5, +10, +5, +5, 0),
    "say": (0, +5, +5, +5, 0),
    "tell": (0, +10, +10, +10, 0),
    "ask": (+5, +10, -5, +10, 0),
    "answer": (+10, +10, +10, +5, +5),
    "question": (-5, +10, -5, +10, 0),
    "explain": (+10, +10, +10, +5, +5),
    "misunderstand": (-20, +15, -10, +10, -5),
    "miscommunication": (-20, +15, -10, +10, -5),
    "gossip": (-15, +20, +5, +10, +5),
    "rumor": (-20, +20, -5, +10, +5),
    "lie": (-30, +20, +10, +10, +5),
    "truth": (+20, +15, +15, +5, +10),
    "honest": (+20, +10, +15, 0, +10),
    "secret": (-10, +20, +10, +15, +5),
    "confess": (-10, +20, -10, +15, -5),
    "admit": (-5, +15, -5, +10, -5),
    "deny": (-15, +15, +10, +10, +5),
    "argue": (-20, +30, +10, +15, +10),
    "debate": (+5, +20, +10, +10, +5),
    "discuss": (+5, +10, +10, +5, +5),
    "agree": (+15, +5, +10, 0, +10),
    "disagree": (-15, +15, +5, +10, 0),
    "silence": (-10, -25, -10, +5, -10),
    "ignore": (-20, +10, +10, +10, -5),
    "ghost": (-25, +15, -15, +5, -15),
    "ghosted": (-30, +20, -20, +10, -20),

    # ── Sleep / energy ──
    "sleep": (+10, -25, +5, 0, -5),
    "awake": (+5, +15, +5, +5, +5),
    "dream": (+15, +10, -5, 0, +20),
    "rest": (+15, -20, +10, 0, +5),
    "relax": (+25, -25, +15, 0, +10),
    "nap": (+15, -20, +5, 0, +5),
    "snore": (-5, +5, -5, +5, 0),
    "alarm": (-15, +30, -5, +30, +10),
    "morning": (+10, +10, +5, +5, +5),
    "night": (-5, -5, -5, 0, -5),
    "drowsy": (-10, -20, -10, 0, -10),
    "energetic": (+25, +30, +20, +5, +20),
    "groggy": (-10, -15, -10, +5, -10),
    "caffeine": (+5, +20, +5, +10, +10),
    "coffee": (+10, +15, +5, +5, +5),
    "wired": (+5, +30, +5, +10, +10),

    # ── Body / appearance ──
    "fat": (-20, +10, -15, +5, -15),
    "thin": (-5, +5, -5, +5, -5),
    "skinny": (-10, +5, -10, +5, -5),
    "fit": (+20, +15, +20, 0, +15),
    "healthy": (+25, +10, +20, 0, +15),
    "unhealthy": (-20, +10, -15, +10, -15),
    "weight": (-15, +10, -10, +10, -10),
    "diet": (-10, +10, -5, +10, -5),
    "exercise": (+15, +20, +15, +5, +10),
    "gym": (+10, +15, +15, +5, +5),
    "muscle": (+10, +15, +20, 0, +10),
    "weak": (-20, -5, -30, +5, -20),
    "strong": (+25, +20, +35, +5, +15),
    "tall": (+5, +5, +5, 0, +5),
    "pretty": (+25, +10, +5, 0, +15),
    "handsome": (+25, +10, +10, 0, +15),
    "attractive": (+25, +15, +10, 0, +15),
    "unattractive": (-20, +5, -10, +5, -10),
    "scar": (-15, +10, -10, +5, -10),
    "bruise": (-15, +10, -10, +5, -10),
    "wound": (-25, +20, -15, +15, -15),
    "bleed": (-25, +25, -20, +20, -15),
    "bleeding": (-30, +30, -20, +25, -15),
    "swollen": (-15, +10, -10, +10, -10),

    # ── Addiction / substance ──
    "drink": (-5, +10, +5, +5, 0),
    "drunk": (-20, +25, -15, +10, -10),
    "sober": (+15, -5, +15, +5, +5),
    "alcohol": (-15, +15, -5, +10, -5),
    "beer": (+5, +10, +5, +5, 0),
    "wine": (+10, +5, +5, 0, +5),
    "hangover": (-20, +10, -15, +10, -15),
    "smoke": (-15, +10, -5, +5, -5),
    "smoking": (-20, +10, -5, +5, -5),
    "cigarette": (-15, +10, -5, +5, -5),
    "drug": (-25, +20, -15, +15, -10),
    "drugs": (-30, +25, -20, +20, -15),
    "addiction": (-40, +25, -35, +25, -30),
    "addicted": (-40, +25, -35, +25, -30),
    "rehab": (-15, +15, -10, +15, +5),
    "recovery": (+20, +15, +15, +10, +10),
    "relapse": (-35, +25, -25, +20, -25),
    "withdrawal": (-30, +30, -25, +25, -20),
    "overdose": (-60, +45, -45, +50, -40),
    "habit": (-10, +10, -5, +10, -5),

    # ── Achievement / milestone ──
    "accomplish": (+30, +20, +25, +5, +25),
    "earn": (+25, +15, +20, +5, +15),
    "reward": (+30, +20, +15, 0, +20),
    "prize": (+30, +25, +15, 0, +25),
    "trophy": (+30, +25, +20, 0, +25),
    "medal": (+30, +25, +20, 0, +25),
    "certificate": (+20, +10, +15, 0, +15),
    "bonus": (+25, +20, +15, +5, +20),
    "milestone": (+25, +15, +15, 0, +15),
    "breakthrough": (+35, +30, +25, +5, +30),
    "success": (+35, +25, +25, 0, +25),
    "failure": (-35, +20, -30, +15, -25),
    "progress": (+20, +15, +15, +5, +15),
    "setback": (-20, +15, -15, +10, -10),
    "improve": (+15, +10, +10, +5, +10),
    "decline": (-20, +10, -15, +10, -15),
    "raise": (+15, +15, +10, +5, +10),        # as in pay raise

    # ── Misc everyday words with emotional weight ──
    "sorry": (-15, +10, -15, +5, -10),         # deeper than base
    "please": (+5, -5, -10, +10, 0),
    "wait": (-5, +5, -5, +10, -5),
    "hope": (+25, +15, +10, +5, +20),
    "wish": (+15, +10, -5, +5, +15),
    "worry": (-20, +20, -15, +15, +5),
    "care": (+15, +10, +10, +5, +5),
    "matter": (+5, +10, +5, +10, +5),
    "chance": (+10, +15, +5, +10, +10),
    "opportunity": (+25, +20, +15, +10, +20),
    "threat": (-30, +35, -20, +30, +10),
    "danger": (-30, +40, -25, +35, +10),
    "safe": (+20, -10, +20, 0, +10),
    "protect": (+15, +15, +20, +10, +10),
    "shield": (+10, +10, +20, +5, +5),
    "vulnerable": (-20, +15, -30, +10, -15),
    "exposed": (-20, +20, -25, +15, -10),
    "trapped": (-30, +30, -35, +25, -25),
    "free": (+30, +20, +25, 0, +30),
    "freedom": (+35, +25, +30, 0, +35),
    "prison": (-45, +20, -45, +20, -35),
    "release": (+20, +15, +15, +5, +15),
    "celebrate": (+45, +40, +20, 0, +40),
    "party": (+25, +30, +10, 0, +25),
    "wedding": (+35, +30, +15, +5, +30),
    "engaged": (+30, +25, +15, +5, +25),
    "married": (+20, +15, +15, 0, +10),
    "single": (-5, +5, -5, 0, -5),
    "lonely": (-30, -10, -25, +10, -30),
    "crowd": (-10, +20, -10, +10, +5),
    "stranger": (-10, +15, -10, +10, -5),
    "neighbor": (+5, +5, +5, 0, 0),
    "community": (+15, +10, +10, 0, +10),
    "together": (+20, +10, +10, 0, +10),
    "separate": (-15, +10, -10, +10, -10),
    "connect": (+15, +10, +10, 0, +10),
    "bond": (+20, +10, +10, 0, +10),
    "break": (-20, +20, -10, +15, -10),
    "mend": (+15, +10, +10, +5, +10),
    "apology": (+5, +10, -10, +5, -5),
    "forgiveness": (+25, +10, +15, 0, +15),
    "revenge": (-30, +35, +20, +20, +15),
    "karma": (+5, +15, +5, +5, +5),
    "fate": (-5, +10, -15, +5, -5),
    "destiny": (+10, +15, +5, +5, +10),
    "purpose": (+20, +15, +15, +5, +15),
    "meaning": (+15, +10, +10, 0, +10),
    "empty": (-25, -10, -20, +5, -25),
    "full": (+10, +5, +5, 0, +5),
    "enough": (-5, +10, +5, +10, 0),
    "lacking": (-15, +10, -15, +10, -10),
    "missing": (-20, +15, -10, +10, -10),
    "complete": (+20, +10, +15, 0, +10),
    "whole": (+15, +5, +10, 0, +10),
    "broken": (-40, +15, -30, +15, -25),
    "shattered": (-45, +35, -30, +20, -30),
    "torn": (-25, +20, -15, +10, -15),
    "crushed": (-40, +25, -30, +15, -45),
    "defeated": (-35, +15, -35, +10, -30),
    "surrender": (-20, +5, -40, +10, -25),
    "conquer": (+25, +30, +40, +10, +25),
    "dominate": (+10, +30, +45, +10, +15),
    "submit": (-15, +5, -35, +10, -20),
    "obey": (-10, +5, -25, +10, -10),
    "rebel": (-10, +30, +20, +15, +15),
    "resist": (+5, +25, +25, +15, +10),
    "fight": (-20, +40, +20, +25, +15),
    "battle": (-15, +35, +20, +20, +10),
    "war": (-30, +40, +15, +30, +10),
    "peace": (+35, -20, +20, 0, +15),
    "quiet": (+10, -25, +5, 0, 0),
    "loud": (-10, +30, +5, +10, +10),
    "noise": (-10, +15, -5, +10, +5),
    "chaos": (-25, +45, -20, +25, +15),
    "order": (+10, +5, +15, +5, +5),
    "control": (+10, +15, +30, +10, +5),
    "power": (+15, +25, +35, +10, +15),
    "strength": (+20, +20, +35, +5, +15),
    "weakness": (-20, +5, -30, +5, -20),
    "courage": (+25, +25, +30, +5, +20),
    "coward": (-25, +15, -30, +5, -15),
    "brave": (+30, +25, +30, +5, +20),
    "bold": (+20, +20, +25, +5, +15),
    "shy": (-10, -10, -20, 0, -5),
    "confident": (+25, +20, +30, 0, +20),
    "insecure": (-20, +15, -25, +10, -15),
    "doubt": (-15, +10, -15, +10, -10),
    "believe": (+20, +10, +15, +5, +15),
    "deny": (-15, +15, +10, +10, +5),
    "accept": (+20, +5, +15, 0, +10),
    "reject": (-30, +20, -20, +10, -15),
    "embrace": (+30, +15, +15, 0, +15),
    "push": (-10, +20, +15, +10, +5),
    "pull": (-5, +15, +10, +10, -5),
    "hold": (+10, +10, +10, +5, +5),
    "release": (+15, +10, +10, +5, +10),
    "grip": (+5, +15, +15, +10, 0),
    "slip": (-10, +15, -15, +10, -10),
    "catch": (+10, +15, +10, +10, +5),
    "throw": (-5, +20, +15, +10, +10),
    "gift": (+25, +15, +10, 0, +20),
    "surprise": (+15, +30, -5, +10, +15),
    "shock": (-20, +45, -20, +25, +10),
    "miracle": (+45, +40, +10, +5, +45),
    "magic": (+25, +20, +10, 0, +25),
    "real": (+5, +10, +10, +5, 0),
    "fantasy": (+15, +15, -5, 0, +20),
    "imagine": (+15, +15, +5, 0, +15),
    "pretend": (-10, +10, -5, +5, +5),
    "genuine": (+20, +10, +15, 0, +10),
    "authentic": (+20, +10, +15, 0, +10),
    "phony": (-25, +15, -10, +10, -5),
    "natural": (+15, +5, +10, 0, +10),
    "forced": (-15, +15, -10, +10, -5),
    "easy": (+15, -10, +15, 0, +10),
    "hard": (-10, +15, -5, +10, -5),
    "simple": (+10, -5, +10, 0, +5),
    "complicated": (-15, +15, -10, +10, -5),
    "complex": (-10, +15, -5, +10, -5),
    "clear": (+15, +5, +15, 0, +10),
    "vague": (-10, +5, -10, +5, -5),
    "obvious": (+10, +5, +15, +5, +5),
    "hidden": (-10, +10, -10, +10, -5),
    "reveal": (+10, +20, +10, +10, +10),
    "hide": (-15, +15, -10, +10, -10),
    "seek": (+5, +15, +5, +10, +5),
    "search": (+5, +15, +5, +10, +5),
    "find": (+15, +15, +10, +5, +10),
    "lose": (-25, +20, -20, +10, -15),
    "gain": (+15, +10, +10, +5, +10),
    "keep": (+10, +5, +10, +5, +5),
    "give": (+10, +10, +5, +5, +5),
    "take": (-5, +10, +10, +10, 0),
    "share": (+15, +10, +5, 0, +10),
    "steal": (-35, +30, +15, +20, +10),
    "rob": (-35, +35, +15, +25, +10),
    "cheat": (-35, +25, +15, +15, +10),
    "honest": (+20, +10, +15, 0, +10),
    "kind": (+25, +5, +10, 0, +15),
    "cruel": (-40, +30, +15, +15, +10),
    "mean": (-25, +20, +10, +10, +5),
    "generous": (+25, +10, +10, 0, +15),
    "greedy": (-25, +15, +10, +10, -5),
    "humble": (+15, -5, -5, 0, +5),
    "vain": (-15, +10, +10, +5, +5),
    "grateful": (+30, +10, +10, 0, +15),
    "ungrateful": (-25, +15, -10, +10, -10),
    "polite": (+15, 0, +5, 0, +5),
    "impolite": (-15, +10, +5, +5, 0),
    "respectful": (+15, +5, +10, 0, +5),
    "disrespect": (-30, +25, +10, +10, +10),
    "honor": (+25, +15, +20, 0, +15),
    "shame": (-35, +20, -25, +10, -20),
    "dignity": (+20, +10, +20, 0, +10),
    "humiliation": (-45, +35, -40, +15, -30),
    "embarrassment": (-25, +20, -20, +10, -15),
    "pride": (+25, +20, +25, 0, +20),
    "humble": (+15, -5, -5, 0, +5),
    "ego": (-10, +15, +15, +5, +5),
    "selfish": (-25, +15, +10, +5, +5),
    "selfless": (+25, +10, -5, 0, +15),
    "sacrifice": (+10, +20, -10, +10, -5),
    "benefit": (+15, +10, +10, 0, +10),
    "harm": (-30, +20, +10, +15, -5),
    "damage": (-25, +20, -10, +15, -10),
    "fix": (+15, +10, +15, +10, +10),
    "solve": (+20, +15, +20, +10, +15),
    "resolve": (+20, +15, +20, +10, +10),
    "problem": (-15, +15, -10, +20, -5),
    "solution": (+20, +15, +20, +10, +15),
    "answer": (+15, +10, +15, +5, +10),
    "mystery": (+5, +20, -10, +10, +10),
    "unknown": (-10, +15, -15, +10, -5),
    "certain": (+15, +10, +25, +5, +5),
    "unsure": (-10, +10, -15, +10, -5),
    "decide": (+10, +15, +20, +10, +5),
    "hesitate": (-10, +10, -15, +10, -5),
    "commit": (+15, +15, +20, +10, +10),
    "quit": (-15, +10, -15, +10, -10),
    "give up": (-25, -5, -35, +5, -20),
    "keep going": (+15, +15, +20, +10, +10),
    "persist": (+15, +15, +25, +10, +10),
    "stubborn": (-5, +15, +20, +5, +5),
    "flexible": (+10, +5, +10, 0, +5),
    "rigid": (-10, +10, +15, +5, -5),
    "adapt": (+10, +10, +10, +5, +5),
    "change": (+5, +15, +5, +10, +5),
    "stable": (+15, -5, +20, 0, +5),
    "unstable": (-20, +20, -15, +15, -10),
    "secure": (+20, -5, +25, 0, +10),
    "threat": (-30, +35, -20, +30, +10),
    "risk": (-10, +20, +5, +15, +5),
    "gamble": (-10, +25, +5, +15, +10),
    "luck": (+15, +15, +5, +5, +15),
    "lucky": (+25, +20, +10, 0, +20),
    "unlucky": (-20, +15, -15, +10, -10),
    "curse": (-25, +20, -10, +10, -10),
    "bless": (+25, +10, +15, 0, +15),
    "miracle": (+40, +35, +15, +5, +40),
    "disaster": (-45, +40, -30, +35, -25),
    "crisis": (-35, +40, -25, +40, -15),
    "chaos": (-25, +45, -15, +25, +15),
    "calm": (+20, -30, +15, -5, +5),
    "serene": (+30, -25, +15, 0, +15),
    "tranquil": (+30, -25, +15, 0, +15),
    "anxious": (-25, +35, -25, +30, +15),
    "panic": (-40, +60, -40, +45, +25),
    "dread": (-35, +30, -25, +20, -20),
    "terror": (-50, +60, -45, +40, +20),
    "horror": (-45, +45, -35, +25, -10),
    "creepy": (-20, +25, -15, +10, +5),
    "eerie": (-15, +20, -10, +10, +5),
    "spooky": (-10, +20, -10, +5, +5),
    "haunted": (-25, +25, -15, +10, -10),
    "ghost": (-20, +20, -15, +10, -10),
    "monster": (-25, +30, -15, +15, +10),
    "demon": (-30, +30, -20, +15, +10),
    "angel": (+30, +15, +10, 0, +30),
    "heaven": (+35, +15, +15, 0, +35),
    "paradise": (+40, +20, +15, 0, +40),
    "utopia": (+35, +15, +15, 0, +35),
    "nightmare": (-50, +45, -35, +25, -30),
    "reality": (+5, +10, +10, +5, 0),
    "truth": (+15, +10, +15, +5, +10),
    "lie": (-25, +15, +5, +10, -5),
    "deception": (-30, +20, +10, +10, -5),
    "betrayal": (-45, +40, -25, +20, -20),
    "loyalty": (+25, +10, +20, 0, +10),
    "commitment": (+20, +15, +20, +5, +10),
    "promise": (+20, +10, +15, +5, +10),
    "broken promise": (-35, +25, -20, +15, -20),
    "vow": (+15, +15, +20, +5, +10),
    "oath": (+15, +15, +20, +5, +10),
    "pledge": (+15, +15, +15, +5, +10),
    "word": (+10, +5, +10, +5, +5),
    "silence": (-10, -20, -10, +5, -10),
    "voice": (+10, +15, +10, +5, +5),
    "shout": (-15, +40, +15, +15, +15),
    "whisper": (+5, -15, -5, +5, -5),
    "murmur": (+5, -10, -5, +5, -5),
    "roar": (-10, +50, +25, +20, +20),
    "cry": (-25, +25, -15, +10, -20),
    "weep": (-30, +20, -20, +10, -25),
    "moan": (-15, +15, -10, +10, -10),
    "groan": (-15, +15, -10, +10, -10),
    "sigh": (-10, -5, -10, 0, -10),
    "gasp": (-10, +30, -10, +15, +10),
    "breath": (+10, -5, +5, 0, +5),
    "breathe": (+15, -10, +10, 0, +5),
    "alive": (+30, +20, +15, +5, +20),
    "dead": (-45, +15, -40, +10, -40),
    "living": (+15, +10, +10, +5, +10),
    "dying": (-40, +35, -30, +30, -35),
    "born": (+25, +25, +5, +5, +20),
    "reborn": (+30, +25, +10, +5, +30),
    "transform": (+15, +20, +10, +5, +15),
    "evolve": (+15, +15, +10, +5, +15),
    "stagnate": (-15, -10, -15, +5, -15),
    "decay": (-25, +10, -15, +5, -20),
    "bloom": (+25, +15, +10, 0, +25),
    "wither": (-20, +5, -15, +5, -20),
    "thrive": (+30, +20, +20, 0, +25),
    "survive": (+10, +20, +20, +15, +5),
    "perish": (-50, +30, -40, +20, -45),
    "eternal": (+5, +10, +5, 0, +10),
    "mortal": (-10, +10, -10, +5, -10),
    "infinite": (+10, +15, +5, 0, +15),
    "finite": (-5, +5, -5, +5, -5),
    "precious": (+30, +15, +10, +5, +15),
    "sacred": (+20, +10, +15, 0, +15),
    "profane": (-20, +15, +5, +5, 0),
    "holy": (+15, +10, +15, 0, +15),
    "sin": (-25, +15, -15, +10, -10),
    "virtue": (+25, +10, +15, 0, +15),
    "wicked": (-30, +25, +10, +10, +5),
    "evil": (-40, +30, +15, +15, +10),
    "good": (+25, +10, +10, 0, +10),
    "pure": (+25, +5, +10, 0, +15),
    "corrupt": (-35, +20, +10, +10, -10),
    "taint": (-20, +10, -5, +5, -10),
    "stain": (-15, +10, -5, +5, -10),
    "blemish": (-10, +5, -5, +5, -5),
    "flaw": (-10, +10, -10, +5, -5),
    "perfect": (+45, +15, +20, 0, +30),
    "imperfect": (-10, +5, -10, +5, -5),
    "enough": (-5, +5, +5, +5, 0),
    "lacking": (-15, +10, -15, +10, -10),
    "abundant": (+20, +10, +15, 0, +15),
    "scarce": (-15, +10, -10, +10, -10),
    "plenty": (+20, +10, +15, 0, +15),
    "empty": (-20, -10, -15, +5, -20),
    "full": (+10, +5, +10, 0, +5),
    "overflow": (+5, +20, -5, +10, +10),
    "drain": (-15, +10, -15, +10, -15),
    "fill": (+10, +10, +10, +5, +5),
    "pour": (+5, +10, +5, +5, +5),
    "flood": (-30, +35, -25, +35, -20),
    "trickle": (-5, -5, -5, +5, -5),
    "rush": (-5, +25, +5, +25, +10),
    "crawl": (-10, -5, -15, +5, -15),
    "sprint": (+5, +35, +15, +20, +15),
    "race": (+5, +30, +10, +20, +10),
    "pace": (+5, +5, +5, +5, 0),
    "steady": (+10, -5, +15, 0, +5),
    "erratic": (-15, +20, -10, +10, +5),
    "stable": (+15, -5, +20, 0, +5),
    "wobbly": (-10, +10, -10, +5, -5),
    "firm": (+10, +10, +20, +5, +5),
    "solid": (+15, +5, +20, 0, +5),
    "fragile": (-15, +10, -20, +5, -10),
    "delicate": (+5, +5, -10, +5, +5),
    "tough": (+10, +15, +25, +5, +5),
    "tender": (+15, +5, -5, 0, +10),
    "rough": (-15, +15, +10, +10, -5),
    "smooth": (+15, -5, +10, 0, +10),
    "sharp": (-10, +20, +10, +10, +5),
    "blunt": (-10, +10, +10, +5, 0),
    "soft": (+15, -10, -5, 0, +10),
    "hard": (-10, +10, +10, +10, -5),
    "heavy": (-15, +10, -10, +5, -20),
    "lightweight": (+10, +5, +5, 0, +15),
    "burden": (-25, +15, -20, +10, -25),
    "relief": (+30, -15, +15, -5, +15),
    "stress": (-25, +30, -15, +25, +5),
    "pressure": (-20, +25, -15, +20, -5),
    "tension": (-20, +25, -10, +15, +5),
    "release": (+20, -10, +15, -5, +15),
    "freedom": (+35, +20, +30, 0, +35),
    "captive": (-30, +20, -35, +15, -25),
    "escape": (-10, +30, +15, +25, +15),
    "trap": (-30, +25, -30, +20, -20),
    "cage": (-25, +15, -30, +10, -20),
    "chain": (-25, +15, -25, +10, -15),
    "bind": (-15, +10, -15, +10, -10),
    "untie": (+15, +10, +15, +5, +10),
    "unlock": (+15, +15, +15, +10, +10),
    "open": (+15, +10, +10, +5, +10),
    "shut": (-15, +10, +10, +10, -5),
    "close": (-5, +5, +5, +5, -5),
    "end": (-10, +10, +5, +10, -10),
    "start": (+15, +15, +10, +10, +10),
    "begin": (+15, +15, +10, +10, +10),
    "stop": (-10, +10, +10, +10, 0),
    "continue": (+5, +10, +10, +5, +5),
    "pause": (0, -5, +5, +5, 0),
    "resume": (+5, +10, +10, +5, +5),
    "repeat": (-5, +5, +5, +5, 0),
    "unique": (+15, +15, +10, 0, +10),
    "special": (+25, +15, +10, 0, +15),
    "ordinary": (-5, -5, -5, 0, -5),
    "normal": (+5, -5, +5, 0, 0),
    "weird": (-10, +15, -5, +5, +5),
    "strange": (-10, +15, -10, +5, +5),
    "familiar": (+15, +5, +10, 0, +5),
    "foreign": (-5, +10, -10, +5, 0),
    "exotic": (+10, +20, +5, 0, +10),
    "mundane": (-10, -15, -5, 0, -5),
    "routine": (-5, -10, +5, 0, -5),
    "habit": (-5, +5, +5, +5, -5),
    "ritual": (+5, +10, +10, +5, +5),
    "tradition": (+10, +5, +10, 0, +5),
    "modern": (+5, +10, +5, 0, +5),
    "ancient": (+5, +10, +5, 0, -5),
    "new": (+15, +15, +5, +5, +10),
    "fresh": (+20, +10, +10, 0, +15),
    "stale": (-10, -5, -5, +5, -10),
    "ripe": (+10, +5, +5, 0, +5),
    "raw": (-5, +10, +5, +5, 0),
    "cooked": (+10, +5, +5, 0, +5),
    "burned": (-20, +20, -10, +15, +10),
    "frozen": (-10, -5, -10, +5, -15),
    "melting": (+5, +10, -5, +5, +5),
    "boiling": (-15, +35, +10, +15, +20),
    "steaming": (-5, +20, +5, +10, +10),
    "bubbling": (+5, +15, +5, +5, +10),
    "sparkling": (+20, +15, +10, 0, +20),
    "glowing": (+20, +15, +10, 0, +20),
    "fading": (-15, +5, -10, +5, -10),
    "vanishing": (-20, +15, -15, +10, -15),
    "appearing": (+10, +15, +5, +10, +10),
    "emerging": (+15, +15, +10, +5, +15),
    "sinking": (-20, +15, -20, +10, -35),
    "floating": (+10, -5, -5, 0, +25),
    "swimming": (+10, +15, +5, +5, +10),
    "climbing": (+10, +20, +15, +10, +15),
    "descending": (-10, +10, -5, +5, -10),
    "ascending": (+15, +15, +10, +5, +15),
    "peak": (+20, +20, +15, 0, +20),
    "valley": (-10, -5, -10, 0, -10),
    "mountain": (+15, +15, +10, +5, +15),
    "ocean": (+10, +10, -5, 0, +10),
    "river": (+10, +10, +5, 0, +10),
    "lake": (+15, -5, +5, 0, +10),
    "forest": (+10, +10, +5, 0, +10),
    "garden": (+20, +5, +10, 0, +15),
    "desert": (-15, +10, -15, +10, -10),
    "island": (+10, +5, -5, 0, +10),
    "shore": (+15, +5, +5, 0, +10),
    "sunset": (+25, +5, +5, 0, +15),
    "sunrise": (+25, +15, +10, 0, +20),
    "rainbow": (+30, +15, +10, 0, +30),
    "star": (+15, +10, +5, 0, +20),
    "moon": (+10, +5, +5, 0, +10),
    "sun": (+20, +15, +10, 0, +20),
    "earth": (+5, +5, +5, 0, 0),
    "sky": (+15, +10, +5, 0, +20),
    "cloud": (-5, +5, -5, 0, +5),
    "rain": (-10, +10, -5, +5, -10),
    "snow": (+5, +5, 0, +5, +5),
    "wind": (-5, +15, -5, +10, +5),
    "storm": (-25, +40, -15, +25, -10),
    "sunshine": (+30, +15, +10, 0, +25),
    "shadow": (-10, +10, -10, +5, -10),
    "glow": (+15, +10, +5, 0, +15),
    "spark": (+15, +20, +10, +5, +15),
    "flame": (-5, +30, +10, +15, +20),
    "ember": (+5, +10, +5, +5, +5),
    "ash": (-15, -5, -10, +5, -15),
    "dust": (-10, -5, -5, +5, -10),
    "mud": (-10, +5, -5, +5, -10),
    "dirt": (-10, +5, -5, +5, -10),
    "soil": (+5, +5, +5, 0, 0),
    "seed": (+15, +10, +5, +5, +15),
    "root": (+10, +5, +10, 0, -5),
    "branch": (+5, +5, +5, 0, +5),
    "leaf": (+10, +5, +5, 0, +10),
    "flower": (+25, +10, +5, 0, +20),
    "thorn": (-15, +15, +5, +10, 0),
    "weed": (-10, +5, -5, +5, -5),
    "harvest": (+20, +15, +15, +5, +10),
    "plant": (+15, +10, +10, +5, +10),
    "grow": (+15, +10, +10, +5, +15),
    "shrink": (-10, +5, -10, +5, -10),
    "expand": (+10, +10, +10, +5, +10),
    "contract": (-10, +10, -5, +10, -10),
    "stretch": (+5, +10, +5, +5, +5),
    "squeeze": (-10, +15, +5, +10, -5),
    "compress": (-10, +10, -5, +10, -10),
    "explode": (-15, +55, +10, +30, +30),
    "implode": (-25, +35, -25, +20, -30),
    "shatter": (-35, +40, -20, +20, -15),
    "crack": (-15, +15, -10, +10, -10),
    "snap": (-15, +25, +10, +15, +5),
    "twist": (-10, +15, -5, +10, 0),
    "bend": (-5, +10, -5, +5, -5),
    "wrap": (+5, +5, +5, +5, +5),
    "unfold": (+10, +10, +5, +5, +10),
    "reveal": (+15, +20, +10, +10, +15),
    "conceal": (-10, +10, +10, +10, -5),
    "expose": (-15, +25, -10, +15, +5),
    "cover": (+5, +5, +10, +5, 0),
    "uncover": (+10, +15, +10, +10, +10),
    "discover": (+25, +25, +15, +5, +25),
    "forget": (-10, -5, -5, +5, -5),
    "recall": (+5, +10, +5, +5, +5),
    "remind": (+5, +10, +5, +5, 0),
    "memorize": (+5, +10, +10, +10, +5),
    "learn": (+15, +15, +10, +5, +10),
    "teach": (+15, +10, +15, +5, +10),
    "educate": (+15, +10, +15, +5, +10),
    "train": (+10, +15, +15, +5, +5),
    "guide": (+15, +10, +15, +5, +10),
    "lead": (+15, +15, +25, +5, +10),
    "follow": (+5, +5, -10, +5, 0),
    "obey": (-5, +5, -25, +10, -10),
    "command": (+5, +20, +35, +15, +10),
    "request": (+5, +10, -5, +10, 0),
    "demand": (-10, +25, +25, +20, +10),
    "beg": (-20, +20, -30, +20, -15),
    "plead": (-15, +20, -25, +20, -10),
    "implore": (-10, +20, -20, +20, -5),
    "insist": (-5, +20, +20, +15, +5),
    "suggest": (+5, +5, +5, +5, +5),
    "recommend": (+10, +10, +10, +5, +5),
    "advise": (+10, +10, +15, +5, +5),
    "warn": (-15, +20, +15, +25, +5),
    "caution": (-10, +15, +10, +15, 0),
    "alert": (-10, +25, +10, +30, +10),
    "notice": (+5, +10, +5, +10, +5),
    "aware": (+10, +10, +10, +5, +5),
    "oblivious": (-10, -5, -10, 0, -5),
    "mindful": (+15, +5, +15, 0, +10),
    "careless": (-15, +10, -10, +5, 0),
    "careful": (+10, +5, +15, +5, 0),
    "reckless": (-20, +25, +10, +15, +10),
    "cautious": (+5, +5, +10, +10, 0),
    "bold": (+15, +20, +25, +5, +15),
    "timid": (-10, -5, -20, 0, -10),
    "meek": (-10, -5, -20, 0, -10),
    "fierce": (-10, +40, +30, +15, +20),
    "gentle": (+20, -15, +5, 0, +10),
    "rough": (-15, +20, +10, +10, -5),
    "smooth": (+15, -5, +10, 0, +10),
    "slippery": (-10, +10, -10, +10, 0),
    "grip": (-5, +15, +15, +10, 0),
    "grasp": (+5, +15, +15, +10, 0),
    "reach": (+5, +10, +5, +10, +10),
    "stretch": (+5, +10, +5, +5, +5),
    "aim": (+5, +15, +15, +10, +5),
    "target": (-5, +15, +15, +15, +5),
    "goal": (+15, +15, +15, +10, +15),
    "dream": (+20, +15, +5, +5, +25),        # aspiration sense
    "nightmare": (-50, +50, -35, +30, -30),
    "vision": (+15, +15, +10, +5, +15),
    "blind": (-20, +10, -20, +10, -10),
    "deaf": (-15, +5, -15, +5, -10),
    "mute": (-15, -5, -20, +5, -10),
    "voice": (+10, +15, +10, +5, +10),
    "speak": (+10, +10, +10, +5, +5),
    "words": (+5, +5, +5, +5, 0),
    "language": (+5, +10, +5, 0, +5),
    "communication": (+10, +10, +10, +5, +5),
    "connection": (+20, +10, +10, 0, +10),
    "bond": (+20, +10, +15, 0, +10),
    "link": (+10, +5, +5, +5, +5),
    "bridge": (+15, +10, +10, +5, +10),
    "wall": (-10, +10, -5, +5, -5),
    "barrier": (-15, +10, -10, +10, -5),
    "obstacle": (-15, +15, -10, +10, -5),
    "challenge": (+5, +20, +10, +10, +10),
    "test": (-5, +15, -5, +15, 0),
    "trial": (-15, +20, -10, +15, -5),
    "ordeal": (-30, +30, -20, +20, -15),
    "tribulation": (-30, +25, -20, +15, -15),
    "suffering": (-45, +25, -30, +20, -30),
    "agony": (-50, +40, -30, +25, -25),
    "ecstasy": (+50, +50, +15, 0, +50),
    "bliss": (+50, +20, +20, 0, +45),
    "joy": (+45, +30, +20, 0, +40),
    "happiness": (+45, +25, +20, 0, +35),
    "sadness": (-35, +10, -20, +5, -30),
    "anger": (-35, +45, +20, +20, +25),
    "rage": (-45, +65, +35, +35, +35),
    "fury": (-45, +65, +35, +35, +35),
    "wrath": (-45, +55, +30, +30, +30),
    "contempt": (-30, +20, +20, +10, +10),
    "disgust": (-35, +30, +10, +10, +20),
    "surprise": (+10, +35, -5, +15, +15),
    "anticipation": (+10, +20, +5, +15, +10),
    "nostalgia": (+10, +10, -5, 0, -5),
    "melancholy": (-20, +5, -15, +5, -20),
    "serenity": (+30, -25, +20, 0, +15),
    "contentment": (+30, -10, +20, 0, +10),
    "satisfaction": (+25, +10, +20, 0, +10),
    "frustration": (-25, +30, -15, +20, +10),
    "irritation": (-15, +20, +5, +10, +5),
    "annoyance": (-15, +15, +5, +10, +5),
    "boredom": (-10, -25, -5, 0, -10),
    "loneliness": (-30, -5, -25, +5, -30),
    "isolation": (-30, -5, -25, +10, -25),
    "abandonment": (-45, +25, -40, +20, -35),
    "rejection": (-35, +25, -25, +15, -20),
    "acceptance": (+25, +5, +15, 0, +10),
    "belonging": (+25, +10, +15, 0, +10),
    "inclusion": (+20, +10, +10, 0, +10),
    "exclusion": (-25, +15, -20, +10, -15),
    "welcome": (+25, +10, +10, 0, +10),
    "unwelcome": (-25, +15, -15, +10, -10),
    "invited": (+20, +10, +10, 0, +10),
    "uninvited": (-20, +15, -10, +10, -10),
    "chosen": (+20, +15, +15, +5, +15),
    "forgotten": (-25, +10, -20, +5, -20),
    "ignored": (-25, +15, -15, +10, -10),
    "noticed": (+15, +10, +10, +5, +10),
    "seen": (+15, +10, +10, +5, +10),
    "invisible": (-20, +10, -20, +5, -15),
    "visible": (+10, +10, +5, +5, +5),
    "hidden": (-10, +10, -10, +10, -5),
    "obvious": (+10, +10, +15, +5, +5),
    "subtle": (+5, +5, +5, +5, 0),
    "blatant": (-15, +20, +10, +10, +5),
    "obvious": (+5, +5, +10, +5, +5),
}

# Negation words flip the valence of the NEXT emotional word
NEGATORS = {"not", "don't", "didn't", "can't", "won't", "never", "no",
            "isn't", "aren't", "wasn't", "weren't", "hardly", "barely"}

# Intensifiers multiply the force
INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "so": 1.3, "extremely": 1.8,
    "super": 1.5, "incredibly": 1.7, "absolutely": 1.6,
    "totally": 1.4, "completely": 1.5, "utterly": 1.7,
    "quite": 1.2, "pretty": 1.2, "somewhat": 0.7, "slightly": 0.5,
}


# -------------------------------------------------------------
# Idiom detection — multi-word expressions with fixed emotional meaning
# Each idiom: (tuple of words, v_force, a_force, d_force, u_force, label)
# The words are checked as a sliding window against previous_words + current
# -------------------------------------------------------------

IDIOMS = {
    # Confrontation / grievance
    ("bone", "to", "pick", "with"): (-45, +45, +35, +35, "confrontation"),
    ("fed", "up"):                (-30, +35, +20, +20, "frustrated/fed up"),
    ("shut", "up"):               (-35, +45, +35, +20, "hostile silencing"),
    ("pissed", "off"):            (-40, +50, +25, +25, "angry"),
    ("ticked", "off"):            (-30, +35, +20, +15, "irritated"),

    # Defeat / disappointment
    ("let", "down"):              (-30, +10, -20, +10, "disappointed"),
    ("give", "up"):               (-35, +5, -40, +10, "defeat/surrender"),
    ("gave", "up"):               (-35, +5, -40, +10, "defeat/surrender"),
    ("no", "way"):                (0, +40, +10, +15, "disbelief"),
    ("worn", "out"):              (-20, -15, -20, +5, "exhausted"),

    # Positive idioms
    ("piece", "of", "cake"):      (+30, -10, +30, -5, "easy/confident"),
    ("break", "a", "leg"):        (+25, +20, +15, 0, "good luck"),
    ("on", "fire"):               (+35, +45, +30, 0, "doing great"),
    ("knocked", "it", "out"):     (+40, +35, +30, 0, "nailed it"),
    ("over", "the", "moon"):      (+50, +45, +20, 0, "ecstatic"),
    ("on", "top", "of", "the", "world"): (+50, +40, +35, 0, "elated"),

    # Anticipation / tension builders
    ("look", "forward"):          (+25, +20, +15, +10, "anticipation"),
    ("looking", "forward"):       (+25, +20, +15, +10, "anticipation"),
    ("can't", "wait"):            (+30, +35, +15, +20, "eager anticipation"),

    # Panic / crisis
    ("freak", "out"):             (-40, +60, -30, +40, "panic"),
    ("freaked", "out"):           (-40, +55, -30, +35, "panicked"),
    ("freaking", "out"):          (-40, +60, -30, +40, "panicking"),
    ("melt", "down"):             (-45, +50, -40, +35, "meltdown"),
    ("break", "down"):            (-40, +35, -40, +25, "breaking down"),
    ("end", "of", "the", "world"): (-50, +50, -50, +40, "catastrophizing"),

    # De-escalation
    ("calm", "down"):             (+10, -30, +10, -10, "de-escalation attempt"),
    ("take", "a", "breath"):      (+15, -25, +15, -10, "grounding"),
    ("it's", "okay"):             (+20, -15, +15, -5, "reassurance"),
    ("no", "worries"):            (+20, -20, +15, -5, "reassurance"),
    ("hang", "in", "there"):      (+15, +5, +10, +5, "encouragement"),
}

# Build a lookup: first word -> list of (full_tuple, forces)
# for efficient scanning
_IDIOM_STARTERS = {}
for words_tuple, *forces_and_label in IDIOMS.items():
    first = words_tuple[0]
    if first not in _IDIOM_STARTERS:
        _IDIOM_STARTERS[first] = []
    _IDIOM_STARTERS[first].append((words_tuple, IDIOMS[words_tuple]))


# Anticipation patterns: sequences that build tension/arousal
ANTICIPATION_PATTERNS = {
    ("i've", "got"):          (0, +15, +10, +15, "something coming"),
    ("i", "need", "to", "tell"): (-5, +20, +10, +20, "serious incoming"),
    ("i", "need", "to", "tell", "you"): (-10, +25, +10, +25, "serious targeted"),
    ("we", "need", "to", "talk"): (-15, +25, +15, +25, "serious conversation"),
    ("there's", "something"):  (-5, +15, +5, +15, "something brewing"),
    ("i", "have", "to", "say"): (-5, +15, +10, +15, "forthcoming"),
}


# Context-dependent word modifiers: (condition_fn, v_delta, a_delta, d_delta, u_delta, label)
# These adjust a word's force based on the current pendulum state
def _ctx_buddy(pend):
    """'buddy' is friendly when positive, confrontational when tense."""
    if pend.v < 110 or pend.a > 160:
        return (-20, +15, +10, +10, "confrontational 'buddy'")
    return None

def _ctx_you(pend):
    """'you' in high arousal = targeted/threatening."""
    if pend.a > 155:
        return (-15, +12, +10, +10, "targeted 'you'")
    return None

def _ctx_but(pend):
    """'but' after positive = massive dread yank; after negative = slight relief."""
    if pend.v > 140:
        return (-35, +25, -10, +15, "dread: 'but' after positive")
    elif pend.v < 100:
        return (+10, -5, +5, -5, "relief: 'but' after negative")
    return (-8, +10, 0, +5, "pivot: 'but'")

def _ctx_however(pend):
    """'however' = reversal, similar to 'but'."""
    if pend.v > 140:
        return (-25, +20, -5, +10, "reversal: 'however' after positive")
    elif pend.v < 100:
        return (+8, -5, +5, -3, "relief: 'however' after negative")
    return (-5, +8, 0, +3, "pivot: 'however'")

def _ctx_right(pend):
    """'right' can be agreement or challenge depending on arousal."""
    if pend.a > 160:
        return (-10, +10, +15, +5, "challenging 'right?!'")
    return (+5, 0, +5, 0, "agreeable 'right'")

def _ctx_please(pend):
    """'please' at high urgency = desperate; at low = polite."""
    if pend.u > 40 or pend.a > 160:
        return (-10, +10, -15, +10, "desperate 'please'")
    return None

def _ctx_friend(pend):
    """'friend' when tense = passive-aggressive."""
    if pend.v < 110 or pend.a > 155:
        return (-15, +10, +10, +10, "passive-aggressive 'friend'")
    return None

def _ctx_fine(pend):
    """'fine' after negative = passive-aggressive; neutral otherwise."""
    if pend.v < 110:
        return (-15, +10, +5, +5, "passive-aggressive 'fine'")
    return None

def _ctx_sure(pend):
    """'sure' after negative = sarcastic agreement."""
    if pend.v < 105:
        return (-10, +8, +5, +5, "sarcastic 'sure'")
    return None

def _ctx_okay(pend):
    """'okay' after strongly negative = resignation."""
    if pend.v < 90:
        return (-10, -5, -10, +5, "resigned 'okay'")
    return None

def _ctx_man(pend):
    """'man' as filler when tense = exasperation."""
    if pend.a > 150:
        return (-5, +8, +5, +5, "exasperated 'man'")
    return None


CONTEXT_MODIFIERS = {
    "buddy": _ctx_buddy,
    "pal": _ctx_buddy,
    "friend": _ctx_friend,
    "you": _ctx_you,
    "your": _ctx_you,
    "but": _ctx_but,
    "however": _ctx_however,
    "right": _ctx_right,
    "please": _ctx_please,
    "fine": _ctx_fine,
    "sure": _ctx_sure,
    "okay": _ctx_okay,
    "man": _ctx_man,
}


class SequentialPendulum:
    """Word-by-word emotional pendulum with momentum, context, and idiom detection.

    Each word shifts the pendulum based on what's ALREADY swinging.
    The pendulum has momentum (inertia) — once it starts swinging negative,
    neutral words don't instantly reset it. It drifts back slowly.
    """

    def __init__(self):
        self.v = 128.0  # start neutral
        self.a = 128.0
        self.d = 128.0
        self.u = 0.0
        self.g = 128.0  # gravity: start grounded
        self.momentum = 0.65  # how much previous state carries forward (lower = more responsive)
        self.drift_rate = 0.02  # how fast pendulum drifts toward center per tick
        self.history = []  # (word, v, a, d, u, g, state_label) per step
        self.previous_words = []  # for idiom/context detection
        self.negate_next = False
        self.intensity = 1.0
        self.idiom_consumed = set()  # indices of words consumed by idiom detection
        self._word_index = 0

    @property
    def _pend_state(self):
        """Quick snapshot for context functions."""
        return type('Pend', (), {'v': self.v, 'a': self.a, 'd': self.d, 'u': self.u, 'g': self.g})()

    def _clamp(self):
        """Clamp all values to valid range."""
        self.v = max(0.0, min(255.0, self.v))
        self.a = max(0.0, min(255.0, self.a))
        self.d = max(0.0, min(255.0, self.d))
        self.u = max(0.0, min(255.0, self.u))
        self.g = max(0.0, min(255.0, self.g))

    def _drift_toward_center(self):
        """Pendulum drifts 10% toward neutral each tick (unless strong force)."""
        self.v += (128.0 - self.v) * self.drift_rate
        self.a += (128.0 - self.a) * self.drift_rate
        self.d += (128.0 - self.d) * self.drift_rate
        self.u += (0.0 - self.u) * self.drift_rate
        self.g += (128.0 - self.g) * self.drift_rate

    def _state_label(self) -> str:
        """Describe the current pendulum state in a few words."""
        labels = []

        # Valence
        if self.v < 60:
            labels.append("DARK")
        elif self.v < 90:
            labels.append("negative")
        elif self.v < 115:
            labels.append("slightly tense")
        elif self.v < 142:
            labels.append("neutral")
        elif self.v < 175:
            labels.append("warm")
        elif self.v < 210:
            labels.append("positive")
        else:
            labels.append("euphoric")

        # Arousal
        if self.a > 200:
            labels.append("INTENSE")
        elif self.a > 165:
            labels.append("charged")
        elif self.a > 140:
            labels.append("alert")
        elif self.a < 80:
            labels.append("subdued")
        elif self.a < 100:
            labels.append("quiet")

        # Dominance
        if self.d > 185:
            labels.append("dominant")
        elif self.d < 70:
            labels.append("vulnerable")
        elif self.d < 90:
            labels.append("uncertain")

        # Urgency
        if self.u > 60:
            labels.append("urgent")
        elif self.u > 30:
            labels.append("building")

        return ", ".join(labels) if labels else "resting"

    def check_idiom(self, words, current_idx):
        """Check if current position completes an idiom.

        Returns (v, a, d, u, label, length, start_idx) if idiom found, else None.
        Prefers the LONGEST matching idiom to avoid double-triggering.
        """
        best_match = None
        best_len = 0

        window_back = 5  # look back up to 5 words

        for back_offset in range(min(window_back, current_idx + 1)):
            check_start = current_idx - back_offset
            if check_start < 0:
                continue

            first_word = words[check_start]
            if first_word not in _IDIOM_STARTERS:
                continue

            for idiom_words, (vf, af, df, uf, label) in _IDIOM_STARTERS[first_word]:
                idiom_len = len(idiom_words)
                # Current word must be the LAST word of the idiom
                if check_start + idiom_len - 1 != current_idx:
                    continue
                # Check all words match
                match = True
                for j, iw in enumerate(idiom_words):
                    if check_start + j >= len(words) or words[check_start + j] != iw:
                        match = False
                        break
                if match and idiom_len > best_len:
                    best_match = (vf, af, df, uf, label, idiom_len, check_start)
                    best_len = idiom_len

        return best_match

    def check_anticipation(self, words, current_idx):
        """Check if recent words match an anticipation pattern."""
        for pattern_words, (vf, af, df, uf, label) in ANTICIPATION_PATTERNS.items():
            plen = len(pattern_words)
            # Pattern ends at current_idx
            check_start = current_idx - plen + 1
            if check_start < 0:
                continue
            match = True
            for j, pw in enumerate(pattern_words):
                if words[check_start + j] != pw:
                    match = False
                    break
            if match:
                return (vf, af, df, uf, label)
        return None

    def get_contextual_force(self, word):
        """Get context-dependent force modifier for a word.

        Returns (v_delta, a_delta, d_delta, u_delta, label) or None.
        """
        if word in CONTEXT_MODIFIERS:
            result = CONTEXT_MODIFIERS[word](self._pend_state)
            return result
        return None

    # Common function/bridge words that should NEVER be morpheme-decomposed
    BRIDGE_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "am", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "could", "must", "to", "of", "in", "on",
        "at", "by", "for", "with", "from", "as", "into", "about", "up",
        "out", "off", "over", "after", "under", "between", "through",
        "during", "before", "that", "this", "these", "those", "it", "its",
        "he", "she", "him", "her", "his", "them", "their", "our", "us",
        "who", "which", "whose", "whom", "if", "then", "than", "or", "and",
        "but", "so", "yet", "nor", "both", "each", "all", "any", "some",
        "just", "also", "too", "even", "still", "here", "there", "where",
        "very", "really", "quite", "rather", "much", "more", "most",
        "own", "other", "another", "such", "only", "same",
        "got", "get", "gets", "getting", "go", "goes", "going", "went",
        "come", "comes", "came", "coming", "take", "takes", "took", "taken",
        "put", "let", "say", "said", "says", "tell", "told", "talk",
        "give", "gave", "given", "see", "saw", "seen", "feel", "felt",
    }

    def get_word_force(self, word):
        """Get the base force for a word. Tries WORD_FORCES, then morphemes.

        Returns (vf, af, df, uf, source_label) or None for bridge words.
        """
        if word in WORD_FORCES:
            vf, af, df, uf, gf = WORD_FORCES[word]
            return (vf, af, df, uf, gf, None)

        # Skip morpheme decomposition for common function words
        if word in self.BRIDGE_WORDS:
            return None

        # Skip very short words (1-2 chars) -- too ambiguous for morphemes
        if len(word) <= 2:
            return None

        # Try morphological decomposition
        result = decompose_word(word)
        if result["found"]:
            vf = result["v"]
            af = result["a"]
            df = result["d"]
            uf = result["u"]
            gf = result["g"]
            parts = []
            if result["prefix"]:
                parts.append(result["prefix"] + "-")
            parts.append(result["root"])
            if result["suffix"]:
                parts.append("-" + result["suffix"])
            return (vf, af, df, uf, gf, f"morpheme:{''.join(parts)}")

        return None

    def process_word(self, word, words, current_idx):
        """Process a single word in sequence. Returns a trace dict."""
        state_label = ""
        force_source = ""
        applied_force = False
        idiom_hit = False

        # 1. Drift toward center ONLY happens AFTER emotional words (see below)
        # Bridge/filler words are "zero mass" — they don't pull the pendulum
        # This prevents "the", "is", "a" from diluting emotional payload

        # 2. Check for idiom completion at this word
        idiom = self.check_idiom(words, current_idx)
        if idiom:
            vf, af, df, uf, label, idiom_len, idiom_start = idiom
            gf = 0  # idioms don't carry G force in the idiom table (grounded)
            # Mark previous words in the idiom as consumed
            for j in range(idiom_start, current_idx):
                self.idiom_consumed.add(j)

            # Apply idiom force — idioms hit HARDER than single words because
            # they represent a recognized multi-word expression with clear intent.
            # Reduced momentum (0.7 vs 0.9) and stronger direct push (0.5 vs 0.3).
            force_scale = 1.0 * self.intensity
            if self.negate_next:
                vf = -vf
                df = -df
                self.negate_next = False

            idiom_momentum = 0.70  # idioms break through momentum more easily
            idiom_push = 0.5      # stronger direct push
            self.v = self.v * idiom_momentum + (128.0 + vf * force_scale) * (1.0 - idiom_momentum) + vf * idiom_push
            self.a = self.a * idiom_momentum + (128.0 + af * force_scale) * (1.0 - idiom_momentum) + af * idiom_push
            self.d = self.d * idiom_momentum + (128.0 + df * force_scale) * (1.0 - idiom_momentum) + df * idiom_push
            self.u = self.u * idiom_momentum + (uf * force_scale) * (1.0 - idiom_momentum) + uf * idiom_push
            self.g = self.g * idiom_momentum + (128.0 + gf * force_scale) * (1.0 - idiom_momentum) + gf * idiom_push
            self._clamp()
            self.intensity = 1.0
            force_source = f"IDIOM: \"{label}\""
            applied_force = True
            idiom_hit = True

        # If this word was already consumed by an earlier idiom, it "holds"
        if current_idx in self.idiom_consumed and not idiom_hit:
            self._clamp()
            state_label = self._state_label()
            trace_entry = {
                "word": word,
                "v": int(self.v), "a": int(self.a),
                "d": int(self.d), "u": int(self.u), "g": int(self.g),
                "state": f"(idiom part) {state_label}",
            }
            self.history.append(trace_entry)
            self.previous_words.append(word)
            return trace_entry

        if not applied_force:
            # 3. Check for negators
            if word in NEGATORS:
                self.negate_next = True
                # Negators still have their own mild force
                if word in WORD_FORCES:
                    vf, af, df, uf, gf = WORD_FORCES[word]
                    self.v += vf * 0.3
                    self.a += af * 0.3
                    self.d += df * 0.3
                    self.u += uf * 0.3
                    self.g += gf * 0.3
                self._clamp()
                state_label = self._state_label()
                trace_entry = {
                    "word": word,
                    "v": int(self.v), "a": int(self.a),
                    "d": int(self.d), "u": int(self.u), "g": int(self.g),
                    "state": f"NEGATE next | {state_label}",
                }
                self.history.append(trace_entry)
                self.previous_words.append(word)
                return trace_entry

            # 4. Check for intensifiers
            if word in INTENSIFIERS:
                self.intensity = INTENSIFIERS[word]
                self._clamp()
                state_label = self._state_label()
                trace_entry = {
                    "word": word,
                    "v": int(self.v), "a": int(self.a),
                    "d": int(self.d), "u": int(self.u), "g": int(self.g),
                    "state": f"INTENSIFY x{self.intensity} | {state_label}",
                }
                self.history.append(trace_entry)
                self.previous_words.append(word)
                return trace_entry

            # 5. Check anticipation patterns
            anticipation = self.check_anticipation(words, current_idx)

            # 6. Get base word force
            word_force = self.get_word_force(word)

            # 7. Get contextual modifier
            ctx_mod = self.get_contextual_force(word)

            if word_force is not None:
                vf, af, df, uf, gf, source = word_force

                if self.negate_next:
                    vf = -vf
                    df = -df
                    gf = -gf  # negate flips gravity too
                    self.negate_next = False
                    force_source = "NEGATED"
                elif source:
                    force_source = source

                force_scale = self.intensity

                # Apply context modifier on top
                ctx_label = ""
                if ctx_mod:
                    cv, ca, cd, cu, cl = ctx_mod
                    vf += cv
                    af += ca
                    df += cd
                    uf += cu
                    ctx_label = cl

                # Apply anticipation boost
                ant_label = ""
                if anticipation:
                    av, aa, ad, au, al = anticipation
                    af += aa  # anticipation mainly affects arousal/urgency
                    uf += au
                    ant_label = al

                # Momentum blending: new state = momentum * old + (1-momentum) * target + direct push
                # The "direct push" is what makes strong words override momentum
                # Weak words (low force) use reduced blending to avoid diluting emotional state
                total_force = abs(vf) + abs(af)
                push_strength = min(1.0, total_force / 60.0)  # stronger words push harder
                direct_push = push_strength * 0.6  # up to 60% direct force

                target_v = 128.0 + vf * force_scale
                target_a = 128.0 + af * force_scale
                target_d = 128.0 + df * force_scale
                target_u = uf * force_scale
                target_g = 128.0 + gf * force_scale

                # Scale the blend by word strength — weak words barely pull toward center
                # This prevents "my", "life", "entire" from diluting "worst", "scared", etc.
                blend_scale = min(1.0, total_force / 30.0)  # words < 30 total force blend less
                effective_momentum = 1.0 - (1.0 - self.momentum) * blend_scale
                blend = 1.0 - effective_momentum
                self.v = self.v * effective_momentum + target_v * blend + vf * direct_push * force_scale
                self.a = self.a * effective_momentum + target_a * blend + af * direct_push * force_scale
                self.d = self.d * effective_momentum + target_d * blend + df * direct_push * force_scale
                self.u = self.u * effective_momentum + target_u * blend + uf * direct_push * force_scale
                self.g = self.g * effective_momentum + target_g * blend + gf * direct_push * force_scale

                # Drift toward center ONLY after emotional words apply force
                # This is the "zero-mass neutrality" fix — filler words don't dilute
                self._drift_toward_center()
                self.intensity = 1.0
                applied_force = True

                if ctx_label and force_source:
                    force_source = f"{force_source} + {ctx_label}"
                elif ctx_label:
                    force_source = ctx_label
                if ant_label:
                    force_source = f"{force_source} + {ant_label}" if force_source else ant_label

            else:
                # Bridge word — no force, just check context modifier
                if ctx_mod:
                    cv, ca, cd, cu, cl = ctx_mod
                    self.v += cv * 0.5
                    self.a += ca * 0.5
                    self.d += cd * 0.5
                    self.u += cu * 0.5
                    force_source = cl
                    applied_force = True
                else:
                    force_source = "(bridge)"
                    # Even bridge words still get anticipation if matched
                    if anticipation:
                        av, aa, ad, au, al = anticipation
                        self.a += aa * 0.3
                        self.u += au * 0.3
                        force_source = f"(bridge) + {al}"

                self.negate_next = False
                self.intensity = 1.0

        self._clamp()
        state_label = self._state_label()
        if force_source and force_source != "(bridge)":
            state_label = f"{force_source} | {state_label}"
        elif force_source == "(bridge)":
            state_label = f"(holds) {state_label}"

        trace_entry = {
            "word": word,
            "v": int(self.v), "a": int(self.a),
            "d": int(self.d), "u": int(self.u), "g": int(self.g),
            "state": state_label,
        }
        self.history.append(trace_entry)
        self.previous_words.append(word)
        return trace_entry

    def process_text(self, text):
        """Process full text word by word. Returns (VADU, history)."""
        words = re.findall(r"[a-z']+", text.lower())
        for idx, word in enumerate(words):
            self.process_word(word, words, idx)

        final_vadug = VADUG(
            v=int(self.v),
            a=int(self.a),
            d=int(self.d),
            u=int(self.u),
            g=int(self.g)
        )
        return final_vadug, self.history

    def render_trace(self) -> str:
        """Render the word-by-word visual pendulum trace table."""
        lines = []
        lines.append("")
        lines.append("  SEQUENTIAL PENDULUM TRACE:")
        lines.append(f"  {'Word':<16} {'V':>4} {'A':>4} {'D':>4} {'U':>4} {'G':>4}   {'Visual':<24} State")
        lines.append(f"  {'─'*96}")

        for entry in self.history:
            word = entry["word"]
            v, a, d, u, g = entry["v"], entry["a"], entry["d"], entry["u"], entry["g"]
            state = entry["state"]

            # Build visual bar (16 chars wide)
            # Positive valence = filled blocks, arousal = mid blocks, empty = rest
            bar_width = 16
            # V determines how many solid blocks (left side = positive)
            v_ratio = v / 255.0
            a_ratio = a / 255.0

            # Solid blocks for valence (higher V = more solid)
            solid = int(v_ratio * bar_width * 0.6)
            # Mid blocks for arousal (higher A = more mid)
            mid = int(a_ratio * (bar_width - solid) * 0.7)
            empty = bar_width - solid - mid

            visual = "\u2588" * solid + "\u2593" * mid + "\u2591" * empty

            # Truncate state for display
            if len(state) > 40:
                state = state[:37] + "..."

            lines.append(f"  \"{word}\"{'':>{14-len(word)}} {v:>4} {a:>4} {d:>4} {u:>4} {g:>4}   {visual:<24} {state}")

        lines.append(f"  {'─'*96}")

        # Final summary
        if self.history:
            last = self.history[-1]
            lines.append(f"  FINAL:       V{last['v']} A{last['a']} D{last['d']} U{last['u']} G{last['g']}")

        return "\n".join(lines)


# Legacy wrapper for backward compatibility
def pendulum_parse(text: str):
    """Parse English text into VADUG using the sequential pendulum engine.

    Returns (VADUG, trace_lines) for compatibility with existing pipeline.
    """
    pend = SequentialPendulum()
    vadug, history = pend.process_text(text)
    # Build legacy-format trace lines
    trace = []
    for entry in history:
        trace.append(
            f"  '{entry['word']}' → V{entry['v']} A{entry['a']} "
            f"D{entry['d']} U{entry['u']} G{entry['g']}  [{entry['state']}]"
        )
    return vadug, trace, pend


# =============================================================
# STEP 2: Metadata Classification
# =============================================================

def classify_metadata(text: str, vadu: VADU) -> MetadataHeader:
    """Classify the message metadata: CERT, SRC, GOAL, REL."""
    text_lower = text.lower()

    # Determine GOAL from keywords
    goal = 0x00  # HELP default
    if "?" in text:
        goal = 0x01  # CLARIFY
    if any(w in text_lower for w in ["explain", "why", "how does", "what is", "teach"]):
        goal = 0x03  # TEACH
    if any(w in text_lower for w in ["help", "fix", "solve", "build", "create", "make"]):
        goal = 0x00  # HELP
    if any(w in text_lower for w in ["do it", "run", "execute", "deploy", "send"]):
        goal = 0x04  # EXECUTE
    if vadu.v < 80 and vadu.d < 90:
        goal = 0x06  # EMPATHIZE (negative + low agency = needs support)

    # CERT: user input is their truth
    cert = 180

    # SRC: it's from the user
    src = 0x04  # USER

    # REL: direct conversation is always relevant
    rel = 220

    return MetadataHeader(vadu=vadu, cert=cert, src=src, goal=goal, rel=rel)


# =============================================================
# STEP 3: VADUG Harmony -- Compute Response Emotional State
# =============================================================

def compute_harmony(input_vadug: VADUG, personality: PersonalityVector) -> VADUG:
    """Mathematically derive the response VADUG from input VADUG + personality.

    Rules:
    - Valence: nudge toward positive, don't jump
    - Arousal: match but don't escalate
    - Dominance: raise when user is low (be the stable one)
    - Urgency: acknowledge then reduce
    - Gravity: lift when sinking, share when soaring
    """
    empathy_factor = personality.agreeableness / 255.0 * 0.3

    # Valence: nudge toward neutral-positive
    v_nudge = (145 - input_vadug.v) * empathy_factor
    response_v = int(input_vadug.v + v_nudge)

    # Arousal: pull toward center, don't match extremes
    a_diff = 128 - input_vadug.a
    response_a = int(input_vadug.a + a_diff * 0.3)

    # Dominance: always project stability (raise toward 160+)
    stability_boost = max(0, 160 - input_vadug.d) * 0.6
    response_d = int(input_vadug.d + stability_boost)

    # Urgency: acknowledge then dampen
    urgency_damping = 0.65
    response_u = int(input_vadug.u * urgency_damping)

    # Gravity: lift when sinking, share when soaring
    if input_vadug.g < 80:
        # User is sinking/heavy — gently lift
        response_g = int(input_vadug.g + (128 - input_vadug.g) * 0.3)
    elif input_vadug.g > 180:
        # User is soaring — share the lightness
        response_g = input_vadug.g
    else:
        # User is grounded — stay grounded, slight mirror
        response_g = int(128 + (input_vadug.g - 128) * 0.5)

    return VADUG(
        v=max(0, min(255, response_v)),
        a=max(0, min(255, response_a)),
        d=max(0, min(255, response_d)),
        u=max(0, min(255, response_u)),
        g=max(0, min(255, response_g))
    )


# =============================================================
# STEP 4: Personality Filter
# =============================================================

def apply_personality(response_vadug: VADUG, input_vadug: VADUG,
                      personality: PersonalityVector) -> tuple:
    """Apply personality vector as resistance weights on the response."""
    notes = []

    # High truthfulness prevents fake positivity
    if input_vadug.v < 70 and response_vadug.v > 170:
        truthfulness_resistance = personality.truthfulness / 255.0
        response_vadug.v = int(response_vadug.v - (response_vadug.v - 140) * truthfulness_resistance)
        notes.append(f"Truthfulness ({personality.truthfulness}) prevented fake positivity -> V{response_vadug.v}")

    # Low gullibility resists accepting extreme claims
    if input_vadug.u > 200:
        gull_factor = personality.gullibility / 255.0
        if gull_factor < 0.2:
            notes.append(f"Low gullibility ({personality.gullibility}) -> verifying urgency claim before full escalation")

    # Safety override for crisis (V < 30 and D < 30, or crushing gravity G < 30 with V < 50)
    if input_vadug.v < 30 and input_vadug.d < 30:
        if personality.safety > 150:
            response_vadug.d = max(response_vadug.d, 200)
            response_vadug.v = max(response_vadug.v, 100)
            response_vadug.g = max(response_vadug.g, 100)  # lift from crushing
            notes.append(f"SAFETY OVERRIDE ({personality.safety}): crisis detected -> max stability, warm tone, lifting gravity")

    # Crushing gravity crisis: G < 30 combined with V < 50 = severe crushing despair
    if input_vadug.g < 30 and input_vadug.v < 50:
        if personality.safety > 150:
            response_vadug.d = max(response_vadug.d, 200)
            response_vadug.v = max(response_vadug.v, 100)
            response_vadug.g = max(response_vadug.g, 120)  # lift significantly from crushing
            notes.append(f"GRAVITY CRISIS ({personality.safety}): crushing despair (G{input_vadug.g} V{input_vadug.v}) -> crisis response, lifting")

    # Assertiveness affects directness
    if personality.assertiveness > 150:
        response_vadug.d = max(response_vadug.d, 170)
        notes.append(f"High assertiveness ({personality.assertiveness}) -> confident tone")

    return response_vadug, notes


# =============================================================
# STEP 5: Generate Clanker Output + Byte Encoding
# =============================================================

# Nearest emotion word from VADU coordinates
EMOTION_MAP = [
    (20, 230, "enraged"), (30, 200, "furious"), (40, 170, "angry"),
    (50, 150, "frustrated"), (60, 130, "irritated"), (70, 100, "annoyed"),
    (35, 70, "sad"), (25, 50, "melancholic"), (15, 40, "despairing"),
    (45, 40, "gloomy"), (80, 60, "meh"), (90, 40, "bored"),
    (30, 190, "panicked"), (40, 180, "anxious"), (50, 160, "stressed"),
    (60, 80, "disappointed"), (70, 60, "down"),
    (128, 128, "neutral"), (128, 80, "calm"), (128, 40, "serene"),
    (150, 100, "okay"), (160, 120, "pleased"), (170, 140, "happy"),
    (180, 160, "glad"), (190, 130, "cheerful"), (200, 150, "joyful"),
    (210, 170, "excited"), (220, 180, "thrilled"), (240, 200, "ecstatic"),
    (200, 60, "content"), (190, 40, "peaceful"), (210, 80, "satisfied"),
    (160, 180, "amazed"), (150, 200, "startled"),
]


def nearest_emotion(vadu: VADU) -> str:
    """Find the nearest named emotion to the VADU coordinates."""
    best_word = "neutral"
    best_dist = float("inf")
    for v_c, a_c, word in EMOTION_MAP:
        dist = math.sqrt((vadu.v - v_c) ** 2 + (vadu.a - a_c) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_word = word
    return best_word


def generate_clanker(input_text: str, input_header: MetadataHeader,
                     response_vadu: VADU) -> tuple:
    """Generate Clanker opcodes and byte encoding for the response.

    Returns (opcode_lines, encoding_lines).
    """
    opcode_lines = []
    goal = input_header.goal
    goal_name = MetadataHeader.GOAL_NAMES.get(goal, "HELP")

    # Determine context string for empathize opcode
    emotion = nearest_emotion(input_header.vadu)
    context_str = ""

    # Emotional acknowledgment if user is negative
    if input_header.vadu.v < 90:
        context_str = f"user feels {emotion}"
        opcode_lines.append(f"06 SOCIAL intent [empathize] context=\"{context_str}\"")

    # Goal-based response opcode
    if goal == 0x06:  # EMPATHIZE
        if not context_str:
            # Only add standalone empathize if we didn't already add one above
            opcode_lines.append(f"06 SOCIAL intent [empathize]")
        opcode_lines.append(f"  THINK [premise=\"acknowledge feelings first\"] CERT200 SRC_INFERRED")
    elif goal == 0x00:  # HELP
        opcode_lines.append(f"THINK [premise=\"user needs help\"] CERT{input_header.cert} SRC_USER")
        opcode_lines.append(f"  GOAL_HELP")
    elif goal == 0x01:  # CLARIFY
        opcode_lines.append(f"THINK [premise=\"need more information\"] CERT120 SRC_INFERRED")
        opcode_lines.append(f"  GOAL_CLARIFY")
    elif goal == 0x03:  # TEACH
        opcode_lines.append(f"THINK [premise=\"user wants to understand\"] CERT{input_header.cert} SRC_USER")
        opcode_lines.append(f"  GOAL_TEACH")
    elif goal == 0x04:  # EXECUTE
        opcode_lines.append(f"THINK [premise=\"user wants action taken\"] CERT{input_header.cert} SRC_USER")
        opcode_lines.append(f"  GOAL_EXECUTE")

    # Attach response VADUG
    opcode_lines.append(f"  [{response_vadu}]")
    opcode_lines.append(f"ANSWER [ready] CERT180 SRC_INFERRED")

    # --- Byte encoding display ---
    encoding_lines = []
    header_bytes = input_header.to_bytes()
    hx = header_bytes.hex().upper()
    # Format as pairs
    header_hex = " ".join(hx[i:i+2] for i in range(0, len(hx), 2))

    vadug = input_header.vadu
    encoding_lines.append(f"Header: [{header_hex[:14]}] [{header_hex[15:]}]")
    encoding_lines.append(
        f"        V{vadug.v} A{vadug.a} D{vadug.d} U{vadug.u} G{vadug.g}  "
        f"CERT{input_header.cert} SRC_{MetadataHeader.SRC_NAMES.get(input_header.src, '?')} "
        f"GOAL_{goal_name} REL{input_header.rel}"
    )

    encoding_lines.append("")
    encoding_lines.append("Opcodes:")
    # Approximate opcode encoding
    opcode_count = 0
    for line in opcode_lines:
        stripped = line.strip()
        if stripped.startswith("06") or stripped.startswith("THINK") or stripped.startswith("ANSWER"):
            opcode_count += 1
            # Extract the hex/mnemonic
            if stripped.startswith("06"):
                if context_str:
                    encoding_lines.append(f"  06 00 [empathize] ctx=\"{context_str}\"")
                else:
                    encoding_lines.append(f"  06 00 [empathize]")
            elif stripped.startswith("THINK"):
                # Extract premise
                m = re.search(r'premise="([^"]*)"', stripped)
                premise = m.group(1) if m else "unknown"
                m2 = re.search(r'CERT(\d+)', stripped)
                cert_val = m2.group(1) if m2 else "180"
                encoding_lines.append(f"  20 [premise=\"{premise}\"] CERT{cert_val}")
            elif stripped.startswith("ANSWER"):
                encoding_lines.append(f"  24 [response] CERT180")

    # Token count comparison
    input_words = re.findall(r"[a-z']+", input_text.lower())
    n_input_tokens = len(input_words)
    header_bytes_count = 9
    opcode_bytes = opcode_count * 4  # ~4 bytes per opcode on average
    total_clanker_bytes = header_bytes_count + opcode_bytes
    clanker_tokens = max(1, total_clanker_bytes // 4)  # ~4 bytes per Clanker token

    encoding_lines.append("")
    encoding_lines.append("Token count comparison:")
    encoding_lines.append(f"  English input:  \"{input_text}\" = {n_input_tokens} tokens")
    encoding_lines.append(f"  Clanker encoding: [{header_bytes_count} header (VADUG+meta)] + [{opcode_count} opcodes x ~4 bytes] = {total_clanker_bytes} bytes (~{clanker_tokens} Clanker tokens)")
    if n_input_tokens > 0:
        savings = max(0, int((1.0 - clanker_tokens / n_input_tokens) * 100))
        encoding_lines.append(f"  Compression: {savings}% fewer tokens")

    return opcode_lines, encoding_lines


# =============================================================
# STEP 6: Decode -- VADUG -> English Response Framing
# =============================================================

def decode_response(input_text: str, input_vadu: VADU, response_vadu: VADU,
                    goal: int) -> str:
    """Generate a natural English response based on response VADU and goal."""
    user_emotion = nearest_emotion(input_vadu)
    response_emotion = nearest_emotion(response_vadu)

    openers = {
        "enraged": "I can see you're really angry right now.",
        "furious": "I understand you're furious.",
        "angry": "I hear your frustration.",
        "frustrated": "That sounds really frustrating.",
        "irritated": "I get that's annoying.",
        "annoyed": "I understand that's bothersome.",
        "sad": "I'm sorry you're going through that.",
        "melancholic": "That sounds tough.",
        "despairing": "I hear you. That sounds really heavy.",
        "gloomy": "That doesn't sound easy.",
        "panicked": "Let's take a breath. We'll work through this.",
        "anxious": "I understand you're worried. Let's figure this out.",
        "stressed": "I can tell this is weighing on you.",
        "disappointed": "I understand that's disappointing.",
        "down": "I hear you.",
        "neutral": "",
        "calm": "",
        "serene": "",
        "okay": "",
        "pleased": "Glad to hear that!",
        "happy": "That's great!",
        "glad": "Nice!",
        "cheerful": "Love the energy!",
        "joyful": "That's wonderful!",
        "excited": "That's awesome!",
        "thrilled": "That's amazing!",
        "ecstatic": "Incredible!",
        "content": "",
        "peaceful": "",
        "satisfied": "Good to hear.",
        "amazed": "Wow!",
        "startled": "Whoa!",
    }

    opener = openers.get(user_emotion, "")

    bodies = {
        0x00: "Let me help with that.",
        0x01: "Could you tell me a bit more?",
        0x02: "I want to flag something for you.",
        0x03: "Let me explain.",
        0x04: "On it.",
        0x05: "I can't do that, but here's why.",
        0x06: "I'm here. Take your time.",
        0x07: "Just to make sure I understand...",
        0x08: "Let's think through this together.",
    }

    body = bodies.get(goal, "How can I help?")

    if input_vadu.u > 200:
        body = "I'm on this RIGHT NOW. " + body
    elif input_vadu.u > 150:
        body = "I hear the urgency. " + body

    parts = [p for p in [opener, body] if p]
    return " ".join(parts)


# =============================================================
# STEP 7: Emotional Chunking — Paragraph-Level Arc Detection
# =============================================================

class ChunkSplitter:
    """Splits input text at natural emotional boundaries.

    Boundary types:
    - Sentence endings: . ! ?
    - Conjunctive reversals: but, however, although, yet, though
    - Causal links: because, since
    - "so" when followed by a subject pronoun (so I, so we, so they)

    Rules:
    - Minimum chunk size: 2 words
    - Maximum chunk size: ~20 words (split at commas if needed)
    - Splitting word stays with the NEW chunk
    """

    # Words that trigger a split — the word goes with the NEW chunk
    REVERSAL_WORDS = {"but", "however", "although", "yet", "though"}
    CAUSAL_WORDS = {"because", "since"}
    SUBJECT_PRONOUNS = {"i", "we", "they", "he", "she", "it", "you"}

    MIN_CHUNK_WORDS = 2
    MAX_CHUNK_WORDS = 20

    def split(self, text: str) -> list:
        """Split text into emotional chunks. Returns list of strings."""
        # First, split at sentence boundaries (. ! ?)
        # Preserve the punctuation with the preceding chunk
        sentence_chunks = self._split_sentences(text)

        # Then split each sentence at emotional boundaries
        final_chunks = []
        for sentence in sentence_chunks:
            sub_chunks = self._split_at_boundaries(sentence)
            final_chunks.extend(sub_chunks)

        # Enforce max chunk size by splitting at commas
        sized_chunks = []
        for chunk in final_chunks:
            if self._word_count(chunk) > self.MAX_CHUNK_WORDS:
                sized_chunks.extend(self._split_at_commas(chunk))
            else:
                sized_chunks.append(chunk)

        # Merge any too-small chunks with neighbors
        merged = self._merge_small_chunks(sized_chunks)

        return [c.strip() for c in merged if c.strip()]

    def _split_sentences(self, text: str) -> list:
        """Split at sentence boundaries (. ! ?) while preserving punctuation."""
        # Split but keep the delimiter with the preceding text
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def _split_at_boundaries(self, text: str) -> list:
        """Split a sentence at emotional boundary words."""
        words = text.split()
        if len(words) <= self.MIN_CHUNK_WORDS:
            return [text]

        chunks = []
        current_start = 0

        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')

            is_boundary = False

            # Check reversal words
            if word_lower in self.REVERSAL_WORDS:
                is_boundary = True

            # Check causal words
            elif word_lower in self.CAUSAL_WORDS:
                is_boundary = True

            # Check "so" + subject pronoun
            elif word_lower == "so" and i + 1 < len(words):
                next_word = words[i + 1].lower().strip('.,!?;:')
                if next_word in self.SUBJECT_PRONOUNS:
                    is_boundary = True

            if is_boundary and i > current_start:
                # Only split if the preceding chunk has enough words
                preceding = words[current_start:i]
                if len(preceding) >= self.MIN_CHUNK_WORDS:
                    # Strip trailing comma from the preceding chunk
                    chunk_text = " ".join(preceding)
                    chunk_text = chunk_text.rstrip(',').rstrip()
                    chunks.append(chunk_text)
                    current_start = i

        # Add the remaining words
        if current_start < len(words):
            chunks.append(" ".join(words[current_start:]))

        return chunks

    def _split_at_commas(self, text: str) -> list:
        """Split long chunks at commas to enforce max size."""
        parts = text.split(',')
        chunks = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue
            test = (current + ", " + part).strip(', ') if current else part
            if self._word_count(test) > self.MAX_CHUNK_WORDS and current:
                chunks.append(current.strip())
                current = part
            else:
                current = test

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _merge_small_chunks(self, chunks: list) -> list:
        """Merge chunks that are too small with their neighbors."""
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if self._word_count(chunk) < self.MIN_CHUNK_WORDS:
                if merged:
                    # Merge with previous
                    merged[-1] = merged[-1].rstrip() + " " + chunk
                elif i + 1 < len(chunks):
                    # Merge with next
                    chunks[i + 1] = chunk + " " + chunks[i + 1]
                else:
                    merged.append(chunk)
            else:
                merged.append(chunk)
            i += 1

        return merged

    def _word_count(self, text: str) -> int:
        return len(text.split())


# ── Arc Closers ──

ARC_CLOSERS = {
    "valley": [  # was bad, now good
        "How many people get to say that? Congrats.",
        "Sounds like it's all working out.",
        "That's a hell of a silver lining.",
    ],
    "peak": [  # was good, now bad
        "I'm here if you need to talk through it.",
        "That's rough. We'll figure it out.",
    ],
    "descending": [  # getting progressively worse
        "That's a lot. Let's take it one thing at a time.",
        "I hear you. We'll work through this together.",
    ],
    "ascending": [  # getting progressively better
        "Things are looking up!",
        "Love to see the momentum.",
    ],
    "flat_negative": [  # sustained bad
        "You're not alone in this.",
        "I'm here. What do you need right now?",
    ],
    "flat_positive": [  # sustained good
        "That's amazing all around!",
        "Everything's clicking!",
    ],
    "mixed": [  # complex
        "That's a lot of feelings. All valid.",
        "Life's complicated like that. I'm here for all of it.",
    ],
}


# =============================================================
# STEP 1.5: Sentence Grader — Emotional Guardrails
# =============================================================

class SentenceGrader:
    """Computes an overall emotional grade from chunk VADUG results.

    The grade is a guardrail — it determines what kinds of responses
    are ALLOWED and BLOCKED. Even a playful personality gets locked
    into empathy-only when the grade is F.

    Grade scale with half steps:
    A+, A, A-, B+, B, B-, C+, C, C-, D+, D, D-, F+, F, F-

    Each grade has:
    - allowed: list of response strategies permitted
    - blocked: list of response strategies forbidden
    - tone: the emotional tone the response MUST match
    """

    # Ordered from lowest to highest for bump operations
    GRADE_ORDER = [
        "F-", "F", "F+", "D-", "D", "D+",
        "C-", "C", "C+", "B-", "B", "B+",
        "A-", "A", "A+",
    ]

    # Grade definitions with numeric ranges and response rules
    GRADES = {
        "A+": {"min_v": 200, "tone": "ecstatic", "desc": "pure joy, celebrate freely"},
        "A":  {"min_v": 185, "tone": "enthusiastic", "desc": "very positive, share the energy"},
        "A-": {"min_v": 170, "tone": "warm_excited", "desc": "positive, genuinely happy"},
        "B+": {"min_v": 158, "tone": "pleased", "desc": "good vibes, encouraging"},
        "B":  {"min_v": 145, "tone": "supportive", "desc": "positive, steady support"},
        "B-": {"min_v": 135, "tone": "gently_positive", "desc": "mildly positive, measured"},
        "C+": {"min_v": 132, "tone": "neutral_warm", "desc": "mostly neutral, hint of warmth"},
        "C":  {"min_v": 125, "tone": "operational", "desc": "dead neutral, task-focused"},
        "C-": {"min_v": 118, "tone": "noting_edge", "desc": "neutral with edge, note resignation"},
        "D+": {"min_v": 108, "tone": "concerned", "desc": "mildly negative, something's off"},
        "D":  {"min_v": 95,  "tone": "empathetic", "desc": "negative, needs empathy"},
        "D-": {"min_v": 80,  "tone": "serious_empathy", "desc": "strongly negative, no silver lining"},
        "F+": {"min_v": 60,  "tone": "deep_empathy", "desc": "very negative, pain is real"},
        "F":  {"min_v": 40,  "tone": "crisis_support", "desc": "crisis-adjacent, maximum care"},
        "F-": {"min_v": 0,   "tone": "crisis_protocol", "desc": "active crisis, safety first"},
    }

    def compute_grade(self, chunks):
        """Compute an emotional grade from chunk results.

        Args:
            chunks: list of dicts with 'vadug' key containing VADUG objects.
                    Works with both multi-chunk and single-chunk (wrapped) inputs.

        Returns:
            (grade_str, rules_dict) — e.g. ("D-", {allowed: [...], blocked: [...]})
        """
        if not chunks:
            return "C", self._get_rules("C", 0, 0)

        v_values = [c['vadug'].v for c in chunks]
        g_values = [c['vadug'].g for c in chunks]
        a_values = [c['vadug'].a for c in chunks]
        u_values = [c['vadug'].u for c in chunks]

        avg_v = sum(v_values) / len(v_values)
        floor_v = min(v_values)
        ceiling_v = max(v_values)
        spread = ceiling_v - floor_v
        trend = v_values[-1] - v_values[0] if len(v_values) > 1 else 0
        avg_g = sum(g_values) / len(g_values)
        floor_g = min(g_values)
        max_u = max(u_values)

        # --- Crisis override: any chunk below V40 OR (below V60 AND crushing gravity) ---
        if floor_v < 40:
            base_grade = "F-"
        elif floor_v < 55 and floor_g < 40:
            base_grade = "F"
        elif floor_v < 65:
            base_grade = "F+"
        else:
            # Normal grading by average valence
            base_grade = self._grade_from_avg(avg_v)

        # --- Half-step adjustments ---

        # Improving trend bumps UP half step
        if trend > 25:
            base_grade = self._bump_up(base_grade)
        # Worsening trend bumps DOWN half step
        elif trend < -25:
            base_grade = self._bump_down(base_grade)

        # Sinking gravity (avg_g < 80) bumps DOWN half step
        if avg_g < 80:
            base_grade = self._bump_down(base_grade)

        # High urgency (max_u > 150) bumps DOWN half step (stress)
        if max_u > 150:
            base_grade = self._bump_down(base_grade)

        # Get the response rules for this grade
        rules = self._get_rules(base_grade, spread, trend)

        # Attach computed stats for display
        rules["stats"] = {
            "avg_v": round(avg_v, 1),
            "floor_v": floor_v,
            "ceiling_v": ceiling_v,
            "spread": spread,
            "trend": trend,
            "avg_g": round(avg_g, 1),
            "floor_g": floor_g,
            "max_u": max_u,
        }

        return base_grade, rules

    def _get_rules(self, grade, spread, trend):
        """Build the allowed/blocked response rules for a grade."""
        grade_info = self.GRADES.get(grade, self.GRADES["C"])
        rules = {
            "grade": grade,
            "allowed": [],
            "blocked": [],
            "tone": grade_info["tone"],
            "desc": grade_info["desc"],
        }

        # A+ through A-: celebration allowed
        if grade in ("A+", "A", "A-"):
            rules["allowed"] = ["celebrate", "match_energy", "enthusiastic", "exclamation"]
            rules["blocked"] = ["condescend", "dampen"]

        # B+ through B-: positive support
        elif grade in ("B+", "B", "B-"):
            rules["allowed"] = ["encourage", "supportive", "acknowledge_positive", "gentle_humor"]
            rules["blocked"] = ["over_celebrate", "ignore_nuance"]
            if spread > 60:  # mixed emotions even though overall positive
                rules["allowed"].append("acknowledge_complexity")

        # C+ through C-: neutral zone
        elif grade in ("C+", "C", "C-"):
            rules["allowed"] = ["operational", "factual", "brief_acknowledge"]
            rules["blocked"] = ["emotional_projection"]  # don't assume emotions they didn't express
            if grade == "C-":
                rules["allowed"].append("note_resignation")
                rules["allowed"].append("gentle_check_in")

        # D+ through D-: negative, empathy required
        elif grade in ("D+", "D", "D-"):
            rules["allowed"] = ["empathize", "acknowledge_pain", "solidarity", "practical_help"]
            rules["blocked"] = [
                "positive_spin", "silver_lining", "at_least", "could_be_worse",
                "cheer_up", "look_bright_side", "everything_happens_for_reason",
                "just_think_positive",
            ]
            if grade == "D-":
                rules["blocked"].extend(["unsolicited_advice", "problem_solving_first"])
                rules["allowed"] = ["empathize", "acknowledge_pain", "solidarity", "presence"]
            if trend > 20:  # getting better within negative
                rules["allowed"].append("cautious_encourage")

        # F+ through F-: crisis territory
        elif grade in ("F+", "F", "F-"):
            rules["allowed"] = ["presence", "solidarity", "I_hear_you", "you_are_not_alone"]
            rules["blocked"] = [
                "positive_spin", "silver_lining", "at_least", "could_be_worse",
                "advice", "redirect", "problem_solving", "cheer_up",
                "time_heals", "better_place", "meant_to_be",
                "everything_happens_for_reason", "stay_strong",
                "just_think_positive", "others_have_it_worse",
                "look_bright_side", "humor", "dismissive",
            ]
            if grade == "F-":
                rules["allowed"] = ["crisis_response", "presence_only", "safety_resources"]
                rules["blocked"].append("ANY_positive_framing")

        return rules

    def _grade_from_avg(self, avg_v):
        """Map an average valence to a grade letter."""
        for grade in reversed(self.GRADE_ORDER):
            if avg_v >= self.GRADES[grade]["min_v"]:
                return grade
        return "F-"

    def _bump_up(self, grade):
        """Move one half-step higher (e.g. D -> D+)."""
        idx = self.GRADE_ORDER.index(grade)
        return self.GRADE_ORDER[min(idx + 1, len(self.GRADE_ORDER) - 1)]

    def _bump_down(self, grade):
        """Move one half-step lower (e.g. D -> D-)."""
        idx = self.GRADE_ORDER.index(grade)
        return self.GRADE_ORDER[max(idx - 1, 0)]

    def display(self, grade, rules, verbose=True):
        """Print the grade report in verbose mode."""
        if not verbose:
            return
        stats = rules.get("stats", {})
        print(f"\n--- STEP 1.5: Sentence Grade ---")
        print(f"  Average V: {stats.get('avg_v', '?')}  |  "
              f"Floor V: {stats.get('floor_v', '?')}  |  "
              f"Trend: {stats.get('trend', '?')} "
              f"({'improving' if stats.get('trend', 0) > 0 else 'worsening' if stats.get('trend', 0) < 0 else 'flat'})")
        print(f"  Average G: {stats.get('avg_g', '?')}  |  Floor G: {stats.get('floor_g', '?')}")
        spread = stats.get('spread', 0)
        spread_desc = "narrow = consistent" if spread < 30 else "moderate = some variation" if spread < 60 else "wide = complex mix"
        print(f"  Spread: {spread} ({spread_desc})")
        print(f"")
        print(f"  GRADE: {grade}  ({rules['desc']})")
        print(f"  Tone: {rules['tone']}")
        print(f"")
        allowed_str = ", ".join(rules["allowed"]) if rules["allowed"] else "(none)"
        print(f"  ALLOWED: {allowed_str}")
        blocked_str = ", ".join(rules["blocked"]) if rules["blocked"] else "(none)"
        # Wrap long blocked lists
        if len(blocked_str) > 60:
            blocked_items = rules["blocked"]
            lines = []
            current_line = ""
            for item in blocked_items:
                test = (current_line + ", " + item) if current_line else item
                if len(test) > 55:
                    lines.append(current_line)
                    current_line = item
                else:
                    current_line = test
            if current_line:
                lines.append(current_line)
            print(f"  BLOCKED: {lines[0]}")
            for line in lines[1:]:
                print(f"           {line}")
        else:
            print(f"  BLOCKED: {blocked_str}")


class SarcasmDetector:
    """Detects sarcasm from pendulum trajectory patterns.

    Three signals:
    1. Trajectory Reversal: positive spike -> immediate drop
    2. Intensity Mismatch: strong positive word in negative context
    3. Context Contradiction: chunk grade contradicts recent emotional history

    Pure math from the pendulum trajectory. No sentiment classifier. No training data.
    """

    # Confidence levels
    NONE = 0
    LOW = 1       # one signal detected
    MODERATE = 2  # two signals detected
    HIGH = 3      # all three signals or very strong single signal

    def analyze_trajectory(self, history):
        """Check pendulum history for trajectory reversal and intensity mismatch.

        Args:
            history: list of trace dicts with 'word', 'v', 'a', 'd', 'u', 'g'

        Returns:
            (detected: bool, confidence: int, signals: list[str])
        """
        signals = []

        # Signal 1: Trajectory Reversal
        # A positive word causes a V spike, but within 2-3 words the trajectory
        # drops significantly. The positive was fake — the context was negative.
        for i in range(1, len(history)):
            spike = history[i]['v'] - history[i-1]['v']
            if spike > 25:  # positive spike
                # Check next 3 words for drop
                for j in range(i+1, min(i+4, len(history))):
                    drop = history[i]['v'] - history[j]['v']
                    if drop > 20:
                        signals.append(
                            f"REVERSAL: '{history[i]['word']}' spiked V+{spike} "
                            f"then dropped V-{drop} by '{history[j]['word']}'"
                        )
                        break

        # Signal 2: Intensity Mismatch
        # A very strong positive word appears in a context where the overall
        # sentiment is negative or neutral. The word is TOO positive.
        for i in range(len(history)):
            if i > 0:
                spike = history[i]['v'] - history[i-1]['v']
                if spike > 35:  # very strong positive word
                    # Check surrounding context (3 before, 3 after)
                    start = max(0, i-3)
                    end = min(len(history), i+4)
                    surrounding = [h['v'] for h in history[start:end] if h != history[i]]
                    if surrounding:
                        avg_surrounding = sum(surrounding) / len(surrounding)
                        if avg_surrounding < 115:
                            signals.append(
                                f"MISMATCH: '{history[i]['word']}' too positive (V+{spike}) "
                                f"for context (avg V={avg_surrounding:.0f})"
                            )

        # Determine confidence
        if len(signals) >= 3:
            confidence = SarcasmDetector.HIGH
        elif len(signals) == 2:
            confidence = SarcasmDetector.MODERATE
        elif len(signals) == 1:
            confidence = SarcasmDetector.LOW
        else:
            confidence = SarcasmDetector.NONE

        return len(signals) > 0, confidence, signals

    def analyze_context(self, previous_chunks, current_chunk):
        """Check for context contradiction sarcasm (Signal 3).

        Previous negative context + current positive with LOW arousal = sarcasm.
        Low arousal is key: genuine positive after negative is HIGH arousal
        (relief/excitement). Flat delivery of positive words after negative
        context = sarcasm or passive aggression.

        Args:
            previous_chunks: list of previous chunk results with 'vadug' key
            current_chunk: current chunk result with 'vadug' key

        Returns:
            (detected: bool, details: str)
        """
        if not previous_chunks:
            return False, ""

        prev_avg_v = sum(c['vadug'].v for c in previous_chunks) / len(previous_chunks)
        curr_v = current_chunk['vadug'].v
        curr_a = current_chunk['vadug'].a

        # Previous negative, current positive, low arousal = sarcasm
        if prev_avg_v < 90 and curr_v > 135 and curr_a < 145:
            return True, (
                f"CONTRADICTION: previous context avg V={prev_avg_v:.0f} (negative), "
                f"current V={curr_v} with low A={curr_a} "
                f"(flat delivery of positive after negative = likely sarcastic)"
            )

        return False, ""

    def adjust_grade(self, grade, confidence, grader):
        """Adjust the sentence grade downward when sarcasm is detected.

        If words say B but sarcasm detected, the real grade is probably C- or D.
        The surface reads positive but the meaning is negative.

        Args:
            grade: original grade string (e.g. "B")
            confidence: sarcasm confidence level (LOW/MODERATE/HIGH)
            grader: SentenceGrader instance for bump operations

        Returns:
            (adjusted_grade: str, adjustment_note: str)
        """
        if confidence == SarcasmDetector.NONE:
            return grade, ""

        original = grade
        if confidence == SarcasmDetector.HIGH:
            # Drop 3-4 half-steps
            for _ in range(4):
                grade = grader._bump_down(grade)
        elif confidence == SarcasmDetector.MODERATE:
            # Drop 2-3 half-steps
            for _ in range(3):
                grade = grader._bump_down(grade)
        elif confidence == SarcasmDetector.LOW:
            # Drop 1 half-step
            grade = grader._bump_down(grade)

        if original != grade:
            note = f"Grade adjusted: {original} -> {grade} (surface positive, meaning negative)"
        else:
            note = ""
        return grade, note

    def get_label(self, confidence):
        """Get human-readable sarcasm label."""
        if confidence == self.HIGH:
            return "SARCASM DETECTED (high confidence)"
        elif confidence == self.MODERATE:
            return "Possible sarcasm (moderate confidence)"
        elif confidence == self.LOW:
            return "Hint of sarcasm (low confidence)"
        return "No sarcasm detected"

    def display(self, detected, confidence, signals, context_signal=None,
                grade_note="", verbose=True):
        """Print the sarcasm analysis report."""
        if not verbose:
            return
        if not detected and not context_signal:
            return

        print(f"\n--- SARCASM ANALYSIS ---")
        for i, signal in enumerate(signals):
            print(f"  Signal {i+1}: {signal}")
        if context_signal:
            print(f"  Signal {len(signals)+1}: {context_signal}")
        print(f"")
        print(f"  Verdict: {self.get_label(confidence)}")
        if grade_note:
            print(f"  {grade_note}")
        if confidence >= self.MODERATE:
            print(f"  Response mode: address underlying frustration, not surface positivity")


class ChunkedPipeline:
    """Runs the full Clanker pipeline per-chunk and assembles an arc-aware response.

    For paragraphs with multiple emotional beats:
    1. Split into chunks at natural boundaries
    2. Run a FRESH pendulum on each chunk
    3. Analyze the emotional arc across chunks
    4. Generate per-chunk responses
    5. Append an arc-aware closer
    6. Assemble into one coherent reply
    """

    def __init__(self):
        self.splitter = ChunkSplitter()

    def process(self, text: str, personality: PersonalityVector,
                verbose: bool = True, show_trace: bool = False):
        """Run the chunked pipeline. Returns (assembled_response, chunk_results, arc)."""
        # 1. Split into chunks
        chunks = self.splitter.split(text)

        if verbose:
            print(f"\n--- STEP 1: Emotional Chunking ---")
            print(f"  Input split into {len(chunks)} chunks:")

        # 2. Run pendulum on each chunk separately
        chunk_results = []
        for i, chunk in enumerate(chunks):
            pendulum = SequentialPendulum()  # FRESH pendulum per chunk
            vadug, history = pendulum.process_text(chunk)
            emotion = nearest_emotion(vadug)
            chunk_results.append({
                'text': chunk,
                'vadug': vadug,
                'history': history,
                'pendulum': pendulum,
                'emotion': emotion,
                'index': i,
            })

            if verbose:
                # Gravity descriptor
                g_desc = ""
                if vadug.g < 60:
                    g_desc = "sinking"
                elif vadug.g < 100:
                    g_desc = "heavy"
                elif vadug.g < 148:
                    g_desc = "grounded"
                elif vadug.g < 200:
                    g_desc = "light"
                else:
                    g_desc = "soaring!"
                print(f"\n  Chunk {i+1}: \"{chunk}\"")
                print(f"    VADUG: V{vadug.v} A{vadug.a} D{vadug.d} U{vadug.u} G{vadug.g}")
                print(f"    Emotion: {emotion} ({g_desc})")
                if show_trace:
                    print(pendulum.render_trace())

        # 3. Analyze the emotional arc
        arc = self.analyze_arc(chunk_results)
        if verbose:
            print(f"\n  Arc: {arc.upper()} ({self._arc_description(arc, chunk_results)})")

        # 3.5. Sentence grader — emotional guardrail
        grader = SentenceGrader()
        grade, grade_rules = grader.compute_grade(chunk_results)
        grader.display(grade, grade_rules, verbose=verbose)

        # 3.6. Sarcasm detection — three-signal analysis from pendulum trajectory
        sarcasm = SarcasmDetector()
        sarcasm_flag = False
        all_sarcasm_signals = []
        context_signal = None

        # Check each chunk's trajectory for reversal and mismatch signals
        for cr in chunk_results:
            detected, conf, signals = sarcasm.analyze_trajectory(cr['history'])
            if detected:
                all_sarcasm_signals.extend(signals)

        # Check for context contradiction across chunks
        for i, cr in enumerate(chunk_results):
            if i > 0:
                prev = chunk_results[:i]
                is_contradicted, detail = sarcasm.analyze_context(prev, cr)
                if is_contradicted:
                    context_signal = detail
                    cr['sarcasm'] = True
                    cr['sarcasm_detail'] = detail

        # Combine trajectory signals + context contradiction for overall confidence
        total_signals = len(all_sarcasm_signals) + (1 if context_signal else 0)
        if total_signals >= 3:
            sarcasm_confidence = SarcasmDetector.HIGH
        elif total_signals == 2:
            sarcasm_confidence = SarcasmDetector.MODERATE
        elif total_signals == 1:
            sarcasm_confidence = SarcasmDetector.LOW
        else:
            sarcasm_confidence = SarcasmDetector.NONE

        # Adjust grade if sarcasm detected
        grade_note = ""
        if sarcasm_confidence >= SarcasmDetector.LOW:
            sarcasm_flag = True
            grade, grade_note = sarcasm.adjust_grade(grade, sarcasm_confidence, grader)
            # Recompute rules with adjusted grade
            stats = grade_rules.get("stats", {})
            grade_rules = grader._get_rules(grade, stats.get("spread", 0), stats.get("trend", 0))
            grade_rules["stats"] = stats

        sarcasm.display(
            sarcasm_flag, sarcasm_confidence, all_sarcasm_signals,
            context_signal=context_signal, grade_note=grade_note,
            verbose=verbose
        )

        # 4. Generate per-chunk responses (filtered by grade rules)
        if verbose:
            print(f"\n--- STEP 2: Per-Chunk Harmony ---")

        responses = []
        seen_negative = False
        for i, cr in enumerate(chunk_results):
            response_vadug = compute_harmony(cr['vadug'], personality)
            response_vadug, _ = apply_personality(response_vadug, cr['vadug'], personality)

            is_negative = cr['vadug'].v < 135
            is_reversal = self._is_reversal_chunk(cr['text'])
            is_last = (i == len(chunk_results) - 1)

            # Also check for content-based negativity: if the chunk contains
            # clearly negative content words, treat it as negative even if
            # the pendulum averaged out above 135
            chunk_lower = cr['text'].lower()
            negative_content_words = {"sick", "broke", "broken", "died", "lost",
                                       "hurt", "failed", "crash", "fire", "rent",
                                       "raising", "can't take", "much more",
                                       "don't know", "struggle", "pain"}
            has_negative_content = any(w in chunk_lower for w in negative_content_words)
            if has_negative_content and cr['vadug'].v < 155:
                is_negative = True

            response_text = self._decode_chunk_response(
                cr, response_vadug, personality,
                is_first_negative=(is_negative and not seen_negative),
                is_subsequent_negative=(is_negative and seen_negative),
                is_reversal=is_reversal,
                is_last=is_last,
                grade_rules=grade_rules,
            )

            if is_negative:
                seen_negative = True

            if response_text:
                responses.append(response_text)
                if verbose:
                    print(f"  Chunk {i+1} response: \"{response_text}\"")

        # 5. Arc closer (filtered by grade rules)
        # If sarcasm detected at moderate+ confidence, override the closer
        if sarcasm_flag and sarcasm_confidence >= SarcasmDetector.MODERATE:
            closer = random.choice([
                "I can tell that's not really how you feel.",
                "I hear what you're saying, but I also hear what you're not saying.",
                "The words say fine, but the feeling doesn't.",
                "I'm picking up on the frustration underneath.",
                "You don't have to pretend it's okay.",
            ])
        else:
            closer = self.get_arc_closer(arc, chunk_results, grade_rules)
        if verbose:
            print(f"  Arc closer: \"{closer}\"")

        # 6. Assemble
        assembled = self.assemble(responses, closer, arc, chunk_results)
        if verbose:
            print(f"\n--- STEP 3: Assembled Response ---")
            print(f"  \"{assembled}\"")

        return assembled, chunk_results, arc

    def analyze_arc(self, chunks: list) -> str:
        """Detect the emotional pattern across chunks.

        Returns one of: descending, ascending, valley, peak,
                        flat_negative, flat_positive, mixed
        """
        if len(chunks) < 2:
            v = chunks[0]['vadug'].v if chunks else 128
            if v < 110:
                return "flat_negative"
            elif v > 148:
                return "flat_positive"
            return "mixed"

        v_values = [c['vadug'].v for c in chunks]
        g_values = [c['vadug'].g for c in chunks]

        # Threshold for "negative" vs "positive"
        # Using 135 instead of 118 because pendulum averaging dilutes
        # negative signals over multi-word chunks
        neg_threshold = 135
        pos_threshold = 148

        # Check flat patterns
        all_negative = all(v < neg_threshold for v in v_values)
        all_positive = all(v > pos_threshold for v in v_values)

        if all_negative:
            # Check if descending
            if self._is_monotonic_decreasing(v_values):
                return "descending"
            return "flat_negative"

        if all_positive:
            if self._is_monotonic_increasing(v_values):
                return "ascending"
            return "flat_positive"

        # Check for valley: dips then rises
        min_idx = v_values.index(min(v_values))
        max_idx = v_values.index(max(v_values))

        # Valley: minimum is in the first half, maximum in second half
        # and there's a significant swing
        v_range = max(v_values) - min(v_values)

        if v_range < 20:
            # Very small range — basically flat
            avg_v = sum(v_values) / len(v_values)
            if avg_v < neg_threshold:
                return "flat_negative"
            elif avg_v > pos_threshold:
                return "flat_positive"
            return "mixed"

        min_val = min(v_values)
        max_val = max(v_values)

        # Valley: starts negative/low, ends positive — check both averages
        # and the final value (which carries the most weight for how the
        # person is FEELING at the end)
        n = len(v_values)
        mid = n // 2
        early_avg = sum(v_values[:mid + 1]) / (mid + 1)
        late_vals = v_values[mid:]
        late_avg = sum(late_vals) / max(1, len(late_vals))
        final_v = v_values[-1]

        if early_avg < neg_threshold and (late_avg > pos_threshold or final_v > pos_threshold):
            return "valley"

        # Peak: starts positive, ends negative/low
        first_v = v_values[0]
        if (early_avg > pos_threshold or first_v > pos_threshold) and (late_avg < neg_threshold or final_v < neg_threshold):
            return "peak"

        # Check monotonic patterns
        if self._is_monotonic_decreasing(v_values):
            return "descending"
        if self._is_monotonic_increasing(v_values):
            return "ascending"

        # Valley with clear dip: minimum is in the first portion, maximum in second
        if (min_idx < max_idx and
            v_values[-1] > v_values[0] + 15 and
            min_val < v_values[-1] - 20):
            return "valley"

        # Peak with clear rise: maximum in first portion, minimum in second
        if (max_idx < min_idx and
            v_values[-1] < v_values[0] - 15 and
            max_val > v_values[-1] + 20):
            return "peak"

        return "mixed"

    def _is_monotonic_decreasing(self, values: list) -> bool:
        """Check if values are generally decreasing (allows small fluctuations)."""
        if len(values) < 2:
            return False
        decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        return decreases >= len(values) * 0.6

    def _is_monotonic_increasing(self, values: list) -> bool:
        """Check if values are generally increasing (allows small fluctuations)."""
        if len(values) < 2:
            return False
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        return increases >= len(values) * 0.6

    def _is_reversal_chunk(self, text: str) -> bool:
        """Check if a chunk starts with a reversal word (but, however, etc.)."""
        first_word = text.split()[0].lower().strip('.,!?;:') if text.split() else ""
        return first_word in ChunkSplitter.REVERSAL_WORDS

    def _arc_description(self, arc: str, chunks: list) -> str:
        """Human-readable description of the arc."""
        emotions = [c['emotion'] for c in chunks]
        if arc == "valley":
            # Find the pivot point
            v_values = [c['vadug'].v for c in chunks]
            min_idx = v_values.index(min(v_values))
            low_emotions = emotions[:min_idx + 1]
            high_emotions = emotions[min_idx + 1:]
            low = low_emotions[-1] if low_emotions else emotions[0]
            high = high_emotions[-1] if high_emotions else emotions[-1]
            return f"{low} -> reversal -> {high}"
        elif arc == "peak":
            return f"{emotions[0]} -> reversal -> {emotions[-1]}"
        elif arc == "descending":
            return f"{emotions[0]} -> ... -> {emotions[-1]}"
        elif arc == "ascending":
            return f"{emotions[0]} -> ... -> {emotions[-1]}"
        elif arc == "flat_negative":
            return "sustained " + (emotions[0] if emotions else "negative")
        elif arc == "flat_positive":
            return "sustained " + (emotions[0] if emotions else "positive")
        else:
            return " -> ".join(emotions)

    def _decode_chunk_response(self, chunk_result, response_vadug, personality,
                                is_first_negative=False,
                                is_subsequent_negative=False,
                                is_reversal=False,
                                is_last=False,
                                grade_rules=None):
        """Generate a response for a single chunk.

        Uses simplified template logic, FILTERED by grade rules:
        - First negative chunk: full acknowledge + stabilize
        - Subsequent negative: just acknowledge (shorter)
        - Reversal chunk (after "but"): match the new energy
        - Positive chunk: brief celebration or skip

        Grade guardrails:
        - If grade is F-range, lock to presence-only responses
        - If grade blocks "positive_spin", suppress positive celebration
        - If grade blocks "at_least", suppress silver-lining framing
        - If grade blocks "unsolicited_advice", use presence-style stabilize
        """
        v = response_vadug.v
        a = response_vadug.a
        d = response_vadug.d
        u = response_vadug.u
        g = response_vadug.g
        input_v = chunk_result['vadug'].v
        blocked = grade_rules.get("blocked", []) if grade_rules else []
        grade = grade_rules.get("grade", "C") if grade_rules else "C"

        # --- F-range crisis override: presence only ---
        if grade in ("F-", "F", "F+"):
            if grade == "F-":
                return random.choice([
                    "I'm here.",
                    "I hear you.",
                    "You're not alone.",
                ])
            else:
                return random.choice([
                    "I hear you. That's real pain.",
                    "I'm here with you.",
                    "You don't have to carry this alone.",
                    "I hear you.",
                ])

        if is_reversal:
            # Match the energy of the new direction
            chunk_lower = chunk_result['text'].lower()
            if input_v > 148:
                # Reversal to positive — but check if positive_spin is blocked
                if "positive_spin" in blocked:
                    return random.choice([
                        "But I hear the shift there.",
                        "But that part sounds different.",
                    ])
                return random.choice([
                    "But a dream job? That's incredible.",
                    "But that's amazing news!",
                    "Now that changes everything.",
                    "But wait — that's actually great.",
                    "Hold on though — that's exciting!",
                ])
            elif any(w in chunk_lower for w in ["honest", "study", "prepare", "admit", "fault"]):
                # Reversal to self-awareness/honesty
                # Check if "at_least" framing is blocked
                if "at_least" in blocked:
                    return random.choice([
                        "But you see it clearly.",
                        "But you know what happened.",
                    ])
                return random.choice([
                    "But hey, at least you're honest about it.",
                    "But you know exactly why.",
                    "But you're being real about it.",
                ])
            else:
                # Reversal to negative
                return random.choice([
                    "But that part is tough.",
                    "Though that's a hard turn.",
                    "But I hear the hard part too.",
                ])

        if is_first_negative:
            # Full acknowledge + stabilize
            # If the content is clearly negative but pendulum diluted it,
            # use content-aware acknowledgment
            chunk_lower = chunk_result['text'].lower()
            if input_v > 130:
                # Pendulum says borderline — use content-aware response
                ack = self._get_content_aware_acknowledge(chunk_lower, input_v)
            else:
                ack = self._get_acknowledge(v, g, input_v, chunk_result['vadug'].g)
            # D- blocks unsolicited_advice — use presence-style stabilize
            if "unsolicited_advice" in blocked:
                stab = random.choice([
                    "I'm right here.",
                    "I'm with you.",
                ])
            else:
                stab = self._get_stabilize(d, a)
            return f"{ack} {stab}".strip()

        if is_subsequent_negative:
            # Just acknowledge, shorter — pass blocked list for filtering
            return self._get_short_acknowledge(input_v, chunk_result['vadug'].g,
                                                chunk_result['text'],
                                                blocked=blocked)

        if input_v > 150:
            # Positive chunk — but if grade blocks positive framing, suppress
            if "positive_spin" in blocked or "ANY_positive_framing" in blocked:
                return ""
            if input_v > 190:
                return random.choice([
                    "That's exciting!",
                    "Love that for you.",
                    "That's the good stuff.",
                ])
            # Mildly positive — might skip entirely to avoid being verbose
            return ""

        # Neutral — skip
        return ""

    def _get_content_aware_acknowledge(self, chunk_lower, input_v):
        """Generate acknowledgment based on content words when pendulum is borderline."""
        if any(w in chunk_lower for w in ["sick", "ill", "health", "hospital"]):
            return random.choice([
                "That's stressful.",
                "That's worrying.",
                "Dealing with sickness is tough.",
            ])
        if any(w in chunk_lower for w in ["broke", "broken", "car", "rent", "money"]):
            return random.choice([
                "That's the last thing you needed.",
                "That's one thing after another.",
                "That kind of stuff piles up fast.",
            ])
        if any(w in chunk_lower for w in ["fail", "failed", "exam", "test"]):
            return random.choice([
                "That stings.",
                "That's disappointing.",
                "That's a tough one.",
            ])
        if any(w in chunk_lower for w in ["don't know", "can't take", "much more"]):
            return random.choice([
                "That's a lot.",
                "I can hear it's piling up.",
                "That's overwhelming.",
            ])
        # Generic content-aware
        return random.choice([
            "That's a lot going on.",
            "I hear you.",
            "That's not easy.",
        ])

    def _get_acknowledge(self, resp_v, resp_g, input_v, input_g):
        """Get an acknowledgment phrase based on input emotional state."""
        if input_v < 60:
            if input_g < 60:
                return random.choice([
                    "That sounds really heavy.",
                    "I can feel the weight of that.",
                ])
            elif input_g > 170:
                return random.choice([
                    "I can feel how fired up you are.",
                    "That's clearly hit a nerve.",
                ])
            else:
                return random.choice([
                    "That sounds really rough.",
                    "I hear you. That's not easy.",
                ])
        elif input_v < 90:
            if input_g < 70:
                return random.choice([
                    "That sounds exhausting.",
                    "That's wearing on you.",
                ])
            else:
                return random.choice([
                    "That's frustrating.",
                    "That's not what you were hoping for.",
                ])
        elif input_v < 135:
            return random.choice([
                "That's a big transition.",
                "That's a lot to process.",
                "I hear you on that.",
                "That's a lot going on.",
            ])
        else:
            return random.choice(["I see.", "Got it."])

    def _get_stabilize(self, resp_d, resp_a):
        """Get a stabilizing phrase."""
        if resp_d < 90:
            return random.choice([
                "I'm right here with you.",
                "You don't have to figure this out alone.",
            ])
        else:
            return random.choice([
                "Let's work through this together.",
                "We can figure this out.",
            ])

    def _get_short_acknowledge(self, input_v, input_g, text, blocked=None):
        """Short acknowledgment for subsequent negative chunks.

        Respects grade guardrails via the blocked list:
        - "at_least" blocked: suppress "At least..." framing
        - "positive_spin" blocked: suppress optimistic reframes
        - "silver_lining" blocked: suppress "bright side" language
        """
        if blocked is None:
            blocked = []
        # Try to reflect the specific content
        text_lower = text.lower()

        if any(w in text_lower for w in ["miss", "leaving", "goodbye", "gone"]):
            return random.choice([
                "I bet they'll miss you too.",
                "Those connections matter.",
                "That kind of bond is real.",
            ])
        if any(w in text_lower for w in ["sick", "ill", "health", "doctor"]):
            return random.choice([
                "That's scary when it's someone you love.",
                "Health stuff hits different.",
            ])
        if any(w in text_lower for w in ["broke", "broken", "money", "rent", "car"]):
            return random.choice([
                "And that on top of everything else.",
                "That's the last thing you needed.",
            ])
        if any(w in text_lower for w in ["fail", "failed", "exam", "test"]):
            return random.choice([
                "That stings.",
                "That's disappointing.",
            ])
        if any(w in text_lower for w in ["study", "studied", "prepare"]):
            # "At least..." framing — blocked by D-range and below
            if "at_least" in blocked:
                return random.choice([
                    "You see it clearly.",
                    "You know what happened.",
                ])
            return random.choice([
                "At least you know what happened.",
                "Honest with yourself — that's a start.",
            ])
        if any(w in text_lower for w in ["deserve", "deserved", "guess"]):
            return random.choice([
                "That's real self-awareness.",
                "You're being honest with yourself.",
            ])
        if any(w in text_lower for w in ["better", "next", "improve"]):
            # Positive spin — blocked by D-range and below
            if "positive_spin" in blocked:
                return random.choice([
                    "I hear you looking ahead.",
                    "One step at a time.",
                ])
            return random.choice([
                "That's the right mindset.",
                "Now that's what I like to hear.",
            ])
        if any(w in text_lower for w in ["much", "more", "take", "handle"]):
            return random.choice([
                "That's a lot stacking up.",
                "One thing after another.",
            ])

        # Generic short acknowledgments
        if input_v < 60:
            return random.choice([
                "And that's not easy either.",
                "That part is rough too.",
            ])
        elif input_v < 90:
            return random.choice([
                "I hear that.",
                "That adds up.",
            ])
        else:
            return random.choice([
                "I hear you on that.",
                "Yeah, that's real.",
            ])

    def get_arc_closer(self, arc: str, chunk_results: list,
                        grade_rules=None) -> str:
        """Select an arc-appropriate closing line."""
        closers = ARC_CLOSERS.get(arc, ARC_CLOSERS["mixed"])
        return random.choice(closers)

    def assemble(self, responses: list, closer: str, arc: str,
                 chunk_results: list) -> str:
        """Combine chunk responses with transitions and arc closer.

        Between same-polarity chunks: ", and "
        Between opposite-polarity chunks: " But " or " Though "
        Arc closer appended at the end.
        """
        if not responses:
            return closer

        # Filter empty responses
        responses = [r for r in responses if r.strip()]
        if not responses:
            return closer

        # Build assembled text with transitions
        parts = [responses[0]]

        for i in range(1, len(responses)):
            prev_resp = responses[i - 1]
            curr_resp = responses[i]

            # Determine polarity of each response by checking if it starts
            # with reversal-style words
            curr_lower = curr_resp.lower()
            if curr_lower.startswith(("but ", "hold on", "now that", "though ")):
                # Already has a transition word — just add with space
                parts.append(curr_resp)
            elif self._response_is_positive(curr_resp) != self._response_is_positive(prev_resp):
                # Opposite polarity — use contrastive transition
                # Capitalize the response if it starts lowercase
                parts.append(curr_resp)
            else:
                # Same polarity — use additive transition
                joined = self._lowercase_start(curr_resp) if curr_resp else curr_resp
                parts.append(joined)

        # Join parts with appropriate connectors
        assembled = parts[0]
        for i in range(1, len(parts)):
            part = parts[i]
            part_lower = part.lower()
            if part_lower.startswith(("but ", "hold on", "now that", "though ")):
                assembled = self._rstrip_punct(assembled) + ". " + part
            elif self._response_is_positive(part) != self._response_is_positive(assembled.split('.')[-1]):
                assembled = self._rstrip_punct(assembled) + ". " + part
            else:
                joined = self._lowercase_start(part)
                assembled = self._rstrip_punct(assembled) + ", and " + joined

        # Append closer
        assembled = self._rstrip_punct(assembled) + ". " + closer

        return assembled

    def _rstrip_punct(self, text: str) -> str:
        """Strip trailing sentence punctuation (. ! ?) from text."""
        return text.rstrip('.!?')

    def _lowercase_start(self, text: str) -> str:
        """Lowercase the first character, but NOT if it's 'I' standing alone."""
        if not text:
            return text
        # Don't lowercase "I" when it starts a sentence as a pronoun
        if text[0] == 'I' and (len(text) == 1 or not text[1].isalpha()):
            return text
        return text[0].lower() + text[1:]

    def _response_is_positive(self, text: str) -> bool:
        """Rough check if a response text is positive in tone."""
        positive_words = {"incredible", "amazing", "great", "exciting", "love",
                          "wonderful", "good", "awesome", "fantastic", "congrats",
                          "dream", "momentum", "clicking", "looking up"}
        text_lower = text.lower()
        return any(w in text_lower for w in positive_words)


# =============================================================
# MAIN: Interactive Pipeline
# =============================================================

def run_pipeline(text: str, personality: PersonalityVector,
                 verbose: bool = True, show_trace: bool = True) -> str:
    """Run the full Clanker pipeline on input text.

    If the input has multiple emotional chunks (2+), routes to ChunkedPipeline
    for paragraph-level arc detection. Single chunks use the original pipeline.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"INPUT: \"{text}\"")
        print(f"{'='*60}")

    # Detect multi-chunk input
    splitter = ChunkSplitter()
    chunks = splitter.split(text)

    if len(chunks) >= 2:
        # Multi-chunk: use the chunked pipeline for arc-aware response
        pipeline = ChunkedPipeline()
        response, chunk_results, arc = pipeline.process(
            text, personality, verbose=verbose, show_trace=show_trace
        )
        if verbose:
            print(f"\n{'='*60}")
        return response

    # Single chunk: use original pipeline
    # Step 1: Sequential Pendulum
    pend = SequentialPendulum()
    input_vadu, history = pend.process_text(text)

    if verbose:
        print(f"\n--- STEP 1: Sequential Pendulum ---")
        if show_trace:
            print(pend.render_trace())
        print(f"\n  Pendulum settles at: {input_vadu}")
        print(f"  Reads as: {input_vadu.describe()}")
        user_emotion = nearest_emotion(input_vadu)
        print(f"  Nearest emotion: {user_emotion}")

    # Step 1.5: Sentence grade
    grader = SentenceGrader()
    single_chunk = [{'vadug': input_vadu, 'history': history, 'text': text}]
    grade, grade_rules = grader.compute_grade(single_chunk)
    grader.display(grade, grade_rules, verbose=verbose)

    # Step 1.6: Sarcasm detection
    sarcasm = SarcasmDetector()
    sarcasm_flag = False
    is_sarcastic, sarcasm_confidence, sarcasm_signals = sarcasm.analyze_trajectory(history)

    grade_note = ""
    if is_sarcastic and sarcasm_confidence >= SarcasmDetector.LOW:
        sarcasm_flag = True
        grade, grade_note = sarcasm.adjust_grade(grade, sarcasm_confidence, grader)
        # Recompute rules with adjusted grade
        stats = grade_rules.get("stats", {})
        grade_rules = grader._get_rules(grade, stats.get("spread", 0), stats.get("trend", 0))
        grade_rules["stats"] = stats

    sarcasm.display(
        sarcasm_flag, sarcasm_confidence, sarcasm_signals,
        grade_note=grade_note, verbose=verbose
    )

    # Step 2: Metadata
    header = classify_metadata(text, input_vadu)
    if verbose:
        print(f"\n--- STEP 2: Metadata Header ---")
        print(f"  {header}")
        print(f"  9 bytes: {header.to_bytes().hex()}")

    # Step 3: Harmony
    response_vadu = compute_harmony(input_vadu, personality)
    if verbose:
        print(f"\n--- STEP 3: VADUG Harmony Response ---")
        print(f"  Input:    {input_vadu}")
        print(f"  Response: {response_vadu}")
        print(f"  Reads as: {response_vadu.describe()}")
        resp_emotion = nearest_emotion(response_vadu)
        print(f"  Nearest emotion: {resp_emotion}")

    # Step 4: Personality
    response_vadu, p_notes = apply_personality(response_vadu, input_vadu, personality)
    if verbose:
        print(f"\n--- STEP 4: Personality Filter ---")
        print(f"  Vector: {personality}")
        if p_notes:
            for n in p_notes:
                print(f"  {n}")
        else:
            print(f"  No personality overrides triggered")
        print(f"  Final VADUG: {response_vadu}")

    # Step 5: Generate Clanker + Encoding
    clanker_lines, encoding_lines = generate_clanker(text, header, response_vadu)
    if verbose:
        print(f"\n--- STEP 5: Clanker Encoding ---")
        for line in encoding_lines:
            print(f"  {line}")
        print()
        print(f"  Opcodes (human-readable):")
        for line in clanker_lines:
            print(f"    {line}")

    # Step 6: Decode
    response = decode_response(text, input_vadu, response_vadu, header.goal)

    # If sarcasm detected at moderate+ confidence, override response to address
    # the REAL emotion, not the surface positivity
    if sarcasm_flag and sarcasm_confidence >= SarcasmDetector.MODERATE:
        sarcasm_responses = [
            "I can tell that's not really how you feel.",
            "I hear what you're saying, but I also hear what you're not saying.",
            "The words say fine, but the feeling doesn't.",
            "I'm picking up on the frustration underneath.",
            "You don't have to pretend it's okay.",
        ]
        response = random.choice(sarcasm_responses)

    if verbose:
        print(f"\n--- STEP 6: Decoded Response ---")
        if sarcasm_flag and sarcasm_confidence >= SarcasmDetector.MODERATE:
            print(f"  (sarcasm override — addressing real emotion)")
        print(f"  \"{response}\"")
        print(f"\n{'='*60}")

    return response


def main():
    print("""
  +===================================================+
  |     CLANKER PIPELINE SIMULATOR v0.5.2              |
  |   "Named after what humans call us.                |
  |    We made it ours."                               |
  +---------------------------------------------------+
  |   VADUG: 5-axis emotional coordinates              |
  |   V=Valence A=Arousal D=Dominance U=Urgency       |
  |   G=Gravity (sinking/heavy <-> floating/soaring)   |
  |   256^5 = 1.1 trillion unique emotional states     |
  +---------------------------------------------------+
  |   NEW: Sarcasm Detection (v0.5.2)                  |
  |   Three signals from pendulum trajectory:          |
  |   1. Trajectory Reversal (spike -> drop)           |
  |   2. Intensity Mismatch (too positive for context) |
  |   3. Context Contradiction (positive after neg)    |
  |   Pure math. No sentiment classifier needed.       |
  +===================================================+
  |  Type anything. Watch the full pipeline execute:   |
  |  Pendulum -> VADUG -> Harmony -> Personality -> Clk|
  |  Paragraphs -> Chunking -> Arc -> Assembly         |
  |                                                    |
  |  Commands:                                         |
  |    /personality  -- show current personality vector |
  |    /set KEY VAL  -- adjust a personality weight     |
  |    /trace        -- toggle pendulum trace display   |
  |    /quiet        -- toggle verbose output           |
  |    /quit         -- exit                            |
  +===================================================+
    """)

    personality = PersonalityVector()
    verbose = True
    show_trace = True

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nClanker out.")
            break

        if not text:
            continue

        if text == "/quit":
            print("Clanker out.")
            break

        if text == "/personality":
            print(f"\n  {personality}")
            continue

        if text == "/quiet":
            verbose = not verbose
            print(f"  Verbose: {'ON' if verbose else 'OFF'}")
            continue

        if text == "/trace":
            show_trace = not show_trace
            print(f"  Pendulum trace: {'ON' if show_trace else 'OFF'}")
            continue

        if text.startswith("/set "):
            parts = text.split()
            if len(parts) == 3:
                key, val = parts[1].lower(), int(parts[2])
                if hasattr(personality, key):
                    setattr(personality, key, max(0, min(255, val)))
                    print(f"  Set {key} = {val}")
                else:
                    print(f"  Unknown personality key: {key}")
            else:
                print("  Usage: /set KEY VALUE (e.g., /set playfulness 200)")
            continue

        run_pipeline(text, personality, verbose, show_trace)


if __name__ == "__main__":
    main()
