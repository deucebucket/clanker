#!/usr/bin/env python3
"""
Clanker Pipeline Simulator — Interactive Demo (v0.9: Math-Based Response Builder)

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
    # ── Gemini-generated vocabulary (merged batch) ──
    "ibuprofen": (20, -10, 15, 10, -5),
    "fever": (-40, 25, -25, 45, 10),
    "dizzy": (-30, 15, -40, 35, -20),
    "nurse": (25, 5, 5, 15, 5),
    "prescription": (10, 0, 10, 25, 5),
    "pharmacy": (15, 5, 10, 15, 5),
    "illness": (-50, 10, -30, 20, 25),
    "symptom": (-20, 15, -10, 35, 5),
    "treatment": (30, 10, 20, 10, -10),
    "vaccine": (45, 10, 20, 15, 0),
    "allergy": (-30, 20, -20, 40, 5),
    "fracture": (-45, 35, -30, 50, 15),
    "concussion": (-55, 20, -45, 55, 20),
    "antibiotic": (20, 0, 15, 15, 0),
    "painkiller": (40, -20, 25, 20, -20),
    "aspirin": (15, -5, 10, 10, -5),
    "bandage": (20, -5, 10, 5, -5),
    "stitches": (-15, 20, -10, 30, 5),
    "wheelchair": (-10, -10, -20, 5, 25),
    "crutches": (-15, 10, -15, 10, 10),
    "paralyzed": (-75, -20, -50, 10, 40),
    "coma": (-80, -40, -50, 50, 40),
    "stroke": (-75, 55, -50, 60, 35),
    "heartattack": (-80, 60, -50, 60, 40),
    "pizza": (60, 25, 10, 20, 10),
    "burger": (50, 20, 10, 15, 15),
    "salad": (40, -5, 15, 0, -20),
    "burnt": (-40, 20, -15, 10, 10),
    "bland": (-25, -20, -5, 0, 5),
    "sushi": (55, 20, 10, 10, -10),
    "steak": (60, 20, 15, 5, 25),
    "chocolate": (70, 25, 10, 15, -10),
    "cake": (65, 20, 10, 5, -15),
    "vegan": (20, 5, 15, 0, -10),
    "takeout": (40, 20, 10, 35, 5),
    "appetizer": (40, 20, 10, 10, -5),
    "dessert": (70, 25, 10, 5, -20),
    "bluetooth": (20, 5, 10, 10, -5),
    "phone": (40, 20, 30, 15, 0),
    "battery": (10, 10, 15, 40, 5),
    "charger": (25, 10, 20, 35, 5),
    "screen": (15, 5, 10, 5, 0),
    "buffering": (-45, 30, -30, 45, 10),
    "app": (25, 10, 15, 10, -5),
    "website": (20, 5, 15, 5, 0),
    "browser": (15, 5, 15, 5, 0),
    "server": (10, 15, 20, 20, 20),
    "database": (10, 5, 25, 10, 25),
    "code": (20, 20, 30, 15, 5),
    "debug": (5, 35, 20, 45, 10),
    "deploy": (45, 50, 35, 50, 10),
    "lit": (65, 55, 30, 5, -30),
    "slay": (70, 45, 40, 0, -25),
    "mid": (-25, -20, -5, 0, 5),
    "cap": (-40, 25, -10, 10, 10),
    "bussin": (75, 40, 25, 5, -20),
    "cringe": (-60, 35, -25, 10, 15),
    "vibe": (50, -10, 20, 0, -20),
    "finna": (5, 20, 10, 35, 0),
    "sus": (-45, 40, -15, 30, 10),
    "deadass": (30, 45, 35, 20, 20),
    "bet": (35, 20, 25, 10, -5),
    "lowkey": (20, -15, 10, 0, -10),
    "highkey": (40, 35, 20, 10, -10),
    "shook": (-20, 55, -35, 10, 5),
    "salty": (-45, 30, -15, 5, 10),
    "basic": (-30, -15, -10, 0, 5),
    "extra": (-25, 30, -10, 5, 5),
    "mood": (30, -5, 15, 0, -10),
    "flex": (35, 30, 35, 5, -5),
    "ratio": (-50, 40, 20, 15, 5),
    "copium": (-35, -5, -20, 5, 10),
    "based": (50, 15, 40, 0, 10),
    "yeet": (40, 50, 25, 15, -35),
    "bruh": (-10, 10, -5, 5, 10),
    "oof": (-30, 15, -20, 5, 10),
    "rip": (-45, 10, -25, 5, 20),
    "goat": (80, 40, 50, 0, -20),
    "simp": (-50, 20, -40, 5, 10),
    "stan": (25, 45, 10, 5, -10),
    "karen": (-65, 50, 15, 20, 25),
    "boomer": (-35, 15, 10, 5, 20),
    "zoomer": (10, 25, 5, 5, -10),
    "chad": (40, 20, 50, 0, 15),
    "npc": (-55, -20, -45, 0, 20),
    "meeting": (-15, 10, -10, 40, 15),
    "email": (0, 15, 10, 45, 0),
    "presentation": (-10, 50, 10, 55, 10),
    "interview": (-10, 55, -20, 55, 15),
    "salary": (50, 20, 40, 10, -10),
    "layoff": (-75, 50, -50, 50, 40),
    "retire": (65, -30, 50, 0, -40),
    "office": (-5, 5, 5, 5, 15),
    "remote": (55, -20, 45, 0, -25),
    "project": (15, 25, 20, 40, 10),
    "client": (5, 30, 5, 45, 10),
    "manager": (0, 20, 35, 30, 20),
    "intern": (10, 30, -30, 25, 5),
    "startup": (40, 55, 20, 50, -10),
    "corporate": (-20, 5, 25, 10, 30),
    "hustle": (30, 50, 40, 45, 5),
    "grind": (-10, 40, 30, 40, 20),
    "networking": (25, 40, 15, 25, 5),
    "invest": (50, 30, 35, 10, 0),
    "loan": (-30, 30, -30, 40, 25),
    "mortgage": (-25, 20, -20, 20, 40),
    "credit": (10, 10, 20, 15, 5),
    "tax": (-55, 30, -40, 50, 30),
    "wealthy": (80, 20, 50, 0, -40),
    "profit": (70, 40, 45, 10, -30),
    "paycheck": (60, 40, 35, 50, -20),
    "tip": (40, 15, 20, 5, -10),
    "donation": (65, 15, 30, 5, -25),
    "charity": (70, 10, 30, 5, -20),
    "inheritance": (75, 35, 45, 10, -30),
    "lottery": (80, 60, 10, 10, -40),
    "jackpot": (80, 60, 40, 5, -40),
    "crush": (60, 55, -20, 15, -25),
    "date": (55, 50, 15, 35, -15),
    "breakup": (-80, 45, -50, 10, 40),
    "marriage": (70, 30, 40, 5, 30),
    "ex": (-35, 30, -20, 10, 15),
    "boyfriend": (65, 40, 25, 10, -15),
    "girlfriend": (65, 40, 25, 10, -15),
    "husband": (70, 20, 35, 5, 15),
    "wife": (70, 20, 35, 5, 15),
    "partner": (65, 15, 35, 5, 5),
    "soulmate": (80, 45, 40, 0, -40),
    "flirt": (55, 50, 20, 25, -20),
    "cuddle": (75, -25, 25, 5, -35),
    "makeup": (50, 30, 30, 10, -20),
    "sore": (-40, 10, -20, 15, 20),
    "sweating": (-25, 35, -20, 30, 10),
    "cramp": (-50, 35, -40, 45, 20),
    "itch": (-40, 30, -35, 45, 5),
    "yawn": (-10, -30, -10, 5, -5),
    "shiver": (-35, 45, -40, 30, 5),
    "faint": (-65, -30, -50, 55, 20),
    "unconscious": (-75, -50, -50, 60, 40),
    "asleep": (40, -50, 20, 0, -30),
    "restless": (-45, 50, -40, 35, 10),
    "wtf": (-40, 55, -20, 55, 5),
    "stfu": (-75, 55, 25, 60, 10),
    "fml": (-70, 20, -50, 30, 35),
    "dumbass": (-70, 40, 15, 20, 15),
    "jackass": (-70, 40, 15, 15, 15),
    "horseshit": (-70, 40, 10, 25, 20),
    "dipshit": (-70, 35, 10, 15, 15),
    "shitty": (-65, 35, -10, 20, 20),
    "fucking": (-50, 60, 30, 45, 5),
    "fucked": (-75, 50, -45, 50, 30),
    "screwed": (-65, 45, -40, 50, 25),
    "pissed": (-70, 55, 20, 45, 15),
    "poser": (-60, 25, -25, 5, 10),
    "wannabe": (-55, 25, -30, 5, 5),
    "sellout": (-75, 40, -10, 20, 20),
    "clinic": (10, 15, 15, 30, 15),
    "patient": (5, 10, -30, 20, 15),
    "bloodpressure": (0, 25, 10, 35, 10),
    "pulse": (5, 35, 10, 40, -5),
    "fatal": (-80, 45, -50, 55, 40),
    "trauma": (-80, 55, -50, 40, 35),
    "torture": (-80, 60, -50, 55, 40),
    "misery": (-80, -10, -50, 10, 40),
    "depression": (-75, -40, -50, 10, 40),
    "outrage": (-70, 55, 30, 45, 15),
    "tea": (45, -20, 20, 5, -15),
    "water": (60, -5, 35, 50, -10),
    "juice": (50, 15, 20, 10, -10),
    "soda": (35, 25, 15, 15, -10),
    "milk": (45, -10, 25, 15, 0),
    "bread": (55, -5, 30, 20, 5),
    "butter": (50, 5, 20, 10, 5),
    "cheese": (65, 15, 25, 10, 10),
    "eggs": (55, 10, 25, 25, 5),
    "bacon": (70, 35, 20, 25, 15),
    "chicken": (55, 10, 25, 15, 10),
    "beef": (55, 10, 25, 15, 20),
    "pork": (50, 10, 25, 15, 15),
    "rice": (50, 0, 30, 15, 5),
    "pasta": (65, 10, 30, 15, 10),
    "apple": (65, 5, 30, 5, -15),
    "banana": (60, 5, 30, 10, -15),
    "orange": (65, 15, 30, 5, -15),
    "berry": (70, 15, 25, 5, -20),
    "vegetable": (65, 0, 35, 5, -15),
    "potato": (60, 5, 35, 15, 10),
    "onion": (20, 15, 20, 10, 5),
    "garlic": (40, 20, 20, 10, 0),
    "salt": (25, 15, 30, 5, 5),
    "sugar": (55, 25, 15, 15, -10),
    "honey": (70, -5, 25, 5, -15),
    "soup": (60, -15, 35, 30, 10),
    "sandwich": (55, 10, 30, 25, 5),
    "cookie": (75, 25, 20, 15, -15),
    "donut": (70, 30, 15, 20, -10),
    "candy": (65, 40, 10, 15, -20),
    "laptop": (45, 20, 35, 30, 15),
    "tablet": (40, 15, 30, 20, 5),
    "mouse": (20, 10, 25, 5, -10),
    "keyboard": (25, 15, 30, 15, 5),
    "monitor": (30, 10, 30, 5, 15),
    "printer": (5, 20, 20, 35, 20),
    "router": (15, 10, 30, 45, 10),
    "internet": (50, 20, 30, 40, -10),
    "social": (35, 35, 10, 10, -10),
    "media": (10, 30, 15, 15, 5),
    "profile": (20, 15, 25, 10, 5),
    "account": (25, 15, 30, 25, 5),
    "login": (15, 20, 30, 40, 5),
    "logout": (10, 15, 35, 20, 5),
    "google": (45, 10, 40, 15, -5),
    "stream": (45, 20, 25, 10, -15),
    "video": (50, 25, 20, 10, -5),
    "audio": (40, 15, 20, 5, -5),
    "camera": (50, 25, 35, 15, 10),
    "lens": (30, 10, 25, 5, 5),
    "flash": (20, 50, 20, 35, -15),
    "focus": (35, 40, 45, 25, -10),
    "capture": (40, 35, 35, 15, -5),
    "storage": (25, 10, 30, 15, 10),
    "system": (15, 10, 30, 15, 20),
    "software": (30, 15, 35, 15, 5),
    "hardware": (25, 10, 35, 10, 25),
    "firewall": (30, 10, 40, 25, 15),
    "encrypt": (35, 15, 45, 30, 10),
    "decrypt": (35, 30, 40, 40, 5),
    "hazard": (-60, 55, -40, 60, 20),
    "patrol": (10, 30, 40, 30, 15),
    "cop": (-10, 40, 30, 45, 15),
    "officer": (15, 25, 40, 35, 20),
    "court": (-10, 40, 20, 45, 35),
    "law": (10, 10, 50, 20, 30),
    "rule": (0, 15, 45, 20, 20),
    "liberty": (75, 35, 50, 5, -40),
    "deceit": (-80, 40, -15, 20, 20),
    "honesty": (75, 5, 45, 0, -15),
    "ally": (60, 30, 40, 20, 10),
    "busy": (-15, 50, 10, 60, 10),
    "exciting": (80, 60, 40, 10, -40),
    "holiday": (80, 20, 45, 10, -40),
    "vacation": (80, -10, 50, 5, -45),
    "explore": (70, 50, 45, 10, -25),
    "gone": (-60, -10, -50, 30, 35),
    "here": (20, -5, 30, 0, 0),
    "everywhere": (25, 35, 20, 10, -15),
    "nowhere": (-45, -20, -40, 0, 20),
    "sometimes": (10, 0, 10, 0, 5),
    "often": (20, 15, 10, 0, 0),
    "rarely": (-10, 5, -10, 0, 10),
    "quick": (45, 50, 30, 55, -25),
    "jump": (55, 55, 35, 20, -40),
    "sit": (30, -35, 30, 0, 10),
    "stand": (35, 10, 40, 10, 20),
    "exist": (20, 0, 30, 0, 20),
    "become": (40, 35, 40, 15, -10),
    "today": (40, 30, 40, 55, 0),
    "tomorrow": (50, 35, 35, 40, -10),
    "yesterday": (10, -10, 20, 10, 15),
    "later": (-10, -10, 10, 35, 10),
    "noon": (40, 15, 40, 30, 0),
    "afternoon": (45, 5, 35, 20, 5),
    "evening": (60, -20, 40, 15, 15),
    "midnight": (30, 5, 20, 20, 35),
    "dawn": (70, 35, 40, 25, -35),
    "dusk": (50, -5, 35, 15, 10),
    "sea": (75, 30, 50, 10, -30),
    "tree": (70, -20, 40, 0, 25),
    "lion": (65, 50, 50, 15, 25),
    "tiger": (60, 55, 50, 20, 25),
    "bear": (45, 40, 50, 20, 35),
    "snake": (-50, 55, -30, 50, 5),
    "insect": (-40, 25, -20, 15, -10),
    "spider": (-55, 50, -30, 45, 5),
    "ant": (10, 20, 15, 10, -10),
    "bee": (30, 45, 20, 40, -15),
    "meat": (55, 10, 30, 10, 15),
    "huge": (35, 40, 50, 15, 40),
    "tiny": (50, 20, -30, 10, -40),
    "wide": (30, 10, 40, 5, 20),
    "narrow": (-15, 25, -20, 15, 10),
    "shallow": (-10, 10, -15, 5, -10),
    "noisy": (-55, 60, -20, 55, 10),
    "poor": (-70, 10, -50, 30, 35),
    "false": (-65, 30, -20, 15, 15),
    "goodbye": (-15, 10, 20, 15, 15),
    "feel": (40, 30, 25, 10, -5),
    "dislike": (-55, 20, -15, 10, 10),
    "frown": (-50, 15, -20, 5, 10),
    "boy": (40, 50, -10, 15, -25),
    "girl": (45, 45, -10, 15, -30),
    "soul": (75, 15, 50, 5, -50),
    "spirit": (60, 45, 35, 10, -50),
    # ── Gemini batch 2: expanded medical, slang, finance, body, profanity ──
    "insulin": (20, -10, 20, 45, 0), "chemotherapy": (-75, 45, -50, 50, 40),
    "ventilator": (-50, 50, -50, 60, 30), "biopsy": (-30, 40, -20, 50, 20),
    "autopsy": (-70, 10, 10, 10, 40), "migraine": (-60, 30, -45, 45, 20),
    "arthritis": (-50, 10, -35, 10, 30), "diabetes": (-45, 20, -30, 30, 20),
    "asthma": (-40, 45, -45, 55, 15), "epilepsy": (-60, 40, -40, 30, 25),
    "vertigo": (-45, 35, -50, 45, 10), "paranoia": (-70, 55, -50, 40, 15),
    "schizophrenia": (-75, 50, -50, 30, 35), "bipolar": (-60, 55, -45, 30, 20),
    "therapist": (50, -15, 20, 15, -10), "psychiatrist": (40, 10, 25, 20, 5),
    "surgeon": (35, 45, 45, 40, 15), "cardiologist": (40, 20, 40, 30, 10),
    "oncologist": (30, 30, 35, 40, 20), "pediatrician": (55, 15, 30, 20, -10),
    "paramedic": (45, 55, 35, 60, 5), "stethoscope": (20, 5, 20, 10, 5),
    "scalpel": (-10, 45, 30, 40, 10), "syringe": (-25, 40, 10, 45, 5),
    "morphine": (50, -50, 30, 40, -20), "fentanyl": (-60, -50, -40, 60, 40),
    "detox": (20, 35, 15, 40, 10), "hospice": (-40, -30, -20, 10, 35),
    "morgue": (-80, -10, -10, 0, 40), "pathogen": (-70, 30, -10, 40, 20),
    "immunity": (70, -5, 45, 5, -20), "placebo": (10, -15, 0, 0, -10),
    "anesthesia": (30, -50, 20, 40, 5), "remission": (75, 10, 45, 5, -30),
    "ramen": (55, 15, 20, 20, 10), "taco": (60, 25, 15, 25, -5),
    "curry": (55, 35, 15, 15, 10), "stew": (50, -10, 30, 15, 20),
    "grill": (35, 30, 30, 20, 5), "fry": (30, 35, 20, 25, 5),
    "steam": (35, -5, 20, 10, -10), "poach": (40, -10, 25, 15, -15),
    "whisk": (20, 25, 15, 10, -5), "fork": (10, 5, 15, 5, 5),
    "spoon": (15, -5, 15, 5, 0), "plate": (15, -10, 15, 5, 10),
    "bowl": (20, -10, 20, 5, 15), "mug": (30, -10, 25, 5, 10),
    "pepper": (20, 35, 10, 5, -5), "ginger": (35, 25, 15, 5, -10),
    "basil": (45, -5, 20, 0, -20), "thyme": (40, -10, 20, 0, -15),
    "rosemary": (45, -10, 20, 0, -15), "cilantro": (30, 15, 10, 0, -10),
    "parsley": (25, 0, 10, 0, -5), "lobster": (65, 30, 20, 10, 20),
    "shrimp": (55, 20, 15, 15, -10), "salmon": (60, 10, 30, 10, 5),
    "tofu": (30, -10, 15, 5, -5), "sausage": (45, 20, 20, 15, 15),
    "omelet": (55, 15, 25, 35, 5), "pancake": (65, 20, 15, 30, -15),
    "waffle": (65, 20, 15, 30, -10), "syrup": (50, 10, 10, 15, 20),
    "bagel": (55, 10, 25, 25, 10), "muffin": (60, 15, 20, 20, -5),
    "pastry": (70, 25, 15, 15, -20), "croissant": (75, 20, 15, 15, -30),
    "baguette": (60, 5, 25, 10, 10), "yogurt": (45, -5, 20, 15, -10),
    "cereal": (40, 10, 20, 40, -5), "AI": (30, 45, 40, 20, -15),
    "robot": (25, 25, 35, 15, 20), "drone": (20, 40, 30, 30, -35),
    "sensor": (15, 15, 25, 15, 5), "signal": (20, 25, 25, 35, -5),
    "cable": (10, 5, 20, 15, 15), "wire": (5, 10, 15, 20, 10),
    "chip": (25, 15, 35, 15, 5), "processor": (30, 20, 40, 10, 10),
    "folder": (15, -5, 20, 5, 5), "file": (15, 0, 20, 10, 0),
    "document": (10, 5, 25, 15, 5), "image": (45, 15, 20, 5, -5),
    "broadcast": (35, 40, 30, 25, 5), "antenna": (10, 10, 20, 15, 10),
    "satellite": (45, 10, 40, 10, -40), "orbit": (40, 5, 40, 5, -50),
    "rocket": (60, 60, 45, 40, -45), "engine": (30, 45, 40, 15, 35),
    "circuit": (20, 20, 30, 10, 10), "pixel": (30, 15, 15, 0, -25),
    "vector": (25, 10, 25, 0, -15), "framework": (25, 10, 35, 5, 15),
    "library": (40, -10, 35, 5, 10), "runtime": (10, 35, 25, 40, 5),
    "interface": (20, 15, 30, 15, 5), "backend": (15, 10, 40, 15, 25),
    "frontend": (40, 20, 30, 15, -10), "fullstack": (50, 45, 45, 30, 10),
    "prototype": (35, 40, 25, 35, 0), "version": (10, 5, 25, 15, 5),
    "patch": (25, 20, 20, 40, 0), "security": (60, 15, 45, 30, 20),
    "auth": (25, 15, 35, 35, 5), "skibidi": (-10, 55, -20, 5, -15),
    "fanum": (20, 40, 0, 10, -10), "gyatt": (40, 55, 10, 5, -20),
    "mog": (35, 45, 45, 5, -5), "mew": (20, 10, 35, 0, -10),
    "aura": (60, 20, 50, 0, -35), "ohio": (-30, 35, -15, 5, 10),
    "grimace": (-40, 30, -25, 5, 15), "pookie": (75, 20, 20, 5, -30),
    "bop": (-45, 40, -30, 10, -15), "opp": (-75, 60, 0, 50, 15),
    "glizzy": (30, 35, 0, 15, -10), "tweaking": (-50, 60, -40, 45, 10),
    "delusional": (-60, 35, -45, 10, 15), "gatekeep": (-55, 25, 30, 15, 15),
    "gaslight": (-80, 50, 35, 25, 25), "girlboss": (40, 50, 45, 10, -20),
    "malewife": (30, -20, -30, 0, -10), "core": (20, 10, 25, 0, -5),
    "coded": (35, 15, 30, 0, -15), "serving": (65, 45, 40, 5, -25),
    "ate": (70, 55, 40, 5, -25), "lore": (40, 15, 25, 5, 10),
    "ratioed": (-70, 55, -40, 40, 15), "maincharacter": (45, 40, 50, 10, -25),
    "villainera": (30, 55, 50, 15, 15), "softlaunch": (40, 20, 20, 25, -15),
    "hardlaunch": (60, 50, 45, 50, -10), "sneaky": (-20, 45, 20, 35, -5),
    "situationship": (-45, 40, -35, 20, 15), "beigeflag": (-15, -10, -15, 0, 5),
    "invoice": (0, 20, 25, 55, 10), "compliance": (10, -10, 35, 30, 20),
    "strategy": (45, 35, 45, 20, 10), "marketing": (30, 45, 30, 35, -10),
    "sales": (35, 55, 35, 55, -5), "logistics": (20, 30, 30, 45, 25),
    "inventory": (15, 15, 25, 40, 30), "warehouse": (5, 25, 20, 35, 40),
    "shipping": (25, 35, 25, 55, 10), "receiving": (30, 25, 25, 50, 5),
    "headquarters": (35, 15, 50, 10, 40), "subsidiary": (10, 5, 20, 5, 15),
    "merger": (30, 50, 45, 40, 35), "acquisition": (35, 55, 50, 45, 35),
    "shares": (40, 45, 35, 25, 10), "dividend": (65, 30, 40, 15, -15),
    "stakeholder": (20, 25, 40, 25, 25), "boardroom": (15, 35, 50, 35, 40),
    "ceo": (50, 55, 50, 45, 40), "founder": (60, 60, 50, 35, 30),
    "equity": (55, 30, 45, 15, 10), "vesting": (45, 25, 35, 10, 15),
    "annuity": (40, -15, 40, 5, 20), "severance": (10, 15, 25, 45, 25),
    "workload": (-45, 40, -30, 50, 35), "sickleave": (-20, -10, -20, 40, 10),
    "maternity": (60, 25, 30, 45, 35), "demotion": (-75, 40, -50, 35, 40),
    "appraisal": (10, 50, -10, 55, 15), "feedback": (25, 30, 15, 40, 5),
    "wealth": (80, 25, 50, 5, -40), "poverty": (-80, 15, -50, 40, 45),
    "currency": (20, 10, 35, 20, 15), "inflation": (-55, 45, -40, 55, 30),
    "deflation": (-45, 30, -35, 40, 20), "interest": (10, 20, 35, 30, 15),
    "principal": (15, 5, 40, 10, 25), "escrow": (20, 10, 40, 25, 35),
    "liquidity": (50, 15, 40, 30, -10), "insolvency": (-75, 50, -50, 55, 40),
    "capital": (45, 20, 50, 10, 35), "revenue": (60, 35, 45, 35, 20),
    "expenditure": (-20, 25, 20, 30, 20), "margin": (25, 30, 35, 40, 10),
    "arbitrage": (40, 50, 45, 55, -5), "portfolio": (40, 15, 45, 10, 25),
    "volatile": (-30, 60, -20, 55, 0), "bullish": (65, 55, 45, 30, -30),
    "bearish": (-65, 45, -40, 25, 35), "recession": (-70, 40, -45, 45, 40),
    "stimulus": (55, 45, 35, 50, -25), "bailout": (30, 50, 30, 60, 20),
    "treasury": (45, 10, 50, 20, 40), "premium": (30, 20, 40, 30, 10),
    "deductible": (-25, 25, -20, 40, 15), "coverage": (55, -5, 45, 15, -15),
    "audit": (-35, 50, -20, 55, 25), "accounting": (10, 5, 40, 25, 15),
    "fiduciary": (50, 10, 50, 20, 35), "acquaintance": (10, 5, 10, 5, 5),
    "colleague": (25, 15, 25, 15, 10), "confidant": (75, 25, 45, 10, -20),
    "mentor": (70, 30, 45, 10, 0), "protege": (60, 40, -10, 20, -10),
    "ancestor": (40, -10, 40, 0, 40), "descendant": (50, 25, 15, 5, -20),
    "sibling": (60, 25, 30, 15, 15), "cousin": (45, 15, 20, 10, 10),
    "nephew": (55, 30, 10, 15, -20), "niece": (55, 30, 10, 15, -25),
    "stepmother": (20, 20, 30, 20, 25), "stepfather": (20, 20, 35, 20, 30),
    "guardian": (65, 15, 50, 30, 25), "ward": (25, 25, -35, 40, 5),
    "rivalry": (-35, 55, 15, 45, 10), "animosity": (-70, 50, 10, 30, 20),
    "adoration": (85, 55, -20, 10, -45), "society": (25, 15, 45, 5, 30),
    "tribe": (40, 35, 40, 15, 15), "clique": (-30, 30, 25, 15, 10),
    "mob": (-65, 65, -10, 60, 15), "audience": (40, 40, 15, 10, -5),
    "following": (50, 45, 30, 15, -10), "fanbase": (60, 55, 25, 10, -15),
    "adversary": (-70, 55, 10, 45, 25), "nemesis": (-85, 60, 20, 50, 35),
    "archrival": (-75, 60, 15, 45, 30), "fling": (45, 55, 10, 30, -25),
    "heart": (65, 45, 45, 55, 10), "brain": (70, 50, 50, 15, -15),
    "lungs": (60, 40, 45, 60, -10), "liver": (45, 20, 40, 10, 25),
    "kidney": (45, 20, 40, 10, 20), "stomach": (35, 30, 35, 45, 15),
    "bone": (40, 15, 50, 5, 40), "skeleton": (-30, 35, 30, 10, 40),
    "skull": (-40, 40, 35, 10, 40), "spine": (50, 35, 50, 10, 40),
    "nerve": (25, 55, 20, 50, -5), "vein": (20, 35, 25, 45, 5),
    "artery": (25, 40, 30, 55, 10), "skin": (55, 10, 35, 5, 0),
    "hormone": (10, 55, 15, 30, -5), "metabolism": (40, 45, 40, 15, -5),
    "reflex": (35, 60, 45, 60, -25), "sensation": (55, 50, 30, 10, -30),
    "perception": (50, 35, 40, 5, -15), "consciousness": (80, 45, 50, 5, -50),
    "stamina": (70, 50, 50, 15, 10), "fatigue": (-65, -35, -50, 15, 35),
    "vigilance": (40, 60, 50, 60, 10), "agility": (70, 55, 45, 15, -45),
    "coordination": (60, 40, 45, 20, 0), "posture": (40, 10, 40, 5, 25),
    "physique": (65, 40, 45, 5, 25), "pussy": (-70, 55, 10, 30, -15),
    "faggot": (-95, 65, 20, 50, 20), "nigger": (-95, 70, 20, 60, 30),
    "retard": (-90, 60, 20, 40, 25), "whore": (-85, 55, 10, 30, 10),
    "slut": (-80, 55, 10, 25, 5), "fuckhead": (-80, 55, 20, 35, 15),
    "shitbird": (-70, 45, 10, 25, 10), "asswipe": (-70, 40, 10, 25, 15),
    "douchebag": (-75, 50, 15, 25, 10), "scumbag": (-85, 50, 20, 30, 25),
    "jackoff": (-70, 40, 10, 20, 10), "jerkoff": (-70, 40, 10, 20, 10),
    "bugger": (-40, 35, 5, 15, 5), "shite": (-55, 40, 5, 25, 15),
    "knob": (-50, 30, 5, 15, 10), "bellend": (-70, 45, 10, 20, 10),
    "cock": (-65, 50, 10, 30, 15), "dickhead": (-80, 55, 15, 30, 15),
    "tit": (-20, 35, 0, 10, -10), "tits": (-20, 40, 0, 15, -15),
    "boob": (-10, 30, 0, 10, -20), "balls": (-40, 45, 10, 20, 10),
    "nutsack": (-50, 40, 5, 15, 15), "arse": (-50, 35, 10, 15, 15),
    "shag": (10, 50, 20, 30, -10), "fuckwit": (-80, 45, 15, 25, 15),
    "shithead": (-80, 50, 15, 30, 15), "assface": (-75, 45, 10, 25, 15),
    "fuckface": (-85, 60, 20, 40, 15), "degenerate": (-85, 40, -40, 20, 35),
    "sociopath": (-90, 55, 30, 45, 35), "psychopath": (-90, 65, 35, 55, 35),
    "narcissist": (-85, 50, 40, 35, 30), "egomaniac": (-80, 55, 45, 30, 25),
    "backstabber": (-90, 60, -20, 40, 25), "leech": (-80, 30, -35, 15, 25),
    "parasite": (-85, 35, -40, 20, 30), "vermin": (-85, 40, -45, 25, 35),
    "pest": (-50, 45, -20, 45, 5), "pervert": (-85, 55, -20, 50, 25),
    "savage": (20, 60, 45, 40, 20), "barbarian": (-65, 55, 35, 30, 30),
    "vandal": (-70, 60, 10, 50, 15), "thug": (-75, 60, 20, 55, 20),
    "delinquent": (-60, 45, -15, 40, 20), "outcast": (-65, -15, -45, 5, 25),
    "pariah": (-80, -20, -50, 5, 35), "blight": (-75, 20, -30, 10, 35),
    "scourge": (-85, 55, 30, 40, 40), "plague": (-90, 60, -40, 60, 40),
    "venom": (-75, 60, 30, 50, 15), "cancerous": (-85, 45, -20, 30, 30),
    "radioactive": (-80, 65, 20, 60, 15), "corrosive": (-75, 45, 10, 40, 20),
    "vile": (-85, 40, -20, 20, 35), "luminescent": (80, 30, 30, 0, -50),
    "transcendent": (90, 40, 50, 5, -55), "ethereal": (80, -20, 40, 0, -60),
    "divine": (90, 35, 50, 5, -55), "angelic": (85, -10, 40, 0, -50),
    "virtuous": (80, 10, 45, 0, 15), "exemplary": (80, 25, 45, 5, 10),
    "paragon": (85, 30, 50, 5, 25), "prodigy": (80, 55, 45, 10, -25),
    "visionary": (85, 60, 50, 15, -35), "pioneer": (75, 55, 45, 20, -15),
    "landmark": (70, 20, 50, 5, 45), "pinnacle": (85, 35, 50, 5, 50),
    "zenith": (85, 30, 50, 5, 50), "apex": (80, 35, 50, 5, 45),
    "ultimate": (85, 55, 50, 10, 30), "sovereign": (75, 20, 50, 10, 40),
    "majestic": (85, 30, 50, 0, 45), "noble": (80, 15, 50, 5, 30),
    "dignified": (75, -5, 45, 0, 25), "stoic": (60, -40, 50, 0, 35),
    "steadfast": (75, 10, 50, 5, 35), "tenacious": (75, 55, 50, 15, 15),
    "formidable": (65, 55, 50, 25, 40), "indomitable": (85, 45, 50, 10, 45),
    "unwavering": (75, 15, 50, 5, 35), "flawless": (90, 15, 50, 0, -35),
    "immaculate": (85, -5, 50, 5, -30), "pristine": (80, -15, 45, 5, -25),
    "exquisite": (85, 35, 40, 0, -35), "sublime": (90, -10, 50, 0, -50),
    "stupendous": (80, 55, 45, 10, -25), "breathtaking": (90, 65, 40, 20, -45),
    "enchanting": (80, 45, 30, 5, -40), "annihilate": (-90, 75, 50, 65, 45),
    "exterminate": (-90, 70, 50, 65, 40), "eradicate": (-85, 65, 45, 60, 35),
    "obliterate": (-90, 75, 50, 65, 45), "massacre": (-95, 80, 50, 70, 45),
    "genocide": (-100, 85, 45, 75, 50), "carnage": (-90, 75, 30, 60, 40),
    "assassination": (-90, 75, 45, 75, 35), "homicide": (-90, 70, 40, 65, 35),
    "casualty": (-80, 45, -50, 40, 35), "fatality": (-85, 50, -50, 50, 40),
    "atrocity": (-95, 75, 10, 50, 45), "brutality": (-85, 65, 35, 55, 35),
    "persecution": (-85, 55, 35, 50, 40), "oppression": (-80, 45, 45, 40, 45),
    "tyranny": (-85, 55, 50, 50, 50), "dictatorship": (-80, 50, 50, 45, 50),
    "conquest": (20, 65, 50, 55, 40), "invasion": (-75, 70, 45, 65, 35),
    "bombardment": (-85, 75, 40, 70, 40), "airstrike": (-80, 75, 45, 75, 20),
    "missile": (-75, 70, 45, 70, 25), "explosive": (-70, 80, 40, 75, 15),
    "detonation": (-80, 85, 45, 80, 25), "shrapnel": (-75, 65, -20, 60, 15),
    "grenade": (-70, 75, 35, 70, 20), "bayonet": (-65, 60, 35, 60, 20),
    "artillery": (-75, 70, 45, 65, 45), "cavalry": (40, 65, 45, 55, 40),
    "infantry": (20, 55, 40, 50, 35), "warfare": (-85, 75, 40, 65, 40),
    "bloodshed": (-90, 75, 20, 65, 30), "refrigerator": (45, 5, 40, 15, 40),
    "dishwasher": (40, 10, 40, 20, 35), "microwave": (25, 20, 35, 45, 20),
    "toaster": (20, 15, 30, 40, 10), "blender": (25, 45, 30, 35, 15),
    "vacuum": (15, 55, 35, 30, 30), "closet": (15, -10, 20, 5, 20),
    "dresser": (20, -10, 25, 5, 30), "curtains": (30, -15, 20, 5, -10),
    "carpet": (35, -20, 25, 0, 25), "rug": (30, -15, 20, 0, 20),
    "lamp": (45, 5, 30, 10, 10), "flashlight": (50, 35, 40, 55, 5),
    "hammer": (15, 45, 40, 25, 30), "screwdriver": (15, 25, 35, 20, 15),
    "wrench": (15, 30, 40, 25, 25), "pliers": (10, 25, 35, 20, 10),
    "saw": (-5, 50, 35, 35, 25), "drill": (10, 55, 40, 40, 20),
    "ladder": (20, 40, 30, 45, 35), "toolbox": (40, 20, 45, 15, 40),
    "bucket": (10, 5, 20, 10, 20), "mop": (5, 15, 20, 30, 20),
    "broom": (10, 10, 20, 25, 15), "trashcan": (-25, 15, 20, 45, 25),
    "recycling": (50, 10, 30, 20, 20), "compost": (30, 5, 25, 15, 30),
    "hose": (25, 20, 25, 35, 15), "lawnmower": (20, 60, 40, 40, 40),
    "shovel": (10, 40, 30, 35, 30), "rake": (10, 30, 25, 35, 15),
    "trowel": (20, 15, 20, 15, 5), "clippers": (15, 25, 30, 20, 5),
    "gloves": (35, -5, 30, 15, 5), "boots": (40, 10, 40, 15, 35),
    "coat": (55, -10, 40, 10, 20),

    # ── Gemini mega-batch: all lexicon files merged ──
    "aa-lava-sharp": (-45, 70, 40, 60, 65),
    "ablation": (-35, 65, 40, 70, 45),
    "abomination": (-100, 80, -55, 55, 50),
    "abrasive": (-45, 50, 15, 15, 15),
    "abscess": (-65, 40, -45, 50, 25),
    "absconding": (-65, 60, -40, 60, 10),
    "absinthe": (30, 50, -15, 10, -40),
    "absolute": (50, 10, 60, 15, 50),
    "absolute-extract": (95, 35, 55, 10, 45),
    "absolution": (90, -20, 50, 40, -40),
    "abstract": (35, 40, 20, 15, -40),
    "abstraction": (25, 15, 35, 10, -15),
    "abyss": (-60, 20, -50, 10, 40),
    "abyssal": (-60, -10, -40, 5, 80),
    "abyssal-plain": (-55, -25, 45, 5, 100),
    "acceleration": (60, 75, 50, 80, -30),
    "accelerometer-axial": (65, 70, 45, 85, 10),
    "accession-record": (40, 20, 45, 45, 35),
    "accord-blend": (90, 45, 50, 15, 10),
    "accountability": (45, 20, 45, 15, 20),
    "acetaminophen": (20, -10, 15, 15, -5),
    "achondrite-meteor": (45, 10, 45, 5, 50),
    "acid": (-70, 60, 40, 60, 10),
    "acid-compliance": (95, 20, 70, 15, 90),
    "acidic": (-35, 55, 40, 45, 15),
    "acquittal": (75, 50, 45, 30, -25),
    "acropolis": (85, 45, 55, 15, 65),
    "actionable": (60, 45, 50, 60, 20),
    "actuarial-science": (40, 20, 55, 55, 50),
    "actuary": (35, 15, 45, 20, 20),
    "actuator": (25, 40, 40, 35, 10),
    "acutance-sharp": (70, 35, 45, 15, 25),
    "adagio": (60, -40, 25, 0, 10),
    "adamantine": (85, 45, 55, 5, 60),
    "additive": (-30, 15, 10, 15, 10),
    "adenosine": (10, -40, 20, 60, 5),
    "adiabatic": (10, 15, 25, 5, 0),
    "admiral": (75, 30, 60, 40, 55),
    "adrenaline": (30, 65, 40, 50, -25),
    "adularescence": (85, 15, 30, 0, -35),
    "advantage": (85, 55, 60, 30, -35),
    "adversarial-attack": (-95, 85, -35, 90, 25),
    "adze": (35, 55, 45, 40, 45),
    "aeon": (65, -15, 55, 0, 50),
    "aeration": (70, 20, 35, 15, -10),
    "aerodynamics": (40, 15, 40, 5, -20),
    "aeropress-plunge": (75, 45, 45, 50, -5),
    "aesthetics": (85, 35, 45, 5, -30),
    "affidavit": (10, 15, 20, 35, 15),
    "affix": (15, 5, 20, 5, 0),
    "aft": (5, -5, 10, 5, 5),
    "afterburner": (45, 60, 45, 50, -35),
    "aftermath": (-55, 35, -35, 45, 30),
    "aftershock": (-70, 60, -45, 70, 15),
    "agency": (85, 45, 60, 25, 10),
    "agile": (50, 45, 35, 25, -20),
    "agile-sprint": (35, 80, 50, 95, 15),
    "aging-barrel": (80, -10, 50, 5, 55),
    "agitation-tank": (25, 45, 35, 45, 25),
    "agora": (60, 55, 35, 20, 15),
    "agribusiness": (25, 40, 55, 30, 50),
    "aileron": (25, 40, 35, 45, -15),
    "aim-bot": (-100, 60, 50, 40, 20),
    "airball": (-70, 25, -50, 15, 20),
    "airfoil": (30, 10, 30, 5, -40),
    "airframe": (20, 5, 45, 5, 30),
    "alabaster": (70, -15, 40, 0, 15),
    "albedo": (55, 10, 40, 5, -45),
    "alchemist": (55, 40, 45, 15, 15),
    "aldehydes": (55, 70, 30, 35, -60),
    "alexandrite": (80, 50, 45, 10, 35),
    "algor-mortis": (-95, -20, 10, 0, 85),
    "algorithm": (30, 35, 45, 15, 15),
    "algorithmic-bias": (-65, 55, 30, 45, 25),
    "aliasing-artifact": (-75, 65, -35, 60, 10),
    "alienation": (-85, -20, -55, 15, 45),
    "alignment": (80, 15, 55, 45, 35),
    "alkali": (10, 30, 25, 20, 15),
    "alkaline": (40, 10, 40, 15, 15),
    "allegory": (55, 25, 40, 5, 30),
    "allegro": (75, 45, 30, 5, -20),
    "allele": (15, 15, 25, 5, 10),
    "allelopathy": (-65, 45, 50, 35, 10),
    "allergen": (-40, 35, -25, 45, 5),
    "alliteration": (40, 30, 20, 10, -10),
    "allophone": (20, 10, 25, 5, -10),
    "alluvial": (60, 15, 30, 10, 15),
    "alpha-male": (20, 55, 50, 15, 35),
    "alt": (20, 15, 20, 5, -5),
    "alt-season": (85, 80, 45, 50, -40),
    "alterity": (35, 30, 35, 10, 0),
    "altimeter": (15, 35, 40, 55, 10),
    "altostratus": (-10, -5, 15, 10, 5),
    "altruistic": (95, 25, 40, 10, -35),
    "amanita": (-90, 50, 35, 60, 20),
    "ambassador": (75, 15, 50, 25, 25),
    "amber": (55, 10, 25, 0, 10),
    "ambergris": (95, 25, 60, 5, 45),
    "ambiguity": (-25, 15, -30, 20, 5),
    "ambivalence": (-10, -10, -20, 15, 5),
    "ambrotype": (80, 25, 50, 5, 85),
    "amendment": (30, 25, 40, 45, 25),
    "amethyst": (75, 20, 40, 5, 20),
    "amortization": (10, -10, 25, 20, 30),
    "amphibian": (20, 5, 10, 5, -5),
    "amphora": (55, 15, 30, 10, 35),
    "amplification": (75, 75, 50, 60, -20),
    "amplifier": (50, 50, 40, 20, 35),
    "amplitude": (35, 55, 40, 30, 15),
    "amulet": (60, 20, 35, 5, 5),
    "amygdala": (-10, 55, -20, 45, 0),
    "amygdaloidal-filling": (45, 10, 40, 5, 40),
    "analgesic": (35, -20, 25, 10, -10),
    "analytics": (40, 25, 40, 15, 10),
    "anamorphic": (80, 55, 50, 15, 40),
    "anamorphic-lens": (85, 55, 50, 15, 40),
    "anchor": (70, -20, 50, 10, 60),
    "andante": (55, -15, 20, 0, 5),
    "anemia": (-40, -10, -35, 15, 20),
    "anemograph-wind": (45, 55, 50, 60, 35),
    "anemometer": (10, 20, 25, 30, 5),
    "aneurysm": (-80, 55, -50, 60, 35),
    "angiosperm": (75, 35, 40, 10, 10),
    "annealing-heat": (75, -20, 45, 15, 35),
    "anodized": (40, 10, 40, 0, 25),
    "anon": (20, 25, 45, 5, 15),
    "antagonist": (-70, 55, 40, 45, 25),
    "anthropogenic": (-65, 45, 50, 55, 40),
    "anti-woke": (-25, 80, 35, 70, 35),
    "antibodies": (65, 15, 45, 30, -20),
    "antibody": (75, 30, 45, 55, -20),
    "anticline": (35, 5, 40, 5, 50),
    "anticyclogenesis": (55, -25, 50, 30, 45),
    "anticyclone": (50, -25, 45, 10, 10),
    "antidepressant": (40, -25, 30, 20, -10),
    "antigen": (-10, 25, 10, 40, 5),
    "antihistamine": (20, -15, 20, 35, -5),
    "antipsychotic": (30, -35, 35, 25, 0),
    "antiseptic": (25, 5, 20, 20, 0),
    "antonym": (25, 15, 25, 5, 5),
    "anubis": (-20, 35, 40, 5, 45),
    "anvil-horn": (40, 10, 55, 5, 95),
    "anxiety": (-65, 55, -50, 55, 10),
    "anxious-avoidant": (-65, 60, -50, 45, 20),
    "apathy": (-60, -50, -40, 0, 35),
    "ape-in": (40, 85, 15, 70, -20),
    "aperture": (55, 45, 45, 35, 15),
    "aperture-stop": (40, 45, 45, 45, 10),
    "apex-predator": (30, 65, 55, 45, 30),
    "apex-turn": (60, 85, 55, 70, 40),
    "api-gateway": (70, 45, 55, 60, 40),
    "apiary": (65, 15, 40, 5, 30),
    "apogee": (55, 25, 45, 15, 15),
    "aporia": (-35, 55, -30, 60, 15),
    "apostate": (-90, 60, -15, 45, 35),
    "apostle": (80, 40, 45, 20, 30),
    "appellation": (70, 10, 55, 10, 40),
    "appels": (30, 65, 30, 60, 5),
    "appendix": (-5, 15, 20, 45, 15),
    "appliqué": (75, 45, 40, 15, 15),
    "appraisal-archival": (50, 35, 55, 40, 45),
    "appreciation": (75, 35, 45, 10, -30),
    "aquamarine": (90, 15, 35, 5, -40),
    "aquaponics": (75, 30, 40, 15, -10),
    "aqueduct": (85, 25, 55, 15, 65),
    "aqueous": (55, -10, 30, 15, -15),
    "aquifer": (80, 5, 45, 10, 45),
    "arabesque": (75, 35, 25, 5, -35),
    "arabica-highland": (85, 20, 45, 5, 15),
    "arable": (65, -10, 40, 10, 40),
    "arachnid": (-60, 55, 25, 45, 5),
    "arbitrage-trade": (60, 85, 55, 95, 15),
    "arbitration": (25, 10, 35, 25, 15),
    "archetype": (50, 15, 45, 5, 40),
    "archipelago": (75, 15, 35, 0, -20),
    "architrave": (65, 15, 45, 5, 50),
    "archway": (55, 10, 40, 5, 35),
    "arctic": (-25, 35, 25, 15, 35),
    "arepa": (55, 15, 20, 10, 5),
    "argot": (-30, 45, 35, 15, 15),
    "aria": (80, 40, 35, 10, -30),
    "aristotelianism": (65, 15, 55, 10, 45),
    "armistice": (85, -50, 50, 60, -30),
    "armoire": (45, -10, 35, 5, 40),
    "armorial": (55, 15, 45, 5, 40),
    "aroma": (85, 15, 25, 0, -30),
    "aromatic": (40, -10, 35, 5, -15),
    "arse-over-tit": (-35, 60, -25, 55, -10),
    "arsenic": (-95, 40, 35, 50, 30),
    "artesian-well-flow": (90, 25, 60, 40, 55),
    "artifact": (65, 20, 40, 5, 35),
    "artisanal": (65, 10, 35, 0, -5),
    "ascender": (25, 10, 20, 5, -25),
    "ascent": (75, 55, 45, 35, -40),
    "asceticism": (-30, -55, 65, 15, 55),
    "ascocarp": (50, 15, 35, 5, 30),
    "asexual": (20, -25, 30, 5, -10),
    "asgard-realm": (95, 45, 65, 10, 70),
    "aspartame": (-40, 10, -5, 10, 5),
    "asphyxia": (-95, 75, -50, 60, 40),
    "asphyxiation": (-100, 85, -70, 100, 55),
    "ass-clown": (-75, 50, 20, 25, 15),
    "assembler": (20, 15, 35, 20, 15),
    "asset": (55, 35, 50, 15, 25),
    "asshat": (-75, 40, 15, 15, 5),
    "asterism": (90, 40, 40, 5, -25),
    "asteroid": (-20, 40, 10, 35, 30),
    "asthenosphere": (30, 45, 45, 10, 65),
    "asthenosphere-flow": (15, 25, 45, 10, 80),
    "astrolabe": (55, 25, 45, 5, 25),
    "astrolabe-brass": (85, 40, 55, 10, 45),
    "asura-demon": (-75, 65, 45, 50, 40),
    "asylum": (60, 30, 15, 55, -10),
    "asymmetric-warfare": (-80, 65, 30, 55, 35),
    "asymptomatic": (20, -40, 15, 10, 5),
    "asymptote": (-20, 35, 35, 30, 20),
    "ate-and-left-no-crumbs": (100, 90, 65, 35, -55),
    "atelier": (85, 45, 55, 35, 35),
    "athleticism": (80, 65, 60, 15, 15),
    "atmospheric-skipping": (30, 50, 35, 35, -60),
    "atoll": (85, -20, 35, 0, -35),
    "atomic-weight": (25, 10, 45, 5, 50),
    "atomism": (30, 20, 45, 5, 25),
    "atonement": (70, 30, 40, 35, 30),
    "atrium": (90, 30, 45, 10, -40),
    "atrophy": (-75, -20, -50, 15, 40),
    "attachment": (60, 20, 35, 5, 15),
    "attachment-style": (40, 45, 30, 25, 15),
    "attic": (10, -20, -15, 5, 20),
    "attractor-strange": (55, 75, 45, 25, -35),
    "attrition": (-75, 30, 15, 35, 45),
    "attrition-churn": (-85, 45, -50, 55, 45),
    "audible-call": (25, 75, 60, 90, 10),
    "audition": (-15, 65, -50, 60, 15),
    "auditory": (45, 35, 35, 15, 0),
    "auger": (25, 45, 40, 35, 40),
    "august": (70, -10, 50, 5, 40),
    "aurora": (90, 30, 40, 0, -60),
    "aurora-borealis": (95, 40, 45, 5, -65),
    "authentication": (40, 20, 45, 40, 15),
    "authorization": (35, 25, 45, 45, 10),
    "autocracy": (-75, 40, 50, 30, 50),
    "automation": (60, 35, 50, 20, -25),
    "autopsy-report": (-75, 25, 50, 35, 70),
    "availability": (95, 25, 60, 50, 30),
    "avalanche": (-85, 75, -50, 65, 50),
    "avalanche-photo": (75, 90, 55, 80, 35),
    "avant-garde": (70, 60, 40, 15, -35),
    "avatar": (70, 45, 45, 25, -25),
    "avionics": (35, 25, 40, 15, 5),
    "awe": (90, 50, 30, 5, -65),
    "awl": (20, 35, 30, 30, 5),
    "axiom": (40, -10, 50, 5, 50),
    "axon": (40, 55, 35, 15, -35),
    "azimuth-bearing": (25, 30, 45, 45, 10),
    "azure": (85, 15, 30, 0, -45),
    "b-tree": (60, 20, 60, 10, 85),
    "baba-ghanoush": (60, -5, 25, 5, 5),
    "baby-gronk": (25, 45, 10, 15, -15),
    "back-emf-voltage": (-45, 75, 45, 65, 25),
    "backlog": (-45, 40, -20, 50, 20),
    "backpropagation": (40, 60, 45, 45, 15),
    "backrooms": (-75, 55, -55, 65, 60),
    "backstage": (20, 55, 20, 60, 15),
    "backstitch": (45, 35, 40, 40, 15),
    "backward-pawn": (-60, 30, -50, 40, 25),
    "bacteriophage": (35, 65, 50, 45, -45),
    "bags-heavy": (-85, 15, -45, 10, 80),
    "bailiff": (10, 20, 35, 30, 25),
    "baklava": (80, 45, 20, 10, -25),
    "balestra": (65, 85, 50, 75, -40),
    "ballast": (25, -10, 45, 5, 60),
    "ballbearing": (25, 35, 30, 10, 15),
    "ballistic-entry": (-65, 85, 45, 90, 65),
    "ballistics": (-40, 55, 45, 50, 30),
    "ballistics-match": (40, 65, 55, 60, 35),
    "balustrade": (45, -5, 35, 0, 25),
    "band-saw": (40, 60, 50, 60, 65),
    "bandgap-energy": (55, 40, 60, 10, 35),
    "bandwidth": (15, 45, 25, 55, 15),
    "bandwidth-gbps": (90, 60, 60, 35, 15),
    "bankruptcy": (-90, 60, -60, 65, 50),
    "bare": (20, 20, 15, 15, 10),
    "barista-workflow": (65, 65, 50, 80, 15),
    "baritone": (55, 10, 40, 0, 40),
    "barograph-pressure": (40, 15, 55, 55, 50),
    "barometer": (15, 10, 25, 20, 15),
    "baron": (55, 20, 45, 10, 40),
    "baroque": (60, 50, 45, 10, 45),
    "barotrauma": (-95, 75, -60, 80, 45),
    "barrage": (-80, 75, 40, 70, 45),
    "barrow": (-20, 10, 30, 5, 60),
    "bas-relief": (70, 15, 40, 5, 45),
    "base-note": (80, 15, 55, 5, 65),
    "baseline": (40, -10, 50, 5, 20),
    "basement": (-15, -10, -20, 5, 40),
    "basidiocarp": (55, 15, 40, 5, 35),
    "basilica": (75, 20, 50, 10, 60),
    "basilisk": (-85, 65, 40, 60, 20),
    "bass-bar": (55, 15, 50, 5, 65),
    "bastestitch": (20, 10, 20, 55, -5),
    "bastion": (75, 30, 65, 45, 75),
    "batholith": (45, 5, 55, 0, 70),
    "batholith-magma": (40, 15, 65, 5, 90),
    "bathymetry": (55, 15, 45, 15, 65),
    "batik-wax": (85, 35, 40, 15, 15),
    "battalion": (30, 60, 45, 50, 45),
    "battery-mgmt-system": (90, 45, 65, 60, 55),
    "bayer-filter": (75, 35, 50, 15, 10),
    "beading": (90, 55, 35, 20, 20),
    "beaker-fill": (25, 15, 25, 35, 10),
    "beamforming": (85, 70, 65, 50, 15),
    "bear-market": (-90, 45, -50, 50, 70),
    "beat-attack": (60, 75, 55, 75, 25),
    "bedrock": (65, -5, 60, 5, 80),
    "bee-veil": (30, 20, 35, 40, 15),
    "beeswax": (55, -10, 30, 0, 35),
    "behavior-mod": (35, 35, 45, 40, 15),
    "behemoth": (35, 50, 60, 20, 60),
    "bell-end": (-80, 50, 15, 30, 15),
    "belladonna": (-85, 45, 45, 60, 35),
    "bellows-blast": (30, 60, 45, 55, 20),
    "bellows-extension": (40, 20, 45, 15, 55),
    "benchmark": (15, 10, 35, 15, 20),
    "benchmarking": (20, 25, 40, 15, 20),
    "bend-heraldic": (40, 10, 40, 5, 15),
    "benediction": (95, -40, 45, 5, -30),
    "benefactor": (80, 15, 50, 5, -10),
    "beneficiary": (75, 25, 40, 10, -20),
    "benthic": (10, -5, 25, 0, 75),
    "benthic-sediment": (10, -10, 35, 5, 95),
    "bernoulli-equation": (65, 45, 60, 25, -40),
    "bernoulli-trial": (35, 45, 40, 25, 5),
    "beta-male": (-50, -10, -45, 5, 20),
    "bevel": (45, 20, 35, 15, 15),
    "bezel-setting": (75, 30, 45, 25, 20),
    "bias-cut": (60, 45, 35, 25, -25),
    "bias-variance-tradeoff": (25, 45, 50, 35, 30),
    "bibimbap": (65, 30, 20, 15, 15),
    "bibliography": (45, -5, 40, 15, 30),
    "bifrost-bridge": (95, 55, 50, 15, -70),
    "bifurcation-point": (-10, 85, 50, 95, 35),
    "bigot": (-100, 65, -10, 45, 35),
    "bijection": (45, 15, 45, 10, 10),
    "bikeshedding": (-55, 50, -15, 35, 15),
    "bilge": (-55, 15, -20, 5, 30),
    "bilirubin": (-10, 10, -5, 25, 10),
    "binary": (10, 5, 40, 5, 20),
    "bind-maneuver": (75, 60, 60, 70, 45),
    "binding-morocco": (80, 15, 50, 5, 55),
    "biodynamic": (60, 10, 35, 5, -15),
    "biogeography": (75, 35, 50, 15, 45),
    "bioluminescent": (95, 65, 40, 10, -50),
    "biopolitics": (-45, 45, 55, 35, 55),
    "biospeleology": (45, 30, 45, 10, 65),
    "biosphere": (90, 35, 55, 20, 10),
    "bipartisan": (60, 15, 40, 25, 10),
    "biplane": (55, 20, 25, 5, -10),
    "birefringence": (50, 35, 35, 10, -5),
    "birkeland-current": (50, 95, 70, 75, -80),
    "biscuit-joint": (65, 25, 40, 20, 25),
    "bisexual": (55, 35, 35, 10, -15),
    "bit-depth": (75, 35, 55, 20, 25),
    "bitwise-op": (25, 35, 40, 20, 5),
    "black-body": (-30, 15, 55, 5, 85),
    "black-ops": (-60, 75, 55, 70, 45),
    "blackmail": (-95, 65, 45, 60, 35),
    "bladder": (10, 15, 25, 50, 20),
    "blasphemy": (-95, 70, 30, 55, 25),
    "blazar-beam": (55, 100, 70, 45, -65),
    "blazon": (60, 25, 40, 10, 25),
    "blicky": (-80, 85, 45, 95, 30),
    "blind-stitch": (60, 15, 45, 40, 5),
    "blitz-package": (40, 90, 55, 95, 20),
    "blizzard": (-65, 60, -40, 60, 30),
    "blockade": (-70, 45, 40, 55, 45),
    "blockchain": (30, 50, 30, 20, 15),
    "blood-feud": (-95, 70, 30, 45, 40),
    "blood-spatter": (-90, 75, 10, 65, 25),
    "blood-sucker": (-95, 50, -30, 40, 25),
    "bloom-filter": (70, 35, 50, 15, 15),
    "bloomer": (70, 35, 40, 5, -30),
    "blow-off-valve": (45, 70, 30, 45, -20),
    "blueshift": (25, 35, 35, 20, -5),
    "bluetooth-le": (80, 25, 50, 35, -25),
    "blunder": (-95, 80, -60, 65, 30),
    "bobbin": (40, 35, 30, 45, -10),
    "bodice": (55, 20, 40, 10, 30),
    "body-mouthfeel": (65, 15, 45, 5, 55),
    "body-tea": (95, 65, 50, 20, -30),
    "body-text": (30, -15, 35, 5, 5),
    "boilerplate": (15, -20, 30, 5, 25),
    "bokeh": (95, 35, 40, 5, -65),
    "bokeh-effect": (90, 45, 40, 10, -55),
    "bollocks": (-45, 40, 5, 20, 10),
    "bolster": (55, -20, 25, 0, 10),
    "boltzmann-brain": (-20, 55, -45, 5, -55),
    "bonded": (75, 20, 40, 5, 20),
    "bookend-vortex": (-75, 90, 50, 95, 25),
    "boolean-algebra": (45, 15, 55, 10, 60),
    "boom-mic": (40, 45, 40, 55, -25),
    "border": (-5, 35, 45, 45, 20),
    "borscht": (50, 5, 30, 10, 20),
    "boson": (45, 50, 45, 15, -60),
    "bottomless-portafilter": (90, 60, 50, 55, 40),
    "boundary": (15, 10, 40, 25, 25),
    "boundary-layer": (10, 15, 20, 10, 0),
    "bouquet": (90, 30, 40, 5, -30),
    "bourbon": (55, 15, 30, 5, 35),
    "bout": (65, 80, 45, 60, 20),
    "bout-committee": (10, 35, 50, 40, 55),
    "bow-echo-storm": (-85, 100, 55, 100, 35),
    "bow-shock-crossing": (45, 95, 75, 95, -70),
    "bow-wave-interstel": (45, 25, 85, 10, -95),
    "bowsprit": (25, 15, 25, 5, 10),
    "brahma-creator": (85, 30, 70, 10, 60),
    "braking-zone": (-25, 85, 50, 90, 60),
    "bratwurst": (55, 20, 20, 15, 25),
    "brazing-rod": (40, 50, 40, 45, 35),
    "breach": (-85, 80, -45, 75, 15),
    "breadcrumbing": (-70, 30, -45, 15, 20),
    "breccia": (30, 15, 35, 5, 45),
    "brecciated-matrix": (25, 35, 45, 20, 75),
    "briar": (-25, 35, 10, 20, 5),
    "bribery": (-80, 50, 25, 35, 25),
    "brick": (-60, 20, -40, 10, 35),
    "bricolage": (75, 45, 40, 15, -10),
    "bridge-maple": (85, 40, 55, 15, 60),
    "brie": (75, -20, 35, 5, -10),
    "briefcase": (25, 20, 35, 25, 25),
    "brigade": (40, 55, 50, 45, 45),
    "brightness-acidity": (75, 55, 30, 10, -25),
    "brilliance": (95, 60, 45, 15, -45),
    "broadside": (-75, 75, 50, 70, 50),
    "brocade": (85, 30, 55, 10, 60),
    "broken-build": (65, 70, 50, 15, 25),
    "bronchitis": (-45, 20, -30, 35, 15),
    "bronze": (55, 10, 40, 0, 35),
    "bronze-age": (55, 40, 45, 15, 50),
    "brooch": (50, 15, 30, 10, 5),
    "brood-comb": (45, 10, 30, 5, 40),
    "brownian-motion": (20, 65, -15, 35, -15),
    "bruschetta": (65, 20, 20, 10, 5),
    "brushless-dc-motor": (95, 75, 65, 50, 65),
    "brute-force": (-50, 90, 55, 95, 45),
    "brute-force-attack": (-45, 95, 60, 95, 40),
    "bruv": (55, 35, 25, 10, 15),
    "bryology": (60, -10, 40, 5, 15),
    "bryophyte": (45, -15, 25, 0, 10),
    "buff": (70, 40, 45, 25, -20),
    "buffer-solution": (70, -25, 50, 25, 15),
    "buffing-wheel": (65, 50, 35, 30, -5),
    "bulgogi": (70, 35, 25, 15, 15),
    "bulkhead": (5, 0, 40, 5, 45),
    "bull-market": (95, 75, 65, 40, 45),
    "bullion-gold": (95, 25, 65, 40, 95),
    "bullpen-session": (20, 45, 30, 50, 25),
    "bunsen-burner": (20, 60, 35, 50, -5),
    "bunt-sacrifice": (30, 45, 25, 75, 5),
    "bunting": (65, 45, 15, 10, -25),
    "buoy": (30, 20, 15, 40, -30),
    "buoyancy": (75, -20, 50, 30, -50),
    "bureau": (30, 5, 30, 10, 30),
    "bureaucracy": (-50, 10, 20, 25, 40),
    "burlesque": (60, 60, 30, 25, -25),
    "burning-in": (45, 55, 50, 40, 25),
    "burr-grinder": (85, 50, 55, 25, 65),
    "burrata": (80, 20, 30, 10, -25),
    "burrito": (65, 30, 15, 35, 20),
    "bustle": (35, 25, 40, 10, 50),
    "buttress": (45, 10, 55, 10, 65),
    "buy-in": (55, 40, 45, 50, 25),
    "byzantine": (50, 45, 45, 10, 50),
    "caching": (50, 10, 40, 30, -10),
    "cacophony": (-85, 75, -30, 60, 5),
    "cactus": (15, 20, 25, 5, 10),
    "cadaver": (-90, -25, -10, 10, 50),
    "cadence": (55, 15, 35, 10, 10),
    "cadency": (30, 15, 35, 10, 25),
    "caeser-shift": (35, 25, 40, 10, 20),
    "calculation": (45, 50, 55, 40, 25),
    "calculus": (15, 45, 35, 30, 40),
    "caldera": (30, 55, 45, 40, 60),
    "caldera-collapse": (-95, 85, 65, 90, 100),
    "calipers": (55, 25, 45, 35, 15),
    "calligraphy-ink": (85, 35, 45, 10, 5),
    "calotype": (70, 15, 40, 5, 45),
    "camaraderie": (80, 40, 45, 10, -25),
    "camber": (15, 25, 30, 15, 10),
    "camping-spawn": (-75, 30, 20, 40, 15),
    "camshaft": (25, 45, 40, 25, 60),
    "canard": (30, 25, 30, 10, -20),
    "cancel-culture": (-90, 85, 45, 90, 40),
    "cancellation-ink": (-40, 35, 25, 30, 10),
    "canidae": (75, 50, 40, 35, 15),
    "cannon": (30, 10, 40, 5, 20),
    "canon": (40, 10, 55, 10, 50),
    "cantilever": (55, 45, 45, 20, -10),
    "canton-flag": (35, 10, 40, 5, 25),
    "canvas": (45, 10, 30, 15, 15),
    "canvas-duck": (40, 10, 45, 10, 65),
    "canyon": (40, 10, 30, 0, 30),
    "cap-height": (35, 5, 40, 5, 10),
    "cap-no-cap": (35, 40, 10, 10, -5),
    "capacitive-reactance": (40, 55, 45, 40, 15),
    "capacitor": (15, 40, 30, 25, 5),
    "capacity": (30, 25, 40, 20, 15),
    "capillary": (15, 20, 20, 15, 0),
    "capital-gains": (70, 45, 45, 25, -25),
    "capitalization": (30, 15, 40, 10, 25),
    "cappuccino": (65, 30, 25, 20, -20),
    "capricious": (-30, 55, -25, 45, -20),
    "capsid": (25, 10, 40, 5, 30),
    "capstan": (15, 25, 35, 15, 40),
    "captivating": (85, 55, 30, 15, -35),
    "caramelization": (65, 25, 25, 15, 10),
    "carat": (45, 35, 40, 50, 15),
    "carbon-dating": (60, 20, 50, 15, 30),
    "carbon-footprint-metric": (-85, 40, 60, 55, 45),
    "carbon-sink": (85, 15, 55, 25, 50),
    "carburetor": (15, 45, 35, 20, 30),
    "carcinogen": (-85, 45, -20, 35, 40),
    "carcinoma": (-80, 40, -50, 45, 40),
    "cardiac": (10, 45, 20, 55, 10),
    "cardinality": (35, 15, 45, 10, 30),
    "carnivore": (10, 60, 45, 50, 25),
    "carrying-capacity": (35, 30, 60, 40, 70),
    "cartilage": (20, -5, 25, 0, 15),
    "caster": (10, 15, 30, 10, 5),
    "castling": (75, 30, 50, 55, 30),
    "catacomb": (-55, -5, -15, 10, 55),
    "catalysis": (65, 55, 45, 50, -10),
    "catalyst": (60, 60, 50, 55, -5),
    "cataract": (55, 55, 40, 20, 40),
    "catharsis": (95, 65, 50, 15, -60),
    "cathedral": (85, 20, 55, 5, 55),
    "catheter": (-45, 30, -50, 50, 20),
    "caucus": (15, 50, 25, 55, 20),
    "caviar": (65, 35, 45, 5, 30),
    "cavitation": (-45, 45, -25, 35, 5),
    "cedar": (50, -10, 35, 0, 20),
    "celestial": (95, 30, 50, 5, -70),
    "celestial-sphere": (95, 25, 65, 5, -85),
    "centaur-object": (25, 45, 55, 30, 65),
    "centrifugal": (10, 65, 45, 55, -20),
    "centrifugation": (15, 80, 50, 60, 40),
    "centrifuge-spin": (15, 75, 45, 60, -20),
    "centripetal": (20, 60, 50, 50, 45),
    "centurion": (55, 35, 55, 30, 50),
    "century-stand": (90, 60, 60, 30, 35),
    "cephalopod": (25, 40, 30, 10, -20),
    "ceramic": (45, -10, 30, 0, 15),
    "cerebellum": (40, 10, 35, 5, 20),
    "cerebrospinal": (40, -20, 35, 50, 20),
    "cerebrum": (50, 30, 50, 10, -20),
    "cerulean": (85, 10, 35, 0, -50),
    "cetacean": (85, 20, 55, 5, 40),
    "ceviche": (70, 40, 20, 15, -10),
    "chain-of-custody": (45, 20, 55, 55, 50),
    "chaise-longue": (70, -40, 40, 5, 25),
    "chalcolithic": (40, 20, 40, 10, 40),
    "challenge-accepted": (90, 80, 60, 40, -40),
    "chamfer": (55, 25, 35, 10, 10),
    "champagne": (80, 55, 35, 30, -40),
    "chandelier": (75, 20, 35, 10, -40),
    "changeup-fade": (55, 45, 25, 40, -15),
    "chaos-theory-butterfly": (65, 95, 45, 75, -45),
    "chaparral-scrub": (40, 30, 30, 20, 25),
    "charcoal": (-10, 25, 15, 20, 15),
    "charcuterie": (65, 20, 30, 5, 25),
    "charge-coupled-device": (90, 40, 65, 20, 75),
    "charge-discharge-rate": (55, 85, 55, 75, 45),
    "charlatan": (-95, 55, 30, 40, 35),
    "chartreuse": (55, 45, 20, 15, -30),
    "chaste": (65, -30, 40, 0, -20),
    "chatoyancy": (80, 35, 35, 5, -10),
    "checkered-flag": (90, 85, 55, 50, 25),
    "checkmate": (100, 85, 60, 60, 60),
    "checksum": (20, 20, 35, 35, 5),
    "cheddar": (60, 5, 30, 5, 15),
    "cherry-ripe-station": (-45, 70, 40, 65, 50),
    "chevron": (45, 25, 35, 10, 15),
    "chiaroscuro": (65, 45, 35, 10, 45),
    "chiffon": (65, -15, 20, 5, -50),
    "chimera": (-20, 55, 15, 30, 10),
    "chink": (-100, 70, -25, 60, 35),
    "chisel": (40, 50, 45, 40, 25),
    "chivalry": (90, 45, 50, 15, 25),
    "chlorophyll": (70, 10, 40, 5, -30),
    "chlorosis": (-55, 20, -40, 35, 15),
    "choad": (-80, 35, 5, 15, 15),
    "choke-point": (-35, 75, 10, 85, 25),
    "cholesterol": (-20, 15, -10, 20, 10),
    "chondrule-stony": (55, 15, 50, 5, 55),
    "chorus-emission": (85, 65, 45, 20, -100),
    "chromatic-aberration": (-55, 50, -25, 35, 10),
    "chromatin": (20, 10, 35, 5, 30),
    "chromatography": (50, 35, 40, 30, 15),
    "chronically-online": (-85, 50, -40, 35, 30),
    "chronicallyonline": (-60, 25, -40, 10, 15),
    "chronograph": (85, 55, 50, 50, 35),
    "chronometer": (45, 10, 45, 10, 15),
    "churros": (75, 35, 15, 15, -15),
    "chypre": (80, 45, 55, 10, 40),
    "cinematography": (85, 40, 50, 15, 25),
    "cipher-block-chain": (75, 45, 65, 35, 55),
    "ciphertext": (15, 35, 45, 45, 35),
    "circle-back": (-55, -20, 15, 45, 25),
    "circuit-breaker": (65, 65, 60, 85, 35),
    "circular-economy": (95, 55, 65, 30, 40),
    "circulation": (50, 40, 45, 20, 5),
    "cirrhosis": (-75, 30, -45, 35, 40),
    "cirrus": (65, -20, 20, 0, -45),
    "cisgender": (50, 10, 40, 5, 10),
    "citadel": (85, 40, 70, 50, 80),
    "citizen": (45, 15, 30, 10, 20),
    "citrine": (80, 45, 35, 5, -10),
    "civet-musk": (-10, 65, 50, 30, 35),
    "cladding-refraction": (75, 35, 55, 20, 45),
    "clamping": (40, 45, 50, 40, 35),
    "clapped": (-85, 45, -35, 15, 20),
    "clapperboard": (65, 70, 45, 80, 30),
    "clarification": (65, 10, 40, 15, 10),
    "clarity": (75, 15, 45, 10, -15),
    "classified": (-10, 45, 60, 55, 45),
    "clavicle": (30, 10, 30, 5, 25),
    "clean-girl": (75, -15, 45, 5, -35),
    "clearance": (70, 50, 40, 55, -25),
    "cleavage": (20, 15, 30, 5, 15),
    "clemency": (90, -10, 50, 45, -30),
    "clickbait": (-60, 55, -10, 45, -15),
    "climax": (90, 75, 50, 60, -15),
    "climax-community": (80, 10, 60, 5, 80),
    "clit": (-30, 55, 10, 25, -5),
    "clit-flicker": (-30, 65, 20, 40, -15),
    "cloisonné-enamel": (85, 40, 50, 10, 35),
    "clout": (25, 55, 30, 30, -15),
    "clusterfuck": (-85, 75, -10, 60, 20),
    "clutch": (85, 65, 50, 60, -30),
    "clutch-plate": (15, 40, 35, 45, 45),
    "clutch-save": (95, 90, 50, 95, -30),
    "cmyk": (30, 15, 35, 10, 15),
    "cnidarian": (35, 25, 10, 30, -45),
    "coatlicue-mother": (25, 70, 65, 30, 85),
    "cobalt": (55, 35, 45, 10, 40),
    "cobra": (-80, 65, 35, 60, 15),
    "coccyx": (10, 10, 20, 5, 35),
    "cochineal-beetle": (65, 50, 20, 15, -5),
    "cochlea": (50, 45, 40, 30, -10),
    "cockpit": (15, 45, 50, 50, 15),
    "cocksplat": (-85, 60, 10, 35, 10),
    "cocksucker": (-95, 70, 35, 50, 20),
    "codependency": (-60, 30, -50, 20, 40),
    "codex-vellum": (85, 15, 55, 5, 80),
    "codicil": (15, 5, 20, 10, 25),
    "codicology": (75, 10, 55, 5, 80),
    "coefficient": (15, 20, 35, 15, 20),
    "coercion": (-90, 55, 50, 50, 35),
    "coesite-high-pres": (35, 20, 60, 15, 95),
    "cogito": (60, 15, 50, 5, 10),
    "cognate": (55, 10, 35, 5, 10),
    "cognitive-bias": (-40, 35, -25, 25, 10),
    "cognitive-dissonance": (-75, 65, -45, 55, 25),
    "cold-brew-concentrate": (80, -35, 45, 20, 25),
    "cold-case": (-50, 20, 15, 10, 45),
    "cold-wallet": (85, -30, 60, 10, 65),
    "coleoptera": (30, 15, 25, 5, 10),
    "collaborator": (-85, 50, -10, 40, 20),
    "collateral": (10, 30, 40, 45, 40),
    "collision": (-85, 75, -50, 70, 30),
    "colloquial": (50, 30, 20, 5, -10),
    "colonnade": (80, 20, 50, 10, 60),
    "colonoscopy": (-40, 40, -40, 45, 25),
    "colony-collapse": (-100, 35, -60, 60, 50),
    "color-grading": (85, 50, 50, 25, 10),
    "colossus": (55, 40, 60, 10, 60),
    "columnar-storage": (75, 25, 60, 10, 70),
    "combinatorics-hard": (-15, 65, 45, 55, 45),
    "combine-harvester": (50, 55, 60, 40, 75),
    "comedy": (85, 60, 35, 15, -30),
    "comet": (55, 45, 30, 25, -35),
    "commensalism": (55, 10, 35, 5, 10),
    "commission": (55, 50, 40, 30, -15),
    "commodity": (10, 5, 25, 10, 30),
    "commodity-crop": (15, 20, 45, 25, 40),
    "commodity-fetishism": (-55, 65, 35, 40, 15),
    "commutation": (75, 30, 45, 40, 20),
    "comparative-linguistics": (60, 25, 50, 10, 40),
    "compartmentalized": (10, 20, 55, 40, 55),
    "compass": (50, 20, 45, 30, 10),
    "compassion": (95, 30, 45, 15, -40),
    "compensation": (55, 45, 35, 25, 5),
    "compiler": (25, 30, 35, 20, 10),
    "complacent": (-35, -40, 15, 0, 20),
    "complementary-metal-ox": (95, 45, 70, 25, 80),
    "complexion": (50, 20, 35, 5, -10),
    "complexity-class-np": (45, 65, 60, 35, 75),
    "complexity-class-p": (75, 35, 65, 10, 65),
    "compliance-audit": (-25, 45, 55, 75, 50),
    "composability": (80, 45, 55, 15, 15),
    "compressor": (10, 55, 40, 45, 40),
    "compromise": (-75, 55, -35, 60, 25),
    "concerto": (70, 45, 45, 10, 25),
    "conchoidal": (-10, 20, 30, 10, 45),
    "conclusion": (60, 40, 50, 45, 35),
    "conditioning": (15, 40, 40, 35, 25),
    "conduction": (15, 20, 25, 5, 10),
    "conductor": (40, 35, 45, 10, 15),
    "conduit": (35, 20, 40, 30, 40),
    "conflagration": (-90, 80, 35, 60, 25),
    "conifer": (60, 5, 45, 0, 45),
    "conjugation": (20, 10, 40, 15, 35),
    "connecting-rod": (20, 50, 45, 30, 55),
    "connective-tissue": (45, -5, 45, 10, 35),
    "consensus-mech": (75, 45, 60, 35, 50),
    "consonance": (80, -30, 40, 0, -10),
    "consonant": (35, 15, 30, 5, 15),
    "constituency": (20, 30, 30, 30, 25),
    "contact-sheet": (75, 25, 45, 20, 30),
    "container": (35, 25, 35, 20, 10),
    "containerization": (75, 40, 55, 25, 35),
    "containerized": (75, 30, 55, 20, 30),
    "continent": (35, 5, 60, 0, 60),
    "contingency": (45, 45, 50, 55, 30),
    "contour": (15, 10, 20, 5, 10),
    "contractor": (15, 20, 25, 25, 10),
    "contrail": (45, 5, 15, 0, -50),
    "convection": (20, 25, 25, 10, -10),
    "convergence": (80, 25, 55, 40, 40),
    "conviction": (-70, 45, 35, 20, 35),
    "coolant": (50, -20, 30, 15, 5),
    "copper": (50, 15, 35, 5, 25),
    "copypasta": (40, 50, 20, 15, -10),
    "coquette-core": (70, 25, 20, 5, -50),
    "coral": (60, 15, 20, 0, -5),
    "corduroy": (65, 15, 45, 10, 45),
    "core-competency": (55, 10, 60, 15, 65),
    "coriolis-effect": (15, 30, 35, 5, 5),
    "coriolis-force-spin": (35, 55, 55, 45, -15),
    "cork-taint": (-85, 40, -40, 20, 25),
    "cornea": (45, 15, 35, 10, 0),
    "cornice": (55, 10, 35, 5, 35),
    "corollary": (35, 10, 35, 15, 20),
    "corps-a-corps": (-40, 85, 10, 75, 40),
    "corridor": (0, 15, -20, 45, 15),
    "corsetry": (40, 55, 50, 30, 45),
    "cortex": (30, 20, 40, 5, 15),
    "corticosteroid": (15, 10, 20, 30, 10),
    "cortisol": (-35, 50, -30, 45, 10),
    "cosine-similarity": (80, 45, 55, 20, 10),
    "cosine-wave": (65, 40, 35, 10, -35),
    "cosmic-inflation": (90, 85, 85, 15, -95),
    "cosmic-ray-shower": (-75, 85, 40, 75, -95),
    "cosmogeny": (95, 55, 60, 15, 80),
    "cosmology": (75, 30, 55, 5, -70),
    "cosplay": (85, 70, 45, 35, -35),
    "cottage-core": (80, -30, 35, 0, -30),
    "cottagecore": (85, -35, 40, 5, -40),
    "couchant": (45, -30, 35, 5, 35),
    "counter": (20, 10, 20, 5, 0),
    "counter-insurgency": (20, 60, 40, 60, 30),
    "counterfeit": (-85, 45, 10, 30, 25),
    "counterintelligence": (45, 65, 55, 60, 25),
    "countersink": (50, 30, 40, 25, 15),
    "countertransference": (-35, 50, -25, 35, 15),
    "courtship": (70, 45, 30, 20, -20),
    "couscous": (50, 0, 25, 10, -10),
    "couturier": (90, 50, 60, 30, 40),
    "covariance": (10, 30, 25, 20, 15),
    "covenant": (85, 20, 55, 15, 55),
    "cradle-to-cradle": (100, 60, 70, 25, 35),
    "cranium": (40, 10, 45, 5, 45),
    "crankshaft": (15, 50, 45, 30, 45),
    "crash-out": (-95, 100, 55, 100, 40),
    "crashout": (-75, 75, 40, 60, 15),
    "crater-rays-bright": (95, 55, 55, 10, -50),
    "crater-rim-uplift": (65, 65, 75, 50, 95),
    "creatinine": (0, 15, 10, 30, 10),
    "credenza": (40, -15, 30, 0, 35),
    "creepypasta": (-65, 85, -35, 40, 20),
    "creole": (50, 25, 30, 5, -5),
    "crepe-de-chine": (75, -10, 35, 5, -35),
    "crepuscular": (45, 15, 20, 10, 5),
    "crescendo": (50, 55, 40, 45, 15),
    "crib-decryption": (65, 65, 50, 85, 10),
    "criminal-profiling": (35, 55, 55, 40, 25),
    "criminogenic": (-70, 45, 30, 20, 35),
    "crimson": (60, 50, 45, 20, 25),
    "crinoline": (45, 30, 45, 10, 55),
    "croisé": (70, 70, 55, 75, 30),
    "crop-rotation": (75, 5, 45, 10, 25),
    "cross-validation": (85, 40, 60, 35, 35),
    "cru": (80, 25, 55, 10, 40),
    "crucible": (30, 70, 40, 45, 55),
    "crucible-melt": (35, 90, 60, 70, 85),
    "crustacean": (15, 10, 20, 5, 15),
    "cryogenic-fuel": (25, 60, 40, 55, 45),
    "cryosphere": (60, -20, 45, 10, 55),
    "cryostat": (35, -45, 50, 35, 65),
    "crypt": (-65, -10, -25, 5, 45),
    "cryptanalysis": (40, 65, 55, 60, 35),
    "cryptocurrency": (20, 70, 20, 35, -35),
    "cryptography": (55, 45, 50, 30, 35),
    "crystal": (60, 25, 25, 5, -15),
    "crystalline": (75, 30, 35, 5, -15),
    "cuffing-season": (75, 65, 35, 70, 45),
    "cufflinks": (55, 20, 35, 15, 10),
    "cultivar": (40, 5, 30, 5, 15),
    "cum-guzzler": (-90, 60, 20, 40, 15),
    "cumguzzler": (-95, 65, 25, 45, 20),
    "cumulonimbus": (-30, 75, 45, 70, 35),
    "cuneiform": (45, 10, 40, 5, 40),
    "cunt": (-85, 60, 30, 40, 10),
    "cupcakke-remix": (80, 95, 40, 60, -50),
    "cupping-protocol": (60, 25, 55, 30, 15),
    "curator-archive": (75, 30, 60, 25, 45),
    "curveball-break": (35, 65, 30, 45, 10),
    "cuvée": (85, 30, 50, 10, 30),
    "cyan": (70, 30, 25, 5, -35),
    "cyanosis": (-80, 45, -50, 60, 25),
    "cyanotype": (85, 20, 35, 5, -60),
    "cyclogenesis": (-45, 75, 40, 70, 35),
    "cyclone": (-75, 65, -50, 65, 35),
    "cynical": (-55, -10, 25, 5, 20),
    "cynicism": (-55, 15, 40, 10, 35),
    "cyst": (-45, 20, -30, 15, 15),
    "cytoplasm": (25, 5, 15, 0, -10),
    "cytoskeleton": (55, 25, 55, 10, 65),
    "daguerreotype": (90, 15, 55, 5, 95),
    "damascus-steel": (95, 55, 65, 15, 75),
    "damask": (80, 20, 50, 5, 50),
    "dampers": (45, -10, 45, 15, 40),
    "dao-vote": (50, 35, 45, 40, 25),
    "dark-academia": (65, 15, 55, 10, 65),
    "dark-energy-repulsion": (-25, 45, 75, 5, -100),
    "dark-matter": (-45, 20, 40, 5, 75),
    "darning": (40, -15, 35, 20, 15),
    "dart-seam": (40, 15, 35, 30, 10),
    "dashboard": (50, 15, 35, 10, 5),
    "datum-point": (40, 10, 45, 25, 35),
    "day-for-night": (35, 40, 35, 20, 45),
    "dead-drop": (-30, 55, 30, 65, 15),
    "dead-reckoning": (15, 65, 45, 75, 20),
    "deadlock": (-100, 60, -60, 85, 50),
    "debugger": (10, 40, 25, 50, 5),
    "decanting": (85, 15, 45, 25, 20),
    "decentralization": (95, 65, 65, 25, 45),
    "decibel-level": (55, 65, 55, 50, 5),
    "deciduous": (55, 20, 35, 15, 20),
    "declarative": (25, 10, 35, 5, -5),
    "declension": (10, 5, 40, 10, 40),
    "decomposer": (25, 15, 30, 10, 25),
    "decompression": (-45, 60, 35, 65, 35),
    "deconstruction": (45, 50, 55, 25, -20),
    "decoy": (55, 60, 40, 65, 10),
    "decryption": (50, 45, 40, 50, 10),
    "deduction": (50, 25, 45, 15, 15),
    "deep-dive": (45, 35, 40, 45, 35),
    "deep-focus": (75, 40, 55, 15, 50),
    "default": (-80, 55, -50, 60, 40),
    "defector": (-85, 65, -20, 60, 25),
    "defendant": (-20, 50, -35, 55, 15),
    "defense-mechanism": (-30, 45, 35, 40, 20),
    "defibrillator": (50, 60, 40, 60, 15),
    "deficit": (-55, 35, -40, 50, 30),
    "deflection": (65, 65, 45, 65, 10),
    "degen": (35, 75, 25, 50, -15),
    "dehydration": (-10, -5, 10, 10, 5),
    "deliberation": (20, 30, 30, 25, 20),
    "deliverable": (15, 35, 25, 55, 10),
    "delta": (45, 10, 30, 5, 15),
    "delta-v": (70, 60, 50, 45, 35),
    "delulu": (-35, 30, -30, 5, -15),
    "delulu-is-solulu": (55, 40, 25, 10, -45),
    "demand-response-sig": (65, 75, 55, 85, 35),
    "demijohn": (50, 5, 35, 10, 55),
    "democracy": (75, 45, 40, 20, 15),
    "dendrite": (45, 50, 20, 10, -40),
    "dendrochronology": (85, 25, 55, 15, 65),
    "dendrology": (80, 15, 55, 10, 50),
    "denim": (40, 15, 35, 5, 25),
    "denouement": (65, -30, 40, 10, 35),
    "dense": (25, 10, 45, 5, 55),
    "density": (20, 5, 40, 5, 50),
    "deontology": (40, 15, 60, 10, 65),
    "deployment-hell": (-95, 90, -50, 95, 40),
    "deposition": (0, 25, 15, 35, 15),
    "deprecation": (-55, 25, -20, 55, 30),
    "depreciation": (-55, 5, -40, 15, 35),
    "depth-gauge": (25, 45, 40, 60, 25),
    "depth-of-field": (75, 40, 45, 20, -10),
    "derecho": (-85, 85, 35, 80, 25),
    "derecho-windfall": (-100, 100, 65, 100, 55),
    "derivative": (20, 40, 40, 30, 15),
    "derivative-gain": (75, 85, 60, 80, 10),
    "derivative-swap": (35, 55, 45, 40, 55),
    "descender": (20, 5, 20, 5, 25),
    "descent": (-45, 40, -35, 45, 35),
    "deserialization": (15, 20, 25, 35, 0),
    "desiccator": (45, -5, 40, 15, 35),
    "desperado": (20, 85, -25, 80, -35),
    "detached": (-20, -35, 20, 5, 10),
    "detachment": (-30, -40, 30, 5, 15),
    "determinant": (45, 35, 60, 20, 85),
    "determinism": (-15, -5, 65, 5, 80),
    "deterritorialization": (30, 60, 40, 25, -35),
    "detritus": (-45, -5, -35, 5, 15),
    "developer-bath": (45, 35, 40, 55, 10),
    "dew": (65, -15, 20, 0, -30),
    "dew-point": (35, 10, 20, 20, -10),
    "dewey-decimal": (20, -10, 55, 15, 45),
    "dexter": (35, 5, 40, 10, 10),
    "dexterity": (75, 60, 55, 20, -35),
    "dharma": (80, 15, 55, 10, 45),
    "diagonal": (40, 35, 35, 20, -15),
    "dialect": (45, 20, 35, 5, 10),
    "dialectic": (40, 45, 35, 20, 15),
    "dialectology": (55, 20, 45, 10, 30),
    "dialysis": (-65, 20, -50, 55, 35),
    "diamond-hands": (85, -20, 55, 5, 60),
    "diaphragm": (45, 25, 40, 55, 10),
    "dick-cheese": (-95, 30, 5, 15, 25),
    "dickcheese": (-100, 35, 10, 20, 30),
    "dicot": (50, 20, 35, 10, 20),
    "didactic": (-20, 10, 40, 15, 30),
    "differential": (20, 35, 35, 10, 35),
    "differential-eq": (-10, 55, 40, 50, 40),
    "differential-privacy": (90, 45, 65, 35, 40),
    "diffie-hellman": (90, 45, 70, 40, 60),
    "diffraction": (40, 30, 30, 15, -35),
    "diffraction-grating": (45, 45, 35, 10, -40),
    "diffuser": (40, 30, 40, 10, 45),
    "diffusion": (30, -10, 20, 10, -20),
    "diffusion-limited": (15, 25, 35, 15, 45),
    "digestion": (45, 15, 40, 20, 10),
    "digestive": (30, 15, 35, 20, 15),
    "diminuendo": (30, -35, 10, 5, 15),
    "dimwit": (-85, -15, -45, 15, 20),
    "dingbat": (45, 40, 15, 10, -20),
    "dingleberry": (-65, 35, -10, 20, 10),
    "diocese": (30, 5, 45, 10, 40),
    "diphthong": (40, 25, 20, 5, -15),
    "diplomat": (60, 10, 45, 20, 15),
    "director": (35, 60, 70, 80, 50),
    "discourse": (40, 35, 45, 25, 20),
    "discovered-attack": (90, 75, 55, 75, -20),
    "disengage": (60, 65, 40, 60, -30),
    "disinformation": (-85, 45, 40, 30, 25),
    "dispersion": (75, 55, 35, 10, -30),
    "display-type": (55, 35, 40, 15, 15),
    "disposition-schedule": (-10, 25, 45, 50, 40),
    "disruptor": (70, 85, 55, 45, -30),
    "dissociation": (-70, -40, -55, 25, -10),
    "dissonance": (-45, 55, -20, 40, 5),
    "distaff-spindle": (60, 10, 35, 15, 45),
    "distillation": (55, 35, 45, 30, 30),
    "distillation-steam": (65, 50, 45, 30, 25),
    "distributed-tracing": (75, 55, 60, 40, 40),
    "diurnal": (55, 25, 30, 5, -15),
    "divergence": (-55, 55, -20, 50, -10),
    "diversification": (45, 20, 40, 10, 10),
    "dividend-yield": (90, 15, 60, 5, 35),
    "dna-sequencing": (70, 40, 60, 30, 40),
    "docket": (0, 20, 15, 50, 15),
    "docking-port": (80, 50, 55, 70, 45),
    "dodecahedron": (65, 45, 50, 5, 45),
    "dodging-tool": (40, 50, 45, 35, -20),
    "dogmatic": (-45, 30, 45, 15, 35),
    "dolly-zoom": (40, 85, 30, 60, -25),
    "dolmen": (40, -5, 45, 0, 70),
    "dolphin": (90, 55, 40, 10, -15),
    "dome": (90, 25, 60, 10, 60),
    "doom-scrolling": (-100, 65, -70, 85, 55),
    "doomer": (-75, -20, -50, 5, 40),
    "doop": (50, 20, 15, 5, -10),
    "doosra-spin": (35, 70, 25, 65, -25),
    "dopamine": (70, 55, 40, 20, -40),
    "doppler-velocity-rad": (65, 75, 60, 85, -20),
    "dossier": (20, 15, 40, 25, 35),
    "double-die-obverse": (85, 70, 45, 50, 45),
    "double-play": (75, 70, 55, 80, 25),
    "douche-canoe": (-70, 40, 15, 20, -20),
    "douchecanoe": (-75, 45, 20, 25, -15),
    "dovetail": (85, 35, 55, 15, 45),
    "dowel": (55, 10, 40, 15, 35),
    "downforce": (65, 55, 50, 25, 65),
    "downsizing": (-80, 55, -50, 50, 40),
    "draft": (-10, 15, -10, 25, -5),
    "drag": (-35, 30, -20, 15, 10),
    "drag-coefficient": (-20, 15, 35, 5, 20),
    "dragon": (60, 65, 55, 55, 50),
    "draping": (75, 25, 45, 15, 10),
    "drawknife": (55, 50, 45, 30, 30),
    "drifting": (75, 80, 40, 45, -25),
    "drill-down": (35, 35, 45, 40, 25),
    "driller": (-65, 95, 50, 95, 50),
    "drive": (25, 15, 30, 15, 15),
    "drone-bee": (10, -5, -20, 5, 10),
    "drop-cap": (60, 30, 35, 10, 25),
    "dropsonde-descent": (55, 85, 50, 90, 65),
    "drumlin": (55, 10, 35, 5, 40),
    "dryer": (40, 25, 40, 30, 40),
    "dualism": (35, 25, 35, 10, 0),
    "duck-out": (-85, 45, -60, 20, 55),
    "duke": (80, 40, 55, 25, 55),
    "dumbfuck": (-90, 50, 10, 20, 15),
    "dungeon": (-80, 40, -50, 55, 50),
    "duration": (15, 5, 30, 15, 15),
    "duress": (-85, 50, -45, 55, 40),
    "duty-cycle": (55, 55, 45, 60, 15),
    "duvet": (75, -45, 40, 5, -25),
    "dynamic-range": (85, 40, 55, 10, 35),
    "e-boy": (60, 70, 30, 25, -40),
    "e-girl": (65, 75, 25, 25, -45),
    "ear": (35, 30, 35, 10, 5),
    "earl": (60, 25, 45, 15, 40),
    "easel": (40, 10, 25, 15, 15),
    "eating-it-up": (95, 85, 55, 30, -45),
    "ebb": (-40, -35, -30, 15, 15),
    "ebony": (35, 5, 50, 0, 45),
    "eccentricity": (15, 35, 30, 25, 15),
    "echinoderm": (10, -5, 15, 0, 10),
    "eclair": (75, 35, 20, 15, -20),
    "eclipse": (20, 50, 15, 40, 10),
    "econometrics": (35, 45, 50, 50, 45),
    "ecosystem": (70, 25, 55, 15, 30),
    "ectotherm": (-10, -15, 10, 0, 5),
    "ecumenical": (65, 15, 40, 25, 20),
    "edema": (-50, 15, -40, 30, 30),
    "edge-computing": (85, 45, 60, 35, -45),
    "edge-lord": (-55, 40, 15, 10, 15),
    "effervescent": (85, 60, 35, 15, -40),
    "efficiency": (65, 15, 50, 10, -20),
    "ego-depletion": (-60, -30, -50, 20, 35),
    "eigenvalue": (55, 45, 55, 25, 40),
    "eigenvector": (45, 45, 45, 30, -10),
    "einherjar-warrior": (80, 75, 60, 25, 55),
    "ejecta-blanket": (55, 75, 60, 60, 65),
    "el-niño": (-50, 65, 25, 70, 10),
    "electorate": (25, 35, 35, 40, 30),
    "electrojet-auroral": (95, 100, 75, 85, -75),
    "electrolyte": (55, 35, 30, 45, -15),
    "electromagnetic-interference": (-85, 75, -45, 80, 30),
    "electron": (15, 45, 20, 10, -40),
    "electrophoresis": (25, 45, 40, 50, -15),
    "elevation": (45, 35, 45, 15, 45),
    "elevator": (20, 40, 35, 40, -10),
    "ellipse": (55, 20, 40, 5, -10),
    "elliptic-curve": (80, 55, 65, 30, 45),
    "embargo": (-60, 35, 40, 45, 40),
    "embassy": (50, 10, 40, 30, 35),
    "embezzlement": (-95, 55, 40, 50, 45),
    "embolism": (-80, 55, -50, 60, 35),
    "embroidery": (85, 40, 45, 10, 25),
    "emerald": (85, 25, 45, 5, 30),
    "emotional-intelligence": (90, 35, 55, 15, -25),
    "empathy": (85, 35, 25, 10, -30),
    "empennage": (20, 10, 30, 5, 15),
    "empirical": (40, 10, 45, 10, 30),
    "empiricism": (60, 10, 55, 20, 50),
    "emulator": (25, 25, 30, 20, 10),
    "emulsion": (35, 10, 20, 10, 5),
    "emulsion-lift": (85, 45, 35, 15, -40),
    "en-garde": (50, 80, 55, 90, 55),
    "en-passant": (40, 55, 30, 60, -10),
    "enabler": (-50, 25, -20, 15, 25),
    "encapsulation": (30, 15, 40, 10, 15),
    "encore": (95, 70, 45, 20, -45),
    "encryption": (65, 20, 55, 40, 30),
    "end-zone-trip": (95, 85, 65, 50, -45),
    "endemic": (-35, 10, 25, 20, 40),
    "endemic-cave-species": (65, 25, 55, 15, 50),
    "endemic-species": (85, 25, 50, 10, 35),
    "endgame": (65, 55, 45, 20, 20),
    "endianness": (10, 15, 35, 15, 5),
    "endocrine": (25, 20, 45, 20, 25),
    "endorphin": (80, 45, 40, 10, -35),
    "endoscope": (-20, 35, -10, 40, 15),
    "endotherm": (40, 35, 30, 10, 10),
    "endothermic": (10, 35, 30, 25, 20),
    "endowment": (75, 10, 45, 5, 20),
    "endpoint": (15, 25, 25, 35, 0),
    "ends": (-20, 45, 30, 40, 35),
    "enemy": (-80, 60, -10, 50, 30),
    "energy-density-whkg": (95, 40, 70, 25, 60),
    "energy-pyramid": (55, 25, 45, 10, 55),
    "enfleurage": (85, -20, 50, 5, 20),
    "engagement": (40, 45, 25, 30, -10),
    "enigma": (20, 35, -15, 25, 10),
    "enlarger-lamp": (65, 40, 50, 25, -45),
    "enmeshment": (-45, 40, -50, 20, 35),
    "ennui": (-65, -55, -40, 0, 40),
    "ensign": (70, 40, 55, 20, 55),
    "entanglement": (-35, 45, -30, 20, 20),
    "entrancing": (85, 45, 25, 10, -40),
    "entropy": (-85, 15, -60, 10, 45),
    "entropy-bit": (25, 55, 40, 15, -15),
    "envelope-viral": (30, 15, 35, 10, 15),
    "enzyme": (50, 55, 45, 40, -10),
    "epaulette": (60, 45, 50, 15, 35),
    "epeirogeny-uplift": (55, 25, 70, 10, 95),
    "ephemeral": (45, 15, -20, 30, -40),
    "epic-fail": (-95, 70, -50, 40, 45),
    "epicureanism": (85, -25, 45, 5, -20),
    "epidemic": (-90, 85, -45, 90, 55),
    "epigraphy": (45, 10, 40, 5, 35),
    "epigraphy-stone": (55, 15, 55, 5, 95),
    "epilogue": (55, 10, 45, 10, 30),
    "epinephrine": (40, 60, 45, 60, -20),
    "epiphany": (100, 60, 50, 45, -60),
    "epiphyte": (65, 15, 30, 0, -35),
    "epiphytic": (70, 20, 35, 5, -45),
    "episteme": (55, 15, 50, 10, 45),
    "epistemology": (45, 25, 45, 5, 25),
    "epithelium": (30, 5, 35, 10, 15),
    "epitome": (80, 20, 50, 0, 15),
    "epoch": (60, 25, 45, 10, 45),
    "equality": (50, -20, 50, 15, 10),
    "equilibrium": (95, -55, 60, 10, 20),
    "equinox": (50, 5, 40, 10, 0),
    "era": (55, 20, 40, 5, 40),
    "erbium-doped-amp": (85, 40, 65, 30, 45),
    "ergodic-hypothesis": (35, 10, 55, 5, 85),
    "ermine": (75, 10, 40, 0, 25),
    "erosion": (-45, 15, -20, 10, 30),
    "error-correction-code": (95, 40, 70, 55, 50),
    "error-variety": (75, 65, 35, 45, 25),
    "erudite": (80, 10, 50, 5, 30),
    "erythrocyte": (35, 10, 25, 10, 0),
    "escape-velocity": (85, 85, 65, 60, -60),
    "escapement": (65, 40, 45, 15, 30),
    "escapement-lever": (65, 65, 50, 45, 35),
    "eschatology": (-40, 65, 50, 25, 95),
    "esg-compliance": (65, 35, 65, 50, 50),
    "esker": (50, 5, 35, 5, 35),
    "esophagus": (30, 15, 30, 45, 20),
    "espionage": (-50, 65, 45, 55, 20),
    "espresso": (55, 55, 30, 50, -15),
    "espresso-crema": (95, 55, 40, 50, 35),
    "estranged": (-75, -20, -50, 10, 45),
    "estrangement": (-75, -20, -50, 10, 45),
    "estrogen": (25, 20, 20, 10, -10),
    "estuary": (55, 15, 25, 5, 10),
    "ether": (20, 45, -15, 25, -45),
    "ethics": (80, 20, 60, 15, 50),
    "ethnomusicology": (90, 45, 55, 10, 35),
    "etude": (30, 35, 20, 15, 5),
    "etymology": (65, 10, 45, 5, 30),
    "euclidean-distance": (65, 35, 55, 15, 25),
    "eudaimonia": (100, 55, 60, 15, -60),
    "eukaryote": (55, 25, 45, 10, 45),
    "euphemism": (30, -15, 15, 10, -5),
    "euphoria": (100, 60, 40, 10, -50),
    "eutrophication": (-90, 55, -45, 60, 45),
    "evangelist": (70, 65, 50, 30, 15),
    "evaporite": (40, 5, 35, 5, 25),
    "evapotranspiration": (45, 10, 30, 5, -20),
    "event-horizon": (-30, 65, -50, 30, 70),
    "eventual-consistency": (35, 35, 45, 25, 45),
    "evergreen": (80, 10, 50, 5, 45),
    "excavation": (-15, 65, 40, 65, 60),
    "execution": (-90, 65, 55, 75, 50),
    "executive": (45, 40, 50, 55, 45),
    "executive-function": (60, 30, 55, 40, 20),
    "exemplar": (80, 25, 45, 5, 10),
    "exfiltration": (60, 75, 40, 80, -10),
    "exhaust-note": (60, 80, 45, 20, 35),
    "exhilaration": (95, 75, 50, 40, -50),
    "exhumation": (-60, 55, 40, 45, 85),
    "existential": (10, 45, -10, 20, 10),
    "exit-strategy": (30, 35, 40, 25, 20),
    "exobase": (25, 10, 40, 10, -90),
    "exosphere": (20, -30, 35, 0, -80),
    "exothermic": (45, 65, 40, 55, -30),
    "exponent": (30, 40, 40, 25, -20),
    "exposure": (50, 45, 40, 35, -10),
    "expulsion": (-95, 85, 45, 90, 40),
    "extortion": (-95, 65, 45, 60, 40),
    "extraction": (50, 75, 45, 80, 25),
    "extraction-yield": (45, 35, 45, 45, 15),
    "extradition": (10, 55, 45, 70, 55),
    "extrinsic-motivation": (35, 45, 35, 40, 10),
    "extrusion": (10, 35, 30, 20, 35),
    "eyeball": (30, 45, 15, 10, -10),
    "f-hole": (85, 40, 45, 10, 35),
    "f-stop-aperture": (45, 25, 50, 45, 15),
    "f7u12": (-70, 95, -20, 80, 10),
    "facade": (25, 35, 45, 20, 35),
    "face-card": (100, 35, 55, 10, -35),
    "factorial": (20, 55, 40, 35, 25),
    "facts": (50, 25, 40, 10, 5),
    "faded": (45, -40, -35, 10, -25),
    "failover": (30, 45, 40, 55, 20),
    "falafel": (65, 25, 25, 20, 10),
    "falcon": (70, 65, 45, 50, -45),
    "fallacy": (-65, 45, -20, 30, 5),
    "fallow": (-15, -35, 20, 5, 35),
    "falsetto": (35, 40, -10, 10, -45),
    "fan-fiction": (50, 55, 15, 25, -25),
    "fancam": (30, 50, 10, 20, -20),
    "fanfic": (25, 40, 15, 10, -15),
    "fanum-tax": (15, 45, 10, 30, -10),
    "far-field-microwave": (60, 85, 55, 65, 35),
    "farce": (50, 65, 20, 25, -15),
    "fastball-four-seam": (45, 75, 50, 60, 20),
    "fatherless": (-70, 45, -30, 20, 20),
    "fathom": (10, -10, 25, 5, 40),
    "fault-line": (-60, 65, 40, 65, 45),
    "fealty": (85, 20, 45, 25, 35),
    "federal": (25, 10, 45, 15, 40),
    "federated-learning": (95, 55, 65, 25, 35),
    "feeding": (-85, 45, -50, 40, 25),
    "feint": (55, 75, 45, 70, -15),
    "felidae": (70, 55, 45, 30, 15),
    "felony": (-80, 55, -20, 40, 40),
    "femtosecond": (20, 85, 15, 95, -95),
    "femur": (40, 10, 50, 5, 45),
    "fenrir-wolf": (-90, 85, 65, 75, 55),
    "fermentation": (40, 5, 20, 0, 10),
    "fern": (75, -20, 25, 0, -15),
    "fertility": (70, 50, 45, 15, -25),
    "fertilizer": (45, 15, 30, 20, 30),
    "fervor": (80, 65, 45, 40, -25),
    "fess": (40, 10, 40, 5, 30),
    "feta": (55, 25, 20, 10, -5),
    "fetid": (-95, 50, -25, 45, 20),
    "fettuccine": (65, 10, 25, 15, 15),
    "fianchetto": (50, 25, 45, 15, 20),
    "fiancée": (75, 55, 35, 15, -20),
    "fiat": (20, 15, 40, 10, 20),
    "fiat-currency": (20, 15, 55, 20, 45),
    "fiber-optics": (80, 50, 55, 40, -50),
    "fibrous": (20, 20, 30, 5, 15),
    "fibula": (35, 10, 35, 5, 40),
    "fidelity": (90, 15, 55, 5, 20),
    "fiduciary-duty": (85, 15, 65, 45, 60),
    "fiduciary-risk": (-75, 55, 40, 60, 45),
    "fiefdom": (40, 25, 45, 20, 50),
    "field-aligned-curr": (40, 85, 65, 65, -85),
    "field-flag": (25, 5, 35, 0, 15),
    "field-goal-unit": (55, 80, 50, 90, 20),
    "file-tool": (15, 35, 35, 20, 15),
    "filigree": (85, 35, 35, 5, -10),
    "film-stock": (55, 10, 45, 20, 45),
    "fimbriation": (45, 15, 25, 5, 5),
    "finale": (65, 55, 45, 55, 25),
    "fingerboard-ebony": (65, 10, 60, 5, 85),
    "fingerprint-latent": (25, 35, 45, 55, 10),
    "finial": (40, 10, 45, 5, -30),
    "fining": (45, 10, 35, 15, 10),
    "firewall-rule": (80, 25, 65, 60, 40),
    "first-crack": (75, 65, 40, 60, -10),
    "fission": (-40, 60, 45, 55, 15),
    "fissure": (-40, 50, 20, 50, 35),
    "fixer-solution": (50, 25, 45, 55, 20),
    "fjord": (80, 20, 50, 5, 55),
    "flacon": (85, 35, 40, 15, 50),
    "flamboyant": (55, 55, 30, 15, -45),
    "flame-war": (-90, 95, 20, 85, 25),
    "flaming-out": (-85, 65, -50, 60, 30),
    "flange": (5, 10, 30, 10, 30),
    "flank": (10, 60, 40, 60, 5),
    "flannel": (90, -45, 40, 5, 10),
    "flaps": (15, 35, 35, 50, 5),
    "flat-white-microfoam": (95, 35, 40, 20, -10),
    "flatware": (35, 5, 20, 10, 15),
    "fleche": (50, 95, 40, 90, -50),
    "fleeting": (-35, 55, -35, 60, -35),
    "fleuret": (45, 55, 35, 45, -15),
    "flexing": (25, 45, 35, 10, -10),
    "floating-point": (20, 40, 35, 25, 10),
    "flop-era": (-85, 35, -55, 50, 65),
    "floptok": (75, 85, 35, 55, -60),
    "flotsam": (-30, 10, -35, 5, -15),
    "flow-state": (100, 65, 60, 5, -80),
    "fluctuation": (-15, 45, -20, 40, 5),
    "fluid-dynamics": (45, 40, 45, 30, 25),
    "fluorescence": (70, 65, 30, 25, -50),
    "flux-borax": (45, 25, 30, 40, 15),
    "flux-transfer-event": (65, 90, 70, 75, -45),
    "fly-end": (15, 10, 30, 5, -15),
    "flywheel": (20, 55, 45, 35, 55),
    "focaccia": (65, 5, 30, 10, 10),
    "focal-length": (35, 30, 40, 25, -15),
    "fodder": (35, -5, 25, 10, 30),
    "foible": (-45, 45, -50, 15, -45),
    "foie-gras": (40, 20, 35, 5, 45),
    "foil": (55, 65, 40, 50, -20),
    "folio-format": (60, 10, 45, 5, 60),
    "folly": (-55, 45, -35, 15, -10),
    "fonds-collection": (55, 10, 50, 5, 65),
    "fondue": (80, 45, 30, 25, 20),
    "font-family": (50, 15, 45, 5, 15),
    "foraging": (45, 40, 20, 25, -10),
    "forever-alone": (-85, -55, -60, 0, 50),
    "forge-hearth": (55, 75, 50, 45, 60),
    "fork-attack": (85, 75, 55, 70, 10),
    "forte": (55, 10, 65, 5, 85),
    "fortissimo": (40, 60, 45, 15, 30),
    "fortress": (65, 10, 60, 10, 60),
    "fortress-chess": (65, -35, 60, 20, 80),
    "forum": (55, 50, 40, 25, 20),
    "fossil": (20, -15, 20, 0, 35),
    "foucault-pendulum": (60, -45, 65, 10, 95),
    "fougère": (85, 40, 50, 10, 30),
    "foundation": (95, -10, 60, 10, 80),
    "foundations": (55, -5, 60, 5, 60),
    "foundry": (-10, 60, 40, 45, 60),
    "fountain-pen": (65, 15, 35, 10, 5),
    "fourier-transform": (95, 60, 65, 30, -10),
    "foxglove": (30, 25, 25, 50, 20),
    "fractal": (65, 45, 30, 5, -45),
    "fractal-set": (85, 60, 35, 15, -50),
    "fragging": (65, 75, 50, 60, -25),
    "fragrant": (90, 25, 30, 5, -40),
    "frailty": (-70, -30, -55, 15, 15),
    "frame-rate": (30, 55, 35, 40, 5),
    "framework-bloat": (-70, 40, -20, 25, 40),
    "freelance": (50, 40, 45, 20, -25),
    "frequency": (25, 45, 35, 35, -10),
    "frequency-analysis": (40, 55, 55, 45, 35),
    "fresco": (85, 35, 45, 10, 35),
    "friction": (-40, 40, 15, 25, 15),
    "frieze": (70, 25, 35, 5, 25),
    "frivolous": (-35, 45, -25, 10, -30),
    "frost": (30, 20, 15, 15, 10),
    "fructose": (25, 20, 10, 5, 0),
    "fuchsia": (70, 60, 25, 20, -30),
    "fuck-face": (-90, 65, 20, 45, 15),
    "fuck-knuckle": (-85, 60, 15, 40, 15),
    "fucknugget": (-70, 45, 10, 20, -5),
    "fuckstick": (-80, 55, 15, 30, 15),
    "fucktard": (-95, 65, 20, 40, 25),
    "fud": (-75, 65, -35, 75, 15),
    "full-stack": (70, 55, 55, 45, 30),
    "fumarole-vent": (-25, 65, 40, 55, 25),
    "fumble": (-65, 45, -50, 40, 25),
    "fumble-recovery": (80, 95, 45, 100, 30),
    "fume-hood": (65, 30, 55, 60, 50),
    "functional": (35, 20, 35, 10, -5),
    "fungus": (-30, 10, -15, 5, 10),
    "furlough": (-65, -10, -45, 35, 35),
    "furry": (45, 65, 15, 20, 30),
    "fuselage": (20, 5, 45, 5, 40),
    "fusillade": (-85, 80, 45, 75, 35),
    "fusion": (70, 60, 50, 15, -20),
    "fusion-crust": (-15, 65, 50, 65, 40),
    "futures": (30, 50, 35, 30, 15),
    "gabardine": (55, 10, 40, 10, 40),
    "gaffer": (60, 55, 55, 65, 35),
    "gagged": (100, 95, -10, 75, -60),
    "galaxy": (80, 30, 50, 0, -50),
    "gale": (-35, 55, -20, 55, 10),
    "gallbladder": (35, 15, 30, 10, 25),
    "galley": (30, 10, 25, 10, 20),
    "galvanized": (30, 5, 45, 0, 40),
    "gambit": (45, 65, 35, 60, -20),
    "game-changer": (85, 75, 50, 35, -45),
    "ganache": (75, 15, 30, 5, 10),
    "ganesha-remover": (95, 40, 65, 25, 35),
    "ganglion": (20, 35, 30, 40, 15),
    "ganking": (-60, 70, 45, 65, 10),
    "gantry": (20, 30, 50, 15, 60),
    "garnet": (70, 30, 45, 5, 40),
    "garrison": (30, 20, 45, 45, 45),
    "garuda-mount": (85, 65, 55, 30, -50),
    "gas": (30, 35, 15, 30, -40),
    "gas-fees": (-90, 60, -30, 80, 15),
    "gasket": (10, 5, 25, 15, 15),
    "gaslighting": (-95, 65, 45, 40, 35),
    "gasoline": (-20, 55, 25, 50, -10),
    "gatekeeping": (-50, 25, 30, 20, 15),
    "gateway": (65, 40, 45, 50, 20),
    "gaussian-noise": (-25, 55, 35, 45, 15),
    "gazpacho": (60, 10, 20, 5, -10),
    "gear-train": (45, 40, 55, 25, 60),
    "geeked": (55, 80, -10, 50, -40),
    "gelato": (85, 35, 25, 15, -40),
    "genotype": (10, 5, 35, 0, 20),
    "geochronology": (60, 5, 55, 5, 60),
    "geomagnetic-pulsation": (55, 55, 60, 45, -60),
    "geomagnetic-storm": (-85, 95, 40, 90, -50),
    "geomorphology": (60, 10, 45, 5, 40),
    "geotropism": (55, 15, 50, 5, 65),
    "getreadywithme": (45, 35, 20, 15, -15),
    "geyser-eruption": (85, 95, 65, 80, 45),
    "ghosting": (-80, 20, -50, 10, 30),
    "gig-economy": (10, 45, 10, 40, 5),
    "gigachad": (85, 45, 50, 5, 50),
    "gimbal": (80, 60, 55, 30, -15),
    "girder": (30, 15, 55, 15, 75),
    "git": (-50, 30, -10, 15, 10),
    "git-gud": (-45, 55, 45, 30, 20),
    "git-push-force": (20, 85, 50, 90, 10),
    "glaciation": (30, 35, 45, 25, 70),
    "glacier": (30, -15, 40, 0, 40),
    "gland": (15, 20, 30, 10, 15),
    "glare": (-45, 55, 15, 45, 10),
    "glass": (25, 35, 15, 10, 5),
    "glass-cannon": (30, 80, -20, 70, -35),
    "glaucoma": (-50, 15, -35, 20, 20),
    "glaze": (-50, 35, -30, 10, 10),
    "glazing": (-60, 40, -40, 10, 10),
    "glazing-hard": (-65, 50, -40, 20, 10),
    "glide": (80, -25, 40, 5, -45),
    "glider": (75, -25, 35, 5, -45),
    "glideslope": (50, 30, 40, 50, -10),
    "glimmer": (70, 35, 15, 10, -35),
    "glissando": (65, 45, 20, 10, -30),
    "glisten": (80, 30, 25, 5, -35),
    "gloom": (-75, -25, -45, 10, 35),
    "glossary": (60, -5, 45, 10, 20),
    "glossary-tech": (55, 5, 45, 25, 35),
    "glossematics": (45, 10, 45, 5, 20),
    "glossy": (65, 35, 35, 5, -20),
    "glucose": (30, 20, 25, 40, 5),
    "glyph": (45, 20, 30, 5, -5),
    "gneiss": (40, 5, 45, 0, 55),
    "gnocchi": (65, 10, 25, 15, 20),
    "god-damned": (-55, 60, 25, 45, 15),
    "goddamned": (-60, 65, 30, 50, 20),
    "godel-incompleteness": (20, 55, 65, 20, 55),
    "godet": (65, 30, 30, 10, -20),
    "godwin-law": (-30, 45, 40, 20, 35),
    "golem": (10, 15, 50, 15, 50),
    "gooch": (-50, 20, 0, 10, 15),
    "googly-delivery": (30, 65, 15, 55, -20),
    "gooning": (-70, 55, -45, 25, 20),
    "gospel": (85, 45, 45, 25, 30),
    "gouda": (65, -10, 30, 5, 10),
    "goulash": (55, 15, 30, 10, 30),
    "gourmand": (90, 55, 35, 20, 25),
    "gourmet": (70, 25, 40, 5, 15),
    "governance": (20, 5, 45, 20, 35),
    "governance-framework": (60, 20, 65, 40, 65),
    "governor": (30, 10, 55, 20, 45),
    "graben": (20, 10, 35, 5, 45),
    "gradient": (70, 45, 35, 10, -35),
    "gradient-descent": (65, 75, 55, 45, -25),
    "grafting": (40, 30, 40, 20, 15),
    "grain-structure": (30, 15, 35, 10, 15),
    "grammar": (30, -10, 45, 15, 35),
    "grand-narrative": (40, 30, 55, 20, 65),
    "grand-slam": (95, 85, 65, 45, -50),
    "grand-unification": (100, 65, 90, 10, 85),
    "grandmaster": (90, 50, 60, 10, 50),
    "granite": (20, -5, 40, 0, 40),
    "granola": (55, 20, 25, 25, 5),
    "gratitude": (95, 10, 45, 10, -30),
    "graupel": (-20, 40, 10, 45, 10),
    "gravimetry": (40, 15, 50, 20, 90),
    "gravity": (20, 10, 50, 5, 50),
    "gravity-slingshot": (90, 75, 55, 45, -40),
    "greenflag": (95, 45, 45, 25, -45),
    "greenwashing-risk": (-95, 65, -45, 75, 20),
    "gregarious": (80, 45, 35, 5, -30),
    "grid-tie-inverter": (80, 45, 65, 50, 55),
    "gridiron-clash": (35, 85, 50, 60, 55),
    "griefing": (-90, 65, 20, 50, 20),
    "grimace-shake": (-10, 55, -5, 20, -20),
    "grimoire": (30, 35, 45, 15, 40),
    "grit-count": (25, 20, 35, 15, 10),
    "gritty": (-30, 45, 15, 15, 15),
    "grizzly": (-30, 60, 50, 60, 55),
    "ground-effect": (35, 40, 35, 30, -5),
    "growth-hacking": (55, 75, 45, 60, -20),
    "grundle": (-55, 15, 0, 10, 20),
    "guard-bell": (55, 30, 60, 15, 70),
    "guerrilla": (-50, 65, 35, 50, 20),
    "guidon": (55, 45, 35, 15, 35),
    "guilloche": (45, 25, 30, 10, 20),
    "guipure": (80, 20, 40, 5, 20),
    "gum-bichromate": (60, 30, 40, 10, 20),
    "gungnir-spear": (85, 75, 70, 55, 70),
    "gunshot-residue": (-65, 60, 25, 75, 15),
    "gunwale": (10, 10, 30, 5, 35),
    "gusset": (45, 20, 40, 30, 15),
    "gustatory": (55, 40, 35, 15, 10),
    "gutter": (15, 5, 25, 10, 10),
    "gymnosperm": (70, 25, 45, 5, 45),
    "gyre-ocean": (45, 30, 40, 15, 35),
    "gyro": (70, 35, 25, 30, 15),
    "gyro-precession": (45, 55, 60, 35, 55),
    "gyroscope-drift": (-60, 65, 40, 60, 25),
    "h-bridge-circuit": (75, 65, 55, 55, 45),
    "habeas-corpus": (85, 45, 65, 90, 45),
    "haberdashery": (55, 10, 40, 15, 25),
    "haboob": (-70, 75, -25, 70, 30),
    "haboob-duststorm": (-95, 85, 45, 90, 60),
    "hackly": (-40, 45, 25, 35, 40),
    "hadron": (25, 55, 40, 20, 30),
    "hail": (-55, 55, -30, 60, 15),
    "hail-mary-pass": (45, 95, 20, 100, -70),
    "hairspring": (60, 50, 45, 40, -45),
    "hairspring-oscillation": (75, 55, 45, 40, -60),
    "halation": (55, 55, 25, 10, -50),
    "half-life": (-40, 45, 35, 55, 35),
    "half-wit": (-90, -25, -55, 10, 25),
    "halftone": (35, 15, 30, 5, 5),
    "halocline": (35, 20, 35, 15, 50),
    "halogen-lamp": (60, 50, 35, 15, -20),
    "halophytic": (35, 20, 45, 10, 35),
    "halting-problem": (-55, 65, 75, 25, 95),
    "halyard": (15, 20, 25, 15, 10),
    "halyard-rope": (25, 20, 35, 40, 15),
    "hammer-planishing": (55, 55, 45, 35, 45),
    "hamming-distance": (65, 35, 50, 20, 10),
    "handler": (15, 35, 60, 45, 35),
    "handshake": (30, 20, 30, 30, 0),
    "hangar": (40, -10, 40, 5, 35),
    "hanging-pawns": (-45, 55, -35, 50, 20),
    "hard-carry": (85, 70, 50, 75, 45),
    "hard-fork": (-35, 70, 45, 55, 30),
    "hard-launch": (90, 85, 50, 65, -45),
    "hardness": (35, 15, 55, 5, 65),
    "harmonic": (95, 25, 50, 5, -35),
    "harpy": (-70, 65, -20, 60, -15),
    "hash": (20, 20, 30, 15, 5),
    "hash-collision": (-85, 75, -20, 80, 20),
    "hashing": (25, 25, 35, 30, 10),
    "haste": (-30, 65, -15, 65, -10),
    "haul": (60, 50, 25, 20, -25),
    "haxxor": (45, 65, 55, 40, 15),
    "haze": (-15, -5, -20, 10, 5),
    "head-shot": (85, 80, 55, 25, 20),
    "headcanon": (40, 25, 30, 5, -10),
    "headcount": (5, 20, 35, 30, 30),
    "hearsay": (-35, 30, -20, 20, 5),
    "heart-note": (85, 40, 45, 10, -35),
    "hedging": (40, 10, 45, 20, 25),
    "hedging-strat": (85, 25, 60, 35, 60),
    "hedonism": (70, 75, 15, 20, -35),
    "hegemony": (-65, 40, 60, 30, 60),
    "heirloom": (70, 15, 40, 0, 30),
    "heisenberg-principle": (35, 65, 50, 25, 15),
    "heisenbug": (-70, 55, -30, 45, -5),
    "helheim-mist": (-80, -35, -50, 10, 65),
    "heliopause-static": (55, -50, 90, 5, -100),
    "heliosphere-edge": (85, 15, 75, 5, -100),
    "helipad": (30, 35, 30, 40, 10),
    "hellenistic": (65, 35, 45, 5, 35),
    "hematoma": (-55, 25, -30, 35, 20),
    "hemisphere": (30, 5, 50, 5, 50),
    "hemline": (40, 15, 35, 15, 20),
    "hemlock": (-95, 35, 45, 60, 40),
    "hemoglobin": (30, 10, 35, 10, 10),
    "hemorrhage": (-80, 60, -50, 60, 30),
    "henchman": (-70, 55, 25, 50, 25),
    "hepatitis": (-65, 25, -40, 35, 30),
    "herald": (70, 45, 40, 50, 20),
    "herbicide": (-65, 20, 15, 35, 15),
    "herbivore": (50, -10, 25, 5, 20),
    "herculean": (75, 55, 50, 10, 45),
    "heretic": (-75, 65, 15, 60, 20),
    "heritage": (60, 0, 45, 5, 35),
    "hermeneutics": (45, 25, 50, 15, 40),
    "herringbone": (60, 20, 40, 5, 30),
    "hessian-matrix": (35, 60, 60, 40, 55),
    "heterosexual": (50, 15, 35, 5, 15),
    "heuristic": (45, 40, 35, 25, -5),
    "heuristic-mental": (45, 35, 35, 25, -10),
    "heuristics": (35, 30, 40, 10, 10),
    "hex": (-80, 55, -40, 60, 15),
    "hexadecimal": (15, 15, 35, 10, 15),
    "hierarchy": (-10, 15, 40, 15, 35),
    "hieroglyph": (60, 25, 40, 5, 30),
    "hilt": (45, 20, 55, 15, 65),
    "hippocampus": (45, 15, 35, 5, 10),
    "hitbox": (5, 40, 25, 45, 10),
    "hits-different": (85, 55, 35, 15, -25),
    "hive-body": (35, 5, 45, 5, 50),
    "hoarfrost": (60, 20, 25, 15, 15),
    "hodl": (75, -10, 50, 5, 40),
    "hohmann-transfer": (70, 45, 50, 30, 30),
    "hoist-side": (15, 10, 30, 5, 20),
    "hollow": (-40, -20, -45, 5, -45),
    "holocaust": (-100, 95, 45, 85, 55),
    "holography": (90, 55, 45, 15, -55),
    "homage": (80, 10, 35, 15, 30),
    "homeostasis": (90, -50, 60, 10, 10),
    "hominid": (50, 30, 45, 10, 20),
    "homomorphic-encryption": (100, 50, 70, 40, 60),
    "homonym": (15, 30, 15, 10, 0),
    "homosexual": (55, 35, 35, 10, -10),
    "honey-extractor": (60, 50, 45, 35, 60),
    "hoplite": (35, 45, 45, 35, 40),
    "horizon": (55, 10, 40, 5, -20),
    "horizontal": (20, -10, 35, 15, 15),
    "horsepower": (80, 75, 60, 20, 55),
    "horst": (35, 10, 40, 5, 50),
    "hotfix": (15, 60, 30, 60, 5),
    "houndstooth": (55, 35, 35, 5, 20),
    "hubris": (-75, 50, 40, 20, 20),
    "huitzilopochtli-sun": (70, 90, 70, 60, 50),
    "hull": (20, 0, 45, 5, 50),
    "humanresources": (-10, 15, 30, 35, 25),
    "humerus": (35, 10, 40, 5, 40),
    "humidity": (-35, 20, -20, 20, 20),
    "humidity-index": (-25, 30, -15, 25, 25),
    "hummus": (70, -5, 30, 15, -15),
    "husbando": (65, 45, -20, 10, -40),
    "hybrid": (55, 45, 40, 15, 10),
    "hybridization": (55, 35, 35, 15, 20),
    "hydraulic": (20, 55, 45, 40, 45),
    "hydroponics": (70, 25, 40, 15, -15),
    "hydrostatic": (40, 10, 45, 20, 65),
    "hydrothermal": (55, 45, 40, 30, 25),
    "hyena": (-75, 60, -15, 45, 15),
    "hyetometer-rain": (45, 20, 40, 35, 25),
    "hymenoptera": (15, 50, 20, 55, -20),
    "hyperbaric": (30, 50, 45, 55, 50),
    "hyperbola": (40, 35, 30, 10, -15),
    "hyperbole": (40, 55, 20, 20, -35),
    "hypercapnia": (-85, 65, -55, 80, 35),
    "hyperparameter": (45, 55, 50, 40, 15),
    "hyperreality": (-20, 65, 25, 20, -50),
    "hypersonic": (40, 60, 45, 45, -50),
    "hypertrophy": (55, 50, 45, 15, 35),
    "hypervisor": (30, 25, 45, 20, 25),
    "hyphae": (35, 15, 25, 5, -10),
    "hypo-clear": (60, 15, 40, 45, 15),
    "hypotenuse": (45, 25, 40, 15, 10),
    "hypothalamus": (30, 55, 50, 40, 0),
    "hypoxia": (-90, 60, -65, 85, 30),
    "ice-wine": (90, 20, 35, 5, 15),
    "ick": (-65, 40, -45, 20, 15),
    "icosahedron": (70, 50, 55, 5, 50),
    "id-instinct": (20, 75, -20, 50, 0),
    "idealism": (80, 45, 45, 15, -60),
    "idealistic": (75, 45, 15, 10, -45),
    "idempotency": (85, 15, 65, 20, 50),
    "idiom": (55, 35, 25, 10, -15),
    "igneous": (45, 20, 45, 5, 50),
    "ikat-weaving": (90, 40, 50, 10, 40),
    "imaginary-number": (25, 40, 20, 15, -40),
    "immutability": (95, 10, 70, 5, 90),
    "impact": (-65, 65, 40, 65, 40),
    "impact-glass-tektite": (75, 45, 55, 20, 25),
    "impeccable": (95, 10, 50, 5, -30),
    "impedance-matching": (85, 40, 60, 35, 45),
    "imperative": (15, 30, 35, 15, 10),
    "impressionism": (85, 25, 30, 5, -45),
    "impulse-control": (50, -15, 60, 45, 25),
    "in-situ": (50, -5, 45, 5, 55),
    "incarceration": (-95, 15, -60, 40, 95),
    "incarnation": (80, 40, 50, 15, 45),
    "incel": (-90, 60, -50, 40, 25),
    "incendiary": (-75, 70, 40, 65, 15),
    "incentive": (65, 50, 45, 20, -25),
    "incision": (-30, 45, 10, 50, 10),
    "inclination": (20, 15, 35, 20, 10),
    "incubation": (-25, 15, 35, 55, 25),
    "incunabula-rare": (90, 35, 60, 10, 85),
    "incunabulum": (90, 25, 60, 5, 85),
    "indexing-latency": (-60, 50, 40, 60, 20),
    "indictment": (-65, 55, -35, 50, 30),
    "indignation": (-60, 65, 35, 55, 25),
    "indigo": (40, -5, 35, 0, 25),
    "induction": (40, 30, 35, 20, 10),
    "inductive-coupling": (50, 60, 50, 45, 25),
    "inductor": (10, 30, 25, 15, 10),
    "industry-plant": (-45, 30, -20, 10, 15),
    "inertia": (15, -45, 55, 5, 70),
    "inertial-measure-unit": (80, 65, 55, 75, 15),
    "infatuation": (60, 75, -40, 50, -45),
    "inferno": (-85, 75, 30, 55, 20),
    "infidelity": (-95, 75, -60, 60, 45),
    "infield-fly": (-20, 40, 20, 65, -10),
    "infiltration": (-75, 70, 45, 70, 10),
    "infinity-loop": (60, 40, 55, 5, -60),
    "inflection": (35, 20, 30, 15, 5),
    "infrastructure": (35, 10, 45, 5, 45),
    "infusion": (55, -5, 25, 5, -15),
    "ingot-mold": (40, 15, 50, 10, 90),
    "initialization-vector": (40, 35, 50, 45, 25),
    "initiative": (80, 75, 55, 80, -30),
    "injection": (35, 20, 40, 15, 5),
    "injunction": (-40, 45, 35, 55, 25),
    "inkwell": (40, 10, 25, 10, 20),
    "inlay": (80, 35, 40, 10, 20),
    "innit": (35, 25, 10, 5, 0),
    "inoculation": (70, 35, 55, 50, 10),
    "inquisitive": (70, 45, 25, 10, -20),
    "insourcing": (35, 15, 35, 15, 10),
    "instance": (15, 15, 25, 20, 10),
    "instantiation": (30, 30, 30, 20, 5),
    "insulator": (35, -30, 40, 5, 20),
    "insurgency": (-70, 65, 25, 55, 25),
    "integral": (15, 45, 40, 35, 40),
    "integral-gain": (60, 45, 55, 55, 25),
    "integrity": (85, 10, 50, 5, 25),
    "intentional-grounding": (-85, 75, -40, 85, 20),
    "intercept": (10, 65, 45, 75, 10),
    "intercooler": (55, 30, 40, 10, 40),
    "interfacing": (30, 5, 35, 20, 20),
    "interference": (-60, 65, 25, 55, 15),
    "interglacial": (65, 20, 40, 15, 35),
    "interoperability": (85, 55, 60, 30, 25),
    "interpreter": (20, 25, 30, 15, 5),
    "interrogation": (-80, 60, 45, 65, 35),
    "intersection": (60, 25, 40, 20, 15),
    "intertextuality": (55, 35, 40, 10, 10),
    "intertidal-zone": (55, 55, 40, 75, 45),
    "interval": (10, 5, 20, 30, 0),
    "intestines": (20, 25, 30, 35, 20),
    "intimacy": (90, 45, 45, 5, -50),
    "intrinsic-motivation": (85, 55, 55, 20, -30),
    "introjection": (-45, 30, -35, 15, 25),
    "intrusion-detection": (65, 70, 55, 90, 30),
    "intubation": (-70, 60, -55, 60, 30),
    "invasive": (-70, 65, 40, 60, 20),
    "invasive-species": (-85, 65, 40, 60, 20),
    "inverted-jenny": (90, 75, 45, 35, 35),
    "invincible": (90, 60, 50, 20, 35),
    "ion-charge": (10, 45, 30, 35, -15),
    "ionopause": (40, 40, 45, 15, -85),
    "ionosphere": (55, 45, 40, 20, -70),
    "ionospheric-reflection": (55, 40, 55, 20, -75),
    "ionospheric-scint": (-65, 65, 35, 55, -85),
    "ipo": (70, 60, 45, 45, 25),
    "irghizite-black": (65, 35, 50, 15, 45),
    "iron": (40, 5, 50, 5, 45),
    "iron-age": (45, 50, 50, 20, 55),
    "irony": (35, 40, 25, 15, 5),
    "irrigation": (65, 15, 45, 25, 15),
    "island-mode-ops": (75, 60, 60, 75, 45),
    "iso-noise": (-60, 50, -40, 45, 10),
    "iso-sensitivity": (20, 60, 40, 55, -5),
    "isobar": (20, 10, 30, 20, 20),
    "isobar-chart": (30, 25, 40, 45, 25),
    "isobaric": (25, 10, 40, 30, 40),
    "isochronous": (75, -50, 60, 10, 45),
    "isolated-pawn": (-55, 40, -45, 35, 15),
    "isomorphism": (70, 25, 45, 10, 10),
    "isostatic-rebound": (45, -10, 60, 5, 90),
    "isotherm": (25, 5, 30, 15, 15),
    "isotope": (15, 30, 40, 25, 45),
    "isotope-decay": (-45, 40, 30, 30, 40),
    "isthmus": (40, 15, 25, 10, 20),
    "italics": (55, 25, 30, 5, -15),
    "ivory": (75, -5, 40, 0, 15),
    "jacobian-matrix": (40, 55, 55, 35, 45),
    "jacquard": (75, 40, 45, 10, 40),
    "jadeite": (85, 15, 50, 5, 55),
    "jargon": (-20, 40, 30, 20, 20),
    "jaundice": (-50, 20, -30, 35, 20),
    "jawline-check": (30, 40, 35, 25, 5),
    "jet-stream": (45, 55, 40, 25, -45),
    "jetsam": (-35, 15, -35, 5, 5),
    "jetty": (35, 15, 30, 5, 35),
    "jiafei-product": (65, 70, 30, 50, -40),
    "jitter-buffer": (-45, 50, -25, 55, 15),
    "jizzstain": (-90, 40, 0, 20, 15),
    "joinery": (85, 30, 55, 15, 45),
    "jolly-roger": (-75, 70, 45, 55, 40),
    "jormungand-serpent": (-95, 75, 70, 65, 85),
    "jotunheim-giant": (-45, 55, 50, 35, 75),
    "joust": (75, 75, 45, 65, 40),
    "jpeg-artifact": (-60, 45, -35, 45, 10),
    "judicial": (35, 10, 50, 20, 50),
    "julia-set-fractal": (90, 65, 60, 15, -70),
    "jurisdiction": (10, 20, 45, 25, 35),
    "jurisprudence": (30, -5, 45, 5, 40),
    "justification": (35, -5, 45, 10, 20),
    "k-nearest-neighbors": (75, 50, 55, 30, 20),
    "kali-fierce": (-35, 95, 75, 65, 45),
    "kalman-filter-state": (85, 60, 65, 50, 35),
    "kamacite-alloy": (45, 10, 50, 5, 90),
    "kanban": (30, 20, 35, 15, 5),
    "karman-line": (90, 55, 60, 20, -50),
    "karst": (35, 20, 25, 5, 40),
    "karst-topography": (55, 35, 45, 15, 70),
    "keel": (30, 0, 50, 0, 60),
    "keepsake": (75, 20, 35, 0, -10),
    "kefir": (40, 10, 20, 15, -10),
    "kelp": (45, 25, 35, 10, 50),
    "kerf": (10, 40, 25, 35, 10),
    "kernel": (20, 10, 45, 10, 30),
    "kernel-panic": (-100, 90, -60, 95, 45),
    "kerning": (75, 15, 40, 20, -10),
    "kerning-pair": (55, 20, 30, 10, -5),
    "kerolox": (50, 70, 50, 45, 55),
    "kerosene": (-25, 50, 25, 45, -5),
    "key-light": (85, 55, 50, 20, -20),
    "keystone-species": (90, 45, 65, 20, 45),
    "kidneys": (50, 20, 45, 10, 20),
    "kilim-flatweave": (75, 20, 55, 5, 55),
    "kimbap": (60, 20, 20, 15, 10),
    "kimchi": (55, 35, 20, 5, 10),
    "kinesiology": (45, 35, 40, 10, 15),
    "kinship": (75, 15, 40, 5, 15),
    "klein-bottle": (55, 55, 45, 10, -50),
    "knife": (-5, 35, 30, 20, 15),
    "knighthood": (85, 55, 50, 20, 45),
    "knob-head": (-80, 50, 15, 30, 15),
    "knobhead": (-85, 55, 20, 35, 20),
    "knot": (15, 20, 20, 35, 0),
    "knuckleball-flutter": (10, 50, -10, 35, -30),
    "kombucha": (45, 20, 15, 10, -10),
    "koreaboo": (-60, 75, -20, 35, 10),
    "kpi": (15, 45, 35, 55, 15),
    "kraken": (-85, 75, 55, 65, 55),
    "kuiper-belt-obj": (65, -25, 70, 10, 90),
    "kukulkan-feathered": (80, 55, 60, 10, 50),
    "kunzite": (75, 35, 30, 5, -15),
    "la-niña": (55, 50, 30, 60, 15),
    "labradorescence": (80, 45, 40, 10, 15),
    "laccolith-intrusion": (35, 20, 60, 5, 85),
    "lackey": (-80, 25, -55, 35, 15),
    "lacquer": (65, 30, 40, 15, 20),
    "lahar-mudflow": (-100, 90, 45, 100, 85),
    "lambda-calculus": (65, 45, 60, 15, 35),
    "lamé": (70, 55, 30, 15, 15),
    "landscape": (80, -15, 40, 0, 25),
    "lapis-lazuli": (80, 10, 45, 5, 50),
    "laplacian-operator": (30, 50, 60, 35, 65),
    "larceny": (-60, 45, 10, 30, 20),
    "larva": (10, -20, -35, 5, 10),
    "lasagna": (70, 15, 30, 15, 30),
    "laser-diode": (85, 75, 55, 60, -25),
    "late-harvest": (85, 25, 40, 10, 20),
    "latency": (-55, 40, -35, 50, 15),
    "latency-ms": (-70, 55, -40, 65, 10),
    "latex": (10, 35, 20, 15, -10),
    "lathe": (10, 45, 40, 25, 50),
    "lathe-bench": (40, 55, 50, 30, 60),
    "latitude": (10, 0, 35, 5, 20),
    "leading": (50, -10, 35, 5, 15),
    "learned-helplessness": (-95, -35, -60, 15, 50),
    "leather": (55, 10, 45, 5, 35),
    "ledger": (20, 10, 40, 15, 25),
    "lees": (15, -5, 20, 5, 35),
    "leetspeak": (50, 45, 35, 5, -25),
    "leeward": (50, -25, 30, 5, -20),
    "leftnocrumbs": (80, 50, 50, 0, -35),
    "leg-break": (25, 60, 20, 50, 15),
    "legacy": (80, 15, 55, 5, 45),
    "legacy-system": (-65, -15, 35, 10, 85),
    "legal-tender": (65, 15, 60, 35, 50),
    "legato": (70, -30, 30, 0, -10),
    "legionnaire": (40, 40, 50, 30, 45),
    "legislation": (20, 15, 45, 35, 45),
    "legs-wine": (50, 15, 25, 5, 10),
    "lemma": (25, 15, 30, 10, 10),
    "lens-flare": (60, 70, 25, 15, -45),
    "lenticular": (55, 15, 30, 5, -30),
    "lepidoptera": (70, 35, 15, 5, -50),
    "lesion": (-55, 25, -30, 35, 20),
    "leukemia": (-80, 35, -50, 40, 40),
    "leukocyte": (35, 20, 35, 30, -5),
    "level-set": (45, -15, 50, 45, 20),
    "leverage": (20, 55, 45, 45, 15),
    "leverage-ratio": (-45, 80, 60, 70, 60),
    "lexicography": (55, 5, 50, 10, 45),
    "lexicon": (60, 15, 50, 10, 25),
    "liability": (-45, 35, -25, 40, 30),
    "libido": (60, 75, 30, 35, -40),
    "libretto": (35, 10, 25, 5, 20),
    "libyan-desert-glass": (90, 20, 65, 10, 35),
    "lichen": (55, -25, 30, 0, 15),
    "lidar-scan-point": (80, 50, 60, 55, 15),
    "liege": (75, 15, 50, 20, 45),
    "life-cycle-assess": (75, 35, 65, 45, 55),
    "lift": (65, 35, 40, 5, -50),
    "ligament": (15, 5, 25, 5, 20),
    "ligature": (70, 20, 40, 5, -20),
    "light-leak": (-55, 65, -30, 45, -35),
    "lighthouse": (85, 40, 50, 55, 45),
    "limerence": (45, 60, -35, 30, -35),
    "limestone": (15, -5, 20, 0, 25),
    "liminal": (10, 30, -35, 35, -10),
    "liminality": (15, 45, -35, 35, -15),
    "limit-point": (40, 35, 45, 50, 20),
    "limited-slip": (75, 50, 60, 25, 55),
    "limitless": (90, 55, 55, 15, -50),
    "line-of-scrimmage": (10, 65, 55, 80, 45),
    "lineage": (55, -5, 45, 5, 40),
    "linear-algebra": (25, 35, 45, 25, 35),
    "linen": (60, -10, 25, 5, -10),
    "linens": (60, -25, 35, 10, -20),
    "lining": (65, -10, 45, 10, 15),
    "liquid": (50, 10, 35, 10, 10),
    "liquid-oxygen": (20, 65, 45, 60, 50),
    "liquidation": (-65, 50, -35, 60, 35),
    "liquidity-pool": (70, 45, 55, 50, 50),
    "lithium-polymer": (85, 60, 55, 50, 45),
    "lithograph": (45, 10, 30, 5, 25),
    "lithophytic": (45, 10, 45, 5, 65),
    "lithosphere": (45, 5, 55, 5, 85),
    "lithospheric-flexure": (35, 10, 55, 5, 95),
    "litigant": (-15, 40, 10, 30, 10),
    "litigation": (-65, 55, -40, 60, 35),
    "liturgy": (55, -5, 40, 15, 45),
    "livestream": (55, 60, 35, 50, -15),
    "livor-mortis": (-95, -15, 15, 0, 90),
    "livvy-dunne": (45, 40, 20, 10, -25),
    "load-balancer": (85, 35, 60, 55, 50),
    "loadbalancer": (45, 20, 45, 40, 15),
    "lobbyist": (-40, 45, 30, 50, 15),
    "lockedin": (60, 55, 50, 45, 15),
    "locket": (65, 25, 25, 10, -20),
    "locust": (-75, 65, -35, 65, -15),
    "loess": (40, -5, 25, 5, 15),
    "log-aggregation": (60, 40, 55, 45, 60),
    "logarithm": (35, 25, 45, 20, 10),
    "loggia": (75, -10, 35, 5, 30),
    "logic": (40, -10, 50, 5, 15),
    "logic-gate-and": (25, 15, 45, 15, 25),
    "logic-gate-xor": (35, 35, 45, 25, 20),
    "longbow": (25, 50, 35, 40, 10),
    "longevity": (75, 10, 50, 5, 25),
    "longevity-scent": (60, 10, 55, 15, 50),
    "longitude": (10, 0, 35, 5, 20),
    "looks-mog": (40, 50, 45, 10, 10),
    "looksmaxxing": (30, 45, 35, 15, 5),
    "loquacious": (35, 45, 15, 15, -10),
    "lorawan-gate": (65, 30, 60, 40, 35),
    "lorenz-equation": (45, 85, 50, 35, -25),
    "lost-media": (35, 60, -10, 50, 45),
    "love-bombing": (-60, 55, 35, 40, -20),
    "lovebombing": (-70, 65, 45, 45, -25),
    "low-code": (45, 20, 30, 15, -15),
    "low-density-parity": (80, 45, 65, 40, 40),
    "low-hanging-fruit": (40, 15, 45, 35, -20),
    "low-key": (40, -35, 35, 10, 55),
    "lowercase": (25, -20, 20, 0, -10),
    "lsm-tree": (65, 30, 55, 15, 75),
    "lubricant": (40, -10, 25, 10, 5),
    "luminary": (85, 40, 50, 5, -20),
    "luminol-glow": (30, 55, 35, 50, -35),
    "luminous": (85, 35, 40, 0, -50),
    "lunar-lander": (95, 65, 60, 55, 70),
    "lunge": (65, 85, 55, 80, -25),
    "lurk-moar": (-60, -30, 20, 10, 25),
    "luster": (70, 25, 35, 0, 15),
    "lut-profile": (45, 25, 40, 15, 5),
    "luthier-bench": (85, 35, 60, 25, 65),
    "lymph-node": (20, 25, 25, 40, 15),
    "lymphatic": (25, 15, 30, 15, 10),
    "lymphocyte": (35, 25, 35, 35, -5),
    "lysosome": (40, 55, 40, 25, -15),
    "macaron": (80, 30, 20, 5, -40),
    "maceration": (40, 25, 30, 15, 25),
    "maceration-tank": (45, 20, 45, 15, 55),
    "mach": (20, 55, 45, 30, -15),
    "mach-meter": (15, 35, 35, 40, 5),
    "machine-learning": (45, 65, 50, 35, 15),
    "macro-lens": (80, 50, 45, 15, -20),
    "madder-root": (55, 10, 35, 5, 40),
    "maelstrom": (-80, 60, -50, 60, 40),
    "magenta": (75, 50, 30, 10, -20),
    "maggot": (-100, 50, -55, 55, 25),
    "magma": (-15, 60, 40, 50, 35),
    "magnetism": (40, 35, 40, 10, 20),
    "magnetogram-flux": (50, 65, 60, 45, -55),
    "magnetometer-bias": (-55, 45, 35, 55, 20),
    "magnetopause": (50, 55, 50, 30, -95),
    "magnetosheath-turb": (-65, 85, 55, 75, -65),
    "magnetosphere": (50, 30, 50, 10, -50),
    "magnetotail-stretch": (-55, 75, 65, 55, -85),
    "magnus-effect": (75, 65, 45, 35, -25),
    "maillard-reaction": (90, 45, 45, 20, 15),
    "main": (40, 25, 40, 5, 10),
    "main-character-syndrome": (-75, 75, 55, 40, 35),
    "main-pop-girl": (95, 75, 60, 30, -50),
    "main-squeeze": (85, 45, 45, 15, -30),
    "mainnet": (60, 45, 50, 55, 35),
    "mainspring": (50, 55, 50, 45, 55),
    "mainspring-tension": (45, 60, 55, 50, 65),
    "malachite": (65, 25, 40, 5, 45),
    "malfeasance": (-85, 50, 30, 35, 30),
    "malice": (-85, 45, 25, 30, 25),
    "malingerer": (-75, 15, -40, 25, 15),
    "mall-goth": (35, 40, 15, 10, 25),
    "malleable": (10, 15, -40, 15, -15),
    "mallet": (55, 45, 45, 25, 45),
    "mammatus": (-20, 55, 20, 45, 10),
    "mammogram": (-10, 45, -20, 50, 20),
    "mammoth": (45, 15, 50, 5, 60),
    "mandelbrot": (95, 60, 45, 10, -60),
    "mandelbrot-set-deep": (100, 60, 65, 15, -75),
    "mandible": (30, 15, 40, 10, 35),
    "mandrel": (15, 20, 35, 10, 35),
    "maneuver": (45, 65, 50, 55, -15),
    "maneuver-burn": (35, 75, 55, 85, 30),
    "mangrove": (50, 10, 45, 5, 35),
    "manifesting": (55, 40, 30, 15, -45),
    "manifesting-this": (75, 45, 40, 15, -55),
    "manifold": (30, 45, 40, 15, 25),
    "manifold-pressure": (25, 55, 45, 50, 45),
    "mannequin": (25, -10, 30, 10, 25),
    "manometer": (10, 20, 30, 40, 10),
    "manorial": (45, 10, 40, 5, 45),
    "mantis": (25, 50, 30, 45, -10),
    "mantle-convection": (45, 55, 65, 15, 95),
    "mantle-plume": (25, 65, 50, 40, 75),
    "manuscript": (50, 10, 30, 5, 25),
    "marble": (50, 5, 35, 0, 35),
    "margin-call": (-100, 95, -60, 100, 50),
    "marginalia-notes": (65, 45, 30, 5, 15),
    "marina": (70, 15, 35, 5, 10),
    "marinade": (50, 15, 25, 10, 5),
    "maritime": (20, 10, 30, 5, 25),
    "market-cap": (75, 50, 65, 30, 80),
    "marking-gauge": (50, 15, 45, 25, 20),
    "markov-chain": (45, 35, 40, 20, 15),
    "marquess": (65, 30, 50, 20, 45),
    "marrow": (30, 0, 30, 5, 35),
    "marsupial": (55, 15, 25, 5, -10),
    "martyr": (-10, 65, 45, 50, 55),
    "mask": (30, 35, 50, 40, 45),
    "mass": (15, 5, 45, 5, 60),
    "masthead": (75, 45, 55, 30, 95),
    "matcha": (60, 25, 30, 15, -15),
    "materialism": (15, 20, 40, 15, 70),
    "matriarch": (60, 20, 50, 10, 40),
    "matrix-math": (10, 35, 45, 25, 30),
    "matte": (40, -30, 25, 0, 10),
    "matte-box": (35, 10, 40, 20, 20),
    "maximum-power-point": (85, 50, 60, 55, 40),
    "maxwell-demon": (-40, 75, 65, 45, 35),
    "me-gusta": (75, 45, 30, 5, -20),
    "meadow": (70, -30, 35, 0, -25),
    "meatrider": (-75, 40, -40, 15, 15),
    "mediation": (40, 10, 30, 25, 10),
    "medulla": (35, 10, 50, 55, 20),
    "megafauna": (45, 15, 55, 5, 40),
    "megalith": (60, 15, 70, 5, 85),
    "meiosis": (45, 50, 35, 35, 10),
    "melanoma": (-75, 40, -45, 50, 35),
    "melee": (-45, 85, 20, 75, 30),
    "melliferous": (75, 15, 30, 5, -15),
    "melodic": (95, 35, 45, 5, -40),
    "melodrama": (-35, 65, -15, 30, 15),
    "meme-coin": (30, 75, -10, 30, -35),
    "menhir": (35, 5, 45, 0, 75),
    "meninges": (15, 30, 40, 55, 30),
    "mentorship": (75, 35, 50, 15, -10),
    "mercury": (10, 40, 20, 35, 40),
    "merge-conflict": (-90, 80, -40, 95, 30),
    "meridian": (25, 5, 40, 10, 35),
    "meridian-transit": (35, 10, 55, 15, 55),
    "mesocyclone": (-80, 80, 45, 75, 40),
    "mesolithic": (35, 10, 35, 5, 45),
    "mesopotamian": (55, 25, 50, 5, 45),
    "mesoscale-convective": (-65, 95, 60, 95, 45),
    "mesosphere": (30, -15, 35, 5, -75),
    "metabolic": (45, 50, 40, 30, 5),
    "metacognition": (70, 30, 55, 10, 5),
    "metadata": (15, 5, 30, 10, 5),
    "metadata-tag": (35, 25, 40, 35, 5),
    "metamorphic": (50, 15, 50, 5, 55),
    "metaphor": (55, 35, 35, 10, -15),
    "metaphysics": (50, 30, 45, 5, -20),
    "metatag": (10, 5, 20, 10, 0),
    "meteor": (30, 50, 20, 45, -20),
    "meticulous": (65, 15, 50, 10, 15),
    "metonymy": (45, 25, 30, 5, -10),
    "metronome": (0, 10, 45, 55, 20),
    "mewing": (20, 15, 35, 0, 10),
    "mewing-streak": (35, 20, 40, 10, 5),
    "michelin": (75, 45, 50, 10, 35),
    "micro-electromechanical": (85, 55, 60, 30, 25),
    "microburst": (-85, 75, -20, 80, 35),
    "microfauna": (15, 10, -10, 5, -40),
    "microfilm-reel": (30, 10, 35, 15, 40),
    "microservice-mesh": (75, 50, 60, 35, 45),
    "microservices": (40, 35, 35, 20, -10),
    "mictlantecuhtli-lord": (-100, 50, 70, 40, 90),
    "middlegame": (55, 60, 45, 50, 30),
    "middleware": (20, 15, 30, 25, 10),
    "milankovitch-cycle": (65, 20, 70, 5, 100),
    "millennium": (65, 10, 55, 5, 50),
    "millinery": (75, 35, 45, 15, 20),
    "mimo-antenna": (90, 65, 60, 45, 10),
    "minaret": (80, 45, 55, 20, -45),
    "minimalism": (45, -45, 50, 5, -15),
    "mining": (15, 40, 30, 20, 35),
    "minion": (-60, 25, -50, 35, 10),
    "mint-mark": (65, 35, 45, 15, 20),
    "miranda-rights": (50, 35, 60, 85, 35),
    "mirror": (5, 20, 15, 5, 5),
    "misandrist": (-100, 65, 30, 45, 35),
    "miscreant": (-85, 60, 15, 45, 25),
    "misdemeanor": (-40, 30, -10, 25, 15),
    "mise-en-scene": (55, 30, 45, 10, 35),
    "misogynist": (-100, 65, 30, 45, 35),
    "mist": (35, -20, -10, 15, -15),
    "miter-saw": (45, 60, 50, 55, 55),
    "mitochondria": (70, 60, 50, 15, 10),
    "mitochondrion": (85, 65, 55, 15, 15),
    "mitosis": (45, 50, 35, 35, 10),
    "mizzen": (20, 10, 20, 5, 15),
    "mjolnir-hammer": (90, 80, 75, 60, 95),
    "mob-mentality": (-75, 75, -20, 65, 20),
    "mob-wife": (60, 45, 50, 15, 30),
    "mobius-strip-loop": (60, 45, 50, 15, -40),
    "modular": (45, 10, 40, 5, 10),
    "modus-operandi": (20, 45, 50, 35, 30),
    "mogged": (35, 50, 50, 10, 10),
    "mogging": (35, 50, 50, 10, 15),
    "mohair-angora": (85, -15, 45, 5, -25),
    "moka-pot-brew": (70, 60, 50, 45, 45),
    "mokume-gane": (90, 45, 55, 10, 55),
    "molarity": (25, 20, 35, 20, 10),
    "moldavite-green": (95, 65, 60, 25, 15),
    "molecule": (45, 30, 40, 5, -10),
    "mollusk": (10, -10, 10, 0, 15),
    "momentum": (55, 55, 45, 45, 20),
    "money-laundering": (-90, 65, 30, 50, 35),
    "monism": (45, -5, 55, 5, 65),
    "monocle": (25, 20, 40, 5, 5),
    "monocoque": (55, 15, 60, 5, 80),
    "monocot": (45, 15, 30, 10, 15),
    "monoculture": (-40, -10, 35, 10, 25),
    "monogamous": (50, -10, 40, 5, 35),
    "monogamy": (55, -5, 45, 5, 30),
    "monolith": (50, 10, 65, 5, 80),
    "monolith-app": (-35, 15, 65, 10, 95),
    "monolithic": (-30, 5, 50, 10, 90),
    "monologue": (35, 30, 45, 15, 25),
    "monophonic": (20, -15, 10, 0, 5),
    "monoplane": (30, 15, 35, 5, -10),
    "monopoly": (-40, 30, 50, 10, 45),
    "monotheism": (45, 10, 55, 5, 55),
    "monotone-voice": (-35, -55, 45, 40, 45),
    "monotreme": (40, 10, 20, 5, 5),
    "monsoon": (20, 50, -15, 55, 25),
    "monte-carlo-sim": (60, 45, 45, 25, 10),
    "moon-phase": (85, 15, 45, 5, -25),
    "moonlighting": (30, 35, 20, 25, -15),
    "moor": (45, -30, 35, 5, 40),
    "moraine": (45, 10, 40, 5, 45),
    "morbidity": (-95, 45, -60, 50, 75),
    "mordant-fixative": (25, 15, 40, 35, 25),
    "morganite": (85, 20, 30, 5, -20),
    "morpheme": (30, 5, 35, 5, 10),
    "morphology-lang": (35, 15, 40, 10, 25),
    "mortality": (-100, 55, -70, 60, 85),
    "mortise": (65, 20, 50, 10, 50),
    "mosaics": (90, 50, 40, 15, 30),
    "moss": (80, -35, 25, 0, -10),
    "mother-fucking": (-60, 80, 35, 60, 10),
    "mother-is-mothering": (95, 60, 60, 15, -40),
    "motherboard": (35, 15, 40, 5, 30),
    "mothered": (85, 55, 45, 5, -25),
    "motherfucker": (-85, 65, 40, 55, 15),
    "motherfucking": (-65, 85, 40, 65, 15),
    "motion": (50, 45, 45, 10, -20),
    "moussaka": (65, 15, 30, 10, 25),
    "mousse": (70, -15, 25, 5, -50),
    "mozzarella": (70, -5, 30, 10, -5),
    "mudslide": (-85, 65, -50, 65, 45),
    "muffing": (-40, 10, -30, 5, 10),
    "multi-junction-solar": (100, 65, 75, 30, 65),
    "muppet": (-50, 40, -20, 10, 5),
    "mural": (65, 25, 40, 10, 35),
    "muscular": (65, 45, 55, 10, 35),
    "muslin": (45, -30, 25, 5, -20),
    "must-juice": (60, 25, 30, 20, 25),
    "mustelid": (35, 60, 35, 40, 10),
    "mutagen": (-75, 50, -15, 30, 25),
    "mutation": (-15, 65, 25, 45, 10),
    "mutex-lock": (20, 25, 45, 40, 45),
    "mutiny": (-90, 60, 45, 60, 40),
    "mutualism-symbiosis": (95, 35, 55, 10, 25),
    "mycelium": (40, 10, 30, 5, -15),
    "mycorrhizae": (90, 20, 50, 10, 30),
    "myelin": (65, -15, 45, 5, 15),
    "n00b": (-75, 35, -45, 10, 20),
    "nacelle": (15, 10, 25, 5, 15),
    "nachos": (75, 55, 15, 40, 15),
    "nadir": (-55, -20, -50, 10, 45),
    "nanometer": (10, 10, -10, 5, -85),
    "nanotechnology": (55, 50, 45, 15, -45),
    "narcissism": (-85, 50, 45, 30, 35),
    "narcissistic": (-90, 50, 45, 20, 35),
    "narcosis": (-70, 55, -45, 50, 20),
    "narrative": (45, 25, 35, 15, 15),
    "nautical": (35, 10, 30, 5, 20),
    "near-field-comm": (75, 45, 55, 60, -15),
    "near-field-rect": (70, 45, 50, 40, 20),
    "nebula": (95, 35, 40, 5, -75),
    "neckbeard": (-80, 15, -40, 5, 50),
    "necropolis": (-75, 10, 35, 10, 60),
    "necrosis": (-95, 35, -55, 65, 50),
    "nectar": (90, 35, 25, 10, -35),
    "nectar-flow": (85, 35, 45, 15, -25),
    "negative-aura": (-80, 35, -45, 10, 40),
    "negative-carrier": (35, 15, 40, 35, 40),
    "negativeaura": (-65, 35, -45, 10, 35),
    "negligence": (-65, 25, -30, 20, 25),
    "negotiation": (25, 45, 35, 45, 15),
    "neolithic": (45, 25, 40, 10, 40),
    "nephelometer-cloud": (55, 30, 45, 25, -15),
    "nepo-baby": (-35, 25, 10, 5, 10),
    "nerf": (-55, 35, -40, 30, 15),
    "neumann-line-iron": (65, 25, 55, 15, 85),
    "neural-network": (55, 70, 55, 40, -10),
    "neurological": (30, 50, 35, 45, 0),
    "neuron": (35, 55, 30, 40, -35),
    "neuroplasticity": (85, 40, 55, 15, -35),
    "neurotic": (-60, 60, -45, 40, 10),
    "neurotransmitter": (65, 65, 40, 30, -30),
    "neutrino": (20, 35, 20, 5, -70),
    "neutron": (20, 20, 30, 5, 10),
    "neutron-star-crust": (35, 75, 75, 10, 100),
    "ngmi": (-80, 30, -45, 15, 25),
    "nightshade": (-90, 45, 40, 60, 35),
    "nihilism": (-70, -30, -50, 0, 35),
    "nimbostratus": (-45, 15, 10, 30, 25),
    "nirvana": (100, -60, 50, 0, -70),
    "nirvana-release": (100, -60, 60, 0, -85),
    "nitrogen-fixation": (60, 10, 40, 5, 10),
    "nitrogen-tap": (85, 45, 35, 30, -35),
    "nitrous-oxide": (30, 95, 40, 85, -30),
    "nitrox": (45, 55, 40, 50, 20),
    "nitwit": (-75, 30, -35, 15, 15),
    "no-cap": (65, 35, 45, 15, 10),
    "no-code": (55, 25, 35, 20, -20),
    "no-scope": (95, 90, 50, 20, -40),
    "noble-gas": (75, -45, 55, 5, -45),
    "noble-rot": (45, 25, 30, 5, 30),
    "nocebo-effect": (-60, 25, -30, 20, 20),
    "nocturnal": (-25, 35, 25, 10, 20),
    "nocturnal-dial": (70, 15, 45, 25, 35),
    "nocturne": (75, -45, 35, 0, 15),
    "nominalism": (25, -5, 40, 5, 15),
    "non-euclidean-geom": (45, 65, 45, 20, -55),
    "nonce-value": (25, 40, 35, 30, 10),
    "noob-tube": (-65, 50, -15, 40, 5),
    "notary": (20, -5, 25, 15, 15),
    "npc-energy": (-85, -45, -60, 5, 45),
    "nuance": (50, 10, 35, 5, 0),
    "nucleocapsid": (35, 20, 40, 10, 35),
    "null-set": (-35, -20, -50, 5, -20),
    "numbers-station": (-40, 65, 55, 80, 60),
    "numismatics": (60, 15, 50, 5, 45),
    "numismatist": (70, 20, 50, 10, 35),
    "nutation-axial": (25, 25, 45, 20, 45),
    "nyquist-frequency": (65, 75, 60, 85, 5),
    "oasis": (75, -20, 30, 15, -15),
    "oatmeal": (60, -25, 35, 30, 15),
    "obelisk": (40, 20, 50, 5, 55),
    "object-oriented": (35, 25, 40, 15, 10),
    "objective": (55, 50, 55, 60, 30),
    "obliquity-ecliptic": (55, 15, 55, 10, 65),
    "observability": (90, 45, 65, 35, 45),
    "observatory": (75, 20, 45, 5, 40),
    "obsidian": (35, 15, 30, 0, 30),
    "obstinate": (-65, 25, 45, 15, 40),
    "obverse-face": (45, 10, 40, 5, 40),
    "oceanic-crust": (40, 5, 55, 5, 90),
    "ochre": (35, -10, 25, 0, 35),
    "octant-measure": (55, 25, 45, 20, 35),
    "octonion": (20, 60, 50, 15, -15),
    "off-spinner": (25, 55, 20, 45, 15),
    "offboarding": (-15, 15, 25, 20, 15),
    "offshore": (-20, 35, 30, 25, 5),
    "offspring": (70, 35, 25, 25, -25),
    "ohio-rizz": (-20, 40, -10, 15, -5),
    "oily": (-35, 15, -15, 10, 20),
    "oleander": (20, 30, 30, 55, 25),
    "olfactory": (40, 30, 30, 10, 5),
    "olfactory-bulb": (55, 40, 40, 20, -5),
    "olfactory-pyramid": (70, 25, 55, 10, 40),
    "oligopoly": (-30, 20, 45, 10, 40),
    "omen": (-30, 55, -20, 65, 10),
    "omnivore": (35, 25, 30, 15, 10),
    "on-god": (85, 65, 55, 40, 35),
    "on-the-radar": (20, 45, 35, 55, 10),
    "onboarding": (45, 35, 20, 40, 0),
    "one-time-pad": (100, -50, 75, 5, 95),
    "ontology": (30, 15, 40, 0, 40),
    "oort-cloud-icy": (75, -40, 80, 5, 100),
    "op-meta": (55, 65, 50, 20, 35),
    "open-source": (95, 60, 50, 10, -25),
    "opening": (60, 55, 40, 55, 10),
    "operative": (40, 55, 50, 40, 15),
    "opposition": (55, 65, 55, 70, 35),
    "opps": (-95, 85, 25, 90, 40),
    "optic-nerve": (70, 50, 50, 55, -10),
    "optical-fiber-core": (95, 45, 65, 35, 85),
    "optimistic": (85, 45, 35, 5, -40),
    "optimization": (55, 25, 45, 10, 10),
    "options": (30, 55, 35, 35, 10),
    "oracle": (75, 45, 50, 30, 10),
    "orbit-decay": (-75, 55, -50, 65, 60),
    "orbital-mechanics": (50, 25, 55, 30, 55),
    "orbital-velocity": (50, 70, 55, 45, 20),
    "orbiting": (-50, 25, -35, 15, 15),
    "orchestration": (45, 40, 45, 35, 15),
    "orchid": (85, 40, 30, 10, -20),
    "ordinality": (35, 10, 45, 15, 50),
    "ordinance": (5, 5, 40, 15, 35),
    "ordnance": (-65, 60, 55, 70, 55),
    "organdy": (55, 10, 25, 5, -30),
    "organelle": (60, 30, 45, 10, -5),
    "organza": (75, 20, 25, 5, -55),
    "ornamentation": (85, 45, 40, 20, 20),
    "orogeny-building": (65, 55, 75, 20, 100),
    "orthogonal-freq-div": (80, 55, 65, 40, 25),
    "orthography": (40, 5, 45, 10, 40),
    "oscillation": (15, 40, 15, 20, 0),
    "oscillator": (20, 45, 25, 15, -10),
    "osiris": (45, 15, 50, 5, 50),
    "osmosis": (35, -5, 25, 15, -15),
    "ostracon": (30, 15, 25, 5, 20),
    "otaku": (40, 55, 10, 20, 25),
    "othering": (-80, 55, 40, 35, 25),
    "ottoman": (55, -30, 25, 0, 20),
    "oud-agarwood": (90, 45, 70, 10, 80),
    "outcrop": (40, 10, 35, 5, 40),
    "outline": (15, 10, 20, 15, 5),
    "outsourcing": (-35, 25, 10, 30, 25),
    "ovary": (60, 35, 40, 15, 10),
    "overcranking": (50, 75, 45, 55, -15),
    "overfitting": (-75, 55, -25, 55, 20),
    "overloading": (-40, 65, 10, 60, 15),
    "overlock": (50, 55, 40, 50, 10),
    "overprint-type": (35, 30, 30, 25, 10),
    "oversteer": (-35, 75, -25, 65, -15),
    "overture": (70, 50, 40, 20, 10),
    "ownage": (85, 65, 60, 15, 30),
    "oxidation": (-30, 25, 15, 20, 10),
    "oxidation-state": (-20, 40, 35, 25, 10),
    "oxymoron": (20, 45, 10, 25, 0),
    "oxytocin": (75, -10, 35, 5, -35),
    "pacemaker": (60, 15, 45, 10, 20),
    "packet": (10, 35, 15, 55, -5),
    "packet-loss": (-95, 75, -55, 85, 25),
    "paddock": (50, 60, 45, 35, 25),
    "paella": (70, 35, 30, 20, 25),
    "pagan": (-95, 65, 20, 50, 15),
    "page-boy": (55, 40, -10, 45, 10),
    "pagoda": (75, 20, 50, 10, 55),
    "pahoehoe-lava": (65, 55, 45, 45, 60),
    "pale-heraldic": (40, 10, 40, 5, 30),
    "paleoclimatology": (65, 15, 50, 10, 60),
    "paleography": (65, 10, 50, 10, 75),
    "paleolithic": (30, 15, 40, 5, 50),
    "paleontology": (75, 15, 45, 5, 40),
    "palette": (55, 35, 30, 20, -10),
    "palimpsest": (55, 10, 40, 5, 65),
    "pallasite-olivine": (95, 45, 65, 15, 85),
    "palsy": (-60, 10, -50, 10, 35),
    "palynology": (55, 40, 40, 25, 20),
    "pancreas": (45, 15, 35, 15, 20),
    "pandemic": (-100, 95, -55, 100, 65),
    "panopticon": (-90, 60, 70, 55, 80),
    "pansexual": (60, 40, 35, 10, -20),
    "pantheism": (75, 25, 45, 5, -20),
    "panther": (55, 60, 45, 45, 20),
    "pantomime": (45, 35, 15, 10, -15),
    "pantone": (65, 25, 45, 15, 10),
    "paper-hands": (-80, 45, -50, 40, -10),
    "papyrology": (50, 10, 40, 5, 25),
    "papyrus-scroll": (75, 10, 45, 5, 35),
    "par-value": (20, 0, 35, 5, 20),
    "parable": (60, 15, 35, 5, 20),
    "parabola": (45, 30, 35, 10, -20),
    "paradigm": (35, 20, 40, 5, 15),
    "paradigm-shift": (65, 50, 55, 20, 45),
    "paradox": (-10, 50, -10, 35, 0),
    "paraffin": (20, -10, 20, 5, 10),
    "parallax": (30, 15, 35, 10, -30),
    "paralysis": (-95, -45, -60, 25, 50),
    "parameter": (20, 15, 45, 25, 20),
    "parapet": (50, 45, 50, 55, 55),
    "parasitic": (-85, 45, -40, 50, 20),
    "parasol": (50, -10, 25, 15, -35),
    "parchment": (45, 5, 30, 5, 15),
    "parity": (15, 10, 30, 25, 10),
    "parmesan": (65, 10, 35, 5, 15),
    "parody": (60, 50, 30, 15, -15),
    "parry": (75, 70, 55, 80, 15),
    "partisan": (-35, 60, 25, 50, 15),
    "pashmina-fiber": (100, -50, 55, 5, -50),
    "pass-interference": (-75, 80, -35, 90, 15),
    "passant": (40, 20, 35, 10, 25),
    "passerine": (60, 40, 15, 5, -40),
    "pastiche": (50, 40, 25, 10, -20),
    "pasture": (80, -30, 40, 0, 25),
    "pathology": (-15, 20, 15, 20, 20),
    "patina-sulfur": (-10, 15, 25, 5, 30),
    "patriarch": (60, 20, 50, 10, 40),
    "pause-boundary": (25, 10, 65, 25, -75),
    "pawn-structure": (50, 15, 50, 15, 45),
    "payload": (15, 40, 40, 45, 35),
    "payload-delivery": (-60, 75, 45, 85, 35),
    "payload-fairing": (55, 45, 45, 25, 60),
    "payroll": (35, 15, 40, 50, 15),
    "peak-era": (100, 85, 65, 45, -60),
    "peak-ring-struct": (85, 45, 70, 25, 90),
    "pearly": (75, -5, 30, 0, -10),
    "pedantic": (-55, 15, 35, 10, 25),
    "pediment": (75, 20, 45, 5, 50),
    "pelagic": (55, 30, 40, 10, 20),
    "pelagic-ocean": (65, 35, 55, 5, -80),
    "pelvis": (35, 15, 45, 10, 40),
    "penance": (-40, 20, -35, 45, 35),
    "pendant": (60, 25, 25, 15, -15),
    "peng": (85, 50, 35, 10, -20),
    "peninsula": (50, 10, 30, 5, 25),
    "pennant": (65, 55, 25, 15, -45),
    "penology": (15, 10, 45, 15, 50),
    "penrose-tiling": (85, 35, 55, 10, 45),
    "pension": (55, -20, 45, 5, 30),
    "pentaprism": (65, 30, 50, 15, 45),
    "peplum": (55, 35, 25, 10, -15),
    "pepperoni": (60, 35, 15, 20, 10),
    "peralkaline": (15, 25, 30, 10, 40),
    "perennial": (75, 5, 45, 0, 30),
    "perforation-gauge": (30, 15, 35, 25, 5),
    "perfumer-nose": (95, 60, 65, 45, 15),
    "peridot": (80, 40, 40, 5, 25),
    "perigee": (45, 35, 40, 25, 25),
    "periodt": (65, 50, 40, 15, -10),
    "periodt-pooh": (85, 75, 55, 60, -30),
    "peripatetic": (55, 40, 40, 10, 15),
    "periwinkle": (75, -20, 20, 0, -40),
    "perjury": (-85, 55, 15, 45, 25),
    "permaculture": (85, 20, 50, 10, 30),
    "permafrost": (-35, -20, 40, 5, 40),
    "permafrost-melt": (-90, 65, -50, 75, 50),
    "perovskite-structure": (90, 55, 65, 25, 45),
    "perpetual-calendar": (90, 30, 60, 5, 50),
    "perpetual-check": (10, 75, -10, 80, 5),
    "perpetual-mechanism": (90, 35, 70, 10, 65),
    "pessimistic": (-70, -20, -10, 5, 30),
    "pesticide": (-60, 30, -10, 35, 20),
    "petri-dish": (20, 35, 25, 40, -15),
    "petroglyph": (50, 15, 35, 5, 35),
    "petrology": (50, 10, 45, 5, 40),
    "ph-balance": (50, 15, 50, 35, 10),
    "phalanx": (50, 35, 55, 30, 55),
    "phantom": (-45, 50, -40, 30, -45),
    "phase-space-orbit": (40, 45, 55, 15, 55),
    "phased-array": (90, 75, 70, 60, 55),
    "phenocryst-crystal": (65, 25, 50, 10, 45),
    "phenomenology": (55, 35, 45, 10, -15),
    "phenotype": (25, 15, 30, 0, 10),
    "pheromones": (40, 50, 35, 25, -30),
    "philatelic-hinge": (25, 10, 20, 30, -10),
    "philatelist": (65, 15, 45, 10, 25),
    "philistine": (-75, 30, -30, 15, 30),
    "philological-audit": (40, 15, 55, 40, 50),
    "phishing-hook": (-95, 65, -35, 70, 20),
    "phloem": (35, 5, 45, 10, 40),
    "phoenix": (90, 60, 50, 15, -40),
    "phoneme": (35, 10, 30, 5, -20),
    "phonetics": (45, 15, 35, 10, -10),
    "phonology": (45, 10, 35, 5, 10),
    "photo-multiplier": (80, 85, 60, 70, 45),
    "photolithography": (60, 55, 50, 30, 40),
    "photon": (90, 60, 35, 10, -80),
    "photonic-crystal": (95, 65, 60, 20, -15),
    "photosynthesis": (80, 25, 50, 10, -20),
    "phototropism": (65, 40, 35, 10, -15),
    "photovoltaic-cell": (95, 25, 65, 20, 60),
    "phreatic-zone": (40, 10, 55, 20, 85),
    "phylloxera": (-95, 55, -20, 45, 30),
    "phytoalexin": (75, 55, 50, 60, -15),
    "phytoplankton": (80, 25, 30, 5, -90),
    "pianissimo": (60, -50, 15, 0, -20),
    "pica": (20, 5, 30, 15, 10),
    "pick-me": (-60, 40, -40, 10, 15),
    "pico-scale": (15, 15, -15, 10, -90),
    "pid-control-loop": (80, 55, 60, 65, 45),
    "pidgin": (30, 35, -10, 15, -5),
    "pierogi": (65, 10, 30, 15, 25),
    "piezoelectric": (65, 75, 45, 55, 10),
    "pillock": (-60, 35, -10, 20, 10),
    "pin-maneuver": (75, 65, 50, 60, 25),
    "pinch-hitter": (40, 75, 40, 85, 10),
    "pine": (50, 5, 35, 0, 15),
    "pinion": (10, 30, 25, 10, 15),
    "pinking": (30, 35, 25, 20, 5),
    "pinking-shears": (50, 45, 40, 35, 30),
    "pinniped": (50, 35, 35, 15, 30),
    "pinstripe": (50, 25, 45, 15, 25),
    "pioneer-species": (70, 55, 45, 25, -20),
    "pipeline": (25, 35, 35, 40, 15),
    "pipette-draw": (40, 30, 40, 55, -10),
    "piping": (50, 30, 35, 20, 10),
    "piss-ant": (-65, 30, -30, 10, 5),
    "piste": (20, 45, 30, 40, 55),
    "pistil": (50, 15, 20, 0, -25),
    "piston": (10, 55, 35, 40, 30),
    "piston-ring": (15, 45, 35, 30, 40),
    "pitch": (10, 45, 30, 45, 0),
    "pituitary": (35, 30, 55, 15, 15),
    "pivot": (25, 50, 30, 55, 5),
    "placebo-effect": (40, -15, 25, 10, -20),
    "placket": (35, 10, 30, 25, 10),
    "plaintext": (75, -20, 50, 10, 10),
    "plaintiff": (5, 35, 20, 25, 10),
    "planchet-blank": (25, 10, 35, 10, 55),
    "planck-epoch": (95, 95, 80, 5, -100),
    "planck-length": (50, 30, 45, 5, -100),
    "plane-tool": (70, 40, 45, 20, 30),
    "planetesimal-acc": (85, 55, 70, 25, 65),
    "plasma": (35, 55, 35, 40, -35),
    "plasma-sheet-dens": (35, 45, 60, 20, -80),
    "plasmaspheric-hiss": (55, -45, 55, 10, -90),
    "plasmid": (45, 35, 40, 20, -15),
    "plasmoid-ejection": (55, 95, 65, 80, -60),
    "plastic": (-10, 5, 15, 5, -15),
    "plate-tectonics-drift": (60, 35, 75, 5, 100),
    "plateau": (40, -15, 40, 0, 45),
    "platelet": (20, 15, 20, 35, 0),
    "platinum": (75, 15, 50, 5, 40),
    "platonic": (65, -15, 40, 5, -15),
    "platonism": (75, 20, 55, 5, -50),
    "pleating": (65, 30, 40, 20, 30),
    "plebeian": (-65, -15, -50, 5, 30),
    "pleochroism": (65, 45, 30, 10, -15),
    "plug": (75, 50, 55, 65, 25),
    "plunge": (-65, 70, -45, 65, 40),
    "pluralism": (65, 35, 40, 10, -15),
    "pneumatic": (15, 50, 35, 35, -20),
    "poignancy": (55, 20, -10, 5, 15),
    "point-in-line": (75, 45, 60, 50, 45),
    "point-size": (25, 10, 35, 20, 5),
    "poison": (-85, 55, 35, 55, 25),
    "poisson-distribution": (45, 40, 45, 30, 25),
    "polarimetric-dual": (75, 55, 65, 70, 10),
    "polaris-northern": (95, 20, 60, 5, -100),
    "polarization": (50, 25, 45, 15, -30),
    "pole-position": (95, 85, 60, 50, 45),
    "polenta": (55, -5, 30, 10, 20),
    "policy-enforcement": (45, 45, 60, 70, 60),
    "pollen": (-20, 55, 15, 60, -45),
    "pollen-basket": (50, 30, 15, 10, -5),
    "polyalphabetic": (30, 45, 55, 20, 50),
    "polyamory": (40, 45, 35, 15, -10),
    "polygamous": (20, 50, 20, 15, -10),
    "polyhedron": (55, 35, 45, 5, 40),
    "polymath": (85, 40, 50, 10, -10),
    "polymerase": (80, 60, 55, 50, -5),
    "polymerization": (35, 45, 40, 20, 30),
    "polymorphic": (35, 25, 35, 15, -5),
    "polynomial": (30, 40, 40, 25, 25),
    "polyphony": (65, 30, 35, 5, 10),
    "polysemy": (35, 20, 30, 5, 5),
    "polytheism": (50, 35, 40, 10, 35),
    "polyurethane": (45, 20, 40, 15, 25),
    "pommel": (20, 15, 45, 10, 55),
    "ponzi-scheme": (-95, 60, 35, 45, 40),
    "pookie-bear": (85, 20, 25, 5, -45),
    "popol-vuh-epic": (85, 35, 55, 5, 65),
    "porcelain": (65, -15, 35, 5, 10),
    "porous": (-10, 10, -20, 5, -15),
    "porphyritic-texture": (50, 15, 45, 10, 50),
    "portal": (75, 55, 50, 55, -10),
    "portrait": (60, 20, 45, 10, 30),
    "portside": (10, 5, 20, 5, 10),
    "positive-aura": (80, 25, 50, 0, -40),
    "post-quantum-crypto": (95, 65, 75, 45, 75),
    "post-structuralism": (35, 40, 40, 20, 10),
    "postmark-date": (45, 25, 35, 40, 15),
    "postulate": (25, 15, 40, 15, 25),
    "potaxie": (85, 80, 40, 35, -55),
    "potentiometer": (10, 15, 25, 10, 5),
    "pour-over-v60": (80, 30, 40, 55, -15),
    "poutine": (70, 50, 20, 35, 35),
    "pragmatic": (55, -5, 50, 15, 30),
    "pragmatics": (50, 15, 45, 15, 15),
    "prat": (-55, 35, -10, 15, 5),
    "praxis": (70, 60, 55, 50, 40),
    "precedent": (20, 10, 40, 5, 30),
    "precession-equinox": (70, 10, 65, 5, 85),
    "precipice": (-40, 70, -45, 70, 55),
    "precipitate": (15, 45, 30, 40, 25),
    "precipitation": (40, 25, 25, 35, 10),
    "prefix": (20, 10, 25, 5, -5),
    "prehistoric": (25, -10, 45, 0, 55),
    "prelude": (55, 35, 30, 40, -10),
    "premise": (20, 5, 35, 10, 15),
    "preservation-acidfree": (90, -10, 55, 10, 30),
    "preservative": (-20, 5, 15, 10, 10),
    "prick": (-70, 50, 15, 25, 10),
    "primates": (50, 45, 40, 15, 15),
    "prime": (80, 50, 50, 5, -35),
    "prime-factor": (55, 35, 45, 20, 40),
    "prime-number": (65, 45, 55, 10, 30),
    "primer": (50, 35, 45, 40, 15),
    "priority": (60, 70, 55, 85, 15),
    "private-key-secure": (95, 65, 75, 95, 85),
    "probability": (35, 40, 30, 45, -5),
    "probable-cause": (25, 45, 55, 60, 45),
    "probate": (-10, 10, 15, 20, 30),
    "probation": (-25, 25, -30, 20, 20),
    "proboscidean": (30, 5, 45, 5, 40),
    "proc-rate": (25, 45, 20, 35, 0),
    "procedural": (20, 15, 30, 10, 10),
    "procrastination": (-65, -20, -50, 45, 35),
    "procurement": (10, 20, 30, 35, 25),
    "prodigious": (75, 50, 50, 5, 35),
    "production": (30, 55, 40, 60, 20),
    "production-ready": (95, 45, 60, 55, 50),
    "productivity": (70, 35, 50, 15, -10),
    "profiler": (20, 35, 30, 40, 5),
    "prognosis": (5, 35, 10, 45, 10),
    "prograde": (60, 50, 45, 40, 15),
    "projection": (25, 35, 35, 20, -10),
    "projection-psych": (-65, 45, -30, 20, 15),
    "prokaryote": (25, 10, 35, 5, 25),
    "prologue": (50, 35, 35, 25, 10),
    "promotion-chess": (95, 80, 55, 65, -50),
    "proof-of-stake": (85, 40, 65, 25, 45),
    "proof-strike": (85, 45, 55, 20, 50),
    "propaganda": (-70, 45, 45, 35, 15),
    "propeller": (25, 50, 30, 40, -10),
    "prophecy": (45, 50, 35, 40, 15),
    "prophet": (75, 55, 50, 40, 20),
    "prophylaxis": (75, 15, 55, 15, 45),
    "propolis": (40, 5, 25, 5, 20),
    "proportional-gain": (65, 65, 55, 60, 15),
    "proprioception": (60, 35, 50, 10, 30),
    "prosciutto": (65, 15, 30, 10, 20),
    "prosecco": (70, 45, 25, 20, -35),
    "prostate": (15, 20, 25, 35, 20),
    "protagonist": (75, 50, 45, 35, -20),
    "protection-money": (-85, 55, 40, 50, 35),
    "proto-language": (50, 10, 55, 5, 70),
    "protocol": (30, 10, 55, 45, 40),
    "proton": (25, 40, 30, 10, 5),
    "protoplanetary-disk": (95, 65, 75, 15, -90),
    "provenance": (70, 15, 50, 10, 45),
    "provenance-lineage": (85, 25, 55, 15, 50),
    "providence": (90, -10, 55, 10, 45),
    "provisioning": (25, 35, 30, 45, 10),
    "proxy": (15, 15, 30, 20, 5),
    "psy-ops": (-75, 60, 50, 55, 20),
    "psychometrics": (40, 30, 45, 45, 25),
    "psychrometer-humid": (35, 25, 45, 40, 20),
    "pteridology": (65, 10, 45, 5, 25),
    "public-key-infra": (85, 20, 65, 50, 65),
    "pulsar": (55, 75, 45, 40, -40),
    "pulse-width-mod": (70, 60, 55, 65, 10),
    "pumice": (55, 20, 20, 5, -40),
    "pump-and-dump": (-95, 90, 35, 85, 20),
    "punter-hangtime": (30, 45, 25, 60, -80),
    "pupa": (15, -25, -30, 5, 15),
    "purfling-inlay": (85, 40, 45, 10, 20),
    "purgatory": (-65, 10, -40, 20, 50),
    "purr": (55, 20, 30, 5, -25),
    "putrid": (-100, 55, -25, 50, 25),
    "pwned": (80, 60, 50, 5, -35),
    "pylon": (40, 35, 55, 25, 70),
    "pyramid": (60, 15, 60, 0, 60),
    "pyranometer-solar": (65, 45, 55, 30, -25),
    "pyroclastic": (-85, 75, 45, 70, 40),
    "pyroclastic-surge": (-100, 100, 55, 100, 65),
    "quadrature-amp-mod": (75, 60, 60, 45, 15),
    "quantization-error": (-65, 55, 45, 50, 15),
    "quantum": (45, 55, 40, 20, -50),
    "quantum-cryptography": (95, 75, 75, 55, 75),
    "quantum-entanglement": (65, 55, 40, 20, -55),
    "quantum-tunnelling": (45, 65, 40, 20, -55),
    "quartz": (45, 10, 25, 0, 15),
    "quasar": (45, 80, 50, 35, -50),
    "quasar-emission": (45, 95, 65, 35, -55),
    "quaternion": (30, 55, 45, 20, -10),
    "quay": (40, 5, 30, 5, 30),
    "queen-bee": (50, 25, 50, 10, 20),
    "queer": (55, 40, 35, 10, -20),
    "quenching-oil": (-15, 85, 40, 90, 45),
    "quetzalcoatl-serpent": (85, 50, 65, 15, 55),
    "quick-scope": (75, 85, 45, 75, -25),
    "quill": (50, 10, 25, 10, -5),
    "quinoa": (45, -5, 30, 5, -5),
    "quisling": (-100, 65, -35, 55, 35),
    "quota": (-35, 55, -15, 60, 25),
    "rabbet": (45, 20, 40, 15, 20),
    "race-condition": (-80, 95, -45, 90, 10),
    "raclette": (75, 35, 30, 20, 25),
    "radar": (15, 45, 45, 55, 5),
    "radar-reflectivity": (45, 65, 55, 80, 30),
    "radiation": (-75, 45, -20, 50, 10),
    "radiation-belt-inj": (-85, 95, 50, 90, -85),
    "radiation-shield": (85, 10, 60, 15, 75),
    "radiative-forcing": (-55, 60, 45, 65, 35),
    "radicand": (20, 15, 30, 10, 15),
    "radio-frequency-id": (65, 35, 50, 55, 10),
    "radioactivity": (-95, 75, -30, 70, 50),
    "radiosonde": (20, 15, 20, 15, -30),
    "radiosonde-ascent": (70, 75, 55, 85, -95),
    "radius": (30, 10, 30, 5, 35),
    "ragged": (-25, 30, -20, 15, -10),
    "ragnarok": (-95, 75, -50, 65, 55),
    "ragnarok-fate": (-100, 95, -60, 100, 80),
    "rampant": (50, 65, 45, 30, 15),
    "rampart": (65, 50, 60, 55, 70),
    "rangeland": (65, -10, 45, 5, 35),
    "rank-and-file": (25, 15, 40, 10, 35),
    "rapport": (70, 20, 40, 10, -20),
    "rapscallion": (-45, 50, 20, 35, -5),
    "raptor": (-55, 70, 45, 65, 20),
    "rare-stacks": (80, 25, 55, 15, 75),
    "rasp": (-15, 45, 30, 20, 10),
    "raster": (15, 25, 25, 15, 10),
    "rate-limiting": (20, 55, 60, 70, 40),
    "rationalism": (65, -15, 55, 15, 40),
    "raw-file": (60, 35, 45, 45, 25),
    "reaction-control": (60, 55, 50, 65, -15),
    "reactive": (45, 45, 35, 35, -10),
    "reagent": (10, 45, 20, 50, -5),
    "reagent-grade": (55, 25, 45, 40, 20),
    "realism": (60, 10, 55, 10, 55),
    "reanimation": (40, 70, 45, 60, -50),
    "recidivism": (-85, 35, -45, 30, 55),
    "reclusive": (-15, -45, 30, 0, 30),
    "recoil": (-55, 60, -40, 60, 10),
    "recombination": (70, 55, 45, 20, -10),
    "reconciled": (70, 15, 40, 15, -10),
    "reconciliation": (80, 50, 45, 35, -25),
    "reconnaissance": (40, 60, 50, 65, 15),
    "reconnection-event": (75, 100, 75, 85, -50),
    "rectenna-efficiency": (75, 55, 60, 30, 45),
    "rectum": (-15, 20, 25, 50, 25),
    "recursion-depth": (30, 65, 40, 60, 45),
    "recursive": (25, 55, 35, 45, 10),
    "recursive-loop-fatal": (-100, 95, -65, 100, 45),
    "recursive-snark": (95, 75, 70, 55, 35),
    "red-giant": (20, 55, 50, 25, 60),
    "red-zone-ops": (20, 85, 55, 95, 40),
    "redaction": (-25, 20, 45, 35, 15),
    "redflag": (-95, 85, -20, 95, 35),
    "redoublement": (45, 80, 40, 85, 10),
    "redshift": (-15, 30, 35, 15, 5),
    "reduction": (35, 15, 20, 15, 5),
    "reduction-cell": (30, 35, 35, 20, 15),
    "redundancy": (35, -5, 45, 5, 30),
    "reed-solomon": (90, 35, 65, 45, 45),
    "reeded-edge": (35, 15, 35, 5, 15),
    "reef": (-35, 45, 10, 50, 30),
    "refactoring": (60, 25, 45, 30, 10),
    "reference-desk": (55, 15, 35, 45, 15),
    "referendum": (40, 55, 35, 60, 20),
    "refraction": (45, 25, 35, 10, -20),
    "refractive-index": (40, 25, 40, 15, 5),
    "refractometer": (35, 15, 40, 25, 10),
    "refugee": (-50, 55, -50, 60, 10),
    "regenerative-braking": (90, 70, 60, 45, 70),
    "regenerative-design": (100, 55, 75, 25, 40),
    "regent": (70, 35, 55, 45, 50),
    "regiment": (35, 55, 45, 45, 45),
    "regmaglypt-thumb": (40, 25, 45, 15, 35),
    "regularization": (70, 30, 60, 25, 45),
    "regulator": (60, 40, 55, 60, 25),
    "rehabilitation": (85, 25, 55, 20, 15),
    "rehearsal": (30, 45, 30, 50, 10),
    "reinforcement": (65, 50, 45, 40, -15),
    "rekt": (-95, 85, -55, 70, 45),
    "rekt-angles": (-25, 45, 15, 10, 5),
    "relative": (15, 15, -10, 15, 0),
    "relic": (55, 10, 40, 5, 45),
    "reliquary": (65, 20, 45, 10, 50),
    "remand": (-30, 40, 30, 45, 25),
    "remise": (40, 75, 35, 85, 5),
    "remittance": (50, 30, 35, 40, -10),
    "remorse": (-75, 35, -50, 30, 30),
    "renaissance": (90, 45, 55, 15, 40),
    "rentfree": (30, 40, 35, 5, -15),
    "reorg": (-50, 55, -30, 60, 25),
    "repeater-chime": (95, 50, 50, 20, 40),
    "repeater-complication": (95, 55, 55, 20, 45),
    "repoussé-tool": (60, 35, 45, 10, 25),
    "reprisal": (-85, 65, 35, 60, 30),
    "republic": (60, 25, 45, 15, 30),
    "resentment": (-85, 55, -20, 30, 35),
    "reservoir-host": (-30, 20, 45, 15, 40),
    "residual-sugar": (55, 15, 20, 5, 10),
    "resilience": (85, 45, 55, 15, 20),
    "resilience-psych": (95, 45, 60, 20, 15),
    "resin": (30, 5, 25, 5, 20),
    "resinous": (45, 15, 25, 5, 25),
    "resistor": (15, 10, 30, 10, 10),
    "resolution": (55, 35, 45, 40, 30),
    "resonance": (85, 65, 45, 25, -25),
    "resonant-inductive": (85, 60, 60, 35, 25),
    "respiratory": (35, 40, 35, 60, -10),
    "resplendent": (90, 40, 45, 5, -45),
    "response": (20, 20, 20, 35, 0),
    "resurrection": (100, 75, 60, 50, -60),
    "retaliation": (-85, 70, 40, 65, 25),
    "retention": (55, 15, 40, 15, 10),
    "retention-rate": (85, 20, 60, 40, 55),
    "retina": (65, 35, 45, 5, -5),
    "retirement": (70, -40, 50, 0, -45),
    "retort-stand": (35, 5, 45, 10, 45),
    "retrograde": (-40, 55, 35, 50, 25),
    "retrovirus": (-75, 50, 45, 60, -35),
    "rev-limiter": (-20, 90, 30, 95, 15),
    "reverb": (50, 10, 20, 5, -20),
    "reverse-tails": (40, 10, 40, 5, 40),
    "revulsion": (-100, 65, -30, 55, 20),
    "reynolds-number": (35, 45, 55, 25, 50),
    "rhapsody": (80, 55, 35, 15, -25),
    "rheology": (30, 20, 40, 10, 45),
    "rheostat": (5, 25, 30, 20, 15),
    "rhizomatic": (65, 45, 35, 15, -15),
    "rhizome": (45, -10, 40, 5, 50),
    "rhodochrosite": (75, 35, 35, 10, 35),
    "ribosome": (55, 45, 40, 10, 5),
    "rick-roll": (70, 65, 35, 20, -60),
    "ricotta": (60, -15, 25, 5, -10),
    "ridge-detail": (20, 15, 35, 25, 5),
    "rigging": (15, 30, 35, 15, 25),
    "right-ascension": (40, 15, 50, 10, -50),
    "right-of-way": (40, 65, 65, 70, 50),
    "rigor-mortis": (-95, -10, 10, 0, 50),
    "rime-ice": (30, 35, 20, 30, 20),
    "ring-current-decay": (45, -25, 60, 30, -75),
    "riposte": (85, 75, 60, 80, -10),
    "risotto": (70, 15, 35, 15, 25),
    "rivet": (15, 35, 35, 20, 35),
    "rizz": (65, 40, 45, 5, -20),
    "rizzler": (85, 60, 55, 30, -25),
    "rng-luck": (10, 65, -30, 40, -15),
    "roach": (-90, 40, -50, 40, 15),
    "roadmap": (40, 15, 35, 10, 5),
    "roasting": (20, 75, 35, 45, -10),
    "robusta-lowland": (35, 45, 50, 10, 35),
    "rodent": (-70, 35, -40, 30, 10),
    "roll": (10, 50, 30, 45, -10),
    "rollback": (-40, 55, -20, 60, 15),
    "rollout": (45, 50, 35, 55, 5),
    "rosin-friction": (50, 55, 35, 30, 10),
    "router-bit": (25, 60, 45, 45, 15),
    "rov-pilot": (60, 55, 50, 40, 40),
    "row-oriented": (45, 15, 55, 20, 65),
    "royal-jelly": (70, 20, 35, 10, -10),
    "rubber-ducking": (65, -10, 35, 10, -20),
    "rudder": (20, 45, 40, 50, 10),
    "rugged": (-100, 85, -60, 80, 50),
    "ruins": (-45, -5, -30, 0, 40),
    "runway": (35, 50, 35, 60, 20),
    "sabotage": (-85, 60, 30, 65, 20),
    "sabre": (60, 85, 55, 65, 25),
    "sack-loss": (-80, 85, -50, 85, 40),
    "sacrament": (80, 25, 45, 20, 50),
    "sacrum": (25, 10, 30, 5, 40),
    "safelight-red": (65, -30, 45, 40, 10),
    "safety-score": (65, 70, 45, 75, 35),
    "saffron": (70, 20, 35, 5, -20),
    "sage": (80, -20, 50, 0, 15),
    "saintly": (95, -10, 45, 5, -50),
    "salami": (55, 10, 25, 10, 25),
    "salinity": (15, 15, 30, 10, 45),
    "salinization": (-75, 20, -40, 30, 45),
    "salt-value": (60, 25, 45, 35, 15),
    "salting": (25, 15, 35, 30, 5),
    "saltire": (55, 30, 40, 10, 20),
    "saltire-cross": (60, 35, 45, 10, 40),
    "salute": (95, 15, 50, 30, 25),
    "salvo": (-70, 70, 45, 65, 40),
    "samsara": (-10, 40, 0, 20, 30),
    "sanction": (-55, 30, 45, 40, 35),
    "sanctioned": (20, 35, 55, 50, 40),
    "sanctum": (85, -35, 50, 5, 40),
    "sandpaper": (-10, 50, 25, 20, 5),
    "sans-serif": (45, 10, 40, 5, -10),
    "saprophytic": (20, -15, 30, 5, 25),
    "sarcophagus": (-65, 20, 45, 15, 75),
    "sateen": (70, -20, 35, 0, -25),
    "satin": (80, -25, 30, 0, -30),
    "satire": (50, 45, 35, 20, 10),
    "sauerkraut": (45, 20, 20, 10, 15),
    "savanna": (65, 10, 35, 5, 10),
    "scaffolding": (20, 55, 35, 60, 40),
    "scalability": (55, 30, 45, 15, 10),
    "scalar": (15, -5, 20, 5, 10),
    "scapula": (35, 15, 35, 5, 35),
    "scattered-disk": (45, 20, 65, 15, 85),
    "scavenger": (-65, 35, -20, 40, 15),
    "schadenfreude": (10, 50, 30, 15, 20),
    "schema": (30, 10, 40, 10, 25),
    "schnitzel": (65, 30, 25, 20, 25),
    "scholasticism": (45, -10, 50, 20, 60),
    "schreibersite-phos": (25, 15, 50, 25, 85),
    "schrödinger-equation": (65, 45, 70, 15, 45),
    "sciatica": (-50, 30, -40, 25, 25),
    "scintillating": (80, 65, 40, 10, -35),
    "sclerotium": (25, -10, 35, 10, 45),
    "scoliosis": (-40, 10, -30, 5, 30),
    "sconce": (45, 10, 20, 5, -10),
    "scorpio": (-70, 60, 35, 55, 10),
    "scp-foundation": (65, 75, 55, 40, 80),
    "screenplay": (40, 35, 40, 30, 20),
    "script-kiddie": (-65, 45, -35, 10, 5),
    "script-kiddy": (-70, 40, -45, 15, 10),
    "scroll-carving": (75, 45, 50, 15, 50),
    "scrote": (-80, 40, 5, 15, 15),
    "scrub": (-75, 25, -45, 5, 20),
    "scrum": (25, 50, 25, 55, 5),
    "scrum-master": (55, 55, 65, 85, 40),
    "sculpture": (70, 15, 45, 5, 50),
    "scupper": (-10, 20, 15, 30, 10),
    "scuttle": (-70, 65, 40, 70, 55),
    "sealing-wax": (50, 25, 30, 20, 15),
    "second-crack": (50, 75, 45, 55, 5),
    "secondary-crater": (-35, 65, 45, 55, 50),
    "secular": (10, -15, 35, 0, 15),
    "secure-base": (95, -30, 60, 10, 35),
    "sedimentary": (45, 5, 40, 5, 45),
    "seed-phrase": (40, 65, 55, 95, 30),
    "seedling": (80, 50, 15, 25, -40),
    "seismology": (35, 55, 45, 60, 35),
    "seizure": (-80, 60, -50, 60, 30),
    "sejant": (40, -15, 35, 5, 30),
    "self-actualization": (100, 45, 60, 5, -70),
    "selvedge": (65, 10, 40, 5, 25),
    "selvedge-denim": (95, 20, 55, 5, 65),
    "semantics": (55, 15, 45, 10, 15),
    "semaphore": (35, 30, 45, 55, 35),
    "semaphore-sig": (35, 50, 40, 65, -10),
    "semiconductor": (50, 45, 50, 15, 5),
    "semiotics": (55, 25, 45, 10, 20),
    "senescence": (-30, -25, 20, 5, 35),
    "sensor-noise": (-65, 55, -40, 50, 15),
    "sensory": (55, 50, 30, 15, -25),
    "sepia": (45, -25, 20, 0, 20),
    "sepia-toning": (75, -20, 40, 5, 45),
    "sepulcher": (-45, -10, 30, 5, 60),
    "sequencing": (85, 45, 60, 30, 40),
    "sequential-gearbox": (65, 65, 55, 50, 40),
    "sequins": (75, 75, 20, 25, -25),
    "sequoia": (95, 20, 60, 5, 70),
    "serialization": (20, 15, 25, 30, 5),
    "serif": (55, -5, 45, 5, 15),
    "serotonin": (75, -20, 45, 5, -30),
    "serverless": (45, 30, 40, 25, -20),
    "serving-cunt": (90, 65, 50, 10, -30),
    "set-theory": (25, 10, 45, 5, 20),
    "settlement": (55, -20, 40, 15, 10),
    "sextant": (50, 20, 40, 5, 20),
    "shadowbanned": (-70, 20, -50, 30, 20),
    "shaman": (60, 50, 40, 30, -15),
    "shannon-entropy": (55, 35, 60, 15, 45),
    "shannon-hartley": (70, 35, 60, 25, 45),
    "sharding": (20, 25, 35, 20, 15),
    "sharding-db": (65, 35, 55, 20, 35),
    "sharding-strategy": (55, 45, 55, 35, 50),
    "shark": (-65, 65, 50, 65, 35),
    "shatter-cone-shock": (-85, 95, 75, 90, 85),
    "shawarma": (75, 40, 30, 30, 20),
    "shellac": (65, 20, 40, 10, 15),
    "sherd": (20, 25, 20, 15, 15),
    "shibori-dye": (80, 30, 35, 15, -20),
    "shilling": (-70, 55, 20, 45, 10),
    "shimmer": (75, 40, 20, 10, -40),
    "shipping-war": (-55, 85, -25, 65, 15),
    "shit-brick": (-80, 45, 10, 25, 30),
    "shit-for-brains": (-90, 55, 10, 30, 20),
    "shitcoin": (-85, 60, -20, 40, -15),
    "shitepoke": (-50, 25, 5, 15, 10),
    "shitmuncher": (-85, 50, 15, 25, 15),
    "shitshow": (-80, 70, -15, 60, 15),
    "shiva-destroyer": (25, 65, 75, 40, 60),
    "shoal": (-20, 30, 0, 45, 20),
    "short-squeeze": (80, 95, 65, 90, -45),
    "shorting": (-30, 60, 30, 50, 5),
    "shortwave-radio": (45, 55, 50, 60, 40),
    "shrine": (80, -20, 45, 5, 30),
    "shutter-angle": (25, 45, 45, 40, 10),
    "shutter-lag": (-70, 55, -45, 60, 15),
    "shutter-speed": (35, 60, 45, 85, -25),
    "shuttle-loom": (45, 50, 40, 35, 45),
    "sidecar-proxy": (45, 35, 50, 30, 25),
    "siderite-iron": (35, 5, 55, 5, 95),
    "siege": (-80, 55, 45, 50, 50),
    "sigil": (25, 40, 35, 10, 5),
    "sigma": (45, 30, 50, 5, 10),
    "sigma-grindset": (50, 60, 50, 35, 40),
    "signal-flags": (45, 45, 35, 70, 15),
    "signal-to-noise": (90, 60, 65, 40, 10),
    "signified": (35, 20, 35, 10, 5),
    "signifier": (30, 20, 35, 10, 5),
    "silage": (20, -5, 25, 10, 40),
    "silhouette": (40, 15, 25, 10, 10),
    "silk": (85, -30, 35, 0, -35),
    "silky": (85, -25, 35, 0, -35),
    "sillage": (65, 45, 50, 10, -40),
    "silly-point": (-10, 75, -20, 80, 10),
    "silver": (70, 20, 40, 5, 20),
    "silver-halide": (35, 10, 35, 5, 15),
    "simile": (45, 25, 30, 5, -10),
    "simp-behavior": (-55, 35, -45, 15, 10),
    "simping": (-55, 45, -45, 20, 10),
    "simpleton": (-80, -25, -55, 10, 20),
    "simulacrum": (-15, 55, 30, 15, -40),
    "simulator": (30, 30, 30, 15, 5),
    "sine-wave": (65, 40, 35, 10, -35),
    "singularity": (-10, 60, 55, 15, 65),
    "sinister": (-45, 35, 30, 25, 15),
    "siren": (-35, 65, 20, 65, -30),
    "skeletal": (20, 10, 40, 5, 45),
    "skepticism": (20, -10, 50, 5, 30),
    "sketchbook": (60, 25, 30, 10, -5),
    "skewer": (80, 70, 50, 65, 15),
    "skibidi-toilet": (-25, 60, -20, 20, -15),
    "skirmish": (-35, 65, 15, 60, 10),
    "skyscraper": (50, 40, 55, 5, 50),
    "slaps": (80, 65, 40, 10, -15),
    "slaughter": (-90, 70, 45, 65, 40),
    "slay-the-house-down-boots": (90, 95, 55, 60, -70),
    "sleeper-cell": (-80, 45, 45, 55, 25),
    "sleipnir-steed": (85, 65, 55, 25, 45),
    "slenderman": (-90, 85, 25, 60, 45),
    "slick-tires": (70, 75, 55, 40, 25),
    "slider-tilt": (25, 70, 35, 55, 15),
    "slimy": (-65, 35, -25, 10, 10),
    "slipstream": (45, 50, 35, 30, -35),
    "slush": (-45, 10, -20, 5, 25),
    "small-caps": (50, 15, 40, 10, 10),
    "smart-contract": (35, 35, 40, 25, 15),
    "smart-grid-node": (90, 55, 65, 40, 50),
    "smartcontract": (35, 35, 40, 25, 15),
    "smocking": (70, 25, 40, 10, 20),
    "smoker-tool": (20, 35, 30, 45, 25),
    "sneak-link": (40, 55, -20, 45, -15),
    "social-loafing": (-50, -20, -35, 10, 25),
    "sociolinguistics": (70, 35, 50, 20, 25),
    "soft-launch": (65, 45, 30, 40, -35),
    "software-defined-radio": (85, 65, 65, 45, 20),
    "solar-wind": (40, 85, 45, 65, -100),
    "solarization": (65, 65, 30, 15, -50),
    "soldering": (20, 40, 30, 35, 10),
    "soldering-iron": (35, 55, 40, 60, 15),
    "solenoid": (15, 45, 25, 40, 5),
    "solfatara-gas": (-45, 55, 35, 50, 20),
    "solicitation": (-10, 30, 15, 40, 5),
    "solid-booster": (45, 85, 55, 75, 75),
    "solid-state-battery": (100, 45, 75, 35, 85),
    "solidarity": (85, 35, 50, 20, 10),
    "soliloquy": (60, -10, 40, 5, 25),
    "solipsism": (-75, -10, 30, 5, -45),
    "solitude": (40, -35, 30, 0, 20),
    "solstice": (60, 25, 35, 15, -10),
    "solulu": (45, 25, 35, 10, -20),
    "solute": (25, 15, 30, 15, 15),
    "solvent": (35, 20, 40, 15, 5),
    "somatic-symptom": (-55, 45, -45, 45, 35),
    "somatization": (-45, 35, -45, 20, 30),
    "sommelier": (55, 20, 45, 15, 15),
    "sonata": (60, 25, 40, 5, 20),
    "soprano": (65, 45, 30, 5, -50),
    "sorbet": (75, 25, 20, 10, -45),
    "soufflé": (75, 40, 25, 15, -55),
    "soulstirring": (85, 55, 35, 10, -35),
    "sound-post": (80, 50, 45, 30, 35),
    "soundboard-spruce": (90, 35, 55, 10, 45),
    "sourcing": (25, 30, 35, 35, 15),
    "sous-vide": (55, 15, 35, 10, 5),
    "southpaw": (35, 30, 35, 15, 10),
    "souvlaki": (65, 30, 25, 25, 10),
    "sovereignty": (65, 20, 50, 10, 50),
    "space-debris": (-85, 60, -45, 70, 40),
    "spacetime": (50, 10, 60, 5, 0),
    "spaghetti-code": (-75, 45, -35, 35, 15),
    "spallation-nuclei": (35, 75, 55, 45, -80),
    "spartan": (55, -30, 45, 5, 35),
    "special-teams": (40, 65, 40, 70, 15),
    "speciation-event": (75, 60, 55, 15, 25),
    "specific-gravity": (20, 5, 50, 10, 75),
    "specific-impulse": (65, 45, 45, 15, 10),
    "specimen-print": (60, 25, 40, 15, 25),
    "spectacle": (80, 70, 45, 20, -40),
    "spectacles": (45, 10, 35, 15, 10),
    "specter": (-55, 60, -40, 35, -40),
    "spectroheliograph": (85, 40, 65, 20, -45),
    "spectrophotometry": (55, 30, 45, 25, 20),
    "spectroscopy": (45, 40, 45, 25, 10),
    "spectrum": (75, 40, 40, 15, -45),
    "speculation": (0, 50, -10, 40, 5),
    "speedrun": (60, 95, 45, 95, -45),
    "speleothem-cave": (75, 15, 50, 5, 80),
    "spellbinding": (90, 60, 35, 15, -45),
    "spic": (-100, 70, -25, 60, 35),
    "spin-the-block": (-60, 95, 50, 95, 15),
    "spinel": (70, 45, 45, 5, 40),
    "spinning-jenny": (45, 55, 40, 35, 70),
    "spintronics": (55, 45, 45, 15, -25),
    "spire": (95, 55, 50, 25, -50),
    "spleen": (35, 10, 30, 15, 25),
    "spoiler": (35, 40, 35, 15, 20),
    "spokeshave": (60, 35, 40, 20, 25),
    "spore": (15, 45, 15, 50, -40),
    "spotless": (80, -10, 40, 0, -30),
    "sprocket": (15, 40, 30, 15, 20),
    "spruce": (45, -5, 35, 0, 25),
    "squid": (20, 35, 20, 30, -20),
    "squire": (60, 50, 15, 35, 15),
    "sriracha": (50, 45, 15, 15, -15),
    "st-elmos-fire": (30, 70, 15, 60, -45),
    "stable-coin": (80, -45, 55, 10, 25),
    "stablecoin": (45, 10, 40, 20, -10),
    "stablecoin-peg": (65, 25, 55, 30, 40),
    "staccato": (15, 50, 25, 20, -15),
    "stack-overflow": (-95, 85, -55, 95, 35),
    "stage-separation": (75, 90, 50, 95, 20),
    "stagflation": (-75, 50, -50, 55, 45),
    "staging": (20, 40, 25, 50, 5),
    "stagnant": (-60, -50, -45, 10, 40),
    "stagnation": (-60, -20, -45, 10, 35),
    "stainless": (80, -5, 45, 0, -25),
    "stakeholder-trust": (95, 35, 65, 10, 35),
    "staking": (55, -15, 45, 5, 30),
    "stalactite-ceiling": (60, 10, 40, 5, 85),
    "stalagmite-floor": (60, 10, 40, 5, 90),
    "stalemate": (-20, -50, 0, 40, 35),
    "stall": (-75, 60, -50, 60, 40),
    "stalwart": (75, 20, 50, 10, 40),
    "stamen": (50, 15, 20, 0, -25),
    "stan-culture": (-30, 95, 15, 80, 20),
    "stan-twitter": (-35, 95, 10, 85, 5),
    "standard-royal": (95, 35, 70, 15, 90),
    "starboard": (15, 5, 20, 5, 10),
    "stash": (65, 45, 50, 55, 50),
    "statant": (35, -5, 40, 5, 40),
    "state-of-charge": (65, 35, 45, 75, 15),
    "state-of-health": (85, 25, 60, 55, 40),
    "static": (-20, 40, 10, 45, 10),
    "stationery": (55, 5, 25, 10, -10),
    "statuary": (80, 25, 55, 10, 65),
    "stature": (45, 15, 45, 5, 45),
    "statute": (10, 0, 50, 10, 40),
    "steadicam": (90, 65, 55, 35, -30),
    "steel": (50, 10, 55, 5, 45),
    "steganography": (45, 55, 55, 40, 15),
    "stem-cell": (75, 40, 55, 15, -10),
    "stemware": (50, 15, 25, 15, -10),
    "stench": (-95, 60, -15, 55, 15),
    "stenographer": (15, 15, 15, 45, 5),
    "stepper": (45, 85, 55, 90, 45),
    "stepper-motor-resolution": (85, 50, 65, 40, 75),
    "sterile": (-35, -20, 20, 10, 15),
    "sticky-wicket": (-55, 55, -40, 60, 25),
    "still-life": (50, -40, 35, 0, 20),
    "stinger": (-60, 70, 30, 80, 5),
    "stishovite-dense": (25, 15, 65, 10, 100),
    "stochastic": (5, 45, 10, 35, 5),
    "stochastic-process": (15, 50, 15, 35, 5),
    "stoicism": (80, -50, 60, 10, 50),
    "stokes-law": (25, 5, 50, 10, 85),
    "stomata": (50, 10, 40, 45, -20),
    "stonewalling": (-75, -10, 30, 35, 45),
    "stoneware": (40, -5, 25, 0, 30),
    "stop-bath": (30, 30, 45, 60, 15),
    "stowaway": (-45, 55, -45, 40, 10),
    "stradivarius": (100, 50, 75, 15, 70),
    "strafe": (-80, 70, 45, 75, 15),
    "strange-matter": (-30, 65, 60, 15, 95),
    "strategic-vision": (95, 65, 70, 20, 45),
    "stratigraphy": (55, 10, 50, 10, 50),
    "stratopause": (35, -5, 40, 5, -70),
    "stratosphere": (40, -15, 45, 0, -60),
    "strike-zone": (10, 55, 60, 95, 20),
    "stroboscopic": (40, 85, 35, 75, -20),
    "structuralism": (25, 10, 50, 15, 55),
    "strut-bar": (35, 15, 40, 5, 55),
    "stucco": (40, 5, 30, 10, 25),
    "stygofauna-water": (55, 20, 50, 10, 55),
    "subaltern": (-40, 25, -50, 20, 30),
    "subduction": (-70, 75, 45, 75, 80),
    "subduction-trench": (-85, 85, 60, 75, 100),
    "subjective": (10, 25, -20, 15, -15),
    "sublimation": (65, 45, 40, 20, -50),
    "sublimation-psych": (70, 35, 45, 10, -25),
    "submersible": (70, 45, 55, 25, 75),
    "subpoena": (-45, 50, -40, 60, 25),
    "subset": (15, 5, 25, 5, 5),
    "subsidy": (50, 5, 30, 15, -10),
    "subterfuge": (-85, 55, 35, 45, 20),
    "subtext": (25, 45, 20, 35, 10),
    "subversion": (-75, 50, -30, 40, 30),
    "succulent": (70, -10, 35, 5, 25),
    "sucrose": (20, 20, 10, 5, 0),
    "suede": (65, -10, 30, 0, 15),
    "suevite-impact": (45, 45, 55, 30, 80),
    "suffix": (20, 10, 25, 5, 5),
    "suffrage": (70, 55, 45, 50, -20),
    "sulfites": (-15, 10, 15, 10, 5),
    "sulfur": (-40, 30, 10, 20, 15),
    "summit": (85, 55, 55, 25, 60),
    "super-box": (40, 10, 40, 5, 45),
    "supercell": (-75, 85, 50, 80, 40),
    "supercharger": (55, 70, 50, 35, 35),
    "superego": (40, 10, 55, 15, 45),
    "superfluidity": (85, 35, 55, 10, -60),
    "supermassive-black-hole": (-50, 85, 75, 20, 100),
    "supernova": (30, 95, 55, 60, -20),
    "superposition": (35, 45, 10, 15, -45),
    "superset": (45, 15, 40, 10, 30),
    "supersonic": (45, 60, 45, 25, -40),
    "superstructure": (60, 45, 55, 30, 65),
    "surge": (65, 65, 45, 55, 10),
    "surjection": (35, 20, 40, 15, 15),
    "surplus": (65, 20, 40, 15, -15),
    "surrealism": (65, 55, 25, 10, -50),
    "surveillance": (-20, 50, 45, 55, 20),
    "suspension": (60, 50, 50, 25, -30),
    "sustainability-audit": (45, 25, 55, 60, 45),
    "sustainable": (65, 5, 40, 10, -5),
    "suture": (20, 15, 20, 30, 5),
    "swamp": (-35, 10, -20, 5, 20),
    "swarming": (15, 75, -30, 65, -35),
    "sweaty-player": (-35, 85, 40, 75, 15),
    "swedish-rhapsody": (-75, 85, 50, 75, 65),
    "sycophant": (-90, 45, -50, 25, 15),
    "syllogism": (30, 20, 40, 5, 25),
    "symbiotic": (85, 25, 50, 15, -10),
    "sympathy": (75, 25, 30, 20, -30),
    "symphony": (85, 40, 50, 10, 50),
    "synapse": (40, 60, 35, 45, -40),
    "synaptic": (55, 60, 30, 45, -45),
    "syncline": (30, 5, 35, 5, 50),
    "syncopation": (55, 50, 25, 10, -25),
    "synecdoche": (40, 30, 30, 5, -5),
    "synergy": (40, 35, 30, 10, -20),
    "synergy-effect": (85, 55, 55, 15, 10),
    "synesthesia": (75, 65, 35, 10, -50),
    "syntax": (45, 10, 45, 15, 35),
    "syntax-error": (-65, 50, -30, 55, 10),
    "syntax-formal": (30, 5, 55, 20, 50),
    "syntax-sugar": (75, 35, 40, 5, -40),
    "syntax-tree": (40, 15, 35, 10, 20),
    "synthwave": (80, 45, 45, 15, -55),
    "tabard": (55, 25, 30, 10, 20),
    "table-saw": (35, 65, 55, 65, 70),
    "table-this": (-50, -35, 40, 20, 50),
    "tableau": (55, -20, 40, 5, 40),
    "tachometer": (40, 65, 45, 80, 10),
    "taciturn": (-10, -35, 35, 0, 25),
    "tactic": (50, 55, 55, 50, 10),
    "tactile": (60, 45, 35, 10, -15),
    "taenite-nickel": (55, 15, 55, 5, 95),
    "taffeta": (65, 35, 35, 10, 20),
    "taiga-biome": (55, 15, 45, 5, 55),
    "tailpiece-gut": (40, 20, 45, 20, 25),
    "tailwind": (70, 25, 40, 20, -30),
    "talisman": (65, 25, 40, 10, 10),
    "talking-stage": (35, 55, -20, 50, 5),
    "tang": (25, 10, 50, 15, 80),
    "tangent-line": (40, 35, 35, 25, -10),
    "tanking": (40, 30, 50, 45, 60),
    "tannin": (20, 35, 25, 10, 15),
    "tanzanite": (85, 55, 45, 15, 45),
    "tapas": (65, 30, 20, 20, -5),
    "tapestry": (50, 5, 25, 0, 15),
    "tapestry-loom": (90, 25, 65, 10, 85),
    "tardigrade": (45, -20, 50, 0, -30),
    "tartan-clan": (85, 35, 65, 10, 55),
    "taupe": (25, -20, 15, 0, 10),
    "tautology": (-30, -10, 10, 5, 10),
    "tax-evasion": (-90, 60, 25, 50, 35),
    "tea-spilled": (80, 85, 40, 95, -20),
    "technical-debt": (-85, 35, -45, 50, 75),
    "technocrane": (80, 60, 60, 40, 75),
    "tectonophysics": (40, 50, 50, 30, 55),
    "telemetry": (35, 25, 40, 30, 10),
    "telemetry-link": (45, 35, 45, 55, 10),
    "teleology": (55, 15, 50, 10, 55),
    "telephoto": (70, 45, 45, 20, 50),
    "telomere": (40, 10, 45, 5, 5),
    "tempering-straw": (65, 40, 50, 30, 25),
    "tempo": (60, 70, 50, 80, -40),
    "tenacity": (80, 60, 55, 20, 25),
    "tenderness": (95, -45, 30, 5, -40),
    "tendon": (15, 5, 25, 5, 20),
    "tenon": (65, 20, 50, 10, 50),
    "tensor": (15, 50, 50, 20, 45),
    "tephra": (-55, 50, 30, 45, 25),
    "tephra-fallout": (-65, 65, 35, 75, 45),
    "tequila": (40, 55, -10, 20, -25),
    "terminally-online": (-95, 45, -50, 40, 35),
    "termination-shock": (65, 95, 85, 50, -95),
    "terracotta": (55, 10, 35, 0, 40),
    "territory": (15, 30, 40, 25, 40),
    "terroir": (75, 15, 45, 5, 40),
    "terroir-coffee": (80, 20, 50, 5, 40),
    "testicle": (40, 45, 25, 25, 5),
    "testimony": (20, 35, 15, 25, 10),
    "testnet": (20, 25, 25, 30, 5),
    "testosterone": (30, 50, 35, 15, 5),
    "tetrahedron": (50, 40, 45, 5, 35),
    "tezcatlipoca-mirror": (-40, 75, 65, 55, 45),
    "thalamus": (40, 25, 45, 10, 5),
    "the-buzzer-uvb76": (-95, 95, 65, 90, 85),
    "the-lincolnshire-poacher": (-55, 75, 45, 70, 55),
    "theology": (65, 25, 55, 10, 60),
    "theology-dogma": (35, 15, 70, 10, 85),
    "theorem": (55, 30, 50, 10, 45),
    "thermal-runaway": (-100, 95, -70, 100, 65),
    "thermocline": (20, 25, 30, 20, 45),
    "thermodynamic-limit": (25, 15, 60, 5, 95),
    "thermodynamics": (35, 45, 50, 35, 40),
    "thermohaline": (55, 25, 45, 20, 65),
    "thermometer": (10, 15, 20, 35, 5),
    "thermosphere": (45, 55, 40, 20, -80),
    "thermospheric-drag": (-45, 25, 45, 15, -90),
    "thermostat": (40, 10, 45, 30, 15),
    "thesaurus": (75, 20, 45, 10, 25),
    "thigmotropism": (50, 55, 40, 25, -5),
    "thimble": (55, 10, 45, 25, 20),
    "thirst-trap": (30, 55, -10, 20, -25),
    "thistle": (-20, 30, 5, 15, 0),
    "thought-leader": (65, 45, 55, 25, 40),
    "threshing": (25, 50, 35, 30, 40),
    "threshold": (25, 45, 30, 55, 15),
    "thrombosis": (-75, 50, -45, 55, 30),
    "throttle": (30, 55, 50, 60, 0),
    "throughput": (40, 20, 35, 10, 5),
    "throughput-ops": (85, 55, 60, 40, 20),
    "thrust": (55, 60, 50, 55, -25),
    "thruster-exhaust": (40, 65, 40, 50, -25),
    "tibia": (35, 10, 40, 5, 40),
    "tilling": (10, 35, 30, 20, 45),
    "timbre": (40, 15, 25, 0, 0),
    "timepiece": (60, 10, 45, 10, 15),
    "tincture": (50, 20, 35, 5, 15),
    "tinnitus": (-45, 35, -40, 20, 10),
    "tintype": (75, 20, 45, 5, 80),
    "titanic": (70, 45, 50, 5, 50),
    "titanium": (75, 20, 55, 10, 30),
    "titration": (25, 50, 35, 55, 10),
    "titre": (40, 25, 45, 35, 15),
    "titrimetry": (35, 40, 40, 55, 10),
    "tlaloc-rain": (40, 65, 55, 75, 40),
    "to-the-moon": (95, 95, 45, 40, -80),
    "toady": (-85, 35, -55, 25, 20),
    "toe-in": (5, 20, 25, 15, 5),
    "toile": (45, 20, 35, 25, 15),
    "token": (25, 30, 20, 15, -5),
    "tomb": (-70, -10, -30, 0, 50),
    "tongs": (25, 45, 40, 55, 20),
    "tongs-mandrel": (45, 30, 50, 40, 55),
    "tongue": (45, 40, 20, 40, -5),
    "top-note": (75, 55, 35, 40, -50),
    "topaz": (80, 40, 45, 5, 35),
    "topography": (20, 5, 35, 5, 30),
    "topology": (55, 35, 45, 10, -10),
    "topology-manifold": (65, 35, 55, 15, 65),
    "topsoil": (70, -5, 45, 10, 50),
    "torque": (20, 55, 45, 40, 25),
    "torque-constant": (65, 35, 60, 20, 85),
    "torque-converter": (25, 30, 40, 20, 50),
    "torque-curve": (65, 55, 50, 15, 60),
    "tort": (-30, 20, 10, 15, 20),
    "tosser": (-75, 45, 10, 25, 10),
    "total-internal-reflection": (65, 55, 50, 15, -45),
    "touch-base": (-15, -15, 10, 40, 10),
    "touch-grass": (-90, 55, 50, 75, 15),
    "touchback": (20, -10, 35, 15, 10),
    "touchgrass": (-50, 30, 35, 55, -10),
    "tourbillon": (95, 60, 55, 10, 45),
    "tourbillon-cage": (100, 70, 65, 20, 30),
    "tourmaline": (75, 50, 40, 10, 30),
    "toxicology-screen": (-20, 50, 40, 70, 30),
    "trace-evidence": (15, 45, 30, 65, 5),
    "tracking": (40, 20, 35, 15, 5),
    "trade-winds": (60, 20, 35, 15, -35),
    "trajectory": (40, 55, 45, 65, 15),
    "transaxle": (40, 40, 50, 20, 70),
    "transceiver": (35, 30, 35, 40, 20),
    "transcendental": (85, 35, 50, 5, -60),
    "transcription": (65, 45, 50, 15, 30),
    "transference": (-25, 45, -15, 30, 10),
    "transform-fault-slip": (-75, 95, 55, 95, 80),
    "transformer": (20, 50, 45, 35, 55),
    "transgender": (55, 40, 35, 15, -15),
    "transgenic": (10, 45, 25, 20, 15),
    "transistor": (25, 15, 35, 5, 5),
    "translation-bio": (65, 45, 50, 15, 30),
    "transliteration": (40, 20, 40, 30, 15),
    "translocation": (45, 30, 40, 25, 20),
    "transparency": (60, 10, 45, 10, -15),
    "transpiration": (60, 15, 20, 5, -30),
    "transubstantiation": (65, 55, 45, 10, 55),
    "trap-house": (-45, 70, 40, 50, 40),
    "trapped-particle": (15, 55, 45, 45, -85),
    "tread-wear": (-55, 20, -30, 45, 15),
    "treason": (-95, 65, -10, 50, 35),
    "treaty": (70, 25, 45, 45, 35),
    "trebuchet": (-10, 55, 40, 50, 60),
    "tremolo": (20, 55, 10, 30, -5),
    "tremor": (-50, 50, -40, 60, 10),
    "trench": (-75, 45, -30, 60, 50),
    "trepidation": (-45, 60, -50, 60, 10),
    "tribology": (25, 35, 45, 15, 35),
    "tributary": (35, 25, 20, 10, -15),
    "tricolor": (55, 25, 40, 10, 20),
    "triglyceride": (-20, 15, -15, 15, 20),
    "trimix": (25, 65, 45, 55, 40),
    "triple-bottom-line": (90, 45, 65, 20, 45),
    "troglobite-blind": (35, 10, 40, 5, 60),
    "troglodyte": (-85, 25, -45, 10, 50),
    "troglodyte-dweller": (25, -15, 35, 5, 75),
    "troglophile-habit": (45, 15, 45, 5, 55),
    "troilite-sulfide": (-45, 35, 45, 20, 80),
    "trojan-asteroid": (35, 35, 60, 25, 75),
    "troling": (-50, 55, 35, 20, -10),
    "troll-face": (40, 55, 25, 10, -15),
    "trophic-cascade": (-40, 65, 45, 35, 30),
    "tropic": (65, 40, 30, 10, -20),
    "troposphere": (30, 10, 35, 10, -40),
    "truffle": (75, 30, 40, 5, 35),
    "trunk": (40, 10, 40, 15, 45),
    "truss": (45, 25, 50, 20, 60),
    "truss-rod": (40, 15, 55, 15, 60),
    "tryhard": (-40, 80, 35, 70, 10),
    "tryna": (15, 20, 15, 50, 0),
    "tuber": (40, -15, 35, 0, 55),
    "tuff": (35, 10, 35, 0, 30),
    "tulle": (60, 25, 15, 10, -55),
    "tumble": (-30, 55, -40, 50, -10),
    "tumulus": (-15, 5, 30, 5, 65),
    "tundra": (-25, -5, -15, 5, 25),
    "tung-oil": (70, 15, 35, 5, 5),
    "tungsten": (55, 45, 35, 15, 25),
    "tuning-peg": (45, 55, 45, 65, 15),
    "turbidity": (-40, 35, -20, 35, 20),
    "turbine": (35, 60, 50, 35, 50),
    "turbocharger": (50, 75, 45, 40, 25),
    "turbulence": (-55, 60, -40, 55, 5),
    "turing-complete": (85, 55, 75, 15, 80),
    "turing-test-pass": (85, 65, 65, 35, 45),
    "turncoat": (-100, 70, -25, 60, 30),
    "turntable": (65, 40, 35, 15, 25),
    "turqouise": (80, 25, 30, 5, -25),
    "turret": (30, 45, 40, 50, 45),
    "twat": (-75, 50, 10, 20, 10),
    "twat-waffle": (-75, 45, 10, 20, -10),
    "twatwaffle": (-80, 50, 15, 25, -5),
    "tweak": (-45, 60, -35, 50, 5),
    "tweed-harris": (70, 10, 60, 5, 75),
    "twill-weave": (50, 10, 40, 5, 35),
    "typeface": (60, 10, 45, 5, 20),
    "typhoon": (-75, 65, -50, 65, 35),
    "tzatziki": (70, -10, 30, 10, -20),
    "ulna": (30, 10, 30, 5, 35),
    "ultramarine": (65, 30, 40, 5, 35),
    "ultrasound": (45, 25, 30, 20, -20),
    "umami": (60, 10, 25, 0, 15),
    "umbilical": (40, 35, 45, 55, 20),
    "un-pack": (25, 20, 35, 30, 10),
    "unboxing": (65, 55, 20, 30, -20),
    "unc": (-10, 15, 15, 5, 25),
    "uncapping-knife": (30, 40, 35, 45, 15),
    "uncirculated": (80, 10, 55, 5, 40),
    "unconformity": (-15, 25, 25, 10, 35),
    "undermining": (-75, 60, 35, 60, 20),
    "understeer": (-45, 60, -35, 60, 15),
    "understudy": (-30, 45, -45, 60, 10),
    "ungulate": (45, 20, 40, 10, 35),
    "union-set": (75, 30, 45, 20, 10),
    "unparalleled": (95, 50, 55, 10, -35),
    "uppercase": (35, 35, 45, 10, 25),
    "upwelling": (65, 45, 40, 20, 10),
    "upwelling-zone": (70, 45, 45, 25, 15),
    "ursidae": (40, 45, 50, 40, 40),
    "usury": (-75, 40, 20, 25, 30),
    "utilitarianism": (55, 15, 50, 30, 40),
    "v-speed": (20, 50, 40, 55, -5),
    "vacuole": (35, 10, 35, 5, 20),
    "vacuum-decay": (-100, 100, 85, 100, 100),
    "vadose-zone": (35, 15, 50, 15, 75),
    "vagabond": (-55, 15, -40, 10, 20),
    "vair": (60, 15, 35, 0, 20),
    "valency": (20, 25, 35, 10, 15),
    "valhalla": (85, 65, 50, 20, 40),
    "valid": (60, 20, 40, 5, -20),
    "validating": (80, 20, 45, 10, -30),
    "validity": (60, 10, 40, 15, 10),
    "valise": (45, 20, 35, 25, 30),
    "valkyrie-chooser": (80, 75, 65, 60, -35),
    "valuation": (35, 40, 35, 30, 15),
    "valuation-multiple": (55, 55, 60, 45, 65),
    "value-add": (75, 35, 45, 20, 15),
    "vanguard": (70, 65, 50, 60, 35),
    "vanilla-js": (35, -25, 45, 5, 25),
    "vanity": (35, 25, 30, 15, 10),
    "vapor": (35, 15, 10, 20, -45),
    "vaporware": (-80, 20, -40, 15, -10),
    "vaporwave": (70, -25, 40, 5, -60),
    "variable": (5, 35, 15, 25, -5),
    "variance": (-15, 35, 15, 25, 10),
    "varietal": (60, 15, 40, 5, 15),
    "varnish": (60, 25, 35, 10, 15),
    "varnish-oil": (95, 25, 45, 10, 30),
    "varroa-mite": (-95, 45, -20, 55, 20),
    "vascular": (30, 55, 40, 55, 15),
    "vassal": (-20, 15, -40, 35, 15),
    "vaudeville": (65, 55, 20, 15, -20),
    "vault": (85, 35, 55, 15, 65),
    "vector-biological": (-40, 55, 35, 60, 5),
    "vector-database": (95, 70, 65, 35, 45),
    "velocity": (45, 70, 50, 75, -40),
    "velvet": (85, -35, 35, 0, -25),
    "vendetta": (-90, 65, 35, 50, 35),
    "vendor": (15, 15, 25, 20, 15),
    "veneer": (60, 15, 35, 5, 10),
    "venerable": (75, -15, 50, 5, 45),
    "veneration": (85, 20, 45, 5, 40),
    "vengeance": (-85, 80, 50, 55, 30),
    "venom-sac": (-70, 55, 35, 65, 20),
    "venturi-effect": (55, 55, 45, 30, -35),
    "vermillion": (65, 55, 30, 10, 10),
    "vernacular": (50, 25, 35, 5, 10),
    "vernalization": (60, 25, 45, 40, 30),
    "vertebrae": (35, 10, 45, 10, 40),
    "vertex": (25, 25, 35, 20, -10),
    "vertical": (25, 15, 40, 15, 55),
    "vesicle": (30, 35, 20, 25, -20),
    "vesicular-basalt": (25, 5, 35, 5, 35),
    "vestibular": (35, 55, 45, 60, -30),
    "vestibule": (60, 15, 35, 20, 25),
    "veto": (-50, 55, 50, 60, 40),
    "vexillum": (50, 15, 45, 5, 65),
    "viaduct": (70, 35, 60, 20, 70),
    "vibration": (20, 50, 20, 35, -10),
    "vibrato": (45, 35, 20, 5, -5),
    "victimology": (-45, 30, -10, 15, 40),
    "viewfinder": (55, 45, 45, 45, -10),
    "vigenere-cipher": (55, 40, 50, 15, 45),
    "vignetting": (-35, 30, 20, 10, 35),
    "vine": (20, 15, 15, 5, -5),
    "viniculture": (65, 15, 40, 5, 25),
    "vintage": (60, 10, 40, 5, 20),
    "violet": (75, 20, 30, 5, -15),
    "viper": (-75, 65, 30, 65, 10),
    "virga": (30, 15, -10, 5, -25),
    "viridian": (50, 10, 30, 0, 20),
    "virion": (20, 30, 30, 45, -20),
    "virtual-power-plant": (95, 65, 70, 30, 40),
    "virtualization": (40, 20, 40, 15, -20),
    "virtue-ethics": (90, 25, 55, 10, 45),
    "virulence": (-95, 75, 50, 80, 40),
    "visceral": (10, 65, 30, 60, 25),
    "viscometer": (15, 15, 30, 25, 40),
    "viscometry": (20, -10, 40, 20, 55),
    "viscosity": (10, -20, 35, 15, 55),
    "viscosity-dynamic": (10, -15, 45, 20, 75),
    "viscosity-index": (15, -10, 35, 10, 45),
    "viscount": (60, 25, 45, 15, 35),
    "viscous": (5, 20, 25, 20, 35),
    "vishnu-preserver": (90, 25, 70, 15, 55),
    "visuals": (90, 45, 45, 15, -45),
    "vitality": (90, 65, 55, 15, -30),
    "viterbi-decoder": (85, 55, 60, 50, 35),
    "vitis-vinifera": (60, 5, 45, 5, 35),
    "vitreous": (65, 30, 35, 5, 20),
    "viverrid": (30, 40, 25, 20, 5),
    "vlog": (40, 30, 25, 15, -10),
    "void": (-65, -45, -55, 0, 0),
    "volant": (65, 50, 35, 20, -45),
    "volatility": (-40, 60, -25, 55, 5),
    "volatility-index": (-55, 90, -25, 85, 5),
    "volcano": (-20, 60, 45, 55, 35),
    "volume": (25, 10, 30, 10, 15),
    "vortex": (-20, 55, -30, 40, -15),
    "vorticity": (-25, 65, 35, 60, 15),
    "vowel": (55, 30, 20, 5, -30),
    "vulnerability": (-70, 60, -50, 65, 10),
    "vulture": (-90, 50, -15, 40, 40),
    "waggle-dance": (65, 55, 30, 40, -40),
    "wagmi": (95, 65, 50, 20, -30),
    "waifu": (65, 45, -20, 10, -40),
    "walk-off-homer": (100, 90, 70, 50, -60),
    "walking-stick": (35, -5, 35, 5, 15),
    "wall-hack": (-95, 65, 45, 50, 15),
    "wanker": (-70, 50, 10, 25, 10),
    "wardrobe": (40, 5, 35, 10, 35),
    "warhead": (-90, 80, 55, 85, 45),
    "warning-card": (-65, 75, -10, 85, 15),
    "warp-thread": (35, 15, 35, 10, 45),
    "wasabi": (35, 60, 10, 25, -20),
    "washed": (-55, -20, -40, 5, 30),
    "washed-up": (-90, -10, -60, 20, 75),
    "washingmachine": (35, 30, 40, 35, 40),
    "wasp": (-65, 65, 20, 65, -25),
    "wastegate": (20, 65, 35, 50, 10),
    "watch-jewel": (80, 20, 50, 5, 15),
    "waterfall": (65, 35, 30, 5, -20),
    "waterfall-method": (-45, -35, 50, 15, 75),
    "watertight": (85, -15, 55, 45, 60),
    "wavelength-div-mux": (90, 55, 65, 45, 35),
    "wavelet-packet": (80, 55, 55, 25, -20),
    "weeb": (25, 65, 10, 25, 15),
    "weft-thread": (35, 15, 35, 10, -10),
    "weirdo": (-55, 35, -25, 20, 10),
    "welding": (25, 60, 40, 50, 30),
    "whale": (85, 15, 50, 5, 60),
    "whale-alert": (25, 75, 55, 85, 45),
    "whiskey": (50, 20, 25, 5, 30),
    "whistler-mode-wave": (45, 75, 40, 25, -95),
    "white-dwarf": (40, 25, 45, 10, 65),
    "white-paper": (45, 10, 40, 15, 20),
    "wicket-maiden": (85, 35, 55, 20, 45),
    "wide-angle": (75, 50, 45, 15, -35),
    "widmanstätten-pat": (95, 35, 60, 10, 75),
    "widowed": (-70, -25, -50, 10, 40),
    "wig-snatched": (95, 95, -20, 80, -85),
    "willow": (55, -35, 25, 0, -20),
    "willow-the-wisp": (-10, 55, -40, 45, -50),
    "windward": (10, 40, 35, 20, 15),
    "wing-load": (10, 30, 35, 20, 25),
    "winter-cluster": (35, -35, 40, 20, 55),
    "wireless-power-trans": (95, 65, 65, 40, 15),
    "woke": (15, 75, 25, 65, 15),
    "wolf": (50, 60, 40, 50, 20),
    "wood-filler": (35, 15, 35, 30, 20),
    "wool": (65, -20, 30, 5, 10),
    "workbench": (75, 20, 60, 25, 75),
    "worker-bee": (30, 45, 10, 40, 5),
    "workflow": (30, 20, 35, 30, 5),
    "working-memory": (45, 40, 45, 50, 10),
    "wraith": (-75, 65, -45, 50, -35),
    "x-height": (30, 5, 30, 5, 5),
    "xenolith-fragment": (20, 10, 45, 5, 55),
    "xenophobe": (-100, 60, -15, 40, 30),
    "xerophytic": (40, 15, 50, 10, 40),
    "xibalba-underworld": (-95, 60, -55, 45, 85),
    "xylem": (35, 5, 45, 10, 40),
    "xylology": (55, 10, 50, 10, 55),
    "yak-shaving": (-45, 40, -25, 30, 20),
    "yapping": (-55, 45, -20, 35, -5),
    "yaw": (5, 45, 25, 45, 0),
    "yearning": (40, 45, -35, 15, -15),
    "yggdrasil-tree": (90, 10, 70, 5, 85),
    "yield": (45, 20, 35, 15, 10),
    "yield-farming": (65, 40, 40, 20, 10),
    "yorker-length": (15, 80, 45, 90, 45),
    "zebra-pattern": (10, 55, 30, 65, 5),
    "zen": (95, -60, 55, 0, -50),
    "zephyr": (70, -35, 20, 0, -40),
    "zero-day": (-100, 95, -40, 100, 25),
    "zigbee-mesh": (70, 35, 55, 30, 15),
    "ziggurat": (70, 35, 55, 10, 75),
    "zircon": (65, 55, 40, 15, 50),
    "zk-proof": (100, 70, 75, 50, 45),
    "zoonotic": (-85, 70, 40, 80, 30),
    "zooted": (50, -30, -35, 10, -20),
    "zugzwang": (-85, 75, -50, 70, 50),
    "zwischenzug": (60, 70, 45, 65, 10),
    "épée": (50, 70, 50, 55, 35),

    # ── Common English patch: everyday words people actually use ──
    "above": (10, 5, 10, 0, -10),
    "across": (10, 10, 10, 5, 0),
    "against": (-15, 20, 10, 15, 10),
    "air": (20, 5, 5, 0, -15),
    "arm": (10, 10, 10, 5, 5),
    "aunt": (35, 10, 20, 0, 10),
    "autumn": (30, -5, 15, 0, 15),
    "became": (10, 15, 10, 10, 5),
    "began": (15, 20, 15, 15, 0),
    "believed": (20, 10, 15, 5, 5),
    "below": (-10, 5, -5, 0, 10),
    "beside": (15, 0, 5, 0, 0),
    "between": (0, 10, 0, 5, 0),
    "bike": (30, 25, 20, 10, -10),
    "black": (-15, 10, 15, 5, 15),
    "blue": (20, -10, 10, 0, -10),
    "boat": (30, 10, 15, 5, 10),
    "body": (10, 10, 10, 5, 10),
    "brought": (15, 15, 15, 10, 5),
    "brown": (5, -5, 10, 0, 15),
    "bus": (0, 10, -5, 15, 10),
    "business": (10, 15, 20, 15, 15),
    "called": (5, 15, 10, 15, 0),
    "came": (10, 15, 10, 10, 0),
    "case": (0, 10, 10, 15, 10),
    "ceiling": (5, 0, 5, 0, 10),
    "chair": (5, -10, 10, 0, 15),
    "chest": (10, 15, 15, 10, 10),
    "christmas": (60, 40, 25, 20, -10),
    "church": (30, -10, 25, 5, 25),
    "day": (10, 5, 5, 5, 0),
    "door": (5, 5, 10, 5, 10),
    "during": (0, 5, 0, 5, 0),
    "education": (30, 10, 20, 10, 10),
    "eye": (20, 15, 10, 5, -5),
    "face": (10, 15, 10, 5, 0),
    "fact": (10, 10, 15, 10, 10),
    "felt": (10, 15, 5, 5, -5),
    "finger": (10, 10, 5, 5, -5),
    "floor": (0, 0, 5, 0, 15),
    "food": (25, 15, 15, 10, 5),
    "foot": (5, 5, 10, 5, 10),
    "forgot": (-20, 15, -15, 10, -5),
    "friday": (40, 30, 20, 10, -10),
    "gave": (25, 15, 20, 5, -5),
    "gold": (50, 25, 35, 5, 15),
    "got": (10, 15, 10, 10, 0),
    "grandfather": (35, 5, 35, 0, 30),
    "grandma": (50, 10, 25, 0, 20),
    "grandmother": (40, 10, 30, 0, 25),
    "grandpa": (45, 5, 30, 0, 25),
    "gray": (-10, -10, 5, 0, 10),
    "green": (30, -5, 15, 0, -5),
    "hair": (15, 5, 5, 0, -5),
    "hand": (15, 10, 15, 5, 5),
    "happened": (-5, 20, -5, 15, 5),
    "head": (5, 10, 10, 5, 5),
    "health": (25, 10, 15, 10, 5),
    "heard": (10, 15, 5, 10, 0),
    "held": (20, 10, 20, 5, 10),
    "history": (15, 5, 15, 0, 20),
    "hour": (0, 10, 5, 15, 5),
    "idea": (25, 20, 15, 10, -5),
    "inside": (10, 5, 5, 5, 10),
    "kept": (10, 5, 15, 5, 10),
    "knew": (15, 5, 15, 5, 5),
    "left": (-15, 15, -10, 10, -5),
    "leg": (5, 10, 10, 5, 10),
    "level": (5, 5, 10, 5, 5),
    "lip": (15, 15, 5, 5, -5),
    "lived": (20, 10, 15, 5, 10),
    "loss": (-45, 20, -30, 15, -30),
    "made": (15, 15, 15, 10, 5),
    "mind": (15, 15, 15, 10, -5),
    "minute": (0, 10, 5, 20, 0),
    "monday": (-20, 15, -5, 15, 5),
    "month": (0, 5, 5, 5, 5),
    "moved": (10, 20, 10, 15, 5),
    "name": (10, 5, 10, 5, 5),
    "neck": (5, 10, 5, 10, 5),
    "needed": (-10, 15, -10, 20, 5),
    "nose": (5, 5, 5, 5, 0),
    "number": (0, 5, 10, 5, 5),
    "opened": (15, 15, 15, 10, 0),
    "outside": (15, 10, 5, 5, -5),
    "paid": (10, 10, 15, 15, 5),
    "park": (40, 10, 15, 0, -10),
    "part": (5, 5, 5, 5, 5),
    "people": (15, 10, 10, 5, 5),
    "person": (10, 5, 10, 5, 5),
    "pink": (35, 15, 5, 0, -15),
    "place": (15, 5, 10, 5, 5),
    "plane": (25, 30, 10, 20, -25),
    "played": (30, 25, 15, 5, -10),
    "point": (5, 10, 10, 10, 0),
    "purple": (25, 10, 15, 0, 5),
    "put": (5, 10, 10, 5, 5),
    "reason": (10, 10, 15, 10, 5),
    "red": (10, 30, 15, 10, 5),
    "restaurant": (35, 20, 15, 10, 5),
    "result": (10, 15, 15, 15, 5),
    "road": (10, 10, 10, 5, 10),
    "roof": (10, 5, 15, 5, 15),
    "said": (5, 10, 10, 5, 5),
    "saturday": (45, 20, 25, 0, -15),
    "seemed": (0, 5, -5, 5, 0),
    "ship": (25, 15, 20, 10, 20),
    "shoulder": (10, 5, 15, 5, 15),
    "showed": (10, 15, 15, 10, 0),
    "side": (5, 5, 5, 5, 5),
    "smile": (50, 20, 15, 0, -15),
    "spring": (50, 25, 20, 0, -20),
    "started": (15, 20, 15, 15, 0),
    "story": (20, 15, 10, 5, 5),
    "street": (5, 10, 5, 5, 5),
    "summer": (55, 30, 25, 0, -15),
    "sunday": (30, -10, 20, 0, 10),
    "table": (5, -5, 10, 0, 15),
    "teacher": (30, 15, 25, 10, 5),
    "tear": (-25, 25, -20, 10, -15),
    "teeth": (5, 10, 10, 5, 5),
    "thought": (10, 10, 10, 5, 0),
    "through": (10, 15, 10, 10, 0),
    "thursday": (5, 10, 5, 10, 0),
    "time": (0, 5, 5, 10, 5),
    "told": (5, 15, 10, 10, 5),
    "took": (5, 15, 10, 10, 5),
    "toward": (10, 10, 10, 10, 0),
    "truck": (5, 15, 20, 10, 25),
    "tuesday": (-10, 10, 0, 10, 5),
    "turned": (5, 15, 10, 10, 5),
    "uncle": (30, 10, 25, 0, 15),
    "until": (-5, 5, -5, 15, 5),
    "walked": (15, 5, 10, 5, 0),
    "watched": (10, 10, 10, 5, 0),
    "wednesday": (-5, 5, 0, 10, 5),
    "week": (0, 5, 5, 5, 5),
    "went": (5, 10, 5, 10, 0),
    "white": (20, -5, 10, 0, -10),
    "window": (15, 5, 10, 0, -5),
    "winter": (-10, 5, 10, 5, 15),
    "without": (-20, 10, -15, 5, -10),
    "woman": (15, 10, 10, 5, 0),
    "year": (5, 5, 5, 5, 10),
    "yellow": (30, 20, 10, 5, -10),

    # ── Gemini lexicon 15-17 ──
    "after": (40, 25, 50, 45, 35),
    "again": (25, 45, 30, 55, 10),
    "airport": (35, 75, 45, 85, 45),
    "all": (60, 35, 75, 15, 40),
    "allow": (65, 15, 50, 20, -15),
    "and": (10, 5, 20, 5, 0),
    "ankle": (10, 15, 25, 10, 15),
    "appear": (15, 25, 10, 25, 0),
    "apron": (40, -10, 25, 5, 15),
    "around": (20, 20, 15, 10, 0),
    "back": (10, 10, 20, 15, 15),
    "backpack": (60, 45, 40, 20, 30),
    "bag": (25, 15, 25, 30, 15),
    "bake-verb": (95, 45, 45, 40, 25),
    "bank": (20, 20, 50, 30, 55),
    "bath": (75, -35, 50, 15, 15),
    "battery-common": (45, 25, 45, 70, 25),
    "beach": (95, 45, 40, 15, -25),
    "because": (45, 20, 55, 30, 20),
    "bed": (85, -45, 50, 0, -50),
    "beer-common": (50, 25, -15, 10, 15),
    "before": (20, 35, 35, 60, 50),
    "belt": (20, 10, 35, 5, 15),
    "bench": (70, -35, 35, 0, 40),
    "big": (35, 25, 55, 10, 70),
    "bit": (15, 5, 5, 5, 0),
    "blackboard": (15, 5, 25, 5, 50),
    "bleach": (-30, 45, 35, 45, 10),
    "blink": (15, 20, 15, 10, -10),
    "blood": (-15, 80, 25, 90, 20),
    "bolt": (10, 25, 35, 20, 50),
    "bookcase": (50, -10, 40, 0, 55),
    "bottle": (25, 10, 20, 25, 15),
    "bottom": (-15, -10, 25, 5, 80),
    "box": (15, 5, 20, 15, 35),
    "box-shipping": (35, 45, 30, 55, 45),
    "brake": (10, 65, 55, 95, 35),
    "break-action": (-75, 75, 25, 60, 20),
    "bring": (45, 35, 40, 55, 10),
    "brush": (45, -5, 25, 15, -5),
    "brush-paint": (65, 25, 40, 15, -5),
    "build": (85, 55, 60, 30, 65),
    "building": (15, 10, 45, 5, 75),
    "bulb": (65, 25, 35, 35, -25),
    "bunch": (50, 25, 25, 10, 15),
    "bush": (40, 0, 20, 5, 30),
    "button": (15, 15, 15, 10, 0),
    "cabinet": (15, 0, 30, 10, 40),
    "can": (10, 5, 15, 20, 20),
    "car": (45, 45, 50, 40, 55),
    "card-payment": (45, 35, 45, 40, 10),
    "carry": (20, 45, 40, 40, 55),
    "cash": (85, 65, 55, 50, 25),
    "century": (65, 5, 85, 15, 100),
    "chalk": (10, 15, 15, 10, 5),
    "chapter": (45, 15, 30, 5, 20),
    "check-payment": (20, 30, 35, 45, 15),
    "chew": (35, 20, 30, 35, 15),
    "child-human": (80, 75, -25, 35, -40),
    "choice": (45, 45, 60, 55, 15),
    "choose": (55, 35, 55, 45, 10),
    "city": (45, 55, 35, 30, 45),
    "classroom": (35, 15, 25, 15, 20),
    "clean-verb": (95, 35, 55, 35, 20),
    "clear-transparent": (85, -20, 50, 15, -55),
    "climb": (70, 85, 55, 45, 75),
    "clip": (15, 15, 20, 15, 5),
    "clothes": (40, 10, 30, 15, 10),
    "clothespin": (15, 15, 20, 15, 5),
    "coffee-common": (70, 60, 35, 60, -15),
    "collar": (15, 10, 25, 5, 10),
    "comb": (40, -10, 25, 15, -10),
    "come": (35, 30, 25, 40, 0),
    "cook-verb": (85, 55, 50, 55, 35),
    "cord": (10, 15, 25, 30, 15),
    "corner": (-5, 15, -15, 10, 15),
    "corner-street": (0, 15, 10, 20, 10),
    "correct": (90, 20, 65, 30, -25),
    "cost": (-15, 50, 30, 65, 30),
    "cotton": (85, -35, 40, 0, -25),
    "cough": (-45, 45, 15, 55, 15),
    "count": (15, 20, 45, 45, 25),
    "countertop": (20, -5, 40, 10, 50),
    "cow": (55, -20, 45, 5, 75),
    "cream": (90, -30, 40, 10, 5),
    "cup": (35, 10, 25, 30, 5),
    "curtain": (40, -15, 20, 5, -20),
    "daily": (25, -15, 45, 30, 35),
    "dark-concept": (-85, 35, -40, 25, 90),
    "day-concept": (90, 45, 55, 15, -60),
    "death-concept": (-100, -50, -100, 65, 100),
    "decade": (45, 10, 75, 35, 100),
    "desk": (25, -5, 40, 15, 65),
    "detergent": (40, 10, 30, 25, 15),
    "dictionary": (55, 0, 55, 10, 60),
    "die-verb": (-100, 55, -100, 70, 95),
    "dig": (30, 60, 45, 35, 65),
    "dozen": (45, 25, 35, 15, 55),
    "drawer": (10, 5, 25, 15, 25),
    "drive-verb": (65, 55, 60, 45, 45),
    "drop": (-30, 50, -40, 45, 60),
    "dry": (15, -15, 25, 10, 5),
    "duck": (60, 35, 15, 15, -10),
    "dusty": (-45, -5, -20, 15, 15),
    "early": (50, 45, 40, 85, -25),
    "east": (45, 25, 25, 10, -10),
    "eat": (85, 35, 45, 60, 15),
    "egg": (65, 10, 35, 40, 15),
    "elbow": (10, 10, 25, 5, 15),
    "envelope": (45, 30, 25, 40, -5),
    "eraser": (40, -15, 20, 15, -15),
    "fabric": (55, -10, 30, 10, -10),
    "fall-verb": (-95, 95, -100, 75, 100),
    "fast": (65, 95, 55, 95, -70),
    "fear-verb": (-85, 85, -55, 90, 30),
    "fence": (15, 5, 45, 5, 45),
    "few": (-15, 5, -20, 5, 10),
    "field": (65, -15, 40, 5, 15),
    "flour": (40, -25, 30, 15, 35),
    "fly-verb": (100, 75, 55, 35, -95),
    "fold": (65, -35, 45, 20, 25),
    "forbid": (-85, 65, 60, 75, 50),
    "form": (5, 20, 40, 55, 35),
    "freezer": (20, -10, 40, 10, 45),
    "fridge": (60, 5, 45, 25, 55),
    "front": (35, 25, 30, 20, -10),
    "fruit": (95, 20, 30, 20, -30),
    "future-concept": (85, 65, 55, 50, -70),
    "gate": (25, 15, 35, 30, 25),
    "glass-cup": (30, 15, 20, 25, 10),
    "glasses": (40, 5, 35, 15, 5),
    "glue": (15, 10, 35, 25, 40),
    "go": (20, 40, 35, 50, -10),
    "grass": (75, -25, 30, 0, -15),
    "grater": (10, 30, 20, 15, 10),
    "ground": (45, -5, 75, 0, 100),
    "group": (35, 30, 40, 15, 25),
    "guilt": (-90, 65, -85, 40, 60),
    "half": (5, -5, -10, 25, 15),
    "hallway": (0, 10, -20, 25, 15),
    "handful": (45, 15, 25, 10, 10),
    "handle": (30, 15, 45, 30, 25),
    "hanger": (20, 5, 25, 15, 5),
    "happen": (10, 45, 5, 50, 0),
    "hard-diff": (-65, 75, 45, 95, 75),
    "hard-difficult": (-55, 65, 40, 85, 60),
    "hat": (40, 5, 20, 5, -15),
    "hate-verb": (-95, 75, 35, 55, 40),
    "heap": (10, 15, 35, 15, 60),
    "heart-organ": (60, 50, 65, 20, 25),
    "help-verb": (95, 40, 50, 65, 10),
    "here-adv": (55, 45, 60, 85, 30),
    "high": (55, 45, 50, 20, -90),
    "hill": (55, 15, 35, 10, 45),
    "hinge": (25, 10, 35, 20, 35),
    "hip": (20, 15, 30, 5, 25),
    "hook": (10, 25, 25, 30, 10),
    "hospital-general": (-15, 45, 35, 80, 50),
    "hotel": (70, 15, 50, 20, 55),
    "humid": (-45, 35, -20, 20, 30),
    "if": (5, 45, 10, 50, -10),
    "ink": (15, 15, 35, 25, 15),
    "iron-metal": (15, 5, 45, 5, 65),
    "ironing-board": (10, 10, 30, 30, 35),
    "jacket": (50, 10, 35, 10, 25),
    "jar": (25, 0, 25, 10, 30),
    "job": (35, 40, 50, 55, 45),
    "kettle": (40, 30, 30, 45, 25),
    "key": (40, 20, 45, 60, 15),
    "keypad": (15, 30, 35, 45, 10),
    "kick": (-55, 85, 45, 75, 35),
    "kill-verb": (-100, 95, 75, 90, 80),
    "knee": (15, 15, 35, 10, 25),
    "knit": (85, -45, 45, 10, 20),
    "ladle": (25, 5, 25, 10, 15),
    "laundry-basket": (10, 15, 25, 45, 40),
    "lawn": (80, -20, 35, 0, 25),
    "lay": (75, -45, 35, 5, -20),
    "least": (-25, -10, -20, 10, 10),
    "leave": (-35, 45, -20, 65, 15),
    "less": (-30, 10, -15, 15, 15),
    "lesson": (45, 15, 20, 25, 10),
    "letter": (70, 35, 35, 45, 5),
    "lid": (15, 10, 35, 20, 10),
    "life-concept": (100, 85, 70, 25, -90),
    "lift-action": (30, 60, 45, 50, 20),
    "light-concept": (100, 55, 55, 15, -95),
    "light-switch": (10, 10, 30, 40, 5),
    "light-weight": (75, 15, 40, 5, -95),
    "light-wt": (85, 25, 45, 10, -100),
    "like-verb": (75, 25, 35, 10, -20),
    "list": (40, 15, 55, 35, 30),
    "living-room": (75, -20, 45, 10, 15),
    "lock": (20, 15, 50, 40, 40),
    "lock-verb": (55, 35, 60, 75, 50),
    "loose": (25, -25, -15, 5, -30),
    "lotion": (70, -25, 30, 10, -15),
    "love-verb": (100, 70, 55, 40, -80),
    "low": (-25, -20, -35, 15, 65),
    "mail": (55, 40, 35, 60, 10),
    "man-human": (25, 15, 40, 5, 35),
    "many": (40, 30, 45, 10, 20),
    "marker": (30, 20, 25, 25, 5),
    "market": (60, 50, 35, 45, 30),
    "math": (10, 50, 25, 45, 40),
    "method": (45, 15, 55, 20, 40),
    "middle": (10, 0, 20, 5, 10),
    "million": (85, 75, 85, 25, 100),
    "mix": (45, 45, 35, 35, 20),
    "monthly": (15, -5, 35, 20, 55),
    "mop-verb": (35, 35, 40, 35, 35),
    "more": (50, 40, 40, 30, -10),
    "most": (55, 35, 55, 20, 20),
    "mouth": (30, 25, 25, 15, 0),
    "move": (35, 55, 40, 45, -10),
    "nail": (0, 30, 20, 25, 25),
    "name-label": (25, 15, 35, 25, 10),
    "napkin": (45, -20, 20, 10, -15),
    "necklace": (75, 25, 30, 5, 10),
    "needle": (-5, 65, 30, 55, 5),
    "news": (10, 75, 30, 95, 20),
    "next": (35, 45, 25, 75, -10),
    "night-concept": (65, -35, 45, 15, 75),
    "none": (-40, -25, -60, 5, 10),
    "north": (10, 15, 25, 5, 10),
    "notebook": (55, 10, 35, 15, 20),
    "now-adv": (55, 75, 60, 100, -50),
    "nut": (55, 15, 40, 15, 50),
    "oil": (25, -5, 35, 15, 20),
    "once": (35, 25, 30, 30, 45),
    "one": (35, 15, 40, 10, 30),
    "opaque": (-10, -10, 30, 5, 55),
    "opener": (35, 15, 35, 30, 10),
    "or": (0, 15, 15, 25, 0),
    "outlet": (5, 25, 35, 45, 35),
    "oven": (45, 65, 40, 55, 55),
    "page": (30, -5, 20, 10, -5),
    "pair": (60, 20, 35, 10, 20),
    "pan": (35, 45, 35, 45, 45),
    "pants": (30, 5, 25, 10, 20),
    "paper": (25, -10, 20, 15, -15),
    "pardon": (25, 5, -15, 35, 5),
    "past-concept": (25, -10, 40, 0, 75),
    "path": (55, 15, 30, 10, 15),
    "peace-state": (100, -90, 70, 0, -80),
    "peeler": (25, 20, 30, 15, 10),
    "pen": (35, 10, 30, 25, 5),
    "pencil": (30, 10, 25, 20, 0),
    "perfume": (75, 30, 30, 10, -35),
    "petal": (85, 15, 15, 0, -55),
    "piece": (20, 10, 15, 10, 10),
    "pig": (40, 10, 35, 5, 55),
    "pile": (15, 10, 30, 15, 55),
    "pinch": (-45, 65, 20, 60, 10),
    "pipe": (10, 10, 40, 20, 55),
    "plan": (55, 35, 55, 45, 35),
    "play-action": (95, 75, 40, 15, -45),
    "pocket": (30, 5, 20, 5, 5),
    "pot": (30, 35, 35, 35, 60),
    "present-concept": (75, 55, 55, 85, 10),
    "previous": (15, 10, 35, 25, 20),
    "puddle": (35, 15, -10, 10, 15),
    "punch": (-85, 95, 55, 95, 45),
    "purse": (30, 15, 30, 15, 10),
    "quarter": (0, 0, -15, 30, 10),
    "rainy": (-15, -5, -20, 35, 25),
    "rat": (-60, 50, -35, 45, 10),
    "razor": (-10, 45, 30, 40, 5),
    "read": (75, -15, 45, 15, 25),
    "reading": (65, -20, 40, 10, 10),
    "ready": (85, 85, 70, 100, -45),
    "report": (10, 35, 45, 65, 40),
    "ride": (85, 65, 45, 25, 30),
    "ring": (80, 40, 45, 10, 20),
    "rock": (15, -5, 45, 0, 75),
    "roller": (55, 35, 40, 30, 15),
    "rope": (35, 30, 45, 35, 55),
    "round": (65, -10, 30, 5, 25),
    "rub": (65, -15, 30, 20, 10),
    "ruler": (25, 5, 40, 15, 20),
    "sand": (55, -10, 25, 5, 15),
    "save-verb": (90, 75, 60, 95, 25),
    "scanner": (20, 30, 35, 30, 35),
    "scent": (85, 25, 35, 10, -25),
    "science": (45, 30, 40, 15, 20),
    "scissors": (5, 55, 35, 45, 15),
    "screw": (5, 30, 25, 25, 20),
    "second": (5, 45, 10, 95, -100),
    "see": (40, 35, 40, 20, -5),
    "seem": (0, 10, -10, 20, 5),
    "set": (25, 10, 35, 15, 35),
    "sew": (60, 20, 45, 25, 10),
    "shade": (70, -35, 35, 10, 25),
    "shampoo": (80, -20, 40, 25, -15),
    "sheep": (65, -30, 30, 5, 40),
    "shelf": (20, -5, 30, 0, 35),
    "shiny": (95, 65, 40, 20, -60),
    "shirt": (35, 5, 25, 10, 10),
    "shoes": (30, 20, 25, 25, 15),
    "show": (55, 40, 45, 35, -5),
    "shower": (60, 30, 45, 40, -15),
    "shrub": (45, -5, 25, 5, 25),
    "sidewalk": (15, 20, 25, 15, 10),
    "sign": (10, 15, 30, 35, 10),
    "sink": (20, 5, 30, 45, 20),
    "slap": (-75, 85, 35, 85, 15),
    "sleepy": (25, -85, -35, 45, 25),
    "sleeve": (20, 5, 20, 5, 5),
    "slow": (-45, -65, 25, 30, 60),
    "small": (45, 10, -45, 5, -35),
    "smell": (55, 35, 30, 15, -15),
    "sneeze": (-10, 55, 20, 65, -10),
    "soap": (75, -25, 45, 35, -20),
    "socket": (5, 25, 30, 45, 35),
    "socks": (55, -20, 20, 5, 5),
    "some": (10, 5, 10, 5, 5),
    "south": (35, 20, 25, 5, 5),
    "spice": (70, 75, 30, 25, -15),
    "sponge": (30, -5, 20, 15, -10),
    "square": (35, 5, 45, 10, 50),
    "stack": (30, 15, 45, 20, 70),
    "stain-wood": (20, 15, 30, 15, 10),
    "stairs": (10, 30, 25, 35, 45),
    "stamp": (45, 15, 35, 25, 10),
    "stapler": (25, 40, 35, 35, 25),
    "start-engine": (75, 85, 55, 75, 35),
    "station": (25, 45, 35, 55, 50),
    "stay": (60, -30, 50, 15, 35),
    "stem": (25, 5, 25, 5, 15),
    "stick": (10, 5, 15, 10, 15),
    "sticky": (-55, 35, -15, 25, 15),
    "still": (55, -45, 50, 15, 45),
    "stir": (50, 25, 35, 30, 15),
    "stone": (20, -5, 40, 0, 65),
    "stop-engine": (45, -25, 60, 65, 55),
    "stormy": (-75, 80, -35, 90, 35),
    "stove": (30, 80, 45, 65, 50),
    "string": (15, 5, 15, 15, -5),
    "student": (55, 35, -15, 20, 5),
    "surprised": (45, 100, 15, 75, -50),
    "swallow": (25, 25, 35, 40, 15),
    "sweat": (-25, 65, -10, 50, 15),
    "sweep": (45, 25, 40, 30, 30),
    "swim": (95, 60, 45, 20, -25),
    "switch": (20, 35, 45, 55, 10),
    "system-general": (25, 15, 50, 10, 65),
    "tablecloth": (50, -15, 30, 5, 20),
    "tape": (30, 10, 25, 25, 0),
    "tape-measure": (35, 15, 35, 25, 15),
    "task": (10, 45, 35, 75, 25),
    "taste": (75, 55, 35, 20, -10),
    "tent": (65, 35, 35, 15, 15),
    "then-adv": (20, 25, 35, 55, 35),
    "there-adv": (15, 15, 30, 55, 45),
    "thick": (25, 15, 55, 10, 75),
    "thousand": (55, 50, 65, 15, 85),
    "thread": (25, 10, 20, 20, -15),
    "throat": (-5, 35, 20, 60, 20),
    "tide": (35, 15, 50, 10, 45),
    "tight": (-25, 40, 30, 50, 30),
    "tire": (15, 15, 35, 45, 50),
    "toe": (10, 15, 15, 5, 5),
    "toilet": (-10, 15, 20, 60, 25),
    "tooth": (10, 15, 25, 10, 15),
    "top": (55, 35, 50, 10, -70),
    "total": (65, 25, 75, 50, 80),
    "touch": (85, 45, 35, 25, -20),
    "towel": (55, -25, 30, 10, -10),
    "town": (55, 25, 30, 15, 25),
    "trash-can": (-30, 15, 20, 50, 25),
    "tray": (20, 5, 25, 15, 20),
    "true": (90, 15, 65, 10, 25),
    "twice": (45, 35, 35, 35, 35),
    "two": (45, 25, 35, 15, 25),
    "umbrella": (55, -5, 40, 50, 10),
    "vent": (15, 25, 30, 30, 15),
    "village": (60, 10, 25, 10, 15),
    "vinegar": (-35, 55, 25, 20, 10),
    "wake": (45, 65, 45, 70, -35),
    "wallet": (35, 20, 45, 30, 15),
    "wash": (85, 25, 50, 40, 15),
    "washer": (15, 5, 20, 15, 10),
    "watch": (25, 15, 40, 25, 10),
    "wave": (65, 55, 40, 35, 20),
    "way": (15, 15, 35, 20, 20),
    "wear": (40, 10, 35, 15, 5),
    "weekend": (100, 55, 50, 20, -50),
    "weekly": (20, -10, 40, 25, 45),
    "west": (25, 15, 25, 5, 10),
    "wet": (10, 20, 10, 20, 10),
    "wheel": (35, 25, 40, 15, 35),
    "while": (35, -25, 40, 15, 10),
    "whiteboard": (25, 10, 30, 10, 40),
    "windy": (20, 65, 15, 45, -20),
    "wine-common": (70, 15, 25, 5, 15),
    "with": (60, 25, 40, 15, 0),
    "woman-human": (30, 15, 40, 5, 20),
    "work-action": (20, 50, 45, 65, 45),
    "wrist": (15, 15, 20, 5, 5),
    "write": (50, 20, 45, 30, 15),
    "writing": (60, 20, 45, 25, 15),
    "yearly": (35, 5, 45, 15, 65),
    "yet": (-10, 40, 15, 65, 10),
    "zero": (-15, -55, -45, 15, 50),
    "zipper": (10, 20, 15, 15, 0),
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
# STEP 6.5: ResponseBuilder — Math-Based Word Selection
# =============================================================

class ResponseBuilder:
    """Builds responses by selecting words from WORD_FORCES based on
    VADUG coordinate distance. Every word is chosen by math, not templates.

    Rules:
    1. No word used twice in a response
    2. No phrase structure repeated across chunks
    3. Track recently used words across conversation
    """

    # Curated adjective sets -- words that work in "That sounds X" / "That's really X"
    # These are all present in WORD_FORCES and selected for natural sentence fit
    NEGATIVE_ADJECTIVES = {
        "tough", "hard", "heavy", "rough", "painful", "awful", "terrible", "horrible",
        "devastating", "exhausting", "overwhelming", "difficult", "sad", "lonely",
        "scary", "stressful", "heartbreaking", "harsh", "brutal", "draining",
        "crushing", "intense", "deep", "frustrating", "agonizing", "dreadful",
        "miserable", "wretched", "hopeless", "helpless", "exhausted", "broken",
        "uncomfortable", "uneasy", "anxious", "frightened", "terrified",
        "disappointed", "embarrassed", "ashamed", "guilty", "stuck", "lost",
        "worried", "stressed", "tired", "sick", "numb", "unfair", "unbearable",
        "meaningless", "pointless", "useless", "bleak", "grim", "somber",
        "melancholy", "bittersweet", "isolating", "toxic", "chaotic",
    }

    POSITIVE_ADJECTIVES = {
        "amazing", "incredible", "wonderful", "fantastic", "beautiful", "brilliant",
        "excellent", "awesome", "magnificent", "spectacular", "phenomenal", "outstanding",
        "remarkable", "extraordinary", "superb", "marvelous", "glorious", "blessed",
        "grateful", "thankful", "proud", "accomplished", "triumphant", "victorious",
        "successful", "thriving", "flourishing", "radiant", "gorgeous", "stunning",
        "ecstatic", "thrilled", "excited", "joyful", "happy", "delicious",
        "inspiring", "vibrant", "colorful", "bright", "warm", "gentle", "peaceful",
        "hopeful", "powerful", "resilient", "strong", "creative", "bold", "brave",
    }

    NOUN_FEELINGS = {
        "grief", "sorrow", "pain", "loss", "anguish", "despair", "heartbreak",
        "struggle", "weight", "darkness", "sadness", "loneliness", "fear",
        "frustration", "anger", "regret", "guilt", "shame", "exhaustion",
        "confusion", "chaos", "stress", "worry", "doubt", "torment",
        "suffering", "burden", "tragedy", "nightmare", "betrayal", "rejection",
        "joy", "love", "pride", "hope", "strength", "courage", "resilience",
        "warmth", "comfort", "peace", "harmony", "triumph", "victory",
        "excitement", "wonder", "gratitude", "relief", "freedom", "growth",
    }

    def __init__(self):
        self.used_words = set()
        self.used_structures = set()
        self.session_history = {}
        self._categorize_words()

    def _categorize_words(self):
        """Categorize WORD_FORCES entries by likely sentence role based on VADUG."""
        self.ACKNOWLEDGE_WORDS = set()
        self.STABILIZE_WORDS = set()
        self.EMOTION_WORDS = set()
        self.ACTION_WORDS = set()
        self.neg_adj_pool = {}
        self.pos_adj_pool = {}
        self.noun_pool = {}

        for word, forces in WORD_FORCES.items():
            v, a, d, u, g = forces
            if v < -10:
                self.ACKNOWLEDGE_WORDS.add(word)
            if d > 15:
                self.STABILIZE_WORDS.add(word)
            if -30 < v < 30 and abs(a) < 30:
                self.EMOTION_WORDS.add(word)
            if v > 0 and u > 5:
                self.ACTION_WORDS.add(word)

            # Build curated pools from WORD_FORCES intersection with our adjective sets
            if word in self.NEGATIVE_ADJECTIVES:
                self.neg_adj_pool[word] = forces
            if word in self.POSITIVE_ADJECTIVES:
                self.pos_adj_pool[word] = forces
            if word in self.NOUN_FEELINGS:
                self.noun_pool[word] = forces

    def find_closest_words(self, target_v, target_a, target_d, target_u, target_g,
                           word_pool=None, n=30):
        """Find words whose VADUG forces are closest to the target coordinates.

        Target values are in 0-255 scale. WORD_FORCES are in force scale (-80 to +80).
        Convert target to force scale: force = target - 128 (for V,A,D,G), force = target (for U).
        """
        target_vf = target_v - 128
        target_af = target_a - 128
        target_df = target_d - 128
        target_uf = target_u
        target_gf = target_g - 128

        pool = word_pool if word_pool else WORD_FORCES
        candidates = []

        for word, forces in pool.items():
            if word in self.used_words:
                continue
            if self.session_history.get(word, 0) > 2:
                continue
            if len(word) < 3:
                continue

            v, a, d, u, g = forces

            distance = math.sqrt(
                3.0 * (v - target_vf)**2 +
                1.0 * (a - target_af)**2 +
                1.0 * (d - target_df)**2 +
                0.5 * (u - target_uf)**2 +
                2.0 * (g - target_gf)**2
            )
            candidates.append((distance, word, v, a, d, u, g))

        return sorted(candidates)[:n]

    def select_word(self, target_v, target_a, target_d, target_u, target_g,
                    word_pool=None):
        """Select the best unused word matching the target VADUG."""
        candidates = self.find_closest_words(target_v, target_a, target_d, target_u, target_g, word_pool)

        if candidates:
            _, word, _, _, _, _, _ = candidates[0]
            self.used_words.add(word)
            self.session_history[word] = self.session_history.get(word, 0) + 1
            return word
        return None

    def _pick_structure(self, structures):
        """Pick the first structure whose opening pattern hasn't been used.

        Uses first 2 words as pattern key to prevent 'That sounds X' and
        'That sounds Y' from both being selected across chunks.
        """
        for s in structures:
            if s is None:
                continue
            words = s.split()
            pattern_key = ' '.join(words[0:2]) if len(words) >= 2 else s
            if pattern_key not in self.used_structures:
                self.used_structures.add(pattern_key)
                return s
        # Fallback: return the first non-None
        for s in structures:
            if s is not None:
                return s
        return None

    def build_acknowledge(self, input_vadug, response_vadug):
        """Build an acknowledgment phrase from WORD_FORCES.

        Selects both an adjective (for 'That sounds X') and a noun (for 'the X
        in that') closest to the input VADUG, then uses the best fitting template.
        """
        # Select an adjective from curated negative adjective pool
        adj_pool = {w: f for w, f in self.neg_adj_pool.items() if w not in self.used_words}
        adj = self.select_word(
            input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
            adj_pool if adj_pool else None
        )

        # Select a noun from curated feeling-noun pool
        noun_pool = {w: f for w, f in self.noun_pool.items()
                     if w not in self.used_words and WORD_FORCES.get(w, (0,))[0] < 0}
        noun = self.select_word(
            input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
            noun_pool if noun_pool else None
        )

        structures = []
        if adj:
            structures.extend([
                f"That sounds {adj}.",
                f"That's really {adj}.",
                f"That's genuinely {adj}.",
            ])
        if noun:
            structures.extend([
                f"I can feel the {noun} in that.",
                f"I hear the {noun}.",
                f"That level of {noun} is real.",
            ])
        if adj:
            structures.append(f"{adj.capitalize()} is the right word for it.")
        structures.append("I hear you.")

        result = self._pick_structure(structures)
        return result if result else "I hear you."

    def build_stabilize(self, response_vadug):
        """Build a stabilizing phrase from WORD_FORCES.

        Picks words that project the RESPONSE vadug (stability, presence).
        """
        stab_pool = {w: WORD_FORCES[w] for w in self.STABILIZE_WORDS
                     if w in WORD_FORCES and w not in self.used_words}
        anchor = self.select_word(
            response_vadug.v, response_vadug.a, response_vadug.d, response_vadug.u, response_vadug.g,
            stab_pool if stab_pool else None
        )

        structures = []
        if anchor:
            if anchor in ("here", "near", "close"):
                structures.append(f"I'm right {anchor} with you.")
            elif anchor in ("strength", "courage", "resilience", "resolve", "grit",
                            "comfort", "warmth"):
                structures.append(f"You've got {anchor} in you.")
            elif anchor in ("handle", "figure", "manage", "work", "face", "tackle",
                            "navigate", "endure"):
                structures.append(f"We'll {anchor} this together.")
            elif anchor in ("together", "beside", "alongside"):
                structures.append(f"We're in this {anchor}.")
            elif anchor in ("strong", "powerful", "resilient", "brave", "capable"):
                structures.append(f"You're stronger than this feels.")
            else:
                structures.append(f"We'll get through this {anchor}.")

        structures.extend([
            "I'm not going anywhere.",
            "You don't have to carry this alone.",
            "Take whatever time you need.",
            "I'm right here.",
        ])

        result = self._pick_structure(structures)
        return result if result else "I'm here."

    # Verbs that work in "Let's X" / "What would X" redirect templates
    REDIRECT_VERBS = {
        "look", "start", "begin", "explore", "try", "work", "figure", "think",
        "plan", "talk", "focus", "tackle", "check", "sort", "help", "fix",
        "learn", "study", "practice", "build", "create", "make", "find",
    }

    def build_redirect(self, response_vadug, grade_rules=None):
        """Build a redirect/next-step phrase. Respects grade rules."""
        if grade_rules and any(b in (grade_rules.get('blocked', []))
                               for b in ['advice', 'redirect', 'problem_solving']):
            return None

        # Only select from known verbs that work in redirect templates
        verb_pool = {w: WORD_FORCES[w] for w in self.REDIRECT_VERBS
                     if w in WORD_FORCES and w not in self.used_words}
        action = self.select_word(
            response_vadug.v, response_vadug.a, response_vadug.d, response_vadug.u, response_vadug.g,
            verb_pool if verb_pool else None
        )

        structures = []
        if action:
            if action in ("look", "start", "begin", "explore", "check"):
                structures.append(f"Let's {action} at this fresh.")
            elif action in ("try", "work", "figure", "think", "plan", "sort",
                            "talk", "focus"):
                structures.append(f"Let's {action} through it.")
            elif action in ("help", "fix", "tackle", "find"):
                structures.append(f"Let's {action} this.")
            else:
                structures.append(f"Let's {action} from here.")

        structures.extend([
            "What do you need from me right now?",
            "Where do you want to start?",
        ])

        result = self._pick_structure(structures)
        return result

    def build_positive_acknowledge(self, input_vadug, response_vadug):
        """Build a positive acknowledgment for happy/excited input."""
        adj_pool = {w: f for w, f in self.pos_adj_pool.items() if w not in self.used_words}
        word = self.select_word(
            input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
            adj_pool if adj_pool else None
        )

        if not word:
            return "That's great to hear."

        structures = [
            f"That's genuinely {word}.",
            f"That sounds {word}!",
            f"That's really {word}.",
            f"I can feel the {word} in that.",
        ]

        result = self._pick_structure(structures)
        return result if result else "That's great to hear."

    # Adjectives that work in "that's genuinely X" / "that part is X" (situation descriptors)
    SITUATION_POS_ADJECTIVES = {
        "amazing", "incredible", "wonderful", "fantastic", "beautiful", "brilliant",
        "excellent", "awesome", "magnificent", "spectacular", "phenomenal", "outstanding",
        "remarkable", "extraordinary", "superb", "marvelous", "glorious", "exciting",
        "inspiring", "vibrant", "stunning", "thrilling", "special", "great",
    }

    def build_reversal_response(self, input_vadug, response_vadug, is_positive_reversal):
        """Build a response for a reversal chunk (after 'but', 'however', etc.)."""
        if is_positive_reversal:
            # Use situation-appropriate positive adjectives only
            sit_pool = {w: WORD_FORCES[w] for w in self.SITUATION_POS_ADJECTIVES
                        if w in WORD_FORCES and w not in self.used_words}
            word = self.select_word(
                input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
                sit_pool if sit_pool else None
            )
            structures = [
                f"But that part is {word}." if word else None,
                f"But wait -- that's {word}!" if word else None,
                f"But hold on, that's genuinely {word}." if word else None,
                "But that changes things.",
                "But that's a different story entirely.",
            ]
        else:
            adj_pool = {w: f for w, f in self.neg_adj_pool.items() if w not in self.used_words}
            noun_pool = {w: f for w, f in self.noun_pool.items()
                         if w not in self.used_words and WORD_FORCES.get(w, (0,))[0] < 0}
            adj = self.select_word(
                input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
                adj_pool if adj_pool else None
            )
            noun = self.select_word(
                input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
                noun_pool if noun_pool else None
            )
            structures = [
                f"But that part is {adj}." if adj else None,
                f"Though that's genuinely {adj}." if adj else None,
                f"But I hear the {noun} in that too." if noun else None,
                "But that part is hard.",
                "But I hear the hard part too.",
            ]

        result = self._pick_structure(structures)
        return result if result else ("But that changes things." if is_positive_reversal else "But that part is hard.")

    def build_chunk_response(self, input_vadug, response_vadug, grade_rules=None,
                              is_first=True, is_reversal=False,
                              is_subsequent_negative=False,
                              is_positive=False, chunk_text=""):
        """Build a complete response for one chunk using WORD_FORCES math."""
        blocked = grade_rules.get("blocked", []) if grade_rules else []
        grade = grade_rules.get("grade", "C") if grade_rules else "C"

        # F-range crisis override: presence only, still math-selected
        if grade in ("F-", "F", "F+"):
            if grade == "F-":
                stab_pool = {w: WORD_FORCES[w] for w in self.STABILIZE_WORDS if w in WORD_FORCES}
                anchor = self.select_word(
                    response_vadug.v, response_vadug.a, response_vadug.d,
                    response_vadug.u, response_vadug.g, stab_pool
                )
                structures = [
                    f"I'm {anchor}." if anchor and anchor in ("here",) else None,
                    "I'm here.",
                    "I hear you.",
                    "You're not alone.",
                ]
                return self._pick_structure(structures) or "I'm here."
            else:
                ack = self.build_acknowledge(input_vadug, response_vadug)
                return ack if ack else "I hear you."

        # Reversal chunks
        if is_reversal:
            is_pos_reversal = input_vadug.v > 148
            return self.build_reversal_response(input_vadug, response_vadug, is_pos_reversal)

        # Positive chunks
        if is_positive and input_vadug.v > 150:
            if "positive_spin" in blocked or "ANY_positive_framing" in blocked:
                return ""
            if input_vadug.v > 190:
                return self.build_positive_acknowledge(input_vadug, response_vadug)
            return ""

        # Dead neutral: skip
        if 135 <= input_vadug.v <= 165 and is_subsequent_negative is False and not is_first:
            return ""

        parts = []

        # Acknowledgment
        if input_vadug.v < 135 or input_vadug.v > 165:
            if is_subsequent_negative:
                # Shorter acknowledgment for follow-on negative chunks
                ack = self.build_acknowledge(input_vadug, response_vadug)
                return ack if ack else "I hear that."
            else:
                ack = self.build_acknowledge(input_vadug, response_vadug)
                if ack:
                    parts.append(ack)

        # Stabilize (for first negative chunk or if dominance is low)
        if is_first and input_vadug.d < 110 and input_vadug.v < 135:
            if "unsolicited_advice" in blocked:
                stab = self._pick_structure([
                    "I'm right here.",
                    "I'm with you.",
                ])
            else:
                stab = self.build_stabilize(response_vadug)
            if stab:
                parts.append(stab)

        # Redirect (only if grade allows)
        if not is_first and not is_reversal and input_vadug.v >= 100:
            red = self.build_redirect(response_vadug, grade_rules)
            if red:
                parts.append(red)

        return ' '.join(parts) if parts else "I hear you."

    def build_full_response(self, chunk_results, arc, grade, grade_rules, personality,
                             verbose=False):
        """Build the complete assembled response for all chunks."""
        responses = []
        seen_negative = False

        for i, chunk in enumerate(chunk_results):
            is_first = (i == 0)
            input_vadug = chunk['vadug']
            response_vadug = compute_harmony(input_vadug, personality)
            response_vadug, _ = apply_personality(response_vadug, input_vadug, personality)

            is_negative = input_vadug.v < 135
            chunk_lower = chunk['text'].lower()

            # Content-based negativity check
            negative_content_words = {"sick", "broke", "broken", "died", "lost",
                                       "hurt", "failed", "crash", "fire", "rent",
                                       "raising", "can't take", "much more",
                                       "don't know", "struggle", "pain"}
            has_negative_content = any(w in chunk_lower for w in negative_content_words)
            if has_negative_content and input_vadug.v < 155:
                is_negative = True

            is_reversal = False
            first_word = chunk['text'].split()[0].lower().strip('.,!?;:') if chunk['text'].split() else ""
            if first_word in ChunkSplitter.REVERSAL_WORDS:
                is_reversal = True

            is_positive = input_vadug.v > 150

            resp = self.build_chunk_response(
                input_vadug,
                response_vadug,
                grade_rules,
                is_first=(is_first and not seen_negative) if is_negative else is_first,
                is_reversal=is_reversal,
                is_subsequent_negative=(is_negative and seen_negative),
                is_positive=is_positive,
                chunk_text=chunk['text'],
            )

            if is_negative:
                seen_negative = True

            if resp and resp.strip() and resp not in responses:
                responses.append(resp)
                if verbose:
                    words_used = sorted(self.used_words)
                    print(f"  Chunk {i+1} -> \"{resp}\"")
                    print(f"    Words selected: {', '.join(words_used[-4:])}")
            elif verbose:
                print(f"  Chunk {i+1} -> (skipped, neutral/duplicate)")

        # Arc closer
        closer = self._build_arc_closer(arc, grade, grade_rules)
        if closer:
            responses.append(closer)
            if verbose:
                print(f"  Arc closer ({arc}) -> \"{closer}\"")

        # Assemble with transitions
        if len(responses) <= 1:
            return responses[0] if responses else "I hear you."

        assembled = responses[0]
        for i, r in enumerate(responses[1:], 1):
            r_lower = r.lower()
            if r_lower.startswith(("but ", "hold on", "now that", "though ")):
                assembled = assembled.rstrip('.!?') + ". " + r
            elif i == len(responses) - 1:
                assembled = assembled.rstrip('.!?') + ". " + r
            else:
                joined = r[0].lower() + r[1:] if r and r[0] != 'I' else r
                assembled = assembled.rstrip('.!?') + ", and " + joined

        return assembled

    def _build_arc_closer(self, arc, grade, grade_rules):
        """Build an arc-aware closing line."""
        if grade in ("F-", "F", "F+"):
            return "I'm here. You're not alone."

        if grade in ("D-", "D"):
            closers = {
                "valley": "But the hard part isn't all there is.",
                "peak": "That's heavy. One step at a time.",
                "descending": "Let's take this one thing at a time.",
                "ascending": "You're finding your way through this.",
                "flat_negative": "I'm here for all of it.",
                "mixed": "All of that matters.",
            }
            return closers.get(arc, "I'm here.")

        closers = {
            "valley": "Not everyone gets that kind of turning point.",
            "peak": "We'll work through the rough part.",
            "descending": "One thing at a time. We'll get there.",
            "ascending": "Things are moving in the right direction.",
            "flat_negative": "You're not carrying this alone.",
            "flat_positive": "That's incredible all around.",
            "mixed": "Life's complicated like that. All of it counts.",
        }
        return closers.get(arc, "")

    def reset_for_new_response(self):
        """Call between responses to reset per-response tracking."""
        self.used_words.clear()
        self.used_structures.clear()


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

        # 4. Generate per-chunk responses via ResponseBuilder (math-based)
        if verbose:
            print(f"\n--- STEP 2: Per-Chunk Harmony (ResponseBuilder) ---")

        builder = ResponseBuilder()
        builder_response = builder.build_full_response(
            chunk_results, arc, grade, grade_rules, personality,
            verbose=verbose
        )

        # If sarcasm detected at moderate+ confidence, override the assembled response
        if sarcasm_flag and sarcasm_confidence >= SarcasmDetector.MODERATE:
            sarcasm_closer = random.choice([
                "I can tell that's not really how you feel.",
                "I hear what you're saying, but I also hear what you're not saying.",
                "The words say fine, but the feeling doesn't.",
                "I'm picking up on the frustration underneath.",
                "You don't have to pretend it's okay.",
            ])
            builder_response = builder_response.rstrip('.!?') + ". " + sarcasm_closer

        # Fallback: if ResponseBuilder produced empty/trivial, use template system
        if not builder_response or builder_response.strip() in ("", "I hear you."):
            if verbose:
                print(f"  (ResponseBuilder produced minimal output, falling back to templates)")
            responses = []
            seen_negative = False
            for i, cr in enumerate(chunk_results):
                response_vadug = compute_harmony(cr['vadug'], personality)
                response_vadug, _ = apply_personality(response_vadug, cr['vadug'], personality)

                is_negative = cr['vadug'].v < 135
                is_reversal = self._is_reversal_chunk(cr['text'])
                is_last = (i == len(chunk_results) - 1)

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
                        print(f"  Chunk {i+1} response (template): \"{response_text}\"")

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

            assembled = self.assemble(responses, closer, arc, chunk_results)
        else:
            assembled = builder_response

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
        """Select an arc-appropriate closing line, filtered by grade guardrails.

        Grade rules override the arc closer when certain strategies are blocked:
        - F-range: presence-only closers regardless of arc
        - D-range: empathy closers, no silver-lining or positive framing
        - "silver_lining" blocked: filter out optimistic closers
        """
        blocked = grade_rules.get("blocked", []) if grade_rules else []
        grade = grade_rules.get("grade", "C") if grade_rules else "C"

        # F-range: override to presence-only closers
        if grade in ("F-", "F", "F+"):
            if grade == "F-":
                return random.choice([
                    "I'm here.",
                    "You're not alone right now.",
                ])
            return random.choice([
                "You're not alone in this.",
                "I'm here. Whatever you need.",
                "I hear you.",
            ])

        # D-range: empathy closers, block any positive/silver-lining arc closers
        if grade in ("D-", "D", "D+"):
            return random.choice([
                "You're not alone in this.",
                "I'm here if you need to talk through it.",
                "That's a lot. I hear you.",
                "I'm here. What do you need right now?",
            ])

        # Normal arc-based closers with filtering
        closers = list(ARC_CLOSERS.get(arc, ARC_CLOSERS["mixed"]))

        # Filter out closers that violate blocked strategies
        if "silver_lining" in blocked or "positive_spin" in blocked:
            # Remove closers with positive/silver-lining language
            positive_words = {"silver lining", "amazing", "clicking", "looking up",
                              "congrats", "incredible", "momentum", "good stuff"}
            closers = [c for c in closers
                       if not any(pw in c.lower() for pw in positive_words)]

        # If all closers got filtered, fall back to neutral
        if not closers:
            closers = [
                "I hear you.",
                "That's a lot to hold.",
                "I'm here for all of it.",
            ]

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

    # Step 6: Decode via ResponseBuilder (math-based word selection)
    builder = ResponseBuilder()
    response = builder.build_chunk_response(
        input_vadu, response_vadu, grade_rules,
        is_first=True, is_reversal=False,
        is_subsequent_negative=False,
        is_positive=(input_vadu.v > 150),
        chunk_text=text,
    )

    # Fallback to template system if ResponseBuilder produced empty/trivial
    if not response or response.strip() == "":
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

    # Step 6.5: Grade guardrail — override response if grade demands it
    grade_override = None
    blocked = grade_rules.get("blocked", [])
    if grade in ("F-", "F", "F+"):
        # Crisis territory: presence only
        if grade == "F-":
            grade_override = random.choice([
                "I'm here.",
                "I hear you.",
                "You're not alone.",
            ])
        else:
            grade_override = random.choice([
                "I hear you. That's real pain.",
                "I'm here with you.",
                "You don't have to carry this alone.",
            ])
    elif grade in ("D-", "D", "D+"):
        # Check if response contains blocked strategies
        response_lower = response.lower()
        has_blocked_content = False
        if "silver_lining" in blocked or "positive_spin" in blocked:
            positive_markers = ["bright side", "at least", "silver lining",
                                "could be worse", "cheer up", "look on the",
                                "everything happens"]
            if any(m in response_lower for m in positive_markers):
                has_blocked_content = True
        if has_blocked_content:
            grade_override = random.choice([
                "I hear you. That's not easy.",
                "That sounds really heavy. I'm here.",
                "I'm sorry you're going through that.",
            ])

    if grade_override:
        response = grade_override

    if verbose:
        print(f"\n--- STEP 6: Decoded Response (ResponseBuilder) ---")
        if sarcasm_flag and sarcasm_confidence >= SarcasmDetector.MODERATE:
            print(f"  (sarcasm override — addressing real emotion)")
        if grade_override:
            print(f"  (grade {grade} guardrail — locked to {grade_rules.get('tone', '?')})")
        print(f"  \"{response}\"")
        print(f"\n{'='*60}")

    return response


def main():
    print("""
  +===================================================+
  |     CLANKER PIPELINE SIMULATOR v0.9                |
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
