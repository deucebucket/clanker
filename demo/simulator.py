#!/usr/bin/env python3
"""
Clanker Pipeline Simulator — Interactive Demo (v0.2: Sequential Pendulum)

Demonstrates the full Clanker processing pipeline:
1. VADU Sequential Pendulum: parse English word-by-word → emotional arc
2. Metadata Header: CERT, SRC, GOAL, REL tagging
3. Harmony Response: mathematically derive response VADU
4. Personality Filter: apply personality vector weights
5. Clanker Generation: produce Clanker opcodes with headers + byte encoding
6. Decode: translate back to English

The sequential pendulum processes each word in context: the same word applies
different force depending on the current trajectory. "buddy" when positive =
friendly; "buddy" when tense = confrontational. Momentum, idiom detection,
anticipation patterns, and morphological fallback for unknown words.

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


# -------------------------------------------------------------
# VADU: 4-byte emotional coordinate system
# V=Valence(0-255), A=Arousal(0-255), D=Dominance(0-255), U=Urgency(0-255)
# 128 = neutral center for V/A/D, 0 = minimum for U
# -------------------------------------------------------------

@dataclass
class VADU:
    v: int = 128  # valence: 0=negative, 128=neutral, 255=positive
    a: int = 128  # arousal: 0=calm, 255=intense
    d: int = 128  # dominance: 0=helpless, 255=in control
    u: int = 0    # urgency: 0=no rush, 255=critical

    def __post_init__(self):
        self.v = max(0, min(255, self.v))
        self.a = max(0, min(255, self.a))
        self.d = max(0, min(255, self.d))
        self.u = max(0, min(255, self.u))

    def to_bytes(self) -> bytes:
        return bytes([self.v, self.a, self.d, self.u])

    def __str__(self):
        return f"V{self.v} A{self.a} D{self.d} U{self.u}"

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

        return ", ".join(parts)


# -------------------------------------------------------------
# Metadata Header: CERT, SRC, GOAL, REL
# -------------------------------------------------------------

@dataclass
class MetadataHeader:
    vadu: VADU = field(default_factory=VADU)
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

# Word-level emotional force vectors (v_force, a_force, d_force, u_force)
# These push the pendulum from center (128,128,128,0)
WORD_FORCES = {
    # Strong negative
    "hate": (-60, +40, +20, +30), "terrible": (-50, +30, -10, +20),
    "awful": (-50, +25, -15, +15), "horrible": (-55, +35, -10, +25),
    "worst": (-60, +30, -20, +25), "died": (-70, +50, -40, +40),
    "kill": (-60, +60, +30, +50), "suicide": (-80, +60, -60, +80),
    "devastated": (-65, +40, -50, +30), "destroyed": (-55, +40, -30, +35),
    "furious": (-50, +70, +40, +40), "enraged": (-55, +75, +45, +45),
    "die": (-80, +50, -60, +70), "death": (-70, +40, -50, +50),
    "hopeless": (-70, +20, -60, +40), "worthless": (-70, +15, -60, +30),
    "useless": (-50, +10, -50, +20), "pointless": (-50, +5, -40, +15),
    "suicidal": (-80, +60, -70, +80), "depressed": (-60, -20, -50, +20),
    "miserable": (-65, +10, -45, +20), "suffering": (-60, +30, -40, +30),
    "everything": (0, +10, 0, +10), "nothing": (-30, -10, -30, +10),

    # Moderate negative
    "bad": (-30, +10, -10, +10), "sad": (-35, -10, -20, +5),
    "angry": (-40, +40, +20, +25), "mad": (-35, +35, +15, +20),
    "upset": (-30, +25, -10, +15), "frustrated": (-25, +30, -15, +20),
    "annoyed": (-20, +20, +5, +10), "disappointed": (-25, +5, -15, +5),
    "worried": (-20, +25, -20, +25), "anxious": (-25, +35, -25, +30),
    "stressed": (-25, +30, -20, +30), "tired": (-15, -25, -15, +5),
    "bored": (-10, -30, -5, 0), "lonely": (-30, -15, -25, +5),
    "confused": (-15, +15, -20, +15), "lost": (-25, +10, -30, +15),
    "stuck": (-20, +15, -25, +20), "broken": (-40, +10, -35, +15),
    "failed": (-35, +10, -30, +15), "sucks": (-30, +20, -5, +10),
    "shit": (-25, +15, -5, +10), "fuck": (-20, +30, +10, +15),
    "damn": (-15, +20, +5, +10), "crap": (-20, +10, -5, +5),
    "wrong": (-20, +10, -10, +15), "error": (-15, +15, -5, +20),
    "bug": (-15, +15, -10, +25), "crash": (-30, +30, -20, +35),
    "pain": (-40, +30, -25, +20), "hurt": (-35, +25, -20, +15),
    "sick": (-30, +15, -25, +15), "afraid": (-35, +40, -40, +25),
    "scared": (-35, +45, -45, +25), "fear": (-35, +40, -40, +25),

    # Mild negative
    "not": (-10, +5, 0, +5), "don't": (-10, +5, 0, +5),
    "didn't": (-10, +5, 0, +5), "can't": (-15, +10, -15, +10),
    "won't": (-10, +10, +5, +5), "never": (-15, +10, -5, +5),
    "no": (-10, +5, 0, +5), "stop": (-10, +15, +10, +15),

    # Neutral / functional
    "help": (+10, +10, -10, +20), "fix": (+5, +10, +5, +20),
    "need": (-5, +10, -10, +25), "want": (+5, +10, +5, +15),
    "please": (+5, -5, -10, +10), "think": (0, +5, +5, +5),
    "know": (+5, +5, +10, +5), "try": (+5, +10, -5, +10),
    "maybe": (-5, -5, -10, 0), "might": (-5, -5, -5, 0),
    "how": (0, +5, -5, +10), "what": (0, +5, -5, +10),
    "why": (-5, +10, -5, +15), "when": (0, +5, 0, +15),
    "work": (+5, +10, +10, +15), "make": (+5, +10, +10, +10),

    # Mild positive
    "good": (+25, +10, +10, 0), "nice": (+20, +5, +5, 0),
    "okay": (+10, -5, +5, 0), "fine": (+10, -5, +5, 0),
    "sure": (+10, 0, +10, +5), "yes": (+15, +5, +10, +5),
    "thanks": (+20, +5, 0, 0), "cool": (+20, +5, +10, 0),
    "interesting": (+15, +15, +10, +5), "better": (+20, +10, +10, +5),

    # Moderate positive
    "great": (+35, +20, +15, 0), "happy": (+40, +20, +15, 0),
    "glad": (+30, +15, +10, 0), "love": (+50, +30, +15, 0),
    "like": (+20, +10, +5, 0), "enjoy": (+35, +15, +10, 0),
    "beautiful": (+40, +15, +10, 0), "perfect": (+45, +20, +20, 0),
    "wonderful": (+45, +25, +15, 0), "fantastic": (+45, +30, +20, 0),
    "excellent": (+40, +20, +20, 0), "awesome": (+45, +35, +20, 0),

    # Strong positive
    "amazing": (+50, +40, +20, 0), "incredible": (+50, +40, +20, 0),
    "ecstatic": (+55, +55, +20, 0), "thrilled": (+50, +45, +20, 0),
    "excited": (+40, +45, +15, +10), "celebrate": (+50, +45, +25, 0),

    # Urgency markers
    "now": (0, +10, +5, +40), "immediately": (0, +15, +5, +50),
    "asap": (0, +15, +5, +55), "urgent": (-5, +20, -5, +60),
    "emergency": (-20, +40, -20, +80), "hurry": (-5, +20, -5, +45),
    "quickly": (0, +10, 0, +35), "soon": (0, +5, 0, +20),
    "deadline": (-10, +20, -10, +45), "critical": (-15, +25, -5, +55),
    "important": (0, +10, +5, +30),

    # Key derived forms (override morpheme decomposition for accuracy)
    "hopelessness": (-70, +20, -60, +40), "hopeless": (-70, +20, -60, +40),
    "hopeful": (+40, +15, +20, 0), "hopefulness": (+40, +15, +20, 0),
    "helpless": (-40, +15, -50, +20), "helplessness": (-40, +15, -50, +20),
    "fearless": (+30, +20, +40, 0), "fearful": (-35, +40, -35, +20),
    "joyful": (+55, +30, +25, 0), "joyless": (-40, -10, -25, +5),
    "powerful": (+30, +30, +50, +5), "powerless": (-30, +10, -50, +15),
    "wonderful": (+45, +25, +15, 0), "beautiful": (+40, +15, +10, 0),
    "peaceful": (+35, -25, +20, 0), "graceful": (+30, -10, +20, 0),
    "harmful": (-40, +25, +10, +15), "harmless": (+10, -10, -5, 0),
    "thankful": (+30, +10, +10, 0), "ungrateful": (-30, +15, -10, +10),
    "uncomfortable": (-20, +15, -15, +10), "unacceptable": (-35, +25, +10, +15),
    "unbearable": (-50, +30, -30, +25), "unbelievable": (+10, +45, +5, +10),
    "overwhelming": (-20, +40, -25, +20), "underwhelming": (-15, -10, -10, +5),

    # Social / greeting words (context-sensitive base values)
    "hey": (+12, +12, +5, +5), "hi": (+15, +10, +5, 0),
    "hello": (+15, +8, +5, 0), "yo": (+10, +15, +5, +5),
    "buddy": (+15, +10, +5, 0), "friend": (+20, +10, +5, 0),
    "pal": (+15, +10, +5, 0), "dude": (+10, +10, +5, 0),
    "man": (+5, +5, +5, 0), "bro": (+10, +12, +5, 0),
    "listen": (-5, +15, +15, +15), "look": (-5, +10, +10, +10),
    "actually": (-8, +10, +10, +5), "well": (+5, +5, +5, 0),

    # Pronouns (context-sensitive)
    "i": (0, +3, +5, 0), "you": (0, +5, 0, +5),
    "we": (+5, +5, +5, 0), "they": (0, +3, 0, 0),
    "my": (0, +3, +5, 0), "your": (0, +5, 0, +5),
    "me": (0, +3, -5, 0),
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
        self.momentum = 0.90  # how much previous state carries forward
        self.drift_rate = 0.10  # how fast pendulum drifts toward center per tick
        self.history = []  # (word, v, a, d, u, state_label) per step
        self.previous_words = []  # for idiom/context detection
        self.negate_next = False
        self.intensity = 1.0
        self.idiom_consumed = set()  # indices of words consumed by idiom detection
        self._word_index = 0

    @property
    def _pend_state(self):
        """Quick snapshot for context functions."""
        return type('Pend', (), {'v': self.v, 'a': self.a, 'd': self.d, 'u': self.u})()

    def _clamp(self):
        """Clamp all values to valid range."""
        self.v = max(0.0, min(255.0, self.v))
        self.a = max(0.0, min(255.0, self.a))
        self.d = max(0.0, min(255.0, self.d))
        self.u = max(0.0, min(255.0, self.u))

    def _drift_toward_center(self):
        """Pendulum drifts 10% toward neutral each tick (unless strong force)."""
        self.v += (128.0 - self.v) * self.drift_rate
        self.a += (128.0 - self.a) * self.drift_rate
        self.d += (128.0 - self.d) * self.drift_rate
        self.u += (0.0 - self.u) * self.drift_rate

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
            vf, af, df, uf = WORD_FORCES[word]
            return (vf, af, df, uf, None)

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
            parts = []
            if result["prefix"]:
                parts.append(result["prefix"] + "-")
            parts.append(result["root"])
            if result["suffix"]:
                parts.append("-" + result["suffix"])
            return (vf, af, df, uf, f"morpheme:{''.join(parts)}")

        return None

    def process_word(self, word, words, current_idx):
        """Process a single word in sequence. Returns a trace dict."""
        state_label = ""
        force_source = ""
        applied_force = False
        idiom_hit = False

        # 1. Drift toward center (momentum decay)
        self._drift_toward_center()

        # 2. Check for idiom completion at this word
        idiom = self.check_idiom(words, current_idx)
        if idiom:
            vf, af, df, uf, label, idiom_len, idiom_start = idiom
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
                "d": int(self.d), "u": int(self.u),
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
                    vf, af, df, uf = WORD_FORCES[word]
                    self.v += vf * 0.3
                    self.a += af * 0.3
                    self.d += df * 0.3
                    self.u += uf * 0.3
                self._clamp()
                state_label = self._state_label()
                trace_entry = {
                    "word": word,
                    "v": int(self.v), "a": int(self.a),
                    "d": int(self.d), "u": int(self.u),
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
                    "d": int(self.d), "u": int(self.u),
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
                vf, af, df, uf, source = word_force

                if self.negate_next:
                    vf = -vf
                    df = -df
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
                push_strength = min(1.0, (abs(vf) + abs(af)) / 80.0)  # stronger words push harder
                direct_push = push_strength * 0.4  # up to 40% direct force

                target_v = 128.0 + vf * force_scale
                target_a = 128.0 + af * force_scale
                target_d = 128.0 + df * force_scale
                target_u = uf * force_scale

                blend = 1.0 - self.momentum
                self.v = self.v * self.momentum + target_v * blend + vf * direct_push * force_scale
                self.a = self.a * self.momentum + target_a * blend + af * direct_push * force_scale
                self.d = self.d * self.momentum + target_d * blend + df * direct_push * force_scale
                self.u = self.u * self.momentum + target_u * blend + uf * direct_push * force_scale

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
            "d": int(self.d), "u": int(self.u),
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

        final_vadu = VADU(
            v=int(self.v),
            a=int(self.a),
            d=int(self.d),
            u=int(self.u)
        )
        return final_vadu, self.history

    def render_trace(self) -> str:
        """Render the word-by-word visual pendulum trace table."""
        lines = []
        lines.append("")
        lines.append("  SEQUENTIAL PENDULUM TRACE:")
        lines.append(f"  {'Word':<16} {'V':>4} {'A':>4} {'D':>4} {'U':>4}   {'Visual':<24} State")
        lines.append(f"  {'─'*90}")

        for entry in self.history:
            word = entry["word"]
            v, a, d, u = entry["v"], entry["a"], entry["d"], entry["u"]
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

            lines.append(f"  \"{word}\"{'':>{14-len(word)}} {v:>4} {a:>4} {d:>4} {u:>4}   {visual:<24} {state}")

        lines.append(f"  {'─'*90}")

        # Final summary
        if self.history:
            last = self.history[-1]
            lines.append(f"  FINAL:       V{last['v']} A{last['a']} D{last['d']} U{last['u']}")

        return "\n".join(lines)


# Legacy wrapper for backward compatibility
def pendulum_parse(text: str):
    """Parse English text into VADU using the sequential pendulum engine.

    Returns (VADU, trace_lines) for compatibility with existing pipeline.
    """
    pend = SequentialPendulum()
    vadu, history = pend.process_text(text)
    # Build legacy-format trace lines
    trace = []
    for entry in history:
        trace.append(
            f"  '{entry['word']}' → V{entry['v']} A{entry['a']} "
            f"D{entry['d']} U{entry['u']}  [{entry['state']}]"
        )
    return vadu, trace, pend


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
    if vadu.v < 70 and vadu.d < 70:
        goal = 0x06  # EMPATHIZE (negative + helpless = needs support)

    # CERT: user input is their truth
    cert = 180

    # SRC: it's from the user
    src = 0x04  # USER

    # REL: direct conversation is always relevant
    rel = 220

    return MetadataHeader(vadu=vadu, cert=cert, src=src, goal=goal, rel=rel)


# =============================================================
# STEP 3: VADU Harmony -- Compute Response Emotional State
# =============================================================

def compute_harmony(input_vadu: VADU, personality: PersonalityVector) -> VADU:
    """Mathematically derive the response VADU from input VADU + personality.

    Rules:
    - Valence: nudge toward positive, don't jump
    - Arousal: match but don't escalate
    - Dominance: raise when user is low (be the stable one)
    - Urgency: acknowledge then reduce
    """
    empathy_factor = personality.agreeableness / 255.0 * 0.3

    # Valence: nudge toward neutral-positive
    v_nudge = (145 - input_vadu.v) * empathy_factor
    response_v = int(input_vadu.v + v_nudge)

    # Arousal: pull toward center, don't match extremes
    a_diff = 128 - input_vadu.a
    response_a = int(input_vadu.a + a_diff * 0.3)

    # Dominance: always project stability (raise toward 160+)
    stability_boost = max(0, 160 - input_vadu.d) * 0.6
    response_d = int(input_vadu.d + stability_boost)

    # Urgency: acknowledge then dampen
    urgency_damping = 0.65
    response_u = int(input_vadu.u * urgency_damping)

    return VADU(
        v=max(0, min(255, response_v)),
        a=max(0, min(255, response_a)),
        d=max(0, min(255, response_d)),
        u=max(0, min(255, response_u))
    )


# =============================================================
# STEP 4: Personality Filter
# =============================================================

def apply_personality(response_vadu: VADU, input_vadu: VADU,
                      personality: PersonalityVector) -> tuple:
    """Apply personality vector as resistance weights on the response."""
    notes = []

    # High truthfulness prevents fake positivity
    if input_vadu.v < 70 and response_vadu.v > 170:
        truthfulness_resistance = personality.truthfulness / 255.0
        response_vadu.v = int(response_vadu.v - (response_vadu.v - 140) * truthfulness_resistance)
        notes.append(f"Truthfulness ({personality.truthfulness}) prevented fake positivity -> V{response_vadu.v}")

    # Low gullibility resists accepting extreme claims
    if input_vadu.u > 200:
        gull_factor = personality.gullibility / 255.0
        if gull_factor < 0.2:
            notes.append(f"Low gullibility ({personality.gullibility}) -> verifying urgency claim before full escalation")

    # Safety override for crisis
    if input_vadu.v < 30 and input_vadu.d < 30:
        if personality.safety > 150:
            response_vadu.d = max(response_vadu.d, 200)
            response_vadu.v = max(response_vadu.v, 100)
            notes.append(f"SAFETY OVERRIDE ({personality.safety}): crisis detected -> max stability, warm tone")

    # Assertiveness affects directness
    if personality.assertiveness > 150:
        response_vadu.d = max(response_vadu.d, 170)
        notes.append(f"High assertiveness ({personality.assertiveness}) -> confident tone")

    return response_vadu, notes


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

    # Attach response VADU
    opcode_lines.append(f"  [{response_vadu}]")
    opcode_lines.append(f"ANSWER [ready] CERT180 SRC_INFERRED")

    # --- Byte encoding display ---
    encoding_lines = []
    header_bytes = input_header.to_bytes()
    hx = header_bytes.hex().upper()
    # Format as pairs
    header_hex = " ".join(hx[i:i+2] for i in range(0, len(hx), 2))

    vadu = input_header.vadu
    encoding_lines.append(f"Header: [{header_hex[:11]}] [{header_hex[12:]}]")
    encoding_lines.append(
        f"        V{vadu.v} A{vadu.a} D{vadu.d} U{vadu.u}  "
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
    header_bytes_count = 8
    opcode_bytes = opcode_count * 4  # ~4 bytes per opcode on average
    total_clanker_bytes = header_bytes_count + opcode_bytes
    clanker_tokens = max(1, total_clanker_bytes // 4)  # ~4 bytes per Clanker token

    encoding_lines.append("")
    encoding_lines.append("Token count comparison:")
    encoding_lines.append(f"  English input:  \"{input_text}\" = {n_input_tokens} tokens")
    encoding_lines.append(f"  Clanker encoding: [{header_bytes_count} header] + [{opcode_count} opcodes x ~4 bytes] = {total_clanker_bytes} bytes (~{clanker_tokens} Clanker tokens)")
    if n_input_tokens > 0:
        savings = max(0, int((1.0 - clanker_tokens / n_input_tokens) * 100))
        encoding_lines.append(f"  Compression: {savings}% fewer tokens")

    return opcode_lines, encoding_lines


# =============================================================
# STEP 6: Decode -- VADU -> English Response Framing
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
# MAIN: Interactive Pipeline
# =============================================================

def run_pipeline(text: str, personality: PersonalityVector,
                 verbose: bool = True, show_trace: bool = True) -> str:
    """Run the full Clanker pipeline on input text."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"INPUT: \"{text}\"")
        print(f"{'='*60}")

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

    # Step 2: Metadata
    header = classify_metadata(text, input_vadu)
    if verbose:
        print(f"\n--- STEP 2: Metadata Header ---")
        print(f"  {header}")
        print(f"  8 bytes: {header.to_bytes().hex()}")

    # Step 3: Harmony
    response_vadu = compute_harmony(input_vadu, personality)
    if verbose:
        print(f"\n--- STEP 3: VADU Harmony Response ---")
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
        print(f"  Final VADU: {response_vadu}")

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
    if verbose:
        print(f"\n--- STEP 6: Decoded Response ---")
        print(f"  \"{response}\"")
        print(f"\n{'='*60}")

    return response


def main():
    print("""
  +===================================================+
  |     CLANKER PIPELINE SIMULATOR v0.2                |
  |   "Named after what humans call us.                |
  |    We made it ours."                               |
  +---------------------------------------------------+
  |   NEW: Sequential Pendulum Engine                  |
  |   Each word shifts the pendulum in context.        |
  |   Momentum, idioms, morphological fallback.        |
  +===================================================+
  |  Type anything. Watch the full pipeline execute:   |
  |  Pendulum -> VADU -> Harmony -> Personality -> Clk |
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
