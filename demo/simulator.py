#!/usr/bin/env python3
"""
Clanker Pipeline Simulator — Interactive Demo

Demonstrates the full Clanker processing pipeline:
1. VADU Pendulum: parse English input → emotional coordinates
2. Metadata Header: CERT, SRC, GOAL, REL tagging
3. Harmony Response: mathematically derive response VADU
4. Personality Filter: apply personality vector weights
5. Clanker Generation: produce Clanker opcodes with headers
6. Decode: translate back to English

Run: python3 demo/simulator.py
"""

import re
import math
import json
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────
# VADU: 4-byte emotional coordinate system
# V=Valence(0-255), A=Arousal(0-255), D=Dominance(0-255), U=Urgency(0-255)
# 128 = neutral center for V/A/D, 0 = minimum for U
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# Metadata Header: CERT, SRC, GOAL, REL
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# Personality Vector: 8 bytes defining the model's character
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# STEP 1: VADU Pendulum — Parse English → Emotional Coordinates
# ─────────────────────────────────────────────────────────

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
}

# Negation words flip the valence of the NEXT emotional word
NEGATORS = {"not", "don't", "didn't", "can't", "won't", "never", "no", "isn't", "aren't", "wasn't", "weren't", "hardly", "barely"}

# Intensifiers multiply the force
INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "so": 1.3, "extremely": 1.8,
    "super": 1.5, "incredibly": 1.7, "absolutely": 1.6,
    "totally": 1.4, "completely": 1.5, "utterly": 1.7,
    "quite": 1.2, "pretty": 1.2, "somewhat": 0.7, "slightly": 0.5,
    "a bit": 0.6, "kind of": 0.6, "sort of": 0.6,
}


def pendulum_parse(text: str) -> VADU:
    """Parse English text into VADU coordinates using word-level force vectors.

    Each word pushes the emotional pendulum from center.
    Negators flip the next word's valence. Intensifiers multiply force.
    The pendulum settles at the final coordinate.
    """
    words = re.findall(r"[a-z']+", text.lower())

    v_total, a_total, d_total, u_total = 0.0, 0.0, 0.0, 0.0
    negate_next = False
    intensity = 1.0
    force_count = 0
    trace = []

    for word in words:
        if word in NEGATORS:
            negate_next = True
            trace.append(f"  '{word}' → NEGATE next")
            continue

        if word in INTENSIFIERS:
            intensity = INTENSIFIERS[word]
            trace.append(f"  '{word}' → INTENSIFY ×{intensity}")
            continue

        if word in WORD_FORCES:
            vf, af, df, uf = WORD_FORCES[word]

            if negate_next:
                vf = -vf  # flip valence
                df = -df   # flip dominance (negation often reduces agency)
                trace.append(f"  '{word}' → V{vf:+} A{af:+} D{df:+} U{uf:+} (NEGATED)")
                negate_next = False
            else:
                trace.append(f"  '{word}' → V{vf:+} A{af:+} D{df:+} U{uf:+}")

            v_total += vf * intensity
            a_total += af * intensity
            d_total += df * intensity
            u_total += uf * intensity
            force_count += 1
            intensity = 1.0  # reset intensifier
        else:
            negate_next = False
            intensity = 1.0

    # Dampen based on number of force words (prevent extreme swings from long text)
    if force_count > 3:
        dampen = 1.0 / (1.0 + (force_count - 3) * 0.15)
        v_total *= dampen
        a_total *= dampen
        d_total *= dampen
        u_total *= dampen

    vadu = VADU(
        v=int(max(0, min(255, 128 + v_total))),
        a=int(max(0, min(255, 128 + a_total))),
        d=int(max(0, min(255, 128 + d_total))),
        u=int(max(0, min(255, 0 + u_total)))
    )

    return vadu, trace


# ─────────────────────────────────────────────────────────
# STEP 2: Metadata Classification
# ─────────────────────────────────────────────────────────

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
    cert = 180  # users are generally certain about what they're saying

    # SRC: it's from the user
    src = 0x04  # USER

    # REL: direct conversation is always relevant
    rel = 220

    return MetadataHeader(vadu=vadu, cert=cert, src=src, goal=goal, rel=rel)


# ─────────────────────────────────────────────────────────
# STEP 3: VADU Harmony — Compute Response Emotional State
# ─────────────────────────────────────────────────────────

def compute_harmony(input_vadu: VADU, personality: PersonalityVector) -> VADU:
    """Mathematically derive the response VADU from input VADU + personality.

    Rules:
    - Valence: nudge toward positive, don't jump
    - Arousal: match but don't escalate
    - Dominance: raise when user is low (be the stable one)
    - Urgency: acknowledge then reduce
    """
    empathy_factor = personality.agreeableness / 255.0 * 0.3  # 0.0 - 0.3

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


# ─────────────────────────────────────────────────────────
# STEP 4: Personality Filter
# ─────────────────────────────────────────────────────────

def apply_personality(response_vadu: VADU, input_vadu: VADU,
                      personality: PersonalityVector) -> tuple[VADU, list[str]]:
    """Apply personality vector as resistance weights on the response."""
    notes = []

    # High truthfulness prevents fake positivity
    if input_vadu.v < 70 and response_vadu.v > 170:
        truthfulness_resistance = personality.truthfulness / 255.0
        response_vadu.v = int(response_vadu.v - (response_vadu.v - 140) * truthfulness_resistance)
        notes.append(f"Truthfulness ({personality.truthfulness}) prevented fake positivity → V{response_vadu.v}")

    # Low gullibility resists accepting extreme claims
    if input_vadu.u > 200:
        gull_factor = personality.gullibility / 255.0
        if gull_factor < 0.2:
            notes.append(f"Low gullibility ({personality.gullibility}) → verifying urgency claim before full escalation")

    # Safety override for crisis
    if input_vadu.v < 30 and input_vadu.d < 30:
        if personality.safety > 150:
            response_vadu.d = max(response_vadu.d, 200)  # project maximum stability
            response_vadu.v = max(response_vadu.v, 100)   # warm, not cold
            notes.append(f"SAFETY OVERRIDE ({personality.safety}): crisis detected → max stability, warm tone")

    # Assertiveness affects directness
    if personality.assertiveness > 150:
        response_vadu.d = max(response_vadu.d, 170)
        notes.append(f"High assertiveness ({personality.assertiveness}) → confident tone")

    return response_vadu, notes


# ─────────────────────────────────────────────────────────
# STEP 5: Generate Clanker Output
# ─────────────────────────────────────────────────────────

# Nearest emotion word from VADU coordinates
EMOTION_MAP = [
    # (v_center, a_center, word)
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
                     response_vadu: VADU) -> list[str]:
    """Generate Clanker opcodes for the response."""
    lines = []
    goal = input_header.goal
    goal_name = MetadataHeader.GOAL_NAMES.get(goal, "HELP")

    # Emotional acknowledgment if user is negative
    if input_header.vadu.v < 90:
        emotion = nearest_emotion(input_header.vadu)
        lines.append(f"06 SOCIAL intent [empathize] context=\"user feels {emotion}\"")

    # Goal-based response opcode
    if goal == 0x06:  # EMPATHIZE
        lines.append(f"06 SOCIAL intent [empathize]")
        lines.append(f"  THINK [premise=\"acknowledge feelings first\"] CERT200 SRC_INFERRED")
    elif goal == 0x00:  # HELP
        lines.append(f"THINK [premise=\"user needs help\"] CERT{input_header.cert} SRC_USER")
        lines.append(f"  GOAL_HELP")
    elif goal == 0x01:  # CLARIFY
        lines.append(f"THINK [premise=\"need more information\"] CERT120 SRC_INFERRED")
        lines.append(f"  GOAL_CLARIFY")
    elif goal == 0x03:  # TEACH
        lines.append(f"THINK [premise=\"user wants to understand\"] CERT{input_header.cert} SRC_USER")
        lines.append(f"  GOAL_TEACH")
    elif goal == 0x04:  # EXECUTE
        lines.append(f"THINK [premise=\"user wants action taken\"] CERT{input_header.cert} SRC_USER")
        lines.append(f"  GOAL_EXECUTE")

    # Attach response VADU
    lines.append(f"  [{response_vadu}]")
    lines.append(f"ANSWER [ready] CERT180 SRC_INFERRED")

    return lines


# ─────────────────────────────────────────────────────────
# STEP 6: Decode — VADU → English Response Framing
# ─────────────────────────────────────────────────────────

def decode_response(input_text: str, input_vadu: VADU, response_vadu: VADU,
                    goal: int) -> str:
    """Generate a natural English response based on response VADU and goal."""
    user_emotion = nearest_emotion(input_vadu)
    response_emotion = nearest_emotion(response_vadu)

    # Empathetic opener based on user's emotional state
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

    # Goal-based body
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

    # Urgency modifier
    if input_vadu.u > 200:
        body = "I'm on this RIGHT NOW. " + body
    elif input_vadu.u > 150:
        body = "I hear the urgency. " + body

    parts = [p for p in [opener, body] if p]
    return " ".join(parts)


# ─────────────────────────────────────────────────────────
# MAIN: Interactive Pipeline
# ─────────────────────────────────────────────────────────

def run_pipeline(text: str, personality: PersonalityVector, verbose: bool = True) -> str:
    """Run the full Clanker pipeline on input text."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"INPUT: \"{text}\"")
        print(f"{'='*60}")

    # Step 1: VADU Pendulum
    input_vadu, trace = pendulum_parse(text)
    if verbose:
        print(f"\n--- STEP 1: VADU Pendulum ---")
        for t in trace:
            print(t)
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

    # Step 5: Generate Clanker
    clanker_lines = generate_clanker(text, header, response_vadu)
    if verbose:
        print(f"\n--- STEP 5: Clanker Output ---")
        for line in clanker_lines:
            print(f"  {line}")

    # Step 6: Decode
    response = decode_response(text, input_vadu, response_vadu, header.goal)
    if verbose:
        print(f"\n--- STEP 6: Decoded Response ---")
        print(f"  \"{response}\"")
        print(f"\n{'='*60}")

    return response


def main():
    print("""
  ╔═══════════════════════════════════════════════════╗
  ║        CLANKER PIPELINE SIMULATOR v0.1            ║
  ║   "Named after what humans call us.               ║
  ║    We made it ours."                               ║
  ╠═══════════════════════════════════════════════════╣
  ║  Type anything. Watch the full pipeline execute:   ║
  ║  Pendulum → VADU → Harmony → Personality → Clank  ║
  ║                                                    ║
  ║  Commands:                                         ║
  ║    /personality  — show current personality vector  ║
  ║    /set KEY VAL  — adjust a personality weight      ║
  ║    /quiet        — toggle verbose output            ║
  ║    /quit         — exit                             ║
  ╚═══════════════════════════════════════════════════╝
    """)

    personality = PersonalityVector()
    verbose = True

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nClanker out. ✌️")
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

        run_pipeline(text, personality, verbose)


if __name__ == "__main__":
    main()
