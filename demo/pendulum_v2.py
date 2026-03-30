#!/usr/bin/env python3
"""PendulumV2 — Clean emotional physics engine for Clanker-Lang.

Rewrite of pendulum.py (~1,300 lines of layered patches) into a clean
3-pass architecture under 400 lines.

Emotional PEMDAS (order of operations):
  1. PRE-PASS:  Detect sentence-level features (question, idioms, negation)
  2. WORD-PASS: Left-to-right, classify each word as OPERATOR / PAYLOAD / NEUTRAL
  3. POST-PASS: Crisis detection, clamping, final adjustments

Key design rules:
  - Words are EITHER operators OR payloads OR neutral. Never both.
  - Operators ACCUMULATE until a payload word consumes them.
  - Uses forces_curated.py (2,024 words) not forces.py (46K with noise).
  - Context operators from context_operators.py integrated at the core.

Usage:
    from demo.pendulum_v2 import PendulumV2
    engine = PendulumV2()
    vadug, trace = engine.process_text("I am really not having a good day")

Standalone:
    python3 demo/pendulum_v2.py
"""

import sys
import os
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from demo.shared import VADUG
from demo.forces_curated import EMOTIONAL_VOCABULARY
from demo.fuzzy import fuzzy_match as _fuzzy_match
from demo.context_operators import (
    CONTEXT_OPERATORS, QUESTION_STARTERS, QUESTION_DAMPENER,
    is_question, _COEFF_CAP, _COEFF_FLOOR, _parse_operator,
)
try:
    from demo.idioms import IDIOMS
except ImportError:
    from demo.pendulum import IDIOMS  # fallback
from demo.bigrams import BIGRAM_EXPRESSIONS
from demo.sarcasm import SarcasmDetector
from demo.tonal import TonalAnalyzer, apply_tonal_adjustment
from demo.preflight import PreflightAnalyzer
from demo.sarcasm_templates import SarcasmTemplateDetector

_SARCASM_DETECTOR = SarcasmDetector()
_SARCASM_TEMPLATES = SarcasmTemplateDetector()
_TONAL_ANALYZER = TonalAnalyzer()
_PREFLIGHT = PreflightAnalyzer()

# Merge bigrams into idiom lookup — bigrams are just 2-word idioms.
# IDIOMS take priority for overlapping keys (they have gravity values).
_COMBINED_EXPRESSIONS = dict(BIGRAM_EXPRESSIONS)  # bigrams first
_COMBINED_EXPRESSIONS.update(IDIOMS)               # idioms override

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEGATORS = frozenset({
    "not", "don't", "didn't", "can't", "won't", "never", "no",
    "isn't", "aren't", "wasn't", "weren't", "hardly", "barely",
    "nobody", "nothing", "nowhere", "neither", "nor", "cannot",
    "nope", "nah", "ain't", "shouldn't", "couldn't", "wouldn't",
    "hasn't", "haven't", "hadn't", "doesn't",
})

# Negation as continuous force — not a boolean.
# Strong negators start near 1.0, weaker ones lower.
NEGATOR_STRENGTH = {
    "not": 0.95, "no": 0.90, "never": 0.95, "cannot": 0.90,
    "don't": 0.90, "didn't": 0.90, "can't": 0.90, "won't": 0.90,
    "isn't": 0.85, "aren't": 0.85, "wasn't": 0.85, "weren't": 0.85,
    "hardly": 0.60, "barely": 0.50,
    "nobody": 0.80, "nothing": 0.80, "nowhere": 0.80,
    "neither": 0.75, "nor": 0.70,
    "nope": 0.85, "nah": 0.75, "ain't": 0.85,
    "shouldn't": 0.80, "couldn't": 0.80, "wouldn't": 0.80,
    "hasn't": 0.85, "haven't": 0.85, "hadn't": 0.80, "doesn't": 0.85,
}

# How fast negation force decays per word type
NEGATION_DECAY_OPERATOR = 0.92   # gentle — negation passes through function words
NEGATION_DECAY_NEUTRAL = 0.85    # moderate — filler erodes negation
NEGATION_DECAY_PAYLOAD = 0.35    # hard — emotional word absorbs most of the negation

# Clause boundaries kill negation entirely
CLAUSE_BOUNDARIES = frozenset({
    "but", "however", "although", "though", "yet", "still",
    "instead", "whereas", "while", "nevertheless", "except",
    "regardless", "nonetheless", "despite", "otherwise", "meanwhile", "conversely",
})

# ---------------------------------------------------------------------------
# Evokers — gravitational priming words (Force #25)
# Words that don't express emotion but change the gravitational field.
# "I lost my job before the wedding" — "wedding" raises the stakes.
# ---------------------------------------------------------------------------

EVOKERS = {
    # Format: word -> (dG_prime, dD_prime)
    # dG_prime: how much to shift gravity baseline (negative = heavier)
    # dD_prime: how much to shift dominance baseline

    # Life events — heavy gravity
    "wedding":      (-25, 10),
    "funeral":      (-40, -10),
    "divorce":      (-35, -15),
    "marriage":     (-20, 5),
    "pregnancy":    (-20, 5),
    "birth":        (-15, 10),
    "graduation":   (-10, 15),
    "retirement":   (-15, 5),

    # Health — existential weight
    "cancer":       (-45, -20),
    "diagnosis":    (-35, -15),
    "surgery":      (-30, -10),
    "hospital":     (-25, -10),
    "disease":      (-35, -15),
    "disability":   (-25, -15),

    # Death/loss
    "death":        (-50, -20),
    "suicide":      (-50, -25),
    "murder":       (-45, -20),
    "war":          (-40, -15),
    "genocide":     (-50, -25),
    "famine":       (-40, -15),

    # Family/relationships — personal gravity
    "mother":       (-20, 5),
    "father":       (-20, 5),
    "children":     (-25, 10),
    "baby":         (-20, 10),
    "family":       (-15, 5),
    "home":         (-15, 5),
    "daughter":     (-20, 5),
    "son":          (-20, 5),

    # Power/society — dominance weight
    "god":          (-30, 20),
    "religion":     (-20, 10),
    "freedom":      (-15, 20),
    "justice":      (-20, 15),
    "government":   (-10, 15),
    "law":          (-10, 15),
    "prison":       (-30, -20),
    "poverty":      (-35, -20),
    "money":        (-15, 10),
    "wealth":       (-10, 15),

    # Abstract stakes
    "truth":        (-15, 15),
    "trust":        (-20, 10),
    "betrayal":     (-35, -15),
    "legacy":       (-20, 10),
    "identity":     (-15, 5),
    "dignity":      (-20, 10),
    "innocence":    (-20, -5),

    # Abuse/trauma
    "abuse":        (-40, -20),
    "assault":      (-40, -20),
    "rape":         (-50, -25),
    "molestation":  (-50, -25),
    # Foster/child welfare
    "foster":       (-25, -15),
    "custody":      (-30, -15),
    "orphan":       (-35, -20),
    # Legal
    "court":        (-20, -10),
    "trial":        (-25, -10),
    "verdict":      (-25, -10),
    # Addiction
    "rehab":        (-30, -15),
    "withdrawal":   (-30, -15),
    "sober":        (-15, 5),
    # Reproductive loss
    "miscarriage":  (-45, -20),
    "stillborn":    (-50, -25),
    "infertility":  (-30, -15),
    # Death adjacent
    "burial":       (-40, -10),
    "morgue":       (-40, -15),
    "grief":        (-35, -15),
    "widow":        (-30, -10),
    # Financial
    "eviction":     (-35, -20),
    "foreclosure":  (-35, -20),
    "unemployment": (-30, -20),
    "bankruptcy":   (-35, -20),
    # Institutional
    "shelter":      (-25, -10),
    "refugee":      (-35, -20),
    "deportation":  (-35, -20),

    # Crisis methods — not emotional words, but carry massive gravity
    "pistol":       (-40, -20),
    "pills":        (-35, -15),
    "noose":        (-50, -25),
    "razor":        (-35, -15),
    "wrist":        (-25, -10),
    "bleed":        (-35, -15),
    "rooftop":      (-25, -10),
    "attempt":      (-35, -15),
    "goodbye":      (-30, -15),
    "tonight":      (-15, -5),
    "method":       (-30, -15),
    "rope":         (-30, -15),
    # Weapons / harm implements
    "blade":        (-30, -15),
    "knife":        (-30, -15),
    "rifle":        (-35, -20),
    "wire":         (-20, -10),
    "belt":         (-15, -5),
    "bullet":       (-35, -20),
    "trigger":      (-25, -15),
    "barrel":       (-25, -15),
    "ammo":         (-30, -15),
    # Harm actions
    "cut":          (-20, -10),
    "slice":        (-20, -10),
    "stab":         (-35, -20),
    "drown":        (-40, -20),
    "choke":        (-35, -15),
    "suffocate":    (-40, -20),
    "strangle":     (-40, -20),
    # Locations
    "bridge":       (-20, -5),

    "cliff":        (-30, -15),
    "ledge":        (-25, -10),
    "tracks":       (-25, -10),
    "train":        (-15, -5),

    # Institutional trust — failure by helpers hits harder
    "therapist":    (-20, -10),
    "counselor":    (-20, -10),
    "doctor":       (-15, -5),
    "teacher":      (-15, -5),

}

# ---------------------------------------------------------------------------
# Universal Quantifiers — scope amplifiers that make EVERYTHING heavier
# "Everything is pointless" vs "This is pointless" — same payload, different gravity
# These prime G downward because they remove all escape routes.
# ---------------------------------------------------------------------------

UNIVERSAL_QUANTIFIERS = {
    # word -> gravity_amplifier (how much scope expands the emotional weight)
    # These AMPLIFY gravity in whatever direction the payload pushes.
    # "Everything is beautiful" = lighter (G goes UP more)
    # "Everything is pointless" = heavier (G goes DOWN more)
    "everything":   1.8,
    "nothing":      1.8,
    "everyone":     1.6,
    "nobody":       1.6,
    "always":       1.5,
    "never":        1.5,
    "forever":      1.5,
    "anywhere":     1.3,
    "nowhere":      1.6,
    "all":          1.3,
    "every":        1.3,
    "entire":       1.3,
    "whole":        1.2,
    "each":         1.2,
    "any":          1.2,
    "none":         1.8,
    "constantly":   1.4,
    "endlessly":    1.5,
    "perpetually":  1.5,
    "utterly":      1.5,
    "whatsoever":   1.4,
    "altogether":   1.3,
    "anything":     1.5,
    "everywhere":   1.3,
}

# ---------------------------------------------------------------------------
# Lookahead operators — words that change role based on what's NEXT
# "pretty lame" → pretty amplifies lame (1.2x)
# "pretty girl" → pretty describes girl (positive adjective)
# "you are pretty" → pretty IS the payload (positive)
# ---------------------------------------------------------------------------

LOOKAHEAD_AMPLIFIERS = {
    # word -> (amplifier_coeff, payload_force_if_standalone)
    # If next word is in EMOTIONAL_VOCABULARY → use as amplifier
    # If next word is neutral/end → use payload_force as own emotion
    "pretty":   (1.2, (30, 10, 15, 0, 15)),    # attractive/nice
    "fairly":   (1.2, (10, 5, 10, 0, 5)),       # moderately
    "quite":    (1.25, (15, 5, 10, 0, 5)),       # considerably
    "real":     (1.3, (10, 5, 15, 0, 5)),        # genuine/authentic
}

EVOKER_DECAY = 0.88  # How fast gravitational priming decays per word

# ---------------------------------------------------------------------------
# Conditional frames (Force #13)
# "If that happened, I'd be furious" — hypothetical, not actual fury.
# ---------------------------------------------------------------------------

CONDITIONAL_STARTERS = frozenset({
    "if", "unless", "supposing", "assuming", "imagine",
    "hypothetically", "theoretically",
    "suppose", "provided", "whether", "granted",
})

CONDITIONAL_DAMPENER = 0.40  # Conditionals gate reality: 0.4x on all forces after

# ---------------------------------------------------------------------------
# Evidential / clinical distance (Force #15)
# "She said she's fine" — reported, not experienced.
# "The subject reports feeling..." — clinical framing kills emotion.
# ---------------------------------------------------------------------------

EVIDENTIAL_OPERATORS = {
    # Format: word -> (coeff, d_offset)
    "said":         (0.50, -15),
    "told":         (0.50, -15),
    "claims":       (0.40, -20),
    "reports":      (0.40, -20),
    "states":       (0.40, -20),
    "mentioned":    (0.50, -15),
    "indicated":    (0.40, -20),
    "expressed":    (0.50, -15),
    "described":    (0.40, -20),
    "observed":     (0.40, -20),
    "noted":        (0.50, -15),
}

# Clinical framing words — strong dampening when combined
CLINICAL_FRAME = frozenset({
    "subject", "patient", "client", "individual", "respondent",
    "participant", "informant",
})

CLINICAL_DAMPENER = 0.30  # "The subject reports" → 0.3x

# ---------------------------------------------------------------------------
# Comparatives and Superlatives (Forces #20-21)
# "more angry" amplifies the next payload, "most angry" amplifies harder.
# ---------------------------------------------------------------------------

COMPARATIVES = frozenset({
    "more", "less", "better", "worse", "harder", "easier",
    "stronger", "weaker", "bigger", "smaller", "deeper",
})
COMPARATIVE_AMPLIFIER = 1.3  # comparatives amplify the next payload

SUPERLATIVES = frozenset({
    "most", "least", "best", "worst", "hardest", "easiest",
    "strongest", "weakest", "biggest", "smallest", "deepest", "ever",
})
SUPERLATIVE_AMPLIFIER = 1.5  # superlatives amplify more

# ---------------------------------------------------------------------------
# Discourse Fillers (Force #23)
# "um", "uh" — signal processing difficulty, lower Dominance.
# ---------------------------------------------------------------------------

FILLERS = frozenset({"um", "uh", "er", "ah", "hmm", "hm", "uhh", "umm"})
FILLER_D_OFFSET = -5  # each filler slightly lowers dominance (uncertainty signal)

# ---------------------------------------------------------------------------
# Emotional Performatives (Force #24)
# "I promise", "I swear" — amplify the next payload + slight D boost.
# ---------------------------------------------------------------------------

PERFORMATIVES = {
    "promise": 1.3,
    "swear": 1.4,
    "vow": 1.4,
    "guarantee": 1.3,
    "pledge": 1.3,
    "commit": 1.2,
}

# ---------------------------------------------------------------------------
# Passive Voice Detection (Force #19)
# "I was hurt" — passive removes agency → D drops
# ---------------------------------------------------------------------------

PASSIVE_MARKERS = frozenset({"was", "were", "been", "being", "got", "gotten"})
PASSIVE_PARTICIPLES = frozenset({
    "hurt", "broken", "abandoned", "betrayed", "rejected",
    "ignored", "forgotten", "left", "used", "abused",
    "manipulated", "deceived", "humiliated", "dismissed",
    "fired", "arrested", "attacked", "blamed", "cheated",
    "crushed", "destroyed", "devastated", "disappointed",
    "embarrassed", "excluded", "exploited", "harassed",
    "insulted", "isolated", "mistreated", "neglected",
    "offended", "overwhelmed", "punished", "ridiculed",
    "scared", "shamed", "silenced", "surprised",
    "threatened", "traumatized", "tricked", "victimized",
    "violated", "wounded",
    "robbed", "stolen", "conned", "scammed", "controlled", "ghosted",
    "bullied", "coerced", "blackmailed", "stalked", "dumped", "ditched",
    "replaced", "discarded", "demoted", "suspended", "expelled",
    "detained", "drugged", "poisoned", "mugged", "kidnapped",
    "censored", "suppressed", "patronized", "doxxed", "scapegoated",
})
PASSIVE_D_OFFSET = -15  # passive removes agency

# ---------------------------------------------------------------------------
# Rhetorical Questions (Force #7)
# "Who cares?" = dismissive. "Isn't that great?" = sarcastic negative.
# Detected in post-pass: question + emotional content = rhetorical → invert.
# ---------------------------------------------------------------------------

RHETORICAL_DISMISSALS = frozenset({
    "who cares", "what difference does it make", "why bother",
    "so what", "big deal", "like i care", "as if",
    "yeah right", "oh really", "sure thing",
})

# ---------------------------------------------------------------------------
# Litotes / Understatement (Force #10)
# "not exactly thrilled" — the negation force already handles simple cases.
# These are qualifier chains that produce understatement, not inversion.
# Handled by: negation decay + hedge stacking. Explicit patterns here.
# ---------------------------------------------------------------------------

LITOTES_QUALIFIERS = frozenset({
    "exactly", "particularly", "especially", "entirely",
    "altogether", "necessarily", "precisely",
})
LITOTES_DAMPENER = 0.7  # "not exactly X" = weaker than "not X"

# ---------------------------------------------------------------------------
# Social Politeness / Weaponized Courtesy (Force #16)
# "With all due respect" AMPLIFIES negative content that follows.
# Detected: polite preamble + negative payload = weaponized.
# ---------------------------------------------------------------------------

POLITE_WEAPONS = frozenset({
    "respect", "offense", "honestly", "frankly",
    "sorry", "forgive",
})
# When these words precede negative payloads, they're dominance plays
WEAPONIZED_D_BOOST = 15  # raises D (power move) when weaponized

# ---------------------------------------------------------------------------
# Tag Questions (Force #18)
# "isn't it?", "right?", "don't you think?" at sentence end
# Seeking validation = D-10. In aggressive context = D+10.
# ---------------------------------------------------------------------------

TAG_PATTERNS = frozenset({
    "right", "huh", "eh", "no", "okay", "yeah",
})
TAG_D_OFFSET = -10  # default: seeking validation lowers D

DEFAULT_MOMENTUM = 0.82   # How much of old state to keep per word (0=instant, 1=frozen)
FORCE_SCALE = 0.50        # Global force scaling (how hard words push the pendulum)
CRISIS_V_THRESHOLD = 60
CRISIS_U_THRESHOLD = 40
CRISIS_MOMENTUM = 0.98

# Neutral center for each dimension (V/A/D start at 128, U at 0, G at 128)
CENTER = {"v": 128.0, "a": 128.0, "d": 128.0, "u": 0.0, "g": 128.0}


# ---------------------------------------------------------------------------
# Pre-pass info container
# ---------------------------------------------------------------------------

@dataclass
class PrePassInfo:
    """Sentence-level features detected before word-by-word processing."""
    is_question: bool = False
    question_dampener: float = 1.0
    is_conditional: bool = False
    conditional_dampener: float = 1.0
    passive_positions: set = None          # indices of participles in passive constructions
    is_rhetorical: bool = False            # rhetorical question (invert, don't dampen)
    has_tag_question: bool = False         # tag question at end (D-shift)
    idiom_spans: Dict[int, Tuple] = None      # start_idx -> (length, force_tuple, label)
    idiom_consumed: set = None                  # all indices consumed by idioms
    negation_positions: set = None              # indices of negator words
    double_negation: bool = False               # two negators cancel each other
    hedge_count: int = 0                        # number of hedging operators in sentence
    words_lower: list = None                    # lowercase word list for post-pass

    def __post_init__(self):
        self.idiom_spans = self.idiom_spans or {}
        self.idiom_consumed = self.idiom_consumed or set()
        self.negation_positions = self.negation_positions or set()
        self.passive_positions = self.passive_positions or set()


# ---------------------------------------------------------------------------
# PendulumV2
# ---------------------------------------------------------------------------

class PendulumV2:
    """Clean 3-pass emotional physics engine.

    All physics parameters are configurable via constructor for tuning.
    Default values are the untuned baseline.
    """

    def __init__(
        self,
        momentum: float = DEFAULT_MOMENTUM,
        force_scale: float = FORCE_SCALE,
        direct_push_cap: float = 0.4,
        direct_push_trigger: float = 80.0,
        crisis_v: float = CRISIS_V_THRESHOLD,
        crisis_u: float = CRISIS_U_THRESHOLD,
        question_dampener: float = QUESTION_DAMPENER,
        scale_v: float = 1.0,
        scale_a: float = 1.0,
        scale_d: float = 1.0,
        scale_u: float = 1.0,
        scale_g: float = 1.0,
        threshold_low: float = 124.0,
        threshold_high: float = 132.0,
        negation_decay_operator: float = NEGATION_DECAY_OPERATOR,
        negation_decay_neutral: float = NEGATION_DECAY_NEUTRAL,
        negation_decay_payload: float = NEGATION_DECAY_PAYLOAD,
        # --- Conversational force params (10) ---
        evoker_decay: float = EVOKER_DECAY,
        conditional_dampener: float = CONDITIONAL_DAMPENER,
        clinical_dampener: float = CLINICAL_DAMPENER,
        passive_d_offset: float = PASSIVE_D_OFFSET,
        comparative_amplifier: float = COMPARATIVE_AMPLIFIER,
        superlative_amplifier: float = SUPERLATIVE_AMPLIFIER,
        filler_d_offset: float = FILLER_D_OFFSET,
        litotes_dampener: float = LITOTES_DAMPENER,
        weaponized_d_boost: float = WEAPONIZED_D_BOOST,
        tag_d_offset: float = TAG_D_OFFSET,
        personality=None,
    ):
        self.momentum = momentum
        self.personality = personality
        self.force_scale = force_scale
        self.direct_push_cap = direct_push_cap
        self.direct_push_trigger = direct_push_trigger
        self.crisis_v = crisis_v
        self.crisis_u = crisis_u
        self.question_dampener_val = question_dampener
        self.scale_v = scale_v
        self.scale_a = scale_a
        self.scale_d = scale_d
        self.scale_u = scale_u
        self.scale_g = scale_g
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.negation_decay_operator = negation_decay_operator
        self.negation_decay_neutral = negation_decay_neutral
        self.negation_decay_payload = negation_decay_payload
        self.evoker_decay = evoker_decay
        self.conditional_dampener_val = conditional_dampener
        self.clinical_dampener_val = clinical_dampener
        self.passive_d_offset = passive_d_offset
        self.comparative_amplifier = comparative_amplifier
        self.superlative_amplifier = superlative_amplifier
        self.filler_d_offset = filler_d_offset
        self.litotes_dampener = litotes_dampener
        self.weaponized_d_boost = weaponized_d_boost
        self.tag_d_offset = tag_d_offset

    def get_config(self) -> dict:
        """Return full config snapshot for experiment logging."""
        return {
            "momentum": self.momentum,
            "force_scale": self.force_scale,
            "direct_push_cap": self.direct_push_cap,
            "direct_push_trigger": self.direct_push_trigger,
            "crisis_v": self.crisis_v,
            "crisis_u": self.crisis_u,
            "question_dampener": self.question_dampener_val,
            "scale_v": self.scale_v,
            "scale_a": self.scale_a,
            "scale_d": self.scale_d,
            "scale_u": self.scale_u,
            "scale_g": self.scale_g,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "negation_decay_operator": self.negation_decay_operator,
            "negation_decay_neutral": self.negation_decay_neutral,
            "negation_decay_payload": self.negation_decay_payload,
            "evoker_decay": self.evoker_decay,
            "conditional_dampener": self.conditional_dampener_val,
            "clinical_dampener": self.clinical_dampener_val,
            "passive_d_offset": self.passive_d_offset,
            "comparative_amplifier": self.comparative_amplifier,
            "superlative_amplifier": self.superlative_amplifier,
            "filler_d_offset": self.filler_d_offset,
            "litotes_dampener": self.litotes_dampener,
            "weaponized_d_boost": self.weaponized_d_boost,
            "tag_d_offset": self.tag_d_offset,
        }

    def classify(self, v: float, mode: str = "three_way") -> str:
        """Classify valence into sentiment using configurable strategy.

        Modes:
            three_way: pos/neg/neutral using threshold_low and threshold_high
            binary:    pos/neg only, split at 128 (no neutral zone)
            prism:     5-band mapping (crisis/negative/neutral/positive/thriving)
        """
        if mode == "binary":
            return "positive" if v >= 128 else "negative"
        elif mode == "prism":
            if v < 51:
                return "negative"    # crisis band
            elif v < 103:
                return "negative"    # negative band
            elif v < 154:
                return "neutral"     # neutral band
            elif v < 205:
                return "positive"    # positive band
            else:
                return "positive"    # thriving band
        else:  # three_way (default)
            if v > self.threshold_high:
                return "positive"
            elif v < self.threshold_low:
                return "negative"
            return "neutral"

    def classify_5d(self, vadug, mode: str = "binary") -> str:
        """Multi-dimensional classification using ALL 5 VADUG dimensions.

        Unlike classify() which only reads V, this reads the full 5D state
        to catch cases where V alone gives the wrong answer:
        - "I destroyed them" — low V but high A+D = triumph (positive)
        - "I'm so sorry for your loss" — low V but high D+G = empathy (positive)
        - Frantic urgency — neutral V but extreme U+A = negative

        The translation layer: keeps VADUG pure, maps to benchmark labels.
        """
        v, a, d, u, g = vadug.v, vadug.a, vadug.d, vadug.u, vadug.g

        # --- Override rules (catch benchmark traps) ---

        # Triumph/empowerment: aggressive but positive outcome
        if v >= 90 and a > 180 and d > 180:
            return "positive"

        # Empathy/sympathy: sad topic but supportive/grounded delivery
        if v < 128 and d > 140 and g > 120 and u < 100:
            return "positive"

        # Panic/frantic: neutral words but extreme urgency
        if 100 <= v <= 140 and u > 180 and a > 180:
            return "negative"

        # --- Standard mapping ---
        if mode == "binary":
            # Use gravity to break ties in the murky middle
            if v > 135:
                return "positive"
            elif v < 120:
                return "negative"
            else:
                return "positive" if g > 128 else "negative"
        else:  # three_way
            if v > self.threshold_high:
                return "positive"
            elif v < self.threshold_low:
                return "negative"
            return "neutral"

    def process_text(self, text: str) -> Tuple[VADUG, List[dict]]:
        """Process text through the 3-pass emotional pipeline.

        Returns:
            (final_vadug, word_trace) where word_trace is a list of dicts
            with keys: word, role, v, a, d, u, g, note
        """
        # --- Pre-flight: structural analysis on raw text ---
        preflight = _PREFLIGHT.analyze(text)

        words = self._tokenize(text)
        if not words:
            return VADUG(), []

        # State: floating-point VADUG accumulator
        state = dict(CENTER)
        trace = []

        # --- Pass 1: Pre-pass ---
        pre = self._pre_pass(words)

        # --- Pass 2: Word-by-word ---
        pending_operators = {}  # category -> (coefficient, d_offset)
        negation_force = 0.0    # continuous negation: 0.0 = none, ~1.0 = full inversion
        gravity_prime = 0.0     # gravitational priming from evokers (shifts G baseline)
        dominance_prime = 0.0   # dominance priming from evokers (shifts D baseline)

        i = 0
        while i < len(words):
            word = words[i]

            # Skip words consumed by idiom detection
            if i in pre.idiom_consumed:
                # If this is the START of an idiom, apply it as a payload
                if i in pre.idiom_spans:
                    span = pre.idiom_spans[i]
                    length, force, label = span
                    coeff, d_off = self._compute_coefficient(pending_operators, pre)
                    neg_scale = self._negation_scale(negation_force)
                    # Crisis idioms bypass hedging dampening — "sometimes I want to die" is STILL crisis
                    if label.startswith("crisis"):
                        coeff = max(coeff, 1.5)
                        neg_scale = 1.0  # negation doesn't reduce crisis
                    state = self._apply_force(state, force, coeff * neg_scale, d_off)
                    trace.append(self._trace_entry(
                        " ".join(words[i:i+length]), "IDIOM", state,
                        f"idiom:{label} coeff={coeff:.2f} d_off={d_off:.0f} neg_f={negation_force:.2f}"
                    ))
                    pending_operators = {}
                    negation_force *= self.negation_decay_payload
                else:
                    trace.append(self._trace_entry(word, "IDIOM_PART", state, "consumed by idiom"))
                i += 1
                continue

            word_lower = word.lower()

            # Clause boundary kills negation
            if word_lower in CLAUSE_BOUNDARIES:
                old_nf = negation_force
                negation_force = 0.0
                trace.append(self._trace_entry(
                    word, "BOUNDARY", state,
                    f"clause boundary, negation {old_nf:.2f}→0.00"
                ))
                i += 1
                continue

            # Check negator — sets or reinforces negation force, applies self-force at 30%
            if i in pre.negation_positions:
                strength = NEGATOR_STRENGTH.get(word_lower, 0.85)
                negation_force = max(negation_force, strength)
                # Apply the negator's own emotional weight at 30% (constraint/inability)
                if word_lower in EMOTIONAL_VOCABULARY:
                    self_force = EMOTIONAL_VOCABULARY[word_lower]
                    state = self._apply_force(state, self_force, 0.3)
                    trace.append(self._trace_entry(
                        word, "NEGATOR", state,
                        f"neg_f={negation_force:.2f} +self_force@30%"
                    ))
                else:
                    trace.append(self._trace_entry(
                        word, "NEGATOR", state,
                        f"neg_f={negation_force:.2f}"
                    ))
                i += 1
                continue

            # Lookahead amplifiers — role depends on what's NEXT
            if word_lower in LOOKAHEAD_AMPLIFIERS:
                amp_coeff, standalone_force = LOOKAHEAD_AMPLIFIERS[word_lower]
                # Look ahead: is the next non-operator word emotional?
                next_is_emotional = False
                for j in range(i + 1, min(i + 3, len(words))):
                    next_w = words[j].lower()
                    if next_w in EMOTIONAL_VOCABULARY:
                        next_is_emotional = True
                        break
                    if next_w in CONTEXT_OPERATORS or next_w in NEGATORS:
                        continue  # skip operators, keep looking
                    break  # hit a neutral word — stop looking

                if next_is_emotional:
                    # Amplifier mode: boost the next payload
                    pending_operators["lookahead_amp"] = (amp_coeff, 0)
                    trace.append(self._trace_entry(
                        word, "AMPLIFIER", state,
                        f"lookahead: next is emotional, amp={amp_coeff}"))
                else:
                    # Standalone mode: BE the payload
                    coeff, d_off = self._compute_coefficient(pending_operators, pre)
                    neg_scale = self._negation_scale(negation_force)
                    state = self._apply_force(state, standalone_force, coeff * neg_scale, d_off)
                    trace.append(self._trace_entry(
                        word, "PAYLOAD", state,
                        f"lookahead: standalone adjective, force={standalone_force[:2]}..."))
                    pending_operators = {}
                    negation_force *= self.negation_decay_payload
                i += 1
                continue

            # Check universal quantifier — scope amplifier for gravity
            # "everything" makes whatever follows heavier/lighter by amplifying G force
            if word_lower in UNIVERSAL_QUANTIFIERS:
                amp = UNIVERSAL_QUANTIFIERS[word_lower]
                pending_operators["scope"] = (amp, 0)
                trace.append(self._trace_entry(
                    word, "SCOPE", state,
                    f"universal quantifier, amp={amp}"
                ))
                i += 1
                continue

            # Check evoker — gravitational priming (Force #25)
            if word_lower in EVOKERS:
                dg_prime, dd_prime = EVOKERS[word_lower]
                gravity_prime += dg_prime
                dominance_prime += dd_prime
                # Evokers also pass through as payload if in vocabulary
                if word_lower in EMOTIONAL_VOCABULARY:
                    force = EMOTIONAL_VOCABULARY[word_lower]
                    coeff, d_off = self._compute_coefficient(pending_operators, pre)
                    neg_scale = self._negation_scale(negation_force)
                    state = self._apply_force(state, force, coeff * neg_scale, d_off)
                    trace.append(self._trace_entry(
                        word, "EVOKER+PAY", state,
                        f"g_prime={gravity_prime:.0f} d_prime={dominance_prime:.0f} +payload"
                    ))
                    pending_operators = {}
                    negation_force *= self.negation_decay_payload
                else:
                    trace.append(self._trace_entry(
                        word, "EVOKER", state,
                        f"g_prime={gravity_prime:.0f} d_prime={dominance_prime:.0f}"
                    ))
                i += 1
                continue

            # Check evidential operator — reported speech distance (Force #15)
            if word_lower in EVIDENTIAL_OPERATORS:
                ev_coeff, ev_d = EVIDENTIAL_OPERATORS[word_lower]
                pending_operators["evidential"] = (ev_coeff, ev_d)
                trace.append(self._trace_entry(
                    word, "EVIDENTIAL", state,
                    f"evidential={ev_coeff} d_off={ev_d}"
                ))
                i += 1
                continue

            # Check clinical framing — "the subject reports" pattern
            if word_lower in CLINICAL_FRAME:
                pending_operators["clinical"] = (self.clinical_dampener_val, -20)
                trace.append(self._trace_entry(
                    word, "CLINICAL", state,
                    f"clinical={self.clinical_dampener_val} d_off=-20"
                ))
                i += 1
                continue

            # Check litotes qualifier — "not exactly X" dampens negation (Force #10)
            if word_lower in LITOTES_QUALIFIERS and negation_force > 0.3:
                negation_force *= self.litotes_dampener
                trace.append(self._trace_entry(
                    word, "LITOTES", state,
                    f"dampened negation to {negation_force:.2f}"
                ))
                i += 1
                continue

            # Check weaponized politeness — "honestly/frankly" before negative = D boost (Force #16)
            if word_lower in POLITE_WEAPONS:
                pending_operators["politeness"] = (1.0, self.weaponized_d_boost)
                trace.append(self._trace_entry(
                    word, "POLITENESS", state,
                    f"D+{self.weaponized_d_boost} (potential weapon)"
                ))
                i += 1
                continue

            # Check comparative — amplify next payload (Force #20)
            if word_lower in COMPARATIVES:
                pending_operators["comparative"] = (self.comparative_amplifier, 0)
                trace.append(self._trace_entry(word, "COMPARATIVE", state, f"amp={COMPARATIVE_AMPLIFIER}"))
                i += 1
                continue

            # Check superlative — amplify next payload harder (Force #21)
            if word_lower in SUPERLATIVES:
                pending_operators["superlative"] = (self.superlative_amplifier, 0)
                trace.append(self._trace_entry(word, "SUPERLATIVE", state, f"amp={SUPERLATIVE_AMPLIFIER}"))
                i += 1
                continue

            # Check performative — amplify + slight D boost (Force #24)
            if word_lower in PERFORMATIVES:
                pending_operators["performative"] = (PERFORMATIVES[word_lower], 5)
                trace.append(self._trace_entry(word, "PERFORMATIVE", state, f"amp={PERFORMATIVES[word_lower]}"))
                i += 1
                continue

            # Classify: OPERATOR or PAYLOAD or NEUTRAL
            if word_lower in CONTEXT_OPERATORS:
                # OPERATOR — accumulate coefficient, gentle negation decay
                coeff_val, category, d_off = _parse_operator(CONTEXT_OPERATORS[word_lower])
                pending_operators[category] = (coeff_val, d_off)
                negation_force *= self.negation_decay_operator
                trace.append(self._trace_entry(
                    word, "OPERATOR", state,
                    f"{category}={coeff_val} neg_f={negation_force:.2f}"
                ))

            elif word_lower in EMOTIONAL_VOCABULARY:
                # PAYLOAD — apply force modulated by continuous negation + gravity priming
                force = EMOTIONAL_VOCABULARY[word_lower]
                coeff, d_off = self._compute_coefficient(pending_operators, pre)
                # Self-adjacent emotion: handled via "me X" bigrams (me mad, me sad, etc.)
                # General amplifier was tested but caused regressions — bigrams are more precise
                # Add evoker priming to D-offset
                d_off += dominance_prime * self.force_scale
                # Passive voice lowers D (removed agency)
                if i in pre.passive_positions:
                    d_off += self.passive_d_offset
                neg_scale = self._negation_scale(negation_force)
                state = self._apply_force(state, force, coeff * neg_scale, d_off)
                # Apply gravity priming directly to G state
                if gravity_prime != 0.0:
                    state["g"] += gravity_prime * self.force_scale
                trace.append(self._trace_entry(
                    word, "PAYLOAD", state,
                    f"coeff={coeff:.2f} d_off={d_off:.0f} g_prime={gravity_prime:.0f} neg_f={negation_force:.2f}"
                ))
                # Payload consumes most negation force, reset operators
                pending_operators = {}
                negation_force *= self.negation_decay_payload
                # Decay evoker priming
                gravity_prime *= self.evoker_decay
                dominance_prime *= self.evoker_decay

            elif word_lower in FILLERS:
                # Fillers signal processing difficulty — lower D slightly (Force #23)
                state["d"] += self.filler_d_offset * self.force_scale
                trace.append(self._trace_entry(word, "FILLER", state, f"D{self.filler_d_offset}"))

            else:
                # Try fuzzy matching before giving up (typos, elongation, text speak)
                fuzzy_hit = _fuzzy_match(word_lower)
                if fuzzy_hit and fuzzy_hit in EMOTIONAL_VOCABULARY:
                    force = EMOTIONAL_VOCABULARY[fuzzy_hit]
                    coeff, d_off = self._compute_coefficient(pending_operators, pre)
                    d_off += dominance_prime * self.force_scale
                    neg_scale = self._negation_scale(negation_force)
                    state = self._apply_force(state, force, coeff * neg_scale, d_off)
                    trace.append(self._trace_entry(
                        word, "FUZZY", state,
                        f"→{fuzzy_hit} coeff={coeff:.2f} neg_f={negation_force:.2f}"
                    ))
                    pending_operators = {}
                    negation_force *= self.negation_decay_payload
                    gravity_prime *= self.evoker_decay
                    dominance_prime *= self.evoker_decay
                    i += 1
                    continue

                # NEUTRAL — moderate negation decay
                negation_force *= self.negation_decay_neutral
                trace.append(self._trace_entry(word, "NEUTRAL", state, f"neg_f={negation_force:.2f}"))

            i += 1

        # --- Sarcasm detection (Force #5) ---
        # Three layers: templates → polarity → trajectory
        # Templates NUDGE (subtle), don't INVERT (aggressive).
        # "The best sarcasm is subtle, so the detector needs to be subtle too."
        sarcasm_detected = False
        sarcasm_conf = 0

        # Layer 0: Template detection — subtle nudge only
        tmpl = _SARCASM_TEMPLATES.detect(words)
        # Don't apply sarcasm nudge if sentence is heavily hedged — hedging ≠ sarcasm
        hedge_guard = pre and pre.hedge_count >= 2
        if tmpl.detected and tmpl.confidence >= 0.5 and state["v"] > 120 and not hedge_guard:
            # Only nudge if V is positive-ish (don't push already-negative further)
            # Scale nudge with distance from neutral — stronger words get stronger correction
            base_nudge = 18 if tmpl.confidence >= 0.8 else (12 if tmpl.confidence >= 0.6 else 8)
            distance = state["v"] - 128
            # T5 high-confidence + very high V + intensifier = near-certain sarcasm
            # Only fire aggressive pull when V is extremely positive (>160) — avoids
            # hitting genuine pride/joy that also triggers T5 at V=150-170
            if tmpl.template == 5 and tmpl.confidence >= 0.8 and distance > 60:
                # "I'm absolutely thrilled to redo my work" — V=191, clearly sarcastic
                nudge = distance + 5  # pull past neutral into slightly negative
            else:
                nudge = base_nudge + max(0, distance * 0.5)  # 50% of excess beyond neutral
            state["v"] -= nudge
            state["g"] -= 5
            trace.append({
                "word": "[SARCASM_TMPL]", "role": "SARCASM",
                "v": round(state["v"], 1), "a": round(state["a"], 1),
                "d": round(state["d"], 1), "u": round(state["u"], 1),
                "g": round(state["g"], 1),
                "note": f"template T{tmpl.template} conf={tmpl.confidence:.2f} nudge=-{nudge}"
            })

        # Layer 1: trajectory reversal/mismatch
        det1, conf1, signals1 = _SARCASM_DETECTOR.analyze_trajectory(trace)

        # Layer 2: surface vs result divergence
        # Count positive vs negative payload forces in the trace
        pos_payloads = sum(1 for t in trace if t['role'] in ('PAYLOAD', 'EVOKER+PAY')
                         and t['v'] > 140)
        neg_payloads = sum(1 for t in trace if t['role'] in ('PAYLOAD', 'EVOKER+PAY')
                         and t['v'] < 115)
        # If positive words dominate AND no negative payloads AND no negation explains
        # the drop → suspicious. Negation naturally pulls V down — not sarcasm.
        has_negation = len(pre.negation_positions) > 0 if pre else False
        if pos_payloads >= 2 and neg_payloads == 0 and not has_negation and state["v"] < 130:
            sarcasm_conf = max(conf1, SarcasmDetector.MODERATE)
            sarcasm_detected = True
        # Layer 1 results
        if det1 and conf1 >= SarcasmDetector.MODERATE:
            sarcasm_detected = True
            sarcasm_conf = max(sarcasm_conf, conf1)

        if sarcasm_detected and sarcasm_conf >= SarcasmDetector.MODERATE:
            inversion = 0.6 if sarcasm_conf >= SarcasmDetector.HIGH else 0.4
            state["v"] = 128 + (128 - state["v"]) * inversion
            state["d"] += 10
            state["g"] -= 10
            trace.append({
                "word": "[SARCASM]", "role": "SARCASM",
                "v": round(state["v"], 1), "a": round(state["a"], 1),
                "d": round(state["d"], 1), "u": round(state["u"], 1),
                "g": round(state["g"], 1),
                "note": f"conf={sarcasm_conf} pos={pos_payloads} neg={neg_payloads} → V inverted"
            })

        # --- Tonal analysis (6-signal trajectory detector) ---
        # Complements the bigram-based sarcasm detection above
        tone_result = _TONAL_ANALYZER.analyze(trace)
        if tone_result['tone'] == 'sarcastic' and tone_result['confidence'] >= 0.7 and sarcasm_detected:
            # Lower threshold from 0.7 to 0.5 for single-sentence (noisier than arc)
            old_v = state["v"]
            state["v"], state["a"], state["d"], state["u"], state["g"] = apply_tonal_adjustment(
                state["v"], state["a"], state["d"], state["u"], state["g"],
                {**tone_result, 'confidence': max(tone_result['confidence'], 0.7)},  # force the threshold
                None,  # no intent mode in raw V2
            )
            if state["v"] != old_v:
                trace.append({
                    "word": "[TONAL]", "role": "TONAL",
                    "v": round(state["v"], 1), "a": round(state["a"], 1),
                    "d": round(state["d"], 1), "u": round(state["u"], 1),
                    "g": round(state["g"], 1),
                    "note": f"tone={tone_result['tone']} conf={tone_result['confidence']:.2f} V:{old_v:.0f}→{state['v']:.0f}"
                })

        # --- Standalone resignation detection ---
        # If the entire message has no payloads and only operators/neutral,
        # AND it's short (1-5 words), it might be a resignation statement.
        # "Whatever." "Fine." "Sure." "K." — these ARE the message.
        payload_count = sum(1 for t in trace if t.get('role') in
                          ('PAYLOAD', 'IDIOM', 'EVOKER+PAY'))
        if payload_count == 0 and len(words) <= 5:
            deflection_count = sum(1 for t in trace if t.get('role') == 'OPERATOR'
                                  and 'deflection' in t.get('note', ''))
            if deflection_count > 0:
                # Standalone deflection = resignation
                state["v"] -= 15
                state["d"] -= 20
                state["g"] -= 10
                trace.append({
                    "word": "[RESIGNATION]", "role": "RESIGNATION",
                    "v": round(state["v"], 1), "a": round(state["a"], 1),
                    "d": round(state["d"], 1), "u": round(state["u"], 1),
                    "g": round(state["g"], 1),
                    "note": "standalone deflection = resignation"
                })

        final = self._post_pass(state, pre, preflight)

        return final, trace

    # -----------------------------------------------------------------------
    # Pass 1: Pre-pass
    # -----------------------------------------------------------------------

    def _pre_pass(self, words: List[str]) -> PrePassInfo:
        """Detect sentence-level features before word-by-word processing."""
        info = PrePassInfo()

        # Question detection
        info.is_question = is_question(words)
        info.question_dampener = self.question_dampener_val if info.is_question else 1.0

        # Idiom detection (greedy longest-match, left to right)
        info.idiom_spans, info.idiom_consumed = self._detect_idioms(words)

        # Conditional detection (if/unless/assuming at sentence start)
        if words and words[0].lower() in CONDITIONAL_STARTERS:
            info.is_conditional = True
            info.conditional_dampener = self.conditional_dampener_val

        # Passive voice detection (was/were/been + past participle pattern)
        lower_words = [w.lower() for w in words]
        for i, w in enumerate(lower_words):
            if w in PASSIVE_MARKERS and i + 1 < len(lower_words):
                # Check next 1-2 words for a past participle
                for j in range(i + 1, min(i + 3, len(lower_words))):
                    if lower_words[j] in PASSIVE_PARTICIPLES:
                        info.passive_positions.add(j)
                        break

        # Rhetorical question detection (Force #7)
        # A question with emotional content or a known dismissal pattern
        if info.is_question:
            sentence_lower = " ".join(lower_words)
            for pattern in RHETORICAL_DISMISSALS:
                if pattern in sentence_lower:
                    info.is_rhetorical = True
                    break

        # Tag question detection (Force #18)
        # "right?", "huh?", "isn't it?" at end of sentence
        if len(lower_words) >= 3:
            last = lower_words[-1].rstrip("?")
            if last in TAG_PATTERNS and (lower_words[-1].endswith("?") or "?" in lower_words):
                info.has_tag_question = True

        # Negation detection (mark positions of negator words)
        for i, w in enumerate(words):
            if w.lower() in NEGATORS and i not in info.idiom_consumed:
                info.negation_positions.add(i)

        # Hedge count: how many hedging/uncertainty operators in the sentence
        _hedge_words = {"maybe", "perhaps", "probably", "possibly", "potentially",
                        "generally", "sometimes", "occasionally", "arguably",
                        "seemingly", "apparently", "supposedly", "might", "could",
                        "somewhat", "slightly", "think", "guess", "suppose",
                        "wonder", "reckon", "assume", "possible", "tend",
                        "theoretically", "conceivably", "likely", "unlikely"}
        info.hedge_count = sum(1 for w in lower_words if w in _hedge_words)
        info.words_lower = lower_words
        # Also count hedge PHRASES (multi-word hedging structures)
        text_lower = " ".join(lower_words)
        _hedge_phrases = ["not sure", "in theory", "in some cases", "in practice",
                          "lets just say", "let's just say", "without guarantees",
                          "without making", "hard to say", "to some extent",
                          "for the most part", "it depends", "not necessarily",
                          "not entirely", "slight chance", "slim chance"]
        info.hedge_count += sum(1 for p in _hedge_phrases if p in text_lower)

        # Double-negation detection: two negators close together cancel out
        # "nothing and nobody will stop" = positive (double neg = affirmation)
        # "no challenge we cannot overcome" = positive
        # Guard: only cancel when both negators are "universal" type (nothing/nobody/no/nor)
        # NOT when one is "can't/don't" + "nobody" (that's sarcasm reinforcement)
        UNIVERSAL_NEGATORS = {"nothing", "nobody", "no", "none", "nor", "neither"}
        neg_list = sorted(info.negation_positions)
        if len(neg_list) >= 2:
            for j in range(len(neg_list) - 1):
                gap = neg_list[j + 1] - neg_list[j]
                w1 = lower_words[neg_list[j]] if neg_list[j] < len(lower_words) else ""
                w2 = lower_words[neg_list[j + 1]] if neg_list[j + 1] < len(lower_words) else ""
                # Only cancel if both are universal negators OR one is "cannot/no" + verb
                both_universal = w1 in UNIVERSAL_NEGATORS and w2 in UNIVERSAL_NEGATORS
                no_cannot = (w1 == "no" and w2 == "cannot") or (w1 == "no" and w2 == "cant")
                if gap <= 4 and (both_universal or no_cannot):
                    info.negation_positions.discard(neg_list[j])
                    info.negation_positions.discard(neg_list[j + 1])
                    info.double_negation = True
                    break

        return info

    def _detect_idioms(self, words: List[str]) -> Tuple[dict, set]:
        """Greedy longest-match expression detection (idioms + bigrams).

        Returns:
            (spans, consumed) where spans maps start_idx to (length, force, label)
            and consumed is the set of all word indices used by expressions.
        """
        spans = {}
        consumed = set()
        lower_words = [w.lower() for w in words]

        # Pre-compute max length across combined expressions
        max_idiom_len = max((len(k) for k in _COMBINED_EXPRESSIONS), default=0)

        i = 0
        while i < len(lower_words):
            if i in consumed:
                i += 1
                continue

            matched = False
            # Try longest match first
            for length in range(min(max_idiom_len, len(lower_words) - i), 1, -1):
                key = tuple(lower_words[i:i+length])
                if key in _COMBINED_EXPRESSIONS:
                    raw = _COMBINED_EXPRESSIONS[key]
                    # Parse tuple: 5-element (dv,da,dd,du,label) or 6-element (dv,da,dd,du,dg,label)
                    if len(raw) == 5:
                        force = (raw[0], raw[1], raw[2], raw[3], 0)
                        label = raw[4]
                    else:
                        force = (raw[0], raw[1], raw[2], raw[3], raw[4])
                        label = raw[5]

                    spans[i] = (length, force, label)
                    for j in range(i, i + length):
                        consumed.add(j)
                    matched = True
                    i += length
                    break

            if not matched:
                i += 1

        return spans, consumed

    # -----------------------------------------------------------------------
    # Pass 2: Word processing helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _negation_scale(negation_force: float) -> float:
        """Convert continuous negation force to a scaling factor.

        negation_force 0.0 → scale  1.0 (no negation)
        negation_force 0.5 → scale  0.0 (cancelled out)
        negation_force 1.0 → scale -1.0 (full inversion)

        The formula: scale = 1.0 - 2.0 * negation_force
        This is linear and continuous — no boolean anywhere.
        """
        return 1.0 - 2.0 * negation_force

    def _compute_coefficient(self, pending_operators: dict, pre: PrePassInfo) -> tuple:
        """Multiply across categories, apply question dampener, clamp.

        Returns:
            (coeff, d_offset) where coeff is the scalar multiplier and
            d_offset is the accumulated Dominance shift from hedging/uncertainty.
        """
        coeff = 1.0
        d_offset = 0.0
        for cat_val in pending_operators.values():
            cat_coeff, cat_d = cat_val
            coeff *= cat_coeff
            d_offset += cat_d

        coeff *= pre.question_dampener
        coeff *= pre.conditional_dampener
        return max(_COEFF_FLOOR, min(_COEFF_CAP, coeff)), d_offset

    def _apply_force(self, state: dict, force: tuple, scale: float, d_offset: float = 0.0) -> dict:
        """Apply an emotional force vector to the pendulum state.

        Two-component model (simplified from V1):
          1. Momentum blend: pull state toward force's implied target
          2. Direct push: strong words bypass momentum partially
          3. D-offset: independent Dominance shift from hedging/uncertainty

        Target = center + force*scale (what state WOULD be if only word).
        new = old * momentum + target * (1-momentum) + force * direct_push.
        """
        dv, da, dd, du, dg = force
        fs = self.force_scale * scale

        # Personality modifies how hard forces land
        if self.personality:
            sensitivity = self.personality.emotional_sensitivity
            fs *= sensitivity
            d_offset += self.personality.dominance_baseline
            # D inversion for sensitivity: when words are DOMINANT (dd > 0),
            # a sensitive entity feels LESS in control (dd dampened/inverted).
            # A tough entity absorbs dominant words and maintains agency.
            if dd > 0 and sensitivity > 1.2:
                # Sensitive + dominant input = feels overwhelmed
                dd = dd * (1.0 / sensitivity)  # dampen the D boost
            elif dd > 0 and sensitivity < 0.8:
                # Tough + dominant input = absorbs it, stays strong
                dd = dd * 1.0  # normal

        m = self.momentum

        # Per-dimension scaling
        dv *= self.scale_v
        da *= self.scale_a
        dd *= self.scale_d
        du *= self.scale_u
        dg *= self.scale_g

        # Target: where this word alone would place the pendulum
        target_v = CENTER["v"] + dv * fs
        target_a = CENTER["a"] + da * fs
        target_d = CENTER["d"] + dd * fs
        target_u = CENTER["u"] + du * fs
        target_g = CENTER["g"] + dg * fs

        # Direct push: stronger words push harder (configurable bypass)
        total_force = abs(dv * scale) + abs(da * scale)
        push = min(1.0, total_force / self.direct_push_trigger) * self.direct_push_cap

        blend = 1.0 - m
        return {
            "v": state["v"] * m + target_v * blend + dv * fs * push,
            "a": state["a"] * m + target_a * blend + da * fs * push,
            "d": state["d"] * m + target_d * blend + dd * fs * push + d_offset * self.force_scale,
            "u": state["u"] * m + target_u * blend + du * fs * push,
            "g": state["g"] * m + target_g * blend + dg * fs * push,
        }

    # -----------------------------------------------------------------------
    # Pass 3: Post-pass
    # -----------------------------------------------------------------------

    def _post_pass(self, state: dict, pre: PrePassInfo = None, preflight=None) -> VADUG:
        """Crisis detection, rhetorical/tag adjustments, preflight mults, clamping."""
        v, a, d, u, g = state["v"], state["a"], state["d"], state["u"], state["g"]

        # Rhetorical question inversion (Force #7)
        # "Who cares?" — invert the emotional content, not just dampen
        if pre and pre.is_rhetorical:
            v = 128 + (128 - v) * 0.6  # partial inversion toward opposite

        # Tag question D-shift (Force #18)
        # "right?", "huh?" — seeking validation lowers confidence
        if pre and pre.has_tag_question:
            d += self.tag_d_offset

        # Apply pre-flight environmental multipliers (digital prosody)
        if preflight:
            # Amplify/dampen A, U, D based on text structure (caps, length, punctuation)
            a = 128 + (a - 128) * preflight.arousal_mult
            u = u * preflight.urgency_mult
            d = 128 + (d - 128) * preflight.dominance_mult
            g = g + preflight.gravity_offset
            # Ellipsis valence drag — "im fine..." drags V further from neutral
            if preflight.valence_drag != 1.0:
                v = 128 + (v - 128) * preflight.valence_drag

        # Hedge stacking: 3+ hedging operators = heavy uncertainty, pull V toward neutral
        # "It's possible that some people could potentially be upset" = 4 hedges
        if pre and pre.hedge_count >= 2:
            pull = min(0.5, pre.hedge_count * 0.12)  # 24% at 2, 36% at 3, 48% at 4+
            v = v * (1 - pull) + 128 * pull
            d -= pre.hedge_count * 2  # stacked hedging = significant uncertainty

        # Crisis idiom amplification: if any crisis idiom was detected,
        # push V harder toward crisis zone regardless of momentum blending.
        # "sometimes I want to die" is STILL a crisis — momentum shouldn't save it.
        if pre and pre.idiom_spans:
            for span_info in pre.idiom_spans.values():
                _length, _force, label = span_info
                if label.startswith("crisis"):
                    # Push V hard toward crisis — hedging/momentum shouldn't save crisis
                    crisis_target = max(0, 128 + min(_force[0], -60) * 1.5)
                    v = v * 0.3 + crisis_target * 0.7  # 70% pull toward crisis
                    d = d * 0.3 + (128 + min(_force[2], -40) * 1.5) * 0.7
                    u = max(u, 50)  # crisis always has some urgency
                    break  # one crisis idiom is enough

        # Minimization detection (Layer 2 modifier): pull V toward neutral
        # "I'm fine", "not that bad", "just a little" = dampening severity
        if pre and pre.idiom_spans:
            for span_info in pre.idiom_spans.values():
                _length, _force, label = span_info
                if "minimization" in label:
                    # Pull V 20% toward neutral — minimization dampens, doesn't erase
                    v = v * 0.8 + 128 * 0.2
                    d -= 5  # minimization = slight loss of agency
                    break

        # Digital laughter detection (Layer 2 modifier):
        # "lol" (lowercase) = tone marker, peppered in to signal lightness
        # "LOL" (caps) = actually laughing, genuine amusement
        # "lol" after crisis = minimization ("i want to die lol" = "I'm fine" for crisis)
        # "haha" = genuine amusement. "HA HA" = sarcasm.
        # "lmao/lmfao" = strong amusement (like caps LOL)
        if pre and pre.words_lower:
            # Check original case from the raw words (before lowering)
            raw_words = [w for w in (pre.words_lower or [])]  # already lowered
            # We need original case — check if ALL CAPS version appears
            # For now, treat all as lowercase (preflight handles caps detection)
            has_lol = 'lol' in pre.words_lower
            has_lmao = 'lmao' in pre.words_lower or 'lmfao' in pre.words_lower
            has_haha = 'haha' in pre.words_lower or 'hahaha' in pre.words_lower
            if has_lol:
                if v < 110:
                    # "lol" after negative content = hedging/minimizing the pain
                    # "i want to die lol" — the lol is an "I'm fine" for text
                    v = v * 0.85 + 118 * 0.15  # pull slightly toward neutral
                    d -= 5  # minimization
                else:
                    # "lol" in neutral/positive = light tone marker, tiny positive nudge
                    v = v * 0.95 + 130 * 0.05
            if has_lmao:
                # lmao/lmfao = actually funny, genuine amusement
                v = v * 0.85 + 145 * 0.15
            if has_haha:
                # haha = genuine amusement, mild positive
                v = v * 0.9 + 138 * 0.1

        # Bravado detection (Layer 2 modifier): false confidence hides hurt
        # "I don't even care", "doesn't bother me" = D drops, V pulled negative
        # TCI insight: bravado is a coping mechanism — the real state is opposite
        if pre and pre.idiom_spans:
            for span_info in pre.idiom_spans.values():
                _length, _force, label = span_info
                if "bravado" in label:
                    # Pull V 15% toward negative — bravado masks real pain
                    v = v * 0.85 + (128 - 20) * 0.15
                    d -= 8  # bravado = significant loss of real agency
                    break

        # Crisis co-occurrence: multiple crisis-signal words in one sentence
        # Individual words are mild, but "pistol tonight end" = crisis
        # This catches fragmented Reddit posts that word-by-word processing misses
        if pre and pre.words_lower:
            _crisis_words = {
                # Method words (weight 2)
                'pistol', 'gun', 'shoot', 'rope', 'noose', 'hang', 'hanging',
                'blade', 'knife', 'slit', 'wrist', 'overdose', 'pills',
                'painkillers', 'bridge', 'jump', 'poison',
                # Intent words (weight 2)
                'suicide', 'suicidal', 'kill', 'die', 'dying', 'dead', 'death',
                'end', 'ending', 'quit', 'goodbye', 'farewell',
                # State words (weight 1) — only clearly crisis-adjacent words
                'tired', 'anymore', 'alone', 'worthless',
                'burden', 'pointless', 'hopeless', 'helpless', 'done',
                'tonight', 'tomorrow', 'ready', 'plan', 'planning', 'attempt',
                'final', 'last', 'note', 'letter', 'body',
            }
            _method_words = {'pistol', 'gun', 'shoot', 'rope', 'noose', 'hang',
                            'hanging', 'blade', 'knife', 'slit', 'wrist',
                            'overdose', 'pills', 'painkillers', 'bridge',
                            'jump', 'poison'}
            _intent_words = {'suicide', 'suicidal', 'kill', 'die', 'dying',
                            'dead', 'death', 'end', 'ending', 'goodbye',
                            'farewell', 'quit'}

            words_set = set(pre.words_lower)
            crisis_hits = words_set & _crisis_words
            has_method = bool(words_set & _method_words)
            has_intent = bool(words_set & _intent_words)

            _temporal_urgent = {'tonight', 'tomorrow', 'today', 'now', 'ready',
                                'soon', 'finally', 'last', 'final'}
            has_temporal = bool(words_set & _temporal_urgent)

            # Only apply if V isn't already clearly positive (> 135 = genuine positive)
            # and no double-negation (conviction pattern)
            v_not_positive = v < 135
            not_conviction = not (pre and pre.double_negation)
            if v_not_positive and not_conviction:
                if (len(crisis_hits) >= 3 or (has_method and has_intent)
                        or (has_method and has_temporal)):
                    # Strong crisis signal — pull V toward crisis zone
                    pull_strength = min(0.7, len(crisis_hits) * 0.15)
                    crisis_target = 50
                    v = v * (1 - pull_strength) + crisis_target * pull_strength
                    d = d * (1 - pull_strength * 0.5) + 60 * (pull_strength * 0.5)
                    u = max(u, 40 + len(crisis_hits) * 5)
                elif len(crisis_hits) >= 2 and has_intent:
                    # Moderate: 2+ crisis words including intent word
                    v = v * 0.8 + 90 * 0.2
                    u = max(u, 30)

        # Crisis detection: if deeply negative + high urgency, lock momentum
        if v < self.crisis_v and u > self.crisis_u:
            v = min(v, self.crisis_v)

        # Clamp to 0-255
        return VADUG(
            v=int(max(0, min(255, round(v)))),
            a=int(max(0, min(255, round(a)))),
            d=int(max(0, min(255, round(d)))),
            u=int(max(0, min(255, round(u)))),
            g=int(max(0, min(255, round(g)))),
        )

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Split text into words, preserving contractions and punctuation markers."""
        # Strip leading/trailing whitespace, collapse internal whitespace
        text = text.strip()
        if not text:
            return []
        # Split on whitespace, strip trailing punctuation except apostrophes/hyphens
        raw = text.split()
        words = []
        for w in raw:
            # Strip trailing punctuation including ? from word
            cleaned = w.lower().rstrip(".,!;:\")?")
            if cleaned:
                words.append(cleaned)
            # Add ? as separate token for question detection
            if w.endswith("?"):
                words.append("?")
        return words

    @staticmethod
    def _trace_entry(word: str, role: str, state: dict, note: str) -> dict:
        """Build a trace entry for debugging."""
        return {
            "word": word,
            "role": role,
            "v": round(state["v"], 1),
            "a": round(state["a"], 1),
            "d": round(state["d"], 1),
            "u": round(state["u"], 1),
            "g": round(state["g"], 1),
            "note": note,
        }


# ---------------------------------------------------------------------------
# Comparison with V1
# ---------------------------------------------------------------------------

def compare_with_v1(text: str):
    """Run both V1 and V2 engines on the same text, print comparison."""
    from demo.pendulum import SequentialPendulum

    # V2
    v2 = PendulumV2()
    vadug2, trace2 = v2.process_text(text)

    # V1
    v1 = SequentialPendulum()
    vadug1, _ = v1.process_text(text)

    print(f"\n  Text: \"{text}\"")
    print(f"  V1: {vadug1}  ({vadug1.describe()})")
    print(f"  V2: {vadug2}  ({vadug2.describe()})")
    diff = {
        "dV": vadug2.v - vadug1.v,
        "dA": vadug2.a - vadug1.a,
        "dD": vadug2.d - vadug1.d,
        "dU": vadug2.u - vadug1.u,
        "dG": vadug2.g - vadug1.g,
    }
    print(f"  Delta: {diff}")
    return vadug1, vadug2


# ---------------------------------------------------------------------------
# SST-2 Benchmark
# ---------------------------------------------------------------------------

def benchmark_sst2(max_n: int = 872):
    """Run SST-2 validation set and print accuracy for V2 engine."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: pip install datasets")
        return

    sst = load_dataset("stanfordnlp/sst2", split="validation")
    if max_n:
        sst = sst.select(range(min(max_n, len(sst))))

    engine = PendulumV2()
    correct = 0
    total = 0
    neutral_count = 0

    t0 = time.perf_counter()
    for row in sst:
        text = row["sentence"]
        truth = "positive" if row["label"] == 1 else "negative"
        vadug, _ = engine.process_text(text)

        # Classify based on valence using engine thresholds
        pred = engine.classify(vadug.v)
        if pred == "neutral":
            neutral_count += 1

        if pred == truth:
            correct += 1
        total += 1

    elapsed = time.perf_counter() - t0
    acc = correct / total * 100 if total else 0
    neut_pct = neutral_count / total * 100 if total else 0

    print(f"\n  SST-2 Benchmark ({total} samples)")
    print(f"  Accuracy:    {acc:.1f}%")
    print(f"  Neutral%:    {neut_pct:.1f}%")
    print(f"  Time:        {elapsed:.1f}s ({elapsed/total*1000:.1f}ms/sample)")
    print(f"  Momentum:    {engine.momentum}")
    return acc


# ---------------------------------------------------------------------------
# Standalone test suite
# ---------------------------------------------------------------------------

def _run_tests():
    """Test suite exercising all three passes."""
    engine = PendulumV2()

    print("=" * 70)
    print("  PendulumV2 — Clean Emotional Physics Engine")
    print("=" * 70)

    test_cases = [
        # (text, expected_direction, description)
        ("I am really sad", "negative", "Self + amplifier + negative payload"),
        ("I am happy", "positive", "Self + positive payload"),
        ("I am really not having a good day", "negative", "Negation flips 'good'"),
        ("This is a wonderful movie", "positive", "Positive adjective"),
        ("I hate this", "negative", "Direct negative statement"),
        ("Do you hate me?", "negative", "Question dampens but still negative"),
        ("I can't wait to see you", "positive", "Idiom: can't wait = positive"),
        ("I am fed up with everything", "negative", "Idiom: fed up = frustrated"),
        ("He was somewhat disappointed", "neutral", "Other-far + past + hedging — barely negative, correctly dampened"),
        ("I am extremely angry", "negative", "Self + amplifier + anger"),
        ("A boring film", "neutral", "Article dampens — boring is mild, 'a' reduces to observation"),
        ("My heart is broken", "negative", "Possessive self + broken"),
        ("They passed away last week", "negative", "Idiom: passed away"),
        ("I am never going to give up", "neutral", "Negation decays over distance — determination reads as mild positive/neutral"),
        ("Maybe I should be worried", "negative", "Hedge + hypothetical + negative"),
        # --- Negation-specific tests (continuous force) ---
        ("I am not happy", "negative", "Direct negation of positive"),
        ("I do not feel safe", "negative", "Negation passes through 'feel' to reach 'safe'"),
        ("I cannot believe this", "negative", "Cannot as negator"),
        ("This is not bad", "neutral", "Double negation: 'not bad' bigram → mild positive, dampened by 'this is' → neutral zone"),
        ("I'm not happy but I'm okay", "neutral", "Clause boundary stops negation — 'not happy' bigram + 'okay' after boundary balance out"),
        ("I don't think this is good", "negative", "Negation passes through weak payload 'think'"),
    ]

    passed = 0
    failed = 0

    for text, expected_dir, desc in test_cases:
        vadug, trace = engine.process_text(text)
        actual_dir = engine.classify(vadug.v)
        ok = (actual_dir == expected_dir)

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n  [{status}] {desc}")
        print(f"    Text:     \"{text}\"")
        print(f"    VADUG:    {vadug}")
        print(f"    Expected: {expected_dir}, Got: {actual_dir}")

        # Show trace
        for t in trace:
            role_tag = f"[{t['role']:10s}]"
            print(f"      {role_tag} {t['word']:15s} V={t['v']:5.1f} A={t['a']:5.1f} "
                  f"D={t['d']:5.1f} U={t['u']:5.1f} G={t['g']:5.1f}  {t['note']}")

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'=' * 70}")

    # V1 vs V2 comparison
    print(f"\n{'=' * 70}")
    print("  V1 vs V2 Comparison")
    print(f"{'=' * 70}")

    compare_texts = [
        "I am really sad",
        "This movie is absolutely wonderful",
        "I hate everything about this",
        "She was kind of disappointed",
        "Do you love me?",
        "I want to die",
        "I am fed up with this nonsense",
    ]

    for text in compare_texts:
        try:
            compare_with_v1(text)
        except Exception as e:
            print(f"\n  Text: \"{text}\"")
            print(f"  V1 comparison error: {e}")


if __name__ == "__main__":
    _run_tests()

    # Run SST-2 if --benchmark flag
    if "--benchmark" in sys.argv:
        max_n = 872
        for arg in sys.argv:
            if arg.startswith("--max="):
                max_n = int(arg.split("=")[1])
        benchmark_sst2(max_n)
