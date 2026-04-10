"""
V9 Root definitions — emotional atoms with 7D charge vectors.

Each root is a named emotional atom. Multiple English surface words map to
the same root. The charge tuple (dV, dA, dD, dU, dG, dW, dI) represents
deltas from neutral center (128), not absolute values.

Charge tuple index map:
  0=dV  Valence      (positive/negative affect)
  1=dA  Arousal      (activation/energy)
  2=dD  Dominance    (power/control)
  3=dU  Urgency      (time pressure)
  4=dG  Gravity      (weight/consequence)
  5=dW  Self-Worth   (self-esteem impact)
  6=dI  Intent       (withdraw=-,deflect=~,neutral=0,connect=+,control=++)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple


class RootCategory(Enum):
    # ── Core emotional states ───────────────────────────────────────────
    POSITIVE_STATE = auto()
    NEGATIVE_STATE = auto()
    POSITIVE_QUALITY = auto()
    NEGATIVE_QUALITY = auto()
    POSITIVE_EVENT = auto()
    NEGATIVE_EVENT = auto()
    SOCIAL_EVAL_POS = auto()
    SOCIAL_EVAL_NEG = auto()
    EXPECTATION_VIOLATION = auto()

    # ── Domain-specific (same polarity math, different context weight) ──
    FINANCIAL_POS = auto()      # approved, funded, raise, bonus, profit
    FINANCIAL_NEG = auto()      # debt, bankrupt, fired, overdue, broke
    MEDICAL_POS = auto()        # healed, cleared, remission, recovered
    MEDICAL_NEG = auto()        # diagnosed, terminal, relapsed, cancer
    ACADEMIC_POS = auto()       # graduated, passed, honors, scholarship
    ACADEMIC_NEG = auto()       # failed, expelled, dropped, flunked
    LEGAL_POS = auto()          # acquitted, pardoned, cleared, innocent
    LEGAL_NEG = auto()          # arrested, convicted, sentenced, sued
    CAREER_POS = auto()         # promoted, hired, raise, recognized
    CAREER_NEG = auto()         # fired, demoted, laid off, passed over
    RELATIONSHIP_POS = auto()   # engaged, married, matched, connected
    RELATIONSHIP_NEG = auto()   # divorced, dumped, ghosted, rejected

    # ── Actions ─────────────────────────────────────────────────────────
    MOTION = auto()
    POSSESSION = auto()
    TRANSFER = auto()
    PERCEPTION = auto()
    COMMUNICATION = auto()
    COGNITION = auto()

    # ── Entities ────────────────────────────────────────────────────────
    SELF_REF = auto()
    OTHER_REF = auto()
    RELATION_REF = auto()
    PERSON = auto()
    OBJECT = auto()

    # ── Operators ───────────────────────────────────────────────────────
    INTENSIFIER = auto()
    NEGATOR = auto()
    CONNECTOR = auto()
    HEDGE = auto()
    COMPRESSOR = auto()
    TEMPORAL = auto()

    # ── Special ─────────────────────────────────────────────────────────
    FORMULAIC = auto()
    COMPOUND_EVENT = auto()
    FILLER = auto()


Charge = Tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True)
class Root:
    name: str
    category: RootCategory
    charge: Charge  # (dV, dA, dD, dU, dG, dW, dI)
    phase: str = "GAS"  # GAS | LIQUID | SOLID


def _r(name: str, cat: RootCategory, charge: Charge, phase: str = "GAS") -> Root:
    """Helper to build a Root with validated charge length."""
    assert len(charge) == 7, f"Root {name}: charge must be 7D, got {len(charge)}D"
    return Root(name=name, category=cat, charge=charge, phase=phase)


P = RootCategory.POSITIVE_STATE
N = RootCategory.NEGATIVE_STATE
PQ = RootCategory.POSITIVE_QUALITY
NQ = RootCategory.NEGATIVE_QUALITY
PE = RootCategory.POSITIVE_EVENT
NE = RootCategory.NEGATIVE_EVENT
SPP = RootCategory.SOCIAL_EVAL_POS
SPN = RootCategory.SOCIAL_EVAL_NEG
EV = RootCategory.EXPECTATION_VIOLATION
CE = RootCategory.COMPOUND_EVENT
F = RootCategory.FORMULAIC
FL = RootCategory.FILLER

ROOTS: dict[str, Root] = {

    # ── Positive states ──────────────────────────────────────────────────────
    "HAPPY":     _r("HAPPY",     P,  ( 23, -5,  11,   0,  17,   7,   0)),
    "EXCITED":   _r("EXCITED",   P,  ( 40, 60,  20,  15,  30,  10,   0)),
    "GRATEFUL":  _r("GRATEFUL",  P,  ( 45,-10,  -5,   0,  20,  20,  30)),
    "RELIEVED":  _r("RELIEVED",  P,  ( 35,-30,  10, -20,  30,  15,   0)),
    "PROUD":     _r("PROUD",     P,  ( 40, 10,  30,   0,  25,  35,   0)),
    "CONTENT":   _r("CONTENT",   P,  ( 25,-20,  10,   0,  15,  15,   0)),
    "AMUSED":    _r("AMUSED",    P,  ( 25, 20,  15,   0,  20,   5,   0)),
    "HOPEFUL":   _r("HOPEFUL",   P,  ( 30, 10,   5,  10,  20,  15,  20)),
    "LOVED":     _r("LOVED",     P,  ( 50, 10,   5,   0,  25,  40,  30), phase="SOLID"),

    # ── Negative states ──────────────────────────────────────────────────────
    "SAD":         _r("SAD",         N,  (-35,-10, -20,   0, -25, -15,   0)),
    "ANGRY":       _r("ANGRY",       N,  (-80,100,  80,  60,  50,   0,  50)),
    "AFRAID":      _r("AFRAID",      N,  (-60, 50, -80,  70,  20, -20, -30)),
    "DISGUSTED":   _r("DISGUSTED",   N,  (-70, 50,  30,  20,  10,   0, -20)),
    "ASHAMED":     _r("ASHAMED",     N,  (-80, 60,-100,  30, -50, -60, -40)),
    "LONELY":      _r("LONELY",      N,  (-50,-20, -40,   0, -30, -30, -30)),
    "ANXIOUS":     _r("ANXIOUS",     N,  (-50, 80,-100,  90,  20, -20,   0)),
    "GUILTY":      _r("GUILTY",      N,  (-70, 40, -80,  30, -40, -50,   0)),
    "JEALOUS":     _r("JEALOUS",     N,  (-60, 70, -30,  40,  20, -40,  30)),
    "FRUSTRATED":  _r("FRUSTRATED",  N,  (-50, 80, -20,  50,  30, -10,   0)),
    "EXHAUSTED":   _r("EXHAUSTED",   N,  (-30,-60, -40,   0, -40, -15, -20)),
    "DEVASTATED":  _r("DEVASTATED",  N,  (-100,80,-100,  60, -70, -50, -30), phase="SOLID"),
    "DESPAIR":     _r("DESPAIR",     N,  (-110,40,-120,  50, -80, -60, -40), phase="SOLID"),

    # ── Positive qualities ───────────────────────────────────────────────────
    "POS_QUALITY": _r("POS_QUALITY", PQ, ( 30,  0,  10,   0,  10,   6,   0)),
    "BEAUTIFUL":   _r("BEAUTIFUL",   PQ, ( 35, 10,   5,   0,  20,   5,   0)),
    "STRONG":      _r("STRONG",      PQ, ( 20, 15,  40,   0,  15,  20,   0)),

    # ── Negative qualities ───────────────────────────────────────────────────
    "NEG_QUALITY": _r("NEG_QUALITY", NQ, (-19,  0,  -6,   0,  -6,  -3,   0)),
    "UGLY":        _r("UGLY",        NQ, (-30, 10, -10,   0, -15, -15,   0)),
    "WEAK":        _r("WEAK",        NQ, (-15,-10, -40,   0, -15, -25,   0)),

    # ── Positive events ──────────────────────────────────────────────────────
    "ACHIEVEMENT": _r("ACHIEVEMENT", PE, ( 50, 30,  40,   0,  30,  40,   0)),
    "HEALING":     _r("HEALING",     PE, ( 40,-10,  15, -10,  25,  20,  20)),

    # ── Negative events ──────────────────────────────────────────────────────
    "LOSS":      _r("LOSS",     NE, (-60, 30, -50,  20, -40, -30, -20)),
    "HARM":      _r("HARM",     NE, (-80, 80, -60,  70, -20, -40,   0), phase="SOLID"),
    "CRISIS":    _r("CRISIS",   NE, (-100,100,-100, 100, -60, -50, -40), phase="SOLID"),
    "BETRAYAL":  _r("BETRAYAL", NE, (-90, 80, -70,  50, -30, -60, -40), phase="SOLID"),
    "MURDER":    _r("MURDER",   NE, (-80, 80, -60,  70, -20, -40,   0), phase="SOLID"),

    # ── Social evaluation ────────────────────────────────────────────────────
    "SOC_EVAL_POS": _r("SOC_EVAL_POS", SPP, ( 20,  0,  15,   0,  10,  15,  15)),
    "SOC_EVAL_NEG": _r("SOC_EVAL_NEG", SPN, (-20, 20, -15,  10, -10, -20, -15)),

    # ── Expectation violation ────────────────────────────────────────────────
    "SURPRISE": _r("SURPRISE", EV, (  0, 60, -30,  30,  10,   0,   0)),

    # ── Actions ──────────────────────────────────────────────────────────────
    "MOTION":      _r("MOTION",      RootCategory.MOTION,        (  0,  5,   0,   0,   0,   0,   0)),
    "POSSESS":     _r("POSSESS",     RootCategory.POSSESSION,    (  0,  0,   5,   0,   0,   0,   0)),
    "TRANSFER":    _r("TRANSFER",    RootCategory.TRANSFER,      (  0,  0,  -5,   0,   0,   0,   0)),
    "PERCEIVE":    _r("PERCEIVE",    RootCategory.PERCEPTION,    (  0,  5,   0,   0,   0,   0,   0)),
    "COMMUNICATE": _r("COMMUNICATE", RootCategory.COMMUNICATION, (  0,  5,   0,   0,   0,   0,   0)),
    "THINK":       _r("THINK",       RootCategory.COGNITION,     (  0,  0,   5,   0,   0,   0,   0)),

    # ── Entities ─────────────────────────────────────────────────────────────
    "SELF":           _r("SELF",           RootCategory.SELF_REF,   (0,0,0,0,0,0,0)),
    "OTHER":          _r("OTHER",          RootCategory.OTHER_REF,  (0,0,0,0,0,0,0)),
    "RELATION":       _r("RELATION",       RootCategory.RELATION_REF,(0,0,0,0,0,0,0)),
    "PERSON_GENERIC": _r("PERSON_GENERIC", RootCategory.PERSON,     (0,0,0,0,0,0,0)),
    "OBJECT_GENERIC": _r("OBJECT_GENERIC", RootCategory.OBJECT,     (0,0,0,0,0,0,0)),

    # ── Operators ────────────────────────────────────────────────────────────
    "INTENSIFY":  _r("INTENSIFY",  RootCategory.INTENSIFIER, (  0, 15,   0,   5,   0,   0,   0)),
    "NEGATE":     _r("NEGATE",     RootCategory.NEGATOR,     (  0,  0,   0,   0,   0,   0,   0)),
    "CONNECT":    _r("CONNECT",    RootCategory.CONNECTOR,   (  0,  0,   0,   0,   0,   0,   0)),
    "HEDGE_OP":   _r("HEDGE_OP",   RootCategory.HEDGE,       (  0, -5,   0,   0,   0,   0,   0)),
    "COMPRESS":   _r("COMPRESS",   RootCategory.COMPRESSOR,  (  0,-10,   0,   0,   0,   0,   0)),
    "TIME":       _r("TIME",       RootCategory.TEMPORAL,    (  0,  0,   0,   0,   0,   0,   0)),

    # ── Formulaic ────────────────────────────────────────────────────────────
    "GREETING": _r("GREETING", F, (0,0,0,0,0,0,0)),
    "THANKS":   _r("THANKS",   F, (0,0,0,0,0,0,0)),
    "APOLOGY":  _r("APOLOGY",  F, (0,0,0,0,0,0,0)),

    # ── Filler ───────────────────────────────────────────────────────────────
    "FILLER_WORD": _r("FILLER_WORD", FL, (0,0,0,0,0,0,0)),

    # ── Mild emotional roots (for words with |dV| 15-30) ────────────────────
    # These prevent 5x amplification when a dV=-5 word gets rounded to dV=-25
    "SLIGHT_POS":  _r("SLIGHT_POS",  RootCategory.POSITIVE_QUALITY, (+12,  0,  +5,   0,  +5,  +3,   0)),
    "SLIGHT_NEG":  _r("SLIGHT_NEG",  RootCategory.NEGATIVE_QUALITY, (-12,  0,  -5,   0,  -5,  -3,   0)),

    # ── Domain: Financial ────────────────────────────────────────────────────
    "FIN_POS":     _r("FIN_POS",     RootCategory.FINANCIAL_POS,  (+30,  +10, +25,   0, +20, +20,   0)),
    "FIN_NEG":     _r("FIN_NEG",     RootCategory.FINANCIAL_NEG,  (-35,  +20, -30, +30, -25, -25, -10)),

    # ── Domain: Medical ──────────────────────────────────────────────────────
    "MED_POS":     _r("MED_POS",     RootCategory.MEDICAL_POS,    (+40, -10, +20, -15, +30, +25, +15)),
    "MED_NEG":     _r("MED_NEG",     RootCategory.MEDICAL_NEG,    (-50, +30, -40, +40, -35, -30, -20), phase="SOLID"),

    # ── Domain: Academic ─────────────────────────────────────────────────────
    "ACAD_POS":    _r("ACAD_POS",    RootCategory.ACADEMIC_POS,   (+30, +15, +25,   0, +15, +25,   0)),
    "ACAD_NEG":    _r("ACAD_NEG",    RootCategory.ACADEMIC_NEG,   (-30, +20, -25, +15, -15, -30, -10)),

    # ── Domain: Legal ────────────────────────────────────────────────────────
    "LEGAL_POS":   _r("LEGAL_POS",   RootCategory.LEGAL_POS,      (+35, -10, +30, -20, +20, +20, +10)),
    "LEGAL_NEG":   _r("LEGAL_NEG",   RootCategory.LEGAL_NEG,      (-40, +40, -50, +50, -20, -25, -15), phase="SOLID"),

    # ── Domain: Career ───────────────────────────────────────────────────────
    "CAREER_POS":  _r("CAREER_POS",  RootCategory.CAREER_POS,     (+35, +15, +30,   0, +20, +30,   0)),
    "CAREER_NEG":  _r("CAREER_NEG",  RootCategory.CAREER_NEG,     (-40, +25, -35, +25, -25, -35, -15)),

    # ── Domain: Relationship ─────────────────────────────────────────────────
    "REL_POS":     _r("REL_POS",     RootCategory.RELATIONSHIP_POS,(+40, +15, +10,   0, +20, +25, +25)),
    "REL_NEG":     _r("REL_NEG",     RootCategory.RELATIONSHIP_NEG,(-45, +30, -20, +20, -25, -35, -25)),

    # ── Compound events ──────────────────────────────────────────────────────
    "EMPLOYMENT_LOSS":   _r("EMPLOYMENT_LOSS",   CE, (-60, 40, -60,  50, -40, -30, -20), phase="SOLID"),
    "MEDICAL_RELIEF":    _r("MEDICAL_RELIEF",    CE, ( 55,-10,  30, -20,  40,  30,  20), phase="SOLID"),
    "RELATIONSHIP_END":  _r("RELATIONSHIP_END",  CE, (-55, 50, -30,  30, -30, -40, -30), phase="SOLID"),
    "DEATH_EUPHEMISM":   _r("DEATH_EUPHEMISM",   CE, (-65, 20, -50,  20, -50, -20, -20), phase="SOLID"),
    "BREAKDOWN":         _r("BREAKDOWN",         CE, (-45, 40, -40,  30, -30, -20, -10)),
    "RECOVERY":          _r("RECOVERY",          CE, ( 40, 10,  20, -10,  25,  20,  15)),
    "BURNOUT":           _r("BURNOUT",            CE, (-40,-30, -50,  20, -35, -25, -20)),
    "ABANDONMENT":       _r("ABANDONMENT",       CE, (-50, 40, -40,  30, -30, -40, -30), phase="SOLID"),
    "SURRENDER":         _r("SURRENDER",         CE, (-35, -10, -50,  10, -20, -20, -30)),
    "FAILURE":           _r("FAILURE",           CE, (-40, 30, -40,  20, -25, -35, -15)),
    "LETDOWN":           _r("LETDOWN",           CE, (-35, 20, -30,  15, -20, -20, -15)),
    "MOVING_ON":         _r("MOVING_ON",         CE, (-20, 10, +10,  10, -10, -10,  +10)),
}
