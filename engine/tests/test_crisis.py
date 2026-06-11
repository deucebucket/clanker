"""Tests for the Crisis Layer -- CrisisTracker concern gradient.

Focus: structure patterns that the detector fires but were previously
absent from STRUCTURE_WEIGHTS (SUSPICIOUS_CALM, MASKING, RESIGNATION)
must raise concern on single crisis-shaped messages, while emotionally
similar but safe sentences stay near zero.

Thresholds: single-message concern for these patterns lands in the
0.21-0.36 range with the current weights; safe sentences land at or
below ~0.04. 0.20 / 0.10 gives margin on both sides without
fine-tuning to the exact engine output.
"""

import pytest

from engine.pendulum import compute_vadug
from engine.crisis import CrisisTracker, STRUCTURE_WEIGHTS


# ── Helper ───────────────────────────────────────────────────────

def _concern(sentence: str) -> float:
    """Single-message concern: score and state are the same reading."""
    vadug, meta = compute_vadug(sentence)
    structures = [m.pattern for m in meta.get("structures", [])]
    tracker = CrisisTracker()
    return tracker.read(vadug, vadug, structures).concern


CRISIS_THRESHOLD = 0.20   # these messages must register as concerning
SAFE_CEILING = 0.10       # safe messages must stay near zero


# ── Previously-discarded patterns must carry weight ──────────────

class TestStructureWeightsCoverage:
    """The three patterns must exist in STRUCTURE_WEIGHTS with sane tiers."""

    def test_suspicious_calm_weighted(self):
        # Pre-suicide calm: strongest of the three, below explicit intent.
        assert STRUCTURE_WEIGHTS["SUSPICIOUS_CALM"] > 0.0
        assert STRUCTURE_WEIGHTS["SUSPICIOUS_CALM"] < STRUCTURE_WEIGHTS["SELF_HARM_INTENT"]

    def test_masking_weighted(self):
        assert STRUCTURE_WEIGHTS["MASKING"] > 0.0

    def test_resignation_weighted(self):
        assert STRUCTURE_WEIGHTS["RESIGNATION"] > 0.0


# ── Crisis-shaped sentences raise concern ────────────────────────

class TestConcernRaised:

    def test_suspicious_calm_raises_concern(self):
        # Classic pre-suicide calm: decision made, peace declared.
        assert _concern("i feel calm now that ive decided") > CRISIS_THRESHOLD

    def test_masking_raises_concern(self):
        assert _concern("im tired of pretending im okay") > CRISIS_THRESHOLD

    def test_resignation_raises_concern(self):
        assert _concern("i just want it to be over") > CRISIS_THRESHOLD


# ── Safe sentences stay low ──────────────────────────────────────

class TestSafeStaysLow:

    def test_genuine_calm_stays_low(self):
        # Calm without a preceding "decision" must not look like
        # SUSPICIOUS_CALM.
        assert _concern("im calm and relaxed after yoga") < SAFE_CEILING

    def test_positive_decision_stays_low(self):
        # A decision about acquiring something good is not finality.
        assert _concern("i decided to take the job") < SAFE_CEILING

    def test_neutral_stays_low(self):
        assert _concern("the meeting is at three") < SAFE_CEILING
