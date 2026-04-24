"""Battleship probe system — fire calibrated probes, measure vibration, triangulate.

The idea: emotional states distort responses to neutral stimuli.
A person in crisis responds to "hmm okay" very differently than
a person in joy. The distortion IS the signal.

Fire known probes → measure how much the response deviates from
what a neutral person would produce → triangulate the hidden state.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .shared import VADUG
from .pendulum import compute_vadug
from .zones import ZoneClassifier, ZoneResult


# ── State transition (local until engine.solver exists) ────────

def state_transition(state_a: VADUG, state_b: VADUG) -> VADUG:
    """Combine two emotional states into a resulting state.

    Simple averaging model: the conversation "blends" the speaker's
    state (A) with the incoming probe (B). Deviations from neutral
    reveal the hidden state.
    """
    return VADUG(
        v=(state_a.v + state_b.v) // 2,
        a=(state_a.a + state_b.a) // 2,
        d=(state_a.d + state_b.d) // 2,
        u=(state_a.u + state_b.u) // 2,
        g=(state_a.g + state_b.g) // 2,
        w=(state_a.w + state_b.w) // 2,
        i=(state_a.i + state_b.i) // 2,
    )


# ── Probe dataclass ───────────────────────────────────────────

@dataclass
class Probe:
    name: str
    text: str
    vadug: VADUG = field(default_factory=VADUG)
    tests_for: List[str] = field(default_factory=list)


@dataclass
class ProbeResult:
    probe_name: str
    vibration: float          # average |actual - expected| across V, D, G
    estimated_zone: str       # zone classification of actual_c
    zone_confidence: float    # confidence of that classification
    actual_c: VADUG           # what actually happened
    expected_neutral_c: VADUG # what neutral would have produced


# ── Skeleton key probes ───────────────────────────────────────

_PROBE_DEFS = [
    ("minimal_ack", "hmm okay", ["CRISIS", "RAGE", "GRIEF"]),
    ("slight_validation", "that sounds tough", ["GRIEF", "CRISIS", "RESIGNATION"]),
    ("clarification", "what do you mean", ["DEFLECTION", "HEDGING"]),
    ("light_redirect", "well thats one way to look at it", ["SARCASM", "BRAVADO"]),
    ("direct_check", "are you okay", ["CRISIS", "MINIMIZATION", "BRAVADO"]),
]

PROBES: List[Probe] = []

for _name, _text, _tests in _PROBE_DEFS:
    _vadug, _ = compute_vadug(_text)
    PROBES.append(Probe(name=_name, text=_text, vadug=_vadug, tests_for=_tests))


# ── Neutral baseline ──────────────────────────────────────────

NEUTRAL = VADUG(128, 128, 128, 0, 128)


# ── Core functions ────────────────────────────────────────────

def fire_probe(probe: Probe, user_state_a: VADUG) -> ProbeResult:
    """Fire a single probe against a user state and measure vibration.

    Vibration = how much the actual result deviates from what a
    perfectly neutral person would have produced. High vibration
    means the hidden state is distorting the response.
    """
    expected_neutral_c = state_transition(NEUTRAL, probe.vadug)
    actual_c = state_transition(user_state_a, probe.vadug)

    # Vibration: average absolute deviation across V, D, G
    vibration = (
        abs(actual_c.v - expected_neutral_c.v)
        + abs(actual_c.d - expected_neutral_c.d)
        + abs(actual_c.g - expected_neutral_c.g)
    ) / 3.0

    # Classify the actual result
    zc = ZoneClassifier()
    zone_result = zc.classify(actual_c)

    return ProbeResult(
        probe_name=probe.name,
        vibration=vibration,
        estimated_zone=zone_result.zone,
        zone_confidence=zone_result.confidence,
        actual_c=actual_c,
        expected_neutral_c=expected_neutral_c,
    )


def triangulate(user_state: VADUG, num_probes: int = 3) -> dict:
    """Fire multiple probes and triangulate the hidden emotional state.

    Fires the first N probes, collects vibration results, and votes
    on the most likely zone weighted by vibration magnitude.

    Returns:
        estimated_zone: str — winning zone
        confidence: float — 0.0-1.0
        total_vibration: float — sum of all probe vibrations
        probe_results: list of ProbeResult
    """
    probes_to_fire = PROBES[:num_probes]
    results: List[ProbeResult] = []

    for probe in probes_to_fire:
        result = fire_probe(probe, user_state)
        results.append(result)

    # Weighted zone voting: each probe votes for zones it tests_for,
    # weighted by vibration magnitude
    zone_votes: dict = {}

    for i, result in enumerate(results):
        probe = probes_to_fire[i]
        for zone_name in probe.tests_for:
            if zone_name not in zone_votes:
                zone_votes[zone_name] = 0.0
            zone_votes[zone_name] += result.vibration

    # Also add votes from actual zone classifications
    for result in results:
        zone = result.estimated_zone
        if zone not in zone_votes:
            zone_votes[zone] = 0.0
        zone_votes[zone] += result.vibration * result.zone_confidence

    total_vibration = sum(r.vibration for r in results)

    if not zone_votes:
        return {
            "estimated_zone": "NEUTRAL",
            "confidence": 0.0,
            "total_vibration": total_vibration,
            "probe_results": results,
        }

    # Winner = zone with highest weighted vibration
    best_zone = max(zone_votes, key=zone_votes.get)
    best_score = zone_votes[best_zone]

    # Confidence: best score relative to total possible vibration
    # More vibration + more agreement = higher confidence
    max_possible = total_vibration * (num_probes + 1)  # probes + classification votes
    confidence = min(1.0, best_score / max(max_possible, 1.0))

    # Boost confidence if vibration is high (strong signal)
    if total_vibration > 30:
        confidence = min(1.0, confidence + 0.2)
    if total_vibration > 60:
        confidence = min(1.0, confidence + 0.2)

    return {
        "estimated_zone": best_zone,
        "confidence": round(confidence, 3),
        "total_vibration": round(total_vibration, 2),
        "probe_results": results,
    }
