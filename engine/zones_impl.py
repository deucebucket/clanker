"""Emotional Zone Classification — match VADUGWI to named emotional states.

Instead of raw V thresholds, classify by which ZONE the coordinates
land in. Different sentences → same zone → same emotional state.

The zones are convergence regions: areas in 7D VADUGWI space where
structurally different sentences resolve to the same emotional meaning.

  "Whatever" → RESIGNATION zone
  "I give up" → RESIGNATION zone
  "Fine do what you want" → RESIGNATION zone

Different words. Same zone. Same state.

Usage:
    from engine.zones import ZoneClassifier
    zc = ZoneClassifier()
    result = zc.classify(vadug)
    print(result.zone)       # "RESIGNATION"
    print(result.confidence) # 0.85
"""

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

from .shared import VADUG


@dataclass
class ZoneResult:
    """Which emotional zone a VADUGWI state lands in."""
    zone: str               # JOY, RAGE, GRIEF, RESIGNATION, etc.
    confidence: float       # 0.0-1.0 how clearly it falls in this zone
    distance: float         # distance to zone center (lower = better match)
    alternatives: list      # other zones it's close to, sorted by distance


# Zone definitions: center point + radius for each dimension
# Derived from convergence analysis of real sentence clusters
ZONES = {
    "JOY": {
        "center": {"v": 156, "d": 146, "g": 137},
        "radius": {"v": 30, "d": 20, "g": 10},
        "description": "high V, high D (agency), light G",
    },
    "RAGE": {
        "center": {"v": 77, "d": 175, "g": 160},
        "radius": {"v": 35, "d": 45, "g": 30},
        "description": "low V, VERY high D (anger IS power), high G",
    },
    "GRIEF": {
        "center": {"v": 105, "d": 100, "g": 113},
        "radius": {"v": 25, "d": 25, "g": 15},
        "description": "moderate-low V, low D (helpless), heavy G",
    },
    "RESIGNATION": {
        "center": {"v": 120, "d": 117, "g": 124},
        "radius": {"v": 15, "d": 10, "g": 6},
        "description": "near-neutral V, consistently low D",
    },
    "ANXIETY": {
        "center": {"v": 101, "d": 93, "g": 134},
        "radius": {"v": 30, "d": 35, "g": 25},
        "description": "low V, low D, HIGH G (ungrounded/floating)",
    },
    "CRISIS": {
        "center": {"v": 81, "d": 82, "g": 89},
        "radius": {"v": 35, "d": 35, "g": 30},
        "description": "low everything — V, D, G all sinking",
    },
    "DEFLECTION": {
        "center": {"v": 124, "d": 122, "g": 128},
        "radius": {"v": 5, "d": 10, "g": 3},
        "description": "near-neutral EVERYTHING (the mask)",
    },
    "EMPOWERMENT": {
        "center": {"v": 149, "d": 131, "g": 131},
        "radius": {"v": 30, "d": 25, "g": 8},
        "description": "high V + moderate-high D (agency)",
    },
    "NEUTRAL": {
        "center": {"v": 128, "d": 128, "g": 128},
        "radius": {"v": 8, "d": 8, "g": 8},
        "description": "dead center — no signal",
    },
}


class ZoneClassifier:
    """Classify VADUGWI coordinates into named emotional zones."""

    def __init__(self):
        self.zones = ZONES

    # Structure patterns that override zone classification
    _CRISIS_PATTERNS = {
        "SELF_REMOVAL", "NO_EXIT", "SELF_NULLIFY", "METHOD_ACQUISITION",
        "SUSPICIOUS_CALM", "BLANKET_APOLOGY", "FAREWELL",
    }
    _NEGATIVE_PATTERNS = {
        "EXHAUSTION", "BETRAYAL", "VICTIMIZATION", "SARCASM_INVERSION",
        "BRAVADO", "CALLING_OUT", "DIRECTED_POSITIVE", "MINIMIZER",
        "EXCLUDED_POSITIVE", "POWER_OVER_SELF",
    }

    def classify(self, vadug: VADUG, structures=None) -> ZoneResult:
        """Find the closest emotional zone for a VADUGWI coordinate.

        Uses weighted Euclidean distance normalized by zone radius.
        Structures override when crisis or strong negative patterns fire.
        """
        # Structure override: if crisis pattern fires, force CRISIS zone
        if structures:
            pattern_names = {s.pattern for s in structures}
            crisis_hit = pattern_names & self._CRISIS_PATTERNS
            if crisis_hit:
                return ZoneResult(
                    zone="CRISIS",
                    confidence=0.85,
                    distance=0.5,
                    alternatives=[("ANXIETY", 1.0), ("GRIEF", 1.2)],
                )

            neg_hit = pattern_names & self._NEGATIVE_PATTERNS
            if neg_hit and vadug.v < 135:
                # Negative structure fired + V below positive threshold
                # Don't let it land in JOY/EMPOWERMENT
                pass  # fall through to distance calc but we'll bias below

        distances = []

        for zone_name, zone in self.zones.items():
            c = zone["center"]
            r = zone["radius"]

            # Normalized distance: how many radii away from center
            dv = abs(vadug.v - c["v"]) / max(r["v"], 1)
            dd = abs(vadug.d - c["d"]) / max(r["d"], 1)
            dg = abs(vadug.g - c["g"]) / max(r["g"], 1)

            # Weighted: V matters most, then D, then G
            dist = (dv * 0.4 + dd * 0.35 + dg * 0.25)

            # Penalize positive zones when negative structures fire
            if structures:
                pattern_names = {s.pattern for s in structures}
                neg_hit = pattern_names & self._NEGATIVE_PATTERNS
                if neg_hit and zone_name in ("JOY", "EMPOWERMENT", "NEUTRAL"):
                    dist += 2.0  # push away from positive zones

            distances.append((zone_name, dist))

        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[1])

        best_zone, best_dist = distances[0]

        # Confidence: inverse of distance, clamped to 0-1
        confidence = max(0.0, min(1.0, 1.0 - best_dist * 0.4))

        # Alternatives: next closest zones
        alternatives = [(name, round(dist, 2)) for name, dist in distances[1:4]]

        return ZoneResult(
            zone=best_zone,
            confidence=round(confidence, 2),
            distance=round(best_dist, 2),
            alternatives=alternatives,
        )

    def classify_cascading(self, vadug: VADUG) -> ZoneResult:
        """Cascading classification — precision first, then coverage.

        Level 1: Strong zone match (distance < 1.0) → high confidence
        Level 2: Near zone boundary (1.0-1.5) → medium confidence, check alternatives
        Level 3: No clear zone → return closest with low confidence + alternatives

        This gives precision when the signal is clear and coverage
        when it's ambiguous — without sacrificing either.
        """
        result = self.classify(vadug)

        if result.distance < 1.0:
            # Strong match — high confidence
            return result

        if result.distance < 1.5:
            # Near boundary — check if alternatives are close
            if result.alternatives and result.alternatives[0][1] < 1.0:
                # Alternative is also close — ambiguous, report both
                alt_name = result.alternatives[0][0]
                result.zone = f"{result.zone}/{alt_name}"
                result.confidence = max(0.0, result.confidence - 0.15)
            return result

        # No clear zone — low confidence
        result.confidence = max(0.0, result.confidence - 0.3)
        return result

    def is_negative_zone(self, zone: str, mode: str = "balanced") -> bool:
        """Check if a zone is negative, with configurable strictness.

        Modes:
            strict:   only CRISIS (highest precision, lowest recall)
            balanced: CRISIS + GRIEF (best accuracy)
            broad:    CRISIS + GRIEF + RESIGNATION + ANXIETY (high recall)
            safety:   everything except JOY, EMPOWERMENT, NEUTRAL (max recall)
        """
        strict = {"CRISIS"}
        balanced = {"CRISIS", "GRIEF"}
        broad = {"CRISIS", "GRIEF", "RESIGNATION", "ANXIETY"}
        safety = {"CRISIS", "GRIEF", "RESIGNATION", "ANXIETY", "RAGE", "DEFLECTION"}

        zones_map = {
            "strict": strict,
            "balanced": balanced,
            "broad": broad,
            "safety": safety,
        }

        check_zones = zones_map.get(mode, balanced)

        # Handle cascading dual-zone labels like "CRISIS/GRIEF"
        for part in zone.split("/"):
            if part in check_zones:
                return True
        return False

    def describe(self, zone_name: str) -> str:
        """Get the description of a zone."""
        if zone_name in self.zones:
            return self.zones[zone_name]["description"]
        return "unknown zone"
