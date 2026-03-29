"""Emotional Zone Classification — match VADUG to named emotional states.

Instead of raw V thresholds, classify by which ZONE the coordinates
land in. Different sentences → same zone → same emotional state.

The zones are convergence regions: areas in 5D VADUG space where
structurally different sentences resolve to the same emotional meaning.

  "Whatever" → RESIGNATION zone
  "I give up" → RESIGNATION zone
  "Fine do what you want" → RESIGNATION zone

Different words. Same zone. Same state.

Usage:
    from demo.zones import ZoneClassifier
    zc = ZoneClassifier()
    result = zc.classify(vadug)
    print(result.zone)       # "RESIGNATION"
    print(result.confidence) # 0.85
"""

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

from demo.shared import VADUG


@dataclass
class ZoneResult:
    """Which emotional zone a VADUG lands in."""
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
    """Classify VADUG coordinates into named emotional zones."""

    def __init__(self):
        self.zones = ZONES

    def classify(self, vadug: VADUG) -> ZoneResult:
        """Find the closest emotional zone for a VADUG coordinate.

        Uses weighted Euclidean distance normalized by zone radius.
        Closer to center = higher confidence.
        """
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

            distances.append((zone_name, dist))

        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[1])

        best_zone, best_dist = distances[0]

        # Confidence: inverse of distance, clamped to 0-1
        # distance of 0 = confidence 1.0
        # distance of 2+ = confidence ~0
        confidence = max(0.0, min(1.0, 1.0 - best_dist * 0.4))

        # Alternatives: next closest zones
        alternatives = [(name, round(dist, 2)) for name, dist in distances[1:4]]

        return ZoneResult(
            zone=best_zone,
            confidence=round(confidence, 2),
            distance=round(best_dist, 2),
            alternatives=alternatives,
        )

    def describe(self, zone_name: str) -> str:
        """Get the description of a zone."""
        if zone_name in self.zones:
            return self.zones[zone_name]["description"]
        return "unknown zone"
