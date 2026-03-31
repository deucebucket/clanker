"""Bidirectional A+B=C Solver — forward read + backward zone targeting.

Forward:  text -> VADUG (wrapper around compute_vadug)
Backward: given A's state + target zone, solve for what B needs to be

The core idea: emotional states are composable. If someone is in state A
and you say B, the resulting state C = weighted_blend(A, B).

A persists at 60% (emotional inertia — you don't forget how you feel).
B has 40% influence (what was just said shifts the state, doesn't replace it).

The solver can also work backwards: given where A is and where you want C
to land (a target zone), sweep B's valence to find valid ranges.
"""

from typing import List, Optional, Tuple

from .shared import VADUG
from .pendulum import compute_vadug
from .zones import ZONES


# ── Forward ────────────────────────────────────────────────────────

def forward(text: str) -> VADUG:
    """Compute VADUG for a text string. Wrapper around compute_vadug."""
    result, _ = compute_vadug(text)
    return result


# ── State transition ───────────────────────────────────────────────

def state_transition(
    a_vadug: VADUG,
    b_vadug: VADUG,
    a_weight: float = 0.6,
) -> VADUG:
    """Compute resulting state C from A + B.

    A is the current emotional state (persists with a_weight).
    B is the new input's emotional reading (has 1 - a_weight influence).

    C = A * a_weight + B * (1 - a_weight), clamped to 0-255.
    """
    b_weight = 1.0 - a_weight
    return VADUG(
        v=int(round(a_vadug.v * a_weight + b_vadug.v * b_weight)),
        a=int(round(a_vadug.a * a_weight + b_vadug.a * b_weight)),
        d=int(round(a_vadug.d * a_weight + b_vadug.d * b_weight)),
        u=int(round(a_vadug.u * a_weight + b_vadug.u * b_weight)),
        g=int(round(a_vadug.g * a_weight + b_vadug.g * b_weight)),
        w=int(round(a_vadug.w * a_weight + b_vadug.w * b_weight)),
    )


# ── Backward: zone targeting ──────────────────────────────────────

def _in_zone(vadug: VADUG, zone_name: str) -> bool:
    """Check if VADUG falls within a zone's radius on V, D, G."""
    zone = ZONES[zone_name]
    c = zone["center"]
    r = zone["radius"]
    return (
        abs(vadug.v - c["v"]) <= r["v"]
        and abs(vadug.d - c["d"]) <= r["d"]
        and abs(vadug.g - c["g"]) <= r["g"]
    )


def solve_for_b_range(
    a_vadug: VADUG,
    target_zone: str,
    temperature_steps: int = 100,
) -> List[Tuple[int, int]]:
    """Sweep B's valence (0-255), return ranges where C lands in target zone.

    For each candidate B valence, construct a synthetic B with neutral A/D/U/G
    and compute C = state_transition(A, B). If C falls in the target zone,
    include that V value.

    Returns list of (start, end) inclusive ranges of valid B valence values.
    """
    valid = []
    # Use finer steps for better resolution, but always cover 0-255
    step = max(1, 256 // temperature_steps)

    for bv in range(0, 256, step):
        # Synthetic B: only V varies, rest neutral
        b = VADUG(v=bv, a=128, d=128, u=0, g=128, w=128)
        c = state_transition(a_vadug, b)
        if _in_zone(c, target_zone):
            valid.append(bv)

    # Collapse to contiguous ranges
    if not valid:
        return []

    ranges = []
    start = valid[0]
    prev = valid[0]
    for bv in valid[1:]:
        if bv - prev > step:
            ranges.append((start, prev))
            start = bv
        prev = bv
    ranges.append((start, prev))
    return ranges


def optimal_b_temperature(
    a_vadug: VADUG,
    target_zone: str,
) -> Optional[int]:
    """Find the optimal B valence to reach the target zone from A.

    Returns the midpoint of the widest valid range, or None if unreachable.
    """
    ranges = solve_for_b_range(a_vadug, target_zone, temperature_steps=256)
    if not ranges:
        return None

    # Find widest range
    widest = max(ranges, key=lambda r: r[1] - r[0])
    return (widest[0] + widest[1]) // 2
