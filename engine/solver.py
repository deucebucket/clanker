"""Bidirectional A+B=C Solver — forward read + backward zone targeting.

Forward:  text -> VADUGWI (wrapper around compute_vadug)
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
    """Compute VADUGWI for a text string. Wrapper around compute_vadug."""
    result, _ = compute_vadug(text)
    return result


# ── State transition ───────────────────────────────────────────────

def state_transition(
    a_vadug: VADUG,
    b_vadug: VADUG,
    a_weight: float = 0.6,
) -> VADUG:
    """Compute resulting state C from A (receiver state) + B (the force/message).

    Base: C = A * a_weight + B * (1 - a_weight), clamped to 0-255.

    FORCE DIRECTION adjustments:
    - B with CONTROL intent (I>200) + negative V = attack on receiver.
      Receiver's D drops (dominated), W drops (worth under attack).
      The force doesn't blend W/D toward B's values -- it PUSHES them down.
    - B with CONNECT intent (I>155) + positive V = support/healing.
      Receiver's W gets lifted, D stabilized.
    - B with WITHDRAW intent (I<40) = the sender is pulling away.
      Receiver's G increases (heavier), connection fading.
    """
    b_weight = 1.0 - a_weight
    CENTER = 128.0

    # Base blend
    c_v = a_vadug.v * a_weight + b_vadug.v * b_weight
    c_a = a_vadug.a * a_weight + b_vadug.a * b_weight
    c_d = a_vadug.d * a_weight + b_vadug.d * b_weight
    c_u = a_vadug.u * a_weight + b_vadug.u * b_weight
    c_g = a_vadug.g * a_weight + b_vadug.g * b_weight
    c_w = a_vadug.w * a_weight + b_vadug.w * b_weight
    c_i = a_vadug.i * a_weight + b_vadug.i * b_weight

    # Force direction adjustments on the RECEIVER
    b_v = b_vadug.v
    b_i = b_vadug.i

    # Attack: negative V directed outward OR pure CONTROL command
    if b_v < 125 and b_i > 60:
        # Negative force aimed outward = attack on receiver
        attack_strength = (128 - b_v) / 128.0  # 0 to 1
        control_boost = max(0, (b_i - 128)) / 127.0  # 0 to 1
        total = attack_strength * (1.0 + control_boost)
        c_d -= total * 15
        c_w -= total * 15
        c_v -= total * 5

    # Pure CONTROL command (high I) -- even if V is only mildly negative
    # "shut up" has high D force but the RECEIVER loses D (being commanded)
    if b_i > 200:
        control_strength = (b_i - 200) / 55.0  # 0 to 1
        c_d -= control_strength * 12  # being controlled drops YOUR D
        # Override the D blend -- receiver doesn't gain power from being commanded
        if b_vadug.d > CENTER:
            # B has high D (commander's power) but that shouldn't transfer to receiver
            d_excess = (c_d - a_vadug.d * a_weight) * 0.3  # dampen the D transfer
            c_d = a_vadug.d * a_weight + d_excess

    # Healing: CONNECT intent + positive V
    elif b_v > 135 and b_i > 155:
        heal_strength = (b_v - 128) / 127.0
        connect_boost = (b_i - 128) / 127.0
        total = heal_strength * (1.0 + connect_boost * 0.5)
        c_w += total * 12  # W lifted (feeling valued)
        c_d += total * 8   # D stabilized (someone has their back)

    # Withdraw: sender pulling away from receiver
    elif b_i < 40:
        withdraw_strength = (40 - b_i) / 40.0
        c_g -= withdraw_strength * 10  # heavier (weight of abandonment)
        c_d -= withdraw_strength * 8   # loss of relational support
        c_w -= withdraw_strength * 5   # worth hit from being left

    # Deflect/dismiss: mild negative, disengaged
    elif b_i > 60 and b_i < 120 and b_v < 128:
        dismiss_strength = (128 - b_v) / 128.0
        c_w -= dismiss_strength * 8  # mild worth hit from being dismissed

    return VADUG(
        v=int(round(max(0, min(255, c_v)))),
        a=int(round(max(0, min(255, c_a)))),
        d=int(round(max(0, min(255, c_d)))),
        u=int(round(max(0, min(255, c_u)))),
        g=int(round(max(0, min(255, c_g)))),
        w=int(round(max(0, min(255, c_w)))),
        i=int(round(max(0, min(255, c_i)))),
    )


# ── Backward: zone targeting ──────────────────────────────────────

def _in_zone(vadug: VADUG, zone_name: str) -> bool:
    """Check if VADUGWI state falls within a zone's radius on V, D, G."""
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
        b = VADUG(v=bv, a=128, d=128, u=0, g=128, w=128, i=128)
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
