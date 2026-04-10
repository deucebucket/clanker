"""Personality system — V5.5."""
from .shared import PersonalityVector

CENTER = 128.0


def apply_personality(context: dict) -> dict:
    """Scale state by personality vector if provided.

    Reads: context["state_*"], context.get("personality")
    Writes: context["state_*"]
    """
    personality = context.get("personality")
    if personality is None:
        return context

    state_v = context["state_v"]
    state_a = context["state_a"]
    state_d = context["state_d"]
    state_u = context["state_u"]
    state_g = context["state_g"]
    state_w = context["state_w"]

    sensitivity = personality.emotional_sensitivity
    state_v = CENTER + (state_v - CENTER) * sensitivity
    state_a = CENTER + (state_a - CENTER) * sensitivity
    state_d = CENTER + (state_d - CENTER) * sensitivity + personality.dominance_baseline
    state_u = state_u * sensitivity
    state_g = CENTER + (state_g - CENTER) * sensitivity + personality.gravity_bias
    state_w = CENTER + (state_w - CENTER) * sensitivity

    context["state_v"] = state_v
    context["state_a"] = state_a
    context["state_d"] = state_d
    context["state_u"] = state_u
    context["state_g"] = state_g
    context["state_w"] = state_w
    return context
