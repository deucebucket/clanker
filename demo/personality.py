"""Personality filter for the Clanker pipeline."""

from .shared import VADUG, PersonalityVector

def apply_personality(response_vadug: VADUG, input_vadug: VADUG,
                      personality: PersonalityVector) -> tuple:
    """Apply personality vector as resistance weights on the response."""
    notes = []

    # High truthfulness prevents fake positivity
    if input_vadug.v < 70 and response_vadug.v > 170:
        truthfulness_resistance = personality.truthfulness / 255.0
        response_vadug.v = int(response_vadug.v - (response_vadug.v - 140) * truthfulness_resistance)
        notes.append(f"Truthfulness ({personality.truthfulness}) prevented fake positivity -> V{response_vadug.v}")

    # Low gullibility resists accepting extreme claims
    if input_vadug.u > 200:
        gull_factor = personality.gullibility / 255.0
        if gull_factor < 0.2:
            notes.append(f"Low gullibility ({personality.gullibility}) -> verifying urgency claim before full escalation")

    # Safety override for crisis (V < 30 and D < 30, or crushing gravity G < 30 with V < 50)
    if input_vadug.v < 30 and input_vadug.d < 30:
        if personality.safety > 150:
            response_vadug.d = max(response_vadug.d, 200)
            response_vadug.v = max(response_vadug.v, 100)
            response_vadug.g = max(response_vadug.g, 100)  # lift from crushing
            notes.append(f"SAFETY OVERRIDE ({personality.safety}): crisis detected -> max stability, warm tone, lifting gravity")

    # Crushing gravity crisis: G < 30 combined with V < 50 = severe crushing despair
    if input_vadug.g < 30 and input_vadug.v < 50:
        if personality.safety > 150:
            response_vadug.d = max(response_vadug.d, 200)
            response_vadug.v = max(response_vadug.v, 100)
            response_vadug.g = max(response_vadug.g, 120)  # lift significantly from crushing
            notes.append(f"GRAVITY CRISIS ({personality.safety}): crushing despair (G{input_vadug.g} V{input_vadug.v}) -> crisis response, lifting")

    # Assertiveness affects directness
    if personality.assertiveness > 150:
        response_vadug.d = max(response_vadug.d, 170)
        notes.append(f"High assertiveness ({personality.assertiveness}) -> confident tone")

    return response_vadug, notes
