"""V9 Pipeline — the full equation decomposition engine.

Pipeline: text → tokenize → decompose → compose → VADUGWI

Public API: compute_vadug(text) → (VADUG, trace_dict)
"""

from typing import Optional, Tuple
from engine_v9.shared import VADUG, PersonalityVector
from engine_v9.tokenizer import tokenize
from engine_v9.decomposer import decompose
from engine_v9.composer import compose


def compute_vadug(
    text: str,
    personality: Optional[PersonalityVector] = None,
    perspective: str = "speaker",
) -> Tuple[VADUG, dict]:
    """Compute VADUGWI coordinates for text using equation decomposition.

    Args:
        text: Input sentence/text
        personality: Optional personality vector (applied as final modifier)
        perspective: "speaker", "listener", or "bystander"

    Returns:
        (VADUG, trace_dict) where trace contains decomposition details
    """
    tokens = tokenize(text)
    equation = decompose(text)
    result = compose(equation)

    # Apply personality if provided
    if personality is not None:
        sensitivity = personality.emotional_sensitivity
        CENTER = 128
        result = VADUG(
            v=max(0, min(255, int(CENTER + (result.v - CENTER) * sensitivity))),
            a=max(0, min(255, int(CENTER + (result.a - CENTER) * sensitivity))),
            d=max(0, min(255, int(CENTER + (result.d - CENTER) * sensitivity + personality.dominance_baseline))),
            u=max(0, min(255, int(result.u * sensitivity))),
            g=max(0, min(255, int(CENTER + (result.g - CENTER) * sensitivity + personality.gravity_bias))),
            w=max(0, min(255, int(CENTER + (result.w - CENTER) * sensitivity))),
            i=result.i,
        )

    trace = {
        "tokens": tokens,
        "equation": {
            "subject": equation.subject.word,
            "nucleus": equation.nucleus.word,
            "nucleus_root": equation.nucleus.root.name,
            "nucleus_gravity": equation.nucleus.gravity,
            "operators": [a.word for a in equation.operators],
            "context": [a.word for a in equation.context],
        },
        "word_count": len(tokens),
    }

    return result, trace
