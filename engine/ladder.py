"""The beat rung: score a longer text beat by beat, then fold with A+B=C.

compute_vadug() runs one pendulum over the whole input, so a multi-sentence
text is treated as one long sentence. compose_beats() splits the text into
beats, gives each beat a fresh pendulum, and folds the beat vectors left to
right with the same blend the turn rung uses (solver.blend_axes, A_WEIGHT
0.6): C = 0.6 * (everything so far) + 0.4 * (this beat).

This is opt-in. compute_vadug() does not call it, because on short chat
sentences it does not help (see docs/vadug-calculation.md, "Beat rung").
"""

import re
from typing import List, Tuple

from .pendulum import compute_vadug
from .shared import VADUG
from .solver import A_WEIGHT, blend_axes

MIN_BEAT_WORDS = 2

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
_CLAUSE_END = re.compile(
    r"(?<=[.!?;,:])\s+|\s+(?=(?:but|because|although|though|however)\b)",
    re.IGNORECASE,
)


def split_beats(text: str, clauses: bool = False) -> List[str]:
    """Split at sentence punctuation; with clauses=True also at commas,
    colons and before but / because / although / though / however.
    Pieces shorter than MIN_BEAT_WORDS are dropped. A text with no usable
    split comes back whole."""
    pattern = _CLAUSE_END if clauses else _SENTENCE_END
    beats = [piece.strip() for piece in pattern.split(text)]
    beats = [beat for beat in beats if len(beat.split()) >= MIN_BEAT_WORDS]
    return beats or [text]


def compose_beats(
    text: str,
    clauses: bool = False,
    a_weight: float = A_WEIGHT,
    perspective: str = "speaker",
) -> Tuple[VADUG, dict]:
    """Score each beat with a fresh pendulum and fold the results.

    Returns (VADUG, meta). meta has the same keys as compute_vadug's trace
    dict (trace and structures are concatenated over beats, so pattern
    based checks still see every match) plus "beats": [(text, VADUG)].
    """
    beats = split_beats(text, clauses=clauses)
    scored = [(beat, *compute_vadug(beat, perspective=perspective)) for beat in beats]

    state = scored[0][1]
    axes = (state.v, state.a, state.d, state.u, state.g, state.w, state.i)
    for _, vadug, _ in scored[1:]:
        current = VADUG(*(int(round(x)) for x in axes))
        axes = blend_axes(current, vadug, a_weight)
    result = VADUG(*(int(round(x)) for x in axes))

    meta = {
        "trace": [entry for _, _, m in scored for entry in m.get("trace", [])],
        "structures": [s for _, _, m in scored for s in m.get("structures", [])],
        "force_flow": scored[-1][2].get("force_flow"),
        "word_count": sum(m.get("word_count", 0) for _, _, m in scored),
        "beats": [(beat, vadug) for beat, vadug, _ in scored],
    }
    return result, meta
