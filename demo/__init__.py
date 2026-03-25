"""Clanker Pipeline Simulator - modular package.

All public classes and functions are re-exported here so that
``from demo import SequentialPendulum, VADUG`` etc. still works.
"""

from .shared import VADUG, VADU, MetadataHeader, PersonalityVector
from .forces import WORD_FORCES
from .pendulum import (
    NEGATORS, INTENSIFIERS, IDIOMS, ANTICIPATION_PATTERNS,
    CONTEXT_MODIFIERS, SequentialPendulum, pendulum_parse,
)
from .personality import apply_personality
from .response import (
    classify_metadata, compute_harmony, EMOTION_MAP, nearest_emotion,
    generate_clanker, decode_response, ResponseBuilder,
)
from .chunker import ChunkSplitter
from .grader import SentenceGrader
from .sarcasm import SarcasmDetector
from .arc import ARC_CLOSERS, ChunkedPipeline, run_pipeline

__all__ = [
    "VADUG", "VADU", "MetadataHeader", "PersonalityVector",
    "WORD_FORCES",
    "NEGATORS", "INTENSIFIERS", "IDIOMS", "ANTICIPATION_PATTERNS",
    "CONTEXT_MODIFIERS", "SequentialPendulum", "pendulum_parse",
    "apply_personality",
    "classify_metadata", "compute_harmony", "EMOTION_MAP", "nearest_emotion",
    "generate_clanker", "decode_response", "ResponseBuilder",
    "ChunkSplitter",
    "SentenceGrader",
    "SarcasmDetector",
    "ARC_CLOSERS", "ChunkedPipeline", "run_pipeline",
]
