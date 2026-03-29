"""Clanker Pipeline Simulator - modular package.

All public classes and functions are re-exported here so that
``from demo import SequentialPendulum, VADUG`` etc. still works.
"""

from .shared import VADUG, VADU, MetadataHeader, PersonalityVector
from .forces import WORD_FORCES
from .pendulum import (
    NEGATORS, INTENSIFIERS, IDIOMS, ANTICIPATION_PATTERNS,
    CONTEXT_MODIFIERS, SequentialPendulum, pendulum_parse,
    Ramp, RAMPS,
)
from .personality import apply_personality
from .response import (
    classify_metadata, compute_harmony, EMOTION_MAP, nearest_emotion,
    generate_clanker, decode_response, ResponseBuilder,
)
from .chunker import ChunkSplitter
from .grader import SentenceGrader
from .sarcasm import SarcasmDetector
from .tonal import TonalAnalyzer, apply_tonal_adjustment
from .intent import (
    IntentDetector,
    GREETING, QUESTION, REQUEST, VENTING, CASUAL, EMOTIONAL, STATEMENT,
    ALL_MODES, PROCESSING_PATHS,
)
from .nonsense import NonsenseDetector, NONSENSE, NON_EMOTIONAL, VALID
from .arc import ARC_CLOSERS, ChunkedPipeline, run_pipeline
from .fuzzy import fuzzy_match
from .pendulum_v2 import PendulumV2
from .classifier import AdaptiveClassifier
from .entropy import EntropyClassifier
from .ring_forces import (
    SARCASM_FORCES, COMEDY_FORCES, HYPERBOLE_FORCES, get_ring_force,
)
from .memory import TurnMemory, SessionMemory, ConversationMemory
from .multipath import MultiPathProcessor
from .bookend import BookendParser
from .word_roles import WordRoleDetector
from .bigrams import BigramDetector
from .trace import PipelineTrace

__all__ = [
    "VADUG", "VADU", "MetadataHeader", "PersonalityVector",
    "WORD_FORCES",
    "NEGATORS", "INTENSIFIERS", "RAMPS", "Ramp",
    "IDIOMS", "ANTICIPATION_PATTERNS",
    "CONTEXT_MODIFIERS", "SequentialPendulum", "pendulum_parse",
    "apply_personality",
    "classify_metadata", "compute_harmony", "EMOTION_MAP", "nearest_emotion",
    "generate_clanker", "decode_response", "ResponseBuilder",
    "ChunkSplitter",
    "SentenceGrader",
    "SarcasmDetector",
    "TonalAnalyzer", "apply_tonal_adjustment",
    "IntentDetector",
    "GREETING", "QUESTION", "REQUEST", "VENTING", "CASUAL", "EMOTIONAL", "STATEMENT",
    "ALL_MODES", "PROCESSING_PATHS",
    "NonsenseDetector", "NONSENSE", "NON_EMOTIONAL", "VALID",
    "ARC_CLOSERS", "ChunkedPipeline", "run_pipeline",
    "fuzzy_match",
    "PendulumV2",
    "AdaptiveClassifier",
    "EntropyClassifier",
    "SARCASM_FORCES", "COMEDY_FORCES", "HYPERBOLE_FORCES", "get_ring_force",
    "TurnMemory", "SessionMemory", "ConversationMemory",
    "BookendParser",
    "WordRoleDetector",
    "BigramDetector",
    "MultiPathProcessor",
    "PipelineTrace",
]
