"""Clanker Pipeline — V2 modular package.

V1 (SequentialPendulum, WORD_FORCES) has been archived to archive/v1/.
All imports now use V2 engine (PendulumV2, EMOTIONAL_VOCABULARY).
"""

from .shared import VADUG, VADU, MetadataHeader, PersonalityVector
from .forces_curated import EMOTIONAL_VOCABULARY
from .pendulum_v2 import PendulumV2
from .personality import apply_personality
from .response import (
    classify_metadata, compute_harmony, EMOTION_MAP, nearest_emotion,
    ResponseBuilder,
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
from .fuzzy import fuzzy_match
from .classifier import AdaptiveClassifier
from .entropy import EntropyClassifier
from .memory import TurnMemory, SessionMemory, ConversationMemory
from .bookend import BookendParser
from .word_roles import WordRoleDetector
from .bigrams import BigramDetector
from .idioms import IDIOMS
from .context_operators import CONTEXT_OPERATORS

__all__ = [
    "VADUG", "VADU", "MetadataHeader", "PersonalityVector",
    "EMOTIONAL_VOCABULARY", "IDIOMS", "CONTEXT_OPERATORS",
    "PendulumV2",
    "apply_personality",
    "classify_metadata", "compute_harmony", "EMOTION_MAP", "nearest_emotion",
    "ResponseBuilder",
    "ChunkSplitter", "SentenceGrader", "SarcasmDetector",
    "TonalAnalyzer", "apply_tonal_adjustment",
    "IntentDetector",
    "GREETING", "QUESTION", "REQUEST", "VENTING", "CASUAL", "EMOTIONAL", "STATEMENT",
    "ALL_MODES", "PROCESSING_PATHS",
    "NonsenseDetector", "NONSENSE", "NON_EMOTIONAL", "VALID",
    "fuzzy_match",
    "AdaptiveClassifier", "EntropyClassifier",
    "TurnMemory", "SessionMemory", "ConversationMemory",
    "BookendParser", "WordRoleDetector", "BigramDetector",
]
