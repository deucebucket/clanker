"""Clanker-LM: adaptive deterministic semantic conversation runtime."""

from .affect import AffectBackend, AffectController, ClankerAffectBackend, HeuristicAffectBackend
from .contracts import validate_public_api_contract, validate_runtime_instance
from .database import Atom, Construction, GrammarRule, LanguageStore, LearnedSenseRecord
from .learning import LearningOutcome, LexicalLearner, PendingDefinition
from .memory import ConversationMemory
from .model import *
from .parser import SemanticParser
from .qa import QuestionAnswerer
from .realize import SurfaceRealizer
from .resolvers import ResolverOutcome, ResolverRegistry, SafeArithmetic
from .runtime import ClankerLM
from .trajectory import CorpusProfile, CorpusProfiler, TrajectoryController

__version__ = "0.2.0"

# The executable contract is validated by ``ClankerLM.__init__`` rather than
# at module import.  Importing the package therefore remains safe when the
# optional V8 engine is not installed.

__all__ = [
    "ClankerLM",
    "ConversationMemory",
    "SemanticParser",
    "QuestionAnswerer",
    "SurfaceRealizer",
    "LanguageStore",
    "Atom",
    "GrammarRule",
    "Construction",
    "LearnedSenseRecord",
    "LexicalLearner",
    "LearningOutcome",
    "PendingDefinition",
    "ResolverRegistry",
    "ResolverOutcome",
    "SafeArithmetic",
    "TrajectoryController",
    "CorpusProfiler",
    "CorpusProfile",
    "AffectBackend",
    "AffectController",
    "ClankerAffectBackend",
    "HeuristicAffectBackend",
    "validate_public_api_contract",
    "validate_runtime_instance",
]
