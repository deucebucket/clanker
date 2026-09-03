"""Deterministic semantic dialogue layer built around the Clanker engine."""

from .affect import ClankerAffectAdapter, RuleAffectAdapter
from .memory import ConversationMemory
from .models import (
    AffectReading,
    AffectVector,
    AnswerContract,
    AnswerStatus,
    Entity,
    Fact,
    GateProfile,
    ParsedUtterance,
    Provenance,
    QuestionFrame,
    SemanticFrame,
    SemanticRole,
    TurnResult,
)
from .runtime import ClankerLM

__all__ = [
    "AffectReading",
    "AffectVector",
    "AnswerContract",
    "AnswerStatus",
    "ClankerAffectAdapter",
    "ClankerLM",
    "ConversationMemory",
    "Entity",
    "Fact",
    "GateProfile",
    "ParsedUtterance",
    "Provenance",
    "QuestionFrame",
    "RuleAffectAdapter",
    "SemanticFrame",
    "SemanticRole",
    "TurnResult",
]
