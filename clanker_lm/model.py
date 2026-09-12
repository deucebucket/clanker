"""Stable public semantic API; implementations live in responsibility-specific contracts.

Public imports and reads of prior pickles resolve through these aliases.
Implementation module paths are explicit; JSON schemas and fields are unchanged.
"""

from .semantic_types.enums import (
    StringEnum,
    EntityKind,
    Gender,
    GrammaticalNumber,
    RefKind,
    SpeechAct,
    QuestionKind,
    WhyKind,
    HowKind,
    ClauseRelationType,
    ClauseRelationDirection,
    ModifierRestriction,
    ModifierGapRole,
    AppositiveRelationType,
    ContentRelationType,
    EmbeddedInterrogativeType,
    EmbeddedInterrogativeStatus,
    InfinitivalRelationType,
    InfinitivalContentStatus,
    GerundRelationType,
    GerundContentStatus,
    AnswerStatus,
    TruthValue,
    SourceKind,
)
from .semantic_types.references import (
    SemanticRef,
    Entity,
    UnresolvedReference,
)
from .semantic_types.events import (
    EventFrame,
    ClauseRelation,
)
from .semantic_types.content import (
    ContentRelation,
    ContentAttachmentAmbiguity,
)
from .semantic_types.questions import (
    QuestionFrame,
)
from .semantic_types.embedded import (
    EmbeddedInterrogativeRelation,
    EmbeddedInterrogativeAttachmentAmbiguity,
)
from .semantic_types.infinitivals import (
    InfinitivalRelation,
    InfinitivalAttachmentAmbiguity,
)
from .semantic_types.gerunds import (
    GerundRelation,
    GerundAttachmentAmbiguity,
)
from .semantic_types.modifiers import (
    EntityModifierRelation,
    AppositiveRelation,
    AppositiveAttachmentAmbiguity,
    ModifierAttachmentAmbiguity,
)
from .semantic_types.parses import (
    ParseResult,
)
from .semantic_types.answers import (
    Evidence,
    AnswerContract,
)
from .semantic_types.affect import (
    AffectVector,
    AffectReading,
)
from .semantic_types.responses import (
    GateDecision,
    CandidateResponse,
    TurnResult,
)
from .semantic_types.serialization import (
    _json_safe,
)

__all__ = (
    'StringEnum',
    'EntityKind',
    'Gender',
    'GrammaticalNumber',
    'RefKind',
    'SpeechAct',
    'QuestionKind',
    'WhyKind',
    'HowKind',
    'ClauseRelationType',
    'ClauseRelationDirection',
    'ModifierRestriction',
    'ModifierGapRole',
    'AppositiveRelationType',
    'ContentRelationType',
    'EmbeddedInterrogativeType',
    'EmbeddedInterrogativeStatus',
    'InfinitivalRelationType',
    'InfinitivalContentStatus',
    'GerundRelationType',
    'GerundContentStatus',
    'AnswerStatus',
    'TruthValue',
    'SourceKind',
    'SemanticRef',
    'Entity',
    'EventFrame',
    'ClauseRelation',
    'ContentRelation',
    'ContentAttachmentAmbiguity',
    'EmbeddedInterrogativeRelation',
    'EmbeddedInterrogativeAttachmentAmbiguity',
    'InfinitivalRelation',
    'InfinitivalAttachmentAmbiguity',
    'GerundRelation',
    'GerundAttachmentAmbiguity',
    'UnresolvedReference',
    'EntityModifierRelation',
    'AppositiveRelation',
    'AppositiveAttachmentAmbiguity',
    'ModifierAttachmentAmbiguity',
    'QuestionFrame',
    'ParseResult',
    'Evidence',
    'AnswerContract',
    'AffectVector',
    'AffectReading',
    'GateDecision',
    'CandidateResponse',
    'TurnResult',
    '_json_safe',
)
