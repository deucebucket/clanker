"""Typed parser-local results and construction profiles."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from .. import lexicon
from ..model import AppositiveRelation, ClauseRelationDirection, ClauseRelationType, ContentRelationType, EmbeddedInterrogativeStatus, EmbeddedInterrogativeType, EventFrame, GerundContentStatus, GerundRelationType, ModifierGapRole, ModifierRestriction, InfinitivalContentStatus, InfinitivalRelationType, SemanticRef, UnresolvedReference

@dataclass
class NPResult:
    ref: Optional[SemanticRef]
    unresolved: List[UnresolvedReference] = field(default_factory=list)
    quantity: Optional[SemanticRef] = None
    entity_ids: List[str] = field(default_factory=list)
    surface: str = ""


@dataclass
class ClauseResult:
    event: Optional[EventFrame]
    unresolved: List[UnresolvedReference] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class ContentSplit:
    matrix_tokens: List[lexicon.Token]
    content_tokens: List[lexicon.Token]
    marker: str
    relation_type: ContentRelationType
    predicate_family: str
    certainty: int
    diagnostics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddedInterrogativePredicateProfile:
    content_status: EmbeddedInterrogativeStatus
    predicate_family: str
    certainty: int
    allows_recipient: bool = False
    allows_direct_answer_request: bool = False


@dataclass
class EmbeddedInterrogativeSplit:
    matrix_tokens: List[lexicon.Token]
    question_tokens: List[lexicon.Token]
    marker: str
    relation_type: EmbeddedInterrogativeType
    profile: EmbeddedInterrogativePredicateProfile
    direct_answer_request: bool = False
    diagnostics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class InfinitivalPredicateProfile:
    relation_type: InfinitivalRelationType
    content_status: InfinitivalContentStatus
    predicate_family: str
    certainty: int
    allows_object_controller: bool = False
    allows_subject_controller: bool = True


@dataclass
class InfinitivalSplit:
    matrix_tokens: List[lexicon.Token]
    complement_tokens: List[lexicon.Token]
    marker: str
    profile: InfinitivalPredicateProfile
    controller_role: str
    certainty: int
    diagnostics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GerundPredicateProfile:
    relation_type: GerundRelationType
    content_status: GerundContentStatus
    predicate_family: str
    certainty: int
    allows_object_controller: bool = False
    allows_subject_controller: bool = True
    phase_entailing: bool = False


@dataclass
class GerundSplit:
    matrix_tokens: List[lexicon.Token]
    complement_tokens: List[lexicon.Token]
    marker: str
    profile: GerundPredicateProfile
    controller_role: str
    certainty: int
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class RelativeSplit:
    main_tokens: List[lexicon.Token]
    modifier_tokens: List[lexicon.Token]
    head_entity_id: str
    marker: str
    gap_role: ModifierGapRole
    restriction: ModifierRestriction
    possessed_entity_id: str = ""
    certainty: int = 230
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class AppositiveSplit:
    main_tokens: List[lexicon.Token]
    relation: AppositiveRelation
    entity_ids: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class SubordinateSplit:
    main_tokens: List[lexicon.Token]
    subordinate_tokens: List[lexicon.Token]
    marker: str
    relation_type: ClauseRelationType
    direction: ClauseRelationDirection
    candidate_types: List[ClauseRelationType] = field(default_factory=list)
    certainty: int = 230
    diagnostics: List[str] = field(default_factory=list)
