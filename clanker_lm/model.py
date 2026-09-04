"""Core semantic data structures for Clanker-LM.

The V8 VADUGWI engine remains the affective kernel.  This module defines the
symbolic layer around it: entities, event frames, typed questions, answer
contracts, evidence provenance, and serializable turn traces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class StringEnum(str, Enum):
    """A Python 3.10-compatible string enum."""

    def __str__(self) -> str:
        return self.value


class EntityKind(StringEnum):
    PERSON = "person"
    THING = "thing"
    PLACE = "place"
    ORGANIZATION = "organization"
    EVENT = "event"
    TIME = "time"
    ABSTRACT = "abstract"
    UNKNOWN = "unknown"


class Gender(StringEnum):
    FEMALE = "female"
    MALE = "male"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class GrammaticalNumber(StringEnum):
    SINGULAR = "singular"
    PLURAL = "plural"
    UNKNOWN = "unknown"


class RefKind(StringEnum):
    ENTITY = "entity"
    LITERAL = "literal"
    VARIABLE = "variable"
    EVENT = "event"


class SpeechAct(StringEnum):
    ASSERT = "assert"
    ASK = "ask"
    ACKNOWLEDGE = "acknowledge"
    CLARIFY = "clarify"
    GREET = "greet"
    COMMAND = "command"
    SOCIAL = "social"
    UNKNOWN = "unknown"


class QuestionKind(StringEnum):
    WHO = "who"
    WHAT = "what"
    WHEN = "when"
    WHERE = "where"
    WHY = "why"
    HOW = "how"
    WHICH = "which"
    WHOSE = "whose"
    YES_NO = "yes_no"
    HOW_MANY = "how_many"
    HOW_MUCH = "how_much"
    WHAT_HAPPENED = "what_happened"
    UNKNOWN = "unknown"


class WhyKind(StringEnum):
    CAUSE = "cause"
    MOTIVE = "motive"
    PURPOSE = "purpose"
    JUSTIFICATION = "justification"
    EVIDENCE = "evidence"
    UNKNOWN = "unknown"


class HowKind(StringEnum):
    METHOD = "method"
    MANNER = "manner"
    MECHANISM = "mechanism"
    PROCESS = "process"
    DEGREE = "degree"
    QUANTITY = "quantity"
    STATE = "state"
    UNKNOWN = "unknown"


class ClauseRelationType(StringEnum):
    CAUSE = "cause"
    TEMPORAL_WHEN = "temporal_when"
    TEMPORAL_OVERLAP = "temporal_overlap"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    TEMPORAL_UNTIL = "temporal_until"
    TEMPORAL_SINCE = "temporal_since"
    CONDITION = "condition"
    EXCEPTION_CONDITION = "exception_condition"
    CONCESSION = "concession"
    PURPOSE = "purpose"
    RESULT = "result"
    AMBIGUOUS = "ambiguous"


class ClauseRelationDirection(StringEnum):
    MAIN_TO_SUBORDINATE = "main_to_subordinate"
    SUBORDINATE_TO_MAIN = "subordinate_to_main"
    SYMMETRIC = "symmetric"
    UNRESOLVED = "unresolved"


class ModifierRestriction(StringEnum):
    RESTRICTIVE = "restrictive"
    NONRESTRICTIVE = "nonrestrictive"


class ModifierGapRole(StringEnum):
    AGENT = "agent"
    PATIENT = "patient"
    POSSESSOR = "possessor"


class AppositiveRelationType(StringEnum):
    IDENTITY = "identity"
    ROLE = "role"
    DESCRIPTION = "description"


class ContentRelationType(StringEnum):
    REPORTED = "reported"
    BELIEVED = "believed"
    KNOWN = "known"
    PERCEIVED = "perceived"


class EmbeddedInterrogativeType(StringEnum):
    """Structural type of a question embedded under a matrix predicate."""

    WH = "wh"
    POLAR = "polar"


class EmbeddedInterrogativeStatus(StringEnum):
    """Epistemic/discourse status contributed by the matrix predicate."""

    ASKED = "asked"
    WONDERED = "wondered"
    KNOWN = "known"
    REMEMBERED = "remembered"
    DISCOVERED = "discovered"
    REQUESTED = "requested"


class InfinitivalRelationType(StringEnum):
    """Syntactic controller relationship for a selected ``to`` complement."""

    SUBJECT_CONTROL = "subject_control"
    OBJECT_CONTROL = "object_control"
    RAISING = "raising"


class InfinitivalContentStatus(StringEnum):
    """Truth-bearing status contributed by the matrix predicate.

    These values describe how the matrix event presents the embedded event;
    none of them assert that the embedded event actually happened.
    """

    PLANNED = "planned"
    INTENDED = "intended"
    HOPED = "hoped"
    DESIRED = "desired"
    DIRECTED = "directed"
    REQUESTED = "requested"
    EVIDENTIAL = "evidential"


class AnswerStatus(StringEnum):
    ANSWERED = "answered"
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    LEXICAL_PROBE = "lexical_probe"
    LEXICAL_LEARNED = "lexical_learned"
    MISSING_REFERENCE = "missing_reference"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    MULTIPLE_MATCHES = "multiple_matches"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    ACKNOWLEDGED = "acknowledged"


class TruthValue(StringEnum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class SourceKind(StringEnum):
    UNKNOWN = "unknown"
    TRAINED = "trained"
    RETRIEVED = "retrieved"
    INFERRED = "inferred"
    ATTRIBUTED = "attributed"
    USER = "user"
    EXTERNAL = "external"
    VERIFIED = "verified"


@dataclass(frozen=True)
class SemanticRef:
    """A typed value used in a semantic frame.

    ``key`` is canonical and comparison-safe.  ``surface`` preserves a useful
    wording for realization.  Entity references use ``entity_id`` as their key;
    literals use a normalized literal string; variables use the requested slot
    name (for example ``patient`` or ``cause``).
    """

    kind: RefKind
    key: str
    surface: str = ""
    value_type: EntityKind = EntityKind.UNKNOWN

    @classmethod
    def entity(cls, entity_id: str, surface: str = "", value_type: EntityKind = EntityKind.UNKNOWN) -> "SemanticRef":
        return cls(RefKind.ENTITY, entity_id, surface, value_type)

    @classmethod
    def literal(cls, value: str, surface: str = "", value_type: EntityKind = EntityKind.ABSTRACT) -> "SemanticRef":
        return cls(RefKind.LITERAL, value, surface or value, value_type)

    @classmethod
    def variable(cls, role: str, value_type: EntityKind = EntityKind.UNKNOWN) -> "SemanticRef":
        return cls(RefKind.VARIABLE, role, f"?{role}", value_type)

    @classmethod
    def event(cls, event_id: str, surface: str = "") -> "SemanticRef":
        return cls(RefKind.EVENT, event_id, surface, EntityKind.EVENT)

    @property
    def is_variable(self) -> bool:
        return self.kind == RefKind.VARIABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "key": self.key,
            "surface": self.surface,
            "value_type": self.value_type.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticRef":
        return cls(
            kind=RefKind(data["kind"]),
            key=str(data["key"]),
            surface=str(data.get("surface", "")),
            value_type=EntityKind(data.get("value_type", EntityKind.UNKNOWN.value)),
        )


@dataclass
class Entity:
    entity_id: str
    canonical_name: str
    kind: EntityKind = EntityKind.UNKNOWN
    gender: Gender = Gender.UNKNOWN
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR
    owner_id: Optional[str] = None
    relation: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    created_turn: int = 0
    last_mentioned_turn: int = 0
    salience: float = 0.0

    def add_alias(self, alias: str) -> None:
        normalized = " ".join(alias.lower().split()).strip()
        if normalized and normalized not in self.aliases:
            self.aliases.append(normalized)

    def to_ref(self, surface: str = "") -> SemanticRef:
        return SemanticRef.entity(self.entity_id, surface or self.canonical_name, self.kind)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "kind": self.kind.value,
            "gender": self.gender.value,
            "number": self.number.value,
            "owner_id": self.owner_id,
            "relation": self.relation,
            "aliases": list(self.aliases),
            "attributes": dict(self.attributes),
            "created_turn": self.created_turn,
            "last_mentioned_turn": self.last_mentioned_turn,
            "salience": self.salience,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Entity":
        return cls(
            entity_id=str(data["entity_id"]),
            canonical_name=str(data["canonical_name"]),
            kind=EntityKind(data.get("kind", EntityKind.UNKNOWN.value)),
            gender=Gender(data.get("gender", Gender.UNKNOWN.value)),
            number=GrammaticalNumber(data.get("number", GrammaticalNumber.SINGULAR.value)),
            owner_id=data.get("owner_id"),
            relation=data.get("relation"),
            aliases=list(data.get("aliases", [])),
            attributes=dict(data.get("attributes", {})),
            created_turn=int(data.get("created_turn", 0)),
            last_mentioned_turn=int(data.get("last_mentioned_turn", 0)),
            salience=float(data.get("salience", 0.0)),
        )


@dataclass
class EventFrame:
    predicate: str
    arguments: Dict[str, SemanticRef] = field(default_factory=dict)
    tense: str = "present"
    aspect: str = "simple"
    polarity: bool = True
    modality: Optional[str] = None
    raw_text: str = ""
    event_id: str = ""
    source: SourceKind = SourceKind.USER
    certainty: int = 230
    turn_index: int = 0
    inferred: bool = False
    discourse_role: str = "main"

    def copy(self, **changes: Any) -> "EventFrame":
        values: Dict[str, Any] = {
            "predicate": self.predicate,
            "arguments": dict(self.arguments),
            "tense": self.tense,
            "aspect": self.aspect,
            "polarity": self.polarity,
            "modality": self.modality,
            "raw_text": self.raw_text,
            "event_id": self.event_id,
            "source": self.source,
            "certainty": self.certainty,
            "turn_index": self.turn_index,
            "inferred": self.inferred,
            "discourse_role": self.discourse_role,
        }
        values.update(changes)
        return EventFrame(**values)

    def fixed_arguments(self) -> Dict[str, SemanticRef]:
        return {key: value for key, value in self.arguments.items() if not value.is_variable}

    def variable_roles(self) -> List[str]:
        return [key for key, value in self.arguments.items() if value.is_variable]

    def signature(self, exclude_roles: Iterable[str] = ()) -> Tuple[Any, ...]:
        excluded = set(exclude_roles)
        args = tuple(
            sorted(
                (role, ref.kind.value, ref.key)
                for role, ref in self.arguments.items()
                if role not in excluded and not ref.is_variable
            )
        )
        return self.predicate, self.tense, self.polarity, self.modality, args

    def proposition_signature(self) -> Tuple[Any, ...]:
        args = tuple(sorted((role, ref.kind.value, ref.key) for role, ref in self.arguments.items() if not ref.is_variable))
        return self.predicate, self.polarity, args

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "arguments": {key: ref.to_dict() for key, ref in self.arguments.items()},
            "tense": self.tense,
            "aspect": self.aspect,
            "polarity": self.polarity,
            "modality": self.modality,
            "raw_text": self.raw_text,
            "event_id": self.event_id,
            "source": self.source.value,
            "certainty": self.certainty,
            "turn_index": self.turn_index,
            "inferred": self.inferred,
            "discourse_role": self.discourse_role,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventFrame":
        return cls(
            predicate=str(data["predicate"]),
            arguments={key: SemanticRef.from_dict(value) for key, value in dict(data.get("arguments", {})).items()},
            tense=str(data.get("tense", "present")),
            aspect=str(data.get("aspect", "simple")),
            polarity=bool(data.get("polarity", True)),
            modality=data.get("modality"),
            raw_text=str(data.get("raw_text", "")),
            event_id=str(data.get("event_id", "")),
            source=SourceKind(data.get("source", SourceKind.USER.value)),
            certainty=int(data.get("certainty", 230)),
            turn_index=int(data.get("turn_index", 0)),
            inferred=bool(data.get("inferred", False)),
            discourse_role=str(data.get("discourse_role", "main")),
        )


@dataclass
class ClauseRelation:
    """A typed relation between one main and one subordinate event.

    Parser instances use event-list indices.  Conversation memory binds those
    indices to stable event IDs without changing the relation semantics.
    """

    relation_type: ClauseRelationType
    main_event_index: int
    subordinate_event_index: int
    marker: str
    direction: ClauseRelationDirection
    certainty: int = 230
    candidate_types: List[ClauseRelationType] = field(default_factory=list)
    relation_id: str = ""
    main_event_id: str = ""
    subordinate_event_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return self.relation_type == ClauseRelationType.AMBIGUOUS

    def copy(self, **changes: Any) -> "ClauseRelation":
        values: Dict[str, Any] = {
            "relation_type": self.relation_type,
            "main_event_index": self.main_event_index,
            "subordinate_event_index": self.subordinate_event_index,
            "marker": self.marker,
            "direction": self.direction,
            "certainty": self.certainty,
            "candidate_types": list(self.candidate_types),
            "relation_id": self.relation_id,
            "main_event_id": self.main_event_id,
            "subordinate_event_id": self.subordinate_event_id,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return ClauseRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.relation_type.value,
            self.main_event_id,
            self.subordinate_event_id,
            self.marker,
            self.direction.value,
            tuple(item.value for item in self.candidate_types),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "main_event_index": self.main_event_index,
            "subordinate_event_index": self.subordinate_event_index,
            "marker": self.marker,
            "direction": self.direction.value,
            "certainty": self.certainty,
            "candidate_types": [item.value for item in self.candidate_types],
            "relation_id": self.relation_id,
            "main_event_id": self.main_event_id,
            "subordinate_event_id": self.subordinate_event_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClauseRelation":
        return cls(
            relation_type=ClauseRelationType(data["relation_type"]),
            main_event_index=int(data.get("main_event_index", -1)),
            subordinate_event_index=int(data.get("subordinate_event_index", -1)),
            marker=str(data.get("marker", "")),
            direction=ClauseRelationDirection(
                data.get("direction", ClauseRelationDirection.UNRESOLVED.value)
            ),
            certainty=max(0, min(255, int(data.get("certainty", 230)))),
            candidate_types=[
                ClauseRelationType(item)
                for item in data.get("candidate_types", [])
            ],
            relation_id=str(data.get("relation_id", "")),
            main_event_id=str(data.get("main_event_id", "")),
            subordinate_event_id=str(data.get("subordinate_event_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class ContentRelation:
    """Typed attribution link from a matrix predicate to finite content.

    Parser instances use event-list indices. Conversation memory binds them to
    stable event IDs. The content event is evidence about what a source said,
    believed, knew, or perceived; it is not automatically an unqualified fact.
    """

    relation_type: ContentRelationType
    matrix_event_index: int
    content_event_index: int
    marker: str
    matrix_predicate: str
    source_entity_id: str
    predicate_family: str
    certainty: int = 210
    relation_id: str = ""
    matrix_event_id: str = ""
    content_event_id: str = ""
    attributed: bool = True
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "ContentRelation":
        values: Dict[str, Any] = {
            "relation_type": self.relation_type,
            "matrix_event_index": self.matrix_event_index,
            "content_event_index": self.content_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "predicate_family": self.predicate_family,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "content_event_id": self.content_event_id,
            "attributed": self.attributed,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return ContentRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.relation_type.value,
            self.matrix_event_id,
            self.content_event_id,
            self.marker,
            self.matrix_predicate,
            self.source_entity_id,
            self.predicate_family,
            self.attributed,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "matrix_event_index": self.matrix_event_index,
            "content_event_index": self.content_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "predicate_family": self.predicate_family,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "content_event_id": self.content_event_id,
            "attributed": self.attributed,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContentRelation":
        return cls(
            relation_type=ContentRelationType(data["relation_type"]),
            matrix_event_index=int(data.get("matrix_event_index", -1)),
            content_event_index=int(data.get("content_event_index", -1)),
            marker=str(data.get("marker", "")),
            matrix_predicate=str(data.get("matrix_predicate", "")),
            source_entity_id=str(data.get("source_entity_id", "")),
            predicate_family=str(data.get("predicate_family", "")),
            certainty=max(0, min(255, int(data.get("certainty", 210)))),
            relation_id=str(data.get("relation_id", "")),
            matrix_event_id=str(data.get("matrix_event_id", "")),
            content_event_id=str(data.get("content_event_id", "")),
            attributed=bool(data.get("attributed", True)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class ContentAttachmentAmbiguity:
    """Explicit unresolved boundary or attribution for a content clause."""

    matrix_surface: str
    content_surface: str
    clause_surface: str
    reason: str
    candidate_boundaries: List[int] = field(default_factory=list)
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix_surface": self.matrix_surface,
            "content_surface": self.content_surface,
            "clause_surface": self.clause_surface,
            "reason": self.reason,
            "candidate_boundaries": list(self.candidate_boundaries),
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContentAttachmentAmbiguity":
        return cls(
            matrix_surface=str(data.get("matrix_surface", "")),
            content_surface=str(data.get("content_surface", "")),
            clause_surface=str(data.get("clause_surface", "")),
            reason=str(data.get("reason", "")),
            candidate_boundaries=[int(item) for item in data.get("candidate_boundaries", [])],
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class EmbeddedInterrogativeRelation:
    """Typed matrix-to-question link for one embedded interrogative.

    The question proposition is stored as a nonassertive event containing its
    typed variable, while this relation retains the interrogative operator,
    attribution source, and matrix discourse status.  Neither a WH variable nor
    a polar proposition is promoted into an unqualified fact by this relation.
    """

    relation_type: EmbeddedInterrogativeType
    content_status: EmbeddedInterrogativeStatus
    matrix_event_index: int
    question_event_index: int
    marker: str
    matrix_predicate: str
    source_entity_id: str
    predicate_family: str
    question_kind: QuestionKind
    requested_role: Optional[str] = None
    answer_type: EntityKind = EntityKind.UNKNOWN
    certainty: int = 200
    relation_id: str = ""
    matrix_event_id: str = ""
    question_event_id: str = ""
    licensed: bool = True
    direct_answer_request: bool = False
    focus_surface: str = ""
    why_kind: WhyKind = WhyKind.UNKNOWN
    how_kind: HowKind = HowKind.UNKNOWN
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "EmbeddedInterrogativeRelation":
        values: Dict[str, Any] = {
            "relation_type": self.relation_type,
            "content_status": self.content_status,
            "matrix_event_index": self.matrix_event_index,
            "question_event_index": self.question_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "predicate_family": self.predicate_family,
            "question_kind": self.question_kind,
            "requested_role": self.requested_role,
            "answer_type": self.answer_type,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "question_event_id": self.question_event_id,
            "licensed": self.licensed,
            "direct_answer_request": self.direct_answer_request,
            "focus_surface": self.focus_surface,
            "why_kind": self.why_kind,
            "how_kind": self.how_kind,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return EmbeddedInterrogativeRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.relation_type.value,
            self.content_status.value,
            self.matrix_event_id,
            self.question_event_id,
            self.marker,
            self.matrix_predicate,
            self.source_entity_id,
            self.predicate_family,
            self.question_kind.value,
            self.requested_role,
            self.answer_type.value,
            self.licensed,
            self.direct_answer_request,
            self.focus_surface,
            self.why_kind.value,
            self.how_kind.value,
        )

    def to_question_frame(self, event: EventFrame) -> "QuestionFrame":
        return QuestionFrame(
            kind=self.question_kind,
            event=event,
            requested_role=self.requested_role,
            answer_type=self.answer_type,
            why_kind=self.why_kind,
            how_kind=self.how_kind,
            raw_text=event.raw_text,
            focus_surface=self.focus_surface,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "content_status": self.content_status.value,
            "matrix_event_index": self.matrix_event_index,
            "question_event_index": self.question_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "predicate_family": self.predicate_family,
            "question_kind": self.question_kind.value,
            "requested_role": self.requested_role,
            "answer_type": self.answer_type.value,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "question_event_id": self.question_event_id,
            "licensed": self.licensed,
            "direct_answer_request": self.direct_answer_request,
            "focus_surface": self.focus_surface,
            "why_kind": self.why_kind.value,
            "how_kind": self.how_kind.value,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddedInterrogativeRelation":
        return cls(
            relation_type=EmbeddedInterrogativeType(data["relation_type"]),
            content_status=EmbeddedInterrogativeStatus(data["content_status"]),
            matrix_event_index=int(data.get("matrix_event_index", -1)),
            question_event_index=int(data.get("question_event_index", -1)),
            marker=str(data.get("marker", "")),
            matrix_predicate=str(data.get("matrix_predicate", "")),
            source_entity_id=str(data.get("source_entity_id", "")),
            predicate_family=str(data.get("predicate_family", "")),
            question_kind=QuestionKind(data.get("question_kind", QuestionKind.UNKNOWN.value)),
            requested_role=data.get("requested_role"),
            answer_type=EntityKind(data.get("answer_type", EntityKind.UNKNOWN.value)),
            certainty=max(0, min(255, int(data.get("certainty", 200)))),
            relation_id=str(data.get("relation_id", "")),
            matrix_event_id=str(data.get("matrix_event_id", "")),
            question_event_id=str(data.get("question_event_id", "")),
            licensed=bool(data.get("licensed", True)),
            direct_answer_request=bool(data.get("direct_answer_request", False)),
            focus_surface=str(data.get("focus_surface", "")),
            why_kind=WhyKind(data.get("why_kind", WhyKind.UNKNOWN.value)),
            how_kind=HowKind(data.get("how_kind", HowKind.UNKNOWN.value)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class EmbeddedInterrogativeAttachmentAmbiguity:
    """Explicit unresolved matrix/question boundary or interrogative scope."""

    matrix_surface: str
    question_surface: str
    clause_surface: str
    reason: str
    candidate_boundaries: List[int] = field(default_factory=list)
    candidate_markers: List[str] = field(default_factory=list)
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix_surface": self.matrix_surface,
            "question_surface": self.question_surface,
            "clause_surface": self.clause_surface,
            "reason": self.reason,
            "candidate_boundaries": list(self.candidate_boundaries),
            "candidate_markers": list(self.candidate_markers),
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "EmbeddedInterrogativeAttachmentAmbiguity":
        return cls(
            matrix_surface=str(data.get("matrix_surface", "")),
            question_surface=str(data.get("question_surface", "")),
            clause_surface=str(data.get("clause_surface", "")),
            reason=str(data.get("reason", "")),
            candidate_boundaries=[int(item) for item in data.get("candidate_boundaries", [])],
            candidate_markers=[str(item) for item in data.get("candidate_markers", [])],
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class InfinitivalRelation:
    """Typed link between a matrix event and one selected infinitive.

    Parser instances use event-list indices. Conversation memory binds those
    indices to stable event IDs. ``licensed`` records matrix polarity: a
    negated plan/request/appearance is retained as a relation but cannot be
    used as positive evidence that the source held that plan or request.
    ``entailed`` is deliberately false for this bounded slice because plans,
    desires, requests, commands, hopes, and appearances do not establish that
    the embedded event occurred.
    """

    relation_type: InfinitivalRelationType
    content_status: InfinitivalContentStatus
    matrix_event_index: int
    complement_event_index: int
    marker: str
    matrix_predicate: str
    source_entity_id: str
    controller_entity_id: str
    embedded_subject_entity_id: str
    predicate_family: str
    certainty: int = 200
    relation_id: str = ""
    matrix_event_id: str = ""
    complement_event_id: str = ""
    licensed: bool = True
    entailed: bool = False
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "InfinitivalRelation":
        values: Dict[str, Any] = {
            "relation_type": self.relation_type,
            "content_status": self.content_status,
            "matrix_event_index": self.matrix_event_index,
            "complement_event_index": self.complement_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "controller_entity_id": self.controller_entity_id,
            "embedded_subject_entity_id": self.embedded_subject_entity_id,
            "predicate_family": self.predicate_family,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "complement_event_id": self.complement_event_id,
            "licensed": self.licensed,
            "entailed": self.entailed,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return InfinitivalRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.relation_type.value,
            self.content_status.value,
            self.matrix_event_id,
            self.complement_event_id,
            self.marker,
            self.matrix_predicate,
            self.source_entity_id,
            self.controller_entity_id,
            self.embedded_subject_entity_id,
            self.predicate_family,
            self.licensed,
            self.entailed,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "content_status": self.content_status.value,
            "matrix_event_index": self.matrix_event_index,
            "complement_event_index": self.complement_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "controller_entity_id": self.controller_entity_id,
            "embedded_subject_entity_id": self.embedded_subject_entity_id,
            "predicate_family": self.predicate_family,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "complement_event_id": self.complement_event_id,
            "licensed": self.licensed,
            "entailed": self.entailed,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InfinitivalRelation":
        return cls(
            relation_type=InfinitivalRelationType(data["relation_type"]),
            content_status=InfinitivalContentStatus(data["content_status"]),
            matrix_event_index=int(data.get("matrix_event_index", -1)),
            complement_event_index=int(data.get("complement_event_index", -1)),
            marker=str(data.get("marker", "to")),
            matrix_predicate=str(data.get("matrix_predicate", "")),
            source_entity_id=str(data.get("source_entity_id", "")),
            controller_entity_id=str(data.get("controller_entity_id", "")),
            embedded_subject_entity_id=str(
                data.get("embedded_subject_entity_id", "")
            ),
            predicate_family=str(data.get("predicate_family", "")),
            certainty=max(0, min(255, int(data.get("certainty", 200)))),
            relation_id=str(data.get("relation_id", "")),
            matrix_event_id=str(data.get("matrix_event_id", "")),
            complement_event_id=str(data.get("complement_event_id", "")),
            licensed=bool(data.get("licensed", True)),
            entailed=bool(data.get("entailed", False)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class InfinitivalAttachmentAmbiguity:
    """Explicit unresolved boundary/controller for a selected infinitive."""

    matrix_surface: str
    complement_surface: str
    clause_surface: str
    reason: str
    candidate_boundaries: List[int] = field(default_factory=list)
    candidate_relation_types: List[InfinitivalRelationType] = field(
        default_factory=list
    )
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix_surface": self.matrix_surface,
            "complement_surface": self.complement_surface,
            "clause_surface": self.clause_surface,
            "reason": self.reason,
            "candidate_boundaries": list(self.candidate_boundaries),
            "candidate_relation_types": [
                item.value for item in self.candidate_relation_types
            ],
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "InfinitivalAttachmentAmbiguity":
        return cls(
            matrix_surface=str(data.get("matrix_surface", "")),
            complement_surface=str(data.get("complement_surface", "")),
            clause_surface=str(data.get("clause_surface", "")),
            reason=str(data.get("reason", "")),
            candidate_boundaries=[
                int(item) for item in data.get("candidate_boundaries", [])
            ],
            candidate_relation_types=[
                InfinitivalRelationType(item)
                for item in data.get("candidate_relation_types", [])
            ],
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class UnresolvedReference:
    surface: str
    reason: str
    candidates: List[str] = field(default_factory=list)
    expected_kind: EntityKind = EntityKind.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface": self.surface,
            "reason": self.reason,
            "candidates": list(self.candidates),
            "expected_kind": self.expected_kind.value,
        }


@dataclass
class EntityModifierRelation:
    """Typed link from one entity to a finite relative-clause event."""

    head_entity_id: str
    modifier_event_index: int
    marker: str
    gap_role: ModifierGapRole
    restriction: ModifierRestriction
    certainty: int = 230
    relation_id: str = ""
    modifier_event_id: str = ""
    modifier_event_signature: str = ""
    possessed_entity_id: str = ""
    inferred: bool = False
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "EntityModifierRelation":
        values: Dict[str, Any] = {
            "head_entity_id": self.head_entity_id,
            "modifier_event_index": self.modifier_event_index,
            "marker": self.marker,
            "gap_role": self.gap_role,
            "restriction": self.restriction,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "modifier_event_id": self.modifier_event_id,
            "modifier_event_signature": self.modifier_event_signature,
            "possessed_entity_id": self.possessed_entity_id,
            "inferred": self.inferred,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return EntityModifierRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.head_entity_id,
            self.modifier_event_id,
            self.marker,
            self.gap_role.value,
            self.restriction.value,
            self.possessed_entity_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head_entity_id": self.head_entity_id,
            "modifier_event_index": self.modifier_event_index,
            "marker": self.marker,
            "gap_role": self.gap_role.value,
            "restriction": self.restriction.value,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "modifier_event_id": self.modifier_event_id,
            "modifier_event_signature": self.modifier_event_signature,
            "possessed_entity_id": self.possessed_entity_id,
            "inferred": self.inferred,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntityModifierRelation":
        return cls(
            head_entity_id=str(data["head_entity_id"]),
            modifier_event_index=int(data.get("modifier_event_index", -1)),
            marker=str(data.get("marker", "")),
            gap_role=ModifierGapRole(
                data.get("gap_role", ModifierGapRole.AGENT.value)
            ),
            restriction=ModifierRestriction(
                data.get("restriction", ModifierRestriction.RESTRICTIVE.value)
            ),
            certainty=max(0, min(255, int(data.get("certainty", 230)))),
            relation_id=str(data.get("relation_id", "")),
            modifier_event_id=str(data.get("modifier_event_id", "")),
            modifier_event_signature=str(data.get("modifier_event_signature", "")),
            possessed_entity_id=str(data.get("possessed_entity_id", "")),
            inferred=bool(data.get("inferred", False)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class AppositiveRelation:
    """Typed identity/description link licensed by explicit apposition."""

    head_entity_id: str
    primary_surface: str
    appositive_surface: str
    relation_type: AppositiveRelationType
    restriction: ModifierRestriction
    appositive_key: str = ""
    role_owner_id: str = ""
    role_name: str = ""
    certainty: int = 230
    relation_id: str = ""
    source: SourceKind = SourceKind.USER
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "AppositiveRelation":
        values: Dict[str, Any] = {
            "head_entity_id": self.head_entity_id,
            "primary_surface": self.primary_surface,
            "appositive_surface": self.appositive_surface,
            "relation_type": self.relation_type,
            "restriction": self.restriction,
            "appositive_key": self.appositive_key,
            "role_owner_id": self.role_owner_id,
            "role_name": self.role_name,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "source": self.source,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return AppositiveRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.head_entity_id,
            self.appositive_key,
            self.relation_type.value,
            self.restriction.value,
            self.role_owner_id,
            self.role_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head_entity_id": self.head_entity_id,
            "primary_surface": self.primary_surface,
            "appositive_surface": self.appositive_surface,
            "relation_type": self.relation_type.value,
            "restriction": self.restriction.value,
            "appositive_key": self.appositive_key,
            "role_owner_id": self.role_owner_id,
            "role_name": self.role_name,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "source": self.source.value,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppositiveRelation":
        return cls(
            head_entity_id=str(data["head_entity_id"]),
            primary_surface=str(data.get("primary_surface", "")),
            appositive_surface=str(data.get("appositive_surface", "")),
            relation_type=AppositiveRelationType(
                data.get("relation_type", AppositiveRelationType.IDENTITY.value)
            ),
            restriction=ModifierRestriction(
                data.get("restriction", ModifierRestriction.NONRESTRICTIVE.value)
            ),
            appositive_key=str(data.get("appositive_key", "")),
            role_owner_id=str(data.get("role_owner_id", "")),
            role_name=str(data.get("role_name", "")),
            certainty=max(0, min(255, int(data.get("certainty", 230)))),
            relation_id=str(data.get("relation_id", "")),
            source=SourceKind(data.get("source", SourceKind.USER.value)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class AppositiveAttachmentAmbiguity:
    """Explicit unresolved appositive identity or attachment choice."""

    primary_surface: str
    appositive_surface: str
    clause_surface: str
    reason: str
    candidate_entity_ids: List[str] = field(default_factory=list)
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_surface": self.primary_surface,
            "appositive_surface": self.appositive_surface,
            "clause_surface": self.clause_surface,
            "reason": self.reason,
            "candidate_entity_ids": list(self.candidate_entity_ids),
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppositiveAttachmentAmbiguity":
        return cls(
            primary_surface=str(data.get("primary_surface", "")),
            appositive_surface=str(data.get("appositive_surface", "")),
            clause_surface=str(data.get("clause_surface", "")),
            reason=str(data.get("reason", "")),
            candidate_entity_ids=list(data.get("candidate_entity_ids", [])),
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class ModifierAttachmentAmbiguity:
    """Explicit unresolved choice between multiple relative attachments."""

    marker: str
    clause_surface: str
    candidate_head_surfaces: List[str]
    reason: str
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker": self.marker,
            "clause_surface": self.clause_surface,
            "candidate_head_surfaces": list(self.candidate_head_surfaces),
            "reason": self.reason,
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModifierAttachmentAmbiguity":
        return cls(
            marker=str(data.get("marker", "")),
            clause_surface=str(data.get("clause_surface", "")),
            candidate_head_surfaces=list(data.get("candidate_head_surfaces", [])),
            reason=str(data.get("reason", "")),
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class QuestionFrame:
    kind: QuestionKind
    event: EventFrame
    requested_role: Optional[str] = None
    answer_type: EntityKind = EntityKind.UNKNOWN
    why_kind: WhyKind = WhyKind.UNKNOWN
    how_kind: HowKind = HowKind.UNKNOWN
    raw_text: str = ""
    unresolved: List[UnresolvedReference] = field(default_factory=list)
    focus_surface: str = ""
    social_convention: Optional[str] = None
    matrix_polarity: Optional[bool] = None
    embedded_polarity: Optional[bool] = None
    embedded_question: Optional["QuestionFrame"] = None
    embedded_interrogative_type: Optional[EmbeddedInterrogativeType] = None
    embedded_marker: str = ""
    embedded_matrix_predicate: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "event": self.event.to_dict(),
            "requested_role": self.requested_role,
            "answer_type": self.answer_type.value,
            "why_kind": self.why_kind.value,
            "how_kind": self.how_kind.value,
            "raw_text": self.raw_text,
            "unresolved": [item.to_dict() for item in self.unresolved],
            "focus_surface": self.focus_surface,
            "social_convention": self.social_convention,
            "matrix_polarity": self.matrix_polarity,
            "embedded_polarity": self.embedded_polarity,
            "embedded_question": (
                self.embedded_question.to_dict()
                if self.embedded_question is not None
                else None
            ),
            "embedded_interrogative_type": (
                self.embedded_interrogative_type.value
                if self.embedded_interrogative_type is not None
                else None
            ),
            "embedded_marker": self.embedded_marker,
            "embedded_matrix_predicate": self.embedded_matrix_predicate,
        }


@dataclass
class ParseResult:
    speech_act: SpeechAct
    raw_text: str
    events: List[EventFrame] = field(default_factory=list)
    relations: List[ClauseRelation] = field(default_factory=list)
    modifiers: List[EntityModifierRelation] = field(default_factory=list)
    modifier_ambiguities: List[ModifierAttachmentAmbiguity] = field(default_factory=list)
    appositives: List[AppositiveRelation] = field(default_factory=list)
    appositive_ambiguities: List[AppositiveAttachmentAmbiguity] = field(default_factory=list)
    contents: List[ContentRelation] = field(default_factory=list)
    content_ambiguities: List[ContentAttachmentAmbiguity] = field(default_factory=list)
    embedded_interrogatives: List[EmbeddedInterrogativeRelation] = field(default_factory=list)
    embedded_interrogative_ambiguities: List[
        EmbeddedInterrogativeAttachmentAmbiguity
    ] = field(default_factory=list)
    infinitivals: List[InfinitivalRelation] = field(default_factory=list)
    infinitival_ambiguities: List[InfinitivalAttachmentAmbiguity] = field(
        default_factory=list
    )
    question: Optional[QuestionFrame] = None
    entities: List[str] = field(default_factory=list)
    unresolved: List[UnresolvedReference] = field(default_factory=list)
    normalized_text: str = ""
    diagnostics: List[str] = field(default_factory=list)

    @property
    def understood(self) -> bool:
        return bool(self.events or self.question or self.speech_act in {SpeechAct.GREET, SpeechAct.SOCIAL})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speech_act": self.speech_act.value,
            "raw_text": self.raw_text,
            "events": [event.to_dict() for event in self.events],
            "relations": [relation.to_dict() for relation in self.relations],
            "modifiers": [modifier.to_dict() for modifier in self.modifiers],
            "modifier_ambiguities": [
                ambiguity.to_dict() for ambiguity in self.modifier_ambiguities
            ],
            "appositives": [item.to_dict() for item in self.appositives],
            "appositive_ambiguities": [
                ambiguity.to_dict() for ambiguity in self.appositive_ambiguities
            ],
            "contents": [item.to_dict() for item in self.contents],
            "content_ambiguities": [
                ambiguity.to_dict() for ambiguity in self.content_ambiguities
            ],
            "embedded_interrogatives": [
                item.to_dict() for item in self.embedded_interrogatives
            ],
            "embedded_interrogative_ambiguities": [
                ambiguity.to_dict()
                for ambiguity in self.embedded_interrogative_ambiguities
            ],
            "infinitivals": [item.to_dict() for item in self.infinitivals],
            "infinitival_ambiguities": [
                ambiguity.to_dict() for ambiguity in self.infinitival_ambiguities
            ],
            "question": self.question.to_dict() if self.question else None,
            "entities": list(self.entities),
            "unresolved": [item.to_dict() for item in self.unresolved],
            "normalized_text": self.normalized_text,
            "diagnostics": list(self.diagnostics),
            "understood": self.understood,
        }


@dataclass
class Evidence:
    event: EventFrame
    matched_roles: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "matched_roles": list(self.matched_roles),
            "score": self.score,
        }


@dataclass
class AnswerContract:
    status: AnswerStatus
    question: Optional[QuestionFrame] = None
    proposition: Optional[EventFrame] = None
    values: List[SemanticRef] = field(default_factory=list)
    truth: TruthValue = TruthValue.UNKNOWN
    evidence: List[Evidence] = field(default_factory=list)
    certainty: int = 0
    source: SourceKind = SourceKind.UNKNOWN
    reason: str = ""
    response_goal: str = "help"
    required_slots: Dict[str, str] = field(default_factory=dict)
    forbidden_claims: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "question": self.question.to_dict() if self.question else None,
            "proposition": self.proposition.to_dict() if self.proposition else None,
            "values": [value.to_dict() for value in self.values],
            "truth": self.truth.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "certainty": self.certainty,
            "source": self.source.value,
            "reason": self.reason,
            "response_goal": self.response_goal,
            "required_slots": dict(self.required_slots),
            "forbidden_claims": list(self.forbidden_claims),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class AffectVector:
    v: int = 128
    a: int = 128
    d: int = 128
    u: int = 0
    g: int = 128
    w: int = 128
    i: int = 128

    def __post_init__(self) -> None:
        for name in ("v", "a", "d", "u", "g", "w", "i"):
            value = int(getattr(self, name))
            object.__setattr__(self, name, max(0, min(255, value)))

    def to_dict(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name in ("v", "a", "d", "u", "g", "w", "i")}

    @classmethod
    def from_object(cls, value: Any) -> "AffectVector":
        return cls(**{name: int(getattr(value, name)) for name in ("v", "a", "d", "u", "g", "w", "i")})

    def distance(self, other: "AffectVector", weights: Optional[Mapping[str, float]] = None) -> float:
        axis_weights = {"v": 1.0, "a": 0.7, "d": 0.8, "u": 1.0, "g": 0.8, "w": 0.7, "i": 0.9}
        if weights:
            axis_weights.update(weights)
        total = 0.0
        divisor = 0.0
        for name, weight in axis_weights.items():
            delta = float(getattr(self, name) - getattr(other, name))
            total += weight * delta * delta
            divisor += weight
        return (total / max(divisor, 1e-9)) ** 0.5


@dataclass
class AffectReading:
    vector: AffectVector
    structures: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    backend: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector.to_dict(),
            "structures": list(self.structures),
            "roles": list(self.roles),
            "metadata": _json_safe(self.metadata),
            "backend": self.backend,
        }


@dataclass
class GateDecision:
    register: str = "neutral"
    severity: str = "low"
    masking: bool = False
    locked_pools: List[str] = field(default_factory=list)
    allowed_pools: List[str] = field(default_factory=list)
    response_act: str = "answer"
    max_sentences: int = 2
    requires_probe: bool = False
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateResponse:
    text: str
    construction_id: str
    semantic_valid: bool = True
    semantic_reason: str = ""
    affect: Optional[AffectVector] = None
    predicted_state: Optional[AffectVector] = None
    affect_distance: float = 0.0
    priority: int = 0
    score: float = 0.0
    atom_ids: List[str] = field(default_factory=list)
    semantic_plan: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "construction_id": self.construction_id,
            "semantic_valid": self.semantic_valid,
            "semantic_reason": self.semantic_reason,
            "affect": self.affect.to_dict() if self.affect else None,
            "predicted_state": self.predicted_state.to_dict() if self.predicted_state else None,
            "affect_distance": self.affect_distance,
            "priority": self.priority,
            "score": self.score,
            "atom_ids": list(self.atom_ids),
            "semantic_plan": list(self.semantic_plan),
        }


@dataclass
class TurnResult:
    input_text: str
    response: str
    parse: ParseResult
    contract: AnswerContract
    gates: GateDecision
    input_affect: AffectReading
    observed_state: AffectVector
    target_state: AffectVector
    predicted_state: AffectVector
    candidates: List[CandidateResponse] = field(default_factory=list)
    memory_revision: int = 0
    learning: Optional[Dict[str, Any]] = None
    resolver: Optional[Dict[str, Any]] = None
    trajectory: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "response": self.response,
            "parse": self.parse.to_dict(),
            "contract": self.contract.to_dict(),
            "gates": self.gates.to_dict(),
            "input_affect": self.input_affect.to_dict(),
            "observed_state": self.observed_state.to_dict(),
            "target_state": self.target_state.to_dict(),
            "predicted_state": self.predicted_state.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "memory_revision": self.memory_revision,
            "learning": _json_safe(self.learning),
            "resolver": _json_safe(self.resolver),
            "trajectory": _json_safe(self.trajectory),
        }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            return repr(value)
    if hasattr(value, "__dict__"):
        return {str(key): _json_safe(item) for key, item in vars(value).items() if not key.startswith("_")}
    return repr(value)
