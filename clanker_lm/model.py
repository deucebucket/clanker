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
        }


@dataclass
class ParseResult:
    speech_act: SpeechAct
    raw_text: str
    events: List[EventFrame] = field(default_factory=list)
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
