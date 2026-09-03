"""Typed intermediate representations for deterministic Clanker-LM dialogue.

The language layer never passes unstructured dictionaries between stages.  Raw
text is converted into these small, serializable objects and each later stage
is allowed to add information without silently changing earlier claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class StringEnum(str, Enum):
    """Enum whose values serialize cleanly as strings on Python 3.10+."""

    def __str__(self) -> str:
        return self.value


class EntityKind(StringEnum):
    PERSON = "person"
    OBJECT = "object"
    PLACE = "place"
    EVENT = "event"
    ORGANIZATION = "organization"
    TIME = "time"
    CONCEPT = "concept"
    BODY_PART = "body_part"
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


class ValueKind(StringEnum):
    ENTITY = "entity"
    TEXT = "text"
    TIME = "time"
    NUMBER = "number"
    BOOLEAN = "boolean"


class SpeechAct(StringEnum):
    STATEMENT = "statement"
    QUESTION = "question"
    COMMAND = "command"
    GREETING = "greeting"
    SOCIAL_CHECKIN = "social_checkin"
    THANKS = "thanks"
    UNKNOWN = "unknown"


class QuestionFamily(StringEnum):
    WH = "wh"
    POLAR = "polar"
    WHY = "why"
    HOW = "how"
    SOCIAL = "social"


class WhyType(StringEnum):
    CAUSE = "cause"
    MOTIVE = "motive"
    PURPOSE = "purpose"
    JUSTIFICATION = "justification"
    EVIDENCE = "evidence"
    RHETORICAL = "rhetorical"
    UNKNOWN = "unknown"


class HowType(StringEnum):
    METHOD = "method"
    MANNER = "manner"
    PROCESS = "process"
    MECHANISM = "mechanism"
    DEGREE = "degree"
    QUANTITY = "quantity"
    CONDITION = "condition"
    SOCIAL = "social"
    UNKNOWN = "unknown"


class SemanticRole(StringEnum):
    AGENT = "agent"
    PATIENT = "patient"
    THEME = "theme"
    RECIPIENT = "recipient"
    LOCATION = "location"
    SOURCE = "source"
    TIME = "time"
    CAUSE = "cause"
    MOTIVE = "motive"
    PURPOSE = "purpose"
    JUSTIFICATION = "justification"
    EVIDENCE = "evidence"
    METHOD = "method"
    MANNER = "manner"
    PROCESS = "process"
    MECHANISM = "mechanism"
    DEGREE = "degree"
    QUANTITY = "quantity"
    ATTRIBUTE = "attribute"
    VALUE = "value"


class Provenance(StringEnum):
    UNKNOWN = "unknown"
    USER = "user"
    RETRIEVED = "retrieved"
    INFERRED = "inferred"
    EXTERNAL = "external"
    VERIFIED = "verified"


class AnswerStatus(StringEnum):
    ANSWERED = "answered"
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    PARTIAL_UNKNOWN = "partial_unknown"
    AMBIGUOUS = "ambiguous"
    NEEDS_CONTEXT = "needs_context"
    RHETORICAL = "rhetorical"


@dataclass(frozen=True)
class RoleValue:
    """A value occupying one semantic role in a proposition."""

    kind: ValueKind
    value: str
    display: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "kind": self.kind.value,
            "value": self.value,
            "display": self.display,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoleValue":
        return cls(
            kind=ValueKind(str(data["kind"])),
            value=str(data["value"]),
            display=str(data.get("display", data["value"])),
        )


@dataclass
class Entity:
    entity_id: str
    canonical_name: str
    display_name: str
    kind: EntityKind = EntityKind.UNKNOWN
    gender: Gender = Gender.UNKNOWN
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR
    relation_to_user: Optional[str] = None
    determiner: Optional[str] = None
    aliases: set[str] = field(default_factory=set)
    salience: float = 1.0
    last_turn: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "kind": self.kind.value,
            "gender": self.gender.value,
            "number": self.number.value,
            "relation_to_user": self.relation_to_user,
            "determiner": self.determiner,
            "aliases": sorted(self.aliases),
            "salience": self.salience,
            "last_turn": self.last_turn,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Entity":
        return cls(
            entity_id=str(data["entity_id"]),
            canonical_name=str(data["canonical_name"]),
            display_name=str(data.get("display_name", data["canonical_name"])),
            kind=EntityKind(str(data.get("kind", EntityKind.UNKNOWN.value))),
            gender=Gender(str(data.get("gender", Gender.UNKNOWN.value))),
            number=GrammaticalNumber(
                str(data.get("number", GrammaticalNumber.SINGULAR.value))
            ),
            relation_to_user=(
                str(data["relation_to_user"])
                if data.get("relation_to_user") is not None
                else None
            ),
            determiner=(
                str(data["determiner"]) if data.get("determiner") is not None else None
            ),
            aliases={str(item) for item in data.get("aliases", [])},
            salience=float(data.get("salience", 1.0)),
            last_turn=int(data.get("last_turn", 0)),
            metadata={str(k): str(v) for k, v in data.get("metadata", {}).items()},
        )


@dataclass
class SemanticFrame:
    predicate: str
    roles: Dict[SemanticRole, RoleValue] = field(default_factory=dict)
    tense: str = "present"
    polarity: bool = True
    modality: Optional[str] = None
    repeated: bool = False
    surface: str = ""

    def role(self, role: SemanticRole) -> Optional[RoleValue]:
        return self.roles.get(role)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "roles": {role.value: value.to_dict() for role, value in self.roles.items()},
            "tense": self.tense,
            "polarity": self.polarity,
            "modality": self.modality,
            "repeated": self.repeated,
            "surface": self.surface,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticFrame":
        return cls(
            predicate=str(data["predicate"]),
            roles={
                SemanticRole(str(role)): RoleValue.from_dict(value)
                for role, value in data.get("roles", {}).items()
            },
            tense=str(data.get("tense", "present")),
            polarity=bool(data.get("polarity", True)),
            modality=(str(data["modality"]) if data.get("modality") else None),
            repeated=bool(data.get("repeated", False)),
            surface=str(data.get("surface", "")),
        )


@dataclass
class Fact:
    fact_id: str
    frame: SemanticFrame
    provenance: Provenance = Provenance.USER
    certainty: int = 230
    turn_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "frame": self.frame.to_dict(),
            "provenance": self.provenance.value,
            "certainty": int(max(0, min(255, self.certainty))),
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fact":
        return cls(
            fact_id=str(data["fact_id"]),
            frame=SemanticFrame.from_dict(data["frame"]),
            provenance=Provenance(str(data.get("provenance", Provenance.USER.value))),
            certainty=int(data.get("certainty", 230)),
            turn_id=int(data.get("turn_id", 0)),
        )


@dataclass(frozen=True)
class UnresolvedReference:
    surface: str
    expected_kind: EntityKind = EntityKind.UNKNOWN
    compatible_entity_ids: Tuple[str, ...] = ()
    reason: str = "no compatible antecedent"


@dataclass
class QuestionFrame:
    family: QuestionFamily
    frame: SemanticFrame
    requested_roles: Tuple[SemanticRole, ...] = ()
    expected_type: str = "unknown"
    wh_word: Optional[str] = None
    why_type: WhyType = WhyType.UNKNOWN
    how_type: HowType = HowType.UNKNOWN
    unresolved_reference: Optional[UnresolvedReference] = None
    rhetorical: bool = False
    surface: str = ""


@dataclass
class ParsedUtterance:
    raw_text: str
    speech_act: SpeechAct
    frame: Optional[SemanticFrame] = None
    question: Optional[QuestionFrame] = None
    unresolved_references: Tuple[UnresolvedReference, ...] = ()
    tokens: Tuple[str, ...] = ()
    register_score: float = 0.0
    severity_score: float = 0.0
    familial: bool = False
    repeated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        object.__setattr__(self, "v", max(0, min(255, int(self.v))))
        object.__setattr__(self, "a", max(0, min(255, int(self.a))))
        object.__setattr__(self, "d", max(0, min(255, int(self.d))))
        object.__setattr__(self, "u", max(0, min(255, int(self.u))))
        object.__setattr__(self, "g", max(0, min(255, int(self.g))))
        object.__setattr__(self, "w", max(0, min(255, int(self.w))))
        object.__setattr__(self, "i", max(0, min(255, int(self.i))))

    def to_dict(self) -> Dict[str, int]:
        return {
            "v": self.v,
            "a": self.a,
            "d": self.d,
            "u": self.u,
            "g": self.g,
            "w": self.w,
            "i": self.i,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AffectVector":
        return cls(**{key: int(value.get(key, 128 if key != "u" else 0)) for key in "vadugwi"})


@dataclass(frozen=True)
class AffectReading:
    vector: AffectVector
    structures: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("engine", "unknown"))


@dataclass(frozen=True)
class GateProfile:
    register: str
    severity: str
    collision_masking: bool
    locked_pools: Tuple[str, ...] = ()
    required_acts: Tuple[str, ...] = ()
    forbidden_acts: Tuple[str, ...] = ()
    rationale: Tuple[str, ...] = ()


@dataclass
class AnswerContract:
    status: AnswerStatus
    question: QuestionFrame
    matching_facts: Tuple[Fact, ...] = ()
    selected_fact: Optional[Fact] = None
    bound_role: Optional[SemanticRole] = None
    bound_value: Optional[RoleValue] = None
    truth_value: Optional[bool] = None
    certainty: int = 0
    provenance: Provenance = Provenance.UNKNOWN
    explanation: str = ""


@dataclass(frozen=True)
class ResponsePlan:
    act: str
    slots: Mapping[str, str]
    gate: GateProfile
    target_mode: str = "support"
    required_tags: Tuple[str, ...] = ()
    forbidden_tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ResponseCandidate:
    candidate_id: str
    text: str
    tags: Tuple[str, ...] = ()
    semantic_signature: str = ""


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    text: str
    score: float
    semantic_valid: bool
    outcome: AffectVector
    reasons: Tuple[str, ...] = ()
    hard_rejected: bool = False
    selected: bool = False


@dataclass
class TurnResult:
    response: str
    parsed: ParsedUtterance
    gates: GateProfile
    input_affect: AffectReading
    response_affect: AffectReading
    resulting_state: AffectVector
    answer: Optional[AnswerContract] = None
    candidate_scores: Tuple[CandidateScore, ...] = ()
    stored_fact_ids: Tuple[str, ...] = ()

    def trace_dict(self) -> Dict[str, Any]:
        answer: Optional[Dict[str, Any]] = None
        if self.answer is not None:
            answer = {
                "status": self.answer.status.value,
                "bound_role": self.answer.bound_role.value if self.answer.bound_role else None,
                "bound_value": (
                    self.answer.bound_value.to_dict() if self.answer.bound_value else None
                ),
                "truth_value": self.answer.truth_value,
                "certainty": self.answer.certainty,
                "provenance": self.answer.provenance.value,
                "explanation": self.answer.explanation,
            }
        return {
            "response": self.response,
            "speech_act": self.parsed.speech_act.value,
            "input_affect": self.input_affect.vector.to_dict(),
            "response_affect": self.response_affect.vector.to_dict(),
            "resulting_state": self.resulting_state.to_dict(),
            "gates": {
                "register": self.gates.register,
                "severity": self.gates.severity,
                "collision_masking": self.gates.collision_masking,
                "locked_pools": list(self.gates.locked_pools),
                "required_acts": list(self.gates.required_acts),
                "forbidden_acts": list(self.gates.forbidden_acts),
                "rationale": list(self.gates.rationale),
            },
            "answer": answer,
            "stored_fact_ids": list(self.stored_fact_ids),
            "candidates": [
                {
                    "id": item.candidate_id,
                    "text": item.text,
                    "score": item.score,
                    "semantic_valid": item.semantic_valid,
                    "hard_rejected": item.hard_rejected,
                    "selected": item.selected,
                    "outcome": item.outcome.to_dict(),
                    "reasons": list(item.reasons),
                }
                for item in self.candidate_scores
            ],
        }
