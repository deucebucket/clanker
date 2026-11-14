"""Enums data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from enum import Enum


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


class GerundRelationType(StringEnum):
    """Syntactic/semantic family of a selected ``-ing`` complement."""

    GERUND_CONTENT = "gerund_content"
    ASPECTUAL_START = "aspectual_start"
    ASPECTUAL_STOP = "aspectual_stop"
    ASPECTUAL_CONTINUATION = "aspectual_continuation"
    PERCEPTION_PARTICIPIAL = "perception_participial"


class GerundContentStatus(StringEnum):
    """Qualified occurrence status contributed by the matrix predicate."""

    ENJOYED = "enjoyed"
    AVOIDED = "avoided"
    BEGUN = "begun"
    STOPPED = "stopped"
    CONTINUED = "continued"
    PERCEIVED = "perceived"


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
