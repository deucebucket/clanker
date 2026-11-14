"""Parses data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .content import ContentAttachmentAmbiguity, ContentRelation
from .embedded import EmbeddedInterrogativeAttachmentAmbiguity, EmbeddedInterrogativeRelation
from .enums import SpeechAct
from .events import ClauseRelation, EventFrame
from .gerunds import GerundAttachmentAmbiguity, GerundRelation
from .infinitivals import InfinitivalAttachmentAmbiguity, InfinitivalRelation
from .modifiers import AppositiveAttachmentAmbiguity, AppositiveRelation, EntityModifierRelation, ModifierAttachmentAmbiguity
from .questions import QuestionFrame
from .references import UnresolvedReference


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
    gerunds: List[GerundRelation] = field(default_factory=list)
    gerund_ambiguities: List[GerundAttachmentAmbiguity] = field(
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
            "gerunds": [item.to_dict() for item in self.gerunds],
            "gerund_ambiguities": [
                ambiguity.to_dict() for ambiguity in self.gerund_ambiguities
            ],
            "question": self.question.to_dict() if self.question else None,
            "entities": list(self.entities),
            "unresolved": [item.to_dict() for item in self.unresolved],
            "normalized_text": self.normalized_text,
            "diagnostics": list(self.diagnostics),
            "understood": self.understood,
        }
