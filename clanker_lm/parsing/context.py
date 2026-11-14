"""Typed, call-local assembly context. No global parser accumulator."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from ..memory import ConversationMemory
from ..model import AppositiveAttachmentAmbiguity, AppositiveRelation, ClauseRelation, ContentAttachmentAmbiguity, ContentRelation, EmbeddedInterrogativeAttachmentAmbiguity, EmbeddedInterrogativeRelation, EntityModifierRelation, EventFrame, GerundAttachmentAmbiguity, GerundRelation, ModifierAttachmentAmbiguity, InfinitivalAttachmentAmbiguity, InfinitivalRelation, ParseResult, QuestionFrame, SpeechAct, UnresolvedReference
from dataclasses import dataclass, field

@dataclass
class ParseContext:
    """One invocation's source, memory handle and construction outputs; never global.

    Handlers append to these original lists in the documented dispatch order.
    No persistence, response generation or learned weights live in this object.
    """
    raw: str
    normalized: str
    memory: ConversationMemory
    events: List[EventFrame] = field(default_factory=list)
    relations: List[ClauseRelation] = field(default_factory=list)
    modifiers: List[EntityModifierRelation] = field(default_factory=list)
    modifier_ambiguities: List[ModifierAttachmentAmbiguity] = field(default_factory=list)
    appositives: List[AppositiveRelation] = field(default_factory=list)
    appositive_ambiguities: List[AppositiveAttachmentAmbiguity] = field(default_factory=list)
    contents: List[ContentRelation] = field(default_factory=list)
    content_ambiguities: List[ContentAttachmentAmbiguity] = field(default_factory=list)
    embedded_interrogatives: List[EmbeddedInterrogativeRelation] = field(default_factory=list)
    embedded_interrogative_ambiguities: List[EmbeddedInterrogativeAttachmentAmbiguity] = field(default_factory=list)
    infinitivals: List[InfinitivalRelation] = field(default_factory=list)
    infinitival_ambiguities: List[InfinitivalAttachmentAmbiguity] = field(default_factory=list)
    gerunds: List[GerundRelation] = field(default_factory=list)
    gerund_ambiguities: List[GerundAttachmentAmbiguity] = field(default_factory=list)
    unresolved: List[UnresolvedReference] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    direct_embedded_question: Optional[QuestionFrame] = None
    primary_event_count: int = 0

    def result(self) -> ParseResult:
        return ParseResult(
            speech_act=(
                SpeechAct.COMMAND
                if self.direct_embedded_question is not None
                else SpeechAct.ASSERT
                if self.events
                else SpeechAct.UNKNOWN
            ),
            raw_text=self.raw,
            events=self.events,
            relations=self.relations,
            modifiers=self.modifiers,
            modifier_ambiguities=self.modifier_ambiguities,
            appositives=self.appositives,
            appositive_ambiguities=self.appositive_ambiguities,
            contents=self.contents,
            content_ambiguities=self.content_ambiguities,
            embedded_interrogatives=self.embedded_interrogatives,
            embedded_interrogative_ambiguities=(
                self.embedded_interrogative_ambiguities
            ),
            infinitivals=self.infinitivals,
            infinitival_ambiguities=self.infinitival_ambiguities,
            gerunds=self.gerunds,
            gerund_ambiguities=self.gerund_ambiguities,
            question=self.direct_embedded_question,
            entities=list(dict.fromkeys(self.entities)),
            unresolved=self.unresolved,
            normalized_text=self.normalized,
            diagnostics=self.diagnostics,
        )
