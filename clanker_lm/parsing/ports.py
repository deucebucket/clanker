"""Narrow structural contract between dispatch and construction handlers."""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import AppositiveAttachmentAmbiguity, ContentAttachmentAmbiguity, EmbeddedInterrogativeAttachmentAmbiguity, EventFrame, GerundAttachmentAmbiguity, GerundRelationType, ModifierAttachmentAmbiguity, InfinitivalAttachmentAmbiguity, InfinitivalRelationType, QuestionFrame, UnresolvedReference
from .types import ClauseResult, ContentSplit, EmbeddedInterrogativeSplit, InfinitivalSplit, GerundSplit, RelativeSplit, AppositiveSplit, SubordinateSplit
from typing import Protocol

class ParserPort(Protocol):
    """Structural interface used by assembly handlers; no runtime backreferences."""

    def _clause_has_resolved_subject(self, tokens: Sequence[lexicon.Token], event: EventFrame) -> bool:
        ...

    def _content_source_entity(self, event: EventFrame) -> str:
        ...

    def _embedded_interrogative_ambiguity(self, tokens: Sequence[lexicon.Token], boundaries: Sequence[int], *, reason: str) -> EmbeddedInterrogativeAttachmentAmbiguity:
        ...

    def _event_signature_key(self, event: EventFrame) -> str:
        ...

    def _gerund_ambiguity(self, tokens: Sequence[lexicon.Token], boundaries: Sequence[int], *, reason: str, candidate_types: Sequence[GerundRelationType]=()) -> GerundAttachmentAmbiguity:
        ...

    def _gerund_controller_entity(self, event: EventFrame, controller_role: str) -> str:
        ...

    def _gerund_source_entity(self, event: EventFrame) -> str:
        ...

    def _infinitival_ambiguity(self, tokens: Sequence[lexicon.Token], boundaries: Sequence[int], *, reason: str, candidate_types: Sequence[InfinitivalRelationType]=()) -> InfinitivalAttachmentAmbiguity:
        ...

    def _infinitival_controller_entity(self, event: EventFrame, controller_role: str) -> str:
        ...

    def _is_factual_phase_matrix(self, event: EventFrame, complement: EventFrame) -> bool:
        ...

    def _parse_clause(self, tokens: Sequence[lexicon.Token], raw: str, memory: ConversationMemory) -> ClauseResult:
        ...

    def _parse_embedded_question(self, tokens: Sequence[lexicon.Token], raw: str, memory: ConversationMemory) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        ...

    def _restore_memory_checkpoint(self, memory: ConversationMemory, checkpoint: Dict[str, object]) -> None:
        ...

    def _split_appositive_clause(self, tokens: Sequence[lexicon.Token], raw: str, memory: ConversationMemory) -> Tuple[Optional[AppositiveSplit], Optional[AppositiveAttachmentAmbiguity]]:
        ...

    def _split_content_clause(self, tokens: Sequence[lexicon.Token]) -> Tuple[Optional[ContentSplit], Optional[ContentAttachmentAmbiguity]]:
        ...

    def _split_embedded_interrogative_clause(self, tokens: Sequence[lexicon.Token]) -> Tuple[Optional[EmbeddedInterrogativeSplit], Optional[EmbeddedInterrogativeAttachmentAmbiguity]]:
        ...

    def _split_gerund_clause(self, tokens: Sequence[lexicon.Token]) -> Tuple[Optional[GerundSplit], Optional[GerundAttachmentAmbiguity]]:
        ...

    def _split_infinitival_clause(self, tokens: Sequence[lexicon.Token]) -> Tuple[Optional[InfinitivalSplit], Optional[InfinitivalAttachmentAmbiguity]]:
        ...

    def _split_relative_clause(self, tokens: Sequence[lexicon.Token], raw: str, memory: ConversationMemory) -> Tuple[Optional[RelativeSplit], Optional[ModifierAttachmentAmbiguity]]:
        ...

    def _split_subordinate_clause(self, tokens: Sequence[lexicon.Token]) -> Optional[SubordinateSplit]:
        ...

    def _surface(self, tokens: Sequence[lexicon.Token]) -> str:
        ...
