"""Scope simple state reports before a downstream consumer uses their terms.

This module does not estimate emotions or assert causal response effects. It
carries the participant, polarity, modality and time already represented by a
supported frame. Callers must provide the reviewed single-word state vocabulary.
No source text is converted to a response template.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import AbstractSet, Optional

from . import lexicon
from .cessation import marker_of
from .recurrence import recurrence_of
from .model import EventFrame, ParseResult, RefKind, SourceKind, SpeechAct


@dataclass(frozen=True)
class ScopedStateReport:
    entity_id: str
    state_term: str
    polarity: bool
    tense: str
    temporal_scope: str
    modality: Optional[str]
    discourse_role: str
    source: str
    event_id: str
    turn_index: int

    @property
    def asserted(self) -> bool:
        return (
            self.polarity
            and not self.modality
            and self.discourse_role in {"", "main"}
            and self.source == SourceKind.USER.value
        )

    @property
    def current_self_report(self) -> bool:
        return self.entity_id == "user" and self.asserted and self.temporal_scope == "current"

    def to_dict(self) -> dict:
        return asdict(self)


def state_report(event: EventFrame, state_terms: AbstractSet[str]) -> Optional[ScopedStateReport]:
    """Return a qualified one-state frame or abstain; do not guess scope.

    Clauses containing multiple state terms or unsupported complement material
    are deliberately not flattened. This is a bounded state-report adapter,
    not a new sentiment parser or automatic seven-axis observer.
    """
    if event.predicate not in {"be", "feel"}:
        return None
    try:
        marker_of(event)
        recurrence_of(event)
    except ValueError:
        return None
    subject = next((event.arguments[k] for k in ("experiencer", "subject", "agent")
                    if k in event.arguments), None)
    state = event.arguments.get("state") or event.arguments.get("value")
    if subject is None or subject.kind != RefKind.ENTITY or state is None:
        return None
    words = [t.norm for t in lexicon.tokenize(state.surface or state.key, include_punctuation=False)]
    matches = [w for w in words if w in state_terms]
    # Preserve conservatism when the old parser leaves temporal adjuncts in
    # a copular value. No conjunction, negation or arbitrary phrase is stripped.
    adjuncts = {"very", "really", "quite", "so", "extremely", "now", "today", "yesterday", "tomorrow", "anymore"}
    if len(matches) != 1 or any(w not in state_terms | adjuncts for w in words):
        return None
    time_ref = event.arguments.get("time")
    time_words = set(words)
    if time_ref is not None:
        time_words |= {t.norm for t in lexicon.tokenize(time_ref.surface or time_ref.key, include_punctuation=False)}
    historical = event.tense == "past" or "yesterday" in time_words
    projected = event.tense == "future" or "tomorrow" in time_words
    if historical and projected:
        temporal = "conflicting"
    elif event.modality or projected:
        temporal = "projected_or_modal"
    elif historical:
        temporal = "historical"
    elif time_ref is not None and (time_ref.surface or time_ref.key).lower().strip() not in {"now", "today"}:
        # A present-tense clause with an unnormalized date is not automatically
        # about this instant. Calendar grounding belongs to the temporal layer.
        temporal = "unspecified"
    elif event.tense == "present":
        temporal = "current"
    else:
        temporal = "unspecified"
    return ScopedStateReport(
        entity_id=subject.key, state_term=matches[0], polarity=event.polarity,
        tense=event.tense, temporal_scope=temporal, modality=event.modality,
        discourse_role=event.discourse_role, source=event.source.value,
        event_id=event.event_id, turn_index=event.turn_index,
    )


def isolated_negated_state(parse: ParseResult, state_terms: AbstractSet[str]) -> Optional[ScopedStateReport]:
    """A negated state is not evidence for either that state or its opposite."""
    if parse.speech_act != SpeechAct.ASSERT or parse.unresolved or len(parse.events) != 1:
        return None
    report = state_report(parse.events[0], state_terms)
    if report and not report.polarity and not report.modality and report.discourse_role in {"", "main"} and report.source == SourceKind.USER.value:
        return report
    return None
