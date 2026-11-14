"""Matching responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import List, Sequence, Tuple
from ..memory import ConversationMemory, EventMatch
from ..model import EntityKind, Evidence, EventFrame, QuestionFrame, QuestionKind, RefKind, SemanticRef, SourceKind

class MatchingComponent:
    """Matching behavior composed by the stable public QuestionAnswerer API."""

    def _match_attribute_query(self, question: QuestionFrame, memory: ConversationMemory) -> List[EventMatch]:
        query = question.event
        subject = query.arguments.get("subject")
        attribute = query.arguments.get("attribute")
        matches: List[EventMatch] = []
        for event in memory.events:
            if event.discourse_role in memory.NONASSERTIVE_DISCOURSE_ROLES:
                continue
            if event.predicate not in {"attribute", "be"}:
                continue
            if subject:
                actual_subject = event.arguments.get("subject") or event.arguments.get("agent")
                if not actual_subject or not memory.refs_equal(subject, actual_subject):
                    continue
            if attribute:
                actual_attribute = event.arguments.get("attribute")
                if actual_attribute and not memory.refs_equal(attribute, actual_attribute):
                    continue
                # No attribute tag means we cannot safely claim that a generic
                # copular value answers a specific color/age/etc. question.
                if actual_attribute is None and attribute.key not in {"state", "identity"}:
                    continue
            value = event.arguments.get("value")
            if value is None:
                continue
            matches.append(EventMatch(event, ["subject", "attribute"], 25.0 + event.certainty / 128.0))

        # Noun-phrase modifiers are stored directly on their entity (for
        # example ``a red car`` or ``blue eyes``).  Expose those explicit
        # attributes as a synthetic evidence frame rather than discarding the
        # information or inventing a value at answer time.
        if subject and subject.kind == RefKind.ENTITY and attribute:
            entity = memory.get_entity(subject.key)
            if entity:
                stored_value = entity.attributes.get(attribute.key)
                if stored_value:
                    synthetic = EventFrame(
                        "attribute",
                        {
                            "subject": subject,
                            "attribute": attribute,
                            "value": SemanticRef.literal(stored_value, stored_value, EntityKind.ABSTRACT),
                        },
                        tense="present",
                        source=SourceKind.USER,
                        certainty=230,
                        turn_index=entity.last_mentioned_turn,
                        inferred=False,
                    )
                    matches.append(EventMatch(synthetic, ["subject", "attribute"], 27.0))
        matches.sort(key=lambda item: (item.score, item.event.turn_index), reverse=True)
        return matches

    def _candidate_roles(self, question: QuestionFrame) -> Tuple[str, ...]:
        requested = question.requested_role or ""
        if question.kind == QuestionKind.WHY:
            return self.WHY_FALLBACKS.get(question.why_kind, (requested,))
        if question.kind == QuestionKind.HOW:
            return self.HOW_FALLBACKS.get(question.how_kind, (requested,))
        by_kind = self.QUESTION_ROLE_FALLBACKS.get(question.kind, {})
        if requested in by_kind:
            return by_kind[requested]
        if requested == "event":
            return ("event",)
        return (requested,)

    @staticmethod
    def _event_aspects_compatible(query: EventFrame, event: EventFrame) -> bool:
        """Keep direct QA inside the stored event's aspect boundary.

        Simple, progressive, perfect, and perfect-progressive propositions are
        distinct facts.  Any licensed implication between them must come from
        an explicit typed relation rather than the general event matcher.
        """

        return query.aspect == event.aspect

    @staticmethod
    def _to_evidence(match: EventMatch) -> Evidence:
        return Evidence(match.event, list(match.matched_roles), match.score)

    @staticmethod
    def _combine_sources(events: Sequence[EventFrame]) -> SourceKind:
        if not events:
            return SourceKind.UNKNOWN
        sources = {event.source for event in events}
        if len(sources) == 1:
            return next(iter(sources))
        if SourceKind.VERIFIED in sources:
            return SourceKind.VERIFIED
        return SourceKind.INFERRED

    @staticmethod
    def _same_anchor_sets(
        positive: Sequence[EventMatch],
        negative: Sequence[EventMatch],
        requested_role: str,
    ) -> bool:
        pos = {item.event.signature(exclude_roles={requested_role}) for item in positive}
        neg = {item.event.signature(exclude_roles={requested_role}) for item in negative}
        return bool(pos & neg)

    @staticmethod
    def _loosely_related(query: EventFrame, memory: ConversationMemory) -> List[EventMatch]:
        """Find facts sharing predicate and at least one argument.

        These are diagnostic evidence only; they never convert UNKNOWN to FALSE.
        """

        results: List[EventMatch] = []
        fixed = query.fixed_arguments()
        for event in memory.events:
            if event.discourse_role in memory.NONASSERTIVE_DISCOURSE_ROLES:
                continue
            if event.predicate != query.predicate:
                continue
            matched = [
                role for role, expected in fixed.items()
                if role in event.arguments and memory.refs_equal(expected, event.arguments[role])
            ]
            if matched:
                results.append(EventMatch(event, matched, len(matched) * 5.0))
        results.sort(key=lambda item: (item.score, item.event.turn_index), reverse=True)
        return results
