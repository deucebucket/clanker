"""Typed question answering over Clanker-LM's symbolic memory."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .memory import ConversationMemory, EventMatch
from .model import (
    AnswerContract,
    AnswerStatus,
    EntityKind,
    Evidence,
    EventFrame,
    HowKind,
    QuestionFrame,
    QuestionKind,
    RefKind,
    SemanticRef,
    SourceKind,
    TruthValue,
    WhyKind,
)


class QuestionAnswerer:
    """Bind a question's typed hole from explicit evidence in memory."""

    WHY_FALLBACKS: Dict[WhyKind, Tuple[str, ...]] = {
        WhyKind.CAUSE: ("cause", "motive", "purpose", "justification", "evidence"),
        WhyKind.MOTIVE: ("motive", "cause", "purpose"),
        WhyKind.PURPOSE: ("purpose", "motive", "cause"),
        WhyKind.JUSTIFICATION: ("justification", "cause", "motive"),
        WhyKind.EVIDENCE: ("evidence", "cause", "justification"),
        WhyKind.UNKNOWN: ("cause", "motive", "purpose", "justification", "evidence"),
    }
    HOW_FALLBACKS: Dict[HowKind, Tuple[str, ...]] = {
        HowKind.METHOD: ("method", "manner", "process", "mechanism"),
        HowKind.MANNER: ("manner", "method"),
        HowKind.MECHANISM: ("mechanism", "cause", "method", "process"),
        HowKind.PROCESS: ("process", "method", "mechanism"),
        HowKind.DEGREE: ("value", "quantity"),
        HowKind.QUANTITY: ("quantity",),
        HowKind.STATE: ("value", "state"),
        HowKind.UNKNOWN: ("method", "manner", "process", "mechanism", "value"),
    }
    QUESTION_ROLE_FALLBACKS: Dict[QuestionKind, Dict[str, Tuple[str, ...]]] = {
        QuestionKind.WHERE: {
            "location": ("location", "destination", "goal"),
            "destination": ("destination", "location", "goal"),
            "source": ("source",),
        },
        QuestionKind.WHEN: {"time": ("time",)},
        QuestionKind.WHOSE: {"possessor": ("possessor", "owner")},
    }

    def answer(self, question: QuestionFrame, memory: ConversationMemory) -> AnswerContract:
        if question.social_convention:
            return AnswerContract(
                status=AnswerStatus.ANSWERED,
                question=question,
                certainty=255,
                source=SourceKind.TRAINED,
                response_goal="social",
                reason=question.social_convention,
            )

        if question.unresolved:
            unresolved = question.unresolved[0]
            if unresolved.candidates:
                return AnswerContract(
                    status=AnswerStatus.AMBIGUOUS_REFERENCE,
                    question=question,
                    certainty=255,
                    source=SourceKind.INFERRED,
                    reason=unresolved.surface,
                    required_slots={
                        "reference": unresolved.surface,
                        "candidate_ids": ",".join(unresolved.candidates),
                    },
                    response_goal="clarify",
                )
            return AnswerContract(
                status=AnswerStatus.MISSING_REFERENCE,
                question=question,
                certainty=255,
                source=SourceKind.INFERRED,
                reason=unresolved.surface,
                required_slots={"reference": unresolved.surface},
                response_goal="clarify",
            )

        if question.kind == QuestionKind.WHAT_HAPPENED:
            return self._answer_event_query(question, memory)
        attributed = self._answer_attributed_content_request(question, memory)
        if attributed is not None:
            return attributed
        if question.kind == QuestionKind.WHOSE:
            ownership = self._answer_possessor(question, memory)
            if ownership is not None:
                return ownership
        if question.kind == QuestionKind.YES_NO:
            return self._answer_yes_no(question, memory)
        if not question.requested_role:
            return AnswerContract(
                status=AnswerStatus.UNSUPPORTED,
                question=question,
                reason="question has no typed open slot",
                response_goal="clarify",
            )
        return self._answer_open_slot(question, memory)


    CONTENT_QUERY_PREDICATES = {
        "say", "tell", "report", "claim",
        "think", "believe", "know",
        "notice", "hear",
    }

    def _answer_attributed_content_request(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        """Answer ``What did X say/think/...`` from typed content links."""

        if (
            question.kind != QuestionKind.WHAT
            or question.requested_role not in {"patient", "content"}
            or question.event.predicate not in self.CONTENT_QUERY_PREDICATES
        ):
            return None
        source_ref = (
            question.event.arguments.get("agent")
            or question.event.arguments.get("experiencer")
            or question.event.arguments.get("subject")
        )
        if source_ref is None or source_ref.kind != RefKind.ENTITY:
            return None
        relations = memory.content_relations_for_source(
            source_ref.key,
            matrix_predicate=question.event.predicate,
        )
        if not relations:
            negated = memory.content_relations_for_source(
                source_ref.key,
                matrix_predicate=question.event.predicate,
                include_negated=True,
            )
            if negated:
                return AnswerContract(
                    status=AnswerStatus.UNKNOWN,
                    question=question,
                    certainty=0,
                    source=SourceKind.ATTRIBUTED,
                    reason="only a negated content attribution is stored",
                    response_goal="answer",
                    required_slots={
                        "unknown_object": "content",
                        "source_entity_id": source_ref.key,
                        "matrix_predicate": question.event.predicate,
                    },
                    forbidden_claims=["invert_negated_attribution"],
                )
            return None

        resolved = []
        for relation in relations:
            content_event = memory.get_event(relation.content_event_id)
            matrix_event = memory.get_event(relation.matrix_event_id)
            if content_event is None or matrix_event is None:
                continue
            if question.event.tense and matrix_event.tense != question.event.tense:
                continue
            resolved.append((relation, matrix_event, content_event))
        if not resolved:
            return None

        latest_turn = max(matrix.turn_index for _relation, matrix, _content in resolved)
        latest = [item for item in resolved if item[1].turn_index == latest_turn]
        signatures = {
            item[2].signature()
            for item in latest
        }
        if len(signatures) > 1:
            values = [
                SemanticRef.event(content.event_id, content.raw_text)
                for _relation, _matrix, content in latest
            ]
            return AnswerContract(
                status=AnswerStatus.MULTIPLE_MATCHES,
                question=question,
                proposition=latest[0][2],
                values=values,
                evidence=[Evidence(content, score=1.0) for _relation, _matrix, content in latest],
                certainty=min(relation.certainty for relation, _matrix, _content in latest),
                source=SourceKind.ATTRIBUTED,
                reason="multiple attributed contents are stored for the latest turn",
                response_goal="clarify",
                required_slots={
                    "attributed": "true",
                    "source_entity_id": source_ref.key,
                    "matrix_predicate": question.event.predicate,
                },
            )

        relation, matrix_event, content_event = max(
            latest,
            key=lambda item: (item[0].certainty, item[2].certainty),
        )
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=content_event,
            values=[SemanticRef.event(content_event.event_id, content_event.raw_text)],
            evidence=[Evidence(content_event, matched_roles=["content"], score=1.0)],
            certainty=min(relation.certainty, content_event.certainty),
            source=SourceKind.ATTRIBUTED,
            reason="bound finite content through an explicit attribution relation",
            response_goal="answer",
            required_slots={
                "attributed": "true",
                "source_entity_id": relation.source_entity_id,
                "matrix_predicate": relation.matrix_predicate,
                "matrix_tense": matrix_event.tense,
                "relation_type": relation.relation_type.value,
            },
        )

    def _answer_possessor(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        patient = question.event.arguments.get("patient")
        if patient is None or patient.kind != RefKind.ENTITY:
            return None
        entity = memory.get_entity(patient.key)
        if entity is None or not entity.owner_id:
            return None
        owner = memory.get_entity(entity.owner_id)
        if owner is None:
            return None
        proposition = EventFrame(
            predicate="own",
            arguments={
                "possessor": owner.to_ref(),
                "patient": entity.to_ref(),
            },
            source=SourceKind.USER,
            certainty=230,
            turn_index=max(entity.last_mentioned_turn, owner.last_mentioned_turn),
        )
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=proposition,
            values=[owner.to_ref()],
            evidence=[Evidence(proposition, matched_roles=["patient"], score=1.0)],
            certainty=230,
            source=SourceKind.USER,
            reason="possessed entity stores an explicit owner relation",
            response_goal="answer",
        )

    def _answer_event_query(self, question: QuestionFrame, memory: ConversationMemory) -> AnswerContract:
        candidates = [
            event
            for event in memory.events
            if event.discourse_role != "content"
        ]
        fixed = question.event.fixed_arguments()
        if fixed:
            filtered: List[EventFrame] = []
            for event in candidates:
                if all(
                    role in event.arguments and memory.refs_equal(expected, event.arguments[role])
                    for role, expected in fixed.items()
                ):
                    filtered.append(event)
            candidates = filtered
        if not candidates:
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                reason="no event in memory matches the requested subject",
                response_goal="answer",
                forbidden_claims=["invent_event"],
            )
        latest_turn = max(event.turn_index for event in candidates)
        latest = [event for event in candidates if event.turn_index == latest_turn]
        if len(latest) > 1:
            return AnswerContract(
                status=AnswerStatus.MULTIPLE_MATCHES,
                question=question,
                proposition=latest[0],
                values=[SemanticRef.event(event.event_id, event.raw_text) for event in latest],
                evidence=[Evidence(event, score=1.0) for event in latest],
                certainty=min(event.certainty for event in latest),
                source=self._combine_sources(latest),
                reason="multiple events occurred in the latest turn",
                response_goal="clarify",
            )
        event = latest[0]
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=event,
            values=[SemanticRef.event(event.event_id, event.raw_text)],
            evidence=[Evidence(event, score=1.0)],
            certainty=event.certainty,
            source=event.source,
            response_goal="answer",
        )

    def _answer_yes_no(self, question: QuestionFrame, memory: ConversationMemory) -> AnswerContract:
        query = question.event
        matches = memory.match_events(query, include_opposite_polarity=True)
        same = [match for match in matches if match.event.polarity == query.polarity]
        opposite = [match for match in matches if match.event.polarity != query.polarity]

        if same and opposite:
            evidence = [self._to_evidence(item) for item in same + opposite]
            return AnswerContract(
                status=AnswerStatus.CONFLICT,
                question=question,
                truth=TruthValue.CONFLICT,
                evidence=evidence,
                certainty=min(item.event.certainty for item in same + opposite),
                source=self._combine_sources([item.event for item in same + opposite]),
                reason="both proposition and negation are stored",
                response_goal="warn",
            )
        if same:
            winner = same[0]
            return AnswerContract(
                status=AnswerStatus.TRUE,
                question=question,
                proposition=winner.event,
                truth=TruthValue.TRUE,
                evidence=[self._to_evidence(item) for item in same],
                certainty=max(item.event.certainty for item in same),
                source=self._combine_sources([item.event for item in same]),
                reason="direct proposition match",
                response_goal="answer",
            )
        if opposite:
            winner = opposite[0]
            return AnswerContract(
                status=AnswerStatus.FALSE,
                question=question,
                proposition=winner.event,
                truth=TruthValue.FALSE,
                evidence=[self._to_evidence(item) for item in opposite],
                certainty=max(item.event.certainty for item in opposite),
                source=self._combine_sources([item.event for item in opposite]),
                reason="explicit negated proposition match",
                response_goal="answer",
            )

        attributed = memory.match_events(
            query,
            include_opposite_polarity=True,
            include_attributed_content=True,
        )
        attributed = [
            match
            for match in attributed
            if match.event.discourse_role == "content"
            and any(
                relation.attributed
                and relation.content_event_id == match.event.event_id
                for relation in memory.content_relations_for_event(match.event.event_id)
            )
        ]
        if attributed:
            sources: List[str] = []
            matrix_predicates: List[str] = []
            for match in attributed:
                for relation in memory.content_relations_for_event(match.event.event_id):
                    if relation.content_event_id != match.event.event_id:
                        continue
                    if relation.source_entity_id not in sources:
                        sources.append(relation.source_entity_id)
                    if relation.matrix_predicate not in matrix_predicates:
                        matrix_predicates.append(relation.matrix_predicate)
            polarities = {match.event.polarity for match in attributed}
            reason = (
                "attributed sources disagree about the proposition"
                if len(polarities) > 1
                else "only attributed content supports the proposition"
            )
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                proposition=query,
                truth=TruthValue.UNKNOWN,
                evidence=[self._to_evidence(item) for item in attributed],
                certainty=0,
                source=SourceKind.ATTRIBUTED,
                reason=reason,
                response_goal="answer",
                required_slots={
                    "attributed": "true",
                    "source_entity_ids": ",".join(sources),
                    "matrix_predicates": ",".join(matrix_predicates),
                    "attributed_polarity_count": str(len(polarities)),
                },
                forbidden_claims=[
                    "promote_attributed_content_to_unqualified_fact",
                    "convert_attributed_content_to_truth",
                ],
                diagnostics=[reason],
            )

        # A related fact is useful context but is not logically equivalent to a
        # negation.  "Sarah bought it" does not prove that Mary had no role.
        related = self._loosely_related(query, memory)
        diagnostics: List[str] = []
        if related:
            diagnostics.append("related facts exist, but none prove or disprove the proposition")
        return AnswerContract(
            status=AnswerStatus.UNKNOWN,
            question=question,
            truth=TruthValue.UNKNOWN,
            evidence=[self._to_evidence(item) for item in related[:3]],
            certainty=0,
            source=SourceKind.UNKNOWN,
            reason="the stored evidence does not establish this proposition",
            response_goal="answer",
            forbidden_claims=["convert_absence_of_evidence_to_false"],
            diagnostics=diagnostics,
        )

    def _answer_open_slot(self, question: QuestionFrame, memory: ConversationMemory) -> AnswerContract:
        requested = question.requested_role or ""
        role_candidates = self._candidate_roles(question)
        query = question.event

        # Attribute questions are normalized against both explicit attribute
        # frames and ordinary copular facts carrying an attribute marker.
        if query.predicate == "attribute":
            matches = self._match_attribute_query(question, memory)
        else:
            matches = memory.related_events(query, requested)

        if not matches:
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                reason="no matching proposition is stored",
                response_goal="answer",
                forbidden_claims=["invent_missing_fact"],
            )

        positive_matches = [match for match in matches if match.event.polarity]
        negative_matches = [match for match in matches if not match.event.polarity]
        if positive_matches and negative_matches and self._same_anchor_sets(positive_matches, negative_matches, requested):
            return AnswerContract(
                status=AnswerStatus.CONFLICT,
                question=question,
                evidence=[self._to_evidence(item) for item in matches],
                certainty=min(item.event.certainty for item in matches),
                source=self._combine_sources([item.event for item in matches]),
                reason="matching positive and negative facts conflict",
                response_goal="warn",
            )

        usable: List[Tuple[EventMatch, str, SemanticRef]] = []
        for match in positive_matches or matches:
            for role in role_candidates:
                value = match.event.arguments.get(role)
                if value is not None and not value.is_variable:
                    usable.append((match, role, value))
                    break

        if not usable:
            # The base event is known, but the requested relation is not.
            best = matches[0]
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                proposition=best.event,
                evidence=[self._to_evidence(item) for item in matches[:3]],
                certainty=max(item.event.certainty for item in matches),
                source=self._combine_sources([item.event for item in matches]),
                reason="base proposition known; requested slot absent",
                response_goal="answer",
                required_slots={"missing_role": requested},
                forbidden_claims=[f"invent_{requested}"],
            )

        grouped: Dict[Tuple[str, str], List[Tuple[EventMatch, str, SemanticRef]]] = defaultdict(list)
        for item in usable:
            value = item[2]
            grouped[(value.kind.value, value.key)].append(item)

        if len(grouped) > 1:
            values = [items[0][2] for items in grouped.values()]
            evidence = [self._to_evidence(item[0]) for items in grouped.values() for item in items]
            return AnswerContract(
                status=AnswerStatus.MULTIPLE_MATCHES,
                question=question,
                proposition=usable[0][0].event,
                values=values,
                evidence=evidence,
                certainty=min(item[0].event.certainty for item in usable),
                source=self._combine_sources([item[0].event for item in usable]),
                reason="more than one distinct value satisfies the open slot",
                response_goal="clarify",
            )

        selected_items = next(iter(grouped.values()))
        selected_match, actual_role, value = max(
            selected_items,
            key=lambda item: (item[0].score, item[0].event.certainty, item[0].event.turn_index),
        )
        proposition = selected_match.event
        diagnostics: List[str] = []
        if actual_role != requested:
            diagnostics.append(f"bound {requested} through compatible role {actual_role}")
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=proposition,
            values=[value],
            evidence=[self._to_evidence(item[0]) for item in selected_items],
            certainty=max(item[0].event.certainty for item in selected_items),
            source=self._combine_sources([item[0].event for item in selected_items]),
            reason=f"bound open slot {requested}",
            response_goal="answer",
            required_slots={"requested_role": requested, "actual_role": actual_role},
            diagnostics=diagnostics,
        )

    def _match_attribute_query(self, question: QuestionFrame, memory: ConversationMemory) -> List[EventMatch]:
        query = question.event
        subject = query.arguments.get("subject")
        attribute = query.arguments.get("attribute")
        matches: List[EventMatch] = []
        for event in memory.events:
            if event.discourse_role == "content":
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
            if event.discourse_role == "content":
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
