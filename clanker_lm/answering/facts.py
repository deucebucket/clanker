"""Facts responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from ..memory import ConversationMemory, EventMatch
from ..model import AnswerContract, AnswerStatus, Evidence, EventFrame, QuestionFrame, RefKind, SemanticRef, SourceKind, TruthValue

class FactsComponent:
    """Facts behavior composed by the stable public QuestionAnswerer API."""

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
            if event.discourse_role
            not in memory.NONASSERTIVE_DISCOURSE_ROLES
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
        matches = [
            match
            for match in memory.match_events(
                query,
                include_opposite_polarity=True,
            )
            if self._event_aspects_compatible(query, match.event)
        ]
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
            if self._event_aspects_compatible(query, match.event)
            and match.event.discourse_role == "content"
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

        gerund = self._answer_gerund_truth_boundary(
            question,
            memory,
        )
        if gerund is not None:
            return gerund

        infinitival = self._answer_infinitival_truth_boundary(
            question,
            memory,
        )
        if infinitival is not None:
            return infinitival

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
            matches = [
                match
                for match in memory.related_events(query, requested)
                if self._event_aspects_compatible(query, match.event)
            ]

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
