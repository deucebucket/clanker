"""Gerund Queries responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import AnswerContract, AnswerStatus, EntityKind, EventFrame, GerundRelation, QuestionFrame, QuestionKind, RefKind, SemanticRef, SourceKind, TruthValue

class GerundQueriesComponent:
    """Gerund Queries behavior composed by the stable public QuestionAnswerer API."""

    def _answer_gerund_request(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        """Answer through a selected gerund/participial matrix relation.

        The answer may expose the selected content, its controller, or one of
        its open arguments, but the embedded frame remains qualified by the
        matrix predicate.  In particular, enjoying, avoiding, or perceiving an
        event never stores that event as an unqualified direct fact.
        """

        predicate = lexicon.lemma(question.event.predicate)
        if predicate not in self.GERUND_QUERY_PREDICATES:
            return None

        source_ref = (
            question.event.arguments.get("agent")
            or question.event.arguments.get("experiencer")
            or question.event.arguments.get("subject")
        )
        relations = list(memory.gerunds)
        if source_ref is not None and not source_ref.is_variable:
            if source_ref.kind != RefKind.ENTITY:
                return None
            relations = [
                item for item in relations
                if item.source_entity_id == source_ref.key
            ]
        relations = [
            item for item in relations
            if item.matrix_predicate == predicate
            and self._gerund_relation_matches_question(item, question, memory)
        ]
        if not relations:
            return None

        if question.kind == QuestionKind.YES_NO:
            # A plain matrix question such as ``Did I see Sarah?`` belongs to
            # ordinary event QA.  This path is only for a selected -ing
            # proposition such as ``Did I see Sarah leaving?``.
            if not any(
                self._gerund_question_selects_complement(
                    question,
                    memory.get_event(item.complement_event_id),
                )
                for item in relations
            ):
                return None

            supporting: List[GerundRelation] = []
            contradicting: List[GerundRelation] = []
            for relation in relations:
                complement = memory.get_event(relation.complement_event_id)
                if complement is None:
                    continue
                matrix_polarity, embedded_polarity = (
                    self._gerund_question_polarities(question, complement)
                )
                exact = relation.licensed == matrix_polarity
                if embedded_polarity is not None:
                    exact = exact and complement.polarity == embedded_polarity
                (supporting if exact else contradicting).append(relation)

            if supporting and contradicting:
                combined = supporting + contradicting
                matrix_events = [
                    event
                    for item in combined
                    if (event := memory.get_event(item.matrix_event_id)) is not None
                ]
                return AnswerContract(
                    status=AnswerStatus.CONFLICT,
                    question=question,
                    proposition=memory.get_event(supporting[0].matrix_event_id),
                    truth=TruthValue.CONFLICT,
                    evidence=self._gerund_evidence(combined, memory),
                    certainty=min(item.certainty for item in combined),
                    source=self._combine_sources(matrix_events),
                    reason=(
                        "stored matrix/embedded polarity pairs conflict with "
                        "the selected -ing proposition"
                    ),
                    response_goal="warn",
                    required_slots={
                        "gerund": "true",
                        "relation_ids": ",".join(
                            item.relation_id for item in combined
                        ),
                    },
                    forbidden_claims=[
                        "promote_gerund_to_unqualified_event",
                    ],
                )

            winner_pool = supporting or contradicting
            if not winner_pool:
                return None
            winner = max(
                winner_pool,
                key=lambda item: (
                    memory.get_event(item.matrix_event_id).turn_index
                    if memory.get_event(item.matrix_event_id) is not None
                    else -1,
                    item.certainty,
                ),
            )
            matrix = memory.get_event(winner.matrix_event_id)
            if matrix is None:
                return None
            truth = bool(supporting)
            return AnswerContract(
                status=AnswerStatus.TRUE if truth else AnswerStatus.FALSE,
                question=question,
                proposition=matrix,
                truth=TruthValue.TRUE if truth else TruthValue.FALSE,
                evidence=self._gerund_evidence([winner], memory),
                certainty=winner.certainty,
                source=matrix.source,
                reason=(
                    "typed gerund/participial proposition match"
                    if truth
                    else "typed opposite gerund/participial proposition match"
                ),
                response_goal="answer",
                required_slots=self._gerund_slots(winner),
                forbidden_claims=[
                    "promote_gerund_to_unqualified_event",
                ],
            )

        licensed = [item for item in relations if item.licensed]
        if not licensed:
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                certainty=0,
                source=SourceKind.ATTRIBUTED,
                reason="only a negated gerund/participial relation is stored",
                response_goal="answer",
                required_slots={
                    "unknown_object": "content",
                    "gerund": "true",
                    "matrix_predicate": predicate,
                },
                forbidden_claims=[
                    "invert_negated_gerund_relation",
                    "promote_gerund_to_unqualified_event",
                ],
            )

        requested = question.requested_role or ""
        tail = self._gerund_question_tail(question)
        if (
            question.kind == QuestionKind.WHAT
            and requested in {"patient", "content"}
            and (
                not tail
                or self._gerund_tail_is_content_placeholder(tail)
            )
        ):
            latest = self._latest_gerund_relations(licensed, memory)
            phrases: Dict[str, Tuple[GerundRelation, SemanticRef]] = {}
            for relation in latest:
                complement = memory.get_event(relation.complement_event_id)
                if complement is None:
                    continue
                phrase = self._gerund_event_phrase(complement, relation, memory)
                phrases.setdefault(
                    phrase,
                    (
                        relation,
                        SemanticRef.literal(phrase, phrase, EntityKind.ABSTRACT),
                    ),
                )
            if not phrases:
                return None
            if len(phrases) > 1:
                selected = [item[0] for item in phrases.values()]
                return AnswerContract(
                    status=AnswerStatus.MULTIPLE_MATCHES,
                    question=question,
                    values=[item[1] for item in phrases.values()],
                    evidence=self._gerund_evidence(selected, memory),
                    certainty=min(item.certainty for item in selected),
                    source=SourceKind.ATTRIBUTED,
                    reason=(
                        "multiple qualified -ing contents are stored for the "
                        "latest turn"
                    ),
                    response_goal="clarify",
                    required_slots={
                        "gerund": "true",
                        "relation_ids": ",".join(
                            item.relation_id for item in selected
                        ),
                    },
                    forbidden_claims=[
                        "promote_gerund_to_unqualified_event",
                    ],
                )
            relation, value = next(iter(phrases.values()))
            complement = memory.get_event(relation.complement_event_id)
            if complement is None:
                return None
            return AnswerContract(
                status=AnswerStatus.ANSWERED,
                question=question,
                proposition=complement,
                values=[value],
                evidence=self._gerund_evidence([relation], memory),
                certainty=relation.certainty,
                source=SourceKind.ATTRIBUTED,
                reason=(
                    "bound selected -ing content through its qualified "
                    "matrix relation"
                ),
                response_goal="answer",
                required_slots=self._gerund_slots(relation),
                forbidden_claims=[
                    "promote_gerund_to_unqualified_event",
                ],
            )

        # Bind WH/open slots through the typed relation rather than through
        # the nonassertive complement in the ordinary fact store.
        bound: List[Tuple[GerundRelation, SemanticRef, str]] = []
        for relation in licensed:
            selected = self._gerund_open_slot_value(
                relation,
                question,
                memory,
            )
            if selected is not None:
                value, actual_role = selected
                bound.append((relation, value, actual_role))
        if not bound:
            return None

        grouped: Dict[Tuple[str, str], List[Tuple[GerundRelation, SemanticRef, str]]] = defaultdict(list)
        for item in bound:
            grouped[(item[1].kind.value, item[1].key)].append(item)
        if len(grouped) > 1:
            representatives = [items[0] for items in grouped.values()]
            selected_relations = [item[0] for item in representatives]
            return AnswerContract(
                status=AnswerStatus.MULTIPLE_MATCHES,
                question=question,
                values=[item[1] for item in representatives],
                evidence=self._gerund_evidence(selected_relations, memory),
                certainty=min(item.certainty for item in selected_relations),
                source=SourceKind.ATTRIBUTED,
                reason="multiple values satisfy the qualified -ing open slot",
                response_goal="clarify",
                required_slots={
                    "gerund": "true",
                    "relation_ids": ",".join(
                        item.relation_id for item in selected_relations
                    ),
                },
                forbidden_claims=[
                    "promote_gerund_to_unqualified_event",
                ],
            )

        candidates = next(iter(grouped.values()))
        relation, value, actual_role = max(
            candidates,
            key=lambda item: (
                memory.get_event(item[0].matrix_event_id).turn_index
                if memory.get_event(item[0].matrix_event_id) is not None
                else -1,
                item[0].certainty,
            ),
        )
        complement = memory.get_event(relation.complement_event_id)
        if complement is None:
            return None
        slots = self._gerund_slots(relation)
        slots.update({
            "requested_role": requested,
            "actual_role": actual_role,
        })
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=complement,
            values=[value],
            evidence=self._gerund_evidence(
                [item[0] for item in candidates],
                memory,
            ),
            certainty=max(item[0].certainty for item in candidates),
            source=SourceKind.ATTRIBUTED,
            reason="bound open slot through a qualified -ing relation",
            response_goal="answer",
            required_slots=slots,
            forbidden_claims=[
                "promote_gerund_to_unqualified_event",
            ],
        )

    @staticmethod
    def _gerund_question_tail(question: QuestionFrame) -> List[str]:
        items = [
            token
            for token in lexicon.tokenize(question.raw_text)
            if token.norm not in lexicon.PUNCTUATION
        ]
        predicate = lexicon.lemma(question.event.predicate)
        predicate_index = next(
            (
                index for index, token in enumerate(items)
                if lexicon.lemma(token.norm) == predicate
            ),
            -1,
        )
        if predicate_index < 0:
            return []
        return [token.norm for token in items[predicate_index + 1 :]]

    @staticmethod
    def _gerund_tail_is_content_placeholder(tail: Sequence[str]) -> bool:
        return bool(tail) and tail[-1] in {
            "do", "doing", "something", "anything",
        }

    def _gerund_question_selects_complement(
        self,
        question: QuestionFrame,
        complement: Optional[EventFrame],
    ) -> bool:
        if complement is None:
            return False
        tail = self._gerund_question_tail(question)
        if self._gerund_tail_is_content_placeholder(tail):
            return True
        return any(
            lexicon.lemma(word) == complement.predicate
            and (word.endswith("ing") or word == complement.predicate)
            for word in tail
        )

    def _gerund_question_polarities(
        self,
        question: QuestionFrame,
        complement: EventFrame,
    ) -> Tuple[bool, Optional[bool]]:
        matrix_polarity = (
            question.matrix_polarity
            if question.matrix_polarity is not None
            else question.event.polarity
        )
        if question.embedded_polarity is not None:
            return matrix_polarity, question.embedded_polarity

        tail = self._gerund_question_tail(question)
        predicate_index = next(
            (
                index for index, word in enumerate(tail)
                if lexicon.lemma(word) == complement.predicate
            ),
            -1,
        )
        if predicate_index >= 0 and any(
            word in lexicon.NEGATORS for word in tail[:predicate_index]
        ):
            return True, False
        return matrix_polarity, None
