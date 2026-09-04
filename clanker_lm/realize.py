"""Template-free deterministic surface realization.

No response sentence or phrase template is stored or formatted.  Each candidate
is assembled from a semantic plan, generic inflection rules, dynamic frame
values, and single-token atoms.  Clanker then evaluates the completed candidate
and selects the mathematically best valid result.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from . import lexicon
from .database import Atom, LanguageStore
from .gates import ContextGate
from .memory import ConversationMemory
from .model import (
    AnswerContract,
    AnswerStatus,
    CandidateResponse,
    Entity,
    EmbeddedInterrogativeRelation,
    EmbeddedInterrogativeType,
    EntityKind,
    EventFrame,
    GateDecision,
    Gender,
    GerundRelation,
    GrammaticalNumber,
    QuestionKind,
    RefKind,
    SemanticRef,
)


Part = Union[str, Atom]


class SurfaceRealizer:
    """Compose contract-preserving replies without whole-sentence templates."""

    def __init__(self, memory: ConversationMemory, store: LanguageStore) -> None:
        self.memory = memory
        self.store = store
        self.store.assert_template_free()

    # ------------------------------------------------------------------
    # Semantic-plan routing
    # ------------------------------------------------------------------

    def realize(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        status = contract.status
        if status == AnswerStatus.ANSWERED:
            candidates = self._answered(contract, gates)
        elif status == AnswerStatus.TRUE:
            candidates = self._polar(contract, gates, positive=True)
        elif status == AnswerStatus.FALSE:
            candidates = self._polar(contract, gates, positive=False)
        elif status == AnswerStatus.UNKNOWN:
            candidates = self._unknown(contract, gates)
        elif status == AnswerStatus.CONFLICT:
            candidates = self._conflict(contract, gates)
        elif status == AnswerStatus.MISSING_REFERENCE:
            candidates = self._missing_reference(contract, gates)
        elif status == AnswerStatus.AMBIGUOUS_REFERENCE:
            candidates = self._ambiguous_reference(contract, gates)
        elif status == AnswerStatus.MULTIPLE_MATCHES:
            candidates = self._multiple_matches(contract, gates)
        elif status == AnswerStatus.LEXICAL_PROBE:
            candidates = self._lexical_probe(contract, gates)
        elif status == AnswerStatus.LEXICAL_LEARNED:
            candidates = self._lexical_learned(contract, gates)
        elif status == AnswerStatus.ACKNOWLEDGED:
            candidates = self._acknowledge(contract, gates)
        else:
            candidates = self._unsupported(contract, gates)

        valid_candidates: List[CandidateResponse] = []
        for candidate in candidates:
            valid, reason = self._validate_surface(candidate.text, contract)
            candidate.semantic_valid = candidate.semantic_valid and valid
            candidate.semantic_reason = reason if not valid else candidate.semantic_reason or reason
            valid_candidates.append(candidate)
        if valid_candidates:
            return valid_candidates
        return self._unsupported(contract, gates)

    def _rule_plan(self, route: str, features: Optional[Mapping[str, Any]] = None) -> List[str]:
        rules = self.store.grammar_rules(route, features=features)
        if not rules:
            return [f"ROUTE:{route}", "RULE:BUILTIN_FAIL_CLOSED"]
        rule = rules[0]
        return [f"ROUTE:{route}", f"RULE:{rule.rule_id}", *rule.children]

    def _atom(self, atom_id: str, fallback: str = "") -> Atom:
        atom = self.store.atom_by_id(atom_id)
        if atom is None:
            # A missing atom is a configuration error, not permission to pull a
            # memorized sentence from code or improvise with a model.
            raise RuntimeError(f"Required language atom is missing: {atom_id}")
        return atom

    def _atoms(
        self,
        category: str,
        *,
        register: str,
        preferred: Sequence[str] = (),
    ) -> List[Atom]:
        by_id = [self.store.atom_by_id(item) for item in preferred]
        selected = [item for item in by_id if item is not None]
        if selected:
            return selected
        return self.store.atom_candidates(category, register=register)

    def _candidate(
        self,
        parts: Sequence[Part],
        *,
        candidate_id: str,
        semantic_plan: Sequence[str],
        priority: int,
    ) -> CandidateResponse:
        text, atom_ids = self._compose(parts)
        return CandidateResponse(
            text=text,
            construction_id=candidate_id,
            semantic_valid=True,
            semantic_reason="composed from a closed semantic plan",
            priority=priority,
            atom_ids=atom_ids,
            semantic_plan=list(semantic_plan),
        )

    @classmethod
    def _compose(cls, parts: Sequence[Part]) -> Tuple[str, List[str]]:
        surfaces: List[str] = []
        atom_ids: List[str] = []
        for part in parts:
            if isinstance(part, Atom):
                surface = part.surface
                atom_ids.append(part.atom_id)
            else:
                surface = str(part)
            surface = re.sub(r"\s+", " ", surface).strip()
            if surface:
                surfaces.append(surface)
        if not surfaces:
            return "", atom_ids
        text = ""
        closing = {".", ",", "?", "!", ":", ";"}
        opening = {"(", "[", "{", "“", "\""}
        for surface in surfaces:
            if not text:
                text = surface
            elif surface in closing:
                text = text.rstrip() + surface
            elif text[-1:] in opening:
                text += surface
            else:
                text += " " + surface
        text = re.sub(r"\s+", " ", text).strip()
        # Capitalization is a grammatical operation applied at every sentence
        # boundary; it is not stored in a response template.
        text = re.sub(
            r"(^|[.!?]\s+)([a-z])",
            lambda match: match.group(1) + match.group(2).upper(),
            text,
        )
        return cls._capitalize(text), atom_ids

    # ------------------------------------------------------------------
    # Contract-specific composition
    # ------------------------------------------------------------------

    def _answered(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        if contract.response_goal == "social":
            return self._social(contract, gates)
        if contract.proposition is None:
            return self._unsupported(contract, gates)
        if contract.required_slots.get("gerund") == "true":
            relation = self._gerund_relation_from_contract(contract)
            if relation is None:
                return self._unsupported(contract, gates)
            clause = self.render_gerund_relation(
                relation,
                capitalize=True,
            )
            return [
                self._candidate(
                    [clause, self._atom("punct.period")],
                    candidate_id="compose.answer.gerund_relation",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        f"GERUND_RELATION:{relation.relation_type.value}",
                        f"CONTENT_STATUS:{relation.content_status.value}",
                        f"MATRIX_PREDICATE:{relation.matrix_predicate}",
                        f"CONTROLLER:{relation.controller_entity_id}",
                        "TRUTH:SELECTED_CONTENT_QUALIFIED_BY_MATRIX",
                    ],
                    priority=119,
                )
            ]
        if contract.required_slots.get("embedded_interrogative") == "true":
            relations = self._embedded_interrogative_relations_from_contract(contract)
            if not relations:
                return self._unsupported(contract, gates)
            clauses = [
                self.render_embedded_interrogative_relation(
                    relation,
                    capitalize=(index == 0),
                )
                for index, relation in enumerate(relations[:4])
            ]
            clause = self.join_phrases(
                clauses,
                conjunction=self._atom("link.and").surface,
            )
            return [
                self._candidate(
                    [clause, self._atom("punct.period")],
                    candidate_id="compose.answer.embedded_interrogative",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        *[
                            f"EMBEDDED_INTERROGATIVE:{item.relation_type.value}"
                            for item in relations[:4]
                        ],
                        *[
                            f"MATRIX_PREDICATE:{item.matrix_predicate}"
                            for item in relations[:4]
                        ],
                        "TRUTH:QUESTION_ATTRIBUTION_ONLY",
                    ],
                    priority=120,
                )
            ]
        if contract.required_slots.get("infinitival") == "true":
            relation = self._infinitival_relation_from_contract(contract)
            if relation is None:
                return self._unsupported(contract, gates)
            clause = self.render_infinitival_relation(
                relation,
                capitalize=True,
            )
            return [
                self._candidate(
                    [clause, self._atom("punct.period")],
                    candidate_id="compose.answer.infinitival_relation",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        f"INFINITIVAL_RELATION:{relation.relation_type.value}",
                        f"CONTENT_STATUS:{relation.content_status.value}",
                        f"MATRIX_PREDICATE:{relation.matrix_predicate}",
                        f"CONTROLLER:{relation.controller_entity_id}",
                    ],
                    priority=118,
                )
            ]
        if contract.required_slots.get("attributed") == "true":
            source_id = contract.required_slots.get("source_entity_id", "")
            predicate = contract.required_slots.get("matrix_predicate", "say")
            tense = contract.required_slots.get("matrix_tense", "present")
            source = self.memory.describe_entity(source_id)
            content = self.render_event(contract.proposition, capitalize=False)
            verb = self._finite_content_verb(predicate, tense)
            return [
                self._candidate(
                    [source, verb, content, self._atom("punct.period")],
                    candidate_id="compose.answer.attributed_content",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        f"ATTRIBUTION_SOURCE:{source_id}",
                        f"ATTRIBUTION_PREDICATE:{predicate}",
                        f"CONTENT_FRAME:{contract.proposition.predicate}",
                    ],
                    priority=116,
                )
            ]
        if (
            contract.question is not None
            and contract.question.kind == QuestionKind.WHOSE
            and contract.values
        ):
            owner = self.render_ref(
                contract.values[0],
                case="subject",
                definite=True,
                capitalize=True,
            )
            return [
                self._candidate(
                    [owner, self._atom("punct.period")],
                    candidate_id="compose.answer.possessor",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        "ROLE:possessor",
                        f"VALUE:{contract.values[0].key}",
                    ],
                    priority=108,
                )
            ]
        clause = self.render_event(contract.proposition, capitalize=True)
        return [
            self._candidate(
                [clause, self._atom("punct.period")],
                candidate_id="compose.answer.proposition",
                semantic_plan=[*self._rule_plan("reply:answer"), f"FRAME:{contract.proposition.predicate}"],
                priority=100,
            )
        ]

    def _polar(self, contract: AnswerContract, gates: GateDecision, *, positive: bool) -> List[CandidateResponse]:
        if contract.proposition is None:
            return self._unknown(contract, gates)
        category = "affirmation" if positive else "denial"
        preferred = (
            ("polarity.yeah", "polarity.yes")
            if positive and gates.register == "casual"
            else ("polarity.yes", "polarity.yeah")
            if positive
            else ("polarity.nope", "polarity.no")
            if gates.register == "casual"
            else ("polarity.no", "polarity.nope")
        )
        particles = self._atoms(category, register=gates.register, preferred=preferred)
        semantic_plan_details: List[str] = []
        if contract.required_slots.get("embedded_memory_probe") == "true":
            known = contract.required_slots.get("embedded_memory_known") == "true"
            if known:
                clause = self.render_event(contract.proposition, capitalize=True)
                semantic_plan_details.extend([
                    "OUTER_EPISTEMIC_QUERY:true",
                    "INNER_QUESTION_ANSWERABLE:true",
                ])
            else:
                inner = contract.question.embedded_question if contract.question else None
                inner_phrase = (
                    self.render_embedded_question_frame(inner, capitalize=False)
                    if inner is not None
                    else self.render_event(contract.proposition, capitalize=False)
                )
                clause, _ = self._compose([
                    self._atom("pronoun.i"),
                    self._atom("aux.do"),
                    self._atom("neg.not"),
                    self._atom("cognition.know"),
                    inner_phrase,
                ])
                semantic_plan_details.extend([
                    "OUTER_EPISTEMIC_QUERY:true",
                    "INNER_QUESTION_ANSWERABLE:false",
                ])
        elif contract.required_slots.get("embedded_interrogative") == "true":
            relations = self._embedded_interrogative_relations_from_contract(contract)
            if not relations:
                return self._unknown(contract, gates)
            clauses = [
                self.render_embedded_interrogative_relation(
                    relation,
                    capitalize=(index == 0),
                )
                for index, relation in enumerate(relations[:4])
            ]
            clause = self.join_phrases(
                clauses,
                conjunction=self._atom("link.but").surface,
            )
            semantic_plan_details.extend([
                *[
                    f"EMBEDDED_INTERROGATIVE:{item.relation_type.value}"
                    for item in relations[:4]
                ],
                "TRUTH:QUESTION_ATTRIBUTION_ONLY",
            ])
        else:
            gerund_relation = (
                self._gerund_relation_from_contract(contract)
                if contract.required_slots.get("gerund") == "true"
                else None
            )
            if contract.required_slots.get("gerund") == "true":
                if gerund_relation is None:
                    return self._unknown(contract, gates)
                clause = self.render_gerund_relation(
                    gerund_relation,
                    capitalize=True,
                )
                semantic_plan_details.append(
                    f"GERUND_RELATION:{gerund_relation.relation_type.value}"
                )
                semantic_plan_details.append(
                    f"CONTENT_STATUS:{gerund_relation.content_status.value}"
                )
            else:
                relation = (
                    self._infinitival_relation_from_contract(contract)
                    if contract.required_slots.get("infinitival") == "true"
                    else None
                )
                clause = (
                    self.render_infinitival_relation(relation, capitalize=True)
                    if relation is not None
                    else self.render_event(contract.proposition, capitalize=True)
                )
                if relation is not None:
                    semantic_plan_details.append(
                        f"INFINITIVAL_RELATION:{relation.relation_type.value}"
                    )
        candidates: List[CandidateResponse] = []
        for index, particle in enumerate(particles[:2]):
            candidates.append(
                self._candidate(
                    [particle, self._atom("punct.period"), clause, self._atom("punct.period")],
                    candidate_id=f"compose.polar.{particle.atom_id}",
                    semantic_plan=[
                        *self._rule_plan(f"reply:{'true' if positive else 'false'}", {"polarity": positive}),
                        f"FRAME:{contract.proposition.predicate}",
                        *semantic_plan_details,
                    ],
                    priority=105 - index,
                )
            )
        return candidates

    def _unknown(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        question = contract.question
        period = self._atom("punct.period")
        i_atom = self._atom("pronoun.i")
        dont = self._atom("neg.dont")
        know = self._atom("cognition.know")
        plan = self._rule_plan("reply:unknown")

        if (
            contract.required_slots.get("direct_embedded_answer_request") == "true"
            and question is not None
        ):
            question_phrase = self.render_embedded_question_frame(
                question,
                capitalize=False,
            )
            return [
                self._candidate(
                    [i_atom, dont, know, question_phrase, period],
                    candidate_id="compose.unknown.direct_embedded_request",
                    semantic_plan=[
                        *plan,
                        f"INNER_QUESTION:{question.kind.value}",
                        "TRUTH:UNKNOWN",
                    ],
                    priority=122,
                )
            ]

        if (
            contract.required_slots.get("embedded_interrogative_query") == "true"
            and question is not None
        ):
            query_phrase = self.render_embedded_interrogative_query(
                question,
                capitalize=False,
            )
            return [
                self._candidate(
                    [
                        i_atom,
                        dont,
                        know,
                        self._atom("question.whether"),
                        query_phrase,
                        period,
                    ],
                    candidate_id="compose.unknown.embedded_interrogative",
                    semantic_plan=[
                        *plan,
                        f"MATRIX_PREDICATE:{question.event.predicate}",
                        "TRUTH:QUESTION_ATTRIBUTION_UNKNOWN",
                    ],
                    priority=120,
                )
            ]

        if (
            contract.required_slots.get("infinitival_evidence") == "true"
            and question is not None
        ):
            relation_ids = [
                item
                for item in contract.required_slots.get(
                    "relation_ids",
                    "",
                ).split(",")
                if item
            ]
            relations = [
                relation
                for relation_id in relation_ids[:3]
                for relation in [
                    next(
                        (
                            item
                            for item in self.memory.infinitivals
                            if item.relation_id == relation_id
                        ),
                        None,
                    )
                ]
                if relation is not None
            ]
            if relations:
                context_parts: List[Part] = []
                for index, relation in enumerate(relations):
                    if index:
                        context_parts.append(self._atom("link.but"))
                    context_parts.append(
                        self.render_infinitival_relation(
                            relation,
                            capitalize=(index == 0),
                        )
                    )
                query_clause = self.render_event(
                    question.event,
                    capitalize=False,
                )
                context_parts.extend(
                    [
                        period,
                        i_atom,
                        dont,
                        know,
                        self._atom("question.whether"),
                        query_clause,
                        period,
                    ]
                )
                return [
                    self._candidate(
                        context_parts,
                        candidate_id="compose.unknown.infinitival_boundary",
                        semantic_plan=[
                            *plan,
                            "EVIDENCE:NONENTAILED_INFINITIVE",
                            f"QUERY_FRAME:{question.event.predicate}",
                        ],
                        priority=120,
                    )
                ]

        if (
            contract.required_slots.get("gerund_evidence") == "true"
            and question is not None
        ):
            relation_ids = [
                item
                for item in contract.required_slots.get(
                    "relation_ids",
                    "",
                ).split(",")
                if item
            ]
            by_id = {
                relation.relation_id: relation
                for relation in self.memory.gerunds
            }
            relations = [
                by_id[relation_id]
                for relation_id in relation_ids[:3]
                if relation_id in by_id
            ]
            if relations:
                context_parts = []
                for index, relation in enumerate(relations):
                    if index:
                        context_parts.append(self._atom("link.but"))
                    context_parts.append(
                        self.render_gerund_relation(
                            relation,
                            capitalize=(index == 0),
                        )
                    )
                query_clause = self.render_event(
                    question.event,
                    capitalize=False,
                )
                context_parts.extend(
                    [
                        period,
                        i_atom,
                        dont,
                        know,
                        self._atom("question.whether"),
                        query_clause,
                        period,
                    ]
                )
                return [
                    self._candidate(
                        context_parts,
                        candidate_id="compose.unknown.gerund_boundary",
                        semantic_plan=[
                            *plan,
                            *[
                                f"GERUND_RELATION:{item.relation_type.value}"
                                for item in relations
                            ],
                            *[
                                f"CONTENT_STATUS:{item.content_status.value}"
                                for item in relations
                            ],
                            "EVIDENCE:QUALIFIED_GERUND_CONTENT",
                            "TRUTH:EMBEDDED_EVENT_UNKNOWN",
                            f"QUERY_FRAME:{question.event.predicate}",
                        ],
                        priority=121,
                    )
                ]

        if contract.proposition is not None and question is not None and question.requested_role:
            omitted = self._reason_roles(question)
            known_clause = self.render_event(contract.proposition, omit_roles=omitted, capitalize=False)
            role_surface, role_atom = self._role_surface(question.requested_role)
            parts: List[Part] = [
                i_atom,
                know,
                known_clause,
                self._atom("punct.comma"),
                self._atom("link.but"),
                self._atom("pronoun.you"),
                self._atom("neg.havent"),
                lexicon.participle_form(self._atom("communication.tell").lemma),
                self._atom("pronoun.me"),
                self._atom("det.the"),
                role_atom or role_surface,
                period,
            ]
            return [
                self._candidate(
                    parts,
                    candidate_id="compose.unknown.known_base",
                    semantic_plan=[*plan, f"KNOWN_FRAME:{contract.proposition.predicate}", f"MISSING_ROLE:{question.requested_role}"],
                    priority=112,
                )
            ]

        if question is not None and question.kind == QuestionKind.YES_NO:
            query_clause = self.render_event(question.event, capitalize=False)
            if contract.required_slots.get("attributed") == "true":
                attributed_parts: List[Part] = []
                seen: set[Tuple[str, str]] = set()
                for evidence in contract.evidence:
                    relations = self.memory.content_relations_for_event(
                        evidence.event.event_id
                    )
                    relation = next(
                        (
                            item
                            for item in relations
                            if item.content_event_id == evidence.event.event_id
                        ),
                        None,
                    )
                    if relation is None:
                        continue
                    key = (relation.source_entity_id, evidence.event.event_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    source = self.memory.describe_entity(relation.source_entity_id)
                    verb = self._finite_content_verb(
                        relation.matrix_predicate,
                        self.memory.get_event(relation.matrix_event_id).tense
                        if self.memory.get_event(relation.matrix_event_id) is not None
                        else "present",
                    )
                    clause = self.render_event(evidence.event, capitalize=False)
                    if attributed_parts:
                        attributed_parts.append(self._atom("link.but"))
                    attributed_parts.extend([source, verb, clause])
                    if len(seen) >= 3:
                        break
                if attributed_parts:
                    attributed_parts.extend(
                        [
                            period,
                            i_atom,
                            dont,
                            know,
                            self._atom("question.whether"),
                            query_clause,
                            period,
                        ]
                    )
                    return [
                        self._candidate(
                            attributed_parts,
                            candidate_id="compose.unknown.attributed_polar",
                            semantic_plan=[
                                *plan,
                                "EVIDENCE:ATTRIBUTED_ONLY",
                                f"QUERY_FRAME:{question.event.predicate}",
                            ],
                            priority=118,
                        )
                    ]
            return [
                self._candidate(
                    [i_atom, dont, know, self._atom("question.whether"), query_clause, period],
                    candidate_id="compose.unknown.polar",
                    semantic_plan=[*plan, f"QUERY_FRAME:{question.event.predicate}"],
                    priority=108,
                )
            ]

        unknown_object = contract.required_slots.get("unknown_object")
        if not unknown_object and question is not None:
            unknown_object = question.requested_role or "answer"
        unknown_object = unknown_object or "answer"
        role_surface, role_atom = self._role_surface(unknown_object)
        return [
            self._candidate(
                [i_atom, dont, know, self._atom("det.the"), role_atom or role_surface, period],
                candidate_id="compose.unknown.slot",
                semantic_plan=[*plan, f"UNKNOWN_SLOT:{unknown_object}"],
                priority=100,
            )
        ]

    def _conflict(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        if (
            contract.required_slots.get("gerund") == "true"
            or contract.required_slots.get("gerund_evidence") == "true"
            or any(
                evidence.event.discourse_role == "gerund"
                for evidence in contract.evidence
            )
        ):
            relation_ids = [
                item
                for item in contract.required_slots.get(
                    "relation_ids",
                    "",
                ).split(",")
                if item
            ]
            by_id = {
                relation.relation_id: relation
                for relation in self.memory.gerunds
            }
            relations: List[GerundRelation] = []
            seen_ids: set[str] = set()
            for relation_id in relation_ids[:4]:
                relation = by_id.get(relation_id)
                if relation is None or relation_id in seen_ids:
                    continue
                relations.append(relation)
                seen_ids.add(relation_id)

            clauses = [
                clause
                for relation in relations
                for clause in [
                    self.render_gerund_relation(
                        relation,
                        capitalize=False,
                    )
                ]
                if clause
            ]
            summary = self.join_phrases(
                clauses,
                conjunction=self._atom("link.but").surface,
            )
            parts: List[Part] = [
                self._atom("pronoun.i"),
                self._atom("aux.have"),
                self._atom("meta.conflicting"),
                self._atom("meta.information"),
            ]
            if summary:
                parts.extend([self._atom("punct.colon"), summary])
            parts.append(self._atom("punct.period"))
            return [
                self._candidate(
                    parts,
                    candidate_id="compose.conflict.gerund_relations",
                    semantic_plan=[
                        *self._rule_plan("reply:unknown"),
                        *[
                            f"GERUND_RELATION:{item.relation_type.value}"
                            for item in relations
                        ],
                        *[
                            f"CONTENT_STATUS:{item.content_status.value}"
                            for item in relations
                        ],
                        "EVIDENCE:CONTRADICTORY_QUALIFIED_GERUND_RELATIONS",
                        "TRUTH:EMBEDDED_EVENTS_REMAIN_MATRIX_QUALIFIED",
                        (
                            "RELATION_LOOKUP:BOUND"
                            if relations
                            else "RELATION_LOOKUP:FAILED_CLOSED"
                        ),
                    ],
                    priority=124,
                )
            ]

        if contract.required_slots.get("embedded_interrogative") == "true":
            relations = self._embedded_interrogative_relations_from_contract(contract)
            clauses = [
                self.render_embedded_interrogative_relation(
                    relation,
                    capitalize=False,
                )
                for relation in relations[:4]
            ]
            summary = self.join_phrases(
                clauses,
                conjunction=self._atom("link.but").surface,
            )
            return [
                self._candidate(
                    [
                        self._atom("pronoun.i"),
                        self._atom("aux.have"),
                        self._atom("meta.conflicting"),
                        self._atom("meta.information"),
                        self._atom("punct.colon"),
                        summary,
                        self._atom("punct.period"),
                    ],
                    candidate_id="compose.conflict.embedded_interrogative",
                    semantic_plan=[
                        *self._rule_plan("reply:unknown"),
                        "EVIDENCE:CONTRADICTORY_QUESTION_ATTRIBUTION",
                    ],
                    priority=122,
                )
            ]

        clauses: List[str] = []
        seen: set[str] = set()
        for evidence in contract.evidence:
            clause = self.render_event(evidence.event, capitalize=False)
            if clause and clause not in seen:
                clauses.append(clause)
                seen.add(clause)
            if len(clauses) >= 3:
                break
        summary = self.join_phrases(clauses, conjunction=self._atom("link.but").surface)
        parts: List[Part] = [
            self._atom("pronoun.i"),
            self._atom("aux.have"),
            self._atom("meta.conflicting"),
            self._atom("meta.information"),
        ]
        if summary:
            parts.extend([self._atom("punct.colon"), summary])
        parts.append(self._atom("punct.period"))
        return [
            self._candidate(
                parts,
                candidate_id="compose.conflict.evidence",
                semantic_plan=[*self._rule_plan("reply:unknown"), "EVIDENCE:CONTRADICTORY"],
                priority=115,
            )
        ]

    def _missing_reference(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        reference = contract.required_slots.get("reference", contract.reason or "it")
        operator_id = "question.who" if self._reference_is_person(reference) else "question.what"
        parts: List[Part] = [
            self._atom(operator_id),
            self._atom("aux.do"),
            self._atom("pronoun.you"),
            self._atom("communication.mean"),
            self._atom("prep.by"),
            reference,
            self._atom("punct.question"),
        ]
        return [
            self._candidate(
                parts,
                candidate_id="compose.probe.missing_reference",
                semantic_plan=[*self._rule_plan("reply:probe"), f"UNRESOLVED:{reference}"],
                priority=120,
            )
        ]

    def _ambiguous_reference(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        ids = [item for item in contract.required_slots.get("candidate_ids", "").split(",") if item]
        labels = [self.memory.describe_entity(entity_id) for entity_id in ids]
        alternatives = self.join_phrases(labels, conjunction=self._atom("link.or").surface)
        return [
            self._candidate(
                [
                    self._atom("aux.do"),
                    self._atom("pronoun.you"),
                    self._atom("communication.mean"),
                    alternatives or contract.required_slots.get("reference", "that"),
                    self._atom("punct.question"),
                ],
                candidate_id="compose.probe.ambiguous_reference",
                semantic_plan=[*self._rule_plan("reply:probe"), *[f"CANDIDATE:{item}" for item in ids]],
                priority=120,
            )
        ]

    def _multiple_matches(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        values = [self.render_ref(value, definite=False) for value in contract.values]
        listing = self.join_phrases(values, conjunction=self._atom("link.and").surface)
        first = self._candidate(
            [
                self._atom("pronoun.i"),
                self._atom("verb.find"),
                self._atom("meta.more"),
                self._atom("prep.than"),
                self._atom("meta.one"),
                self._atom("meta.answer"),
                self._atom("punct.colon"),
                listing,
                self._atom("punct.period"),
                self._atom("question.which"),
                self._atom("meta.one"),
                self._atom("aux.do"),
                self._atom("pronoun.you"),
                self._atom("communication.mean"),
                self._atom("punct.question"),
            ],
            candidate_id="compose.probe.multiple_values",
            semantic_plan=[*self._rule_plan("reply:probe"), *[f"VALUE:{value.key}" for value in contract.values]],
            priority=118,
        )
        return [first]

    def _lexical_probe(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        term = contract.required_slots.get("term", contract.reason or "word")
        axis = contract.required_slots.get("probe_axis", "definition")
        plan = [*self._rule_plan("reply:probe"), f"LEXEME:{term}", f"PROBE_AXIS:{axis}"]
        if axis in {"definition", "meaning"}:
            # A user asking for an unknown definition gets an explicit
            # uncertainty statement followed by the smallest useful probe.
            # Every element is still composed from atoms, inflection, and the
            # dynamic term; no completed response sentence is stored.
            if contract.required_slots.get("semantic_position") == "definition_query":
                means = lexicon.present_form(
                    self._atom("communication.mean").lemma,
                    third_person_singular=True,
                )
                parts: List[Part] = [
                    self._atom("pronoun.i"),
                    self._atom("aux.do"),
                    self._atom("neg.not"),
                    self._atom("cognition.know"),
                    self._atom("question.what"),
                    term,
                    means,
                    self._atom("punct.period"),
                    self._atom("learn.example"),
                    self._atom("punct.question"),
                ]
                return [
                    self._candidate(
                        parts,
                        candidate_id="compose.lexical.unknown_definition_probe",
                        semantic_plan=[*plan, "TRUTH:UNKNOWN", "ACT:REQUEST_EXAMPLE"],
                        priority=130,
                    )
                ]
            parts = [
                self._atom("question.what"),
                self._atom("aux.does"),
                term,
                self._atom("communication.mean"),
                self._atom("punct.question"),
            ]
            return [self._candidate(parts, candidate_id="compose.lexical.definition_probe", semantic_plan=plan, priority=125)]
        atom_id = {
            "polarity": "learn.polarity",
            "semantic_class": "learn.meaning",
            "intensity": "learn.intense",
            "example": "learn.example",
        }.get(axis, "learn.meaning")
        atom = self.store.atom_by_id(atom_id)
        if atom is None and axis == "polarity":
            atom = self._atom("learn.positive")
        assert atom is not None
        return [
            self._candidate(
                [atom, self._atom("punct.question")],
                candidate_id=f"compose.lexical.one_word.{axis}",
                semantic_plan=plan,
                priority=125,
            )
        ]

    def _lexical_learned(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        if contract.proposition is not None:
            return self._answered(contract, gates)
        return self._acknowledge(contract, gates)

    def _acknowledge(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        """Realize only candidates inside the planner-selected act class."""

        if contract.response_goal == "social" or gates.response_act == "social":
            return self._social(contract, gates)

        period = self._atom("punct.period")
        route_plan = self._rule_plan("reply:acknowledge")
        response_act = gates.response_act or "neutral_acknowledge"

        if response_act == "safety_probe" or gates.severity == "critical":
            return [
                self._candidate(
                    [*self._safe_question_parts()],
                    candidate_id="compose.acknowledge.safety_probe",
                    semantic_plan=[*route_plan, "RESPONSE_ACT:safety_probe", "ACT:SAFETY_CHECK"],
                    priority=140,
                )
            ]

        if response_act == "positive_acknowledge":
            positive = self.render_event(
                EventFrame(
                    "sound",
                    {
                        "subject": SemanticRef.literal(
                            "that", self._atom("demonstrative.that").surface
                        ),
                        "state": SemanticRef.literal(
                            "good", self._atom("evaluation.good").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            return [
                self._candidate(
                    [positive, period],
                    candidate_id="compose.acknowledge.positive",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:positive_acknowledge",
                        "ACT:POSITIVE_RECOGNITION",
                    ],
                    priority=128,
                )
            ]

        if response_act == "empathic_acknowledge":
            sorry = self.render_event(
                EventFrame(
                    "be",
                    {
                        "subject": SemanticRef.entity(
                            "assistant", "I", EntityKind.PERSON
                        ),
                        "value": SemanticRef.literal(
                            "sorry", self._atom("empathy.sorry").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            rough = self.render_event(
                EventFrame(
                    "sound",
                    {
                        "subject": SemanticRef.literal(
                            "that", self._atom("demonstrative.that").surface
                        ),
                        "state": SemanticRef.literal(
                            "rough", self._atom("evaluation.rough").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            return [
                self._candidate(
                    [sorry, period],
                    candidate_id="compose.acknowledge.loss.sorry",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:empathic_acknowledge",
                        "ACT:EMPATHY",
                    ],
                    priority=132,
                ),
                self._candidate(
                    [rough, period],
                    candidate_id="compose.acknowledge.loss.rough",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:empathic_acknowledge",
                        "ACT:VALIDATE",
                    ],
                    priority=124,
                ),
            ]

        if response_act == "serious_followup":
            sorry = self.render_event(
                EventFrame(
                    "be",
                    {
                        "subject": SemanticRef.entity(
                            "assistant", "I", EntityKind.PERSON
                        ),
                        "value": SemanticRef.literal(
                            "sorry", self._atom("empathy.sorry").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            serious = self.render_event(
                EventFrame(
                    "sound",
                    {
                        "subject": SemanticRef.literal(
                            "that", self._atom("demonstrative.that").surface
                        ),
                        "state": SemanticRef.literal(
                            "serious", self._atom("evaluation.serious").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            question = self._what_happened_question()
            candidates = [
                self._candidate(
                    [serious, period, *question],
                    candidate_id="compose.acknowledge.serious",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:serious_followup",
                        "ACT:SEVERITY_RECOGNITION",
                        "ACT:OPEN_FOLLOWUP",
                    ],
                    priority=124,
                )
            ]
            if gates.masking or gates.register == "casual":
                candidates.insert(
                    0,
                    self._candidate(
                        [sorry, period, serious, period, *question],
                        candidate_id="compose.acknowledge.masked_serious",
                        semantic_plan=[
                            *route_plan,
                            "RESPONSE_ACT:serious_followup",
                            "ACT:EMPATHY",
                            "ACT:SEVERITY_RECOGNITION",
                            "ACT:OPEN_FOLLOWUP",
                        ],
                        priority=132,
                    ),
                )
            return candidates

        if response_act == "empathic_followup":
            rough = self.render_event(
                EventFrame(
                    "sound",
                    {
                        "subject": SemanticRef.literal(
                            "that", self._atom("demonstrative.that").surface
                        ),
                        "state": SemanticRef.literal(
                            "rough", self._atom("evaluation.rough").surface
                        ),
                    },
                ),
                capitalize=True,
            )
            return [
                self._candidate(
                    [rough, period, *self._what_happened_question()],
                    candidate_id="compose.acknowledge.rough",
                    semantic_plan=[
                        *route_plan,
                        "RESPONSE_ACT:empathic_followup",
                        "ACT:VALIDATE",
                        "ACT:OPEN_FOLLOWUP",
                    ],
                    priority=120,
                )
            ]

        # Neutral acknowledgment uses a deterministic discourse-cycle instead
        # of randomness.  Only one candidate is emitted, so affective ranking
        # cannot escape the selected response-act class or repeat one form on
        # every compatible turn.
        variant = max(0, self.memory.turn_index - 1) % 3
        if variant == 0:
            verb = self._atom("ack.understand")
            clause = self.render_event(
                EventFrame(
                    verb.lemma,
                    {"agent": SemanticRef.entity("assistant", "I", EntityKind.PERSON)},
                ),
                capitalize=True,
            )
            parts: List[Part] = [clause, period]
            candidate_id = "compose.acknowledge.neutral.understand"
            act_plan = "ACT:UNDERSTAND"
        elif variant == 1:
            verb = self._atom("ack.hear")
            clause = self.render_event(
                EventFrame(
                    verb.lemma,
                    {
                        "agent": SemanticRef.entity(
                            "assistant", "I", EntityKind.PERSON
                        ),
                        "patient": SemanticRef.entity(
                            "user", "you", EntityKind.PERSON
                        ),
                    },
                ),
                capitalize=True,
            )
            parts = [clause, period]
            candidate_id = "compose.acknowledge.neutral.hear"
            act_plan = "ACT:HEAR"
        else:
            parts = [
                self._atom("pronoun.i"),
                self._atom("ack.get"),
                self._atom("demonstrative.that"),
                period,
            ]
            candidate_id = "compose.acknowledge.neutral.get"
            act_plan = "ACT:GET"

        return [
            self._candidate(
                parts,
                candidate_id=candidate_id,
                semantic_plan=[
                    *route_plan,
                    "RESPONSE_ACT:neutral_acknowledge",
                    f"DISCOURSE_VARIANT:{variant}",
                    act_plan,
                ],
                priority=110,
            )
        ]

    def _social(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        convention = contract.question.social_convention if contract.question else contract.reason
        period = self._atom("punct.period")
        if convention == "wellbeing_check":
            return [
                self._candidate(
                    [
                        self._atom("pronoun.i"),
                        self._atom("aux.am"),
                        self._atom("state.ready"),
                        self._atom("prep.to"),
                        self._atom("process.work"),
                        period,
                    ],
                    candidate_id="compose.social.wellbeing",
                    semantic_plan=[*self._rule_plan("reply:social"), "STATE:READY", "PURPOSE:WORK"],
                    priority=110,
                )
            ]
        if convention == "activity_check":
            return [
                self._candidate(
                    [self._atom("pronoun.i"), self._atom("aux.am"), self._atom("state.working"), period],
                    candidate_id="compose.social.activity",
                    semantic_plan=[*self._rule_plan("reply:social"), "ACTIVITY:WORK"],
                    priority=110,
                )
            ]
        preferred = "social.hey" if gates.register == "casual" else "social.hello"
        return [
            self._candidate(
                [self._atom(preferred), period],
                candidate_id=f"compose.social.{preferred}",
                semantic_plan=[*self._rule_plan("reply:social"), "ACT:GREET"],
                priority=108,
            )
        ]

    def _unsupported(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        return [
            self._candidate(
                [
                    self._atom("pronoun.i"),
                    self._atom("neg.cannot"),
                    self._atom("verb.determine"),
                    self._atom("det.the"),
                    self._atom("meta.answer"),
                    self._atom("punct.period"),
                ],
                candidate_id="compose.unsupported",
                semantic_plan=["ROUTE:FAIL_CLOSED", "EPISTEMIC:UNSUPPORTED"],
                priority=1,
            )
        ]

    def _what_happened_question(self) -> List[Part]:
        happen = self._atom("event.happen")
        return [self._atom("question.what"), lexicon.past_form(happen.lemma), self._atom("punct.question")]

    def _safe_question_parts(self) -> List[Part]:
        return [self._atom("aux.are"), self._atom("pronoun.you"), self._atom("evaluation.safe"), self._atom("punct.question")]

    def _role_surface(self, role: str) -> Tuple[str, Optional[Atom]]:
        normalized = role.lower().replace("_", " ").strip()
        candidates = self.store.atom_candidates(
            "semantic_role_noun",
            register="neutral",
            features={"role": normalized.replace(" ", "_")},
        )
        if candidates:
            return candidates[0].surface, candidates[0]
        if normalized in {"answer", "timezone", "calculation", "meaning", "example", "information"}:
            id_map = {
                "answer": "meta.answer",
                "timezone": "meta.timezone",
                "calculation": "meta.calculation",
                "meaning": "meta.meaning",
                "example": "meta.example",
                "information": "meta.information",
            }
            atom = self.store.atom_by_id(id_map[normalized])
            if atom:
                return atom.surface, atom
        return normalized or "answer", None

    @staticmethod
    def _reference_is_person(reference: str) -> bool:
        lowered = reference.lower().strip(" .?!\"'“”")
        return lowered in {"he", "she", "him", "her", "they", "them", "who"}

    @staticmethod
    def _reason_roles(question: Any) -> Tuple[str, ...]:
        if not question:
            return ("cause", "motive", "purpose", "justification", "evidence", "method", "manner", "process", "mechanism")
        return tuple({
            question.requested_role or "",
            "cause", "motive", "purpose", "justification", "evidence",
            "method", "manner", "process", "mechanism",
        } - {""})

    # ------------------------------------------------------------------
    # Event grammar
    # ------------------------------------------------------------------
    @staticmethod
    def _finite_content_verb(predicate: str, tense: str) -> str:
        if tense == "past":
            return lexicon.past_form(predicate)
        if tense == "future":
            return "will " + predicate
        return lexicon.present_form(predicate, third_person_singular=True)

    def _embedded_interrogative_relations_from_contract(
        self,
        contract: AnswerContract,
    ) -> List[EmbeddedInterrogativeRelation]:
        relation_ids = [
            item
            for item in contract.required_slots.get("relation_ids", "").split(",")
            if item
        ]
        if relation_ids:
            by_id = {
                item.relation_id: item
                for item in self.memory.embedded_interrogatives
            }
            return [
                by_id[relation_id]
                for relation_id in relation_ids
                if relation_id in by_id
            ]

        matrix_ids = {
            item
            for item in contract.required_slots.get("matrix_event_ids", "").split(",")
            if item
        }
        question_ids = {
            item
            for item in contract.required_slots.get("question_event_ids", "").split(",")
            if item
        }
        return [
            item
            for item in self.memory.embedded_interrogatives
            if (not matrix_ids or item.matrix_event_id in matrix_ids)
            and (not question_ids or item.question_event_id in question_ids)
        ]

    def render_embedded_interrogative_relation(
        self,
        relation: EmbeddedInterrogativeRelation,
        *,
        capitalize: bool = False,
    ) -> str:
        matrix = self.memory.get_event(relation.matrix_event_id)
        question_event = self.memory.get_event(relation.question_event_id)
        if matrix is None or question_event is None:
            return ""
        matrix_clause = self.render_event(matrix, capitalize=False)
        question_clause = self.render_embedded_question(
            relation,
            question_event,
            capitalize=False,
        )
        return self._finish_clause(
            " ".join(item for item in (matrix_clause, question_clause) if item),
            capitalize,
        )

    def render_embedded_question(
        self,
        relation: EmbeddedInterrogativeRelation,
        event: EventFrame,
        *,
        capitalize: bool = False,
    ) -> str:
        frame = relation.to_question_frame(event)
        return self.render_embedded_question_frame(
            frame,
            marker=relation.marker,
            capitalize=capitalize,
        )

    def render_embedded_question_frame(
        self,
        question: Any,
        *,
        marker: str = "",
        capitalize: bool = False,
    ) -> str:
        event = question.event
        operator = marker or getattr(question, "embedded_marker", "")
        if not operator:
            operator = {
                QuestionKind.WHO: "who",
                QuestionKind.WHAT: "what",
                QuestionKind.WHEN: "when",
                QuestionKind.WHERE: "where",
                QuestionKind.WHY: "why",
                QuestionKind.HOW: "how",
                QuestionKind.WHICH: "which",
                QuestionKind.WHOSE: "whose",
                QuestionKind.YES_NO: "whether",
            }.get(question.kind, "whether")

        requested_role = question.requested_role
        if question.kind == QuestionKind.YES_NO:
            clause = self.render_event(event, capitalize=False)
            return self._finish_clause(f"{operator} {clause}", capitalize)

        if requested_role == "possessor":
            focus = getattr(question, "focus_surface", "")
            clause = self.render_event(event, capitalize=False)
            tail = clause
            normalized_focus = focus.lower().strip()
            for prefix in (
                f"the {normalized_focus} ",
                f"a {normalized_focus} ",
                f"an {normalized_focus} ",
                f"{normalized_focus} ",
            ):
                if normalized_focus and clause.lower().startswith(prefix):
                    tail = clause[len(prefix):]
                    break
            terms = [operator]
            if focus:
                terms.append(focus)
            terms.append(tail)
            return self._finish_clause(" ".join(terms), capitalize)

        if requested_role in {"agent", "subject", "experiencer"}:
            tail = self._predicate_tail_for_question(event)
            focus = getattr(question, "focus_surface", "")
            terms = [operator]
            if focus and focus.lower() != operator.lower():
                terms.append(focus)
            terms.append(tail)
            return self._finish_clause(" ".join(terms), capitalize)

        clause = self.render_event(event, capitalize=False)
        focus = getattr(question, "focus_surface", "")
        terms = [operator]
        if focus and question.kind in {QuestionKind.WHICH, QuestionKind.WHOSE}:
            terms.append(focus)
        terms.append(clause)
        return self._finish_clause(" ".join(terms), capitalize)

    def render_embedded_interrogative_query(
        self,
        question: Any,
        *,
        capitalize: bool = False,
    ) -> str:
        matrix_clause = self.render_event(question.event, capitalize=False)
        inner = question.embedded_question
        if inner is None:
            return self._finish_clause(matrix_clause, capitalize)
        inner_clause = self.render_embedded_question_frame(
            inner,
            marker=question.embedded_marker,
            capitalize=False,
        )
        return self._finish_clause(
            f"{matrix_clause} {inner_clause}",
            capitalize,
        )

    def _infinitival_relation_from_contract(
        self,
        contract: AnswerContract,
    ) -> Optional[Any]:
        relation_id = contract.required_slots.get("relation_id", "")
        if relation_id:
            return next(
                (
                    item
                    for item in self.memory.infinitivals
                    if item.relation_id == relation_id
                ),
                None,
            )
        matrix_event_id = contract.required_slots.get("matrix_event_id", "")
        complement_event_id = contract.required_slots.get(
            "complement_event_id",
            "",
        )
        return next(
            (
                item
                for item in self.memory.infinitivals
                if (
                    not matrix_event_id
                    or item.matrix_event_id == matrix_event_id
                )
                and (
                    not complement_event_id
                    or item.complement_event_id == complement_event_id
                )
            ),
            None,
        )

    def render_infinitival_relation(
        self,
        relation: Any,
        *,
        capitalize: bool = False,
    ) -> str:
        matrix = self.memory.get_event(relation.matrix_event_id)
        complement = self.memory.get_event(relation.complement_event_id)
        if matrix is None or complement is None:
            return ""
        matrix_clause = self.render_event(matrix, capitalize=False)
        complement_clause = self._render_infinitive_event(
            complement,
            relation.controller_entity_id,
        )
        text = f"{matrix_clause} to {complement_clause}".strip()
        return self._finish_clause(text, capitalize)

    def _render_infinitive_event(
        self,
        event: EventFrame,
        controller_entity_id: str,
    ) -> str:
        args = dict(event.arguments)
        for role in ("agent", "subject", "experiencer", "possessor", "patient"):
            value = args.get(role)
            if (
                value is not None
                and value.kind == RefKind.ENTITY
                and value.key == controller_entity_id
            ):
                args.pop(role, None)
                break

        negative = not event.polarity
        prefix = "not " if negative else ""
        if event.predicate == "be":
            complement = self._render_copular_complement(args)
            return f"{prefix}be {complement}".strip()

        pieces = [f"{prefix}{event.predicate}".strip()]
        patient = args.get("patient") or args.get("state")
        if patient:
            pieces.append(
                self.render_ref(
                    patient,
                    case="object",
                    definite=self._should_be_definite(patient),
                )
            )
        recipient = args.get("recipient")
        if recipient:
            pieces.extend(
                [
                    "to",
                    self.render_ref(
                        recipient,
                        case="object",
                        definite=True,
                    ),
                ]
            )
        for role, preposition in (
            ("destination", "to"),
            ("source", "from"),
            ("location", "at"),
            ("time", ""),
            ("method", "by"),
            ("manner", ""),
        ):
            value = args.get(role)
            if value is None:
                continue
            phrase = self.render_ref(
                value,
                case="object",
                definite=role in {"destination", "source", "location"},
            )
            if role == "time":
                phrase = self._render_time_phrase(phrase)
            if preposition:
                pieces.extend([preposition, phrase])
            else:
                pieces.append(phrase)
        return " ".join(item for item in pieces if item).strip()

    def _gerund_relation_from_contract(
        self,
        contract: AnswerContract,
    ) -> Optional[GerundRelation]:
        relation_id = contract.required_slots.get("relation_id", "")
        if relation_id:
            return next(
                (
                    item
                    for item in self.memory.gerunds
                    if item.relation_id == relation_id
                ),
                None,
            )
        matrix_event_id = contract.required_slots.get("matrix_event_id", "")
        complement_event_id = contract.required_slots.get(
            "complement_event_id",
            "",
        )
        return next(
            (
                item
                for item in self.memory.gerunds
                if (
                    not matrix_event_id
                    or item.matrix_event_id == matrix_event_id
                )
                and (
                    not complement_event_id
                    or item.complement_event_id == complement_event_id
                )
            ),
            None,
        )

    def render_gerund_relation(
        self,
        relation: GerundRelation,
        *,
        capitalize: bool = False,
    ) -> str:
        matrix = self.memory.get_event(relation.matrix_event_id)
        complement = self.memory.get_event(relation.complement_event_id)
        if matrix is None or complement is None:
            return ""
        matrix_clause = self.render_event(matrix, capitalize=False)
        complement_clause = self._render_gerund_event(
            complement,
            relation.controller_entity_id,
        )
        text = " ".join(
            item for item in (matrix_clause, complement_clause) if item
        )
        return self._finish_clause(text, capitalize)

    def _render_gerund_event(
        self,
        event: EventFrame,
        controller_entity_id: str,
    ) -> str:
        args = dict(event.arguments)
        for role in ("agent", "subject", "experiencer", "possessor", "patient"):
            value = args.get(role)
            if (
                value is not None
                and value.kind == RefKind.ENTITY
                and value.key == controller_entity_id
            ):
                args.pop(role, None)
                break

        prefix = "not " if not event.polarity else ""
        gerund = lexicon.gerund_form(event.predicate)
        if event.predicate == "be":
            complement = self._render_copular_complement(args)
            return f"{prefix}{gerund} {complement}".strip()

        pieces = [f"{prefix}{gerund}".strip()]
        patient = args.get("patient") or args.get("state")
        if patient:
            pieces.append(
                self.render_ref(
                    patient,
                    case="object",
                    # The selected complement already carries its licensed
                    # nominal surface (``groceries`` versus ``the door``).
                    # Discourse definiteness from the later question must not
                    # rewrite that embedded object.
                    definite=False,
                )
            )
        recipient = args.get("recipient")
        if recipient:
            pieces.extend(
                [
                    "to",
                    self.render_ref(
                        recipient,
                        case="object",
                        definite=True,
                    ),
                ]
            )
        for role, preposition in (
            ("destination", "to"),
            ("source", "from"),
            ("location", "at"),
            ("time", ""),
            ("method", "by"),
            ("manner", ""),
            ("purpose", "to"),
            ("cause", "because"),
            ("topic", "about"),
        ):
            value = args.get(role)
            if value is None:
                continue
            phrase = self.render_ref(
                value,
                case="object",
                definite=role in {"destination", "source", "location"},
            )
            stored_preposition = args.get(f"{role}_preposition")
            if stored_preposition:
                preposition = self.render_ref(stored_preposition)
            if role == "time":
                phrase = (
                    phrase
                    if stored_preposition
                    else self._render_time_phrase(phrase)
                )
            if role == "purpose" and phrase.lower().startswith("to "):
                pieces.append(phrase)
            elif preposition:
                pieces.extend([preposition, phrase])
            else:
                pieces.append(phrase)
        quantity = args.get("quantity")
        if quantity and not args.get("patient"):
            pieces.append(self.render_ref(quantity))
        return " ".join(item for item in pieces if item).strip()

    def render_event(
        self,
        event: EventFrame,
        *,
        polarity: Optional[bool] = None,
        omit_roles: Iterable[str] = (),
        omit_variables: bool = True,
        capitalize: bool = False,
    ) -> str:
        args = {
            role: value
            for role, value in event.arguments.items()
            if role not in set(omit_roles) and not (omit_variables and value.is_variable)
        }
        positive = event.polarity if polarity is None else polarity

        if event.predicate == "attribute":
            subject = args.get("subject")
            value = args.get("value")
            attribute = args.get("attribute")
            if subject and value:
                text = f"{self.render_ref(subject, case='subject')} {self._be_form(event, subject, positive)} {self.render_ref(value)}"
            elif subject and attribute:
                text = f"{self.render_ref(subject, case='subject')} has a known {self.render_ref(attribute)}"
            else:
                text = "the attribute is known"
            return self._finish_clause(text, capitalize)

        subject_role, subject_ref = self._select_subject(event, args)
        if not subject_ref:
            return self._finish_clause(event.raw_text.strip().rstrip(".?!") or event.predicate, capitalize)
        subject_text = self.render_ref(subject_ref, case="subject", definite=True)

        if event.predicate == "be":
            complement = self._render_copular_complement(args)
            verb = (
                self._verb_form("be", event, subject_ref, positive)
                if event.aspect in {"progressive", "perfect_progressive"}
                else self._be_form(event, subject_ref, positive)
            )
            text = f"{subject_text} {verb} {complement}".strip()
            return self._finish_clause(text, capitalize)

        if event.predicate == "own":
            patient = args.get("patient")
            verb = self._verb_form("have", event, subject_ref, positive)
            # Body parts and ordinary possessed objects retain their natural
            # determiner; previously mentioned standalone things are definite.
            definite = bool(patient and self._should_be_definite(patient))
            text = f"{subject_text} {verb} {self.render_ref(patient, definite=definite) if patient else ''}".strip()
            return self._finish_clause(text, capitalize)

        verb = self._verb_form(event.predicate, event, subject_ref, positive)
        pieces = [subject_text, verb]

        # Passive event: patient selected as subject and a separate agent exists.
        if subject_role == "patient" and "agent" in args and event.predicate not in lexicon.UNACCUSATIVE_VERBS:
            aux = self._be_form(event, subject_ref, positive=True)
            participle = lexicon.participle_form(event.predicate)
            if not positive:
                aux = self._negative_be(event, subject_ref)
            pieces = [subject_text, aux, participle, "by", self.render_ref(args["agent"], case="object")]
        else:
            patient = args.get("patient") or args.get("state")
            if patient and patient != subject_ref:
                if args.get("quantity") and patient.kind == RefKind.ENTITY:
                    quantity_text = self.render_ref(args["quantity"])
                    entity = self.memory.get_entity(patient.key)
                    patient_text = entity.canonical_name if entity else self.render_ref(patient, case="object", definite=False)
                    patient_text = re.sub(r"^(?:a|an|the)\s+", "", patient_text, flags=re.IGNORECASE)
                    # A parser may preserve the original surface (for example
                    # ``three cars``) while also storing a typed quantity slot.
                    # Realization uses the entity's canonical head so the
                    # quantity is expressed exactly once.
                    patient_text = re.sub(
                        r"^(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|\d+(?:\.\d+)?)\s+)+",
                        "",
                        patient_text,
                        flags=re.IGNORECASE,
                    )
                    pieces.append(f"{quantity_text} {patient_text}")
                else:
                    pieces.append(self.render_ref(patient, case="object", definite=self._should_be_definite(patient)))
            recipient = args.get("recipient")
            if recipient:
                pieces.extend(["to", self.render_ref(recipient, case="object", definite=True)])

        adjunct_order = [
            ("destination", "to"),
            ("source", "from"),
            ("location", "at"),
            ("time", ""),
            ("method", "by"),
            ("manner", ""),
            ("purpose", "to"),
            ("cause", "because"),
            ("motive", "because"),
            ("justification", "because"),
            ("evidence", "because"),
            ("topic", "about"),
        ]
        used_reason = False
        for role, prefix in adjunct_order:
            value = args.get(role)
            if not value:
                continue
            if role in {"cause", "motive", "justification", "evidence"}:
                if used_reason:
                    continue
                used_reason = True
            phrase = self.render_ref(value, case="object", definite=role in {"location", "destination", "source"})
            stored_preposition = args.get(f"{role}_preposition")
            if stored_preposition:
                prefix = self.render_ref(stored_preposition)
            if role == "time":
                phrase = phrase if stored_preposition else self._render_time_phrase(phrase)
            if role == "purpose" and phrase.lower().startswith("to "):
                pieces.append(phrase)
            elif prefix:
                pieces.extend([prefix, phrase])
            else:
                pieces.append(phrase)

        quantity = args.get("quantity")
        if quantity and not args.get("patient"):
            pieces.append(self.render_ref(quantity))

        text = " ".join(piece for piece in pieces if piece).strip()
        return self._finish_clause(text, capitalize)

    def _select_subject(self, event: EventFrame, args: Dict[str, SemanticRef]) -> Tuple[str, Optional[SemanticRef]]:
        for role in ("agent", "subject", "experiencer", "possessor"):
            if role in args:
                return role, args[role]
        if "patient" in args:
            return "patient", args["patient"]
        return "", None

    def _render_copular_complement(self, args: Dict[str, SemanticRef]) -> str:
        if "location" in args:
            prep = self.render_ref(args.get("location_preposition")) or "at"
            return f"{prep} {self.render_ref(args['location'], definite=True)}"
        if "time" in args:
            prep = self.render_ref(args.get("time_preposition"))
            phrase = self.render_ref(args["time"])
            return f"{prep} {phrase}".strip() if prep else self._render_time_phrase(phrase)
        if "value" in args:
            return self.render_ref(args["value"], definite=False)
        if "state" in args:
            return self.render_ref(args["state"], definite=False)
        return ""

    def _verb_form(self, predicate: str, event: EventFrame, subject: SemanticRef, positive: bool) -> str:
        entity = self.memory.get_entity(subject.key) if subject.kind == RefKind.ENTITY else None
        plural = bool(entity and entity.number == GrammaticalNumber.PLURAL)
        first_person = subject.kind == RefKind.ENTITY and subject.key == "assistant"
        second_person = subject.kind == RefKind.ENTITY and subject.key == "user"
        third_singular = not plural and not first_person and not second_person

        if event.aspect == "perfect_progressive":
            gerund = lexicon.gerund_form(predicate)
            negation = "" if positive else " not"
            if event.tense == "future":
                modal = event.modality or "will"
                auxiliary = f"{modal}{negation} have been"
            elif event.modality:
                auxiliary = f"{event.modality}{negation} have been"
            elif event.tense == "past":
                auxiliary = f"had{negation} been"
            else:
                have = "has" if third_singular else "have"
                auxiliary = f"{have}{negation} been"
            return f"{auxiliary} {gerund}"

        if event.aspect == "progressive":
            gerund = lexicon.gerund_form(predicate)
            if event.modality:
                auxiliary = f"{event.modality}{'' if positive else ' not'} be"
            elif event.tense == "future":
                auxiliary = f"will{'' if positive else ' not'} be"
            else:
                auxiliary = (
                    self._be_form(event, subject, True)
                    if positive
                    else self._negative_be(event, subject)
                )
            return f"{auxiliary} {gerund}"

        if event.aspect == "perfect":
            participle = lexicon.participle_form(predicate)
            if event.tense == "future":
                auxiliary = "will have"
            elif event.tense == "past":
                auxiliary = "had"
            else:
                auxiliary = "has" if third_singular else "have"
            return f"{auxiliary}{'' if positive else ' not'} {participle}"

        if positive:
            if event.modality:
                return f"{event.modality} {predicate}"
            if event.tense == "past":
                return lexicon.past_form(predicate, plural_subject=plural)
            if event.tense == "future":
                return f"will {predicate}"
            return lexicon.present_form(
                predicate,
                third_person_singular=third_singular,
                plural_subject=plural,
                first_person=first_person,
            )

        if event.modality:
            return f"{event.modality} not {predicate}"
        if predicate == "be":
            return self._negative_be(event, subject)
        if event.tense == "past":
            return f"did not {predicate}"
        if event.tense == "future":
            return f"will not {predicate}"
        auxiliary = "does" if third_singular else "do"
        return f"{auxiliary} not {predicate}"

    def _be_form(self, event: EventFrame, subject: SemanticRef, positive: bool) -> str:
        if not positive:
            return self._negative_be(event, subject)
        entity = self.memory.get_entity(subject.key) if subject.kind == RefKind.ENTITY else None
        plural = bool(entity and entity.number == GrammaticalNumber.PLURAL)
        first_person = subject.kind == RefKind.ENTITY and subject.key == "assistant"
        second_person = subject.kind == RefKind.ENTITY and subject.key == "user"
        if event.tense == "past":
            return "were" if plural or second_person else "was"
        if event.tense == "future":
            return "will be"
        if first_person:
            return "am"
        if plural or second_person:
            return "are"
        return "is"

    def _negative_be(self, event: EventFrame, subject: SemanticRef) -> str:
        base = self._be_form(event.copy(polarity=True), subject, True)
        if base == "will be":
            return "will not be"
        return f"{base} not"

    def render_ref(
        self,
        ref: Optional[SemanticRef],
        *,
        case: str = "object",
        definite: bool = False,
        capitalize: bool = False,
    ) -> str:
        if ref is None:
            return ""
        if ref.kind == RefKind.VARIABLE:
            return ref.surface or f"the {ref.key}"
        if ref.kind == RefKind.EVENT:
            event = next((item for item in self.memory.events if item.event_id == ref.key), None)
            text = self.render_event(event, capitalize=False) if event else (ref.surface or "the event")
            return self._capitalize(text) if capitalize else text
        if ref.kind == RefKind.LITERAL:
            text = ref.surface or ref.key
            return self._capitalize(text) if capitalize else text

        entity = self.memory.get_entity(ref.key)
        if not entity:
            text = ref.surface or ref.key
            return self._capitalize(text) if capitalize else text
        if entity.entity_id == "user":
            text = {"subject": "you", "object": "you", "possessive": "your"}.get(case, "you")
        elif entity.entity_id == "assistant":
            text = {"subject": "I", "object": "me", "possessive": "my"}.get(case, "me")
        elif entity.relation and entity.owner_id:
            owner = "your" if entity.owner_id == "user" else "my" if entity.owner_id == "assistant" else self.memory.describe_entity(entity.owner_id) + "'s"
            text = f"{owner} {entity.relation}"
        else:
            text = entity.canonical_name
            if definite and entity.kind in {EntityKind.THING, EntityKind.PLACE} and not self._has_determiner(text) and not self._looks_proper(entity):
                text = "the " + text
            elif not definite and entity.kind == EntityKind.THING and ref.surface:
                text = ref.surface
        return self._capitalize(text) if capitalize else text

    @staticmethod
    def _looks_proper(entity: Entity) -> bool:
        return bool(entity.canonical_name[:1].isupper())

    @staticmethod
    def _has_determiner(text: str) -> bool:
        first = text.lower().split()[0] if text.split() else ""
        return first in lexicon.DETERMINERS | lexicon.POSSESSIVES

    def _should_be_definite(self, ref: SemanticRef) -> bool:
        entity = self.memory.get_entity(ref.key) if ref.kind == RefKind.ENTITY else None
        return bool(entity and entity.last_mentioned_turn < self.memory.turn_index)


    @staticmethod
    def _render_time_phrase(phrase: str) -> str:
        lower = phrase.lower().strip()
        if not lower or lower.startswith(("on ", "at ", "in ", "before ", "after ", "during ", "since ", "until ")):
            return phrase
        first = lower.split()[0]
        if first in lexicon.DAYS or first in lexicon.MONTHS:
            return "on " + phrase
        if re.fullmatch(r"\d{1,2}(:\d{2})?(am|pm)?", lower.replace(" ", "")):
            return "at " + phrase
        return phrase

    def _predicate_tail_for_question(self, event: EventFrame) -> str:
        args = dict(event.arguments)
        args.pop("agent", None)
        args.pop("subject", None)
        temp = event.copy(arguments={"agent": SemanticRef.literal("someone", "someone", EntityKind.PERSON), **args})
        rendered = self.render_event(temp, capitalize=False)
        if rendered.startswith("someone "):
            return rendered[len("someone ") :]
        return rendered

    # ------------------------------------------------------------------
    # Validation and punctuation
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Semantic validation and punctuation
    # ------------------------------------------------------------------

    def _validate_surface(self, text: str, contract: AnswerContract) -> Tuple[bool, str]:
        lower = text.lower()
        if not text.strip():
            return False, "empty composition"
        if contract.status == AnswerStatus.ANSWERED and contract.values:
            value = self.render_ref(contract.values[0], definite=False).lower()
            ref = contract.values[0]
            entity = self.memory.get_entity(ref.key) if ref.kind == RefKind.ENTITY else None
            aliases = [value]
            if (
                ref.kind == RefKind.EVENT
                and contract.required_slots.get("gerund") == "true"
            ):
                relation = self._gerund_relation_from_contract(contract)
                complement = (
                    self.memory.get_event(relation.complement_event_id)
                    if relation is not None
                    else None
                )
                if complement is not None and ref.key == complement.event_id:
                    aliases.extend(
                        [
                            lexicon.gerund_form(complement.predicate).lower(),
                            self._render_gerund_event(
                                complement,
                                relation.controller_entity_id,
                            ).lower(),
                        ]
                    )
            if entity:
                aliases.extend([entity.canonical_name.lower(), *entity.aliases])
            if not any(alias and alias in lower for alias in aliases):
                return False, "answer surface omits the bound value"
        if contract.status == AnswerStatus.TRUE and not lower.startswith(("yes", "yeah")):
            return False, "truth answer lacks an affirmation atom"
        if contract.status == AnswerStatus.FALSE and not lower.startswith(("no", "nope")):
            return False, "false answer lacks a denial atom"
        if contract.status == AnswerStatus.UNKNOWN and not any(marker in lower for marker in ("don't know", "do not know", "haven't told", "cannot determine")):
            return False, "unknown answer asserts beyond evidence"
        if contract.status in {AnswerStatus.MISSING_REFERENCE, AnswerStatus.AMBIGUOUS_REFERENCE, AnswerStatus.MULTIPLE_MATCHES, AnswerStatus.LEXICAL_PROBE} and not text.rstrip().endswith("?"):
            return False, "probe composition is not interrogative"
        if contract.status == AnswerStatus.CONFLICT and "conflict" not in lower:
            return False, "conflict composition hides disagreement"
        return True, "semantic contract preserved by compositional grammar"

    @staticmethod
    def _finish_clause(text: str, capitalize: bool) -> str:
        text = re.sub(r"\s+", " ", text).strip().rstrip(".?!")
        return SurfaceRealizer._capitalize(text) if capitalize else text

    @staticmethod
    def _capitalize(text: str) -> str:
        if not text:
            return text
        return text[0].upper() + text[1:]

    @staticmethod
    def join_phrases(items: Sequence[str], *, conjunction: str = "and") -> str:
        clean = [item for item in items if item]
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} {conjunction} {clean[1]}"
        return ", ".join(clean[:-1]) + f", {conjunction} {clean[-1]}"
