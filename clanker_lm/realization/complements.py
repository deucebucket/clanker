"""Complements responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import Any, Optional
from .. import lexicon
from ..model import AnswerContract, EventFrame, GerundRelation, RefKind

class ComplementsComponent:
    """Complements behavior composed by the stable public SurfaceRealizer API."""

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
