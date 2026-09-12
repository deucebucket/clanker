"""Embedded assembly stage. Returns whether the clause has been handled."""

from __future__ import annotations
from typing import Optional, Sequence
from ... import lexicon
from ...model import EmbeddedInterrogativeRelation, EntityKind, SemanticRef, SourceKind, UnresolvedReference
from ..ports import ParserPort
from ..context import ParseContext

def apply(parser: ParserPort, clause: Sequence[lexicon.Token],
          connector: Optional[str], ctx: ParseContext) -> bool:
    """Handle this construction or return False to continue the ordered pipeline."""
    embedded_split, embedded_ambiguity = (
        parser._split_embedded_interrogative_clause(clause)
    )
    if embedded_ambiguity is not None:
        ctx.embedded_interrogative_ambiguities.append(embedded_ambiguity)
        ctx.unresolved.append(
            UnresolvedReference(
                surface=embedded_ambiguity.question_surface,
                reason=embedded_ambiguity.reason,
            )
        )
        ctx.diagnostics.extend(embedded_ambiguity.diagnostics)
        ctx.diagnostics.append(
            "embedded interrogative boundary/scope remains ambiguous"
        )
        return True

    if embedded_split is not None:
        matrix_result = parser._parse_clause(
            embedded_split.matrix_tokens,
            ctx.raw,
            ctx.memory,
        )
        if (
            matrix_result.event is not None
            and embedded_split.direct_answer_request
            and not parser._content_source_entity(matrix_result.event)
        ):
            matrix_result.event.arguments["agent"] = SemanticRef.entity(
                "assistant",
                "you",
                EntityKind.PERSON,
            )
            if "assistant" not in matrix_result.entities:
                matrix_result.entities.append("assistant")
            matrix_result.diagnostics.append(
                "imperative answer request supplies assistant agent"
            )
        question, question_unresolved, question_entities, question_diagnostics = (
            parser._parse_embedded_question(
                embedded_split.question_tokens,
                ctx.raw,
                ctx.memory,
            )
        )
        source_entity_id = (
            parser._content_source_entity(matrix_result.event)
            if matrix_result.event is not None
            else None
        )
        if (
            matrix_result.event is not None
            and question is not None
            and source_entity_id
            and not question_unresolved
        ):
            matrix_result.event.discourse_role = (
                "main" if ctx.primary_event_count == 0 else "coordinate"
            )
            question.event.discourse_role = "interrogative"
            question.event.source = SourceKind.ATTRIBUTED
            question.event.certainty = min(
                question.event.certainty,
                embedded_split.profile.certainty,
            )
            question.embedded_matrix_predicate = (
                matrix_result.event.predicate
            )
            if connector is not None:
                matrix_result.diagnostics.insert(
                    0,
                    f"coordinate connector={connector}",
                )
            ctx.primary_event_count += 1
            matrix_index = len(ctx.events)
            ctx.events.append(matrix_result.event)
            question_index = len(ctx.events)
            ctx.events.append(question.event)
            ctx.embedded_interrogatives.append(
                EmbeddedInterrogativeRelation(
                    relation_type=embedded_split.relation_type,
                    content_status=(
                        embedded_split.profile.content_status
                    ),
                    matrix_event_index=matrix_index,
                    question_event_index=question_index,
                    marker=embedded_split.marker,
                    matrix_predicate=matrix_result.event.predicate,
                    source_entity_id=source_entity_id,
                    predicate_family=(
                        embedded_split.profile.predicate_family
                    ),
                    question_kind=question.kind,
                    requested_role=question.requested_role,
                    answer_type=question.answer_type,
                    certainty=embedded_split.profile.certainty,
                    licensed=matrix_result.event.polarity,
                    direct_answer_request=(
                        embedded_split.direct_answer_request
                    ),
                    focus_surface=question.focus_surface,
                    why_kind=question.why_kind,
                    how_kind=question.how_kind,
                    diagnostics=[
                        *embedded_split.diagnostics,
                        *question_diagnostics,
                        (
                            "positive matrix licenses attributed "
                            "question content"
                            if matrix_result.event.polarity
                            else "negated matrix does not license "
                            "positive question attribution"
                        ),
                    ],
                )
            )
            if embedded_split.direct_answer_request:
                ctx.direct_embedded_question = question
            ctx.unresolved.extend(matrix_result.unresolved)
            ctx.entities.extend(matrix_result.entities)
            ctx.entities.extend(question_entities)
            ctx.diagnostics.extend(matrix_result.diagnostics)
            ctx.diagnostics.extend(question_diagnostics)
            ctx.diagnostics.extend(embedded_split.diagnostics)
            ctx.diagnostics.append(
                "embedded interrogative relation="
                f"{embedded_split.relation_type.value} "
                f"kind={question.kind.value} "
                f"source={source_entity_id}"
            )
            return True

        reason = (
            "embedded interrogative could not be parsed with its "
            "licensed matrix source"
        )
        ambiguity = parser._embedded_interrogative_ambiguity(
            clause,
            [len(embedded_split.matrix_tokens)],
            reason=reason,
        )
        ctx.embedded_interrogative_ambiguities.append(ambiguity)
        ctx.unresolved.extend(matrix_result.unresolved)
        ctx.unresolved.extend(question_unresolved)
        ctx.diagnostics.extend(matrix_result.diagnostics)
        ctx.diagnostics.extend(question_diagnostics)
        ctx.diagnostics.extend(ambiguity.diagnostics)
        return True
    return False
