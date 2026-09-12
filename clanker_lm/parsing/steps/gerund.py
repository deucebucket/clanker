"""Gerund assembly stage. Returns whether the clause has been handled."""

from __future__ import annotations
import copy
from typing import Optional, Sequence
from ... import lexicon
from ...model import GerundRelation, GerundRelationType, SourceKind, UnresolvedReference
from ..ports import ParserPort
from ..context import ParseContext

def apply(parser: ParserPort, clause: Sequence[lexicon.Token],
          connector: Optional[str], ctx: ParseContext) -> bool:
    """Handle this construction or return False to continue the ordered pipeline."""
    gerund_split, gerund_ambiguity = parser._split_gerund_clause(clause)
    if gerund_ambiguity is not None:
        ctx.gerund_ambiguities.append(gerund_ambiguity)
        ctx.unresolved.append(
            UnresolvedReference(
                surface=gerund_ambiguity.complement_surface,
                reason=gerund_ambiguity.reason,
            )
        )
        ctx.diagnostics.extend(gerund_ambiguity.diagnostics)
        ctx.diagnostics.append(
            "gerund complement boundary/controller remains ambiguous"
        )
        return True

    if gerund_split is not None:
        memory_checkpoint = copy.deepcopy(ctx.memory.__dict__)
        matrix_result = parser._parse_clause(
            gerund_split.matrix_tokens,
            ctx.raw,
            ctx.memory,
        )
        if matrix_result.event is not None:
            if (
                matrix_result.event.predicate == "keep"
                and "possessor" in matrix_result.event.arguments
                and "agent" not in matrix_result.event.arguments
            ):
                # KEEP is possessive in ordinary clauses, but its
                # selected -ing construction is aspectual.  The matrix
                # subject is therefore an actor/source, not a possessor.
                matrix_result.event.arguments["agent"] = (
                    matrix_result.event.arguments.pop("possessor")
                )
                matrix_result.diagnostics.append(
                    "aspectual keep subject normalized as agent"
                )
            source_entity_id = parser._gerund_source_entity(
                matrix_result.event
            )
            controller_entity_id = parser._gerund_controller_entity(
                matrix_result.event,
                gerund_split.controller_role,
            )
            if source_entity_id and controller_entity_id:
                internal_alias = ctx.memory.ensure_internal_alias(
                    controller_entity_id
                )
                complement_result = parser._parse_clause(
                    [
                        lexicon.Token(
                            internal_alias,
                            internal_alias,
                            -1,
                        ),
                        *gerund_split.complement_tokens,
                    ],
                    ctx.raw,
                    ctx.memory,
                )
                if (
                    complement_result.event is not None
                    and not complement_result.unresolved
                ):
                    matrix_result.event.discourse_role = (
                        "main"
                        if ctx.primary_event_count == 0
                        else "coordinate"
                    )
                    participial = (
                        gerund_split.profile.relation_type
                        == GerundRelationType.PERCEPTION_PARTICIPIAL
                    )
                    complement_result.event.discourse_role = (
                        "participle" if participial else "gerund"
                    )
                    complement_result.event.source = SourceKind.ATTRIBUTED
                    complement_result.event.tense = "nonfinite"
                    complement_result.event.aspect = (
                        "participle"
                        if participial
                        else "gerund"
                    )
                    complement_result.event.certainty = min(
                        complement_result.event.certainty,
                        gerund_split.certainty,
                    )
                    if connector is not None:
                        matrix_result.diagnostics.insert(
                            0,
                            f"coordinate connector={connector}",
                        )
                    ctx.primary_event_count += 1

                    matrix_index = len(ctx.events)
                    ctx.events.append(matrix_result.event)
                    complement_index = len(ctx.events)
                    ctx.events.append(complement_result.event)
                    licensed = matrix_result.event.polarity
                    phase_entailed = (
                        gerund_split.profile.phase_entailing
                        and parser._is_factual_phase_matrix(
                            matrix_result.event,
                            complement_result.event,
                        )
                    )
                    ctx.gerunds.append(
                        GerundRelation(
                            relation_type=(
                                gerund_split.profile.relation_type
                            ),
                            content_status=(
                                gerund_split.profile.content_status
                            ),
                            matrix_event_index=matrix_index,
                            complement_event_index=complement_index,
                            marker=gerund_split.marker,
                            matrix_predicate=matrix_result.event.predicate,
                            source_entity_id=source_entity_id,
                            controller_entity_id=controller_entity_id,
                            embedded_subject_entity_id=(
                                controller_entity_id
                            ),
                            predicate_family=(
                                gerund_split.profile.predicate_family
                            ),
                            certainty=gerund_split.certainty,
                            licensed=licensed,
                            entailed=phase_entailed,
                            diagnostics=[
                                *gerund_split.diagnostics,
                                (
                                    "matrix polarity=positive"
                                    if matrix_result.event.polarity
                                    else "matrix polarity=negative"
                                ),
                                (
                                    "embedded polarity=positive"
                                    if complement_result.event.polarity
                                    else "embedded polarity=negative"
                                ),
                                (
                                    "licensed phase entailment remains "
                                    "derived and nonassertive"
                                    if phase_entailed
                                    else "modal, future-scheduled, or "
                                    "progressive phase matrix remains "
                                    "non-entailed"
                                    if licensed
                                    and gerund_split.profile.phase_entailing
                                    else "selected -ing content remains "
                                    "qualified and nonassertive"
                                    if licensed
                                    else "negated matrix does not license "
                                    "positive -ing content"
                                ),
                            ],
                        )
                    )
                    ctx.unresolved.extend(matrix_result.unresolved)
                    ctx.entities.extend(matrix_result.entities)
                    ctx.entities.extend(complement_result.entities)
                    ctx.diagnostics.extend(matrix_result.diagnostics)
                    ctx.diagnostics.extend(complement_result.diagnostics)
                    ctx.diagnostics.extend(gerund_split.diagnostics)
                    ctx.diagnostics.append(
                        "gerund relation="
                        f"{gerund_split.profile.relation_type.value} "
                        f"status={gerund_split.profile.content_status.value} "
                        f"controller={controller_entity_id}"
                    )
                    return True

                reason = (
                    "embedded -ing event could not be parsed with its "
                    "licensed controller"
                )
                ambiguity = parser._gerund_ambiguity(
                    clause,
                    [len(gerund_split.matrix_tokens)],
                    reason=reason,
                    candidate_types=[
                        gerund_split.profile.relation_type
                    ],
                )
                parser._restore_memory_checkpoint(
                    ctx.memory,
                    memory_checkpoint,
                )
                ctx.gerund_ambiguities.append(ambiguity)
                ctx.unresolved.extend(complement_result.unresolved)
                ctx.diagnostics.extend(complement_result.diagnostics)
                ctx.diagnostics.extend(ambiguity.diagnostics)
                return True

            missing = "source" if not source_entity_id else "controller"
            ambiguity = parser._gerund_ambiguity(
                clause,
                [len(gerund_split.matrix_tokens)],
                reason=(
                    f"matrix gerund predicate lacks a resolved {missing} entity"
                ),
                candidate_types=[gerund_split.profile.relation_type],
            )
            parser._restore_memory_checkpoint(
                ctx.memory,
                memory_checkpoint,
            )
            ctx.gerund_ambiguities.append(ambiguity)
            ctx.unresolved.extend(matrix_result.unresolved)
            ctx.diagnostics.extend(matrix_result.diagnostics)
            ctx.diagnostics.extend(ambiguity.diagnostics)
            return True

        ambiguity = parser._gerund_ambiguity(
            clause,
            [len(gerund_split.matrix_tokens)],
            reason="selected -ing matrix event could not be parsed",
            candidate_types=[gerund_split.profile.relation_type],
        )
        parser._restore_memory_checkpoint(ctx.memory, memory_checkpoint)
        ctx.gerund_ambiguities.append(ambiguity)
        ctx.diagnostics.extend(matrix_result.diagnostics)
        ctx.diagnostics.extend(ambiguity.diagnostics)
        return True
    return False
