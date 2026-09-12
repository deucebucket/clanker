"""Infinitival assembly stage. Returns whether the clause has been handled."""

from __future__ import annotations
from typing import Optional, Sequence
from ... import lexicon
from ...model import InfinitivalRelation, SourceKind, UnresolvedReference
from ..ports import ParserPort
from ..context import ParseContext

def apply(parser: ParserPort, clause: Sequence[lexicon.Token],
          connector: Optional[str], ctx: ParseContext) -> bool:
    """Handle this construction or return False to continue the ordered pipeline."""
    infinitival_split, infinitival_ambiguity = (
        parser._split_infinitival_clause(clause)
    )
    if infinitival_ambiguity is not None:
        ctx.infinitival_ambiguities.append(infinitival_ambiguity)
        ctx.unresolved.append(
            UnresolvedReference(
                surface=infinitival_ambiguity.complement_surface,
                reason=infinitival_ambiguity.reason,
            )
        )
        ctx.diagnostics.extend(infinitival_ambiguity.diagnostics)
        ctx.diagnostics.append(
            "infinitival complement boundary/controller remains ambiguous"
        )
        return True

    if infinitival_split is not None:
        matrix_result = parser._parse_clause(
            infinitival_split.matrix_tokens,
            ctx.raw,
            ctx.memory,
        )
        if matrix_result.event is not None:
            source_entity_id = parser._content_source_entity(
                matrix_result.event
            )
            controller_entity_id = parser._infinitival_controller_entity(
                matrix_result.event,
                infinitival_split.controller_role,
            )
            if source_entity_id and controller_entity_id:
                internal_alias = ctx.memory.ensure_internal_alias(
                    controller_entity_id
                )
                embedded_tokens = [
                    lexicon.Token(
                        internal_alias,
                        internal_alias,
                        -1,
                    ),
                    *infinitival_split.complement_tokens,
                ]
                complement_result = parser._parse_clause(
                    embedded_tokens,
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
                    complement_result.event.discourse_role = "infinitive"
                    complement_result.event.source = SourceKind.ATTRIBUTED
                    complement_result.event.tense = "infinitive"
                    complement_result.event.certainty = min(
                        complement_result.event.certainty,
                        infinitival_split.certainty,
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
                    ctx.infinitivals.append(
                        InfinitivalRelation(
                            relation_type=(
                                infinitival_split.profile.relation_type
                            ),
                            content_status=(
                                infinitival_split.profile.content_status
                            ),
                            matrix_event_index=matrix_index,
                            complement_event_index=complement_index,
                            marker=infinitival_split.marker,
                            matrix_predicate=(
                                matrix_result.event.predicate
                            ),
                            source_entity_id=source_entity_id,
                            controller_entity_id=controller_entity_id,
                            embedded_subject_entity_id=(
                                controller_entity_id
                            ),
                            predicate_family=(
                                infinitival_split.profile.predicate_family
                            ),
                            certainty=infinitival_split.certainty,
                            licensed=matrix_result.event.polarity,
                            entailed=False,
                            diagnostics=[
                                *infinitival_split.diagnostics,
                                (
                                    "positive matrix licenses the "
                                    "non-entailed complement relation"
                                    if matrix_result.event.polarity
                                    else "negated matrix does not "
                                    "license positive complement content"
                                ),
                            ],
                        )
                    )
                    ctx.unresolved.extend(matrix_result.unresolved)
                    ctx.entities.extend(matrix_result.entities)
                    ctx.entities.extend(complement_result.entities)
                    ctx.diagnostics.extend(matrix_result.diagnostics)
                    ctx.diagnostics.extend(complement_result.diagnostics)
                    ctx.diagnostics.extend(infinitival_split.diagnostics)
                    ctx.diagnostics.append(
                        "infinitival relation="
                        f"{infinitival_split.profile.relation_type.value} "
                        f"status={infinitival_split.profile.content_status.value} "
                        f"controller={controller_entity_id}"
                    )
                    return True

                reason = (
                    "embedded infinitive could not be parsed with its "
                    "licensed controller"
                )
                ambiguity = parser._infinitival_ambiguity(
                    clause,
                    [
                        len(infinitival_split.matrix_tokens)
                    ],
                    reason=reason,
                    candidate_types=[
                        infinitival_split.profile.relation_type
                    ],
                )
                ctx.infinitival_ambiguities.append(ambiguity)
                ctx.unresolved.extend(complement_result.unresolved)
                ctx.diagnostics.extend(complement_result.diagnostics)
                ctx.diagnostics.extend(ambiguity.diagnostics)
                return True

            missing = (
                "source"
                if not source_entity_id
                else "controller"
            )
            ambiguity = parser._infinitival_ambiguity(
                clause,
                [len(infinitival_split.matrix_tokens)],
                reason=(
                    f"matrix infinitival predicate lacks a resolved {missing} entity"
                ),
                candidate_types=[
                    infinitival_split.profile.relation_type
                ],
            )
            ctx.infinitival_ambiguities.append(ambiguity)
            ctx.unresolved.extend(matrix_result.unresolved)
            ctx.diagnostics.extend(matrix_result.diagnostics)
            ctx.diagnostics.extend(ambiguity.diagnostics)
            return True

        ctx.diagnostics.extend(matrix_result.diagnostics)
        ctx.diagnostics.append(
            f"infinitival split fallback marker={infinitival_split.marker}"
        )
    return False
