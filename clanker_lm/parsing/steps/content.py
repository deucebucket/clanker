"""Content assembly stage. Returns whether the clause has been handled."""

from __future__ import annotations
import hashlib
from typing import Optional, Sequence
from ... import lexicon
from ...model import ContentAttachmentAmbiguity, ContentRelation, SourceKind, UnresolvedReference
from ..ports import ParserPort
from ..context import ParseContext

def apply(parser: ParserPort, clause: Sequence[lexicon.Token],
          connector: Optional[str], ctx: ParseContext) -> bool:
    """Handle this construction or return False to continue the ordered pipeline."""
    content_split, content_ambiguity = parser._split_content_clause(clause)
    if content_ambiguity is not None:
        ctx.content_ambiguities.append(content_ambiguity)
        ctx.unresolved.append(
            UnresolvedReference(
                surface=content_ambiguity.content_surface,
                reason=content_ambiguity.reason,
            )
        )
        ctx.diagnostics.extend(content_ambiguity.diagnostics)
        ctx.diagnostics.append("content-clause boundary remains ambiguous")
        return True

    if content_split is not None:
        matrix_result = parser._parse_clause(
            content_split.matrix_tokens,
            ctx.raw,
            ctx.memory,
        )
        content_result = parser._parse_clause(
            content_split.content_tokens,
            ctx.raw,
            ctx.memory,
        )
        if matrix_result.event and content_result.event:
            source_entity_id = parser._content_source_entity(
                matrix_result.event
            )
            if not source_entity_id:
                ambiguity = ContentAttachmentAmbiguity(
                    matrix_surface=parser._surface(content_split.matrix_tokens),
                    content_surface=parser._surface(content_split.content_tokens),
                    clause_surface=parser._surface(clause),
                    reason="matrix content predicate lacks a resolved source entity",
                    ambiguity_id=(
                        "content-"
                        + hashlib.sha256(
                            " ".join(token.norm for token in clause).encode("utf-8")
                        ).hexdigest()[:16]
                    ),
                    diagnostics=[
                        "content source unresolved; durable assertion suppressed"
                    ],
                )
                ctx.content_ambiguities.append(ambiguity)
                ctx.unresolved.extend(matrix_result.unresolved)
                ctx.unresolved.extend(content_result.unresolved)
                ctx.diagnostics.extend(ambiguity.diagnostics)
                return True

            matrix_result.event.discourse_role = (
                "main" if ctx.primary_event_count == 0 else "coordinate"
            )
            content_result.event.discourse_role = "content"
            content_result.event.source = SourceKind.ATTRIBUTED
            content_result.event.certainty = min(
                content_result.event.certainty,
                content_split.certainty,
            )
            if connector is not None:
                matrix_result.diagnostics.insert(
                    0,
                    f"coordinate connector={connector}",
                )
            ctx.primary_event_count += 1

            matrix_index = len(ctx.events)
            ctx.events.append(matrix_result.event)
            content_index = len(ctx.events)
            ctx.events.append(content_result.event)
            ctx.contents.append(
                ContentRelation(
                    relation_type=content_split.relation_type,
                    matrix_event_index=matrix_index,
                    content_event_index=content_index,
                    marker=content_split.marker,
                    matrix_predicate=matrix_result.event.predicate,
                    source_entity_id=source_entity_id,
                    predicate_family=content_split.predicate_family,
                    certainty=content_split.certainty,
                    attributed=matrix_result.event.polarity,
                    diagnostics=[
                        *content_split.diagnostics,
                        (
                            "matrix predicate licenses attributed evidence"
                            if matrix_result.event.polarity
                            else "negated matrix predicate does not license attributed evidence"
                        ),
                    ],
                )
            )
            ctx.unresolved.extend(matrix_result.unresolved)
            ctx.unresolved.extend(content_result.unresolved)
            ctx.entities.extend(matrix_result.entities)
            ctx.entities.extend(content_result.entities)
            ctx.diagnostics.extend(matrix_result.diagnostics)
            ctx.diagnostics.extend(content_result.diagnostics)
            ctx.diagnostics.extend(content_split.diagnostics)
            ctx.diagnostics.append(
                "content relation="
                f"{content_split.relation_type.value} "
                f"marker={content_split.marker} "
                f"source={source_entity_id}"
            )
            return True

        ctx.diagnostics.extend(matrix_result.diagnostics)
        ctx.diagnostics.extend(content_result.diagnostics)
        ctx.diagnostics.append(
            f"content split fallback marker={content_split.marker}"
        )
    return False
