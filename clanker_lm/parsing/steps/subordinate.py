"""Subordinate assembly stage. Returns whether the clause has been handled."""

from __future__ import annotations
from typing import Optional, Sequence
from ... import lexicon
from ...model import ClauseRelation, ClauseRelationType, EntityKind, SemanticRef
from ..ports import ParserPort
from ..context import ParseContext

def apply(parser: ParserPort, clause: Sequence[lexicon.Token],
          connector: Optional[str], ctx: ParseContext) -> bool:
    """Handle this construction or return False to continue the ordered pipeline."""
    subordinate_split = parser._split_subordinate_clause(clause)
    if subordinate_split is not None:
        main_result = parser._parse_clause(
            subordinate_split.main_tokens,
            ctx.raw,
            ctx.memory,
        )
        subordinate_result = parser._parse_clause(
            subordinate_split.subordinate_tokens,
            ctx.raw,
            ctx.memory,
        )
        if main_result.event and subordinate_result.event:
            main_result.event.discourse_role = (
                "main" if ctx.primary_event_count == 0 else "coordinate"
            )
            subordinate_result.event.discourse_role = "subordinate"
            if connector is not None:
                main_result.diagnostics.insert(
                    0,
                    f"coordinate connector={connector}",
                )
            ctx.primary_event_count += 1

            main_index = len(ctx.events)
            ctx.events.append(main_result.event)
            subordinate_index = len(ctx.events)
            ctx.events.append(subordinate_result.event)

            relation = ClauseRelation(
                relation_type=subordinate_split.relation_type,
                main_event_index=main_index,
                subordinate_event_index=subordinate_index,
                marker=subordinate_split.marker,
                direction=subordinate_split.direction,
                certainty=subordinate_split.certainty,
                candidate_types=list(subordinate_split.candidate_types),
                diagnostics=list(subordinate_split.diagnostics),
            )
            ctx.relations.append(relation)

            subordinate_surface = parser._surface(
                subordinate_split.subordinate_tokens
            )
            subordinate_key = lexicon.normalize_phrase(
                token.norm
                for token in subordinate_split.subordinate_tokens
            )
            if relation.relation_type == ClauseRelationType.CAUSE:
                cause_role = (
                    "motive"
                    if main_result.event.predicate in lexicon.VOLITIONAL_VERBS
                    else "cause"
                )
                cause_ref = SemanticRef.literal(
                    subordinate_key,
                    subordinate_surface,
                    EntityKind.ABSTRACT,
                )
                main_result.event.arguments[cause_role] = cause_ref
                main_result.event.arguments.setdefault("cause", cause_ref)
            elif relation.relation_type == ClauseRelationType.PURPOSE:
                main_result.event.arguments["purpose"] = SemanticRef.literal(
                    subordinate_key,
                    subordinate_surface,
                    EntityKind.ABSTRACT,
                )

            ctx.unresolved.extend(main_result.unresolved)
            ctx.unresolved.extend(subordinate_result.unresolved)
            ctx.entities.extend(main_result.entities)
            ctx.entities.extend(subordinate_result.entities)
            ctx.diagnostics.extend(main_result.diagnostics)
            ctx.diagnostics.extend(subordinate_result.diagnostics)
            ctx.diagnostics.extend(subordinate_split.diagnostics)
            ctx.diagnostics.append(
                "clause relation="
                f"{relation.relation_type.value} "
                f"marker={relation.marker} "
                f"direction={relation.direction.value}"
            )
            return True

        ctx.diagnostics.extend(main_result.diagnostics)
        ctx.diagnostics.extend(subordinate_result.diagnostics)
        ctx.diagnostics.append(
            f"subordinate split fallback marker={subordinate_split.marker}"
        )
    return False
