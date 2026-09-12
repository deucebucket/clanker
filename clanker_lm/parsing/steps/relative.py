"""Relative assembly stage. Returns whether the clause has been handled."""

from __future__ import annotations
from typing import Optional, Sequence
from ... import lexicon
from ...model import EntityModifierRelation
from ..ports import ParserPort
from ..context import ParseContext

def apply(parser: ParserPort, clause: Sequence[lexicon.Token],
          connector: Optional[str], ctx: ParseContext) -> bool:
    """Handle this construction or return False to continue the ordered pipeline."""
    relative_split, modifier_ambiguity = parser._split_relative_clause(
        clause,
        ctx.raw,
        ctx.memory,
    )
    if modifier_ambiguity is not None:
        ctx.modifier_ambiguities.append(modifier_ambiguity)
        ctx.diagnostics.extend(modifier_ambiguity.diagnostics)
        ctx.diagnostics.append(
            f"relative attachment ambiguous marker={modifier_ambiguity.marker}"
        )

    if relative_split is not None:
        main_result = parser._parse_clause(
            relative_split.main_tokens,
            ctx.raw,
            ctx.memory,
        )
        modifier_result = parser._parse_clause(
            relative_split.modifier_tokens,
            ctx.raw,
            ctx.memory,
        )
        if main_result.event and modifier_result.event:
            main_result.event.discourse_role = (
                "main" if ctx.primary_event_count == 0 else "coordinate"
            )
            modifier_result.event.discourse_role = "modifier"
            if connector is not None:
                main_result.diagnostics.insert(
                    0,
                    f"coordinate connector={connector}",
                )
            ctx.primary_event_count += 1

            main_index = len(ctx.events)
            ctx.events.append(main_result.event)
            modifier_index = len(ctx.events)
            ctx.events.append(modifier_result.event)
            ctx.modifiers.append(
                EntityModifierRelation(
                    head_entity_id=relative_split.head_entity_id,
                    modifier_event_index=modifier_index,
                    modifier_event_signature=parser._event_signature_key(
                        modifier_result.event
                    ),
                    marker=relative_split.marker,
                    gap_role=relative_split.gap_role,
                    restriction=relative_split.restriction,
                    certainty=relative_split.certainty,
                    possessed_entity_id=relative_split.possessed_entity_id,
                    diagnostics=list(relative_split.diagnostics),
                )
            )
            ctx.unresolved.extend(main_result.unresolved)
            ctx.unresolved.extend(modifier_result.unresolved)
            ctx.entities.extend(main_result.entities)
            ctx.entities.extend(modifier_result.entities)
            ctx.diagnostics.extend(main_result.diagnostics)
            ctx.diagnostics.extend(modifier_result.diagnostics)
            ctx.diagnostics.extend(relative_split.diagnostics)
            ctx.diagnostics.append(
                "entity modifier="
                f"{relative_split.restriction.value} "
                f"marker={relative_split.marker} "
                f"gap={relative_split.gap_role.value}"
            )
            return True
        ctx.diagnostics.extend(main_result.diagnostics)
        ctx.diagnostics.extend(modifier_result.diagnostics)
        ctx.diagnostics.append(
            f"relative split fallback marker={relative_split.marker}"
        )
    return False
