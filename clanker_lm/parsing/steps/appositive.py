"""Appositive assembly stage. Returns whether the clause has been handled."""

from __future__ import annotations
from typing import Optional, Sequence
from ... import lexicon
from ...model import UnresolvedReference
from ..ports import ParserPort
from ..context import ParseContext

def apply(parser: ParserPort, clause: Sequence[lexicon.Token],
          connector: Optional[str], ctx: ParseContext) -> bool:
    """Handle this construction or return False to continue the ordered pipeline."""
    appositive_split, appositive_ambiguity = parser._split_appositive_clause(
        clause,
        ctx.raw,
        ctx.memory,
    )
    if appositive_ambiguity is not None:
        ctx.appositive_ambiguities.append(appositive_ambiguity)
        ctx.unresolved.append(
            UnresolvedReference(
                surface=appositive_ambiguity.appositive_surface,
                reason=appositive_ambiguity.reason,
                candidates=list(appositive_ambiguity.candidate_entity_ids),
            )
        )
        ctx.diagnostics.extend(appositive_ambiguity.diagnostics)
        ctx.diagnostics.append("appositive identity remains ambiguous")
        return True

    if appositive_split is not None:
        result = parser._parse_clause(
            appositive_split.main_tokens,
            ctx.raw,
            ctx.memory,
        )
        if result.event:
            result.event.discourse_role = (
                "main" if ctx.primary_event_count == 0 else "coordinate"
            )
            if connector is not None:
                result.diagnostics.insert(
                    0,
                    f"coordinate connector={connector}",
                )
            ctx.primary_event_count += 1
            ctx.events.append(result.event)
            ctx.appositives.append(appositive_split.relation)
            ctx.unresolved.extend(result.unresolved)
            ctx.entities.extend(appositive_split.entity_ids)
            ctx.entities.extend(result.entities)
            ctx.diagnostics.extend(result.diagnostics)
            ctx.diagnostics.extend(appositive_split.diagnostics)
            ctx.diagnostics.append(
                "appositive relation="
                f"{appositive_split.relation.relation_type.value} "
                f"restriction={appositive_split.relation.restriction.value}"
            )
            return True
        ctx.diagnostics.extend(result.diagnostics)
        ctx.diagnostics.append("appositive split fallback")
    return False
