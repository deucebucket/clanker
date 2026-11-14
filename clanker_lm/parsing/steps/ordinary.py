"""Ordinary assembly stage. Returns whether the clause has been handled."""

from __future__ import annotations
import copy
from typing import Optional, Sequence
from ... import lexicon
from ..ports import ParserPort
from ..context import ParseContext

def apply(parser: ParserPort, clause: Sequence[lexicon.Token],
          connector: Optional[str], ctx: ParseContext) -> bool:
    """Handle this construction or return False to continue the ordered pipeline."""
    unresolved_pronoun_possible = any(
        token.norm in lexicon.PRONOUN_FEATURES
        and lexicon.PRONOUN_FEATURES[token.norm][0] is None
        for token in clause
    )
    ordinary_checkpoint = (
        copy.deepcopy(ctx.memory.__dict__)
        if unresolved_pronoun_possible
        else None
    )
    result = parser._parse_clause(clause, ctx.raw, ctx.memory)
    if (
        result.event is not None
        and result.unresolved
        and not parser._clause_has_resolved_subject(
            clause,
            result.event,
        )
    ):
        if ordinary_checkpoint is not None:
            parser._restore_memory_checkpoint(
                ctx.memory,
                ordinary_checkpoint,
            )
        result.event = None
        result.entities = []
        result.diagnostics.append(
            "event with unresolved grammatical subject suppressed"
        )
    if result.event:
        result.event.discourse_role = (
            "main" if ctx.primary_event_count == 0 else "coordinate"
        )
        if connector is not None:
            result.diagnostics.insert(0, f"coordinate connector={connector}")
        ctx.primary_event_count += 1
        ctx.events.append(result.event)
    ctx.unresolved.extend(result.unresolved)
    ctx.entities.extend(result.entities)
    ctx.diagnostics.extend(result.diagnostics)
    return True
