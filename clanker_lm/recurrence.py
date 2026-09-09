"""Typed recurrence for simple state reports, not a response phrase table.

A positive 'again' report supports the qualified state at the report's scope.
It asserts recurrence but does not fabricate a prior event, duration or cause.
Negated 'again' has distinct scope and is deliberately outside this slice.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from . import lexicon
from .model import RefKind
from .cessation import CESSATION_ROLE

RECURRENCE_ROLE = "state_recurrence"
RECURRENCE_KIND = "recurred"
RECURRENCE_MARKER = "again"
TIME_CUES = frozenset({"now", "today", "yesterday", "tomorrow", "tonight"})


@dataclass(frozen=True)
class RecurrenceScan:
    tokens: tuple[lexicon.Token, ...]
    marker: str | None = None
    error: str | None = None


def scan_recurrence(items: Sequence[lexicon.Token], predicate: str,
                    verb_index: int, *, cessation: bool = False) -> RecurrenceScan:
    """Remove exactly one licensed local modifier before NP parsing.

    Only BE/FEEL: final, final-before-deictic, after the main verb, or directly
    before FEEL. Capitalized/quoted names are not stripped as operators.
    Clause segmentation and reporting scope remain the native parser's job.
    """
    original = tuple(items)
    if predicate not in {"be", "feel"}:
        return RecurrenceScan(original)
    positions = [i for i,t in enumerate(original)
                 if t.norm == RECURRENCE_MARKER and t.text == RECURRENCE_MARKER]
    if not positions:
        return RecurrenceScan(original)
    words = [t.norm for t in original]
    if len(positions) != 1:
        return RecurrenceScan(original, error="repeated recurrence operator")
    if cessation or any(w in lexicon.NEGATORS for w in words):
        return RecurrenceScan(original, error="negated or ceased recurrence requires separate scope")
    if not 0 <= verb_index < len(original):
        return RecurrenceScan(original, error="unbound recurrence predicate")
    i = positions[0]
    if any(t.text in {'"', "'", '“', '”'} for t in original):
        return RecurrenceScan(original, error="quoted recurrence attachment is unsupported")
    allowed = {verb_index + 1, len(words) - 1}
    if words[-1] in TIME_CUES:
        allowed.add(len(words) - 2)
    if predicate == "feel":
        allowed.add(verb_index - 1)
    if i not in allowed or i == 0 or i == verb_index:
        return RecurrenceScan(original, error="unsupported recurrence attachment")
    # An operator alone is not the state's complement, and after an article
    # it may belong to an NP. Do not turn names/items into phase operators.
    rest = [t for j,t in enumerate(original) if j != i]
    shifted = verb_index - (i < verb_index)
    if (len(rest) <= shifted + 1 or (i and words[i-1] in {"the", "a", "an", "my", "your"})
            or not any(t.norm not in TIME_CUES for t in rest[shifted+1:])):
        return RecurrenceScan(original, error="recurrence has no supported state complement")
    return RecurrenceScan(tuple(rest), RECURRENCE_MARKER)


def recurrence_of(event) -> str | None:
    """Validate the operator wherever an existing frame is consumed."""
    ref = event.arguments.get(RECURRENCE_ROLE)
    if ref is None:
        return None
    if (ref.kind != RefKind.LITERAL or ref.key != RECURRENCE_KIND
            or ref.surface != RECURRENCE_MARKER or event.predicate not in {"be", "feel"}
            or not event.polarity or event.aspect != "simple"
            or CESSATION_ROLE in event.arguments):
        raise ValueError("invalid or unsupported state recurrence operator")
    return RECURRENCE_MARKER


def preserve_state_occurrence(event) -> bool:
    """Phase-sensitive state observations must retain their turn identity.

    Stable non-state facts retain the existing deduplication behavior. Same-turn
    storage remains idempotent. This does not change evidence support weights.
    """
    if event.predicate == "feel":
        return "state" in event.arguments
    if event.predicate == "be":
        attribute = event.arguments.get("attribute")
        return bool(attribute and attribute.kind == RefKind.LITERAL
                    and attribute.key == "state" and "value" in event.arguments)
    return False
