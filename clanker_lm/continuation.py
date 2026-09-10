"""Qualified state continuation and preserved fronted discourse operators.

STILL inside a positive simple BE/FEEL clause marks reported persistence, not
renewal. Fronted 'Still,' is concessive; fronted 'Again,' leaves temporal versus
speech-act repetition unresolved. Neither alone fabricates earlier events.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from . import lexicon
from .model import RefKind
from .cessation import CESSATION_ROLE
from .recurrence import RECURRENCE_ROLE
from .state_vocabulary import STATE_TERMS

CONTINUATION_ROLE = 'state_continuation'
CONTINUATION_KIND = 'continued'
DISCOURSE_ROLE = 'state_discourse'
FRONTED = {'again': 'recurrence_or_reiteration', 'still': 'concession'}
TIME_CUES = frozenset({'now','today','yesterday','tomorrow','tonight'})


@dataclass(frozen=True)
class OperatorScan:
    tokens: tuple[lexicon.Token, ...]
    marker: str | None = None
    error: str | None = None


def fronted_operator(items: Sequence[lexicon.Token]) -> OperatorScan:
    original = tuple(items)
    if len(original) < 2 or original[0].norm not in FRONTED or original[1].norm != ',':
        return OperatorScan(original)
    # The comma licenses discourse position, not a temporal interpretation.
    return OperatorScan(original[2:], original[0].norm)


def scan_continuation(items: Sequence[lexicon.Token], predicate: str,
                      verb_index: int, *, phase: bool = False) -> OperatorScan:
    original = tuple(items)
    if predicate not in {'be','feel'}:
        return OperatorScan(original)  # progressive non-state verbs retain their old path
    positions = [i for i,t in enumerate(original) if t.norm == 'still' and t.text == 'still']
    if not positions:
        return OperatorScan(original)
    # Only the reviewed affective-state inventory belongs to this operator
    # path. Presence/location (e.g. a channel check) must retain its own parser
    # and lifecycle interpretation instead of acquiring a state-only role.
    if not any(t.norm in STATE_TERMS for t in original[max(0,verb_index+1):]):
        return OperatorScan(original)
    if len(positions) != 1 or not 0 <= verb_index < len(original):
        return OperatorScan(original, error='repeated or unbound continuation operator')
    if phase or any(t.norm in lexicon.NEGATORS for t in original):
        return OperatorScan(original, error='negative or combined continuation requires separate scope')
    i = positions[0]
    allowed = {verb_index + 1}
    if predicate == 'feel' or any(t.norm in lexicon.AUXILIARIES for t in original[:verb_index]):
        allowed.add(verb_index - 1)
    # Final STILL is possibly an immobility adjective; do not equate it with
    # medial temporal STILL. Nor can a quoted/proper-name token be removed.
    if (i not in allowed or i == 0 or any(t.text in {'"',"'",'“','”'} for t in original)
            or not any(t.norm not in TIME_CUES for t in original[max(i,verb_index)+1:])):
        return OperatorScan(original, error='unsupported continuation attachment')
    remaining = tuple(t for j,t in enumerate(original) if j != i)
    return OperatorScan(remaining, 'still')


def continuation_of(event) -> str | None:
    ref = event.arguments.get(CONTINUATION_ROLE)
    if ref is None:
        return None
    if (ref.kind != RefKind.LITERAL or ref.key != CONTINUATION_KIND or ref.surface != 'still'
            or event.predicate not in {'be','feel'} or not event.polarity
            or event.aspect != 'simple' or CESSATION_ROLE in event.arguments
            or RECURRENCE_ROLE in event.arguments or DISCOURSE_ROLE in event.arguments):
        raise ValueError('invalid or unsupported state continuation')
    return 'still'


def discourse_of(event) -> str | None:
    ref = event.arguments.get(DISCOURSE_ROLE)
    if ref is None:
        return None
    if (ref.kind != RefKind.LITERAL or ref.surface not in FRONTED
            or ref.key != FRONTED[ref.surface] or event.predicate not in {'be','feel'}
            or event.aspect != 'simple' or event.discourse_role not in {'','main','coordinate'}
            or any(k in event.arguments for k in (CESSATION_ROLE,RECURRENCE_ROLE,CONTINUATION_ROLE))):
        raise ValueError('invalid or unsupported fronted state discourse')
    return ref.surface
