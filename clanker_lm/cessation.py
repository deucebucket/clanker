"""Bounded state-cessation operator, shared by parsing and interpretation.

No completed replies are stored. The operation denies continuation of one
qualified state; it never establishes a positive opposite or a prior event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from . import lexicon

CESSATION_ROLE = "state_change"
CESSATION_KIND = "ceased"
CESSATION_MARKERS = {"no_longer", "not_anymore", "not_any_more", "not_any_longer"}


@dataclass(frozen=True)
class CessationScan:
    tokens: tuple[lexicon.Token, ...]
    marker: str | None = None
    error: str | None = None


def scan_cessation(items: Sequence[lexicon.Token], predicate: str, verb_index: int) -> CessationScan:
    """Identify a local state-phase modifier before NP/negation stripping.

    Only BE/FEEL are licensed. Clause segmentation remains the main parser's
    job. More than one negation, multiple phase markers or malformed placement
    abstains explicitly; negated conjunction is rejected by state_components.
    Tokens are removed only at licensed operator positions, never globally.
    """
    original = tuple(items)
    if predicate not in {"be", "feel"}:
        return CessationScan(original)
    words = [t.norm for t in items]
    n = len(words)
    marker, removed = None, set()
    pairs = [i for i in range(n-1) if words[i:i+2] == ["no", "longer"]]
    if pairs:
        if len(pairs) != 1:
            return CessationScan(original, error="ambiguous repeated cessation operator")
        i = pairs[0]
        # After a copula/FEEL, or immediately before the main verb.
        if i not in {verb_index+1, verb_index-2}:
            return CessationScan(original, error="unsupported cessation attachment")
        marker, removed = "no_longer", {i, i+1}
    else:
        end = n - (1 if n and words[-1] in {"now", "today", "yesterday", "tomorrow", "tonight"} else 0)
        if end and words[end-1] == "anymore":
            marker, removed = "not_anymore", {end-1}
        elif end >= 2 and words[end-2:end] == ["any", "more"]:
            marker, removed = "not_any_more", {end-2, end-1}
        else:
            for i in range(n-2):
                if words[i:i+3] == ["not", "any", "longer"]:
                    marker, removed = "not_any_longer", {i, i+1, i+2}
                    if i not in {verb_index+1, verb_index-3}:
                        return CessationScan(original, error="unsupported cessation attachment")
                    break
    if marker is None:
        return CessationScan(original)
    neg = [i for i, w in enumerate(words) if w in lexicon.NEGATORS]
    if len(neg) != 1:
        return CessationScan(original, error="cessation requires exactly one scoped negator")
    neg_index = neg[0]
    if marker == "no_longer" and neg_index not in removed:
        return CessationScan(original, error="unbound cessation negation")
    if marker != "no_longer":
        if words[neg_index] != "not" or neg_index not in {verb_index-1, verb_index+1, verb_index-3}:
            return CessationScan(original, error="unbound cessation negation")
        removed.add(neg_index)
    # Two surface phase expressions cannot collapse to one operator.
    remainder = [w for i,w in enumerate(words) if i not in removed]
    if "anymore" in remainder or "longer" in remainder or any(
            remainder[i:i+2] == ["any", "more"] for i in range(len(remainder)-1)):
        return CessationScan(original, error="multiple or unsupported cessation qualifiers")
    return CessationScan(tuple(t for i,t in enumerate(items) if i not in removed), marker)


def marker_of(event) -> str | None:
    ref = event.arguments.get(CESSATION_ROLE)
    if ref is None:
        return None
    if ref.kind.value != "literal" or ref.key != CESSATION_KIND or ref.surface not in CESSATION_MARKERS:
        raise ValueError("invalid state cessation operator")
    if event.predicate not in {"be", "feel"} or event.polarity or event.aspect != "simple":
        raise ValueError("cessation requires a negative simple state frame")
    return ref.surface


def withdrawal_matches(denied: str, asserted: str) -> bool:
    """A broad denial withdraws qualified occurrences, not the reverse.

    'No longer very angry' does not entail 'not angry'. Exact qualified denial
    can retire its matching qualified report. This is syntactic subsumption,
    not an unmeasured numerical scale of adjective intensity.
    """
    a, b = denied.split(), asserted.split()
    # ANYMORE is a phase marker, not part of the denied adjective predicate.
    if a and a[-1] == "anymore":
        a = a[:-1]
    if b and b[-1] == "anymore":
        b = b[:-1]
    return bool(a and b and (a == b or len(a) == 1 and a[0] == b[-1]))
