"""
V9 Decomposer — Event Nucleus Detection + Role Assignment.

Core V9 innovation: find the emotional center of mass first, then assign
everything else relative to it. This makes decomposition order-independent —
"Laid off from work, I just got" and "I just got laid off from work" produce
the same nucleus.

Pipeline:
  Pass 1 — Map each token to a Root; compute gravity.
  Pass 2 — Find nucleus: highest-gravity atom (ties → later position wins).
  Pass 3 — Assign roles relative to nucleus:
              subject   = first atom with a subject category
              operators = atoms with operator categories
              context   = remaining charged atoms (not filler/formulaic/connector/temporal)
  Fallback — If no subject found, use implicit SELF atom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine_v9.roots import Root, RootCategory, ROOTS
from engine_v9.root_map import map_to_root
from engine_v9.tokenizer import tokenize

# ---------------------------------------------------------------------------
# Role category sets
# ---------------------------------------------------------------------------

_SUBJECT_CATEGORIES: frozenset[RootCategory] = frozenset({
    RootCategory.SELF_REF,
    RootCategory.OTHER_REF,
    RootCategory.RELATION_REF,
    RootCategory.PERSON,
})

_OPERATOR_CATEGORIES: frozenset[RootCategory] = frozenset({
    RootCategory.NEGATOR,
    RootCategory.INTENSIFIER,
    RootCategory.HEDGE,
    RootCategory.COMPRESSOR,
})

# Categories that are pure background — never appear in context
# TEMPORAL is intentionally excluded: "yesterday", "today", etc. carry
# meaningful temporal anchoring and should appear as context atoms.
_SKIP_CONTEXT_CATEGORIES: frozenset[RootCategory] = frozenset({
    RootCategory.FILLER,
    RootCategory.FORMULAIC,
    RootCategory.CONNECTOR,
    RootCategory.OBJECT,      # generic fallback — zero charge, no semantics
})

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    """A single token with its resolved root and computed gravity."""
    word: str
    root: Root
    position: int   # 0-based index in the token stream
    gravity: float  # computed by _compute_gravity()


@dataclass
class Equation:
    """Structured decomposition of a sentence into VADUGWI equation form."""
    subject: Atom                      # WHO the event belongs to
    nucleus: Atom                      # highest-gravity emotional center
    context: list[Atom] = field(default_factory=list)    # supporting charged atoms
    operators: list[Atom] = field(default_factory=list)  # modifiers (negators, intensifiers…)
    all_atoms: list[Atom] = field(default_factory=list)  # every atom in order


# ---------------------------------------------------------------------------
# Gravity computation
# ---------------------------------------------------------------------------

def _compute_gravity(root: Root) -> float:
    """
    Compute gravitational mass of a root from its charge vector.

    Primary signal: abs(dV) — valence is the strongest pull.
    Secondary: sum of abs values of the remaining 6 dimensions, weighted 0.2.
    """
    dV = abs(root.charge[0])
    secondary = sum(abs(root.charge[i]) for i in range(1, 7)) * 0.2
    return dV + secondary


# ---------------------------------------------------------------------------
# Implicit self — used when no explicit subject is found
# ---------------------------------------------------------------------------

_IMPLICIT_SELF: Atom = Atom(
    word="<implicit>",
    root=ROOTS["SELF"],
    position=-1,
    gravity=0.0,
)

# ---------------------------------------------------------------------------
# Generic nucleus fallback — for empty / all-filler input
# ---------------------------------------------------------------------------

_GENERIC_NUCLEUS: Atom = Atom(
    word="<generic>",
    root=ROOTS["OBJECT_GENERIC"],
    position=-1,
    gravity=0.0,
)

# ---------------------------------------------------------------------------
# Main decompose function
# ---------------------------------------------------------------------------

def decompose(text: str) -> Equation:
    """
    Decompose a sentence into its V9 Equation form.

    Args:
        text: Raw input sentence.

    Returns:
        Equation with subject, nucleus, operators, context, and all_atoms.
    """
    tokens = tokenize(text)

    # Empty input → minimal equation
    if not tokens:
        return Equation(
            subject=_IMPLICIT_SELF,
            nucleus=_GENERIC_NUCLEUS,
            context=[],
            operators=[],
            all_atoms=[],
        )

    # ── Pass 1: build atom list ──────────────────────────────────────────────
    all_atoms: list[Atom] = []
    for pos, token in enumerate(tokens):
        root = map_to_root(token)
        gravity = _compute_gravity(root)
        all_atoms.append(Atom(word=token, root=root, position=pos, gravity=gravity))

    # ── Pass 2: find nucleus ─────────────────────────────────────────────────
    # Highest gravity wins. Ties break to LATER position (punchline rule).
    nucleus = all_atoms[0]
    for atom in all_atoms[1:]:
        if atom.gravity >= nucleus.gravity:  # >= means later wins on tie
            nucleus = atom

    # If nucleus has zero gravity (all filler/generic), use last non-operator atom
    if nucleus.gravity == 0.0:
        candidates = [
            a for a in all_atoms
            if a.root.category not in _OPERATOR_CATEGORIES
            and a.root.category not in _SKIP_CONTEXT_CATEGORIES
        ]
        nucleus = candidates[-1] if candidates else all_atoms[-1]

    # ── Pass 3: assign roles relative to nucleus ─────────────────────────────
    subject: Optional[Atom] = None
    operators: list[Atom] = []
    context: list[Atom] = []

    for atom in all_atoms:
        if atom is nucleus:
            continue  # nucleus is its own thing

        cat = atom.root.category

        # Subject: first atom in a subject category
        if cat in _SUBJECT_CATEGORIES:
            if subject is None:
                subject = atom
            # additional subject-category atoms treated as context
            else:
                context.append(atom)
            continue

        # Operators: negators, intensifiers, hedges, compressors
        # Only apply if within 3 tokens of nucleus (council consensus)
        if cat in _OPERATOR_CATEGORIES:
            dist = abs(atom.position - nucleus.position)
            if dist <= 3:
                operators.append(atom)
            # else: distant operator, ignore (e.g., "neither" far from nucleus)
            continue

        # Skip filler, formulaic, connectors, temporal, bare objects
        if cat in _SKIP_CONTEXT_CATEGORIES:
            continue

        # Everything else is context (charged atoms that aren't the nucleus)
        context.append(atom)

    # Fallback: no explicit subject → implicit self
    if subject is None:
        subject = _IMPLICIT_SELF

    return Equation(
        subject=subject,
        nucleus=nucleus,
        operators=operators,
        context=context,
        all_atoms=all_atoms,
    )


def decompose_molecules(molecules) -> Equation:
    """Decompose from pre-bonded Molecules instead of raw text.

    Molecules carry bonded charges (potentially inverted, amplified, etc.)
    which gives more accurate gravity than raw atom charges.

    Args:
        molecules: list of Molecule objects from bond_resolve()

    Returns:
        Equation with nucleus selected from molecule charges.
    """
    if not molecules:
        return Equation(
            subject=_IMPLICIT_SELF,
            nucleus=_GENERIC_NUCLEUS,
        )

    # Convert molecules to atoms using their bonded surface charge
    all_atoms = []
    for mol in molecules:
        # Create a synthetic root with the molecule's bonded charge
        charge = tuple(mol.surface_charge)
        synth_root = Root(
            name=mol.root.name,
            category=mol.root.category,
            charge=charge,
            phase=mol.root.phase,
        )
        atom = Atom(
            word=" ".join(mol.words),
            root=synth_root,
            position=mol.position,
            gravity=mol.gravity,
        )
        all_atoms.append(atom)

    # Same logic as decompose: find nucleus, assign roles
    nucleus = all_atoms[0]
    for atom in all_atoms[1:]:
        if atom.gravity >= nucleus.gravity:
            nucleus = atom

    if nucleus.gravity == 0.0:
        candidates = [
            a for a in all_atoms
            if a.root.category not in _OPERATOR_CATEGORIES
            and a.root.category not in _SKIP_CONTEXT_CATEGORIES
        ]
        nucleus = candidates[-1] if candidates else all_atoms[-1]

    subject = None
    operators = []
    context = []

    for atom in all_atoms:
        if atom is nucleus:
            continue
        cat = atom.root.category
        if cat in _SUBJECT_CATEGORIES:
            if subject is None:
                subject = atom
            else:
                context.append(atom)
            continue
        if cat in _OPERATOR_CATEGORIES:
            dist = abs(atom.position - nucleus.position)
            if dist <= 3:
                operators.append(atom)
            continue
        if cat in _SKIP_CONTEXT_CATEGORIES:
            continue
        context.append(atom)

    if subject is None:
        subject = _IMPLICIT_SELF

    return Equation(
        subject=subject,
        nucleus=nucleus,
        operators=operators,
        context=context,
        all_atoms=all_atoms,
    )
