"""V9 Molecular Bonding Layer — word interaction through bond sites.

Words are atoms with bonding surfaces. Adjacent atoms react based on
compatible bond sites, producing molecules with emergent charge.

Pipeline position: tokenize → root_map → **bond_resolve** → decompose → compose

Bond types (8 fundamental interaction primitives):
  AMPLIFIER   — scales partner's charge magnitude ("very", "fucking")
  INVERTER    — flips partner's charge sign ("not", "never")
  DIRECTIONAL — creates force vector toward/away ("off", "out", "at")
  EVALUATOR   — applies judgment to entity ("good", "terrible")
  ANCHOR      — locks to reference frame: time/possession/cause ("just", "my", "because")
  DOMAIN      — sets context field: financial/medical/etc. ("cost", "hospital")
  BINDER      — merges atoms into compound unit ("of", articles)
  PRAGMATIC   — meta-language: enables conditional flip under contradiction ("love" in sarcasm)

Each Root has bond_sites — a tuple of BondSite descriptors.
Bond compatibility is checked via bitmask AND (one CPU cycle).

Two-layer output: Molecule preserves surface_charge AND context_charge
separately. Post-composition contradiction detection uses the gap
between layers to detect sarcasm/irony.
"""

from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import List, Tuple, Optional

from .roots import Root, ROOTS, RootCategory
from .root_map import map_to_root
from .tokenizer import tokenize


# ── Bond type bitmask ────────────────────────────────────────────

class BondType(IntFlag):
    """Bitmask bond types — use bitwise AND for O(1) compatibility check."""
    AMPLIFIER   = auto()   # 0x01
    INVERTER    = auto()   # 0x02
    DIRECTIONAL = auto()   # 0x04
    EVALUATOR   = auto()   # 0x08
    ANCHOR      = auto()   # 0x10
    DOMAIN      = auto()   # 0x20
    BINDER      = auto()   # 0x40
    PRAGMATIC   = auto()   # 0x80


# ── Bond site descriptor ────────────────────────────────────────

@dataclass(frozen=True)
class BondSite:
    """One interaction site on a Root's bonding surface."""
    type: BondType
    direction: int = 1     # -1=LEFT (modifies left neighbor), +1=RIGHT, 0=BOTH
    strength: int = 1      # 1-3, bond eagerness


# ── Molecule (two-layer charge) ──────────────────────────────────

@dataclass
class Molecule:
    """Bonded atom or compound with two charge layers.

    surface_charge: what the words SAY (literal meaning)
    context_charge: what the bond context IMPLIES
    When these diverge → sarcasm/irony/masking.
    """
    words: List[str]
    root: Root                           # primary root (highest gravity in the molecule)
    surface_charge: List[int]            # 7D — from the primary evaluative word
    context_charge: List[int]            # 7D — from bonded context
    gravity: float = 0.0
    flags: set = field(default_factory=set)  # "sarcasm", "masking", "formulaic"
    position: int = 0                    # position in sentence


# ── Reaction table (8×8 = 64 entries) ────────────────────────────
# Left bond type × Right bond type → reaction function
# Only populated for compatible reactions. Missing entries = no bond.

def _react_amplifier_right(left_charge, right_charge, left_strength):
    """AMPLIFIER on left amplifies right's charge."""
    factor = 1.0 + left_strength * 0.5  # strength 1→1.5x, 2→2.0x, 3→2.5x
    return [int(c * factor) for c in right_charge]

def _react_inverter_right(left_charge, right_charge, left_strength):
    """INVERTER on left flips AND amplifies right's charge.

    Negation in language is stronger than a simple sign flip.
    "Not happy" is MORE than "mildly unhappy" — it's a deliberate
    contradiction that carries emphasis. Factor of 1.5 matches the
    linguistic weight of negation.
    """
    factor = -1.0 * (1.0 + left_strength * 0.15)  # strength 2 → -1.3x
    return [int(c * factor) for c in right_charge]

def _react_directional_merge(left_charge, right_charge, left_strength):
    """DIRECTIONAL bonds create force vectors — merge charges with sign alignment."""
    return [l + r for l, r in zip(left_charge, right_charge)]

def _react_evaluator_domain(left_charge, right_charge, left_strength):
    """EVALUATOR + DOMAIN — evaluation in domain context. Both charges preserved."""
    return None  # signals two-layer preservation

def _react_evaluator_evaluator(left_charge, right_charge, left_strength):
    """Two evaluators — compound evaluation. Sum charges."""
    return [l + r for l, r in zip(left_charge, right_charge)]

def _react_pragmatic_context(left_charge, right_charge, left_strength):
    """PRAGMATIC + negative context — enable sarcasm detection."""
    return None  # signals two-layer preservation with sarcasm potential


# Reaction lookup: (left_type, right_type) → reaction_fn
# None value = preserve both charges as separate layers (two-layer molecule)
REACTIONS = {
    # AMPLIFIER acts on anything to its right
    (BondType.AMPLIFIER, BondType.EVALUATOR):   _react_amplifier_right,
    (BondType.AMPLIFIER, BondType.DOMAIN):       _react_amplifier_right,
    (BondType.AMPLIFIER, BondType.PRAGMATIC):    _react_amplifier_right,

    # INVERTER flips anything to its right
    (BondType.INVERTER, BondType.EVALUATOR):     _react_inverter_right,
    (BondType.INVERTER, BondType.DOMAIN):         _react_inverter_right,
    (BondType.INVERTER, BondType.AMPLIFIER):      _react_inverter_right,
    (BondType.INVERTER, BondType.PRAGMATIC):      _react_inverter_right,

    # DIRECTIONAL bonds (merge into force vector)
    (BondType.DIRECTIONAL, BondType.DIRECTIONAL): _react_directional_merge,
    (BondType.EVALUATOR, BondType.DIRECTIONAL):   _react_directional_merge,

    # AMPLIFIER + DIRECTIONAL = aggressive command ("fuck off", "get out")
    (BondType.AMPLIFIER, BondType.DIRECTIONAL):   _react_directional_merge,

    # EVALUATOR + DOMAIN = evaluation in context (two-layer)
    (BondType.EVALUATOR, BondType.DOMAIN):        _react_evaluator_domain,
    (BondType.DOMAIN, BondType.EVALUATOR):         _react_evaluator_domain,

    # EVALUATOR + EVALUATOR = compound
    (BondType.EVALUATOR, BondType.EVALUATOR):     _react_evaluator_evaluator,

    # PRAGMATIC + DOMAIN/EVALUATOR = sarcasm potential (two-layer)
    (BondType.PRAGMATIC, BondType.DOMAIN):         _react_pragmatic_context,
    (BondType.PRAGMATIC, BondType.EVALUATOR):      _react_pragmatic_context,
    (BondType.EVALUATOR, BondType.PRAGMATIC):      _react_pragmatic_context,
}


# ── Root bond site assignments ───────────────────────────────────
# These map RootCategory → default bond sites.
# Individual roots can override via ROOT_BOND_OVERRIDES.

_AMP = BondSite(BondType.AMPLIFIER, direction=1, strength=2)
_INV = BondSite(BondType.INVERTER, direction=1, strength=2)
_DIR = BondSite(BondType.DIRECTIONAL, direction=0, strength=1)
_EVA = BondSite(BondType.EVALUATOR, direction=0, strength=1)
_EVA_STRONG = BondSite(BondType.EVALUATOR, direction=0, strength=2)
_ANC = BondSite(BondType.ANCHOR, direction=1, strength=1)
_DOM = BondSite(BondType.DOMAIN, direction=0, strength=1)
_BND = BondSite(BondType.BINDER, direction=0, strength=1)
_PRA = BondSite(BondType.PRAGMATIC, direction=0, strength=1)

CATEGORY_BONDS = {
    # Emotional states — EVALUATOR (they emit judgment)
    RootCategory.POSITIVE_STATE:    (_EVA_STRONG, _PRA),
    RootCategory.NEGATIVE_STATE:    (_EVA_STRONG,),
    RootCategory.POSITIVE_QUALITY:  (_EVA,),
    RootCategory.NEGATIVE_QUALITY:  (_EVA,),
    RootCategory.POSITIVE_EVENT:    (_EVA_STRONG,),
    RootCategory.NEGATIVE_EVENT:    (_EVA_STRONG,),
    RootCategory.SOCIAL_EVAL_POS:   (_EVA,),
    RootCategory.SOCIAL_EVAL_NEG:   (_EVA,),
    RootCategory.EXPECTATION_VIOLATION: (_EVA, _PRA),

    # Domain categories — DOMAIN + EVALUATOR
    RootCategory.FINANCIAL_POS:     (_DOM, _EVA),
    RootCategory.FINANCIAL_NEG:     (_DOM, _EVA),
    RootCategory.MEDICAL_POS:       (_DOM, _EVA),
    RootCategory.MEDICAL_NEG:       (_DOM, _EVA),
    RootCategory.ACADEMIC_POS:      (_DOM, _EVA),
    RootCategory.ACADEMIC_NEG:      (_DOM, _EVA),
    RootCategory.LEGAL_POS:         (_DOM, _EVA),
    RootCategory.LEGAL_NEG:         (_DOM, _EVA),
    RootCategory.CAREER_POS:        (_DOM, _EVA),
    RootCategory.CAREER_NEG:        (_DOM, _EVA),
    RootCategory.RELATIONSHIP_POS:  (_DOM, _EVA),
    RootCategory.RELATIONSHIP_NEG:  (_DOM, _EVA),

    # Actions — DIRECTIONAL
    RootCategory.MOTION:        (_DIR,),
    RootCategory.TRANSFER:      (_DIR,),
    RootCategory.POSSESSION:    (_BND,),
    RootCategory.PERCEPTION:    (_DIR,),
    RootCategory.COMMUNICATION: (_DIR,),
    RootCategory.COGNITION:     (_BND,),

    # Entities — BINDER (they receive bonds)
    RootCategory.SELF_REF:      (_BND,),
    RootCategory.OTHER_REF:     (_BND,),
    RootCategory.RELATION_REF:  (_BND,),
    RootCategory.PERSON:        (_BND,),
    RootCategory.OBJECT:        (),

    # Operators
    RootCategory.INTENSIFIER:   (_AMP,),
    RootCategory.NEGATOR:       (_INV,),
    RootCategory.CONNECTOR:     (),
    RootCategory.HEDGE:         (BondSite(BondType.AMPLIFIER, direction=1, strength=-1),),  # negative amplification = dampening
    RootCategory.COMPRESSOR:    (BondSite(BondType.AMPLIFIER, direction=1, strength=-1),),
    RootCategory.TEMPORAL:      (_ANC,),

    # Special
    RootCategory.FORMULAIC:     (_PRA,),
    RootCategory.COMPOUND_EVENT:(_EVA_STRONG, _DOM),
    RootCategory.FILLER:        (),
}


def _get_bond_sites(root: Root) -> Tuple[BondSite, ...]:
    """Get bond sites for a root — category default with optional override."""
    return CATEGORY_BONDS.get(root.category, ())


def _compute_gravity(charge):
    """Gravity from charge vector — for molecule ranking."""
    v_mag = abs(charge[0])
    other = sum(abs(x) for x in charge[1:]) * 0.2
    return v_mag + other


# ── Sarcasm contradiction threshold ──────────────────────────────

SARCASM_THRESHOLD = 30  # surface vs context divergence needed to trigger


# ── Main bonding pass ────────────────────────────────────────────

def bond_resolve(text: str) -> List[Molecule]:
    """Run the bonding pass on text → list of Molecules.

    Single left-to-right scan, O(n).
    Adjacent atoms with compatible bond sites react.
    Output: list of Molecules (some are single atoms, some are fused compounds).
    """
    tokens = tokenize(text)
    if not tokens:
        return []

    # Map tokens to roots
    atoms = []
    for i, token in enumerate(tokens):
        root = map_to_root(token)
        atoms.append({
            "word": token,
            "root": root,
            "charge": list(root.charge),
            "sites": _get_bond_sites(root),
            "position": i,
            "bonded": False,
        })

    # Left-to-right bonding pass with window=3
    # For each atom, try to bond with right neighbors within window.
    # Closest compatible match wins. Still O(n) — fixed window.
    BOND_WINDOW = 3
    molecules = []
    i = 0
    while i < len(atoms):
        left = atoms[i]

        if not left["bonded"]:
            # Try to bond with nearest compatible right neighbor (window=3)
            bond_result = None
            bond_partner_idx = None
            for offset in range(1, min(BOND_WINDOW + 1, len(atoms) - i)):
                right = atoms[i + offset]
                if right["bonded"]:
                    continue
                result = _try_bond(left, right)
                if result is not None:
                    bond_result = result
                    bond_partner_idx = i + offset
                    break  # closest match wins

            if bond_result is not None:
                molecules.append(bond_result)
                left["bonded"] = True
                atoms[bond_partner_idx]["bonded"] = True
                # Don't skip to bond_partner_idx — atoms between might still bond elsewhere
                i += 1
                continue

        # No bond — emit as single-atom molecule
        if not left["bonded"]:
            mol = Molecule(
                words=[left["word"]],
                root=left["root"],
                surface_charge=list(left["charge"]),
                context_charge=[0] * 7,
                gravity=_compute_gravity(left["charge"]),
                position=left["position"],
            )
            molecules.append(mol)

        i += 1

    # Post-bond sarcasm detection
    _detect_sarcasm(molecules)

    return molecules


def _try_bond(left: dict, right: dict) -> Optional[Molecule]:
    """Try to bond two adjacent atoms. Returns Molecule if compatible, None otherwise."""
    left_sites = left["sites"]
    right_sites = right["sites"]

    if not left_sites or not right_sites:
        return None

    # Check all site combinations for a reaction
    for ls in left_sites:
        for rs in right_sites:
            # Direction check: left site must point right (or both), right site must accept from left
            if ls.direction == -1:  # left site points left, can't bond rightward
                continue
            if rs.direction == 1:   # right site points right, can't accept from left
                continue

            reaction = REACTIONS.get((ls.type, rs.type))
            if reaction is not None:
                result_charge = reaction(left["charge"], right["charge"], ls.strength)

                if result_charge is None:
                    # Two-layer molecule — preserve both charges separately
                    # Surface = the evaluative word, context = the domain/context word
                    surface = left["charge"] if ls.type == BondType.EVALUATOR or ls.type == BondType.PRAGMATIC else right["charge"]
                    context = right["charge"] if surface is left["charge"] else left["charge"]
                    mol = Molecule(
                        words=[left["word"], right["word"]],
                        root=left["root"] if _compute_gravity(left["charge"]) >= _compute_gravity(right["charge"]) else right["root"],
                        surface_charge=list(surface),
                        context_charge=list(context),
                        gravity=max(_compute_gravity(left["charge"]), _compute_gravity(right["charge"])),
                        position=left["position"],
                    )
                    return mol
                else:
                    # Merged molecule — single charge
                    # Use the higher-gravity root as the molecule's root
                    primary = left if _compute_gravity(left["charge"]) >= _compute_gravity(right["charge"]) else right
                    mol = Molecule(
                        words=[left["word"], right["word"]],
                        root=primary["root"],
                        surface_charge=result_charge,
                        context_charge=[0] * 7,
                        gravity=_compute_gravity(result_charge),
                        position=left["position"],
                    )
                    return mol

    return None


def _detect_sarcasm(molecules: List[Molecule]):
    """Post-bond sarcasm detection — flag molecules with charge contradiction.

    When surface_charge is positive and context_charge is negative (or vice versa)
    beyond the threshold, flag as sarcasm and invert the surface charge.
    """
    for mol in molecules:
        if not any(mol.context_charge):
            continue  # no context layer, can't be sarcasm

        surface_v = mol.surface_charge[0]
        context_v = mol.context_charge[0]

        # Positive surface + negative context = sarcasm candidate
        if surface_v > 0 and context_v < 0:
            divergence = surface_v - context_v  # both contribute to gap
            if divergence > SARCASM_THRESHOLD:
                mol.flags.add("sarcasm")
                # Invert surface charge
                mol.surface_charge = [-c for c in mol.surface_charge]
                mol.gravity = _compute_gravity(mol.surface_charge)

        # Negative surface + positive context = also possible (ironic complaint about good thing)
        elif surface_v < 0 and context_v > 0:
            divergence = context_v - surface_v
            if divergence > SARCASM_THRESHOLD:
                mol.flags.add("irony")
                mol.surface_charge = [-c for c in mol.surface_charge]
                mol.gravity = _compute_gravity(mol.surface_charge)
