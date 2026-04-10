# Council Prompt: V9 Word Bonding Surface Layer

## Context

Clanker V9 is an equation decomposition engine that computes 7D emotional coordinates (VADUGWI) from text. Currently each word maps to a Root with a category and a 7D charge vector. The equation decomposer finds the highest-gravity atom (event nucleus), assigns roles (SUBJECT, CONTEXT, OPERATORS), and the composer resolves the final state.

**What's working:** Domain-specific categories (FINANCIAL_POS, MEDICAL_NEG, etc.), compound bond tables, phase system (SOLID/LIQUID/GAS), structure detection (45+ patterns).

**What's NOT working:** The engine treats words as isolated charges that add up. It doesn't model how words INTERACT with each other. "Fuck" + "off" = rejection, but neither word alone carries that meaning. "Good" + "morning" = formulaic neutral, but "good" alone is positive. The compound bond table handles known pairs, but can't generalize to new combinations.

## The Proposal: Molecular Bonding for Words

Instead of treating words as point charges, treat them as atoms with **bonding surfaces** — a set of interaction sites that define HOW each word can react with neighboring words. The emotional meaning of a phrase emerges from the BONDS between words, not from summing individual charges.

### Chemistry Analogy

- Carbon has 4 bonding sites. Hydrogen has 1. You don't need a lookup table for CH4 — the bonding rules produce methane from the atoms' properties.
- Similarly, "fuck" has an INTENSIFIER site and a DIRECTIONAL site. "Off" has a SEPARATION site and a DIRECTIONAL site. When their DIRECTIONAL sites bond, the molecule "fuck off" = aggressive separation = rejection. Emergent, not listed.

### Reactions We Already Detect (implicitly)

| Reaction | Chemistry | Current V9 Implementation |
|----------|-----------|--------------------------|
| Neutralization | acid + base = water | "good morning" → FORMULAIC (hardcoded) |
| Amplification | catalyst | INTENSIFIER × charge = amplified charge |
| Inversion | sign flip | NEGATOR × charge = inverted charge |
| Dissolution | solvent | SOLVENT dissolves LIQUID phase atoms |
| Crystallization | two gases → solid | Compound bond table ("laid off" → SOLID event) |
| Precipitation | solution context | "I'm fine" after crisis = MASKING (anomaly detector) |

### What's Missing

Each word currently has: `name, category, charge[7], phase`

Each word NEEDS: `bond_sites[]` — a set of interaction descriptors that define what this word can react with and how.

## Questions for the Council

As a physicist consultant, please address:

### 1. Bond Site Types

What are the fundamental interaction types? I'm estimating 10-15. Candidates:

- INTENSIFIER_SITE (amplifies bonded partner's charge)
- INVERTER_SITE (flips bonded partner's sign)
- DIRECTIONAL_SITE (creates force vector when bonded to direction word)
- SEPARATION_SITE (disconnection, removal, away-from)
- EVALUATION_SITE (judgment — bonds with person-refs to create social evaluation)
- FORMULAIC_SITE (neutralizes in greeting/ritual context)
- REGISTER_SITE (marks the sentence register — casual, formal, literary)
- TEMPORAL_SITE (anchors to time — "just", "always", "never")
- POSSESSION_SITE (ownership/control bond)
- CAUSATION_SITE (cause-effect link)

Are these the right types? What's missing? What's redundant?

### 2. Bond Resolution Rules

When two atoms are adjacent and share a compatible bond site, what happens?

- Does the bond produce a NEW charge vector (emergent molecule)?
- Does it modify one atom's charge based on the other's properties?
- Is it directional (left atom acts on right, or right on left)?
- Can an atom bond with multiple neighbors simultaneously?
- What's the bond strength? Does distance matter (proximity decay)?

### 3. Data Structure

What should the Root dataclass look like with bonding?

Current:
```python
@dataclass(frozen=True)
class Root:
    name: str
    category: RootCategory
    charge: Tuple[int, int, int, int, int, int, int]  # VADUGWI deltas
    phase: str = "GAS"
```

Proposed (needs your input):
```python
@dataclass(frozen=True)
class Root:
    name: str
    category: RootCategory
    charge: Tuple[int, int, int, int, int, int, int]
    phase: str = "GAS"
    bond_sites: Tuple[BondSite, ...] = ()  # interaction surface
```

What does BondSite look like? How is it encoded?

### 4. Integration with Equation Decomposition

V9 currently runs: tokenize → root_map → decompose (find nucleus) → compose (sum charges)

Where does bonding resolve in this pipeline?

Option A: Bonding resolves BEFORE decomposition — adjacent atoms react, producing molecular atoms, then the decomposer finds the nucleus among molecules.

Option B: Bonding resolves DURING composition — the composer knows about bonds and adjusts charges based on neighbor interactions.

Option C: Bonding IS the composition — the entire sentence is resolved through sequential bond reactions, no separate "sum charges" step.

### 5. Sarcasm as Chemistry

The immediate test case: "I love paying fifty dollars for a tiny plate of lettuce."

- "love" = POSITIVE_STATE, strong positive charge
- "paying fifty dollars" = FINANCIAL_NEG context
- "tiny plate of lettuce" = negative evaluation of value

How do the bonding rules produce the sarcastic inversion? What bond sites on "love" react with the financial/evaluative context to flip the sign?

### 6. Scale and Feasibility

- How many bond sites per word on average? (1-3? 5-10?)
- Is this O(n) per sentence or does it require O(n²) pairwise checks?
- Can this work with the existing ~500 roots, or does it require per-word bond site definitions for all 4,500 vocabulary words?

## Current V9 Stats (for reference)

- 18 modules, 45+ structure patterns, 41 root categories, ~2,500 mapped words
- Verified set: 60.5% accuracy (V8: 78%)
- Novel conversational: 54.2% accuracy
- Sarcasm: 0% detection (27 inversions missed on novel set)
- Domain categories added: Financial, Medical, Academic, Legal, Career, Relationship
- Phase system: SOLID (never flips) / LIQUID (context flips) / GAS (neutral default)
- Equation decomposition is order-independent (Yoda test passes)
- Running in PMS on Bonsai-8B (1-bit model) — VADUGWI conditioning works

## Constraints

- Must stay O(n) per sentence, <1ms
- Must be deterministic (no ML in the bond resolver)
- Must work with the existing root system (extend, don't replace)
- Bond sites should be definable at the root level (~500 roots), not per-word (4,500+ words)
- The engine will eventually be compiled to a binary — keep data structures simple
