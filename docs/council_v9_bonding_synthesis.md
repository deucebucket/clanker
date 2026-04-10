# Council V9 Bonding Synthesis — 4-LLM Consensus

**Date:** 2026-04-09
**Council:** Gemini, Grok, Claude, GPT
**Topic:** Molecular bonding surface layer for V9 equation decomposition

## Chosen Approach

Grok's 9×9 reaction table + Claude's two-layer molecule (surface vs context charge) + GPT's "store the ability to flip" principle.

- Bonding pass runs BEFORE decomposition (Option A — unanimous)
- 8 bond types, bitmask compatibility, O(n) single pass
- Molecules preserve surface charge AND context charge separately
- Sarcasm detection = post-composition contradiction detector on the two layers
- 500 roots carry bond sites, words inherit

## Alternative Approaches (preserved for future testing)

### Gemini's Approach
- Valence-1/2/Aromatic classification of bond types
- Bitmask encoding for binary compilation
- QUANTIFIER_SITE for magnitude scaling ("fifty dollars")
- Sarcasm as phase transition (crystallization from bond mismatch)
- Full doc: `docs/council_v9_bonding.md` (prompt) + Gemini response

### Grok's Approach
- 9 bond types with CONTRAST as sarcasm trigger
- 9×9 reaction table (81 entries, constant-time lookup)
- Mutate working copies, don't create new Roots mid-sentence
- molecule_flags (sarcasm, masking) instead of new objects
- Direction: OUT/IN/BOTH on each site
- Full emergent sarcasm from bonding rules alone

### Claude's Approach
- 8 bond types with MASK (suppresses charge to 0.3, anomaly detector sees underneath)
- Molecule dataclass with surface_charge AND context_charge
- Sarcasm is NOT a bond reaction — it's a charge contradiction detector
- Post-composition check: if surface > 0 and context < 0 and divergence > threshold → flip
- Direction: LEFT/RIGHT/BOTH on each site
- Start with 50 highest-frequency roots, iterate

### GPT's Approach
- 8 types: SCALAR, POLARITY, VECTOR, BINDING, EVALUATION, CONTEXT FIELD, TEMPORAL, PRAGMATIC
- PRAGMATIC site = meta-language intent (sarcasm, masking, formulaic)
- "Store the ability to flip under contradiction, not the flip itself"
- Sarcasm = field-level contradiction between expectation and context
- Molecule formation only for BINDING + VECTOR alignment or high-cohesion patterns
- "8 levers that combine into 10,000 behaviors"

## Full Consensus Points

1. Option A — bonding before decomposition (4/4)
2. ~8 bond types, not 15 (4/4)
3. Bitmask compatibility for constant-time matching (4/4)
4. 500 roots carry bond sites (4/4)
5. Flat data structures for binary compilation (4/4)
6. O(n) single pass, left-to-right (4/4)
7. Sarcasm involves positive-negative charge mismatch (4/4)

## Disagreements

| Topic | Options | Chosen |
|-------|---------|--------|
| Sarcasm mechanism | Emergent (Grok/Gemini) vs Detector (Claude) vs Pragmatic (GPT) | Hybrid: reaction table + detector |
| Bond output | Modify in place (Grok) vs New molecule (Claude) vs Layered (GPT) | Two-layer molecule |
| REGISTER as bond type | Yes (Claude/Gemini) vs No, sentence-level (Grok) vs Not yet needed (GPT) | Not a bond type — sentence-level property |
| Number of types | 9 (Grok) vs 8 (Claude/GPT) vs Valence-grouped (Gemini) | 8 |
