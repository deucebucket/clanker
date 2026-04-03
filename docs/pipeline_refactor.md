# Pendulum Pipeline Refactor

## Current Problem
`pendulum.py` is a 500-line monolith. One function does everything.
Can't test stages independently, can't swap components, can't bypass stages.

## Proposed Architecture

```
text
  → Stage 1: TOKENIZE (split, clean, compound resolution)
  → Stage 2: CLASSIFY (word roles from word_classifier.py)
  → Stage 3: PROXIMITY (compute coefficients from proximity.py)
  → Stage 4: FORCES (accumulate word forces with coefficients)
  → Stage 5: BLEND (adaptive momentum, apply forces to state)
  → Stage 6: STRUCTURES (detect patterns, apply adjustments)
  → Stage 7: ANOMALY (trajectory analysis from anomaly.py)
  → Stage 8: SATURATION (tanh compression)
  → Stage 9: CLAMP (0-255 final output)
  → VADUGWI
```

## Design Principles

1. Each stage is a FUNCTION that takes input and returns output
2. Each stage is OPTIONAL -- if missing, pipeline skips it
3. Stages are composable: `pipeline = [tokenize, classify, proximity, forces, blend]`
4. Testing: run any subset of stages
5. Swapping: replace `blend` with `impulse_blend` without touching other stages
6. Bypass: `pipeline = [tokenize, classify, forces, blend]` (no proximity)

## Implementation Plan

1. Extract each stage from compute_vadug() into its own function
2. Create a Pipeline class that chains stages
3. compute_vadug() becomes a thin wrapper: `Pipeline.default().run(text)`
4. Each stage function has a clear signature:
   - tokenize(text) → words
   - classify(words) → roles
   - proximity(roles) → coefficients
   - forces(roles, coefficients) → accumulated_evidence
   - blend(state, evidence) → new_state
   - structures(roles, state) → adjustments
   - anomaly(state, history) → anomaly_flags
   - saturate(state) → compressed_state
   - clamp(state) → VADUG

## When to Do This
After the anomaly detector is ported and tested.
This is a refactor -- no new features, just better organization.
Test before and after: all benchmarks must hold.
