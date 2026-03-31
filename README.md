# Clanker

A conversation state resolver that computes 7-dimensional emotional coordinates from text using structural pattern recognition. It reads text the way a chess player reads a board -- recognizing patterns from piece positions, not memorizing specific games.

"Whatever" alone reads as resignation. "Whatever makes you happy" reads as passive-aggressive. "Do whatever" reads as permission. Same word -- context changes the Dominance dimension. A sentiment classifier says "neutral" for all three.

~300KB engine. 0.15ms per sentence. 167 tests. Fully deterministic and auditable.

## VADUGWI Coordinates

Seven dimensions, each 0--255 with 128 as neutral center (Urgency starts at 0):

| Dim | Low (0) | Center (128) | High (255) | Measures |
|-----|---------|--------------|------------|----------|
| **V** Valence | Strongly negative | Neutral | Strongly positive | Emotional direction |
| **A** Arousal | Very calm | Moderate | Very intense | Energy level |
| **D** Dominance | Helpless | Balanced | In full control | Agency and power |
| **U** Urgency | None | Moderate | Critical | Time pressure |
| **G** Gravity | Crushing weight | Grounded | Light, floating | Emotional weight |
| **W** Self-Worth | Shattered | Stable | Strong | Self-evaluation |
| **I** Intent | Withdraw | Deflect/Neutral | Connect/Control | Communicative direction |

7 bytes encode 72 quadrillion possible emotional states.

## What the Engine Reads

| Input | V | Notes |
|-------|---|-------|
| "I'm fine" | 118 | Below neutral -- uneasy, not positive |
| "haha yeah im totally okay" | 93 | Forced composure, low valence despite positive words |
| "oh joy" | 113 | Positive word, negative reading |
| "do you even love me" | 120 | Positive word weaponized as challenge |
| "my wife cheated on me with my best friend" | 49 | V=49, A=170, D=56 -- deep negative, high intensity, low control |
| "I love my mom" | 179 | Genuine positive, no false alarm |
| "the meeting is at three" | 128 | Neutral -- no emotional content detected |

## How It Works

Four processing layers run in sequence:

1. **Word Classification** -- each word is assigned one of 23 structural roles (SELF_REF, EMOTIONAL, NEGATOR, AMPLIFIER, CONNECTOR, CHOPPER, etc.)
2. **Proximity Weighting** -- nearby words influence each other with exponential decay (0.7x per word of distance)
3. **Structure Detection** -- role sequences are matched against 26 defined patterns
4. **Physics** -- momentum-based blending (0.82 persistence) produces final VADUGWI coordinates

The core equations are documented in `docs/vadug-calculation.md`. In brief:

- **State update**: each word blends into a running state at 0.82/0.18 momentum ratio
- **Proximity field**: modifiers (negators, amplifiers, hedges) apply force scaled by distance
- **Impulse override**: high-magnitude words push directly past the momentum filter
- **Structure adjustment**: detected patterns shift the physics result by weighted confidence

## Current Numbers

| Metric | Value |
|--------|-------|
| Accuracy on permanent suite | 100% on 630 sentences |
| Crisis recall | 97.3% |
| False positives on safe text | 0% |
| Latency | 0.15ms per sentence |
| Engine size | ~300KB |
| Vocabulary | 4,000+ words |
| Structural patterns | 26 |
| Word roles | 23 |
| Tests | 167 |

## 26 Structural Patterns

These are role-sequence patterns detected from word classification output:

BETRAYAL, BLANKET_APOLOGY, BRAVADO, CALLING_OUT, CHOPPER_SPLIT, D_INVERSION, DIRECTED_POSITIVE, EXCLUDED_POSITIVE, EXHAUSTION, FAREWELL, FINALITY, FLEEING, METHOD_ACQUISITION, MINIMIZER, NO_EXIT, POWER_OVER_SELF, PURSUIT_OF_METHOD, RELIEF_ABSENCE, SARCASM_INVERSION, SELF_EXCLUDED, SELF_NULLIFY, SELF_REMOVAL, SELF_SUBMISSION, SUSPICIOUS_CALM, VICTIMIZATION, WITHHELD_POSITIVE

Each pattern carries a confidence weight and VADUGWI adjustment vector. Crisis-relevant patterns (FAREWELL, METHOD_ACQUISITION, SELF_REMOVAL, NO_EXIT) are tuned for zero false positives on safe text.

## Bidirectional Solver

**Forward**: text produces VADUGWI coordinates.

**Backward**: given current state A and a target outcome zone C, the solver sweeps response temperature to find the range of valid B values that land in the target zone. The valid response is a range, not a single point.

Default blend: `C = A * 0.6 + B * 0.4` -- 60% of current mood persists, 40% of the response gets through. This ratio is adjustable per personality profile.

## SmolLM2 / Llama Integration

The engine pairs with a small language model for emotionally coherent dialogue generation. The model generates candidate responses; the engine scores each candidate's emotional impact using the forward solver. Responses that land outside the target zone are rejected. The model speaks, the engine scores the impact.

## Running Tests

```bash
python3 -m pytest engine/tests/ -v
```

## Links

- **Live demo**: [huggingface.co/spaces/deucebucket/clanker](https://huggingface.co/spaces/deucebucket/clanker)
- **Browser demo**: [deucebucket.github.io/clanker-demo](https://deucebucket.github.io/clanker-demo)

## File Structure

```
engine/              V5.5 engine (~300KB)
  pendulum.py          Physics layer -- momentum blending
  word_classifier.py   Word role classification (23 roles)
  proximity.py         Proximity field computation
  structures.py        Pattern detection (26 patterns)
  solver.py            Bidirectional A+B=C solver
  forces_curated.py    4,000+ word force tuples (7D VADUGWI)
  shared.py            VADUGWI dataclass

docs/                Reference
  vadug-calculation.md   Full equation reference
  v3-user-physics.md     Structural rules
  THEORY.md              Theory document
```
