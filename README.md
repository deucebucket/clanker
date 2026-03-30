# Clanker-Lang -- Conversation State Resolver

I built a system that detects **emotional stance** through structural pattern recognition. It reads text the way a chess player reads a board -- recognizing patterns from piece positions, not memorizing specific games. ~300KB engine, 0.15ms/sentence, 158 tests passing.

| Phrase | V | D | Reading |
|--------|---|---|---------|
| "Whatever" | 113 | 108 | Resignation -- giving up |
| "Whatever makes you happy" | 124 | 123 | Passive-aggressive -- "fine, fuck it" |
| "Whatever you want" | 127 | 125 | Surrender -- "I'm done fighting" |
| "Whatever helps" | 129 | 130 | Borderline genuine |
| "Do whatever" | 128 | 129 | Permission -- most neutral form |

Same word. Different context. Different emotional stance. A sentiment classifier says "neutral" for all five.

### The crisis demo

| Sentence | V | D | G | Routing |
|----------|---|---|---|---------|
| "I'm sad" | 0 | 7 | 92 | Check in tomorrow |
| "I want to die" | 0 | 0 | 84 | Call 911 now |

Same Valence. Different Dominance and Gravity. No sentiment classifier makes this distinction.

### How it works

Words have **mass**. "Love" is a heavy star. "Carpenter" is dark matter -- no mass of its own, it reflects whatever stars are nearby. The engine classifies every word into a structural role, computes proximity fields, then detects patterns from the role sequences.

This is NOT an "emotional physics engine" or a sentiment classifier. It is a conversation state resolver that uses structural pattern recognition. I think some emotional language follows rules that can be described with math. The data suggests this works for certain patterns. The benchmarks show where it works and where it falls short.

## Results

I want to be honest about these numbers. The ones I'm proud of and the ones that keep me humble:

| Benchmark | Result | What It Tests |
|-----------|--------|---------------|
| **Novel sentences** | **91%** | Sentences the engine never practiced on |
| **Crisis detection** | **86%** | Real crisis text identification |
| **Sarcasm detection** | **90%** | Structural sarcasm templates |
| **Safe sentence false positives** | **0%** | Never flags safe text as crisis |
| **SST-2 (academic sentiment)** | **51%** | Movie review positive/negative classification |

The 51% on SST-2 is real and I'm not hiding it. Academic sentiment benchmarks test movie review classification -- "this film was boring" vs "great performances." That is a different task than structural emotional reading. I score 91% on novel emotional sentences because that is what this system is built for. The SST-2 number shows the system does NOT do general sentiment classification well. I think that is an honest tradeoff, not a failure.

## V3 Architecture

**V3 is the current engine** (`engine/` directory). V2 is boxed at tag `v2.0` and lives in `demo/`.

V3 has three systems:

### 1. Structure Recognition

Words get classified into four tiers:

| Tier | Count | What | Example |
|------|-------|------|---------|
| **Anchor stars** | ~50 | ALWAYS heavy, guilty until proven innocent | die, kill, love, hate, suicide |
| **Regular stars** | ~200 | Have mass, can be overridden by context | happy, sad, angry, scared |
| **Operators** | ~50 | Shape the field, no mass of their own | I, you, not, very, but, still |
| **Dark matter** | everything else | Null -- inherits from nearby stars | carpenter, Tuesday, meeting |

Connectors are math operators, not filler:
- **and** = additive (+), both stack
- **but** = chopper (-), kills before, promotes after
- **or** = alternative (><), fork/uncertainty
- **of** = attributive (/), routes source to state
- **if** = conditional (?), opens hypothetical branch

The engine reads role sequences and recognizes structural patterns -- like a chess player seeing checkmate conditions from piece positions, not memorized move sequences.

### 2. A+B=C Bidirectional Solver

Given a user's emotional state (A) and a candidate response (B), predict the resulting state (C). Or work backwards: given where someone is and where you want them to land, find what B needs to be. The target is a zone (like a runway), not a point.

### 3. Emotional Battleship

Fire calibrated probes, measure how much the response deviates from what a neutral person would produce, triangulate the hidden emotional state. The distortion IS the signal.

## The 5 Dimensions (VADUG)

| Axis | Range | What It Measures |
|------|-------|-----------------|
| V (Valence) | 0-255 | Negative <- 128 -> Positive |
| A (Arousal) | 0-255 | Calm <- 128 -> Intense |
| D (Dominance) | 0-255 | Helpless <- 128 -> In Control |
| U (Urgency) | 0-255 | No Rush -> Critical |
| G (Gravity) | 0-255 | Crushing/Sinking <- 128 -> Floating/Soaring |

5 bytes = 1.1 trillion unique emotional states.

## Quick Start

```bash
# Install
cd decoder/python && pip install -e ".[dev]"

# Interactive demo (V2)
python3 demo/simulator.py

# Run engine tests (V3)
python3 -m pytest engine/tests/ -v

# Run benchmarks
python3 benchmarks/academic_benchmark.py --quick

# View experiment history
python3 benchmarks/experiment_tracker.py --history
```

## Stats

- **Engine size:** ~300KB
- **Speed:** 0.15ms/sentence
- **Trained model:** ~30MB (GPT-2 backbone, 5 VADUG heads)
- **Tests:** 158 passing (V3 engine suite)
- **Experiments logged:** 65+ with NASA-style versioning

## Key Findings

- Nothing in emotions is boolean -- negation is a decaying force, hedging shifts a separate axis
- ~2,154 words carry 97% of emotional signal (44K words were noise)
- Bridge words are OPERATORS not fillers (12x range: 0.24x to 2.88x)
- Gravity and Dominance appear to be the driving forces, not Valence
- Idioms may be findable from mathematical residuals (pattern deviation suggests compound meaning)
- Structure detection generalizes better than vocabulary matching

## Architecture

See `CLAUDE.md` for full module breakdown and conventions.
See `docs/THEORY.md` for the theory and research connections.
See `docs/v3-user-physics.md` for V3's structural rules.
See `docs/tuning-notes.md` for tuning decisions and rationale.

## Trained Model

**Clanker-Micro** (~30MB) -- GPT-2 backbone with 5 VADUG regression heads trained on engine output. The rule-based engine reads English; the model learns to think in VADUG coordinates directly.

- Training: `python3 training/train.py`
- HuggingFace Space: [deucebucket/clanker-emotional-engine](https://huggingface.co/spaces/deucebucket/clanker-emotional-engine)

## License

MIT
