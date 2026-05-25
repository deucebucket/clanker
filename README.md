# Clanker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19383636.svg)](https://doi.org/10.5281/zenodo.19383636)

A conversation state resolver that detects emotional stance through structural pattern recognition. Computes 7-dimensional emotional coordinates (VADUGWI) from text using deterministic, explainable transformations. Every output can be traced through explicit math. You can ask WHY and get a real answer.

"Whatever" alone reads as resignation (V=93, D=97). "Whatever makes you happy" reads as passive-aggressive (V=30, D=113). "Do whatever" reads as dismissive permission (V=93, D=97). Same word -- context changes the coordinates. A sentiment classifier says "neutral" for all three.

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
| "I'm fine" | 83 | Below neutral -- uneasy, not positive |
| "haha yeah im totally okay" | 15 | Forced composure, bravado mask detected |
| "oh joy" | 29 | Positive word, negative reading |
| "do you even love me" | 152 | Positive word, but A=154 D=133 -- challenge energy |
| "my wife cheated on me with my best friend" | 14 | V=14, A=151, D=56 -- deep negative, high intensity, low control |
| "I love my mom" | 184 | Genuine positive, no false alarm |
| "the meeting is at three" | 128 | Neutral -- no emotional content detected |

## How It Works

Four processing layers run in sequence:

1. **Word Classification** -- each word is assigned structural roles (SELF_REF, EMOTIONAL, NEGATOR, AMPLIFIER, CONNECTOR, CHOPPER, etc.)
2. **Proximity Weighting** -- nearby words influence each other with exponential decay (0.7x per word of distance)
3. **Structure Detection** -- 61 chess-like patterns detected from role sequences
4. **Physics** -- 9-stage pipeline: tokenize, classify, interpret context, coefficients, accumulate forces, structure adjustment, W-V coupling, personality, tanh saturation

Additional systems:
- **Force Flow** -- WHO does WHAT to WHOM directional analysis
- **Phase System** -- SOLID (never flips), LIQUID (context-dependent), GAS (neutral) word states
- **Crisis Detection** -- continuous 0.0-1.0 concern gradient, zero false positives on safe text
- **Anomaly Detection** -- deflection, masking, velocity, resonance patterns
- **Bidirectional Solver** -- given state A and target zone C, find valid response range B

The core equations are documented in `docs/vadug-calculation.md`.

## Quick Start

```bash
git clone https://github.com/deucebucket/clanker.git
cd clanker
pip install -r requirements.txt
python3 -m pytest engine/tests/ -v
```

```python
from engine.pendulum import compute_vadug

result, context = compute_vadug("whatever makes you happy")
print(f"V={result.v}, A={result.a}, D={result.d}, U={result.u}, G={result.g}, W={result.w}, I={result.i}")
# V=30, A=141, D=113, U=5, G=143, W=102, I=161
```

## Current Numbers

| Metric | Value |
|--------|-------|
| Ground truth (developer-verified) | 41/41 (100%) |
| Stress test (275 real sentences, 11 categories) | 201/275 (73.1%) |
| Crisis recall | 70.6% (36/51) |
| Crisis false positive (safe + dark humor + metaphor) | 0/75 (0.0%) |
| SST-2 benchmark | 63.9% (VADER: 55.7%, RoBERTa: 69%) |
| Real-text spot-check (5 corpora) | ~59% on 167 sentences |
| Throughput | 2,000-6,000 sentences/sec |
| Latency | 0.17ms per sentence |
| Vocabulary | 4,544 curated words |
| Structural patterns | 61 |
| Tests | 207 |

Tested against consensus of 4 frontier AI models (Gemini, Claude Opus, GPT-4, Grok).

## Known Weaknesses

| Category | Accuracy |
|----------|----------|
| Slang positive | 44% |
| Grief | 52% |
| Passive aggressive | 68% |

Positive inflation reduced but not eliminated. These are areas of active development.

## File Structure

```
engine/              V8 engine
  pendulum.py          Physics layer -- 9-stage pipeline
  word_classifier.py   Structural role classification
  proximity.py         Proximity field computation
  structures.py        Pattern detection (61 patterns)
  solver.py            Bidirectional A+B=C solver
  force_flow.py        WHO does WHAT to WHOM
  forces_curated.py    4,544 word force tuples (7D VADUGWI)
  crisis.py            Crisis detection (0.0-1.0 gradient)
  phase.py             SOLID/LIQUID/GAS word states
  anomaly.py           Anomaly detection (4 detectors)
  shared.py            VADUGWI dataclass
  vocabulary.py        Vocabulary loader

engine_v9/           V9 engine (experimental)

docs/                Reference
  vadug-calculation.md   Full equation reference
  THEORY.md              Theory document
  v3-user-physics.md     Structural rules
  SPEC.md                Full engine specification
```

## Running Tests

```bash
# V8 engine tests
python3 -m pytest engine/tests/ -v

# Full barrage (ground truth + stress + crisis + throughput)
python3 benchmarks/full_barrage.py

# Crisis benchmark
python3 benchmarks/crisis_benchmark.py

# Academic benchmark (vs VADER, TextBlob, RoBERTa)
python3 benchmarks/academic_benchmark.py --quick
```

## Links

- **Live demo**: [huggingface.co/spaces/deucebucket/clanker](https://huggingface.co/spaces/deucebucket/clanker)
- **Browser demo**: [deucebucket.github.io/clanker-demo](https://deucebucket.github.io/clanker-demo)
- **Theory**: [docs/THEORY.md](docs/THEORY.md)
- **Full specification**: [SPEC.md](SPEC.md)

## License

Licensed under the [MIT License](LICENSE).

## Author

Jerry Mares ([deucebucket](https://github.com/deucebucket))

DOI: [10.5281/zenodo.19383636](https://doi.org/10.5281/zenodo.19383636)
