# Clanker-Lang

A rule-based emotional physics engine. Processes text into 5-dimensional emotional coordinates. No ML. 0.3ms/sentence. Every number is auditable.

## Try It

```bash
python3 demo/simulator.py
```

```
Input:  "I lost my job today and I feel terrible"
Output: V=80 A=230 D=98 U=104 G=91

        Valence:   80  (very negative)
        Arousal:   230 (extremely intense)
        Dominance: 98  (helpless)
        Urgency:   104 (urgent)
        Gravity:   91  (sinking)

        Intent: EMOTIONAL
        Tone: sincere
        Grade: D
```

## Engines

Two engines exist:

| Engine | Architecture | Vocabulary | Lines |
|--------|-------------|-----------|-------|
| **V1** (`demo/pendulum.py`) | Layered pipeline with multi-path processing | 46,101 words | ~1,400 |
| **V2** (`demo/pendulum_v2.py`) | Clean 3-pass PEMDAS (pre-pass, word-pass, post-pass) | 2,049 curated words | ~535 |

V2 is the active development engine. Clean math: `Force = Payload * Context(WHO * TENSE * INTENSITY) * Negation * Physics`

## Benchmarks

Tested on 2,872 sentences from published academic datasets.

| Engine | SST-2 | GoEmotions | TweetEval | Type | Speed |
|--------|-------|-----------|-----------|------|-------|
| **Clanker V1** | **60.9%** | 51.6% | 72.0% | Rule-based physics | 0.7ms |
| Clanker V2 | 40.1%* | 53.1% | 42.9%* | Rule-based PEMDAS | 0.09ms |
| VADER | 55.7% | **60.6%** | **74.1%** | Rule-based lexicon | 0.06ms |
| TextBlob | 53.8% | 57.8% | 50.7% | Pattern-based | 0.16ms |
| RoBERTa | 69.0% | 62.1% | 77.7% | 125M param transformer | 5ms |

*V2 baseline is untuned. Genetic algorithm optimization in progress. See `benchmarks/experiment_log.jsonl` for versioned test results.

Note: these benchmarks reduce 5D VADUG output to positive/negative/neutral. The 5 dimensions distinguish anger from sadness from fear -- something 1D systems cannot do.

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

# Interactive demo
python3 demo/simulator.py

# Run benchmarks (V1 + V2 + VADER + TextBlob + RoBERTa)
python3 benchmarks/academic_benchmark.py

# Run V2 genetic optimizer
python3 benchmarks/gpu_optimizer_v2.py

# View experiment history
python3 benchmarks/experiment_tracker.py --history
```

## How It Works

### V2 Engine (3-pass PEMDAS)

Every word is classified as one of three types:
- **OPERATORS** -- context multipliers (I=1.8x, a=0.6x, very=1.3x, was=0.85x)
- **PAYLOADS** -- emotional force carriers (2,049 curated words, 97% of signal)
- **NEUTRAL** -- transparent, zero force

Processing order:
1. **Pre-pass** -- detect questions, idioms, negation positions
2. **Word-pass** -- left-to-right: operators accumulate, payloads consume them with momentum physics
3. **Post-pass** -- crisis detection, clamping, final VADUG

84 context operators across 17 categories create a measured 12x range on the same word.

### V1 Engine (7-layer pipeline)

1. Intent Detection -- question, venting, greeting, emotional content
2. Bookend Parsing -- who->whom relationship structure
3. Pendulum -- word-by-word with 46K calibrated forces, exponential decay
4. Context Modifiers -- trajectory-dependent word meaning
5. Tonal Analysis -- sarcasm, deadpan, deflection detection
6. Entropy Check -- flat trajectory = neutral
7. Adaptive Classification -- intent-aware thresholds

## Experiment Tracking

Every test run is versioned with NASA-style logging:

```bash
python3 benchmarks/experiment_tracker.py --history     # all runs
python3 benchmarks/experiment_tracker.py --best        # top performers
python3 benchmarks/experiment_tracker.py --compare EXP-0001 EXP-0042
```

Each experiment logs: ID, theory name, full config snapshot, per-benchmark results, deltas vs baseline.

## Key Findings

- 2,049 words carry 97% of emotional signal (44K words = noise)
- Bridge words are OPERATORS not fillers (12x range: 0.24x to 2.88x)
- Forces and physics are coupled (can't fix one without the other)
- NRC VAD lexicon has systematic negativity bias (positive words 2-4x weaker)
- Idioms discoverable from mathematical residuals (pattern deviation = idiom)
- Sarcasm can't be rule-based (needs model context)

## Architecture

See `CLAUDE.md` for full module breakdown and conventions.
See `docs/THEORY.md` for the unified theory of emotional mechanics.
See `docs/tuning-notes.md` for tuning decisions and rationale.

## License

MIT
