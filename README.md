# Clanker-Lang

A rule-based emotional physics engine that processes text into 5-dimensional emotional coordinates. Nothing in emotions is boolean -- every word is a continuous force, every negation decays over distance, every hedge shifts confidence without flipping polarity. 0.09ms/sentence. Every number is auditable.

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

## The Engine

**V2** (`demo/pendulum_v2.py`) -- a 3-pass PEMDAS emotional physics engine.

`Force = Payload * Context(WHO * TENSE * INTENSITY) * Negation * Physics`

- **~2,000 curated words** carry 97% of emotional signal (replaced a 46K-word V1 lexicon -- 44K words were noise)
- **84 context operators** across 17 categories create a measured 12x range on the same word
- **17 of 25 conversational forces** implemented: continuous negation, hedging with D-offsets, evokers (gravitational priming), conditionals, evidential/clinical, comparatives, superlatives, passive voice, fillers, performatives, hyperbole idioms, euphemisms
- **Remaining 8 forces**: rhetorical questions, sarcasm, irony, understatement, repetition, code-switching, reported speech, discourse markers

## Benchmarks

Tested on 2,872 sentences from published academic datasets (SST-2, GoEmotions, TweetEval).

| Engine | SST-2 | GoEmotions | TweetEval | Composite | Type | Speed |
|--------|-------|-----------|-----------|-----------|------|-------|
| **Clanker V2** | **60.9%** | 56.8% | 62.9% | **60.2%** | Rule-based physics | 0.09ms |
| VADER | 55.7% | **60.6%** | **74.1%** | 63.5% | Rule-based lexicon | 0.06ms |
| TextBlob | 53.8% | 57.8% | 50.7% | 54.1% | Pattern-based | 0.16ms |
| RoBERTa | 69.0% | 62.1% | 77.7% | 69.6% | 125M param transformer | 5ms |

V2 beats VADER on SST-2 by +5pp with 17 of 25 forces active. 8 forces remain unimplemented. See `benchmarks/experiment_log.jsonl` for 34 versioned experiments with NASA-style logging.

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

# Run benchmarks (Clanker V2 + VADER + TextBlob + RoBERTa)
python3 benchmarks/academic_benchmark.py

# Run genetic optimizer (RTX 3090, ~1M evals)
python3 benchmarks/gpu_optimizer_v2.py

# View experiment history
python3 benchmarks/experiment_tracker.py --history
```

## How It Works

### The 25-Force Conversational Universe

Human conversation generates 25 distinct emotional forces. Each force changes how words land:

Negation doesn't flip a boolean -- it applies a decaying force that weakens over distance. "I don't hate you" is not the same as "I love you." Hedging doesn't just dampen -- it shifts Dominance (confidence) independently of Valence. "I might be angry" and "I am angry" have different D-coordinates, not just different intensities.

The engine models each force as continuous physics, not if/else rules.

### 3-Pass PEMDAS Pipeline

Every word is classified as one of three types:
- **OPERATORS** -- context multipliers (I=1.8x, a=0.6x, very=1.3x, was=0.85x)
- **PAYLOADS** -- emotional force carriers (~2,000 curated words, 97% of signal)
- **NEUTRAL** -- transparent, zero force

Processing order:
1. **Pre-pass** -- detect questions, idioms, negation positions, conditionals
2. **Word-pass** -- left-to-right: operators accumulate, payloads consume them with momentum physics
3. **Post-pass** -- crisis detection, clamping, final VADUG

## Experiment Tracking

Every test run is versioned with NASA-style logging:

```bash
python3 benchmarks/experiment_tracker.py --history     # all runs
python3 benchmarks/experiment_tracker.py --best        # top performers
python3 benchmarks/experiment_tracker.py --compare EXP-0001 EXP-0042
```

Each experiment logs: ID, theory name, full config snapshot, per-benchmark results, deltas vs baseline.

## Key Findings

- **Nothing in emotions is boolean** -- negation is a decaying force, hedging shifts a separate axis, intensity is continuous
- ~2,000 words carry 97% of emotional signal (44K words = noise)
- Bridge words are OPERATORS not fillers (12x range: 0.24x to 2.88x)
- Gravity and Dominance are the driving forces, not Valence
- Forces and physics are coupled (can't fix one without the other)
- Idioms discoverable from mathematical residuals (pattern deviation = idiom)
- Sarcasm can't be rule-based (needs model context)

## Trained Model

**Clanker-Micro** (7.7M params) -- GPT-2 backbone with 5 VADUG regression heads trained on engine output. The rule-based engine reads English; the model learns to think in VADUG coordinates directly.

- Training: `python3 training/train.py`
- HuggingFace Space: [deucebucket/clanker-emotional-engine](https://huggingface.co/spaces/deucebucket/clanker-emotional-engine)

## Architecture

See `CLAUDE.md` for full module breakdown and conventions.
See `docs/THEORY.md` for the unified theory of emotional mechanics.
See `docs/tuning-notes.md` for tuning decisions and rationale.

## License

MIT
