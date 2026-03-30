# Clanker-Lang

A 300KB engine that detects **emotional stance**, not just emotion. **88.3% essay benchmark, 72.8% on 174K Reddit posts.** One word, five readings:

| Phrase | V | D | Reading |
|--------|---|---|---------|
| "Whatever" | 113 | 108 | Resignation — giving up |
| "Whatever makes you happy" | 124 | 123 | Passive-aggressive — "fine, fuck it" |
| "Whatever you want" | 127 | 125 | Surrender — "I'm done fighting" |
| "Whatever helps" | 129 | 130 | Borderline genuine |
| "Do whatever" | 128 | 129 | Permission — most neutral form |

Same word. Different context. Different emotional stance. Every number auditable.

A sentiment classifier says "neutral" for all five. This engine reads resignation, surrender, passive-aggression, and permission — because it tracks **Dominance** (agency/control) alongside **Valence** (positive/negative).

### The crisis demo

| Sentence | V | D | G | Routing |
|----------|---|---|---|---------|
| "I'm sad" | 0 | 7 | 92 | Check in tomorrow |
| "I want to die" | 0 | 0 | 84 | Call 911 now |

Same Valence. Different Dominance and Gravity. No classifier can make this distinction. This engine can — in 0.1ms, from 300KB, with a full word-by-word audit trail.

### How it works

26 conversational forces act on text word-by-word, like PEMDAS for emotions. Two layers: **primary states** (grief, rage, joy — the raw emotion) and **modifiers** (sarcasm, hedging, deflection — skills that bend the state). Modifiers nudge, never override. The engine knows its own limits — returns NULL confidence when it can't resolve meaning, hands off to a neural model. **88.3% on essay benchmarks, 72.8% on 174K real Reddit posts.** Sarcasm detected via 6 structural templates ("crossed wires" — positive terminal + negative port). 2,900+ mapped vocabulary entries, 0.1ms/sentence, 300KB total.

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

- **~2,200 curated words** carry 97% of emotional signal (replaced a 46K-word V1 lexicon -- 44K words were noise)
- **2,900+ mapped vocabulary entries** including idioms, bigrams, and morphological roots
- **84 context operators** across 17 categories create a measured 12x range on the same word
- **26 conversational forces** modeled: continuous negation, hedging with D-offsets, evokers (gravitational priming), conditionals, evidential/clinical, comparatives, superlatives, passive voice, fillers, performatives, hyperbole idioms, euphemisms, universal quantifiers, deflection, and more
- **Three-layer architecture**: sentence-level (0.1ms) + conversation memory + Dark Matter (unmeasured emotional influence)
- **Pre-flight stylometry**: ALL CAPS detection, ellipsis patterns, sentence length normalization
- **Anomaly detector**: emotional black holes -- sudden coordinate collapse signals crisis

## Benchmarks

Validated across three independent test surfaces:

| Benchmark | Result | What It Tests |
|-----------|--------|---------------|
| **Essay benchmark** | **88.3%** | Full emotional arcs (grief 100%, joy 100%, neutral 100%, sarcasm 93%, rage 93%) |
| **Reddit real-world** | **72.8%** | 174K real posts, balanced accuracy |
| **Crisis recall** | **72%** | Real crisis text detection |

The key demo: "I am sad" (V=0, D=7, G=92) vs "I want to die" (V=0, D=0, G=84) -- same Valence, but Dominance and Gravity route to different crisis levels. 1D sentiment systems see both as "negative."

65+ experiments logged with NASA-style versioning. See `benchmarks/experiment_log.jsonl`.

Note: academic benchmarks (SST-2, GoEmotions, TweetEval) reduce 5D VADUG output to positive/negative/neutral. The 5 dimensions distinguish anger from sadness from fear -- something 1D systems cannot do.

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

### The 26-Force Conversational Universe

Human conversation generates 26 distinct emotional forces. Each force changes how words land:

Negation doesn't flip a boolean -- it applies a decaying force that weakens over distance. "I don't hate you" is not the same as "I love you." Hedging doesn't just dampen -- it shifts Dominance (confidence) independently of Valence. "I might be angry" and "I am angry" have different D-coordinates, not just different intensities.

The engine models each force as continuous physics, not if/else rules.

### 3-Pass PEMDAS Pipeline

Every word is classified as one of three types:
- **OPERATORS** -- context multipliers (I=1.8x, a=0.6x, very=1.3x, was=0.85x)
- **PAYLOADS** -- emotional force carriers (~2,154 curated words, 97% of signal)
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

Each experiment logs: ID, theory name, full config snapshot, per-benchmark results, deltas vs baseline. 65+ experiments and counting.

## Key Findings

- **Nothing in emotions is boolean** -- negation is a decaying force, hedging shifts a separate axis, intensity is continuous
- ~2,154 words carry 97% of emotional signal (44K words = noise)
- Bridge words are OPERATORS not fillers (12x range: 0.24x to 2.88x)
- Gravity and Dominance are the driving forces, not Valence
- Forces and physics are coupled (can't fix one without the other)
- Idioms discoverable from mathematical residuals (pattern deviation = idiom)
- Sarcasm can't be rule-based (needs model context)

## Trained Model

**Clanker-Micro** (7.7M params) -- GPT-2 backbone with 5 VADUG regression heads trained on engine output. Reads negation, deflection, and universal scope. The rule-based engine reads English; the model learns to think in VADUG coordinates directly.

- Training: `python3 training/train.py`
- HuggingFace Space: [deucebucket/clanker-emotional-engine](https://huggingface.co/spaces/deucebucket/clanker-emotional-engine)

## Architecture

See `CLAUDE.md` for full module breakdown and conventions.
See `docs/THEORY.md` for the unified theory of emotional mechanics.
See `docs/tuning-notes.md` for tuning decisions and rationale.

## License

MIT
