# Clanker Engine Tuning Notes

## Default Personality: Emotionally Perceptive Observer

Not neutral (robot). Not empath (oversensitive). A well-calibrated emotional sensor.

```
gullibility:     100  (skeptical but open)
agreeableness:   160  (empathetic but not pushover)
suggestibility:   80  (resistant to manipulation)
truthfulness:    200  (honest about readings)
safety:          180  (handles hard content, has limits)
curiosity:       160  (interested in emotional nuance)
assertiveness:   120  (present but not dominating)
playfulness:     100  (serious-leaning, appropriate humor)
```

Willingness (W) = 0.42 → Force multiplier = 1.09x (9% more perceptive than neutral)

## Layer Ablation Results (v0.15.0, 2,872 sentences)

| Layer | Effect when disabled | Decision |
|-------|---------------------|----------|
| Context modifiers | **+0.6pp (HURTS)** | Fixed: gated to V<100 or V>155 only |
| Recency weighting | +0.2pp | Keep — helps arcs (designed for multi-sentence) |
| Exponential decay | +0.1pp on benchmark | Keep — enables entropy detection |
| Bookend parsing | 0.0pp | Keep — helps crisis detection (isolation -13, self_loop -15) |
| Idiom detection | 0.0pp | Keep — helps specific phrases |
| Crisis momentum lock | 0.0pp | Keep — SAFETY CRITICAL (8/8 detection) |
| Shift markers | -0.0pp | Keep — helps arc reversal |
| Word role detection | -0.0pp | Keep — fixes "poor girl" type errors |
| Morpheme decomposition | -0.0pp | Keep — fallback for unknown words |
| Bigram detection | -0.1pp | Keep — helps "give up", "fall in love" |

## Key Tuning Decisions

### Force scaling (genetic algorithm, 15,161 evaluations)
- Valence: 1.15x
- Arousal: 2.25x (was dampened to 0.77x — WRONG)
- Dominance: 2.49x
- Urgency: 1.23x
- Gravity: 1.36x
- Self-Worth: 1.0x (default scaling, tuning pending)
- Intent: 1.0x (V5.5 addition — default scaling, tuning pending)

### Pendulum physics
- Momentum: 0.99 for strong words, exponential decay for weak
- Decay lambda: 0.15 (slower decay = longer emotional memory)
- Spike threshold: 10 (low — more words qualify as emotional)
- Drift rate: 0.01 (minimal center pull)
- Force scale cap: 2.5x max (prevents triple multiplier clipping)

### Classification thresholds (intent-adaptive)
- EMOTIONAL/VENTING: [123, 124] (very tight — force a decision)
- STATEMENT/DEFAULT: [114, 130]
- QUESTION/REQUEST: [105, 150] (wide — neutral is fine)

### Context modifiers
- Only activate when V < 100 or V > 155 (not in neutral zone)
- 31 chameleon words including lol, love, fine, ok, sorry
- REPLACE base force, not add on top

### Entropy neutral detection
- Fires when: high entropy + low variance + V near center
- More willing for STATEMENT/QUESTION intent
- Completely flat trajectory → always neutral

## What Doesn't Help on Academic Benchmarks (but helps real conversations)
- Relationship gravity (daughter vs someone)
- Bookend parsing (isolation, self_loop patterns)
- Morpheme decomposition (truly unknown words)
- Crisis momentum lock (safety-critical, not measured by sentiment benchmarks)

These are REAL-WORLD features that academic benchmarks don't test.
