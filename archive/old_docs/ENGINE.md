# Clanker Pendulum Engine — Reference Implementation

*How to derive VADUG coordinates from natural language input.*

This document specifies the reference implementation for parsing natural language
into Clanker VADUG emotional coordinates. The Pendulum Engine is not part of the
Clanker language specification itself — it is one implementation of VADUG detection.
Other implementations (neural, rule-based, hybrid) are equally valid as long as
they produce valid VADUG coordinates.

## 1. Overview

Clanker processes natural language input through a sequential pendulum that tracks emotional state word by word across five dimensions (VADUG). Unlike traditional sentiment analysis which averages word scores, the pendulum models emotional DYNAMICS — how each word shifts the trajectory based on what came before it.

## 2. How It Works

1. The pendulum starts at center (V128 A128 D128 U0 G128 — neutral, grounded).
2. Each word applies a force vector that SHIFTS the pendulum.
3. The force depends on:
   - The word's base emotional weight (from morphological roots).
   - The pendulum's CURRENT position (context-dependent force).
   - Recent word history (idiom detection, anticipation patterns).
4. The pendulum has MOMENTUM — it resists sudden changes.
5. Neutral words barely move it; strong emotional words can yank it.
6. The final position after all words = the VADUG coordinate for the message.

## 3. Context-Dependent Forces

The same word applies different force depending on current state:

- **"buddy"** when pendulum is positive → friendly (V+15)
- **"buddy"** when pendulum is negative/tense → confrontational (V-10, A+20)
- **"you"** in high-arousal context → targeted/threatening (V-15, A+20)
- **"but"** after positive trajectory → dread, reversal incoming (V-40, A+20)
- **"but"** after negative trajectory → relief possible (V+10, A-5)

## 4. Momentum and Inertia

The pendulum maintains 85-90% of its current state between words. This means:

- Once swinging negative, neutral words don't reset it — it drifts slowly.
- Strong emotional words can override momentum.
- Emotional trajectories build over sentences, not just sum individual words.

## 5. Idiom Recognition

Multi-word expressions that carry compound emotional meaning:

| Idiom            | V   | A   | D   | U   | G   | Meaning      |
|------------------|-----|-----|-----|-----|-----|--------------|
| "bone to pick"   | -25 | +30 | +25 | +25 | +10 | grievance (rises) |
| "piece of cake"  | +20 | -15 | +20 | —   | +15 | easy (light)      |
| "fed up"         | -30 | +25 | -10 | +15 | +20 | frustrated (rises)|
| "break a leg"    | +25 | +20 | +10 | —   | +10 | good luck (light) |

## 6. Morphological Fallback

When a word isn't in the direct dictionary, it decomposes into prefix + root + suffix:

- ~30 prefix modifiers (un-, dis-, over-, mis-...)
- ~1000 root morphemes with emotional weights
- ~40 suffix modifiers (-less, -ful, -ous, -ive, -ness...)
- ~1070 entries covering millions of words through composition

Example: `"hopelessness"` = hope(V+55) + -less(negate → V-55) + -ness(state → V-55)

## 7. Anticipation Patterns

Certain word sequences build tension before the payload arrives:

- **"I've got"** → something coming, arousal builds
- **"I need to tell you"** → serious incoming, urgency rises
- **"listen"** → attention demanded, dominance shifts
- **"actually"** → correction coming, slight negative shift

## 8. The Emotional Arc

The pendulum doesn't produce a single score — it produces a TRAJECTORY:

```
"I love you, but I broke your vase"
  "I"     → V128 G128 (neutral, grounded)
  "love"  → V200 G210 (soaring positive, light)
  "you"   → V205 G215 (warm, directed, floating)
  "but"   → V150 G160 (YANK — dread, gravity drops)
  "I"     → V148 G155 (holds tense)
  "broke" → V100 G100 (negative, guilt, sinking)
  "your"  → V95  G95  (directed at you — makes it personal, heavier)
  "vase"  → V90  G100 (object, slight recovery)

  Arc: neutral → love/soaring → DREAD/dropping → guilt/sinking → settling
  Final: V90 A160 D85 U40 G100
```

The model trained on these arcs learns emotional PHYSICS — how emotions flow, build, crash, and recover through language. The gravity axis adds the physical dimension: love lifts (G210), guilt sinks (G100), despair crushes (G15).

## 9. Why This Matters for Clanker Models

A Clanker model trained on sequential pendulum traces learns to predict emotional trajectories, not just next tokens. It can:

- Anticipate when someone is about to get angry (rising A, falling V, rising G — anger boils upward).
- Model how different responses will shift the user's emotional state across all five dimensions.
- Plan multi-turn emotional arcs for therapeutic or de-escalation purposes.
- Understand that "I'm fine" after bad news means the opposite of "I'm fine" in isolation.
- Distinguish hate (rises, G180) from despair (crushes, G15) from dislike (sinks, G90) — emotions that share similar V/A/D but diverge sharply in gravity.

This is emotional dynamics, not sentiment classification.
