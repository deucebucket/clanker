---
title: Clanker-Lang Emotional Physics Engine
emoji: "\U0001F527"
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: "5.50.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
startup_duration_timeout: 1h
---

# Clanker-Lang: Glass-Box Emotional Physics Engine

Rule-based emotional physics. 5D VADUG output. 0.3ms/sentence. Zero ML. Every number auditable.

## Benchmarks

| Benchmark | Result | Categories at 100% |
|-----------|--------|--------------------|
| **Essay benchmark** | **91.7%** | Grief, Rage, Joy, Neutral, Fear |
| **Reddit 174K posts** | **72.8%** | Balanced accuracy |
| **Sarcasm detection** | **93.3%** | 6 structural templates |

26 conversational forces. 2,900+ mapped vocabulary entries. Zero training, zero GPU.

## What This Demo Shows

**Left panel**: Chat with personality presets
**Right panel**: Glass-box debug showing every step:

1. Intent Detection -- GREETING/QUESTION/REQUEST/VENTING/EMOTIONAL/CASUAL/STATEMENT
2. Word-by-Word Pendulum Trace -- each word's V/A/D/U/G delta
3. Tonal Analysis -- sarcasm, deadpan, hyperbole, deflection
4. VADUG Result -- 5-axis emotional coordinate
5. Entropy Check -- trajectory analysis for neutral detection
6. Grade + Response -- personality-shaped emotional response

## The 5 Dimensions (VADUG)

| Axis | Range | Meaning |
|------|-------|---------|
| V (Valence) | 0-255 | Negative to Positive |
| A (Arousal) | 0-255 | Calm to Intense |
| D (Dominance) | 0-255 | Helpless to In Control |
| U (Urgency) | 0-255 | No Rush to Critical |
| G (Gravity) | 0-255 | Crushing/Sinking to Floating/Soaring |

## Engine Stats (v0.14.1)

- 46,101 calibrated word forces
- 99 idioms (17 crisis, 22 gravity metaphors)
- 31 context modifiers (chameleon words)
- 24 intensity ramps with multi-word decay
- 55 relationship gravity words
- Exponential decay physics (spike + recovery)
- Entropy-based neutral detection
- 0.3ms/sentence on CPU, 2.3 MB total

## Personality Presets

- **Therapist**: High agreeableness, low assertiveness, safety-first
- **Best Friend**: Playful, agreeable, curious
- **Drill Sergeant**: Maximum assertiveness, zero playfulness
- **Stoic**: Low suggestibility, measured responses
- **Empath**: Maximum agreeableness, high curiosity
- **Chaos Goblin**: High playfulness, low safety, maximum curiosity
