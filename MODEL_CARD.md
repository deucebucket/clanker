# Clanker-Micro: 6D Emotional Coordinate Model

**22.6M parameters | 6 dimensions | 0.15ms/sentence | 300KB engine**

Clanker-Micro predicts 6 continuous emotional coordinates (VADUGW) from English text. Unlike sentiment classifiers that output discrete buckets (positive/negative/angry/sad), Clanker produces a precise point in 6-dimensional emotional space.

## What It Measures

| Dimension | Name | Scale | What It Captures |
|-----------|------|-------|-----------------|
| **V** | Valence | 0-255 | How positive or negative (128 = neutral) |
| **A** | Arousal | 0-255 | Energy level -- calm to intense |
| **D** | Dominance | 0-255 | Power/control -- helpless to in command |
| **U** | Urgency | 0-255 | Time pressure -- none to critical |
| **G** | Gravity | 0-255 | Emotional weight -- crushing to floating |
| **W** | Self-Worth | 0-255 | Self-assessment -- shattered to strong |

**Total state space:** 256^6 = **281 trillion** unique emotional coordinates.

## Why 6 Buckets Aren't Enough

A standard emotion classifier puts "I hate this job" and "I hate myself" in the same bucket: **anger** or **sadness**.

Clanker sees them differently:

| Sentence | V | W | Reading |
|----------|---|---|---------|
| "I hate this job" | 90 | 95 | Negative, but self-worth intact -- external frustration |
| "I hate myself" | 85 | 92 | Negative AND self-worth dropping -- self-directed |

The **W dimension** tells you whether someone is blaming the world or blaming themselves. No other model captures this.

## Direction Matters

"He hit me" and "I hit him" are the same words. Every sentiment model scores them identically.

Clanker resolves WHO does WHAT to WHOM:

| Sentence | V | D | W | Reading |
|----------|---|---|---|---------|
| "he hit me" | 99 | 117 | 106 | Negative -- I'm the target, D drops |
| "i hit him" | 105 | 114 | 111 | Negative -- I'm the actor, D holds |
| "she cheated on me" | 35 | 83 | 79 | Devastating -- betrayal + self-worth collapse |
| "i cheated on her" | 65 | 79 | 94 | Bad -- but self-worth less affected |

## Structural Pattern Recognition

The model detects 26 structural patterns -- not from keywords, but from the arrangement of word roles in a sentence. Like a chess player reading piece positions, not memorizing specific games.

| Sentence | Pattern | V | W | What Was Detected |
|----------|---------|---|---|-------------------|
| "haha yeah im totally okay" | BRAVADO | 122 | 152 | Overcompensation mask -- protesting too much |
| "oh great another meeting" | SARCASM_INVERSION | 119 | 128 | Positive words, negative intent |
| "my wife cheated on me with my best friend" | BETRAYAL | 0 | 80 | Intimate trust weaponized |
| "nobody would even notice" | -- | 127 | 128 | Isolation -- zero people would observe absence |
| "im a burden to everyone" | SELF_NULLIFY | 86 | 96 | User calculating self as obstruction |
| "my father never once said he was proud" | WITHHELD_POSITIVE | 120 | 117 | Positive emotion that was never expressed |

## Negation & Context Resolution

"Stopped" flips based on what stopped. No other model handles this:

| Sentence | V | Reading |
|----------|---|---------|
| "i stopped smoking" | 136 | Positive -- bad habit ended |
| "she stopped loving me" | 117 | Negative -- love ended |
| "the pain stopped" | 143 | Positive -- suffering ended |
| "i stopped trying" | 125 | Negative -- effort ended |

"Without" changes meaning based on what's absent:

| Sentence | V | Pattern | Reading |
|----------|---|---------|---------|
| "i can finally afford groceries without stress" | 135 | RELIEF_ABSENCE | Positive -- stress is gone |
| "they left without saying goodbye" | 122 | FINALITY | Negative -- closure was denied |
| "i havent had a panic attack in a month" | 136 | RELIEF_ABSENCE | Positive -- progress report |

## Self-Worth Trajectory

W tracks across a conversation. Each message updates the running self-assessment:

| Message | V | W | Trajectory |
|---------|---|---|-----------|
| "things have been hard lately" | 122 | 128 | Neutral -- reporting situation |
| "i just feel like im failing at everything" | 122 | 120 | W dropping -- self-assessment starting to erode |
| "i am nothing" | 75 | 87 | W collapsed -- self-worth destroyed |
| "maybe i can try again tomorrow" | 130 | 129 | W recovering -- resilience signal |

## Model Details

- **Architecture:** GPT-2 backbone (256 embed, 12 layers, 8 heads) + 3 task heads
- **Parameters:** 22.6M
- **Output heads:**
  - Word roles: 26 structural role classes (67.5% accuracy)
  - Sentence patterns: 26 pattern detectors (98.9% accuracy)
  - VADUGW: 6 continuous coordinates (MAE 2.3 on 0-255 scale)
- **Training data:** 141K sentences from EmpatheticDialogues, EmoBank, and curated sources
- **Inference speed:** ~0.15ms/sentence on CPU
- **Size:** 87MB weights

## Datasets

Trained and calibrated against:
- [EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) (Facebook, 25K conversations)
- [EmoBank](https://github.com/JULIELab/EmoBank) (JULIE Lab, 10K VAD-scored sentences)
- [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (Google, 58K Reddit comments)
- Curated crisis, sarcasm, and relationship sentences from clinical TCI field experience

## Intended Use

- **Emotional layer for LLMs:** Attach VADUGW scores to messages so models understand emotional context
- **Crisis detection:** 97.3% recall on crisis language, 0.5% false positive rate
- **Conversation state tracking:** Monitor emotional trajectory across multi-turn dialogue
- **NPC emotional systems:** Game characters with persistent emotional state
- **Therapy/coaching support tools:** Track self-worth trajectory over sessions

## Limitations

- English only (the dimensional framework is language-agnostic, vocabulary is not)
- Slang/hyperbole requires conversation context ("i literally died laughing" reads literal per-sentence)
- Academic sentiment benchmarks (SST-2) measure a different task -- movie review polarity, not emotional coordinates
- The model approximates the engine -- edge cases may differ from engine-grade accuracy

## How To Use

```python
from clanker import score

result = score("i am nothing without you")
print(result)
# VADUGW(v=80, a=125, d=98, u=17, g=136, w=88)
# Pattern: SELF_NULLIFY
# Reading: deeply negative, low self-worth, conditional on relationship
```

### As emotional context for an LLM:

```python
vadugw = score(user_message)

system_prompt = f"""The user's emotional state:
  Valence: {vadugw.v} ({'negative' if vadugw.v < 118 else 'neutral' if vadugw.v < 138 else 'positive'})
  Self-Worth: {vadugw.w} ({'low' if vadugw.w < 100 else 'stable' if vadugw.w < 148 else 'healthy'})
  Dominance: {vadugw.d} ({'feels powerless' if vadugw.d < 100 else 'neutral'})

{'The user is directing negativity at themselves, not the situation.' if vadugw.w < vadugw.v else ''}
Respond with appropriate care."""
```

## The Standard

VADUGW is a proposed open standard for emotional coordinates in AI systems. The dimensions, the scale (0-255, 128=neutral), and the interpretation are public. Any system can produce or consume VADUGW scores.

The engine that produces the most accurate scores is proprietary. The standard is open.

Like RGB for color. The standard is free. The best camera is not.

---

*Built from TCI field experience. The math comes from watching real humans in crisis and recognizing the patterns no classifier catches.*

*Jerry Mares | [deucebucket](https://github.com/deucebucket)*
