# Clanker-Micro: 7D Emotional Coordinate Model

**22.6M parameters | 7 dimensions | 0.15ms/sentence | ~300KB engine**

Clanker-Micro predicts 7 continuous emotional coordinates (VADUGWI) from English text. Each sentence maps to a point in 7-dimensional emotional space, producing granular readings beyond what discrete classification provides.

**Every output can be traced through explicit, deterministic transformations, with each contributing factor visible and attributable.** You can ask WHY and get a real answer.

Transformers give fluency. Rule systems give control. Clanker is the bridge that lets you combine them -- fluency, control, and explainability -- instead of choosing one.

---

## Dimensions

| Dim | Name | Scale | What It Captures |
|-----|------|-------|-----------------|
| **V** | Valence | 0-255 | Positive ↔ negative (128 = neutral) |
| **A** | Arousal | 0-255 | Calm ↔ intense |
| **D** | Dominance | 0-255 | Helpless ↔ in control |
| **U** | Urgency | 0-255 | No time pressure ↔ critical |
| **G** | Gravity | 0-255 | Crushing weight ↔ floating |
| **W** | Self-Worth | 0-255 | Shattered ↔ strong self-assessment |
| **I** | Intent | 0-255 | Withdraw (0) ↔ deflect (64) ↔ neutral (128) ↔ connect (192) ↔ control (255) |

Total state space: 256^7 = 72 quadrillion unique emotional coordinates.

---

## Benchmarks

### Academic Datasets (external, not used in training)

| Dataset | Clanker V5 | VADER | Notes |
|---------|-----------|-------|-------|
| **SST-2** (Stanford Sentiment, movie reviews) | 69.0% | 55.7% | 872 validation sentences |
| **GoEmotions** (Google, Reddit comments) | 75.5% | 60.6% | 1,000 test samples, binary pos/neg |
| **TweetEval** (Twitter sentiment) | 65.5% | 74.1% | 1,000 test samples, excluding neutral |

SST-2 and GoEmotions are scored cold -- no training data from these datasets was used. TweetEval contains Twitter-specific formatting (handles, hashtags, URLs) that the engine does not currently process.

### Frontier AI Consensus Benchmark

Tested against consensus of 4 frontier AI models (Gemini, Claude Opus, GPT-4, Grok) on 521 Fallout 76 dialogue lines. Each model independently graded every sentence on all 7 VADUGWI dimensions.

| Metric | Score |
|--------|-------|
| Agreement with strong consensus (3+ of 4 agree) | **76.3%** on 131 high-confidence sentences |
| Closest alignment | Gemini (71%) and Claude Opus (70%) |
| Grading style | Reads emotional subtext (nuanced camp), not neutral-default (conservative camp) |

For comparison, Gemini and Grok only agree with each other 58% of the time. GPT-4 and Grok agree 88% but both default neutral on ambiguous lines. The engine reads more like Gemini and Claude -- the graders that detect sarcasm, condescension, and passive aggression.

### Internal Benchmarks

| Test | Score | Details |
|------|-------|---------|
| Core categories (8 types) | 100% | 64/64 -- crisis, positive, sarcasm, bravado, relationship, fight, internet, body |
| Permanent test suite | 100% | 630/630 accumulated test sentences |
| Novel sentence batches | 100% | 10/10 per run (randomly generated) |
| Hard novel stress test | 80% | 40/50 diverse real-world sentences |
| Essay emotional arcs | 85.8% | 103/120 across 8 essay types |
| Crisis recall | 97.3% | 177/182 crisis sentences detected |
| Crisis false positive | 0.5% | 1/207 safe sentences incorrectly flagged |

### Speed and Size

| Metric | Value |
|--------|-------|
| Inference speed | ~0.15ms/sentence (~10,000 sentences/sec on CPU) |
| Engine size | ~300KB (no GPU required) |
| Model size | 87MB (22.6M parameters) |

---

## How It Works (Overview)

The engine treats language as physics. Words carry emotional mass and exert force on neighboring words through proximity fields. Sentence structure is analyzed through pattern recognition -- role sequences form recognizable configurations that modify the emotional coordinates.

The system resolves directional force between entities: "he hit me" and "i hit him" produce different dominance and self-worth scores because the force flows in opposite directions.

26 structural patterns are detected, including sarcasm inversion, bravado masking, crisis indicators, betrayal, self-nullification, accountability, and withdrawal.

---

## Example Outputs

### Self-Worth Detection

| Sentence | V | A | D | U | G | W | I |
|----------|---|---|---|---|---|---|---|
| "i hate this job" | 90 | 162 | 158 | 49 | 160 | 95 | 128 |
| "i hate myself" | 85 | 155 | 155 | 32 | 154 | 92 | 19 |

Same valence. Different self-worth. Different intent (neutral vs withdraw).

### Directional Force

| Sentence | V | D | W | I | Reading |
|----------|---|---|---|---|---------|
| "he hit me" | 99 | 117 | 106 | 57 | Being hit -- D and W drop |
| "i hit him" | 105 | 114 | 111 | 231 | Hitting -- control intent |
| "she cheated on me" | 35 | 83 | 79 | 49 | Betrayal -- W collapses |

### Structural Patterns

| Sentence | V | Pattern | What Was Detected |
|----------|---|---------|-------------------|
| "haha yeah im totally okay" | 122 | BRAVADO | Overcompensation mask |
| "oh great another meeting" | 119 | SARCASM_INVERSION | Positive words, negative intent |
| "my wife cheated with my best friend" | 0 | BETRAYAL | Intimate trust weaponized |
| "im a burden to everyone" | 86 | SELF_NULLIFY | User calculating self as obstruction |
| "my father never once said he was proud" | 120 | WITHHELD_POSITIVE | Positive emotion never expressed |

### Negation and Absence

| Sentence | V | Reading |
|----------|---|---------|
| "i stopped smoking" | 136 | Positive -- bad habit ceased |
| "she stopped loving me" | 117 | Negative -- love ceased |
| "the pain stopped" | 143 | Positive -- suffering ended |
| "i havent had a panic attack in a month" | 136 | Positive -- progress report |

### Accountability vs Deflection

| Sentence | V | D | I | Reading |
|----------|---|---|---|---------|
| "i was wrong and im sorry" | 106 | 102 | 177 | Connect -- taking accountability |
| "it wasnt my fault" | 118 | 121 | 85 | Deflect -- rejecting blame |
| "i hate myself" | 85 | 155 | 19 | Withdraw -- self-attack |
| "shut up" | 116 | 141 | 210 | Control -- commanding silence |

---

## Conversation State Transition (A + B = C)

The engine supports forward and backward emotional state computation:

- **Forward:** Given receiver state A and message B, compute resulting state C
- **Backward:** Given current state A and desired state C, find message characteristics B that achieve it

State transitions account for force direction:
- CONTROL intent messages drop the receiver's dominance and self-worth
- CONNECT intent messages lift the receiver's self-worth
- WITHDRAW intent increases the receiver's emotional weight

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Backbone | GPT-2 (256 embed, 12 layers, 8 heads) |
| Parameters | 22.6M |
| Output heads | 3: word roles (26 classes), sentence patterns (26 patterns), VADUGWI (7D) |
| Training data | 141K sentences (EmpatheticDialogues + curated sources) |
| Role accuracy | 66.5% |
| Pattern accuracy | 98.7% |
| VADUGWI MAE | 2.19 (mean absolute error on 0-255 scale) |

---

## Training Data Sources

- [EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) (Facebook Research, 25K conversations)
- [EmoBank](https://github.com/JULIELab/EmoBank) (JULIE Lab, 10K VAD-annotated sentences)
- [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (Google Research, 58K Reddit comments)
- Curated crisis, sarcasm, and relationship sentences from clinical field experience

Bayesian vocabulary calibration was performed against the EmpatheticDialogues corpus (30K sentence sample) to empirically ground word-level emotional weights.

---

## Limitations

- **English only.** The dimensional framework (VADUGWI) is language-agnostic, but the vocabulary and structural patterns are English-specific. Multilingual support would require per-language vocabulary files.
- **Slang and hyperbole.** "I literally died laughing" reads literally per-sentence. Conversation context (state tracking across messages) resolves these cases but is not available in single-sentence mode.
- **Academic sentiment benchmarks.** SST-2 measures movie review polarity, which is a related but different task from emotional coordinate prediction. The engine was not trained on SST-2 data.
- **Tweet-specific formatting.** Twitter handles (@user), hashtags (#topic), and URLs are not processed, contributing to lower TweetEval scores.
- **The model approximates the engine.** Edge cases may produce different results between the 22.6M model and the full engine. The engine is the source of truth.

---

## Two Products, One Engine

### 1. Control Layer for Models
VADUGWI scoring that any language model can use. Score text, track emotional state across conversations, steer responses toward target outcomes with the A+B=C solver. The engine reads the force. The model generates the words. The bridge between fluency and explainability.

### 2. Emotional Baseline Assessment
A 35-question situational test drawn from a pool of 630 calibrated probes. Measures emotional interpretation bias across all 7 dimensions without asking direct clinical questions. "Your partner is quiet on the drive home" -- the user's rating reveals their bias, not the sentence's meaning. Detects negative interpretation bias, self-worth patterns, control sensitivity, intent projection, and emotional range compression. Non-threatening. No "do you want to die." Just "how does this situation make you feel, 1-10."

### 3. NPC Emotional Systems
Game characters with persistent 7D emotional state. Each interaction shifts the NPC's VADUGWI through state_transition. NPCs that remember being hurt (W drops), that withdraw after betrayal (I shifts), that build trust over time (W rises). The ACEs/PCEs framework initializes NPC personality from backstory -- trauma shapes how they receive emotional force.

---

## Intended Use

- **Control layer for LLMs**: emotional context, conversation steering, response selection
- **Crisis detection**: 97.3% recall with step-level attribution for clinicians
- **Emotional assessment**: non-threatening baseline test, trauma/bias detection
- **NPC systems**: persistent emotional state, personality from backstory
- **Conversation tracking**: multi-turn trajectory, self-worth monitoring
- **Therapy support**: session-over-session W trajectory, accountability detection
- **Research**: dimensional emotion modeling, implicit bias measurement

---

## How To Use

```python
from clanker import score

result = score("i am nothing without you")
# VADUGWI(v=80, a=125, d=98, u=17, g=136, w=88, i=27)
# Pattern: SELF_NULLIFY
# Reading: deeply negative, low self-worth, withdrawing
```

### As emotional context for an LLM:

```python
vadugwi = score(user_message)

system_prompt = f"""The user's emotional state:
  Valence: {vadugwi.v} ({'negative' if vadugwi.v < 118 else 'neutral' if vadugwi.v < 138 else 'positive'})
  Self-Worth: {vadugwi.w} ({'low' if vadugwi.w < 100 else 'stable' if vadugwi.w < 148 else 'healthy'})
  Intent: {vadugwi.i} ({'withdrawing' if vadugwi.i < 40 else 'deflecting' if vadugwi.i < 80 else 'neutral' if vadugwi.i < 148 else 'connecting' if vadugwi.i < 200 else 'controlling'})

Respond with appropriate care and tone."""
```

---

## The VADUGWI Standard

VADUGWI is proposed as an open standard for emotional coordinates in AI systems. The 7 dimensions, the 0-255 scale, and the interpretation framework are public. Any system can produce or consume VADUGWI scores.

Full specification: [SPEC.md](SPEC.md)

---

*Built from Therapeutic Crisis Intervention field experience.*

*Jerry Mares | [deucebucket](https://github.com/deucebucket)*
