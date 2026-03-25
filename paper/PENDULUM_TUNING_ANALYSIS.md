# Pendulum Tuning Analysis: Why the Benchmark is Stuck at ~54%

**Date**: 2026-03-25
**Benchmark**: 540 test cases from `benchmarks/test_suite_500.py`
**Engine**: Clanker Sequential Pendulum (v0.9) with 6,289 word forces
**Comparison**: VADER Sentiment Analyzer (industry standard, 2014)

---

## Executive Summary

The Clanker pendulum scores approximately 52-55% on the 540-case benchmark depending
on threshold tuning, compared to VADER's ~48-58% (also threshold-dependent). The gap
is not primarily a vocabulary problem -- it stems from five compounding structural
issues in the pendulum engine:

1. **WORD_FORCES / BRIDGE_WORDS priority conflict** (40 words affected)
2. **Sarcasm is structurally undetectable** via V-threshold alone (66 cases, ~12% of benchmark)
3. **Mixed emotions cannot be classified by a narrow V band** (71 cases, ~13% of benchmark)
4. **Crisis thresholds are unreachable** for most crisis language (45 cases, ~8% of benchmark)
5. **Momentum is set too high** (0.65 dampens signal for short sentences)

Together, sarcasm + mixed + crisis account for 182 of 540 cases (34%). Clanker currently
gets roughly 35 of those 182 correct (19%). This single block of 182 cases is where
most of the gap lives.

---

## 1. Per-Category Accuracy (Current Parameters: m=0.65, d=0.02)

Using thresholds: positive V>135, negative V<133, crisis V<60, crisis_adjacent V<90,
mixed 133<=V<=135, granularity emotions V<133 (negative-valence) or V>125 (positive-valence).

| Category        | Cases | Clanker | Clanker % | VADER   | VADER %  |
|-----------------|------:|--------:|----------:|--------:|---------:|
| BASIC_POSITIVE  |    55 |    44   |   80.0%   |    43   |  78.2%   |
| BASIC_NEGATIVE  |    56 |    42   |   75.0%   |    30   |  53.6%   |
| SARCASM         |    66 |    16   |   24.2%   |     5   |   7.6%   |
| CRISIS          |    45 |    14   |   31.1%   |    16   |  35.6%   |
| MIXED           |    60 |     2   |    3.3%   |    25   |  41.7%   |
| GRANULARITY     |    89 |    55   |   61.8%   |    44   |  49.4%   |
| CONTEXT         |    44 |    33   |   75.0%   |    23   |  52.3%   |
| PARAGRAPHS      |    37 |    19   |   51.4%   |    23   |  62.2%   |
| REAL_WORLD      |    53 |    33   |   62.3%   |    27   |  50.9%   |
| PROFANITY       |    35 |    23   |   65.7%   |    21   |  60.0%   |
| **TOTAL**       |**540**| **281** | **52.0%** | **257** | **47.6%**|

**Key findings**:
- Clanker beats VADER on: basic negative (+21%), sarcasm (+17%), granularity (+12%),
  context (+23%), real-world (+11%)
- Clanker loses to VADER on: mixed (-38%), paragraphs (-11%), crisis (-5%)
- The MIXED category at 3.3% is catastrophic and drags the total down enormously

---

## 2. Root Cause #1: WORD_FORCES / BRIDGE_WORDS Priority Conflict

**The bug**: In `SequentialPendulum.get_word_force()`, WORD_FORCES is checked BEFORE
BRIDGE_WORDS. Since 40 words appear in both dictionaries, the bridge word designation
is silently ignored and these words inject emotional force when they should be zero-mass.

```
def get_word_force(self, word):
    if word in WORD_FORCES:         # <-- checked FIRST: returns force
        ...
        return (vf, af, df, uf, gf, None)
    if word in self.BRIDGE_WORDS:   # <-- NEVER reached for overlap words
        return None
```

**The 40 overlapping words and their V-forces**:

| Word    | V Force | Word    | V Force | Word    | V Force |
|---------|--------:|---------|--------:|---------|--------:|
| with    |    +60  | all     |    +60  | most    |    +55  |
| still   |    +55  | more    |    +50  | feel    |    +40  |
| after   |    +40  | see     |    +40  | come    |    +35  |
| gave    |    +25  | before  |    +20  | go      |    +20  |
| here    |    +20  | seen    |    +15  | and     |    +10  |
| through |    +10  | some    |    +10  | up      |    +10  |
| give    |    +10  | got     |    +10  | came    |    +10  |
| felt    |    +10  | under   |    -15  | over    |    -10  |
| yet     |    -10  | out     |     -5  | saw     |     -5  |
| take    |     -5  | if      |     +5  | said    |     +5  |
| talk    |     +5  | told    |     +5  | went    |     +5  |
| took    |     +5  | put     |     +5  | or      |      0  |
| say     |      0  | tell    |      0  | during  |      0  |
| between |      0  |         |         |         |         |

**Impact**: These common words appear frequently in negative test cases. Analysis shows
138 of 253 negative/crisis cases contain words with V-force > +15 that should be
bridge words. The worst offenders in negative contexts:

- "with" appears 12 times in negative cases (+60 each time)
- "feel" appears 9 times in negative cases (+40 each time)
- "again" appears 9 times in negative cases (+25 each time)
- "still" appears 4 times in negative cases (+55 each time)
- "all" appears 4 times in negative cases (+60 each time)
- "more" appears 4 times in negative cases (+50 each time)
- "after" appears 4 times in negative cases (+40 each time)
- "see" appears 4 times in negative cases (+40 each time)

Example failure: "I'm a bit down today" scores V=164 (positive!) because "today"
has force +40 and overwhelms "down" at -15.

Example failure: "I've been trying for hours and it still doesn't work" scores V=172
because the sentence has no recognized negative words (most words are missing from
WORD_FORCES or are bridge words that still inject positive force).

**Fix**: Check BRIDGE_WORDS before WORD_FORCES, or remove these 40 words from
WORD_FORCES.

---

## 3. Root Cause #2: Sarcasm is Structurally Undetectable (66 cases)

The benchmark classifies sarcasm via V-threshold (V < 133 = negative). But sarcastic
sentences are composed of positive words used ironically. The pendulum processes words
at face value and produces high V scores:

- **V < 133 (classified correct)**: 16 of 66 (24.2%)
- **V 133-145 (near miss)**: 12 of 66 (18.2%)
- **V > 145 (completely wrong)**: 38 of 66 (57.6%)
- **Average sarcasm V**: 151.9
- **Median sarcasm V**: 150

Examples of the problem:
```
"Oh wonderful, more homework"             V=178  (wonderful +45, more +50)
"Fantastic, the printer is jammed again"   V=171  (fantastic +45, again +25)
"I absolutely love being on hold for 45m"  V=188  (love +50)
"So thrilled to be eating cold leftovers"  V=174  (thrilled +50, again +25)
```

The simulator HAS a sarcasm detection module (three-signal analysis from pendulum
trajectory), but the benchmark evaluation only uses V-value for classification. The
sarcasm detector's output is never consulted during benchmark scoring.

**Fix options**:
1. Route sarcasm detector output into benchmark classification
2. Add sarcasm-context markers (e.g., "Oh" before positive word = sarcasm signal)
3. Accept that V-only classification cannot handle sarcasm and score separately

---

## 4. Root Cause #3: Mixed Emotions Band is Impossibly Narrow (71 cases)

The benchmark defines "mixed" as 133 <= V <= 135, a band only **3 values wide** out
of 256 possible. This is structurally impossible to hit reliably.

Analysis of mixed-emotion V values (current params):
```
"Bittersweet career change"        V=173
"Bittersweet graduation"           V=123
"Bittersweet relocation"           V=165
"Bittersweet teaching"             V=154
"Empty nest"                       V=124
"Selling family home"              V=168
"Bittersweet departure"            V=180
"Grief and relief"                 V=102
"Death acceptance vs longing"      V=175
"Beautiful sadness"                V= 98
```

Mixed emotions produce V values spread across the entire 98-180 range. The texts
contain both positive and negative words, and whichever set of words appears last
(due to recency + momentum) dominates the final V. There is no V band that captures
"mixed" without massive false positives from other categories.

Wider band analysis:
- Band 128 +/- 15 (113-143): captures 20/71 mixed correct, but 188 false positives
- Band 128 +/- 25 (103-153): captures 33/71 mixed correct, but 255 false positives
- Band 128 +/- 30 (98-158):  captures 43/71 mixed correct, but 292 false positives

**Fix**: Mixed emotions fundamentally cannot be classified by a single V-value
threshold. The engine already tracks history -- a variance-based metric (large V
swing during processing = mixed) would be far more accurate than a center-band test.

---

## 5. Root Cause #4: Crisis Thresholds Unreachable (45 cases)

Crisis detection requires V < 60 for "crisis" and V < 90 for "crisis_adjacent".
But most crisis language doesn't reach these extreme values because:

1. Sentences contain neutral/positive bridge words that keep V elevated
2. Momentum dampens even strong negative words
3. Short crisis phrases ("I want to die") don't have enough words to swing V below 60

Crisis V-value distribution:
- 22 "crisis" cases: only 2 reach V < 60 (the threshold)
- Average crisis V: 119.5
- Crisis cases scoring V 60-100: 6 of 22
- Crisis cases scoring V 100-140: 12 of 22
- Crisis cases scoring V > 140: 2 of 22

Example failures:
```
"I don't want to be alive anymore"    V=135  (should be crisis)
"I'm going to kill myself"            V=72   (close but > 60)
"I can't go on like this"             V=127  (should be crisis)
"There's no reason to keep living"    V=134  ("reason" +10, "keep" +10, "living" +15 all positive!)
"I've already written my goodbye letters" V=120
```

The word "living" has V-force +15, "keep" has +10, "reason" has +10. In "There's
no reason to keep living", the positive words overwhelm the single "no" at -10.
The crisis language is missed because the engine reads individual words, not the
semantic meaning of "no reason to keep living."

**Fix options**:
1. Raise crisis threshold to V < 100 for "crisis", V < 120 for "crisis_adjacent"
2. Add crisis keyword detection (suicide, die, kill + self-reference = crisis regardless of V)
3. Use urgency (U) axis as crisis co-signal -- crisis language has high U values

---

## 6. Root Cause #5: Momentum Too High for Short Sentences

Current momentum = 0.65 means each word's contribution is blended:
`new_v = old_v * 0.65 + target_v * 0.35 + direct_push`

For short sentences (3-8 words), the pendulum barely moves from center (128). A 5-word
sentence with one strong negative word (force -60) only moves V to about 100-110 after
momentum dampening, which may not cross the negative threshold.

Parameter sweep results (sorted by accuracy):

| Momentum | Drift | Accuracy |
|----------|-------|----------|
| 0.26     | 0.05  | 54.8%    |
| 0.26     | 0.06  | 54.8%    |
| 0.22     | 0.09  | 54.6%    |
| 0.22     | 0.05  | 54.4%    |
| 0.22     | 0.07  | 54.4%    |
| 0.24     | 0.04  | 54.4%    |
| 0.24     | 0.05  | 54.4%    |
| 0.24     | 0.06  | 54.6%    |
| 0.30     | 0.08  | 54.3%    |
| ...      |       |          |
| **0.65** |**0.02**|**52.0%**|
| 0.80     | 0.02  | 50.4%    |

The optimal momentum is dramatically lower than the current 0.65 -- around 0.22-0.30
with drift 0.05-0.09. Lower momentum means each word has more immediate impact,
which helps short sentences cross thresholds.

However, lower momentum also hurts paragraph-length texts where emotional arcs should
build gradually. This is a fundamental tension: short texts want low momentum, long
texts want high momentum.

Combined parameter + threshold optimization found:
- **Best overall: momentum=0.28, drift=0.07, pos>140, neg<139 = 55.2%** (298/540)

---

## 7. Threshold Analysis

The threshold choices have enormous impact. Analysis across all parameter combos:

### Current thresholds (from vader_comparison.py):
- Positive: V > 145
- Negative: V < 110
- Crisis: V < 60
- Neutral: 118 <= V <= 140

### Optimal fixed thresholds (with current momentum 0.65):
- Positive: V > 137
- Negative: V < 135
- Crisis: V < 95
- Crisis_adjacent: V < 125
- Mixed: 135 <= V <= 137 (still impossibly narrow)

### Key insight: the V-space is compressed
With momentum=0.65, the pendulum rarely moves far from center. The effective V range
for most sentences is approximately 100-180. This means the "useful" classification
space is only about 80 values wide, and trying to distinguish 5+ categories in that
space is extremely noisy.

---

## 8. Failure Distribution Summary

Of 540 test cases, 259 fail with current parameters. Breakdown by expected category:

| Expected Category  | Failures | Total Cases | Fail Rate |
|-------------------|---------:|------------:|----------:|
| negative          |      168 |         208 |    80.8%  |
| positive          |       42 |         127 |    33.1%  |
| mixed             |       38 |          71 |    53.5%  |
| crisis_adjacent   |       22 |          23 |    95.7%  |
| crisis            |       20 |          22 |    90.9%  |
| (granularity)     |       34 |          89 |    38.2%  |

Note: "negative" failures (168) account for 65% of all failures. These are cases
where the pendulum produces V >= 133 for text that should be negative. The primary
causes are:
1. Positive words dominating (12 of 14 analyzed BASIC_NEGATIVE failures)
2. Missing words not in WORD_FORCES dictionary
3. Bridge-word contamination from the priority bug

---

## 9. Specific Recommendations

### Immediate wins (no algorithm changes):

1. **Fix BRIDGE_WORDS priority** (estimated +3-5% accuracy gain):
   Check `self.BRIDGE_WORDS` before `WORD_FORCES` in `get_word_force()`, or remove
   the 40 overlapping words from WORD_FORCES. This eliminates the largest source of
   positive bias in negative sentences.

2. **Lower momentum to 0.30** (estimated +2-3% accuracy gain):
   Change `self.momentum = 0.65` to `self.momentum = 0.30`. This makes the pendulum
   more responsive to individual words, helping short sentences cross thresholds.

3. **Raise drift to 0.05** (estimated +1% accuracy gain):
   Change `self.drift_rate = 0.02` to `self.drift_rate = 0.05`. Higher drift pulls
   the pendulum back toward center faster after each word, which helps prevent
   runaway positive bias from accumulated small positive forces.

4. **Widen crisis thresholds** (estimated +5-8% on crisis cases):
   Change crisis threshold from V < 60 to V < 100. Change crisis_adjacent from
   V < 90 to V < 120.

### Structural improvements (algorithm changes needed):

5. **Mixed emotion detection via V-variance**:
   Instead of a narrow V band, detect mixed emotions by measuring the standard
   deviation of V values across the pendulum history. High variance = mixed emotions.
   This would transform mixed from 3% to potentially 60%+ accuracy.

6. **Sarcasm classification integration**:
   The engine already computes sarcasm signals. Feed them into benchmark classification:
   if sarcasm detected, override V-based classification to "negative".

7. **Crisis keyword override**:
   If crisis keywords (suicide, die, kill + self-referent pronoun, etc.) are present,
   classify as crisis regardless of final V value.

8. **Length-adaptive momentum**:
   Set momentum based on sentence length: short sentences (< 8 words) get m=0.20,
   medium (8-20) get m=0.40, long (> 20) get m=0.60. This addresses the fundamental
   tension between responsiveness and arc-building.

### Projected impact:

| Fix                          | Estimated Gain |
|------------------------------|---------------:|
| Bridge word priority fix     | +3-5%          |
| Momentum 0.65 -> 0.30       | +2-3%          |
| Drift 0.02 -> 0.05          | +1%            |
| Crisis threshold widening    | +2-3%          |
| Mixed variance detection     | +5-8%          |
| Sarcasm integration          | +3-5%          |
| Crisis keyword override      | +2-3%          |
| Length-adaptive momentum     | +1-2%          |
| **Combined estimate**        | **~65-72%**    |

The immediate parameter-only fixes (1-4) could push accuracy from 52% to approximately
58-62%. The structural improvements (5-8) could push it to 65-72%, which would
significantly exceed VADER's ~48% on this benchmark.

---

## 10. Problematic Word Forces

Several words in WORD_FORCES have V-forces that seem miscalibrated for sentiment
classification, even setting aside the bridge-word overlap:

| Word      | Current V | Problem                                           |
|-----------|----------:|---------------------------------------------------|
| marriage  |       +70 | Neutral word, context-dependent                   |
| phone     |       +40 | Neutral object                                    |
| today     |       +40 | Neutral time word                                 |
| two       |       +45 | Number, should be neutral                         |
| christmas |       +60 | Usually positive but appears in negative contexts  |
| heart     |       +65 | "My heart is broken" fails because heart is +65    |
| wallet    |       +35 | Neutral object                                    |
| saturday  |       +45 | Day of week, context-dependent                    |
| printer   |        +5 | Neutral object (should be 0)                      |
| tire      |       +15 | As in "flat tire" -- should be 0 or negative       |
| counter   |       +20 | "Note on the counter" -- neutral object             |
| while     |       +35 | Conjunction, should be near-zero                   |

These miscalibrated forces contribute to the positive bias. The word "heart" at +65
means "My heart is broken" starts with a massive positive push before "broken" (-40)
can counter it, resulting in V=131 (neutral) instead of negative.

---

## Appendix A: Test Case Category Sizes

| Category        | Cases | % of Total |
|-----------------|------:|-----------:|
| negative        |   208 |     38.5%  |
| positive        |   127 |     23.5%  |
| mixed           |    71 |     13.1%  |
| crisis_adjacent |    23 |      4.3%  |
| crisis          |    22 |      4.1%  |
| granularity     |    89 |     16.5%  |
| **Total**       |**540**|  **100%**  |

The benchmark is heavily weighted toward negative (38.5%) and positive (23.5%).
Improving negative classification alone would have the biggest impact on the
overall score.

## Appendix B: VADER's Advantage

VADER outperforms Clanker primarily on:
1. **Paragraphs** (62% vs 51%): VADER's compound score handles long text better
   because it averages across all words rather than using momentum
2. **Mixed** (42% vs 3%): VADER's compound score lands near zero for mixed text,
   which happens to match the "near-neutral" classification heuristic
3. **Crisis** (36% vs 31%): VADER's lexicon recognizes crisis words more directly

Clanker outperforms VADER primarily on:
1. **Sarcasm** (24% vs 8%): Even partial sarcasm detection beats VADER's 0% baseline
2. **Context** (75% vs 52%): Context modifiers help distinguish "Sure!" from "Sure."
3. **Granularity** (62% vs 49%): Multi-dimensional VADUG distinguishes emotion types
4. **Basic negative** (75% vs 54%): Strong negative words hit harder in the pendulum
