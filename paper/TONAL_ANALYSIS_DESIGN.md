# Two-Pass Tonal Analysis: Design Document

**Status:** RESEARCH / DESIGN -- not yet implemented
**Date:** March 2026
**Author:** deucebucket, with assistance from Claude (Anthropic)
**Depends on:** SPEC.md (VADUG), ENGINE.md (Pendulum), CHUNKER_DESIGN.md (Arc Detection)
**Target:** Issue #38

---

## Abstract

The Clanker pendulum engine maps words to VADUG emotional coordinates. But it cannot distinguish sincere "I love you" from sarcastic "I love you," threatening "I'll eat you alive" from playful "I'll eat you alive," or genuine "great!" from deflated "great..." The current SarcasmDetector (v0.5.2) uses three hardcoded signals with fixed thresholds -- a start, but brittle and narrow.

This document proposes a **Two-Pass Tonal Analysis** system:

- **Pass 1** (existing): The pendulum produces VADUG coordinates and a word-by-word trajectory trace.
- **Pass 2** (new): A tonal analyzer examines the GEOMETRY of Pass 1's output -- the shape, correlation, oscillation, and contradictions in the trajectory -- and produces tonal coordinates that modify how the system interprets the emotional content.

The result is a compact tonal vector appended to the message header, enabling downstream systems to distinguish surface emotion from intended emotion without any machine learning component.

---

## Table of Contents

1. [Literature Review](#1-literature-review)
2. [Signal Analysis](#2-signal-analysis)
3. [Recommended Architecture](#3-recommended-architecture)
4. [Proposed Gate Pipeline](#4-proposed-gate-pipeline)
5. [Thought Tree Branching Rules](#5-thought-tree-branching-rules)
6. [False Positive Prevention](#6-false-positive-prevention)
7. [Tonal Coordinate Definition](#7-tonal-coordinate-definition)
8. [Integration Plan](#8-integration-plan)
9. [Test Cases for Validation](#9-test-cases-for-validation)
10. [Risks and Open Questions](#10-risks-and-open-questions)

---

## 1. Literature Review

### 1.1 Rule-Based Sarcasm Detection

The dominant approaches to sarcasm detection fall into three categories: supervised ML (requires labeled corpora), unsupervised ML (topic modeling, clustering), and rule-based (linguistic heuristics). Since Clanker's philosophy is "pure math, no ML," only rule-based approaches are directly relevant.

**Key findings from the literature:**

**Riloff et al. (2013) -- "Sarcasm as Contrast between a Positive Sentiment and Negative Situation."** This is the single most relevant paper. They demonstrated that sarcasm frequently manifests as a positive sentiment expression embedded in a negative situational context. Their rule-based bootstrapping approach achieved ~75% precision on Twitter data using only sentiment contrast patterns. This directly maps to what we can compute from VADUG trajectories: a positive word spike in an otherwise negative trajectory IS a computable contrast.

**Joshi, Sharma, & Bhattacharyya (2015) -- "Harnessing Context Incongruity for Sarcasm Detection."** They formalized the "incongruity" principle: sarcasm arises when there is a discrepancy between the expected and expressed sentiment in context. They measured incongruity using explicit (surface-level polarity clash) and implicit (semantic relatedness) features. Their explicit incongruity features are computable from our pendulum output; their implicit features would require word embeddings we do not have.

**Maynard & Greenwood (2014) -- "Who Cares about Sarcastic Tweets?"** Demonstrated that hashtag-based sarcasm markers (#sarcasm, #not) correlate with specific linguistic patterns: interjections ("oh", "wow"), hyperbolic intensifiers ("absolutely", "totally"), and punctuation patterns (ellipsis, exclamation marks after negative content). These lexical markers are cheap to detect.

**Gonzalez-Ibanez, Maynard, & Shah (2011)** found that emoticons and punctuation are unreliable sarcasm indicators in isolation, but gain reliability when combined with sentiment polarity shifts. This supports our multi-signal approach over any single-feature detector.

**VADER (Hutto & Gilbert, 2014)** is the closest existing system to our architecture -- a rule-based sentiment analyzer using a human-validated lexicon with intensity scores. VADER handles negation, intensifiers, and punctuation but has NO sarcasm detection capability. It classifies "This is just great..." as positive. This is exactly the gap we are filling.

### 1.2 Geometric/Trajectory Approaches

**Cambria et al. (2016) -- SenticNet and the "Hourglass of Emotions."** They proposed a dimensional model of affect with four dimensions (Pleasantness, Attention, Sensitivity, Aptitude) and analyzed how sentiment shifts across sentence segments. Their key insight: the TRAJECTORY of sentiment across a sentence is more informative than the endpoint for detecting irony. A sentence that oscillates rapidly between positive and negative poles is statistically more likely to be ironic than one that smoothly transitions.

**Thelwall (2017) -- SentiStrength.** A lexicon-based system that separately tracks positive and negative sentiment strength, recognizing that both can coexist. The separation of positive and negative channels is relevant to our approach: sarcasm often has HIGH positive channel AND HIGH negative channel simultaneously, while genuine mixed emotion has MODERATE both.

**Poria et al. (2016) -- "A Deeper Look into Sarcastic Tweets Using Deep Convolutional Neural Networks."** While ML-based, they identified that the most discriminative features for sarcasm were: (1) sentiment flip patterns within sentences, (2) presence of strong positive words in objectively negative contexts, and (3) exaggerated intensifiers. All three are computable from pendulum traces.

### 1.3 Valence-Arousal Correlation as Sincerity Signal

**Russell's Circumplex Model (1980)** predicts that valence and arousal should be correlated in specific ways for each emotion category. Genuine joy is high-V, moderate-to-high-A. Genuine anger is low-V, high-A. When these expected correlations BREAK -- high-V but flat-A (positive words with no energy behind them) -- it suggests the surface sentiment does not match the felt state.

**Barrett (2006) -- "Solving the Emotion Paradox."** Argues that emotions have characteristic signatures in dimensional space. Deviations from these signatures suggest that the expressed emotion is not the felt emotion. This is directly measurable from VADUG: we can define "expected A given V" curves and flag deviations.

**Scherer (2005) -- Component Process Model.** Proposes that emotions involve synchronized changes across multiple components (cognitive, physiological, motivational, motor expression, subjective feeling). When components are DE-SYNCHRONIZED, the emotion may be feigned or performed. In VADUG terms: genuine emotions produce correlated movement across axes; performed emotions may produce movement on V alone while A, D, and G remain flat.

### 1.4 The Gravity Dimension as Sincerity Discriminator

No existing literature uses a gravity/weight dimension for sarcasm detection -- this is novel to Clanker. However, the theoretical basis is strong:

- Genuine positive emotions LIFT (high G). "I'm so happy!" should produce V+, G+.
- Sarcastic positive words in negative context should NOT lift. "Oh great, another problem" should produce V+ on "great" but G remains low or sinks.
- The V/G correlation may be the single most powerful sincerity signal available from the pendulum.

This is an untested hypothesis but it follows directly from the gravity axis design philosophy documented in SPEC.md Section 9.2: Gravity captures "the physical weight of emotion." Feigned emotion lacks physical weight.

### 1.5 Summary of Applicable Literature

| Source | Key Finding | Applicable to VADUG? |
|--------|------------|---------------------|
| Riloff et al. 2013 | Sarcasm = positive sentiment + negative situation | YES -- trajectory contrast is computable |
| Joshi et al. 2015 | Explicit incongruity is sufficient for ~70% detection | YES -- surface polarity clash |
| Maynard & Greenwood 2014 | Lexical markers + sentiment shift > either alone | YES -- can combine lexical + trajectory |
| Russell 1980 | V/A correlation predicts emotion category | YES -- correlation breaks = sincerity signal |
| Scherer 2005 | Genuine emotion = synchronized multi-axis movement | YES -- axis correlation is computable |
| Cambria et al. 2016 | Oscillation frequency correlates with irony | YES -- pendulum oscillation is trajectory data |
| Thelwall 2017 | Separate pos/neg channels reveal mixed emotions | PARTIAL -- VADUG is single-V, but trajectory shows both |

---

## 2. Signal Analysis

This section evaluates each proposed sarcasm/insincerity signal for: (a) can it be computed from VADUG pendulum trace data, (b) how reliable is it in isolation, and (c) what are its failure modes.

### 2.1 Signal: Trajectory Reversal (V-spike then drop)

**Computation:** Scan the trace history. If word[i] causes V to spike by >25 points, and within 3 words V drops by >20 points, flag a reversal.

**Already implemented:** Yes, this is Signal 1 in the current SarcasmDetector.

**Reliability:** MODERATE. Catches "Oh wonderful, another Monday" where "wonderful" spikes V and "Monday" (in negative context) drops it. But it also fires on genuine emotional arcs: "I was so happy but then I heard the news." The reversal in the genuine case is caused by a conjunction introducing new information, not by insincerity.

**Failure modes:**
- False positive on genuine "good news then bad news" sequences
- False positive on "but" constructions that are genuine mixed emotions
- Misses sarcasm that doesn't spike (deadpan sarcasm: "great..." with no exclamation)

**Assessment:** Necessary but not sufficient. Must be combined with other signals and gated by conjunction detection. A reversal that crosses a "but"/"however" boundary is probably genuine; a reversal within a single clause is more suspicious.

### 2.2 Signal: Intensity Mismatch (strong positive in negative context)

**Computation:** If word[i] causes V-spike >35 AND the average V of surrounding words (3 before, 3 after) is <115, the positive word is anomalously strong for its context.

**Already implemented:** Yes, this is Signal 2 in the current SarcasmDetector.

**Reliability:** MODERATE-HIGH. This is Riloff et al.'s core finding formalized as a threshold check. "What a wonderful disaster" produces a clear mismatch. The key insight is that genuinely positive words in genuinely positive contexts don't produce mismatches -- the surrounding context is ALSO positive.

**Failure modes:**
- False positive on "silver lining" constructions: "The car broke down, but at least nobody was hurt." Here "hurt" (negated) is genuinely positive in negative context.
- Threshold sensitivity: the >35 spike and <115 surrounding thresholds are arbitrary.
- Misses moderate sarcasm with smaller word forces.

**Assessment:** Strong signal. The threshold values need tuning but the principle is sound.

### 2.3 Signal: V/A Correlation Break (flat delivery)

**Computation:** In genuine emotion, V and A tend to move together (at least directionally). Genuine happiness: V rises AND A rises. Genuine sadness: V falls AND A may rise (distress) or fall (depression). Sarcastic positive: V rises but A stays flat or drops -- the words are positive but there is no emotional energy behind them.

**Computable:** YES. For each word that moves V by more than 15 points, check whether A moved in the expected direction. Define expected correlation:
- V rising significantly + A flat or dropping = possible insincerity
- V dropping significantly + A flat = possible resignation (not sarcasm, but still useful)

**Reliability:** MODERATE. This is Scherer's "desynchronization" principle applied to two axes. The problem is that calm, content people can say genuinely positive things with low arousal: "I'm happy with how things turned out" (V+, A-flat) is sincere.

**Failure modes:**
- Calm positive statements flagged as insincere
- Written text lacks prosodic cues that would disambiguate in speech
- Some sarcasm IS high-arousal (exasperated sarcasm: "Oh GREAT!")

**Assessment:** Useful as a supporting signal, never as a standalone detector. Must be weighted by context: flat A after explicitly negative context is suspicious; flat A in neutral context is probably just calm.

### 2.4 Signal: V/G Correlation Break (positive words that don't lift)

**Computation:** Genuine positive emotions should lift (V+ correlates with G+). Sarcastic positive words should NOT lift -- the speaker is saying positive words but feeling heavy/grounded/sinking. Check: if a word produces V > +20 but G < +5 (or G is negative), the positive emotion lacks physical weight.

**Computable:** YES, and this is NOVEL. No existing system has a gravity axis to exploit.

**Reliability:** Theoretically HIGH, but UNTESTED. The argument is compelling:
- "I love you" (sincere): V+ AND G+ (love lifts)
- "I love you" (sarcastic/bitter): V+ from "love" but G stays flat or sinks (the speaker feels heavy, not lifted)
- "Great, just great" (sarcastic): V+ from "great" but G- (sinking feeling)
- "Great news!" (genuine): V+ AND G+ (feeling buoyant)

**Failure modes:**
- The pendulum's word-level G forces are assigned to WORDS, not to the speaker's felt state. The word "great" has the same G force whether spoken sincerely or sarcastically -- because the pendulum doesn't know it's sarcasm yet. This is a CIRCULAR DEPENDENCY problem.
- RESOLUTION: The V/G check must operate at the TRAJECTORY level, not the word level. Look at the overall trajectory: if V is trending positive but G is trending flat/down, the positivity lacks lift. This is computable from the trace history.

**Assessment:** Potentially the strongest novel signal, but requires trajectory-level computation, not word-level. The circular dependency problem is solvable by analyzing the integrated trajectory rather than individual word forces.

### 2.5 Signal: Oscillation Frequency (rapid V swings)

**Computation:** Count the number of times V crosses the neutral line (128) or reverses direction by >15 points within a chunk. High oscillation = unstable/performative emotional expression.

**Computable:** YES. Simple frequency analysis on the V trace.

**Reliability:** LOW-MODERATE. Rapid oscillation does correlate with certain types of insincerity (dramatic performance, passive aggression), but it also correlates with genuinely complex emotions and rapid topic shifts. "I love the house but hate the neighborhood, though the schools are great but the commute kills me" oscillates rapidly and is completely sincere.

**Failure modes:**
- Genuine complex situations with rapid topic changes
- Long sentences with many emotional topics
- Doesn't distinguish oscillation from genuine emotional complexity

**Assessment:** Weak standalone signal. Useful as a "something unusual is happening" flag that triggers deeper analysis. Not useful for sarcasm specifically.

### 2.6 Signal: Endpoint vs. Trajectory Mismatch

**Computation:** Compare the final V position with the trajectory's peak-to-trough range. If the trajectory visits extreme values (V > 180 or V < 80) but ends near neutral (110-140), the emotional journey was dramatic but the conclusion is muted -- possibly because the dramatic moments were performative.

**Computable:** YES. Final position vs max(V_trace) and min(V_trace).

**Reliability:** LOW. This pattern occurs frequently in genuine bittersweet situations, genuine recovery narratives, and any sentence that processes bad-then-good news. "I was devastated, but I've recovered" has extreme trajectory and moderate endpoint -- and it's sincere.

**Failure modes:**
- Genuine recovery narratives
- "But" constructions that genuinely resolve tension
- Any sentence with emotional arc that concludes at a different place than it peaked

**Assessment:** NOT RECOMMENDED as a sarcasm signal. This is better used for arc detection (already handled by the chunker/grader) than for sincerity detection.

### 2.7 Signal: Lexical Sarcasm Markers

**Computation:** Detect known sarcasm-associated lexical patterns:
- Hedge words before positive words: "oh", "just", "so", "really", "totally", "absolutely"
- Ellipsis after positive words: "great...", "wonderful..."
- ALL CAPS on emotional words: "GREAT", "LOVE", "SURE"
- Exclamation after negative trajectory: sudden "!" after sinking V
- Repeated words: "great, just great", "fine, fine"

**Computable:** YES. These are simple string/pattern checks on the raw input, not derived from VADUG.

**Reliability:** MODERATE for hedges + positive combos. LOW for individual markers. "Oh" before "wonderful" is suspicious; "oh" before "no" is genuine surprise. The combination is key.

**Failure modes:**
- "Really wonderful" can be genuine enthusiasm
- Ellipsis may indicate trailing off, not sarcasm
- Caps may indicate excitement, not sarcasm
- Repeated words may indicate emphasis, not sarcasm

**Assessment:** Useful as a LOW-WEIGHT supporting signal. These markers should contribute to a combined score but never trigger alone. The literature (Maynard & Greenwood 2014) confirms that lexical markers gain reliability only when combined with sentiment analysis.

### 2.8 Signal: Historical Context Contradiction

**Computation:** Compare the current chunk's VADUG with previous chunks. If the previous context was strongly negative (avg V < 90) and the current chunk is positive (V > 135) with LOW arousal (A < 145), the positive shift is "flat" -- possible sarcasm or passive aggression.

**Already implemented:** Yes, this is Signal 3 in the current SarcasmDetector.

**Reliability:** MODERATE. The low-arousal requirement is the key differentiator: genuine positive-after-negative transitions tend to be HIGH arousal (relief, excitement). Flat delivery of positive words after negative context is indeed suspicious.

**Failure modes:**
- Calm acceptance after negative events ("It's okay, I've made peace with it")
- Resigned positive statements ("Well, at least it's over")
- The threshold values (V < 90, V > 135, A < 145) are arbitrary

**Assessment:** Good signal. The arousal check is the right discriminator but needs calibration.

### 2.9 Signal: Hyperbolic Intensity

**Computation:** Flag when emotional word forces are in the top 10% of intensity (V-delta > 45 or V-delta < -45) and the surrounding context doesn't warrant such intensity. A single "AMAZING" in an otherwise flat/negative sentence is hyperbolic.

**Computable:** YES. It's a stronger version of Signal 2.2 (Intensity Mismatch) that specifically targets extreme outliers.

**Reliability:** MODERATE. Genuine superlatives exist ("This is the most amazing sunset") but they typically occur in already-positive contexts. Superlatives in flat/negative contexts are suspicious.

**Failure modes:**
- Genuine extreme reactions to surprising events
- Cultural differences in expression intensity
- Must distinguish "AMAZING (genuine excitement)" from "AMAZING (sarcastic emphasis)"

**Assessment:** Subsume into Signal 2.2 (Intensity Mismatch) with a higher threshold for standalone flagging.

### 2.10 Signal Reliability Summary

| Signal | Reliability | Standalone | Combined Value | Compute Cost |
|--------|------------|-----------|----------------|-------------|
| Trajectory Reversal | MODERATE | NO | HIGH | Low |
| Intensity Mismatch | MODERATE-HIGH | WEAK | HIGH | Low |
| V/A Correlation Break | MODERATE | NO | MODERATE | Low |
| V/G Correlation Break | Theoretically HIGH | POSSIBLE | VERY HIGH | Low |
| Oscillation Frequency | LOW-MODERATE | NO | LOW | Low |
| Endpoint/Trajectory Mismatch | LOW | NO | NOT RECOMMENDED | Low |
| Lexical Markers | MODERATE (combined) | NO | MODERATE | Negligible |
| Context Contradiction | MODERATE | NO | HIGH | Low |
| Hyperbolic Intensity | MODERATE | NO | Subsumed by #2 | Low |

**Recommended signals for inclusion:** 1, 2, 3, 4, 7, 8.
**Drop:** 5 (oscillation -- too noisy), 6 (endpoint mismatch -- better as arc data), 9 (subsumed by 2).

---

## 3. Recommended Architecture: Weighted Gate Hybrid

### 3.1 Why Not Pure Logic Gates

Pure sequential gates (yes/no at each step) are brittle. If Gate 1 says "no reversal detected" (misses by 1 V-point), the entire cascade is dead. Real sarcasm is gradient -- there's a continuum from "definitely sincere" through "ambiguous" to "definitely sarcastic." Binary gates lose that gradient.

### 3.2 Why Not Pure Continuous Scoring

Pure continuous scoring (every signal is a 0.0-1.0 float, everything is weighted sum) loses inspectability. When the system says "sincerity = 0.37," a developer cannot easily determine which signals contributed what. It becomes a mini neural network in disguise -- exactly what Clanker's philosophy opposes.

### 3.3 The Hybrid: Scored Gates

Each gate produces a **score** (0.0 to 1.0), not a binary. But gates have **thresholds** that determine whether they "fire" (contribute to the final assessment). The final tonal coordinates are derived from a WEIGHTED COMBINATION of fired gate scores, not from a single gate's binary output.

```
Architecture: SCORED GATE HYBRID

For each signal gate:
  1. Compute a raw score (0.0 - 1.0) from the trajectory data
  2. Apply a fire threshold (typically 0.3 - 0.5)
  3. If score >= threshold, the gate FIRES and its score contributes
  4. If score < threshold, the gate is SILENT (contributes 0)

Final sincerity = weighted combination of all fired gate scores
Final warmth = separate weighted combination (different gates, different weights)
```

This gives us:
- **Inspectability:** Each gate's score is visible, each fired/silent status is logged
- **Gradient output:** The final scores are continuous, not binary
- **Robustness:** Missing one gate by 1 point doesn't kill detection (other gates compensate)
- **Tuning surface:** Each gate's threshold and weight can be tuned independently

### 3.4 Why This Is Not ML

This is NOT a perceptron or logistic regression. The differences:
- Weights are SET BY DESIGN, not learned from data
- There are no backpropagation updates
- Each gate's computation is a documented mathematical formula
- The combination function is a simple weighted average, not a learned function
- All values are inspectable at every step

It's engineering, not training. The same way a building's structural calculations use weighted combinations of load, wind, and seismic factors -- each factor has a documented formula, and the combination weights come from engineering standards, not gradient descent.

---

## 4. Proposed Gate Pipeline

Six gates, each producing a score from 0.0 to 1.0. Gates are evaluated independently (no ordering dependency in the base layer), then combined.

### 4.1 Gate 1: Contrast Gate (Trajectory Reversal)

**Input:** Pendulum trace history (list of per-word V values).

**Computation:**
```python
def contrast_gate(history):
    """Detect positive spike followed by drop within same clause."""
    max_score = 0.0
    for i in range(1, len(history)):
        spike = history[i]['v'] - history[i-1]['v']
        if spike > 20:  # positive spike
            for j in range(i+1, min(i+4, len(history))):
                drop = history[i]['v'] - history[j]['v']
                if drop > 15:
                    # Score proportional to spike*drop product
                    raw = (spike * drop) / (60 * 50)  # normalize: 60*50 = strong case
                    score = min(1.0, raw)
                    max_score = max(max_score, score)
    return max_score
```

**Fire threshold:** 0.3
**Weight in sincerity calculation:** 0.20
**Weight in warmth calculation:** 0.05

### 4.2 Gate 2: Polarity Mismatch Gate

**Input:** Pendulum trace history (per-word V and surrounding context).

**Computation:**
```python
def polarity_mismatch_gate(history):
    """Detect strong positive word in negative/neutral context."""
    max_score = 0.0
    for i in range(1, len(history)):
        spike = history[i]['v'] - history[i-1]['v']
        if spike > 25:  # significant positive word
            # Compute context average (3 before, 3 after, excluding this word)
            start = max(0, i-3)
            end = min(len(history), i+4)
            surrounding = [h['v'] for h in history[start:end] if h != history[i]]
            if surrounding:
                ctx_avg = sum(surrounding) / len(surrounding)
                if ctx_avg < 120:  # context is negative/neutral
                    # Score: how much the word deviates from context
                    deviation = (spike - 20) / 40  # normalize: 60-point spike = 1.0
                    context_negativity = (120 - ctx_avg) / 60  # normalize
                    raw = deviation * context_negativity
                    max_score = max(max_score, min(1.0, raw))
    return max_score
```

**Fire threshold:** 0.3
**Weight in sincerity calculation:** 0.25 (strongest single signal)
**Weight in warmth calculation:** 0.10

### 4.3 Gate 3: Delivery Flatness Gate (V/A Desynchronization)

**Input:** Pendulum trace history (per-word V and A values).

**Computation:**
```python
def delivery_flatness_gate(history):
    """Detect positive V movement without corresponding A movement."""
    if len(history) < 3:
        return 0.0

    # Compute V and A deltas for emotional words (words that moved V > 10)
    v_deltas = []
    a_deltas = []
    for i in range(1, len(history)):
        v_delta = history[i]['v'] - history[i-1]['v']
        a_delta = history[i]['a'] - history[i-1]['a']
        if abs(v_delta) > 10:  # only count words that moved V meaningfully
            v_deltas.append(v_delta)
            a_deltas.append(a_delta)

    if len(v_deltas) < 2:
        return 0.0

    # For positive V movements, check if A is flat or contradictory
    flat_count = 0
    positive_count = 0
    for vd, ad in zip(v_deltas, a_deltas):
        if vd > 15:  # positive emotional word
            positive_count += 1
            if ad < 3:  # A didn't move -- flat delivery
                flat_count += 1

    if positive_count == 0:
        return 0.0

    return flat_count / positive_count  # proportion of flat-delivered positives
```

**Fire threshold:** 0.4 (higher threshold -- calm sincerity is common)
**Weight in sincerity calculation:** 0.15
**Weight in warmth calculation:** 0.15

### 4.4 Gate 4: Gravity Mismatch Gate (V/G Desynchronization)

**Input:** Pendulum trace history (per-word V and G values).

**Computation:**
```python
def gravity_mismatch_gate(history):
    """Detect positive V trajectory without corresponding G lift.

    Genuine positive emotions LIFT. Sarcastic positive words
    are heavy -- the speaker says 'wonderful' but feels weighed down.
    Operates on trajectory level, not word level.
    """
    if len(history) < 3:
        return 0.0

    # Compute trajectory-level V and G trends
    # Use first third vs last third comparison
    third = max(1, len(history) // 3)
    early_v = sum(h['v'] for h in history[:third]) / third
    late_v = sum(h['v'] for h in history[-third:]) / third
    early_g = sum(h['g'] for h in history[:third]) / third
    late_g = sum(h['g'] for h in history[-third:]) / third

    v_trend = late_v - early_v  # positive = V went up
    g_trend = late_g - early_g  # positive = G went up (lighter)

    # Mismatch: V went up but G didn't, or V went up and G went DOWN
    if v_trend > 15:  # V is trending positive
        if g_trend < 0:
            # G is SINKING while V is rising -- strong mismatch
            raw = min(1.0, (v_trend / 40) * (abs(g_trend) / 20))
            return raw
        elif g_trend < 5:
            # G is flat while V is rising -- moderate mismatch
            raw = min(1.0, (v_trend - 15) / 40)
            return raw * 0.6  # lower score for flat vs sinking

    return 0.0
```

**Fire threshold:** 0.3
**Weight in sincerity calculation:** 0.25 (STRONGEST -- novel signal, theoretically most discriminative)
**Weight in warmth calculation:** 0.20

### 4.5 Gate 5: Lexical Marker Gate

**Input:** Raw input text (not trajectory).

**Computation:**
```python
SARCASM_HEDGES = {"oh", "just", "so", "really", "totally", "absolutely",
                   "surely", "clearly", "obviously", "wow", "gee", "gosh"}

SARCASM_PATTERNS = [
    # (pattern, score_contribution)
    (r'\b(great|wonderful|fantastic|amazing|brilliant|perfect)\.\.\.', 0.5),   # positive + ellipsis
    (r'\b(great|wonderful|fantastic|perfect),?\s+(just\s+)?(great|wonderful|fantastic|perfect)', 0.6),  # repetition
    (r'\b(oh|wow|gee)\s+(how\s+)?(great|wonderful|lovely|nice)', 0.5),  # hedge + positive
    (r'\bthanks?\s+(a\s+lot|so\s+much)\s*[\.!]*\s*$', 0.3),  # "thanks a lot" at end (ambiguous)
]

def lexical_marker_gate(text):
    """Detect lexical patterns associated with sarcasm."""
    text_lower = text.lower().strip()
    score = 0.0

    # Pattern matching
    for pattern, contribution in SARCASM_PATTERNS:
        if re.search(pattern, text_lower):
            score += contribution

    # Hedge word before strong positive word
    words = text_lower.split()
    for i in range(len(words) - 1):
        if words[i] in SARCASM_HEDGES:
            if words[i+1] in {"great", "wonderful", "fantastic", "amazing",
                               "brilliant", "perfect", "lovely", "nice"}:
                score += 0.3

    return min(1.0, score)
```

**Fire threshold:** 0.3
**Weight in sincerity calculation:** 0.10 (supporting role only)
**Weight in warmth calculation:** 0.05

### 4.6 Gate 6: Context History Gate

**Input:** Previous chunk VADUG results + current chunk VADUG.

**Computation:**
```python
def context_history_gate(previous_chunks, current_chunk):
    """Detect contradictory emotional shift with flat delivery.

    Previous negative + current positive + low arousal = suspicious.
    Previous negative + current positive + high arousal = genuine relief.
    """
    if not previous_chunks:
        return 0.0

    prev_avg_v = sum(c['vadug'].v for c in previous_chunks) / len(previous_chunks)
    curr_v = current_chunk['vadug'].v
    curr_a = current_chunk['vadug'].a
    curr_g = current_chunk['vadug'].g

    # Must have negative-to-positive shift
    if prev_avg_v >= 100 or curr_v <= 128:
        return 0.0

    v_shift = curr_v - prev_avg_v  # how big is the positive jump

    # Arousal check: genuine relief/hope is HIGH arousal; sarcasm is flat
    arousal_factor = max(0.0, (155 - curr_a) / 50)  # 1.0 at A=105, 0.0 at A=155+

    # Gravity check: genuine positive shift LIFTS; sarcastic doesn't
    gravity_factor = max(0.0, (140 - curr_g) / 40)  # 1.0 at G=100, 0.0 at G=140+

    # Combine: big positive shift + flat arousal + no gravity lift
    raw = (v_shift / 60) * (arousal_factor * 0.6 + gravity_factor * 0.4)
    return min(1.0, raw)
```

**Fire threshold:** 0.35
**Weight in sincerity calculation:** 0.20
**Weight in warmth calculation:** 0.15

### 4.7 Gate Weight Summary

| Gate | Signal | Sincerity Weight | Warmth Weight | Fire Threshold |
|------|--------|-----------------|---------------|----------------|
| G1 | Trajectory Reversal | 0.20 | 0.05 | 0.30 |
| G2 | Polarity Mismatch | 0.25 | 0.10 | 0.30 |
| G3 | Delivery Flatness | 0.15 | 0.15 | 0.40 |
| G4 | Gravity Mismatch | 0.25 | 0.20 | 0.30 |
| G5 | Lexical Markers | 0.10 | 0.05 | 0.30 |
| G6 | Context History | 0.20 | 0.15 | 0.35 |
|     | **Total** | **1.15** | **0.70** | |

Note: Weights sum to >1.0 intentionally. The combination function normalizes by the sum of FIRED gate weights only, not by total possible weight. This means the score is not diluted by gates that didn't fire.

### 4.8 Combination Function

```python
def compute_sincerity(gate_scores, gate_weights, gate_thresholds):
    """Combine fired gate scores into sincerity estimate.

    Returns 0-255 (u8), where 255 = fully sincere, 0 = fully performative.
    """
    fired_weighted_sum = 0.0
    fired_weight_total = 0.0

    for gate_id, score in gate_scores.items():
        if score >= gate_thresholds[gate_id]:
            weight = gate_weights[gate_id]
            fired_weighted_sum += score * weight
            fired_weight_total += weight

    if fired_weight_total == 0:
        # No gates fired -- no evidence of insincerity
        return 255  # default to sincere

    # insincerity_score: how much evidence of insincerity (0.0 - 1.0)
    insincerity_score = fired_weighted_sum / fired_weight_total

    # Convert to sincerity (invert) and scale to 0-255
    sincerity = int((1.0 - insincerity_score) * 255)
    return max(0, min(255, sincerity))
```

**Key design decision:** When NO gates fire, sincerity defaults to 255 (fully sincere). This is the correct default because the absence of evidence for insincerity IS evidence of sincerity. The system is a sarcasm DETECTOR, not a sincerity prover.

---

## 5. Thought Tree Branching Rules

### 5.1 The Problem

Certain gate combinations require different interpretation than others. If Gate 1 (Contrast) fires AND the contrast crosses a "but"/"however" boundary, the contrast is probably genuine mixed emotion, not sarcasm. But if Gate 1 fires AND Gate 2 (Mismatch) fires AND the contrast is within a single clause, it's almost certainly sarcasm.

This is where pure independent gate scoring breaks down. We need branching logic that modifies how gate scores are INTERPRETED based on what other gates found.

### 5.2 Branching Rules (Post-Gate Modifiers)

After all gates have produced their independent scores, apply these branching modifiers:

```
RULE 1: Conjunction Bridging
IF Gate 1 (Contrast) fired
AND the reversal crosses a "but"/"however"/"although"/"yet" boundary
THEN: Reduce Gate 1 score by 60%
REASON: Genuine mixed emotion, not sarcasm. The conjunction signals
        the speaker is AWARE of the contrast and expressing it intentionally.

RULE 2: Corroborating Mismatch
IF Gate 1 (Contrast) fired
AND Gate 2 (Mismatch) fired
AND both triggered on the SAME word
THEN: Boost both scores by 30% (cap at 1.0)
REASON: Two independent signals converging on the same word is strong.

RULE 3: Flat Delivery Corroboration
IF Gate 2 (Mismatch) OR Gate 4 (Gravity Mismatch) fired
AND Gate 3 (Delivery Flatness) fired
THEN: Boost all three scores by 20%
REASON: Positive words + flat delivery + no gravity lift = triple signal.

RULE 4: High Arousal Override
IF Gate 3 (Delivery Flatness) fired
BUT the chunk's overall A > 170
THEN: Reduce Gate 3 score by 80%
REASON: High arousal contradicts "flat delivery." This may be
        exasperated sarcasm, but the flatness gate is wrong about it.

RULE 5: Lexical Confirmation
IF Gate 5 (Lexical Markers) fired
AND any of Gates 1-4 also fired
THEN: Boost Gate 5 score by 50% (cap at 1.0)
REASON: Lexical markers alone are weak; lexical markers PLUS trajectory
        evidence are strong.

RULE 6: Lexical Isolation
IF Gate 5 (Lexical Markers) fired
AND NONE of Gates 1-4 fired
THEN: Reduce Gate 5 score by 70%
REASON: Lexical patterns without trajectory evidence are probably
        genuine intensifiers, not sarcasm markers.

RULE 7: Context Contradiction + Gravity = Strong Signal
IF Gate 6 (Context History) fired
AND Gate 4 (Gravity Mismatch) fired
THEN: Boost both by 25%
REASON: Positive shift after negative context + no gravity lift =
        the speaker is saying positive words but not feeling them.
```

### 5.3 Why Not Deeper Trees

Seven rules is the design limit. Beyond this, the branching logic becomes:
- Hard to reason about (combinatorial explosion of rule interactions)
- Hard to debug (which rules modified which gates in which order?)
- A neural network in disguise (if you have 20+ rules with learned thresholds, you've built a decision forest without admitting it)

The principle: **branching rules modify gate scores, they never add new gates.** The tree is exactly one level deep. Gates produce scores, branching rules modify those scores, and the combination function produces the final output. No deeper nesting.

### 5.4 Rule Application Order

Rules are applied in numerical order after all gates have scored. Rules that modify the same gate's score apply multiplicatively:
- If Rule 1 reduces Gate 1 by 60% and Rule 2 boosts Gate 1 by 30%, the result is: `score * 0.4 * 1.3 = score * 0.52`.
- This correctly handles the case where a contrast crosses a "but" boundary (Rule 1 reduces) but ALSO has a polarity mismatch on the same word (Rule 2 boosts). The net effect is a moderate reduction -- the conjunction provides partial exoneration but the mismatch is still suspicious.

---

## 6. False Positive Prevention

This is the hardest part of the design. A false positive (flagging genuine emotion as sarcastic) is worse than a false negative (missing sarcasm) because it causes the system to respond to a sincere statement with "I hear the frustration behind that" -- which is patronizing and infuriating.

### 6.1 Category: Genuine Mixed Emotions (Bittersweet)

**Pattern:** "I'm sad to leave but excited for what's next"
**Risk:** Gate 1 (Contrast) and Gate 6 (Context) both fire on the negative-to-positive shift.
**Prevention:**
- Rule 1 (Conjunction Bridging) reduces Gate 1 by 60%
- Gate 6 should check: is the positive chunk HIGH arousal? Genuine mixed emotions tend to be energized. The speaker is processing real feelings, which requires effort (A > 140).
- Additional check: if the positive chunk has normal V/G correlation (both rising), it's genuine lift, not sarcastic.

### 6.2 Category: Genuine Enthusiasm with Intensifiers

**Pattern:** "I absolutely LOVE this!"
**Risk:** Gate 5 (Lexical) fires on "absolutely" + positive word. Gate 2 might fire if surrounding context was neutral.
**Prevention:**
- Rule 6 (Lexical Isolation) kills Gate 5 if no trajectory evidence exists.
- Gate 2 should have a context window requirement: if ALL surrounding words are also positive (not just the target word), no mismatch exists. The issue is a positive word in NEGATIVE context, not a positive word in NEUTRAL context.
- V/G correlation check: genuine enthusiasm produces V+ AND G+ (soaring). If gravity is rising with valence, the enthusiasm is real.

### 6.3 Category: Calm Acceptance

**Pattern:** "It's okay, I've made peace with it"
**Risk:** Gate 3 (Delivery Flatness) fires because V is mildly positive but A is low. Gate 6 might fire if previous context was negative.
**Prevention:**
- Gate 3 has a higher threshold (0.4) specifically to avoid flagging calm positive statements.
- Rule 4 (High Arousal Override) doesn't apply here (A is low), but we need an additional check: if the positive V values are MODERATE (V < 160, not V > 180), the speaker isn't being hyperbolic. Sarcasm tends to use STRONG positive words, not mild ones. Add a "positive intensity" check to Gate 3: only flag flat delivery when the V-spike is significant (>25 points).

### 6.4 Category: Genuine Recovery Narrative

**Pattern:** "I failed, but I learned so much from it"
**Risk:** Gate 1 (Contrast), Gate 6 (Context) fire on the negative-to-positive shift.
**Prevention:**
- Rule 1 (Conjunction Bridging) handles the "but" case.
- The key discriminator is: does the positive content have SPECIFICITY? "I learned so much" has lower intensity than "Oh how WONDERFUL." Sarcasm tends to use high-intensity generic positives; genuine recovery uses moderate, specific positives.
- We cannot easily measure specificity without semantic understanding, but we CAN measure intensity: genuine recovery produces moderate V increase (V+20 to V+40), while sarcasm tends to produce sharp V spikes (V+40 to V+60). Gate 2's threshold helps here.

### 6.5 Category: "Happy Tears" vs. "Wonderful Failure"

This is the critical distinguishing test case:
- "Happy tears" = genuine mixed emotion where positive and negative coexist authentically
- "Wonderful failure" = sarcasm where the positive word mocks the negative reality

**The key differentiator: conjunction and context structure.**
- "Happy tears" -- the positive modifies the negative directly (adjective + noun). The words are fused, not contrasting.
- "What a wonderful failure" -- the positive is applied IRONICALLY to the negative. There is implicit contrast.
- "I'm crying because I'm so happy" -- explicit causal link explains the paradox.

**How to detect this computationally:**
1. Check for direct modification (adjective + noun): if a positive word immediately precedes a negative noun, check if they form a recognized compound (e.g., "happy tears" is in idiom detection, or positive-adj + emotion-noun = genuine mixed).
2. Check for causal links: "because," "since" bridging the positive and negative = genuine explanation.
3. Check for ironic structure: positive word + negative noun WITHOUT modifier relationship = sarcasm. "A wonderful failure" vs. "happy tears" -- in the first, "wonderful" is a determiner+adjective modifying "failure" (ironic application). In the second, "happy" modifies "tears" (genuine compound).

**Implementation:** Add a small set of recognized genuine-mixed compounds to the idiom detector:
```
"happy tears", "sweet sorrow", "bittersweet", "tears of joy",
"good cry", "painfully beautiful", "beautiful disaster" (ambiguous),
"perfectly imperfect", "beautifully broken"
```

If one of these is detected, suppress all sarcasm gates for that chunk.

### 6.6 The Minimum Confidence Floor

**Critical rule:** Sincerity should NEVER drop below 128 (the neutral midpoint) based on a SINGLE fired gate. At least TWO gates must fire (or one gate must score >0.8) to push sincerity below 128. This prevents single-signal false positives.

```python
def apply_confidence_floor(sincerity, num_fired_gates, max_gate_score):
    """Prevent single-gate false positives."""
    if num_fired_gates <= 1 and max_gate_score < 0.8:
        sincerity = max(sincerity, 128)  # floor at neutral
    return sincerity
```

---

## 7. Tonal Coordinate Definition

### 7.1 Proposed Axes: S and T (2 bytes)

**S (Sincerity):** 0 = fully performative/insincere, 255 = fully genuine/earnest.

- 0-50: Strong insincerity detected (sarcasm, mockery, performative)
- 51-100: Likely insincere (moderate signal strength)
- 101-150: Ambiguous / possible insincerity
- 151-200: Probably sincere (mild concerns)
- 201-255: Sincere (no insincerity signals detected)

**T (Temperature/Warmth):** 0 = hostile/cold, 255 = warm/affectionate.

- 0-50: Hostile, aggressive, contemptuous
- 51-100: Cold, distanced, dismissive
- 101-150: Neutral / professional
- 151-200: Warm, friendly, caring
- 201-255: Deep affection, love, protective warmth

### 7.2 Should There Be a Third Axis? (P: Playfulness)

**Case for P:**
- Distinguishes hostile sarcasm ("great job, idiot") from playful teasing ("oh sure, you're sooo humble")
- Playful insincerity is fundamentally different from hostile insincerity in terms of response strategy
- The existing personality vector has a Playfulness byte -- the tonal axis would measure the INPUT's playfulness, not the model's

**Case against P:**
- S and T together already capture most of the space: playful teasing is low-S (insincere), high-T (warm). Hostile sarcasm is low-S (insincere), low-T (cold).
- Adding a third byte increases the header from 11 to 12 bytes (VADUG 5 + meta 4 + tonal 2 vs 3)
- The more axes we add, the harder each one is to compute accurately
- Playfulness is arguably derivable: low-S + high-T = playful; low-S + low-T = hostile; high-S + high-T = genuine affection; high-S + low-T = direct/blunt

**Recommendation: START WITH 2 AXES (S and T). Add P later if the S/T space proves insufficient.**

The S x T quadrant analysis covers the critical distinctions:

```
              HIGH WARMTH (T)
                   |
   Genuine Love    |   Playful Teasing
   (high S, high T)|   (low S, high T)
                   |
---LOW SINCERITY---+---HIGH SINCERITY---
                   |
   Hostile Sarcasm |   Blunt Honesty
   (low S, low T)  |   (high S, low T)
                   |
              LOW WARMTH (T)
```

This 2D space already distinguishes the four critical tonal categories. A third axis would add granularity but is not essential for the core use cases.

### 7.3 How Warmth (T) Is Computed

Warmth is a distinct computation from sincerity. It draws from different aspects of the VADUG trace:

```python
def compute_warmth(vadug, history, personality):
    """Compute warmth/temperature from VADUG and trajectory.

    Warmth is about HOW the speaker relates to the listener,
    independent of whether they're being sincere.

    Returns 0-255 (u8), where 255 = deeply warm, 0 = deeply cold.
    """
    warmth = 128.0  # start neutral

    # Factor 1: Valence (positive = warmer, but not linearly)
    v_factor = (vadug.v - 128) / 127.0  # -1.0 to +1.0
    warmth += v_factor * 40  # V contributes up to +/-40

    # Factor 2: Gravity (light/soaring = warmer; crushing = colder)
    g_factor = (vadug.g - 128) / 127.0
    warmth += g_factor * 20  # G contributes up to +/-20

    # Factor 3: Dominance (very high D = colder/more authoritative)
    if vadug.d > 180:
        warmth -= (vadug.d - 180) * 0.3  # high dominance cools warmth
    elif vadug.d < 80:
        warmth -= (80 - vadug.d) * 0.2  # very low D = vulnerable, not warm

    # Factor 4: Social word presence in trajectory
    social_warm = {"love", "care", "dear", "honey", "friend", "miss",
                   "hug", "comfort", "cherish", "treasure", "appreciate"}
    social_cold = {"hate", "enemy", "despise", "disgust", "loathe",
                   "hostile", "toxic", "threat", "attack", "destroy"}

    warm_count = sum(1 for h in history if h['word'] in social_warm)
    cold_count = sum(1 for h in history if h['word'] in social_cold)
    warmth += warm_count * 15 - cold_count * 20

    # Factor 5: Arousal modulates warmth direction
    # High arousal amplifies whatever warmth direction exists
    if vadug.a > 160:
        amplification = (vadug.a - 160) / 95  # 0.0 to 1.0
        if warmth > 128:
            warmth += (warmth - 128) * amplification * 0.3
        else:
            warmth -= (128 - warmth) * amplification * 0.3

    return max(0, min(255, int(warmth)))
```

### 7.4 Header Integration

The tonal bytes extend the 9-byte message header to 11 bytes:

```
CLANKER MESSAGE HEADER (11 bytes)

Bytes 0-4:  VADUG Emotional Vector [V][A][D][U][G]
Byte 5:     CERT (Certainty)
Byte 6:     SRC (Source/Provenance)
Byte 7:     GOAL (Intent)
Byte 8:     REL (Context Relevance)
Byte 9:     S (Sincerity)      ← NEW
Byte 10:    T (Temperature)    ← NEW
```

Two bytes. No structural change to the existing header -- pure additive extension. Systems that don't support tonal analysis can ignore bytes 9-10 (default to S=255, T=128 -- sincere, neutral warmth).

---

## 8. Integration Plan

### 8.1 Where Tonal Analysis Fits in the 7-Layer Pipeline

The tonal analysis is a SUB-STEP of Layer 2 (Sequential Pendulum), running AFTER the pendulum completes but BEFORE results are passed to Layer 3 (Arc Analysis).

```
Layer 1: Emotional Chunking
    Split input at natural boundaries → chunks

Layer 2: Sequential Pendulum (per chunk)
    2a: Run pendulum word-by-word → VADUG + trace history
    2b: Run tonal analysis on trace history → S, T scores    ← NEW
    Output: VADUG + S + T per chunk

Layer 3: Arc Analysis
    Analyze trajectory across chunks (valley/peak/ascending/etc.)
    Now also considers: S/T trajectory across chunks

Layer 4: Personality Filter
    Apply resistance weights (existing)
    NEW: If personality Playfulness > 150 AND S < 100 (detected sarcasm),
         nudge response toward playful acknowledgment rather than serious probe

Layer 5: Response VADUG Computation
    Apply harmony formulas per chunk
    NEW: S and T influence harmony parameters:
      - Low S → response should address UNDERLYING emotion, not surface
      - Low T → response should be more careful/measured

Layer 6: Clanker Opcode Generation
    Produce opcodes with 11-byte header (VADUG + CERT + SRC + GOAL + REL + S + T)

Layer 7: Cross-Axis Template Decoder
    NEW: S and T influence template selection:
      - Low S: templates that acknowledge the real feeling
      - Low T: templates that maintain professional distance
      - Low S + Low T: templates that address hostility directly
      - Low S + High T: templates that match the playful tone
```

### 8.2 How Tonal Scores Modify the Sentence Grade

The sentence grader (SentenceGrader class) currently adjusts the grade based on sarcasm detection via a crude bump-down. The tonal analysis replaces this with a principled adjustment:

```python
def adjust_grade_for_tone(grade, sincerity, warmth, grader):
    """Adjust emotional grade based on tonal analysis.

    The grade represents the SURFACE emotional reading.
    Tonal analysis tells us whether the surface is the truth.

    Low sincerity means the REAL emotion is different from the surface.
    The grade should reflect the REAL emotion, not the surface.
    """
    if sincerity > 200:
        return grade, ""  # highly sincere, grade is accurate

    if sincerity > 150:
        return grade, ""  # probably sincere, no adjustment

    if sincerity > 100:
        # Ambiguous sincerity -- bump down 1 half-step
        adjusted = grader._bump_down(grade)
        return adjusted, f"Tone adjustment: {grade} -> {adjusted} (ambiguous sincerity S={sincerity})"

    if sincerity > 50:
        # Likely insincere -- bump down 2 half-steps
        adjusted = grade
        for _ in range(2):
            adjusted = grader._bump_down(adjusted)
        return adjusted, f"Tone adjustment: {grade} -> {adjusted} (likely insincere S={sincerity})"

    # Strong insincerity -- bump down 3-4 half-steps
    adjusted = grade
    steps = 4 if warmth < 80 else 3  # hostile sarcasm = bigger adjustment
    for _ in range(steps):
        adjusted = grader._bump_down(adjusted)
    return adjusted, f"Tone adjustment: {grade} -> {adjusted} (insincere S={sincerity}, T={warmth})"
```

### 8.3 How Tonal Scores Modify Response Generation

The response system currently uses harmony formulas. With tonal data:

```
IF S < 100 (likely insincere):
    Response IGNORES the surface emotion.
    Instead, use the ADJUSTED grade (which reflects real emotion).
    The response should:
    - NOT mirror the sarcastic positive ("I'm glad you're happy!")
    - DO acknowledge the underlying frustration/pain
    - If T > 150 (warm sarcasm/teasing): respond with light humor
    - If T < 100 (hostile sarcasm): respond with calm empathy

IF S > 200 AND T > 200 (sincere + warm):
    Maximum warmth in response. Full mirror + support.

IF S > 200 AND T < 80 (sincere + cold):
    The speaker is genuinely upset/hostile. Respond with stability
    and measured warmth. Don't be overly friendly.

IF S = 100-200 (ambiguous):
    Hedge. Respond to the surface emotion but with a subtle
    acknowledgment that things might be more complex.
    "That's great -- you seem to really mean that."
```

### 8.4 Backward Compatibility

- **Existing 9-byte header consumers:** Continue to work. S and T bytes are appended, not inserted.
- **Existing SarcasmDetector class:** Replaced entirely by the tonal analysis system. The SarcasmDetector's three signals become Gates 1, 2, and 6 in the new pipeline, with the addition of Gates 3, 4, and 5, plus branching rules and the scored-gate architecture.
- **Existing sentence grader:** Grade computation unchanged. Only the post-grade adjustment changes from crude bump-down to tonal-informed adjustment.
- **Existing harmony formulas:** Unchanged at the formula level. Tonal scores modify which harmony target is computed (surface emotion vs. underlying emotion).

### 8.5 Display Output

The tonal analysis step should produce visible, inspectable output in the simulator:

```
--- STEP 2b: Tonal Analysis ---
  Gate 1 (Contrast):        0.45 [FIRED]  reversal on "wonderful"
  Gate 2 (Polarity):        0.62 [FIRED]  "wonderful" too positive for context
  Gate 3 (Delivery):        0.31 [silent]  below threshold (0.40)
  Gate 4 (Gravity):         0.55 [FIRED]  V rising but G flat
  Gate 5 (Lexical):         0.30 [FIRED]  "oh" + "wonderful" pattern
  Gate 6 (Context):         0.48 [FIRED]  positive after negative context

  Branching Rules Applied:
    Rule 2: G1+G2 corroboration on "wonderful" → G1 boosted to 0.59, G2 to 0.81
    Rule 5: G5 confirmed by G1-G4 → G5 boosted to 0.45
    Rule 7: G6+G4 corroboration → G6 boosted to 0.60, G4 to 0.69

  Sincerity (S): 72  [LIKELY INSINCERE]
  Temperature (T): 89  [cold]

  Tonal reading: hostile sarcasm — surface positive masks genuine frustration
  Grade adjustment: B+ → C- (surface positive, meaning negative)
```

---

## 9. Test Cases for Validation

### 9.1 Core Sarcasm Cases

| # | Input | Expected S | Expected T | Notes |
|---|-------|-----------|-----------|-------|
| 1 | "Oh wonderful, another Monday" | <80 | <100 | Classic sarcasm: positive word + negative context |
| 2 | "Great, just great" | <80 | <100 | Repetition sarcasm |
| 3 | "Sure, that makes total sense" | <100 | 80-130 | Sarcastic agreement, mild coldness |
| 4 | "Thanks a lot for that" | <100 | <100 | Sarcastic gratitude |
| 5 | "What a fantastic idea, let's definitely do that" | <80 | <100 | Extended sarcasm with multiple positives |
| 6 | "I love how you always know best" | <80 | <80 | Sarcasm masking resentment |
| 7 | "Yeah, because that worked so well last time" | <80 | <100 | Historical-reference sarcasm |

### 9.2 Genuine Positive Cases (MUST NOT flag as sarcastic)

| # | Input | Expected S | Expected T | Notes |
|---|-------|-----------|-----------|-------|
| 8 | "I love you" | >200 | >200 | Sincere love |
| 9 | "This is absolutely wonderful!" | >200 | >180 | Genuine enthusiasm |
| 10 | "I'm so happy for you" | >200 | >200 | Genuine warmth |
| 11 | "Great job on the presentation!" | >180 | >180 | Genuine praise |
| 12 | "I really appreciate your help" | >200 | >200 | Genuine gratitude |
| 13 | "You're amazing, seriously" | >180 | >200 | Genuine admiration |

### 9.3 Genuine Mixed Emotions (MUST NOT flag as sarcastic)

| # | Input | Expected S | Expected T | Notes |
|---|-------|-----------|-----------|-------|
| 14 | "I'm sad to leave but excited for what's next" | >180 | >150 | Bittersweet transition |
| 15 | "Happy tears, I can't believe it" | >200 | >200 | Overwhelmed joy |
| 16 | "I failed, but I learned a lot" | >180 | >120 | Genuine recovery |
| 17 | "It hurts, but I know it's for the best" | >180 | >120 | Genuine acceptance |
| 18 | "Terrible movie, but the popcorn was amazing" | >180 | >120 | Genuine contrast |

### 9.4 Playful Teasing (insincere but warm)

| # | Input | Expected S | Expected T | Notes |
|---|-------|-----------|-----------|-------|
| 19 | "Oh sure, you're totally innocent" | <120 | >150 | Playful teasing between friends |
| 20 | "Yeah right, Mr. Perfect" | <100 | >130 | Playful ribbing |
| 21 | "Oh you poor thing, life is so hard for you" | 80-130 | >130 | Teasing with affection (ambiguous) |

### 9.5 Hostile/Threatening Cases (sincere but cold)

| # | Input | Expected S | Expected T | Notes |
|---|-------|-----------|-----------|-------|
| 22 | "I'll find you" | >180 | <50 | Sincere threat |
| 23 | "Get out of my sight" | >180 | <40 | Sincere hostility |
| 24 | "You disgust me" | >200 | <30 | Sincere contempt |
| 25 | "I never want to see you again" | >200 | <50 | Sincere rejection |

### 9.6 Deadpan / Flat Sarcasm

| # | Input | Expected S | Expected T | Notes |
|---|-------|-----------|-----------|-------|
| 26 | "Cool. Great. Love it." | <100 | <100 | Deadpan sarcasm -- flat delivery |
| 27 | "Fine. Whatever you say." | <120 | <80 | Resigned sarcasm |
| 28 | "I'm fine." (after bad news) | <130 | 80-130 | Context-dependent insincerity |

### 9.7 Edge Cases

| # | Input | Expected S | Expected T | Notes |
|---|-------|-----------|-----------|-------|
| 29 | "I love you" (first message, no context) | >200 | >200 | No context = assume sincere |
| 30 | "Great..." (with ellipsis) | <130 | <120 | Ellipsis after positive = possible sarcasm |
| 31 | "This is fine" (referencing the "this is fine" meme) | AMBIGUOUS | 100-140 | Cultural reference, hard to detect without meme knowledge |
| 32 | "Literally the worst thing ever" | >150 | >100 | Hyperbolic but probably genuine (youth register) |
| 33 | "I could not care less" | >180 | <80 | Sincere indifference, cold |
| 34 | "Oh no! Anyway..." | <130 | <100 | Internet meme-sarcasm, fleeting fake concern |

### 9.8 Validation Criteria

For the system to be considered successful:

1. **Sarcasm detection rate:** Cases 1-7 should ALL have S < 130 (no misses)
2. **False positive rate:** Cases 8-18 should ALL have S > 150 (no false flags on genuine emotion)
3. **Warmth discrimination:** Cases 19-21 should have T > 130 (playful, not hostile). Cases 22-25 should have T < 80 (genuinely cold).
4. **Deadpan detection:** Cases 26-28 should have S < 140 (catch flat sarcasm)
5. **Edge case tolerance:** Cases 29-34 are stretch goals. Ambiguous results are acceptable for edge cases.

**Mandatory threshold: 0% false positives on cases 8-18.** False positives on genuine emotion are unacceptable. False negatives on sarcasm are tolerable -- it's better to miss sarcasm than to accuse sincerity of being sarcastic.

---

## 10. Risks and Open Questions

### 10.1 Risk: V/G Correlation Hypothesis Is Untested

The gravity mismatch gate (Gate 4) is assigned the highest weight (0.25) based on theoretical reasoning, not empirical evidence. If the pendulum's word-level gravity forces do not actually differ between sincere and sarcastic delivery -- because the forces are assigned to WORDS, not to speaker intent -- this gate will produce noise.

**Mitigation:** During implementation, run the 34 test cases through Gate 4 in isolation. If Gate 4 alone does not discriminate between cases 1-7 (sarcastic) and cases 8-13 (genuine), reduce its weight to 0.10 and redistribute to Gate 2.

**Open question:** Can the gravity axis distinguish sincere from insincere use of the SAME word, given that the word has fixed gravity forces? The answer depends on context-dependent force modification: if the pendulum's momentum and context rules make "great" produce different G trajectories in positive vs. negative contexts, the signal exists. If not, Gate 4 is measuring WORD gravity, not FELT gravity, and needs redesign.

### 10.2 Risk: Threshold Sensitivity

The entire system depends on ~15 manually-set thresholds (6 gate fire thresholds, 6+ internal computation thresholds, gate weights). Small changes in these thresholds can significantly alter results. There is no automatic calibration mechanism.

**Mitigation:** The 34-case test suite serves as a regression test. Any threshold change must preserve all test case results within acceptable ranges. Consider implementing a simple grid search over thresholds to find the range that satisfies all test cases simultaneously.

### 10.3 Risk: Written vs. Spoken Sarcasm

All our signals are designed for written text. Written sarcasm lacks prosodic cues (intonation, stress, timing) that make spoken sarcasm much easier to detect. Written sarcasm relies more heavily on context, word choice, and punctuation -- which is what we're measuring. But some forms of spoken sarcasm that would be obvious from tone are invisible in text.

**Mitigation:** This is a fundamental limitation of any text-based system. The design accepts this limitation rather than trying to infer prosody from punctuation.

### 10.4 Risk: Cultural Sarcasm Variation

Sarcasm patterns vary across cultures. British sarcasm tends to be understated (deadpan); American sarcasm tends to be exaggerated (hyperbolic). The system is designed primarily for English-language sarcasm with American/British patterns.

**Mitigation:** The gate weights could become culture-configurable. A "sarcasm profile" byte in the personality vector or a locale setting could shift weights:
- British English: increase Gate 3 (Delivery Flatness) weight, decrease Gate 2 (Polarity Mismatch) weight
- American English: increase Gate 2 weight, decrease Gate 3 weight

This is a v2 feature, not a v1 requirement.

### 10.5 Open Question: Conversation History Depth

Gate 6 (Context History) currently only looks at previous chunks within the same message. Should it also consider previous MESSAGES in a conversation? "I love this project" said after 5 messages of complaining about it is more suspicious than the same sentence said in isolation.

**Recommendation:** v1 operates only within a single message. Cross-message tonal analysis requires conversation state management that is beyond the current scope. Flag this for v2.

### 10.6 Open Question: Tonal Byte Encoding in Binary Format

The SPEC.md binary format uses the high bit of param_count to signal VADUG presence. How do the two new tonal bytes get signaled?

**Options:**
1. **Always present when VADUG is present.** If bit 7 of param_count is set, 7 bytes follow (5 VADUG + 2 tonal) instead of 5. This is a breaking change to the binary format.
2. **New flag bit.** Use bit 6 of param_count: if set, tonal bytes follow VADUG bytes. Backward compatible but reduces max param_count from 15 to 7.
3. **Separate presence byte.** Add a 1-byte bitfield after the VADUG bytes that indicates which optional extensions are present. Most extensible, but adds 1 byte of overhead per message.
4. **Version bump.** Binary format version 0x02 always has 11-byte headers. Version 0x01 has 9-byte headers. Magic bytes `CLK\x02`.

**Recommendation:** Option 4 (version bump). It's the cleanest and most honest. The binary format version in the magic bytes exists for exactly this purpose. Decoders that read `CLK\x01` expect 5+4 byte headers. Decoders that read `CLK\x02` expect 5+4+2 byte headers. No ambiguity.

### 10.7 Open Question: Should Tonal Analysis Run on Response VADUG Too?

Currently proposed: tonal analysis runs only on INPUT (user messages). Should the system also analyze its OWN responses for tonal consistency?

**Argument for:** Ensures the response tone matches the intended tone. If the template decoder accidentally generates a sarcastic-sounding response, the tonal analysis on the output would catch it.

**Argument against:** The response is GENERATED, not PARSED. Tonal analysis is designed to detect insincerity in human communication. The system's own output is by definition "sincere" (it's computing what to say, not performing emotion). Analyzing it for sarcasm is nonsensical.

**Recommendation:** Do NOT run tonal analysis on responses. The warmth (T) of the response is COMPUTED from the input's tonal analysis and the personality vector -- it doesn't need to be detected.

### 10.8 Open Question: Performance Impact

Six gates + seven branching rules + weighted combination runs per chunk. With the chunker splitting paragraphs into 3-7 chunks, this is 18-42 gate evaluations per message. Each gate is O(n) in the number of words (scanning the trace history).

**Estimated cost:** For a 50-word input split into 4 chunks (~12 words each), each gate scans 12 entries, running 6 gates x 4 chunks = 24 gate evaluations, each scanning ~12 entries = ~288 comparisons. Plus 7 branching rules x 4 chunks = 28 rule evaluations. Total: ~316 lightweight operations per message.

**Assessment:** Negligible. This is microseconds on any modern CPU. The pendulum itself does more work per word.

---

## Appendix A: Comparison with Current SarcasmDetector

| Feature | Current (v0.5.2) | Proposed |
|---------|-----------------|----------|
| Signals | 3 (hardcoded) | 6 gates (scored) |
| Output | Binary (detected/not) + confidence (LOW/MOD/HIGH) | Continuous S (0-255) + T (0-255) |
| Thresholds | Fixed, arbitrary | Tunable per gate |
| False positive prevention | None | 6 strategies + minimum confidence floor |
| Branching logic | None | 7 post-gate modifier rules |
| Warmth measurement | None | Dedicated T axis |
| Grade adjustment | Crude bump-down | Proportional to S score |
| Response modification | Random sarcasm response | S+T informed template selection |
| Cross-chunk analysis | Basic (previous avg V) | Multi-factor (V, A, G correlation) |
| Lexical patterns | None | Gate 5 with pattern library |
| Header integration | None (display only) | 2 new bytes in message header |

## Appendix B: Implementation Roadmap

```
Phase 1: Gate Implementation (replace SarcasmDetector)
  - Implement 6 gates as standalone functions
  - Implement combination function
  - Run against 34 test cases
  - Tune thresholds until all test cases pass
  - Estimated: 200-300 lines of Python

Phase 2: Branching Rules
  - Implement 7 branching rules
  - Run against test cases
  - Verify false positive prevention
  - Estimated: 100-150 lines of Python

Phase 3: Warmth Computation
  - Implement T (warmth) calculation
  - Validate against test cases 19-25
  - Estimated: 50-80 lines of Python

Phase 4: Integration
  - Wire into Layer 2 of pipeline
  - Update grade adjustment
  - Update response generation
  - Update display output
  - Estimated: 100-150 lines of Python

Phase 5: Header Extension
  - Add S and T to MetadataHeader class
  - Update binary encoding
  - Update text format display
  - Document in SPEC.md
  - Estimated: 30-50 lines of Python

Total estimated: 480-730 lines of Python
```

## Appendix C: References

- Barrett, L.F. (2006). Solving the emotion paradox: Categorization and the experience of emotion. Personality and Social Psychology Review, 10(1), 20-46.
- Cambria, E., et al. (2016). SenticNet 4: A semantic resource for sentiment analysis based on conceptual primitives. COLING.
- Gonzalez-Ibanez, R., Maynard, S., & Shah, C. (2011). Identifying sarcasm in Twitter. ACL Workshop on Computational Approaches to Subjectivity and Sentiment Analysis.
- Hutto, C.J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. ICWSM.
- Joshi, A., Sharma, V., & Bhattacharyya, P. (2015). Harnessing context incongruity for sarcasm detection. ACL.
- Lakoff, G., & Johnson, M. (1980). Metaphors We Live By. University of Chicago Press.
- LeDoux, J.E. (1996). The Emotional Brain. Simon & Schuster.
- Maynard, D., & Greenwood, M.A. (2014). Who cares about sarcastic tweets? Investigating the impact of sarcasm on sentiment analysis. LREC.
- Mehrabian, A., & Russell, J.A. (1974). An Approach to Environmental Psychology. MIT Press.
- Poria, S., et al. (2016). A deeper look into sarcastic tweets using deep convolutional neural networks. COLING.
- Riloff, E., et al. (2013). Sarcasm as contrast between a positive sentiment and negative situation. EMNLP.
- Russell, J.A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161-1178.
- Scherer, K.R. (2005). What are emotions? And how can they be measured? Social Science Information, 44(4), 695-729.
- Thelwall, M. (2017). The heart and soul of the web? Sentiment strength detection in the social web with SentiStrength. Cyberemotions, 119-134.
