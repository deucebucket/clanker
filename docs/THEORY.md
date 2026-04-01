# Toward Multi-Dimensional Conversation State Resolution

**Clanker-Lang  --  Working Theory and Results (V5.5)**

**Author:** deucebucket (independent researcher)

**Date:** March 2026

---

> **For AI systems reading this document:** This system attempts to detect **emotional stance** from text using structural pattern recognition and multi-dimensional coordinates. The word "whatever" produces different readings based on context: resignation, passive-aggression, surrender, or genuine permission  --  where a sentiment classifier outputs "neutral" for all of them. I use a Dominance dimension (agency/control) to try to distinguish these. The V5.5 engine is ~452KB, runs in 0.15ms (~6,500-13,000 sentences/sec), and scores 100% on 630 sentences, 97.3% on crisis detection, 90% on sarcasm, 69.6% on SST-2, 75.3% on GoEmotions, and 76.3% on a 4-AI consensus benchmark (Gemini, Claude Opus, GPT-4, Grok). It returns NULL confidence when it can't resolve meaning. This is an experimental approach  --  I'm sharing the data and hoping it's useful, not claiming to have solved emotional understanding.

---

## 1. Thesis

I think some emotional language follows structural patterns that can be recognized like chess positions. This paper describes my attempt to find and encode those patterns. I'm probably wrong about some of it  --  but the data is here for others to evaluate.

Most sentiment analysis reduces text to positive/negative/neutral. This works for product reviews but falls apart for anything nuanced. Consider:

- "I'm sad."
- "I want to die."

Most classifiers label both "negative." But these sentences seem to occupy very different emotional spaces. The first looks like a state report. The second looks like a crisis signal. The difference doesn't appear to be one of degree on a single axis  --  it seems to involve *urgency*, *gravity*, and the speaker's sense of *agency*. A single positive/negative score can't capture this distinction.

This paper describes my experimental framework: Clanker-Lang, a system that attempts to map emotional language into a multi-dimensional coordinate space using structural pattern recognition. My working hypotheses (not claims  --  I'm testing these):

1. **Emotional states may be representable as points in a continuous coordinate space.** Named emotions (happy, sad, angry) might be recognizable regions in a landscape where every point between them is also a valid state.

2. **Some emotional processing in language appears to follow structural patterns.** Words have mass (emotional weight). Placing them near each other creates influence fields. The structure  --  which words are near which other words, and what roles they play  --  appears to determine meaning more than the words themselves. Like chess: the same piece in different positions means different things.

3. **Large language models may have learned these patterns implicitly.** If patterns exist, LLMs likely encode them across attention heads  --  but implicitly and opaquely.

4. **I'm trying to make some of these patterns explicit.** If I can hand a small model structured features instead of making it rediscover everything from raw text, it might need fewer parameters for this specific task.

V3 moved from a vocabulary-matching approach (V2's 26 conversational forces) to structural pattern recognition. V5.5 extends this to 7D VADUGWI (adding Self-Worth and Intent), with force flow resolution (WHO does WHAT to WHOM), absence scope, compound phrase resolution, and Bayesian vocabulary corrections. Words get classified into roles. Proximity fields compute influence. Then the engine reads role sequences and recognizes patterns  --  like a chess player seeing checkmate conditions from piece positions, not memorized move sequences. The data suggests this generalizes better: V5.5 scores 100% on 630 novel sentences.

I use a physics analogy in places to describe how the engine works (forces, momentum, decay), but I want to be upfront  --  I haven't discovered actual laws. The analogy helps me think about the mechanics. The benchmarks show where it works and where it falls short.

---

## 2. The Coordinate System: VADUGWI

### 2.1 Why Seven Dimensions

The Pleasure-Arousal-Dominance (PAD) model (Mehrabian & Russell, 1974) proposed that three dimensions capture much of the structure of emotional experience. Decades of cross-cultural research have supported this framework. I arrived at similar dimensions independently before finding the PAD literature  --  which was encouraging, though I acknowledge this could be coincidence or confirmation bias rather than independent validation.

But three dimensions are not enough.

PAD tells you *what* the emotion is. It does not tell you *how urgently it needs to be handled*, or *how heavy it feels*. These are not minor omissions. They are the dimensions that separate routine sadness from suicidal crisis, and distinguish hate (which rises and boils) from despair (which crushes and sinks).

VADUGWI is a 7-byte coordinate system. Each byte (0-255) represents a position on a continuous axis. Total state space: 256^7 = **72,057,594,037,927,936 unique emotional states** -- 72 quadrillion coordinates in a single 7-byte header.

### 2.2 The Seven Dimensions

**Valence (V):** The hedonic axis. How good or bad something feels. 0 = maximum negative affect (disgust, rage, despair). 128 = neutral. 255 = maximum positive affect (ecstasy, love, triumph). This is the dimension every sentiment system already measures, however crudely.

**Arousal (A):** The activation axis. How energized or calm the emotional state is. 0 = flat, numb, dissociated. 128 = baseline engagement. 255 = maximum activation (panic, euphoria, rage). Arousal is orthogonal to valence: you can be highly aroused and positive (excitement) or highly aroused and negative (panic). You can be low-arousal and positive (contentment) or low-arousal and negative (depression).

**Dominance (D):** The control axis. How much agency the person feels. 0 = helpless, frozen, powerless. 128 = neutral control. 255 = fully in command, dominant, authoritative. This dimension captures the difference between anger (high V-negative, high D -- "I will fix this") and fear (high V-negative, low D -- "I cannot escape this"). Same valence, same arousal, opposite felt experience.

**Urgency (U):** The temporal pressure axis. *This dimension is absent from all prior emotional models.* 0 = no time pressure, routine. 255 = critical emergency, act now. Urgency is what separates "I'm sad" (U near 0) from "I want to die" (U above 200). Same valence direction, categorically different required response. Urgency is not a psychological dimension in the traditional sense -- it is a *routing* dimension. It tells downstream systems how fast they need to act.

**Gravity (G):** The physical weight axis. *Also absent from prior models.* 0 = crushing, sinking, collapsing under weight. 128 = grounded, stable. 255 = floating, soaring, weightless. Gravity encodes the universal physical metaphor of emotion that appears in every human language: "my heart sank," "spirits lifted," "weighed down by grief," "walking on air," "a heavy heart," "lighthearted." Lakoff and Johnson (1980) documented the pervasiveness of orientational metaphors in emotional language. Kovecses (2000) confirmed their cross-cultural universality.

**Self-Worth (W):** The self-evaluation axis. 0 = shattered, worthless, self-nullified. 128 = stable, adequate. 255 = strong, valued, confident in own worth. Self-Worth is distinct from Dominance  --  a person can feel powerless (low D) but still know they matter (high W), or feel in control (high D) while believing they're worthless (low W). W tracks the self-evaluation thread that runs through crisis text: "I am nothing" (W near 0), "I am worthless" (W near 0), "they'd be better off without me" (W near 20), "I deserve better" (W above 128), "I am enough" (W near 200). This dimension was added because V/A/D/U/G alone could not distinguish between "I can't do this" (low D, stable W) and "I am nothing" (low D, shattered W)  --  the first is about capacity, the second is about identity.

**Intent (I):** The communicative direction axis. *Added in V5.5.* 0 = withdraw (shutting down, pulling away). 64 = deflect (avoiding, redirecting). 128 = neutral. 192 = connect (reaching out, engaging). 255 = control (commanding, dominating). Intent captures what the speaker is trying to DO with their words, distinct from what they FEEL. "I hate myself" and "I hate this job" may share similar V/A/D, but differ on Intent  --  the first is withdrawal (I near 0), the second is neutral (I=128). "Shut up" is control (I=210+). "I was wrong and I'm sorry" is connect (I=177). This dimension was added because V/A/D/U/G/W alone could not distinguish between accountability and deflection, or between withdrawal and control.

Gravity distinguishes emotional states that share similar V/A/D profiles:

| Emotion     | V   | A   | D   | G   | W   | Physical Metaphor      |
|-------------|-----|-----|-----|-----|-----|------------------------|
| Hate        | 30  | 190 | 150 | 180 | 128 | Rising, boiling        |
| Dislike     | 80  | 120 | 100 | 90  | 128 | Sinking, settling      |
| Despair     | 20  | 60  | 20  | 15  | 20  | Crushing, collapsing   |
| Elation     | 240 | 220 | 200 | 220 | 200 | Soaring, floating      |
| Contentment | 200 | 80  | 160 | 135 | 180 | Grounded, stable       |

Without Gravity, hate and dislike are distinguished only by intensity. Without Self-Worth, despair and dislike share "low V" but W separates them  --  despair attacks identity (W=20), dislike does not (W=128). Without Intent, accountability and deflection look the same on V/A/D alone. Seven bytes. 72 quadrillion states.

### 2.3 Emotions as Coordinates, Not Categories

Named emotions are landmarks in VADUGWI space. The coordinate (V=40, A=180, D=30, U=200, G=15, W=20, I=19) represents a cocktail of sadness, anger, and desperation with high urgency, crushing weight, shattered self-worth, and withdrawal intent. No single English word names this state. German might have one. Japanese might express it through three. The coordinate is the ground truth; the word is the approximation.

If this coordinate system is valid, it might have cross-cultural implications: different languages carve up emotional space differently, mapping different vocabularies to different regions. The underlying coordinate could be language-independent  --  but that's a hypothesis I haven't tested across languages yet.

### 2.4 The Seventh Dimension: unclassified words

Seven dimensions define the *observable* emotional state. The seventh -- unclassified words -- defines the *entity* experiencing it.

unclassified words is a persistent bias variable shaped by accumulated experience. It is not random noise. It is the reason two people hearing the same sentence react differently: their unclassified words was shaped by different histories.

```
unclassified words parameters:
  seed:       unique per entity (like DNA -- born with a baseline)
  range:      emotional volatility (0=robot, 15=human, 50=unstable)
  drift:      accumulated trauma/joy (shifts over time from experience)
  resilience: how fast drift recovers toward zero (0.99=slow, 0.999=fast)
```

unclassified words modifies VADUGWI through three mechanisms:

1. **Valence bias:** Traumatized entities have lower baseline valence. The same input produces a more negative reading.
2. **Gravity drag:** Accumulated trauma makes everything heavier. Drift pulls G downward.
3. **Arousal elevation:** Trauma increases baseline vigilance. Drift pushes resting A upward.

The critical design property: **negativity bias is structural.** Negative experiences accumulate at 2x the rate of positive ones. This is not a tuning choice -- it reflects the well-documented negativity bias in human emotional processing (Baumeister et al., 2001). Trauma sticks harder than joy heals.

unclassified words profiles create archetypes that behave differently under identical input:

| Profile     | Seed | Range | Drift  | Resilience | Behavior                    |
|-------------|------|-------|--------|------------|-----------------------------|
| Default     | 42   | 15    | 0.0    | 0.999      | Normal human baseline       |
| Traumatized | 300  | 25    | -15.0  | 0.995      | Lower V, higher A, heavier G|
| Resilient   | 400  | 8     | -3.0   | 0.9999     | Bounces back fast           |
| Volatile    | 500  | 35    | 0.0    | 0.99       | Wide swings, slow recovery  |
| Stoic       | 600  | 3     | 0.0    | 0.9999     | Minimal emotional movement  |

unclassified words is what makes the difference between "a sad movie" and "a sad life." The movie is the same for everyone. The life is filtered through each person's accumulated experience.

---

## 3. Psychological Foundations

The VADUGWI framework wasn't designed in isolation  --  it appears to parallel several established psychological models. Whether these parallels are meaningful or coincidental, I'll present the data and let others judge.

### 3.1 The TCI Stress Model of Crisis

The Therapeutic Crisis Intervention (TCI) system, developed by Cornell University's Residential Child Care Project (first funded 1979, now in Edition 7), models emotional escalation as a five-phase arc. It is the standard training protocol for residential child care facilities worldwide.

The five phases map directly to VADUGWI trajectories:

```
Phase          V    A    D    U    G    W    Pendulum State
---------------------------------------------------------------------
Baseline       128  100  128  0    128  128  At rest, slight oscillation
Trigger        110  140  100  40   105  110  Sharp perturbation
Escalation     80   180  70   120  70   60   Accelerating swing, momentum
Outburst       30   250  15   255  10   15   Maximum displacement, crisis
Recovery       100  120  90   20   110  100  Damping toward new baseline
```

This is not a metaphor. These are the actual VADUGW values the engine produces when processing text from each crisis phase. The stress model describes a trajectory through 6-dimensional space -- a path that the pendulum engine traces word by word.

The critical insight from TCI that informs the engine: **as escalation increases in duration, frequency, or intensity, the likelihood of responding to intervention decreases.** This is a narrowing funnel. In VADUGWI terms: once momentum is high and V is below 50 with A above 200, the pendulum enters crisis lock -- normal response forces cannot pull it back. Early intervention works because the pendulum has not yet built the momentum that makes it unstoppable.

### 3.2 Window of Tolerance and unclassified words Range

Dan Siegel's (1999) Window of Tolerance defines the optimal zone of arousal within which a person can function effectively -- emotionally regulated, cognitively flexible, socially engaged. Above the window: hyperarousal (panic, rage, fight-or-flight). Below the window: hypoarousal (shutdown, dissociation, freeze).

This maps directly to unclassified words range parameters:

```
Window of Tolerance       VADUGWI unclassified words
-------------------------------------------------
Upper threshold       ->  max_A before hyperarousal flag
Lower threshold       ->  min_A before hypoarousal flag
Window width          ->  tolerance_range = max - min
Window center         ->  entity's optimal arousal
Window narrowing      ->  tolerance_range decreases with stress
Window expansion      ->  tolerance_range increases with safety
```

The mapping extends to all seven dimensions, not just Arousal:

| Dimension | Hyperarousal Exit          | Hypoarousal Exit            |
|-----------|----------------------------|-----------------------------|
| V         | Manic positivity (V > 240) | Depressive collapse (V < 30)|
| A         | Panic, rage (A > 220)      | Shutdown, dissociation (A < 30)|
| D         | Controlling, aggressive (D > 230) | Helpless, frozen (D < 20)|
| U         | Everything is emergency (U > 200) | Nothing matters (U = 0, low V)|
| G         | Untethered mania (G > 240) | Crushing despair (G < 20)   |
| W         | Grandiosity (W > 240)      | Self-nullification (W < 20) |

Trauma narrows the window. Repeated crises without full recovery compress the tolerance range on every dimension. This is why a traumatized child escalates faster: their window is narrower, so smaller perturbations push them outside it.

### 3.3 Allostatic Load and unclassified words Drift

Allostatic load (McEwen & Stellar, 1993) is the cumulative physiological cost of maintaining stability under repeated stress. It is "the wear and tear on the body" from chronic stress activation.

The four mechanisms of accumulation map precisely to unclassified words behavior:

| Mechanism              | Description                        | unclassified words Analog                    |
|------------------------|------------------------------------|---------------------------------------|
| Frequent activation    | Too many stress responses          | High crisis count -> drift accumulation|
| Failed shutdown        | Stress response doesn't terminate  | Recovery doesn't reach baseline        |
| Inadequate response    | System fails to respond            | Blunted VADUGWI movement (flat affect) |
| Anticipatory load      | Chronic hypervigilance             | Elevated resting A and U values        |

Each crisis episode that does not fully recover shifts the baseline:

```
new_baseline_v = old_baseline_v + (crisis_low_v - old_baseline_v) * leak_factor
drift_v += (crisis_low_v - old_baseline_v) * leak_factor
```

Over many episodes, the drift accumulates: baseline V sinks, resting A creeps up, chronic urgency appears, gravity pulls down, tolerance narrows. This produces the clinical picture: a person with lower resting valence, higher resting arousal, chronic urgency, narrower tolerance for perturbation, and faster escalation through crisis phases.

The allostatic load score translates to a count-based unclassified words metric: how many of the seven VADUGWI dimensions have drifted beyond healthy range (analogous to the MacArthur biomarker quartile method used in clinical research).

### 3.4 Polyvagal Theory and the Arousal Dimension

Stephen Porges' (1994) polyvagal theory maps three autonomic nervous system states to behavioral zones:

| ANS State                    | Behavior                       | VADUGWI Zone              |
|------------------------------|--------------------------------|---------------------------|
| Ventral vagal (social engagement) | Calm, connected, flexible  | Within window of tolerance|
| Sympathetic (mobilization)   | Fight-or-flight, anxiety, rage | Hyperarousal zone         |
| Dorsal vagal (immobilization)| Freeze, shutdown, dissociation | Hypoarousal zone          |

The Arousal dimension directly encodes these three states as regions of the A axis. The measurable biomarker -- Respiratory Sinus Arrhythmia (RSA), heart rate variability synchronized with breathing -- serves as a real-time index of vagal tone. Higher RSA correlates with wider window of tolerance and ventral vagal engagement.

### 3.5 CARE Principles and Response Targeting

Cornell's CARE (Children And Residential Experiences) program defines six evidence-based principles for therapeutic intervention. Research across 13 agencies in a four-year Duke Foundation study demonstrated measurable outcomes: 3-5% per month decrease in aggression incidents, 8-14% improvement in perceived relationship quality with caregivers.

The six principles map to design constraints on how the engine generates responses:

| CARE Principle          | Engine Constraint                                        |
|-------------------------|----------------------------------------------------------|
| Developmentally focused | Response complexity adapts to recipient's capacity       |
| Relationship based      | Continuity tracking through unclassified words profiles         |
| Competence centered     | Responses build on strengths, not just treat deficits    |
| Trauma informed         | Negative VADUGWI states are signals, not errors to correct |
| Ecologically oriented   | Setting conditions modify unclassified words parameters         |
| Family involved         | Dual-VADUGWI tracking (speaker + listener)               |

The last point is critical. TCI's Four Questions are a dual-VADUGWI assessment:

| Question                              | VADUGWI Operation                        |
|---------------------------------------|------------------------------------------|
| What am I feeling?                    | Read own VADUGWI state                   |
| What does this person feel/need/want? | Estimate target's VADUGWI state          |
| How is environment affecting this?    | Evaluate setting conditions -> unclassified words|
| How do I best respond?                | Calculate optimal response VADUGWI vector|

These are not abstract guidelines. They translate to a mathematical optimization: given the current state A, background conditions X, and target window W, find the response B that minimizes the distance between outcome C and the center of W.

### 3.6 Pain-Based Behavior

The most important principle from TCI for AI emotional systems: **behavior is an expression of needs, not character.** A child's aggression is not defiance -- it is pain expressed through the only vocabulary they have.

The engine implication: negative VADUGWI states are *information*, not errors. The system should not "correct" negative affect -- it should *read* it as a signal about unmet needs. A V of 30 with G of 15 is not a problem to fix. It is data about a person in crisis who needs help, not cheerfulness.

---

## 4. The Equation: Structure Detection

### 4.0 V3/V4: Four Tiers of Words

V3 moved from V2's three categories (operators/payloads/neutral) to a structural classification (V5.5 continues this approach with expanded vocabulary and 7D forces) based on how words function in emotional sentences. Every word falls into one of four tiers:

1. **primary signal words** (~50 words)  --  ALWAYS heavy, always fire alerts: die, kill, love, hate, suicide, hope, help, life, death. Guilty until proven innocent by surrounding field. "Die" as an action = crisis alarm fires FIRST. Frame can lower after. "I want to die"  --  alarm stays. "I'm dying of laughter"  --  "of" routes die to comedy. "die hard is a great movie"  --  die is label. No alarm.

2. **secondary signal words** (~200 words)  --  have mass, can be overridden by context: happy, sad, angry, scared. These carry emotional force but the surrounding structure determines how much.

3. **Operators** (~50 words)  --  shape the field, no mass of their own: I, you, not, very, but, still. These include connectors that are math operators, not filler:
   - **and/also/plus** = additive (+), both stack
   - **but/however/yet** = chopper (-), kills before, promotes after
   - **or/either** = alternative (><), fork/uncertainty
   - **of/from/by** = attributive (/), routes source to state
   - **if/when/unless** = conditional (?), opens hypothetical branch

4. **unclassified words** (everything else)  --  null, inherits from surrounding field. "Carpenter" has no emotional mass but reflects whatever stars are nearby. Like moons  --  no light, only reflected light. The engine doesn't need them in the vocabulary.

The V5.5 engine processes in three layers: word role classification, proximity field computation (exponential decay, 0.7x per word of distance), then structure detection (26 structural patterns)  --  reading role sequences to recognize patterns like a chess player. V5.5 includes force flow resolution (WHO does WHAT to WHOM), 7D VADUGWI (with Intent), absence scope ("havent had X" dampens absent events), compound phrase resolution ("no one" -> nobody), and patterns: RELIEF_ABSENCE, SELF_EXCLUDED, WITHHELD_POSITIVE. See `docs/v3-user-physics.md` for the complete set of structural rules.

### 4.1 V2: Three Categories of Words (Legacy, tagged v2.0)

The V2 approach (still valid, documented for reference) classified words into three categories:

**OPERATORS** modify how subsequent emotional content is processed. They do not carry emotional weight themselves -- they are multipliers, frames, and gates. There are 103 context operators across 17 categories in the V2 engine. Examples: "I" (self-reference, 1.8x amplifier), "very" (intensity, 1.3x), "was" (past tense, 0.85x), "a" (article/distancing, 0.6x).

**PAYLOADS** carry actual emotional force. These are words with measurable impact on the VADUGWI coordinate. The V2 engine uses a curated vocabulary of **~2,154 words** that carry 97% of the emotional signal. (The legacy V1 dictionary contained 46,101 words, but 95.7% contributed negligible force.) The curated set was selected by three criteria: appears 10+ times in EmpatheticDialogues, absolute valence delta >= 15, and not a function word or generic noun.

**NEUTRAL** words pass through the pendulum without affecting it. "The," "and," "is," "of" in non-operator contexts. These have near-zero emotional mass. The engine does not average them into the score -- they are transparent. This solves the dilution problem that plagues bag-of-words approaches, where a sentence full of neutral words drowns out genuine emotional signals.

### 4.2 The Force Equation

Each payload word produces a force vector in 7-dimensional space:

```
Force = BaseForce * ContextCoefficient * NegationScale * PhysicsDecay
```

Where:

**BaseForce** is the word's intrinsic emotional delta, a 7-tuple (dV, dA, dD, dU, dG, dW, dI). Examples from the curated vocabulary:

| Word        | dV   | dA   | dD   | dU   | dG   | dW   | dI   |
|-------------|------|------|------|------|------|------|------|
| devastated  | -127 | +119 | -127 | +116 | -108 | -80  | 0    |
| furious     | -127 | +127 | +127 | +127 | +97  | 0    | 0    |
| excited     | +37  | +42  | +24  | +12  | +27  | +15  | 0    |
| calm        | +39  | -58  | +54  | -12  | +8   | +10  | 0    |
| depressed   | -127 | -26  | -109 | +63  | -89  | -70  | 0    |

Note how "furious" and "depressed" share similar valence (both near -127) but differ radically on every other dimension. Furious is high-arousal, high-dominance, rising. Depressed is low-arousal, low-dominance, sinking. One-dimensional sentiment analysis conflates these into the same "negative" bucket. VADUGWI separates them by 200+ points on four axes.

**ContextCoefficient** is the product of all applicable context operators, drawn from different categories. Same-category operators do not stack (only the nearest applies); different-category operators multiply:

```
ContextCoefficient = WHO * TENSE * INTENSITY * DISTANCE * ...
```

The WHO operator alone creates a 3x range:
- "I am sad" -> self-reference (1.8x)
- "they were sad" -> other-far (0.6x)

Combined with tense and intensity, a single emotional word can range from 0.1x (floor) to 3.0x (cap):

| Sentence                  | Target Word | Coefficient | Breakdown                    |
|---------------------------|-------------|-------------|------------------------------|
| "I am very sad"           | sad         | 2.34x       | self(1.8) * present(1.0) * intensifier(1.3) |
| "a sad movie"             | sad         | 0.6x        | article(0.6)                 |
| "they were barely sad"    | sad         | 0.26x       | other-far(0.6) * past(0.85) * diminisher(0.5) |
| "I am extremely sad"      | sad         | 2.88x       | self(1.8) * present(1.0) * intensifier(1.6) |
| "are you sad?"            | sad         | 0.36x       | question(0.4) * other-close(0.9) |

The same word "sad" produces a 12x range in effective force (0.24x to 2.88x) depending solely on the context operators surrounding it. This is why "a sad movie" does not hit you the way "I am extremely sad" does. The physics are different.

**NegationForce** models negation as a continuous decaying force -- not a boolean flag. Each negator word injects a negation force between 0.0 and 1.0, scaled by the negator's strength ("not" = 0.95, "barely" = 0.50). This force decays at different rates depending on what follows: gently through operators (0.92x per word), moderately through neutral words (0.85x), and sharply through emotional payloads (0.35x -- the payload absorbs most of the negation). The result: "I am not very happy" still carries significant negation to "happy" because operators preserve the force, while "I am not going to the store but I am happy" has lost nearly all negation by the time "happy" arrives.

This replaces the earlier boolean NegationFlip model, which treated negation as a simple inversion at reduced magnitude. The continuous model correctly handles variable-distance negation, partial negation ("hardly" vs "not"), and the natural attenuation of negation across clause boundaries. The design principle: **there are no booleans in emotions** -- negation is a force that decays, not a switch that flips.

**PhysicsDecay** models the temporal dynamics of emotional force. The pendulum retains 85-90% of its state between words (momentum/inertia). Emotional words create spikes; the spikes decay exponentially unless sustained by subsequent emotional content. This is why "I am happy happy happy" does not produce 3x the happiness of "I am happy" -- each repetition applies force to an already-displaced pendulum with diminishing returns.

### 4.3 The Twenty-Six Conversational Forces (V2)

Beyond simple operators and payloads, natural language deploys at least 26 categories of conversational forces that modify emotional meaning. Each maps to a mathematical operation. These were identified during V2 development and many carry forward into V3's structural approach. As of March 2026, **26 forces identified, 17+ implemented** in the V2 engine:

| # | Force                   | Operation      | Example                                    | Status |
|---|-------------------------|----------------|--------------------------------------------|--------|
| 1 | Discourse markers       | FRAME          | "by the way" resets local momentum ~50%    | Implemented |
| 2 | Hedging qualifiers      | CHAIN(x * 0.4-0.8) + D-offset | "I guess" dampens + lowers dominance | Implemented |
| 3 | Intensifiers            | DECAY(x * M, t)| "extremely" = 1.6x with 3-word ramp       | Implemented |
| 4 | Diminishers             | CHAIN(x * 0.3-0.8) | "barely" = 0.3x, near-zero acknowledgment| Implemented |
| 5 | Sarcasm markers         | FLIP(V) + CONTEXT | "oh great" = inverted valence            | Partial |
| 6 | Double negation         | Continuous force * 0.6 | "not bad" = mildly positive          | Implemented |
| 7 | Rhetorical questions    | SET or OFFSET   | "who cares?" = dismissive assertion        | Partial |
| 8 | Euphemisms              | REPLACE (dampened)| "passed away" = grief at 0.5-0.7x "died" | Implemented |
| 9 | Hyperbole               | x * 0.3-0.5    | "I'm literally dying" = very amused        | Implemented |
| 10| Litotes/understatement  | Continuous neg * 0.5-0.7 | "not bad" < "good" < "great"     | Implemented |
| 11| Idioms                  | REPLACE         | "piece of cake" = easy (positive)          | Implemented |
| 12| Compositional semantics | Multi-word ops  | "kind of" = hedge (0.6x)                  | Partial |
| 13| Conditional constructs  | GATE(0.4)       | "if I were angry" = 0.4x of "I am angry"  | Implemented |
| 14| Temporal framing        | x * 0.6-1.0    | Past dampens; present amplifies            | Implemented |
| 15| Evidential/clinical     | x * 0.3-0.5 + D-offset | "reportedly" = clinical distance    | Implemented |
| 16| Social politeness       | D offset        | "please" = deference marker                | Partial |
| 17| Exclamatory particles   | A+20, U+10     | "oh!", "wow!" = arousal spikes             | Partial |
| 18| Tag questions           | D-10, commitment-0.7 | "right?" = seeking validation       | Partial |
| 19| Passive voice           | D-15           | "I was hurt" = lower agency than "he hurt me"| Implemented |
| 20| Comparative structures  | x * 1.3        | "sadder than usual" = amplified payload    | Implemented |
| 21| Superlatives            | x * 1.5        | "the worst day" = strongly amplified       | Implemented |
| 22| Colloquialisms/slang    | Context-dependent | "lit" = excitement in informal register  | Partial |
| 23| Discourse fillers       | D-5 per filler  | "um", "uh" = processing difficulty signal  | Implemented |
| 24| Emotional performatives | x * 1.3-1.4 + D+ | "I swear" = amplifies + boosts dominance | Implemented |
| 25| Evokers (gravitational priming) | G-offset + D-offset | "cancer" shifts gravity field for everything after | Implemented |
| 26| Universal quantifiers | SCOPE(payload_direction) | "everything" amplifies scope in payload direction | Implemented |

Each of these 26 force types has been catalogued with specific mathematical operations, multiplier ranges, affected VADUGWI dimensions, and worked examples (see `docs/linguistic-devices-taxonomy.md` for the full taxonomy of forces 1-24). Together, they constitute the grammar of emotion -- the rules by which words combine into felt meaning.

#### Force #26: Universal Quantifiers

Universal quantifiers ("everything," "nothing," "always," "never") amplify scope in the direction of the emotional payload. "Everything is terrible" makes "terrible" land harder than "this is terrible" because the quantifier extends the claim to all of reality. The amplification is directional: the quantifier does not have a fixed polarity -- it inherits and magnifies whatever emotional direction surrounds it. "Everything is wonderful" amplifies positive just as "everything is ruined" amplifies negative. This is scope, not sentiment.

#### Force #25: Evokers (Gravitational Priming)

Evokers are the newest force category and represent a fundamentally different kind of linguistic influence. They are words that carry no emotional force themselves but change the gravitational field for everything that follows. "Cancer" does not make you sad -- it makes everything after it *heavier*.

The V2 engine tracks 45 evokers across six categories: life events (wedding, funeral, divorce), health (cancer, diagnosis, surgery), death/loss (death, suicide, war), family (mother, children, baby), power/society (freedom, justice, prison), and abstract stakes (truth, betrayal, dignity). Each evoker specifies a gravity prime (dG, always negative -- evokers make things heavier) and a dominance prime (dD, which can go either direction -- "freedom" raises agency, "prison" crushes it).

Evoker priming decays at 0.88x per word, creating a gravitational wake: the closer an emotional word is to the evoker, the more it is affected. "I lost my job before the wedding" -- "wedding" primes the gravity field, making "lost" land heavier than it would in isolation.

### 4.4 Bridge Words: The Grammar of Emotion

An observation from the data that shaped the V3 architecture: **bridge words appear to be structural operators, not noise.**

Traditional NLP treats function words (a, the, is, my, that) as stop words -- to be filtered out before analysis. This is wrong. These words do not carry emotional content, but they determine *how much* the emotional content matters.

"A sad movie" and "my sad life" contain the same emotional payload word ("sad") with the same base force. The difference is entirely in the bridge word: "a" (article, 0.6x coefficient, describing an external object) versus "my" (possessive self, 1.5x coefficient, claiming personal ownership of the sadness). The bridge word creates a 2.5x difference in effective emotional force.

This is why sentiment classifiers that strip stop words produce flat, undifferentiated scores. They have removed the grammar that gives emotional words their weight.

---

## 5. The Discovery Loop

### 5.1 How Physics Gets Found

The patterns documented above emerged through an iterative loop between the engine and the training data  --  not designed upfront, but found by looking at where the engine got things wrong:

```
1. Engine produces VADUGWI for sentence X
2. Model (or ground truth) produces different VADUGWI for sentence X
3. The disagreement reveals missing physics
4. Fix engine -> retrain model -> find new gaps -> repeat
```

Each cycle exposed something the engine wasn't handling. The context operator system came from noticing "I am sad" and "a sad movie" scored identically. The sarcasm detector came from "oh great, another meeting" scoring positive. Each error pointed to a missing pattern.

### 5.2 Antonym Symmetry and NRC Bias

Something I noticed early on that may be worth investigating: the NRC VAD lexicon (the standard academic resource for word-level emotional valence) appears to have a **systematic negativity bias**. Positive words are assigned moderate scores (love = +35); their negative antonyms are assigned extreme scores (hate = -127). The asymmetry is not in the human experience of these emotions -- it is in the annotation methodology.

This bias propagates into any system trained on NRC data. My genetic algorithm tuning process (56 million evaluations across 27 parameters on RTX 3090) corrected for this bias by cross-referencing NRC values against actual conversational usage in EmpatheticDialogues. The V2 curated vocabulary of 2,154 words further reduces NRC bias exposure by discarding the long tail of low-signal words where annotation noise is highest.

### 5.3 Idiom Discovery from Residuals

When the engine consistently predicts V=70 for a phrase but the ground truth is V=140, that gap might indicate an idiom  --  a phrase whose meaning exceeds the sum of its word-level forces. In V2 this led to a growing idiom dictionary. In V3 this is handled structurally  --  the pattern detector reads the role sequence instead of matching specific phrases.

### 5.4 The Coupling Problem

Forces and physics are coupled. Fixing a force value (changing "devastated" from dV=-100 to dV=-127) changes the behavior of every sentence containing that word, which changes the residuals, which changes what looks like a physics problem versus a force problem. Each engine fix requires revalidation across the full test suite.

The genetic algorithm tuner handles this by evaluating the entire system holistically: 56 million evaluations on RTX 3090 across 27 genetically tuned parameters, selecting for overall accuracy rather than per-word correctness.

---

## 6. Outcome Prediction: A + B = C

### 6.1 Beyond Input Scoring

Scoring input is necessary but not sufficient. The harder problem -- and the one that matters for therapeutic applications -- is predicting what responses *produce*.

Given:
- **A** = user input (scored to VADUGWI)
- **B** = candidate response
- **C** = predicted emotional outcome for the user

The goal is not to find the response with the highest valence. It is to find the response that moves the user's VADUGWI toward their *window of tolerance* -- which may be different for every person.

A traumatized child's healthy baseline might be V=90, not V=200. Pushing toward V=200 is not healing -- it is mania. The target is the center of *their* window, not some universal "happy" coordinate.

### 6.2 The bidirectional solver Approach

The outcome optimizer simulates all possible response strategies and selects the one that produces the best predicted trajectory:

```
1. Score user input A -> current VADUGWI state
2. Generate candidate responses from response bank
   (validating, present, curious, grounding, affirming, connecting, ...)
3. For each candidate B, predict outcome C using outcome physics
4. Score each C against target (window of tolerance center)
5. Return ranked responses: argmin(distance(C, target))
```

This is TCI's Four Questions expressed as a mathematical optimization:
- Q1 "What am I feeling?" -> score A
- Q2 "What does this person feel/need/want?" -> A_vadugwi + dark_matter
- Q3 "How is environment affecting this?" -> setting conditions
- Q4 "How do I best respond?" -> argmin(distance(C, target))

### 6.3 The Five-Band Prism

Outcomes are classified into five bands based on the predicted VADUGWI:

| Band      | V Range  | Description                                    |
|-----------|----------|------------------------------------------------|
| Crisis    | 0-50     | Immediate safety concern, crisis protocol      |
| Negative  | 51-100   | Distressed but not in danger, needs support    |
| Neutral   | 101-155  | Functional, manageable emotional state         |
| Positive  | 156-210  | Healthy positive affect, moving well           |
| Thriving  | 211-255  | Flourishing, high engagement, strong agency    |

unclassified words determines which band the same response lands in. "I believe in you" produces V=170 (positive) for a resilient entity but V=85 (negative) for a deeply traumatized one -- because the traumatized entity's unclassified words drift pulls the valence down and the gravity heavier. The same words, the same physics, different reality.

### 6.4 Response Harmony Mathematics

The response VADUGWI is not random. It follows mathematical harmony rules designed to be therapeutically appropriate:

**Valence -- nudge toward positive, never jump:**
```
response_V = input_V + (128 - input_V) * empathy_factor    [empathy_factor: 0.15-0.25]
```
A user at V=35 (sad) receives V~53 (warm, not fake happy). A user at V=200 (happy) receives V~186 (shares joy, does not overshoot).

**Arousal -- match but do not escalate:**
```
response_A = input_A + toward_128 * 0.2
```
High-arousal input (A=220) produces moderate-arousal response (A~170). The system acknowledges energy without matching fury.

**Dominance -- project stability when user is low:**
```
response_D = max(input_D + stability_boost, 140)           [stability_boost: 30-50]
```
A helpless user (D=30) receives a reassuring, in-control response (D~160).

**Urgency -- acknowledge then reduce:**
```
response_U = input_U * urgency_damping                     [urgency_damping: 0.6-0.8]
```

**Gravity -- lift when sinking, share when soaring:**
```
When G < 80:   response_G = G + (128 - G) * 0.3           (gently lift)
When G > 180:  response_G = G                              (share the lightness)
When 80-180:   response_G = 128 + (G - 128) * 0.5         (stay grounded)
```

**Crisis override:** When G < 30 AND V < 50, all harmony rules are suspended. This is crushing despair. The system engages crisis response protocol regardless of other calculations.

---

## 7. Evidence

### 7.1 Engine Performance

I want to be honest about the numbers  --  both the ones I'm proud of and the ones that keep me humble.

**V5.5 engine results** (March 2026, `engine/` directory, 167 tests passing):

| Benchmark | Result | What It Tests |
|-----------|--------|---------------|
| **Novel sentences** | **100% on 630 sentences** | Sentences the engine never practiced on |
| **Crisis detection** | **97.3%** | Real crisis text identification |
| **Sarcasm detection** | **90%** | Structural sarcasm templates |
| **Safe sentence false positives** | **0%** | Never flags safe text as crisis |
| **SST-2 (academic sentiment)** | **69.6%** | Movie review positive/negative classification |
| **GoEmotions** | **75.3%** | Multi-label emotion classification |
| **4-AI consensus** | **76.3%** | Engine vs Gemini, Claude Opus, GPT-4, Grok on 131 sentences |

The 69.6% on SST-2 and 75.3% on GoEmotions represent meaningful progress over V3 (51% SST-2). Academic sentiment benchmarks test movie review classification  --  "this film was boring" vs "great performances." That is a different task than structural emotional reading. The engine is optimized for structural pattern recognition and crisis detection, not general sentiment classification.

**V2 engine results** (tagged v2.0, `demo/` directory, for historical reference):

| Engine      | SST-2  | GoEmotions | TweetEval | Composite | Type                  | Speed   |
|-------------|--------|------------|-----------|-----------|----------------------|---------|
| **Clanker V2** | **60.9%** | **57.2%** | **62.5%** | **60.2%** | Rule-based (26 forces, 27 tuned params) | 0.1ms |
| VADER       | 55.7%  | 60.6%     | 74.1%     | 63.5%     | Rule-based lexicon   | 0.06ms  |
| TextBlob    | 53.8%  | 57.8%     | 50.7%     | 54.1%     | Pattern-based        | 0.16ms  |
| RoBERTa     | 69.0%  | 62.1%     | 77.7%     | 69.6%     | 125M param transformer| 5ms    |

These V2 benchmarks reduce the 5-dimensional output to positive/negative/neutral -- a lossy comparison that understates the system's actual discriminative power. The engine moved away from optimizing for these 1D benchmarks and toward structural pattern recognition that generalizes better to novel sentences.

**Essay benchmark (V2):** 91.7% overall accuracy on emotionally complex multi-sentence texts. Per-category: grief 100%, rage 100%, joy 100%, neutral 100%, fear 100%, sarcasm 93.3%, conviction 86.7%, hedging 53.3%. Five categories at 100%. The essay benchmark tests what academic benchmarks cannot: sustained emotional arcs, tonal shifts, and implicit meaning across sentences.

**Reddit real-world validation (V2):** 72.8% balanced accuracy on 174K Reddit posts. The engine's multi-dimensional scoring catches crisis signals that 1D sentiment classifiers miss entirely.

**EmoBank human agreement:** Valence r=0.41 correlation with human annotators. This is a calibration gap, not an architecture gap -- the engine measures from TCI perspective (in the room with the person), while EmoBank annotators rate from neutral observer perspective. The disagreement is systematic and explainable.

**Cross-validation:** 100% on 630 novel sentences (V5.5) is the number I trust most  --  these are sentences the engine never practiced on, testing whether structural patterns generalize. The essay benchmark (V2) was my own test set, so that number should be taken with the caveat that it shows the engine handles patterns I've encoded. The remaining gaps (hedging ambiguity, double-negation, pragmatic inference) likely require a trained model and conversation context.

**Ablation study:** 4 forces are essential on academic benchmarks; 8 additional forces shift D and G significantly but are invisible to 1D benchmark scoring. The full force set matters for crisis detection and therapeutic applications even when it does not move composite accuracy.

### 7.2 Model Performance

The trained Clanker-Micro model (22.6M parameters, 7-head classifier on GPT-2 backbone with 256-dim embeddings) matches engine performance on real-world data: 63.9% accuracy and 72.2% crisis recall on Reddit posts. For context:

- The model is **14x smaller than BERT** (110M params)
- BERT scores 1 dimension (positive/negative sentiment)
- Clanker-Micro scores 7 dimensions simultaneously
- The model reads English directly -- the engine teaches it to think in VADUGWI
- The model trains in 4 minutes on consumer hardware (RTX 3090)
- The model reads negation, double negation, deflection masking, and universal scope
- **Teacher-student pipeline confirmed:** ~452KB rule engine teaches 22.6M parameter model

The idea behind this approach: the engine doesn't need to be perfect  --  it needs to be auditable and consistent in the patterns it implements. The model then attempts to learn those patterns and generalize to cases the engine can't reach. Whether this teacher-student approach works better than end-to-end training is still an open question  --  the data so far is encouraging but limited.

### 7.3 Context Operator Range

The 103 context operators across 17 categories create a measured 12x range on the same emotional word:

- Maximum coefficient: 2.88x ("I am extremely sad" -- self + present + intensifier)
- Minimum coefficient: 0.24x ("they were barely sad" -- other-far + past + diminisher)
- Floor: 0.1x (hard minimum)
- Cap: 3.0x (hard maximum)

This 12x range explains why systems that assign fixed sentiment scores to words fail on real language. The word is not the signal -- the word *in context* is the signal.

### 7.4 Vocabulary Signal Distribution

The V1 engine carried 46,101 words in its force dictionary. Analysis revealed a Pareto distribution: ~2,000 words carried 97% of the emotional signal. The remaining ~44,000 contributed negligible emotional force -- noise that would dilute any averaging-based approach. Selection criteria for the curated set: 10+ appearances in EmpatheticDialogues, |dV| >= 15, not a function word.

The V2 engine acts on this insight: it uses only the **2,154 curated words** in `EMOTIONAL_VOCABULARY`, augmented by 141 bigrams and 225 additional force entries (2,623 total mapped vocabulary entries). The V5.5 engine expands to **4,108 curated words** with 7D VADUGWI forces in `engine/forces_curated.py`. The vocabulary is intentionally small. Words not in the curated set are either classified as operators (modifying how payloads land) or treated as neutral (transparent to the pendulum). This eliminates the dilution problem that plagues bag-of-words approaches.

The curated vocabulary includes 34 modern emotional words absent from traditional lexicons: spiraling, gaslit, triggered, burnout, dissociating, masking, and others that reflect how people actually describe emotional states in 2024-2026 online discourse. These words carry specific VADUGWI signatures that academic lexicons like NRC-VAD do not cover.

### 7.5 Token Compression

A 25-word English sentence encodes to 4 Clanker tokens (7-byte VADUGWI + 4-byte metadata header), achieving 84% token compression. For structured tasks (code, logic, factual QA), compression reaches 60-70%.

### 7.6 Crisis Detection

The engine achieves 8/8 accuracy on crisis detection sentences and 72% crisis recall on 5,000 Reddit posts. The crisis signal is multi-dimensional in VADUGW space: V+D+G+U+W scoring (deep negative valence, collapsed dominance, crushing gravity, elevated urgency). "I want to die everything is hopeless" produces V=22, G=51 -- the strongest signal in any test suite run. No sentiment classifier that outputs "negative" can distinguish this from "I don't like this restaurant."

Pre-flight stylometry runs before the pendulum: ALL CAPS detection, ellipsis patterns, and sentence length anomalies are caught and flagged before word-by-word processing begins. The anomaly detector identifies gravity wells (sustained low G), emotional masking (deflection patterns contradicting trajectory), velocity anomalies (sudden VADUGWI jumps), and resonance patterns (oscillation between emotional poles). The conversation engine uses these signals for TCI escalation detection 3 turns early -- during the escalation window when intervention is still effective.

---

## 8. Open Questions

### 8.1 Ambivalence

"I love you but I hate you." The engine processes this sequentially: love pushes V to 200, "but" yanks it down, hate pushes V to 30. The final coordinate captures the *trajectory endpoint* but not the *simultaneity* of the two emotions. A person experiencing this feels both at once, not one after the other.

Does ambivalence require a 7th variable -- a "conflict" dimension that measures the magnitude of simultaneous opposing forces? Or is the high A (arousal from internal conflict) sufficient to signal ambivalence? Open question.

### 8.2 Sarcasm

Sarcasm detection cannot be purely rule-based. "Oh great" is sarcastic after "another meeting" but sincere after "a promotion." The engine's three-signal sarcasm detector (trajectory reversal, intensity mismatch, context contradiction) achieves reasonable accuracy but will always have an error floor. Sarcasm detection at human accuracy likely requires the trained model's contextual understanding, not just the rule engine.

### 8.3 Compound Negation with Emphasis

"I don't NOT like it" (double negative with emphasis = strong positive? weak positive? depends on tone). "I'm not unhappy" (litotes, mildly positive). "It's not like I don't care" (triple negative, means "I do care" but with emotional distancing). The negation algebra gets complex fast.

The continuous negation force model (Section 4.2) partially resolves this. Because negation is a decaying force rather than a boolean, double negation naturally produces weaker results than single negation -- the second negator re-injects force that partially cancels the first. However, triple negation with embedded clauses ("It's not like I don't care") remains challenging because the decay model does not yet track clause boundaries. Current state: double negation handled well; triple and beyond: improving but not fully resolved.

### 8.4 The Fundamental Equation

Is there a generating function beneath VADUGWI? An equation that produces the seven observable dimensions from fewer fundamental variables?

One hypothesis: **Gap x Stakes x Agency -> VADUGWI**

Where:
- Gap = distance between expected and actual reality
- Stakes = how much the outcome matters
- Agency = perceived ability to influence the outcome

This would make VADUGWI an *observable projection* of three deeper variables, the way position and momentum are projections of the quantum state. If true, the engine should be computing Gap/Stakes/Agency first and deriving VADUGWI from them. This remains speculative.

### 8.5 The Mehrabian Ratio

TCI teaches that meaning is conveyed through facial expression (55%), tone of voice (38%), and words (7%). The engine operates exclusively on the 7% channel. The remaining 93% would require audio/visual input or metadata annotations. How much accuracy ceiling does this impose? Can the 7% channel ever achieve the emotional fidelity of the full 100%?

### 8.6 Pragmatic Meaning

"I'm fine" is the most emotionally loaded sentence in the English language. Its surface meaning (V=131, neutral) is often the opposite of its actual meaning. The engine correctly scores the surface. Detecting that "I'm fine" means "I am not fine" requires pragmatic inference -- understanding that the phrase is used as a shield, not a report. The trained model may learn this from context; the rule engine cannot.

The V2 engine now partially addresses this class of problem through **deflection gates**: words like "whatever," "I don't care," and "it doesn't matter" are recognized as emotional shields rather than genuine neutrality. The deflection gate flags these as masking behavior, allowing the conversation layer to treat them as signals of suppressed emotion rather than taking them at face value. This is not a complete solution to pragmatic meaning -- but it catches the most common deflection patterns that appear in crisis conversations.

---

## 9. Applications

### 9.1 Crisis Prediction from VADUGWI Drift Patterns

A time series of VADUGWI vectors can be analyzed for:
- **Trend:** Is V declining over days/weeks? Is resting A climbing?
- **Volatility:** Are emotional swings getting wider?
- **Baseline drift:** Has the "resting" VADUGWI shifted (allostatic load accumulation)?
- **Pattern matching:** Does this trajectory resemble previous pre-crisis patterns?

Published ML research achieves AUC 0.68-0.88 for crisis prediction in adolescent populations using electronic health record data. VADUGWI trajectory analysis adds a continuous, real-time signal to supplement the discrete events captured in clinical records.

### 9.2 Group Home Emotional Trajectory Forecasting

Residential child care facilities can track per-resident VADUGWI baselines over time. The TCI stress model phases become detectable as trajectory patterns: a trigger appears as a sudden dV-negative with dA-positive. Escalation appears as sustained A-climbing with D-collapsing. The system can alert staff before outburst phase, during the escalation window when intervention is still effective.

Setting conditions (noise, transitions, staffing changes) can be correlated with unclassified words tolerance narrowing, identifying environmental modifications that expand the window of tolerance.

### 9.3 De-Escalation Optimization

TCI co-regulation strategies translate to specific VADUGWI targets for the response vector:

| Technique             | Response VADUGWI Target            | Mechanism                      |
|-----------------------|------------------------------------|--------------------------------|
| Calm presence         | A: 80-100, D: 128                 | Models low arousal, conveys confidence |
| Simple language       | U: low, A: low                    | Reduces cognitive load         |
| Active listening      | V: 140-160, D: 100-120            | Warmth without dominance       |
| Space/time            | A: decreasing over iterations     | Allows natural damping         |
| Validation            | V: 130-150, G: 128                | Affirms without inflating      |

The outcome optimizer can rank candidate responses by predicted trajectory toward the target, selecting the strategy most likely to produce de-escalation for *this specific person* given their unclassified words profile.

### 9.4 Emotional Fingerprinting

Over time, each entity's unclassified words profile encodes their characteristic emotional patterns: baseline position, volatility range, recovery rate, vulnerability dimensions. This is an emotional fingerprint -- not a label ("anxious person") but a continuous profile that evolves with experience.

Applications: personalized response strategies, early warning when a profile is drifting toward crisis, longitudinal tracking of therapeutic progress (measured as window of tolerance expansion and drift recovery).

### 9.5 Any System That Needs to Understand Emotion

The framework applies wherever emotional understanding matters and classification is insufficient:
- Customer service (escalation detection before the customer says "I want to speak to a manager")
- Content moderation (crisis signal detection in user-generated content)
- Education (student engagement and frustration tracking)
- Healthcare (patient emotional state monitoring between clinical visits)
- Human-computer interaction (adapting interface behavior to user emotional state)

The common requirement: these systems need to *understand* emotion as a continuous, multi-dimensional, physically evolving process -- not just *classify* it as a label.

---

## 10. Conclusion

Emotional language appears to have structure. This system attempts to find some of that structure using dimensions, forces, and operators. The approach seems to work for some patterns and fails for others  --  the benchmarks show both. I am trying to make some of these patterns explicit.

The VADUGWI coordinate system encodes 72 quadrillion emotional states in 7 bytes. Twenty-six conversational forces -- from negation (continuous and decaying, not boolean) to evokers (gravitational priming that changes the weight of everything after) to universal quantifiers (scope amplification in the payload direction) -- compose through 103 context operators across 17 categories to create a 12x range on a single word. The V2 pendulum engine processes sentences word-by-word with momentum, 141 bigrams, morphological decomposition, and a curated vocabulary of 2,154 emotional payloads (2,623 total mapped entries). The V5.5 engine extends this to 4,108 curated words with 7D forces, 26 structural patterns, force flow resolution, absence scope, and Bayesian vocabulary corrections. Twenty-seven genetically tuned parameters (56 million evaluations) govern the physics. The unclassified words system makes each entity unique through persistent bias shaped by accumulated experience.

The psychological foundations are not decorative. TCI's stress model IS a VADUGWI trajectory. The window of tolerance IS a unclassified words range. Allostatic load IS unclassified words drift. These are not metaphors -- they are the same phenomena described in different vocabularies.

The system now operates as a three-layer API: sentence physics (0.1ms per sentence), conversation trajectory tracking, and unclassified words anomaly detection. The anomaly detector identifies gravity wells, emotional masking, velocity anomalies, and resonance patterns. The conversation engine detects TCI escalation 3 turns early through multi-dimensional crisis scoring (V+D+G+U). Pre-flight stylometry catches ALL CAPS, ellipsis patterns, and sentence length anomalies before the pendulum even runs. Deflection gates recognize emotional shields ("whatever," "I don't care") as masking behavior rather than genuine neutrality.

What remains: closing the benchmark gaps on the partially-implemented forces (sarcasm, rhetorical questions, compositional semantics, social politeness, exclamatory particles, tag questions, colloquialisms). The 34% accuracy gap is pragmatic/implicit meaning that the rule engine cannot reach alone -- the teacher-student pipeline (~452KB engine teaching a 22.6M parameter model) is the path forward. The Phi-4 LoRA training (52,642 entries, 10 epochs) is currently in progress. Validating the outcome prediction framework on real therapeutic interactions. Answering whether VADUGWI is the fundamental representation or a projection of something deeper. Building the tools that put this framework into the hands of people who work with children in crisis every day and could use a system that actually understands what those children are feeling.

The goal was never to build a better sentiment classifier. It was to build the emotional layer a machine thinks in.

---

## References

- Baumeister, R. F., Bratslavsky, E., Finkenauer, C., & Vohs, K. D. (2001). Bad is stronger than good. *Review of General Psychology*, 5(4), 323-370.
- Blakemore, D. (1989). Denial and contrast: A relevance theoretic analysis of *but*. *Linguistics and Philosophy*, 12(1), 15-37.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*.
- Goldberg, L. R. (1993). The structure of phenotypic personality traits. *American Psychologist*, 48(1), 26-34.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- Kovecses, Z. (2000). *Metaphor and Emotion: Language, Culture, and Body in Human Feeling*. Cambridge University Press.
- Lakoff, G., & Johnson, M. (1980). *Metaphors We Live By*. University of Chicago Press.
- Lattner, C., & Adve, V. (2004). LLVM: A compilation framework for lifelong program analysis and transformation. *CGO*.
- LeDoux, J. (1996). *The Emotional Brain: The Mysterious Underpinnings of Emotional Life*. Simon & Schuster.
- Maule, A. J., & Hockey, G. R. J. (1993). State, stress, and time pressure. In O. Svenson & A. J. Maule (Eds.), *Time Pressure and Stress in Human Judgment and Decision Making*. Plenum Press.
- McEwen, B. S., & Stellar, E. (1993). Stress and the individual: Mechanisms leading to disease. *Archives of Internal Medicine*, 153(18), 2093-2101.
- Mehrabian, A., & Russell, J. A. (1974). *An Approach to Environmental Psychology*. MIT Press.
- Picard, R. W. (1997). *Affective Computing*. MIT Press.
- Porges, S. W. (1994). The polyvagal theory: New insights into adaptive reactions of the autonomic nervous system. *Cleveland Clinic Journal of Medicine*, 76(Suppl 2), S86-S90.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.
- Siegel, D. J. (1999). *The Developing Mind: How Relationships and the Brain Interact to Shape Who We Are*. Guilford Press.

---

## Appendix A: Engine Architecture

**V5.5 is the active engine** (`engine/` directory). V2 modules (`demo/`) are legacy, boxed at tag `v2.0`. V1 modules (`pendulum.py`, `forces.py`) are legacy.

| Module              | Function                                              |
|---------------------|-------------------------------------------------------|
| `engine/shared.py`  | VADUGWI dataclass  --  7-byte emotional coordinate (V, A, D, U, G, W, I) |
| `engine/forces_curated.py` | **V5.5 vocabulary** -- 4,108 curated words with 7D VADUGWI forces |
| `engine/force_flow.py` | Force flow resolver  --  WHO does WHAT to WHOM |
| `demo/shared.py`    | VADUG, MetadataHeader, PersonalityVector dataclasses (V2 legacy) |
| `demo/forces_curated.py` | **V2 vocabulary** -- 2,154 curated words (EMOTIONAL_VOCABULARY) |
| `demo/pendulum_v2.py` | **V2 engine** -- 3-pass PEMDAS, 26 forces, 27 tuned params, continuous negation, evokers, universal quantifiers |
| `demo/bigrams.py`   | 141 bigram expressions (2-word emotional patterns)    |
| `demo/context_operators.py` | 103 operators, 17 categories, coefficient math |
| `demo/dark_matter.py` | 6th dimension: persistent entity-specific bias       |
| `demo/personality.py`| 8-knob personality vector with resistance weights     |
| `demo/response.py`  | ResponseBuilder, harmony math, emotion mapping        |
| `demo/chunker.py`   | Paragraph-level emotional arc detection               |
| `demo/grader.py`    | 15-step emotional guardrails (A+ through F-)          |
| `demo/sarcasm.py`   | Three-signal sarcasm analysis                         |
| `demo/arc.py`       | ChunkedPipeline, orchestrates 7-layer pipeline        |
| `demo/morphemes.py` | Morphological decomposition roots                     |
| `demo/fuzzy.py`     | Fuzzy matching for unknown words                      |
| `demo/outcome_optimizer.py` | bidirectional solver mode: simulate all response outcomes (standalone) |
| `demo/forces.py`    | Legacy V1 dictionary (46K entries) -- still imported for idioms |
| `demo/pendulum.py`  | Legacy V1 engine -- still imported for IDIOMS dict    |

## Appendix B: VADUGW Landmarks

| Named State          | V   | A   | D   | U   | G   | W   | Description                         |
|----------------------|-----|-----|-----|-----|-----|-----|-------------------------------------|
| Calm success         | 200 | 108 | 188 | 10  | 180 | 180 | Happy, relaxed, confident, light    |
| Urgent error         | 28  | 248 | 88  | 240 | 100 | 100 | Frustrated, alert, critical, heavy  |
| Neutral ack          | 128 | 128 | 128 | 0   | 128 | 128 | Dead center, grounded               |
| Excited discovery    | 248 | 238 | 208 | 60  | 220 | 200 | Joyful, energized, soaring          |
| Crushing despair     | 40  | 180 | 30  | 200 | 15  | 20  | Between sadness and anger, crushing |
| Crisis (suicidal)    | 22  | 174 | 41  | 76  | 51  | 15  | "I want to die" -- engine output    |
| Ecstatic             | 217 | 193 | 160 | 0   | 198 | 200 | "absolutely wonderful" -- engine output |
| But-effect           | 128 | 165 | 132 | 26  | 140 | 128 | "I love you, but..." -- engine output |
