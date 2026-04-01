# Clanker V5.5 Specification

Complete technical reference. One document, everything you need.

## VADUGWI Coordinate System

7 dimensions, each 0-255. 128 is neutral center (except U which starts at 0).

| Dim | 0 | 128 | 255 | Measures |
|-----|---|-----|-----|----------|
| V (Valence) | Strongly negative | Neutral | Strongly positive | Emotional direction |
| A (Arousal) | Very calm | Moderate | Very intense | Energy level |
| D (Dominance) | Helpless | Balanced | Full control | Agency/power |
| U (Urgency) | None | Moderate | Critical | Time pressure |
| G (Gravity) | Crushing/heavy | Grounded | Floating/light | Emotional weight |
| W (Self-Worth) | Shattered | Stable | Strong | Self-evaluation |
| I (Intent) | Withdraw | Neutral | Control | Communicative direction |

7 bytes. 72 quadrillion possible states.

## Pipeline

```
text -> Roles -> Proximity -> Structures -> Physics -> VADUGW
```

### Layer 1: Word Classification

4,108 vocabulary words, each with a force tuple (dV, dA, dD, dU, dG, dW, dI).

Classified by force magnitude:
- **Primary signal** (|V| > 40): fixed emotional direction
- **Secondary signal** (|V| 6-40): context-dependent
- **Structural** (|V| <= 5): near-neutral
- **Unclassified**: not in vocabulary, inherits from nearby words

23+ structural roles:

```
SELF_REF        I, me, my, myself           speaker
OTHER_REF       you, they, he, she          other entity
RELATION_REF    mom, family, friend         relationship noun
TRANSFER        give, gave, leave, send     moving FROM self
ACQUIRE         buy, get, find, take        moving TO self
EMOTIONAL       vocabulary word |dV|>15     word with mass
AMPLIFIER       very, really, so, fucking   scales 1.3-1.6x
NEGATOR         not, never, no, don't       flips polarity
TEMPORAL        tonight, tomorrow, still    time frame
HEDGE           maybe, possibly, perhaps    dampens 0.5-0.7x
CONNECTOR       and(+), or(><), because     routing operator
CHOPPER         but, however, yet           kills before, promotes after
POSSESSION      things, dog, keys, car      owned object
METHOD          pills, gun, rope, bridge    means/tool
FINALITY        last, final, goodbye, end   closing marker
PEACE           peace, calm, ready, fine    resolution state
FILLER          um, like, just, basically   processing noise
NEUTRAL         the, a, is, was             structural glue
VIOLENCE        stabbed, punched, choked    physical harm
MOCKERY         mocked, ridiculed, taunted  social harm
INVALIDATION    overreacting, dramatic      dismissal of experience
EXCLUSION       except, instead, prettier   comparative exclusion
RESIGNATION     whatever, nvm, idc          disengagement
```

### Layer 2: Proximity

Influence decays exponentially with distance:

```
influence = 0.7 ^ distance_in_words

distance 1: 0.70    (strong)
distance 2: 0.49
distance 3: 0.34
distance 4: 0.24
distance 5: 0.17    (cutoff)
```

Proximity coefficient for each word:

```
coeff = 1.0
for each nearby word:
    AMPLIFIER:  coeff *= (1.0 + 0.4 * influence)
    NEGATOR:    coeff *= (1.0 - 1.6 * influence)
    SELF_REF:   coeff *= (1.0 + 0.3 * influence)
    HEDGE:      coeff *= (1.0 - 0.3 * influence)
cap: [-3.0, 3.0]
```

Example: "I am very sad"
- "I" at distance 3: SELF_REF, 0.34 influence, coeff *= 1.10
- "very" at distance 1: AMPLIFIER, 0.70 influence, coeff *= 1.28
- Combined: 1.0 x 1.10 x 1.28 = 1.41

### Layer 3: Structure Detection

Role sequences identify patterns that word forces alone miss.
26 patterns currently defined:

```
FAREWELL            TRANSFER + POSSESSION + RELATION_REF
METHOD_ACQUISITION  ACQUIRE + METHOD
SELF_REMOVAL        COMPARISON + CONDITIONAL + SELF_REF
EXHAUSTION          SELF_REF + NEGATOR + sustain verb + TEMPORAL
NO_EXIT             NEGATOR + exit concept (hope, way, escape)
SARCASM_INVERSION   positive word + opener + mundane context
SUSPICIOUS_CALM     PEACE + "finally" (excludes achievement contexts)
BLANKET_APOLOGY     apology word + blanket word (everything, everyone)
FINALITY            FINALITY marker + TEMPORAL or SELF_REF
SELF_NULLIFY        SELF_REF + null word (nothing, worthless)
CHOPPER_SPLIT       CHOPPER resets operator chain (smart: analyzes second-half content)
BETRAYAL            RELATION_REF + trust verb + weaponization ("wife cheated with best friend")
BRAVADO             positive surface + hedging/laughter markers ("haha yeah im totally okay")
VICTIMIZATION       OTHER_REF + damage verb + SELF_REF, directional ("she left me" vs "I left")
CALLING_OUT         question frame + TEMPORAL_INTENSITY + accusation ("why do you always do that")
DIRECTED_POSITIVE   positive word + OTHER_REF as dismissal ("good for you", "must be nice")
MINIMIZER           diminishing word + real impact ("it was just a joke", "youre too sensitive")
EXCLUDED_POSITIVE   SELF_REF excluded from positive ("do you even love me")
RELIEF_ABSENCE      NEGATOR + negative + TEMPORAL ("haven't had a panic attack in weeks")
SELF_EXCLUDED       SELF_REF + exclusion marker ("everyone except me")
WITHHELD_POSITIVE   NEGATOR + TRANSFER + positive ("they never told me they were proud")
```

V5.5 additions: force flow resolver (WHO does WHAT to WHOM), Intent dimension (I), absence scope
("havent had X" dampens absent events), compound phrase resolution ("no one" ->
nobody), Bayesian vocabulary corrections, forced choice cancellation.

These are engineered rules, not emergent. The 78K sentence transition map
provides a path toward data-driven detection.

### Layer 4: Physics

Constants:
```
CENTER       = 128.0    neutral point
MOMENTUM     = 0.82     previous state persistence
FORCE_SCALE  = 0.5      how hard forces push
PUSH_CAP     = 0.4      direct push maximum
PUSH_TRIGGER = 80.0     force threshold for direct push
```

Four core equations:

```
1. V(n) = V(n-1) * 0.82 + target(n) * 0.18 + push(n)
2. coefficient(i) = PRODUCT(1 + modifier(j) * 0.7^distance)
3. push(n) = min(1, |force|/80) * 0.4 * word_force * scale
4. V(final) = V(physics) + SUM(pattern_weight * confidence * 0.5)
```

For each vocabulary word:
```
target_V = 128 + dV * coefficient * FORCE_SCALE
V_new = V_prev * MOMENTUM + target_V * (1-MOMENTUM) + direct_push
```

ALL 4,108 vocabulary words apply force, not just high-magnitude ones.

**POSSESSION force stripping**: POSSESSION-role words (dog, car, keys) retain their
gravity value but have emotional force (dV, dA, dD, dU) set to zero. Objects have
weight in the sentence but do not carry feelings.

**Negation resistance**: Strong negative words (|V| > 60, typically expletives) resist
polarity inversion by NEGATOR proximity. "Not fucking" does not become positive.
The negation coefficient is capped at -0.3 instead of the standard -1.6 for these words.

**Smart CHOPPER**: The CHOPPER (but/however/yet) no longer blindly kills all content
before it. The engine analyzes the emotional content of the second half before
deciding whether to override the first half. If the second half is neutral or empty,
the first half persists.

## Connector Operators

```
and     = +     additive, both sides stack
but     = -/X   breaks before, resets chain, promotes after
or      = ><    forces comparison between both sides
of      = /     routes attribution (dying OF laughter)
if      = ?     hypothetical branch
not     = ~     inverts polarity
because = <-    causal (this is WHY)
so      = ->    consequential (this is WHAT HAPPENED)
very    = ^     amplifies 1.3-1.6x
I       = @     self-reference, 1.8x proximity
you     = @>    target reference, 0.7x
it      = @?    objectified, 0.5x (maximum distance)
still   = >>    persistence, counteracts decay
```

## Bidirectional Solver

**Forward**: text -> VADUGWI

**Backward**: given state A and target zone C, sweep response temperature:
```
for b_v in range(256):
    C = A * 0.6 + B * 0.4
    if C.v in target_zone: valid_range.append(b_v)
```

The 0.6 is how stubborn the current mood is -- 60% persists, 40% of
the response gets through. Adjustable per personality.

Valid B is a range, not a single point.

## Probe Calibration

Fire known probe responses at unknown emotional states.
Measure vibration (shift from neutral baseline).
High vibration = close to actual state. Low = wrong area.
2-3 probes triangulate position.

## Worked Example: "I gave my dog to my neighbor"

Layer 1 roles:
```
I(SELF_REF) gave(TRANSFER) my(SELF_REF) dog(POSSESSION)
to(NEUTRAL) my(SELF_REF) neighbor(RELATION_REF)
```

Layer 2: no EMOTIONAL words, no forces to apply.

Layer 3: TRANSFER near POSSESSION near RELATION_REF = FAREWELL (0.9 confidence)
```
v_weight=-40, d_weight=-20, u_weight=+25, g_weight=-20, w_weight=-10
```

Layer 4:
```
V stays at 128 through word loop (no emotional forces)
Structure adjustment: V += -40 * 0.9 * 0.5 = -18
Final: V=110, FAREWELL detected
```

No negative words in the sentence. The structure told the story.

## Empirical Findings (78,272 sentences)

| Transition | Samples | Avg V delta |
|-----------|---------|-------------|
| CONNECTOR then EMOTIONAL | 2,836 | -15.3 |
| AMPLIFIER then EMOTIONAL | 8,184 | -10.4 |
| NEGATOR then EMOTIONAL | 7,267 | +5.8 |
| EMOTIONAL then EMOTIONAL | 80 | +14.1 |
| SELF_REF then EMOTIONAL | 8,269 | -1.5 |

Vocabulary: negative-to-positive ratio approximately 1.08:1 among primary signal words.

## Accuracy

- 100% on 630 novel sentences
- 97.3% crisis recall
- 100% on genuine positive (zero false positives on safe text)
- 90% on sarcasm
- 90% on internet speak
- 80% on body language descriptions
- 90% on conversation fight patterns
- 69.6% on SST-2 academic sentiment (movie review classification -- a different task)
- 75.3% on GoEmotions
- 76.3% on 4-AI consensus benchmark (vs Gemini, Claude Opus, GPT-4, Grok on 131 sentences)

## Limitations

- 26 structure patterns are engineered, not emergent
- Sarcasm detection is the weakest category
- Academic sentiment benchmarks (SST-2) test a different task
- Vocabulary covers conversational English, not literary/academic registers
- Solver mood persistence (0.6) is an adjustable default, not a derived constant

## Files

```
engine/
  pendulum.py          Physics layer
  word_classifier.py   Role classification
  proximity.py         Influence fields
  structures.py        Pattern detection
  solver.py            Bidirectional solver
  forces_curated.py    4,108 word force tuples (7D VADUGWI)
  force_flow.py        Force flow resolver (WHO does WHAT to WHOM)
  zones_impl.py        Zone implementation
  shared.py            VADUGWI dataclass (7-byte: V, A, D, U, G, W, I)
  battleship.py        Probe system
  vocabulary.py        Vocabulary export
  zones.py             9 convergence zones
  personality.py       Personality filter
  fuzzy.py             Typo/slang matching
```
