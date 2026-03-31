# How VADUGWI is Calculated - V5.5 Formula Reference

## The V5.5 Sentence Equation

```
VADUGWI = Physics( Structures( Proximity( Roles(words) ) ) )
```

Four layers, bottom-up:

1. **Roles**: Each word gets a structural role (18 types)
2. **Proximity**: Distance-based influence fields between words (decay 0.7x/word)
3. **Structures**: Pattern recognition on role sequences (26 patterns)
4. **Physics**: Momentum-based force application, produces final VADUGWI

## Layer 1: Word Role Classification

Every word falls into one of four tiers:

```
primary signal words:   die, kill, love, hate, suicide, hope, help, life, death
                (~50 words, always heavy, always fire alerts)

secondary signal words:  happy, sad, angry, scared, grateful, lonely...
                (~200 words, have mass, in VOCABULARY with |dV| > 15)

OPERATORS:      I(1.8x), you(0.7x), very(1.4x), not(flip), but(chop),
                if(?), and(+), or(><), of(/), because(<-), so(->)
                (~50 words, shape the field, no mass of their own)

unclassified words:    carpenter, tuesday, meeting, building...
                (everything else, no emotional mass, inherits from
                proximity to nearby stars)
```

### The 18 Structural Roles
```
SELF_REF:       I, me, my, myself          (speaker process)
OTHER_REF:      you, they, he, she          (other entity)
RELATION_REF:   mom, family, friend, boo    (relationship noun)
TRANSFER:       give, gave, leave, send     (moving FROM self)
ACQUIRE:        buy, get, find, take        (moving TO self)
EMOTIONAL:      any VOCABULARY word |dV|>15 (star with mass)
AMPLIFIER:      very, really, so, fucking   (scales next word 1.3-1.6x)
NEGATOR:        not, never, no, don't       (flips/decays)
TEMPORAL:       tonight, tomorrow, still    (time frame)
HEDGE:          maybe, possibly, perhaps    (uncertainty dampener)
CONNECTOR:      and(+), or(><), because(<-) (routing operators)
CHOPPER:        but, however, yet           (kills before, promotes after)
POSSESSION:     things, dog, keys, car      (owned object)
METHOD:         pills, gun, rope, bridge    (means/tool)
FINALITY:       last, final, goodbye, end   (closing marker)
PEACE:          peace, calm, ready, fine    (resolution state)
FILLER:         um, like, just, basically   (processing noise)
NEUTRAL:        the, a, is, was             (structural glue)
```

## Layer 2: Proximity Weighting

Each word creates an influence field. Nearby words modify each other.

```
influence = 0.7 ^ distance     (exponential decay)

distance=1: 0.70 influence     (adjacent, strong)
distance=2: 0.49
distance=3: 0.34
distance=4: 0.24
distance=5: 0.17              (weak, practical cutoff)
```

### Proximity Coefficient
For each EMOTIONAL word, nearby modifiers change the coefficient:

```python
coeff = 1.0
for each nearby word:
    if AMPLIFIER:  coeff *= (1.0 + 0.4 * influence)   # boost
    if NEGATOR:    coeff *= (1.0 - 1.6 * influence)   # flip (can go negative)
    if SELF_REF:   coeff *= (1.0 + 0.3 * influence)   # personalize
    if HEDGE:      coeff *= (1.0 - 0.3 * influence)   # dampen
```

Cap: [-3.0, 3.0]

### Example
"I am very sad" -> roles: [SELF_REF, NEUTRAL, AMPLIFIER, EMOTIONAL]

For "sad" at position 3:
- "I" at distance 3: SELF_REF, influence = 0.7^3 = 0.34, coeff *= 1.10
- "very" at distance 1: AMPLIFIER, influence = 0.70, coeff *= 1.28
- Combined: 1.0 x 1.10 x 1.28 = 1.41

## Layer 3: Structure Detection (26 Patterns)

Role sequences form patterns, like chess openings:

```
FAREWELL:           TRANSFER + POSSESSION + RELATION_REF
                    "I gave my dog to my neighbor"

METHOD_ACQUISITION: ACQUIRE + METHOD
                    "just bought a bunch of pills"

SELF_REMOVAL:       COMPARISON + CONDITIONAL + SELF_REF
                    "they'd be happier if I wasn't here"

EXHAUSTION:         SELF_REF + NEGATOR + SUSTAIN_VERB + TEMPORAL
                    "I can't take this anymore"

NO_EXIT:            NEGATOR + EXIT_CONCEPT
                    "there is no hope"

SARCASM_INVERSION:  POSITIVE_EMOTIONAL + NEGATIVE_CONTEXT
                    "oh great another monday" (output != intent)

SUSPICIOUS_CALM:    PEACE + "finally"
                    "I finally feel at peace" (decision made)

BLANKET_APOLOGY:    APOLOGY + BLANKET_WORD
                    "im sorry for everything"

FINALITY:           FINALITY_MARKER + (TEMPORAL | SELF_REF)
                    "this is the last time"

SELF_NULLIFY:       SELF_REF + NULL_WORD
                    "I am nothing"

CHOPPER_SPLIT:      CHOPPER present
                    "I love you but I'm leaving" (resets at chopper)

RELIEF_ABSENCE:     NEGATOR + NEGATIVE_EMOTIONAL + TEMPORAL
                    "I haven't had a panic attack in weeks"

SELF_EXCLUDED:      SELF_REF + EXCLUDED_MARKER
                    "everyone was invited except me"

WITHHELD_POSITIVE:  NEGATOR + TRANSFER + POSITIVE_EMOTIONAL
                    "they never told me they were proud"
```

(Plus 12 additional patterns -- see `engine/structures.py` for the complete set of 26.)

## Layer 4: Physics (Pendulum)

### Constants
```
CENTER = 128.0      (neutral point for V, A, D, G, W, I)
MOMENTUM = 0.82     (how much previous state persists)
FORCE_SCALE = 0.5   (how hard forces push)
PUSH_CAP = 0.4      (direct push maximum)
PUSH_TRIGGER = 80.0 (force threshold for direct push)
```

### Per-Word Force Application (EMOTIONAL words only)

```python
fs = FORCE_SCALE * |proximity_coefficient|
sign = +1 if coefficient >= 0 else -1

target_V = 128 + dV * fs * sign

# Direct push for strong forces
total_force = |dV * coeff| + |dA * coeff|
push = min(1.0, total_force / 80.0) * 0.4

# Momentum blend
V = V_prev * 0.82 + target_V * 0.18 + dV * fs * sign * push
```

### Structure Adjustments (after word loop)
```python
for each detected structure:
    V += structure.v_weight * confidence * FORCE_SCALE
    D += structure.d_weight * confidence * FORCE_SCALE
    U = max(U, structure.u_weight * confidence)
    G += structure.g_weight * confidence * FORCE_SCALE
    W += structure.w_weight * confidence * FORCE_SCALE
```

### Clamp
All values clamped to 0-255.

## Worked Example: "I gave my dog to my neighbor"

### Layer 1: Roles
```
I        -> SELF_REF
gave     -> TRANSFER
my       -> SELF_REF
dog      -> POSSESSION
to       -> NEUTRAL
my       -> SELF_REF
neighbor -> RELATION_REF
```

### Layer 2: Proximity
No EMOTIONAL words. All roles are structural. No forces to apply.

### Layer 3: Structures
```
find_role_pairs(TRANSFER, POSSESSION) -> (1, 3, 0.49)  found
find_role_pairs(TRANSFER, RELATION_REF) -> (1, 6, 0.12) found
-> FAREWELL detected, confidence=0.9
   v_weight=-40, d_weight=-20, u_weight=+25, g_weight=-20, w_weight=-10
```

### Layer 4: Physics
```
No EMOTIONAL words -> V stays at 128 through word loop
Structure adjustment: V += -40 * 0.9 * 0.5 = -18
Final: V = 110
```

Result: V=110 (negative), U elevated, FAREWELL pattern detected.
The engine read the STRUCTURE, not the words. It doesn't know "dog" or
"neighbor." It knows TRANSFER + POSSESSION + RELATION_REF = farewell.

## A+B=C Bidirectional Solver

### Forward: text -> VADUGWI
Run the 4-layer pipeline.

### Backward: A + desired_C -> B range
```python
for b_v in range(256):
    B = VADUGWI(v=b_v, a=128, d=128, u=0, g=128, w=128, i=128)
    C = A * 0.6 + B * 0.4
    if C.v in target_zone:
        valid_range.append(b_v)
```

Valid B is a RANGE, not a point. Like landing a plane, anywhere on the runway works.

## Connector Operators

```
and     = +    (additive, both sides stack)
but     = -    (chopper, kills before, promotes after)
or      = ><   (comparison, forces choice, creates tension)
of      = /    (attribution, routes source to state)
if      = ?    (conditional, opens hypothetical branch)
because = <-   (causal, this is WHY)
so      = ->   (consequential, this is WHAT HAPPENED)
also    = +=   (additive with emphasis)
```
