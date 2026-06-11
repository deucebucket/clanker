# How VADUGWI is Calculated - V8.4 Formula Reference

All constants and worked values below verified against the shipped engine
(`engine/pendulum.py`, `engine/proximity.py`, `engine/force_flow.py`) on
2026-06-11.

## The Sentence Equation

```
VADUGWI = Physics( Structures( Proximity( Roles(words) ) ) )
```

Four layers, bottom-up:

1. **Roles**: Each word gets a structural role (27 types)
2. **Proximity**: Distance-based influence fields between words (decay 0.90x/word)
3. **Structures**: Pattern recognition on role sequences (66 patterns)
4. **Physics**: Momentum-based force application, produces final VADUGWI

## Layer 1: Word Role Classification

Every word falls into one of four tiers:

```
primary signal words:   die, kill, love, hate, suicide, hope, help, life, death
                (~50 words, always heavy, always fire alerts)

secondary signal words:  happy, sad, angry, scared, grateful, lonely...
                (~200 words, have mass, in VOCABULARY with |dV| > 15)

OPERATORS:      I, you, very, not, but, if, and, or, of, because, so
                (~50 words, shape the field, no mass of their own)

unclassified words:    carpenter, tuesday, meeting, building...
                (everything else, no emotional mass, inherits from
                proximity to nearby stars)
```

### The 27 Structural Roles

```
SELF_REF:        I, me, my, myself          (speaker process)
OTHER_REF:       you, they, he, she          (other entity)
RELATION_REF:    mom, family, friend, boo    (relationship noun)
TRANSFER:        give, gave, leave, send     (moving FROM self)
ACQUIRE:         buy, get, find, take        (moving TO self)
EMOTIONAL:       any VOCABULARY word |dV|>15 (star with mass)
AMPLIFIER:       very, really, so, fucking   (scales nearby words)
NEGATOR:         not, never, no, don't       (flips/decays)
TEMPORAL:        tonight, tomorrow, still    (time frame)
HEDGE:           maybe, possibly, perhaps    (uncertainty dampener)
CONNECTOR:       and(+), or(><), because(<-) (routing operators)
CHOPPER:         but, however, yet           (kills before, promotes after)
COMPRESSOR:      just, only, merely          (shrinks what follows)
POSSESSION:      things, keys, car           (owned object)
METHOD:          pills, gun, rope, bridge    (means/tool)
FINALITY:        last, final, goodbye, end   (closing marker)
PEACE:           peace, calm, ready, fine    (resolution state)
FILLER:          um, like, basically         (processing noise)
NEUTRAL:         the, a, is, was             (structural glue)
INVERSION:       structural meaning flip
SURPRISE:        pattern interrupt (A-spike, not V-direction)
POWER:           use, control, command       (power verb family)
SUBMISSION:      obey, surrender             (power inversion)
PULL_TOWARD / PULL_AWAY / PULL_RESOLVED:  chase/flee/escape verb family
REGISTER_CASUAL: lol, lmao, bro              (SOLVENT -- dissolves LIQUID atoms)
```

## Layer 2: Proximity Weighting

Each word creates an influence field. Nearby words modify each other.

```
influence = PROXIMITY_DECAY ^ distance      PROXIMITY_DECAY = 0.90

distance=1: 0.90 influence     (adjacent, strong)
distance=2: 0.81
distance=3: 0.73
distance=4: 0.66
distance=5: 0.59               (wide field -- champion v2 widened the decay)
```

### Proximity Coefficient

For each EMOTIONAL word, nearby modifiers change the coefficient
(`ROLE_MODIFIERS` in `engine/proximity.py`, champion v2 values):

```python
coeff = 1.0
for each nearby word:
    if AMPLIFIER:   coeff *= (1.0 + 0.965 * influence)   # boost
    if NEGATOR:     coeff *= (1.0 - 2.464 * influence)   # flip (can go negative)
    if SELF_REF:    coeff *= (1.0 + 0.466 * influence)   # personalize
    if HEDGE:       coeff *= (1.0 - 0.416 * influence)   # dampen
    if COMPRESSOR:  coeff *= (1.0 - 0.301 * influence)   # compress
```

Cap: [-2.63, +2.63] (`COEFFICIENT_CAP`)

### Example

"I am very sad" -> roles: [SELF_REF, NEUTRAL, AMPLIFIER, EMOTIONAL]

For "sad" at position 3:
- "very" at distance 1: AMPLIFIER, influence = 0.90, coeff *= 1.87
- "I" at distance 3: SELF_REF, influence = 0.90^3 = 0.73, coeff *= 1.34
- Combined: 1.0 x 1.87 x 1.34 = 2.50 (engine trace: coeff=2.503)

## Layer 3: Structure Detection (66 Patterns)

Role sequences form patterns, like chess openings. A sample:

```
FAREWELL:           TRANSFER + POSSESSION + RELATION_REF
DIVESTITURE:        giving possessions away ("ive been giving my stuff away")
METHOD_ACQUISITION: ACQUIRE + METHOD ("just bought a bunch of pills")
SELF_REMOVAL:       COMPARISON + CONDITIONAL + SELF_REF
EXHAUSTION:         SELF_REF + NEGATOR + SUSTAIN_VERB + TEMPORAL
NO_EXIT:            NEGATOR + EXIT_CONCEPT ("there is no hope")
SARCASM_INVERSION:  POSITIVE_EMOTIONAL + NEGATIVE_CONTEXT ("oh great another monday")
SUSPICIOUS_CALM:    PEACE + "finally" ("I finally feel at peace")
BLANKET_APOLOGY:    APOLOGY + BLANKET_WORD ("im sorry for everything")
SELF_NULLIFY:       SELF_REF + NULL_WORD ("I am nothing")
CHOPPER_SPLIT:      CHOPPER present ("I love you but I'm leaving")
MASKING:            self-status claim + dismissive imperative ("im fine dont worry about it")
PERSISTENT_ABSENCE: ghost possession -- possessive of absent person + retention verb
TEMPORAL_GRIEVANCE: "finally/for once" on ANOTHER's action in acknowledgment frame
EXCLUSION_CONTRAST: "glad one of us is having fun", "must be nice to..."
IRONIC_DEFERENCE:   "no no youre clearly the expert" (surrender as assertion)
FAINT_PRAISE:       "interesting choice but ok" (requires dismissive cue)
RETROSPECTIVE_HOPE: hope-verb + closed outcome + transactional frame ("hope it was worth it")
```

See the `detectors` list in `engine/structures.py:detect_all` for the
complete set of 66. Each match carries a confidence and a weight vector
(v, a, d, u, g, w).

## Layer 4: Physics (Pendulum)

### Constants (champion v2, genetically optimized 2026-04-03)

```
CENTER = 128.0              (neutral point for V, A, D, G, W, I)
M_BASE = 0.557              (base momentum: how much previous state persists)
M_AROUSAL_SCALE = 0.25      (high A = stickier state)
M_NEGATIVITY_BIAS = 1.15    (negative states are stickier than positive)
M_POSITIVITY_EASE = 0.90    (positive transitions are easier)
M_MIN = 0.30, M_MAX = 0.95  (momentum floor/ceiling)
FORCE_SCALE = 1.405         (how hard forces push)
DIRECT_PUSH_CAP = 1.0       (direct push maximum)
DIRECT_PUSH_TRIGGER = 86.2  (force threshold for direct push)
SATURATION = 120.0          (tanh saturation replaces hard clamp)
```

Momentum is adaptive, not fixed: `m_eff = M_BASE + (A - 128)/255 * 0.25`,
scaled by the negativity bias when moving toward negative V, clamped to
[0.30, 0.95].

### Per-Word Force Application (EMOTIONAL words only)

```python
target_V = 128 + dV * coeff * FORCE_SCALE

# Direct push for strong forces
total_force = sum of |d* x coeff| across dimensions
push = min(1.0, total_force / 86.2) * 1.0 * sign * |dV| * FORCE_SCALE

# Adaptive momentum blend
V = V_prev * m_eff + target_V * (1 - m_eff) + push
```

### Structure Adjustments (after word loop)

```python
for each detected structure:
    V += structure.v_weight * confidence * FORCE_SCALE
    A += structure.a_weight * confidence * FORCE_SCALE
    D += structure.d_weight * confidence * FORCE_SCALE
    U += structure.u_weight * confidence * FORCE_SCALE
    G += structure.g_weight * confidence * FORCE_SCALE
    W += structure.w_weight * confidence * FORCE_SCALE
```

(Special-cased patterns -- RECOVERY_MILESTONE, SARCASM_INVERSION,
AMBIGUITY_HOLD -- have their own application rules in `apply_structures`.)

### W-V Coupling (Stage 6)

Self-worth modulates valence asymmetrically: low W amplifies negative V
(`1 + 0.5 * e^(-w_norm)`, capped 1.8x) and suppresses positive V
(`1 - 0.3 * (1 - e^(w_norm))`), where `w_norm = (W - 128)/128`.

### Saturation

All dimensions pass through `128 + 120 * tanh((x - 128)/120)`, then clamp
to 0-255. Extreme readings compress smoothly instead of hitting a wall.

## W -- Attribution-Routed Self-Worth

W answers "how much of this force's valence is about MY worth?" Valence is
routed into W through an attribution coefficient R derived from the
force-flow arc (WHO does WHAT to WHOM), replacing the old binary
SELF_REF-within-4-words gate:

```
dW = dV * coeff * FORCE_SCALE * 0.7 * R
```

| Flow | Example | R |
|------|---------|---|
| Self-declarative (self acts on self, negative) | "i am worthless" | 1.5 |
| Self-declarative (positive) | "i am proud of myself" | 0.7 |
| Force aimed at self (negative) | "he told me im nothing" | 1.2 |
| Force aimed at self (positive) | being loved/supported | 0.9 |
| Guilt (self harms other) | "i hurt her" | 0.43 (effective ~0.30 after damp) |
| Atmospheric, no self token | "the meeting is at three" | 0 -- W untouched |

Measured: "i am worthless" W=34, "he told me im nothing" W=32,
"i hurt her" W=93, "i am proud of myself" W=156.

When the flow resolves no real entity tokens, W falls back to the legacy
SELF_REF-proximity gate. Self-directed crisis patterns (SUSPICIOUS_CALM,
EXHAUSTION, RESIGNATION, DIVESTITURE, NO_EXIT, METHOD_ACQUISITION...) carry
deepened w_weights. Measured effect: crisis-set W median 88 vs 128 on the
safe set.

## I -- Agency Axis

The directional intent logic (withdraw 0 / deflect 64 / neutral 128 /
connect 192 / control 255) answers WHERE the force aims. The agency axis
answers whether the speaker still owns their own motion:

- **Futility** phrases ("whats the point", "why bother", "i give up",
  "no point") sink I to 64 -- powerlessness. Negated futility ("didnt give
  up") is perseverance and does not fire.
- **Agency** phrases ("i want to <verb>", "im going to <verb>", "i will",
  "let me", "i need to") lift I to 168 -- but only from the neutral band
  (90-160). The marker must bind to an action, not a destination:
  "im going to fix this" is agency; "im going to the store" is travel.
- A strong directional read (self-destruction withdraw, attack control) is
  **never overridden** by either axis.
- Tier-1 passive-aggression patterns also lift neutral I to 168: the
  grievance is a directed jab, not neutral information.

## Worked Example: "I gave my dog to my neighbor"

### Layer 1: Roles
```
I        -> SELF_REF
gave     -> TRANSFER
my       -> SELF_REF
dog      -> RELATION_REF   (animals are relations, not objects)
to       -> CONNECTOR
my       -> SELF_REF
neighbor -> RELATION_REF
```

### Layer 3: Structures
```
DIVESTITURE detected, confidence=0.65
  v_weight=-35, d_weight=-20, u_weight=+35, g_weight=+40, w_weight=-38
```

### Layer 4: Physics
```
Structure adjustment: V += -35 * 0.65 * 1.405 = -32
Final (engine output): V=83 A=133 D=113 U=31 G=177 W=101 I=128
```

Result: V=83 (negative), U elevated, W depressed -- giving away a loved
being reads as divestiture, a crisis-adjacent act. The engine read the
STRUCTURE, not the words. It doesn't know "dog" or "neighbor." It knows
TRANSFER + possession-of-relation = giving away what you love.

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
