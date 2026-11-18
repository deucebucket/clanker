# How VADUGWI is Calculated - V8.6 Formula Reference

All constants and worked values below were checked against the shipped
engine (`engine/pendulum.py`, `engine/proximity.py`, `engine/force_flow.py`)
on 2026-11-18. What V8.6 changed, with the held-out numbers behind each
change, is in [V8.6: what changed and why](#v86-what-changed-and-why) at
the end.

## The Sentence Equation

```
VADUGWI = Physics( Structures( Proximity( Roles(words) ) ) )
```

Four layers, bottom-up:

1. **Roles**: Each word gets a structural role (27 types)
2. **Proximity**: Distance-based influence fields between words (decay 0.90x/word; negation is forward-only over 3 words)
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
    if SELF_REF:    coeff *= (1.0 + 0.466 * influence)   # personalize
    if HEDGE:       coeff *= (1.0 - 0.416 * influence)   # dampen
    if COMPRESSOR:  coeff *= (1.0 - 0.301 * influence)   # compress
```

Cap: [-2.63, +2.63] (`COEFFICIENT_CAP`)

### Negation (V8.6)

Negation is not part of the symmetric field. A negator acts forward, on
the next `NEGATION_WINDOW = 3` words, and fades by `NEGATION_DECAY = 0.5`
per extra word:

```python
gap = target_position - negator_position          # 1, 2 or 3; otherwise no effect
coeff *= 1.0 + (NEGATION_FLIP - 1.0) * NEGATION_DECAY ** (gap - 1)   # NEGATION_FLIP = -0.75

gap=1: x -0.75      ("not happy": flipped and damped)
gap=2: x +0.125     ("not very happy": nearly cancelled)
gap=3: x +0.5625
```

A negator after the word, or more than three words before it, does
nothing. Expletives and deception verbs still resist (strength x 0.15).
Until V8.5 negation was `coeff *= (1.0 - 2.464 * influence)` inside the
field above: symmetric, reaching 21 words, crossing zero about 7 words out.
`ROLE_MODIFIERS["NEGATOR"]` is still defined but no longer read.

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

### Structure reliability (V8.6)

Each structure's **valence** push is multiplied by
`STRUCTURE_RELIABILITY.get(pattern, 1.0)` (`engine/structure_reliability.py`).
The table is generated, not hand-edited: `python -m evaluation.vadugwi
fit-reliability` replays the open training pool with each pattern's push
switched off and counts how often the push had moved V toward the label
versus away from it. A pattern is set to 0.0 only if it changed V on at
least 20 training sentences and its toward-rate is below one half with
posterior probability above 0.95 (uniform prior). Currently gated:

| Pattern | toward | away | note |
|---|---|---|---|
| MUNDANE_HYPERBOLE | 87 | 224 | fired on 21% of the pool, +V push |
| AMBIGUITY_HOLD | 38 | 54 | borderline (posterior 0.952) |

Detection is unchanged: a gated pattern still appears in the trace and
still moves A, D, U, G and W, so pattern-based crisis checks see it.

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

### Per-Word Force Application

```python
target_V = 128 + dV * coeff * FORCE_SCALE

# Direct push for strong forces
total_force = sum of |d*| across dimensions
push = min(1.0, total_force / 86.2) * 1.0 * sign * |dV| * FORCE_SCALE

# Adaptive momentum blend -- only on an axis the word is charged on (V8.6)
if dV != 0:
    V = V_prev * m_eff + target_V * (1 - m_eff) + push
```

**Axis inertness (V8.6).** The same `if d != 0` guard applies to A, D, U
and G. A zero component exerts no force on its axis. Until V8.5 any word
with a charge on one axis pulled every other axis back toward its rest
value, so trailing vocabulary words with `dV == 0` drained the valence of
the words before them (W already had this guard).

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

The V line is scaled by the structure's reliability (see Layer 3).

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
Final (engine output): V=93 A=135 D=120 U=31 G=177 W=101 I=128
```

(V8.5 gave V=83 A=133 D=113; the difference is axis inertness.)

Result: V=93 (negative), U elevated, W depressed -- giving away a loved
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
    C = state_transition(A, B)      # 0.6/0.4 blend, then the directional pushes
    if C in target_zone:            # a zone is a box on V, D and G
        valid_range.append(b_v)
```

Valid B is a RANGE, not a point. Like landing a plane, anywhere on the runway works.

### The blend, and the beat rung (V8.6)

`solver.blend_axes(A, B, a_weight=A_WEIGHT)` is the one convex blend,
`C = 0.6 * A + 0.4 * B` on all seven axes. `state_transition` starts from
it for a turn and then adds its intent-conditioned pushes.

`engine.ladder.compose_beats(text)` applies the same blend one rung down.
It splits a longer text into beats (sentences, or clauses with
`clauses=True`), scores each beat with a fresh pendulum, and folds left to
right: `C = 0.6 * (so far) + 0.4 * (this beat)`. It is **opt-in**;
`compute_vadug` still runs one pendulum over the whole input. See the
results below for why.

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

## V8.6: what changed and why

### Version history

| Version | Date | What |
|---|---|---|
| V8.4 | 2026-04 to 2026-06 | champion-v2 constants (fitted 2026-04-03), 66 structures, W and I channels. The previous edition of this document. |
| V8.5 | 2026-06-11 to 2026-06-15 | slang register overhaul, vocabulary neutralizer, bravado fix, orthographic arousal. No constant in this document changed, so the V8.4 title stayed. This is what `origin/main` 92bd354 ships. |
| V8.6 | 2026-11-18 | structure reliability table, axis inertness, forward negation, shared blend and opt-in beat rung. No fitted constant was re-tuned. |

### How it was measured

`evaluation/vadugwi/` (see its README). Rules and constants were chosen on
the open training pool only (`council_v2_train`, 1,492 sentences, plus half
of EmoBank). Held-out sets were measured after the choices were fixed:

- **Sealed holdout v2**: 373 sentences, sealed 2026-06-11, 3 classes at the
  repo thresholds (neg < 110, pos >= 145). 62.2% of it is negative, so
  "always negative" scores 62.2%. Accuracy alone is a weak yardstick here;
  balanced accuracy and the threshold-free AUC (pos vs neg) are reported
  with it.
- **EmoBank test half**: 5,010 sentences with human valence ratings, other
  genres (news, letters, fiction). Pearson r.
- **SST-2 validation**: 872 movie-review sentences, pos/neg. AUC.

Intervals are 95% paired bootstrap over items (2,000 draws, fixed seed).

### What V8.5 measured as

| | in-repo stress (275) | holdout v2 accuracy | balanced acc. | AUC | EmoBank r | SST-2 AUC |
|---|---|---|---|---|---|---|
| V8.5 | 271 (98.5%) | 154/373 (41.3%) | 0.443 | 0.678 | 0.298 | 0.649 |

### What the diagnosis found (training pool)

- **The fitted physics constants do not carry held-out signal.** Sweeping
  `PROXIMITY_DECAY` from 0.0001 to 0.95, zeroing each role modifier,
  changing `SATURATION` or `COEFFICIENT_CAP`: accuracy moves by under half
  a point and AUC by under 0.01. Only overall gain (`FORCE_SCALE`, the push
  trigger) moves accuracy, by shifting sentences across the fixed
  thresholds.
- **The ranking power is the lexicon's.** A bare sum of word charges has
  AUC 0.690; the full engine has 0.705. Knowing only a sentence's category
  gives 0.71.
- **Structures mostly move the operating point.** Removing all of them
  takes in-repo stress from 98% to 62% and training-pool accuracy from
  0.49 to 0.40, but AUC only from 0.705 to 0.703, and it *raises* the
  correlation with human ratings on EmoBank (0.30 to 0.35).
- **Negation as fitted was worse than none** on the 185 training sentences
  that contain a negator: AUC 0.59 with it, 0.63 without, 0.78 with the
  forward rule that shipped (0.73-0.78 across the forward settings tried).

### Held-out result of each change

Each row is measured against the row above it.

| Change | holdout v2 accuracy | EmoBank test r | SST-2 AUC | in-repo stress |
|---|---|---|---|---|
| V8.5 baseline | 0.413 | 0.298 | 0.649 | 271/275 |
| + structure reliability | 0.429, +0.016 [0.000, +0.032] | 0.326, +0.028 [+0.013, +0.043] | 0.659, +0.010 [-0.012, +0.032] | 268 |
| + axis inertness | 0.453, +0.024 [+0.003, +0.048] | 0.352, +0.026 [+0.015, +0.036] | 0.673, +0.015 [+0.002, +0.028] | 267 |
| + forward negation | 0.461, +0.008 [-0.005, +0.021] | 0.364, +0.012 [+0.002, +0.023] | 0.691, +0.018 [+0.006, +0.031] | 266 |
| **V8.6 vs V8.5** | **172/373 (46.1%), +0.048 [+0.019, +0.078]** | **+0.066 [+0.045, +0.085]** | **+0.042 [+0.017, +0.069]** | **-5** |

Holdout v2 balanced accuracy went 0.443 to 0.472 (+0.029 [+0.001,
+0.056]); its AUC went 0.678 to 0.692 (+0.014 [-0.014, +0.042], not
separable from zero). V8.6 is still below the 62.2% majority-class line on
holdout v2. Holdout v1 (45 sentences) moved from 28 to 29 correct, which
is noise at that size.

### Costs

- In-repo stress 271 to 266 of 275. Ground truth stays 40/41.
- Crisis benchmark: recall unchanged at 49/51. Its binary rule now flags
  **3 of 75 safe sentences (was 0)**: three mundane-negative sentences
  that the old drain pulled back to neutral now read as plainly negative
  (V 58-93) with low W. This came in with axis inertness.
- Five engine tests that pinned a V band on a slang sense-split sentence
  are `xfail(strict)`. Each band was held by zero-charge words draining V
  toward centre. The fix they need is lexical (the literal sense of
  "cooked", "fire", "sheesh" should carry less charge).
- The worked values in this document moved by a few points, as noted.

### Tried and not shipped

| Candidate | Held-out result | Why not |
|---|---|---|
| `PROXIMITY_DECAY` 0.5 or 0.3 for the remaining operators | holdout v2 accuracy +0.003 / -0.003, EmoBank r -0.002 / -0.003 | No held-out effect either way; the decay of the non-negation operators is not identified by any data here. Left at 0.90. The locality question was real for negation, which now has its own short forward scope. |
| Higher gain (`FORCE_SCALE` 2.2) | holdout v2 accuracy +0.032 [+0.008, +0.056], balanced accuracy +0.009 [-0.027, +0.044], AUC -0.004 | It moves neutral predictions to negative on a pool that is 62% negative. That fits the pool's class mix, not the engine. |
| Refit word charges by ridge regression on the training pool (388 words) | holdout v2 accuracy +0.062 [+0.024, +0.102], AUC +0.045; EmoBank r **-0.023** [-0.034, -0.011]; SST-2 AUC **-0.022** [-0.038, -0.008]; in-repo stress -8.4 points | Helps the pool it was fitted on and hurts both other domains. The same refit on EmoBank did not transfer to the council pool either. |
| Beat rung as the default (`compose_beats`) | EmoBank test, multi-clause items (n=2,389): r 0.318 to 0.340, +0.022 [-0.004, +0.046]; multi-sentence (n=326): +0.049 [-0.030, +0.125]; holdout v2 multi-clause (n=41): accuracy -0.073 [-0.195, +0.049] | Not separable from zero. On the EmoBank fitting half the blend weight peaked at 0.5-0.7, which is where the turn rule's 0.6 already sits, so the code is shipped opt-in with that weight and not wired into `compute_vadug`. |
| Neutral-W coupling without the 1.5x negative amplification | training pool accuracy 0.520 to 0.499, crisis recall 49 to 46 | Costs crisis recall. |

### What this does not fix

- The lexicon is the ceiling. Agreement with human valence ratings is
  r = 0.36; for arousal and dominance it is about 0.14 and 0.09 (EmoBank
  fitting half).
- Everything here is English. The role lists, the lexicon and the forward
  direction of negation are English-specific.
- The rungs above the turn (session, relationship) are still not wired.

