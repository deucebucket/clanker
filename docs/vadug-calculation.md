# How VADUG is Calculated — Complete Formula Reference

## The Sentence Equation

```
V_final = C(w₀) + Σᵢ [ F(wᵢ) × R(i) × G(i) × W × cap(2.5) × rec(i/n) × mom(i) ]
```

Where:
- `C(w₀)` = first-word calibration (45 starter words prime the initial state)
- `F(wᵢ)` = word force lookup (46,101 entries, each a 5-tuple: dv, da, dd, du, dg)
- `R(i)` = ramp multiplier (if intensity word active: 0.5x to 2.0x, decays per word)
- `G(i)` = relationship gravity (55 words: daughter=1.7x, friend=1.3x, stranger=0.9x, decays 0.3x/word)
- `W` = personality willingness (empath=1.3x, stoic=0.8x, default=1.0x)
- `cap(2.5)` = maximum combined multiplier (prevents clipping)
- `rec(i/n)` = recency weight (0.9 + 0.4 × position/length — later words weigh more)
- `mom(i)` = exponential decay momentum (strong words: 0.99 carry; weak words: exponential recovery toward center)

## Step-by-Step Processing

### Step 0: Initialize
```
V = 128, A = 128, D = 128, U = 0, G = 128  (neutral center)
momentum = 0.99
drift_rate = 0.01
```

### Step 1: First-Word Calibration
45 starter words prime the pendulum before processing begins:
```
"I"      → V=120 (self-referential, expect emotion)
"nobody" → V=108, G=108 (negation opener)
"help"   → V=110, U=30 (urgent)
"hey"    → V=138 (greeting, warm)
"why"    → V=122, A=135, U=15 (questioning)
```

### Step 2: Bookend Parsing
First + last significant word determine relational geometry:
```
"Nobody ... me"    → isolation pattern → V -= 15, G -= 10
"I ... myself"     → self_loop pattern → V -= 20, G -= 15
"You ... me"       → accusation pattern → V -= 10, A += 15
"Help ... please"  → plea pattern → U += 20
```

### Step 3: For Each Word

#### 3a. Check Idiom (multi-word expressions)
99 idioms, 5 or 6-element tuples:
```
("want", "to", "die") → (-95, +35, -60, +70, -65, "crisis")
("over", "the", "moon") → (+50, +40, +20, 0, +55, "ecstatic")
("weight", "of", "the", "world") → (-60, +30, -40, +30, -70, "crushing gravity")
```
If idiom matches, SKIP individual word processing. Apply idiom force directly.

#### 3b. Check Ramp (intensity amplifier)
24 ramp words:
```
"extremely" → multiplier 1.6, length 3, decay 0.6
"very"      → multiplier 1.3, length 2, decay 0.6
"slightly"  → multiplier 0.7, length 1 (dampener)
"barely"    → multiplier 0.5, length 1 (heavy dampener)
```
Ramps CHAIN: "really really" → 1.35 × 1.35 = 1.82x compound.
Ramp decays: word 1 = full, word 2 = ×0.6, word 3 = ×0.36.

#### 3c. Check Bigram (word pair override)
53 bigram pairs:
```
("give", "up")     → (-35, -10, -35, 5, -30, "surrender")
("fall", "love")   → (+40, +30, -10, 15, +30, "romance")
("not", "good")    → (-15, +5, -5, +5, -5, "negation")
```
If bigram found within 3 positions, OVERRIDE individual word force.

#### 3d. Look Up Word Force
46,101 entries: `WORD_FORCES["angry"] = (-89, 53, 44, 79, 54)`
- dv = valence delta (negative = more negative)
- da = arousal delta
- dd = dominance delta
- du = urgency delta
- dg = gravity delta

#### 3e. Word Role Resolution
66 descriptor words (moist, dark, heavy, etc.) inherit emotion from nearby subject:
```
"moist" + "cake" → dv becomes positive (cake is positive)
"moist" + "wound" → dv stays negative (wound is negative)
Only for mild descriptors (|dv| <= 30). Strong descriptors keep own force.
```

#### 3f. Context Modifier (chameleon words)
31 words that change meaning based on pendulum trajectory:
```
"love" when V < 100 → sarcastic: (-20, +15, -5, +5, -5)
"love" when V > 155 → genuine: (+40, +25, +15, 0, +25)
"lol" when V < 60   → shield: (-5, -5, -5, 0, -5) [amplifies pain]
"lol" when V > 170  → genuine: (+5, +5, 0, 0, +5)
"ok" when V < 80    → defeated: (-10, -10, -10, 0, -10)
```
Context modifiers REPLACE the base word force. Only activate when V < 100 or V > 155.

#### 3g. Negation
"not", "never", "no" → flip dv and dg of NEXT word.

#### 3h. Compute Force Scale
```
force_scale = intensity × recency_weight × relationship_gravity × willingness
force_scale = min(force_scale, 2.5)  # cap prevents clipping
```

#### 3i. Apply Ramp (if active)
```
force_scale *= ramp_multiplier
```

#### 3j. Exponential Decay Momentum
```
if word_strength > 10:  # STRONG word
    effective_momentum = 0.99  (spike — carries through)
    track spike for decay
else:  # WEAK word
    decay_factor = e^(-0.15 × words_since_spike)
    effective_momentum = 0.5 + 0.49 × decay_factor
    (ranges from 0.99 just after spike to 0.5 far from spike)
```

#### 3k. Blend New State
```
target_v = 128 + vf × force_scale
blend = 1 - effective_momentum
direct_push = min(1.0, word_strength / 60) × 0.6

V = V × effective_momentum + target_v × blend + vf × direct_push × force_scale
A = A × effective_momentum + target_a × blend + af × direct_push × force_scale
(same for D, U, G)

Clamp all to [0, 255]
```

### Step 4: Post-Processing

#### 4a. Ending Weight (for multi-sentence arcs)
If arc_span >= 15 between opening and ending 30% of words:
```
V = V × 0.7 + ending_v × 0.3
```

#### 4b. Tonal Analysis
7 signal detectors on the trajectory:
- Valence whiplash (positive spike then negative drop)
- Peak and fade
- Intensity mismatch
- Deadpan, hyperbolic, understated
If sarcastic detected on non-emotional intent: `V = 128 - (V - 128) × 0.6`

#### 4c. Entropy Neutral Detection
```
entropy = Shannon entropy of V trajectory bins
variance = trajectory variance
spikes = count of |V_delta| > 10

If entropy > 0.8 AND variance < 30 AND |V - 128| < 10 AND spikes < 4:
    → Override to NEUTRAL
```

### Step 5: Output Mode
```
three_color:      blended = V × 0.9 + G × 0.1; thresholds [124, 128]
dimensional:      raw V, A, D, U, G
emotion_label:    weighted Euclidean distance to 31 emotion prototypes
crisis_check:     V < 50 AND G < 50
binary_sentiment: blended >= 126 → positive, else negative
```

## Key Parameters (all tuned)

| Parameter | Value | Tuned By |
|-----------|-------|----------|
| Momentum (strong words) | 0.99 | Bracket tournament (2,187 configs) |
| Decay lambda | 0.15 | Genetic algorithm |
| Spike threshold | 10 | Genetic algorithm |
| Drift rate | 0.01 | Cross-dataset sweep |
| Force cap | 2.5x | Layer audit |
| Recency range | [0.9, 1.3] | Bracket tournament |
| Entropy threshold | 0.8 | Parameter sweep (3,372 sentences) |
| Entropy V margin | 10 | Parameter sweep |
| Entropy spike max | 4 | Parameter sweep |
| Context mod gate | V<100 or V>155 | Ablation testing |
| 3-color thresholds | [124, 128] V+G blend | Universal sweep (7,720 sentences) |
| Force scaling V | 1.15x | Genetic algorithm (EmoBank 10K) |
| Force scaling A | 2.25x | Genetic algorithm |
| Force scaling D | 2.49x | Genetic algorithm |
| Force scaling U | 1.23x | Genetic algorithm |
| Force scaling G | 1.36x | Genetic algorithm |
