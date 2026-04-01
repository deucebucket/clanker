# Linguistic Devices Taxonomy

A comprehensive catalog of every linguistic device that modifies emotional meaning in text.
Each device maps to a mathematical operation on the VADUGWI coordinate system.
This is the equation tree for the Clanker-Lang emotional engine.

---

## How to Read This Document

Each device entry specifies:

- **Operation type** -- the mathematical primitive it maps to
- **Affects** -- which VADUGWI dimensions are primarily modified
- **Examples** -- concrete instances with expected behavior
- **Engine notes** -- how this translates to pendulum physics

### Operation Types

| Symbol | Name | Description |
|--------|------|-------------|
| `x * M` | **Multiplier** | Scales the magnitude of forces. `x * 1.5` = amplify, `x * 0.5` = dampen |
| `x * -1` | **Flip** | Inverts the sign of a force (positive becomes negative, vice versa) |
| `x + C` | **Offset** | Adds a constant delta to one or more dimensions |
| `SET(x)` | **Set** | Overrides current value -- ignores what came before |
| `FRAME(x)` | **Frame** | Sets a baseline expectation; does not move the pendulum but changes how subsequent words are interpreted |
| `CHAIN(f)` | **Chain Modifier** | Modifies the NEXT operation rather than the current state |
| `GATE(p)` | **Gate** | Marks the attached content as conditional -- multiplies force by probability p (0.0 to 1.0) |
| `DECAY(x, t)` | **Ramp/Decay** | Applies force x that decays over t subsequent words |
| `REPLACE` | **Idiom Replace** | Entire multi-word unit replaces individual word forces with a single fixed vector |
| `CONTEXT(tag)` | **Context Tag** | Attaches metadata that downstream modules (sarcasm, grading) consume |

---

## 1. Pragmatic Devices / Discourse Markers

**Operation:** `FRAME` -- sets interpretive context for what follows

**Affects:** All dimensions indirectly (by changing how subsequent forces are weighted)

Discourse markers do not carry emotional content themselves. They signal how to INTERPRET the emotional content that follows. They are frame-setters.

| Marker | Effect | Notes |
|--------|--------|-------|
| "in general" | `FRAME(generalization)` -- reduces specificity, dampens U | What follows is abstract, not personal |
| "by the way" | `FRAME(tangent)` -- resets local momentum by ~50% | Signals topic shift; emotional carryover should decay |
| "of course" | `CHAIN(x * 1.2)` on next clause -- mild amplifier of certainty | Implies shared knowledge; raises D slightly |
| "as a matter of fact" | `CHAIN(x * 1.3)` + `FRAME(correction)` | Signals the speaker is correcting a misconception; raises D, A |
| "to be honest" | `FRAME(confession)` + `CHAIN(x * 1.15)` | What follows has elevated emotional authenticity |
| "at the end of the day" | `FRAME(summary)` -- weights the following clause as the takeaway | Elevates gravity of next statement |
| "having said that" | `FRAME(contrast)` -- signals incoming contradiction | Previous emotional direction may reverse |
| "the thing is" | `FRAME(revelation)` + offset A+10, U+10 | Builds tension for what follows |
| "on top of that" | `FRAME(escalation)` + `CHAIN(x * 1.2)` | Amplifies cumulative emotional load |
| "for what it's worth" | `FRAME(hedge)` + `CHAIN(x * 0.7)` | Dampens the authority of what follows; lowers D |

### Examples

1. "By the way, your mother called." -- `FRAME(tangent)` resets emotional momentum; "mother called" reads as neutral information
2. "To be honest, I'm struggling." -- `FRAME(confession)` amplifies the vulnerability in "struggling"
3. "Of course it went wrong." -- `FRAME(expected)` + amplifies resignation; "went wrong" reads as inevitable, not surprising
4. "As a matter of fact, I did the whole thing myself." -- `FRAME(correction)` + boosts D (dominance/pride)
5. "The thing is, nobody asked." -- `FRAME(revelation)` + builds to dismissal

---

## 2. Hedging Qualifiers

**Operation:** `CHAIN(x * 0.4 to 0.8)` -- dampens the next emotional force

**Affects:** All dimensions (reduces magnitude), especially D (lowers certainty/dominance)

Hedges reduce the speaker's commitment to the emotional content. They are multipliers less than 1.0, applied to the next content word or clause.

| Hedge | Multiplier | D Offset | Notes |
|-------|-----------|----------|-------|
| "maybe" | `x * 0.5` | D-15 | Strong uncertainty |
| "perhaps" | `x * 0.5` | D-10 | Formal uncertainty |
| "probably" | `x * 0.7` | D-5 | Leans toward true |
| "possibly" | `x * 0.4` | D-15 | Weak commitment |
| "I think" | `x * 0.7` | D-10 | Subjectivity marker |
| "I guess" | `x * 0.5` | D-20 | Reluctant, low confidence |
| "generally" | `x * 0.8` | D-5 | Weakens to trend |
| "sometimes" | `x * 0.6` | D-5 | Reduces frequency/certainty |
| "kind of" | `x * 0.6` | D-10 | Informal dampener |
| "sort of" | `x * 0.6` | D-10 | Informal dampener |
| "in a way" | `x * 0.5` | D-10 | Partial commitment |
| "to some extent" | `x * 0.6` | D-5 | Formal partial |
| "it seems like" | `x * 0.6` | D-15 | Distance from claim |

### Examples

1. "I'm maybe a little upset." -- `x * 0.5` on "upset" forces, then "a little" applies `x * 0.7` on top = `x * 0.35` total
2. "I think we should be worried." -- `x * 0.7` on "worried" forces; the worry is presented as opinion, not fact
3. "I guess that's fine." -- `x * 0.5` on "fine" PLUS D-20; reads as reluctant acceptance, not actual satisfaction
4. "It seems like things are getting better." -- `x * 0.6` on "better" -- improvement is tentative
5. "Probably nothing to worry about." -- `x * 0.7` on "nothing to worry" -- the reassurance itself is weakened

---

## 3. Intensifiers / Amplifiers

**Operation:** `DECAY(x * M, length)` -- ramp amplifier with word-length decay

**Affects:** Primarily V, A (amplifies existing emotional direction and energy)

These are already implemented in the engine as the Ramp system. Included here for completeness with extended coverage.

| Intensifier | Multiplier | Ramp Length | Decay | Notes |
|-------------|-----------|-------------|-------|-------|
| "very" | x * 1.3 | 2 | 0.6 | Baseline amplifier |
| "really" | x * 1.35 | 2 | 0.6 | Slightly stronger than "very" |
| "extremely" | x * 1.6 | 3 | 0.6 | Steep, long ramp |
| "absolutely" | x * 1.7 | 3 | 0.6 | Near-maximum amplifier |
| "so" | x * 1.25 | 2 | 0.6 | Casual amplifier |
| "incredibly" | x * 1.5 | 3 | 0.6 | Strong amplifier |
| "utterly" | x * 1.6 | 2 | 0.6 | Carries negative connotation |
| "completely" | x * 1.5 | 2 | 0.6 | Totality marker |
| "deeply" | x * 1.35 | 2 | 0.6 | Implies G shift (heavier) |
| "profoundly" | x * 1.5 | 2 | 0.6 | Formal; implies G shift |
| "ridiculously" | x * 1.5 | 2 | 0.6 | Informal; can signal sarcasm |
| "insanely" | x * 1.5 | 2 | 0.6 | Informal hyperbolic amplifier |
| "wildly" | x * 1.4 | 2 | 0.6 | Implies chaos (A boost) |
| "unbelievably" | x * 1.6 | 2 | 0.6 | Marks content as exceeding expectation |
| "downright" | x * 1.4 | 1 | -- | Short, punchy amplifier |
| "straight-up" | x * 1.4 | 1 | -- | Slang; also signals honesty |
| "hella" | x * 1.4 | 1 | -- | Regional slang amplifier |
| "mad" (as adverb) | x * 1.3 | 1 | -- | NYC slang: "mad tired" |

### Examples

1. "I'm extremely happy." -- `DECAY(x * 1.6, 3)` on "happy": V boosted significantly
2. "That was utterly devastating." -- `DECAY(x * 1.6, 2)` on "devastating": already strong negative gets amplified to near-floor V
3. "She's so tired." -- `DECAY(x * 1.25, 2)` on "tired": moderate amplification of fatigue
4. "Absolutely incredible work." -- `DECAY(x * 1.7, 3)` cascades through "incredible" and "work"
5. "I deeply regret what happened." -- `DECAY(x * 1.35, 2)` on "regret" + G offset (heavier/sinking)

---

## 4. Diminishers / Downtoners

**Operation:** `CHAIN(x * 0.3 to 0.8)` -- reduces magnitude of next emotional force

**Affects:** All dimensions (compresses toward neutral), especially A (lowers intensity)

Diminishers are the inverse of amplifiers. They pull emotional forces toward center. Some are already in the Ramp system (somewhat, slightly, barely). Extended list here.

| Diminisher | Multiplier | A Offset | Notes |
|-----------|-----------|----------|-------|
| "just" | x * 0.6 | A-10 | Minimizer: "just a scratch" |
| "only" | x * 0.5 | A-10 | Restricts scope |
| "merely" | x * 0.4 | A-15 | Strong formal diminisher |
| "barely" | x * 0.3 | A-15 | Near-zero acknowledgment |
| "slightly" | x * 0.5 | A-10 | Small degree |
| "somewhat" | x * 0.6 | -- | Mild reduction |
| "a little" | x * 0.5 | A-5 | Casual small-degree |
| "a bit" | x * 0.5 | A-5 | Casual small-degree |
| "a tad" | x * 0.4 | A-5 | Very mild; informal |
| "mildly" | x * 0.5 | A-10 | Moderate diminisher |
| "partly" | x * 0.5 | -- | Partial scope |
| "almost" | x * 0.8 | -- | Near-miss: close but not there |
| "nearly" | x * 0.8 | -- | Near-miss |
| "not quite" | x * 0.7 | D-5 | Shortfall marker |
| "kind of" | x * 0.6 | D-10 | Overlaps with hedging |
| "hardly" | x * 0.3 | A-15 | Near-negation diminisher |

### Special Case: "just"

"Just" is context-dependent:
- **Diminisher:** "It's just a scratch." -- `x * 0.6` on "scratch"
- **Temporal:** "I just got here." -- no emotional effect (temporal marker)
- **Emphatic:** "Just stop!" -- `x * 1.3` (amplifier when imperative)

### Examples

1. "I'm a little worried." -- `x * 0.5` on "worried": concern is minimized
2. "It only hurts a bit." -- `x * 0.5` on "hurts" then `x * 0.5` again from "a bit" = `x * 0.25` total
3. "She barely noticed." -- `x * 0.3` on whatever emotional content "noticed" carries
4. "I'm mildly annoyed." -- `x * 0.5` on "annoyed": irritation is held in check
5. "It was merely a suggestion." -- `x * 0.4` on "suggestion": strips emotional weight entirely

---

## 5. Sarcasm Markers

**Operation:** `FLIP(V)` + `CONTEXT(sarcasm)` -- inverts valence; flags for downstream detection

**Affects:** V (flipped), A (elevated -- sarcasm is emotionally hot), D (often elevated -- sarcasm implies superiority)

Sarcasm inverts the surface emotional meaning. The engine currently detects sarcasm via trajectory analysis (reversal, intensity mismatch, context contradiction). These lexical markers should BOOST sarcasm confidence when detected.

| Marker | Operation | Notes |
|--------|-----------|-------|
| "yeah right" | `FLIP(V)` + A+20 | Direct inversion marker |
| "sure" (standalone) | `FLIP(V)` + D+15 | Agreement that means disagreement |
| "oh great" | `FLIP(V)` + A+15 | Faux enthusiasm |
| "thanks a lot" | `FLIP(V)` + A+10 | Bitter gratitude |
| "how wonderful" | `FLIP(V)` + A+20 | Faux positive |
| "what a surprise" | `FLIP(V)` + A+10, D+10 | Pretend shock |
| "oh joy" | `FLIP(V)` + A+15 | Mock excitement |
| "no kidding" | `FLIP(V)` or literal | Context-dependent |
| "tell me about it" | depends on tone | Can be genuine agreement or sarcastic |
| "good for you" | `FLIP(V)` when dismissive | Can be genuine |
| "nice one" | `FLIP(V)` + D+10 | After a failure |
| "real smart" | `FLIP(V)` + D+20 | Condescension |
| "oh please" | `FLIP(V)` + D+15, A+15 | Dismissive disbelief |

### Detection Heuristics

Sarcasm markers alone are not sufficient. Confidence requires corroboration:
- Marker + negative context = HIGH confidence sarcasm
- Marker alone = LOW confidence (could be literal)
- Marker + positive context = likely literal

### Examples

1. "Oh great, another meeting." -- `FLIP(V)` on "great"; "meeting" is mildly negative context; HIGH confidence
2. "Yeah right, like that'll work." -- `FLIP(V)` explicit; "like that'll work" confirms disbelief
3. "Thanks a lot for nothing." -- "nothing" confirms the flip; gratitude is bitter
4. "Sure, because that always goes well." -- `FLIP(V)` on "well"; temporal "always" in sarcastic frame = "never"
5. "What a surprise that it broke again." -- `FLIP(V)` on "surprise"; "broke again" confirms negative frame

---

## 6. Double Negation / Negative Composites

**Operation:** `FLIP(V)` applied twice = net positive, OR `FLIP(V)` + offset

**Affects:** V (direction), D (often raised -- double negation implies deliberate construction)

Double negation produces a POSITIVE meaning but weaker than direct positive statement. "Not bad" is less positive than "good." This is a flip + dampening.

| Pattern | Operation | Resulting Valence | Notes |
|---------|-----------|-------------------|-------|
| "not" + negative word | `FLIP(V) * 0.6` | Mild positive | "not bad" = mildly good |
| "nobody" + negative | `FLIP(V) * 0.5` | Weak positive | "nobody complained" = OK |
| "can't stop" + positive | Positive + A+20 | Amplified positive | "can't stop laughing" = very amused |
| "can't stop" + negative | Negative + A+20, U+15 | Amplified negative | "can't stop crying" = distressed |
| "never" + negative | `FLIP(V) * 0.7` | Moderate positive | "never disappoints" = reliably good |
| "nothing" + negative | `FLIP(V) * 0.5` | Weak positive | "nothing wrong" = acceptable |
| "no" + negative noun | `FLIP(V) * 0.5` | Weak positive | "no problem" = fine |
| "without" + negative | `FLIP(V) * 0.6` | Mild positive | "without hesitation" = confident |
| "won't" + negative | `FLIP(V) * 0.6` | Mild positive | "won't fail" = will succeed |
| "hardly" + negative | `FLIP(V) * 0.3` | Very weak positive | "hardly the worst" = barely OK |

### Key Rule

Double negation resolves to positive but at REDUCED magnitude compared to the direct positive equivalent:
- "not bad" < "good" < "great"
- "never disappoints" < "always satisfies" < "always delights"

The reduction factor should be 0.5-0.7 of the equivalent direct positive.

### Examples

1. "Nobody had a bad time." -- `FLIP(V)` on "bad" via "nobody", dampened: reads as "it was OK" not "it was great"
2. "I can't stop smiling." -- no flip; "can't stop" is a continuation marker, amplifies "smiling" with A+20
3. "Not terrible, actually." -- `FLIP(V) * 0.6` on "terrible": mild positive + "actually" adds surprise frame
4. "There's nothing wrong with it." -- `FLIP(V) * 0.5` on "wrong": passable, not enthusiastic
5. "She never fails to impress." -- `FLIP(V) * 0.7` on "fails": consistent positive, D+10 (reliability)

---

## 7. Rhetorical Questions

**Operation:** `FRAME(rhetorical)` + `SET` or `OFFSET` depending on implied answer

**Affects:** V (set by implied answer), A+15 to A+30 (questions create engagement), D (varies)

Rhetorical questions do not seek information. They assert an emotional position disguised as a question. The implied answer is the real emotional content.

| Question | Implied Meaning | Operation |
|----------|----------------|-----------|
| "Who cares?" | Nobody cares / it doesn't matter | V-40, A+15, D+20 (dismissive) |
| "What's the point?" | There is no point | V-50, A-10, D-40 (helpless) |
| "Why bother?" | It's not worth doing | V-40, A-15, D-30 (defeated) |
| "How hard can it be?" | It should be easy | V+10, A+15, D+30 (confident/dismissive) |
| "Are you kidding me?" | This is unacceptable | V-30, A+40, D+20 (outraged) |
| "What did I do to deserve this?" | I don't deserve this (suffering) | V-40, A+25, D-30 (victimhood) |
| "Can you believe it?" | This is remarkable | V+20 or V-20, A+30 (depends on context) |
| "Who does that?" | Nobody should do that | V-30, A+20, D+15 (judgment) |
| "What's wrong with you?" | You are behaving badly | V-40, A+30, D+25 (accusation) |
| "Do I look like I care?" | I don't care | V-10, A+20, D+30 (aggressive indifference) |
| "How many times do I have to say this?" | I've said it enough | V-30, A+35, D+20, U+20 (exasperation) |

### Detection

Rhetorical questions are identified by:
1. Question mark at end of clause
2. Known rhetorical patterns ("who cares", "what's the point")
3. Question in context where answer is self-evident
4. Negative implied answer + high arousal

### Examples

1. "Who even cares about this anymore?" -- `SET(V=88, A=143, D=148)` -- dismissive, not actually asking
2. "What's the point of trying?" -- `SET(V=78, A=118, D=88)` -- defeated; "trying" gets negated by frame
3. "Are you serious right now?" -- `SET(V=98, A=168, D=148)` -- disbelief/outrage
4. "Why would anyone do that?" -- judgment frame; the "anyone" universalizes the condemnation
5. "Oh, so now you care?" -- sarcasm + rhetorical; double-device: `FLIP` on "care" inside rhetorical frame

---

## 8. Euphemisms

**Operation:** `REPLACE` with dampened equivalent -- same direction, reduced magnitude

**Affects:** V (same direction as literal, but closer to neutral), A (reduced), G (often shifted)

Euphemisms encode the same referent as their literal counterpart but with emotional padding. "Passed away" and "died" point at the same event, but the euphemism dampens the blow. They function like idioms: the multi-word unit replaces individual word scores.

| Euphemism | Literal Equivalent | Operation | Notes |
|-----------|-------------------|-----------|-------|
| "passed away" | died | `REPLACE(-55, +30, -35, +20)` | Already in IDIOMS |
| "let go" (from job) | fired | `REPLACE(-40, +20, -30, +15)` | Softened termination |
| "between jobs" | unemployed | `REPLACE(-25, +10, -20, +10)` | Self-protective framing |
| "downsizing" | mass layoffs | `REPLACE(-35, +25, -25, +20)` | Corporate distancing |
| "collateral damage" | civilian deaths | `REPLACE(-60, +10, +20, +15)` | Military sanitization; D raised (institutional) |
| "passed on" | died | `REPLACE(-50, +25, -30, +15)` | Already in IDIOMS |
| "with the Lord" | dead | `REPLACE(-30, -10, +15, 0)` | Religious consolation |
| "no longer with us" | dead | `REPLACE(-45, +15, -20, +10)` | Formal distancing |
| "put to sleep" | euthanized | `REPLACE(-50, -15, +10, +10)` | Veterinary/medical |
| "adult beverages" | alcohol | `REPLACE(+5, +5, 0, 0)` | Humor-coded |
| "enhanced interrogation" | torture | `REPLACE(-80, +30, +40, +20)` | Political sanitization |
| "friendly fire" | killed by own side | `REPLACE(-70, +30, -20, +30)` | Military euphemism |
| "correctional facility" | prison | `REPLACE(-30, +5, +10, 0)` | Institutional softening |
| "economically disadvantaged" | poor | `REPLACE(-25, +5, -15, +5)` | Formal softening |

### Key Principle

Euphemisms preserve DIRECTION but reduce MAGNITUDE:
- `euphemism_force = literal_force * 0.5 to 0.7`
- Exception: political/military euphemisms may also shift D upward (institutional authority masks suffering)

### Examples

1. "She passed away last night." -- `REPLACE` overrides "passed" and "away" individual scores; reads as grief but softened
2. "We had to let him go." -- `REPLACE` for employment context; "had to" adds reluctance frame
3. "I'm between opportunities right now." -- euphemism for unemployment; V is mildly negative, not crushing
4. "There was some collateral damage." -- sanitized language; the euphemism itself signals emotional distancing
5. "She's in a better place now." -- already in IDIOMS as consolation; V is mildly negative (acknowledges loss) but G+20 (uplift)

---

## 9. Hyperbole

**Operation:** `x * 0.3 to 0.5` (DAMPENING, not amplification) -- hyperbole overstates, so the actual emotional content is less than surface

**Affects:** A (surface-high but actual-moderate), V (same direction, reduced magnitude)

This is counterintuitive. Hyperbole LOOKS like amplification but actually signals that the speaker is exaggerating for effect. "I'm literally dying" does not mean death. The engine must recognize hyperbolic patterns and REDUCE toward the intended meaning.

| Hyperbole | Literal Meaning | Operation | Notes |
|-----------|----------------|-----------|-------|
| "I'm dying" (amusement) | Very amused | `SET(V+40, A+40)` -- not death | Context: laughter/humor |
| "I could eat a horse" | Very hungry | `SET(V-15, A+20, U+15)` | Not about horses |
| "a million times" | Many times | `CHAIN(x * 1.3)` | Mild amplifier, not literal |
| "the worst thing ever" | Very bad | `x * 0.6` on "worst" | "Ever" is hyperbolic scope |
| "I'm literally dying" | Strongly affected | `SET(V+30, A+50)` or `SET(V-40, A+50)` | "Literally" as hyperbolic marker |
| "I waited forever" | Waited a long time | offset U+20 | Not literal eternity |
| "everyone knows" | Many people know | `CHAIN(x * 1.1)` | Mild universal claim |
| "I've told you a thousand times" | Repeated request | A+30, U+25 | Exasperation |
| "it weighs a ton" | Very heavy | `x * 0.5` on literal weight | Physical metaphor |
| "I nearly died" (embarrassment) | Very embarrassed | `SET(V-30, A+35, D-30)` | Not literal near-death |
| "my head is going to explode" | Very stressed/frustrated | `SET(V-35, A+40, U+25)` | Stress metaphor |
| "I'm drowning in work" | Very busy/overwhelmed | `SET(V-30, A+30, D-40, U+30)` | Metaphorical drowning |

### Detection Heuristics

Hyperbole is detected by:
1. Death/destruction vocabulary in non-crisis context
2. Impossible quantities ("a million", "a thousand")
3. Physical impossibilities ("head exploding", "eating a horse")
4. "Literally" used non-literally (informal register marker)

### Key Principle

Hyperbole = the speaker has ALREADY amplified. The engine should NOT amplify again. Instead, map to the INTENDED emotional state, which is typically 30-50% of the surface literal meaning.

### Examples

1. "I'm literally going to die if I have to sit through another meeting." -- Not death; V-20, A+30, U+10
2. "This is the best day of my entire life!" -- Hyperbolic scope; `x * 0.6` on "best": very good day, not peak life event
3. "I waited a million years for this." -- "million years" = long wait; U+20, A+15
4. "Everyone hates Mondays." -- "everyone" is hyperbolic universal; `x * 0.8` dampening on the claim
5. "My heart literally stopped." -- Shock/surprise; V varies by context, A+40

---

## 10. Litotes / Understatement

**Operation:** `FLIP(V) * 0.5 to 0.7` -- negation of negative produces weak positive

**Affects:** V (positive but dampened), D+10 (understatement implies control/composure)

Litotes is the deliberate use of understatement via double negation. "Not bad" means "good but I'm not going to say so directly." It always produces a WEAKER version of the positive equivalent.

| Litotes | Positive Equivalent | V Result | Notes |
|---------|-------------------|----------|-------|
| "not bad" | good | V+20 (not V+40) | Classic litotes |
| "not terrible" | acceptable/OK | V+15 | Barely positive |
| "not the worst" | tolerable | V+10 | Faint praise |
| "not half bad" | quite good | V+30 | "Half" amplifies the litotes |
| "not unpleasant" | somewhat pleasant | V+15 | Formal/reserved |
| "not without merit" | has some value | V+15, D+10 | Academic praise |
| "no small feat" | impressive achievement | V+30, D+20 | Elevated litotes |
| "not exactly thrilled" | disappointed/unhappy | V-20 | Litotes for negative (inverted) |
| "not entirely convinced" | skeptical/doubtful | V-10, D-10 | Hedged doubt |
| "not the brightest" | stupid | V-25, D-15 | Litotes as insult |
| "doesn't suck" | decent | V+15 | Informal |
| "could be worse" | not great but OK | V+5 | Minimal positivity |
| "not too shabby" | decent/good | V+25 | Informal positive |

### Scaling Rule

```
litotes_V = positive_equivalent_V * 0.5 to 0.7
litotes_D = positive_equivalent_D + 10  (composure bonus)
```

### Examples

1. "The food was not bad." -- V+20 (good would be V+35); speaker is reserved
2. "It's not the worst idea I've heard." -- V+10; faint, almost grudging approval
3. "She's not exactly thrilled about it." -- INVERTED litotes: V-20; she's unhappy
4. "That was no small accomplishment." -- V+30, D+20; genuine praise via understatement
5. "Not too shabby for a first attempt." -- V+25; positive with a frame of low expectations

---

## 11. Idioms

**Operation:** `REPLACE` -- entire phrase maps to a single fixed emotional vector

**Affects:** All dimensions (idiom vector overrides component words)

Idioms are non-compositional. "Kick the bucket" has nothing to do with kicking or buckets. The engine must detect the multi-word unit and replace individual word forces with a single idiom vector. Already extensively implemented in `pendulum.py`.

| Idiom | VADUGWI Vector | Category |
|-------|-------------|----------|
| "kick the bucket" | V-60, A+30, D-40, U+20 | Death |
| "over the moon" | V+50, A+40, D+20, U0, G+55 | Joy |
| "break a leg" | V+25, A+20, D+15, U0 | Good luck |
| "piece of cake" | V+30, A-10, D+30, U-5 | Easy |
| "under the weather" | V-25, A-15, D-20, U+5 | Ill |
| "hit the nail on the head" | V+25, A+20, D+30, U0 | Accuracy |
| "spill the beans" | V-10, A+30, D-10, U+20 | Reveal secret |
| "bite the bullet" | V-15, A+25, D+20, U+15 | Endure hardship |
| "stab in the back" | V-55, A+45, D-25, U+35 | Betrayal |
| "walking on eggshells" | V-20, A+30, D-30, U+20 | Anxiety/caution |
| "throw in the towel" | V-35, A-10, D-40, U+10 | Give up |
| "add insult to injury" | V-40, A+35, D-20, U+20 | Escalation |
| "cry over spilled milk" | V-15, A+10, D-10, U+5 | Wasted regret |
| "burn bridges" | V-40, A+30, D+20, U+20 | Destructive choice |
| "bend over backwards" | V+10, A+25, D-20, U+15 | Extreme effort |

### Engine Integration

Idioms are detected via sliding window over previous_words + current_word. When matched, the idiom vector REPLACES all individual word forces for those tokens. The idiom label is attached as metadata for downstream grading.

### Examples

1. "He finally kicked the bucket." -- `REPLACE` overrides "kicked", "the", "bucket" with death vector
2. "She's over the moon about the news." -- `REPLACE` with joy vector including G+55 (soaring)
3. "I'm walking on eggshells around him." -- `REPLACE` with anxiety vector; D-30 is key (loss of agency)
4. "Don't cry over spilled milk." -- `REPLACE` + imperative "don't" frame = advice to stop regretting
5. "She really bent over backwards for us." -- `REPLACE` + "really" ramp = amplified effort vector

---

## 12. Compositional Semantics (Multi-word Operators)

**Operation:** Various -- these are multi-word phrases that function as single operators

**Affects:** Depends on specific phrase; acts as frame-setter, contrast marker, or scope modifier

These are phrases where meaning emerges from composition but is NOT idiomatic (the meaning IS derivable from parts, unlike idioms). They function as structural operators on the emotional content.

| Phrase | Operation | Function |
|--------|-----------|----------|
| "in general" | `FRAME(generalization)` + `CHAIN(x * 0.8)` | Weakens specificity |
| "of course" | `CHAIN(x * 1.2)` + D+10 | Certainty amplifier |
| "on the other hand" | `FRAME(contrast)` + momentum reset | Signals reversal of emotional direction |
| "as a result" | `FRAME(consequence)` + U+10 | Causal link; makes next content feel inevitable |
| "in spite of" | `FRAME(concession)` | Acknowledges X but asserts Y; Y gets weight |
| "regardless of" | `FRAME(override)` + D+15 | Dismisses previous; asserts control |
| "for the most part" | `CHAIN(x * 0.8)` | Generalizer with exceptions implied |
| "on balance" | `FRAME(summary)` | Signals net assessment coming |
| "at the same time" | `FRAME(parallel)` | Two emotional states coexist |
| "in other words" | `FRAME(restatement)` | Rephrasing; emotional content should match previous |
| "more importantly" | `CHAIN(x * 1.3)` + U+10 | Escalation marker |
| "first of all" | `FRAME(list_start)` + A+10 | Beginning of enumeration; often signals frustration buildup |
| "last but not least" | `CHAIN(x * 1.2)` | Signals importance of final item |
| "all things considered" | `FRAME(summary)` + D+10 | Weighed conclusion |
| "needless to say" | `CHAIN(x * 1.2)` + D+15 | Redundancy marker; implies obviousness |

### Contrast Operators (Special Class)

These reset emotional momentum and signal direction change:

| Operator | Reset Strength | Notes |
|----------|---------------|-------|
| "but" | 70% momentum reset | Most common; strong pivot |
| "however" | 60% momentum reset | Formal "but" |
| "yet" | 50% momentum reset | Softer contrast |
| "although" | 40% momentum reset | Concessive; first clause acknowledged |
| "on the other hand" | 80% momentum reset | Strong explicit contrast |
| "nevertheless" | 50% momentum reset | Formal; persistence through adversity |
| "instead" | 90% momentum reset | Near-complete replacement |
| "rather" | 80% momentum reset | Preference/correction |

### Examples

1. "In general, the team is doing well, but there are concerns." -- "in general" dampens, "but" resets, "concerns" carries
2. "Of course she's upset -- you lied to her." -- "of course" amplifies certainty of "upset"
3. "On the other hand, it could be much worse." -- 80% momentum reset; new direction is mildly positive
4. "As a result, everyone was on edge." -- causal frame; "on edge" gets U+10 from consequence framing
5. "First of all, I never said that." -- "first of all" signals the start of a defensive enumeration; A+10 for buildup

---

## 13. Conditional Frames

**Operation:** `GATE(p)` -- multiplies emotional force by probability factor

**Affects:** All dimensions (force is scaled by likelihood); U (raised by uncertainty)

Conditional frames mark content as hypothetical. The emotional impact is real but GATED by the perceived likelihood. "If the building collapses" carries fear, but less than "the building is collapsing."

| Conditional | Gate Factor | U Offset | Notes |
|-------------|-----------|----------|-------|
| "if" | p=0.5 | U+10 | Default uncertainty |
| "when" (future) | p=0.8 | U+15 | Expected to happen |
| "unless" | p=0.3 (on the "unless" clause) | U+15 | Low probability contingency |
| "in case" | p=0.3 | U+10 | Precautionary |
| "even if" | p=0.5 but irrelevant | D+15 | Defiance marker; outcome claimed regardless |
| "what if" | p=0.4 | U+20, A+15 | Anxiety-producing speculation |
| "assuming" | p=0.7 | U+5 | Working hypothesis |
| "provided that" | p=0.6 | U+10 | Conditional with structure |
| "suppose" | p=0.4 | U+10 | Hypothetical invitation |
| "hypothetically" | p=0.2 | U+5 | Explicitly unreal; low gate |
| "as long as" | p=0.7 | U+10 | Conditional maintenance |
| "once" (future) | p=0.9 | U+5 | High-confidence future |

### Key Principle

The gate factor does NOT eliminate emotional content -- it scales it. A gated fear (p=0.5) is still fear, just at half intensity. The uncertainty itself adds U.

### Conditional Inversion

The emotional content of the conditional clause and the main clause often oppose each other:
- "If it rains, we'll stay home." -- rain=negative, staying home=depends
- "Unless you apologize, we're done." -- apologize=positive, done=negative; the threat is primary

### Examples

1. "If I lose my job, I don't know what I'll do." -- `GATE(0.5)` on "lose my job" fear; still carries significant V-50 * 0.5 = V-25
2. "When this is over, everything will be better." -- `GATE(0.8)` on "better"; high confidence relief
3. "What if they say no?" -- `GATE(0.4)` on rejection anxiety; U+20 for the uncertainty itself
4. "Unless something changes, I'm leaving." -- threat is primary; "leaving" is GATED at 0.7 (likely)
5. "Even if it hurts, I have to do this." -- `GATE` is overridden by "even if" defiance; D+15

---

## 14. Temporal Markers

**Operation:** `FRAME(temporal)` + scope/frequency modifiers

**Affects:** A (chronic vs acute), U (ongoing vs resolved), G (weight accumulation)

Temporal markers change the emotional SCOPE. "I'm sad" is acute. "I'm always sad" is chronic -- fundamentally different emotional state with different VADUGWI implications.

| Marker | Operation | Emotional Effect |
|--------|-----------|-----------------|
| "always" | `CHAIN(x * 1.4)` + U+15 | Universalizes; makes it inescapable |
| "never" | `CHAIN(x * 1.4)` + U+15 + D-15 | Absolute negation; hopelessness when negative |
| "sometimes" | `CHAIN(x * 0.5)` | Intermittent; reduces consistency |
| "often" | `CHAIN(x * 0.8)` | Frequent but not universal |
| "rarely" | `CHAIN(x * 0.3)` | Infrequent; minimal impact |
| "used to" | `FRAME(past)` + `CHAIN(x * 0.6)` | Past state; implies change; nostalgia or relief |
| "anymore" | `FRAME(cessation)` | Signals state change; "not anymore" = relief or loss |
| "still" | `CHAIN(x * 1.2)` + U+10 | Persistence; ongoing against expectation |
| "already" | A+10, U+10 | Faster than expected; surprise or impatience |
| "finally" | A+15, D+10 | Long-awaited resolution; relief or exasperation |
| "eventually" | `GATE(0.8)` + U-10 | Distant but expected; reduces immediate urgency |
| "suddenly" | A+30, U+25 | Abrupt onset; shock/surprise |
| "gradually" | A-15, U-10 | Slow onset; muted emotional impact |
| "lately" | `FRAME(recent_trend)` | Recent pattern; implies change |
| "forever" | `CHAIN(x * 1.5)` (often hyperbolic) | Permanence or hyperbole |
| "temporarily" | `CHAIN(x * 0.5)` + U-10 | Bounded duration; reduces stakes |
| "constantly" | `CHAIN(x * 1.3)` + U+15 | Relentless; exhausting |
| "once" (past) | `FRAME(past)` + `CHAIN(x * 0.4)` | Single past instance; diminished |

### Acute vs Chronic Scaling

| Pattern | Type | A Modifier | G Modifier |
|---------|------|-----------|-----------|
| "right now" | Acute | A+20 | -- |
| "this week" | Subacute | A+10 | -- |
| "for months" | Chronic | A-10 | G-15 (heavier) |
| "for years" | Deep chronic | A-20 | G-30 (crushing) |
| "my whole life" | Pervasive | A-25 | G-40 (weighted down) |

### Examples

1. "I'm always tired." -- "always" makes "tired" chronic: V-28 * 1.4, U+15, G-15 (accumulating weight)
2. "She used to be happy." -- `FRAME(past)` + dampened "happy"; implies current unhappiness by contrast
3. "I finally got the job!" -- "finally" adds A+15, D+10 to the positive "got the job"
4. "Suddenly everything changed." -- A+30, U+25 onset; "changed" is context-dependent
5. "I've been struggling for years." -- chronic frame: A-20, G-30; "struggling" force is persistent and heavy

---

## 15. Evidential Markers

**Operation:** `FRAME(source)` + D adjustment based on certainty level

**Affects:** D (speaker's confidence), V (dampened by distance from direct experience)

Evidential markers tell us HOW the speaker knows what they're claiming. Direct experience has full emotional weight. Hearsay, inference, and speculation progressively reduce it.

| Marker | Certainty | D Offset | Force Multiplier | Notes |
|--------|-----------|----------|-----------------|-------|
| "I know" | 1.0 | D+15 | x * 1.0 | Direct knowledge |
| "I feel" | 0.9 (subjective) | D+5 | x * 0.9 | Emotional truth; high personal validity |
| "I think" | 0.7 | D-10 | x * 0.7 | Opinion; hedged |
| "I believe" | 0.75 | D-5 | x * 0.8 | Conviction without certainty |
| "I suspect" | 0.5 | D-15 | x * 0.5 | Low confidence inference |
| "apparently" | 0.6 (hearsay) | D-20 | x * 0.6 | Second-hand information |
| "supposedly" | 0.4 | D-25 | x * 0.4 | Skeptical hearsay |
| "allegedly" | 0.3 | D-30 | x * 0.3 | Formal distancing; doubt implied |
| "they say" | 0.5 | D-20 | x * 0.5 | Attributed to others |
| "I heard" | 0.5 | D-15 | x * 0.5 | Second-hand |
| "it turns out" | 0.9 | D+10, A+10 | x * 1.0 | Discovery; surprise element |
| "as far as I can tell" | 0.6 | D-10 | x * 0.6 | Limited observation |
| "from what I understand" | 0.6 | D-10 | x * 0.6 | Second-hand synthesis |
| "clearly" | 1.0 | D+20 | x * 1.1 | Asserted as obvious |
| "obviously" | 1.0 | D+25 | x * 1.15 | Strong assertion of self-evidence |

### Key Principle

Evidential markers do NOT change what emotion is expressed -- they change how COMMITTED the speaker is to it. "I think I'm upset" and "I know I'm upset" express the same emotion at different confidence levels. The engine should scale force AND adjust D accordingly.

### Examples

1. "I feel like something is wrong." -- "I feel" = subjective truth; force at 0.9; D+5
2. "Apparently they're getting divorced." -- hearsay; force at 0.6; D-20; the speaker distances
3. "Obviously this is a disaster." -- "obviously" = maximum assertion; force at 1.15; D+25
4. "I suspect he's lying." -- low confidence; force at 0.5; D-15; accusation is tentative
5. "It turns out everything was fine." -- discovery + relief; force at 1.0 with A+10 surprise

---

## 16. Social Politeness

**Operation:** `OFFSET` on D and V; `CONTEXT(politeness)` metadata

**Affects:** D (lowered by deference, raised by authority), V (mild positive from social lubrication)

Politeness markers serve social function more than emotional content. They adjust the power dynamic (D) between speaker and addressee. Their emotional content is typically mild.

| Marker | V Offset | D Offset | A Offset | Notes |
|--------|---------|---------|---------|-------|
| "please" | V+5 | D-15 | -- | Deference; requesting |
| "thank you" | V+15 | D-5 | -- | Gratitude (genuine) |
| "thanks" | V+10 | D-5 | -- | Casual gratitude |
| "excuse me" | V+5 | D-20 | -- | Apology for intrusion |
| "I'm sorry" | V+5 or V-15 | D-20 | -- | Apology (context-dependent) |
| "pardon" | V+5 | D-25 | -- | Formal deference |
| "if you don't mind" | V+5 | D-25 | -- | Extreme deference |
| "would you mind" | V+5 | D-20 | -- | Polite request form |
| "I appreciate it" | V+20 | D0 | -- | Genuine gratitude |
| "with all due respect" | V-5 | D+15 | A+10 | Prelude to disagreement |
| "no offense" | V-10 | D+10 | A+10 | Prelude to offensive content |
| "I don't mean to be rude" | V-10 | D+5 | A+10 | Same as above |
| "forgive me" | V-5 | D-25 | A+5 | Seeking absolution |
| "my apologies" | V-5 | D-20 | -- | Formal sorry |
| "kindly" | V+5 | D+10 | -- | Formal request; veiled authority |

### Weaponized Politeness

Some politeness markers signal the OPPOSITE of their surface meaning:

| Pattern | Surface | Actual |
|---------|---------|--------|
| "with all due respect" + criticism | Respectful | Disrespectful; D+15 (dominance) |
| "no offense, but" | Non-offensive | Offensive content incoming; A+10 |
| "bless your heart" (Southern US) | Blessing | Condescension; D+20, V-15 |
| "I'm sorry you feel that way" | Apology | Non-apology; D+20, V-10 |
| "please, by all means" | Permission | Sarcastic permission; `FLIP` possible |

### Examples

1. "Please help me, I don't know what to do." -- "please" adds D-15 (supplication); "don't know" already low-D
2. "Thank you so much for everything." -- V+15, "so much" amplifies; genuine gratitude
3. "With all due respect, that's completely wrong." -- weaponized: D+15 for speaker, then strong negative claim
4. "No offense, but your idea won't work." -- A+10 tension marker; the criticism is primary
5. "I'm sorry you feel that way." -- non-apology: D+20 (maintains position), V-10 (dismissive)

---

## 17. Exclamations and Interjections

**Operation:** `OFFSET` on A (all exclamations raise arousal) + V offset by type

**Affects:** A (always elevated), V (direction depends on interjection), G (some shift)

Interjections are pure emotional bursts. They carry arousal first, valence second. They have no compositional meaning -- "wow" doesn't modify anything, it IS the emotional signal.

| Interjection | V Offset | A Offset | D Offset | G Offset | Notes |
|-------------|---------|---------|---------|---------|-------|
| "oh!" | varies | A+20 | -- | -- | Surprise; valence from context |
| "wow" | V+20 | A+30 | -- | G+10 | Positive surprise |
| "damn" | V-15 | A+25 | D+10 | -- | Frustration/emphasis |
| "yay" | V+35 | A+35 | D+10 | G+20 | Joy/celebration |
| "ugh" | V-25 | A+20 | D-5 | G-10 | Disgust/annoyance |
| "whoa" | V+5 | A+30 | D-10 | G+15 | Surprise/awe |
| "ouch" | V-20 | A+25 | D-15 | -- | Pain |
| "yikes" | V-20 | A+30 | D-15 | G+10 | Alarm |
| "aha" | V+15 | A+25 | D+15 | -- | Discovery |
| "hmm" | V0 | A+5 | D+5 | -- | Contemplation |
| "ew" | V-30 | A+25 | D+5 | G+15 | Disgust/repulsion |
| "aww" | V+25 | A+15 | D-10 | -- | Tenderness |
| "phew" | V+15 | A-10 | D+10 | G-5 | Relief |
| "sheesh" | V-15 | A+20 | D+5 | -- | Exasperation |
| "huh" | V0 | A+10 | D-5 | -- | Confusion |
| "oh no" | V-25 | A+30 | D-15 | G-10 | Alarm/dread |
| "hell yeah" | V+35 | A+40 | D+25 | G+20 | Emphatic enthusiasm |
| "holy shit" | V varies | A+40 | D-10 | G+15 | Intense surprise |
| "bruh" | V-10 | A+15 | D+10 | -- | Disbelief (slang) |
| "lol" | V+15 | A+10 | D+5 | G+5 | Amusement marker |
| "omg" | V varies | A+35 | D-10 | -- | Intense surprise (text) |
| "wtf" | V-15 | A+35 | D+5 | -- | Shock/outrage (text) |

### Exclamation Marks

Exclamation marks themselves are arousal amplifiers:
- Single `!` -- A+10
- Double `!!` -- A+20
- Triple `!!!` -- A+25 (diminishing returns beyond 3)
- Question + exclamation `?!` -- A+25, confusion + surprise

### Examples

1. "Wow, that's incredible!" -- V+20, A+30 from "wow" + A+10 from `!` + "incredible" force
2. "Ugh, not this again." -- V-25, A+20 from "ugh" + "not this again" = recurring frustration
3. "Yay! We did it!" -- V+35, A+35 from "yay" + double `!` = A+45 total
4. "Oh no, what happened?" -- V-25, A+30; genuine concern
5. "Damn, that was close." -- V-15, A+25; near-miss relief mixed with alarm

---

## 18. Tag Questions

**Operation:** `CHAIN(D-10 to D-25)` -- reduces speaker's dominance; seeks validation

**Affects:** D (lowered -- seeking agreement), A+5 to A+10 (mild tension from uncertainty)

Tag questions convert statements into requests for confirmation. They signal the speaker is not fully committed to their assertion. Emotionally, they reduce dominance and add mild anxiety.

| Tag | D Offset | A Offset | Notes |
|-----|---------|---------|-------|
| "right?" | D-10 | A+5 | Casual confirmation seeking |
| "isn't it?" | D-15 | A+5 | Standard tag |
| "don't you think?" | D-20 | A+10 | Actively soliciting agreement |
| "wouldn't you say?" | D-20 | A+10 | Formal solicitation |
| "you know?" | D-10 | A+5 | Casual; phatic |
| "am I wrong?" | D-15 | A+10 | Vulnerable; opens to challenge |
| "isn't that so?" | D-15 | A+5 | Formal |
| "no?" | D-10 | A+5 | Romance-language influenced |
| "huh?" | D-5 | A+10 | Casual/confused |
| "yeah?" | D-5 | A+5 | Casual confirmation |

### Aggressive Tags (Inverted D)

Some tags RAISE dominance instead of lowering it:

| Tag | D Offset | A Offset | Notes |
|-----|---------|---------|-------|
| "got it?" | D+20 | A+15 | Commanding |
| "understand?" | D+20 | A+15 | Authoritative |
| "capisce?" | D+25 | A+15 | Intimidating |
| "or what?" | D+15 | A+20 | Threatening |
| "or else" | D+20 | A+20, U+20 | Explicit threat |

### Examples

1. "This is the right decision, isn't it?" -- D-15; the speaker doubts their own claim
2. "We should go, right?" -- D-10; seeking group consensus
3. "That was rude, don't you think?" -- D-20; inviting validation of judgment
4. "You'll have it done by Friday, got it?" -- D+20; authoritative, not seeking agreement
5. "Pretty cool, huh?" -- D-5; casual enthusiasm seeking shared excitement

---

## 19. Passive Voice Emotional Shifts

**Operation:** `OFFSET` on D (agency removal) + `CONTEXT(passive_agent)`

**Affects:** D (lowered for the one acted upon, removed for the actor), V (responsibility is diffused)

Passive voice removes the actor from the emotional equation. "Mistakes were made" does not assign blame. "I made mistakes" does. The emotional impact shifts from personal accountability (high D on actor) to diffused responsibility (low D, ambiguous agent).

| Active Form | Passive Form | D Shift | V Shift | Notes |
|-------------|-------------|---------|---------|-------|
| "I broke it" | "It was broken" | D-30 (actor removed) | V+10 (less blame) | Accountability vanishes |
| "They fired me" | "I was let go" | D-20 (victim framing) | V+10 (softened) | Euphemism + passive |
| "You hurt me" | "I was hurt" | D-15 (attacker removed) | -- | Focus shifts to suffering |
| "I made a mistake" | "Mistakes were made" | D-40 (no agent) | V+15 (blame diffused) | Classic political passive |
| "He betrayed us" | "We were betrayed" | D-25 (actor removed) | -- | Focus on victimhood |
| "She ruined everything" | "Everything was ruined" | D-30 | V+5 | Diffused responsibility |
| "They rejected me" | "I was rejected" | D-20 | -- | Focus on experience |
| "We failed" | "It wasn't achieved" | D-35 | V+10 | Maximum diffusion |

### Key Principle

Passive voice does three things:
1. **Removes the agent** -- lowers D because nobody is "doing" the action
2. **Elevates the patient** -- the one acted upon becomes the subject
3. **Diffuses emotional responsibility** -- anger at a passive construction is harder to sustain than anger at a named actor

### Detection

Passive voice is detected by:
- Auxiliary "was/were/been/being" + past participle
- Missing agent (no "by X" clause)
- Subject receives the action

### Examples

1. "Mistakes were made." -- D-40; no one to blame; V+15 (blame is absorbed by the void)
2. "I was told I'm not good enough." -- D-20 (agent hidden); the judgment feels institutional, not personal
3. "Your application has been denied." -- D-30 (bureaucratic passive); U+15 (institutional power)
4. "It was decided that layoffs were necessary." -- maximum diffusion; D-40, V+15
5. "She was hurt by what happened." -- actor is "what happened" (abstract); D-25

---

## 20. Comparative Structures

**Operation:** `FRAME(comparison)` + `OFFSET` based on direction

**Affects:** V (directional shift), D (winner gains, loser loses), A+5 to A+15 (tension from evaluation)

Comparatives create a RELATIVE emotional frame. "Better than expected" is positive not because of absolute goodness but because of exceeded expectations. The emotional force comes from the GAP between compared items.

| Structure | Operation | Notes |
|-----------|-----------|-------|
| "better than" | V+15 to V+30 (gap-dependent) | Positive gap |
| "worse than" | V-15 to V-30 | Negative gap |
| "more X than Y" | Amplifies X force | Relative emphasis |
| "less X than Y" | Dampens X, implies Y | Inverse comparison |
| "as X as Y" | Equalizer; neutral | Similarity claim |
| "not as X as" | X is lacking | Negative gap disguised as comparison |
| "rather X than Y" | Preference frame | X is positive, Y is negative |
| "compared to" | `FRAME(baseline)` | Sets reference point |

### Gap-Based Scaling

The emotional force of a comparison scales with the perceived gap:

| Gap Size | Example | V Modifier |
|----------|---------|-----------|
| Small gap | "slightly better" | V+10 |
| Medium gap | "better" | V+20 |
| Large gap | "much better" | V+30 |
| Extreme gap | "infinitely better" | V+40 (hyperbolic cap) |

### Direction Matters

- "Better than I thought" -- positive surprise: V+25, A+15
- "Worse than I thought" -- negative surprise: V-25, A+15
- "Better than nothing" -- faint praise: V+5 (low baseline)
- "Worse than death" -- hyperbolic: V-40 (but dampened as hyperbole)

### Examples

1. "This is way better than last time." -- V+30 (large gap); positive momentum
2. "It's worse than I imagined." -- V-30, A+15 (surprise intensifies negative)
3. "She's more talented than anyone I know." -- superlative via comparison; V+35, D+20
4. "Not as bad as it could have been." -- litotes + comparison: V+10 (mild relief)
5. "I'd rather fail trying than never try at all." -- preference frame: "fail trying" is elevated, "never try" is condemned

---

## 21. Superlatives

**Operation:** `CHAIN(x * 1.5 to 2.0)` -- maximum amplification of the base adjective

**Affects:** V (extreme directional push), A+15 to A+25 (intensity from extremity), D+10 (certainty of claim)

Superlatives push to the extremes of the emotional range. They assert that something is at the maximum or minimum of a scale. Often hyperbolic.

| Superlative | V Effect | A Effect | Notes |
|-------------|---------|---------|-------|
| "best" | V * 1.8 | A+20 | Peak positive |
| "worst" | V * 1.8 (negative) | A+20 | Peak negative |
| "most beautiful" | V * 1.7 | A+15 | Peak aesthetic positive |
| "most terrible" | V * 1.7 (negative) | A+15 | Peak negative |
| "least" | V * 0.3 | A-10 | Minimum of positive scale |
| "greatest" | V * 1.8 | A+20, D+15 | Peak achievement |
| "smallest" | depends on context | -- | Size superlative; emotional context-dependent |
| "first ever" | V * 1.5 | A+25 | Novelty amplifier |
| "last ever" | V * 1.3 | A+20, G-10 | Finality; heavier |
| "best ever" | V * 2.0 | A+25 | Lifetime peak (often hyperbolic) |
| "worst ever" | V * 2.0 (negative) | A+25 | Lifetime nadir (often hyperbolic) |

### Hyperbole Interaction

Superlatives are frequently hyperbolic. Apply hyperbole dampening (x * 0.5) when context suggests exaggeration:
- "Worst day ever" in casual context -- hyperbolic: actual force = V * 2.0 * 0.5 = V * 1.0
- "Worst day ever" after genuine catastrophe -- literal: actual force = V * 2.0

### Scope Amplifiers

Superlatives interact with scope words:

| Scope | Modifier | Notes |
|-------|----------|-------|
| "of the day" | x * 1.0 | Short scope; minimal amplification |
| "of the year" | x * 1.2 | Medium scope |
| "of my life" | x * 1.5 | Personal scope; maximum weight |
| "in the world" | x * 1.3 | Universal scope; often hyperbolic |
| "in history" | x * 1.5 | Maximum scope (almost always hyperbolic) |
| "ever" | x * 1.5 | Unbounded scope |

### Examples

1. "This is the best day of my life." -- "best" * 1.8 + "of my life" * 1.5 = massive positive; but check for hyperbole
2. "That was the worst movie I've ever seen." -- "worst" * 1.8 + "ever" * 1.5; casual context = hyperbole dampening
3. "She's the most compassionate person I know." -- "most" * 1.7 on "compassionate"; genuine praise
4. "This is the least helpful thing you could have said." -- "least" * 0.3 on "helpful" = nearly unhelpful; criticism
5. "Greatest achievement in the history of this company." -- "greatest" * 1.8 + "history" scope * 1.5; ceremonial amplification

---

## 22. Colloquialisms / Slang

**Operation:** `REPLACE` or `OFFSET` -- slang words carry fixed emotional vectors that differ from standard meanings

**Affects:** All dimensions; slang tends toward higher A (energy), variable V

Slang creates a register shift. The emotional content differs from the literal dictionary meaning. "That's sick" (slang) is POSITIVE; "I feel sick" is negative. Context and register detection is critical.

| Slang Term | Slang Meaning | V | A | D | G | Notes |
|-----------|--------------|---|---|---|---|-------|
| "lit" | Exciting/great | +35 | +40 | +15 | +15 | Gen Z positive |
| "dead" | Very amused | +30 | +35 | -10 | +10 | "I'm dead" = dying laughing |
| "sick" (positive) | Cool/impressive | +30 | +30 | +15 | +10 | Register-dependent |
| "slay" | Excel/dominate | +35 | +35 | +30 | +15 | Empowerment |
| "based" | Admirably authentic | +25 | +20 | +25 | +5 | Approval of boldness |
| "cap" / "no cap" | Lie / truth | V varies | +15 | +10 | -- | "no cap" = honest |
| "fire" | Excellent | +35 | +35 | +15 | +15 | Approval |
| "mid" | Mediocre | -15 | -10 | -10 | -5 | Dismissive |
| "sus" | Suspicious | -15 | +20 | +10 | -- | Distrust |
| "vibe" / "vibing" | Relaxed/enjoying | +25 | -10 | +10 | +10 | Calm positive |
| "bet" | Agreement/OK | +10 | +10 | +10 | -- | Casual affirmation |
| "fam" | Close friend/group | +15 | +5 | +5 | -- | Solidarity marker |
| "lowkey" | Quietly/subtly | `CHAIN(x * 0.6)` | -- | -- | -- | Diminisher (slang) |
| "highkey" | Openly/very | `CHAIN(x * 1.4)` | -- | -- | -- | Amplifier (slang) |
| "no cap" | Truthfully | D+15 | A+5 | -- | -- | Evidential marker (slang) |
| "salty" | Bitter/resentful | -25 | +25 | -10 | -- | Informal negative |
| "shook" | Shocked/overwhelmed | -10 | +35 | -15 | +10 | Surprise/alarm |
| "slaps" | Excellent (music/food) | +30 | +30 | +10 | +10 | Sensory positive |
| "rizz" | Charisma/charm | +20 | +20 | +25 | +10 | Social competence |
| "L" / "taking an L" | Loss/failure | -30 | +15 | -20 | -10 | Defeat |
| "W" / "big W" | Win/success | +30 | +20 | +20 | +10 | Victory |
| "yeet" | Throw/discard with force | +10 | +35 | +20 | +15 | Chaotic energy |
| "cringe" | Embarrassing | -25 | +20 | -15 | -- | Second-hand embarrassment |
| "cope" | Self-deception | -15 | +10 | -15 | -- | Dismissal of coping |
| "goated" | Greatest | +40 | +25 | +25 | +15 | Supreme praise |

### Register Detection

The engine must detect register to disambiguate:
- "That's sick!" (enthusiastic context) = positive slang
- "I feel sick." (health context) = literal negative
- "He's dead." (reaction context) = amusement slang
- "He's dead." (news context) = literal death

### Examples

1. "That concert was absolutely lit." -- "lit" = V+35, A+40 + "absolutely" ramp x1.7
2. "Bro I'm literally dead." -- "dead" (slang) = V+30, A+35; "literally" is hyperbolic marker
3. "She ate that performance, no cap." -- "ate" (slang: excelled) + "no cap" (truthfully) = genuine high praise
4. "That take is mid at best." -- "mid" = V-15, A-10 + "at best" = even the ceiling is mediocre
5. "Lowkey shook about tomorrow." -- "lowkey" * 0.6 on "shook": quietly anxious

---

## 23. Code-Switching Signals / Discourse Fillers

**Operation:** `CONTEXT(filler)` -- mostly emotional noise, but some carry social/cognitive signals

**Affects:** Minimal direct effect; some adjust D or signal processing difficulty

Fillers and code-switching markers are often stripped in NLP. But they carry real emotional information: hesitation signals uncertainty (D-), repetition signals emphasis (A+), and social markers signal group belonging.

| Filler | Operation | Notes |
|--------|-----------|-------|
| "basically" | `FRAME(simplification)` | Speaker is translating; D+5 (explaining) |
| "like" (filler) | D-5 per instance | Casual register; hedging |
| "you know" | D-5, `CONTEXT(phatic)` | Seeking connection, not information |
| "I mean" | `FRAME(correction)` + D-5 | Self-repair; rethinking |
| "literally" (filler) | `CHAIN(x * 1.2)` or hyperbole marker | Depends on context |
| "actually" | `FRAME(correction)` + D+10 | Counter-expectation; mild assertiveness |
| "honestly" | `CHAIN(x * 1.15)` + D+10 | Signals elevated authenticity |
| "so" (sentence-initial) | `FRAME(narrative)` | Story-starting marker |
| "well" (sentence-initial) | `FRAME(hedged_response)` + D-5 | Signals reluctance or qualification incoming |
| "anyway" | `FRAME(topic_shift)` + momentum reset 50% | Dismisses previous, moves on |
| "whatever" | V-15, D-10 | Dismissive resignation |
| "look" (sentence-initial) | D+15, A+10 | Commanding attention; assertive |
| "listen" | D+15, A+10 | Same as "look" |
| "right" (filler) | D-5 | Checking comprehension |
| "okay so" | `FRAME(setup)` | Preparing to explain |
| "the thing is" | `FRAME(revelation)` + A+10 | Building to important point |

### Frequency as Signal

Filler density itself carries information:
- Low fillers: formal, confident, rehearsed (D+)
- High fillers: informal, uncertain, spontaneous (D-)
- Sudden increase in fillers: emotional difficulty processing (A+, D-)

### Examples

1. "I mean, like, I don't know, it's just... whatever." -- high filler density = processing difficulty; D-30 cumulative
2. "Look, I'm going to be honest with you." -- "look" D+15 + "honest" D+10 = assertive honesty frame
3. "Basically what happened was everything fell apart." -- "basically" simplification frame; main content is the emotional payload
4. "Well, it's not exactly what I hoped for." -- "well" signals hesitation; litotes follows
5. "Actually, that's not true at all." -- "actually" D+10; counter-correction; assertive denial

---

## 24. Emotional Performatives

**Operation:** `CHAIN(x * 1.2 to 1.5)` + `D+15 to D+30` -- signals elevated commitment to the emotional content

**Affects:** D (raised -- speaker is taking a strong position), A+10 to A+20 (emotional intensity), V amplified

Performatives are speech acts where saying the thing IS the action. "I promise" does not describe a promise -- it IS a promise. These elevate the emotional stakes and speaker commitment.

| Performative | D Offset | A Offset | Force Multiplier | Notes |
|-------------|---------|---------|-----------------|-------|
| "I promise" | D+25 | A+15 | x * 1.4 | Binding commitment |
| "I swear" | D+30 | A+20 | x * 1.5 | Maximum commitment |
| "trust me" | D+20 | A+10 | x * 1.3 | Requesting faith |
| "believe me" | D+25 | A+15 | x * 1.4 | Insistent credibility |
| "I guarantee" | D+30 | A+15 | x * 1.5 | Formal binding |
| "mark my words" | D+25 | A+20 | x * 1.4 | Prophetic authority |
| "I assure you" | D+20 | A+10 | x * 1.3 | Formal reassurance |
| "take my word for it" | D+20 | A+10 | x * 1.3 | Authority claim |
| "I vow" | D+30 | A+20 | x * 1.5 | Ritual binding |
| "cross my heart" | D+20 | A+15 | x * 1.3 | Informal solemn |
| "I confess" | D-10 | A+20 | x * 1.3 | Vulnerability; D drops |
| "I admit" | D-5 | A+15 | x * 1.2 | Concession; mild vulnerability |
| "I warn you" | D+25 | A+25, U+20 | x * 1.5 | Threat frame |
| "I beg you" | D-30 | A+25 | x * 1.5 | Maximum supplication |
| "I demand" | D+30 | A+25, U+20 | x * 1.5 | Maximum authority |

### Commitment Escalation

Performatives can stack:
- "I promise" = D+25
- "I swear to God I promise" = D+30 + D+25 = high commitment
- But over-stacking reads as desperation (D should curve back down after 2+ performatives)

### Inverse Correlation

Frequent performatives can signal DECREASED trustworthiness:
- A single "trust me" is normal
- Repeated "trust me, believe me, I swear" = protest-too-much; engine should apply skepticism discount (x * 0.8 on the third+ performative)

### Examples

1. "I promise this will work out." -- D+25, x * 1.4 on "work out": elevated positive commitment
2. "Trust me, you don't want to go there." -- D+20, x * 1.3 on negative: amplified warning
3. "I swear I didn't know." -- D+30, x * 1.5 on "didn't know": strong denial; defensive
4. "I confess I was wrong." -- D-10 (vulnerability), x * 1.3 on "wrong": amplified admission
5. "I beg you, don't leave." -- D-30 (maximum supplication), x * 1.5 on "don't leave": desperate pleading

---

## Summary: The Equation Tree

### Operation Priority (Processing Order)

When multiple devices co-occur, apply in this order:

1. **Frame setters** -- establish context (conditionals, discourse markers, temporal markers)
2. **Idiom/euphemism replacement** -- override individual word forces
3. **Negation** -- flip valence
4. **Sarcasm detection** -- may flip valence again
5. **Intensifiers/diminishers** -- scale magnitude (ramps with decay)
6. **Hedges/evidentials** -- reduce commitment
7. **Performatives** -- boost commitment
8. **Tag questions** -- adjust D
9. **Exclamations** -- add A burst
10. **Comparative/superlative** -- relative scaling
11. **Passive voice** -- D adjustment
12. **Politeness** -- social D adjustment
13. **Hyperbole check** -- dampen if exaggerated

### Device Interaction Matrix

| When X meets Y | Result |
|----------------|--------|
| Intensifier + Hyperbole | Amplification is DAMPENED (hyperbole already overstates) |
| Hedge + Performative | Conflict: hedge wins on V, performative wins on D |
| Sarcasm + Politeness | Weaponized politeness: surface D-low, actual D-high |
| Negation + Negation | Double negative: mild positive |
| Conditional + Temporal | Gate factor interacts with scope: "if always" = chronic hypothetical |
| Diminisher + Litotes | Compound dampening: "just not bad" = barely OK |
| Superlative + Euphemism | Rare but possible: "the most peaceful passing" |
| Exclamation + Rhetorical Q | "Who cares?!" -- A from `?!` + rhetorical SET |
| Slang + Intensifier | "Hella based" -- slang vector * amplifier ramp |
| Filler density + Hedge | Compound uncertainty: D drops significantly |
| Passive + Euphemism | Maximum diffusion: "lessons were learned" |
| Performative + Sarcasm | "Oh, I PROMISE this'll be fun." -- performative flipped |

### VADUGWI Impact Summary by Category

| Category | V | A | D | U | G |
|----------|---|---|---|---|---|
| Pragmatic devices | frame | -- | +/- | -- | -- |
| Hedging qualifiers | dampen | -- | -- | -- | -- |
| Intensifiers | amplify | + | -- | -- | -- |
| Diminishers | dampen | - | -- | -- | -- |
| Sarcasm markers | FLIP | + | + | -- | -- |
| Double negation | flip+dampen | -- | -- | -- | -- |
| Rhetorical questions | SET | + | +/- | -- | -- |
| Euphemisms | dampen | - | -- | -- | +/- |
| Hyperbole | dampen (actual) | + | -- | -- | -- |
| Litotes | flip+dampen | -- | + | -- | -- |
| Idioms | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE |
| Compositional semantics | frame/reset | -- | -- | -- | -- |
| Conditional frames | GATE | -- | -- | + | -- |
| Temporal markers | scope | +/- | -- | +/- | +/- |
| Evidential markers | scale | -- | +/- | -- | -- |
| Social politeness | +/- | -- | +/- | -- | -- |
| Exclamations | +/- | + | -- | -- | +/- |
| Tag questions | -- | + | - | -- | -- |
| Passive voice | + | -- | - | -- | -- |
| Comparatives | +/- | + | +/- | -- | -- |
| Superlatives | amplify | + | + | -- | -- |
| Colloquialisms | REPLACE | + | -- | -- | -- |
| Code-switching | -- | -- | - | -- | -- |
| Performatives | amplify | + | + | +/- | -- |

---

## Implementation Notes

### What Exists Today

The engine already handles:
- **Negation** (`NEGATORS` in `pendulum.py`) -- `FLIP(V)` on next emotional word
- **Intensifiers/Diminishers** (`RAMPS` in `pendulum.py`) -- decay-based amplifiers
- **Idioms** (`IDIOMS` in `pendulum.py`) -- multi-word replacement
- **Sarcasm** (`SarcasmDetector` in `sarcasm.py`) -- trajectory-based detection
- **Word forces** (`WORD_FORCES` in `forces.py`) -- 6,400+ individual word vectors

### What Needs Building

In rough priority order:

1. **Euphemism detection** -- extend IDIOMS with euphemism vectors and metadata tags
2. **Discourse markers / frame setters** -- new pre-processing pass that sets context before word-by-word processing
3. **Hedges and evidentials** -- multiplier chain system (similar to ramps but for dampening)
4. **Conditional gating** -- `GATE(p)` system for if/when/unless clauses
5. **Temporal scope modifiers** -- chronic vs acute detection with G adjustment
6. **Rhetorical question detection** -- pattern matching + implied answer mapping
7. **Comparative/superlative scaling** -- gap-based relative force calculation
8. **Passive voice detection** -- auxiliary + participle pattern -> D adjustment
9. **Performative detection** -- commitment amplifier chain
10. **Tag question detection** -- D adjustment at clause end
11. **Politeness layer** -- social D adjustments
12. **Slang register detection** -- context-dependent vector selection
13. **Hyperbole detection** -- dampening when surface meaning is physically impossible
14. **Litotes resolution** -- double-negative to weak-positive converter
15. **Filler density tracking** -- cumulative D adjustment from processing difficulty signals

### Architecture Suggestion

Each category maps to one of three engine integration points:

| Integration Point | Categories |
|-------------------|-----------|
| **Pre-pass** (before word-by-word) | Frames, conditionals, rhetorical Qs, passive voice, register detection |
| **Word-level** (during pendulum) | Intensifiers, diminishers, negation, hedges, evidentials, performatives, exclamations, slang |
| **Post-pass** (after word-by-word) | Sarcasm validation, hyperbole check, filler density, tag question adjustment |
