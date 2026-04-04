# Council Round 5 — 4-LLM Consensus Synthesis

## Date: 2026-04-03
## Brothers: GPT-4, Claude, Gemini, Grok

## The Consensus (ALL 4 agree)

The remaining 41% error rate is **physics, not vocabulary**. All 4 brothers
identified the same 6 problems and converged on the same solutions.

### The Missing Layer (GPT's Key Insight)

> "You're doing: words → forces → result.
> What you need: words → INTERPRETATION → forces → result."

All 4 brothers agree: there's a missing **context interpretation layer**
between word classification and force computation. This layer reads
pragmatic function (discourse vs negation, intensifier vs emotional,
instructional vs conversational) BEFORE forces are applied.

## The 6 Physics Problems + Consensus Solutions

### 1. DISCOURSE MARKERS (Highest Priority)
- **Problem**: "no" always = NEGATOR. "no we good" reads V=17.
- **All 4 agree**: Positional + lookahead reclassification. Sentence-initial
  "no" + positive content ahead → reclassify as AMPLIFIER/FILLER.
- **GPT**: "Words need contextual function tags, not fixed behavior."
- **Claude**: "Pipeline ordering is critical: discourse-marker reclassification
  must precede negation math."
- **Grok**: Full pseudocode with 3-token lookahead + register check.
- **Gemini**: 3-token window, forward_v_sum > 20 threshold.
- **STATUS**: ✅ IMPLEMENTED in V8.1 interpret_context()

### 2. EXPLETIVE-AS-INTENSIFIER
- **Problem**: "shit you are right" → V=26. Expletive crashes positive.
- **All 4 agree**: Sentence-initial expletive + positive trailing content →
  strip negative charge, optionally amplify.
- **Claude**: "trailing_valence > 30 threshold. Start conservative (40)."
- **Grok**: "next_clause_positive check, 1.4x amplifier if true."
- **GPT**: "They're not negative forces. They're multipliers applied to
  the following clause."
- **STATUS**: ✅ IMPLEMENTED in V8.1 interpret_context()

### 3. LITERARY COMPLEXITY PENALTY
- **Problem**: Long subordinate-clause sentences trigger false sarcasm/nullity.
- **All 4 agree**: Sentence complexity → dampen structure detection confidence.
- **Claude**: C = (WC/10) + (CC*0.5) + (SC*1.0). Sigmoid decay above C>3.
- **Grok**: tokenCount + subordinateDepth → dampener floor at 0.4.
- **GPT**: "Not everything should enter the emotional system."
- **STATUS**: ❌ NOT YET IMPLEMENTED. Needs complexity scoring in structures.

### 4. NEGATOR INVERSION (Full Sign Flip)
- **Problem**: "isn't very good" → still positive. Negation too weak.
- **All 4 agree**: FULL MULTIPLICATIVE INVERSION, not additive dampening.
- **Claude**: Option B (-0.7x) recommended. "Not good" ≠ "bad" intensity.
- **Grok**: "FULL inversion: forceMap[j] = -1.0 * forceMap[j] - 25"
- **GPT**: "Stop softly nudging negation. NEGATOR = polarity inverter."
- **Gemini**: "target['mod_v'] = target['raw_v'] * -1.2"
- **STATUS**: ✅ IMPLEMENTED in V8.1 interpret_context() (uses -1.0 flip)
- **NOTE**: Claude warns consumed negators must become FILLER to prevent
  double-negation via proximity. ✅ IMPLEMENTED.

### 5. REGISTER DETECTION
- **Problem**: Instructional/expository prose reads positive.
- **All 4 agree**: Sentence-level register classification → force dampening.
- **Claude**: 3 registers (CONVERSATIONAL=1.0, LITERARY=0.6, EXPOSITORY=0.3).
  Quoted dialogue inside literary = CONVERSATIONAL override.
- **Grok**: keyword + pattern classifier → global force scalar.
- **Gemini**: 80% reduction for instructional text (aggressive).
- **GPT**: "Mode switch, not a tweak."
- **STATUS**: ⚠️ PARTIAL. Basic instructional cue detection in V8.1 (0.4x).
  Needs LITERARY register and quoted-dialogue detection.

### 6. TENSE / COUNTERFACTUAL
- **Problem**: "trusted him" reads as trust-present. "supposed to grow old"
  reads positive.
- **All 4 agree**: Past tense + positive = dampen. Counterfactual + positive = invert.
- **Claude**: Two-pass: past-tense dampening (0.3x) + counterfactual inversion (-0.5x).
  "Past tense is genuinely ambiguous. Dampening > full inversion."
- **Grok**: Counterfactual indicators → -0.75x inversion on main predicate.
- **GPT**: "That's not sentiment. That's timeline violation."
- **Gemini**: Past tense detected by -ed suffix + irregular table.
- **STATUS**: ✅ IMPLEMENTED in V8.1 accumulate_forces() (counterfactual -0.75x,
  past_trust 0.5x dampening).

## Pipeline Order (All 4 Agree)

1. Tokenize + vocab lookup
2. Classify (role assignment)
3. **Register detection** (sentence-level dampener)
4. **Discourse marker reclassification** (before negation!)
5. **Expletive-as-intensifier**
6. **Tense/counterfactual modification**
7. Force accumulation (with register dampening)
8. **Complexity dampener** (on structure confidence)
9. **Negator inversion** (after discourse markers consumed)
10. Structure detection
11. Final V summation

## Projected Impact (Claude's Estimate)

Conservative: 59% → 71%
Median: 59% → 76-79%
Optimistic: 59% → 83%

## What's Implemented vs Remaining

| Fix | Implemented? | Impact |
|-----|-------------|--------|
| Discourse markers | ✅ V8.1 | "no we good" V=17→216 |
| Expletive intensifier | ✅ V8.1 | Partial (needs better affirm detection) |
| Complexity dampening | ❌ | Gutenberg/Philosophy biggest gap |
| Negator inversion | ✅ V8.1 | "not happy" properly neg |
| Register detection | ⚠️ Partial | Basic instructional. Needs LITERARY. |
| Tense/counterfactual | ✅ V8.1 | "supposed to" inverted |

## Key Quote

GPT: "You're not building a replacement for models. You're building a
control layer. And for once, your 'I explain it like I'm dumb' thing
actually helped you."
