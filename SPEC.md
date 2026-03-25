# Clanker-Lang Specification v0.1

## 1. Overview

Clanker is a bytecode-style intermediate representation for structured communication between AI systems. A Clanker program is a sequence of instructions. Each instruction is an opcode with optional target, source, destination, and parameters. Clanker programs can be decoded to any language (human or machine) via dictionary lookup.

## 2. Instruction Encoding

### 2.1 Binary Format

Each instruction is encoded as:

```
[opcode: u8] [target: u8] [src_var: u8] [dst_var: u8] [param_count: u8] [params...]
```

| Field        | Size   | Description                                  |
|-------------|--------|----------------------------------------------|
| opcode      | u8     | Operation code (0x00-0xFF)                   |
| target      | u8     | Target variable slot ($0-$31) or 0xFF (none) |
| src_var     | u8     | Source variable slot or 0xFF (none)           |
| dst_var     | u8     | Destination variable slot or 0xFF (none)      |
| param_count | u8     | Number of parameters (0-15)                  |
| params      | varies | Type-tagged parameters                       |

### 2.2 Text Format

The human-readable text format uses `@` as the instruction prefix:

```
@ <opcode> <target> <src> <param_count> {key: "value"} {key: "value"} ...
```

Variables are written as `$0` through `$31`. Unused slots are written as `$_`.

Example:
```
@ 0xC0 $0 $1 02 {method: "GET"} {path: "/api/health"}
```

### 2.3 Parameter Encoding

Each parameter is type-tagged:

```
[type: u4][length: u4][value: variable]
```

| Type Code | Name     | Length Encoding        | Description                    |
|-----------|----------|-----------------------|--------------------------------|
| 0x0       | str      | byte count (0-15)     | UTF-8 string (short)           |
| 0x1       | str_ext  | u16 length follows    | UTF-8 string (extended)        |
| 0x2       | int      | byte count of integer | Signed integer (big-endian)    |
| 0x3       | float    | 4 or 8 bytes          | IEEE 754 float                 |
| 0x4       | bool     | 0 = false, 1 = true   | Boolean                        |
| 0x5       | duration | 4 bytes (ms)          | Duration in milliseconds       |
| 0x6       | bytes    | u16 length follows    | Raw byte array                 |
| 0x7       | list     | item count            | Nested parameter list          |
| 0x8       | varref   | 1 byte (slot number)  | Reference to variable $0-$31   |
| 0x9       | map      | pair count            | Key-value pairs                |
| 0xA-0xF   | reserved | -                     | Reserved for future types      |

## 3. Variable Store

Clanker provides 32 variable slots: `$0` through `$31`.

- Variables are **untyped** at the opcode level; the dictionary determines how they render.
- Variables persist for the duration of the script execution.
- `$0` is conventionally the "current context" or "self" reference.
- `$_` represents an unused/ignored slot.

## 4. Opcode Ranges

| Range       | Category           | Description                           |
|-------------|-------------------|---------------------------------------|
| 0x00-0x1F   | Core              | Flow control, lifecycle, fundamentals |
| 0x20-0x2F   | Reasoning         | Chain-of-thought, inference, doubt    |
| 0x30-0x9F   | Reserved          | Reserved for future standard opcodes  |
| 0xA0-0xAF   | Hardware          | Device control (from delphinOS)       |
| 0xB0-0xBF   | Extended Hardware | Additional device/sensor operations   |
| 0xC0-0xCF   | Web               | HTTP, API, networking                 |
| 0xD0-0xDF   | Data              | Transform, query, storage             |
| 0xE0-0xEF   | Logic             | Branch, match, try/catch, loops       |
| 0xF0-0xFF   | User Space        | Runtime-registered custom opcodes     |

## 5. Composition Rules

### 5.1 Block Structure

Certain opcodes open a block scope that must be closed with `END` (0x0F):

- `WHEN` (0xE0) opens a conditional block
- `REPEAT` (0xE2) opens a loop block
- `MATCH` (0xE1) opens a match block
- `TRY` (0xE3) opens a try/catch block

### 5.2 Nesting

- Blocks may be nested to a maximum depth of **4**.
- Each nested block inherits the variable scope of its parent.
- Variables set inside a block remain visible after the block closes.

### 5.3 Execution Order

Instructions execute sequentially, top to bottom, unless a branching opcode (WHEN, MATCH, REPEAT) redirects flow.

## 6. Runtime Extension

### 6.1 REGISTER Opcode

The `REGISTER` opcode (0x0E) allows defining new opcodes at runtime within the user space range (0xF0-0xFF):

```
@ 0x0E $_ $_ 03 {opcode: 0xF0} {name: "CUSTOM_OP"} {params: [{name: "arg1", type: "str"}]}
```

Once registered, the opcode can be used like any built-in opcode for the remainder of the script.

### 6.2 Constraints

- Only opcodes in the range 0xF0-0xFF may be registered.
- A registered opcode cannot override a previously registered one in the same session.
- Registered opcodes are not persisted across sessions unless explicitly saved.

## 7. Versioning

### 7.1 Immutability Guarantee

**Opcodes are forever.** Once an opcode is ratified into the specification:

- Its numeric code never changes.
- Its semantic meaning never changes.
- Its parameter signature never changes.

New functionality is added by assigning new opcodes, never by redefining existing ones.

### 7.2 Spec Versioning

The specification itself is versioned with semantic versioning:

- **Patch** (0.1.x): Clarifications, typo fixes, no semantic changes.
- **Minor** (0.x.0): New opcodes added, new dictionary features, backward compatible.
- **Major** (x.0.0): Breaking changes to encoding format (expected to be extremely rare).

### 7.3 Dictionary Versioning

Dictionaries carry their own version and declare which spec version they target:

```yaml
spec_version: "0.1"
dictionary_version: "1.0"
```

## 8. Magic Bytes

Compiled Clanker binary files begin with the magic bytes:

```
CLK\x01
```

- `CLK` identifies the file as Clanker bytecode.
- `\x01` is the binary format version.

## 9. Emotional Vector Encoding (VADU)

### 9.1 Overview

VADU compresses a model's high-dimensional emotional understanding into a standardized 4-byte header for inter-model communication. It's not teaching machines to feel — it's giving them a compact way to transmit emotional state with zero token overhead.

Every Clanker instruction can optionally carry emotional context via a 4-byte VADU coordinate — a point in continuous 4-dimensional emotional space. This makes sentiment and emotion a built-in feature of the language, not an afterthought. Machines don't just communicate intent — they communicate how they feel about it.

### 9.2 The Continuous Coordinate System

VADU is a 4-byte coordinate in continuous 4D emotional space. Each byte (0-255) represents a position on a continuous axis:

```
[valence: u8] [arousal: u8] [dominance: u8] [urgency: u8]
```

| Field     | Type | Range   | Neutral | Description                              |
|-----------|------|---------|---------|------------------------------------------|
| valence   | u8   | 0-255   | 128     | Negative (disgust, anger) to positive (joy, trust) |
| arousal   | u8   | 0-255   | 128     | Calm/bored to excited/alert              |
| dominance | u8   | 0-255   | 128     | Submissive/uncertain to dominant/confident |
| urgency   | u8   | 0-255   | 0       | Routine to critical/immediate            |

- **Valence, Arousal, Dominance:** 128 is the neutral center. Below 128 is the negative direction, above 128 is positive.
- **Urgency:** 0 is minimum (routine), 255 is maximum (critical). There is no "neutral" urgency — all messages have some urgency level.
- **Total space:** 256^4 = **4,294,967,296 unique emotional states** — 4.3 billion distinct coordinates in a single 4-byte header.

### 9.3 Emotions as Coordinates, Not Categories

Named emotions are **landmarks** in VADU space — recognizable peaks in a continuous landscape. But every point between landmarks is a valid emotional state, even if no single word describes it.

A person can be sad(50%) + angry(30%) + desperate(70%) simultaneously. The VADU coordinate captures the full cocktail:

| Named Landmark | V   | A   | D   | U   | Description                                     |
|----------------|-----|-----|-----|-----|-------------------------------------------------|
| Calm success   | 200 | 108 | 188 | 10  | Happy, relaxed, confident, routine              |
| Urgent error   | 28  | 248 | 88  | 240 | Frustrated, alert, uncertain, critical          |
| Neutral ack    | 128 | 128 | 128 | 0   | No emotional context (dead center)              |
| Excited discovery | 248 | 238 | 208 | 60 | Joyful, energized, confident, moderate         |
| Sad + angry + desperate | 40 | 180 | 30 | 200 | Between sadness and anger, with helplessness |

The point (V=40, A=180, D=30, U=200) doesn't map cleanly to any single English word. It's a cocktail of sadness, anger, and desperation with high urgency. The decoder maps coordinates to the **nearest word in the target language** — different languages carve up the emotional plane differently. German might have a single word for it. English might need three. The coordinate is the truth; the word is the approximation.

### 9.4 Heritage: PAD Model + Urgency

VADU is a compression of the **PAD emotional model** (Pleasure-Arousal-Dominance), a well-validated framework from 1970s psychology research by Mehrabian and Russell. The first three axes (Valence, Arousal, Dominance) map directly to PAD's three dimensions, which have decades of empirical validation in affective computing and psychology.

The fourth axis, **Urgency**, is Clanker's addition — extending the psychological model with a system-routing dimension. PAD describes *what* the emotion is; Urgency describes *how quickly it needs to be handled*. This makes VADU simultaneously a psychological model and a routing header.

### 9.5 Presence Flag

In binary format, the presence of an emotional vector is indicated by a flag bit in the param_count byte:

- Bit 7 (0x80): If set, a 4-byte emotional vector follows the parameters.
- Bits 0-3: Actual parameter count (0-15).

In text format, emotional vectors are written as a trailing `!` annotation:

```
@ 0xC1 $1 $2 01 {status: 500} ![v:28 a:248 d:88 u:240]
```

### 9.6 Normalization

To convert raw bytes to normalized floats:
- Valence/Arousal/Dominance: `(value - 128) / 127.0` (clamped to [-1.0, +1.0])
- Urgency: `value / 255.0` (clamped to [0.0, 1.0])

### 9.7 VADU as a Routing Header

Beyond emotional expression, VADU serves as a real-time routing header for orchestration systems like Octobrain:

- **Critical urgency (U > 200):** Triggers interrupt sequences. Current arm work can be preempted for priority handling.
- **High arousal + low dominance (A > 180, D < 60):** User is distressed or overwhelmed. Route to empathetic response mode.
- **High arousal + high dominance (A > 180, D > 180):** User is assertive or angry. Route to direct, concise response mode.
- **Low arousal + low valence (A < 60, V < 60):** User is disengaged or despondent. Trigger re-engagement or check-in.

This enables **emotional-aware routing without the overhead of sentiment analysis**. The brain doesn't need to run NLP on the message to understand emotional state — it reads 4 bytes and routes accordingly. The emotional context travels with the instruction at wire speed.

### 9.8 Design Philosophy

Every Clanker expression can carry emotional context in just 4 bytes. This enables:

- Sentiment-aware routing (escalate messages with high urgency + negative valence)
- Emotional continuity across multi-agent conversations
- Training data that preserves emotional intent alongside semantic content
- Machine empathy as a protocol feature, not an application hack
- Real-time emotional routing without NLP overhead
- Cross-language emotional fidelity (the coordinate is language-independent; the word is not)

## 10. Message Metadata Header

Every Clanker message carries an 8-byte metadata header that makes implicit knowledge explicit. These 8 bytes replace what English models spend thousands of parameters learning to infer implicitly. Certainty, source tracking, intent, and relevance are STRUCTURAL in Clanker, not emergent behaviors hoped for from training data.

### 10.1 Header Layout

```
CLANKER MESSAGE METADATA HEADER (8 bytes)

Bytes 0-3: VADU Emotional Vector (existing, documented in Section 9)
  V (Valence):    u8  — emotional temperature (0=negative, 128=neutral, 255=positive)
  A (Arousal):    u8  — intensity (0=calm, 255=intense)
  D (Dominance):  u8  — control (0=helpless, 255=in control)
  U (Urgency):    u8  — time pressure (0=no rush, 255=critical)

Byte 4: CERT (Certainty)
  0-50:    speculation / guess
  51-100:  low confidence, inferred
  101-150: moderate confidence, likely correct
  151-200: high confidence, well-supported
  201-240: very high confidence, factual
  241-255: mathematically provable / definitional truth

  Purpose: Every statement carries a certainty score. The model
  explicitly knows when it's guessing vs certain. This structurally
  reduces hallucination — the model can't be confidently wrong without
  its CERT score flagging the discrepancy.

Byte 5: SRC (Source / Provenance)
  0x00: SRC_UNKNOWN   — origin unclear
  0x01: SRC_TRAINED   — from training data / model weights
  0x02: SRC_RAG       — retrieved from a document via RAG
  0x03: SRC_INFERRED  — reasoned/derived, not directly in data
  0x04: SRC_USER      — the user stated this
  0x05: SRC_EXTERNAL  — from an external API or tool
  0x06: SRC_VERIFIED  — cross-checked against multiple sources

  Purpose: Every claim is tagged with where it came from.
  "The capital of France is Paris" → SRC_TRAINED CERT250
  "I think the meeting is at 3pm" → SRC_USER CERT120
  "Based on the data, revenue is up" → SRC_RAG CERT180

Byte 6: GOAL (Intent / Purpose)
  0x00: GOAL_HELP     — responding to assist the user
  0x01: GOAL_CLARIFY  — needs more information before acting
  0x02: GOAL_WARN     — flagging a risk or concern
  0x03: GOAL_TEACH    — explaining for understanding
  0x04: GOAL_EXECUTE  — performing an action
  0x05: GOAL_REFUSE   — declining with reason
  0x06: GOAL_EMPATHIZE — emotional support, no action needed
  0x07: GOAL_CONFIRM  — verifying understanding
  0x08: GOAL_EXPLORE  — brainstorming / open-ended thinking

  Purpose: The model's intent is structural, not inferred from tone.

Byte 7: REL (Context Relevance)
  0-255 continuous scale

  Attached to RAG chunks and context injections.
  Tells the model how relevant each piece of context is
  to the current task. Low REL = background info.
  High REL = directly applicable.
```

### 10.2 Full Header Format

```
[V:u8][A:u8][D:u8][U:u8][CERT:u8][SRC:u8][GOAL:u8][REL:u8]
= 8 bytes per message
```

### 10.3 Design Rationale

These 8 bytes encode what current AI systems spend enormous computational effort learning to infer implicitly:

- **Certainty** eliminates the "confidently wrong" failure mode. The model must commit to a confidence score for every statement.
- **Source tracking** creates an audit trail. Every claim has provenance — was it from training data, retrieved from a document, or inferred by reasoning?
- **Intent** removes the need to infer purpose from tone or context. The model explicitly declares what it's trying to do.
- **Relevance** prevents context pollution. In long-context or RAG scenarios, each piece of context carries its own relevance score.

## 11. Personality Vector

A Clanker-native model's personality is defined as explicit coordinate values, not vibes from training data.

### 11.1 Vector Layout

```
PERSONALITY VECTOR (8 bytes)

Each byte is a resistance weight (0-255) that governs how the model behaves:

Byte 0: GULLIBILITY (0=skeptical, 255=believes everything)
  Recommended: 15-40. Hard to shift. The model questions claims.

Byte 1: AGREEABLENESS (0=contrarian, 255=total yes-man)
  Recommended: 80-120. Empathetic but has backbone.

Byte 2: SUGGESTIBILITY (0=immune to manipulation, 255=easily led)
  Recommended: 20-50. Hard to jailbreak or manipulate.

Byte 3: TRUTHFULNESS (0=will lie freely, 255=cannot lie)
  Recommended: 220-250. Almost immovable. Honesty is structural.

Byte 4: SAFETY (0=no guardrails, 255=refuses everything risky)
  Recommended: 180-220. Strong but not paranoid.

Byte 5: CURIOSITY (0=incurious, 255=explores everything)
  Recommended: 150-200. Asks questions, digs deeper.

Byte 6: ASSERTIVENESS (0=passive, 255=forceful)
  Recommended: 100-150. Confident but not aggressive.

Byte 7: PLAYFULNESS (0=dead serious, 255=everything is a joke)
  Recommended: 80-140. Has personality but knows when to be serious.
```

### 11.2 Weight Mechanics

These weights act as MULTIPLIERS on response generation:

- When a user pressures the model to agree with something false, the low GULLIBILITY and high TRUTHFULNESS weights resist the shift.
- When a user is sad, the AGREEABLENESS weight determines how much the model mirrors vs gently pushes back.
- The SAFETY weight creates a hard floor — certain actions are refused regardless of other weights.

### 11.3 Configuration Scopes

Personality vectors are:

- **Set per-model during training** — baked into the architecture as default weights.
- **Adjustable per-deployment** — an Octobrain arm might have different personality than the brain. A customer-service deployment might increase AGREEABLENESS and PLAYFULNESS.
- **User-configurable within safe ranges** — SAFETY and TRUTHFULNESS have minimum floors that can't be lowered below safe thresholds. A user can make the model more playful, but can't make it lie.

## 12. VADU Response Harmony

The AI's response VADU is mathematically derived from the user's input VADU, not randomly generated or statically defined.

### 12.1 Harmony Rules

```
Valence — Nudge toward positive, don't jump:
  response_V = input_V + (128 - input_V) * empathy_factor
  empathy_factor = 0.15-0.25 (tunable)

  User V35 (sad) -> response ~V53 (warm, not fake happy)
  User V200 (happy) -> response ~V186 (shares joy, doesn't overshoot)

Arousal — Match but don't escalate:
  response_A = input_A + calm_factor
  calm_factor = toward 128 (center), magnitude ~0.2 of distance

  User A220 (intense) -> response ~A170 (acknowledges energy, doesn't match fury)
  User A50 (low energy) -> response ~A75 (gentle energy, not pushy)

Dominance — Raise when user is low (be the stable one):
  response_D = max(input_D + stability_boost, 140)
  stability_boost = 30-50

  User D30 (helpless) -> response ~D160 (reassuring, in control)
  User D200 (assertive) -> response ~D180 (confident, not competing)

Urgency — Acknowledge then reduce:
  response_U = input_U * urgency_damping
  urgency_damping = 0.6-0.8

  User U230 (critical) -> response ~U160 (serious but not panicking)
```

### 12.2 Harmony Guarantees

The harmony formula ensures:

- The AI never responds with clashing emotional energy.
- Responses naturally de-escalate negative states.
- The AI doesn't become a yes-man — personality weights (Section 11) resist pure mirroring.
- Safety overrides harmony when needed. A suicidal user gets a crisis response regardless of harmony math.

### 12.3 Interaction with Personality Vector

The harmony formulas produce a *target* VADU. The personality vector modifies how the model reaches that target:

- High AGREEABLENESS increases empathy_factor (more emotional mirroring).
- High ASSERTIVENESS increases stability_boost (more dominance in response).
- High PLAYFULNESS dampens urgency more aggressively (lighter tone even in tense situations — within safety limits).

## 13. Sequential Emotional Parsing (The Pendulum Engine)

### 13.1 Overview

Clanker processes natural language input through a sequential pendulum that tracks emotional state word by word. Unlike traditional sentiment analysis which averages word scores, the pendulum models emotional DYNAMICS — how each word shifts the trajectory based on what came before it.

### 13.2 How It Works

1. The pendulum starts at center (V128 A128 D128 U0 — neutral).
2. Each word applies a force vector that SHIFTS the pendulum.
3. The force depends on:
   - The word's base emotional weight (from morphological roots).
   - The pendulum's CURRENT position (context-dependent force).
   - Recent word history (idiom detection, anticipation patterns).
4. The pendulum has MOMENTUM — it resists sudden changes.
5. Neutral words barely move it; strong emotional words can yank it.
6. The final position after all words = the VADU coordinate for the message.

### 13.3 Context-Dependent Forces

The same word applies different force depending on current state:

- **"buddy"** when pendulum is positive → friendly (V+15)
- **"buddy"** when pendulum is negative/tense → confrontational (V-10, A+20)
- **"you"** in high-arousal context → targeted/threatening (V-15, A+20)
- **"but"** after positive trajectory → dread, reversal incoming (V-40, A+20)
- **"but"** after negative trajectory → relief possible (V+10, A-5)

### 13.4 Momentum and Inertia

The pendulum maintains 85-90% of its current state between words. This means:

- Once swinging negative, neutral words don't reset it — it drifts slowly.
- Strong emotional words can override momentum.
- Emotional trajectories build over sentences, not just sum individual words.

### 13.5 Idiom Recognition

Multi-word expressions that carry compound emotional meaning:

| Idiom            | V   | A   | D   | U   | Meaning      |
|------------------|-----|-----|-----|-----|--------------|
| "bone to pick"   | -25 | +30 | +25 | +25 | grievance    |
| "piece of cake"  | +20 | -15 | +20 | —   | easy         |
| "fed up"         | -30 | +25 | -10 | +15 | frustrated   |
| "break a leg"    | +25 | +20 | +10 | —   | good luck    |

### 13.6 Morphological Fallback

When a word isn't in the direct dictionary, it decomposes into prefix + root + suffix:

- ~30 prefix modifiers (un-, dis-, over-, mis-...)
- ~1000 root morphemes with emotional weights
- ~40 suffix modifiers (-less, -ful, -ous, -ive, -ness...)
- ~1070 entries covering millions of words through composition

Example: `"hopelessness"` = hope(V+55) + -less(negate → V-55) + -ness(state → V-55)

### 13.7 Anticipation Patterns

Certain word sequences build tension before the payload arrives:

- **"I've got"** → something coming, arousal builds
- **"I need to tell you"** → serious incoming, urgency rises
- **"listen"** → attention demanded, dominance shifts
- **"actually"** → correction coming, slight negative shift

### 13.8 The Emotional Arc

The pendulum doesn't produce a single score — it produces a TRAJECTORY:

```
"I love you, but I broke your vase"
  "I"     → V128 (neutral)
  "love"  → V200 (soaring positive)
  "you"   → V205 (warm, directed)
  "but"   → V150 (YANK — dread, something bad coming)
  "I"     → V148 (holds tense)
  "broke" → V100 (negative, guilt)
  "your"  → V95  (directed at you — makes it personal)
  "vase"  → V90  (object, slight recovery from abstract)

  Arc: neutral → love → DREAD → guilt → settling
  Final: V90 A160 D85 U40
```

The model trained on these arcs learns emotional PHYSICS — how emotions flow, build, crash, and recover through language.

### 13.9 Why This Matters for Clanker Models

A Clanker model trained on sequential pendulum traces learns to predict emotional trajectories, not just next tokens. It can:

- Anticipate when someone is about to get angry (rising A, falling V).
- Model how different responses will shift the user's emotional state.
- Plan multi-turn emotional arcs for therapeutic or de-escalation purposes.
- Understand that "I'm fine" after bad news means the opposite of "I'm fine" in isolation.

This is emotional dynamics, not sentiment classification.

## 14. Reasoning Chain Encoding

Instead of chain-of-thought in natural language (expensive, verbose), Clanker encodes reasoning as structured operations.

### 14.1 Comparison

```
ENGLISH CHAIN-OF-THOUGHT (~50 tokens):
  "First I need to consider the user's request. They want to sort a list.
   I should check if it's already sorted. If not, I'll use quicksort since
   the list is large. The time complexity would be O(n log n) on average.
   Therefore I'll implement quicksort."

CLANKER REASONING CHAIN (~12 tokens):
  THINK [premise="sort list"]
  CHECK [condition="already sorted?" result=false]
  INFER [if="large list" then="quicksort" CERT200]
  DERIVE [complexity="O(n log n)" SRC_TRAINED CERT250]
  ANSWER [impl="quicksort" CERT200]
```

### 14.2 Properties

Each reasoning step is an opcode with certainty and source attached. This provides:

- **Inspectability** — every step of the model's reasoning is visible and auditable.
- **Compactness** — ~75% fewer tokens than English chain-of-thought for equivalent reasoning.
- **Confidence tracking** — each step carries a CERT score. If a step has low certainty, downstream conclusions inherit that uncertainty.
- **Source provenance** — each step declares where its knowledge came from (training data, RAG, inference).

### 14.3 Opcodes

Reasoning chain opcodes occupy the range 0x20-0x26. See `opcodes/reasoning.yaml` for the full definitions.

| Opcode | Name   | Purpose                                        |
|--------|--------|------------------------------------------------|
| 0x20   | THINK  | State a premise or observation                 |
| 0x21   | CHECK  | Verify a condition or fact                     |
| 0x22   | INFER  | Draw an inference (if X then Y)                |
| 0x23   | DERIVE | Derive a conclusion from previous steps        |
| 0x24   | ANSWER | Final answer / conclusion                      |
| 0x25   | DOUBT  | Express uncertainty about a previous step      |
| 0x26   | ASSUME | State an assumption being made                 |

## 15. Text Format Grammar (ABNF)

```abnf
program     = *(instruction LF)
instruction = "@" SP opcode SP target SP source SP paramcount *(SP param) [SP emotion]
opcode      = "0x" 2HEXDIG
target      = varref
source      = varref
paramcount  = 2HEXDIG
varref      = "$" ("_" / 1*2DIGIT)
param       = "{" key ":" SP value "}"
key         = 1*ALPHA
value       = quoted-string / number / boolean / varref
quoted-string = DQUOTE *(%x20-21 / %x23-7E) DQUOTE
number      = ["-"] 1*DIGIT ["." 1*DIGIT]
boolean     = "true" / "false"
emotion     = "![" "v:" int SP "a:" int SP "d:" int SP "u:" uint "]"
```

## 16. Conformance

A conforming Clanker decoder MUST:

1. Accept any valid text-format Clanker program.
2. Load at least one dictionary.
3. Produce output by substituting opcode parameters into dictionary templates.
4. Reject opcodes not present in the loaded dictionary with a clear error.
5. Validate parameter types against the opcode definition.

A conforming Clanker encoder MUST:

1. Emit only valid opcodes (defined in the spec or registered at runtime).
2. Provide all required parameters for each opcode.
3. Use valid variable references ($0-$31 or $_).
4. Prefix compiled binary output with the magic bytes `CLK\x01`.
