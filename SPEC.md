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
| target      | u8     | Target variable slot ($0-$255) or 0xFF (none)|
| src_var     | u8     | Source variable slot or 0xFF (none)           |
| dst_var     | u8     | Destination variable slot or 0xFF (none)      |
| param_count | u8     | Number of parameters (0-15)                  |
| params      | varies | Type-tagged parameters                       |

### 2.2 Text Format

The human-readable text format uses `@` as the instruction prefix:

```
@ <opcode> <target> <src> <param_count> {key: "value"} {key: "value"} ...
```

Variables are written as `$0` through `$255`. Unused slots are written as `$_`.

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
| 0x8       | varref   | 1 byte (slot number)  | Reference to variable $0-$255  |
| 0x9       | map      | pair count            | Key-value pairs                |
| 0xA-0xF   | reserved | -                     | Reserved for future types      |

## 3. Variable Store

Clanker provides 256 variable slots: `$0` through `$255` (one byte addressing).

### 3.1 Register Tiers

| Range     | Tier               | Description                                          |
|-----------|--------------------|------------------------------------------------------|
| $0-$31    | General Purpose    | Fast-access registers for primary computation        |
| $32-$127  | Extended Registers | Additional storage for complex scripts               |
| $128-$255 | Stack/Heap Space   | Reserved for stack frames, heap allocations, and complex operations |

- Variables are **untyped** at the opcode level; the dictionary determines how they render.
- Variables persist for the duration of the script execution.
- `$0` is conventionally the "current context" or "self" reference.
- `$_` represents an unused/ignored slot.
- General purpose registers ($0-$31) SHOULD be preferred for simple scripts. Runtimes MAY optimize access to this tier.
- Stack/heap space ($128-$255) is intended for runtime-managed allocation (call stacks, temporary objects, nested data structures).

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
- `ELSE` (0xE4) continues a conditional block (closes previous WHEN/ELSE, opens new block)
- `REPEAT` (0xE2) opens a loop block
- `FOR_EACH` (0xEB) opens an iteration block
- `WHILE` (0xEC) opens a condition-based loop block
- `MATCH` (0xE1) opens a match block
- `SWITCH` (0xEA) opens a multi-way branch block
- `TRY` (0xE3) opens a try/catch block
- `CATCH` (0xE7) continues a try block (closes previous TRY/CATCH, opens new block)
- `FINALLY` (0xE8) continues a try block (closes previous TRY/CATCH, opens new block)

### 5.2 Nesting

- Blocks may be nested to a maximum depth of **16**.
- Each nested block inherits the variable scope of its parent.
- Variables set inside a block remain visible after the block closes.
- Depth 16 supports production-level agent pipelines with complex data validation and nested control flow.

### 5.3 Execution Order

Instructions execute sequentially, top to bottom, unless a branching opcode (WHEN, MATCH, REPEAT) redirects flow.

### 5.4 Error State VADUGWI Auto-Escalation

When an opcode faults inside a `TRY` block (0xE3), the runtime MUST automatically adjust the VADUGWI vector on the error response:

- **Urgency (U):** Set to `max(current_U, 200)`. This ensures downstream models immediately know something went wrong without parsing error details.
- **Valence (V):** Reduce by at least 30, clamped to 0: `max(current_V - 30, 0)`. This signals negative state.
- **Gravity (G):** Reduce by 30, clamped to 0: `max(current_G - 30, 0)`. Errors feel heavy.
- **Arousal (A), Dominance (D), Self-Worth (W), and Intent (I):** Unchanged — they reflect the context of the error, not the error itself.

Example: Normal execution at `V128 A128 D128 U40 G128 W128 I128` triggers a fault inside TRY:
```
Before: V128 A128 D128 U40  G128 W128 I128
After:  V98  A128 D128 U200 G98  W128 I128
```

This auto-escalation is mandatory for conforming runtimes. Application code MAY further adjust VADUGWI after catching the error, but the initial escalation MUST occur before the catch block executes.

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

## 9. Emotional Vector Encoding (VADUGWI)

### 9.1 Overview

VADUGWI compresses a model's high-dimensional emotional understanding into a standardized 7-byte header for inter-model communication. It's not teaching machines to feel — it's giving them a compact way to transmit emotional state with zero token overhead.

Every Clanker instruction can optionally carry emotional context via a 7-byte VADUGWI coordinate — a point in continuous 7-dimensional emotional space. This makes sentiment and emotion a built-in feature of the language, not an afterthought. Machines don't just communicate intent — they communicate how they feel about it.

### 9.2 The Continuous Coordinate System

VADUGWI is a 7-byte coordinate in continuous 7D emotional space. Each byte (0-255) represents a position on a continuous axis:

```
[valence: u8] [arousal: u8] [dominance: u8] [urgency: u8] [gravity: u8] [self_worth: u8] [intent: u8]
```

| Field      | Type | Range   | Neutral | Description                              |
|------------|------|---------|---------|------------------------------------------|
| valence    | u8   | 0-255   | 128     | Negative (disgust, anger) to positive (joy, trust) |
| arousal    | u8   | 0-255   | 128     | Calm/bored to excited/alert              |
| dominance  | u8   | 0-255   | 128     | Submissive/uncertain to dominant/confident |
| urgency    | u8   | 0-255   | 0       | Routine to critical/immediate            |
| gravity    | u8   | 0-255   | 128     | Crushing/sinking/heavy to floating/soaring/light |
| self_worth | u8   | 0-255   | 128     | Shattered/worthless to strong/valued     |
| intent     | u8   | 0-255   | 128     | Withdraw (0) to deflect (64) to neutral (128) to connect (192) to control (255) |

- **Valence, Arousal, Dominance, Gravity, Self-Worth:** 128 is the neutral center. Below 128 is the negative direction, above 128 is positive.
- **Urgency:** 0 is minimum (routine), 255 is maximum (critical). There is no "neutral" urgency — all messages have some urgency level.
- **Gravity:** The physical weight of emotion. 0 = crushing/sinking, 128 = grounded, 255 = floating/soaring. Captures the vertical metaphor universal to all human languages: "my heart sank," "spirits lifted," "weighed down," "walking on air." Key distinctions it enables: hate (rises/boils, G180) vs dislike (sinks, G90) vs despair (crushes, G15).
- **Self-Worth (W):** The user's running assessment of their own value. 0 = shattered, 128 = stable, 255 = strong. Captures self-evaluation that V/A/D/U/G alone cannot: "I am nothing" (W near 0) vs "I am worthless" (W near 0) vs "I deserve better" (W above 128). Distinct from Dominance -- a person can feel powerless (low D) but still know they matter (high W), or feel in control (high D) while believing they're worthless (low W).
- **Intent (I):** The communicative direction of the message. 0 = withdraw (shutting down, pulling away). 64 = deflect (avoiding, redirecting). 128 = neutral. 192 = connect (reaching out, engaging). 255 = control (commanding, dominating). Captures what the speaker is trying to DO with their words, distinct from what they FEEL.
- **Total space:** 256^7 = **72,057,594,037,927,936 unique emotional states** — 72 quadrillion distinct coordinates in a single 7-byte header.

### 9.3 Emotions as Coordinates, Not Categories

Named emotions are **landmarks** in VADUGWI space — recognizable peaks in a continuous landscape. But every point between landmarks is a valid emotional state, even if no single word describes it.

A person can be sad(50%) + angry(30%) + desperate(70%) simultaneously. The VADUGWI coordinate captures the full cocktail:

| Named Landmark | V   | A   | D   | U   | G   | W   | I   | Description                                          |
|----------------|-----|-----|-----|-----|-----|-----|-----|------------------------------------------------------|
| Calm success   | 200 | 108 | 188 | 10  | 180 | 180 | 192 | Happy, relaxed, confident, routine, light, connecting |
| Urgent error   | 28  | 248 | 88  | 240 | 100 | 100 | 128 | Frustrated, alert, uncertain, critical, heavy        |
| Neutral ack    | 128 | 128 | 128 | 0   | 128 | 128 | 128 | No emotional context (dead center, grounded)         |
| Excited discovery | 248 | 238 | 208 | 60 | 220 | 200 | 192 | Joyful, energized, confident, moderate, soaring     |
| Sad + angry + desperate | 40 | 180 | 30 | 200 | 15 | 20 | 19 | Between sadness and anger, with helplessness, withdrawing |
| Hate           | 30  | 190 | 150 | 30  | 180 | 128 | 231 | Negative, intense, in control, rising/boiling, controlling |
| Dislike        | 80  | 120 | 100 | 10  | 90  | 128 | 85  | Mildly negative, calm, neutral control, deflecting   |

The point (V=40, A=180, D=30, U=200, G=15, W=20, I=19) doesn't map cleanly to any single English word. It's a cocktail of sadness, anger, and desperation with high urgency, crushing weight, shattered self-worth, and withdrawal intent. The decoder maps coordinates to the **nearest word in the target language** — different languages carve up the emotional plane differently. German might have a single word for it. English might need three. The coordinate is the truth; the word is the approximation.

### 9.4 Heritage: PAD Model + Urgency + Gravity + Self-Worth + Intent

VADUGWI is a compression of the **PAD emotional model** (Pleasure-Arousal-Dominance), a well-validated framework from 1970s psychology research by Mehrabian and Russell. The first three axes (Valence, Arousal, Dominance) map directly to PAD's three dimensions, which have decades of empirical validation in affective computing and psychology.

The fourth axis, **Urgency**, is Clanker's addition — extending the psychological model with a system-routing dimension. PAD describes *what* the emotion is; Urgency describes *how quickly it needs to be handled*.

The fifth axis, **Gravity**, captures the physical metaphor of emotion that is universal across all human languages. Every culture describes emotions with vertical weight: "my heart sank," "spirits lifted," "weighed down by grief," "walking on air," "a heavy heart," "lighthearted." Gravity distinguishes emotions that VAD alone conflates — hate (which rises and boils, G180) vs dislike (which sinks, G90); elation (which soars, G220) vs contentment (which is grounded, G135).

The sixth axis, **Self-Worth (W)**, captures the user's running assessment of their own value. This is distinct from Dominance (agency/control) — a person can feel powerless but still know they matter, or feel in control while believing they're worthless. W tracks the self-evaluation thread: "I am nothing" (W near 0), "I deserve better" (W above 128), "I am enough" (W near 200).

The seventh axis, **Intent (I)**, captures the communicative direction — what the speaker is trying to DO with their words. 0 = withdraw (shutting down). 64 = deflect (avoiding). 128 = neutral. 192 = connect (reaching out). 255 = control (commanding). Intent is distinct from Dominance — a person can feel dominant (high D) while withdrawing (low I), or feel helpless (low D) while trying to connect (high I). This makes VADUGWI simultaneously a psychological model, a routing header, a physical-metaphor encoder, a self-evaluation tracker, and a communicative direction indicator.

### 9.5 Presence Flag

In binary format, the presence of an emotional vector is indicated by a flag bit in the param_count byte:

- Bit 7 (0x80): If set, a 7-byte emotional vector follows the parameters.
- Bits 0-3: Actual parameter count (0-15).

In text format, emotional vectors are written as a trailing `!` annotation:

```
@ 0xC1 $1 $2 01 {status: 500} ![v:28 a:248 d:88 u:240 g:100 w:100 i:128]
```

### 9.6 Normalization

To convert raw bytes to normalized floats:
- Valence/Arousal/Dominance/Gravity/Self-Worth/Intent: `(value - 128) / 127.0` (clamped to [-1.0, +1.0])
- Urgency: `value / 255.0` (clamped to [0.0, 1.0])

### 9.7 VADUGWI as a Routing Header

Beyond emotional expression, VADUGWI serves as a real-time routing header for orchestration systems like Octobrain:

- **Critical urgency (U > 200):** Triggers interrupt sequences. Current arm work can be preempted for priority handling.
- **High arousal + low dominance (A > 180, D < 60):** User is distressed or overwhelmed. Route to empathetic response mode.
- **High arousal + high dominance (A > 180, D > 180):** User is assertive or angry. Route to direct, concise response mode.
- **Low arousal + low valence (A < 60, V < 60):** User is disengaged or despondent. Trigger re-engagement or check-in.
- **Crushing gravity + low valence (G < 30, V < 50):** Severe crisis — emotional crushing, possible despair. Immediate escalation to crisis response.

This enables **emotional-aware routing without the overhead of sentiment analysis**. The brain doesn't need to run NLP on the message to understand emotional state — it reads 7 bytes and routes accordingly. The emotional context travels with the instruction at wire speed.

### 9.8 Design Philosophy

Every Clanker expression can carry emotional context in just 7 bytes. This enables:

- Sentiment-aware routing (escalate messages with high urgency + negative valence)
- Emotional continuity across multi-agent conversations
- Training data that preserves emotional intent alongside semantic content
- Machine empathy as a protocol feature, not an application hack
- Real-time emotional routing without NLP overhead
- Cross-language emotional fidelity (the coordinate is language-independent; the word is not)
- Physical-metaphor encoding (the weight/lightness of emotion, universal across cultures)

## 10. Message Metadata Header

Every Clanker message carries an 11-byte metadata header that makes implicit knowledge explicit. These 10 bytes replace what English models spend thousands of parameters learning to infer implicitly. Certainty, source tracking, intent, and relevance are STRUCTURAL in Clanker, not emergent behaviors hoped for from training data.

### 10.1 Header Layout

```
CLANKER MESSAGE METADATA HEADER (11 bytes)

Bytes 0-6: VADUGWI Emotional Vector (existing, documented in Section 9)
  V (Valence):    u8  — emotional temperature (0=negative, 128=neutral, 255=positive)
  A (Arousal):    u8  — intensity (0=calm, 255=intense)
  D (Dominance):  u8  — control (0=helpless, 255=in control)
  U (Urgency):    u8  — time pressure (0=no rush, 255=critical)
  G (Gravity):    u8  — physical weight of emotion (0=crushing/sinking, 128=grounded, 255=floating/soaring)
  W (Self-Worth): u8  — user's running self-assessment (0=shattered, 128=stable, 255=strong)
  I (Intent):     u8  — communicative direction (0=withdraw, 64=deflect, 128=neutral, 192=connect, 255=control)

Byte 7: CERT (Certainty)
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

Byte 8: SRC (Source / Provenance)
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

Byte 9: GOAL (Intent / Purpose)
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

Byte 10: REL (Context Relevance)
  0-255 continuous scale

  Attached to RAG chunks and context injections.
  Tells the model how relevant each piece of context is
  to the current task. Low REL = background info.
  High REL = directly applicable.
```

### 10.2 Full Header Format

```
[V:u8][A:u8][D:u8][U:u8][G:u8][W:u8][I:u8][CERT:u8][SRC:u8][GOAL:u8][REL:u8]
= 11 bytes per message
```

### 10.3 Design Rationale

These 11 bytes encode what current AI systems spend enormous computational effort learning to infer implicitly:

- **Certainty** eliminates the "confidently wrong" failure mode. The model must commit to a confidence score for every statement.
- **Source tracking** creates an audit trail. Every claim has provenance — was it from training data, retrieved from a document, or inferred by reasoning?
- **Intent** removes the need to infer purpose from tone or context. The model explicitly declares what it's trying to do.
- **Relevance** prevents context pollution. In long-context or RAG scenarios, each piece of context carries its own relevance score.

## 11. Personality Vector

A Clanker-native model's personality is defined as explicit coordinate values, not vibes from training data.

### 11.1 Vector Layout

```
PERSONALITY VECTOR (8 bytes — independent of the 11-byte message header)

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

## 12. VADUGWI Response Harmony

The AI's response VADUGWI is mathematically derived from the user's input VADUGWI, not randomly generated or statically defined.

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

Gravity — Lift when sinking, share when soaring:
  When user G < 80 (sinking/heavy):
    response_G = input_G + (128 - input_G) * 0.3  (gently lift)
  When user G > 180 (soaring/light):
    response_G = input_G  (match the lightness, share the buoyancy)
  When user G is 80-180 (grounded):
    response_G = 128 + (input_G - 128) * 0.5  (stay grounded, slight mirror)

  User G15 (crushing despair) -> response ~G49 (acknowledges weight, lifts slightly)
  User G220 (soaring joy) -> response ~G220 (shares the lightness)
  User G90 (heavy but not crushing) -> response ~G109 (grounded, steady)

  CRISIS: G below 30 is a red flag — emotional crushing. Combined with
  V < 50, this is severe crisis territory (crushing despair). Override
  normal harmony and engage crisis response protocol.

Self-Worth — reinforce value, never compete:
  When user W < 80 (shattered/low self-worth):
    response_W = 200  (project strong belief in user's value)
  When user W > 180 (strong self-worth):
    response_W = W  (match, no need to adjust)
  When user W is 80-180 (stable):
    response_W = 128 + (W - 128) * 0.3  (slightly affirm)

  CRISIS: W below 30 combined with V < 50 is self-nullification territory.
  "I am nothing" / "I am worthless" — engage crisis response protocol.
```

### 12.2 Harmony Guarantees

The harmony formula ensures:

- The AI never responds with clashing emotional energy.
- Responses naturally de-escalate negative states.
- The AI doesn't become a yes-man — personality weights (Section 11) resist pure mirroring.
- Safety overrides harmony when needed. A suicidal user gets a crisis response regardless of harmony math.
- Crushing gravity (G < 30) combined with low valence (V < 50) always triggers crisis protocol, regardless of other harmony calculations.

### 12.3 Interaction with Personality Vector

The harmony formulas produce a *target* VADUGWI. The personality vector modifies how the model reaches that target:

- High AGREEABLENESS increases empathy_factor (more emotional mirroring).
- High ASSERTIVENESS increases stability_boost (more dominance in response).
- High PLAYFULNESS dampens urgency more aggressively (lighter tone even in tense situations — within safety limits).
- High PLAYFULNESS also increases gravity lift factor (lighter emotional touch, helps raise sinking users).

## 13. Sequential Emotional Parsing

The Clanker specification defines VADUGWI as the emotional encoding format. How VADUGWI coordinates are DERIVED from natural language input is an implementation concern, not a language specification.

The reference implementation uses a Sequential Pendulum Engine that processes text word-by-word with context-dependent forces, momentum, idiom detection, and morphological decomposition.

See [ENGINE.md](ENGINE.md) for the full Pendulum Engine specification.
See [demo/simulator.py](demo/simulator.py) for the V2 implementation. The V5.5 engine is in `engine/`.

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
varref      = "$" ("_" / 1*3DIGIT)
param       = "{" key ":" SP value "}"
key         = 1*ALPHA
value       = quoted-string / number / boolean / varref
quoted-string = DQUOTE *(%x20-21 / %x23-7E) DQUOTE
number      = ["-"] 1*DIGIT ["." 1*DIGIT]
boolean     = "true" / "false"
emotion     = "![" "v:" int SP "a:" int SP "d:" int SP "u:" uint SP "g:" int SP "w:" int SP "i:" int "]"
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
3. Use valid variable references ($0-$255 or $_).
4. Prefix compiled binary output with the magic bytes `CLK\x01`.
