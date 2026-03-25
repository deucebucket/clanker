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
| 0x20-0x9F   | Reserved          | Reserved for future standard opcodes  |
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

## 10. Text Format Grammar (ABNF)

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

## 11. Conformance

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
