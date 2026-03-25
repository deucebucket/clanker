# Phin-Lang Specification v0.1

## 1. Overview

Phin is a bytecode-style intermediate representation for structured communication between AI systems. A Phin program is a sequence of instructions. Each instruction is an opcode with optional target, source, destination, and parameters. Phin programs can be decoded to any language (human or machine) via dictionary lookup.

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

Phin provides 32 variable slots: `$0` through `$31`.

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

## 8. Text Format Grammar (ABNF)

```abnf
program     = *(instruction LF)
instruction = "@" SP opcode SP target SP source SP paramcount *(SP param)
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
```

## 9. Conformance

A conforming Phin decoder MUST:

1. Accept any valid text-format Phin program.
2. Load at least one dictionary.
3. Produce output by substituting opcode parameters into dictionary templates.
4. Reject opcodes not present in the loaded dictionary with a clear error.
5. Validate parameter types against the opcode definition.

A conforming Phin encoder MUST:

1. Emit only valid opcodes (defined in the spec or registered at runtime).
2. Provide all required parameters for each opcode.
3. Use valid variable references ($0-$31 or $_).
