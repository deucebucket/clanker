# Adding New Opcodes

This guide explains how to propose and register new opcodes in Phin-Lang.

## Guiding Principles

1. **Opcodes are forever.** Once ratified, an opcode's code and meaning never change. Think carefully before proposing.
2. **One concept, one opcode.** Each opcode should represent exactly one atomic operation.
3. **Parameters vary, opcodes don't.** If two operations differ only in their data, they should be the same opcode with different parameters — not two opcodes.
4. **Bias toward fewer opcodes.** The total opcode space is 256. Use it wisely.

## Opcode Ranges

| Range     | Owner       | Status       |
|-----------|------------|--------------|
| 0x00-0x1F | Core       | Partially allocated |
| 0x20-0x9F | Reserved   | Unallocated  |
| 0xA0-0xAF | Hardware   | Partially allocated |
| 0xB0-0xBF | Ext. HW    | Unallocated  |
| 0xC0-0xCF | Web        | Partially allocated |
| 0xD0-0xDF | Data       | Partially allocated |
| 0xE0-0xEF | Logic      | Partially allocated |
| 0xF0-0xFF | User Space | Runtime only |

## Proposing a Standard Opcode

### Step 1: Check if it already exists

Read through the existing opcode YAML files in `opcodes/`. Your proposed operation might already be covered by an existing opcode with different parameters.

### Step 2: Open an issue

Create a GitHub issue with the label `spec` and include:

- **Opcode name** (UPPER_SNAKE_CASE)
- **Proposed code** (hex, within the appropriate range)
- **Description** (one clear sentence)
- **Parameters** (name, type, required/optional for each)
- **Rationale** (why this needs its own opcode)
- **Example** (at least one Phin instruction using this opcode)
- **Dictionary samples** (how it would render in English and at least one code language)

Example issue:

```
Title: Opcode proposal: CACHE (0xC8)

Description: Cache a response or computed value with a TTL.

Params:
  - key: str, required — Cache key
  - ttl: duration, required — Time to live
  - strategy: str, optional — Cache strategy (lru, fifo, ttl)

Rationale: Caching is a fundamental web operation. Currently requires
combining STORE with application-specific logic. A dedicated opcode
makes caching intent explicit.

Example:
  @ 0xC8 $0 $_ 02 {key: "user_123"} {ttl: "5m"}

English: cache "user_123" for 5m
Python:  cache.set("user_123", data, ttl=300)
```

### Step 3: Discussion

The community reviews the proposal. Key questions:

- Can this be expressed with existing opcodes?
- Is the scope too broad or too narrow?
- Are the parameter types correct?
- Does the proposed code conflict with anything?

### Step 4: Ratification

Once consensus is reached:

1. The opcode is added to the appropriate YAML file in `opcodes/`.
2. All existing dictionaries should add a translation (or at minimum, English).
3. The spec version is bumped (minor version).
4. The opcode is **permanent** — it cannot be removed or changed.

## Runtime Registration (User Space)

For project-specific or experimental opcodes, use the REGISTER opcode (0x0E) to define custom opcodes in the user space range (0xF0-0xFF):

```
@ 0x0E $_ $_ 03 {opcode: 0xF0} {name: "MY_CUSTOM_OP"} {params: [{name: "arg1", type: "str"}]}
```

### Rules for User Space

- Only 0xF0-0xFF are available (16 slots).
- Registered opcodes exist only for the current script session.
- They are not persisted or shared unless explicitly saved.
- If a user-space opcode proves widely useful, propose it as a standard opcode.

## Checklist for Opcode Authors

- [ ] Name is UPPER_SNAKE_CASE and unique across all opcode files
- [ ] Code is within the correct range for the category
- [ ] Code does not conflict with any existing opcode
- [ ] Description is one clear sentence
- [ ] All parameters have name, type, and required/optional
- [ ] Parameter types are defined in `rules/types.yaml`
- [ ] At least one example Phin instruction is provided
- [ ] English dictionary entry is provided
- [ ] At least one code dictionary entry is provided
- [ ] Rationale explains why a new opcode is needed
