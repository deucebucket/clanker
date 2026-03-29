# Opcode Definitions

Each YAML file in this directory defines opcodes for a specific range of the Clank opcode space.

## Files

| File            | Range       | Category           |
|----------------|-------------|-------------------|
| core.yaml      | 0x00-0x1F   | Flow control, lifecycle, fundamentals |
| hardware.yaml  | 0xA0-0xAF   | Device control (from delphinOS) |
| web.yaml       | 0xC0-0xCF   | HTTP, API, networking |
| data.yaml      | 0xD0-0xDF   | Data transformation, query, storage |
| logic.yaml     | 0xE0-0xEF   | Branching, matching, loops, error handling |

## Format

Every opcode entry follows this structure:

```yaml
opcodes:
  0xNN:
    name: OPCODE_NAME
    description: "What this opcode does"
    params:
      - {name: param_name, type: param_type, required: true}
      - {name: optional_param, type: param_type, optional: true}
```

## Rules

1. **Opcodes are forever.** Once ratified, the meaning and code of an opcode never changes.
2. **Names are UPPER_SNAKE_CASE** and must be unique across all files.
3. **Descriptions** should be one clear sentence.
4. **Parameter types** must be defined in `rules/types.yaml`.
5. **Reserved ranges** (0x20-0x9F) should not be used until allocated by a spec update.
6. **User space** (0xF0-0xFF) is for runtime-registered custom opcodes only.

## Adding New Opcodes

See [docs/adding-opcodes.md](../docs/adding-opcodes.md) for the proposal process.
