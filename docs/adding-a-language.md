# Adding a Language Dictionary

This guide walks through contributing a new dictionary to Clanker-Lang. A dictionary translates Clanker opcodes to a specific human or programming language.

## Overview

Adding a new language requires exactly one file: a YAML dictionary. No code changes needed.

## Step 1: Choose Your Language Type

| Type   | Directory            | Examples                 |
|--------|---------------------|--------------------------|
| Human  | `dictionaries/human/` | en.yaml, zh.yaml         |
| Code   | `dictionaries/code/`  | python.yaml, rust.yaml   |
| Other  | `dictionaries/other/` | pseudocode.yaml          |

## Step 2: Create the File

Name the file using the ISO 639-1 code for human languages (e.g., `es.yaml` for Spanish, `fa.yaml` for Farsi) or the language name for code languages (e.g., `go.yaml`, `cpp.yaml`).

## Step 3: Add the Header

Every dictionary must start with these fields:

```yaml
language: es          # ISO 639-1 code or language name
name: Spanish         # Human-readable name
spec_version: "0.1"  # Clanker spec version this targets
dictionary_version: "1.0"
kind: human           # "human", "code", or "other"
```

## Step 4: Add Opcode Translations

The `opcodes` section maps opcode hex codes to translation entries.

### For Human Languages

```yaml
opcodes:
  0x00:
    template: "hecho"
    with_param: "hecho: \"{message}\""
  0x01:
    template: "esperar {duration}"
  0x02:
    template: "establecer {name} como {value}"
```

- `template` — The default rendering. Use `{param_name}` for parameter placeholders.
- `with_param` — Alternate rendering when an optional parameter is present.

### For Code Languages

```yaml
opcodes:
  0x00:
    template: "pass"
    with_param: "print(\"{message}\")"
  0x01:
    template: "time.sleep({duration})"
    imports: ["import time"]
```

- `template` — A code snippet with placeholders.
- `imports` — Any required imports for this opcode.
- `block_open` / `block_close` — For opcodes that open/close blocks.

## Step 5: Cover All Defined Opcodes

Your dictionary should include entries for ALL opcodes defined in the `opcodes/` YAML files. Check these files for the current list:

- `opcodes/core.yaml` (0x00-0x15)
- `opcodes/hardware.yaml` (0xA0-0xAB)
- `opcodes/web.yaml` (0xC0-0xC7)
- `opcodes/data.yaml` (0xD0-0xD4)
- `opcodes/logic.yaml` (0xE0-0xE3)

You can start with a subset and expand over time, but aim for completeness.

## Step 6: Test Your Dictionary

Use the reference decoder to verify your translations render correctly:

```python
from clanker_decoder import decode, DictionaryLoader

loader = DictionaryLoader()
script = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/hello"}\n@ 0x00'
print(decode(script, "es", loader=loader))
```

You can also run the test suite:

```bash
cd decoder/python
pip install -e ".[dev]"
pytest tests/ -v
```

## Step 7: Submit

1. Fork the repository.
2. Add your dictionary file.
3. Add tests for your language in `decoder/python/tests/`.
4. Open a pull request.

## Template Placeholders Reference

Placeholders correspond to the `name` field of each opcode's `params` in the opcode YAML files. For example, if `opcodes/web.yaml` defines:

```yaml
0xC0:
  name: ENDPOINT
  params:
    - {name: method, type: str, required: true}
    - {name: path, type: str, required: true}
```

Then your dictionary template can use `{method}` and `{path}`:

```yaml
0xC0:
  template: "define {method} endpoint at {path}"
```

## Tips

- Keep translations natural. "define GET endpoint at /hello" reads well; "GET endpoint define at /hello" does not.
- For code dictionaries, produce syntactically valid snippets where possible.
- Test edge cases: empty strings, special characters, long parameter values.
- Look at `en.yaml` as a reference for completeness.
