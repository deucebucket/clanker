# Contributing to Clanker

Contributions are welcome. This document covers the basics.

## Getting Started

```bash
git clone https://github.com/deucebucket/clanker.git
cd clanker
pip install -r requirements.txt
python3 -m pytest engine/tests/ -v
```

## What to Contribute

- Bug fixes with test cases
- New structural patterns (with evidence from real text)
- Vocabulary corrections (wrong force values on existing words)
- Benchmark corpus expansion (real conversational text, not synthetic)
- Documentation improvements
- Language dictionaries for the decoder (`dictionaries/`)

## What Not to Change

- Opcode definitions are immutable. Never redefine an existing opcode.
- Force tuples are 5-wide: `(dv, da, dd, du, dg)` -- deltas, not absolute values. W and I are computed dimensions (attribution routing and the agency axis), not stored per-word.
- The physics pipeline order in `pendulum.py` is load-bearing. Don't reorder stages.
- Never tune against `benchmarks/holdout_probes.json` or `benchmarks/holdout_v2_probes.json` -- they are evaluation-only and v2 is sealed (never display its sentences). See `benchmarks/HOLDOUT_PROTOCOL.md`.

## Code Style

- Hex codes uppercase: `0xFF` not `0xff`
- YAML 2-space indentation
- Opcode names UPPER_SNAKE_CASE
- Parameters lowercase_with_underscores
- No comments unless the WHY is non-obvious

## Running Tests

```bash
# Engine tests (333 tests)
python3 -m pytest engine/tests/ -v

# Full benchmark suite
python3 benchmarks/full_barrage.py
```

## Pull Requests

- One change per PR
- Include test cases for behavioral changes
- Run the full test suite before submitting
- Describe what changed and why in the PR description

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
