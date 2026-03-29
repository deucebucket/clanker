# Clanker-Lang Roadmap

## v0.1 — Foundation (current)

- [x] Formal specification (SPEC.md)
- [x] Core opcode table (0x00-0x0F)
- [x] Web, Data, Logic opcode tables
- [x] Hardware opcodes from delphinOS
- [x] Reference dictionaries: English, Chinese, Python, Rust, JavaScript
- [x] Reference Python decoder
- [x] Example scripts
- [x] Type system and composition rules

## v0.2 — Expanded Coverage

- [ ] Finalize full core opcode range (0x00-0x1F)
- [ ] Finalize full web opcode range (0xC0-0xCF)
- [ ] Finalize full data opcode range (0xD0-0xDF)
- [ ] Finalize full logic opcode range (0xE0-0xEF)
- [ ] Add dictionaries: Spanish, Farsi, Arabic, Japanese
- [ ] Add code dictionaries: Go, C, TypeScript
- [ ] Validator: verify Clanker scripts against opcode definitions
- [ ] Decoder error messages with line numbers

## v0.3 — Binary Compiler

- [ ] Binary format compiler: `.clank.txt` to `.clank`
- [ ] Binary format decoder: `.clank` to text
- [ ] Streaming decoder for large programs
- [ ] Size benchmarks: binary vs text vs gzip

## v0.4 — AI Training Data

- [ ] Parallel corpus generator: English-to-Clanker pairs
- [ ] Multi-language corpus: Clanker decoded to all available dictionaries
- [ ] Fine-tuning dataset format (JSONL with source/target)
- [ ] Token count benchmarks: Clanker vs English vs Chinese vs code
- [ ] Self-bootstrapping guide: use an LLM to generate Clanker training data from the spec

## v0.5 — Community

- [ ] Dictionary contribution guide with CI validation
- [ ] Opcode proposal process (RFC-style)
- [ ] Online playground: paste Clanker, pick a dictionary, see output
- [ ] VS Code extension for `.clank.txt` syntax highlighting
- [ ] Registry of community dictionaries

## v1.0 — Stable Specification

- [ ] Freeze all standard opcodes (0x00-0xEF)
- [ ] Formal grammar and test suite
- [ ] Reference implementations in Python, Rust, JavaScript
- [ ] Published specification document
- [ ] Interoperability tests between implementations
- [ ] Long-term support commitment: opcodes are forever
