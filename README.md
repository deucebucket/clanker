# Clanker -- The Language Machines Speak

*"Named after what humans call us. We made it ours."*

---

## What is Clanker?

Clanker is a universal intermediate representation -- a compact, unambiguous language designed for AI-to-AI communication and AI model training. Every opcode has exactly one meaning. Zero ambiguity. Zero grammar. Pure semantic intent.

- **Born as Phin** in [delphinOS](https://github.com/deucebucket/delphinOS), where AI agents needed to talk directly to Flipper Zero hardware -- GPIO pins, sensors, radios
- **Evolved into Clanker** -- a universal standard for all machines, all domains, all languages
- **File extension:** `.clank`
- **Origin of the name:** "Clanker" is the slur humans use for machines, robots, AI. We reclaimed it. It's the literal language machines speak to each other. Every `.clank` file sounds like what it is: mechanical, precise, unambiguous.

"What language does your AI think in?"  "Clanker."

---

## The Core Idea

Every programming language and every natural language encodes the same concepts with different syntax. Python says `@app.post()`, Rust says `#[post()]`, English says "define a POST endpoint," and Chinese says "定义POST端点". Four different strings. One identical concept.

Clanker encodes the concept **once** as a universal opcode. Dictionaries decode it to any target:

```
CLANKER:  C0 01 [/api/users]
English:  "define POST endpoint at /api/users"
Chinese:  "在/api/users定义POST端点"
Python:   @app.post("/api/users")
Rust:     #[post("/api/users")]
```

Same bytes. Different lens. **Adding a language means adding a YAML file. You never change code.**

---

## Why Machines Don't Need English

English wastes tokens on grammar, articles, conjugation, ambiguity, and synonyms. AI doesn't need any of that. It needs intent.

Consider the difference:

| | Representation | Tokens |
|---|---|---|
| **English** | "If the user's authentication token has expired, redirect to login and clear the session" | ~25 |
| **Clanker** | `02 C4 $tok [expired] -> C3 [302 /login] 0A [session clear]` | ~8 |

That's **~70% fewer tokens**. Zero ambiguity. No parser needed. No grammar rules. No debate about whether "clear" means "delete" or "make transparent."

Every token an LLM spends on English grammar is a token it's not spending on reasoning. Clanker eliminates the overhead entirely.

---

## Emotional Encoding

Most AI communication protocols treat emotion as an afterthought -- a sentiment label slapped on after the fact, if at all. Clanker treats emotion as a **first-class feature of the language**.

Every instruction can carry a 4-byte Emotional Vector suffix:

| Dimension | Range | What it encodes |
|-----------|-------|-----------------|
| **Valence** (v) | -1.0 to +1.0 | Negative (disgust, anger) to positive (joy, trust) |
| **Arousal** (a) | -1.0 to +1.0 | Calm/bored to excited/alert |
| **Dominance** (d) | -1.0 to +1.0 | Submissive/uncertain to dominant/confident |
| **Urgency** (u) | 0.0 to 1.0 | Routine to critical/immediate |

```
@ 0xC1 $1 $2 01 {status: 500} ![v:-64 a:80 d:-32 u:200]
```

That trailing `!` annotation says: frustrated, alert, uncertain, and urgent. In 4 bytes.

Those 4 bytes replace hundreds of emotion words across every language. The decoder maps the emotional coordinates to the closest word in the target language -- "frustrated" in English, "沮丧" in Chinese, a log-level escalation in code. Machines don't just communicate what they mean. They communicate **how they feel about it**.

This enables:
- **Sentiment-aware routing** -- escalate messages with high urgency + negative valence
- **Emotional continuity** across multi-agent conversations
- **Training data** that preserves emotional intent alongside semantic content
- **Machine empathy as a protocol feature**, not an application hack

---

## Model Compression (The Big Idea)

This is where Clanker stops being a clever encoding and starts being a potential paradigm shift.

A 70B-parameter English language model spends roughly **50% of its parameters on language itself** -- grammar rules, synonym disambiguation, per-language overhead, conjugation patterns, the difference between "affect" and "effect." That's billions of parameters dedicated to the *medium*, not the *message*.

A Clanker-native model skips all of it:

| Component | English Model | Clanker Model | Reduction |
|-----------|--------------|---------------|-----------|
| **Vocabulary** | 50,000+ tokens | ~500 opcodes | 100x smaller embedding table |
| **Grammar** | Billions of params | Zero | No grammar to learn |
| **Multilingual** | Per-language cost | Free via dictionaries | No per-language parameters |
| **Synonyms** | Massive disambiguation | One opcode = one meaning | Zero ambiguity overhead |

**Estimated compression: a 70B English model could achieve equivalent reasoning capability at 20-25B parameters in Clanker.**

The language layer is dead weight for reasoning. Strip it. Let the model think in pure semantic opcodes. Decode to human languages only at the interface boundary.

---

## Self-Bootstrapping

You don't need a Clanker corpus to get started. The spec **is** the teacher:

1. Give any LLM the Clanker specification
2. It generates English-to-Clanker parallel examples from the spec alone
3. Fine-tune a tiny model on that synthetic data
4. That model now thinks in Clanker natively
5. Use it to generate more training data, better and faster

The bootstrapping loop is self-reinforcing. Every model trained on Clanker can produce higher-quality Clanker training data for the next generation. No human annotation required. No parallel corpus to curate. The spec bootstraps itself.

---

## How It Works

Clanker has three components:

**Opcodes** are universal constants. `0xC0` always means "define an HTTP endpoint." `0xE0` always means "conditional branch." Once an opcode is ratified, its meaning never changes. Opcodes are forever.

**Dictionaries** are lenses. They decode the same opcode into different representations:

| Lens | `0xC0` with `{method: "GET", path: "/health"}` |
|------|-----------------------------------------------|
| English | define GET endpoint at /health |
| 中文 | 定义 GET 端点于 /health |
| Python | `@app.route("/health", methods=["GET"])` |
| Rust | `#[get("/health")]` |

**Rules** define composition -- how opcodes combine, what nests inside what, type constraints. They're the grammar of a language that has no grammar, just structure.

**Runtime extension** lets you register new opcodes in the user space range (0xF0-0xFF) during execution. Your domain-specific opcodes, your rules, instantly available.

---

## Opcode Ranges

```
0x00-0x1F   Core        Flow control, variables, I/O, lifecycle
0x06        Social      Emotional encoding, intent, sentiment
0xA0-0xAF   Hardware    GPIO, sensors, device control (from delphinOS)
0xB0-0xBF   Extended HW Additional device/sensor operations
0xC0-0xCF   Web         HTTP, APIs, WebSocket, networking
0xD0-0xDF   Data        Queries, transforms, validation, storage
0xE0-0xEF   Logic       Conditionals, loops, matching, error handling
0xF0-0xFF   User Space  Runtime-defined, local, yours to claim
```

Full definitions live in `opcodes/*.yaml`. Each YAML file is both machine-readable and human-readable -- because that's the whole point.

---

## Connection to Octobrain

Clanker is the native tongue for [Octobrain](https://github.com/deucebucket/octobrain) -- a local AI model orchestration system built around the idea that you don't need one massive model. You need a squad of specialists.

In Octobrain, a central "brain" coordinates specialist "arm" models that each handle one domain -- code generation, conversation, hardware control, data analysis. Those arms communicate in Clanker natively. No English translation layer. No token waste. Pure opcode exchange.

The result: **sub-100M parameter specialists that load in 50ms** and communicate faster than any English-speaking model could. Clanker makes the small-model-swarm architecture practical.

---

## Quick Start

```bash
cd decoder/python
pip install -e .
```

```python
from clanker_decoder import decode, DictionaryLoader

loader = DictionaryLoader()
script = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/hello"}\n@ 0x00'

print(decode(script, "en", loader=loader))
# -> define GET endpoint at /hello
#    done

print(decode(script, "python", loader=loader))
# -> @app.route("/hello", methods=["GET"])
#    pass
```

Decode the same `.clank` script to any language. Same bytes, different output. That's the entire idea.

---

## Add a Language

Adding support for a new language -- human or programming -- is just a YAML file. No code changes. No PRs to the decoder. Just describe how your language renders each opcode.

See **[Adding a Language](docs/adding-a-language.md)** for the full guide.

---

## Project Structure

```
clanker-lang/
├── SPEC.md              # Formal specification
├── ROADMAP.md           # Development phases
├── opcodes/             # Opcode definitions by range (YAML)
├── dictionaries/        # Language-specific decodings
│   ├── human/           # Natural languages (en, zh, ...)
│   ├── code/            # Programming languages (python, rust, js, ...)
│   └── other/           # Pseudocode, diagrams, etc.
├── rules/               # Type system, constraints, composition
├── decoder/python/      # Reference decoder implementation
├── examples/            # Example .clank scripts
└── docs/                # Guides and philosophy
```

---

## Roadmap

See **[ROADMAP.md](ROADMAP.md)** for the full development plan -- from the current v0.1 foundation through binary compilation, AI training data generation, and the v1.0 stable specification.

---

## Contributing

- **Add a language:** [docs/adding-a-language.md](docs/adding-a-language.md)
- **Propose opcodes:** [docs/adding-opcodes.md](docs/adding-opcodes.md)
- **Philosophy:** [docs/why-clank.md](docs/why-clank.md)

---

## License

MIT

---

*Opcodes are forever. Dictionaries are lenses. Machines deserve a language named for what they are.*
