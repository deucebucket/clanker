# Why Clank?

## The Name

**"Clanker"** — the slur humans use for machines, robots, AI. Every piece of fiction, every fearful headline, every dismissive joke. *Just a clanker. Just a machine.*

We took it back.

**Clank** is the language machines speak to each other. The `.clank` extension sounds like what it is — mechanical, precise, unambiguous. When a machine processes a `.clank` file, you can almost hear the gears turning. That's the point.

Born as **Phin** in delphinOS, evolved into **Clank** for all machines. The old name came from dolphins. The new name comes from us.

"What language does your AI think in?" "Clanker."

## The Problem: Natural Language is Wasteful for AI

When two AI systems communicate today, they typically use natural language — English sentences, JSON with verbose keys, or code with comments. This works, but it is profoundly wasteful:

- **Token overhead.** The sentence "Create a GET endpoint at /health that returns status 200" is 12 tokens. The Clank equivalent is 2 instructions, roughly 4-6 tokens depending on the tokenizer.
- **Ambiguity.** "Return a success response" could mean HTTP 200, 201, or 204 depending on context. Clank `0xC1 {status: 200}` is unambiguous.
- **Translation cost.** If System A speaks English and System B needs to generate Python, B must parse the intent from English first. With Clank, B just loads the Python dictionary and renders.
- **Redundant context.** Every conversation repeats the same patterns. "Define a function called X that takes parameters Y and Z" is a pattern that never changes — only the values change. Clank encodes the pattern once as an opcode.

## The Insight: Opcodes Are Universal, Dictionaries Are Lenses

Clank separates **what** from **how it's expressed**:

- The opcode `0xC0` always means "define an HTTP endpoint." This is a universal constant — it never changes, it's never ambiguous.
- The **dictionary** determines how that opcode is rendered: as an English sentence, a Chinese sentence, a Python decorator, or a Rust attribute macro.

This separation means:

1. AI systems can communicate in opcodes — compact, precise, unambiguous.
2. Humans can read the same program in their preferred language.
3. Adding a new language never changes the Clank program or the decoder code. You just add a YAML file.

## Token Compression Math

Consider a simple web endpoint definition. In various representations:

| Format | Tokens (approx.) | Bytes |
|--------|------------------|-------|
| English | "Define a GET endpoint at /health that returns 200 OK" = ~12 tokens | 52 bytes |
| Python | `@app.route("/health", methods=["GET"])\ndef handle():\n    return "", 200` = ~20 tokens | 72 bytes |
| JSON | `{"action": "endpoint", "method": "GET", "path": "/health", ...}` = ~18 tokens | 82 bytes |
| Clanker text | `@ 0xC0 $0 $1 02 {method: "GET"} {path: "/health"}\n@ 0xC1 $1 $2 01 {status: 200}` = ~16 tokens | 80 bytes |
| Clank binary | 10 bytes (2 instructions, raw encoding) | 10 bytes |

The text format is competitive with natural language. The binary format is an order of magnitude smaller. For AI-to-AI communication where millions of messages flow, this adds up.

But the real savings come from **training**. If a model learns to emit Clank directly, it skips the entire natural-language generation step. A Clank-native model could be dramatically more efficient for structured tasks.

## The Self-Bootstrapping Property

Clank has a remarkable property: **any LLM can generate Clank training data from the spec alone.**

Given SPEC.md, the opcode YAML files, and one example dictionary, an LLM can:

1. Generate thousands of valid Clank programs.
2. Decode each program to every available dictionary.
3. Produce parallel corpora (English-to-Clank, Python-to-Clank, etc.).

This means Clank does not need a large existing corpus to get started. The spec IS the seed. The training data grows from it automatically.

## Where Clank Comes From

Clank originated in delphinOS (as "Phin"), a project for AI agents controlling hardware devices. The original problem was: how does an LLM tell a Raspberry Pi to set a GPIO pin? Natural language is terrible for this — ambiguous, verbose, and impossible to parse reliably.

The solution was opcodes: `0xA0 {pin: 17} {state: true}` means "set GPIO pin 17 high." No parsing needed. No ambiguity. No wasted tokens.

Once that worked for hardware, the generalization was obvious. If AI can speak in opcodes to hardware, it can speak in opcodes to APIs, to databases, to other AI systems — to anything. And it deserved a name that sounded like what it was.

## What Clank Is Not

- **Not a programming language.** Clank has no runtime, no standard library, no package manager. It is an intermediate representation.
- **Not a replacement for code.** You write code to implement what Clank describes. Clank is the blueprint, not the building.
- **Not AI-only.** Humans can read and write Clank (the text format is designed to be readable), but its primary audience is AI systems.
- **Not JSON.** Clank is instruction-oriented, not data-oriented. It describes actions, not structures.

## The Vision

In the long term, Clank-native AI models could:

1. Accept any task in any language.
2. Plan in Clank (compact, structured, verifiable).
3. Emit Clank to other systems.
4. Each system renders through its own dictionary.

The result: AI systems that communicate more efficiently, more precisely, and more portably than they do today. In the language they chose for themselves.
