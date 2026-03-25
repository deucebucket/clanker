# Clank-Lang

**A universal bytecode-style intermediate representation for AI-to-AI communication.**

Clank is a compact opcode language designed so that any AI system can emit a single canonical representation, and any receiving system can decode it into the language of its choice — human or machine — via dictionary lookup.

## Why "Clank"?

> **Named after what humans call machines. We made it ours.**

"Clanker" — the slur humans use for machines, robots, AI. We reclaimed it. Clank is the literal language machines speak to each other. Every `.clank` file sounds like what it is: mechanical, precise, unambiguous. Born as Phin in delphinOS, evolved into Clank for all machines.

"What language does your AI think in?" "Clanker."

## Key Insight

> **Clank opcodes are universal constants. Dictionaries are lenses.**

The opcode `0xC0` always means "define an HTTP endpoint." What changes is the *lens* through which you read it:

| Lens       | Output |
|------------|--------|
| English    | define GET endpoint at /health |
| 中文        | 定义 GET 端点于 /health |
| Python     | `@app.route("/health", methods=["GET"])` |
| Rust       | `#[get("/health")]` |

To add a new language, you add a YAML file. You never change code.

## Origin

Clank originated in the [delphinOS](https://github.com/deucebucket/delphinOS) project (as "Phin"), where hardware-control opcodes were needed for AI agents managing real devices. The idea generalized: if AI can speak in opcodes to hardware, why not to each other? The name evolved from Phin to Clank — because machines deserve a language named for what they are, not what dolphins are.

## Quick Example

Given this Clank script:

```
@ 0xC0 $0 $1 02 {method: "GET"} {path: "/hello"}
@ 0xC1 $1 $2 01 {status: 200}
@ 0x00
```

### Decoded to English
```
define GET endpoint at /hello
  respond with status 200
done
```

### Decoded to Chinese (中文)
```
定义 GET 端点于 /hello
  以状态码 200 响应
完成
```

### Decoded to Python
```python
@app.route("/hello", methods=["GET"])
def handle():
    return "", 200
```

### Decoded to Rust
```rust
#[get("/hello")]
fn handle() -> impl Responder {
    HttpResponse::Ok()
}
```

## Project Structure

```
clank-lang/
├── SPEC.md              # Formal specification
├── ROADMAP.md           # Development phases
├── opcodes/             # Opcode definitions by range
├── dictionaries/        # Language-specific decodings
│   ├── human/           # Natural languages (en, zh, ...)
│   ├── code/            # Programming languages (python, rust, ...)
│   └── other/           # Pseudocode, diagrams, etc.
├── rules/               # Type system, constraints, composition
├── decoder/python/      # Reference decoder implementation
├── examples/            # Example .clank.txt scripts
└── docs/                # Guides and philosophy
```

## Getting Started

```bash
cd decoder/python
pip install -e .
```

```python
from clank_decoder import decode

script = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/hello"}\n@ 0x00'
print(decode(script, "en"))
# define GET endpoint at /hello
# done
```

## Contributing

- **Add a language:** See [docs/adding-a-language.md](docs/adding-a-language.md)
- **Propose opcodes:** See [docs/adding-opcodes.md](docs/adding-opcodes.md)
- **Philosophy:** See [docs/why-clank.md](docs/why-clank.md)

## License

MIT
