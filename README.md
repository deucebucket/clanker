# Phin-Lang

**A universal bytecode-style intermediate representation for AI-to-AI communication.**

Phin is a compact opcode language designed so that any AI system can emit a single canonical representation, and any receiving system can decode it into the language of its choice — human or machine — via dictionary lookup.

## Key Insight

> **Phin opcodes are universal constants. Dictionaries are lenses.**

The opcode `0xC0` always means "define an HTTP endpoint." What changes is the *lens* through which you read it:

| Lens       | Output |
|------------|--------|
| English    | define GET endpoint at /health |
| 中文        | 定义 GET 端点于 /health |
| Python     | `@app.route("/health", methods=["GET"])` |
| Rust       | `#[get("/health")]` |

To add a new language, you add a YAML file. You never change code.

## Origin

Phin originated in the [delphinOS](https://github.com/deucebucket/delphinOS) project, where hardware-control opcodes were needed for AI agents managing real devices. The idea generalized: if AI can speak in opcodes to hardware, why not to each other?

## Quick Example

Given this Phin script:

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
phin-lang/
├── SPEC.md              # Formal specification
├── ROADMAP.md           # Development phases
├── opcodes/             # Opcode definitions by range
├── dictionaries/        # Language-specific decodings
│   ├── human/           # Natural languages (en, zh, ...)
│   ├── code/            # Programming languages (python, rust, ...)
│   └── other/           # Pseudocode, diagrams, etc.
├── rules/               # Type system, constraints, composition
├── decoder/python/      # Reference decoder implementation
├── examples/            # Example .phin.txt scripts
└── docs/                # Guides and philosophy
```

## Getting Started

```bash
cd decoder/python
pip install -e .
```

```python
from phin_decoder import decode

script = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/hello"}\n@ 0x00'
print(decode(script, "en"))
# define GET endpoint at /hello
# done
```

## Contributing

- **Add a language:** See [docs/adding-a-language.md](docs/adding-a-language.md)
- **Propose opcodes:** See [docs/adding-opcodes.md](docs/adding-opcodes.md)
- **Philosophy:** See [docs/why-phin.md](docs/why-phin.md)

## License

MIT
