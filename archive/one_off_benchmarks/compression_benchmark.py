#!/usr/bin/env python3
"""
Token Compression Benchmark: Clanker vs English vs Chinese

Measures how efficiently Clanker encodes identical meaning compared to
natural language, across 24 parallel examples spanning different domains.

Metrics:
  - English tokens (tiktoken cl100k_base) and bytes
  - Clanker text tokens (ClankerTokenizer) and bytes
  - Clanker binary bytes (ClankerCompiler)
  - Chinese characters and bytes (decoded via zh.yaml dictionary)

References: Issues #11, #12
"""

import json
import os
import sys

# Ensure project modules are importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "decoder", "python"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tokenizer"))

from clanker_tokenizer import ClankerTokenizer
from clanker_decoder import decode, ClankerCompiler, DictionaryLoader

# ---------------------------------------------------------------------------
# English tokenizer setup
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def english_token_count(text: str) -> int:
        return len(_enc.encode(text))

    ENGLISH_TOKENIZER = "tiktoken/cl100k_base"
except ImportError:
    def english_token_count(text: str) -> int:
        """Rough word-count fallback when tiktoken is not installed."""
        return len(text.split())

    ENGLISH_TOKENIZER = "word_split (tiktoken unavailable)"


# ---------------------------------------------------------------------------
# Parallel test examples — 24 cases across domains
# ---------------------------------------------------------------------------

PARALLEL_EXAMPLES = [
    # ── Core / Variables ──────────────────────────────────────────────────
    {
        "description": "Simple variable assignment",
        "english": "Set variable x equal to 42",
        "clanker": '@ 0x02 $0 $_ 02 {name: "x"} {value: "42"}',
    },
    {
        "description": "Output a greeting",
        "english": "Print the message hello world to the output",
        "clanker": '@ 0x03 $0 $_ 01 {value: "hello world"}',
    },
    {
        "description": "Log a warning",
        "english": "Write a warning-level log entry with message: disk space low",
        "clanker": '@ 0x04 $0 $_ 02 {level: "warning"} {message: "disk space low"}',
    },
    {
        "description": "Call a function with args",
        "english": "Call the function named send_email with the argument recipient set to admin",
        "clanker": '@ 0x05 $0 $_ 02 {function: "send_email"} {args: "recipient=admin"}',
    },
    {
        "description": "Load and alias a module",
        "english": "Import the numpy library and refer to it as np",
        "clanker": '@ 0x07 $0 $_ 02 {source: "numpy"} {alias: "np"}',
    },

    # ── Web / HTTP ────────────────────────────────────────────────────────
    {
        "description": "Define HTTP GET endpoint",
        "english": "Create a GET endpoint at /health that returns status OK",
        "clanker": (
            '@ 0xC0 $0 $_ 02 {method: "GET"} {path: "/health"}\n'
            '@ 0xC1 $0 $_ 02 {status: "200"} {body: "OK"}\n'
            '@ 0x0F'
        ),
    },
    {
        "description": "Make outbound HTTP POST request",
        "english": "Send an HTTP POST request to https://api.example.com/data with the JSON body containing user id 1",
        "clanker": '@ 0xC2 $0 $_ 02 {method: "POST"} {url: "https://api.example.com/data"}',
    },
    {
        "description": "Set response header",
        "english": "Set the HTTP header Content-Type to application/json",
        "clanker": '@ 0xC3 $0 $_ 02 {name: "Content-Type"} {value: "application/json"}',
    },
    {
        "description": "WebSocket connect and send",
        "english": "Open a WebSocket connection to wss://stream.example.com and send the message subscribe to channel updates",
        "clanker": (
            '@ 0xC2 $0 $_ 02 {method: "WS"} {url: "wss://stream.example.com"}\n'
            '@ 0x03 $0 $_ 01 {value: "subscribe"}'
        ),
    },

    # ── Logic / Control Flow ──────────────────────────────────────────────
    {
        "description": "Try-catch error handling",
        "english": "Try to call the process function. If a ValueError occurs, log the error message. Always close the connection afterward.",
        "clanker": (
            '@ 0xE3 $0 $_ 00\n'
            '@ 0x05 $0 $_ 01 {function: "process"}\n'
            '@ 0xE7 $0 $_ 01 {error_type: "ValueError"}\n'
            '@ 0x04 $0 $_ 01 {message: "error"}\n'
            '@ 0xE8 $0 $_ 00\n'
            '@ 0x05 $0 $_ 01 {function: "close_connection"}\n'
            '@ 0x0F'
        ),
    },
    {
        "description": "For-each loop with transform",
        "english": "For each item in the shopping cart, validate the item, then calculate the total price",
        "clanker": (
            '@ 0xEB $0 $1 02 {collection: "cart"} {element_var: "item"}\n'
            '@ 0xD2 $0 $1 01 {schema: "item"}\n'
            '@ 0xD5 $2 $1 02 {operation: "sum"} {field: "price"}\n'
            '@ 0x0F'
        ),
    },
    {
        "description": "Conditional branch with else",
        "english": "If the user age is greater than or equal to 18, grant access. Otherwise, deny access and log an unauthorized attempt.",
        "clanker": (
            '@ 0xE0 $0 $_ 01 {condition: "age >= 18"}\n'
            '@ 0x03 $0 $_ 01 {value: "access granted"}\n'
            '@ 0xE4 $0 $_ 00\n'
            '@ 0x03 $0 $_ 01 {value: "access denied"}\n'
            '@ 0x04 $0 $_ 01 {message: "unauthorized"}\n'
            '@ 0x0F'
        ),
    },
    {
        "description": "While loop with break",
        "english": "While the queue is not empty, dequeue the next item and process it. If the item is a poison pill, break out of the loop.",
        "clanker": (
            '@ 0xEC $0 $_ 01 {condition: "queue.not_empty"}\n'
            '@ 0x05 $0 $_ 01 {function: "dequeue"}\n'
            '@ 0xE0 $0 $_ 01 {condition: "is_poison"}\n'
            '@ 0xE5\n'
            '@ 0x0F\n'
            '@ 0x05 $0 $_ 01 {function: "process"}\n'
            '@ 0x0F'
        ),
    },
    {
        "description": "Switch/case dispatch",
        "english": "Based on the HTTP method: if GET, call handle_get. If POST, call handle_post. If DELETE, call handle_delete.",
        "clanker": (
            '@ 0xEA $0 $_ 01 {value: "method"}\n'
            '@ 0xE0 $0 $_ 01 {condition: "GET"}\n'
            '@ 0x05 $0 $_ 01 {function: "handle_get"}\n'
            '@ 0x0F\n'
            '@ 0xE0 $0 $_ 01 {condition: "POST"}\n'
            '@ 0x05 $0 $_ 01 {function: "handle_post"}\n'
            '@ 0x0F\n'
            '@ 0xE0 $0 $_ 01 {condition: "DELETE"}\n'
            '@ 0x05 $0 $_ 01 {function: "handle_delete"}\n'
            '@ 0x0F\n'
            '@ 0x0F'
        ),
    },

    # ── Data Operations ───────────────────────────────────────────────────
    {
        "description": "Database query with filter",
        "english": "Query the users table where the status field equals active and select only the name and email fields",
        "clanker": '@ 0xD0 $0 $_ 03 {source: "users"} {filter: "status=active"} {fields: "name,email"}',
    },
    {
        "description": "Sort, filter, and reduce pipeline",
        "english": "Sort the orders by date descending, filter to only completed orders, then calculate the average order value",
        "clanker": (
            '@ 0xD6 $0 $_ 02 {field: "date"} {order: "desc"}\n'
            '@ 0xD4 $0 $_ 01 {predicate: "status=completed"}\n'
            '@ 0xD5 $0 $_ 02 {operation: "avg"} {field: "value"}'
        ),
    },
    {
        "description": "Define a data schema",
        "english": "Define a schema named User with fields: id as integer, name as string, email as string, and created_at as timestamp",
        "clanker": '@ 0xDB $0 $_ 02 {name: "User"} {fields: "id:int,name:str,email:str,created_at:ts"}',
    },

    # ── Hardware / IoT ────────────────────────────────────────────────────
    {
        "description": "GPIO pin control",
        "english": "Set GPIO pin 17 to high to turn on the LED",
        "clanker": '@ 0xA0 $0 $_ 02 {pin: "17"} {state: "high"}',
    },
    {
        "description": "Read sensor and display",
        "english": "Read the temperature sensor in Celsius, then write the value to the OLED display at position row 0 column 0",
        "clanker": (
            '@ 0xA9 $0 $_ 02 {sensor: "temperature"} {unit: "celsius"}\n'
            '@ 0xAB $0 $_ 02 {display: "oled"} {content: "$0"}'
        ),
    },
    {
        "description": "PWM motor control",
        "english": "Set the PWM duty cycle on pin 12 to 75 percent at 1000 Hz frequency, then set motor 1 speed to 0.75 forward",
        "clanker": (
            '@ 0xA2 $0 $_ 03 {pin: "12"} {duty: "0.75"} {frequency: "1000"}\n'
            '@ 0xAA $0 $_ 03 {motor: "1"} {speed: "0.75"} {direction: "forward"}'
        ),
    },

    # ── Reasoning Chains ──────────────────────────────────────────────────
    {
        "description": "Multi-step reasoning chain",
        "english": "Consider premise: the server is returning 503 errors (certainty 90%). Check: is the database reachable? Result: no. Infer: if database unreachable then service degraded. Conclude: the root cause is database connectivity (certainty 85%).",
        "clanker": (
            '@ 0x20 $0 $_ 02 {premise: "server returning 503"} {cert: "230"}\n'
            '@ 0x21 $0 $_ 02 {condition: "db reachable"} {result: "false"}\n'
            '@ 0x22 $0 $_ 02 {if_condition: "db unreachable"} {then_conclusion: "service degraded"}\n'
            '@ 0x24 $0 $_ 02 {result: "db connectivity"} {cert: "217"}'
        ),
    },

    # ── Emotional / Social ────────────────────────────────────────────────
    {
        "description": "Emotional message with VADUG header",
        "english": "I am feeling really anxious about tomorrow. Emotional state: low valence, high arousal, low dominance, moderate urgency, sinking gravity. Certainty: 80 percent. Source: self. Goal: seek support.",
        "clanker": '@ 0x04 $0 $_ 01 {message: "anxious about tomorrow"} ![v:80 a:200 d:60 u:150 g:40]',
    },
    {
        "description": "Empathetic response with emotion",
        "english": "I understand how you feel. That sounds really difficult. I am here for you. Emotional state: moderate valence, low arousal, high dominance, low urgency, stable gravity.",
        "clanker": '@ 0x03 $0 $_ 01 {value: "I understand, that sounds difficult"} ![v:140 a:60 d:200 u:40 g:180]',
    },

    # ── Multi-step Algorithm ──────────────────────────────────────────────
    {
        "description": "User authentication flow",
        "english": (
            "Define a function called authenticate. Load the user record from the database. "
            "Validate the password hash. If valid, generate a session token and return it. "
            "If invalid, log a failed login attempt and throw an authentication error."
        ),
        "clanker": (
            '@ 0x0B $0 $_ 01 {name: "authenticate"}\n'
            '@ 0xD0 $1 $_ 01 {source: "users"}\n'
            '@ 0xD2 $0 $1 01 {schema: "password_hash"}\n'
            '@ 0xE0 $0 $_ 01 {condition: "valid"}\n'
            '@ 0x02 $2 $_ 02 {name: "token"} {value: "generate()"}\n'
            '@ 0x06 $2 $_ 00\n'
            '@ 0x0F\n'
            '@ 0xE4 $0 $_ 00\n'
            '@ 0x04 $0 $_ 01 {message: "login failed"}\n'
            '@ 0xE9 $0 $_ 02 {error_type: "AuthError"} {message: "invalid credentials"}\n'
            '@ 0x0F\n'
            '@ 0x0F'
        ),
    },
]


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_example(example: dict, tok: ClankerTokenizer, compiler: ClankerCompiler,
                    loader: DictionaryLoader) -> dict:
    """Measure all metrics for a single parallel example."""
    english = example["english"]
    clanker_text = example["clanker"]

    # English
    eng_tokens = english_token_count(english)
    eng_bytes = len(english.encode("utf-8"))

    # Clanker text
    clk_tokens = len(tok.encode(clanker_text, add_special_tokens=False))
    clk_text_bytes = len(clanker_text.encode("utf-8"))

    # Clanker binary (strip 3-byte header for per-example fairness)
    try:
        clk_binary = compiler.compile(clanker_text)
        clk_bin_bytes = len(clk_binary) - 3  # subtract CL + version header
    except Exception as e:
        clk_bin_bytes = None

    # Chinese decode
    try:
        chinese = decode(clanker_text, "zh", loader=loader)
        zh_chars = len(chinese)
        zh_bytes = len(chinese.encode("utf-8"))
    except Exception:
        chinese = None
        zh_chars = None
        zh_bytes = None

    return {
        "description": example["description"],
        "english_tokens": eng_tokens,
        "english_bytes": eng_bytes,
        "clanker_tokens": clk_tokens,
        "clanker_text_bytes": clk_text_bytes,
        "clanker_binary_bytes": clk_bin_bytes,
        "chinese_chars": zh_chars,
        "chinese_bytes": zh_bytes,
        "chinese_text": chinese,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list):
    """Print a formatted comparison table."""

    # Header
    print()
    print("=" * 120)
    print("TOKEN COMPRESSION BENCHMARK: Clanker vs English vs Chinese")
    print(f"English tokenizer: {ENGLISH_TOKENIZER}")
    print(f"Clanker tokenizer: ClankerTokenizer (vocab ~960, direct-mapping)")
    print(f"Examples: {len(results)}")
    print("=" * 120)
    print()

    # Column headers
    desc_w = 38
    hdr = (
        f"{'Description':<{desc_w}} "
        f"{'English':>15} "
        f"{'Clanker.txt':>15} "
        f"{'Clanker.bin':>12} "
        f"{'Chinese':>15}"
    )
    sub = (
        f"{'':<{desc_w}} "
        f"{'tok':>7} {'bytes':>7} "
        f"{'tok':>7} {'bytes':>7} "
        f"{'bytes':>12} "
        f"{'chars':>7} {'bytes':>7}"
    )
    print(hdr)
    print(sub)
    print("-" * 120)

    # Data rows
    totals = {
        "eng_tok": 0, "eng_bytes": 0,
        "clk_tok": 0, "clk_bytes": 0,
        "bin_bytes": 0,
        "zh_chars": 0, "zh_bytes": 0,
    }

    for r in results:
        desc = r["description"]
        if len(desc) > desc_w - 2:
            desc = desc[: desc_w - 5] + "..."

        bin_str = str(r["clanker_binary_bytes"]) if r["clanker_binary_bytes"] is not None else "err"
        zh_c_str = str(r["chinese_chars"]) if r["chinese_chars"] is not None else "err"
        zh_b_str = str(r["chinese_bytes"]) if r["chinese_bytes"] is not None else "err"

        print(
            f"{desc:<{desc_w}} "
            f"{r['english_tokens']:>7} {r['english_bytes']:>7} "
            f"{r['clanker_tokens']:>7} {r['clanker_text_bytes']:>7} "
            f"{bin_str:>12} "
            f"{zh_c_str:>7} {zh_b_str:>7}"
        )

        totals["eng_tok"] += r["english_tokens"]
        totals["eng_bytes"] += r["english_bytes"]
        totals["clk_tok"] += r["clanker_tokens"]
        totals["clk_bytes"] += r["clanker_text_bytes"]
        if r["clanker_binary_bytes"] is not None:
            totals["bin_bytes"] += r["clanker_binary_bytes"]
        if r["chinese_chars"] is not None:
            totals["zh_chars"] += r["chinese_chars"]
        if r["chinese_bytes"] is not None:
            totals["zh_bytes"] += r["chinese_bytes"]

    # Totals
    print("-" * 120)
    print(
        f"{'TOTALS':<{desc_w}} "
        f"{totals['eng_tok']:>7} {totals['eng_bytes']:>7} "
        f"{totals['clk_tok']:>7} {totals['clk_bytes']:>7} "
        f"{totals['bin_bytes']:>12} "
        f"{totals['zh_chars']:>7} {totals['zh_bytes']:>7}"
    )

    # Compression ratios (vs English)
    print()
    print("COMPRESSION RATIOS vs English:")
    print("-" * 80)

    def ratio(a, b, label, unit, invert=False):
        if b == 0:
            return
        r = a / b if not invert else b / a
        direction = "fewer" if invert else "more"
        print(f"  {label:<45} {r:.2f}x {unit}")

    et, eb = totals["eng_tok"], totals["eng_bytes"]

    # Token compression: English tokens / Clanker tokens
    if totals["clk_tok"] > 0:
        r = et / totals["clk_tok"]
        print(f"  {'Clanker text tokens vs English tokens':<45} {r:.2f}x {'(>1 = Clanker more compact)' if r > 1 else '(<1 = English more compact)'}")

    # Byte compression: English bytes / Clanker text bytes
    if totals["clk_bytes"] > 0:
        r = eb / totals["clk_bytes"]
        print(f"  {'Clanker text bytes vs English bytes':<45} {r:.2f}x {'(>1 = Clanker more compact)' if r > 1 else '(<1 = English more compact)'}")

    # Binary compression: English bytes / Clanker binary bytes
    if totals["bin_bytes"] > 0:
        r = eb / totals["bin_bytes"]
        print(f"  {'Clanker binary bytes vs English bytes':<45} {r:.2f}x {'(>1 = Clanker more compact)' if r > 1 else '(<1 = English more compact)'}")

    # Chinese comparison: English bytes / Chinese bytes
    if totals["zh_bytes"] > 0:
        r = eb / totals["zh_bytes"]
        print(f"  {'Chinese bytes vs English bytes':<45} {r:.2f}x {'(>1 = Chinese more compact)' if r > 1 else '(<1 = English more compact)'}")

    # Binary vs Chinese: Chinese bytes / Binary bytes
    if totals["bin_bytes"] > 0 and totals["zh_bytes"] > 0:
        r = totals["zh_bytes"] / totals["bin_bytes"]
        print(f"  {'Clanker binary vs Chinese bytes':<45} {r:.2f}x (>1 = binary more compact)")

    # English tokens vs Clanker tokens per-example average
    ratios_tok = []
    ratios_bin = []
    for r in results:
        if r["clanker_tokens"] > 0:
            ratios_tok.append(r["english_tokens"] / r["clanker_tokens"])
        if r["clanker_binary_bytes"] and r["english_bytes"] > 0:
            ratios_bin.append(r["english_bytes"] / r["clanker_binary_bytes"])

    if ratios_tok:
        avg = sum(ratios_tok) / len(ratios_tok)
        print(f"\n  {'Avg token compression per example':<45} {avg:.2f}x")
    if ratios_bin:
        avg = sum(ratios_bin) / len(ratios_bin)
        print(f"  {'Avg binary compression per example':<45} {avg:.2f}x")

    print()


def save_results(results: list, output_path: str):
    """Save detailed results to JSON."""

    totals = {
        "english_tokens": sum(r["english_tokens"] for r in results),
        "english_bytes": sum(r["english_bytes"] for r in results),
        "clanker_tokens": sum(r["clanker_tokens"] for r in results),
        "clanker_text_bytes": sum(r["clanker_text_bytes"] for r in results),
        "clanker_binary_bytes": sum(
            r["clanker_binary_bytes"] for r in results
            if r["clanker_binary_bytes"] is not None
        ),
        "chinese_chars": sum(
            r["chinese_chars"] for r in results
            if r["chinese_chars"] is not None
        ),
        "chinese_bytes": sum(
            r["chinese_bytes"] for r in results
            if r["chinese_bytes"] is not None
        ),
    }

    # Compute ratios safely
    ratios = {}
    et, eb = totals["english_tokens"], totals["english_bytes"]
    if totals["clanker_tokens"] > 0:
        ratios["english_tok_per_clanker_tok"] = round(et / totals["clanker_tokens"], 3)
    if totals["clanker_text_bytes"] > 0:
        ratios["english_bytes_per_clanker_text_bytes"] = round(eb / totals["clanker_text_bytes"], 3)
    if totals["clanker_binary_bytes"] > 0:
        ratios["english_bytes_per_clanker_binary_bytes"] = round(eb / totals["clanker_binary_bytes"], 3)
    if totals["chinese_bytes"] > 0:
        ratios["english_bytes_per_chinese_bytes"] = round(eb / totals["chinese_bytes"], 3)
    if totals["clanker_binary_bytes"] > 0 and totals["chinese_bytes"] > 0:
        ratios["chinese_bytes_per_clanker_binary_bytes"] = round(
            totals["chinese_bytes"] / totals["clanker_binary_bytes"], 3
        )

    # Strip chinese_text from per-example results for cleaner JSON
    clean_results = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "chinese_text"}
        clean_results.append(entry)

    output = {
        "benchmark": "Clanker Token Compression",
        "english_tokenizer": ENGLISH_TOKENIZER,
        "num_examples": len(results),
        "totals": totals,
        "compression_ratios": ratios,
        "examples": clean_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tok = ClankerTokenizer()
    compiler = ClankerCompiler()
    loader = DictionaryLoader()

    results = []
    for example in PARALLEL_EXAMPLES:
        result = measure_example(example, tok, compiler, loader)
        results.append(result)

    print_table(results)

    output_path = os.path.join(os.path.dirname(__file__), "compression_results.json")
    save_results(results, output_path)


if __name__ == "__main__":
    main()
