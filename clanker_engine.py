#!/usr/bin/env python3
"""Clanker VADUGWI Engine -- compiled standalone binary.

Usage:
    # Score a single sentence
    ./clanker-engine score "whatever makes you happy"

    # Score with perspective
    ./clanker-engine score --perspective bystander "she is fat"

    # Score with personality
    ./clanker-engine score --personality stoic.yaml "you are worthless"

    # Score a file (one sentence per line)
    ./clanker-engine file input.txt --output results.json

    # Run as local API server
    ./clanker-engine serve --port 7860

    # Interactive mode
    ./clanker-engine interactive

License: MIT
DOI: 10.5281/zenodo.19383636
Author: Jerry Mares
"""

import argparse
import json
import sys
import time
import os

# Add engine path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.pendulum import compute_vadug
from engine.solver import state_transition
from engine.shared import VADUG
from engine.shared import PersonalityVector


def load_personality(path):
    """Load personality from YAML-like config file."""
    if not path or not os.path.exists(path):
        return None
    config = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                try:
                    config[key] = int(val)
                except ValueError:
                    config[key] = val
    return PersonalityVector(
        gullibility=config.get('gullibility', 25),
        agreeableness=config.get('agreeableness', 100),
        suggestibility=config.get('suggestibility', 30),
        truthfulness=config.get('truthfulness', 235),
        safety=config.get('safety', 200),
        curiosity=config.get('curiosity', 170),
        assertiveness=config.get('assertiveness', 120),
        playfulness=config.get('playfulness', 100),
    )


def format_result(vadug, meta, compact=False):
    """Format VADUGWI result as dict."""
    structs = [s.pattern for s in meta.get("structures", [])]
    ff = meta.get("force_flow")
    result = {
        "v": vadug.v, "a": vadug.a, "d": vadug.d,
        "u": vadug.u, "g": vadug.g, "w": vadug.w, "i": vadug.i,
    }
    if not compact:
        result["structures"] = structs
        if ff:
            result["force_flow"] = {
                "actor": ff.actor_role,
                "target": ff.target_role,
                "negated": ff.negated,
            }
        result["word_count"] = meta.get("word_count", 0)
        result["trace"] = meta.get("trace", [])
    return result


def cmd_score(args):
    """Score a single sentence."""
    text = " ".join(args.text)
    personality = load_personality(args.personality) if args.personality else None
    perspective = args.perspective or "speaker"

    start = time.perf_counter()
    vadug, meta = compute_vadug(text, personality=personality, perspective=perspective)
    elapsed = time.perf_counter() - start

    result = format_result(vadug, meta, compact=args.compact)
    result["text"] = text
    result["perspective"] = perspective
    result["ms"] = round(elapsed * 1000, 3)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        structs = result.get("structures", [])
        struct_str = ", ".join(structs) if structs else "none"
        print(f"V={vadug.v} A={vadug.a} D={vadug.d} U={vadug.u} G={vadug.g} W={vadug.w} I={vadug.i}")
        print(f"Structures: {struct_str}")
        print(f"Perspective: {perspective}")
        print(f"Time: {elapsed*1000:.3f}ms")


def cmd_file(args):
    """Score an entire file."""
    personality = load_personality(args.personality) if args.personality else None
    perspective = args.perspective or "auto"

    with open(args.input) as f:
        lines = [l.strip() for l in f if l.strip() and len(l.strip()) > 3]

    results = []
    start = time.perf_counter()
    for line in lines:
        try:
            vadug, meta = compute_vadug(line, personality=personality, perspective=perspective)
            entry = format_result(vadug, meta, compact=True)
            entry["text"] = line
            results.append(entry)
        except Exception as e:
            results.append({"text": line, "error": str(e)})
    elapsed = time.perf_counter() - start

    summary = {
        "total": len(results),
        "time_sec": round(elapsed, 3),
        "per_sentence_ms": round(elapsed / len(results) * 1000, 3) if results else 0,
        "sentences_per_sec": round(len(results) / elapsed) if elapsed > 0 else 0,
        "v_mean": round(sum(r.get("v", 128) for r in results) / len(results), 1) if results else 0,
        "w_mean": round(sum(r.get("w", 128) for r in results) / len(results), 1) if results else 0,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Scored {len(results)} lines in {elapsed:.2f}s ({summary['sentences_per_sec']} sent/sec)")
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(summary, indent=2))


def cmd_interactive(args):
    """Interactive scoring mode."""
    personality = load_personality(args.personality) if args.personality else None
    perspective = args.perspective or "auto"
    state = VADUG()

    print("Clanker VADUGWI Engine -- Interactive Mode")
    print(f"Perspective: {perspective}")
    print("Type a sentence to score. Type 'quit' to exit.")
    print("Type 'state' to see current running state.")
    print("Type 'reset' to reset state to neutral.")
    print("-" * 50)

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() == 'quit':
            break
        if text.lower() == 'state':
            print(f"State: V={state.v} A={state.a} D={state.d} U={state.u} G={state.g} W={state.w} I={state.i}")
            continue
        if text.lower() == 'reset':
            state = VADUG()
            print("State reset to neutral.")
            continue

        vadug, meta = compute_vadug(text, personality=personality, perspective=perspective)
        structs = [s.pattern for s in meta.get("structures", [])]
        struct_str = ", ".join(structs) if structs else "none"

        # Update running state via A+B=C
        state = state_transition(state, vadug)

        print(f"Score: V={vadug.v} A={vadug.a} D={vadug.d} U={vadug.u} G={vadug.g} W={vadug.w} I={vadug.i}")
        print(f"State: V={state.v} A={state.a} D={state.d} U={state.u} G={state.g} W={state.w} I={state.i}")
        if structs:
            print(f"Patterns: {struct_str}")


def cmd_serve(args):
    """Run as local HTTP API server."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    personality = load_personality(args.personality) if args.personality else None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if parsed.path == "/score":
                text = params.get("text", [""])[0]
                perspective = params.get("perspective", ["auto"])[0]
                if not text:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "missing ?text= parameter"}).encode())
                    return

                start = time.perf_counter()
                vadug, meta = compute_vadug(text, personality=personality, perspective=perspective)
                elapsed = time.perf_counter() - start

                result = format_result(vadug, meta, compact=False)
                result["text"] = text
                result["perspective"] = perspective
                result["ms"] = round(elapsed * 1000, 3)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            elif parsed.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "engine": "vadugwi-v5.5"}).encode())

            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "not found", "endpoints": ["/score?text=...", "/health"]}).encode())

        def do_POST(self):
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()

            if self.path == "/score":
                try:
                    data = json.loads(body)
                    text = data.get("text", "")
                    perspective = data.get("perspective", "auto")
                    pers_config = data.get("personality")
                except:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "invalid JSON"}).encode())
                    return

                pers = None
                if pers_config and isinstance(pers_config, dict):
                    pers = PersonalityVector(**{k: v for k, v in pers_config.items() if hasattr(PersonalityVector, k)})

                start = time.perf_counter()
                vadug, meta = compute_vadug(text, personality=pers or personality, perspective=perspective)
                elapsed = time.perf_counter() - start

                result = format_result(vadug, meta, compact=False)
                result["text"] = text
                result["perspective"] = perspective
                result["ms"] = round(elapsed * 1000, 3)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            elif self.path == "/batch":
                try:
                    data = json.loads(body)
                    sentences = data.get("sentences", [])
                    perspective = data.get("perspective", "auto")
                except:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "invalid JSON"}).encode())
                    return

                results = []
                start = time.perf_counter()
                for text in sentences:
                    vadug, meta = compute_vadug(text, personality=personality, perspective=perspective)
                    r = format_result(vadug, meta, compact=True)
                    r["text"] = text
                    results.append(r)
                elapsed = time.perf_counter() - start

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "results": results,
                    "total": len(results),
                    "ms": round(elapsed * 1000, 3),
                }).encode())

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # silent

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"VADUGWI Engine serving on http://localhost:{args.port}")
    print(f"  GET  /score?text=...&perspective=auto")
    print(f"  POST /score  {{\"text\": \"...\", \"perspective\": \"auto\", \"personality\": {{...}}}}")
    print(f"  POST /batch  {{\"sentences\": [...]}}")
    print(f"  GET  /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Clanker VADUGWI Engine -- 7D emotional scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # score
    p_score = sub.add_parser("score", help="Score a sentence")
    p_score.add_argument("text", nargs="+", help="Text to score")
    p_score.add_argument("--perspective", choices=["speaker", "listener", "bystander", "auto"])
    p_score.add_argument("--personality", help="Path to personality YAML file")
    p_score.add_argument("--json", action="store_true", help="Output as JSON")
    p_score.add_argument("--compact", action="store_true", help="Minimal output")

    # file
    p_file = sub.add_parser("file", help="Score an entire file")
    p_file.add_argument("input", help="Input text file (one sentence per line)")
    p_file.add_argument("--output", "-o", help="Output JSON file")
    p_file.add_argument("--perspective", choices=["speaker", "listener", "bystander", "auto"], default="auto")
    p_file.add_argument("--personality", help="Path to personality YAML file")

    # interactive
    p_int = sub.add_parser("interactive", help="Interactive scoring mode with running state")
    p_int.add_argument("--perspective", choices=["speaker", "listener", "bystander", "auto"], default="auto")
    p_int.add_argument("--personality", help="Path to personality YAML file")

    # serve
    p_serve = sub.add_parser("serve", help="Run as local HTTP API server")
    p_serve.add_argument("--port", type=int, default=7860)
    p_serve.add_argument("--personality", help="Default personality YAML file")

    args = parser.parse_args()
    if args.command == "score":
        cmd_score(args)
    elif args.command == "file":
        cmd_file(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
