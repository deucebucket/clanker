"""Interactive command-line entry point for ``python -m clanker_lm``."""

from __future__ import annotations

import argparse
import json
import sys

from .runtime import ClankerLM


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic Clanker-LM semantic conversation runtime"
    )
    parser.add_argument("--db", help="SQLite session database path")
    parser.add_argument("--session", default="default", help="session identifier")
    parser.add_argument("--once", help="process one message and exit")
    parser.add_argument("--trace", action="store_true", help="print the full decision trace")
    parser.add_argument(
        "--strict-clanker",
        action="store_true",
        help="fail instead of using the fallback affect adapter",
    )
    args = parser.parse_args(argv)

    with ClankerLM(
        session_id=args.session,
        db_path=args.db,
        strict_clanker=args.strict_clanker,
    ) as runtime:
        if args.once:
            result = runtime.process(args.once)
            print(result.response)
            if args.trace:
                print(json.dumps(result.trace_dict(), indent=2, default=str))
            return 0

        print("Clanker-LM deterministic dialogue. Type /quit, /reset, or /trace.")
        show_trace = args.trace
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                return 0
            if text == "/reset":
                runtime.reset()
                print("Session reset.")
                continue
            if text == "/trace":
                show_trace = not show_trace
                print(f"Trace {'on' if show_trace else 'off'}.")
                continue
            result = runtime.process(text)
            print(result.response)
            if show_trace:
                print(json.dumps(result.trace_dict(), indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
