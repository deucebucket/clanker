"""Command-line entry point for the conversation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import DATA_DIR, SOURCE_DIR, compile_corpora, verify_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile, verify, or run Clanker-LM conversation evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    compile_parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    subparsers.add_parser("verify")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--split", choices=("heldout", "development"), default="development")
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--failures", type=Path)
    args = parser.parse_args()

    if args.command == "compile":
        print(json.dumps(compile_corpora(source_dir=args.source_dir, output_dir=args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_corpus(), indent=2, sort_keys=True))
        return 0
    from .runner import run_evaluation

    report = run_evaluation(split=args.split, output_path=args.output, failures_path=args.failures)
    if args.output:
        print(json.dumps({
            "split": report["split"],
            "corpus_sha256": report["corpus_sha256"],
            "semantic_fingerprint": report["semantic_fingerprint"],
            "failure_count": report["failure_count"],
            "output": str(args.output),
            "failures": str(args.failures) if args.failures else None,
        }, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
