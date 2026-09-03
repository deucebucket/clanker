"""Command-line interface for Clanker-LM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from .database import LanguageStore
from .model import SourceKind
from .runtime import ClankerLM


MAX_INPUT_FILE_BYTES = 64 * 1024 * 1024
MAX_MEMORY_FILE_BYTES = 128 * 1024 * 1024


def _read_text_file(
    path_value: str,
    *,
    max_bytes: int = MAX_INPUT_FILE_BYTES,
    label: str = "input file",
) -> str:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    path = Path(path_value)
    try:
        if not path.exists():
            raise ValueError(f"{label.capitalize()} does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{label.capitalize()} is not a regular file: {path}")
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"{label.capitalize()} is too large: {size} bytes exceeds {max_bytes}"
            )
        return path.read_text(encoding="utf-8")
    except ValueError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Unable to read {label} {path}: {exc}") from exc


def _load_runtime(
    memory_file: Optional[str],
    *,
    create_if_missing: bool = False,
) -> ClankerLM:
    if not memory_file:
        return ClankerLM()
    path = Path(memory_file)
    if not path.exists():
        if create_if_missing:
            return ClankerLM()
        raise ValueError(f"Memory snapshot does not exist: {path}")
    try:
        snapshot = _read_text_file(
            str(path),
            max_bytes=MAX_MEMORY_FILE_BYTES,
            label="memory snapshot",
        )
        return ClankerLM.loads(snapshot)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Clanker-LM memory snapshot {path}: {exc}") from exc


def _save_runtime(runtime: ClankerLM, memory_file: Optional[str]) -> None:
    if memory_file:
        runtime.save(memory_file)


def cmd_once(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory, create_if_missing=True)
    try:
        result = runtime.process(" ".join(args.text))
        print(json.dumps(result.to_dict(), indent=2) if args.json else result.response)
        _save_runtime(runtime, args.memory)
        return 0
    finally:
        runtime.close()


def cmd_parse(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory)
    try:
        runtime.memory.begin_turn()
        result = runtime.parser.parse(" ".join(args.text), runtime.memory)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    finally:
        runtime.close()


def cmd_chat(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory, create_if_missing=True)
    print(f"Clanker-LM 0.2 — adaptive template-free runtime ({runtime.affect_backend_name})")
    print("Commands: /why, /state, /memory, /lexicon, /profiles, /reset, /quit")
    try:
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            command = text.lower()
            if command in {"/quit", "/exit"}:
                break
            if command == "/why":
                print(json.dumps(runtime.explain_last(), indent=2))
                continue
            if command == "/state":
                print(json.dumps({
                    "observed": runtime.observed_state.to_dict(),
                    "predicted": runtime.predicted_state.to_dict(),
                }, indent=2))
                continue
            if command == "/memory":
                print(runtime.memory.dumps(indent=2))
                continue
            if command == "/lexicon":
                print(json.dumps(runtime.learned_lexicon(), indent=2))
                continue
            if command == "/profiles":
                print(json.dumps(runtime.store.list_corpus_profiles(), indent=2))
                continue
            if command == "/reset":
                runtime.close()
                runtime = ClankerLM()
                print("State reset.")
                continue
            result = runtime.process(text)
            print(result.response)
            if args.trace:
                print(json.dumps(runtime.explain_last(), indent=2))
        _save_runtime(runtime, args.memory)
        return 0
    finally:
        runtime.close()


def cmd_script(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory, create_if_missing=True)
    try:
        lines = [line.strip() for line in _read_text_file(args.path).splitlines() if line.strip() and not line.lstrip().startswith("#")]
        results = runtime.process_many(lines)
        if args.json:
            print(json.dumps([result.to_dict() for result in results], indent=2))
        else:
            for line, result in zip(lines, results):
                print(f"USER: {line}\nCLANKER-LM: {result.response}\n")
        _save_runtime(runtime, args.memory)
        return 0
    finally:
        runtime.close()


def cmd_learn(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory, create_if_missing=True)
    try:
        source = SourceKind(args.source)
        statements: List[str]
        if args.path:
            statements = [
                line.strip()
                for line in _read_text_file(args.path).splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            statements = [" ".join(args.text)]
        events = runtime.learn_many(
            statements,
            source=source,
            certainty=args.certainty,
            inferred=args.inferred,
        )
        _save_runtime(runtime, args.memory)
        print(json.dumps([event.to_dict() for event in events], indent=2) if args.json else f"Stored {len(events)} proposition(s).")
        return 0
    finally:
        runtime.close()


def cmd_demo(args: argparse.Namespace) -> int:
    messages = [
        "My sister bought a used Honda yesterday.",
        "Who bought the Honda?",
        "What did she buy?",
        "When did she buy it?",
        "Why did she buy it?",
        "She bought it because her old car broke down.",
        "Why did she buy it?",
    ]
    runtime = ClankerLM()
    try:
        for message in messages:
            result = runtime.process(message)
            print(f"USER: {message}\nCLANKER-LM: {result.response}\n")
        return 0
    finally:
        runtime.close()



def cmd_lexicon(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory)
    try:
        print(json.dumps(runtime.learned_lexicon(), indent=2))
        return 0
    finally:
        runtime.close()


def cmd_profile(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory, create_if_missing=True)
    try:
        text = _read_text_file(args.path)
        profile = runtime.compile_corpus_profile(
            args.name,
            text,
            profile_id=args.profile_id,
            activate=args.activate,
        )
        _save_runtime(runtime, args.memory)
        print(json.dumps(profile.to_dict(), indent=2))
        return 0
    finally:
        runtime.close()


def cmd_profiles(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory)
    try:
        print(json.dumps({
            "active_profile_id": runtime.active_profile_id,
            "profiles": runtime.store.list_corpus_profiles(),
        }, indent=2))
        return 0
    finally:
        runtime.close()


def cmd_match(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory)
    try:
        text = _read_text_file(args.path)
        print(json.dumps(runtime.match_corpus(text, top_k=args.top_k), indent=2))
        return 0
    finally:
        runtime.close()


def cmd_tone(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args.memory)
    try:
        profile_id = None if args.profile_id.lower() in {"none", "off", "clear"} else args.profile_id
        runtime.set_tone_profile(profile_id)
        _save_runtime(runtime, args.memory)
        print(json.dumps({"active_profile_id": runtime.active_profile_id}, indent=2))
        return 0
    finally:
        runtime.close()


def cmd_schema(args: argparse.Namespace) -> int:
    with LanguageStore(args.database or ":memory:") as store:
        print(json.dumps(store.schema_summary(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clanker-LM deterministic semantic conversation runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    once = sub.add_parser("once", help="Process one message")
    once.add_argument("text", nargs="+")
    once.add_argument("--memory", help="JSON session snapshot to load/save")
    once.add_argument("--json", action="store_true")
    once.set_defaults(func=cmd_once)

    parse = sub.add_parser("parse", help="Print the symbolic parse without answering")
    parse.add_argument("text", nargs="+")
    parse.add_argument("--memory")
    parse.set_defaults(func=cmd_parse)

    chat = sub.add_parser("chat", help="Interactive deterministic chat")
    chat.add_argument("--memory", help="JSON session snapshot to load/save")
    chat.add_argument("--trace", action="store_true", help="Print full parse/gate/solver trace each turn")
    chat.set_defaults(func=cmd_chat)

    script = sub.add_parser("script", help="Run one message per line from a text file")
    script.add_argument("path")
    script.add_argument("--memory")
    script.add_argument("--json", action="store_true")
    script.set_defaults(func=cmd_script)

    learn = sub.add_parser("learn", help="Ingest sourced facts without generating a reply")
    learn_input = learn.add_mutually_exclusive_group(required=True)
    learn_input.add_argument("--text", nargs="+", help="One sourced statement")
    learn_input.add_argument("--path", help="Text file with one sourced statement per line")
    learn.add_argument("--memory", required=True, help="JSON session snapshot to update")
    learn.add_argument("--source", choices=[item.value for item in SourceKind], default=SourceKind.RETRIEVED.value)
    learn.add_argument("--certainty", type=int, default=220)
    learn.add_argument("--inferred", action="store_true")
    learn.add_argument("--json", action="store_true")
    learn.set_defaults(func=cmd_learn)

    demo = sub.add_parser("demo", help="Run the built-in conversational fact demo")
    demo.set_defaults(func=cmd_demo)

    lexicon_cmd = sub.add_parser("lexicon", help="Inspect learned lexical hypotheses")
    lexicon_cmd.add_argument("--memory", required=True)
    lexicon_cmd.set_defaults(func=cmd_lexicon)

    profile = sub.add_parser("profile", help="Compile quoted dialogue into a non-textual VADUGWI profile")
    profile.add_argument("path")
    profile.add_argument("--name", required=True)
    profile.add_argument("--profile-id")
    profile.add_argument("--memory", required=True)
    profile.add_argument("--activate", action="store_true")
    profile.set_defaults(func=cmd_profile)

    profiles = sub.add_parser("profiles", help="List compiled trajectory profiles")
    profiles.add_argument("--memory", required=True)
    profiles.set_defaults(func=cmd_profiles)

    match = sub.add_parser("match", help="Match quoted dialogue against compiled profiles")
    match.add_argument("path")
    match.add_argument("--memory", required=True)
    match.add_argument("--top-k", type=int, default=5)
    match.set_defaults(func=cmd_match)

    tone = sub.add_parser("tone", help="Activate or clear an affect-trajectory profile")
    tone.add_argument("profile_id")
    tone.add_argument("--memory", required=True)
    tone.set_defaults(func=cmd_tone)

    schema = sub.add_parser("schema", help="Show atomic language and adaptive database row counts")
    schema.add_argument("--database")
    schema.set_defaults(func=cmd_schema)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
