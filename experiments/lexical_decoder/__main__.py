"""Explicit research chat with no network or artificial token delay."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .affinity import AffinityStore, canonical
from .decoder import DecoderConfig
from .grammar import DecodeError
from .runtime import ReceiptChat


def atomic_json(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".lexical-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(canonical(value) + "\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", help="one turn; otherwise interactive stdin")
    parser.add_argument("--pack", type=Path, help="explicit development SQLite count pack")
    parser.add_argument("--snapshot", type=Path, help="opt-in private session persistence")
    parser.add_argument("--receipt", type=Path, help="write the LAST executed receipt; contains session words")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--affect-weight", type=int, default=24)
    parser.add_argument("--usage-scope", help="enable explicitly submitted /hear examples in a private scope")
    args = parser.parse_args(argv)
    pack = None
    try:
        pack = AffinityStore.load(args.pack) if args.pack else None
        with ReceiptChat(pack=pack, config=DecoderConfig(beam_width=args.beam, affect_weight=args.affect_weight), usage_scope=args.usage_scope) as chat:
            if args.snapshot and args.snapshot.exists():
                if args.snapshot.stat().st_size > 8 * 1024 * 1024:
                    raise DecodeError("snapshot exceeds 8 MiB limit")
                chat.restore(json.loads(args.snapshot.read_text()))
            if args.once is not None:
                messages = (args.once,)
            else:
                print("Experimental lexical chat — real V8; /quit exits; /receipt inspects the last turn.", file=sys.stderr)
                messages = sys.stdin
            for message in messages:
                message = message.strip()
                if message == "/quit":
                    break
                if not message:
                    continue
                if message.startswith("/hear ") or message.startswith("/retract "):
                    fields = message.split(maxsplit=2)
                    if len(fields) != 3:
                        raise DecodeError("use /hear ID text or /retract ID reason")
                    if fields[0] == "/hear":
                        changed = chat.observe_usage(fields[2], evidence_id=fields[1],
                                                     source_id=args.usage_scope or "", consent=True)
                    else:
                        changed = chat.retract_usage(fields[1], reason=fields[2])
                    print(canonical({"control": fields[0], "changed": changed,
                                     "ledger_sha256": chat.usage.hash}), file=sys.stderr)
                    if args.snapshot:
                        atomic_json(args.snapshot, chat.snapshot())
                    continue
                if message == "/receipt":
                    print(canonical(chat.receipt)); continue
                result = chat.process(message)
                for chunk in chat.committed_chunks():
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                sys.stdout.write("\n")
                if args.trace:
                    print(canonical(chat.receipt))
                if args.receipt:
                    atomic_json(args.receipt, chat.receipt)
                if args.snapshot:
                    atomic_json(args.snapshot, chat.snapshot())
                if chat.receipt["coverage"] != "supported":
                    print("Coverage declined: " + chat.receipt["coverage"], file=sys.stderr)
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"lexical decoder: {exc}\n")
    finally:
        if pack is not None:
            pack.close()


if __name__ == "__main__":
    raise SystemExit(main())
