"""Bounded, immutable SQLite n-gram counts with executed probability receipts.

This is a development compiler, not a global-learning or promotion endpoint.
No sentence is retained. Source attestations are not independently verified.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

TOKENIZER_VERSION = "word-punctuation-v1"
TOKEN = re.compile(r"\d{1,2}:\d{2}|[^\W\d_]+(?:['’][^\W\d_]+)?|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)
MAX_TOKENS = 200_000
MAX_ROWS = 200_000


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def tokens(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or len(text) > 16_384:
        raise ValueError("text must be a string of at most 16384 characters")
    return tuple(x.replace("’", "'").lower() for x in TOKEN.findall(text))


class AffinityStore:
    """One read-only logical count snapshot; bounded cache, no runtime updates."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        try:
            meta = connection.execute("SELECT payload FROM metadata").fetchall()
            if len(meta) != 1:
                raise ValueError("exactly one pack manifest is required")
            self.manifest = json.loads(meta[0][0])
            if set(self.manifest) != {"format", "tokenizer", "source_id", "source_sha256", "license", "purpose", "counts_sha256"}:
                raise ValueError("unknown or missing pack manifest fields")
            if self.manifest["format"] != 1 or self.manifest["tokenizer"] != TOKENIZER_VERSION:
                raise ValueError("unsupported affinity format or tokenizer")
            if self.manifest["purpose"] != "development" or self.manifest["license"] not in {"MIT", "CC0-1.0"}:
                raise ValueError("only explicitly licensed development packs are supported")
            if not isinstance(self.manifest["source_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", self.manifest["source_id"]):
                raise ValueError("invalid source identifier")
            if not isinstance(self.manifest["source_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", self.manifest["source_sha256"]):
                raise ValueError("invalid source digest")
            rows = connection.execute("SELECT history, token, n FROM counts ORDER BY history, token LIMIT ?", (MAX_ROWS + 1,)).fetchall()
            if len(rows) > MAX_ROWS:
                raise ValueError("affinity row budget exceeded")
            for history, token, n in rows:
                h = json.loads(history)
                if not isinstance(h, list) or len(h) > 2 or history != canonical(h):
                    raise ValueError("invalid history")
                if any(not isinstance(w, str) or tokens(w) != (w,) for w in [*h, token]):
                    raise ValueError("counts must contain normalized single tokens")
                if type(n) is not int or not 1 <= n <= MAX_TOKENS:
                    raise ValueError("invalid count")
            if digest(rows) != self.manifest["counts_sha256"]:
                raise ValueError("affinity count digest mismatch")
            unigrams = {t: n for h, t, n in rows if h == "[]"}
            if sum(unigrams.values()) > MAX_TOKENS:
                raise ValueError("affinity token budget exceeded")
            if any(t not in unigrams or n > unigrams[t] for h, t, n in rows):
                raise ValueError("context count is inconsistent with unigram count")
            if any(w not in unigrams for h, _, _ in rows for w in json.loads(h)):
                raise ValueError("unknown history token")
            by_key = {(h, t): n for h, t, n in rows}
            partitions: Counter[tuple[str, str]] = Counter()
            for h, t, n in rows:
                history = json.loads(h)
                if history:
                    suffix = canonical(history[1:])
                    if n > by_key.get((suffix, t), 0):
                        raise ValueError("higher-order count exceeds its backoff count")
                    partitions[(suffix, t)] += n
            if any(n > by_key.get(key, 0) for key, n in partitions.items()):
                raise ValueError("higher-order partitions exceed backoff support")
            self.vocabulary = frozenset(unigrams)
            self.pack_hash = digest(self.manifest)
            connection.execute("PRAGMA query_only=ON")
            self._lookup = lru_cache(maxsize=2048)(self._query)
        except Exception:
            connection.close()
            raise

    @classmethod
    def compile(cls, sentences: Iterable[str], *, source_id: str, license: str = "MIT", purpose: str = "development") -> "AffinityStore":
        if purpose != "development" or license not in {"MIT", "CC0-1.0"}:
            raise ValueError("training/held-out/private/promoted inputs are not accepted by this development compiler")
        counts: Counter[tuple[str, str]] = Counter()
        source = hashlib.sha256()
        total = 0
        for index, sentence in enumerate(sentences):
            if index >= 20_000:
                raise ValueError("sentence budget exceeded")
            seq = tokens(sentence)
            total += len(seq)
            if total > MAX_TOKENS:
                raise ValueError("token budget exceeded")
            raw = sentence.encode("utf-8")
            source.update(len(raw).to_bytes(8, "big")); source.update(raw)
            for position, word in enumerate(seq):
                for order in range(min(2, position) + 1):
                    counts[(canonical(seq[position-order:position]), word)] += 1
            if len(counts) > MAX_ROWS:
                raise ValueError("row budget exceeded")
        rows = [(h, t, n) for (h, t), n in sorted(counts.items())]
        manifest = {
            "format": 1, "tokenizer": TOKENIZER_VERSION, "source_id": source_id,
            "source_sha256": source.hexdigest(), "license": license, "purpose": purpose,
            "counts_sha256": digest(rows),
        }
        connection = sqlite3.connect(":memory:")
        connection.executescript("CREATE TABLE metadata(payload TEXT NOT NULL); CREATE TABLE counts(history TEXT NOT NULL, token TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY(history, token)) WITHOUT ROWID;")
        with connection:
            connection.execute("INSERT INTO metadata VALUES(?)", (canonical(manifest),))
            connection.executemany("INSERT INTO counts VALUES(?,?,?)", rows)
        return cls(connection)

    @classmethod
    def from_payload(cls, payload: dict) -> "AffinityStore":
        if not isinstance(payload, dict) or set(payload) != {"manifest", "rows"}:
            raise ValueError("invalid count payload")
        rows = payload["rows"]
        if not isinstance(rows, list) or len(rows) > MAX_ROWS:
            raise ValueError("invalid or excessive count rows")
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 3 or type(row[2]) is not int:
                raise ValueError("count rows must be history/token/integer triples")
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript("CREATE TABLE metadata(payload TEXT NOT NULL); CREATE TABLE counts(history TEXT NOT NULL, token TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY(history, token)) WITHOUT ROWID;")
            with connection:
                connection.execute("INSERT INTO metadata VALUES(?)", (canonical(payload["manifest"]),))
                connection.executemany("INSERT INTO counts VALUES(?,?,?)", rows)
            return cls(connection)
        except Exception:
            connection.close()
            raise

    def to_payload(self) -> dict:
        rows = self.connection.execute("SELECT history, token, n FROM counts ORDER BY history, token").fetchall()
        return {"manifest": dict(self.manifest), "rows": [list(r) for r in rows]}

    @classmethod
    def load(cls, path: str | Path) -> "AffinityStore":
        path = Path(path).resolve(strict=True)
        if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("pack must be a regular SQLite file no larger than 32 MiB")
        # Copy a read-only source into private memory: later disk mutations
        # cannot change scores underneath a receipt's frozen pack identity.
        source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        destination = sqlite3.connect(":memory:")
        try:
            source.backup(destination)
        finally:
            source.close()
        return cls(destination)

    def save(self, path: str | Path) -> None:
        # Exclusive creation prevents accidental replacement of a reviewed pack.
        path = Path(path)
        with path.open("xb"):
            pass
        out = sqlite3.connect(str(path))
        try:
            self.connection.backup(out)
        finally:
            out.close()

    def probability(self, history: Sequence[str], word: str) -> dict:
        h = tuple(w.replace("’", "'").lower() for w in history[-2:])
        word = word.replace("’", "'").lower()
        if tokens(word) != (word,):
            raise ValueError("probability query requires one normalized token")
        return dict(self._lookup(h, word))

    def _query(self, requested: tuple[str, ...], word: str) -> tuple:
        used = requested
        while True:
            denominator = self.connection.execute("SELECT COALESCE(SUM(n),0) FROM counts WHERE history=?", (canonical(used),)).fetchone()[0]
            if denominator or not used:
                break
            used = used[1:]
        row = self.connection.execute("SELECT n FROM counts WHERE history=? AND token=?", (canonical(used), word)).fetchone()
        count = row[0] if row else 0
        # Add-half smoothing, including one explicit unknown-word class.
        numerator = 2 * count + 1
        normalizer = 2 * denominator + len(self.vocabulary) + 1
        p = numerator / normalizer
        return tuple({
            "requested_context": requested, "used_context": used,
            "backoff_steps": len(requested) - len(used), "count": count,
            "context_total": denominator, "vocabulary_size": len(self.vocabulary),
            "smoothing": "add-half-with-unknown-class",
            "probability_kind": "word" if word in self.vocabulary else "unknown_class",
            "numerator": numerator, "denominator": normalizer,
            "probability": p, "cost_micros": round(-math.log(p) * 1_000_000),
            "pack_sha256": self.pack_hash,
        }.items())

    def close(self) -> None:
        self._lookup.cache_clear()
        self.connection.close()
