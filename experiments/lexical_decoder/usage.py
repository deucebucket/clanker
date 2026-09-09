"""Opt-in scoped usage weights with immutable evidence and reversible activation.

A weight is an observed count, not an invented parameter. Retraction changes
active counts but keeps the evidence record. No self-output reinforcement.
Source/consent flags are caller attestations, NOT authentication or a detector
of concealed held-out material. Global promotion is intentionally absent.
"""
from __future__ import annotations

import copy
import json
import sqlite3
from collections import Counter
from functools import lru_cache
import re

from .affinity import AffinityStore, canonical, digest, tokens, TOKENIZER_VERSION
from .grammar import DecodeError

MAX_OBSERVATIONS = 256
MAX_EVENT_TOKENS = 128
MAX_TOTAL_TOKENS = 16_000


def count_rows(sequence):
    result = Counter()
    for i, token in enumerate(sequence):
        for n in range(min(2, i) + 1):
            result[(canonical(sequence[i-n:i]), token)] += 1
    return [[h, t, n] for (h, t), n in sorted(result.items())]


class UsageLedger:
    def __init__(self, scope_id: str):
        if not isinstance(scope_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", scope_id):
            raise DecodeError("invalid usage scope")
        self.scope_id = scope_id
        self.observations = []
        self.retractions = []

    @property
    def hash(self):
        return digest(self.to_dict())

    def observe(self, text: str, *, evidence_id: str, source_id: str,
                source_role: str, purpose: str, consent: bool, quoted: bool = False):
        if type(consent) is not bool or not consent or type(quoted) is not bool or quoted:
            raise DecodeError("usage observation requires explicit consent and unquoted evidence")
        if source_role != "user" or purpose not in {"development", "session_usage"}:
            raise DecodeError("assistant, held-out, evaluation and simulated usage is not eligible")
        for name, value in (("evidence_id", evidence_id), ("source_id", source_id)):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", value):
                raise DecodeError("invalid " + name)
        sequence = tokens(text)
        if not 1 <= len(sequence) <= MAX_EVENT_TOKENS:
            raise DecodeError("usage observation token budget")
        record = {"evidence_id": evidence_id, "source_id": source_id, "source_role": source_role,
                  "purpose": purpose, "consent": consent, "quoted": quoted,
                  "content_sha256": digest(sequence), "token_count": len(sequence), "rows": count_rows(sequence)}
        for old in self.observations:
            if old["evidence_id"] == evidence_id:
                if old != record:
                    raise DecodeError("evidence ID is immutable")
                return False
            if old["source_id"] == source_id and old["content_sha256"] == record["content_sha256"]:
                return False  # Relabeling an identical example is not independent support.
        if len(self.observations) >= MAX_OBSERVATIONS or sum(x["token_count"] for x in self.observations) + len(sequence) > MAX_TOTAL_TOKENS:
            raise DecodeError("usage ledger budget")
        self.observations.append(record)
        return True

    def retract(self, evidence_id: str, *, reason: str):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1024:
            raise DecodeError("a bounded correction reason is required")
        if not any(x["evidence_id"] == evidence_id for x in self.observations):
            raise DecodeError("unknown usage evidence")
        if any(x["evidence_id"] == evidence_id for x in self.retractions):
            return False
        self.retractions.append({"evidence_id": evidence_id, "reason_sha256": digest(reason)})
        return True

    def to_dict(self):
        return {"schema": "scoped-usage-v1", "scope_id": self.scope_id,
                "observations": copy.deepcopy(self.observations), "retractions": copy.deepcopy(self.retractions)}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"schema", "scope_id", "observations", "retractions"} or data["schema"] != "scoped-usage-v1":
            raise DecodeError("invalid usage ledger schema")
        result = cls(data["scope_id"])
        observations, retractions = data["observations"], data["retractions"]
        if not isinstance(observations, list) or not isinstance(retractions, list) or len(observations) > MAX_OBSERVATIONS or len(retractions) > len(observations):
            raise DecodeError("usage ledger budget")
        ids, seen_contents, total = set(), set(), 0
        for row in observations:
            keys = {"evidence_id", "source_id", "source_role", "purpose", "consent", "quoted", "content_sha256", "token_count", "rows"}
            if not isinstance(row, dict) or set(row) != keys or row["source_role"] != "user" or row["purpose"] not in {"development", "session_usage"} or row["consent"] is not True or row["quoted"] is not False:
                raise DecodeError("invalid usage evidence")
            for key in ("evidence_id", "source_id"):
                if not isinstance(row[key], str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", row[key]):
                    raise DecodeError("invalid usage identity")
            if not isinstance(row["content_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", row["content_sha256"]):
                raise DecodeError("invalid evidence digest")
            identity = (row["source_id"], row["content_sha256"])
            if row["evidence_id"] in ids or identity in seen_contents:
                raise DecodeError("duplicate usage evidence")
            if type(row["token_count"]) is not int or not 1 <= row["token_count"] <= MAX_EVENT_TOKENS:
                raise DecodeError("invalid observation token count")
            counts = row["rows"]
            if not isinstance(counts, list) or len(counts) > 3 * MAX_EVENT_TOKENS:
                raise DecodeError("invalid observation count rows")
            by_key = {}
            for count in counts:
                if not isinstance(count, (list, tuple)) or len(count) != 3:
                    raise DecodeError("invalid count triple")
                h, t, n = count
                if not isinstance(h, str) or not isinstance(t, str) or type(n) is not int or not 1 <= n <= MAX_EVENT_TOKENS:
                    raise DecodeError("invalid count value")
                history = json.loads(h)
                if not isinstance(history, list) or len(history) > 2 or canonical(history) != h:
                    raise DecodeError("invalid count history")
                if any(not isinstance(w, str) or tokens(w) != (w,) for w in [*history, t]):
                    raise DecodeError("invalid count token")
                if (h, t) in by_key:
                    raise DecodeError("duplicate count triple")
                by_key[h, t] = n
            unigrams = {t: n for (h, t), n in by_key.items() if h == "[]"}
            if sum(unigrams.values()) != row["token_count"]:
                raise DecodeError("observation count total mismatch")
            partitions = Counter()
            for (h, t), n in by_key.items():
                history = json.loads(h)
                if t not in unigrams or any(w not in unigrams for w in history):
                    raise DecodeError("unknown count token")
                if history:
                    suffix = canonical(history[1:])
                    partitions[suffix, t] += n
            if any(n > by_key.get(key, 0) for key, n in partitions.items()):
                raise DecodeError("context counts exceed backoff support")
            ids.add(row["evidence_id"]); seen_contents.add(identity)
            total += row["token_count"]
        if total > MAX_TOTAL_TOKENS:
            raise DecodeError("usage ledger token budget")
        retired = set()
        for r in retractions:
            if not isinstance(r, dict) or set(r) != {"evidence_id", "reason_sha256"} or r["evidence_id"] not in ids or r["evidence_id"] in retired:
                raise DecodeError("invalid usage retraction")
            if not isinstance(r["reason_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", r["reason_sha256"]):
                raise DecodeError("invalid correction digest")
            retired.add(r["evidence_id"])
        result.observations = copy.deepcopy(observations)
        result.retractions = copy.deepcopy(retractions)
        return result

    def active_rows(self):
        retired = {x["evidence_id"] for x in self.retractions}
        counts = Counter()
        for observation in self.observations:
            if observation["evidence_id"] not in retired:
                for h, t, n in observation["rows"]:
                    counts[(h, t)] += n
        return counts

    def compile(self, base: AffinityStore):
        return SessionAffinity(base, self)


class SessionAffinity(AffinityStore):
    """Immutable per-turn view; learning never mutates the active decoder pack."""
    def __init__(self, base, ledger):
        base_payload = base.to_payload()
        counts = Counter({(h, t): n for h, t, n in base_payload["rows"]})
        counts.update(ledger.active_rows())
        rows = [(h, t, n) for (h, t), n in sorted(counts.items()) if n > 0]
        self.manifest = {"format": "session-affinity-v1", "tokenizer": TOKENIZER_VERSION,
                         "purpose": "session_usage", "scope_id": ledger.scope_id,
                         "base_pack_sha256": base.pack_hash, "ledger_sha256": ledger.hash,
                         "counts_sha256": digest(rows), "source_attestations_independently_verified": False,
                         "observations": len(ledger.observations), "retractions": len(ledger.retractions)}
        self.pack_hash = digest(self.manifest)
        self.vocabulary = frozenset(t for h, t, n in rows if h == "[]")
        self.connection = sqlite3.connect(":memory:")
        self.connection.executescript("CREATE TABLE counts(history TEXT, token TEXT, n INTEGER, PRIMARY KEY(history, token)) WITHOUT ROWID;")
        self.connection.executemany("INSERT INTO counts VALUES(?,?,?)", rows)
        self.connection.commit()
        self.connection.execute("PRAGMA query_only=ON")
        self._lookup = lru_cache(maxsize=2048)(self._query)

    def to_payload(self):
        raise DecodeError("session affinity is restored from its scoped ledger, not promoted as a base pack")

    def save(self, path):
        raise DecodeError("session affinity cannot be exported as a reviewed global pack")
