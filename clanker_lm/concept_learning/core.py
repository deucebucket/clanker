"""A bounded, scoped learning ledger for explicit type relations.

The starter vocabulary is an internal ontology convention, not a world corpus.
Only two proof rules are supported: subclass transitivity and membership along
subclass edges. Negative evidence is exact, never closed-world negation. No
sentences are stored as replies, and frequency cannot outrank a conflicting
premise. The immutable ledger is evidence, not an authenticated source of truth.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


SCHEMA = "clanker-concept-ledger-v1"
SEED_VERSION = "internal-entity-sorts-v1"
# Definitional internal sorts only. No private people, textbook claims, example
# replies, affect constants, or domain-specific invented words appear here.
SEED_EDGES = tuple((name, "is_a", "entity") for name in ("person", "object", "event", "state", "concept"))
SEED_HASH = digest([SEED_VERSION, SEED_EDGES])
_RESERVED = frozenset("a an the every not no is are might may must could should would and or but if unless because again still longer".split())


def symbol(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", value) or value in _RESERVED:
        raise ValueError("a bounded, unambiguous concept symbol is required")
    return value


def identifier(value: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError("a bounded evidence/context identity is required")
    return value


@dataclass(frozen=True)
class Claim:
    subject: str
    relation: str
    object: str
    positive: bool = True

    def __post_init__(self) -> None:
        if self.relation not in {"is_a", "instance_of"} or type(self.positive) is not bool:
            raise ValueError("unsupported relation or polarity")
        symbol(self.object)
        if self.relation == "is_a":
            symbol(self.subject)
        else:
            identifier(self.subject)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.subject, self.relation, self.object


@dataclass(frozen=True)
class ProofLimits:
    max_depth: int = 16
    max_expansions: int = 2048

    def __post_init__(self):
        if type(self.max_depth) is not int or not 1 <= self.max_depth <= 64:
            raise ValueError("invalid proof depth")
        if type(self.max_expansions) is not int or not 1 <= self.max_expansions <= 100000:
            raise ValueError("invalid proof expansion budget")


class ConceptLedger:
    """Separate append-only learning overlay over a pinned read-only seed.

    One instance belongs to one account/compartment. The host must establish
    that identity. `lineage` is caller-supplied provenance, not authentication.
    Repeated claims from one lineage add no active support, even with new IDs.
    Corrections explicitly retract their original premise; queries rederive
    proofs from the active ledger so stale cached conclusions cannot survive.
    """
    MAX_RECORDS = 2048
    MAX_CONCEPTS = 256

    def __init__(self, scope: str, *, limits: ProofLimits | None = None):
        self.scope = identifier(scope)
        self.limits = limits or ProofLimits()
        self.records: list[dict] = []

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "scope": self.scope, "seed_hash": SEED_HASH,
                "limits": asdict(self.limits), "records": json.loads(canonical(self.records))}

    @classmethod
    def from_dict(cls, value: dict, *, expected_scope: str | None = None) -> "ConceptLedger":
        if (not isinstance(value, dict) or set(value) != {"schema", "scope", "seed_hash", "limits", "records"}
                or value["schema"] != SCHEMA or value["seed_hash"] != SEED_HASH):
            raise ValueError("concept snapshot needs an explicit schema/seed migration")
        if expected_scope is not None and value["scope"] != expected_scope:
            raise ValueError("concept snapshot belongs to another compartment")
        out = cls(value["scope"], limits=ProofLimits(**value["limits"]))
        if not isinstance(value["records"], list) or len(value["records"]) > cls.MAX_RECORDS:
            raise ValueError("invalid concept ledger size")
        for record in value["records"]:
            if not isinstance(record, dict):
                raise ValueError("invalid concept record")
            if record.get("op") == "teach":
                out.teach(Claim(**record["claim"]), evidence_id=record["evidence_id"],
                          lineage=record["lineage"], content_sha256=record["content_sha256"])
            elif record.get("op") == "withdraw":
                out.withdraw(record["target"], reason=record["reason"])
            else:
                raise ValueError("unknown concept ledger operation")
            if not out.records or out.records[-1] != record:
                raise ValueError("concept record contents or chain changed")
        if len(out.records) != len(value["records"]):
            raise ValueError("duplicate concept records")
        return out

    @property
    def generation(self) -> str:
        return digest([SCHEMA, self.scope, SEED_HASH, asdict(self.limits), self.records[-1]["id"] if self.records else None])

    def _append(self, body: dict) -> dict:
        if len(self.records) >= self.MAX_RECORDS:
            raise ValueError("concept evidence budget exceeded")
        data = {**body, "scope": self.scope, "previous": self.records[-1]["id"] if self.records else "0" * 64}
        record = {**data, "id": digest(data)}
        self.records.append(record)
        return json.loads(canonical(record))

    def active(self) -> list[dict]:
        withdrawn = {r["target"] for r in self.records if r["op"] == "withdraw"}
        return [r for r in self.records if r["op"] == "teach" and r["id"] not in withdrawn]

    def concepts(self) -> set[str]:
        items = {term for a, _, b in SEED_EDGES for term in (a, b)}
        for r in self.records:
            if r["op"] == "teach":
                c = r["claim"]
                items.add(c["object"])
                if c["relation"] == "is_a":
                    items.add(c["subject"])
        return items

    def teach(self, claim: Claim, *, evidence_id: str, lineage: str, content_sha256: str) -> dict:
        if not isinstance(claim, Claim):
            raise TypeError("a validated Claim is required")
        identifier(evidence_id); identifier(lineage)
        if not isinstance(content_sha256, str) or not re.fullmatch("[0-9a-f]{64}", content_sha256):
            raise ValueError("source content SHA-256 required")
        for r in self.records:
            if r["op"] == "teach" and r["evidence_id"] == evidence_id:
                if (r["lineage"] != lineage or r["claim"] != asdict(claim) or r["content_sha256"] != content_sha256):
                    raise ValueError("evidence identity cannot be rebound")
                return {"record_id": r["id"], "added": False, "reason": "same_evidence"}
        for r in self.active():
            if r["lineage"] == lineage and r["claim"] == asdict(claim):
                return {"record_id": r["id"], "added": False, "reason": "same_lineage_claim"}
        new = {claim.object} | ({claim.subject} if claim.relation == "is_a" else set())
        if len(self.concepts() | new) > self.MAX_CONCEPTS:
            raise ValueError("concept vocabulary budget exceeded")
        record = self._append({"op": "teach", "claim": asdict(claim), "evidence_id": evidence_id,
                               "lineage": lineage, "content_sha256": content_sha256})
        return {"record_id": record["id"], "added": True, "reason": "new_scoped_premise"}

    def withdraw(self, record_id: str, *, reason: str) -> dict:
        identifier(record_id); identifier(reason)
        targets = [r for r in self.records if r["op"] == "teach" and r["id"] == record_id]
        if not targets:
            raise ValueError("only this compartment's learned premise may be withdrawn")
        old = next((r for r in self.records if r["op"] == "withdraw" and r["target"] == record_id), None)
        if old:
            return {"record_id": old["id"], "added": False}
        r = self._append({"op": "withdraw", "target": record_id, "reason": reason})
        return {"record_id": r["id"], "added": True}

    def _edges(self):
        records = [{"id": "seed:" + digest(c), "claim": asdict(Claim(*c)),
                    "lineage": "reviewed-internal-ontology", "evidence_id": SEED_VERSION}
                   for c in SEED_EDGES] + self.active()
        positive, negative = {}, {}
        for r in records:
            c = Claim(**r["claim"])
            table = positive if c.positive else negative
            table.setdefault(c.key, []).append(r)
        for table in (positive, negative):
            for key in table:
                table[key].sort(key=lambda r: r["id"])
        return positive, negative

    def resolve(self, query: Claim) -> dict:
        """Find one deterministic, uncontradicted supporting derivation.

        Negative inclusion means 'not every X is Y', NOT 'no X is Y'; it never
        propagates to individual members. Every step cites an active premise.
        Contradicted intermediate subpaths cannot support further conclusions.
        Search is bounded. No proof, or an exhausted search, is UNKNOWN, never
        false. A direct exact denial is FALSE only if the positive search has
        completed without a compatible proof or unresolved conflict frontier.
        """
        if not isinstance(query, Claim) or not query.positive:
            raise ValueError("resolve expects a positive, typed question")
        positive, negative = self._edges()
        adj: dict[tuple[str, str], list[tuple[str, tuple]]] = {}
        for key in positive:
            adj.setdefault(key[:2], []).append((key[2], key))
        for choices in adj.values():
            choices.sort()
        direct_negative = negative.get(query.key, [])
        pending = deque([(query.subject, query.relation, (query.subject,), ())])
        explored = 0; truncated = False; blocked: set[str] = set()
        found: tuple[tuple, ...] | None = None
        # Reflexivity is a logical convention about a named type, not proof
        # that it has an actual instance. Unknown names are not auto-grounded.
        if query.relation == "is_a" and query.subject == query.object and query.subject in self.concepts():
            found = ()
        while pending and found is None:
            node, relation, nodes, path = pending.popleft()
            choices = adj.get((node, relation), [])
            if len(path) >= self.limits.max_depth:
                truncated |= bool(choices)
                continue
            for nxt, key in choices:
                explored += 1
                if explored > self.limits.max_expansions:
                    truncated = True; pending.clear(); break
                # The first membership node is an instance identity, not a
                # concept symbol. Equal spelling across those namespaces is
                # not a cycle (an instance ID may itself be 'object').
                visited_concepts = nodes[1:] if query.relation == "instance_of" else nodes
                if nxt in visited_concepts:
                    continue
                denial_keys = []
                for i, prior in enumerate(nodes):
                    rel = query.relation if i == 0 else "is_a"
                    denial = (prior, rel, nxt)
                    # Keep exact target contradiction visible as CONFLICT.
                    if denial in negative and not (i == 0 and nxt == query.object):
                        denial_keys.append(denial)
                if denial_keys:
                    for denied in denial_keys:
                        blocked.update(r["id"] for r in negative[denied])
                    continue
                extended = path + (key,)
                if nxt == query.object:
                    found = extended
                    break
                pending.append((nxt, "is_a", nodes + (nxt,), extended))
        if found is not None:
            status = "conflict" if direct_negative else "true"
        elif direct_negative and not truncated and not blocked:
            status = "false"
        else:
            status = "unknown"
        steps = []
        for key in found or ():
            witnesses = positive[key]
            steps.append({"claim": asdict(Claim(*key)), "record_ids": [r["id"] for r in witnesses],
                          "lineages": sorted({r["lineage"] for r in witnesses}),
                          "evidence_ids": [r["evidence_id"] for r in witnesses]})
        body = {"schema": SCHEMA, "scope": self.scope, "generation": self.generation,
                "query": asdict(query), "status": status, "steps": steps,
                "denial_record_ids": [r["id"] for r in direct_negative],
                "blocked_dependency_ids": sorted(blocked), "search_exhausted": truncated,
                "explored_edges": min(explored, self.limits.max_expansions),
                "rules": ["subclass_transitivity", "membership_inheritance"],
                "epistemic_basis": "relative_to_active_definitions_and_reported_membership",
                "external_truth_verified": False}
        return {**body, "receipt_sha256": digest(body)}

    def verify(self, receipt: dict) -> None:
        if not isinstance(receipt, dict) or receipt.get("scope") != self.scope:
            raise ValueError("concept proof has the wrong compartment")
        if self.resolve(Claim(**receipt["query"])) != receipt:
            raise ValueError("concept proof is stale, modified, or not reproducible")
