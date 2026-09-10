"""Bounded goal-directed reasoning over explicitly supplied concept evidence.

Pure data in/data out: no model, network, source text, generated prose, or writes
back into the knowledge store. The two supported rule families are membership
inheritance and subclass transitivity, not arbitrary theorem proving. A trace
records executed operations; it is not a claim about subjective thought.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable

SCHEMA = "clanker-goal-workspace-v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class ReasoningLimits:
    max_records: int = 1024
    max_work: int = 1024
    max_depth: int = 16
    max_bridges: int = 16

    def __post_init__(self) -> None:
        for key, ceiling in (("max_records", 8192), ("max_work", 16384),
                             ("max_depth", 32), ("max_bridges", 64)):
            n = getattr(self, key)
            if type(n) is not int or not 1 <= n <= ceiling:
                raise ValueError("invalid reasoning limit: " + key)


@dataclass(frozen=True)
class Goal:
    relation: str
    subject: str
    target: str

    def __post_init__(self) -> None:
        if self.relation not in {"member", "subclass"}:
            raise ValueError("unlicensed goal relation")
        for value in (self.subject, self.target):
            if (not isinstance(value, str) or not 1 <= len(value) <= 180
                    or any(c.isspace() or ord(c) < 32 for c in value)):
                raise ValueError("goal requires bounded atomic identities")

    def triple(self) -> list[str]:
        return [self.relation, self.subject, self.target]


class _Budget(Exception):
    pass


class _Work:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.count = 0
        self.steps: list[dict] = []

    def step(self, operation: str, **data: Any) -> None:
        if self.count >= self.maximum:
            raise _Budget()
        self.steps.append({"step": self.count, "operation": operation, **data})
        self.count += 1


class EvidenceView:
    """Per-call immutable-by-copy indexes; no dependence on mutable cache keys."""
    def __init__(self, rows: Iterable[dict], scope: str, ledger_id: str,
                 limits: ReasoningLimits):
        if not isinstance(scope, str) or not scope or len(scope) > 180:
            raise ValueError("reasoning scope required")
        if (not isinstance(ledger_id, str) or len(ledger_id) != 64
                or any(c not in "0123456789abcdef" for c in ledger_id)):
            raise ValueError("pinned ledger fingerprint required")
        self.scope, self.ledger_id = scope, ledger_id
        collected = []
        for row in rows:
            if len(collected) >= limits.max_records:
                raise ValueError("reasoning evidence capacity exceeded")
            if (not isinstance(row, dict) or row.get("scope") != scope
                    or row.get("kind") != "assert" or type(row.get("positive")) is not bool):
                raise ValueError("unscoped or invalid reasoning evidence")
            allowed = {"kind", "scope", "relation", "subject", "target", "positive",
                       "evidence_id", "lineage", "input_hash", "turn", "sequence", "previous", "sha256"}
            if set(row) - allowed or any(
                    type(value) not in {str, bool, int}
                    or (isinstance(value, str) and len(value.encode("utf-8")) > 720)
                    or (type(value) is int and not 0 <= value <= 2**63 - 1)
                    for value in row.values()):
                raise ValueError("reasoning evidence must be bounded scalar provenance")
            Goal(row.get("relation"), row.get("subject"), row.get("target"))
            if not isinstance(row.get("evidence_id"), str) or not row["evidence_id"]:
                raise ValueError("missing evidence identity")
            collected.append(json.loads(canonical(row)))
        self.rows = tuple(sorted(collected, key=lambda r: r["evidence_id"]))
        self.by_id = {row["evidence_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError("duplicate evidence identity")
        self.sha256 = fingerprint(self.rows)
        self.adj: dict[tuple, list[dict]] = {}
        self.reverse: dict[str, list[dict]] = {}
        self.against: dict[tuple, list[str]] = {}
        for row in self.rows:
            triple = (row["relation"], row["subject"], row["target"])
            if not row["positive"]:
                self.against.setdefault(triple, []).append(row["evidence_id"])
            else:
                self.adj.setdefault(triple[:2], []).append(row)
                if row["relation"] == "subclass":
                    self.reverse.setdefault(row["target"], []).append(row)
        for edges in (*self.adj.values(), *self.reverse.values()):
            edges.sort(key=lambda r: (r["target"], r["subject"], r["evidence_id"]))


def _target_region(view: EvidenceView, target: str, work: _Work,
                   limits: ReasoningLimits) -> tuple[dict[str, tuple[str, ...]], bool]:
    """Backward subgoals: a known superclass chain that ends at the target."""
    routes: dict[str, tuple[str, ...]] = {target: ()}
    queue = deque([target])
    depth_limited = False
    while queue:
        node = queue.popleft()
        for edge in view.reverse.get(node, ()):
            work.step("EXPAND_TARGET_SUBGOAL", goal=["subclass", edge["subject"], target],
                      evidence_id=edge["evidence_id"])
            child = edge["subject"]
            if child in routes:
                continue
            if len(routes[node]) >= limits.max_depth:
                depth_limited = True
                continue
            routes[child] = (edge["evidence_id"],) + routes[node]
            queue.append(child)
    return routes, depth_limited


def _search(view: EvidenceView, goal: Goal, work: _Work, limits: ReasoningLimits,
            focused: dict[str, tuple[str, ...]] | None) -> tuple[dict, dict[str, tuple[str, ...]]]:
    queue = deque([(goal.relation, goal.subject, (), (), (), ())])
    visited: set[tuple] = set()
    reachable: dict[str, tuple[str, ...]] = {}
    clean = contested = None
    limited = False
    # A negative assertion is not a disjointness axiom, and absence is not false.
    direct = sorted(view.against.get(tuple(goal.triple()), ()))
    while queue:
        kind, current, path, denials, rules, ancestors = queue.popleft()
        sig = (kind, current, bool(denials), tuple(sorted(ancestors)))
        if sig in visited:
            continue
        visited.add(sig)
        work.step("VISIT_SUBGOAL", relation=kind, subject=current, target=goal.target,
                  depth=len(path), inherited_counterevidence=list(denials))
        edges = view.adj.get((kind, current), ())
        if len(path) >= limits.max_depth:
            limited |= bool(edges)
            continue
        for row in edges:
            end = row["target"]
            work.step("CONSIDER_PREMISE", evidence_id=row["evidence_id"],
                      triple=[kind, current, end])
            if focused is not None and end not in focused:
                work.step("PRUNE_IRRELEVANT", evidence_id=row["evidence_id"],
                          reason="no indexed positive path to requested class")
                continue
            used = path + (row["evidence_id"],)
            contrary = list(view.against.get((kind, current, end), ()))
            contrary.extend(view.against.get((goal.relation, goal.subject, end), ()))
            for ancestor in ancestors:
                contrary.extend(view.against.get(("subclass", ancestor, end), ()))
            against = tuple(sorted(set(denials) | set(contrary)))
            applied = rules + (() if not path else (
                "membership_inheritance" if goal.relation == "member" else "subclass_transitivity",))
            work.step("APPLY_RULE" if path else "BIND_DIRECT_PREMISE",
                      conclusion=[goal.relation, goal.subject, end],
                      rule=applied[-1] if path else "direct_assertion",
                      support=list(used), opposing_support=list(against))
            if not against:
                reachable.setdefault(end, used)
            if end == goal.target:
                witness = {"premises": list(used), "opposing_premises": list(against), "rules": list(applied)}
                if not against and clean is None:
                    clean = witness
                elif against and contested is None:
                    contested = witness
            if end in ancestors:
                work.step("PRUNE_CYCLE", class_id=end)
            else:
                queue.append(("subclass", end, used, against, applied, ancestors + (end,)))
    if clean:
        status, witness = ("conflict" if direct else "true"), clean
    elif limited:
        status, witness = "unknown", None
    elif contested:
        status, witness = "conflict", contested
    elif direct:
        status, witness = "false", None
    else:
        status, witness = "unknown", None
    return {"status": status, "witness": witness, "direct_denials": direct,
            "truncated": limited, "visited": len(visited)}, reachable


def solve(rows: Iterable[dict], scope: str, ledger_id: str, query: Goal,
          *, limits: ReasoningLimits | None = None, find_bridge: bool = False) -> dict:
    """Plan -> inspect -> derive -> challenge -> optionally propose one subgoal.

    All limits are work limits, not just trace truncation. If the search is
    incomplete it reports UNKNOWN, never a proof that no path exists. A bridge
    is a question candidate tested in a disposable view, not accepted knowledge.
    """
    if type(find_bridge) is not bool:
        raise ValueError("explicit bridge-search flag required")
    cfg = limits or ReasoningLimits()
    view = EvidenceView(rows, scope, ledger_id, cfg)
    work = _Work(cfg.max_work)
    result = {"status": "unknown", "witness": None, "direct_denials": [],
              "truncated": True, "visited": 0}
    proposed = None
    candidates = []
    main_complete = False
    bridge_limited = False
    try:
        work.step("PIN_CONTEXT", scope=scope, ledger_id=ledger_id,
                  evidence_sha256=view.sha256, indexed_records=len(view.rows), goal=query.triple())
        region, reverse_limited = _target_region(view, query.target, work, cfg)
        if reverse_limited:
            # A pruned reverse frontier is not a complete relevance filter.
            work.step("REVERSE_DEPTH_LIMIT", target=query.target)
            raise _Budget()
        work.step("PLAN_RELEVANT_REGION", classes=len(region), rule_families=2)
        result, _ = _search(view, query, work, cfg, region)
        main_complete = not result["truncated"]
        work.step("CHECK_RESULT", status=result["status"], witness=result["witness"],
                  direct_denials=result["direct_denials"], complete=main_complete)
        if find_bridge and main_complete and result["status"] == "unknown":
            # Diagnose existing clean anchors on the other side of the gap.
            _unused, left = _search(view, query, work, cfg, None)
            if _unused["truncated"]:
                bridge_limited = True
            else:
                pairs = []
                for child, before in left.items():
                    for parent, after in region.items():
                        work.step("CONSIDER_MISSING_LINK", triple=["subclass", child, parent])
                        if child == parent or view.against.get(("subclass", child, parent)):
                            continue
                        if any(e["target"] == parent for e in view.adj.get(("subclass", child), ())):
                            continue
                        # Prefer using evidence on both sides, then the shortest
                        # supported route. This is a utility rule, not truth odds.
                        score = (-int(bool(before)) - int(bool(after)), len(before) + len(after), child, parent)
                        pairs.append((score, child, parent, before, after))
                for rank, child, parent, before, after in sorted(pairs)[:cfg.max_bridges]:
                    work.step("TRIAL_ASSUMPTION", triple=["subclass", child, parent],
                              accepted_as_fact=False, structural_rank=list(rank))
                    hypothesis_id = "hypothesis:" + fingerprint([scope, query.triple(), child, parent])
                    if hypothesis_id in view.by_id:
                        continue
                    assumption = {"kind": "assert", "scope": scope, "relation": "subclass",
                                  "subject": child, "target": parent, "positive": True,
                                  "evidence_id": hypothesis_id}
                    # Reserve one additional view row without weakening the base
                    # input capacity. No assumption is written to caller state.
                    if len(view.rows) == cfg.max_records:
                        bridge_limited = True
                        break
                    sandbox = EvidenceView((*view.rows, assumption), scope, ledger_id, cfg)
                    trial, _ = _search(sandbox, query, work, cfg, None)
                    candidate = {"goal": ["subclass", child, parent], "rank": list(rank),
                                 "support_before": list(before), "support_after": list(after),
                                 "hypothetical_status": trial["status"], "hypothetical_truncated": trial["truncated"],
                                 "accepted_as_fact": False}
                    candidates.append(candidate)
                    if trial["status"] == "true" and not trial["truncated"]:
                        proposed = candidate
                        work.step("SELECT_CLARIFICATION", goal=candidate["goal"],
                                  reason="one unaccepted premise closes an evidenced gap")
                        break
    except _Budget:
        if not main_complete:
            result = {"status": "unknown", "witness": None,
                      "direct_denials": sorted(view.against.get(tuple(query.triple()), ())),
                      "truncated": True, "visited": sum(s["operation"] == "VISIT_SUBGOAL" for s in work.steps)}
        else:
            bridge_limited = True
        proposed = None
    used = set(result["direct_denials"])
    if result["witness"]:
        used.update(result["witness"]["premises"])
        used.update(result["witness"]["opposing_premises"])
    output = {"schema": SCHEMA, "scope": scope, "ledger_id": ledger_id,
              "evidence_sha256": view.sha256, "query": query.triple(),
              "limits": asdict(cfg), "find_bridge": find_bridge, **result,
              "evidence": [view.by_id[e] for e in sorted(used)],
              "steps": work.steps, "work_used": work.count,
              "bridge_candidates": candidates, "bridge": proposed,
              "bridge_search_limited": bridge_limited,
              "truth_basis": "relative_to_user_taught_premises_not_independent_world_verification"}
    output["sha256"] = fingerprint(output)
    return output


def verify(receipt: dict, rows: Iterable[dict], scope: str, ledger_id: str) -> None:
    """Replay the computation, not just the receipt's self-reported checksum."""
    if not isinstance(receipt, dict) or receipt.get("scope") != scope:
        raise ValueError("reasoning receipt belongs to another context")
    q = receipt.get("query")
    if not isinstance(q, list) or len(q) != 3:
        raise ValueError("invalid reasoning query")
    if type(receipt.get("find_bridge")) is not bool:
        raise ValueError("invalid reasoning mode")
    expected = solve(rows, scope, ledger_id, Goal(*q),
                     limits=ReasoningLimits(**receipt.get("limits", {})),
                     find_bridge=receipt["find_bridge"])
    if receipt != expected:
        raise ValueError("reasoning steps, evidence or result changed")
