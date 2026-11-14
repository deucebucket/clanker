"""Task-gated activation over an ALREADY AUTHORIZED graph view.

This module selects references, not truth or identity. A context-reference edge
may expose a concept label without recursively activating the concept's other
relations. It never supplies a proof premise. The caller must enforce account
access BEFORE constructing this index; a scope label is not authentication.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Iterable


SCHEMA = "clanker-activation-v1"
MAX_INDEX_NODES = 6000
MAX_INDEX_EDGES = 18000


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def identifier(value: str) -> None:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value.encode("utf-8")) > 512
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError("invalid activation identifier")


class Plane(str, Enum):
    PERSONAL = "personal"
    SEMANTIC = "semantic"
    SIMULATION = "simulation"
    AUXILIARY = "auxiliary"


class Task(str, Enum):
    PERSONAL_RECALL = "personal_recall"
    CONCEPT_EXPLANATION = "concept_explanation"
    CROSS_DOMAIN_COMPARISON = "cross_domain_comparison"


class LinkKind(str, Enum):
    ASSOCIATION = "association"
    CONTEXT_REFERENCE = "context_reference"


@dataclass(frozen=True)
class Node:
    key: str
    plane: Plane

    def __post_init__(self) -> None:
        identifier(self.key)
        if not isinstance(self.plane, Plane):
            raise ValueError("a typed memory plane is required")


@dataclass(frozen=True)
class Link:
    key: str
    source: str
    target: str
    kind: LinkKind = LinkKind.ASSOCIATION

    def __post_init__(self) -> None:
        for value in (self.key, self.source, self.target):
            identifier(value)
        if not isinstance(self.kind, LinkKind):
            raise ValueError("a typed link kind is required")


@dataclass(frozen=True)
class Policy:
    """Trusted adapter configuration, NOT a permission claimed in user text.

    Tasks describe retrieval intent; purpose records the independently chosen
    session mode. This first kernel does not implement a login or teaching policy.
    """
    context: str
    purpose: str = "private_chat"
    revision: int = 1
    allowed_tasks: tuple[Task, ...] = (Task.PERSONAL_RECALL,)

    def __post_init__(self) -> None:
        identifier(self.context)
        if self.purpose not in {"private_chat", "public_chat", "study", "teaching", "evaluation"}:
            raise ValueError("unsupported activation purpose")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("invalid policy revision")
        if (type(self.allowed_tasks) is not tuple or not self.allowed_tasks
                or len(set(self.allowed_tasks)) != len(self.allowed_tasks)
                or any(not isinstance(t, Task) for t in self.allowed_tasks)):
            raise ValueError("invalid allowed task set")


@dataclass(frozen=True)
class Request:
    roots: tuple[str, ...]
    task: Task
    max_depth: int = 3
    max_nodes: int = 150
    max_edges: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.task, Task):
            raise ValueError("typed task required")
        if (type(self.roots) is not tuple or not 1 <= len(self.roots) <= 8
                or len(set(self.roots)) != len(self.roots)):
            raise ValueError("invalid activation roots")
        for root in self.roots:
            identifier(root)
        for value, low, high in ((self.max_depth, 0, 8), (self.max_nodes, 1, 300),
                                 (self.max_edges, 1, 4096)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("invalid activation budget")
        if len(self.roots) > self.max_nodes:
            raise ValueError("roots exceed node budget")


class ActivationIndex:
    """Immutable bounded metadata index; graph payloads remain in their store."""

    def __init__(self, nodes: Iterable[Node], links: Iterable[Link], *,
                 context: str, content_generation: str):
        identifier(context)
        identifier(content_generation)
        ns: dict[str, Node] = {}
        es: dict[str, Link] = {}
        for n in nodes:
            if not isinstance(n, Node) or n.key in ns or len(ns) >= MAX_INDEX_NODES:
                raise ValueError("invalid/oversized activation nodes")
            ns[n.key] = n
        adjacency: dict[str, list[Link]] = {key: [] for key in ns}
        for e in links:
            if (not isinstance(e, Link) or e.key in es or len(es) >= MAX_INDEX_EDGES
                    or e.source not in ns or e.target not in ns):
                raise ValueError("invalid/oversized activation links")
            if e.kind == LinkKind.CONTEXT_REFERENCE and not (
                    ns[e.source].plane in {Plane.PERSONAL, Plane.SIMULATION}
                    and ns[e.target].plane == Plane.SEMANTIC):
                raise ValueError("context references require personal/simulation -> semantic")
            es[e.key] = e
            adjacency[e.source].append(e)
            if e.source != e.target and e.kind != LinkKind.CONTEXT_REFERENCE:
                adjacency[e.target].append(e)
        self.nodes = MappingProxyType(ns)
        self.links = MappingProxyType(es)
        self.adjacency = MappingProxyType({key: tuple(sorted(edges, key=lambda e: e.key))
                                          for key, edges in adjacency.items()})
        self.context = context
        self.generation = fingerprint({"content": content_generation,
            "nodes": [asdict(ns[k]) for k in sorted(ns)],
            "links": [asdict(es[k]) for k in sorted(es)]})

    def _allowed(self, source: str, edge: Link, task: Task) -> tuple[str | None, str]:
        target = edge.target if source == edge.source else edge.source
        a, b = self.nodes[source].plane, self.nodes[target].plane
        if edge.kind == LinkKind.CONTEXT_REFERENCE:
            if source != edge.source:
                return None, "context_reference_is_not_a_reverse_memory_lookup"
            if task == Task.PERSONAL_RECALL:
                return "reference_only", "context_label_only_no_expansion"
            if task == Task.CROSS_DOMAIN_COMPARISON:
                return "active", "explicit_cross_domain_comparison"
            return None, "concept_explanation_does_not_expand_personal_context"
        if a != b:
            return None, "cross_plane_association_requires_a_typed_bridge"
        allowed_planes = ({Plane.PERSONAL} if task == Task.PERSONAL_RECALL else
                          {Plane.SEMANTIC} if task == Task.CONCEPT_EXPLANATION else
                          {Plane.PERSONAL, Plane.SEMANTIC})
        if a not in allowed_planes:
            return None, "plane_outside_task"
        return "active", "same_plane_association"

    def select(self, request: Request, policy: Policy) -> dict:
        if not isinstance(request, Request) or not isinstance(policy, Policy):
            raise TypeError("typed request and policy required")
        # Validate access configuration before even looking up root existence.
        if policy.context != self.context or request.task not in policy.allowed_tasks:
            raise PermissionError("activation is outside the bound policy")
        allowed_roots = ({Plane.PERSONAL} if request.task == Task.PERSONAL_RECALL else
                         {Plane.SEMANTIC} if request.task == Task.CONCEPT_EXPLANATION else
                         {Plane.PERSONAL, Plane.SEMANTIC})
        if any(r not in self.nodes or self.nodes[r].plane not in allowed_roots for r in request.roots):
            raise ValueError("root unavailable for this task")
        active = {r: 0 for r in sorted(request.roots)}
        references: dict[str, int] = {}
        queue = deque((r, 0) for r in sorted(request.roots))
        steps: list[dict] = []
        reached_edges: set[str] = set()
        stops: set[str] = set()
        expanded = 0
        while queue:
            source, depth = queue.popleft()
            if active.get(source) != depth:
                continue
            expanded += 1
            for edge in self.adjacency[source]:
                if len(steps) >= request.max_edges:
                    stops.add("edge_budget")
                    queue.clear()
                    break
                target = edge.target if source == edge.source else edge.source
                mode, reason = self._allowed(source, edge, request.task)
                step = {"source": source, "edge": edge.key, "target": target,
                        "depth": depth + 1, "decision": "withheld", "reason": reason}
                steps.append(step)
                if mode is None:
                    continue
                if depth >= request.max_depth:
                    if target not in active and target not in references:
                        stops.add("depth_budget")
                    step["reason"] = "depth_budget"
                    continue
                existing = active.get(target) if mode == "active" else active.get(target, references.get(target))
                if existing is not None and existing <= depth + 1:
                    reached_edges.add(edge.key)
                    step.update(decision="already_selected")
                    continue
                if target not in active and target not in references and len(active) + len(references) >= request.max_nodes:
                    stops.add("node_budget")
                    step["reason"] = "node_budget"
                    continue
                reached_edges.add(edge.key)
                step["decision"] = mode
                if mode == "reference_only":
                    references[target] = depth + 1
                else:
                    references.pop(target, None)
                    active[target] = depth + 1
                    queue.append((target, depth + 1))
        result = {"schema": SCHEMA, "context": self.context,
                  "graph_generation": self.generation,
                  "policy": asdict(policy), "request": asdict(request),
                  "active": sorted(active), "reference_only": sorted(references),
                  "traversed_edges": sorted(reached_edges), "steps": steps,
                  "status": "incomplete" if stops else "complete",
                  "stop_reasons": sorted(stops),
                  "work": {"expanded_nodes": expanded, "inspected_edges": len(steps),
                           "selected_nodes": len(active) + len(references)},
                  "context_edges_are_proof_premises": False,
                  "authentication_or_identity_inferred": False}
        result["digest"] = fingerprint(result)
        return result

    def verify(self, receipt: dict, request: Request, policy: Policy) -> None:
        # Rerun rather than trust a rehashed result. The receipt is not a token
        # granting access or a durable authority to read an older memory version.
        if not isinstance(receipt, dict) or canonical(receipt) != canonical(self.select(request, policy)):
            raise ValueError("activation receipt does not reproduce under current bindings")
