"""JSON-backed construction graph traversal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import ResponseCandidate, ResponsePlan


class _SafeSlots(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


class ConstructionGraph:
    """Traverse act nodes and emit only constructions allowed by hard gates."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        source = Path(path) if path else Path(__file__).with_name("data") / "constructions.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported construction graph version")
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("construction graph requires a nodes list")
        self.nodes: Dict[str, Mapping[str, Any]] = {}
        for node in nodes:
            node_id = str(node.get("id", ""))
            if not node_id or node_id in self.nodes:
                raise ValueError(f"invalid or duplicate construction node: {node_id!r}")
            self.nodes[node_id] = node
        if "root" not in self.nodes:
            raise ValueError("construction graph requires root node")

    def traverse(self, plan: ResponsePlan) -> Tuple[ResponseCandidate, ...]:
        results: List[ResponseCandidate] = []
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node is None:
                return
            node_type = node.get("type")
            if node_type == "root":
                for child in node.get("children", []):
                    visit(str(child))
                return
            if node_type == "act":
                if node.get("act") != plan.act:
                    return
                for child in node.get("children", []):
                    visit(str(child))
                return
            if node_type != "construction":
                return
            candidate = self._render_node(node, plan)
            if candidate is not None:
                results.append(candidate)

        visit("root")
        results.sort(key=lambda item: item.candidate_id)
        return tuple(results)

    def _render_node(
        self, node: Mapping[str, Any], plan: ResponsePlan
    ) -> Optional[ResponseCandidate]:
        allowed_registers = set(node.get("register", ["neutral", "casual"]))
        allowed_severity = set(node.get("severity", ["low", "moderate", "high", "critical"]))
        tags = tuple(str(tag) for tag in node.get("tags", []))
        tag_set = set(tags)
        if plan.gate.register not in allowed_registers:
            return None
        if plan.gate.severity not in allowed_severity:
            return None
        if set(plan.required_tags) - tag_set:
            return None
        if set(plan.forbidden_tags) & tag_set:
            return None
        if set(plan.gate.locked_pools) & tag_set:
            return None
        required_slots = [str(slot) for slot in node.get("required_slots", [])]
        if any(slot not in plan.slots or not str(plan.slots[slot]).strip() for slot in required_slots):
            return None
        reference = str(plan.slots.get("reference", "")).lower()
        node_id = str(node.get("id", ""))
        if node_id == "probe_object" and reference not in {"it", "this", "that"}:
            return None
        if node_id == "probe_person" and reference in {"it", "this", "that"}:
            return None
        try:
            text = str(node["template"]).format_map(_SafeSlots(plan.slots))
        except (KeyError, ValueError):
            return None
        text = " ".join(text.split()).replace(" .", ".").replace(" ?", "?")
        return ResponseCandidate(
            candidate_id=str(node["id"]),
            text=text,
            tags=tags,
            semantic_signature=f"act:{plan.act}",
        )
