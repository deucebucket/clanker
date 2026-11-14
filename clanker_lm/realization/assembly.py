"""Assembly responsibilities; extracted without changing decision rules."""

from __future__ import annotations
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple
from ..database import Atom
from ..model import CandidateResponse
from .types import Part

class AssemblyComponent:
    """Assembly behavior composed by the stable public SurfaceRealizer API."""

    def _rule_plan(self, route: str, features: Optional[Mapping[str, Any]] = None) -> List[str]:
        rules = self.store.grammar_rules(route, features=features)
        if not rules:
            return [f"ROUTE:{route}", "RULE:BUILTIN_FAIL_CLOSED"]
        rule = rules[0]
        return [f"ROUTE:{route}", f"RULE:{rule.rule_id}", *rule.children]

    def _atom(self, atom_id: str, fallback: str = "") -> Atom:
        atom = self.store.atom_by_id(atom_id)
        if atom is None:
            # A missing atom is a configuration error, not permission to pull a
            # memorized sentence from code or improvise with a model.
            raise RuntimeError(f"Required language atom is missing: {atom_id}")
        return atom

    def _atoms(
        self,
        category: str,
        *,
        register: str,
        preferred: Sequence[str] = (),
    ) -> List[Atom]:
        by_id = [self.store.atom_by_id(item) for item in preferred]
        selected = [item for item in by_id if item is not None]
        if selected:
            return selected
        return self.store.atom_candidates(category, register=register)

    def _candidate(
        self,
        parts: Sequence[Part],
        *,
        candidate_id: str,
        semantic_plan: Sequence[str],
        priority: int,
    ) -> CandidateResponse:
        text, atom_ids = self._compose(parts)
        return CandidateResponse(
            text=text,
            construction_id=candidate_id,
            semantic_valid=True,
            semantic_reason="composed from a closed semantic plan",
            priority=priority,
            atom_ids=atom_ids,
            semantic_plan=list(semantic_plan),
        )

    @classmethod
    def _compose(cls, parts: Sequence[Part]) -> Tuple[str, List[str]]:
        surfaces: List[str] = []
        atom_ids: List[str] = []
        for part in parts:
            if isinstance(part, Atom):
                surface = part.surface
                atom_ids.append(part.atom_id)
            else:
                surface = str(part)
            surface = re.sub(r"\s+", " ", surface).strip()
            if surface:
                surfaces.append(surface)
        if not surfaces:
            return "", atom_ids
        text = ""
        closing = {".", ",", "?", "!", ":", ";"}
        opening = {"(", "[", "{", "“", "\""}
        for surface in surfaces:
            if not text:
                text = surface
            elif surface in closing:
                text = text.rstrip() + surface
            elif text[-1:] in opening:
                text += surface
            else:
                text += " " + surface
        text = re.sub(r"\s+", " ", text).strip()
        # Capitalization is a grammatical operation applied at every sentence
        # boundary; it is not stored in a response template.
        text = re.sub(
            r"(^|[.!?]\s+)([a-z])",
            lambda match: match.group(1) + match.group(2).upper(),
            text,
        )
        return cls._capitalize(text), atom_ids

    @staticmethod
    def _finish_clause(text: str, capitalize: bool) -> str:
        text = re.sub(r"\s+", " ", text).strip().rstrip(".?!")
        return AssemblyComponent._capitalize(text) if capitalize else text

    @staticmethod
    def _capitalize(text: str) -> str:
        if not text:
            return text
        return text[0].upper() + text[1:]

    @staticmethod
    def join_phrases(items: Sequence[str], *, conjunction: str = "and") -> str:
        clean = [item for item in items if item]
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} {conjunction} {clean[1]}"
        return ", ".join(clean[:-1]) + f", {conjunction} {clean[-1]}"
