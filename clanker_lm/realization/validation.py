"""Validation responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import Tuple
from .. import lexicon
from ..model import AnswerContract, AnswerStatus, RefKind

class ValidationComponent:
    """Validation behavior composed by the stable public SurfaceRealizer API."""

    def _validate_surface(self, text: str, contract: AnswerContract) -> Tuple[bool, str]:
        lower = text.lower()
        if not text.strip():
            return False, "empty composition"
        if contract.status == AnswerStatus.ANSWERED and contract.values:
            value = self.render_ref(contract.values[0], definite=False).lower()
            ref = contract.values[0]
            entity = self.memory.get_entity(ref.key) if ref.kind == RefKind.ENTITY else None
            aliases = [value]
            if (
                ref.kind == RefKind.EVENT
                and contract.required_slots.get("gerund") == "true"
            ):
                relation = self._gerund_relation_from_contract(contract)
                complement = (
                    self.memory.get_event(relation.complement_event_id)
                    if relation is not None
                    else None
                )
                if complement is not None and ref.key == complement.event_id:
                    aliases.extend(
                        [
                            lexicon.gerund_form(complement.predicate).lower(),
                            self._render_gerund_event(
                                complement,
                                relation.controller_entity_id,
                            ).lower(),
                        ]
                    )
            if entity:
                aliases.extend([entity.canonical_name.lower(), *entity.aliases])
            if not any(alias and alias in lower for alias in aliases):
                return False, "answer surface omits the bound value"
        if contract.status == AnswerStatus.TRUE and not lower.startswith(("yes", "yeah")):
            return False, "truth answer lacks an affirmation atom"
        if contract.status == AnswerStatus.FALSE and not lower.startswith(("no", "nope")):
            return False, "false answer lacks a denial atom"
        if contract.status == AnswerStatus.UNKNOWN and not any(marker in lower for marker in ("don't know", "do not know", "haven't told", "cannot determine")):
            return False, "unknown answer asserts beyond evidence"
        if contract.status in {AnswerStatus.MISSING_REFERENCE, AnswerStatus.AMBIGUOUS_REFERENCE, AnswerStatus.MULTIPLE_MATCHES, AnswerStatus.LEXICAL_PROBE} and not text.rstrip().endswith("?"):
            return False, "probe composition is not interrogative"
        if contract.status == AnswerStatus.CONFLICT and "conflict" not in lower:
            return False, "conflict composition hides disagreement"
        return True, "semantic contract preserved by compositional grammar"
