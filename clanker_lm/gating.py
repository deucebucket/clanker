"""Context gates and response-act planning."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .memory import ConversationMemory
from .models import (
    AffectReading,
    EntityKind,
    GateProfile,
    ParsedUtterance,
    ResponsePlan,
    SemanticRole,
    SpeechAct,
    ValueKind,
)
from .normalize import normalize_alias


class GateEngine:
    """Lock socially invalid language pools before candidate generation."""

    def evaluate(self, parsed: ParsedUtterance, affect: AffectReading) -> GateProfile:
        structure_set = set(affect.structures)
        severity_score = parsed.severity_score
        if affect.vector.u >= 190 or (
            affect.vector.v <= 55 and affect.vector.g <= 55 and affect.vector.u >= 120
        ):
            severity_score = max(severity_score, 0.96)
        elif affect.vector.u >= 110 or (
            affect.vector.v <= 75 and affect.vector.g <= 80
        ):
            severity_score = max(severity_score, 0.75)

        if severity_score >= 0.9:
            severity = "critical"
        elif severity_score >= 0.65:
            severity = "high"
        elif severity_score >= 0.3:
            severity = "moderate"
        else:
            severity = "low"

        register = "casual" if parsed.register_score >= 0.35 else "neutral"
        masking = bool(
            {"MASKING", "BRAVADO"} & structure_set
            or (register == "casual" and severity in {"high", "critical"})
        )

        locked: set[str] = set()
        required: List[str] = []
        forbidden: set[str] = {"unsupported_claim", "invented_entity"}
        rationale: List[str] = []

        if parsed.unresolved_references:
            required.append("context_probe")
            locked.update({"factual_assertion", "assumption"})
            rationale.append("a reference has no unique antecedent")

        if parsed.speech_act in {SpeechAct.QUESTION, SpeechAct.SOCIAL_CHECKIN}:
            required.append("answer_or_explicit_unknown")
            forbidden.add("topic_shift")
            rationale.append("the input opens a typed information slot")

        if severity == "critical":
            locked.update(
                {
                    "humor",
                    "playful",
                    "slang",
                    "celebration",
                    "minimization",
                    "casual_dismissal",
                }
            )
            required.extend(["acknowledge_severity", "safety_check"])
            forbidden.update({"premature_advice", "false_reassurance"})
            rationale.append("critical severity requires direct safety-oriented language")
        elif severity == "high":
            locked.update({"humor", "playful", "celebration", "minimization"})
            required.append("acknowledge_severity")
            forbidden.update({"premature_advice", "false_reassurance"})
            rationale.append("high-severity content blocks levity and minimization")
        elif severity == "moderate":
            locked.add("high_drama")
            forbidden.add("minimization")
            rationale.append("moderate severity permits warmth but not dismissal")

        if masking:
            locked.update({"formal", "clinical", "performative_empathy", "humor"})
            required.append("respect_masking_distance")
            rationale.append("casual delivery collides with severe content")
        elif register == "casual" and severity == "low":
            locked.update({"formal", "clinical", "high_drama"})
            rationale.append("casual low-severity input should not trigger formal language")

        if affect.vector.v < 100 and "context_probe" not in required:
            required.extend(["acknowledge", "invite_elaboration"])
        elif affect.vector.v > 170 and severity == "low":
            required.append("celebrate")

        return GateProfile(
            register=register,
            severity=severity,
            collision_masking=masking,
            locked_pools=tuple(sorted(locked)),
            required_acts=tuple(dict.fromkeys(required)),
            forbidden_acts=tuple(sorted(forbidden)),
            rationale=tuple(rationale),
        )


class ResponsePlanner:
    """Convert a parsed disclosure into a graph-traversal response plan."""

    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory

    def plan_statement(
        self,
        parsed: ParsedUtterance,
        gates: GateProfile,
        affect: AffectReading,
    ) -> ResponsePlan:
        if parsed.unresolved_references:
            unresolved = parsed.unresolved_references[0]
            reference = unresolved.surface.lower()
            if unresolved.compatible_entity_ids:
                descriptions = [
                    self.memory.describe_entity(entity_id, prefer_pronoun=False)
                    for entity_id in unresolved.compatible_entity_ids[:3]
                    if entity_id in self.memory.entities
                ]
                options = " or ".join(descriptions)
            else:
                options = ""
            return ResponsePlan(
                act="context_probe",
                slots={"reference": reference, "options": options},
                gate=gates,
                target_mode="clarify",
                required_tags=("context_probe",),
                forbidden_tags=("factual_assertion",),
            )

        if parsed.speech_act == SpeechAct.GREETING:
            return ResponsePlan(
                act="greeting",
                slots={},
                gate=gates,
                target_mode="social",
            )

        frame = parsed.frame
        if frame is None:
            return ResponsePlan(
                act="clarify_unknown",
                slots={},
                gate=gates,
                target_mode="clarify",
            )

        slots = self._frame_slots(parsed)
        attribute = normalize_alias(slots.get("attribute", ""))
        predicate = frame.predicate

        if gates.severity == "critical":
            return ResponsePlan(
                act="critical_safety",
                slots=slots,
                gate=gates,
                target_mode="safety",
                required_tags=("safety_check",),
            )
        if predicate == "be" and any(
            marker in attribute
            for marker in {"sick", "ill", "hospital", "cancer", "hurt", "injured"}
        ):
            return ResponsePlan(
                act=("masked_illness" if gates.collision_masking else "serious_illness"),
                slots=slots,
                gate=gates,
                target_mode="support",
                required_tags=("acknowledge", "inquiry"),
            )
        agent = frame.roles.get(SemanticRole.AGENT)
        if (
            predicate == "hurt"
            and agent is not None
            and agent.kind == ValueKind.ENTITY
            and self.memory.entities.get(agent.value)
            and self.memory.entities[agent.value].kind == EntityKind.BODY_PART
        ):
            return ResponsePlan(
                act="pain_check",
                slots=slots,
                gate=gates,
                target_mode="support",
                required_tags=("acknowledge", "inquiry"),
            )
        patient = frame.roles.get(SemanticRole.PATIENT)
        if predicate in {"piss_off", "upset", "hurt", "hit", "cheat"} and (
            patient is None or patient.value == "user"
        ):
            return ResponsePlan(
                act="conflict_support",
                slots=slots,
                gate=gates,
                target_mode="deescalate",
                required_tags=("acknowledge", "inquiry"),
            )
        if affect.vector.v >= 170 and gates.severity == "low":
            return ResponsePlan(
                act="celebrate",
                slots=slots,
                gate=gates,
                target_mode="celebrate",
                required_tags=("celebrate",),
            )
        if affect.vector.v < 105:
            return ResponsePlan(
                act="general_support",
                slots=slots,
                gate=gates,
                target_mode="support",
                required_tags=("acknowledge", "inquiry"),
            )
        return ResponsePlan(
            act="fact_ack",
            slots=slots,
            gate=gates,
            target_mode="neutral",
            required_tags=("acknowledge",),
        )

    def _frame_slots(self, parsed: ParsedUtterance) -> Dict[str, str]:
        assert parsed.frame is not None
        frame = parsed.frame
        slots: Dict[str, str] = {
            "repeat_phrase": " again" if frame.repeated or parsed.repeated else "",
            "again_word": "again" if frame.repeated or parsed.repeated else "",
            "predicate": frame.predicate.replace("_", " "),
        }
        agent = frame.roles.get(SemanticRole.AGENT)
        if agent is not None and agent.kind == ValueKind.ENTITY:
            slots["agent_display"] = self.memory.describe_entity(
                agent.value, prefer_pronoun=False
            )
            slots["agent_pronoun"] = self.memory.pronoun(agent.value, subject=True)
        else:
            slots["agent_display"] = "they"
            slots["agent_pronoun"] = "they"
        patient = frame.roles.get(SemanticRole.PATIENT)
        if patient is not None and patient.kind == ValueKind.ENTITY:
            slots["patient_display"] = self.memory.describe_entity(
                patient.value, subject=False, prefer_pronoun=True
            )
        attribute = frame.roles.get(SemanticRole.ATTRIBUTE)
        slots["attribute"] = attribute.display if attribute else "not doing well"
        return slots
