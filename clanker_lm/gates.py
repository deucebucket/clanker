"""Contextual gating, response-act planning, and collision masking."""

from __future__ import annotations

from typing import List, Optional

from . import lexicon
from .database import LanguageStore
from .memory import ConversationMemory
from .model import AffectReading, AnswerStatus, GateDecision, ParseResult
from .response_policy import ResponseActPlanner


class ContextGate:
    """Convert parsed/affective context into legal response pools and acts."""

    def __init__(self, store: LanguageStore) -> None:
        self.store = store
        self.response_policy = ResponseActPlanner()

    def decide(
        self,
        text: str,
        parse: ParseResult,
        affect: AffectReading,
        memory: ConversationMemory,
        *,
        answer_status: Optional[AnswerStatus] = None,
    ) -> GateDecision:
        words = [token.norm for token in lexicon.tokenize(text, include_punctuation=False)]
        casual = any(word in lexicon.CASUAL_MARKERS for word in words)
        profanity = any(word in {"fuck", "fucking", "shit", "damn", "pissed", "ass", "hell"} for word in words)
        familial = any(word in lexicon.RELATIONS for word in words)
        vector = affect.vector
        structures = set(affect.structures)

        severity_level = 0
        rationale: List[str] = []
        if vector.u >= 200 or (vector.v <= 50 and vector.g <= 45):
            severity_level = 3
            rationale.append("critical VADUGWI region")
        elif vector.u >= 120 or vector.g <= 75 or structures & {
            "FAREWELL",
            "METHOD_ACQUISITION",
            "SELF_REMOVAL",
            "NO_EXIT",
            "CRISIS",
        }:
            severity_level = 2
            rationale.append("high urgency/gravity or crisis structure")
        elif vector.v < 105 or vector.a > 175:
            severity_level = 1
            rationale.append("negative or high-arousal input")

        # Literal semantic severity supplies a hard floor.  A boundary affect
        # read may refine or raise this floor but cannot erase an explicit
        # severe family disclosure.
        severe_family_terms = {
            "sick",
            "ill",
            "dying",
            "died",
            "dead",
            "missing",
            "hospital",
            "cancer",
            "hurt",
            "injured",
            "overdose",
            "suicide",
            "killed",
        }
        severe_family_hit = familial and any(word in severe_family_terms for word in words)
        if severe_family_hit:
            severity_level = max(severity_level, 2)
            rationale.append("explicit severe family content sets a high-severity floor")
        elif familial and severity_level >= 1:
            rationale.append("familial relevance raises response care without severity escalation")

        if casual or profanity:
            initial_register = "casual"
        elif vector.v >= 170 and severity_level == 0:
            initial_register = "positive"
        else:
            initial_register = "neutral"

        policy = self.response_policy.plan(
            text,
            parse,
            affect,
            answer_status=answer_status,
            initial_severity={0: "low", 1: "moderate", 2: "high", 3: "critical"}[severity_level],
            initial_register=initial_register,
        )
        severity_level = self.response_policy.SEVERITY_TO_LEVEL[policy.severity]
        register = policy.register
        response_act = policy.response_act
        rationale.extend(policy.rationale)

        masking = casual and severity_level >= 2
        if masking:
            register = "casual"
            rationale.append("casual register collides with severe content")

        requires_probe = bool(parse.unresolved) or response_act in {"probe", "safety_probe"}

        feature_map = {
            "severity_level": severity_level,
            "register": "casual" if casual else register,
            "masking": masking,
            "familial": familial,
            "profanity": profanity,
            "response_act": response_act,
        }
        locked: List[str] = []
        allowed: List[str] = []
        for rule in self.store.applicable_gate_rules(feature_map):
            for pool in rule["lock_pools"]:
                if pool not in locked:
                    locked.append(pool)
            for pool in rule["allow_pools"]:
                if pool not in allowed:
                    allowed.append(pool)
            rationale.append(f"gate:{rule['id']}")

        if answer_status in {
            AnswerStatus.UNKNOWN,
            AnswerStatus.CONFLICT,
            AnswerStatus.UNSUPPORTED,
            AnswerStatus.MISSING_REFERENCE,
            AnswerStatus.AMBIGUOUS_REFERENCE,
        }:
            if "humor" not in locked:
                locked.append("humor")
            rationale.append("uncertainty locks humor")

        severity = {0: "low", 1: "moderate", 2: "high", 3: "critical"}[severity_level]
        return GateDecision(
            register=register,
            severity=severity,
            masking=masking,
            locked_pools=locked,
            allowed_pools=allowed,
            response_act=response_act,
            max_sentences=policy.max_sentences,
            requires_probe=requires_probe,
            rationale=rationale,
        )

    @staticmethod
    def severity_number(decision: GateDecision) -> int:
        return {"low": 0, "moderate": 1, "high": 2, "critical": 3}.get(decision.severity, 0)
