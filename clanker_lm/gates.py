"""Contextual gating and collision-masking rules for Clanker-LM."""

from __future__ import annotations

from typing import List, Optional

from . import lexicon
from .database import LanguageStore
from .memory import ConversationMemory
from .model import AffectReading, AnswerStatus, GateDecision, ParseResult, SpeechAct


class ContextGate:
    """Convert parsed/affective context into legal response pools."""

    def __init__(self, store: LanguageStore) -> None:
        self.store = store

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
        elif vector.u >= 120 or vector.g <= 75 or structures & {"FAREWELL", "METHOD_ACQUISITION", "SELF_REMOVAL", "NO_EXIT", "CRISIS"}:
            severity_level = 2
            rationale.append("high urgency/gravity or crisis structure")
        elif vector.v < 105 or vector.a > 175:
            severity_level = 1
            rationale.append("negative or high-arousal input")

        # Severity gates cannot depend exclusively on an affect backend's
        # aggregate coordinate.  The semantic content itself supplies a hard
        # floor for explicit severe family disclosures; otherwise a backend
        # boundary read could turn "my mom is really sick" into a glib low-
        # severity acknowledgment.  VADUGWI still controls finer movement and
        # can raise this floor to critical.
        severe_family_terms = {
            "sick", "ill", "dying", "died", "dead", "missing", "hospital",
            "cancer", "hurt", "injured", "overdose", "suicide", "killed",
        }
        severe_family_hit = familial and any(word in severe_family_terms for word in words)
        if severe_family_hit:
            severity_level = max(severity_level, 2)
            rationale.append("explicit severe family content sets a high-severity floor")
        elif familial and severity_level >= 1:
            rationale.append("familial relevance raises response care without severity escalation")

        masking = casual and severity_level >= 2
        if masking:
            rationale.append("casual register collides with severe content")

        if masking:
            register = "casual"
        elif casual or profanity:
            register = "casual"
        elif vector.v >= 170 and severity_level == 0:
            register = "positive"
        else:
            register = "neutral"

        requires_probe = bool(parse.unresolved)
        if requires_probe:
            response_act = "probe"
            rationale.append("unresolved reference halts normal equation")
        elif parse.speech_act == SpeechAct.ASK:
            response_act = "social" if parse.question and parse.question.social_convention else "answer"
        elif parse.speech_act in {SpeechAct.GREET, SpeechAct.SOCIAL}:
            response_act = "social"
        else:
            response_act = "acknowledge"

        if answer_status in {AnswerStatus.MISSING_REFERENCE, AnswerStatus.AMBIGUOUS_REFERENCE, AnswerStatus.MULTIPLE_MATCHES}:
            response_act = "probe"

        feature_map = {
            "severity_level": severity_level,
            "register": "casual" if casual else "neutral",
            "masking": masking,
            "familial": familial,
            "profanity": profanity,
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
            max_sentences=1 if severity_level == 0 and response_act == "answer" else 2,
            requires_probe=requires_probe,
            rationale=rationale,
        )

    @staticmethod
    def severity_number(decision: GateDecision) -> int:
        return {"low": 0, "moderate": 1, "high": 2, "critical": 3}.get(decision.severity, 0)
