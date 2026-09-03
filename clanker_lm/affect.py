"""Clanker affect adapter, target policy, and candidate back-solving."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import Any, Callable, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .answers import SemanticValidator
from .models import (
    AffectReading,
    AffectVector,
    AnswerContract,
    CandidateScore,
    GateProfile,
    ResponseCandidate,
)
from .normalize import HIGH_SEVERITY_MARKERS, PROFANITY, SEVERE_MARKERS, tokenize


class AffectAdapter(Protocol):
    def analyze(self, text: str) -> AffectReading:
        ...

    def transition(self, current: AffectVector, message: AffectVector) -> AffectVector:
        ...


class ClankerAffectAdapter:
    """Bridge the language layer to the existing VADUGWI kernel."""

    def __init__(self, personality: Any = None, *, strict: bool = False) -> None:
        self.personality = personality
        self.strict = strict
        try:
            from engine.pendulum import compute_vadug
            from engine.shared import VADUG
            from engine.solver import state_transition
        except ImportError:
            if strict:
                raise
            self._compute_vadug = None
            self._state_transition = None
            self._vadug_type = None
            self._fallback = RuleAffectAdapter()
        else:
            self._compute_vadug = compute_vadug
            self._state_transition = state_transition
            self._vadug_type = VADUG
            self._fallback = None

    @property
    def using_clanker(self) -> bool:
        return self._compute_vadug is not None

    def analyze(self, text: str) -> AffectReading:
        if self._compute_vadug is None:
            return self._fallback.analyze(text)
        result, meta = self._compute_vadug(text, personality=self.personality)
        structures = tuple(
            getattr(item, "pattern", str(item)) for item in meta.get("structures", [])
        )
        return AffectReading(
            vector=AffectVector(
                v=result.v,
                a=result.a,
                d=result.d,
                u=result.u,
                g=result.g,
                w=result.w,
                i=result.i,
            ),
            structures=structures,
            metadata={"engine": "clanker", "word_count": meta.get("word_count", 0)},
        )

    def transition(self, current: AffectVector, message: AffectVector) -> AffectVector:
        if self._state_transition is None or self._vadug_type is None:
            return self._fallback.transition(current, message)
        a = self._vadug_type(**current.to_dict())
        b = self._vadug_type(**message.to_dict())
        result = self._state_transition(a, b)
        return AffectVector(**result.__dict__)


class RuleAffectAdapter:
    """Dependency-free fallback used only when the Clanker kernel is absent."""

    POSITIVE = {
        "happy",
        "great",
        "love",
        "won",
        "success",
        "excited",
        "good",
        "awesome",
        "finally",
    }
    NEGATIVE = {
        "sad",
        "angry",
        "mad",
        "hate",
        "hurt",
        "sick",
        "ill",
        "pissed",
        "lost",
        "bad",
        "awful",
        "scared",
        "worried",
    }

    def analyze(self, text: str) -> AffectReading:
        items = [token.norm for token in tokenize(text)]
        pos = sum(1 for item in items if item in self.POSITIVE)
        neg = sum(1 for item in items if item in self.NEGATIVE)
        profane = sum(1 for item in items if item in PROFANITY)
        severe = any(item in HIGH_SEVERITY_MARKERS for item in items)
        moderate = any(item in SEVERE_MARKERS for item in items)
        v = 128 + min(80, pos * 28) - min(100, neg * 30)
        a = 128 + min(90, (pos + neg + profane) * 16 + text.count("!") * 10)
        u = 210 if severe else 80 if moderate else 0
        g = 45 if severe else 95 if moderate else 128
        d = 100 if neg else 145 if pos else 128
        w = 110 if neg else 145 if pos else 128
        i = 180 if "?" in text else 128
        return AffectReading(AffectVector(v, a, d, u, g, w, i), (), {"engine": "fallback"})

    def transition(self, current: AffectVector, message: AffectVector) -> AffectVector:
        values = {}
        for key in "vadugwi":
            a = getattr(current, key)
            b = getattr(message, key)
            values[key] = round(a * 0.6 + b * 0.4)
        return AffectVector(**values)


class TargetPolicy:
    """Calculate a desired post-response state, not desired wording."""

    def target(
        self,
        current: AffectVector,
        gates: GateProfile,
        mode: str,
    ) -> AffectVector:
        if mode == "safety":
            return AffectVector(
                v=min(128, max(current.v, 90)),
                a=max(115, min(current.a, 165)),
                d=max(current.d, 118),
                u=max(100, min(current.u, 190)),
                g=max(current.g, 95),
                w=max(current.w, 110),
                i=205,
            )
        if mode == "celebrate":
            return AffectVector(
                v=min(225, current.v + 18),
                a=min(210, current.a + 10),
                d=min(205, current.d + 10),
                u=max(0, int(current.u * 0.7)),
                g=min(220, current.g + 15),
                w=min(210, current.w + 12),
                i=190,
            )
        if mode in {"clarify", "factual", "neutral", "social"}:
            return AffectVector(
                v=self._move_toward(current.v, 140, 0.35),
                a=self._move_toward(current.a, 118, 0.30),
                d=self._move_toward(current.d, 145, 0.30),
                u=int(current.u * 0.70),
                g=self._move_toward(current.g, 135, 0.30),
                w=self._move_toward(current.w, 135, 0.20),
                i=178,
            )
        # Support/de-escalation.  Severe grief is not pushed into artificial joy.
        target_v = 112 if gates.severity in {"high", "critical"} else 145
        return AffectVector(
            v=self._move_toward(current.v, target_v, 0.42),
            a=self._move_toward(current.a, 108, 0.42),
            d=self._move_toward(current.d, 145, 0.38),
            u=int(current.u * 0.58),
            g=self._move_toward(current.g, 128, 0.42),
            w=self._move_toward(current.w, 145, 0.35),
            i=192,
        )

    @staticmethod
    def _move_toward(value: int, target: int, amount: float) -> int:
        return round(value + (target - value) * amount)


@dataclass(frozen=True)
class Selection:
    candidate: ResponseCandidate
    response_reading: AffectReading
    outcome: AffectVector
    scores: Tuple[CandidateScore, ...]


class CandidateScorer:
    """Back-solve candidate responses through the actual state transition."""

    _WEIGHTS: Mapping[str, float] = {
        "v": 1.25,
        "a": 1.0,
        "d": 0.9,
        "u": 1.15,
        "g": 1.0,
        "w": 1.0,
        "i": 0.8,
    }

    def __init__(
        self,
        affect: AffectAdapter,
        validator: Optional[SemanticValidator] = None,
        target_policy: Optional[TargetPolicy] = None,
    ) -> None:
        self.affect = affect
        self.validator = validator or SemanticValidator()
        self.target_policy = target_policy or TargetPolicy()

    def choose(
        self,
        candidates: Sequence[ResponseCandidate],
        *,
        current: AffectVector,
        gates: GateProfile,
        mode: str,
        contract: Optional[AnswerContract] = None,
    ) -> Selection:
        if not candidates:
            candidates = (
                ResponseCandidate(
                    "hard-fallback",
                    "I need a little more information.",
                    ("clarify", "no_claim"),
                    "fallback",
                ),
            )
        target = self.target_policy.target(current, gates, mode)
        scored: List[tuple[float, str, ResponseCandidate, AffectReading, AffectVector, CandidateScore]] = []
        locked = set(gates.locked_pools)
        for candidate in candidates:
            semantic_valid, semantic_reasons = self.validator.validate(candidate, contract)
            reading = self.affect.analyze(candidate.text)
            outcome = self.affect.transition(current, reading.vector)
            distance = self._distance(outcome, target)
            reasons = list(semantic_reasons)
            penalty = 0.0
            forbidden_tags = locked & set(candidate.tags)
            if forbidden_tags:
                semantic_valid = False
                reasons.append(f"locked tags: {', '.join(sorted(forbidden_tags))}")
            if gates.collision_masking and "formal" in candidate.tags:
                semantic_valid = False
                reasons.append("formal language breaks the masking gate")
            if gates.severity in {"high", "critical"} and reading.vector.v > 190:
                penalty += (reading.vector.v - 190) * 3
                reasons.append("excessively positive for severe context")
            if mode == "factual" and "factual" not in candidate.tags:
                penalty += 35.0
            if mode == "factual" and "direct" in candidate.tags:
                penalty -= 2.0
            if mode == "factual" and "concise" in candidate.tags:
                penalty += 1.0
            if mode == "clarify" and "concise" in candidate.tags:
                penalty -= 1.0
            total = distance + penalty
            hard_rejected = not semantic_valid
            score_record = CandidateScore(
                candidate_id=candidate.candidate_id,
                text=candidate.text,
                score=round(total, 6),
                semantic_valid=semantic_valid,
                outcome=outcome,
                reasons=tuple(reasons),
                hard_rejected=hard_rejected,
            )
            scored.append(
                (
                    total,
                    candidate.candidate_id,
                    candidate,
                    reading,
                    outcome,
                    score_record,
                )
            )

        eligible = [item for item in scored if not item[5].hard_rejected]
        if not eligible:
            details = "; ".join(
                f"{item[2].candidate_id}: {', '.join(item[5].reasons)}"
                for item in scored
            )
            raise RuntimeError(
                "No response candidate satisfied the semantic and contextual "
                f"hard gates. {details}"
            )

        eligible.sort(key=lambda item: (item[0], item[1]))
        winner = eligible[0]
        winner_id = winner[2].candidate_id
        all_scores = sorted(
            (
                replace(item[5], selected=item[2].candidate_id == winner_id)
                for item in scored
            ),
            key=lambda item: (item.hard_rejected, item.score, item.candidate_id),
        )
        return Selection(
            candidate=winner[2],
            response_reading=winner[3],
            outcome=winner[4],
            scores=tuple(all_scores),
        )

    def _distance(self, actual: AffectVector, target: AffectVector) -> float:
        total = 0.0
        for key, weight in self._WEIGHTS.items():
            delta = (getattr(actual, key) - getattr(target, key)) / 255.0
            total += weight * delta * delta
        return sqrt(total) * 100.0
