"""Adapter between Clanker-LM and the existing V8 VADUGWI engine."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .model import (
    AffectReading,
    AffectVector,
    AnswerContract,
    AnswerStatus,
    CandidateResponse,
    GateDecision,
)


try:
    from engine.pendulum import compute_vadug as _V8_COMPUTE_VADUG
    from engine.shared import VADUG as _V8_VADUG
    from engine.solver import state_transition as _V8_STATE_TRANSITION
except (ImportError, ModuleNotFoundError) as exc:
    _V8_COMPUTE_VADUG = None
    _V8_VADUG = None
    _V8_STATE_TRANSITION = None
    _V8_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _V8_IMPORT_ERROR = None


class AffectBackend:
    name = "abstract"

    def analyze(self, text: str) -> AffectReading:
        raise NotImplementedError

    def transition(self, state: AffectVector, message: AffectVector) -> AffectVector:
        raise NotImplementedError


class ClankerAffectBackend(AffectBackend):
    """Use ``engine.pendulum`` and ``engine.solver`` without modifying them."""

    name = "clanker-v8"

    def __init__(self, *, personality: Any = None, perspective: str = "speaker") -> None:
        if (
            _V8_COMPUTE_VADUG is None
            or _V8_VADUG is None
            or _V8_STATE_TRANSITION is None
        ):
            error = ModuleNotFoundError(
                "Clanker V8 engine imports are unavailable; use "
                "HeuristicAffectBackend explicitly or install the engine package"
            )
            if _V8_IMPORT_ERROR is not None:
                raise error from _V8_IMPORT_ERROR
            raise error
        self._compute_vadug = _V8_COMPUTE_VADUG
        self._state_transition = _V8_STATE_TRANSITION
        self._vadug_type = _V8_VADUG
        self.personality = personality
        self.perspective = perspective

    def analyze(self, text: str) -> AffectReading:
        vector, meta = self._compute_vadug(
            text,
            personality=self.personality,
            perspective=self.perspective,
        )
        meta = meta or {}
        structures = [
            getattr(item, "pattern", str(item))
            for item in meta.get("structures", [])
        ]
        roles = [
            getattr(item, "role", str(item))
            for item in meta.get("roles", [])
        ]
        return AffectReading(
            vector=AffectVector.from_object(vector),
            structures=structures,
            roles=roles,
            metadata=self._safe_metadata(meta),
            backend=self.name,
        )

    def transition(self, state: AffectVector, message: AffectVector) -> AffectVector:
        a = self._vadug_type(**state.to_dict())
        b = self._vadug_type(**message.to_dict())
        return AffectVector.from_object(self._state_transition(a, b))

    @staticmethod
    def _safe_metadata(meta: Any) -> dict:
        # Keep the trace useful while avoiding live class instances that cannot
        # be serialized by the CLI/API layer.
        safe: dict = {}
        if not isinstance(meta, dict):
            return safe
        for key, value in meta.items():
            if key in {"roles", "structures", "force_flow", "trace_entries"}:
                if isinstance(value, list):
                    safe[key] = [
                        dict(vars(item)) if hasattr(item, "__dict__") else str(item)
                        for item in value
                    ]
                elif hasattr(value, "__dict__"):
                    safe[key] = dict(vars(value))
                else:
                    safe[key] = value
            elif isinstance(value, (str, int, float, bool, type(None))):
                safe[key] = value
        return safe


class HeuristicAffectBackend(AffectBackend):
    """Dependency-free fallback used only when the V8 engine is unavailable."""

    name = "heuristic-fallback"
    NEGATIVE = {"sad", "angry", "hate", "hurt", "hurts", "hurting", "sick", "ill", "bad", "terrible", "awful", "pissed", "died", "dead", "lost", "cry", "crying"}
    POSITIVE = {"happy", "love", "great", "good", "excited", "proud", "won", "passed", "better", "safe"}
    URGENT = {"now", "immediately", "emergency", "help", "tonight", "danger", "dying"}

    def analyze(self, text: str) -> AffectReading:
        words = re.findall(r"[a-z']+", text.lower())
        neg = sum(word in self.NEGATIVE for word in words)
        pos = sum(word in self.POSITIVE for word in words)
        urgent = sum(word in self.URGENT for word in words)
        casual = any(word in {"bruh", "bro", "lol", "lmao", "dude"} for word in words)
        v = 128 + min(90, pos * 28) - min(100, neg * 32)
        a = 128 + min(90, (neg + pos + urgent) * 14)
        u = min(255, urgent * 80)
        g = 128 + min(70, pos * 18) - min(100, neg * 22)
        i = 175 if "?" in text or any(word in {"help", "tell", "explain"} for word in words) else 128
        structures = ["MASKING"] if casual and neg >= 1 else []
        return AffectReading(
            vector=AffectVector(v=v, a=a, d=128, u=u, g=g, w=128, i=i),
            structures=structures,
            roles=[],
            metadata={"negative_hits": neg, "positive_hits": pos, "urgent_hits": urgent},
            backend=self.name,
        )

    def transition(self, state: AffectVector, message: AffectVector) -> AffectVector:
        values = {}
        for axis in ("v", "a", "d", "u", "g", "w", "i"):
            values[axis] = round(getattr(state, axis) * 0.6 + getattr(message, axis) * 0.4)
        return AffectVector(**values)


class AffectController:
    """Calculate response targets and backsolve candidate delivery through Clanker."""

    def __init__(self, backend: Optional[AffectBackend] = None) -> None:
        if backend is not None:
            self.backend = backend
        else:
            try:
                self.backend = ClankerAffectBackend()
            except (ImportError, ModuleNotFoundError):
                self.backend = HeuristicAffectBackend()

    def analyze(self, text: str) -> AffectReading:
        return self.backend.analyze(text)

    def observe(self, previous: AffectVector, input_vector: AffectVector) -> AffectVector:
        return self.backend.transition(previous, input_vector)

    def target_for(
        self,
        observed: AffectVector,
        contract: AnswerContract,
        gates: GateDecision,
    ) -> AffectVector:
        """Choose a reachable conversational state, not a forced emotional jump."""

        if gates.severity == "critical":
            return AffectVector(
                v=max(observed.v, 135),
                a=min(max(observed.a, 110), 150),
                d=max(observed.d, 165),
                u=max(observed.u, 200),
                g=max(observed.g, 108),
                w=max(observed.w, 165),
                i=max(observed.i, 205),
            )
        if gates.masking:
            return AffectVector(
                v=max(observed.v, 128),
                a=min(observed.a, 145),
                d=max(observed.d, 150),
                u=observed.u,
                g=max(observed.g, 112),
                w=max(observed.w, 150),
                i=max(observed.i, 195),
            )
        if contract.status in {
            AnswerStatus.MISSING_REFERENCE,
            AnswerStatus.AMBIGUOUS_REFERENCE,
            AnswerStatus.MULTIPLE_MATCHES,
            AnswerStatus.LEXICAL_PROBE,
        }:
            return AffectVector(
                v=max(125, min(150, observed.v)),
                a=min(observed.a, 125),
                d=max(observed.d, 150),
                u=observed.u,
                g=max(observed.g, 125),
                w=max(observed.w, 128),
                i=max(observed.i, 192),
            )
        if contract.status in {AnswerStatus.UNKNOWN, AnswerStatus.UNSUPPORTED, AnswerStatus.CONFLICT}:
            return AffectVector(
                v=max(118, min(145, observed.v)),
                a=min(observed.a, 130),
                d=max(observed.d, 145),
                u=observed.u,
                g=max(observed.g, 120),
                w=max(observed.w, 128),
                i=max(observed.i, 172),
            )
        if contract.status in {AnswerStatus.ACKNOWLEDGED, AnswerStatus.LEXICAL_LEARNED} and observed.v < 105:
            return AffectVector(
                v=min(150, observed.v + 28),
                a=max(90, observed.a - 25),
                d=min(170, observed.d + 18),
                u=observed.u,
                g=min(150, observed.g + 24),
                w=min(170, observed.w + 20),
                i=max(observed.i, 190),
            )
        return AffectVector(
            v=max(140, min(175, observed.v + 12)),
            a=max(90, min(140, observed.a)),
            d=max(148, observed.d),
            u=observed.u,
            g=max(135, observed.g),
            w=max(132, observed.w),
            i=max(178, observed.i),
        )

    def rank_candidates(
        self,
        candidates: Sequence[CandidateResponse],
        observed: AffectVector,
        target: AffectVector,
    ) -> Tuple[CandidateResponse, List[CandidateResponse]]:
        if not candidates:
            raise ValueError("At least one candidate is required")
        scored: List[CandidateResponse] = []
        for candidate in candidates:
            reading = self.backend.analyze(candidate.text)
            predicted = self.backend.transition(observed, reading.vector)
            distance = predicted.distance(target)
            invalid_penalty = 10000.0 if not candidate.semantic_valid else 0.0
            # Priority is a language-graph preference, but affective fit can
            # override small differences.  One priority point is 0.04 distance.
            score = distance + invalid_penalty - candidate.priority * 0.04
            candidate.affect = reading.vector
            candidate.predicted_state = predicted
            candidate.affect_distance = distance
            candidate.score = score
            scored.append(candidate)
        scored.sort(key=lambda item: (item.score, -item.priority, item.construction_id))
        return scored[0], scored
