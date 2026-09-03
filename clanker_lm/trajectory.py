"""Multi-turn VADUGWI calibration and corpus trajectory fingerprints.

Only mathematical summaries and cryptographic hashes are persisted.  No source
utterances, response sentences, book quotations, or reconstruction templates
are stored in trajectory tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .affect import AffectController
from .database import LanguageStore
from .model import (
    AffectReading,
    AffectVector,
    AnswerContract,
    AnswerStatus,
    CandidateResponse,
    GateDecision,
    ParseResult,
    QuestionKind,
    SpeechAct,
)


AXES = ("v", "a", "d", "u", "g", "w", "i")


@dataclass
class TrajectoryFinalization:
    finalized: bool = False
    trajectory_id: Optional[int] = None
    context_key: str = ""
    residual: Dict[str, float] = field(default_factory=dict)
    success: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finalized": self.finalized,
            "trajectory_id": self.trajectory_id,
            "context_key": self.context_key,
            "residual": dict(self.residual),
            "success": self.success,
        }


class TrajectoryController:
    """Learn systematic response-transition residuals online."""

    MIN_SAMPLES = 2
    MAX_CORRECTION = 28.0

    def __init__(self, store: LanguageStore) -> None:
        self.store = store

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def incoming_act(parse: ParseResult) -> str:
        if parse.question:
            return f"ask:{parse.question.kind.value}"
        if parse.speech_act == SpeechAct.ASSERT and parse.events:
            predicate = parse.events[-1].predicate
            return f"assert:{predicate}"
        return parse.speech_act.value

    @staticmethod
    def response_act(contract: AnswerContract, gates: GateDecision) -> str:
        if contract.status in {AnswerStatus.LEXICAL_PROBE}:
            return "probe:lexical"
        if contract.status in {AnswerStatus.MISSING_REFERENCE, AnswerStatus.AMBIGUOUS_REFERENCE, AnswerStatus.MULTIPLE_MATCHES}:
            return "probe:reference"
        if contract.status in {AnswerStatus.TRUE, AnswerStatus.FALSE}:
            return "answer:polar"
        if contract.status == AnswerStatus.ANSWERED:
            return "answer:factual"
        if contract.status in {AnswerStatus.UNKNOWN, AnswerStatus.UNSUPPORTED, AnswerStatus.CONFLICT}:
            return f"answer:{contract.status.value}"
        return gates.response_act

    @staticmethod
    def _bin(value: int, *, urgency: bool = False) -> int:
        width = 43 if urgency else 32
        return min(7, max(0, int(value) // width))

    def context_key(
        self,
        *,
        parse: ParseResult,
        contract: AnswerContract,
        gates: GateDecision,
        input_vector: AffectVector,
        observed: AffectVector,
        profile_id: Optional[str],
    ) -> str:
        input_bins = "".join(str(self._bin(getattr(input_vector, axis), urgency=axis == "u")) for axis in AXES)
        observed_bins = "".join(str(self._bin(getattr(observed, axis), urgency=axis == "u")) for axis in AXES)
        fields = (
            self.incoming_act(parse),
            self.response_act(contract, gates),
            contract.status.value,
            gates.severity,
            "mask" if gates.masking else "plain",
            input_bins,
            observed_bins,
            profile_id or "none",
        )
        return "|".join(fields)

    def finalize_pending(
        self,
        *,
        reaction_vector: AffectVector,
        observed_next: AffectVector,
    ) -> TrajectoryFinalization:
        pending = self.store.pending_trajectory()
        if not pending:
            return TrajectoryFinalization()
        predicted = AffectVector(**json.loads(pending["predicted_after_json"]))
        target = AffectVector(**json.loads(pending["target_vector_json"]))
        residual = {
            axis: float(getattr(observed_next, axis) - getattr(predicted, axis))
            for axis in AXES
        }
        success = max(0.0, min(1.0, 1.0 - observed_next.distance(target) / 160.0))
        trajectory_id = int(pending["trajectory_id"])
        self.store.finalize_trajectory(
            trajectory_id,
            observed_next=observed_next,
            reaction_vector=reaction_vector,
            residual=residual,
            success=success,
        )
        return TrajectoryFinalization(
            finalized=True,
            trajectory_id=trajectory_id,
            context_key=str(pending["context_key"]),
            residual=residual,
            success=success,
        )

    def adjust_target(self, target: AffectVector, context_key: str) -> Tuple[AffectVector, Dict[str, Any]]:
        stat = self.store.transition_stat(context_key)
        if not stat or int(stat["sample_count"]) < self.MIN_SAMPLES:
            return target, {"applied": False, "sample_count": int(stat["sample_count"]) if stat else 0}
        count = int(stat["sample_count"])
        reliability = min(0.70, 0.18 + math.log2(count + 1) * 0.10)
        mean_residual: Mapping[str, float] = stat["mean_residual"]
        values: Dict[str, int] = {}
        corrections: Dict[str, float] = {}
        for axis in AXES:
            # If reality tends to land below prediction on an axis, request a
            # correspondingly stronger target next time.  Corrections are
            # bounded so learned data cannot override hard safety gates.
            correction = max(
                -self.MAX_CORRECTION,
                min(self.MAX_CORRECTION, -float(mean_residual.get(axis, 0.0)) * reliability),
            )
            corrections[axis] = correction
            values[axis] = round(getattr(target, axis) + correction)
        adjusted = AffectVector(**values)
        return adjusted, {
            "applied": True,
            "sample_count": count,
            "reliability": reliability,
            "mean_residual": dict(mean_residual),
            "correction": corrections,
            "success_mean": float(stat["success_mean"]),
        }

    def record(
        self,
        *,
        input_text: str,
        response_text: str,
        parse: ParseResult,
        contract: AnswerContract,
        gates: GateDecision,
        input_vector: AffectVector,
        state_before: AffectVector,
        target: AffectVector,
        selected: CandidateResponse,
        predicted_after: AffectVector,
        context_key: str,
        profile_id: Optional[str],
    ) -> int:
        response_vector = selected.affect or AffectVector()
        return self.store.record_trajectory(
            input_hash=self._hash(input_text),
            response_hash=self._hash(response_text),
            incoming_act=self.incoming_act(parse),
            response_act=self.response_act(contract, gates),
            context_key=context_key,
            input_vector=input_vector,
            state_before=state_before,
            target_vector=target,
            response_vector=response_vector,
            predicted_after=predicted_after,
            profile_id=profile_id,
        )


@dataclass
class CorpusProfile:
    profile_id: str
    name: str
    quote_count: int
    centroid: AffectVector
    variance: Dict[str, float]
    delta_centroid: Dict[str, float]
    act_distribution: Dict[str, float]
    transition_matrix: Dict[str, float]
    fingerprint: str
    source_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "quote_count": self.quote_count,
            "centroid": self.centroid.to_dict(),
            "variance": dict(self.variance),
            "delta_centroid": dict(self.delta_centroid),
            "act_distribution": dict(self.act_distribution),
            "transition_matrix": dict(self.transition_matrix),
            "fingerprint": self.fingerprint,
            "source_hash": self.source_hash,
            "metadata": dict(self.metadata),
        }


class CorpusProfiler:
    """Compile dialogue into non-textual VADUGWI and discourse signatures."""

    QUOTE_RE = re.compile(r"[\"“]([^\"”]{2,2000})[\"”]", re.DOTALL)
    PLAY_RE = re.compile(r"^\s*[A-Z][A-Z0-9 _'-]{1,40}:\s*(.+?)\s*$", re.MULTILINE)
    WINDOW_SIZES = (4, 8, 16, 32, 64)

    def __init__(self, store: LanguageStore, affect: AffectController) -> None:
        self.store = store
        self.affect = affect

    @classmethod
    def extract_dialogue(cls, text: str) -> List[str]:
        quotes = [re.sub(r"\s+", " ", item).strip() for item in cls.QUOTE_RE.findall(text)]
        if quotes:
            return [item for item in quotes if item]
        return [re.sub(r"\s+", " ", item).strip() for item in cls.PLAY_RE.findall(text) if item.strip()]

    @staticmethod
    def classify_act(text: str) -> str:
        lowered = text.strip().lower()
        words = re.findall(r"[a-z']+", lowered)
        first = words[0] if words else ""
        if text.rstrip().endswith("?"):
            if first in {"why", "how", "what", "who", "where", "when", "which", "whose"}:
                return f"question:{first}"
            return "question:polar"
        if first in {"sorry", "apologies"}:
            return "apology"
        if first in {"yes", "yeah", "yep", "no", "nope"}:
            return "agreement" if first in {"yes", "yeah", "yep"} else "rejection"
        if first in {"please", "stop", "go", "come", "tell", "give", "leave"}:
            return "directive"
        if any(word in {"feel", "felt", "sad", "angry", "afraid", "love", "hate"} for word in words):
            return "disclosure"
        if text.rstrip().endswith("!"):
            return "exclamation"
        return "assertion"

    def compile_text(self, name: str, text: str, *, profile_id: Optional[str] = None) -> CorpusProfile:
        quotes = self.extract_dialogue(text)
        if not quotes:
            raise ValueError("No quoted or speaker-labelled dialogue was found")
        return self.compile_quotes(name, quotes, source_text=text, profile_id=profile_id)

    def compile_quotes(
        self,
        name: str,
        quotes: Sequence[str],
        *,
        source_text: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> CorpusProfile:
        clean = [re.sub(r"\s+", " ", item).strip() for item in quotes if item.strip()]
        if not clean:
            raise ValueError("At least one dialogue turn is required")
        readings = [self.affect.analyze(item) for item in clean]
        vectors = [item.vector for item in readings]
        acts = [self.classify_act(item) for item in clean]
        centroid = self._centroid(vectors)
        variance = self._variance(vectors, centroid)
        deltas = [
            {axis: float(getattr(right, axis) - getattr(left, axis)) for axis in AXES}
            for left, right in zip(vectors, vectors[1:])
        ]
        delta_centroid = {
            axis: sum(item[axis] for item in deltas) / len(deltas) if deltas else 0.0
            for axis in AXES
        }
        act_counts = Counter(acts)
        act_distribution = {key: value / len(acts) for key, value in sorted(act_counts.items())}
        transition_counts = Counter(f"{left}->{right}" for left, right in zip(acts, acts[1:]))
        transition_total = sum(transition_counts.values()) or 1
        transition_matrix = {
            key: value / transition_total for key, value in sorted(transition_counts.items())
        }
        vector_blob = self.pack_vectors(vectors)
        packed_deltas = self.pack_signed_deltas(vectors)
        act_blob = "\x1f".join(acts).encode("utf-8")
        length_blob = bytes(min(255, len(re.findall(r"\S+", item))) for item in clean)
        fingerprint = hashlib.sha256(vector_blob + b"\x00" + act_blob + b"\x00" + length_blob).hexdigest()
        source_material = source_text if source_text is not None else "\x1e".join(clean)
        source_hash = hashlib.sha256(source_material.encode("utf-8")).hexdigest()
        identifier = profile_id or "profile_" + fingerprint[:16]
        chunks = self._chunks(vectors)
        profile = CorpusProfile(
            profile_id=identifier,
            name=name,
            quote_count=len(vectors),
            centroid=centroid,
            variance=variance,
            delta_centroid=delta_centroid,
            act_distribution=act_distribution,
            transition_matrix=transition_matrix,
            fingerprint=fingerprint,
            source_hash=source_hash,
            metadata={
                "format_version": 1,
                "stored_raw_text": False,
                "vector_bytes_per_turn": 7,
                "act_count": len(act_counts),
            },
        )
        payload = profile.to_dict()
        payload["trajectory_blob"] = vector_blob
        payload["delta_blob"] = packed_deltas
        payload["chunks"] = chunks
        self.store.upsert_corpus_profile(payload)
        return profile

    def match_text(self, text: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        quotes = self.extract_dialogue(text)
        if not quotes:
            raise ValueError("No dialogue was found for matching")
        transient = self._transient(quotes)
        results: List[Dict[str, Any]] = []
        for summary in self.store.list_corpus_profiles():
            stored = self.store.get_corpus_profile(str(summary["profile_id"]))
            if not stored:
                continue
            centroid = AffectVector(**stored["centroid"])
            centroid_distance = transient["centroid"].distance(centroid)
            delta_distance = math.sqrt(
                sum(
                    (transient["delta_centroid"].get(axis, 0.0) - float(stored["delta_centroid"].get(axis, 0.0))) ** 2
                    for axis in AXES
                ) / len(AXES)
            )
            act_keys = set(transient["act_distribution"]) | set(stored["act_distribution"])
            act_distance = sum(
                abs(transient["act_distribution"].get(key, 0.0) - float(stored["act_distribution"].get(key, 0.0)))
                for key in act_keys
            ) / 2.0
            score = centroid_distance + 0.45 * delta_distance + 28.0 * act_distance
            results.append(
                {
                    "profile_id": stored["profile_id"],
                    "name": stored["name"],
                    "score": score,
                    "centroid_distance": centroid_distance,
                    "delta_distance": delta_distance,
                    "act_distance": act_distance,
                    "fingerprint_exact": transient["fingerprint"] == stored["fingerprint"],
                }
            )
        results.sort(key=lambda item: (not item["fingerprint_exact"], item["score"], item["profile_id"]))
        return results[: max(1, int(top_k))]

    def adjust_target(
        self,
        target: AffectVector,
        profile_id: Optional[str],
        *,
        turn_index: int,
        severity: str,
    ) -> Tuple[AffectVector, Dict[str, Any]]:
        if not profile_id:
            return target, {"applied": False}
        profile = self.store.get_corpus_profile(profile_id)
        if not profile:
            return target, {"applied": False, "reason": "unknown profile"}
        centroid = AffectVector(**profile["centroid"])
        delta = profile["delta_centroid"]
        trajectory = self.unpack_vectors(bytes(profile["trajectory_blob"]))
        phase_index = (max(1, int(turn_index)) - 1) % max(1, len(trajectory))
        phase_vector = trajectory[phase_index] if trajectory else centroid
        next_vector = trajectory[(phase_index + 1) % len(trajectory)] if len(trajectory) > 1 else phase_vector
        strength = 0.08 if severity in {"high", "critical"} else 0.20
        values: Dict[str, int] = {}
        for axis in AXES:
            phase_motion = (getattr(next_vector, axis) - getattr(phase_vector, axis)) * 0.35
            profile_goal = getattr(phase_vector, axis) + phase_motion
            values[axis] = round(getattr(target, axis) * (1.0 - strength) + profile_goal * strength)
        adjusted = AffectVector(**values)
        return adjusted, {
            "applied": True,
            "profile_id": profile_id,
            "profile_name": profile["name"],
            "strength": strength,
            "turn_index": turn_index,
            "phase_index": phase_index,
            "phase_vector": phase_vector.to_dict(),
            "next_vector": next_vector.to_dict(),
            "centroid": centroid.to_dict(),
            "delta_centroid": dict(delta),
        }

    def _transient(self, quotes: Sequence[str]) -> Dict[str, Any]:
        vectors = [self.affect.analyze(item).vector for item in quotes]
        acts = [self.classify_act(item) for item in quotes]
        centroid = self._centroid(vectors)
        deltas = [
            {axis: float(getattr(right, axis) - getattr(left, axis)) for axis in AXES}
            for left, right in zip(vectors, vectors[1:])
        ]
        delta_centroid = {
            axis: sum(item[axis] for item in deltas) / len(deltas) if deltas else 0.0
            for axis in AXES
        }
        counts = Counter(acts)
        act_distribution = {key: value / len(acts) for key, value in counts.items()}
        length_blob = bytes(min(255, len(re.findall(r"\S+", item))) for item in quotes)
        fingerprint = hashlib.sha256(
            self.pack_vectors(vectors)
            + b"\x00"
            + "\x1f".join(acts).encode("utf-8")
            + b"\x00"
            + length_blob
        ).hexdigest()
        return {
            "centroid": centroid,
            "delta_centroid": delta_centroid,
            "act_distribution": act_distribution,
            "fingerprint": fingerprint,
        }

    @staticmethod
    def _centroid(vectors: Sequence[AffectVector]) -> AffectVector:
        return AffectVector(**{
            axis: round(sum(getattr(item, axis) for item in vectors) / len(vectors))
            for axis in AXES
        })

    @staticmethod
    def _variance(vectors: Sequence[AffectVector], centroid: AffectVector) -> Dict[str, float]:
        return {
            axis: sum((getattr(item, axis) - getattr(centroid, axis)) ** 2 for item in vectors) / len(vectors)
            for axis in AXES
        }

    @staticmethod
    def pack_vectors(vectors: Sequence[AffectVector]) -> bytes:
        return bytes(getattr(vector, axis) for vector in vectors for axis in AXES)

    @staticmethod
    def pack_signed_deltas(vectors: Sequence[AffectVector]) -> bytes:
        values: List[int] = []
        for left, right in zip(vectors, vectors[1:]):
            for axis in AXES:
                values.append(max(0, min(255, 128 + getattr(right, axis) - getattr(left, axis))))
        return bytes(values)

    @staticmethod
    def unpack_vectors(blob: bytes) -> List[AffectVector]:
        if len(blob) % len(AXES):
            raise ValueError("packed trajectory length is not divisible by seven")
        return [
            AffectVector(**dict(zip(AXES, blob[index : index + len(AXES)])))
            for index in range(0, len(blob), len(AXES))
        ]

    def _chunks(self, vectors: Sequence[AffectVector]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for size in self.WINDOW_SIZES:
            if len(vectors) < size:
                continue
            stride = max(1, size // 2)
            for start in range(0, len(vectors) - size + 1, stride):
                window = vectors[start : start + size]
                deltas = [
                    AffectVector(**{
                        axis: max(0, min(255, 128 + getattr(right, axis) - getattr(left, axis)))
                        for axis in AXES
                    })
                    for left, right in zip(window, window[1:])
                ]
                vector_blob = self.pack_vectors(window)
                delta_blob = self.pack_vectors(deltas)
                chunks.append(
                    {
                        "start_index": start,
                        "window_size": size,
                        "vector_blob": vector_blob,
                        "delta_blob": delta_blob,
                        "vector_hash": hashlib.blake2b(vector_blob, digest_size=8).hexdigest(),
                        "delta_hash": hashlib.blake2b(delta_blob, digest_size=8).hexdigest(),
                    }
                )
        if not chunks:
            vector_blob = self.pack_vectors(vectors)
            chunks.append(
                {
                    "start_index": 0,
                    "window_size": len(vectors),
                    "vector_blob": vector_blob,
                    "delta_blob": b"",
                    "vector_hash": hashlib.blake2b(vector_blob, digest_size=8).hexdigest(),
                    "delta_hash": hashlib.blake2b(b"", digest_size=8).hexdigest(),
                }
            )
        return chunks
