"""Active, evidence-tracked lexical learning for Clanker-LM.

Unknown words are not inserted directly into the canonical vocabulary.  The
learner stores occurrences and user explanations as auditable evidence,
maintains versioned sense hypotheses, re-evaluates earlier contexts whenever
new evidence arrives, and can split a surface form into multiple senses when
corrective context is persistently bimodal.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import lexicon
from .affect import AffectController
from .database import LanguageStore, LearnedSenseRecord
from .memory import ConversationMemory
from .model import (
    AffectReading,
    AffectVector,
    AnswerContract,
    AnswerStatus,
    EntityKind,
    EventFrame,
    ParseResult,
    RefKind,
    SemanticRef,
    SourceKind,
)


META_DEFINITION_RE = re.compile(
    r"^\s*(?P<term>[A-Za-z][A-Za-z'’-]*)\s+(?:means?|is\s+slang\s+for)\s+(?P<definition>.+?)\s*[.?!]*\s*$",
    re.IGNORECASE,
)
DEFINITION_QUERY_RE = re.compile(
    r"^\s*(?:what\s+does\s+|what\s+do\s+you\s+mean\s+by\s+|define\s+)(?P<term>[A-Za-z][A-Za-z'’-]*)(?:\s+mean)?\s*[?!.]*\s*$",
    re.IGNORECASE,
)
QUOTED_TERM_RE = re.compile(r"[\"'“”‘’]([A-Za-z][A-Za-z'’-]*)[\"'“”‘’]")


POSITIVE_WORDS = {
    "positive", "good", "great", "excellent", "amazing", "awesome", "cool",
    "love", "loved", "happy", "fun", "beautiful", "impressive", "win",
}
NEGATIVE_WORDS = {
    "negative", "bad", "awful", "terrible", "disappointing", "overhyped",
    "hate", "hated", "sad", "angry", "annoying", "gross", "broken", "wrong",
}
INTENSE_WORDS = {"intense", "extreme", "powerful", "strong", "wild", "huge", "very"}
DESCRIPTIVE_WORDS = {"descriptive", "neutral", "literal", "technical", "category", "type"}

SEMANTIC_HINTS: Dict[str, Set[str]] = {
    "positive_evaluation": POSITIVE_WORDS,
    "negative_evaluation": NEGATIVE_WORDS,
    "intensity": INTENSE_WORDS,
    "descriptive": DESCRIPTIVE_WORDS,
    "illness_state": {"ill", "illness", "sick", "disease", "health", "medical"},
    "emotion_state": {"emotion", "feeling", "feel", "mood", "angry", "sad", "happy"},
    "action": {"action", "verb", "doing", "does", "behavior", "behaviour"},
    "person_descriptor": {"person", "someone", "people", "guy", "girl", "character"},
    "object_descriptor": {"object", "thing", "item", "device", "tool"},
}


@dataclass
class PendingDefinition:
    term: str
    normalized: str
    term_id: int
    context: str
    part_of_speech: str
    semantic_position: str
    probe_axis: str = "definition"
    attempts: int = 0
    created_turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "normalized": self.normalized,
            "term_id": self.term_id,
            "context": self.context,
            "part_of_speech": self.part_of_speech,
            "semantic_position": self.semantic_position,
            "probe_axis": self.probe_axis,
            "attempts": self.attempts,
            "created_turn": self.created_turn,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PendingDefinition":
        return cls(
            term=str(data["term"]),
            normalized=str(data["normalized"]),
            term_id=int(data["term_id"]),
            context=str(data.get("context", "")),
            part_of_speech=str(data.get("part_of_speech", "unknown")),
            semantic_position=str(data.get("semantic_position", "unknown")),
            probe_axis=str(data.get("probe_axis", "definition")),
            attempts=int(data.get("attempts", 0)),
            created_turn=int(data.get("created_turn", 0)),
        )


@dataclass
class LearningOutcome:
    handled: bool = False
    contract: Optional[AnswerContract] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    suppress_fact_storage: bool = False


class LexicalLearner:
    """Learn word senses from context and direct user correction."""

    TRUST_THRESHOLD = 0.68
    PROBE_THRESHOLD = 0.50
    MAX_PROBE_ATTEMPTS = 3

    def __init__(
        self,
        store: LanguageStore,
        affect: AffectController,
        memory: ConversationMemory,
        *,
        scope_id: str = "default",
    ) -> None:
        self.store = store
        self.affect = affect
        self.memory = memory
        self.scope_id = scope_id
        self.pending: Optional[PendingDefinition] = None
        self._static_known = self._collect_static_known_words()

    # ------------------------------------------------------------------
    # Public routing
    # ------------------------------------------------------------------

    def preparse(self, text: str, input_affect: AffectReading) -> LearningOutcome:
        """Handle definition queries or a reply to the current lexical probe."""

        query = DEFINITION_QUERY_RE.match(text)
        if query:
            term = query.group("term")
            return self._definition_query(term)

        direct = META_DEFINITION_RE.match(text)

        # A pending lexical probe owns the next plausible explanatory turn.
        # This must run before ordinary ``X means Y`` detection so replies such
        # as ``It means very intense`` bind to the pending word instead of
        # accidentally creating a learned sense for the pronoun ``it``.
        if self.pending:
            if direct:
                reply_term = self.normalize_term(direct.group("term"))
                if reply_term in {self.pending.normalized, "it", "that", "this"}:
                    return self._consume_definition(
                        direct.group("definition"),
                        input_affect,
                        raw_explanation=text,
                    )
            if self._looks_like_definition_reply(text):
                return self._consume_definition(text, input_affect, raw_explanation=text)

        if direct:
            term = direct.group("term")
            definition = direct.group("definition")
            normalized = self.normalize_term(term)
            senses = self.store.learned_senses(normalized, min_confidence=0.0)
            if not senses:
                self._open_pending(
                    term,
                    text,
                    part_of_speech="unknown",
                    semantic_position="metalinguistic_subject",
                    input_affect=input_affect,
                )
            elif self.pending is None or self.pending.normalized != normalized:
                row = self.store.term_row(normalized)
                assert row is not None
                self.pending = PendingDefinition(
                    term=term,
                    normalized=normalized,
                    term_id=int(row["term_id"]),
                    context=text,
                    part_of_speech=senses[0].part_of_speech,
                    semantic_position="metalinguistic_subject",
                    created_turn=self.memory.turn_index,
                )
            return self._consume_definition(definition, input_affect, raw_explanation=text)

        return LearningOutcome()

    def postparse(
        self,
        text: str,
        parse: ParseResult,
        input_affect: AffectReading,
    ) -> LearningOutcome:
        """Detect a high-confidence unknown lexical item after semantic parse."""

        candidate = self.detect_unknown(text, parse)
        if not candidate:
            return LearningOutcome()
        term, part_of_speech, semantic_position = candidate
        self._open_pending(
            term,
            text,
            part_of_speech=part_of_speech,
            semantic_position=semantic_position,
            input_affect=input_affect,
        )
        assert self.pending is not None
        return LearningOutcome(
            handled=True,
            contract=self._probe_contract(self.pending),
            metadata={
                "action": "lexical_probe",
                "term": term,
                "part_of_speech": part_of_speech,
                "semantic_position": semantic_position,
                "probe_axis": self.pending.probe_axis,
            },
            suppress_fact_storage=True,
        )

    def apply_overlay(self, text: str, reading: AffectReading) -> AffectReading:
        """Blend trusted learned senses into an affect reading.

        The V8 kernel remains unchanged.  Learned vocabulary is an auditable
        overlay and can be removed without altering canonical engine weights.
        """

        tokens = lexicon.tokenize(text, include_punctuation=False)
        matches: List[Tuple[str, LearnedSenseRecord]] = []
        for token in tokens:
            senses = self.store.learned_senses(
                self.normalize_term(token.norm),
                min_confidence=self.PROBE_THRESHOLD,
                include_disputed=False,
            )
            if not senses:
                continue
            if len(senses) == 1:
                selected = senses[0]
            else:
                selected = min(
                    senses,
                    key=lambda sense: sense.vector.distance(reading.vector),
                )
            matches.append((token.norm, selected))

        if not matches:
            return reading

        values = reading.vector.to_dict()
        neutral = {"v": 128, "a": 128, "d": 128, "u": 0, "g": 128, "w": 128, "i": 128}
        for _, sense in matches:
            weight = min(0.82, max(0.20, sense.confidence * 0.72))
            for axis in values:
                delta = getattr(sense.vector, axis) - neutral[axis]
                values[axis] = round(values[axis] + delta * weight)
        vector = AffectVector(**values)
        metadata = dict(reading.metadata)
        metadata["learned_lexemes"] = [
            {
                "term": term,
                "sense_id": sense.sense_id,
                "semantic_class": sense.semantic_class,
                "confidence": sense.confidence,
                "version": sense.version,
            }
            for term, sense in matches
        ]
        return AffectReading(
            vector=vector,
            structures=list(reading.structures),
            roles=list(reading.roles),
            metadata=metadata,
            backend=f"{reading.backend}+learned-overlay",
        )

    def observe_known_context(
        self,
        text: str,
        base_reading: AffectReading,
    ) -> Dict[str, Any]:
        """Accumulate weak evidence whenever an already learned term reappears."""

        tokens = lexicon.tokenize(text, include_punctuation=False)
        seen: Set[str] = set()
        updates: List[Dict[str, Any]] = []
        for token in tokens:
            normalized = self.normalize_term(token.norm)
            if normalized in seen:
                continue
            senses = self.store.learned_senses(normalized, min_confidence=0.0)
            if not senses:
                continue
            seen.add(normalized)
            term = self.store.term_row(normalized)
            if not term:
                continue
            # Remove only this lexical item so surrounding descriptors and the
            # rest of the proposition provide independent corrective evidence.
            surrounding = re.sub(
                rf"(?i)(?<![A-Za-z0-9']){re.escape(token.text)}(?![A-Za-z0-9'])",
                " ",
                text,
            )
            surrounding = re.sub(r"\s+", " ", surrounding).strip()
            context_reading = self.affect.analyze(surrounding) if surrounding else base_reading
            semantic_class, polarity = self._infer_semantic_class(surrounding)
            signal = context_reading.vector.distance(AffectVector())
            informative = semantic_class != "unknown" or polarity != 0 or signal >= 18.0
            weight = 0.42 if informative else 0.12
            self.store.add_lexical_evidence(
                term_id=int(term["term_id"]),
                evidence_kind="context_reuse",
                raw_context=text,
                raw_explanation="",
                context_hash=self._hash(text),
                context_features={
                    "surrounding_hash": self._hash(surrounding),
                    "informative": informative,
                    "base_backend": base_reading.backend,
                },
                vector=context_reading.vector,
                semantic_class=semantic_class,
                polarity=polarity,
                support_weight=weight,
                contradiction_weight=0.0,
                observed_turn=self.memory.turn_index,
            )
            refreshed = self._recompute(
                int(term["term_id"]),
                part_of_speech=senses[0].part_of_speech,
            )
            updates.append(
                {
                    "term": token.text,
                    "evidence_kind": "context_reuse",
                    "informative": informative,
                    "polarity": polarity,
                    "semantic_class": semantic_class,
                    "sense_count": len(refreshed),
                    "versions": [sense.version for sense in refreshed],
                }
            )
        return {"action": "lexical_context_update", "updates": updates} if updates else {}

    # ------------------------------------------------------------------
    # Unknown detection
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_term(term: str) -> str:
        return term.lower().replace("’", "'").strip(".,!?;:\"' ")

    def _collect_static_known_words(self) -> Set[str]:
        known: Set[str] = set(self.store.known_atom_surfaces())
        for name, value in vars(lexicon).items():
            if name.startswith("_"):
                continue
            if isinstance(value, (set, frozenset)):
                known.update(str(item).lower() for item in value if isinstance(item, str))
            elif isinstance(value, dict):
                known.update(str(item).lower() for item in value if isinstance(item, str))
                for item in value.values():
                    if isinstance(item, str):
                        known.add(item.lower())
                    elif isinstance(item, tuple):
                        known.update(str(part).lower() for part in item if isinstance(part, str))
        try:
            from engine.vocabulary import VOCABULARY

            known.update(str(item).lower() for item in VOCABULARY)
        except (ImportError, ModuleNotFoundError):
            pass
        return known

    def known_words(self) -> Set[str]:
        known = set(self._static_known)
        for item in self.store.learned_terms_summary():
            if any(float(sense["confidence"]) >= self.PROBE_THRESHOLD for sense in item["senses"]):
                known.add(str(item["normalized"]))
        for entity in self.memory.entities.values():
            for phrase in [entity.canonical_name, *entity.aliases]:
                known.update(self.normalize_term(part) for part in phrase.split())
        return known

    def detect_unknown(
        self,
        text: str,
        parse: ParseResult,
    ) -> Optional[Tuple[str, str, str]]:
        known = self.known_words()
        tokens = lexicon.tokenize(text, include_punctuation=False)
        unknown = [
            token
            for token in tokens
            if self._is_unknown_token(token.text, token.norm, known)
        ]
        if not unknown:
            return None

        quoted = {self.normalize_term(item) for item in QUOTED_TERM_RE.findall(text)}
        for token in unknown:
            if token.norm in quoted:
                return token.text, "unknown", "quoted_term"

        # Copular values are high-confidence descriptive/predicate positions.
        for event in parse.events:
            if event.predicate == "be":
                value = event.arguments.get("value") or event.arguments.get("state")
                if value:
                    for token in unknown:
                        if self._ref_contains(value, token.norm):
                            return token.text, "adjective", "copular_complement"
            if event.predicate in {"feel", "seem", "sound", "look"}:
                state = event.arguments.get("state") or event.arguments.get("value")
                if state:
                    for token in unknown:
                        if self._ref_contains(state, token.norm):
                            return token.text, "adjective", "state_complement"

        words = [token.norm for token in tokens]
        for token in unknown:
            index = words.index(token.norm)
            previous = words[index - 1] if index else ""
            following = words[index + 1] if index + 1 < len(words) else ""
            if previous in lexicon.INTENSIFIERS:
                return token.text, "adjective", "intensified_descriptor"
            if token.norm.endswith("ing") and following in lexicon.PRONOUN_FEATURES:
                return token.text, "verb", "unknown_predicate"
            if previous in {"keeps", "keep", "kept", "started", "start"} and token.norm.endswith("ing"):
                return token.text, "verb", "unknown_predicate"

        if not parse.understood:
            token = unknown[0]
            return token.text, "unknown", "parse_blocker"
        return None

    def _is_unknown_token(self, surface: str, normalized: str, known: Set[str]) -> bool:
        if not normalized or normalized in known:
            return False
        if normalized.isdigit() or re.fullmatch(r"[$€£]?\d+(?:\.\d+)?", normalized):
            return False
        if len(normalized) < 3:
            return False
        base = lexicon.lemma(normalized)
        if base in known:
            return False
        # Capitalized tokens are treated as proper names/entities unless the
        # same spelling is already the pending metalinguistic subject.
        if surface[:1].isupper() and not (self.pending and self.pending.normalized == normalized):
            return False
        return True

    @staticmethod
    def _ref_contains(ref: SemanticRef, token: str) -> bool:
        return token in {part.lower() for part in (ref.surface or ref.key).split()}

    # ------------------------------------------------------------------
    # Evidence and hypothesis update
    # ------------------------------------------------------------------

    def _open_pending(
        self,
        term: str,
        context: str,
        *,
        part_of_speech: str,
        semantic_position: str,
        input_affect: AffectReading,
    ) -> None:
        normalized = self.normalize_term(term)
        term_id = self.store.touch_unknown_term(term, normalized, self.memory.turn_index)
        self.pending = PendingDefinition(
            term=term,
            normalized=normalized,
            term_id=term_id,
            context=context,
            part_of_speech=part_of_speech,
            semantic_position=semantic_position,
            probe_axis="definition",
            attempts=0,
            created_turn=self.memory.turn_index,
        )
        masked = re.sub(rf"\b{re.escape(term)}\b", "", context, flags=re.IGNORECASE)
        context_reading = self.affect.analyze(masked.strip() or context)
        semantic_class, polarity = self._infer_semantic_class(masked)
        self.store.add_lexical_evidence(
            term_id=term_id,
            evidence_kind="occurrence",
            raw_context=context,
            raw_explanation="",
            context_hash=self._hash(context),
            context_features={
                "part_of_speech": part_of_speech,
                "semantic_position": semantic_position,
                "base_backend": input_affect.backend,
            },
            vector=context_reading.vector,
            semantic_class=semantic_class,
            polarity=polarity,
            support_weight=0.30,
            contradiction_weight=0.0,
            observed_turn=self.memory.turn_index,
        )
        self._recompute(term_id, part_of_speech=part_of_speech)

    def _consume_definition(
        self,
        definition: str,
        input_affect: AffectReading,
        *,
        raw_explanation: str,
    ) -> LearningOutcome:
        pending = self.pending
        if not pending:
            return LearningOutcome()
        fragment = self._definition_fragment(definition)
        reading = self.affect.analyze(fragment)
        semantic_class, polarity = self._infer_semantic_class(fragment)
        if polarity > 0:
            vector = self._force_polarity(reading.vector, positive=True)
        elif polarity < 0:
            vector = self._force_polarity(reading.vector, positive=False)
        else:
            vector = reading.vector
        vector = self._apply_semantic_constraints(vector, semantic_class)
        informative = (
            semantic_class != "unknown"
            or polarity != 0
            or reading.vector.distance(AffectVector()) >= 18.0
        )
        support = (1.0 + min(1.0, len(fragment.split()) / 8.0)) if informative else 0.30
        self.store.add_lexical_evidence(
            term_id=pending.term_id,
            evidence_kind="user_definition",
            raw_context=pending.context,
            raw_explanation=raw_explanation,
            context_hash=self._hash(pending.context + "\n" + raw_explanation),
            context_features={
                "probe_axis": pending.probe_axis,
                "part_of_speech": pending.part_of_speech,
                "semantic_position": pending.semantic_position,
                "informative": informative,
            },
            vector=vector,
            semantic_class=semantic_class,
            polarity=polarity,
            support_weight=support,
            contradiction_weight=0.0,
            observed_turn=self.memory.turn_index,
        )
        senses = self._recompute(pending.term_id, part_of_speech=pending.part_of_speech)
        pending.attempts += 1

        best = senses[0] if senses else None
        if best and best.confidence >= self.TRUST_THRESHOLD:
            metadata = {
                "action": "lexical_learned",
                "term": pending.term,
                "semantic_class": best.semantic_class,
                "confidence": best.confidence,
                "version": best.version,
                "sense_count": len(senses),
                "vector": best.vector.to_dict(),
            }
            contract = AnswerContract(
                status=AnswerStatus.LEXICAL_LEARNED,
                proposition=EventFrame(
                    predicate="mean",
                    arguments={
                        "agent": SemanticRef.literal(pending.normalized, pending.term, EntityKind.ABSTRACT),
                        "patient": SemanticRef.literal(
                            best.semantic_class,
                            best.semantic_class.replace("_", " "),
                            EntityKind.ABSTRACT,
                        ),
                    },
                    source=SourceKind.USER,
                    certainty=round(best.confidence * 255),
                    raw_text=raw_explanation,
                ),
                certainty=round(best.confidence * 255),
                source=SourceKind.USER,
                reason="learned lexical sense from accumulated evidence",
                response_goal="lexical_learned",
                required_slots={"term": pending.term},
            )
            self.pending = None
            return LearningOutcome(
                handled=True,
                contract=contract,
                metadata=metadata,
                suppress_fact_storage=True,
            )

        if pending.attempts >= self.MAX_PROBE_ATTEMPTS:
            # Keep the evidence even when it is insufficient.  The next natural
            # occurrence can strengthen or correct the hypothesis without
            # trapping the user in a clarification loop.
            metadata = {
                "action": "lexical_saved_unresolved",
                "term": pending.term,
                "attempts": pending.attempts,
                "confidence": best.confidence if best else 0.0,
            }
            contract = AnswerContract(
                status=AnswerStatus.LEXICAL_LEARNED,
                certainty=round((best.confidence if best else 0.0) * 255),
                source=SourceKind.USER,
                reason="evidence saved; sense remains provisional",
                response_goal="lexical_saved",
                required_slots={"term": pending.term},
            )
            self.pending = None
            return LearningOutcome(True, contract, metadata, True)

        pending.probe_axis = self._next_probe_axis(senses, semantic_class, polarity)
        return LearningOutcome(
            handled=True,
            contract=self._probe_contract(pending),
            metadata={
                "action": "lexical_reprobe",
                "term": pending.term,
                "probe_axis": pending.probe_axis,
                "confidence": best.confidence if best else 0.0,
                "saved_evidence": True,
            },
            suppress_fact_storage=True,
        )

    def _recompute(
        self,
        term_id: int,
        *,
        part_of_speech: str,
    ) -> List[LearnedSenseRecord]:
        evidence = self.store.lexical_evidence(term_id)
        if not evidence:
            return []
        positive = [item for item in evidence if int(item["polarity"]) > 0 or item["vector"].v >= 145]
        negative = [item for item in evidence if int(item["polarity"]) < 0 or item["vector"].v <= 111]
        positive_weight = sum(float(item["support_weight"]) for item in positive)
        negative_weight = sum(float(item["support_weight"]) for item in negative)
        total_weight = sum(float(item["support_weight"]) for item in evidence)

        positive_explicit = any(
            item["evidence_kind"] == "user_definition" and int(item.get("polarity", 0)) > 0
            for item in positive
        )
        negative_explicit = any(
            item["evidence_kind"] == "user_definition" and int(item.get("polarity", 0)) < 0
            for item in negative
        )
        split = (
            positive_weight >= 1.0
            and negative_weight >= 1.0
            and positive_explicit
            and negative_explicit
            and min(positive_weight, negative_weight) / max(total_weight, 1e-9) >= 0.25
        )
        groups: List[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]
        if split:
            groups = [
                ("positive", positive, {"context_polarity": "positive"}),
                ("negative", negative, {"context_polarity": "negative"}),
            ]
        else:
            groups = [("default", evidence, {})]

        senses_payload: List[Dict[str, Any]] = []
        for index, (label, group, conditions) in enumerate(groups, start=1):
            if not group:
                continue
            mean = self._weighted_vector(group)
            semantic_class = self._weighted_semantic_class(group)
            weight = sum(float(item["support_weight"]) for item in group)
            coherence = self._coherence(group, mean)
            explicit_count = sum(
                item["evidence_kind"] == "user_definition"
                and (
                    item.get("semantic_class") != "unknown"
                    or int(item.get("polarity", 0)) != 0
                    or bool(item.get("context_features", {}).get("informative"))
                )
                for item in group
            )
            confidence = min(
                0.98,
                0.20 + 0.18 * explicit_count + 0.10 * min(4.0, weight) + 0.28 * coherence,
            )
            if explicit_count and semantic_class != "unknown":
                confidence = min(0.98, confidence + 0.12)
            if semantic_class == "unknown" and not any(int(item.get("polarity", 0)) for item in group):
                confidence = min(confidence, 0.48)
            if confidence >= 0.88:
                sense_status = "trusted"
            elif confidence >= self.TRUST_THRESHOLD:
                sense_status = "locally_trusted"
            elif confidence >= 0.40:
                sense_status = "provisional"
            else:
                sense_status = "unresolved"
            if split:
                sense_status = "split_" + sense_status
            senses_payload.append(
                {
                    "sense_index": index,
                    "scope_type": "session",
                    "scope_id": self.scope_id,
                    "part_of_speech": part_of_speech,
                    "semantic_class": semantic_class,
                    "register": self._infer_register(group),
                    "confidence": confidence,
                    "status": sense_status,
                    "support_weight": weight,
                    "contradiction_weight": max(0.0, total_weight - weight),
                    "vector": mean.to_dict(),
                    "conditions": conditions,
                }
            )
        term_status = "split_into_multiple_senses" if split else (
            senses_payload[0]["status"] if senses_payload else "unresolved"
        )
        self.store.replace_senses(term_id, senses_payload, status=term_status)
        row = next(
            (
                item
                for item in self.store.learned_terms_summary()
                if int(item["term_id"]) == term_id
            ),
            None,
        )
        if not row:
            return []
        return self.store.learned_senses(str(row["normalized"]), min_confidence=0.0)

    @staticmethod
    def _weighted_vector(items: Sequence[Mapping[str, Any]]) -> AffectVector:
        axes = ("v", "a", "d", "u", "g", "w", "i")
        total = sum(float(item["support_weight"]) for item in items) or 1.0
        values = {
            axis: round(
                sum(getattr(item["vector"], axis) * float(item["support_weight"]) for item in items)
                / total
            )
            for axis in axes
        }
        return AffectVector(**values)

    @staticmethod
    def _weighted_semantic_class(items: Sequence[Mapping[str, Any]]) -> str:
        weights: Dict[str, float] = {}
        for item in items:
            label = str(item.get("semantic_class", "unknown"))
            if label == "unknown":
                continue
            weights[label] = weights.get(label, 0.0) + float(item["support_weight"])
        return max(weights, key=lambda label: (weights[label], label)) if weights else "unknown"

    @staticmethod
    def _coherence(items: Sequence[Mapping[str, Any]], mean: AffectVector) -> float:
        if len(items) <= 1:
            return 0.55
        total = sum(float(item["support_weight"]) for item in items) or 1.0
        distance = sum(
            item["vector"].distance(mean) * float(item["support_weight"])
            for item in items
        ) / total
        return max(0.0, min(1.0, 1.0 - distance / 105.0))

    @staticmethod
    def _infer_register(items: Sequence[Mapping[str, Any]]) -> str:
        casual = sum(
            float(item["support_weight"])
            for item in items
            if item.get("context_features", {}).get("register") == "casual"
        )
        return "casual" if casual > 0.75 else "neutral"

    @staticmethod
    def _force_polarity(vector: AffectVector, *, positive: bool) -> AffectVector:
        values = vector.to_dict()
        values["v"] = max(166, vector.v) if positive else min(90, vector.v)
        values["g"] = max(150, vector.g) if positive else min(104, vector.g)
        return AffectVector(**values)

    @staticmethod
    def _apply_semantic_constraints(
        vector: AffectVector,
        semantic_class: str,
    ) -> AffectVector:
        """Convert explicit definitional categories into axis constraints.

        The user's semantic explanation is stronger evidence than a weak
        sentence-level affect read.  For example, ``means intense`` must raise
        arousal even when the surrounding definition sentence itself is calm.
        These are bounded floors/ceilings, not replacement sentence vectors.
        """

        values = vector.to_dict()
        if semantic_class == "intensity":
            values["a"] = max(values["a"], 184)
            values["i"] = max(values["i"], 146)
        elif semantic_class == "positive_evaluation":
            values["v"] = max(values["v"], 166)
            values["g"] = max(values["g"], 150)
        elif semantic_class == "negative_evaluation":
            values["v"] = min(values["v"], 90)
            values["g"] = min(values["g"], 104)
        elif semantic_class == "illness_state":
            values["v"] = min(values["v"], 100)
            values["g"] = min(values["g"], 105)
        elif semantic_class == "action":
            values["i"] = max(values["i"], 148)
        return AffectVector(**values)

    @staticmethod
    def _definition_fragment(text: str) -> str:
        match = META_DEFINITION_RE.match(text)
        if match:
            return match.group("definition").strip()
        lowered = text.lower()
        for marker in ("like ", "basically ", "it means ", "means "):
            index = lowered.find(marker)
            if index >= 0:
                return text[index + len(marker) :].strip(" .?!")
        return text.strip(" .?!")

    @staticmethod
    def _infer_semantic_class(text: str) -> Tuple[str, int]:
        words = {token.norm for token in lexicon.tokenize(text, include_punctuation=False)}
        positive = len(words & POSITIVE_WORDS)
        negative = len(words & NEGATIVE_WORDS)
        polarity = 1 if positive > negative else -1 if negative > positive else 0
        best_label = "unknown"
        best_score = 0
        for label, hints in SEMANTIC_HINTS.items():
            score = len(words & hints)
            if score > best_score:
                best_label = label
                best_score = score
        if best_label == "unknown":
            if polarity > 0:
                best_label = "positive_evaluation"
            elif polarity < 0:
                best_label = "negative_evaluation"
        return best_label, polarity

    def _next_probe_axis(
        self,
        senses: Sequence[LearnedSenseRecord],
        semantic_class: str,
        polarity: int,
    ) -> str:
        best = senses[0] if senses else None
        if polarity == 0 and (not best or 112 <= best.vector.v <= 144):
            return "polarity"
        if semantic_class == "unknown":
            return "semantic_class"
        if not best or 108 <= best.vector.a <= 148:
            return "intensity"
        return "example"

    def _probe_contract(self, pending: PendingDefinition) -> AnswerContract:
        return AnswerContract(
            status=AnswerStatus.LEXICAL_PROBE,
            certainty=255,
            source=SourceKind.INFERRED,
            reason="unknown lexical item blocks a confident interpretation",
            response_goal="lexical_probe",
            required_slots={
                "term": pending.term,
                "probe_axis": pending.probe_axis,
                "part_of_speech": pending.part_of_speech,
                "semantic_position": pending.semantic_position,
            },
            forbidden_claims=["invent_word_meaning", "promote_single_example_to_global_truth"],
        )

    def _definition_query(self, term: str) -> LearningOutcome:
        normalized = self.normalize_term(term)
        senses = self.store.learned_senses(normalized, min_confidence=0.0)
        if senses:
            best = senses[0]
            proposition = EventFrame(
                predicate="mean",
                arguments={
                    "agent": SemanticRef.literal(normalized, term, EntityKind.ABSTRACT),
                    "patient": SemanticRef.literal(
                        best.semantic_class,
                        best.semantic_class.replace("_", " "),
                        EntityKind.ABSTRACT,
                    ),
                },
                source=SourceKind.INFERRED,
                certainty=round(best.confidence * 255),
            )
            return LearningOutcome(
                handled=True,
                contract=AnswerContract(
                    status=AnswerStatus.ANSWERED,
                    proposition=proposition,
                    values=[proposition.arguments["patient"]],
                    certainty=proposition.certainty,
                    source=SourceKind.INFERRED,
                    reason="versioned learned lexical hypothesis",
                    response_goal="answer",
                    required_slots={"requested_role": "patient"},
                ),
                metadata={
                    "action": "lexical_lookup",
                    "term": term,
                    "sense_count": len(senses),
                    "confidence": best.confidence,
                    "version": best.version,
                },
                suppress_fact_storage=True,
            )

        input_affect = self.affect.analyze(term)
        self._open_pending(
            term,
            term,
            part_of_speech="unknown",
            semantic_position="definition_query",
            input_affect=input_affect,
        )
        assert self.pending is not None
        return LearningOutcome(
            handled=True,
            contract=self._probe_contract(self.pending),
            metadata={"action": "lexical_probe", "term": term, "probe_axis": "definition"},
            suppress_fact_storage=True,
        )

    def _looks_like_definition_reply(self, text: str) -> bool:
        words = [token.norm for token in lexicon.tokenize(text, include_punctuation=False)]
        if not words:
            return False
        if words[0] in lexicon.QUESTION_WORDS or words[0] in lexicon.YES_NO_STARTERS:
            return False
        if len(words) <= 12:
            return True
        return any(marker in words for marker in {"means", "like", "basically", "positive", "negative"})

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Snapshot support
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "pending": self.pending.to_dict() if self.pending else None,
        }

    def restore(self, data: Mapping[str, Any]) -> None:
        self.scope_id = str(data.get("scope_id", self.scope_id))
        pending = data.get("pending")
        self.pending = PendingDefinition.from_dict(pending) if pending else None
