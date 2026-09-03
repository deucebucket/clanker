"""End-to-end adaptive, template-free Clanker-LM runtime."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .affect import AffectBackend, AffectController, ClankerAffectBackend
from .contracts import validate_public_api_contract, validate_runtime_instance
from .database import LanguageStore
from .gates import ContextGate
from .learning import LearningOutcome, LexicalLearner
from .memory import ConversationMemory
from .model import (
    AffectVector,
    AnswerContract,
    AnswerStatus,
    EventFrame,
    ParseResult,
    SourceKind,
    SpeechAct,
    TurnResult,
)
from .parser import SemanticParser
from .qa import QuestionAnswerer
from .realize import SurfaceRealizer
from .resolvers import ResolverOutcome, ResolverRegistry
from .trajectory import CorpusProfile, CorpusProfiler, TrajectoryController


class ClankerLM:
    """Deterministic semantic runtime driven by the existing Clanker engine.

    The runtime does not predict tokens and stores no response sentences.  It
    parses input into semantic frames, learns unknown lexical senses through
    evidence, binds live semantic commands through deterministic resolvers,
    composes legal replies from atomic words and grammar, and uses Clanker's
    A+B=C transition to select the candidate that best reaches the target state.
    """

    SNAPSHOT_VERSION = 3
    COMPATIBLE_SNAPSHOT_VERSIONS = {1, 2, 3}
    MAX_BATCH_MESSAGES = 10_000

    def __init__(
        self,
        *,
        memory: Optional[ConversationMemory] = None,
        language_store: Optional[LanguageStore] = None,
        affect_backend: Optional[AffectBackend] = None,
        personality: Any = None,
        perspective: str = "speaker",
        default_timezone: str = "America/Chicago",
        clock: Optional[Callable[[], datetime]] = None,
        learning_scope: str = "default",
        active_profile_id: Optional[str] = None,
    ) -> None:
        validate_public_api_contract(type(self), ConversationMemory, LanguageStore)
        self.memory = memory or ConversationMemory()
        self.store = language_store or LanguageStore()
        if affect_backend is None and personality is not None:
            try:
                affect_backend = ClankerAffectBackend(
                    personality=personality,
                    perspective=perspective,
                )
            except (ImportError, ModuleNotFoundError):
                affect_backend = None
        self.affect = AffectController(affect_backend)
        self.parser = SemanticParser()
        self.answerer = QuestionAnswerer()
        self.gate = ContextGate(self.store)
        self.realizer = SurfaceRealizer(self.memory, self.store)
        self.learner = LexicalLearner(
            self.store,
            self.affect,
            self.memory,
            scope_id=learning_scope,
        )
        self.resolvers = ResolverRegistry(
            self.store,
            default_timezone=default_timezone,
            clock=clock,
        )
        self.trajectory = TrajectoryController(self.store)
        self.profiler = CorpusProfiler(self.store, self.affect)
        self.active_profile_id = active_profile_id
        self.observed_state = AffectVector()
        self.predicted_state = AffectVector()
        self.last_result: Optional[TurnResult] = None
        self._lock = threading.RLock()
        validate_runtime_instance(self)

    @property
    def affect_backend_name(self) -> str:
        return self.affect.backend.name

    @property
    def default_timezone(self) -> str:
        return self.resolvers.default_timezone

    def process(self, text: str) -> TurnResult:
        with self._lock:
            self.memory.begin_turn()

            base_affect = self.affect.analyze(text)
            input_affect = self.learner.apply_overlay(text, base_affect)
            # Carry the assistant's selected response into the next turn.
            # Before the first response both states are neutral; afterwards the
            # predicted post-response state is the conversational prior.
            state_before = self.predicted_state if self.last_result is not None else self.observed_state
            observed = self.affect.observe(state_before, input_affect.vector)
            finalized = self.trajectory.finalize_pending(
                reaction_vector=input_affect.vector,
                observed_next=observed,
            )

            learning = self.learner.preparse(text, input_affect)
            resolver = ResolverOutcome()
            if learning.handled:
                parse = self._synthetic_parse(text, learning)
                assert learning.contract is not None
                contract = learning.contract
            else:
                resolver = self.resolvers.resolve(text)
                if resolver.handled:
                    parse = self._synthetic_resolver_parse(text, resolver)
                    assert resolver.contract is not None
                    contract = resolver.contract
                else:
                    parse = self.parser.parse(text, self.memory)
                    learning = self.learner.postparse(text, parse, input_affect)
                    if learning.handled:
                        assert learning.contract is not None
                        contract = learning.contract
                    else:
                        passive_learning = self.learner.observe_known_context(text, base_affect)
                        if passive_learning:
                            learning.metadata.update(passive_learning)
                        contract = self._resolve_contract(parse)

            gates = self.gate.decide(
                text,
                parse,
                input_affect,
                self.memory,
                answer_status=contract.status,
            )
            candidates = self.realizer.realize(contract, gates)

            base_target = self.affect.target_for(observed, contract, gates)
            profile_target, profile_adjustment = self.profiler.adjust_target(
                base_target,
                self.active_profile_id,
                turn_index=self.memory.turn_index,
                severity=gates.severity,
            )
            context_key = self.trajectory.context_key(
                parse=parse,
                contract=contract,
                gates=gates,
                input_vector=input_affect.vector,
                observed=observed,
                profile_id=self.active_profile_id,
            )
            corrected_target, transition_adjustment = self.trajectory.adjust_target(
                profile_target,
                context_key,
            )
            target = self._enforce_target_floors(
                corrected_target,
                base_target=base_target,
                gates=gates,
            )

            selected, scored = self.affect.rank_candidates(candidates, observed, target)
            predicted = selected.predicted_state or observed

            trajectory_id = self.trajectory.record(
                input_text=text,
                response_text=selected.text,
                parse=parse,
                contract=contract,
                gates=gates,
                input_vector=input_affect.vector,
                state_before=state_before,
                target=target,
                selected=selected,
                predicted_after=predicted,
                context_key=context_key,
                profile_id=self.active_profile_id,
            )

            self.observed_state = observed
            self.predicted_state = predicted
            trajectory_metadata = {
                "previous_finalization": finalized.to_dict(),
                "current_trajectory_id": trajectory_id,
                "context_key": context_key,
                "profile_adjustment": profile_adjustment,
                "transition_adjustment": transition_adjustment,
                "base_target": base_target.to_dict(),
                "final_target": target.to_dict(),
                "stores_raw_text": False,
            }
            result = TurnResult(
                input_text=text,
                response=selected.text,
                parse=parse,
                contract=contract,
                gates=gates,
                input_affect=input_affect,
                observed_state=observed,
                target_state=target,
                predicted_state=predicted,
                candidates=scored,
                memory_revision=self.memory.revision,
                learning=learning.metadata or None,
                resolver=resolver.metadata if resolver.handled else None,
                trajectory=trajectory_metadata,
            )
            self.last_result = result
            return result

    @staticmethod
    def _synthetic_parse(text: str, outcome: LearningOutcome) -> ParseResult:
        status = outcome.contract.status if outcome.contract else AnswerStatus.UNSUPPORTED
        speech_act = SpeechAct.ASK if status == AnswerStatus.ANSWERED else SpeechAct.CLARIFY
        return ParseResult(
            speech_act=speech_act,
            raw_text=text,
            normalized_text=" ".join(text.lower().split()),
            diagnostics=[
                "active lexical learner handled the turn",
                f"learning_status={status.value}",
            ],
        )

    @staticmethod
    def _synthetic_resolver_parse(text: str, outcome: ResolverOutcome) -> ParseResult:
        command = str(outcome.metadata.get("command", "semantic_command"))
        return ParseResult(
            speech_act=SpeechAct.ASK,
            raw_text=text,
            normalized_text=" ".join(text.lower().split()),
            diagnostics=[
                "deterministic semantic resolver handled the turn",
                f"command={command}",
            ],
        )

    @staticmethod
    def _enforce_target_floors(
        target: AffectVector,
        *,
        base_target: AffectVector,
        gates: Any,
    ) -> AffectVector:
        values = target.to_dict()
        if gates.severity == "critical":
            for axis in ("d", "u", "w", "i"):
                values[axis] = max(values[axis], getattr(base_target, axis))
            values["a"] = min(values["a"], max(150, base_target.a))
        elif gates.masking or gates.severity == "high":
            values["i"] = max(values["i"], min(195, base_target.i))
            values["w"] = max(values["w"], min(150, base_target.w))
        return AffectVector(**values)

    def _resolve_contract(self, parse: ParseResult) -> AnswerContract:
        if parse.question:
            return self.answerer.answer(parse.question, self.memory)

        if parse.unresolved:
            unresolved = parse.unresolved[0]
            if unresolved.candidates:
                return AnswerContract(
                    status=AnswerStatus.AMBIGUOUS_REFERENCE,
                    certainty=255,
                    source=SourceKind.INFERRED,
                    reason=unresolved.surface,
                    response_goal="clarify",
                    required_slots={
                        "reference": unresolved.surface,
                        "candidate_ids": ",".join(unresolved.candidates),
                    },
                )
            return AnswerContract(
                status=AnswerStatus.MISSING_REFERENCE,
                certainty=255,
                source=SourceKind.INFERRED,
                reason=unresolved.surface,
                response_goal="clarify",
                required_slots={"reference": unresolved.surface},
            )

        if parse.speech_act == SpeechAct.ASSERT and parse.events:
            stored = [self.memory.add_event(event) for event in parse.events]
            self.memory.add_clause_relations(parse.relations, stored)
            self.memory.add_entity_modifier_relations(parse.modifiers, stored)
            return AnswerContract(
                status=AnswerStatus.ACKNOWLEDGED,
                proposition=stored[-1],
                evidence=[],
                certainty=min(event.certainty for event in stored),
                source=SourceKind.USER,
                reason=f"stored {len(stored)} proposition(s)",
                response_goal="acknowledge",
            )

        if parse.speech_act in {SpeechAct.GREET, SpeechAct.SOCIAL}:
            return AnswerContract(
                status=AnswerStatus.ACKNOWLEDGED,
                certainty=255,
                source=SourceKind.TRAINED,
                reason="social convention",
                response_goal="social",
            )

        return AnswerContract(
            status=AnswerStatus.UNSUPPORTED,
            certainty=0,
            source=SourceKind.UNKNOWN,
            reason="the deterministic parser did not produce a supported frame",
            response_goal="clarify",
        )

    def process_many(
        self,
        messages: Iterable[str],
        *,
        continue_on_error: bool = False,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        max_messages: Optional[int] = None,
    ) -> List[TurnResult]:
        """Process a bounded message batch with explicit failure policy.

        Fail-fast remains the default so callers never silently lose a turn.
        Batch importers may opt into ``continue_on_error`` and receive each
        rejected item through ``on_error``. Failed items are omitted from the
        returned list; successful turns preserve their original order. The
        hard batch ceiling prevents an unbounded iterable from consuming the
        runtime indefinitely; callers should chunk larger imports explicitly.
        """

        if max_messages is None:
            limit = self.MAX_BATCH_MESSAGES
        elif isinstance(max_messages, bool) or not isinstance(max_messages, int):
            raise TypeError("max_messages must be an integer or None")
        else:
            limit = max_messages
        if not 1 <= limit <= self.MAX_BATCH_MESSAGES:
            raise ValueError(
                f"max_messages must be between 1 and {self.MAX_BATCH_MESSAGES}"
            )

        results: List[TurnResult] = []
        for index, message in enumerate(messages):
            if index >= limit:
                raise ValueError(
                    f"message batch exceeds the configured limit of {limit}"
                )
            try:
                results.append(self.process(message))
            except Exception as exc:
                # This is an intentional batch boundary. Individual runtime
                # calls still expose their original exception unless the
                # caller explicitly requests continuation.
                if not continue_on_error:
                    raise
                if on_error is not None:
                    on_error(message, exc)
        return results

    def learn(
        self,
        text: str,
        *,
        source: SourceKind = SourceKind.RETRIEVED,
        certainty: int = 220,
        inferred: bool = False,
    ) -> List[EventFrame]:
        """Ingest trusted text as semantic evidence without generating a reply."""

        with self._lock:
            self.memory.begin_turn()
            parsed = self.parser.parse(text, self.memory)
            if parsed.unresolved:
                surfaces = ", ".join(item.surface for item in parsed.unresolved)
                raise ValueError(f"Cannot learn text with unresolved references: {surfaces}")
            if not parsed.events:
                raise ValueError("Cannot learn text that does not produce a supported proposition")
            stored: List[EventFrame] = []
            for event in parsed.events:
                event.source = source
                event.certainty = max(0, min(255, int(certainty)))
                event.inferred = bool(inferred)
                stored.append(self.memory.add_event(event))
            self.memory.add_clause_relations(parsed.relations, stored)
            return stored

    def learn_many(
        self,
        statements: Iterable[str],
        *,
        source: SourceKind = SourceKind.RETRIEVED,
        certainty: int = 220,
        inferred: bool = False,
    ) -> List[EventFrame]:
        stored: List[EventFrame] = []
        for statement in statements:
            stored.extend(
                self.learn(
                    statement,
                    source=source,
                    certainty=certainty,
                    inferred=inferred,
                )
            )
        return stored

    def remember_event(self, event: EventFrame) -> EventFrame:
        """Store an already-structured event supplied by a trusted adapter."""

        with self._lock:
            if not event.predicate or any(ref.is_variable for ref in event.arguments.values()):
                raise ValueError("Stored evidence must be a closed proposition")
            if event.turn_index <= 0:
                self.memory.begin_turn()
                event.turn_index = self.memory.turn_index
            return self.memory.add_event(event)

    # ------------------------------------------------------------------
    # Adaptive lexical and corpus interfaces
    # ------------------------------------------------------------------

    def learned_lexicon(self) -> List[Dict[str, Any]]:
        return self.store.learned_terms_summary()

    def compile_corpus_profile(
        self,
        name: str,
        text: str,
        *,
        profile_id: Optional[str] = None,
        activate: bool = False,
    ) -> CorpusProfile:
        with self._lock:
            profile = self.profiler.compile_text(name, text, profile_id=profile_id)
            if activate:
                self.active_profile_id = profile.profile_id
            return profile

    def match_corpus(self, text: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            return self.profiler.match_text(text, top_k=top_k)

    def set_tone_profile(self, profile_id: Optional[str]) -> None:
        if profile_id is not None and not self.store.get_corpus_profile(profile_id):
            raise KeyError(f"Unknown corpus profile: {profile_id}")
        self.active_profile_id = profile_id

    def explain_last(self) -> Dict[str, Any]:
        if not self.last_result:
            return {"error": "no turn has been processed"}
        result = self.last_result
        return {
            "selected_response": result.response,
            "parse": result.parse.to_dict(),
            "answer_contract": result.contract.to_dict(),
            "gates": result.gates.to_dict(),
            "affect_backend": result.input_affect.backend,
            "input_vector": result.input_affect.vector.to_dict(),
            "observed_state": result.observed_state.to_dict(),
            "target_state": result.target_state.to_dict(),
            "predicted_state": result.predicted_state.to_dict(),
            "candidate_ranking": [candidate.to_dict() for candidate in result.candidates],
            "memory_revision": result.memory_revision,
            "learning": result.learning,
            "resolver": result.resolver,
            "trajectory": result.trajectory,
            "template_free": self.store.schema_summary()["template_tables"] == 0,
        }

    # ------------------------------------------------------------------
    # Snapshot persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "snapshot_version": self.SNAPSHOT_VERSION,
                "memory": self.memory.to_dict(),
                "observed_state": self.observed_state.to_dict(),
                "predicted_state": self.predicted_state.to_dict(),
                "affect_backend": self.affect_backend_name,
                "language_overlay": self.store.export_overlay(),
                "learner": self.learner.to_dict(),
                "default_timezone": self.default_timezone,
                "active_profile_id": self.active_profile_id,
            }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        language_store: Optional[LanguageStore] = None,
        affect_backend: Optional[AffectBackend] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> "ClankerLM":
        version = int(data.get("snapshot_version", 0))
        if version not in cls.COMPATIBLE_SNAPSHOT_VERSIONS:
            raise ValueError(f"Unsupported Clanker-LM snapshot version: {version}")
        store = language_store or LanguageStore()
        if version >= 2 and data.get("language_overlay"):
            store.import_overlay(data["language_overlay"])
        runtime = cls(
            memory=ConversationMemory.from_dict(data["memory"]),
            language_store=store,
            affect_backend=affect_backend,
            default_timezone=str(data.get("default_timezone", "America/Chicago")),
            clock=clock,
            active_profile_id=data.get("active_profile_id"),
        )
        runtime.observed_state = AffectVector(**dict(data.get("observed_state", {})))
        runtime.predicted_state = AffectVector(**dict(data.get("predicted_state", {})))
        if version >= 2 and data.get("learner"):
            runtime.learner.restore(data["learner"])
        return runtime

    def dumps(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def loads(
        cls,
        text: str,
        *,
        language_store: Optional[LanguageStore] = None,
        affect_backend: Optional[AffectBackend] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> "ClankerLM":
        return cls.from_dict(
            json.loads(text),
            language_store=language_store,
            affect_backend=affect_backend,
            clock=clock,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps(indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        language_store: Optional[LanguageStore] = None,
        affect_backend: Optional[AffectBackend] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> "ClankerLM":
        return cls.loads(
            Path(path).read_text(encoding="utf-8"),
            language_store=language_store,
            affect_backend=affect_backend,
            clock=clock,
        )

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "ClankerLM":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
