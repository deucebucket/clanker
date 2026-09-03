"""End-to-end deterministic Clanker-LM conversational runtime."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from .affect import AffectAdapter, CandidateScorer, ClankerAffectAdapter
from .answers import AnswerEngine, AnswerRealizer
from .constructions import ConstructionGraph
from .gating import GateEngine, ResponsePlanner
from .memory import ConversationMemory
from .models import (
    AnswerStatus,
    Fact,
    Provenance,
    SemanticFrame,
    SpeechAct,
    TurnResult,
)
from .normalize import split_sentences
from .parser import SemanticParser
from .persistence import SQLiteSessionStore


class ClankerLM:
    """Symbolic dialogue runtime with Clanker as the affective kernel.

    Processing order:
        text -> semantic frames -> evidence binding -> response contract
             -> contextual gates -> construction candidates
             -> Clanker back-solve -> verified deterministic response
    """

    def __init__(
        self,
        *,
        session_id: str = "default",
        db_path: Optional[str | Path] = None,
        affect: Optional[AffectAdapter] = None,
        personality=None,
        strict_clanker: bool = False,
        construction_path: Optional[str | Path] = None,
    ) -> None:
        self.store = SQLiteSessionStore(db_path) if db_path is not None else None
        loaded = self.store.load(session_id) if self.store else None
        self.memory = loaded or ConversationMemory(session_id=session_id)
        self.affect = affect or ClankerAffectAdapter(
            personality=personality, strict=strict_clanker
        )
        self.parser = SemanticParser(self.memory)
        self.gates = GateEngine()
        self.planner = ResponsePlanner(self.memory)
        self.answer_engine = AnswerEngine(self.memory)
        self.answer_realizer = AnswerRealizer(self.memory)
        self.graph = ConstructionGraph(construction_path)
        self.scorer = CandidateScorer(self.affect)
        self.last_result: Optional[TurnResult] = None

    def process(
        self,
        text: str,
        *,
        provenance: Provenance = Provenance.USER,
        certainty: int = 230,
    ) -> TurnResult:
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")

        self.memory.begin_turn()
        input_affect = self.affect.analyze(text)
        receiver_state = self.affect.transition(
            self.memory.running_state, input_affect.vector
        )

        segments = split_sentences(text)
        parsed_segments = [self.parser.parse(segment) for segment in segments]
        parsed = parsed_segments[-1]
        if len(parsed_segments) > 1:
            parsed = replace(
                parsed,
                register_score=max(item.register_score for item in parsed_segments),
                severity_score=max(item.severity_score for item in parsed_segments),
                familial=any(item.familial for item in parsed_segments),
                repeated=any(item.repeated for item in parsed_segments),
                metadata={
                    **parsed.metadata,
                    "segment_count": len(parsed_segments),
                },
            )

        stored: list[str] = []
        # Earlier declarative sentences in the same turn become available to a
        # final question immediately.
        for item in parsed_segments[:-1]:
            stored.extend(
                self._remember_parsed_fact(item, provenance=provenance, certainty=certainty)
            )

        gate_profile = self.gates.evaluate(parsed, input_affect)
        answer = None
        if parsed.speech_act in {SpeechAct.QUESTION, SpeechAct.SOCIAL_CHECKIN} and parsed.question:
            answer = self.answer_engine.answer(parsed.question)
            candidates = self.answer_realizer.candidates(answer)
            if answer.status == AnswerStatus.NEEDS_CONTEXT:
                mode = "clarify"
            elif answer.status == AnswerStatus.RHETORICAL:
                mode = "support"
            elif parsed.speech_act == SpeechAct.SOCIAL_CHECKIN:
                mode = "social"
            else:
                mode = "factual"
        else:
            stored.extend(
                self._remember_parsed_fact(parsed, provenance=provenance, certainty=certainty)
            )
            plan = self.planner.plan_statement(parsed, gate_profile, input_affect)
            candidates = self.graph.traverse(plan)
            mode = plan.target_mode

        selection = self.scorer.choose(
            candidates,
            current=receiver_state,
            gates=gate_profile,
            mode=mode,
            contract=answer,
        )
        self.memory.running_state = selection.outcome
        if self.store:
            self.store.save(self.memory)

        result = TurnResult(
            response=selection.candidate.text,
            parsed=parsed,
            gates=gate_profile,
            input_affect=input_affect,
            response_affect=selection.response_reading,
            resulting_state=selection.outcome,
            answer=answer,
            candidate_scores=selection.scores,
            stored_fact_ids=tuple(stored),
        )
        self.last_result = result
        return result

    def reply(self, text: str, **kwargs) -> str:
        return self.process(text, **kwargs).response

    def add_fact(
        self,
        frame: SemanticFrame,
        *,
        provenance: Provenance = Provenance.RETRIEVED,
        certainty: int = 220,
    ) -> Fact:
        if self.memory.turn == 0:
            self.memory.begin_turn()
        fact = self.memory.remember_fact(
            frame, provenance=provenance, certainty=certainty
        )
        if self.store:
            self.store.save(self.memory)
        return fact

    def reset(self) -> None:
        session_id = self.memory.session_id
        if self.store:
            self.store.reset(session_id)
        self.memory = ConversationMemory(session_id=session_id)
        self.parser = SemanticParser(self.memory)
        self.planner = ResponsePlanner(self.memory)
        self.answer_engine = AnswerEngine(self.memory)
        self.answer_realizer = AnswerRealizer(self.memory)
        self.last_result = None

    def close(self) -> None:
        if self.store:
            self.store.close()
            self.store = None

    def _remember_parsed_fact(
        self,
        parsed,
        *,
        provenance: Provenance,
        certainty: int,
    ) -> Tuple[str, ...]:
        if (
            parsed.speech_act != SpeechAct.STATEMENT
            or parsed.frame is None
            or parsed.unresolved_references
            or parsed.frame.predicate in {"unknown", "rhetorical"}
        ):
            return ()
        fact = self.memory.remember_fact(
            parsed.frame,
            provenance=provenance,
            certainty=certainty,
        )
        return (fact.fact_id,)

    def __enter__(self) -> "ClankerLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
