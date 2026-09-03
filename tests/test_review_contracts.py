"""Regression contracts for automated review findings."""

from __future__ import annotations

from dataclasses import fields
from typing import List, Tuple

import pytest

import clanker_lm.affect as affect_module
from clanker_lm.__main__ import _load_runtime, _read_text_file
from clanker_lm.affect import ClankerAffectBackend, HeuristicAffectBackend
from clanker_lm.database import LearnedSenseRecord
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import AnswerStatus
from clanker_lm.runtime import ClankerLM


def test_learned_sense_record_schema_is_complete() -> None:
    names = {field.name for field in fields(LearnedSenseRecord)}
    assert {
        "sense_id",
        "term_id",
        "sense_index",
        "surface",
        "part_of_speech",
        "semantic_class",
        "register",
        "confidence",
        "status",
        "version",
        "support_weight",
        "contradiction_weight",
        "vector",
        "conditions",
    } <= names


def test_conversation_memory_exposes_monotonic_turn_index() -> None:
    memory = ConversationMemory()
    assert memory.turn_index == 0
    assert memory.begin_turn() == 1
    assert memory.turn_index == 1
    assert memory.begin_turn() == 2
    assert memory.turn_index == 2


def test_process_many_can_continue_after_an_item_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        original_process = runtime.process

        def flaky_process(message: str):
            if message == "<error>":
                raise ValueError("synthetic batch failure")
            return original_process(message)

        monkeypatch.setattr(runtime, "process", flaky_process)
        errors: List[Tuple[str, str]] = []
        results = runtime.process_many(
            ["Hello.", "<error>", "Hello again."],
            continue_on_error=True,
            on_error=lambda message, exc: errors.append((message, str(exc))),
        )

        assert len(results) == 2
        assert errors == [("<error>", "synthetic batch failure")]
    finally:
        runtime.close()


def test_process_many_remains_fail_fast_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        def failing_process(message: str):
            raise ValueError(message)

        monkeypatch.setattr(runtime, "process", failing_process)
        with pytest.raises(ValueError, match="stop"):
            runtime.process_many(["stop"])
    finally:
        runtime.close()

def test_lexical_learning_review_paths_are_executable() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        assert runtime.learner._static_known
        assert all(isinstance(item, str) for item in runtime.learner._static_known)

        learned = runtime.process("Glorp means negative and disappointing.")
        assert learned.contract.status == AnswerStatus.LEXICAL_LEARNED
        senses = runtime.store.learned_senses("glorp", min_confidence=0.0)
        assert senses
        assert all(isinstance(item, LearnedSenseRecord) for item in senses)
        assert all(item.confidence >= 0.0 for item in senses)

        queried = runtime.process("What does glorp mean?")
        assert queried.contract.status == AnswerStatus.ANSWERED
    finally:
        runtime.close()


def test_optional_v8_import_probe_is_defined_at_module_load() -> None:
    assert hasattr(affect_module, "_V8_COMPUTE_VADUG")
    assert hasattr(affect_module, "_V8_VADUG")
    assert hasattr(affect_module, "_V8_STATE_TRANSITION")
    if affect_module._V8_IMPORT_ERROR is None:
        backend = ClankerAffectBackend()
        assert backend.name == "clanker-v8"


def test_corrupt_memory_snapshot_gets_a_helpful_error(tmp_path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid Clanker-LM memory snapshot"):
        _load_runtime(str(path))


def test_cli_text_reader_rejects_oversized_input(tmp_path) -> None:
    path = tmp_path / "oversized.txt"
    path.write_text("0123456789", encoding="utf-8")
    with pytest.raises(ValueError, match="too large"):
        _read_text_file(str(path), max_bytes=5)


def test_process_many_rejects_unbounded_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        monkeypatch.setattr(runtime, "process", lambda message: message)
        with pytest.raises(ValueError, match="exceeds the configured limit"):
            runtime.process_many(["one", "two", "three"], max_messages=2)
        with pytest.raises(ValueError, match="between 1"):
            runtime.process_many([], max_messages=0)
    finally:
        runtime.close()


def test_corpus_chunk_queries_are_bounded() -> None:
    text = (
        '"One." "Two." "Three." "Four." '
        '"Five." "Six." "Seven." "Eight."'
    )
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        profile = runtime.compile_corpus_profile("bounded", text)
        assert len(runtime.store.corpus_chunks(profile.profile_id, limit=1)) == 1
        with pytest.raises(ValueError, match="corpus chunk limit"):
            runtime.store.corpus_chunks(profile.profile_id, limit=0)
    finally:
        runtime.close()
