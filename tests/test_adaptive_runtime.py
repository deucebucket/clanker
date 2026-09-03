from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from clanker_lm import ClankerLM, HeuristicAffectBackend, LanguageStore
from clanker_lm.model import (
    AffectVector,
    AnswerContract,
    AnswerStatus,
    GateDecision,
    ParseResult,
    SpeechAct,
)
from clanker_lm.resolvers import SafeArithmetic
from clanker_lm.trajectory import CorpusProfiler, TrajectoryController


def fixed_clock() -> datetime:
    return datetime(2026, 9, 3, 16, 5, tzinfo=timezone.utc)


def test_language_store_has_no_sentence_templates_or_surface_phrases():
    with LanguageStore() as store:
        store.assert_template_free()
        tables = {
            row["name"]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert not ({"constructions", "construction_slots", "graph_edges"} & tables)
        assert all(
            not any(character.isspace() for character in row["surface"])
            for row in store.connection.execute("SELECT surface FROM atoms")
        )
        assert all(
            "template" not in row["name"].lower()
            for row in store.connection.execute("PRAGMA table_info(grammar_rules)")
        )


def test_candidates_expose_atomic_and_semantic_generation_trace(runtime: ClankerLM):
    runtime.process("Sarah bought a car.")
    result = runtime.process("Who bought the car?")
    assert result.contract.status == AnswerStatus.ANSWERED
    selected = next(item for item in result.candidates if item.text == result.response)
    assert selected.semantic_plan
    assert "DECLARATIVE_CLAUSE" in selected.semantic_plan
    assert "punct.period" in selected.atom_ids
    assert not any("template" in step.lower() for step in selected.semantic_plan)


def test_unknown_word_opens_one_evidence_backed_definition_probe(runtime: ClankerLM):
    result = runtime.process("That movie was glorp.")
    assert result.contract.status == AnswerStatus.LEXICAL_PROBE
    assert result.response == "What does glorp mean?"
    assert result.learning and result.learning["term"].lower() == "glorp"
    assert not runtime.memory.events
    rows = runtime.store.learned_terms_summary()
    assert len(rows) == 1 and rows[0]["normalized"] == "glorp"
    assert runtime.store.lexical_evidence(rows[0]["term_id"])


def test_unhelpful_definition_is_saved_then_reduced_to_one_word_probe(runtime: ClankerLM):
    runtime.process("That movie was glorp.")
    result = runtime.process("Something.")
    assert result.contract.status == AnswerStatus.LEXICAL_PROBE
    assert result.response == "Polarity?"
    assert result.learning and result.learning["saved_evidence"]
    assert len(result.response.rstrip("?").split()) == 1
    evidence = runtime.store.lexical_evidence(
        runtime.store.learned_terms_summary()[0]["term_id"]
    )
    assert any(item["raw_explanation"] == "Something." for item in evidence)


def test_definition_promotes_a_local_sense_and_affect_overlay_reuses_it(runtime: ClankerLM):
    runtime.process("That movie was glorp.")
    learned = runtime.process("Negative, like disappointing and overhyped.")
    assert learned.contract.status == AnswerStatus.LEXICAL_LEARNED
    assert "negative evaluation" in learned.response.lower()
    sense = runtime.store.learned_senses("glorp")[0]
    assert sense.confidence >= 0.68
    assert sense.vector.v < 128

    reused = runtime.process("The sequel was glorp.")
    assert reused.contract.status == AnswerStatus.ACKNOWLEDGED
    assert reused.input_affect.vector.v < 128
    assert reused.input_affect.backend.endswith("+learned-overlay")
    assert reused.learning and reused.learning["action"] == "lexical_context_update"


def test_corrective_context_versions_and_splits_conflicting_word_senses(runtime: ClankerLM):
    runtime.process("That movie was glorp.")
    runtime.process("Negative, like disappointing.")
    first_version = runtime.store.learned_senses("glorp")[0].version
    corrected = runtime.process("Glorp means amazing and excellent.")
    assert corrected.contract.status == AnswerStatus.LEXICAL_LEARNED
    senses = runtime.store.learned_senses("glorp", min_confidence=0.0)
    assert len(senses) == 2
    assert all(item.version > first_version for item in senses[:1])
    assert {item.conditions.get("context_polarity") for item in senses} == {
        "positive",
        "negative",
    }
    term = runtime.store.term_row("glorp")
    assert term and term["status"] == "split_into_multiple_senses"
    evidence = runtime.store.lexical_evidence(term["term_id"])
    assert all(item["reanalyzed_version"] >= 2 for item in evidence)


def test_learned_word_query_and_overlay_survive_snapshot(tmp_path):
    path = tmp_path / "adaptive.json"
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("That movie was glorp.")
        runtime.process("Negative, like disappointing.")
        runtime.save(path)
    finally:
        runtime.close()

    loaded = ClankerLM.load(path, affect_backend=HeuristicAffectBackend())
    try:
        answer = loaded.process("What does glorp mean?")
        assert answer.contract.status == AnswerStatus.ANSWERED
        assert "negative" in answer.response.lower()
        assert loaded.store.learned_senses("glorp")
    finally:
        loaded.close()


def test_live_time_date_and_calculation_are_typed_ephemeral_observations():
    runtime = ClankerLM(
        affect_backend=HeuristicAffectBackend(),
        clock=fixed_clock,
        default_timezone="America/Chicago",
    )
    try:
        time_result = runtime.process("What time is it in Tokyo?")
        assert time_result.contract.status == AnswerStatus.ANSWERED
        assert "1:05 AM" in time_result.response
        assert time_result.resolver and time_result.resolver["command"] == "CURRENT_TIME"
        assert time_result.resolver["expires_at"]

        date_result = runtime.process("What's today's date?")
        assert "Thursday, September 3, 2026" in date_result.response
        assert date_result.resolver and date_result.resolver["command"] == "CURRENT_DATE"

        calculation = runtime.process("What is 2 + 3 * 4?")
        assert calculation.response == "2 + 3 * 4 is 14."
        assert calculation.resolver and calculation.resolver["command"] == "CALCULATE"

        observations = runtime.store.connection.execute(
            "SELECT * FROM resolver_observations ORDER BY observation_id"
        ).fetchall()
        assert len(observations) == 3
        assert observations[0]["expires_at"] is not None
        assert all(row["request_hash"] and len(row["request_hash"]) == 64 for row in observations)
    finally:
        runtime.close()


def test_unknown_timezone_fails_closed_without_guessing():
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend(), clock=fixed_clock)
    try:
        result = runtime.process("What time is it in Atlantis?")
        assert result.contract.status == AnswerStatus.UNKNOWN
        assert "timezone" in result.response.lower()
        assert "guess_timezone" in result.contract.forbidden_claims
    finally:
        runtime.close()


def test_safe_arithmetic_rejects_code_and_unbounded_power():
    assert str(SafeArithmetic.evaluate("2 + 3 * 4")) == "14"
    with pytest.raises(ValueError):
        SafeArithmetic.evaluate("__import__('os').system('id')")
    with pytest.raises(ValueError):
        SafeArithmetic.evaluate("2 ** 999")
    with pytest.raises(ZeroDivisionError):
        SafeArithmetic.evaluate("1 / 0")


def test_trajectory_statistics_learn_signed_residual_and_correct_future_target():
    with LanguageStore() as store:
        controller = TrajectoryController(store)
        context = "assert:feel|acknowledge|acknowledged|moderate|plain|0000000|0000000|none"
        for index in range(2):
            trajectory_id = store.record_trajectory(
                input_hash=f"i{index}",
                response_hash=f"r{index}",
                incoming_act="assert:feel",
                response_act="acknowledge",
                context_key=context,
                input_vector=AffectVector(v=80, a=170),
                state_before=AffectVector(v=100),
                target_vector=AffectVector(v=145),
                response_vector=AffectVector(v=150),
                predicted_after=AffectVector(v=140),
                profile_id=None,
            )
            store.finalize_trajectory(
                trajectory_id,
                observed_next=AffectVector(v=120),
                reaction_vector=AffectVector(v=90),
                residual={"v": -20, "a": 0, "d": 0, "u": 0, "g": 0, "w": 0, "i": 0},
                success=0.5,
            )
        stat = store.transition_stat(context)
        assert stat and stat["sample_count"] == 2
        assert stat["mean_residual"]["v"] == pytest.approx(-20.0)
        adjusted, meta = controller.adjust_target(AffectVector(v=145), context)
        assert meta["applied"]
        assert adjusted.v > 145


def test_runtime_finalizes_previous_turn_without_storing_sentences(runtime: ClankerLM):
    first = runtime.process("Sarah bought a car.")
    second = runtime.process("Who bought the car?")
    assert first.trajectory and second.trajectory
    assert second.trajectory["previous_finalization"]["finalized"]
    rows = runtime.store.connection.execute(
        "SELECT input_hash, response_hash, observed_next_json FROM trajectory_turns ORDER BY trajectory_id"
    ).fetchall()
    assert len(rows) == 2
    assert all(len(row["input_hash"]) == 64 and len(row["response_hash"]) == 64 for row in rows)
    assert rows[0]["observed_next_json"] is not None
    columns = {row["name"] for row in runtime.store.connection.execute("PRAGMA table_info(trajectory_turns)")}
    assert "input_text" not in columns and "response_text" not in columns


def test_corpus_profile_stores_only_math_and_can_match_exact_dialogue():
    text = (
        'He said, "I cannot believe you did that." '
        'She answered, "I was trying to help." '
        'He asked, "Why did you hide it?" '
        'She said, "Because I was afraid."'
    )
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        profile = runtime.compile_corpus_profile("argument-scene", text, activate=True)
        assert profile.quote_count == 4
        assert profile.metadata["stored_raw_text"] is False
        stored = runtime.store.get_corpus_profile(profile.profile_id)
        assert stored
        assert b"I cannot believe" not in stored["trajectory_blob"]
        assert b"I cannot believe" not in stored["delta_blob"]
        textual_metadata = {key: value for key, value in stored.items() if not isinstance(value, bytes)}
        assert "I cannot believe" not in json.dumps(textual_metadata)
        assert len(stored["trajectory_blob"]) == profile.quote_count * 7
        chunks = runtime.store.corpus_chunks(profile.profile_id)
        assert chunks and all(isinstance(item["vector_blob"], bytes) for item in chunks)
        matches = runtime.match_corpus(text)
        assert matches[0]["profile_id"] == profile.profile_id
        assert matches[0]["fingerprint_exact"]

        result = runtime.process("Sarah bought a car.")
        assert result.trajectory["profile_adjustment"]["applied"]
        assert result.trajectory["profile_adjustment"]["profile_id"] == profile.profile_id
    finally:
        runtime.close()


def test_packed_vadugwi_trajectory_is_seven_bytes_per_turn():
    vectors = [AffectVector(v=1, a=2, d=3, u=4, g=5, w=6, i=7), AffectVector()]
    blob = CorpusProfiler.pack_vectors(vectors)
    assert len(blob) == 14
    assert CorpusProfiler.unpack_vectors(blob) == vectors


def test_unknown_definition_query_states_uncertainty_and_binds_it_reply(runtime: ClankerLM):
    probe = runtime.process("What does zorb mean?")
    assert probe.contract.status == AnswerStatus.LEXICAL_PROBE
    assert probe.response == "I do not know what zorb means. Example?"
    assert probe.response not in {
        row["surface"]
        for row in runtime.store.connection.execute("SELECT surface FROM atoms")
    }
    selected = next(item for item in probe.candidates if item.text == probe.response)
    assert "TRUTH:UNKNOWN" in selected.semantic_plan
    assert "ACT:REQUEST_EXAMPLE" in selected.semantic_plan

    learned = runtime.process("It means super intense.")
    assert learned.contract.status == AnswerStatus.LEXICAL_LEARNED
    assert learned.learning and learned.learning["term"].lower() == "zorb"
    assert runtime.store.term_row("it") is None
    zorb_sense = runtime.store.learned_senses("zorb")[0]
    assert zorb_sense.semantic_class == "intensity"
    assert zorb_sense.vector.a >= 170

    answer = runtime.process("What does zorb mean?")
    assert answer.contract.status == AnswerStatus.ANSWERED
    assert "intensity" in answer.response.lower()
