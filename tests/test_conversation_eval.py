import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import evaluation.conversations.runner as conversation_runner

from clanker_lm.database import LanguageStore
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AffectVector,
    AnswerStatus,
    EventFrame,
    ParseResult,
    QuestionFrame,
    QuestionKind,
    SemanticRef,
    SpeechAct,
    TruthValue,
    UnresolvedReference,
)
from evaluation.conversations.corpus import (
    DATA_DIR,
    MANIFEST_PATH,
    SOURCE_DIR,
    CorpusIntegrityError,
    _canonical_json,
    _manifest_constituents,
    _validate_source_document,
    assert_production_tree,
    compile_corpora,
    load_manifest,
    load_split,
    production_tree_sha256,
    verify_corpus,
)
from evaluation.conversations.runner import (
    FrozenLookupTrajectory,
    _aggregate_resources,
    _answer_exact,
    _classification,
    _correction_store_digest,
    _drift,
    _entity_exact,
    _enforce_zero_execution_errors,
    _metric_summary,
    _paired_mode_differences,
    _pooled_cluster_mean,
    _publish_artifacts,
    _semantic_parse_exact,
    _validate_aggregate_artifacts,
    _assert_provenance_unchanged,
    _whole_interaction_trajectory_metrics,
    run_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_has_frozen_whole_conversation_corpus():
    manifest = load_manifest()
    heldout = manifest["splits"]["heldout"]
    development = manifest["splits"]["development"]
    assert manifest["baseline_code_commit"] == "9ae77f072f8afda0b1d2b757ab492757cabff0f8"
    assert heldout["turn_count"] >= 500
    assert development["turn_count"] >= 60
    assert heldout["domain_turns"] == {
        "public_domain_drama": 120,
        "public_domain_novel": 120,
        "public_domain_real_human": 120,
        "synthetic_adversarial": 160,
    }
    assert heldout["training_eligible"] is False
    assert heldout["teacher_replay_eligible"] is False
    assert DATA_DIR.joinpath("ROOT.sha256").read_text().strip() == manifest["corpus_root_sha256"]


def test_all_conversations_are_content_addressed_and_whole():
    for split in ("heldout", "development"):
        for conversation in load_split(split, purpose="evaluation"):
            assert len(conversation["turns"]) >= 4
            assert conversation["conversation_id"].endswith(conversation["conversation_sha256"])
            assert len(conversation["conversation_sha256"]) == 64
            assert conversation["lineage_id"]


@pytest.mark.parametrize("purpose", ["training", "teacher_replay", "promotion"])
def test_heldout_fails_closed_for_learning_purposes(purpose):
    with pytest.raises(CorpusIntegrityError, match="forbidden"):
        load_split("heldout", purpose=purpose)


def test_sources_have_typed_licensed_provenance():
    for path in sorted(SOURCE_DIR.glob("*.json")):
        document = json.loads(path.read_text())
        _validate_source_document(path, document)
        for source in document["sources"]:
            assert source["source_url"].startswith("https://")
            assert source["source_download_url"].startswith("https://")
            assert source["authoritative_source_url"].startswith("https://")
            assert source["provenance_evidence_url"].startswith("https://")
            assert len(source["raw_source_sha256"]) == 64
            if source["domain"].startswith("public_domain_"):
                assert source["supervision_level"] == "weak"
            if source["domain"] in {"synthetic_adversarial", "open_development"}:
                assert source["structural_only"] is True


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("source_schema_version", True),
        ("is_real_human", "false"),
        ("is_public_domain", 1),
        ("training_eligible", "false"),
        ("teacher_replay_eligible", 0),
        ("publication_year", True),
    ],
)
def test_source_schema_rejects_coercible_types(field, bad_value):
    path = SOURCE_DIR / "development_v1.json"
    document = json.loads(path.read_text())
    if field == "source_schema_version":
        document[field] = bad_value
    else:
        document["sources"][0][field] = bad_value
    with pytest.raises(CorpusIntegrityError):
        _validate_source_document(path, document)


def test_source_schema_rejects_container_and_participant_corruption():
    path = SOURCE_DIR / "development_v1.json"
    original = json.loads(path.read_text())
    corruptions = []
    item = copy.deepcopy(original)
    item["sources"] = {}
    corruptions.append(item)
    item = copy.deepcopy(original)
    item["conversations"][0]["turns"] = "not-an-array"
    corruptions.append(item)
    item = copy.deepcopy(original)
    item["conversations"][0]["participants"].append(item["conversations"][0]["participants"][0])
    corruptions.append(item)
    for document in corruptions:
        with pytest.raises(CorpusIntegrityError):
            _validate_source_document(path, document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update({"unexpected": True}),
        lambda item: item["sources"][0].update({"unexpected": True}),
        lambda item: item["sources"][0].update({"retrieval_date": "2026-02-30"}),
        lambda item: item["sources"][0].update({"is_real_human": True}),
        lambda item: item["sources"][0].update({"license_name": "Proprietary"}),
        lambda item: item["sources"][0].update({"rights_index_url": 7}),
        lambda item: item["sources"][0].update({"rights_index_url": "file:///rights"}),
        lambda item: item["conversations"][0].update({"lineage_id": True}),
        lambda item: item["conversations"][0].update({"template_id": 7}),
        lambda item: item["conversations"][0].update(
            {"participants": item["conversations"][0]["participants"][:1]}
        ),
        lambda item: item["conversations"][0]["turns"][0]["annotation_overrides"].update(
            {"ambiguity": "false"}
        ),
        lambda item: item["conversations"][0]["turns"][0]["annotation_overrides"]["semantic"].update(
            {"scored": "false"}
        ),
    ],
)
def test_source_schema_rejects_nested_and_rights_corruption(mutation):
    path = SOURCE_DIR / "development_v1.json"
    document = json.loads(path.read_text())
    mutation(document)
    with pytest.raises(CorpusIntegrityError):
        _validate_source_document(path, document)


def test_public_literary_source_rejects_structural_only_marker():
    path = SOURCE_DIR / "heldout_drama_v1.json"
    document = json.loads(path.read_text())
    document["sources"][0]["structural_only"] = "false"
    with pytest.raises(CorpusIntegrityError, match="structural_only"):
        _validate_source_document(path, document)


@pytest.mark.parametrize(
    "license_name",
    ["Not public domain; proprietary", "public domain status disputed", "CC0 revoked"],
)
def test_rights_gate_rejects_negated_or_disputed_license_names(license_name):
    path = SOURCE_DIR / "development_v1.json"
    document = json.loads(path.read_text())
    document["sources"][0]["license_name"] = license_name
    with pytest.raises(CorpusIntegrityError, match="public-domain grant"):
        _validate_source_document(path, document)


def test_structural_answer_alignment_rejects_cross_conversation_labels():
    path = SOURCE_DIR / "development_v1.json"
    document = json.loads(path.read_text())
    turn = next(
        turn
        for conversation in document["conversations"]
        for turn in conversation["turns"]
        if turn["annotation_overrides"].get("expected_answer", {}).get("scored")
    )
    turn["annotation_overrides"]["expected_answer"]["predicate"] = "unrelated_schedule"
    with pytest.raises(CorpusIntegrityError, match="answer predicate"):
        _validate_source_document(path, document)
    del turn["annotation_overrides"]["expected_answer"]
    with pytest.raises(CorpusIntegrityError, match="lacks expected_answer"):
        _validate_source_document(path, document)


def test_failed_compile_does_not_overwrite_existing_artifacts(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for name in (
        "development_v1.json",
        "heldout_drama_v1.json",
        "heldout_nasa_v1.json",
        "heldout_novel_v1.json",
    ):
        shutil.copy2(SOURCE_DIR / name, source_dir / name)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sentinel = output_dir / "heldout_v1.jsonl"
    sentinel.write_bytes(b"unchanged")
    with pytest.raises(CorpusIntegrityError, match="at least 500"):
        compile_corpora(source_dir=source_dir, output_dir=output_dir)
    assert sentinel.read_bytes() == b"unchanged"
    assert not (output_dir / "manifest_v1.json").exists()


def test_conflicting_duplicate_source_ids_fail_closed(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    original = json.loads((SOURCE_DIR / "development_v1.json").read_text())
    (source_dir / "one.json").write_text(json.dumps(original))
    conflicting = copy.deepcopy(original)
    conflicting["sources"][0]["title"] += " changed"
    for conversation in conflicting["conversations"]:
        conversation["source_conversation_id"] += "-copy"
        conversation["lineage_id"] += "-copy"
    (source_dir / "two.json").write_text(json.dumps(conflicting))
    with pytest.raises(CorpusIntegrityError, match="conflicting duplicate source_id"):
        compile_corpora(source_dir=source_dir, output_dir=tmp_path / "out")


def test_unused_source_document_provenance_changes_constituent_root(tmp_path):
    source_dir = tmp_path / "sources"
    shutil.copytree(SOURCE_DIR, source_dir)
    first = compile_corpora(source_dir=source_dir, output_dir=tmp_path / "first")
    path = source_dir / "development_v1.json"
    document = json.loads(path.read_text())
    document["sources"][0]["rights_note"] += " Second independently reviewed note."
    path.write_text(json.dumps(document, indent=2) + "\n")
    second = compile_corpora(source_dir=source_dir, output_dir=tmp_path / "second")
    assert first["splits"]["heldout"]["sha256"] == second["splits"]["heldout"]["sha256"]
    assert first["corpus_root_sha256"] != second["corpus_root_sha256"]


@pytest.mark.parametrize("field", ["allowed_uses", "training_eligible", "teacher_replay_eligible"])
def test_manifest_policy_tampering_fails_even_when_split_bytes_do_not_change(tmp_path, field):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    manifest_path = data / "manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["heldout"][field] = (
        ["evaluation", "training"] if field == "allowed_uses" else True
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CorpusIntegrityError):
        load_manifest(manifest_path)

    manifest["corpus_root_sha256"] = hashlib.sha256(
        _canonical_json(_manifest_constituents(manifest)).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (data / "ROOT.sha256").write_text(manifest["corpus_root_sha256"] + "\n")
    with pytest.raises(CorpusIntegrityError, match="policy"):
        load_split("heldout", purpose="training", manifest_path=manifest_path)


def test_production_tree_digest_rejects_tracked_and_untracked_byte_changes(tmp_path):
    for directory in ("clanker_lm", "engine"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    shutil.copy2(ROOT / "clanker_engine.py", tmp_path / "clanker_engine.py")
    expected = production_tree_sha256(tmp_path)
    assert_production_tree(expected, repo_root=tmp_path)
    model_path = tmp_path / "clanker_lm/model.py"
    model_path.write_bytes(model_path.read_bytes() + b"\n# dirty tracked production byte\n")
    with pytest.raises(CorpusIntegrityError, match="production module bytes"):
        assert_production_tree(expected, repo_root=tmp_path)
    model_path.write_bytes((ROOT / "clanker_lm/model.py").read_bytes())
    (tmp_path / "clanker_lm/untracked_runtime.py").write_text("VALUE = 1\n")
    with pytest.raises(CorpusIntegrityError, match="production module bytes"):
        assert_production_tree(expected, repo_root=tmp_path)


def test_production_tree_digest_binds_runtime_seed_asset(tmp_path):
    for directory in ("clanker_lm", "engine"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    shutil.copy2(ROOT / "clanker_engine.py", tmp_path / "clanker_engine.py")
    expected = production_tree_sha256(tmp_path)
    seed_path = tmp_path / "clanker_lm/data/language_seed.json"
    seed_path.write_bytes(seed_path.read_bytes() + b"\n")
    with pytest.raises(CorpusIntegrityError, match="production module bytes"):
        assert_production_tree(expected, repo_root=tmp_path)


def test_compiler_and_full_constituent_root_are_reproducible():
    result = verify_corpus()
    assert result["verified"] is True
    assert result["heldout_turns"] == 520
    assert result["exact_split_overlap"] == 0
    assert result["near_duplicate_split_overlap"] == 0
    assert result["production_text_leakage_hits"] == 0


def test_frozen_lookup_cannot_be_mutated_by_observations():
    lookup = LanguageStore()
    observations = LanguageStore()
    try:
        with lookup.connection:
            lookup.connection.execute(
                "INSERT INTO transition_stats(context_key, sample_count, mean_residual_json, "
                "mean_reaction_json, success_mean) VALUES (?, ?, ?, ?, ?)",
                ("key", 2, json.dumps({axis: 10.0 for axis in "vadugwi"}),
                 json.dumps(AffectVector().to_dict()), 0.5),
            )
        before = _correction_store_digest(lookup)
        controller = FrozenLookupTrajectory(observations, lookup)
        adjusted, metadata = controller.adjust_target(AffectVector(), "key")
        assert metadata["applied"] is True
        assert adjusted != AffectVector()
        assert _correction_store_digest(lookup) == before
        assert observations.schema_summary()["transition_stats"] == 0
    finally:
        lookup.close()
        observations.close()


def _answer_result(
    *,
    predicate="lend",
    value="Sol",
    requested_role="recipient",
    status=AnswerStatus.ANSWERED,
    truth=TruthValue.TRUE,
    proposition=True,
):
    question = SimpleNamespace(
        requested_role=requested_role,
        event=EventFrame(predicate),
    )
    contract = SimpleNamespace(
        status=status,
        truth=truth,
        proposition=EventFrame(predicate) if proposition else None,
        values=[SemanticRef.literal(value)] if value is not None else [],
        question=question,
    )
    return SimpleNamespace(contract=contract)


def test_structured_answer_exactness_uses_contract_value_predicate_and_question_role():
    expected = {"scored": True, "values": ["Sol"], "predicate": "lend", "requested_roles": ["recipient"]}
    assert _answer_exact(expected, _answer_result(), expected_status="answered", expected_truth="true")
    assert not _answer_exact(expected, _answer_result(value="Mira"), expected_status="answered", expected_truth="true")
    assert not _answer_exact(expected, _answer_result(predicate="give"), expected_status="answered", expected_truth="true")
    assert not _answer_exact(expected, _answer_result(requested_role="agent"), expected_status="answered", expected_truth="true")
    assert not _answer_exact(expected, _answer_result(), expected_status="unknown", expected_truth="true")


@pytest.mark.parametrize(
    ("status", "truth"),
    [
        (AnswerStatus.UNKNOWN, TruthValue.UNKNOWN),
        (AnswerStatus.CONFLICT, TruthValue.CONFLICT),
        (AnswerStatus.FALSE, TruthValue.FALSE),
        (AnswerStatus.UNSUPPORTED, TruthValue.UNKNOWN),
    ],
)
def test_nonclaim_answer_exactness_uses_typed_question_event(status, truth):
    expected = {
        "scored": True,
        "values": [],
        "predicate": "originate",
        "requested_roles": ["direction"],
    }
    result = _answer_result(
        predicate="originate",
        value=None,
        requested_role="direction",
        status=status,
        truth=truth,
        proposition=False,
    )
    assert _answer_exact(
        expected,
        result,
        expected_status=status.value,
        expected_truth=truth.value,
    )
    result.contract.question.event = EventFrame("unrelated")
    assert not _answer_exact(
        expected,
        result,
        expected_status=status.value,
        expected_truth=truth.value,
    )


def test_semantic_exactness_canonicalizes_atoms_selects_gold_event_and_requires_exact_roles():
    runtime = SimpleNamespace(memory=ConversationMemory())
    conversation = {
        "participant_bindings": {"user": "participant:user", "clanker": "participant:clanker"}
    }
    turn = {"speaker": "user", "addressee": "clanker"}
    event = EventFrame(
        "own",
        arguments={"theme": SemanticRef.entity("compass", "brass compass")},
    )
    question = QuestionFrame(QuestionKind.WHO, event, requested_role="owner")
    result = SimpleNamespace(parse=ParseResult(
        speech_act=SpeechAct.ASK,
        raw_text="",
        events=[EventFrame("introduce"), event, EventFrame("trailing")],
        question=question,
    ))
    expected = {
        "predicate": "own",
        "roles": {"theme": "brass_compass", "requested_role": "owner"},
        "scored": True,
    }
    assert _semantic_parse_exact(
        expected, result, runtime=runtime, conversation=conversation, turn=turn
    )
    event.arguments["location"] = SemanticRef.entity("vault", "vault")
    assert not _semantic_parse_exact(
        expected, result, runtime=runtime, conversation=conversation, turn=turn
    )
    expected["partial_roles"] = True
    assert _semantic_parse_exact(
        expected, result, runtime=runtime, conversation=conversation, turn=turn
    )
    question.requested_role = "patient"
    assert not _semantic_parse_exact(
        expected, result, runtime=runtime, conversation=conversation, turn=turn
    )


def test_entity_exactness_uses_turn_relative_local_ids_and_ambiguity_sets():
    memory = ConversationMemory()
    runtime = SimpleNamespace(memory=memory)
    conversation = {
        "participant_bindings": {"USER": "participant:user", "CLANKER": "participant:clanker"}
    }
    turn = {"speaker": "USER", "addressee": "CLANKER"}
    event = EventFrame(
        "call",
        arguments={
            "agent": SemanticRef.entity("user", "I"),
            "patient": SemanticRef.entity("assistant", "you"),
        },
    )
    result = SimpleNamespace(
        parse=ParseResult(speech_act=SpeechAct.ASSERT, raw_text="", events=[event])
    )
    assert _entity_exact(
        {"roles": {}, "mentions": {"I": "user", "you": "clanker"}},
        result,
        runtime=runtime,
        conversation=conversation,
        turn=turn,
    )
    ambiguous = SimpleNamespace(
        parse=ParseResult(
            speech_act=SpeechAct.CLARIFY,
            raw_text="",
            unresolved=[UnresolvedReference("they", "ambiguous", ["user", "assistant"])],
        )
    )
    assert _entity_exact(
        {"roles": {}, "mentions": {"they": ["user", "clanker"]}},
        ambiguous,
        runtime=runtime,
        conversation=conversation,
        turn=turn,
    )


def test_entity_exactness_handles_theme_role_and_turn_relative_identity_without_whitelist():
    runtime = SimpleNamespace(memory=ConversationMemory())
    conversation = {
        "participant_bindings": {"user": "participant:user", "clanker": "participant:clanker"}
    }
    user_turn = {"speaker": "user", "addressee": "clanker"}
    result = SimpleNamespace(parse=ParseResult(
        speech_act=SpeechAct.ASSERT,
        raw_text="",
        events=[EventFrame("label", arguments={
            "theme": SemanticRef.entity("label_printer", "label printer"),
            "agent": SemanticRef.entity("user", "I"),
        })],
    ))
    assert _entity_exact(
        {"roles": {"theme": "entity:label_printer", "agent": "participant:user"}, "mentions": {}},
        result,
        runtime=runtime,
        conversation=conversation,
        turn=user_turn,
    )
    clanker_turn = {"speaker": "clanker", "addressee": "user"}
    assert not _entity_exact(
        {"roles": {"agent": "participant:user"}, "mentions": {}},
        result,
        runtime=runtime,
        conversation=conversation,
        turn=clanker_turn,
    )


def test_paired_mode_differences_pair_identical_turns():
    base = [
        {"conversation_id": "c1", "turn_id": "t1", "domain": "d", "mae_v": 4.0},
        {"conversation_id": "c2", "turn_id": "t1", "domain": "d", "mae_v": 2.0},
    ]
    stateful = [
        {"conversation_id": "c1", "turn_id": "t1", "domain": "d", "mae_v": 2.0},
        {"conversation_id": "c2", "turn_id": "t1", "domain": "d", "mae_v": 1.0},
    ]
    compared = _paired_mode_differences(
        {"sentence_only": base, "stateful": stateful, "transition_corrected": stateful},
        seed_base="fixed",
    )
    metric = compared["stateful_minus_sentence_only"]["metrics"]["mae_v"]
    assert metric["value"] == -1.5
    assert metric["n_conversations"] == 2


def test_cluster_bootstrap_preserves_turn_weighted_statistic_for_uneven_conversations():
    records = [
        {"conversation_id": "short", "turn_id": "s1", "domain": "d", "score": 0.0},
        {"conversation_id": "long", "turn_id": "l1", "domain": "d", "score": 1.0},
        {"conversation_id": "long", "turn_id": "l2", "domain": "d", "score": 1.0},
        {"conversation_id": "long", "turn_id": "l3", "domain": "d", "score": 1.0},
    ]
    summary = _metric_summary(records, "score", seed_material="uneven")
    assert summary["value"] == 0.75
    low, high = summary["ci95_conversation_cluster_bootstrap"]
    assert low <= summary["value"] <= high
    assert _pooled_cluster_mean([[0.0], [1.0, 1.0, 1.0]]) == 0.75
    assert _pooled_cluster_mean([[0.0], [1.0, 1.0, 1.0]]) != 0.5


def test_metric_summary_is_deterministic_for_fixed_seed_and_records():
    records = []
    for conversation in ("one", "two", "three"):
        for index in range(3):
            records.append({
                "conversation_id": conversation,
                "turn_id": f"t{index}",
                "domain": "d",
                "outcome": "continued",
                "score": float(index % 2),
            })
    assert _metric_summary(records, "score", seed_material="fixed") == _metric_summary(
        records, "score", seed_material="fixed"
    )


def test_classification_preserves_undefined_draws_and_defined_zero_f1():
    no_labels = [
        {
            "conversation_id": "c",
            "turn_id": "t",
            "domain": "d",
            "expected_status": "answered",
            "actual_status": "answered",
        }
    ]
    absent = _classification(no_labels, "conflict", seed_material="absent")
    assert absent["precision"] is None
    assert absent["recall"] is None
    assert absent["f1"] is None
    assert absent["bootstrap_valid_draws"] == {"precision": 0, "recall": 0, "f1": 0}
    assert absent["ci95_conversation_cluster_bootstrap"] == {}

    zero = _classification(
        [
            {"conversation_id": "c1", "turn_id": "t", "domain": "d", "expected_status": "conflict", "actual_status": "answered"},
            {"conversation_id": "c2", "turn_id": "t", "domain": "d", "expected_status": "answered", "actual_status": "conflict"},
        ],
        "conflict",
        seed_material="zero",
    )
    assert zero["precision"] == 0.0
    assert zero["recall"] == 0.0
    assert zero["f1"] == 0.0


def test_development_semantic_report_is_reproducible():
    first = run_evaluation(split="development")
    second = run_evaluation(split="development")
    assert first["semantic_fingerprint"] == second["semantic_fingerprint"]
    assert first["paired_mode_differences"] == second["paired_mode_differences"]
    for mode in ("sentence_only", "stateful", "transition_corrected"):
        assert "mae_v" not in first["modes"][mode]["overall"]["metrics"]
        for section in ("overall", "by_domain", "by_outcome", "by_supervision_level"):
            assert first["modes"][mode][section] == second["modes"][mode][section]


def test_drift_reports_terminal_bias_and_axis_slopes():
    records = []
    for index in range(3):
        item = {"conversation_id": "c", "next_state_distance": float(index + 1)}
        for axis in "vadugwi":
            item[f"residual_{axis}"] = float(index - 1)
            item[f"absolute_residual_{axis}"] = float(index + 1)
        records.append(item)
    drift = _drift(records)
    assert drift["terminal_signed_bias_by_axis"]["v"] == 1.0
    assert drift["absolute_residual_slope_by_axis"]["v"] == 2.0


def test_trajectory_direction_uses_declared_whole_interaction_g0():
    before = {axis: 100 for axis in "vadugwi"}
    after_current = {axis: 160 for axis in "vadugwi"}
    observed_next = {axis: 150 for axis in "vadugwi"}
    predicted = {axis: 170 for axis in "vadugwi"}
    target = {axis: 180 for axis in "vadugwi"}
    metrics = _whole_interaction_trajectory_metrics(before, observed_next, predicted, target)
    assert metrics["direction_v"] == 1.0
    assert (predicted["v"] - after_current["v"]) * (
        observed_next["v"] - after_current["v"]
    ) < 0


def test_resource_growth_subtracts_seed_baseline_and_reports_slope():
    rows = {"atoms": 3, "transition_stats": 0}
    shapes = []
    for turn_number, growth in ((1, 10), (2, 20)):
        shapes.append({
            "conversation_id": "c",
            "turn_number": turn_number,
            "baseline": {
                "memory": {"entities": 2, "events": 0, "relations": 0, "serialized_bytes": 100},
                "store": {"allocated_bytes": 4096, "rows": rows},
            },
            "current": {
                "memory": {"entities": 2, "events": turn_number, "relations": 0, "serialized_bytes": 100 + growth},
                "store": {"allocated_bytes": 4096 + growth, "rows": {"atoms": 3, "transition_stats": turn_number}},
            },
        })
    result = _aggregate_resources(shapes)
    assert result["memory_serialized_bytes_growth_mean"] == 15.0
    assert result["sqlite_row_growth_maxima"]["atoms"] == 0
    assert result["sqlite_row_growth_maxima"]["transition_stats"] == 2
    assert result["growth_slopes_per_turn"]["sqlite_allocated_bytes"] == 10.0


def test_aggregate_artifact_guard_checks_every_turn_and_recursive_payload_keys():
    conversations = load_split("heldout", purpose="evaluation")
    last_text = conversations[-1]["turns"][-1]["text"]
    with pytest.raises(CorpusIntegrityError, match="turn content"):
        _validate_aggregate_artifacts({"nested": {"value": last_text}}, [], conversations)
    with pytest.raises(CorpusIntegrityError, match="payload key"):
        _validate_aggregate_artifacts({"nested": {"raw_text": "redacted"}}, [], conversations)
    with pytest.raises(CorpusIntegrityError, match="payload key"):
        _validate_aggregate_artifacts({"nested": {"body": "invented reply"}}, [], conversations)
    with pytest.raises(CorpusIntegrityError, match="turn content"):
        _validate_aggregate_artifacts({last_text: 1}, [], conversations)
    _validate_aggregate_artifacts({"candidate_count": 3, "turn_id": "heldout-id"}, [], conversations)


def test_provenance_race_guard_rejects_changed_commit_or_hash():
    initial = {
        "evaluation_commit": "a" * 40,
        "evaluator_sha256": "b" * 64,
        "compiler_sha256": "c" * 64,
        "production_code_sha256": "d" * 64,
    }
    changed = dict(initial)
    changed["evaluation_commit"] = "e" * 40
    with pytest.raises(CorpusIntegrityError, match="changed while"):
        _assert_provenance_unchanged(initial, {}, current=changed)


def test_artifact_publish_stages_every_file_before_replacing(tmp_path):
    report_path = tmp_path / "report.json"
    failures_path = tmp_path / "report_failures.jsonl"
    checksum_path = tmp_path / "report.sha256"
    for path in (report_path, failures_path, checksum_path):
        path.write_text("sentinel\n")
    with pytest.raises(RuntimeError, match="simulated provenance race"):
        _publish_artifacts(
            {"aggregate": 1},
            [{"turn_id": "id", "category": "metric"}],
            output_path=report_path,
            failures_path=failures_path,
            provenance_check=lambda: (_ for _ in ()).throw(RuntimeError("simulated provenance race")),
        )
    for path in (report_path, failures_path, checksum_path):
        assert path.read_text() == "sentinel\n"


@pytest.mark.parametrize("fail_at", [2, 3])
def test_artifact_publish_rolls_back_every_public_target_on_replace_failure(
    tmp_path, monkeypatch, fail_at
):
    report_path = tmp_path / "report.json"
    failures_path = tmp_path / "report_failures.jsonl"
    checksum_path = tmp_path / "report.sha256"
    for path in (report_path, failures_path, checksum_path):
        path.write_text("sentinel\n")
    original = conversation_runner._atomic_replace
    calls = 0

    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("simulated replace failure")
        original(source, target)

    monkeypatch.setattr(conversation_runner, "_atomic_replace", fail_once)
    with pytest.raises(OSError, match="simulated replace failure"):
        _publish_artifacts(
            {"aggregate": 2},
            [{"turn_id": "new", "category": "metric"}],
            output_path=report_path,
            failures_path=failures_path,
            provenance_check=lambda: None,
        )
    for path in (report_path, failures_path, checksum_path):
        assert path.read_text() == "sentinel\n"


def test_execution_errors_fail_the_release_runner():
    with pytest.raises(RuntimeError, match="1 execution error"):
        _enforce_zero_execution_errors([
            {
                "conversation_id": "c",
                "turn_id": "t",
                "category": "execution_error",
                "exception": "ValueError",
            }
        ])
    _enforce_zero_execution_errors([{"category": "semantic_parse_exact"}])


def test_baseline_is_aggregate_only_and_exact_post_106():
    report_path = ROOT / "evaluation/conversations/baselines/post_106_heldout_v1.json"
    failure_path = ROOT / "evaluation/conversations/baselines/post_106_heldout_v1_failures.jsonl"
    report = json.loads(report_path.read_text())
    assert report["production_code_commit"] == "9ae77f072f8afda0b1d2b757ab492757cabff0f8"
    assert set(report["modes"]) == {"sentence_only", "stateful", "transition_corrected"}
    assert report["development_correction_bundle"]["lookup_store_unchanged"] is True
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", report["evaluation_commit"], "HEAD"],
        cwd=ROOT,
        check=True,
    )
    assert report["corpus_root_sha256"] == load_manifest()["corpus_root_sha256"]
    expected_hashes = {}
    for line in report_path.with_suffix(".sha256").read_text().splitlines():
        digest, filename = line.split()
        expected_hashes[filename] = digest
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == expected_hashes[report_path.name]
    assert hashlib.sha256(failure_path.read_bytes()).hexdigest() == expected_hashes[failure_path.name]
    conversations = load_split("heldout", purpose="evaluation")
    failures = [json.loads(line) for line in failure_path.read_text().splitlines() if line]
    _validate_aggregate_artifacts(report, failures, conversations)


def test_distribution_artifacts_exclude_evaluation_corpus(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    sdist = next(tmp_path.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdist) as archive:
        sdist_names = archive.getnames()
    assert not any("evaluation/conversations" in name for name in wheel_names)
    assert not any("evaluation/conversations" in name for name in sdist_names)
