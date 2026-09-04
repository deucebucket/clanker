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
    _validate_source_document,
    compile_corpora,
    load_manifest,
    load_split,
    verify_corpus,
)
from evaluation.conversations.runner import (
    FrozenLookupTrajectory,
    _aggregate_resources,
    _answer_exact,
    _correction_store_digest,
    _drift,
    _entity_exact,
    _metric_summary,
    _paired_mode_differences,
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
    document["provenance_review_note"] = "second independently reviewed note"
    path.write_text(json.dumps(document, indent=2) + "\n")
    second = compile_corpora(source_dir=source_dir, output_dir=tmp_path / "second")
    assert first["splits"]["heldout"]["sha256"] == second["splits"]["heldout"]["sha256"]
    assert first["corpus_root_sha256"] != second["corpus_root_sha256"]


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


def _answer_result(*, predicate="lend", value="Sol", requested_role="recipient"):
    question = SimpleNamespace(requested_role=requested_role)
    contract = SimpleNamespace(
        status=AnswerStatus.ANSWERED,
        truth=TruthValue.TRUE,
        proposition=EventFrame(predicate),
        values=[SemanticRef.literal(value)],
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
        {"I": "user", "you": "clanker"},
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
        {"they": ["user", "clanker"]},
        ambiguous,
        runtime=runtime,
        conversation=conversation,
        turn=turn,
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


def test_development_semantic_report_is_reproducible():
    first = run_evaluation(split="development")
    second = run_evaluation(split="development")
    assert first["semantic_fingerprint"] == second["semantic_fingerprint"]
    assert first["paired_mode_differences"] == second["paired_mode_differences"]
    for mode in ("sentence_only", "stateful", "transition_corrected"):
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
    first_text = load_split("heldout", purpose="evaluation")[0]["turns"][0]["text"]
    assert first_text not in report_path.read_text()
    assert first_text not in failure_path.read_text()


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
