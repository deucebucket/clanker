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
import evaluation.conversations.corpus as conversation_corpus
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
    _production_reference_hits,
    _validate_source_document,
    assert_production_tree,
    compile_corpora,
    load_manifest,
    load_split,
    production_tree_sha256,
    selected_manifest_path,
    verify_additive_generations,
    verify_corpus,
    verify_generation_history,
)
from evaluation.conversations.runner import (
    FAILURE_CATEGORIES,
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
    load_published_artifacts,
    run_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_has_frozen_whole_conversation_corpus():
    manifest = load_manifest()
    heldout = manifest["splits"]["heldout"]
    development = manifest["splits"]["development"]
    assert manifest["baseline_code_commit"] == "c8c0bf4ccd5e73b1bd6bbe99762c87c4a549665e"
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
    assert selected_manifest_path().parent.joinpath("ROOT.sha256").read_text().strip() == manifest["corpus_root_sha256"]


def test_shipped_generation_layout_has_no_divergent_flat_fallback():
    assert (DATA_DIR / "CURRENT").is_file()
    for name in ("ROOT.sha256", "manifest_v1.json", "heldout_v1.jsonl", "development_v1.jsonl"):
        assert not (DATA_DIR / name).exists()


def test_direct_historical_generation_path_cannot_bypass_current_selection():
    selected = selected_manifest_path()
    historical = next(
        directory
        for directory in (DATA_DIR / "generations").iterdir()
        if directory != selected.parent
    )
    with pytest.raises(CorpusIntegrityError, match="not selected by CURRENT"):
        load_manifest(historical / "manifest_v1.json")


def test_documented_static_exclusion_boundary_is_not_overclaimed():
    contract = (ROOT / "evaluation/conversations/README.md").read_text()
    assert "cannot govern arbitrary text or paths deliberately supplied" in contract
    assert "owned by issue #110" in contract
    assert "raw-source attestation remains a project claim" in contract


def test_ci_guards_every_preexisting_corpus_generation_path():
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()
    assert 'additive_base="${{ github.event.before }}"' in workflow
    assert 'python -m evaluation.conversations verify-additive --base "$additive_base"' in workflow
    assert "python -m evaluation.conversations verify-history --ref HEAD" in workflow
    assert "fetch-depth: 0" in workflow


def _write_history_ledger(data: Path) -> None:
    payload = conversation_corpus._history_payload_from_disk(data)
    (data / "HISTORY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (data / "HISTORY.json").chmod(0o644)


def test_candidate_ancestry_gate_rejects_branch_local_generation_deletion(tmp_path):
    repo = tmp_path / "repo"
    data = repo / "evaluation/conversations/data"
    generations = data / "generations"
    generations.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    for generation, marker in (("a" * 64, "old"), ("b" * 64, "new")):
        directory = generations / generation
        directory.mkdir()
        for name in sorted(conversation_corpus.GENERATION_FILES):
            (directory / name).write_text(f"{marker}:{name}\n")
            (directory / name).chmod(0o644)
        _write_history_ledger(data)
        (data / "CURRENT").write_text(generation + "\n")
        (data / "CURRENT").chmod(0o644)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", f"select {marker}"], cwd=repo, check=True)

    assert verify_generation_history("HEAD", repo_root=repo, data_dir=data)[
        "verified_generations"
    ] == 2
    shutil.rmtree(generations / ("a" * 64))
    _write_history_ledger(data)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "delete old generation"], cwd=repo, check=True)
    with pytest.raises(CorpusIntegrityError, match="selection ancestry"):
        verify_generation_history("HEAD", repo_root=repo, data_dir=data)

    subprocess.run(["git", "checkout", "HEAD^", "--", str(
        Path("evaluation/conversations/data/generations") / ("a" * 64)
    )], cwd=repo, check=True)
    _write_history_ledger(data)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "restore immutable history"], cwd=repo, check=True)
    assert verify_generation_history("HEAD", repo_root=repo, data_dir=data)[
        "verified_generations"
    ] == 2
    old_root = generations / ("a" * 64) / "ROOT.sha256"
    old_root.chmod(0o755)
    with pytest.raises(CorpusIntegrityError, match="mode"):
        verify_generation_history("HEAD", repo_root=repo, data_dir=data)


def test_shipped_history_covers_every_selected_generation():
    result = verify_generation_history("HEAD")
    assert result["verified_generations"] >= 8
    assert result["current_generation"] == load_manifest()["corpus_root_sha256"]


def test_additive_guard_rejects_new_files_and_symlinks_inside_old_generation(tmp_path):
    repo = tmp_path / "repo"
    data = repo / "evaluation/conversations/data"
    generation = "a" * 64
    generation_dir = data / "generations" / generation
    generation_dir.mkdir(parents=True)
    for name in ("heldout_v1.jsonl", "development_v1.jsonl", "manifest_v1.json", "ROOT.sha256"):
        (generation_dir / name).write_text(f"{name}\n")
    (data / "CURRENT").write_text(generation + "\n")
    _write_history_ledger(data)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=repo, check=True)
    assert verify_additive_generations("HEAD", repo_root=repo, data_dir=data) == {
        "baseline_present": True,
        "verified_generations": 1,
    }

    extra = generation_dir / "EXTRA.raw"
    extra.write_text("not sealed\n")
    with pytest.raises(CorpusIntegrityError, match="inventory"):
        verify_additive_generations("HEAD", repo_root=repo, data_dir=data)
    extra.unlink()
    extra.symlink_to("ROOT.sha256")
    with pytest.raises(CorpusIntegrityError, match="inventory"):
        verify_additive_generations("HEAD", repo_root=repo, data_dir=data)
    extra.unlink()
    root_path = generation_dir / "ROOT.sha256"
    root_path.chmod(0o755)
    with pytest.raises(CorpusIntegrityError, match="mode"):
        verify_additive_generations("HEAD", repo_root=repo, data_dir=data)
    root_path.chmod(0o644)
    (data / "CURRENT").chmod(0o755)
    with pytest.raises(CorpusIntegrityError, match="CURRENT pointer was modified"):
        verify_additive_generations("HEAD", repo_root=repo, data_dir=data)


def test_additive_guard_rejects_baseline_generations_without_pointer(tmp_path):
    repo = tmp_path / "repo"
    data = repo / "evaluation/conversations/data"
    generation_dir = data / "generations" / ("a" * 64)
    generation_dir.mkdir(parents=True)
    for name in ("heldout_v1.jsonl", "development_v1.jsonl", "manifest_v1.json", "ROOT.sha256"):
        (generation_dir / name).write_text(f"{name}\n")
    _write_history_ledger(data)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "generation without pointer"], cwd=repo, check=True)
    (data / "CURRENT").write_text("a" * 64 + "\n")
    with pytest.raises(CorpusIntegrityError, match="without a CURRENT pointer"):
        verify_additive_generations("HEAD", repo_root=repo, data_dir=data)


def test_additive_guard_allows_tooling_only_pointer_promotion(tmp_path):
    repo = tmp_path / "repo"
    data = repo / "evaluation/conversations/data"
    old_generation = "a" * 64
    old_dir = data / "generations" / old_generation
    old_dir.mkdir(parents=True)
    immutable = {
        "corpus_version": "conversation-v1",
        "content_address": "sha256:" + "1" * 64,
        "baseline_code_commit": "2" * 40,
        "production_code_sha256": "3" * 64,
        "production_code_files": [],
        "splits": {"heldout": {"sha256": "1" * 64}, "development": {"sha256": "4" * 64}},
        "sources": [],
        "compiler": "compiler",
        "immutability_policy": "frozen",
    }
    old_manifest = {
        **immutable,
        "compiler_sha256": "5" * 64,
        "evaluator_sha256": "6" * 64,
        "schema_sha256": {"schema": "7" * 64},
        "corpus_root_sha256": old_generation,
    }
    for name in ("heldout_v1.jsonl", "development_v1.jsonl"):
        (old_dir / name).write_text("{}\n")
    (old_dir / "manifest_v1.json").write_text(json.dumps(old_manifest))
    (old_dir / "ROOT.sha256").write_text(old_generation + "\n")
    (data / "CURRENT").write_text(old_generation + "\n")
    _write_history_ledger(data)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "first generation"], cwd=repo, check=True)

    new_generation = "b" * 64
    new_dir = data / "generations" / new_generation
    shutil.copytree(old_dir, new_dir)
    promoted = copy.deepcopy(old_manifest)
    promoted.update({
        "compiler_sha256": "8" * 64,
        "evaluator_sha256": "9" * 64,
        "schema_sha256": {"schema": "0" * 64},
        "corpus_root_sha256": new_generation,
    })
    (new_dir / "manifest_v1.json").write_text(json.dumps(promoted))
    (new_dir / "ROOT.sha256").write_text(new_generation + "\n")
    (data / "CURRENT").write_text(new_generation + "\n")
    _write_history_ledger(data)
    assert verify_additive_generations("HEAD", repo_root=repo, data_dir=data) == {
        "baseline_present": True,
        "verified_generations": 1,
    }

    promoted["splits"]["heldout"]["sha256"] = "f" * 64
    (new_dir / "manifest_v1.json").write_text(json.dumps(promoted))
    with pytest.raises(CorpusIntegrityError, match="history generation bytes|changed frozen"):
        verify_additive_generations("HEAD", repo_root=repo, data_dir=data)


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


def test_corpus_pointer_failure_keeps_prior_generation_selected(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    shutil.copytree(SOURCE_DIR, source_dir)
    output_dir = tmp_path / "data"
    first = compile_corpora(source_dir=source_dir, output_dir=output_dir)
    old_manifest_path = selected_manifest_path(output_dir / "manifest_v1.json")
    old_text = load_split(
        "development", purpose="development", manifest_path=output_dir / "manifest_v1.json"
    )[0]["turns"][0]["text"]
    original = conversation_corpus._atomic_replace_path

    document_path = source_dir / "development_v1.json"
    document = json.loads(document_path.read_text())
    document["conversations"][0]["turns"][0]["text"] += " Atomic publication probe."
    document_path.write_text(json.dumps(document, indent=2) + "\n")

    def fail_before_select(source, target):
        if target.name == "CURRENT":
            raise OSError("simulated pointer failure")
        original(source, target)

    monkeypatch.setattr(conversation_corpus, "_atomic_replace_path", fail_before_select)
    with pytest.raises(OSError, match="simulated pointer failure"):
        compile_corpora(source_dir=source_dir, output_dir=output_dir)
    selected = load_manifest(output_dir / "manifest_v1.json")
    assert selected["corpus_root_sha256"] == first["corpus_root_sha256"]
    assert selected_manifest_path(output_dir / "manifest_v1.json") == old_manifest_path
    assert load_split(
        "development", purpose="development", manifest_path=output_dir / "manifest_v1.json"
    )[0]["turns"][0]["text"] == old_text


def test_corpus_process_death_before_pointer_keeps_prior_generation(tmp_path):
    source_dir = tmp_path / "sources"
    shutil.copytree(SOURCE_DIR, source_dir)
    output_dir = tmp_path / "data"
    first = compile_corpora(source_dir=source_dir, output_dir=output_dir)
    old_text = load_split(
        "development", purpose="development", manifest_path=output_dir / "manifest_v1.json"
    )[0]["turns"][0]["text"]

    document_path = source_dir / "development_v1.json"
    document = json.loads(document_path.read_text())
    document["conversations"][0]["turns"][0]["text"] += " Process-death publication probe."
    document_path.write_text(json.dumps(document, indent=2) + "\n")
    code = """
import os
from pathlib import Path
import evaluation.conversations.corpus as corpus
original = corpus._atomic_replace_path
def kill_before_select(source, target):
    if target.name == 'CURRENT':
        os._exit(73)
    original(source, target)
corpus._atomic_replace_path = kill_before_select
corpus.compile_corpora(source_dir=Path(__import__('sys').argv[1]), output_dir=Path(__import__('sys').argv[2]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(source_dir), str(output_dir)], cwd=ROOT
    )
    assert result.returncode == 73
    selected = load_manifest(output_dir / "manifest_v1.json")
    assert selected["corpus_root_sha256"] == first["corpus_root_sha256"]
    assert load_split(
        "development", purpose="development", manifest_path=output_dir / "manifest_v1.json"
    )[0]["turns"][0]["text"] == old_text


def test_corpus_fsync_failure_before_selection_keeps_prior_generation(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    shutil.copytree(SOURCE_DIR, source_dir)
    output_dir = tmp_path / "data"
    first = compile_corpora(source_dir=source_dir, output_dir=output_dir)
    document_path = source_dir / "development_v1.json"
    document = json.loads(document_path.read_text())
    document["conversations"][0]["turns"][0]["text"] += " Durability probe."
    document_path.write_text(json.dumps(document, indent=2) + "\n")
    original = conversation_corpus._fsync_directory
    output_syncs = 0

    def fail_after_history(path):
        nonlocal output_syncs
        if path == output_dir:
            output_syncs += 1
            if output_syncs == 2:
                raise OSError("simulated directory fsync failure")
        original(path)

    monkeypatch.setattr(conversation_corpus, "_fsync_directory", fail_after_history)
    with pytest.raises(OSError, match="fsync"):
        compile_corpora(source_dir=source_dir, output_dir=output_dir)
    assert load_manifest(output_dir / "manifest_v1.json")["corpus_root_sha256"] == first[
        "corpus_root_sha256"
    ]


def test_corpus_pointer_generation_name_must_match_constituent_root(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    selected = selected_manifest_path(data / "manifest_v1.json")
    counterfeit = data / "generations" / ("f" * 64)
    shutil.copytree(selected.parent, counterfeit)
    (data / "CURRENT").write_text("f" * 64 + "\n")
    with pytest.raises(CorpusIntegrityError, match="immutable history"):
        load_manifest(data / "manifest_v1.json")


@pytest.mark.parametrize("pointer_value", [None, "not-a-sha256\n"])
def test_corpus_generation_layout_requires_valid_current_pointer(tmp_path, pointer_value):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    pointer = data / "CURRENT"
    if pointer_value is None:
        pointer.unlink()
        match = "required"
    else:
        pointer.write_text(pointer_value)
        match = "malformed"
    with pytest.raises(CorpusIntegrityError, match=match):
        load_manifest(data / "manifest_v1.json")


def test_corpus_generation_rejects_extra_and_symlink_members(tmp_path):
    data = tmp_path / "extra-data"
    shutil.copytree(DATA_DIR, data)
    selected = selected_manifest_path(data / "manifest_v1.json").parent
    (selected / "EXTRA.raw").write_text("not sealed\n")
    with pytest.raises(CorpusIntegrityError, match="inventory"):
        load_manifest(data / "manifest_v1.json")

    symlink_data = tmp_path / "symlink-data"
    shutil.copytree(DATA_DIR, symlink_data)
    selected = selected_manifest_path(symlink_data / "manifest_v1.json").parent
    root_path = selected / "ROOT.sha256"
    root_path.unlink()
    root_path.symlink_to("manifest_v1.json")
    with pytest.raises(CorpusIntegrityError, match="inventory"):
        load_manifest(symlink_data / "manifest_v1.json")


def test_corpus_generation_parent_symlink_fails_for_load_and_compile(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    external = tmp_path / "external-generations"
    (data / "generations").replace(external)
    (data / "generations").symlink_to(external, target_is_directory=True)
    with pytest.raises(CorpusIntegrityError, match="generations parent"):
        load_manifest(data / "manifest_v1.json")

    compile_output = tmp_path / "compile-data"
    compile_output.mkdir()
    compile_external = tmp_path / "compile-external"
    compile_external.mkdir()
    (compile_output / "generations").symlink_to(compile_external, target_is_directory=True)
    with pytest.raises(CorpusIntegrityError, match="generations parent"):
        compile_corpora(source_dir=SOURCE_DIR, output_dir=compile_output)
    assert not any(compile_external.iterdir())


def test_corpus_data_directory_symlink_fails_before_selection_or_compile(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    external = tmp_path / "external-data"
    data.replace(external)
    data.symlink_to(external, target_is_directory=True)
    with pytest.raises(CorpusIntegrityError, match="traverse a symlink"):
        load_manifest(data / "manifest_v1.json")
    with pytest.raises(CorpusIntegrityError, match="traverse a symlink"):
        compile_corpora(source_dir=SOURCE_DIR, output_dir=data)


def test_split_loader_hashes_and_parses_one_immutable_buffer(tmp_path, monkeypatch):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    manifest_path = data / "manifest_v1.json"
    split_path = selected_manifest_path(manifest_path).parent / "development_v1.jsonl"
    original_read_bytes = Path.read_bytes
    swapped = False

    def swap_after_read(path):
        nonlocal swapped
        payload = original_read_bytes(path)
        if path == split_path and not swapped:
            swapped = True
            with path.open("wb") as stream:
                stream.write(b"{}\n")
        return payload

    monkeypatch.setattr(Path, "read_bytes", swap_after_read)
    conversations = load_split("development", purpose="development", manifest_path=manifest_path)
    assert swapped is True
    assert len(conversations) == 10


@pytest.mark.parametrize("field", ["allowed_uses", "training_eligible", "teacher_replay_eligible"])
def test_manifest_policy_tampering_fails_even_when_split_bytes_do_not_change(tmp_path, field):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    manifest_path = selected_manifest_path(data / "manifest_v1.json")
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
    manifest_path.with_name("ROOT.sha256").write_text(manifest["corpus_root_sha256"] + "\n")
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


def test_production_tree_digest_binds_packaged_web_runtime_assets(tmp_path):
    for directory in ("clanker_lm", "engine"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    shutil.copy2(ROOT / "clanker_engine.py", tmp_path / "clanker_engine.py")
    expected = production_tree_sha256(tmp_path)
    asset_path = tmp_path / "clanker_lm/web_assets/app.js"
    asset_path.write_bytes(asset_path.read_bytes() + b"\n")
    with pytest.raises(CorpusIntegrityError, match="production module bytes"):
        assert_production_tree(expected, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("relative_path", "payload"),
    [
        ("clanker_engine.py", "load_split('heldout', purpose='evaluation')\n"),
        ("engine/probe.py", "from evaluation.conversations.corpus import load_split\n"),
        ("clanker_lm/probe.py", "Path('evaluation/conversations/data') / 'CURRENT'\n"),
        (
            "clanker_lm/dynamic_probe.py",
            "import importlib\nSPLIT = 'held' + 'out'\n"
            "importlib.import_module('evaluation' + '.conversations.corpus').load_split("
            "split=SPLIT, purpose='evaluation')\n",
        ),
        (
            "engine/pointer_probe.py",
            "from pathlib import Path\nPOINTER = 'CUR' + 'RENT'\n"
            "ROOT = Path('evaluation') / f\"{'conversations'}\" / 'data' / POINTER\n",
        ),
        (
            "clanker_lm/join_probe.py",
            "import importlib\nPARTS = ('evaluation', 'conversations', 'corpus')\n"
            "NAME = '.'.join(PARTS)\nimportlib.import_module(NAME)\n",
        ),
        (
            "engine/format_probe.py",
            "from pathlib import Path\nBASE = '{}/{}'.format('evaluation', 'conversations')\n"
            "NAME = ('cur' + 'rent').upper()\nROOT = Path(BASE) / 'data' / NAME\n",
        ),
        (
            "clanker_lm/getattr_probe.py",
            "import importlib\nNAME = ('IMPORT', '_MODULE')\n"
            "LOADER = getattr(importlib, ''.join(NAME).lower())\nLOADER('unrelated')\n",
        ),
        (
            "clanker_lm/web_assets/reference_probe.js",
            "const root = ['evaluation','conversations','data'].join('/');\n"
            "const pointer = ['CUR','RENT'].join('');\n",
        ),
        (
            "clanker_lm/web_assets/fragment_probe.js",
            "const decoy = 'harmless';\n"
            "const root = ['eval','uation','/','con','ver','sations','/data'].join('');\n"
            "const pointer = ['C','U','R','R','E','N','T'].join('');\n",
        ),
        ("clanker_lm/web_assets/current_probe.js", "const pointer = 'CURRENT';\n"),
        (
            "clanker_lm/web_assets/current_join_probe.js",
            "const pointer = ['CUR','RENT'].join('');\n",
        ),
    ],
)
def test_production_reference_scan_covers_every_declared_path(tmp_path, relative_path, payload):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    manifest = {"content_address": "sha256:" + "a" * 64}
    assert _production_reference_hits(manifest, repo_root=tmp_path) == [relative_path]


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


def test_question_role_cannot_be_borrowed_by_another_same_predicate_event():
    runtime = SimpleNamespace(memory=ConversationMemory())
    conversation = {
        "participant_bindings": {"user": "participant:user", "clanker": "participant:clanker"}
    }
    turn = {"speaker": "user", "addressee": "clanker"}
    assertion = EventFrame(
        "own", arguments={"theme": SemanticRef.entity("compass", "brass compass")}
    )
    queried = EventFrame(
        "own",
        arguments={
            "theme": SemanticRef.entity("map", "paper map"),
            "owner": SemanticRef.variable("owner"),
        },
    )
    result = SimpleNamespace(parse=ParseResult(
        speech_act=SpeechAct.ASK,
        raw_text="",
        events=[assertion, queried],
        question=QuestionFrame(QuestionKind.WHO, queried, requested_role="owner"),
    ))
    contaminated = {
        "predicate": "own",
        "roles": {"theme": "brass_compass", "requested_role": "owner"},
        "scored": True,
    }
    assert not _semantic_parse_exact(
        contaminated, result, runtime=runtime, conversation=conversation, turn=turn
    )
    correct = copy.deepcopy(contaminated)
    correct["roles"]["theme"] = "paper_map"
    assert _semantic_parse_exact(
        correct, result, runtime=runtime, conversation=conversation, turn=turn
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
    addressed_human = SimpleNamespace(parse=ParseResult(
        speech_act=SpeechAct.ASSERT,
        raw_text="",
        events=[EventFrame("tell", arguments={
            "patient": SemanticRef.entity("assistant", "you"),
        })],
    ))
    assert _entity_exact(
        {"roles": {"patient": "participant:user"}, "mentions": {}},
        addressed_human,
        runtime=runtime,
        conversation=conversation,
        turn=clanker_turn,
    )
    assert not _entity_exact(
        {"roles": {"patient": "participant:clanker"}, "mentions": {}},
        addressed_human,
        runtime=runtime,
        conversation=conversation,
        turn=clanker_turn,
    )


def test_paired_mode_differences_pair_identical_turns():
    base = [
        {"conversation_id": "c1", "turn_id": "t1", "domain": "d", "mae_v": 4.0, "target_distance_improvement": 1.0},
        {"conversation_id": "c2", "turn_id": "t1", "domain": "d", "mae_v": 2.0, "target_distance_improvement": 2.0},
    ]
    stateful = [
        {"conversation_id": "c1", "turn_id": "t1", "domain": "d", "mae_v": 2.0, "target_distance_improvement": 3.0},
        {"conversation_id": "c2", "turn_id": "t1", "domain": "d", "mae_v": 1.0, "target_distance_improvement": 4.0},
    ]
    compared = _paired_mode_differences(
        {"sentence_only": base, "stateful": stateful, "transition_corrected": stateful},
        seed_base="fixed",
    )
    metric = compared["stateful_minus_sentence_only"]["metrics"]["mae_v"]
    assert metric["value"] == -1.5
    assert metric["n_conversations"] == 2
    assert compared["stateful_minus_sentence_only"]["metrics"][
        "target_distance_improvement"
    ]["value"] == 2.0


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
    for expected_status, actual_status in (
        ("conflict", "answered"),
        ("answered", "conflict"),
    ):
        one_sided = _classification(
            [{
                "conversation_id": "c",
                "turn_id": "t",
                "domain": "d",
                "expected_status": expected_status,
                "actual_status": actual_status,
            }],
            "conflict",
            seed_material=f"{expected_status}|{actual_status}",
        )
        assert one_sided["f1"] == 0.0
        assert one_sided["bootstrap_valid_draws"]["f1"] == 10_000
        assert one_sided["ci95_conversation_cluster_bootstrap"]["f1"] == [0.0, 0.0]


@pytest.fixture(scope="module")
def development_reports():
    first = run_evaluation(split="development")
    second = run_evaluation(split="development")
    return first, second


def _failure_rows_from_report(report):
    failures = []
    for mode, mode_report in report["modes"].items():
        for category in FAILURE_CATEGORIES:
            for cluster in mode_report["overall"]["metrics"][category]["clusters"]:
                for observation in cluster["observations"]:
                    if observation["value"] == 0.0:
                        failures.append({
                            "conversation_id": cluster["conversation_id"],
                            "turn_id": observation["turn_id"],
                            "domain": cluster["domain"],
                            "mode": mode,
                            "category": category,
                        })
    return failures


def test_development_semantic_report_is_reproducible(development_reports):
    first, second = development_reports
    assert first["semantic_fingerprint"] == second["semantic_fingerprint"]
    assert first["paired_mode_differences"] == second["paired_mode_differences"]
    for mode in ("sentence_only", "stateful", "transition_corrected"):
        assert "mae_v" not in first["modes"][mode]["overall"]["metrics"]
        for section in ("overall", "by_domain", "by_outcome", "by_supervision_level"):
            assert first["modes"][mode][section] == second["modes"][mode][section]


def test_aggregate_report_schema_rejects_generated_payload_aliases_and_bad_types(
    development_reports,
):
    report = development_reports[0]
    conversations = load_split("development", purpose="development")
    identities = [
        (conversation["conversation_id"], turn["turn_id"], conversation["domain"])
        for conversation in conversations
        for turn in conversation["turns"]
    ]
    failures = _failure_rows_from_report(report)
    assert len(failures) == report["failure_count"]
    _validate_aggregate_artifacts(report, failures, conversations)

    mutations = []
    item = copy.deepcopy(report)
    item["development_correction_bundle"]["answer"] = "invented generated reply"
    mutations.append(item)
    item = copy.deepcopy(report)
    item["paired_mode_differences"]["stateful_minus_sentence_only"]["result"] = {
        "content": "invented generated reply"
    }
    mutations.append(item)
    item = copy.deepcopy(report)
    first_metric = next(iter(item["modes"]["sentence_only"]["overall"]["metrics"].values()))
    first_metric["value"] = {"answer": "invented generated reply"}
    mutations.append(item)
    item = copy.deepcopy(report)
    item["failure_count"] = str(item["failure_count"])
    mutations.append(item)
    item = copy.deepcopy(report)
    item["limitations"][0] = "type-correct but undeclared generated response"
    mutations.append(item)
    item = copy.deepcopy(report)
    item["paired_mode_differences"]["stateful_minus_sentence_only"]["pairing"] = (
        "type-correct but undeclared generated response"
    )
    mutations.append(item)
    item = copy.deepcopy(report)
    item["environment"]["processor"] = "type-correct generated response"
    mutations.append(item)
    item = copy.deepcopy(report)
    item["environment"]["machine"] = "invented_generated_reply"
    mutations.append(item)
    item = copy.deepcopy(report)
    item["modes"]["sentence_only"]["resource_growth"]["sqlite_row_growth_maxima"][
        "invented_generated_reply"
    ] = 1
    mutations.append(item)
    item = copy.deepcopy(report)
    item["failure_counts"] = {"bogus": item["failure_count"]}
    mutations.append(item)
    item = copy.deepcopy(report)
    item["modes"]["sentence_only"]["execution_errors"] = 7
    mutations.append(item)
    item = copy.deepcopy(report)
    item["modes"]["sentence_only"]["overall"]["metrics"] = {}
    mutations.append(item)
    item = copy.deepcopy(report)
    del item["modes"]["sentence_only"]["overall"]["metrics"]["semantic_parse_exact"]
    mutations.append(item)
    item = copy.deepcopy(report)
    item["modes"]["sentence_only"]["overall"]["metrics"]["dialogue_act_correct"][
        "n_turns"
    ] -= 1
    mutations.append(item)
    item = copy.deepcopy(report)
    item["modes"]["sentence_only"]["by_domain"] = {}
    mutations.append(item)
    item = copy.deepcopy(report)
    item["modes"]["sentence_only"]["by_outcome"] = {}
    mutations.append(item)
    item = copy.deepcopy(report)
    item["modes"]["sentence_only"]["by_supervision_level"] = {}
    mutations.append(item)
    item = copy.deepcopy(report)
    item["paired_mode_differences"]["stateful_minus_sentence_only"]["metrics"] = {}
    mutations.append(item)

    negative_paths = (
        ("summary turn count", lambda item: item["modes"]["sentence_only"]["overall"].__setitem__("turn_count", -1)),
        ("metric turn count", lambda item: next(iter(item["modes"]["sentence_only"]["overall"]["metrics"].values())).__setitem__("n_turns", -1)),
        ("classification count", lambda item: item["modes"]["sentence_only"]["overall"]["unknown_classification"].__setitem__("tp", -1)),
        ("outcome count", lambda item: item["modes"]["sentence_only"]["overall"]["outcome_counts"].__setitem__(next(iter(item["modes"]["sentence_only"]["overall"]["outcome_counts"])), -1)),
        ("latency count", lambda item: item["modes"]["sentence_only"]["latency"].__setitem__("n", -1)),
        ("paired count", lambda item: item["paired_mode_differences"]["stateful_minus_sentence_only"].__setitem__("paired_turn_count", -1)),
        ("correction count", lambda item: item["development_correction_bundle"].__setitem__("source_conversation_count", -1)),
    )
    for _label, mutate in negative_paths:
        item = copy.deepcopy(report)
        mutate(item)
        mutations.append(item)
    item = copy.deepcopy(report)
    item["development_correction_bundle"]["sample_count"] += 1
    mutations.append(item)
    item = copy.deepcopy(report)
    metric = item["modes"]["sentence_only"]["overall"]["metrics"]["dialogue_act_correct"]
    metric.update({"value": 99, "n_turns": 1, "n_conversations": 1, "ci95_conversation_cluster_bootstrap": [1, 0]})
    mutations.append(item)
    item = copy.deepcopy(report)
    classification = item["modes"]["sentence_only"]["overall"]["unknown_classification"]
    classification["precision"] = 1.0 if classification["precision"] != 1.0 else 0.0
    mutations.append(item)
    item = copy.deepcopy(report)
    classification = item["modes"]["sentence_only"]["overall"]["unknown_classification"]
    classification["tp"] += 1
    mutations.append(item)
    item = copy.deepcopy(report)
    classification = item["modes"]["sentence_only"]["overall"]["unknown_classification"]
    classification["bootstrap_valid_draws"]["f1"] -= 1
    mutations.append(item)
    item = copy.deepcopy(report)
    classification = item["modes"]["sentence_only"]["overall"]["unknown_classification"]
    if "f1" in classification["ci95_conversation_cluster_bootstrap"]:
        classification["ci95_conversation_cluster_bootstrap"]["f1"][0] = 0.123456789
    mutations.append(item)
    item = copy.deepcopy(report)
    calibration = item["modes"]["sentence_only"]["overall"]["uncertainty_calibration"]
    calibration.update({"ece_10_bin": 99, "n": 0, "bins": []})
    mutations.append(item)
    item = copy.deepcopy(report)
    calibration = item["modes"]["sentence_only"]["overall"]["uncertainty_calibration"]
    calibration["bins"][0]["confidence_sum"] += 0.25
    mutations.append(item)
    item = copy.deepcopy(report)
    calibration = item["modes"]["sentence_only"]["overall"]["uncertainty_calibration"]
    calibration["bins"][0]["lower"] += 0.05
    mutations.append(item)
    item = copy.deepcopy(report)
    del item["modes"]["sentence_only"]["overall"]["metrics"]["brier"]
    mutations.append(item)
    for field, value in (
        ("source_conversation_count", report["development_correction_bundle"]["source_conversation_count"] + 1),
        ("source_corpus_sha256", "0" * 64),
        ("production_runtime_commit", "0" * 40),
    ):
        item = copy.deepcopy(report)
        item["development_correction_bundle"][field] = value
        mutations.append(item)
    item = copy.deepcopy(report)
    item["semantic_fingerprint"] = "0" * 64
    mutations.append(item)

    for corrupt in mutations:
        with pytest.raises(CorpusIntegrityError):
            _validate_aggregate_artifacts(corrupt, failures, conversations)

    corrupt_failures = copy.deepcopy(failures)
    corrupt_failures[0]["exception"] = "InventedGeneratedReply"
    with pytest.raises(CorpusIntegrityError, match="unexpected field"):
        _validate_aggregate_artifacts(report, corrupt_failures, conversations)
    duplicate_report = copy.deepcopy(report)
    duplicate_report["failure_count"] += 1
    duplicate_report["failure_counts"][failures[0]["category"]] += 1
    with pytest.raises(CorpusIntegrityError, match="duplicate"):
        _validate_aggregate_artifacts(
            duplicate_report, failures + [copy.deepcopy(failures[0])], conversations
        )


def test_report_sufficient_statistics_reject_self_consistent_forgery(
    development_reports,
):
    report = development_reports[0]
    conversations = load_split("development", purpose="development")
    failures = _failure_rows_from_report(report)

    def refresh_fingerprint(item):
        semantic_payload = {
            "modes": {
                mode: {
                    "overall": item["modes"][mode]["overall"],
                    "by_domain": item["modes"][mode]["by_domain"],
                    "by_outcome": item["modes"][mode]["by_outcome"],
                    "by_supervision_level": item["modes"][mode]["by_supervision_level"],
                    "execution_errors": item["modes"][mode]["execution_errors"],
                }
                for mode in conversation_runner.MODES
            },
            "paired_mode_differences": item["paired_mode_differences"],
        }
        item["semantic_fingerprint"] = hashlib.sha256(
            _canonical_json(semantic_payload).encode("utf-8")
        ).hexdigest()

    forged = copy.deepcopy(report)
    metric = forged["modes"]["sentence_only"]["overall"]["metrics"][
        "dialogue_act_correct"
    ]
    metric["clusters"][0]["observations"][0]["value"] = 0.5
    metric["value"] = sum(
        observation["value"]
        for cluster in metric["clusters"]
        for observation in cluster["observations"]
    ) / metric["n_turns"]
    refresh_fingerprint(forged)
    with pytest.raises(CorpusIntegrityError, match="binary observations"):
        _validate_aggregate_artifacts(forged, failures, conversations)

    forged = copy.deepcopy(report)
    metric = forged["modes"]["sentence_only"]["overall"]["metrics"]["truth_correct"]
    metric["ci95_conversation_cluster_bootstrap"] = [0.123, 0.987]
    refresh_fingerprint(forged)
    with pytest.raises(CorpusIntegrityError, match="deterministic observations"):
        _validate_aggregate_artifacts(forged, failures, conversations)

    forged = copy.deepcopy(report)
    comparison = forged["paired_mode_differences"]["stateful_minus_sentence_only"]
    metric = comparison["metrics"]["truth_correct"]
    metric["clusters"][0]["observations"][0]["value"] = 1.0
    metric["value"] = sum(
        observation["value"]
        for cluster in metric["clusters"]
        for observation in cluster["observations"]
    ) / metric["n_turns"]
    refresh_fingerprint(forged)
    with pytest.raises(CorpusIntegrityError):
        _validate_aggregate_artifacts(forged, failures, conversations)

    forged = copy.deepcopy(report)
    summary = forged["modes"]["sentence_only"]["overall"]
    unknown = summary["unknown_classification"]
    records = [
        {
            "domain": cluster["domain"],
            "conversation_id": cluster["conversation_id"],
            **observation,
        }
        for cluster in unknown["clusters"]
        for observation in cluster["observations"]
    ]
    records[0]["actual_status"] = "unknown"
    summary["unknown_classification"] = _classification(
        records,
        "unknown",
        seed_material=(
            f"{report['corpus_sha256']}|metric-v1|sentence_only|overall|unknown"
        ),
    )
    refresh_fingerprint(forged)
    with pytest.raises(CorpusIntegrityError, match="predictions disagree|answer-status"):
        _validate_aggregate_artifacts(forged, failures, conversations)

    forged = copy.deepcopy(report)
    calibration = forged["modes"]["sentence_only"]["overall"]["uncertainty_calibration"]
    calibration["bins"][0]["correct_sum"] += 0.5
    calibration["bins"][0]["accuracy"] = (
        calibration["bins"][0]["correct_sum"] / calibration["bins"][0]["n"]
    )
    refresh_fingerprint(forged)
    with pytest.raises(CorpusIntegrityError, match="integral"):
        _validate_aggregate_artifacts(forged, failures, conversations)


def test_report_provenance_resources_and_failure_membership_fail_closed(
    development_reports,
):
    report = development_reports[0]
    conversations = load_split("development", purpose="development")
    failures = _failure_rows_from_report(report)

    forged = copy.deepcopy(report)
    forged["development_correction_bundle"]["sha256"] = "0" * 64
    forged["development_correction_bundle"]["sha256_after_evaluation"] = "0" * 64
    with pytest.raises(CorpusIntegrityError, match="not reproducible"):
        _validate_aggregate_artifacts(forged, failures, conversations)

    forged = copy.deepcopy(report)
    forged["evaluation_commit"] = "0" * 40
    with pytest.raises(CorpusIntegrityError, match="local ancestry"):
        _validate_aggregate_artifacts(forged, failures, conversations)

    for path, value in (("latency", -1.0), ("maxrss", -1)):
        forged = copy.deepcopy(report)
        if path == "latency":
            forged["modes"]["sentence_only"]["latency"]["mean_ms"] = value
        else:
            forged["modes"]["sentence_only"]["resource_growth"][
                "process_lifetime_maxrss_peak"
            ] = value
        with pytest.raises(CorpusIntegrityError, match="domain"):
            _validate_aggregate_artifacts(forged, failures, conversations)

    if failures:
        forged_failures = copy.deepcopy(failures)
        replacement = next(
            (failure, cluster["conversation_id"], cluster["domain"], observation["turn_id"])
            for failure in forged_failures
            for cluster in report["modes"][failure["mode"]]["overall"]["metrics"]
            [failure["category"]]["clusters"]
            for observation in cluster["observations"]
            if observation["value"] == 1.0
        )
        failure, conversation_id, domain, turn_id = replacement
        failure["conversation_id"] = conversation_id
        failure["domain"] = domain
        failure["turn_id"] = turn_id
        with pytest.raises(CorpusIntegrityError, match="membership|identity"):
            _validate_aggregate_artifacts(report, forged_failures, conversations)


def test_drift_reports_terminal_bias_and_axis_slopes():
    records = []
    for index in range(3):
        item = {
            "conversation_id": "c", "turn_id": f"t{index}", "domain": "open_development",
            "next_state_distance": float(index + 1),
        }
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


def test_weighted_rms_next_state_distance_cannot_exceed_255():
    records = [{
        "domain": "synthetic_adversarial",
        "conversation_id": "synthetic",
        "turn_id": "t01",
        "next_state_distance": 256.0,
    }]
    metric = conversation_runner._metric_summary(
        records, "next_state_distance", seed_material="distance-bound"
    )
    with pytest.raises(CorpusIntegrityError, match="declared metric domain"):
        conversation_runner._validate_metric_summary(
            metric,
            "distance",
            metric_name="next_state_distance",
            seed_material="distance-bound",
        )


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
    with pytest.raises(CorpusIntegrityError, match="discriminator"):
        _validate_aggregate_artifacts({"candidate_count": 3, "turn_id": "heldout-id"}, [], conversations)


def test_aggregate_artifact_guard_rejects_short_embedded_and_fragmented_turns():
    conversations = load_split("heldout", purpose="evaluation")
    raw_turns = [turn["text"] for conversation in conversations for turn in conversation["turns"]]
    shortest = min(raw_turns, key=len)
    with pytest.raises(CorpusIntegrityError, match="turn content"):
        _validate_aggregate_artifacts(
            {"candidate_count": 1, "identifier": f"prefix {shortest} suffix"},
            [],
            conversations,
        )
    compact_turns = [conversation_runner._norm(text).replace(" ", "") for text in raw_turns]
    for length, use_keys in ((13, False), (39, True)):
        target = next(text for text in compact_turns if len(text) == length)
        for cut in range(1, len(target)):
            artifact = (
                {target[:cut]: {target[cut:]: 1}}
                if use_keys
                else {"left_fragment": target[:cut], "right_fragment": target[cut:]}
            )
            with pytest.raises(CorpusIntegrityError, match="fragmented held-out"):
                _validate_aggregate_artifacts(artifact, [], conversations)


@pytest.mark.parametrize("report", [{}, {"arbitrary": "free-form generated response"}])
def test_aggregate_artifact_guard_requires_report_discriminator_and_exact_schema(report):
    conversations = load_split("development", purpose="development")
    with pytest.raises(CorpusIntegrityError, match="discriminator"):
        _validate_aggregate_artifacts(report, [], conversations)


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


def test_artifact_publish_pointer_failure_keeps_prior_generation_selected(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    failures_path = tmp_path / "report_failures.jsonl"
    _publish_artifacts(
        {"aggregate": 1},
        [{"turn_id": "old", "category": "metric"}],
        output_path=report_path,
        failures_path=failures_path,
        provenance_check=lambda: None,
    )
    old_report, old_failures, old_generation = load_published_artifacts(report_path, validate=False)
    original = conversation_runner._atomic_replace

    def fail_once(source, target):
        if target.suffix == ".current":
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
    selected_report, selected_failures, selected_generation = load_published_artifacts(report_path, validate=False)
    assert selected_report == old_report
    assert selected_failures == old_failures
    assert selected_generation == old_generation


def test_artifact_publish_process_death_before_pointer_keeps_prior_generation(tmp_path):
    report_path = tmp_path / "report.json"
    failure_path = tmp_path / "report_failures.jsonl"
    _publish_artifacts(
        {"aggregate": 1},
        [{"turn_id": "old", "category": "metric"}],
        output_path=report_path,
        failures_path=failure_path,
        provenance_check=lambda: None,
    )
    old_report, old_failures, old_generation = load_published_artifacts(report_path, validate=False)
    code = """
import os
from pathlib import Path
import evaluation.conversations.runner as runner
original = runner._atomic_replace
def kill_before_select(source, target):
    if target.suffix == '.current':
        os._exit(73)
    original(source, target)
runner._atomic_replace = kill_before_select
runner._publish_artifacts(
    {'aggregate': 2},
    [{'turn_id': 'new', 'category': 'metric'}],
    output_path=Path(__import__('sys').argv[1]),
    failures_path=Path(__import__('sys').argv[2]),
    provenance_check=lambda: None,
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(report_path), str(failure_path)], cwd=ROOT
    )
    assert result.returncode == 73
    selected_report, selected_failures, selected_generation = load_published_artifacts(report_path, validate=False)
    assert selected_report == old_report
    assert selected_failures == old_failures
    assert selected_generation == old_generation


def test_report_fsync_failure_before_selection_keeps_prior_generation(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    failure_path = tmp_path / "report_failures.jsonl"
    _publish_artifacts(
        {"aggregate": 1}, [{"turn_id": "old", "category": "metric"}],
        output_path=report_path, failures_path=failure_path, provenance_check=lambda: None,
    )
    old_report, old_failures, old_generation = load_published_artifacts(
        report_path, validate=False
    )
    original = conversation_runner._fsync_report_directory
    parent_syncs = 0

    def fail_before_selection(path):
        nonlocal parent_syncs
        if path == tmp_path:
            parent_syncs += 1
            if parent_syncs == 2:
                raise OSError("simulated directory fsync failure")
        original(path)

    monkeypatch.setattr(conversation_runner, "_fsync_report_directory", fail_before_selection)
    with pytest.raises(OSError, match="fsync"):
        _publish_artifacts(
            {"aggregate": 2}, [{"turn_id": "new", "category": "metric"}],
            output_path=report_path, failures_path=failure_path, provenance_check=lambda: None,
        )
    report, failures, generation = load_published_artifacts(report_path, validate=False)
    assert (report, failures, generation) == (old_report, old_failures, old_generation)


def test_artifact_generation_has_no_backup_basename_collision(tmp_path):
    report_path = tmp_path / ".backup-0"
    failure_path = tmp_path / ".backup-1.jsonl"
    expected_report = {"aggregate": 1}
    expected_failures = [{"turn_id": "id", "category": "metric"}]
    _publish_artifacts(
        expected_report,
        expected_failures,
        output_path=report_path,
        failures_path=failure_path,
        provenance_check=lambda: None,
    )
    report, failures, _ = load_published_artifacts(report_path, validate=False)
    assert report == expected_report
    assert failures == expected_failures


def test_report_generation_rejects_extra_and_symlink_members(tmp_path):
    report_path = tmp_path / "report.json"
    failure_path = tmp_path / "report_failures.jsonl"
    report = {"aggregate": 1}
    failures = [{"turn_id": "id", "category": "metric"}]
    _publish_artifacts(
        report,
        failures,
        output_path=report_path,
        failures_path=failure_path,
        provenance_check=lambda: None,
    )
    _, _, generation = load_published_artifacts(report_path, validate=False)
    (generation / "EXTRA.raw").write_text("not sealed\n")
    with pytest.raises(CorpusIntegrityError, match="inventory"):
        load_published_artifacts(report_path, validate=False)
    with pytest.raises(CorpusIntegrityError, match="inventory"):
        _publish_artifacts(
            report,
            failures,
            output_path=report_path,
            failures_path=failure_path,
            provenance_check=lambda: None,
        )

    symlink_report = tmp_path / "symlink.json"
    symlink_failures = tmp_path / "symlink_failures.jsonl"
    _publish_artifacts(
        report,
        failures,
        output_path=symlink_report,
        failures_path=symlink_failures,
        provenance_check=lambda: None,
    )
    _, _, generation = load_published_artifacts(symlink_report, validate=False)
    selected_failure = generation / symlink_failures.name
    selected_failure.unlink()
    selected_failure.symlink_to(symlink_report.name)
    with pytest.raises(CorpusIntegrityError, match="inventory"):
        load_published_artifacts(symlink_report, validate=False)


def test_report_loader_hashes_and_parses_one_immutable_buffer(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _publish_artifacts(
        {"aggregate": 1},
        [{"turn_id": "id", "category": "metric"}],
        output_path=report_path,
        failures_path=None,
        provenance_check=lambda: None,
    )
    _, _, generation = load_published_artifacts(report_path, validate=False)
    selected_report = generation / report_path.name
    original_read_bytes = Path.read_bytes
    swapped = False

    def swap_after_read(path):
        nonlocal swapped
        payload = original_read_bytes(path)
        if path == selected_report and not swapped:
            swapped = True
            with path.open("wb") as stream:
                stream.write(b'{"aggregate": 999}\n')
        return payload

    monkeypatch.setattr(Path, "read_bytes", swap_after_read)
    report, _, _ = load_published_artifacts(report_path, validate=False)
    assert swapped is True
    assert report == {"aggregate": 1}


def test_report_loader_rejects_pointer_escape_and_duplicate_checksum_rows(tmp_path):
    report_path = tmp_path / "report.json"
    _publish_artifacts(
        {"aggregate": 1},
        [{"turn_id": "id", "category": "metric"}],
        output_path=report_path,
        failures_path=None,
        provenance_check=lambda: None,
    )
    pointer_path = report_path.with_suffix(".current")
    original_target = pointer_path.readlink()
    pointer_path.unlink()
    pointer_path.symlink_to("../outside")
    with pytest.raises(CorpusIntegrityError, match="pointer target"):
        load_published_artifacts(report_path, validate=False)

    pointer_path.unlink()
    pointer_path.symlink_to(original_target)
    generation = pointer_path.resolve()
    checksum_path = generation / "report.sha256"
    digest = hashlib.sha256((generation / "report.json").read_bytes()).hexdigest()
    checksum_path.write_text(
        f"{digest}  report.json\n{digest}  report.json\n"
    )
    with pytest.raises(CorpusIntegrityError, match="checksum is malformed"):
        load_published_artifacts(report_path, validate=False)


def test_report_loader_rejects_symlink_generation_parent(tmp_path):
    report_path = tmp_path / "report.json"
    _publish_artifacts(
        {"aggregate": 1},
        [],
        output_path=report_path,
        failures_path=None,
        provenance_check=lambda: None,
    )
    generations = tmp_path / "report.generations"
    external = tmp_path / "external-report-generations"
    generations.replace(external)
    generations.symlink_to(external, target_is_directory=True)
    with pytest.raises(CorpusIntegrityError, match="generation store"):
        load_published_artifacts(report_path, validate=False)


def test_report_parent_symlink_fails_before_load_or_publish(tmp_path):
    publish = tmp_path / "publish"
    publish.mkdir()
    report_path = publish / "report.json"
    _publish_artifacts(
        {"aggregate": 1}, [], output_path=report_path, failures_path=None,
        provenance_check=lambda: None,
    )
    external = tmp_path / "external-publish"
    publish.replace(external)
    publish.symlink_to(external, target_is_directory=True)
    with pytest.raises(CorpusIntegrityError, match="traverse a symlink"):
        load_published_artifacts(report_path, validate=False)
    with pytest.raises(CorpusIntegrityError, match="traverse a symlink"):
        _publish_artifacts(
            {"aggregate": 2}, [], output_path=report_path, failures_path=None,
            provenance_check=lambda: None,
        )


@pytest.mark.parametrize(
    "target_name",
    ["report.json", "report_failures.jsonl", "report.sha256", "report.current", "report.generations"],
)
def test_report_publication_rejects_existing_symlink_targets(tmp_path, target_name):
    target = tmp_path / target_name
    target.symlink_to(tmp_path / "outside")
    with pytest.raises(CorpusIntegrityError, match="symlink|store"):
        _publish_artifacts(
            {"aggregate": 1}, [], output_path=tmp_path / "report.json",
            failures_path=tmp_path / "report_failures.jsonl", provenance_check=lambda: None,
        )


def test_report_loader_rejects_existing_public_role_symlink(tmp_path):
    report_path = tmp_path / "report.json"
    _publish_artifacts(
        {"aggregate": 1}, [], output_path=report_path, failures_path=None,
        provenance_check=lambda: None,
    )
    report_path.unlink()
    report_path.symlink_to(tmp_path / "outside")
    with pytest.raises(CorpusIntegrityError, match="missing or inconsistent"):
        load_published_artifacts(report_path, validate=False)


def test_report_conventional_paths_follow_one_selected_generation(tmp_path):
    report_path = tmp_path / "report.json"
    failure_path = tmp_path / "report_failures.jsonl"
    _publish_artifacts(
        {"aggregate": 1}, [{"turn_id": "old", "category": "metric"}],
        output_path=report_path, failures_path=failure_path, provenance_check=lambda: None,
    )
    _publish_artifacts(
        {"aggregate": 2}, [{"turn_id": "new", "category": "metric"}],
        output_path=report_path, failures_path=failure_path, provenance_check=lambda: None,
    )
    report, failures, generation = load_published_artifacts(report_path, validate=False)
    assert report == {"aggregate": 2}
    assert failures == [{"turn_id": "new", "category": "metric"}]
    assert report_path.read_bytes() == (generation / report_path.name).read_bytes()
    assert failure_path.read_bytes() == (generation / failure_path.name).read_bytes()
    assert report_path.with_suffix(".sha256").read_bytes() == (
        generation / "report.sha256"
    ).read_bytes()


def test_report_loader_rejects_generation_mode_change(tmp_path):
    report_path = tmp_path / "report.json"
    _publish_artifacts(
        {"aggregate": 1}, [], output_path=report_path, failures_path=None,
        provenance_check=lambda: None,
    )
    _, _, generation = load_published_artifacts(report_path, validate=False)
    (generation / "report.json").chmod(0o755)
    with pytest.raises(CorpusIntegrityError, match="mode"):
        load_published_artifacts(report_path, validate=False)


def test_report_loader_validates_schema_by_default(tmp_path):
    report_path = tmp_path / "report.json"
    _publish_artifacts(
        {"aggregate": 1}, [], output_path=report_path, failures_path=None,
        provenance_check=lambda: None,
    )
    with pytest.raises(CorpusIntegrityError, match="discriminator"):
        load_published_artifacts(report_path)
    assert load_published_artifacts(report_path, validate=False)[0] == {"aggregate": 1}


def test_report_loader_validates_corpus_and_commit_provenance_by_default(
    tmp_path, development_reports,
):
    report = development_reports[0]
    failures = _failure_rows_from_report(report)
    report_path = tmp_path / "development.json"
    _publish_artifacts(
        report, failures, output_path=report_path, failures_path=None,
        provenance_check=lambda: None,
    )
    loaded_report, loaded_failures, _generation = load_published_artifacts(report_path)
    assert loaded_report == report
    assert loaded_failures == failures


@pytest.mark.parametrize("failure_name", ["report.json", "report.sha256"])
def test_artifact_publish_rejects_colliding_targets_before_mutation(tmp_path, failure_name):
    report_path = tmp_path / "report.json"
    report_path.write_text("sentinel\n")
    with pytest.raises(CorpusIntegrityError, match="must be distinct"):
        _publish_artifacts(
            {"aggregate": 1},
            [],
            output_path=report_path,
            failures_path=tmp_path / failure_name,
            provenance_check=lambda: None,
        )
    assert report_path.read_text() == "sentinel\n"


@pytest.mark.parametrize("failure_name", ["report.current", "report.generations"])
def test_artifact_publish_rejects_reserved_target_collisions(tmp_path, failure_name):
    report_path = tmp_path / "report.json"
    with pytest.raises(CorpusIntegrityError, match="publication metadata"):
        _publish_artifacts(
            {"aggregate": 1},
            [],
            output_path=report_path,
            failures_path=tmp_path / failure_name,
            provenance_check=lambda: None,
        )
    assert not report_path.with_suffix(".current").exists()


def test_artifact_publish_requires_jsonl_failure_target_before_mutation(tmp_path):
    report_path = tmp_path / "report.json"
    with pytest.raises(CorpusIntegrityError, match=".jsonl"):
        _publish_artifacts(
            {"aggregate": 1}, [], output_path=report_path,
            failures_path=tmp_path / "failures.txt", provenance_check=lambda: None,
        )
    assert not report_path.with_suffix(".current").exists()


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


def test_baseline_is_aggregate_only_and_exact_post_113():
    report_path = ROOT / "evaluation/conversations/baselines/post_113_heldout_v1.json"
    if not report_path.with_suffix(".current").exists():
        pytest.skip("baseline is published only after the immutable core is accepted")
    report, failures, generation_dir = load_published_artifacts(report_path)
    assert report["production_code_commit"] == "c8c0bf4ccd5e73b1bd6bbe99762c87c4a549665e"
    assert set(report["modes"]) == {"sentence_only", "stateful", "transition_corrected"}
    assert report["development_correction_bundle"]["lookup_store_unchanged"] is True
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", report["evaluation_commit"], "HEAD"],
        cwd=ROOT,
        check=True,
    )
    assert report["corpus_root_sha256"] == load_manifest()["corpus_root_sha256"]
    selected_report_path = generation_dir / report_path.name
    failure_path = generation_dir / "post_113_heldout_v1_failures.jsonl"
    expected_hashes = {}
    for line in generation_dir.joinpath("post_113_heldout_v1.sha256").read_text().splitlines():
        digest, filename = line.split()
        expected_hashes[filename] = digest
    assert hashlib.sha256(selected_report_path.read_bytes()).hexdigest() == expected_hashes[report_path.name]
    assert hashlib.sha256(failure_path.read_bytes()).hexdigest() == expected_hashes[failure_path.name]
    conversations = load_split("heldout", purpose="evaluation")
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
