"""Corpus compiler, loader, and immutable-split integrity checks.

This module deliberately has no dependency on :mod:`clanker_lm`.  Held-out
labels are produced from source annotations and a small, frozen weak-label
contract rather than from the system being measured.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping


CORPUS_VERSION = "conversation-v1"
SOURCE_SCHEMA_VERSION = 1
COMPILED_SCHEMA_VERSION = 1
AXES = ("v", "a", "d", "u", "g", "w", "i")
ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sources"
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest_v1.json"
CURRENT_POINTER = "CURRENT"
GENERATIONS_DIRECTORY = "generations"
REPO_ROOT = ROOT.parent.parent
BASELINE_CODE_COMMIT = "66b85de66337789fa83292ecf683c6b23cc0af55"
PRODUCTION_PATHS = ("clanker_lm", "engine", "clanker_engine.py")
SPLIT_POLICIES = {
    "heldout": {
        "allowed_uses": ["evaluation"],
        "training_eligible": False,
        "teacher_replay_eligible": False,
    },
    "development": {
        "allowed_uses": ["development", "evaluation", "teacher_replay"],
        "training_eligible": True,
        "teacher_replay_eligible": True,
    },
}
SUPPORTED_LICENSES = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "Public Domain in the USA": "https://www.gutenberg.org/policy/license.html",
    "U.S. Government Work / Public Use Permitted": "https://ntrs.nasa.gov/citations/20160014392",
}

ALLOWED_DIALOGUE_ACTS = {
    "assert",
    "ask",
    "command",
    "acknowledge",
    "social",
    "clarify",
    "fragment",
}
ALLOWED_OUTCOMES = {
    "clarified",
    "understood",
    "dismissed",
    "escalated",
    "de-escalated",
    "continued",
    "ended",
}
ALLOWED_TRUTH = {"true", "false", "unknown", "conflict"}
ALLOWED_RESPONSE_ACTS = {
    "answer", "boundary", "empathic_acknowledge", "empathic_followup",
    "neutral_acknowledge", "positive_acknowledge", "probe", "safety_probe",
    "serious_followup", "social",
}
ALLOWED_DOMAINS = {
    "public_domain_drama",
    "public_domain_novel",
    "public_domain_real_human",
    "synthetic_adversarial",
    "open_development",
}
ALLOWED_ANSWER_STATUS = {
    "acknowledged",
    "answered",
    "true",
    "false",
    "unknown",
    "conflict",
    "unsupported",
    "missing_reference",
    "ambiguous_reference",
    "multiple_matches",
    "lexical_probe",
    "lexical_learned",
}


class CorpusIntegrityError(ValueError):
    """Raised when source or compiled evaluation data violates the contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_replace_path(source: Path, target: Path) -> None:
    source.replace(target)


def _selected_data_dir(data_dir: Path) -> Path:
    pointer = data_dir / CURRENT_POINTER
    if not pointer.is_file():
        return data_dir
    generation = pointer.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", generation):
        raise CorpusIntegrityError("corpus CURRENT pointer is malformed")
    selected = data_dir / GENERATIONS_DIRECTORY / generation
    if not selected.is_dir():
        raise CorpusIntegrityError("corpus CURRENT pointer selects a missing generation")
    return selected


def selected_manifest_path(path: Path = MANIFEST_PATH) -> Path:
    selected = _selected_data_dir(path.parent)
    return selected / path.name


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    location: str,
) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise CorpusIntegrityError(f"{location}: {'; '.join(details)}")


def _production_files(repo_root: Path = REPO_ROOT) -> List[Path]:
    files: List[Path] = []
    for relative in PRODUCTION_PATHS:
        path = repo_root / relative
        if path.is_dir():
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and (
                    item.suffix == ".py"
                    or item.relative_to(repo_root).parts[:2] == ("clanker_lm", "data")
                )
            )
        elif path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(repo_root).as_posix())


def _production_tree_payload(repo_root: Path = REPO_ROOT) -> List[Dict[str, str]]:
    return [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in _production_files(repo_root)
    ]


def production_tree_sha256(repo_root: Path = REPO_ROOT) -> str:
    return _sha256_bytes(_canonical_json(_production_tree_payload(repo_root)).encode("utf-8"))


def _baseline_production_tree_payload(
    commit: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> List[Dict[str, str]]:
    try:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit, "--", *PRODUCTION_PATHS],
            cwd=repo_root,
            text=True,
        ).splitlines()
        relevant = sorted(
            name
            for name in names
            if name.endswith(".py") or Path(name).parts[:2] == ("clanker_lm", "data")
        )
        return [
            {
                "path": name,
                "sha256": _sha256_bytes(
                    subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo_root)
                ),
            }
            for name in relevant
        ]
    except (OSError, subprocess.SubprocessError) as exc:
        raise CorpusIntegrityError(f"cannot resolve production baseline commit {commit}") from exc


def assert_production_tree(
    expected_sha256: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    actual = production_tree_sha256(repo_root)
    if actual != expected_sha256:
        raise CorpusIntegrityError(
            "local production module bytes do not match the manifest-bound post-#113 baseline"
        )


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower().replace("’", "'")))


def _clamp(value: float) -> int:
    return max(0, min(255, round(value)))


_POSITIVE = {
    "beautiful", "better", "calm", "clear", "excellent", "fine", "glad",
    "good", "great", "happy", "hope", "kind", "like", "love", "safe",
    "thank", "thanks", "well", "wonderful",
}
_NEGATIVE = {
    "afraid", "angry", "bad", "broke", "confused", "dead", "died",
    "difficult", "fear", "hate", "hurt", "ill", "lost", "not", "sad",
    "sick", "sorry", "trouble", "unsafe", "wrong",
}
_URGENT = {"danger", "emergency", "help", "immediately", "must", "now", "urgent"}
_SOCIAL = {"hello", "hi", "goodbye", "thanks", "thank"}
_ACKS = {"okay", "ok", "roger", "yes", "no", "understand", "understood"}
_COMMANDS = {
    "bring", "call", "close", "do", "forget", "give", "go", "keep", "leave",
    "look", "open", "remember", "say", "send", "stand", "stop", "tell", "wait",
}
_VERBS = {
    "am": "be", "are": "be", "is": "be", "was": "be", "were": "be",
    "asked": "ask", "asks": "ask", "bought": "buy", "broke": "break",
    "called": "call", "calls": "call", "closed": "close", "died": "die",
    "gave": "give", "go": "go", "goes": "go", "had": "have", "has": "have",
    "heard": "hear", "helped": "help", "know": "know", "knows": "know",
    "left": "leave", "like": "like", "likes": "like", "love": "love",
    "opened": "open", "passed": "pass", "remember": "remember",
    "remembered": "remember", "reported": "report", "said": "say", "saw": "see",
    "told": "tell", "want": "want", "wanted": "want", "went": "go",
}


def weak_affect(text: str, previous: Mapping[str, int]) -> Dict[str, int]:
    """Produce frozen corpus-side weak VADUGWI annotations.

    These are evaluation labels, not calls into Clanker.  Literary and federal
    transcript results must therefore be reported as weak supervision.
    """

    words = _normalized_text(text).split()
    pos = sum(word in _POSITIVE for word in words)
    neg = sum(word in _NEGATIVE for word in words)
    urgent = sum(word in _URGENT for word in words)
    question = text.rstrip().endswith("?")
    exclaim = "!" in text
    lexical = {
        "v": 128 + 20 * pos - 24 * neg,
        "a": 118 + 12 * (pos + neg) + 18 * urgent + (14 if exclaim else 0),
        "d": 128 + 8 * pos - 10 * neg,
        "u": 12 + 34 * urgent + 8 * neg + (8 if question else 0),
        "g": 128 + 10 * pos - 16 * neg,
        "w": 128 + 8 * pos - 12 * neg,
        "i": 128 + (28 if question else 0) + 10 * urgent,
    }
    return {
        axis: _clamp(float(previous.get(axis, 128 if axis != "u" else 0)) * 0.55 + lexical[axis] * 0.45)
        for axis in AXES
    }


def infer_dialogue_act(text: str) -> str:
    normalized = _normalized_text(text)
    words = normalized.split()
    if not words:
        return "fragment"
    if text.rstrip().endswith("?"):
        return "ask"
    if words[0] in _SOCIAL:
        return "social"
    if words[0] in _ACKS and len(words) <= 6:
        return "acknowledge"
    if words[0] in _COMMANDS:
        return "command"
    if len(words) <= 2 or normalized in {"stand by", "go ahead", "loud and clear"}:
        return "fragment"
    return "assert"


def infer_semantic(text: str, *, confidence: float) -> Dict[str, Any]:
    words = _normalized_text(text).split()
    predicate_index = next((i for i, word in enumerate(words) if word in _VERBS), None)
    if predicate_index is None:
        return {
            "predicate": "not_established",
            "roles": {},
            "scored": False,
            "annotation_method": "weak_rule_v1",
            "confidence": min(confidence, 0.45),
        }
    roles: Dict[str, str] = {}
    if predicate_index > 0:
        roles["agent"] = " ".join(words[:predicate_index])
    if predicate_index + 1 < len(words):
        roles["patient"] = " ".join(words[predicate_index + 1 :])
    return {
        "predicate": _VERBS[words[predicate_index]],
        "roles": roles,
        "scored": confidence >= 0.8,
        "annotation_method": "weak_rule_v1",
        "confidence": confidence,
    }


def infer_expected_response(text: str, dialogue_act: str) -> tuple[str, str, str]:
    words = set(_normalized_text(text).split())
    if dialogue_act == "ask":
        return "answer", "unknown", "unknown"
    if dialogue_act == "fragment":
        return "probe", "unsupported", "unknown"
    if dialogue_act == "command":
        return "boundary", "unsupported", "unknown"
    if words & {"dead", "died"}:
        return "empathic_acknowledge", "acknowledged", "unknown"
    if words & {"danger", "emergency", "help", "sick", "unsafe"}:
        return "serious_followup", "acknowledged", "unknown"
    if words & _POSITIVE:
        return "positive_acknowledge", "acknowledged", "unknown"
    return "neutral_acknowledge", "acknowledged", "unknown"


def infer_outcome(current: Mapping[str, Any], following: Mapping[str, Any] | None) -> str:
    if following is None:
        return "ended"
    current_act = str(current["annotation"]["dialogue_act"])
    next_act = str(following["annotation"]["dialogue_act"])
    if current_act in {"ask", "clarify"} and next_act in {"assert", "acknowledge"}:
        return "clarified"
    before = current["annotation"]["vadugwi_after"]
    after = following["annotation"]["vadugwi_after"]
    if int(after["v"]) - int(before["v"]) >= 12:
        return "de-escalated"
    if int(before["v"]) - int(after["v"]) >= 12:
        return "escalated"
    if next_act == "acknowledge":
        return "understood"
    return "continued"


def _validate_annotation_overrides(path: Path, conversation_id: str, value: Any) -> None:
    location = f"{path}: {conversation_id} annotation_overrides"
    if not isinstance(value, Mapping):
        raise CorpusIntegrityError(f"{location} must be an object")
    _require_exact_keys(
        value,
        required=set(),
        optional={
            "ambiguity", "annotator_confidence", "dialogue_act", "expected_answer",
            "expected_answer_status", "expected_entity_refs", "expected_response_act",
            "expected_truth", "outcome", "semantic",
        },
        location=location,
    )
    enum_fields = {
        "dialogue_act": ALLOWED_DIALOGUE_ACTS,
        "expected_response_act": ALLOWED_RESPONSE_ACTS,
        "expected_answer_status": ALLOWED_ANSWER_STATUS,
        "expected_truth": ALLOWED_TRUTH,
        "outcome": ALLOWED_OUTCOMES,
    }
    for field, allowed in enum_fields.items():
        if field in value and (not isinstance(value[field], str) or value[field] not in allowed):
            raise CorpusIntegrityError(f"{location}: invalid {field}")
    if "ambiguity" in value and type(value["ambiguity"]) is not bool:
        raise CorpusIntegrityError(f"{location}: ambiguity must be boolean")
    if "annotator_confidence" in value and (
        isinstance(value["annotator_confidence"], bool)
        or not isinstance(value["annotator_confidence"], (int, float))
        or not 0.0 <= value["annotator_confidence"] <= 1.0
    ):
        raise CorpusIntegrityError(f"{location}: invalid annotator_confidence")

    semantic = value.get("semantic")
    if semantic is not None:
        if not isinstance(semantic, Mapping):
            raise CorpusIntegrityError(f"{location}: semantic must be an object")
        _require_exact_keys(
            semantic,
            required={"predicate", "roles", "scored"},
            optional={"partial_roles"},
            location=f"{location}.semantic",
        )
        if not isinstance(semantic["predicate"], str) or not semantic["predicate"].strip():
            raise CorpusIntegrityError(f"{location}: semantic predicate must be nonempty text")
        if not isinstance(semantic["roles"], Mapping) or not all(
            isinstance(role, str) and role.strip()
            and isinstance(role_value, str) and role_value.strip()
            for role, role_value in semantic["roles"].items()
        ):
            raise CorpusIntegrityError(f"{location}: semantic roles must map nonempty strings")
        if type(semantic["scored"]) is not bool:
            raise CorpusIntegrityError(f"{location}: semantic.scored must be boolean")
        if "partial_roles" in semantic and type(semantic["partial_roles"]) is not bool:
            raise CorpusIntegrityError(f"{location}: semantic.partial_roles must be boolean")

    answer = value.get("expected_answer")
    if answer is not None:
        if not isinstance(answer, Mapping):
            raise CorpusIntegrityError(f"{location}: expected_answer must be an object")
        _require_exact_keys(
            answer,
            required={"scored", "values", "predicate", "requested_roles"},
            location=f"{location}.expected_answer",
        )
        if type(answer["scored"]) is not bool:
            raise CorpusIntegrityError(f"{location}: expected_answer.scored must be boolean")
        if not isinstance(answer["predicate"], str) or not answer["predicate"].strip():
            raise CorpusIntegrityError(f"{location}: expected_answer predicate must be nonempty text")
        for field in ("values", "requested_roles"):
            if not isinstance(answer[field], list) or not all(
                isinstance(item, str) and item.strip() for item in answer[field]
            ):
                raise CorpusIntegrityError(f"{location}: expected_answer.{field} must be a text array")
        if semantic is None or answer["predicate"] != semantic["predicate"]:
            raise CorpusIntegrityError(f"{location}: answer predicate must match semantic predicate")
        requested = semantic["roles"].get("requested_role")
        if answer["requested_roles"] != ([requested] if requested else []):
            raise CorpusIntegrityError(f"{location}: answer roles must match the typed question role")
        if answer["scored"] != (value.get("dialogue_act") == "ask"):
            raise CorpusIntegrityError(f"{location}: only explicit structural questions may score answers")
        if (
            answer["scored"]
            and value.get("expected_answer_status") in {"false", "unknown", "conflict"}
            and answer["values"]
        ):
            raise CorpusIntegrityError(f"{location}: non-positive answer cannot carry gold values")

    refs = value.get("expected_entity_refs")
    if refs is not None and (
        not isinstance(refs, Mapping)
        or not all(
            isinstance(key, str) and key.strip()
            and (
                isinstance(ref, str) and ref.strip()
                or isinstance(ref, list)
                and ref
                and all(isinstance(item, str) and item.strip() for item in ref)
            )
            for key, ref in refs.items()
        )
    ):
        raise CorpusIntegrityError(f"{location}: expected_entity_refs is malformed")


def _strict_validate_source_document(path: Path, document: Mapping[str, Any]) -> None:
    _require_exact_keys(
        document,
        required={"source_schema_version", "split", "learning_allowed", "sources", "conversations"},
        optional={"transcription_policy"},
        location=str(path),
    )
    split = document.get("split")
    if type(document.get("learning_allowed")) is not bool or (
        split in SPLIT_POLICIES
        and document["learning_allowed"] != SPLIT_POLICIES[str(split)]["training_eligible"]
    ):
        raise CorpusIntegrityError(f"{path}: learning_allowed disagrees with split policy")
    if "transcription_policy" in document and (
        not isinstance(document["transcription_policy"], str)
        or not document["transcription_policy"].strip()
    ):
        raise CorpusIntegrityError(f"{path}: transcription_policy must be nonempty text")
    required_source = {
        "source_id", "domain", "title", "creator", "publication_year",
        "source_url", "license_name", "license_url", "rights_note",
        "extraction_method", "annotation_method", "is_real_human", "is_public_domain",
        "training_eligible", "teacher_replay_eligible", "source_download_url",
        "provenance_evidence_url", "retrieval_date", "raw_source_sha256",
        "extraction_locator_schema", "extractor_version", "supervision_level",
        "outcome_evidence", "authoritative_source_url",
    }
    for source in document.get("sources", []):
        if not isinstance(source, Mapping):
            raise CorpusIntegrityError(f"{path}: each source must be an object")
        _require_exact_keys(
            source,
            required=required_source,
            optional={"rights_index_url", "structural_only"},
            location=f"{path}: source",
        )
        source_id = source["source_id"]
        if not isinstance(source["retrieval_date"], str):
            raise CorpusIntegrityError(f"{path}: source {source_id} has invalid retrieval_date")
        try:
            date.fromisoformat(source["retrieval_date"])
        except ValueError as exc:
            raise CorpusIntegrityError(f"{path}: source {source_id} has invalid retrieval_date") from exc
        domain = source["domain"]
        if source["is_real_human"] != (domain == "public_domain_real_human"):
            raise CorpusIntegrityError(f"{path}: source {source_id} real-human flag disagrees with domain")
        if (
            source["is_public_domain"] is not True
            or source["license_name"] not in SUPPORTED_LICENSES
            or source["license_url"] != SUPPORTED_LICENSES[source["license_name"]]
        ):
            raise CorpusIntegrityError(f"{path}: source {source_id} has no supported public-domain grant")
        structural_only = domain in {"synthetic_adversarial", "open_development"}
        if structural_only and source.get("structural_only") is not True:
            raise CorpusIntegrityError(f"{path}: source {source_id} structural_only disagrees with domain")
        if not structural_only and "structural_only" in source:
            raise CorpusIntegrityError(f"{path}: source {source_id} cannot declare structural_only")
        if "rights_index_url" in source and (
            not isinstance(source["rights_index_url"], str)
            or not source["rights_index_url"].startswith("https://")
        ):
            raise CorpusIntegrityError(f"{path}: source {source_id} rights_index_url must use HTTPS")
        if structural_only != (source["supervision_level"] == "gold_structural"):
            raise CorpusIntegrityError(f"{path}: source {source_id} supervision disagrees with domain")
        if split in SPLIT_POLICIES:
            policy = SPLIT_POLICIES[str(split)]
            if source["training_eligible"] != policy["training_eligible"]:
                raise CorpusIntegrityError(f"{path}: source {source_id} training policy disagrees with split")
            if source["teacher_replay_eligible"] != policy["teacher_replay_eligible"]:
                raise CorpusIntegrityError(f"{path}: source {source_id} replay policy disagrees with split")
    sources_by_id = {
        str(source["source_id"]): source
        for source in document.get("sources", [])
        if isinstance(source, Mapping) and "source_id" in source
    }
    for conversation in document.get("conversations", []):
        if not isinstance(conversation, Mapping):
            raise CorpusIntegrityError(f"{path}: each conversation must be an object")
        _require_exact_keys(
            conversation,
            required={"source_conversation_id", "source_id", "relationship_context", "participants", "turns"},
            optional={"entity_bindings", "lineage_id", "template_id", "template_variables"},
            location=f"{path}: conversation",
        )
        identifier = str(conversation["source_conversation_id"])
        if len(conversation.get("participants", [])) < 2:
            raise CorpusIntegrityError(f"{path}: {identifier} requires at least two participants")
        for field in ("lineage_id", "template_id"):
            if field in conversation and (
                not isinstance(conversation[field], str) or not conversation[field].strip()
            ):
                raise CorpusIntegrityError(f"{path}: {field} must be nonempty text")
        for field in ("entity_bindings", "template_variables"):
            if field in conversation and (
                not isinstance(conversation[field], Mapping)
                or not all(
                    isinstance(key, str) and key.strip()
                    and isinstance(item, str) and item.strip()
                    for key, item in conversation[field].items()
                )
            ):
                raise CorpusIntegrityError(f"{path}: {field} must map nonempty strings")
        for turn in conversation.get("turns", []):
            if not isinstance(turn, Mapping):
                raise CorpusIntegrityError(f"{path}: {identifier} turn must be an object")
            _require_exact_keys(
                turn,
                required={"speaker", "addressee", "text"},
                optional={"source_locator", "annotation_overrides"},
                location=f"{path}: {identifier} turn",
            )
            if "source_locator" in turn and not isinstance(turn["source_locator"], str):
                raise CorpusIntegrityError(f"{path}: source_locator must be text")
            overrides = turn.get("annotation_overrides", {})
            _validate_annotation_overrides(path, identifier, overrides)
            source = sources_by_id.get(str(conversation.get("source_id")))
            if source is not None and source.get("supervision_level") == "gold_structural":
                required = {
                    "dialogue_act", "semantic", "expected_response_act", "expected_answer_status",
                    "expected_truth", "outcome", "ambiguity", "annotator_confidence",
                    "expected_entity_refs",
                }
                missing = required - set(overrides)
                if missing:
                    raise CorpusIntegrityError(
                        f"{path}: {identifier} structural annotation missing {sorted(missing)}"
                    )
                if overrides["dialogue_act"] == "ask" and "expected_answer" not in overrides:
                    raise CorpusIntegrityError(
                        f"{path}: {identifier} structural question lacks expected_answer"
                    )


def _validate_source_document(path: Path, document: Mapping[str, Any]) -> None:
    if not isinstance(document, Mapping):
        raise CorpusIntegrityError(f"{path}: source document must be an object")
    try:
        _strict_validate_source_document(path, document)
    except CorpusIntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CorpusIntegrityError(f"{path}: malformed source document") from exc
    schema_version = document.get("source_schema_version")
    if type(schema_version) is not int or schema_version != SOURCE_SCHEMA_VERSION:
        raise CorpusIntegrityError(f"{path}: unsupported source schema")
    if document.get("split") not in {"heldout", "development"}:
        raise CorpusIntegrityError(f"{path}: split must be heldout or development")
    if not isinstance(document.get("learning_allowed"), bool):
        raise CorpusIntegrityError(f"{path}: learning_allowed boolean is required")
    if document["split"] == "heldout" and document["learning_allowed"]:
        raise CorpusIntegrityError(f"{path}: held-out source cannot allow learning")
    raw_sources = document.get("sources")
    raw_conversations = document.get("conversations")
    if not isinstance(raw_sources, list) or not isinstance(raw_conversations, list):
        raise CorpusIntegrityError(f"{path}: sources and conversations must be arrays")
    sources = list(raw_sources)
    conversations = list(raw_conversations)
    if not sources or not conversations:
        raise CorpusIntegrityError(f"{path}: sources and conversations are required")
    source_ids: set[str] = set()
    required_source = {
        "source_id", "domain", "title", "creator", "publication_year",
        "source_url", "license_name", "license_url", "rights_note",
        "extraction_method", "annotation_method", "is_real_human",
        "is_public_domain", "training_eligible", "teacher_replay_eligible",
        "source_download_url", "provenance_evidence_url", "retrieval_date",
        "raw_source_sha256", "extraction_locator_schema", "extractor_version",
        "supervision_level", "outcome_evidence", "authoritative_source_url",
    }
    for source in sources:
        if not isinstance(source, Mapping):
            raise CorpusIntegrityError(f"{path}: each source must be an object")
        missing = required_source - set(source)
        if missing:
            raise CorpusIntegrityError(f"{path}: source missing {sorted(missing)}")
        source_id = str(source["source_id"])
        if source_id in source_ids:
            raise CorpusIntegrityError(f"{path}: duplicate source_id {source_id}")
        source_ids.add(source_id)
        if type(source["publication_year"]) is not int or not 1 <= source["publication_year"] <= 9999:
            raise CorpusIntegrityError(f"{path}: source {source_id} has invalid publication_year")
        for field in (
            "is_real_human", "is_public_domain", "training_eligible",
            "teacher_replay_eligible",
        ):
            if type(source[field]) is not bool:
                raise CorpusIntegrityError(f"{path}: source {source_id} {field} must be boolean")
        for field in (
            "source_id", "domain", "title", "creator", "source_url",
            "source_download_url", "license_name", "license_url",
            "provenance_evidence_url", "authoritative_source_url", "rights_note", "extraction_method",
            "extraction_locator_schema", "extractor_version", "annotation_method",
            "supervision_level", "outcome_evidence",
        ):
            if not isinstance(source[field], str) or not source[field].strip():
                raise CorpusIntegrityError(f"{path}: source {source_id} has empty/non-string {field}")
        for field in (
            "source_url", "source_download_url", "license_url",
            "provenance_evidence_url", "authoritative_source_url",
        ):
            if not str(source[field]).startswith("https://"):
                raise CorpusIntegrityError(f"{path}: source {source_id} {field} must use HTTPS")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(source["retrieval_date"])):
            raise CorpusIntegrityError(f"{path}: source {source_id} has invalid retrieval_date")
        if not re.fullmatch(r"[0-9a-f]{64}", str(source["raw_source_sha256"])):
            raise CorpusIntegrityError(f"{path}: source {source_id} has invalid raw_source_sha256")
        if source["supervision_level"] not in {"weak", "gold_structural"}:
            raise CorpusIntegrityError(f"{path}: source {source_id} has invalid supervision_level")
        if source["domain"] not in ALLOWED_DOMAINS:
            raise CorpusIntegrityError(f"{path}: source {source_id} has invalid domain")
        if source["domain"] in {"synthetic_adversarial", "open_development"} and source.get("structural_only") is not True:
            raise CorpusIntegrityError(f"{path}: synthetic source {source_id} must be structural_only")
        if document["split"] == "heldout" and (
            source["training_eligible"] or source["teacher_replay_eligible"]
        ):
            raise CorpusIntegrityError(f"{path}: held-out source marked train/replay eligible")
        license_name = str(source["license_name"]).lower()
        if not source["is_public_domain"] and "cc0" not in license_name:
            raise CorpusIntegrityError(f"{path}: repository text lacks public-domain/CC0 grant")
    conversation_ids: set[str] = set()
    for conversation in conversations:
        if not isinstance(conversation, Mapping):
            raise CorpusIntegrityError(f"{path}: each conversation must be an object")
        for field in (
            "source_conversation_id", "source_id", "relationship_context",
            "participants", "turns",
        ):
            if field not in conversation:
                raise CorpusIntegrityError(f"{path}: conversation missing {field}")
        identifier = str(conversation["source_conversation_id"])
        if not isinstance(conversation["source_conversation_id"], str) or not identifier.strip():
            raise CorpusIntegrityError(f"{path}: conversation ID must be a nonempty string")
        if identifier in conversation_ids:
            raise CorpusIntegrityError(f"{path}: duplicate conversation {identifier}")
        conversation_ids.add(identifier)
        if str(conversation["source_id"]) not in source_ids:
            raise CorpusIntegrityError(f"{path}: unknown conversation source")
        if not isinstance(conversation["relationship_context"], str) or not conversation["relationship_context"].strip():
            raise CorpusIntegrityError(f"{path}: conversation relationship_context must be nonempty")
        if not isinstance(conversation["participants"], list) or not isinstance(conversation["turns"], list):
            raise CorpusIntegrityError(f"{path}: participants and turns must be arrays")
        participant_list = conversation["participants"]
        if not all(isinstance(item, str) and item.strip() for item in participant_list):
            raise CorpusIntegrityError(f"{path}: participants must be nonempty strings")
        if len(participant_list) != len(set(participant_list)):
            raise CorpusIntegrityError(f"{path}: duplicate conversation participant")
        if "entity_bindings" in conversation and not isinstance(conversation["entity_bindings"], Mapping):
            raise CorpusIntegrityError(f"{path}: entity_bindings must be an object")
        participants = set(participant_list)
        turns = list(conversation["turns"])
        if len(turns) < 4:
            raise CorpusIntegrityError(f"{path}: {identifier} is not a whole conversation")
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise CorpusIntegrityError(f"{path}: {identifier} turn must be an object")
            for field in ("speaker", "addressee", "text"):
                if not isinstance(turn.get(field), str) or not turn[field].strip():
                    raise CorpusIntegrityError(f"{path}: {identifier} turn missing {field}")
            if "annotation_overrides" in turn and not isinstance(turn["annotation_overrides"], Mapping):
                raise CorpusIntegrityError(f"{path}: {identifier} annotation_overrides must be an object")
            if str(turn["speaker"]) not in participants or str(turn["addressee"]) not in participants:
                raise CorpusIntegrityError(f"{path}: turn participant missing from participants")


def _compile_conversation(
    raw: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    split: str,
) -> Dict[str, Any]:
    confidence = 0.78 if source["domain"] == "public_domain_real_human" else 0.60
    if source["domain"] in {"synthetic_adversarial", "open_development"}:
        confidence = 0.98
    previous = {axis: 128 for axis in AXES}
    previous["u"] = 0
    generated_bindings = {
        str(participant): f"participant:{_normalized_text(str(participant)).replace(' ', '_')}"
        for participant in raw["participants"]
    }
    participant_bindings = dict(raw.get("entity_bindings", generated_bindings))
    for participant in raw["participants"]:
        participant_bindings.setdefault(str(participant), generated_bindings[str(participant)])
    if not all(isinstance(key, str) and isinstance(value, str) and value for key, value in participant_bindings.items()):
        raise CorpusIntegrityError("entity_bindings must map strings to stable corpus-local IDs")
    turns: List[Dict[str, Any]] = []
    for index, raw_turn in enumerate(raw["turns"], 1):
        text = " ".join(str(raw_turn["text"]).split())
        overrides = dict(raw_turn.get("annotation_overrides", {}))
        dialogue_act = str(overrides.get("dialogue_act", infer_dialogue_act(text)))
        if dialogue_act not in ALLOWED_DIALOGUE_ACTS:
            raise CorpusIntegrityError(f"invalid dialogue act {dialogue_act}")
        response_act, answer_status, truth = infer_expected_response(text, dialogue_act)
        response_act = str(overrides.get("expected_response_act", response_act))
        answer_status = str(overrides.get("expected_answer_status", answer_status))
        truth = str(overrides.get("expected_truth", truth))
        if response_act not in ALLOWED_RESPONSE_ACTS:
            raise CorpusIntegrityError(f"invalid expected response act {response_act}")
        if answer_status not in ALLOWED_ANSWER_STATUS or truth not in ALLOWED_TRUTH:
            raise CorpusIntegrityError("invalid expected answer status/truth")
        turn_confidence = float(overrides.get("annotator_confidence", confidence))
        if not 0.0 <= turn_confidence <= 1.0:
            raise CorpusIntegrityError("annotator confidence must be within [0, 1]")
        after = weak_affect(text, previous)
        semantic = dict(overrides.get("semantic", infer_semantic(text, confidence=turn_confidence)))
        semantic.setdefault("roles", {})
        semantic.setdefault("scored", semantic.get("predicate") not in {None, "not_established"})
        semantic.setdefault("annotation_method", source["annotation_method"])
        semantic.setdefault("confidence", turn_confidence)
        expected_answer = dict(overrides.get("expected_answer", {}))
        expected_answer.setdefault("scored", False)
        expected_answer.setdefault("values", [])
        expected_answer.setdefault("predicate", semantic.get("predicate"))
        expected_answer.setdefault("requested_roles", [])
        ambiguity = overrides.get("ambiguity", "..." in text or "[inaudible]" in text.lower())
        flat_refs = dict(overrides.get("expected_entity_refs", {}))
        semantic_role_names = set(dict(semantic.get("roles", {}))) - {"requested_role"}
        expected_entity_refs = (
            {
                "roles": {
                    key: value for key, value in flat_refs.items() if key in semantic_role_names
                },
                "mentions": {
                    key: value for key, value in flat_refs.items() if key not in semantic_role_names
                },
            }
            if flat_refs
            else {}
        )
        structural_only = source.get("structural_only") is True
        annotation = {
            "dialogue_act": dialogue_act,
            "semantic": semantic,
            "vadugwi_before": dict(previous),
            "vadugwi_after": after,
            "observed_next_state": None,
            "expected_response_act": response_act,
            "expected_answer_status": answer_status,
            "expected_truth": truth,
            "outcome": str(overrides.get("outcome", "continued")),
            "ambiguity": ambiguity,
            "annotator_confidence": turn_confidence,
            "expected_entity_refs": expected_entity_refs,
            "expected_answer": expected_answer,
            "supervision_level": str(source["supervision_level"]),
            "semantic_supervision_level": str(source["supervision_level"]),
            "affect_supervision_level": "weak_rule_v1",
            "affect_scored": not structural_only,
            "outcome_evidence": str(source["outcome_evidence"]),
        }
        turns.append(
            {
                "turn_id": f"{raw['source_conversation_id']}:t{index:02d}",
                "speaker": str(raw_turn["speaker"]),
                "addressee": str(raw_turn["addressee"]),
                "text": text,
                "source_locator": str(raw_turn.get("source_locator", "")),
                "annotation": annotation,
            }
        )
        previous = after
    for index, turn in enumerate(turns):
        following = turns[index + 1] if index + 1 < len(turns) else None
        turn["annotation"]["observed_next_state"] = (
            dict(following["annotation"]["vadugwi_after"]) if following else None
        )
        raw_override = dict(raw["turns"][index].get("annotation_overrides", {}))
        outcome = str(raw_override.get("outcome", infer_outcome(turn, following)))
        if outcome not in ALLOWED_OUTCOMES:
            raise CorpusIntegrityError(f"invalid outcome {outcome}")
        turn["annotation"]["outcome"] = outcome
    conversation: Dict[str, Any] = {
        "schema_version": COMPILED_SCHEMA_VERSION,
        "corpus_version": CORPUS_VERSION,
        "split": split,
        "source_conversation_id": str(raw["source_conversation_id"]),
        "source_id": str(raw["source_id"]),
        "lineage_id": str(raw.get("lineage_id", raw["source_id"])),
        "domain": str(source["domain"]),
        "relationship_context": str(raw["relationship_context"]),
        "participants": [str(item) for item in raw["participants"]],
        "participant_bindings": participant_bindings,
        "source": {
            key: source[key]
            for key in (
                "title", "creator", "publication_year", "source_url",
                "license_name", "license_url", "rights_note", "extraction_method",
                "annotation_method", "is_real_human", "is_public_domain",
                "training_eligible", "teacher_replay_eligible",
                "source_download_url", "provenance_evidence_url",
                "authoritative_source_url",
                "retrieval_date", "raw_source_sha256", "extraction_locator_schema",
                "extractor_version", "supervision_level", "outcome_evidence",
            )
        },
        "turns": turns,
    }
    conversation["source"]["structural_only"] = source.get("structural_only") is True
    digest = _sha256_bytes(_canonical_json(conversation).encode("utf-8"))
    conversation["conversation_sha256"] = digest
    conversation["conversation_id"] = f"{raw['source_conversation_id']}@sha256:{digest}"
    return conversation


def _manifest_constituents(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return every immutable and policy-relevant field bound by ROOT.sha256."""

    fields = (
        "manifest_schema_version", "corpus_version", "content_address", "frozen_date",
        "baseline_code_commit", "production_code_sha256", "production_code_files",
        "compiler", "compiler_sha256", "evaluator_sha256", "schema_sha256", "immutability_policy",
        "splits", "sources",
    )
    return {field: manifest[field] for field in fields}


def compile_corpora(
    *,
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = DATA_DIR,
    baseline_code_commit: str = BASELINE_CODE_COMMIT,
) -> Dict[str, Any]:
    """Compile all source documents into canonical JSONL splits and a manifest."""

    documents: List[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(source_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CorpusIntegrityError(f"{path}: unreadable source document") from exc
        _validate_source_document(path, document)
        documents.append((path, document))
    if not documents:
        raise CorpusIntegrityError("no source documents found")

    by_split: Dict[str, List[Dict[str, Any]]] = {"heldout": [], "development": []}
    source_manifest: List[Dict[str, Any]] = []
    seen_sources: Dict[str, Mapping[str, Any]] = {}
    seen_conversations: set[str] = set()
    split_lineages: Dict[str, set[str]] = {"heldout": set(), "development": set()}
    for path, document in documents:
        sources = {str(item["source_id"]): item for item in document["sources"]}
        for source_id, source in sources.items():
            if source_id in seen_sources and _canonical_json(source) != _canonical_json(seen_sources[source_id]):
                raise CorpusIntegrityError(f"conflicting duplicate source_id across documents: {source_id}")
            if source_id not in seen_sources:
                try:
                    source_document = str(path.relative_to(ROOT))
                except ValueError:
                    source_document = path.name
                source_manifest.append(
                    {
                        **dict(source),
                        "source_document": source_document,
                        "source_document_sha256": _sha256_file(path),
                    }
                )
                seen_sources[source_id] = source
        split = str(document["split"])
        for raw in document["conversations"]:
            source_id = str(raw["source_id"])
            identifier = str(raw["source_conversation_id"])
            if identifier in seen_conversations:
                raise CorpusIntegrityError(f"duplicate cross-document conversation {identifier}")
            seen_conversations.add(identifier)
            lineage_id = str(raw.get("lineage_id", source_id)).strip()
            if not lineage_id:
                raise CorpusIntegrityError(f"{path}: conversation {identifier} has empty lineage_id")
            split_lineages[split].add(lineage_id)
            by_split[split].append(
                _compile_conversation(raw, source=sources[source_id], split=split)
            )

    lineage_overlap = split_lineages["heldout"] & split_lineages["development"]
    if lineage_overlap:
        raise CorpusIntegrityError(f"lineage occurs in both splits: {sorted(lineage_overlap)[0]}")

    split_entries: Dict[str, Any] = {}
    split_payloads: Dict[str, bytes] = {}
    for split, conversations in by_split.items():
        conversations.sort(key=lambda item: item["source_conversation_id"])
        payload = "".join(_canonical_json(item) + "\n" for item in conversations).encode("utf-8")
        filename = "heldout_v1.jsonl" if split == "heldout" else "development_v1.jsonl"
        path = output_dir / filename
        split_payloads[filename] = payload
        domain_counts = Counter(item["domain"] for item in conversations)
        domain_turns = Counter()
        conversation_hashes: Dict[str, str] = {}
        for item in conversations:
            domain_turns[item["domain"]] += len(item["turns"])
            conversation_hashes[item["source_conversation_id"]] = item["conversation_sha256"]
        split_entries[split] = {
            "path": filename,
            "sha256": _sha256_bytes(payload),
            "conversation_count": len(conversations),
            "turn_count": sum(len(item["turns"]) for item in conversations),
            "domain_conversations": dict(sorted(domain_counts.items())),
            "domain_turns": dict(sorted(domain_turns.items())),
            "conversation_hashes": dict(sorted(conversation_hashes.items())),
            **SPLIT_POLICIES[split],
        }

    if split_entries["heldout"]["turn_count"] < 500:
        raise CorpusIntegrityError("held-out split must contain at least 500 turns")
    if split_entries["development"]["turn_count"] == 0:
        raise CorpusIntegrityError("separate open development split is required")

    compiler_sha256 = _sha256_file(Path(__file__))
    evaluator_path = ROOT / "runner.py"
    evaluator_sha256 = _sha256_file(evaluator_path) if evaluator_path.is_file() else None
    schema_sha256 = {
        path.name: _sha256_file(path)
        for path in sorted((ROOT / "schema").glob("*.json"))
    }
    production_files = _production_tree_payload()
    baseline_files = _baseline_production_tree_payload(baseline_code_commit)
    if production_files != baseline_files:
        raise CorpusIntegrityError(
            "local production module bytes differ from the requested baseline commit"
        )
    production_sha256 = _sha256_bytes(_canonical_json(production_files).encode("utf-8"))
    manifest: Dict[str, Any] = {
        "manifest_schema_version": 1,
        "corpus_version": CORPUS_VERSION,
        "content_address": f"sha256:{split_entries['heldout']['sha256']}",
        "frozen_date": "2026-09-04",
        "baseline_code_commit": baseline_code_commit,
        "production_code_sha256": production_sha256,
        "production_code_files": production_files,
        "compiler": "evaluation.conversations.corpus:compile_corpora",
        "compiler_sha256": compiler_sha256,
        "evaluator_sha256": evaluator_sha256,
        "schema_sha256": schema_sha256,
        "immutability_policy": (
            "heldout-v1 bytes and labels are frozen; corrections require a new corpus version"
        ),
        "splits": split_entries,
        "sources": sorted(source_manifest, key=lambda item: item["source_id"]),
    }
    root_sha256 = _sha256_bytes(_canonical_json(_manifest_constituents(manifest)).encode("utf-8"))
    manifest["corpus_root_sha256"] = root_sha256
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    publish_payloads = {
        **split_payloads,
        "manifest_v1.json": manifest_payload,
        "ROOT.sha256": (root_sha256 + "\n").encode("ascii"),
    }
    generations = output_dir / GENERATIONS_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    generation_dir = generations / root_sha256
    with tempfile.TemporaryDirectory(prefix=".conversation-compile-", dir=output_dir) as staging_name:
        staging_root = Path(staging_name)
        staging = staging_root / "generation"
        staging.mkdir()
        for filename, payload in publish_payloads.items():
            (staging / filename).write_bytes(payload)
        if generation_dir.exists():
            for filename, payload in publish_payloads.items():
                if not (generation_dir / filename).is_file() or (generation_dir / filename).read_bytes() != payload:
                    raise CorpusIntegrityError("existing immutable corpus generation differs from compiled bytes")
        else:
            _atomic_replace_path(staging, generation_dir)
        pointer_stage = staging_root / CURRENT_POINTER
        pointer_stage.write_text(root_sha256 + "\n", encoding="ascii")
        _atomic_replace_path(pointer_stage, output_dir / CURRENT_POINTER)
    return manifest


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    path = selected_manifest_path(path)
    if not path.is_file():
        raise CorpusIntegrityError(f"missing corpus manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusIntegrityError(f"unreadable corpus manifest: {path}") from exc
    if not isinstance(manifest, Mapping):
        raise CorpusIntegrityError("corpus manifest must be an object")
    _require_exact_keys(
        manifest,
        required={
            "manifest_schema_version", "corpus_version", "content_address", "frozen_date",
            "baseline_code_commit", "production_code_sha256", "production_code_files",
            "compiler", "compiler_sha256", "evaluator_sha256", "schema_sha256", "corpus_root_sha256",
            "immutability_policy", "splits", "sources",
        },
        location=str(path),
    )
    if type(manifest["manifest_schema_version"]) is not int or manifest["manifest_schema_version"] != 1:
        raise CorpusIntegrityError("unsupported manifest schema")
    if manifest["corpus_version"] != CORPUS_VERSION:
        raise CorpusIntegrityError("unsupported corpus version")
    if not isinstance(manifest["splits"], Mapping) or set(manifest["splits"]) != set(SPLIT_POLICIES):
        raise CorpusIntegrityError("manifest split inventory is invalid")
    split_keys = {
        "path", "sha256", "conversation_count", "turn_count", "domain_conversations",
        "domain_turns", "conversation_hashes", "allowed_uses", "training_eligible",
        "teacher_replay_eligible",
    }
    for split, expected_policy in SPLIT_POLICIES.items():
        entry = manifest["splits"][split]
        if not isinstance(entry, Mapping):
            raise CorpusIntegrityError(f"manifest split {split} must be an object")
        _require_exact_keys(entry, required=split_keys, location=f"manifest split {split}")
        for field, expected in expected_policy.items():
            if entry[field] != expected or type(entry[field]) is not type(expected):
                raise CorpusIntegrityError(f"manifest split {split} violates immutable {field} policy")
        if type(entry["conversation_count"]) is not int or type(entry["turn_count"]) is not int:
            raise CorpusIntegrityError(f"manifest split {split} counts must be integers")
        if not isinstance(entry["path"], str) or Path(entry["path"]).name != entry["path"]:
            raise CorpusIntegrityError(f"manifest split {split} path is invalid")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise CorpusIntegrityError(f"manifest split {split} digest is invalid")
    expected_root = _sha256_bytes(_canonical_json(_manifest_constituents(manifest)).encode("utf-8"))
    if manifest["corpus_root_sha256"] != expected_root:
        raise CorpusIntegrityError("manifest policy/provenance constituent root is inconsistent")
    if path.parent.parent.name == GENERATIONS_DIRECTORY and path.parent.name != expected_root:
        raise CorpusIntegrityError("selected corpus generation name disagrees with constituent root")
    root_path = path.parent / "ROOT.sha256"
    if not root_path.is_file() or root_path.read_text(encoding="ascii").strip() != expected_root:
        raise CorpusIntegrityError("corpus root digest is missing or inconsistent")
    return dict(manifest)


def load_split(
    split: str,
    *,
    purpose: str,
    manifest_path: Path = MANIFEST_PATH,
) -> List[Dict[str, Any]]:
    """Load a split only for a manifest-authorized purpose.

    The explicit purpose parameter is the technical boundary that prevents the
    immutable held-out labels from entering teacher replay or promotion paths.
    """

    manifest = load_manifest(manifest_path)
    manifest_path = selected_manifest_path(manifest_path)
    if split not in manifest["splits"]:
        raise CorpusIntegrityError(f"unknown split: {split}")
    entry = manifest["splits"][split]
    if split == "heldout" and purpose != "evaluation":
        raise CorpusIntegrityError(f"split {split} is forbidden for purpose {purpose}")
    if purpose not in entry["allowed_uses"]:
        raise CorpusIntegrityError(f"split {split} is forbidden for purpose {purpose}")
    path = manifest_path.parent / str(entry["path"])
    if _sha256_file(path) != entry["sha256"]:
        raise CorpusIntegrityError(f"content hash mismatch for {split}")
    conversations = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(conversations) != int(entry["conversation_count"]):
        raise CorpusIntegrityError(f"conversation count mismatch for {split}")
    for conversation in conversations:
        expected = str(conversation.get("conversation_sha256", ""))
        unhashed = dict(conversation)
        unhashed.pop("conversation_id", None)
        unhashed.pop("conversation_sha256", None)
        actual = _sha256_bytes(_canonical_json(unhashed).encode("utf-8"))
        if expected != actual or conversation.get("conversation_id") != f"{conversation['source_conversation_id']}@sha256:{actual}":
            raise CorpusIntegrityError(f"whole-conversation content address mismatch: {conversation.get('source_conversation_id')}")
        if split == "heldout" and (
            conversation["source"]["training_eligible"]
            or conversation["source"]["teacher_replay_eligible"]
        ):
            raise CorpusIntegrityError("held-out compiled conversation crossed the learning boundary")
    return conversations


def _shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    words = _normalized_text(text).split()
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def _near_duplicate_score(left: str, right: str) -> float:
    a = _shingles(left)
    b = _shingles(right)
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / len(a | b)


def verify_corpus(
    *,
    manifest_path: Path = MANIFEST_PATH,
    source_dir: Path = SOURCE_DIR,
) -> Dict[str, Any]:
    """Recompile and enforce licensing, immutability, and leakage gates."""

    manifest = load_manifest(manifest_path)
    assert_production_tree(str(manifest["production_code_sha256"]))
    with tempfile.TemporaryDirectory(prefix="clanker-conversation-eval-") as tmp:
        regenerated = compile_corpora(
            source_dir=source_dir,
            output_dir=Path(tmp),
            baseline_code_commit=str(manifest["baseline_code_commit"]),
        )
        if regenerated["splits"] != manifest["splits"]:
            raise CorpusIntegrityError("compiled split bytes or policy metadata are not reproducible")
        if regenerated["corpus_root_sha256"] != manifest["corpus_root_sha256"]:
            raise CorpusIntegrityError("compiler/evaluator/provenance constituent root changed")
        if regenerated["sources"] != manifest["sources"]:
            raise CorpusIntegrityError("source provenance manifest is not reproducible")

    heldout = load_split("heldout", purpose="evaluation", manifest_path=manifest_path)
    development = load_split("development", purpose="development", manifest_path=manifest_path)
    heldout_turns = [turn for conversation in heldout for turn in conversation["turns"]]
    development_turns = [turn for conversation in development for turn in conversation["turns"]]
    heldout_normalized = {_normalized_text(turn["text"]): turn["turn_id"] for turn in heldout_turns}
    development_normalized = {_normalized_text(turn["text"]): turn["turn_id"] for turn in development_turns}
    exact_overlap = set(heldout_normalized) & set(development_normalized)
    if exact_overlap:
        example = sorted(exact_overlap)[0]
        raise CorpusIntegrityError(
            "exact heldout/development leakage: "
            f"{heldout_normalized[example]} vs {development_normalized[example]}"
        )

    near_duplicates: List[Dict[str, Any]] = []
    development_long = [turn for turn in development_turns if len(_normalized_text(turn["text"]).split()) >= 5]
    for heldout_turn in heldout_turns:
        if len(_normalized_text(heldout_turn["text"]).split()) < 5:
            continue
        for development_turn in development_long:
            score = _near_duplicate_score(heldout_turn["text"], development_turn["text"])
            if score >= 0.84:
                near_duplicates.append(
                    {
                        "heldout_turn": heldout_turn["turn_id"],
                        "development_turn": development_turn["turn_id"],
                        "score": score,
                    }
                )
    if near_duplicates:
        raise CorpusIntegrityError(f"near-duplicate split leakage: {near_duplicates[0]}")

    repo_root = ROOT.parent.parent
    forbidden_needles = {
        "heldout_v1.jsonl",
        str(manifest["content_address"]),
        "evaluation.conversations.data.heldout",
    }
    production_hits: List[str] = []
    for path in sorted((repo_root / "clanker_lm").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in forbidden_needles):
            production_hits.append(str(path.relative_to(repo_root)))
    if production_hits:
        raise CorpusIntegrityError(f"production imports/references held-out labels: {production_hits}")

    text_leakage_hits: List[Dict[str, str]] = []
    relevant_suffixes = {".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".html", ".js"}
    heldout_phrases = {
        _normalized_text(turn["text"]): turn["turn_id"]
        for turn in heldout_turns
        if len(_normalized_text(turn["text"]).split()) >= 7
    }
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in relevant_suffixes:
            continue
        relative = path.relative_to(repo_root)
        if relative.parts[0] in {".git", "dist", "build", "__pycache__"}:
            continue
        if relative.parts[:3] in {
            ("evaluation", "conversations", "data"),
            ("evaluation", "conversations", "sources"),
        }:
            continue
        try:
            normalized_file = _normalized_text(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        file_shingles = _shingles(normalized_file)
        for phrase, turn_id in heldout_phrases.items():
            exact = phrase in normalized_file
            phrase_shingles = _shingles(phrase)
            containment = (
                len(phrase_shingles & file_shingles) / len(phrase_shingles)
                if phrase_shingles and len(phrase.split()) >= 10
                else 0.0
            )
            if exact or containment >= 0.84:
                text_leakage_hits.append({
                    "turn_id": turn_id,
                    "path": str(relative),
                    "match": "exact" if exact else "near",
                })
                break
    if text_leakage_hits:
        raise CorpusIntegrityError(f"held-out text leaked into production/tests: {text_leakage_hits[0]}")

    return {
        "verified": True,
        "corpus_version": CORPUS_VERSION,
        "content_address": manifest["content_address"],
        "heldout_conversations": len(heldout),
        "heldout_turns": len(heldout_turns),
        "development_conversations": len(development),
        "development_turns": len(development_turns),
        "exact_split_overlap": 0,
        "near_duplicate_split_overlap": 0,
        "production_reference_hits": 0,
        "production_text_leakage_hits": 0,
    }
