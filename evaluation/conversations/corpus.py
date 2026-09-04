"""Corpus compiler, loader, and immutable-split integrity checks.

This module deliberately has no dependency on :mod:`clanker_lm`.  Held-out
labels are produced from source annotations and a small, frozen weak-label
contract rather than from the system being measured.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter
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


def _validate_source_document(path: Path, document: Mapping[str, Any]) -> None:
    if not isinstance(document, Mapping):
        raise CorpusIntegrityError(f"{path}: source document must be an object")
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
        ambiguity = bool(overrides.get("ambiguity", "..." in text or "[inaudible]" in text.lower()))
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
            "expected_entity_refs": dict(overrides.get("expected_entity_refs", {})),
            "expected_answer": expected_answer,
            "supervision_level": str(source["supervision_level"]),
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
    digest = _sha256_bytes(_canonical_json(conversation).encode("utf-8"))
    conversation["conversation_sha256"] = digest
    conversation["conversation_id"] = f"{raw['source_conversation_id']}@sha256:{digest}"
    return conversation


def compile_corpora(
    *,
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = DATA_DIR,
    baseline_code_commit: str = "9ae77f072f8afda0b1d2b757ab492757cabff0f8",
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
            "allowed_uses": ["evaluation"] if split == "heldout" else ["development", "evaluation", "teacher_replay"],
            "training_eligible": split == "development",
            "teacher_replay_eligible": split == "development",
        }

    if split_entries["heldout"]["turn_count"] < 500:
        raise CorpusIntegrityError("held-out split must contain at least 500 turns")
    if split_entries["development"]["turn_count"] == 0:
        raise CorpusIntegrityError("separate open development split is required")

    compiler_sha256 = _sha256_file(Path(__file__))
    evaluator_path = ROOT / "runner.py"
    evaluator_sha256 = _sha256_file(evaluator_path) if evaluator_path.is_file() else None
    constituent_root = {
        "compiler_sha256": compiler_sha256,
        "evaluator_sha256": evaluator_sha256,
        "source_document_sha256": {
            item["source_document"]: item["source_document_sha256"]
            for item in source_manifest
        },
        "split_sha256": {
            split: entry["sha256"] for split, entry in sorted(split_entries.items())
        },
    }
    root_sha256 = _sha256_bytes(_canonical_json(constituent_root).encode("utf-8"))
    manifest = {
        "manifest_schema_version": 1,
        "corpus_version": CORPUS_VERSION,
        "content_address": f"sha256:{split_entries['heldout']['sha256']}",
        "frozen_date": "2026-09-04",
        "baseline_code_commit": baseline_code_commit,
        "compiler": "evaluation.conversations.corpus:compile_corpora",
        "compiler_sha256": compiler_sha256,
        "evaluator_sha256": evaluator_sha256,
        "corpus_root_sha256": root_sha256,
        "immutability_policy": (
            "heldout-v1 bytes and labels are frozen; corrections require a new corpus version"
        ),
        "splits": split_entries,
        "sources": sorted(source_manifest, key=lambda item: item["source_id"]),
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    publish_payloads = {
        **split_payloads,
        "manifest_v1.json": manifest_payload,
        "ROOT.sha256": (root_sha256 + "\n").encode("ascii"),
    }
    with tempfile.TemporaryDirectory(prefix=".conversation-compile-", dir=output_dir) as staging_name:
        staging = Path(staging_name)
        for filename, payload in publish_payloads.items():
            (staging / filename).write_bytes(payload)
        for filename in sorted(publish_payloads):
            (staging / filename).replace(output_dir / filename)
    return manifest


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    if not path.is_file():
        raise CorpusIntegrityError(f"missing corpus manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("corpus_version") != CORPUS_VERSION:
        raise CorpusIntegrityError("unsupported corpus version")
    root_path = path.parent / "ROOT.sha256"
    if not root_path.is_file() or root_path.read_text(encoding="ascii").strip() != manifest.get("corpus_root_sha256"):
        raise CorpusIntegrityError("corpus root digest is missing or inconsistent")
    return manifest


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
    if split not in manifest["splits"]:
        raise CorpusIntegrityError(f"unknown split: {split}")
    entry = manifest["splits"][split]
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
    with tempfile.TemporaryDirectory(prefix="clanker-conversation-eval-") as tmp:
        regenerated = compile_corpora(
            source_dir=source_dir,
            output_dir=Path(tmp),
            baseline_code_commit=str(manifest["baseline_code_commit"]),
        )
        for split, entry in manifest["splits"].items():
            if regenerated["splits"][split]["sha256"] != entry["sha256"]:
                raise CorpusIntegrityError(f"compiled {split} bytes are not reproducible")
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
        if relative.parts[0] in {".git", "evaluation", "dist", "build", "__pycache__"}:
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
