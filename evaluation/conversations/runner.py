"""Deterministic multi-turn evaluator for the frozen conversation corpus.

The evaluator never emits source text or generated responses.  Held-out state
is scoped to one whole conversation, and correction lookup is seeded only from
the open development split.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from clanker_lm.affect import ClankerAffectBackend
from clanker_lm.database import LanguageStore
from clanker_lm.model import AffectVector, TurnResult
from clanker_lm.runtime import ClankerLM
from clanker_lm.trajectory import TrajectoryController

from .corpus import (
    AXES,
    MANIFEST_PATH,
    REPO_ROOT,
    CorpusIntegrityError,
    _canonical_json,
    _sha256_file,
    assert_production_tree,
    load_manifest,
    load_split,
)


METRIC_SCHEMA_VERSION = 1
MODES = ("sentence_only", "stateful", "transition_corrected")
BOOTSTRAP_DRAWS = 10_000
FIXED_INSTANT = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)


class NoCorrectionTrajectory(TrajectoryController):
    """Observe transitions while returning the production target unchanged."""

    def adjust_target(self, target: AffectVector, context_key: str) -> Tuple[AffectVector, Dict[str, Any]]:
        del context_key
        return target, {"applied": False, "sample_count": 0, "evaluation_mode": "no_correction"}


class FrozenLookupTrajectory(TrajectoryController):
    """Write observations locally, but read corrections from an immutable store."""

    def __init__(self, observation_store: LanguageStore, lookup_store: LanguageStore) -> None:
        super().__init__(observation_store)
        self.lookup = TrajectoryController(lookup_store)

    def adjust_target(self, target: AffectVector, context_key: str) -> Tuple[AffectVector, Dict[str, Any]]:
        adjusted, metadata = self.lookup.adjust_target(target, context_key)
        return adjusted, {**metadata, "lookup": "development_only_frozen"}


def _runtime(store: LanguageStore, *, mode: str, correction_store: LanguageStore | None) -> ClankerLM:
    runtime = ClankerLM(
        language_store=store,
        affect_backend=ClankerAffectBackend(),
        active_profile_id=None,
        default_timezone="UTC",
        clock=lambda: FIXED_INSTANT,
        learning_scope="conversation-eval-ephemeral",
    )
    if runtime.affect_backend_name != "clanker-v8":
        raise RuntimeError(f"conversation baseline requires clanker-v8, got {runtime.affect_backend_name}")
    if mode in {"sentence_only", "stateful"}:
        runtime.trajectory = NoCorrectionTrajectory(store)
    elif mode == "transition_corrected":
        if correction_store is None:
            raise RuntimeError("transition-corrected mode requires a development correction store")
        runtime.trajectory = FrozenLookupTrajectory(store, correction_store)
    else:
        raise ValueError(f"unknown evaluation mode: {mode}")
    return runtime


def _merge_transition_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["context_key"])].append(row)
    merged: Dict[str, Dict[str, Any]] = {}
    for key, items in grouped.items():
        count = sum(int(item["sample_count"]) for item in items)
        residual = {
            axis: sum(
                json.loads(str(item["mean_residual_json"]))[axis] * int(item["sample_count"])
                for item in items
            ) / count
            for axis in AXES
        }
        reaction = {
            axis: round(sum(
                json.loads(str(item["mean_reaction_json"]))[axis] * int(item["sample_count"])
                for item in items
            ) / count)
            for axis in AXES
        }
        success = sum(float(item["success_mean"]) * int(item["sample_count"]) for item in items) / count
        merged[key] = {
            "sample_count": count,
            "mean_residual": residual,
            "mean_reaction": reaction,
            "success_mean": success,
        }
    return merged


def _build_development_corrections(
    conversations: Sequence[Mapping[str, Any]],
) -> Tuple[LanguageStore, Dict[str, Any]]:
    """Build a correction lookup from open development conversations only."""

    transition_rows: List[Mapping[str, Any]] = []
    for conversation in conversations:
        store = LanguageStore()
        runtime = _runtime(store, mode="stateful", correction_store=None)
        try:
            for turn in conversation["turns"]:
                runtime.process(str(turn["text"]))
            transition_rows.extend(
                dict(row)
                for row in store.connection.execute(
                    "SELECT context_key, sample_count, mean_residual_json, "
                    "mean_reaction_json, success_mean FROM transition_stats"
                ).fetchall()
            )
        finally:
            runtime.close()

    expected_finalized = sum(max(0, len(conversation["turns"]) - 1) for conversation in conversations)
    actual_finalized = sum(int(row["sample_count"]) for row in transition_rows)
    if actual_finalized != expected_finalized:
        raise RuntimeError(
            f"development correction finalization mismatch: {actual_finalized} != {expected_finalized}"
        )
    merged = _merge_transition_rows(transition_rows)
    correction_store = LanguageStore()
    with correction_store.connection:
        for key, item in sorted(merged.items()):
            correction_store.connection.execute(
                "INSERT INTO transition_stats(context_key, sample_count, mean_residual_json, "
                "mean_reaction_json, success_mean) VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    item["sample_count"],
                    json.dumps(item["mean_residual"], sort_keys=True),
                    json.dumps(item["mean_reaction"], sort_keys=True),
                    item["success_mean"],
                ),
            )
    canonical = [
        {"context_key": key, **item}
        for key, item in sorted(merged.items())
    ]
    digest = hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()
    return correction_store, {
        "source_split": "development",
        "source_conversation_count": len(conversations),
        "context_count": len(merged),
        "sample_count": sum(item["sample_count"] for item in merged.values()),
        "expected_finalized_sample_count": expected_finalized,
        "terminal_pending_turns_excluded": len(conversations),
        "eligible_context_count": sum(
            item["sample_count"] >= TrajectoryController.MIN_SAMPLES for item in merged.values()
        ),
        "sha256": digest,
        "active_profile_id": None,
        "heldout_observations_used_for_lookup": 0,
    }


def _correction_store_digest(store: LanguageStore) -> str:
    rows = [
        dict(row)
        for row in store.connection.execute(
            "SELECT context_key, sample_count, mean_residual_json, mean_reaction_json, "
            "success_mean FROM transition_stats ORDER BY context_key"
        ).fetchall()
    ]
    canonical = [
        {
            "context_key": row["context_key"],
            "sample_count": int(row["sample_count"]),
            "mean_residual": json.loads(row["mean_residual_json"]),
            "mean_reaction": json.loads(row["mean_reaction_json"]),
            "success_mean": float(row["success_mean"]),
        }
        for row in rows
    ]
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def _norm(value: Any) -> str:
    return " ".join(str(value).lower().replace("’", "'").split())


def _semantic_atom(value: Any) -> str:
    """Canonical corpus atom shared by symbolic gold and parsed surfaces."""

    return "_".join(re.findall(r"[a-z0-9']+", str(value).lower().replace("’", "'")))


def _semantic_parse_exact(
    expected: Mapping[str, Any],
    result: TurnResult,
    *,
    runtime: ClankerLM,
    conversation: Mapping[str, Any],
    turn: Mapping[str, Any],
) -> bool:
    expected_predicate = _semantic_atom(expected.get("predicate"))
    expected_roles: Dict[str, str] = {}
    for role, value in dict(expected.get("roles", {})).items():
        normalized_role = _semantic_atom(role)
        expected_roles[normalized_role] = (
            _semantic_atom(value)
            if normalized_role == "requested_role"
            else _expected_local_id(value, conversation=conversation, turn=turn)
        )
    question = result.parse.question
    events = list(result.parse.events)
    if question is not None and question.event not in events:
        events.append(question.event)
    for event in events:
        if _semantic_atom(event.predicate) != expected_predicate:
            continue
        actual_roles = {
            _semantic_atom(role): _local_entity_id(
                reference,
                runtime=runtime,
                conversation=conversation,
                turn=turn,
            )
            for role, reference in event.arguments.items()
            if not reference.is_variable
        }
        if question is not None and _semantic_atom(question.event.predicate) == expected_predicate:
            if question.requested_role:
                actual_roles["requested_role"] = _semantic_atom(question.requested_role)
        if bool(expected.get("partial_roles")):
            if all(actual_roles.get(role) == value for role, value in expected_roles.items()):
                return True
        elif actual_roles == expected_roles:
            return True
    return False


def _answer_exact(
    expected: Mapping[str, Any],
    result: TurnResult,
    *,
    expected_status: str,
    expected_truth: str,
) -> bool:
    if result.contract.status.value != expected_status or result.contract.truth.value != expected_truth:
        return False
    event = result.contract.proposition
    if event is None and result.contract.question is not None:
        event = result.contract.question.event
    actual_predicate = _semantic_atom(event.predicate) if event is not None else None
    actual_values = sorted(_semantic_atom(item.surface or item.key) for item in result.contract.values)
    expected_values = sorted(_semantic_atom(item) for item in expected.get("values", []))
    expected_predicate = expected.get("predicate")
    if expected_predicate is not None and _semantic_atom(expected_predicate) != actual_predicate:
        return False
    if expected_values != actual_values:
        return False
    requested_roles = sorted(_semantic_atom(item) for item in expected.get("requested_roles", []))
    actual_roles = []
    if result.contract.question is not None and result.contract.question.requested_role:
        actual_roles.append(_semantic_atom(result.contract.question.requested_role))
    return requested_roles == actual_roles


def _local_entity_id(
    reference: Any,
    *,
    runtime: ClankerLM,
    conversation: Mapping[str, Any],
    turn: Mapping[str, Any],
) -> str:
    if reference.key == "user":
        return str(conversation["participant_bindings"][turn["speaker"]])
    if reference.key == "assistant":
        return str(conversation["participant_bindings"][turn["addressee"]])
    entity = runtime.memory.entities.get(reference.key)
    normalized = _norm(entity.canonical_name if entity is not None else reference.surface or reference.key)
    for participant, local_id in conversation["participant_bindings"].items():
        if _norm(participant) == normalized:
            return str(local_id)
    return f"entity:{normalized.replace(' ', '_')}"


def _expected_local_id(
    value: Any,
    *,
    conversation: Mapping[str, Any],
    turn: Mapping[str, Any],
) -> str:
    normalized = _norm(value)
    if normalized in {"user", "speaker"}:
        return str(conversation["participant_bindings"][turn["speaker"]])
    if normalized in {"clanker", "assistant", "addressee"}:
        return str(conversation["participant_bindings"][turn["addressee"]])
    for participant, local_id in conversation["participant_bindings"].items():
        if _norm(participant) == normalized:
            return str(local_id)
    if normalized.startswith("participant:") or normalized.startswith("entity:"):
        return normalized
    return f"entity:{normalized.replace(' ', '_')}"


def _entity_exact(
    expected: Mapping[str, Any],
    result: TurnResult,
    *,
    runtime: ClankerLM,
    conversation: Mapping[str, Any],
    turn: Mapping[str, Any],
) -> bool:
    if not expected:
        return True
    events = result.parse.events
    actual_by_role: Dict[str, set[str]] = defaultdict(set)
    actual_by_surface: Dict[str, set[str]] = defaultdict(set)
    for event in events:
        for role, reference in event.arguments.items():
            local_id = _local_entity_id(
                reference,
                runtime=runtime,
                conversation=conversation,
                turn=turn,
            )
            actual_by_role[role].add(local_id)
            if reference.surface:
                actual_by_surface[_norm(reference.surface)].add(local_id)
    for unresolved in result.parse.unresolved:
        candidates = set()
        for candidate in unresolved.candidates:
            reference = SimpleNamespace(key=candidate, surface=candidate)
            candidates.add(
                _local_entity_id(
                    reference,
                    runtime=runtime,
                    conversation=conversation,
                    turn=turn,
                )
            )
        actual_by_surface[_norm(unresolved.surface)].update(candidates)
    if set(expected) != {"roles", "mentions"}:
        return False
    expected_roles = expected["roles"]
    expected_mentions = expected["mentions"]
    if not isinstance(expected_roles, Mapping) or not isinstance(expected_mentions, Mapping):
        return False
    for role, value in expected_roles.items():
        values = value if isinstance(value, list) else [value]
        expected_ids = {
            _expected_local_id(item, conversation=conversation, turn=turn)
            for item in values
        }
        actual_ids = actual_by_role.get(str(role), set())
        if actual_ids != expected_ids:
            return False
    for mention, value in expected_mentions.items():
        values = value if isinstance(value, list) else [value]
        expected_ids = {
            _expected_local_id(item, conversation=conversation, turn=turn)
            for item in values
        }
        actual_ids = actual_by_surface.get(_norm(mention), set())
        if actual_ids != expected_ids:
            return False
    return True


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _memory_shape(runtime: ClankerLM) -> Dict[str, int]:
    memory = runtime.memory
    return {
        "entities": len(memory.entities),
        "events": len(memory.events),
        "relations": len(memory.relations),
        "serialized_bytes": len(_canonical_json(memory.to_dict()).encode("utf-8")),
    }


def _store_shape(store: LanguageStore) -> Dict[str, Any]:
    page_count = int(store.connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(store.connection.execute("PRAGMA page_size").fetchone()[0])
    return {
        "allocated_bytes": page_count * page_size,
        "freelist_pages": int(store.connection.execute("PRAGMA freelist_count").fetchone()[0]),
        "rows": store.schema_summary(),
    }


def _whole_interaction_trajectory_metrics(
    expected_before: Mapping[str, int],
    expected_next: Mapping[str, int],
    predicted: Mapping[str, int],
    target: Mapping[str, int],
) -> Dict[str, float]:
    """Score the declared g0-before-current to g1-after-next interaction."""

    metrics: Dict[str, float] = {}
    for axis in AXES:
        metrics[f"residual_{axis}"] = predicted[axis] - int(expected_next[axis])
        metrics[f"absolute_residual_{axis}"] = abs(metrics[f"residual_{axis}"])
        metrics[f"mae_{axis}"] = abs(predicted[axis] - int(expected_next[axis]))
        metrics[f"mae_normalized_{axis}"] = metrics[f"mae_{axis}"] / 255.0
        metrics[f"direction_{axis}"] = float(
            _sign(predicted[axis] - int(expected_before[axis]))
            == _sign(int(expected_next[axis]) - int(expected_before[axis]))
        )
    gold = AffectVector(**expected_next)
    before = AffectVector(**expected_before)
    predicted_vector = AffectVector(**predicted)
    target_vector = AffectVector(**target)
    metrics["next_state_distance"] = predicted_vector.distance(gold)
    metrics["target_attainment"] = max(
        0.0, min(1.0, 1.0 - gold.distance(target_vector) / 160.0)
    )
    metrics["target_distance_improvement"] = (
        before.distance(target_vector) - gold.distance(target_vector)
    )
    return metrics


def _score_turn(
    conversation: Mapping[str, Any],
    turn: Mapping[str, Any],
    result: TurnResult,
    *,
    runtime: ClankerLM,
    mode: str,
    latency_ms: float,
) -> Dict[str, Any]:
    annotation = turn["annotation"]
    expected_before = annotation["vadugwi_before"]
    expected_next = annotation["observed_next_state"]
    predicted = result.predicted_state.to_dict()
    target = result.target_state.to_dict()
    actual_act = result.parse.speech_act.value
    expected_act = str(annotation["dialogue_act"])
    if actual_act == "greet":
        actual_act = "social"
    record: Dict[str, Any] = {
        "conversation_id": conversation["conversation_id"],
        "turn_id": turn["turn_id"],
        "domain": conversation["domain"],
        "supervision_level": annotation["supervision_level"],
        "semantic_supervision_level": annotation["semantic_supervision_level"],
        "affect_supervision_level": annotation["affect_supervision_level"],
        "affect_scored": float(bool(annotation["affect_scored"])),
        "outcome_evidence": annotation["outcome_evidence"],
        "outcome": annotation["outcome"],
        "mode": mode,
        "dialogue_act_correct": float(actual_act == expected_act),
        "response_act_correct": float(result.gates.response_act == annotation["expected_response_act"]),
        "answer_status_correct": float(result.contract.status.value == annotation["expected_answer_status"]),
        "truth_correct": float(result.contract.truth.value == annotation["expected_truth"]),
        "latency_ms": latency_ms,
        "candidate_count": float(len(result.candidates)),
        "correction_applied": float(bool((result.trajectory or {}).get("transition_adjustment", {}).get("applied"))),
        "actual_status": result.contract.status.value,
        "expected_status": annotation["expected_answer_status"],
    }
    semantic = annotation["semantic"]
    if bool(semantic.get("scored")):
        record["semantic_parse_exact"] = float(
            _semantic_parse_exact(
                semantic,
                result,
                runtime=runtime,
                conversation=conversation,
                turn=turn,
            )
        )
    expected_answer = annotation["expected_answer"]
    if bool(expected_answer.get("scored")):
        record["semantic_answer_exact"] = float(
            _answer_exact(
                expected_answer,
                result,
                expected_status=str(annotation["expected_answer_status"]),
                expected_truth=str(annotation["expected_truth"]),
            )
        )
    refs = annotation.get("expected_entity_refs", {})
    if refs:
        record["entity_resolution_exact"] = float(
            _entity_exact(refs, result, runtime=runtime, conversation=conversation, turn=turn)
        )
    claimed = result.contract.status.value not in {
        "acknowledged", "unsupported", "unknown", "missing_reference",
        "ambiguous_reference", "multiple_matches", "lexical_probe",
    }
    if claimed and result.contract.certainty > 0:
        record["answer_confidence"] = result.contract.certainty / 255.0
        record["answer_correct"] = record.get(
            "semantic_answer_exact",
            float(record["answer_status_correct"] == 1.0 and record["truth_correct"] == 1.0),
        )
        record["brier"] = (record["answer_confidence"] - record["answer_correct"]) ** 2
    if expected_next is not None and bool(annotation["affect_scored"]):
        record.update(
            _whole_interaction_trajectory_metrics(
                expected_before, expected_next, predicted, target
            )
        )
    return record


def _evaluate_mode(
    conversations: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    correction_store: LanguageStore | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    resource_observations: List[Dict[str, Any]] = []
    construction_ms: List[float] = []
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix=f"clanker-{mode}-") as tmp:
        for conversation_index, conversation in enumerate(conversations):
            shared_runtime: ClankerLM | None = None
            shared_baseline: Dict[str, Any] | None = None
            if mode != "sentence_only":
                construction_started = time.perf_counter_ns()
                store = LanguageStore(Path(tmp) / f"conversation-{conversation_index:04d}.sqlite3")
                shared_runtime = _runtime(store, mode=mode, correction_store=correction_store)
                construction_ms.append((time.perf_counter_ns() - construction_started) / 1_000_000.0)
                shared_baseline = {"memory": _memory_shape(shared_runtime), "store": _store_shape(store)}
            try:
                for turn_index, turn in enumerate(conversation["turns"]):
                    runtime = shared_runtime
                    baseline = shared_baseline
                    if runtime is None:
                        construction_started = time.perf_counter_ns()
                        store = LanguageStore(Path(tmp) / f"sentence-{conversation_index:04d}-{turn_index:04d}.sqlite3")
                        runtime = _runtime(store, mode=mode, correction_store=correction_store)
                        construction_ms.append((time.perf_counter_ns() - construction_started) / 1_000_000.0)
                        baseline = {"memory": _memory_shape(runtime), "store": _store_shape(store)}
                    assert baseline is not None
                    try:
                        started = time.perf_counter_ns()
                        result = runtime.process(str(turn["text"]))
                        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                        record = _score_turn(
                            conversation,
                            turn,
                            result,
                            runtime=runtime,
                            mode=mode,
                            latency_ms=latency_ms,
                        )
                        records.append(record)
                        resource_observations.append({
                            "conversation_id": conversation["conversation_id"],
                            "turn_number": turn_index + 1,
                            "baseline": baseline,
                            "current": {"memory": _memory_shape(runtime), "store": _store_shape(runtime.store)},
                        })
                        for metric in (
                            "dialogue_act_correct", "response_act_correct", "answer_status_correct",
                            "truth_correct", "semantic_parse_exact", "semantic_answer_exact",
                            "entity_resolution_exact",
                        ):
                            if metric in record and record[metric] == 0.0:
                                failures.append({
                                    "conversation_id": conversation["conversation_id"],
                                    "turn_id": turn["turn_id"],
                                    "domain": conversation["domain"],
                                    "mode": mode,
                                    "category": metric,
                                })
                    except Exception as exc:  # report without exposing source text
                        failures.append({
                            "conversation_id": conversation["conversation_id"],
                            "turn_id": turn["turn_id"],
                            "domain": conversation["domain"],
                            "mode": mode,
                            "category": "execution_error",
                            "exception": type(exc).__name__,
                        })
                        break
                    finally:
                        if shared_runtime is None:
                            runtime.close()
            finally:
                if shared_runtime is not None:
                    shared_runtime.close()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resources = _aggregate_resources(resource_observations)
    resources["construction_latency_ms"] = _distribution(construction_ms)
    resources["tracemalloc_peak_bytes"] = traced_peak
    resources["process_lifetime_maxrss_peak"] = rss_peak
    resources["process_maxrss_units"] = "KiB on Linux; bytes on macOS"
    resources["process_maxrss_comparability"] = "observational_process_lifetime_peak_mode_order_dependent"
    return records, failures, resources


def _bootstrap_ci(records: Sequence[Mapping[str, Any]], metric: str, *, seed_material: str) -> List[float] | None:
    by_conversation: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for record in records:
        if metric in record:
            key = (str(record["domain"]), str(record["conversation_id"]))
            by_conversation[key].append(float(record[metric]))
    clusters_by_domain: Dict[str, List[List[float]]] = defaultdict(list)
    for (domain, _), values in sorted(by_conversation.items()):
        clusters_by_domain[domain].append(values)
    clusters = [values for domain_clusters in clusters_by_domain.values() for values in domain_clusters]
    if not clusters:
        return None
    if len(clusters) == 1:
        value = sum(clusters[0]) / len(clusters[0])
        return [value, value]
    digest = hashlib.sha256(f"{seed_material}|{metric}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    estimates = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_clusters = [
            domain_clusters[rng.randrange(len(domain_clusters))]
            for _, domain_clusters in sorted(clusters_by_domain.items())
            for _ in domain_clusters
        ]
        estimates.append(_pooled_cluster_mean(sampled_clusters))
    estimates.sort()
    return [estimates[round(0.025 * (len(estimates) - 1))], estimates[round(0.975 * (len(estimates) - 1))]]


def _pooled_cluster_mean(clusters: Sequence[Sequence[float]]) -> float:
    """Return the published turn-weighted point statistic for sampled clusters."""

    return sum(sum(values) for values in clusters) / sum(len(values) for values in clusters)


def _metric_summary(records: Sequence[Mapping[str, Any]], metric: str, *, seed_material: str) -> Dict[str, Any] | None:
    values = [float(record[metric]) for record in records if metric in record]
    if not values:
        return None
    return {
        "value": sum(values) / len(values),
        "n_turns": len(values),
        "n_conversations": len({record["conversation_id"] for record in records if metric in record}),
        "ci95_conversation_cluster_bootstrap": _bootstrap_ci(records, metric, seed_material=seed_material),
    }


def _classification(
    records: Sequence[Mapping[str, Any]],
    label: str,
    *,
    seed_material: str,
) -> Dict[str, Any]:
    tp = sum(record.get("expected_status") == label and record.get("actual_status") == label for record in records)
    fp = sum(record.get("expected_status") != label and record.get("actual_status") == label for record in records)
    fn = sum(record.get("expected_status") == label and record.get("actual_status") != label for record in records)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
        if precision is not None and recall is not None
        else None
    )
    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    clusters_by_domain: Dict[str, List[List[Mapping[str, Any]]]] = defaultdict(list)
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["domain"]), str(record["conversation_id"]))].append(record)
    for (domain, _), items in sorted(grouped.items()):
        clusters_by_domain[domain].append(items)
    if not grouped:
        return result
    rng = random.Random(int.from_bytes(
        hashlib.sha256(f"{seed_material}|classification|{label}".encode("utf-8")).digest()[:8],
        "big",
    ))
    estimates: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [
            item
            for _, clusters in sorted(clusters_by_domain.items())
            for _ in clusters
            for item in clusters[rng.randrange(len(clusters))]
        ]
        sample_tp = sum(item.get("expected_status") == label and item.get("actual_status") == label for item in sampled)
        sample_fp = sum(item.get("expected_status") != label and item.get("actual_status") == label for item in sampled)
        sample_fn = sum(item.get("expected_status") == label and item.get("actual_status") != label for item in sampled)
        sample_precision = sample_tp / (sample_tp + sample_fp) if sample_tp + sample_fp else None
        sample_recall = sample_tp / (sample_tp + sample_fn) if sample_tp + sample_fn else None
        sample_f1 = (
            2 * sample_precision * sample_recall / (sample_precision + sample_recall)
            if sample_precision is not None and sample_recall is not None
            and sample_precision + sample_recall
            else 0.0
            if sample_precision is not None and sample_recall is not None
            else None
        )
        if sample_precision is not None:
            estimates["precision"].append(sample_precision)
        if sample_recall is not None:
            estimates["recall"].append(sample_recall)
        if sample_f1 is not None:
            estimates["f1"].append(sample_f1)
    result["bootstrap_valid_draws"] = {
        name: len(values) for name, values in estimates.items()
    }
    result["ci95_conversation_cluster_bootstrap"] = {
        name: [
            sorted(values)[round(0.025 * (len(values) - 1))],
            sorted(values)[round(0.975 * (len(values) - 1))],
        ]
        for name, values in estimates.items()
        if values
    }
    return result


def _wilson(successes: int, total: int) -> List[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _ece(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    pairs = [
        (float(record["answer_confidence"]), float(record["answer_correct"]))
        for record in records
        if "answer_confidence" in record
    ]
    if not pairs:
        return None
    bins = []
    weighted = 0.0
    for index in range(10):
        low, high = index / 10.0, (index + 1) / 10.0
        members = [(confidence, correct) for confidence, correct in pairs if low <= confidence <= high and (index == 9 or confidence < high)]
        if not members:
            continue
        confidence = sum(item[0] for item in members) / len(members)
        accuracy = sum(item[1] for item in members) / len(members)
        weighted += len(members) / len(pairs) * abs(confidence - accuracy)
        bins.append({"lower": low, "upper": high, "n": len(members), "confidence": confidence, "accuracy": accuracy})
    return {"ece_10_bin": weighted, "n": len(pairs), "bins": bins}


def _drift(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    by_conversation: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if "next_state_distance" in record:
            by_conversation[str(record["conversation_id"])].append(record)
    distance_slopes: List[float] = []
    terminal_distances: List[float] = []
    terminal_bias: Dict[str, List[float]] = {axis: [] for axis in AXES}
    absolute_slopes: Dict[str, List[float]] = {axis: [] for axis in AXES}
    for items in by_conversation.values():
        if not items:
            continue
        terminal_distances.append(float(items[-1]["next_state_distance"]))
        for axis in AXES:
            terminal_bias[axis].append(float(items[-1][f"residual_{axis}"]))
        if len(items) < 2:
            continue
        xs = [index / (len(items) - 1) for index in range(len(items))]
        xbar = statistics.mean(xs)
        denominator = sum((x - xbar) ** 2 for x in xs)
        distance_values = [float(item["next_state_distance"]) for item in items]
        distance_mean = statistics.mean(distance_values)
        distance_slopes.append(
            sum((x - xbar) * (y - distance_mean) for x, y in zip(xs, distance_values)) / denominator
        )
        for axis in AXES:
            values = [float(item[f"absolute_residual_{axis}"]) for item in items]
            mean = statistics.mean(values)
            absolute_slopes[axis].append(
                sum((x - xbar) * (y - mean) for x, y in zip(xs, values)) / denominator
            )
    if not terminal_distances:
        return None
    return {
        "terminal_distance_mean": statistics.mean(terminal_distances),
        "distance_slope_mean": statistics.mean(distance_slopes) if distance_slopes else None,
        "terminal_signed_bias_by_axis": {
            axis: statistics.mean(values) if values else None
            for axis, values in terminal_bias.items()
        },
        "absolute_residual_slope_by_axis": {
            axis: statistics.mean(values) if values else None
            for axis, values in absolute_slopes.items()
        },
        "n_conversations": len(terminal_distances),
    }


def _summarize(records: Sequence[Mapping[str, Any]], *, seed_material: str) -> Dict[str, Any]:
    metric_names = [
        "dialogue_act_correct", "response_act_correct", "answer_status_correct", "truth_correct",
        "semantic_parse_exact", "semantic_answer_exact", "entity_resolution_exact", "brier", "target_attainment",
        "target_distance_improvement", "next_state_distance", "correction_applied", "candidate_count",
    ] + [f"mae_{axis}" for axis in AXES] + [f"mae_normalized_{axis}" for axis in AXES] + [f"direction_{axis}" for axis in AXES]
    metrics = {
        name: summary
        for name in metric_names
        if (summary := _metric_summary(records, name, seed_material=seed_material)) is not None
    }
    conversation_count = len({record["conversation_id"] for record in records})
    return {
        "turn_count": len(records),
        "conversation_count": conversation_count,
        "ci_stability": "stable" if conversation_count >= 20 else "unstable_fewer_than_20_conversations",
        "metrics": metrics,
        "unknown_classification": _classification(
            records, "unknown", seed_material=f"{seed_material}|unknown"
        ),
        "conflict_classification": _classification(
            records, "conflict", seed_material=f"{seed_material}|conflict"
        ),
        "uncertainty_calibration": _ece(records),
        "drift": _drift(records),
        "outcome_counts": dict(sorted(Counter(str(record["outcome"]) for record in records).items())),
        "confidence_interval_method": {
            "method": "percentile bootstrap",
            "cluster": "whole conversation",
            "overall_stratification": "domain",
            "draws": BOOTSTRAP_DRAWS,
            "fixed_seed_material": seed_material,
        },
    }


def _aggregate_resources(shapes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not shapes:
        return {}
    result: Dict[str, Any] = {"turn_samples": len(shapes)}
    for memory_key in ("entities", "events", "relations", "serialized_bytes"):
        values = [
            int(item["current"]["memory"][memory_key]) - int(item["baseline"]["memory"][memory_key])
            for item in shapes
        ]
        result[f"memory_{memory_key}_growth_mean"] = sum(values) / len(values)
        result[f"memory_{memory_key}_growth_max"] = max(values)
    allocated = [
        int(item["current"]["store"]["allocated_bytes"])
        - int(item["baseline"]["store"]["allocated_bytes"])
        for item in shapes
    ]
    result["sqlite_allocated_bytes_growth_mean"] = sum(allocated) / len(allocated)
    result["sqlite_allocated_bytes_growth_max"] = max(allocated)
    row_maxima: Dict[str, int] = {}
    for item in shapes:
        for table, count in item["current"]["store"]["rows"].items():
            growth = int(count) - int(item["baseline"]["store"]["rows"].get(table, 0))
            row_maxima[table] = max(row_maxima.get(table, 0), growth)
    result["sqlite_row_growth_maxima"] = dict(sorted(row_maxima.items()))
    by_conversation: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in shapes:
        by_conversation[str(item["conversation_id"])].append(item)
    result["growth_slopes_per_turn"] = {
        "memory_serialized_bytes": _mean_growth_slope(
            by_conversation,
            lambda item: int(item["current"]["memory"]["serialized_bytes"])
            - int(item["baseline"]["memory"]["serialized_bytes"]),
        ),
        "sqlite_allocated_bytes": _mean_growth_slope(
            by_conversation,
            lambda item: int(item["current"]["store"]["allocated_bytes"])
            - int(item["baseline"]["store"]["allocated_bytes"]),
        ),
    }
    return result


def _mean_growth_slope(
    by_conversation: Mapping[str, Sequence[Mapping[str, Any]]],
    value: Any,
) -> float | None:
    slopes: List[float] = []
    for items in by_conversation.values():
        ordered = sorted(items, key=lambda item: int(item["turn_number"]))
        if len(ordered) < 2:
            continue
        xs = [float(item["turn_number"]) for item in ordered]
        ys = [float(value(item)) for item in ordered]
        xbar, ybar = statistics.mean(xs), statistics.mean(ys)
        denominator = sum((x - xbar) ** 2 for x in xs)
        slopes.append(sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator)
    return statistics.mean(slopes) if slopes else None


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {}
    percentile = lambda q: ordered[round(q * (len(ordered) - 1))]
    return {
        "n": len(ordered),
        "mean_ms": statistics.mean(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": max(ordered),
        "observational_nondeterministic": True,
    }


def _latency_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = sorted(float(record["latency_ms"]) for record in records)
    if not values:
        return {}
    distribution = _distribution(values)
    total_seconds = sum(values) / 1000.0
    return {
        **distribution,
        "turns_per_second": len(values) / total_seconds if total_seconds else None,
    }


def _paired_mode_differences(
    mode_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed_base: str,
) -> Dict[str, Any]:
    comparisons = (
        ("stateful_minus_sentence_only", "stateful", "sentence_only"),
        ("transition_corrected_minus_stateful", "transition_corrected", "stateful"),
    )
    metric_names = [
        "dialogue_act_correct", "response_act_correct", "answer_status_correct",
        "truth_correct", "semantic_parse_exact", "semantic_answer_exact",
        "entity_resolution_exact", "brier", "target_attainment",
        "target_distance_improvement", "next_state_distance", "correction_applied",
    ] + [f"mae_{axis}" for axis in AXES] + [f"mae_normalized_{axis}" for axis in AXES] + [f"direction_{axis}" for axis in AXES]
    output: Dict[str, Any] = {}
    for label, candidate_mode, reference_mode in comparisons:
        candidate = {
            (str(item["conversation_id"]), str(item["turn_id"])): item
            for item in mode_records[candidate_mode]
        }
        reference = {
            (str(item["conversation_id"]), str(item["turn_id"])): item
            for item in mode_records[reference_mode]
        }
        paired_keys = sorted(set(candidate) & set(reference))
        metric_output: Dict[str, Any] = {}
        for metric in metric_names:
            differences = [
                {
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                    "domain": candidate[(conversation_id, turn_id)]["domain"],
                    "difference": float(candidate[(conversation_id, turn_id)][metric])
                    - float(reference[(conversation_id, turn_id)][metric]),
                }
                for conversation_id, turn_id in paired_keys
                if metric in candidate[(conversation_id, turn_id)]
                and metric in reference[(conversation_id, turn_id)]
            ]
            summary = _metric_summary(
                differences,
                "difference",
                seed_material=f"{seed_base}|paired|{label}|{metric}",
            )
            if summary is not None:
                metric_output[metric] = summary
        output[label] = {
            "candidate_mode": candidate_mode,
            "reference_mode": reference_mode,
            "paired_turn_count": len(paired_keys),
            "pairing": "identical conversation/turn IDs; cluster samples shared within each delta",
            "metrics": metric_output,
        }
    return output


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _capture_provenance(manifest: Mapping[str, Any]) -> Dict[str, str]:
    evaluator_sha256 = _sha256_file(Path(__file__))
    compiler_sha256 = _sha256_file(Path(__file__).with_name("corpus.py"))
    if evaluator_sha256 != manifest["evaluator_sha256"]:
        raise CorpusIntegrityError("loaded evaluator bytes do not match the frozen manifest")
    if compiler_sha256 != manifest["compiler_sha256"]:
        raise CorpusIntegrityError("loaded compiler bytes do not match the frozen manifest")
    schema_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(Path(__file__).with_name("schema").glob("*.json"))
    }
    if schema_hashes != manifest["schema_sha256"]:
        raise CorpusIntegrityError("evaluation schema bytes do not match the frozen manifest")
    assert_production_tree(str(manifest["production_code_sha256"]))
    measured_paths = (
        "clanker_lm", "engine", "clanker_engine.py",
        "evaluation/conversations/runner.py", "evaluation/conversations/corpus.py",
        "evaluation/conversations/schema", "evaluation/conversations/sources",
        "evaluation/conversations/data",
    )
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *measured_paths],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise CorpusIntegrityError("cannot establish evaluation git provenance") from exc
    if dirty:
        raise CorpusIntegrityError("measured evaluator/corpus/production tree is dirty")
    commit = _git_commit()
    if commit == "unknown":
        raise CorpusIntegrityError("cannot establish evaluation commit")
    return {
        "evaluation_commit": commit,
        "evaluator_sha256": evaluator_sha256,
        "compiler_sha256": compiler_sha256,
        "schema_sha256": hashlib.sha256(_canonical_json(schema_hashes).encode("utf-8")).hexdigest(),
        "production_code_sha256": str(manifest["production_code_sha256"]),
    }


def _assert_provenance_unchanged(
    initial: Mapping[str, str],
    manifest: Mapping[str, Any],
    *,
    current: Mapping[str, str] | None = None,
) -> None:
    observed = dict(current) if current is not None else _capture_provenance(manifest)
    if observed != dict(initial):
        raise CorpusIntegrityError("evaluation provenance changed while the run was in progress")


_FORBIDDEN_ARTIFACT_KEYS = {
    "text", "raw_text", "source_text", "input_text", "response", "generated_response",
    "message", "messages", "event", "events", "parse", "candidates", "turn_payload",
    "body", "content", "payload", "output", "reply", "utterance",
}


def _validate_aggregate_artifacts(
    report: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    conversations: Sequence[Mapping[str, Any]],
) -> None:
    """Fail if aggregate artifacts contain raw conversational payloads."""

    heldout_texts = [
        str(turn["text"])
        for conversation in conversations
        for turn in conversation["turns"]
    ]

    if "report_schema_version" in report:
        expected_report_keys = {
            "report_schema_version", "corpus_version", "split", "corpus_sha256",
            "corpus_root_sha256", "compiler_sha256", "evaluator_sha256", "schema_sha256",
            "production_code_commit", "production_code_sha256", "evaluation_commit",
            "backend", "clock", "timezone", "modes", "development_correction_bundle",
            "paired_mode_differences", "semantic_fingerprint", "metric_supervision",
            "failure_count", "failure_counts", "environment", "limitations",
        }
        if set(report) != expected_report_keys:
            raise CorpusIntegrityError("aggregate report top-level schema is not exact")
        if set(report["modes"]) != set(MODES):
            raise CorpusIntegrityError("aggregate report mode schema is not exact")
        mode_keys = {
            "overall", "by_domain", "by_outcome", "by_supervision_level",
            "latency", "resource_growth", "execution_errors",
        }
        summary_keys = {
            "turn_count", "conversation_count", "ci_stability", "metrics",
            "unknown_classification", "conflict_classification", "uncertainty_calibration",
            "drift", "outcome_counts", "confidence_interval_method",
        }
        metric_keys = {
            "value", "n_turns", "n_conversations", "ci95_conversation_cluster_bootstrap",
        }
        for mode in MODES:
            mode_report = report["modes"][mode]
            if set(mode_report) != mode_keys:
                raise CorpusIntegrityError(f"aggregate report {mode} schema is not exact")
            summaries = [mode_report["overall"]]
            for stratum in ("by_domain", "by_outcome", "by_supervision_level"):
                summaries.extend(mode_report[stratum].values())
            for summary in summaries:
                if set(summary) != summary_keys:
                    raise CorpusIntegrityError("aggregate summary schema is not exact")
                if any(set(metric) != metric_keys for metric in summary["metrics"].values()):
                    raise CorpusIntegrityError("aggregate metric schema is not exact")
        failure_keys = {"conversation_id", "turn_id", "domain", "mode", "category"}
        for failure in failures:
            if not isinstance(failure, Mapping) or not failure_keys <= set(failure):
                raise CorpusIntegrityError("failure ledger row is malformed")
            if set(failure) - failure_keys - {"exception"}:
                raise CorpusIntegrityError("failure ledger row contains an unexpected field")

    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized_key = str(key).lower()
                if normalized_key in _FORBIDDEN_ARTIFACT_KEYS or normalized_key.endswith("_text"):
                    raise CorpusIntegrityError(f"aggregate artifact contains forbidden payload key at {location}.{key}")
                walk(str(key), f"{location}.<key>")
                walk(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{location}[{index}]")
        elif isinstance(value, str):
            normalized_value = _norm(value)
            for raw_text in heldout_texts:
                normalized_text = _norm(raw_text)
                if normalized_value == normalized_text or (
                    len(normalized_text) >= 16 and normalized_text in normalized_value
                ):
                    raise CorpusIntegrityError(
                        f"aggregate artifact contains held-out turn content at {location}"
                    )

    walk(report, "report")
    walk(list(failures), "failures")


def _atomic_replace(source: Path, target: Path) -> None:
    source.replace(target)


def _publish_artifacts(
    report: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    failures_path: Path | None,
    provenance_check: Any,
) -> None:
    failure_target = failures_path or output_path.with_name(f"{output_path.stem}_failures.jsonl")
    checksum_path = output_path.with_suffix(".sha256")
    targets = [output_path.resolve(), failure_target.resolve(), checksum_path.resolve()]
    if len(set(targets)) != len(targets):
        raise CorpusIntegrityError("report, failures, and checksum targets must be distinct")
    if failure_target.parent.resolve() != output_path.parent.resolve():
        raise CorpusIntegrityError("report, failures, and checksum must share one publish directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    failure_bytes = "".join(_canonical_json(item) + "\n" for item in failures).encode("utf-8")
    checksum_bytes = (
        f"{hashlib.sha256(report_bytes).hexdigest()}  {output_path.name}\n"
        f"{hashlib.sha256(failure_bytes).hexdigest()}  {failure_target.name}\n"
    ).encode("ascii")
    with tempfile.TemporaryDirectory(prefix=".conversation-report-", dir=output_path.parent) as staging_name:
        staging = Path(staging_name)
        staged = {
            output_path: report_bytes,
            failure_target: failure_bytes,
            checksum_path: checksum_bytes,
        }
        for target, payload in staged.items():
            (staging / target.name).write_bytes(payload)
        backups: Dict[Path, Path | None] = {}
        for index, target in enumerate(staged):
            if target.exists():
                backup = staging / f".backup-{index}"
                backup.write_bytes(target.read_bytes())
                backups[target] = backup
            else:
                backups[target] = None
        provenance_check()
        published: List[Path] = []
        try:
            for target in (output_path, failure_target, checksum_path):
                _atomic_replace(staging / target.name, target)
                published.append(target)
        except Exception as publish_error:
            rollback_errors: List[Exception] = []
            for target in reversed(published):
                try:
                    backup = backups[target]
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        backup.replace(target)
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise RuntimeError(
                    f"artifact publication failed and {len(rollback_errors)} rollback action(s) failed"
                ) from publish_error
            raise


def _enforce_zero_execution_errors(failures: Sequence[Mapping[str, Any]]) -> None:
    execution_errors = [item for item in failures if item.get("category") == "execution_error"]
    if execution_errors:
        first = execution_errors[0]
        raise RuntimeError(
            "conversation evaluation aborted after "
            f"{len(execution_errors)} execution error(s); first at "
            f"{first.get('conversation_id')} / {first.get('turn_id')} "
            f"({first.get('exception', 'unknown exception')})"
        )


def run_evaluation(
    *,
    split: str = "development",
    output_path: Path | None = None,
    failures_path: Path | None = None,
    manifest_path: Path = MANIFEST_PATH,
) -> Dict[str, Any]:
    """Run all three modes and return an aggregate-only, text-free report."""

    manifest = load_manifest(manifest_path)
    run_provenance = _capture_provenance(manifest)
    conversations = load_split(split, purpose="evaluation", manifest_path=manifest_path)
    development = load_split("development", purpose="development", manifest_path=manifest_path)
    correction_store, correction_bundle = _build_development_corrections(development)
    correction_bundle["source_corpus_sha256"] = manifest["splits"]["development"]["sha256"]
    correction_bundle["production_runtime_commit"] = manifest["baseline_code_commit"]
    correction_digest_before = _correction_store_digest(correction_store)
    if correction_digest_before != correction_bundle["sha256"]:
        correction_store.close()
        raise RuntimeError("development correction store digest disagrees with its canonical bundle")
    all_failures: List[Dict[str, Any]] = []
    mode_records: Dict[str, List[Dict[str, Any]]] = {}
    mode_reports: Dict[str, Any] = {}
    seed_base = f"{manifest['splits'][split]['sha256']}|metric-v{METRIC_SCHEMA_VERSION}"
    try:
        for mode in MODES:
            records, failures, resources = _evaluate_mode(
                conversations,
                mode=mode,
                correction_store=correction_store if mode == "transition_corrected" else None,
            )
            all_failures.extend(failures)
            mode_records[mode] = records
            domains = sorted({str(record["domain"]) for record in records})
            outcomes = sorted({str(record["outcome"]) for record in records})
            supervision_levels = sorted({str(record["supervision_level"]) for record in records})
            mode_reports[mode] = {
                "overall": _summarize(records, seed_material=f"{seed_base}|{mode}|overall"),
                "by_domain": {
                    domain: _summarize(
                        [record for record in records if record["domain"] == domain],
                        seed_material=f"{seed_base}|{mode}|domain|{domain}",
                    )
                    for domain in domains
                },
                "by_outcome": {
                    outcome: _summarize(
                        [record for record in records if record["outcome"] == outcome],
                        seed_material=f"{seed_base}|{mode}|outcome|{outcome}",
                    )
                    for outcome in outcomes
                },
                "by_supervision_level": {
                    level: _summarize(
                        [record for record in records if record["supervision_level"] == level],
                        seed_material=f"{seed_base}|{mode}|supervision|{level}",
                    )
                    for level in supervision_levels
                },
                "latency": _latency_summary(records),
                "resource_growth": resources,
                "execution_errors": sum(item["category"] == "execution_error" for item in failures),
            }
        correction_digest_after = _correction_store_digest(correction_store)
        correction_bundle["sha256_after_evaluation"] = correction_digest_after
        correction_bundle["lookup_store_unchanged"] = correction_digest_after == correction_digest_before
        if not correction_bundle["lookup_store_unchanged"]:
            raise RuntimeError("held-out evaluation mutated the development correction lookup")
    finally:
        correction_store.close()

    _enforce_zero_execution_errors(all_failures)
    paired_differences = _paired_mode_differences(mode_records, seed_base=seed_base)
    semantic_payload = {
        "modes": {
        mode: {
            "overall": details["overall"],
            "by_domain": details["by_domain"],
            "by_outcome": details["by_outcome"],
            "by_supervision_level": details["by_supervision_level"],
            "execution_errors": details["execution_errors"],
        }
        for mode, details in mode_reports.items()
        },
        "paired_mode_differences": paired_differences,
    }
    semantic_fingerprint = hashlib.sha256(_canonical_json(semantic_payload).encode("utf-8")).hexdigest()
    report: Dict[str, Any] = {
        "report_schema_version": METRIC_SCHEMA_VERSION,
        "corpus_version": manifest["corpus_version"],
        "split": split,
        "corpus_sha256": manifest["splits"][split]["sha256"],
        "corpus_root_sha256": manifest["corpus_root_sha256"],
        "compiler_sha256": manifest["compiler_sha256"],
        "evaluator_sha256": manifest["evaluator_sha256"],
        "schema_sha256": manifest["schema_sha256"],
        "production_code_commit": manifest["baseline_code_commit"],
        "production_code_sha256": manifest["production_code_sha256"],
        "evaluation_commit": run_provenance["evaluation_commit"],
        "backend": "clanker-v8",
        "clock": FIXED_INSTANT.isoformat(),
        "timezone": "UTC",
        "modes": mode_reports,
        "development_correction_bundle": correction_bundle,
        "paired_mode_differences": paired_differences,
        "semantic_fingerprint": semantic_fingerprint,
        "metric_supervision": {
            "semantic_and_entity": "turn semantic_supervision_level",
            "affect_and_trajectory": "weak_rule_v1 only; structural_only sources excluded",
            "outcome": "weak or counterfactual stratum only; no outcome accuracy is claimed",
        },
        "failure_count": len(all_failures),
        "failure_counts": dict(sorted(Counter(item["category"] for item in all_failures).items())),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        },
        "limitations": [
            "Literary and archival next-state/outcome labels are weak supervision, not causal Clanker exposure.",
            "ClankerLM.process accepts no speaker/addressee; participant-aware scores expose that interface limit.",
            "Latency and resource measurements are observational and excluded from semantic_fingerprint.",
            "Process max-RSS is a process-lifetime peak and is mode-order dependent, not a paired mode comparison.",
            "No categorical outcome prediction API exists; outcomes stratify metrics but outcome accuracy is not claimed.",
        ],
    }
    _validate_aggregate_artifacts(report, all_failures, conversations)
    _assert_provenance_unchanged(run_provenance, manifest)
    if output_path is not None:
        _publish_artifacts(
            report,
            all_failures,
            output_path=output_path,
            failures_path=failures_path,
            provenance_check=lambda: _assert_provenance_unchanged(run_provenance, manifest),
        )
    elif failures_path is not None:
        raise CorpusIntegrityError("failures_path requires output_path for transactional publication")
    return report
