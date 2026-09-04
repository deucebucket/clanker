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
    ALLOWED_DOMAINS,
    ALLOWED_OUTCOMES,
    MANIFEST_PATH,
    REPO_ROOT,
    CorpusIntegrityError,
    _canonical_json,
    _sha256_file,
    assert_production_tree,
    load_manifest,
    load_split,
    selected_manifest_path,
)


METRIC_SCHEMA_VERSION = 1
MODES = ("sentence_only", "stateful", "transition_corrected")
BOOTSTRAP_DRAWS = 10_000
FIXED_INSTANT = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
PAIRING_DESCRIPTION = "identical conversation/turn IDs; cluster samples shared within each delta"
REPORT_LIMITATIONS = (
    "Literary and archival next-state/outcome labels are weak supervision, not causal Clanker exposure.",
    "ClankerLM.process accepts no speaker/addressee; participant-aware scores expose that interface limit.",
    "Latency and resource measurements are observational and excluded from semantic_fingerprint.",
    "Process max-RSS is a process-lifetime peak and is mode-order dependent, not a paired mode comparison.",
    "No categorical outcome prediction API exists; outcomes stratify metrics but outcome accuracy is not claimed.",
    "Static exclusion cannot govern arbitrary operator-supplied text or paths; official replay/promotion enforcement is tracked by issue 110.",
)
METRIC_NAMES = frozenset({
    "dialogue_act_correct", "response_act_correct", "answer_status_correct", "truth_correct",
    "semantic_parse_exact", "semantic_answer_exact", "entity_resolution_exact", "candidate_count",
    "brier", "target_attainment", "target_distance_improvement", "next_state_distance",
    "correction_applied",
} | {f"mae_{axis}" for axis in AXES} | {f"mae_normalized_{axis}" for axis in AXES}
  | {f"direction_{axis}" for axis in AXES})
FAILURE_CATEGORIES = frozenset({
    "dialogue_act_correct", "response_act_correct", "answer_status_correct", "truth_correct",
    "semantic_parse_exact", "semantic_answer_exact", "entity_resolution_exact",
})
SUPERVISION_LEVELS = frozenset({"weak", "weak_rule_v1", "gold_structural"})
SUMMARY_REQUIRED_METRICS = frozenset({
    "dialogue_act_correct", "response_act_correct", "answer_status_correct", "truth_correct",
    "candidate_count", "correction_applied",
})
PAIRED_REQUIRED_METRICS = frozenset({
    "dialogue_act_correct", "response_act_correct", "answer_status_correct", "truth_correct",
    "correction_applied",
})
PLATFORM_IMPLEMENTATIONS = frozenset({"cpython", "pypy", "graalpy"})
PLATFORM_SYSTEMS = frozenset({"linux", "darwin", "windows", "freebsd", "openbsd"})
PLATFORM_MACHINES = frozenset({
    "x86_64", "amd64", "aarch64", "arm64", "i386", "i686", "ppc64le", "s390x",
    "riscv64", "unknown",
})
SQLITE_ROW_TABLES = frozenset({
    "atoms", "grammar_rules", "gate_rules", "learned_terms", "learned_senses",
    "lexical_evidence", "resolver_observations", "trajectory_turns", "transition_stats",
    "corpus_profiles", "trajectory_chunks", "template_tables",
})


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
    if question is not None and not any(event is question.event for event in events):
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
        if question is not None and event is question.event:
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
    for participant, local_id in conversation["participant_bindings"].items():
        if _norm(participant) == normalized:
            return str(local_id)
    if normalized == "speaker":
        return str(conversation["participant_bindings"][turn["speaker"]])
    if normalized == "addressee":
        return str(conversation["participant_bindings"][turn["addressee"]])
    if normalized == "assistant":
        return "participant:clanker"
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
    if bool(expected_answer.get("scored")):
        record["answer_confidence"] = result.contract.certainty / 255.0
        record["answer_correct"] = record["semantic_answer_exact"]
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
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else None
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["domain"]), str(record["conversation_id"]))].append(record)
    clusters = []
    for (domain, conversation_id), items in sorted(grouped.items()):
        cluster_tp = sum(
            item.get("expected_status") == label and item.get("actual_status") == label
            for item in items
        )
        cluster_fp = sum(
            item.get("expected_status") != label and item.get("actual_status") == label
            for item in items
        )
        cluster_fn = sum(
            item.get("expected_status") == label and item.get("actual_status") != label
            for item in items
        )
        clusters.append({
            "domain": domain,
            "conversation_id": conversation_id,
            "tp": cluster_tp,
            "fp": cluster_fp,
            "fn": cluster_fn,
            "tn": len(items) - cluster_tp - cluster_fp - cluster_fn,
        })
    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clusters": clusters,
    }
    clusters_by_domain: Dict[str, List[List[Mapping[str, Any]]]] = defaultdict(list)
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
        sample_f1_denominator = 2 * sample_tp + sample_fp + sample_fn
        sample_f1 = 2 * sample_tp / sample_f1_denominator if sample_f1_denominator else None
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
        bins.append({
            "lower": low,
            "upper": high,
            "n": len(members),
            "confidence_sum": sum(item[0] for item in members),
            "correct_sum": sum(item[1] for item in members),
            "squared_error_sum": sum((item[0] - item[1]) ** 2 for item in members),
            "confidence": confidence,
            "accuracy": accuracy,
        })
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
            "pairing": PAIRING_DESCRIPTION,
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


def _require_artifact_keys(value: Any, expected: set[str], location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CorpusIntegrityError(f"{location} schema is not exact")
    return value


def _is_artifact_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_metric_summary(
    value: Any,
    location: str,
    *,
    metric_name: str,
    paired: bool = False,
    max_turns: int | None = None,
    max_conversations: int | None = None,
) -> None:
    metric = _require_artifact_keys(
        value,
        {"value", "n_turns", "n_conversations", "ci95_conversation_cluster_bootstrap"},
        location,
    )
    if not _is_artifact_number(metric["value"]):
        raise CorpusIntegrityError(f"{location}.value must be finite numeric")
    if (
        type(metric["n_turns"]) is not int
        or type(metric["n_conversations"]) is not int
        or metric["n_turns"] < 0
        or metric["n_conversations"] < 0
        or metric["n_turns"] == 0
        or metric["n_conversations"] == 0
        or metric["n_conversations"] > metric["n_turns"]
        or (max_turns is not None and metric["n_turns"] > max_turns)
        or (max_conversations is not None and metric["n_conversations"] > max_conversations)
    ):
        raise CorpusIntegrityError(f"{location} counts are invalid")
    interval = metric["ci95_conversation_cluster_bootstrap"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(_is_artifact_number(item) for item in interval)
        or interval[0] > interval[1]
    ):
        raise CorpusIntegrityError(f"{location} confidence interval is invalid")
    max_distance = math.sqrt(len(AXES)) * 255.0
    unit_metrics = {
        "dialogue_act_correct", "response_act_correct", "answer_status_correct", "truth_correct",
        "semantic_parse_exact", "semantic_answer_exact", "entity_resolution_exact", "brier",
        "target_attainment", "correction_applied",
    } | {f"mae_normalized_{axis}" for axis in AXES} | {f"direction_{axis}" for axis in AXES}
    if metric_name in unit_metrics:
        bounds = (-1.0, 1.0) if paired else (0.0, 1.0)
    elif metric_name.startswith("mae_"):
        bounds = (-255.0, 255.0) if paired else (0.0, 255.0)
    elif metric_name == "next_state_distance":
        bounds = (-max_distance, max_distance) if paired else (0.0, max_distance)
    elif metric_name == "target_distance_improvement":
        bounds = (-2.0 * max_distance, 2.0 * max_distance) if paired else (-max_distance, max_distance)
    elif metric_name == "candidate_count":
        bounds = (0.0, math.inf)
    else:
        raise CorpusIntegrityError(f"{location} metric domain is not declared")
    if any(not bounds[0] <= point <= bounds[1] for point in [metric["value"], *interval]):
        raise CorpusIntegrityError(f"{location} value is outside its declared metric domain")


def _validate_classification(
    value: Any,
    location: str,
    *,
    max_turns: int,
    label: str,
    seed_material: str,
) -> None:
    item = _require_artifact_keys(
        value,
        {
            "tp", "fp", "fn", "precision", "recall", "f1", "clusters", "bootstrap_valid_draws",
            "ci95_conversation_cluster_bootstrap",
        },
        location,
    )
    if any(type(item[field]) is not int or item[field] < 0 for field in ("tp", "fp", "fn")):
        raise CorpusIntegrityError(f"{location} counts must be nonnegative integers")
    if item["tp"] + item["fp"] + item["fn"] > max_turns:
        raise CorpusIntegrityError(f"{location} counts exceed the summary turn count")
    if not isinstance(item["clusters"], list):
        raise CorpusIntegrityError(f"{location} clusters must be an array")
    cluster_rows: List[Dict[str, Any]] = []
    cluster_ids: set[tuple[str, str]] = set()
    for index, raw_cluster in enumerate(item["clusters"]):
        cluster = _require_artifact_keys(
            raw_cluster,
            {"domain", "conversation_id", "tp", "fp", "fn", "tn"},
            f"{location}.clusters[{index}]",
        )
        identity = (cluster["domain"], cluster["conversation_id"])
        if (
            not all(isinstance(part, str) and part for part in identity)
            or identity in cluster_ids
            or any(
                type(cluster[field]) is not int or cluster[field] < 0
                for field in ("tp", "fp", "fn", "tn")
            )
        ):
            raise CorpusIntegrityError(f"{location} cluster is invalid")
        cluster_ids.add(identity)
        for (expected_status, actual_status), count in (
            ((label, label), cluster["tp"]),
            (("other", label), cluster["fp"]),
            ((label, "other"), cluster["fn"]),
            (("other", "other"), cluster["tn"]),
        ):
            for offset in range(count):
                cluster_rows.append({
                    "domain": cluster["domain"],
                    "conversation_id": cluster["conversation_id"],
                    "turn_id": f"{index}:{expected_status}:{actual_status}:{offset}",
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                })
    if len(cluster_rows) != max_turns:
        raise CorpusIntegrityError(f"{location} cluster population is inconsistent")
    rebuilt = _classification(cluster_rows, label, seed_material=seed_material)
    if dict(item) != rebuilt:
        raise CorpusIntegrityError(
            f"{location} disagrees with deterministic cluster statistics"
        )
    if any(
        item[field] is not None and not _is_artifact_number(item[field])
        for field in ("precision", "recall", "f1")
    ):
        raise CorpusIntegrityError(f"{location} point estimate is invalid")
    expected_points = {
        "precision": item["tp"] / (item["tp"] + item["fp"])
        if item["tp"] + item["fp"] else None,
        "recall": item["tp"] / (item["tp"] + item["fn"])
        if item["tp"] + item["fn"] else None,
        "f1": 2 * item["tp"] / (2 * item["tp"] + item["fp"] + item["fn"])
        if 2 * item["tp"] + item["fp"] + item["fn"] else None,
    }
    if any(
        (item[field] is None) != (expected is None)
        or (
            expected is not None
            and not math.isclose(float(item[field]), expected, rel_tol=0.0, abs_tol=1e-15)
        )
        for field, expected in expected_points.items()
    ):
        raise CorpusIntegrityError(f"{location} point estimates disagree with counts")
    valid = _require_artifact_keys(
        item["bootstrap_valid_draws"], {"precision", "recall", "f1"}, f"{location}.valid_draws"
    )
    if any(type(count) is not int or not 0 <= count <= BOOTSTRAP_DRAWS for count in valid.values()):
        raise CorpusIntegrityError(f"{location} valid-draw count is invalid")
    intervals = item["ci95_conversation_cluster_bootstrap"]
    expected_intervals = {field for field, count in valid.items() if count > 0}
    if not isinstance(intervals, Mapping) or set(intervals) != expected_intervals:
        raise CorpusIntegrityError(f"{location} interval schema is invalid")
    if any(
        not isinstance(interval, list) or len(interval) != 2
        or not all(_is_artifact_number(point) and 0.0 <= point <= 1.0 for point in interval)
        or interval[0] > interval[1]
        for interval in intervals.values()
    ):
        raise CorpusIntegrityError(f"{location} interval is invalid")


def _validate_summary(value: Any, location: str, *, expected_seed: str) -> None:
    summary = _require_artifact_keys(
        value,
        {
            "turn_count", "conversation_count", "ci_stability", "metrics",
            "unknown_classification", "conflict_classification", "uncertainty_calibration",
            "drift", "outcome_counts", "confidence_interval_method",
        },
        location,
    )
    if (
        type(summary["turn_count"]) is not int
        or type(summary["conversation_count"]) is not int
        or summary["turn_count"] < 0
        or summary["conversation_count"] < 0
        or summary["conversation_count"] > summary["turn_count"]
    ):
        raise CorpusIntegrityError(f"{location} counts are invalid")
    expected_stability = (
        "stable" if summary["conversation_count"] >= 20
        else "unstable_fewer_than_20_conversations"
    )
    if summary["ci_stability"] != expected_stability:
        raise CorpusIntegrityError(f"{location}.ci_stability is invalid")
    if not isinstance(summary["metrics"], Mapping):
        raise CorpusIntegrityError(f"{location}.metrics must be an object")
    if not SUMMARY_REQUIRED_METRICS <= set(summary["metrics"]):
        raise CorpusIntegrityError(f"{location}.metrics is missing required measurements")
    for metric_name, metric in summary["metrics"].items():
        if metric_name not in METRIC_NAMES:
            raise CorpusIntegrityError(f"{location} metric name is invalid")
        _validate_metric_summary(
            metric,
            f"{location}.metrics.{metric_name}",
            metric_name=metric_name,
            max_turns=summary["turn_count"],
            max_conversations=summary["conversation_count"],
        )
    _validate_classification(
        summary["unknown_classification"], f"{location}.unknown",
        max_turns=summary["turn_count"], label="unknown",
        seed_material=f"{expected_seed}|unknown",
    )
    _validate_classification(
        summary["conflict_classification"], f"{location}.conflict",
        max_turns=summary["turn_count"], label="conflict",
        seed_material=f"{expected_seed}|conflict",
    )
    calibration = summary["uncertainty_calibration"]
    brier = summary["metrics"].get("brier")
    if (calibration is None) != (brier is None):
        raise CorpusIntegrityError(f"{location}.calibration availability disagrees with Brier")
    if calibration is not None:
        calibration = _require_artifact_keys(
            calibration, {"ece_10_bin", "n", "bins"}, f"{location}.calibration"
        )
        if (
            not _is_artifact_number(calibration["ece_10_bin"])
            or type(calibration["n"]) is not int
            or not 1 <= calibration["n"] <= summary["turn_count"]
            or calibration["n"] != brier["n_turns"]
            or not 0.0 <= calibration["ece_10_bin"] <= 1.0
        ):
            raise CorpusIntegrityError(f"{location}.calibration has invalid values")
        if not isinstance(calibration["bins"], list):
            raise CorpusIntegrityError(f"{location}.calibration bins must be an array")
        for index, bin_item in enumerate(calibration["bins"]):
            bin_item = _require_artifact_keys(
                bin_item,
                {
                    "lower", "upper", "n", "confidence_sum", "correct_sum",
                    "squared_error_sum", "confidence", "accuracy",
                },
                f"{location}.calibration.bins[{index}]",
            )
            if type(bin_item["n"]) is not int or bin_item["n"] <= 0 or not all(
                _is_artifact_number(bin_item[field])
                for field in (
                    "lower", "upper", "confidence_sum", "correct_sum",
                    "squared_error_sum", "confidence", "accuracy",
                )
            ) or not (
                0.0 <= bin_item["lower"] < bin_item["upper"] <= 1.0
                and 0.0 <= bin_item["confidence_sum"] <= bin_item["n"]
                and 0.0 <= bin_item["correct_sum"] <= bin_item["n"]
                and 0.0 <= bin_item["squared_error_sum"] <= bin_item["n"]
                and 0.0 <= bin_item["confidence"] <= 1.0
                and 0.0 <= bin_item["accuracy"] <= 1.0
            ):
                raise CorpusIntegrityError(f"{location}.calibration bin is invalid")
            if not (
                math.isclose(
                    bin_item["confidence"], bin_item["confidence_sum"] / bin_item["n"],
                    rel_tol=0.0, abs_tol=1e-15,
                )
                and math.isclose(
                    bin_item["accuracy"], bin_item["correct_sum"] / bin_item["n"],
                    rel_tol=0.0, abs_tol=1e-15,
                )
            ):
                raise CorpusIntegrityError(
                    f"{location}.calibration bin disagrees with sufficient statistics"
                )
            lower_index = round(bin_item["lower"] * 10)
            if not (
                0 <= lower_index <= 9
                and math.isclose(
                    bin_item["lower"], lower_index / 10.0, rel_tol=0.0, abs_tol=1e-15
                )
                and math.isclose(
                    bin_item["upper"], (lower_index + 1) / 10.0,
                    rel_tol=0.0, abs_tol=1e-15,
                )
            ):
                raise CorpusIntegrityError(
                    f"{location}.calibration bin boundary is not canonical"
                )
        if [item["lower"] for item in calibration["bins"]] != sorted(
            {item["lower"] for item in calibration["bins"]}
        ):
            raise CorpusIntegrityError(
                f"{location}.calibration bins are duplicated or unordered"
            )
        if sum(bin_item["n"] for bin_item in calibration["bins"]) != calibration["n"]:
            raise CorpusIntegrityError(f"{location}.calibration bin counts are inconsistent")
        if not math.isclose(
            sum(bin_item["squared_error_sum"] for bin_item in calibration["bins"])
            / calibration["n"],
            brier["value"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise CorpusIntegrityError(f"{location}.calibration disagrees with Brier score")
        recomputed_ece = sum(
            bin_item["n"] / calibration["n"]
            * abs(bin_item["confidence"] - bin_item["accuracy"])
            for bin_item in calibration["bins"]
        )
        if not math.isclose(
            calibration["ece_10_bin"], recomputed_ece, rel_tol=0.0, abs_tol=1e-15
        ):
            raise CorpusIntegrityError(f"{location}.calibration ECE is inconsistent")
    drift = summary["drift"]
    if drift is not None:
        drift = _require_artifact_keys(
            drift,
            {
                "terminal_distance_mean", "distance_slope_mean",
                "terminal_signed_bias_by_axis", "absolute_residual_slope_by_axis",
                "n_conversations",
            },
            f"{location}.drift",
        )
        if (
            type(drift["n_conversations"]) is not int
            or not 0 <= drift["n_conversations"] <= summary["conversation_count"]
        ):
            raise CorpusIntegrityError(f"{location}.drift count is invalid")
        if any(
            drift[field] is not None and not _is_artifact_number(drift[field])
            for field in ("terminal_distance_mean", "distance_slope_mean")
        ):
            raise CorpusIntegrityError(f"{location}.drift estimate is invalid")
        for field in ("terminal_signed_bias_by_axis", "absolute_residual_slope_by_axis"):
            axes = _require_artifact_keys(drift[field], set(AXES), f"{location}.drift.{field}")
            if any(value is not None and not _is_artifact_number(value) for value in axes.values()):
                raise CorpusIntegrityError(f"{location}.drift axis value is invalid")
    if not isinstance(summary["outcome_counts"], Mapping) or not all(
        key in ALLOWED_OUTCOMES and type(count) is int and count >= 0
        for key, count in summary["outcome_counts"].items()
    ) or sum(summary["outcome_counts"].values()) != summary["turn_count"]:
        raise CorpusIntegrityError(f"{location}.outcome_counts is invalid")
    method = _require_artifact_keys(
        summary["confidence_interval_method"],
        {"method", "cluster", "overall_stratification", "draws", "fixed_seed_material"},
        f"{location}.confidence_interval_method",
    )
    if (
        method["method"] != "percentile bootstrap"
        or method["cluster"] != "whole conversation"
        or method["overall_stratification"] != "domain"
        or method["draws"] != BOOTSTRAP_DRAWS
        or method["fixed_seed_material"] != expected_seed
    ):
        raise CorpusIntegrityError(f"{location}.confidence_interval_method is invalid")


def _validate_report_schema(report: Mapping[str, Any], failures: Sequence[Mapping[str, Any]]) -> None:
    expected_report_keys = {
        "report_schema_version", "corpus_version", "split", "corpus_sha256",
        "corpus_root_sha256", "compiler_sha256", "evaluator_sha256", "schema_sha256",
        "production_code_commit", "production_code_sha256", "evaluation_commit",
        "backend", "clock", "timezone", "modes", "development_correction_bundle",
        "paired_mode_differences", "semantic_fingerprint", "metric_supervision",
        "failure_count", "failure_counts", "environment", "limitations",
    }
    report = _require_artifact_keys(report, expected_report_keys, "aggregate report")
    if type(report["report_schema_version"]) is not int or report["report_schema_version"] != 1:
        raise CorpusIntegrityError("aggregate report schema version is invalid")
    if report["corpus_version"] != "conversation-v1" or report["split"] not in {
        "heldout", "development",
    } or report["backend"] != "clanker-v8" or report["timezone"] != "UTC":
        raise CorpusIntegrityError("aggregate report identity field is invalid")
    hash_fields = {
        "corpus_sha256", "corpus_root_sha256", "compiler_sha256", "evaluator_sha256",
        "production_code_sha256", "semantic_fingerprint",
    }
    if any(
        not isinstance(report[field], str) or not re.fullmatch(r"[0-9a-f]{64}", report[field])
        for field in hash_fields
    ):
        raise CorpusIntegrityError("aggregate report digest field is invalid")
    if any(
        not isinstance(report[field], str) or not re.fullmatch(r"[0-9a-f]{40}", report[field])
        for field in ("production_code_commit", "evaluation_commit")
    ):
        raise CorpusIntegrityError("aggregate report commit field is invalid")
    if report["clock"] != FIXED_INSTANT.isoformat():
        raise CorpusIntegrityError("aggregate report clock is invalid")
    if type(report["failure_count"]) is not int or report["failure_count"] != len(failures):
        raise CorpusIntegrityError("aggregate report failure_count is invalid")
    if not isinstance(report["failure_counts"], Mapping) or not all(
        key in FAILURE_CATEGORIES and type(count) is int and count >= 0
        for key, count in report["failure_counts"].items()
    ) or sum(report["failure_counts"].values()) != report["failure_count"]:
        raise CorpusIntegrityError("aggregate report failure_counts is invalid")
    expected_schema_names = {
        "conversation-v1.schema.json", "failure-v1.schema.json",
        "report-v1.schema.json", "source-v1.schema.json",
    }
    if not isinstance(report["schema_sha256"], Mapping) or set(report["schema_sha256"]) != expected_schema_names or not all(
        isinstance(name, str) and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
        for name, digest in report["schema_sha256"].items()
    ):
        raise CorpusIntegrityError("schema hashes are invalid")
    supervision = _require_artifact_keys(
        report["metric_supervision"],
        {"semantic_and_entity", "affect_and_trajectory", "outcome"},
        "metric supervision",
    )
    if dict(supervision) != {
        "semantic_and_entity": "turn semantic_supervision_level",
        "affect_and_trajectory": "weak_rule_v1 only; structural_only sources excluded",
        "outcome": "weak or counterfactual stratum only; no outcome accuracy is claimed",
    }:
        raise CorpusIntegrityError("metric supervision values are invalid")
    environment = _require_artifact_keys(
        report["environment"],
        {"python_version", "implementation", "operating_system", "machine"},
        "environment",
    )
    version = environment["python_version"]
    if not isinstance(version, list) or len(version) != 3 or any(type(item) is not int for item in version):
        raise CorpusIntegrityError("environment Python version is invalid")
    if (
        environment["implementation"] not in PLATFORM_IMPLEMENTATIONS
        or environment["operating_system"] not in PLATFORM_SYSTEMS
        or environment["machine"] not in PLATFORM_MACHINES
    ):
        raise CorpusIntegrityError("environment platform identifiers are invalid")
    if report["limitations"] != list(REPORT_LIMITATIONS):
        raise CorpusIntegrityError("limitations are not the declared finite set")
    modes = _require_artifact_keys(report["modes"], set(MODES), "modes")
    mode_keys = {
        "overall", "by_domain", "by_outcome", "by_supervision_level",
        "latency", "resource_growth", "execution_errors",
    }
    for mode in MODES:
        mode_report = _require_artifact_keys(modes[mode], mode_keys, f"modes.{mode}")
        if type(mode_report["execution_errors"]) is not int or mode_report["execution_errors"] < 0:
            raise CorpusIntegrityError(f"modes.{mode}.execution_errors must be nonnegative integer")
        seed_prefix = f"{report['corpus_sha256']}|metric-v{METRIC_SCHEMA_VERSION}|{mode}"
        _validate_summary(
            mode_report["overall"], f"modes.{mode}.overall",
            expected_seed=f"{seed_prefix}|overall",
        )
        for stratum in ("by_domain", "by_outcome", "by_supervision_level"):
            if not isinstance(mode_report[stratum], Mapping):
                raise CorpusIntegrityError(f"modes.{mode}.{stratum} must be object")
            allowed_keys = (
                ALLOWED_DOMAINS if stratum == "by_domain"
                else ALLOWED_OUTCOMES if stratum == "by_outcome"
                else SUPERVISION_LEVELS
            )
            if set(mode_report[stratum]) - set(allowed_keys):
                raise CorpusIntegrityError(f"modes.{mode}.{stratum} has invalid keys")
            seed_label = {
                "by_domain": "domain", "by_outcome": "outcome",
                "by_supervision_level": "supervision",
            }[stratum]
            for key, summary in mode_report[stratum].items():
                _validate_summary(
                    summary, f"modes.{mode}.{stratum}.{key}",
                    expected_seed=f"{seed_prefix}|{seed_label}|{key}",
                )
                if (
                    summary["turn_count"] > mode_report["overall"]["turn_count"]
                    or summary["conversation_count"] > mode_report["overall"]["conversation_count"]
                ):
                    raise CorpusIntegrityError(f"modes.{mode}.{stratum}.{key} exceeds overall counts")
        if (
            sum(item["turn_count"] for item in mode_report["by_domain"].values())
            != mode_report["overall"]["turn_count"]
            or sum(item["conversation_count"] for item in mode_report["by_domain"].values())
            != mode_report["overall"]["conversation_count"]
        ):
            raise CorpusIntegrityError(f"modes.{mode}.by_domain counts disagree with overall")
        if set(mode_report["by_outcome"]) != set(mode_report["overall"]["outcome_counts"]):
            raise CorpusIntegrityError(f"modes.{mode}.by_outcome keys disagree with outcome counts")
        if any(
            mode_report["by_outcome"][outcome]["turn_count"] != count
            for outcome, count in mode_report["overall"]["outcome_counts"].items()
        ):
            raise CorpusIntegrityError(f"modes.{mode}.by_outcome counts disagree with overall")
        if not isinstance(mode_report["latency"], Mapping) or not isinstance(
            mode_report["resource_growth"], Mapping
        ):
            raise CorpusIntegrityError(f"modes.{mode} observations must be objects")
        latency = _require_artifact_keys(
            mode_report["latency"],
            {
                "n", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms",
                "observational_nondeterministic", "turns_per_second",
            },
            f"modes.{mode}.latency",
        )
        if (
            type(latency["n"]) is not int
            or latency["n"] < 0
            or latency["n"] != mode_report["overall"]["turn_count"]
            or latency["observational_nondeterministic"] is not True
        ):
            raise CorpusIntegrityError(f"modes.{mode}.latency types are invalid")
        if not all(
            _is_artifact_number(latency[field])
            for field in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "turns_per_second")
        ):
            raise CorpusIntegrityError(f"modes.{mode}.latency values are invalid")
        resource_keys = {
            "turn_samples", "sqlite_allocated_bytes_growth_mean",
            "sqlite_allocated_bytes_growth_max", "sqlite_row_growth_maxima",
            "growth_slopes_per_turn", "construction_latency_ms", "tracemalloc_peak_bytes",
            "process_lifetime_maxrss_peak", "process_maxrss_units", "process_maxrss_comparability",
        } | {
            f"memory_{memory_key}_growth_{stat}"
            for memory_key in ("entities", "events", "relations", "serialized_bytes")
            for stat in ("mean", "max")
        }
        resources = _require_artifact_keys(
            mode_report["resource_growth"], resource_keys, f"modes.{mode}.resource_growth"
        )
        if (
            type(resources["turn_samples"]) is not int
            or resources["turn_samples"] < 0
            or resources["turn_samples"] != mode_report["overall"]["turn_count"]
            or type(resources["tracemalloc_peak_bytes"]) is not int
            or resources["tracemalloc_peak_bytes"] < 0
        ):
            raise CorpusIntegrityError(f"modes.{mode}.resource counts are invalid")
        numeric_resources = resource_keys - {
            "sqlite_row_growth_maxima", "growth_slopes_per_turn", "construction_latency_ms",
            "process_maxrss_units", "process_maxrss_comparability",
        }
        if any(not _is_artifact_number(resources[field]) for field in numeric_resources):
            raise CorpusIntegrityError(f"modes.{mode}.resource values are invalid")
        if not isinstance(resources["process_maxrss_units"], str) or not isinstance(
            resources["process_maxrss_comparability"], str
        ):
            raise CorpusIntegrityError(f"modes.{mode}.resource labels are invalid")
        if resources["process_maxrss_units"] != "KiB on Linux; bytes on macOS" or resources[
            "process_maxrss_comparability"
        ] != "observational_process_lifetime_peak_mode_order_dependent":
            raise CorpusIntegrityError(f"modes.{mode}.resource labels are invalid")
        if (
            not isinstance(resources["sqlite_row_growth_maxima"], Mapping)
            or set(resources["sqlite_row_growth_maxima"]) != SQLITE_ROW_TABLES
            or not all(
                type(count) is int and count >= 0
                for count in resources["sqlite_row_growth_maxima"].values()
            )
        ):
            raise CorpusIntegrityError(f"modes.{mode}.resource row counts are invalid")
        slopes = _require_artifact_keys(
            resources["growth_slopes_per_turn"],
            {"memory_serialized_bytes", "sqlite_allocated_bytes"},
            f"modes.{mode}.resource slopes",
        )
        if any(value is not None and not _is_artifact_number(value) for value in slopes.values()):
            raise CorpusIntegrityError(f"modes.{mode}.resource slopes are invalid")
        construction = _require_artifact_keys(
            resources["construction_latency_ms"],
            {"n", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "observational_nondeterministic"},
            f"modes.{mode}.construction latency",
        )
        expected_constructions = (
            mode_report["overall"]["turn_count"]
            if mode == "sentence_only"
            else mode_report["overall"]["conversation_count"]
        )
        if (
            type(construction["n"]) is not int
            or construction["n"] < 0
            or construction["n"] != expected_constructions
            or construction["observational_nondeterministic"] is not True
        ):
            raise CorpusIntegrityError(f"modes.{mode}.construction latency types are invalid")
        if not all(
            _is_artifact_number(construction[field])
            for field in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
        ):
            raise CorpusIntegrityError(f"modes.{mode}.construction latency values are invalid")
    correction_keys = {
        "source_split", "source_conversation_count", "context_count", "sample_count",
        "expected_finalized_sample_count", "terminal_pending_turns_excluded",
        "eligible_context_count", "sha256", "active_profile_id",
        "heldout_observations_used_for_lookup", "source_corpus_sha256",
        "production_runtime_commit", "sha256_after_evaluation", "lookup_store_unchanged",
    }
    correction = _require_artifact_keys(
        report["development_correction_bundle"], correction_keys, "development correction bundle"
    )
    if any(
        type(correction[field]) is not int or correction[field] < 0
        for field in (
            "source_conversation_count", "context_count", "sample_count",
            "expected_finalized_sample_count", "terminal_pending_turns_excluded",
            "eligible_context_count", "heldout_observations_used_for_lookup",
        )
    ) or type(correction["lookup_store_unchanged"]) is not bool:
        raise CorpusIntegrityError("development correction bundle types are invalid")
    if correction["active_profile_id"] is not None or correction["source_split"] != "development" or not all(
        isinstance(correction[field], str)
        for field in (
            "source_split", "sha256", "source_corpus_sha256", "production_runtime_commit",
            "sha256_after_evaluation",
        )
    ):
        raise CorpusIntegrityError("development correction bundle provenance is invalid")
    if correction["heldout_observations_used_for_lookup"] != 0 or correction["lookup_store_unchanged"] is not True:
        raise CorpusIntegrityError("development correction bundle violates held-out isolation")
    if (
        correction["sample_count"] != correction["expected_finalized_sample_count"]
        or correction["terminal_pending_turns_excluded"] != correction["source_conversation_count"]
        or correction["context_count"] > correction["sample_count"]
        or correction["eligible_context_count"] > correction["context_count"]
        or correction["sha256_after_evaluation"] != correction["sha256"]
    ):
        raise CorpusIntegrityError("development correction bundle counts are inconsistent")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", correction[field])
        for field in ("sha256", "source_corpus_sha256", "sha256_after_evaluation")
    ) or not re.fullmatch(r"[0-9a-f]{40}", correction["production_runtime_commit"]):
        raise CorpusIntegrityError("development correction bundle digest is invalid")
    paired = _require_artifact_keys(
        report["paired_mode_differences"],
        {"stateful_minus_sentence_only", "transition_corrected_minus_stateful"},
        "paired mode differences",
    )
    for name, comparison in paired.items():
        comparison = _require_artifact_keys(
            comparison,
            {"candidate_mode", "reference_mode", "paired_turn_count", "pairing", "metrics"},
            f"paired mode differences.{name}",
        )
        if (
            type(comparison["paired_turn_count"]) is not int
            or comparison["paired_turn_count"] < 0
            or not isinstance(comparison["metrics"], Mapping)
        ):
            raise CorpusIntegrityError(f"paired mode differences.{name} types are invalid")
        expected_modes = (
            ("stateful", "sentence_only")
            if name == "stateful_minus_sentence_only"
            else ("transition_corrected", "stateful")
        )
        if (
            (comparison["candidate_mode"], comparison["reference_mode"]) != expected_modes
            or comparison["pairing"] != PAIRING_DESCRIPTION
        ):
            raise CorpusIntegrityError(f"paired mode differences.{name} labels are invalid")
        expected_paired_turns = min(
            report["modes"][expected_modes[0]]["overall"]["turn_count"],
            report["modes"][expected_modes[1]]["overall"]["turn_count"],
        )
        if comparison["paired_turn_count"] != expected_paired_turns:
            raise CorpusIntegrityError(f"paired mode differences.{name} count is inconsistent")
        for metric_name, metric in comparison["metrics"].items():
            if metric_name not in METRIC_NAMES:
                raise CorpusIntegrityError(f"paired.{name} metric name is invalid")
            _validate_metric_summary(
                metric,
                f"paired.{name}.{metric_name}",
                metric_name=metric_name,
                paired=True,
                max_turns=comparison["paired_turn_count"],
                max_conversations=min(
                    report["modes"][expected_modes[0]]["overall"]["conversation_count"],
                    report["modes"][expected_modes[1]]["overall"]["conversation_count"],
                ),
            )
        if not PAIRED_REQUIRED_METRICS <= set(comparison["metrics"]):
            raise CorpusIntegrityError(f"paired.{name} is missing required measurements")
    failure_keys = {"conversation_id", "turn_id", "domain", "mode", "category"}
    for failure in failures:
        if not isinstance(failure, Mapping) or not failure_keys <= set(failure):
            raise CorpusIntegrityError("failure ledger row is malformed")
        if set(failure) != failure_keys:
            raise CorpusIntegrityError("failure ledger row contains an unexpected field")
        if not all(isinstance(value, str) for value in failure.values()):
            raise CorpusIntegrityError("failure ledger values must be text")
        if failure["mode"] not in MODES or failure["domain"] not in ALLOWED_DOMAINS or failure[
            "category"
        ] not in FAILURE_CATEGORIES:
            raise CorpusIntegrityError("failure ledger categorical value is invalid")

    actual_failure_counts = dict(sorted(Counter(item["category"] for item in failures).items()))
    if dict(report["failure_counts"]) != actual_failure_counts:
        raise CorpusIntegrityError("failure_counts disagrees with the failure ledger")
    for mode in MODES:
        if report["modes"][mode]["execution_errors"] != 0:
            raise CorpusIntegrityError("publishable aggregate reports cannot contain execution errors")


def _validate_no_conversation_payloads(
    report: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    conversation_texts: Sequence[str],
) -> None:
    string_values: List[str] = []
    key_values: List[str] = []
    content_values: List[str] = []

    def walk(value: Any, location: str, *, is_key: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized_key = str(key).lower()
                if normalized_key in _FORBIDDEN_ARTIFACT_KEYS or normalized_key.endswith("_text"):
                    raise CorpusIntegrityError(
                        f"aggregate artifact contains forbidden payload key at {location}.{key}"
                    )
                walk(str(key), f"{location}.<key>", is_key=True)
                walk(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{location}[{index}]")
        elif isinstance(value, str):
            string_values.append(value)
            (key_values if is_key else content_values).append(value)
            normalized_value = _norm(value)
            for raw_text in conversation_texts:
                normalized_text = _norm(raw_text)
                if normalized_value == normalized_text or (
                    normalized_text and normalized_text in normalized_value
                ):
                    raise CorpusIntegrityError(
                        f"aggregate artifact contains held-out turn content at {location}"
                    )

    walk(report, "report")
    walk(list(failures), "failures")
    normalized_joined = " ".join(_norm(value) for value in string_values)
    compact_slots = {_norm(value).replace(" ", "") for value in string_values}
    compact_joined = (
        "".join(_norm(value).replace(" ", "") for value in key_values),
        "".join(_norm(value).replace(" ", "") for value in content_values),
    )
    for raw_text in conversation_texts:
        normalized_text = _norm(raw_text)
        compact_text = normalized_text.replace(" ", "")
        split_across_two_slots = any(
            compact_text[:cut] in compact_slots and compact_text[cut:] in compact_slots
            for cut in range(1, len(compact_text))
        )
        if normalized_text and (
            normalized_text in normalized_joined
            or any(compact_text in joined for joined in compact_joined)
            or split_across_two_slots
        ):
            raise CorpusIntegrityError("aggregate artifact contains fragmented held-out turn content")


def _validate_aggregate_artifacts(
    report: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    conversations: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Fail if aggregate artifacts contain raw conversational payloads."""

    heldout_texts = [
        str(turn["text"])
        for conversation in conversations
        for turn in conversation["turns"]
    ]
    _validate_no_conversation_payloads(report, failures, heldout_texts)
    turn_domains = {
        (str(conversation["conversation_id"]), str(turn["turn_id"])): str(conversation["domain"])
        for conversation in conversations
        for turn in conversation["turns"]
    }

    if not isinstance(report, Mapping) or report.get("report_schema_version") != METRIC_SCHEMA_VERSION:
        raise CorpusIntegrityError("aggregate report discriminator is missing or unsupported")
    _validate_report_schema(report, failures)
    bound_manifest = manifest if manifest is not None else load_manifest()
    split = str(report["split"])
    if split not in bound_manifest["splits"] or any(
        report[field] != bound_manifest[manifest_field]
        for field, manifest_field in (
            ("corpus_root_sha256", "corpus_root_sha256"),
            ("compiler_sha256", "compiler_sha256"),
            ("evaluator_sha256", "evaluator_sha256"),
            ("schema_sha256", "schema_sha256"),
            ("production_code_commit", "baseline_code_commit"),
            ("production_code_sha256", "production_code_sha256"),
        )
    ) or report["corpus_sha256"] != bound_manifest["splits"][split]["sha256"]:
        raise CorpusIntegrityError("aggregate report provenance disagrees with the selected manifest")
    correction = report["development_correction_bundle"]
    development = bound_manifest["splits"]["development"]
    if (
        correction["source_conversation_count"] != development["conversation_count"]
        or correction["terminal_pending_turns_excluded"] != development["conversation_count"]
        or correction["expected_finalized_sample_count"]
        != development["turn_count"] - development["conversation_count"]
        or correction["source_corpus_sha256"] != development["sha256"]
        or correction["production_runtime_commit"] != bound_manifest["baseline_code_commit"]
    ):
        raise CorpusIntegrityError("development correction bundle disagrees with the selected manifest")

    semantic_payload = {
        "modes": {
            mode: {
                "overall": report["modes"][mode]["overall"],
                "by_domain": report["modes"][mode]["by_domain"],
                "by_outcome": report["modes"][mode]["by_outcome"],
                "by_supervision_level": report["modes"][mode]["by_supervision_level"],
                "execution_errors": report["modes"][mode]["execution_errors"],
            }
            for mode in MODES
        },
        "paired_mode_differences": report["paired_mode_differences"],
    }
    expected_fingerprint = hashlib.sha256(
        _canonical_json(semantic_payload).encode("utf-8")
    ).hexdigest()
    if report["semantic_fingerprint"] != expected_fingerprint:
        raise CorpusIntegrityError("semantic fingerprint disagrees with aggregate metrics")

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
        if set(failure) != failure_keys:
            raise CorpusIntegrityError("failure ledger row contains an unexpected field")

    failure_identities = [
        (
            str(failure["conversation_id"]), str(failure["turn_id"]),
            str(failure["mode"]), str(failure["category"]),
        )
        for failure in failures
    ]
    if len(failure_identities) != len(set(failure_identities)):
        raise CorpusIntegrityError("failure ledger contains duplicate rows")
    failure_metric_counts = Counter(
        (str(failure["mode"]), str(failure["category"])) for failure in failures
    )
    for mode in MODES:
        overall_metrics = report["modes"][mode]["overall"]["metrics"]
        for category in FAILURE_CATEGORIES:
            metric = overall_metrics.get(category)
            expected_failures = (
                metric["n_turns"] - round(metric["value"] * metric["n_turns"])
                if metric is not None
                else 0
            )
            if failure_metric_counts[(mode, category)] != expected_failures:
                raise CorpusIntegrityError("failure ledger counts disagree with zero-valued metrics")

    expected_populations: Dict[str, Dict[str, tuple[int, int]]] = {
        "by_domain": {}, "by_outcome": {}, "by_supervision_level": {},
    }
    population_rows: Dict[str, Dict[str, List[tuple[str, Mapping[str, Any]]]]] = {
        "by_domain": defaultdict(list),
        "by_outcome": defaultdict(list),
        "by_supervision_level": defaultdict(list),
    }
    all_rows: List[tuple[str, Mapping[str, Any]]] = []
    population_members: Dict[str, Dict[str, set[str]]] = {
        "by_domain": defaultdict(set),
        "by_outcome": defaultdict(set),
        "by_supervision_level": defaultdict(set),
    }
    population_turns: Dict[str, Counter[str]] = {
        "by_domain": Counter(), "by_outcome": Counter(), "by_supervision_level": Counter(),
    }
    for conversation in conversations:
        conversation_id = str(conversation["conversation_id"])
        for turn in conversation["turns"]:
            all_rows.append((conversation_id, turn))
            labels = {
                "by_domain": str(conversation["domain"]),
                "by_outcome": str(turn["annotation"]["outcome"]),
                "by_supervision_level": str(turn["annotation"]["supervision_level"]),
            }
            for stratum, label in labels.items():
                population_turns[stratum][label] += 1
                population_members[stratum][label].add(conversation_id)
                population_rows[stratum][label].append((conversation_id, turn))
    for stratum in expected_populations:
        expected_populations[stratum] = {
            label: (count, len(population_members[stratum][label]))
            for label, count in population_turns[stratum].items()
        }
    total_turns = sum(len(conversation["turns"]) for conversation in conversations)
    total_conversations = len(conversations)

    def expected_metric_counts(
        rows: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> Dict[str, tuple[int, int]]:
        metric_rows: Dict[str, List[str]] = {
            name: [conversation_id for conversation_id, _turn in rows]
            for name in SUMMARY_REQUIRED_METRICS
        }
        semantic_rows = [
            conversation_id
            for conversation_id, turn in rows
            if turn["annotation"]["semantic"]["scored"] is True
        ]
        answer_rows = [
            conversation_id
            for conversation_id, turn in rows
            if turn["annotation"]["expected_answer"]["scored"] is True
        ]
        entity_rows = [
            conversation_id
            for conversation_id, turn in rows
            if bool(turn["annotation"]["expected_entity_refs"])
        ]
        affect_rows = [
            conversation_id
            for conversation_id, turn in rows
            if turn["annotation"]["affect_scored"] is True
            and turn["annotation"]["observed_next_state"] is not None
        ]
        if semantic_rows:
            metric_rows["semantic_parse_exact"] = semantic_rows
        if answer_rows:
            metric_rows["semantic_answer_exact"] = answer_rows
            metric_rows["brier"] = answer_rows
        if entity_rows:
            metric_rows["entity_resolution_exact"] = entity_rows
        if affect_rows:
            for name in (
                "target_attainment", "target_distance_improvement", "next_state_distance",
                *(f"mae_{axis}" for axis in AXES),
                *(f"mae_normalized_{axis}" for axis in AXES),
                *(f"direction_{axis}" for axis in AXES),
            ):
                metric_rows[name] = affect_rows
        return {
            name: (len(conversation_ids), len(set(conversation_ids)))
            for name, conversation_ids in metric_rows.items()
        }

    expected_metrics = {
        "overall": expected_metric_counts(all_rows),
        **{
            f"{stratum}:{label}": expected_metric_counts(rows)
            for stratum, groups in population_rows.items()
            for label, rows in groups.items()
        },
    }

    def validate_metric_population(
        summary: Mapping[str, Any],
        expected_key: str,
        rows: Sequence[tuple[str, Mapping[str, Any]]],
        location: str,
    ) -> None:
        expected = expected_metrics[expected_key]
        observed_names = set(summary["metrics"])
        if observed_names != set(expected):
            raise CorpusIntegrityError(f"{location} metric population is incomplete")
        for name, counts in expected.items():
            observed = summary["metrics"][name]
            if (observed["n_turns"], observed["n_conversations"]) != counts:
                raise CorpusIntegrityError(f"{location}.{name} counts are inconsistent")
        for label, field in (("unknown", "unknown_classification"), ("conflict", "conflict_classification")):
            gold_support = sum(
                turn["annotation"]["expected_answer_status"] == label
                for _conversation_id, turn in rows
            )
            classification = summary[field]
            if classification["tp"] + classification["fn"] != gold_support:
                raise CorpusIntegrityError(f"{location}.{field} gold support is inconsistent")
            expected_clusters: Dict[str, Dict[str, Any]] = {}
            for conversation_id in {conversation_id for conversation_id, _turn in rows}:
                turn_rows = [
                    turn for candidate_id, turn in rows if candidate_id == conversation_id
                ]
                expected_clusters[conversation_id] = {
                    "domain": turn_domains[
                        (conversation_id, str(turn_rows[0]["turn_id"]))
                    ],
                    "turn_count": len(turn_rows),
                    "gold_support": sum(
                        turn["annotation"]["expected_answer_status"] == label
                        for turn in turn_rows
                    ),
                }
            observed_clusters = classification["clusters"]
            if {
                cluster["conversation_id"] for cluster in observed_clusters
            } != set(expected_clusters):
                raise CorpusIntegrityError(
                    f"{location}.{field} cluster identities are inconsistent"
                )
            for cluster in observed_clusters:
                expected_cluster = expected_clusters[cluster["conversation_id"]]
                if (
                    cluster["domain"] != expected_cluster["domain"]
                    or sum(cluster[key] for key in ("tp", "fp", "fn", "tn"))
                    != expected_cluster["turn_count"]
                    or cluster["tp"] + cluster["fn"]
                    != expected_cluster["gold_support"]
                ):
                    raise CorpusIntegrityError(
                        f"{location}.{field} cluster population is inconsistent"
                    )
        affect_conversations = {
            conversation_id
            for conversation_id, turn in rows
            if turn["annotation"]["affect_scored"] is True
            and turn["annotation"]["observed_next_state"] is not None
        }
        if affect_conversations:
            if (
                summary["drift"] is None
                or summary["drift"]["n_conversations"] != len(affect_conversations)
            ):
                raise CorpusIntegrityError(f"{location}.drift population is inconsistent")
        elif summary["drift"] is not None:
            raise CorpusIntegrityError(f"{location}.drift must be unavailable without affect rows")

    for mode in MODES:
        mode_report = report["modes"][mode]
        if (
            mode_report["overall"]["turn_count"] != total_turns
            or mode_report["overall"]["conversation_count"] != total_conversations
        ):
            raise CorpusIntegrityError(f"aggregate report {mode} overall population is incomplete")
        validate_metric_population(
            mode_report["overall"], "overall", all_rows, f"aggregate report {mode}.overall"
        )
        for stratum, expected in expected_populations.items():
            observed = mode_report[stratum]
            if set(observed) != set(expected) or any(
                (
                    observed[label]["turn_count"],
                    observed[label]["conversation_count"],
                ) != counts
                for label, counts in expected.items()
            ):
                raise CorpusIntegrityError(f"aggregate report {mode} {stratum} population is incomplete")
            for label in expected:
                validate_metric_population(
                    observed[label],
                    f"{stratum}:{label}",
                    population_rows[stratum][label],
                    f"aggregate report {mode}.{stratum}.{label}",
                )

    paired_expected = {
        name: counts for name, counts in expected_metrics["overall"].items()
        if name != "candidate_count"
    }
    for comparison_name, comparison in report["paired_mode_differences"].items():
        if set(comparison["metrics"]) != set(paired_expected):
            raise CorpusIntegrityError(
                f"paired mode differences.{comparison_name} metric population is incomplete"
            )
        for name, counts in paired_expected.items():
            metric = comparison["metrics"][name]
            if (metric["n_turns"], metric["n_conversations"]) != counts:
                raise CorpusIntegrityError(
                    f"paired mode differences.{comparison_name}.{name} counts are inconsistent"
                )

    for failure in failures:
        identity = (str(failure.get("conversation_id", "")), str(failure.get("turn_id", "")))
        if identity not in turn_domains or failure.get("domain") != turn_domains[identity]:
            raise CorpusIntegrityError("failure ledger identity does not match the evaluated corpus")


def _atomic_replace(source: Path, target: Path) -> None:
    source.replace(target)


def _report_generation_id(
    *,
    report_name: str,
    report_sha256: str,
    failure_name: str,
    failure_sha256: str,
    checksum_name: str,
    checksum_sha256: str,
) -> str:
    return hashlib.sha256(_canonical_json({
        "report": {"name": report_name, "sha256": report_sha256},
        "failures": {"name": failure_name, "sha256": failure_sha256},
        "checksum": {"name": checksum_name, "sha256": checksum_sha256},
    }).encode("utf-8")).hexdigest()


def _validate_report_generation(directory: Path, expected_names: set[str]) -> None:
    try:
        members = list(directory.iterdir())
    except OSError as exc:
        raise CorpusIntegrityError("published report generation is unreadable") from exc
    if {member.name for member in members} != expected_names or any(
        member.is_symlink() or not member.is_file() for member in members
    ):
        raise CorpusIntegrityError("published report generation file inventory is not exact")


def _reject_report_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise CorpusIntegrityError(f"controlled report path cannot traverse a symlink: {current}")
        if current.parent == current:
            return
        current = current.parent


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
    pointer_path = output_path.with_suffix(".current")
    generations = output_path.parent / f"{output_path.stem}.generations"
    _reject_report_symlink_chain(output_path.parent)
    _reject_report_symlink_chain(failure_target.parent)
    controlled_targets = (output_path, failure_target, checksum_path, pointer_path, generations)
    if any(path.is_symlink() for path in controlled_targets):
        raise CorpusIntegrityError("report publication target cannot be a symlink")
    targets = [output_path.resolve(), failure_target.resolve(), checksum_path.resolve()]
    if len(set(targets)) != len(targets):
        raise CorpusIntegrityError("report, failures, and checksum targets must be distinct")
    if failure_target.parent.resolve() != output_path.parent.resolve():
        raise CorpusIntegrityError("report, failures, and checksum must share one publish directory")
    reserved = {pointer_path.resolve(), generations.resolve()}
    if reserved & set(targets) or pointer_path.resolve() == generations.resolve():
        raise CorpusIntegrityError("report artifact targets collide with atomic publication metadata")
    if failure_target.suffix != ".jsonl":
        raise CorpusIntegrityError("failure target must use .jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    failure_bytes = "".join(_canonical_json(item) + "\n" for item in failures).encode("utf-8")
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    failure_sha256 = hashlib.sha256(failure_bytes).hexdigest()
    checksum_bytes = (
        f"{report_sha256}  {output_path.name}\n"
        f"{failure_sha256}  {failure_target.name}\n"
    ).encode("ascii")
    checksum_sha256 = hashlib.sha256(checksum_bytes).hexdigest()
    generation_id = _report_generation_id(
        report_name=output_path.name,
        report_sha256=report_sha256,
        failure_name=failure_target.name,
        failure_sha256=failure_sha256,
        checksum_name=checksum_path.name,
        checksum_sha256=checksum_sha256,
    )
    generation_dir = generations / generation_id
    pointer_bytes = (json.dumps({
        "generation": generation_id,
        "report": output_path.name,
        "failures": failure_target.name,
        "checksum": checksum_path.name,
    }, sort_keys=True) + "\n").encode("ascii")
    if generations.is_symlink() or (generations.exists() and not generations.is_dir()):
        raise CorpusIntegrityError("report generation store must be a regular directory")
    generations.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".conversation-report-", dir=output_path.parent) as staging_name:
        staging_root = Path(staging_name)
        staging = staging_root / "generation"
        staging.mkdir()
        for name, payload in (
            (output_path.name, report_bytes),
            (failure_target.name, failure_bytes),
            (checksum_path.name, checksum_bytes),
        ):
            (staging / name).write_bytes(payload)
        provenance_check()
        if generation_dir.exists():
            if not generation_dir.is_dir() or generation_dir.is_symlink():
                raise CorpusIntegrityError("existing report generation is not a regular directory")
            _validate_report_generation(
                generation_dir, {output_path.name, failure_target.name, checksum_path.name}
            )
            expected = {
                output_path.name: report_bytes,
                failure_target.name: failure_bytes,
                checksum_path.name: checksum_bytes,
            }
            if any(
                not (generation_dir / name).is_file()
                or (generation_dir / name).read_bytes() != payload
                for name, payload in expected.items()
            ):
                raise CorpusIntegrityError("existing immutable report generation differs")
        else:
            _atomic_replace(staging, generation_dir)
        pointer_stage = staging_root / pointer_path.name
        pointer_stage.write_bytes(pointer_bytes)
        provenance_check()
        _atomic_replace(pointer_stage, pointer_path)


def load_published_artifacts(output_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Path]:
    """Resolve one atomically selected report generation and verify its hashes."""

    _reject_report_symlink_chain(output_path.parent)
    pointer_path = output_path.with_suffix(".current")
    if output_path.is_symlink() or pointer_path.is_symlink():
        raise CorpusIntegrityError("published report target cannot be a symlink")
    try:
        if pointer_path.is_symlink():
            raise OSError("pointer is a symlink")
        pointer = json.loads(pointer_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusIntegrityError("published report pointer is missing or malformed") from exc
    if not isinstance(pointer, Mapping) or set(pointer) != {
        "generation", "report", "failures", "checksum",
    }:
        raise CorpusIntegrityError("published report pointer schema is invalid")
    generation = pointer["generation"]
    if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{64}", generation):
        raise CorpusIntegrityError("published report generation ID is invalid")
    for field in ("report", "failures", "checksum"):
        if not isinstance(pointer[field], str) or Path(pointer[field]).name != pointer[field]:
            raise CorpusIntegrityError("published report filename is invalid")
    pointer_names = [pointer[field] for field in ("report", "failures", "checksum")]
    if len(set(pointer_names)) != 3:
        raise CorpusIntegrityError("published report role filenames must be distinct")
    if (
        pointer["report"] != output_path.name
        or pointer["checksum"] != output_path.with_suffix(".sha256").name
        or not pointer["failures"].endswith(".jsonl")
    ):
        raise CorpusIntegrityError("published report pointer selects unexpected filenames")
    generations = output_path.parent / f"{output_path.stem}.generations"
    if generations.is_symlink() or not generations.is_dir():
        raise CorpusIntegrityError("published report generation store is missing or invalid")
    generation_dir = generations / generation
    if not generation_dir.is_dir() or generation_dir.is_symlink():
        raise CorpusIntegrityError("published report generation is missing or not a regular directory")
    _validate_report_generation(
        generation_dir, {pointer["report"], pointer["failures"], pointer["checksum"]}
    )
    report_path = generation_dir / pointer["report"]
    failure_path = generation_dir / pointer["failures"]
    checksum_path = generation_dir / pointer["checksum"]
    public_targets = (
        output_path.parent / pointer["report"],
        output_path.parent / pointer["failures"],
        output_path.parent / pointer["checksum"],
    )
    if any(path.is_symlink() for path in public_targets):
        raise CorpusIntegrityError("published report role target cannot be a symlink")
    try:
        checksum_bytes = checksum_path.read_bytes()
        checksum_lines = [line.split() for line in checksum_bytes.decode("ascii").splitlines()]
        if (
            len(checksum_lines) != 2
            or any(len(line) != 2 for line in checksum_lines)
            or len({line[1] for line in checksum_lines}) != 2
            or any(not re.fullmatch(r"[0-9a-f]{64}", line[0]) for line in checksum_lines)
        ):
            raise ValueError("checksum must contain exactly two rows")
        expected_hashes = {name: digest for digest, name in checksum_lines}
    except (OSError, UnicodeError, ValueError) as exc:
        raise CorpusIntegrityError("published report checksum is malformed") from exc
    if set(expected_hashes) != {report_path.name, failure_path.name}:
        raise CorpusIntegrityError("published report checksum names are invalid")
    payloads: Dict[Path, bytes] = {}
    for path in (report_path, failure_path):
        try:
            payloads[path] = path.read_bytes()
        except OSError as exc:
            raise CorpusIntegrityError("published report generation is unreadable") from exc
        if expected_hashes.get(path.name) != hashlib.sha256(payloads[path]).hexdigest():
            raise CorpusIntegrityError("published report generation hash mismatch")
    selected_generation = _report_generation_id(
        report_name=report_path.name,
        report_sha256=expected_hashes[report_path.name],
        failure_name=failure_path.name,
        failure_sha256=expected_hashes[failure_path.name],
        checksum_name=checksum_path.name,
        checksum_sha256=hashlib.sha256(checksum_bytes).hexdigest(),
    )
    if selected_generation != generation:
        raise CorpusIntegrityError("published report generation identifier mismatch")
    try:
        report = json.loads(payloads[report_path].decode("utf-8"))
        failures = [
            json.loads(line)
            for line in payloads[failure_path].decode("utf-8").splitlines()
            if line
        ]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusIntegrityError("published report generation payload is malformed") from exc
    if not isinstance(report, Mapping) or any(not isinstance(item, Mapping) for item in failures):
        raise CorpusIntegrityError("published report generation payload roles are invalid")
    return report, failures, generation_dir


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

    manifest_path = selected_manifest_path(manifest_path)
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
            "python_version": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
            "implementation": platform.python_implementation().lower(),
            "operating_system": platform.system().lower() or "unknown",
            "machine": platform.machine().lower() or "unknown",
        },
        "limitations": list(REPORT_LIMITATIONS),
    }
    _validate_aggregate_artifacts(report, all_failures, conversations, manifest=manifest)
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
