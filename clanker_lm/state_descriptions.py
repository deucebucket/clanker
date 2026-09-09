"""Qualified state components shared by memory and compositional realization.

This is a bounded grammatical interpretation of existing event frames, not an
emotion detector or a response phrase catalogue. Original evidence is immutable;
projections preserve person, scope, intensifiers and explicit temporal cues.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

from . import lexicon
from .model import (AnswerContract, AnswerStatus, EntityKind, Evidence, EventFrame,
                    HowKind, QuestionKind, RefKind, SemanticRef, SourceKind)
from .state_scope import ScopedStateReport, state_report

STATE_TERMS = frozenset({
    "angry", "calm", "sad", "happy", "afraid", "scared", "anxious", "worried",
    "frustrated", "disappointed", "relieved", "upset", "excited", "proud",
    "ashamed", "lonely", "hopeful", "hopeless", "overwhelmed", "tired",
    "exhausted", "better", "content", "grateful", "jealous", "nervous",
})
DEICTICS = frozenset({"now", "today", "yesterday", "tomorrow", "tonight"})
MODIFIERS = frozenset({"very", "really", "quite", "so", "extremely"})
MAX_COMPONENTS = 4
SCHEMA = "qualified-state-components-v1"


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def content_hash(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class StateComponent:
    index: int
    report: ScopedStateReport
    event: EventFrame
    temporal_cue: str | None


def state_components(event: EventFrame) -> tuple[StateComponent, ...]:
    """Project a simple state or an unambiguous positive AND list.

    Negated conjunction is deliberately not distributed: not(A and B) does
    not establish not-A and not-B. OR/BUT, missing conjuncts, embedded clauses,
    conflicting times and mixed per-conjunct times are unsupported.
    """
    args = event.arguments
    subject_roles = [r for r in ("subject", "experiencer", "agent") if r in args]
    value_roles = [r for r in ("state", "value") if r in args]
    if (event.predicate not in {"be", "feel"} or event.aspect != "simple"
            or len(subject_roles) != 1 or len(value_roles) != 1
            or set(args) - {subject_roles[0], value_roles[0], "attribute", "time"}
            or args[subject_roles[0]].kind != RefKind.ENTITY):
        return ()
    if "attribute" in args and (args["attribute"].kind != RefKind.LITERAL
                                 or args["attribute"].key != "state"):
        return ()
    value_role = value_roles[0]
    value_ref = args[value_role]
    if value_ref.kind != RefKind.LITERAL:
        # The legacy FEEL parser represents some bare state complements as
        # entity references (e.g. better_1). Only a lower-case, fully licensed
        # adjective construction is projected as a state; this does not change
        # that stored entity or turn arbitrary objects/names into feelings.
        if (event.predicate != "feel" or value_ref.kind != RefKind.ENTITY
                or not value_ref.surface or value_ref.surface != value_ref.surface.lower()):
            return ()
    words = [t.norm for t in lexicon.tokenize(args[value_role].surface or args[value_role].key)]
    if not words or len(words) > 24:
        return ()
    # An explicit temporal argument and the suffix must agree. No old binding
    # is overwritten. Other date expressions are outside this first slice.
    time_ref = args.get("time")
    if time_ref and (time_ref.kind != RefKind.LITERAL or time_ref.key not in DEICTICS):
        return ()
    cue = time_ref.key if time_ref else None
    if words[-1] in DEICTICS:
        if cue and cue != words[-1]:
            return ()
        cue = words.pop()
    # 'anymore' is meaningful to denial; retain it in the original projected
    # complement rather than turning it into an independent time fact.
    anymore = bool(words and words[-1] == "anymore")
    if anymore:
        if event.polarity:
            return ()
        words.pop()
    parts: list[list[str]] = [[]]
    for token in words:
        if token == "and":
            if not parts[-1] or len(parts) >= MAX_COMPONENTS:
                return ()
            parts.append([])
        else:
            parts[-1].append(token)
    if len(parts) > 1 and (not event.polarity or event.modality):
        return ()
    labels = []
    for part in parts:
        if (not part or part[-1] not in STATE_TERMS
                or any(w not in MODIFIERS for w in part[:-1])):
            return ()
        labels.append(part[-1])
    if len(set(labels)) != len(labels):
        return ()
    result = []
    for index, part in enumerate(parts):
        projected_args = dict(args)
        text = " ".join(part + (["anymore"] if anymore else []))
        projected_args[value_role] = SemanticRef.literal(
            lexicon.normalize_phrase(part + (["anymore"] if anymore else [])), text, EntityKind.ABSTRACT)
        if cue:
            projected_args["time"] = SemanticRef.literal(cue, cue, EntityKind.TIME)
        projected = event.copy(arguments=projected_args)
        scoped = state_report(projected, STATE_TERMS)
        if scoped is None or scoped.temporal_scope == "conflicting":
            return ()
        result.append(StateComponent(index, scoped, projected, cue))
    return tuple(result)


def state_question_scope(question):
    """Return subject, temporal class and cue, or abstain from other questions."""
    event = question.event
    if (question.kind != QuestionKind.HOW or question.how_kind != HowKind.STATE
            or question.unresolved or event.predicate not in {"be", "feel"}
            or event.aspect != "simple" or event.modality or not event.polarity
            or event.tense not in {"present", "past"}
            or question.requested_role not in {"state", "value"}
            or not any(v.is_variable for r,v in event.arguments.items() if r in {"state", "value"})):
        return None
    subjects = [v for r, v in event.arguments.items() if r in {"subject", "agent", "experiencer"}]
    if (len(subjects) != 1 or subjects[0].kind != RefKind.ENTITY
            or set(event.arguments) - {"subject", "agent", "experiencer", "state", "value", "time"}
            or any(not v.is_variable for r,v in event.arguments.items() if r in {"state", "value"})):
        return None
    cue_ref = event.arguments.get("time")
    if cue_ref and (cue_ref.kind != RefKind.LITERAL or cue_ref.key not in DEICTICS):
        return None
    cue = cue_ref.key if cue_ref else None
    if event.tense == "present" and cue not in {None, "now", "today"}:
        return None
    if event.tense == "past" and cue in {"tomorrow", "now"}:
        return None
    return subjects[0].key, "current" if event.tense == "present" else "historical", cue


def answer_from_appraisals(question, memory, rows: Iterable[dict]) -> AnswerContract | None:
    """Revalidate indexed graph evidence before binding a complete state set.

    Distinct supported state labels can coexist. A later direct denial removes
    that label only in the same temporal bucket. Reports, modality, or general
    recovery never silently change another person's or historical appraisal.
    Deictic cues are matched within this recorded session; no calendar dates
    or actual feelings are inferred by this module.
    """
    scope = state_question_scope(question)
    if scope is None:
        return None
    person, temporal, cue = scope
    rows = tuple(rows)
    if not rows:
        return None
    active: dict[tuple[str, str | None], tuple[dict, StateComponent, EventFrame]] = {}
    for row in sorted(rows, key=lambda x: (x.get("turn", -1), x.get("id", ""))):
        event = memory.get_event(row.get("event_id", ""))
        if (not event or content_hash(event.to_dict()) != row.get("event_signature_hash")
                or event.source != SourceKind.USER or event.discourse_role not in {"main", "coordinate"}
                or event.modality or row.get("experiencer_id") != person):
            continue
        for part in state_components(event):
            s = part.report
            if (s.entity_id != person or s.state_term != row.get("label")
                    or s.temporal_scope != temporal or part.index != row.get("component_index", 0)
                    or part.temporal_cue != row.get("temporal_cue")
                    or s.polarity != row.get("polarity") or s.tense != row.get("tense")
                    or row.get("source_kind") != event.source.value
                    or row.get("reporter_ids") != ["user"]):
                continue
            if cue and part.temporal_cue != cue and not (cue == "now" and part.temporal_cue is None):
                continue
            key = (s.state_term, part.temporal_cue if temporal == "historical" else None)
            if not s.polarity:
                active.pop(key, None)
            else:
                active[key] = (row, part, event)
    selected = sorted(active.values(), key=lambda x: (x[2].turn_index, x[2].event_id, x[1].index))
    if not selected or len(selected) > MAX_COMPONENTS:
        return AnswerContract(status=AnswerStatus.UNKNOWN, question=question,
            certainty=0, source=SourceKind.UNKNOWN, response_goal="answer",
            reason="no complete bounded, source-qualified state set is available",
            forbidden_claims=["invent_missing_fact"],
            required_slots={"appraisal_lookup": person})
    entries = [{"event_id": event.event_id, "event_hash": content_hash(event.to_dict()),
                "component_index": part.index, "appraisal_id": row["id"]}
               for row, part, event in selected]
    evidence_by_id = {event.event_id: event for _row, _part, event in selected}
    values = []
    for _row, part, _event in selected:
        value = part.event.arguments.get("state") or part.event.arguments["value"]
        if value not in values:
            values.append(value)
    return AnswerContract(status=AnswerStatus.ANSWERED, question=question,
        proposition=selected[0][1].event.copy(discourse_role="main"), values=values,
        evidence=[Evidence(e, matched_roles=["experiencer", "state", "time"], score=1.0)
                  for e in evidence_by_id.values()],
        certainty=min(e.certainty for e in evidence_by_id.values()), source=SourceKind.USER,
        response_goal="answer", reason="bound complete state set through revalidated appraisal records",
        required_slots={"appraisal_lookup": person,
            "retrieved_appraisal_ids": ",".join(row["id"] for row, _, _ in selected),
            "requested_role": question.requested_role or "state",
            "state_bundle": stable_json({"schema": SCHEMA, "question_hash": content_hash(question.to_dict()),
                                         "entries": entries})})


def resolve_state_bundle(contract, memory) -> tuple[EventFrame, ...]:
    """Check bindings again at realization; reject stale or edited evidence."""
    if (contract.status != AnswerStatus.ANSWERED or contract.source != SourceKind.USER
            or contract.question is None or contract.certainty < 160):
        raise ValueError("state bundle requires an evidenced state answer")
    raw = contract.required_slots.get("state_bundle", "")
    if len(raw) > 16384:
        raise ValueError("state bundle budget exceeded")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid state bundle") from exc
    if (not isinstance(data, dict) or set(data) != {"schema", "question_hash", "entries"}
            or data["schema"] != SCHEMA or data["question_hash"] != content_hash(contract.question.to_dict())
            or not isinstance(data["entries"], list) or not 1 <= len(data["entries"]) <= MAX_COMPONENTS):
        raise ValueError("state bundle identity or bounds mismatch")
    scope = state_question_scope(contract.question)
    if scope is None:
        raise ValueError("unsupported state question")
    subject, temporal, cue = scope
    expected_ids = [entry.get("appraisal_id") for entry in data["entries"] if isinstance(entry, dict)]
    if (len(expected_ids) != len(data["entries"]) or any(not isinstance(x, str) for x in expected_ids)
            or contract.required_slots.get("retrieved_appraisal_ids") != ",".join(expected_ids)
            or contract.required_slots.get("appraisal_lookup") != subject):
        raise ValueError("state receipt binding changed")
    output, seen = [], set()
    evidence = {e.event.event_id: e.event for e in contract.evidence}
    for entry in data["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"event_id", "event_hash", "component_index", "appraisal_id"}:
            raise ValueError("invalid state component binding")
        e = memory.get_event(entry["event_id"])
        if (not e or e.event_id not in evidence or content_hash(e.to_dict()) != entry["event_hash"]
                or e.to_dict() != evidence[e.event_id].to_dict() or e.source != SourceKind.USER
                or e.discourse_role not in {"main", "coordinate"}):
            raise ValueError("state evidence changed or lost its source")
        components = state_components(e)
        index = entry["component_index"]
        if type(index) is not int or not 0 <= index < len(components) or (e.event_id, index) in seen:
            raise ValueError("invalid or duplicate state component")
        expected_id = f"appraisal:{e.turn_index}:{e.event_id}" + (f":component:{index}" if index else "")
        if entry["appraisal_id"] != expected_id:
            raise ValueError("appraisal component identity changed")
        seen.add((e.event_id,index))
        component = components[index]
        s = component.report
        if (s.entity_id != subject or not s.polarity or s.modality or s.temporal_scope != temporal
                or (cue and component.temporal_cue != cue and not (cue == "now" and component.temporal_cue is None))):
            raise ValueError("state component does not answer the qualified question")
        output.append(component.event.copy(discourse_role="main"))
    selected_values = {(e.arguments.get("state") or e.arguments["value"]).key for e in output}
    if selected_values != {v.key for v in contract.values}:
        raise ValueError("state answer values were dropped or added")
    if not contract.proposition or contract.proposition.to_dict() != output[0].to_dict():
        raise ValueError("state answer proposition changed")
    return tuple(output)


def resolve_reported_state(contract, memory):
    """Return one source/matrix/content chain, never an unqualified inner fact.

    Restricted to questions explicitly requesting reported/believed finite
    state content. Checks the live relation and original events; IDs supplied
    in a contract alone are not sufficient evidence.
    """
    if (contract.status != AnswerStatus.ANSWERED or contract.source != SourceKind.ATTRIBUTED
            or contract.required_slots.get("attributed") != "true" or contract.question is None
            or contract.proposition is None or contract.certainty < 160):
        raise ValueError("qualified reporting answer required")
    q = contract.question
    if (q.kind != QuestionKind.WHAT or q.requested_role not in {"patient", "content"}
            or not q.event.polarity or q.event.modality or q.unresolved):
        raise ValueError("reporting answer must preserve the outer question")
    reporter = q.event.arguments.get("agent") or q.event.arguments.get("subject") or q.event.arguments.get("experiencer")
    if not reporter or reporter.kind != RefKind.ENTITY:
        raise ValueError("reporting source is unresolved")
    content = memory.get_event(contract.proposition.event_id)
    if (not content or content.to_dict() != contract.proposition.to_dict()
            or content.source != SourceKind.ATTRIBUTED or content.discourse_role != "content"
            or not state_components(content) or len(contract.values) != 1
            or contract.values[0].kind != RefKind.EVENT or contract.values[0].key != content.event_id
            or not any(e.event.to_dict() == content.to_dict() for e in contract.evidence)):
        raise ValueError("reported state content changed or is unsupported")
    matches = [r for r in memory.contents if r.content_event_id == content.event_id
               and r.source_entity_id == reporter.key and r.matrix_predicate == q.event.predicate]
    if len(matches) != 1:
        raise ValueError("reporting relation is missing or ambiguous")
    relation = matches[0]
    matrix = memory.get_event(relation.matrix_event_id)
    allowed = {"say": "reported", "tell": "reported", "report": "reported",
               "think": "believed", "believe": "believed"}
    subject = (matrix.arguments.get("agent") or matrix.arguments.get("subject")) if matrix else None
    if (not matrix or not subject or subject.kind != RefKind.ENTITY or subject.key != reporter.key
            or not relation.attributed or allowed.get(matrix.predicate) != relation.relation_type.value
            or matrix.predicate != relation.matrix_predicate or matrix.source != SourceKind.USER
            or matrix.discourse_role not in {"", "main"} or not matrix.polarity
            or matrix.modality or matrix.aspect != "simple" or matrix.inferred
            or q.event.tense != matrix.tense or content.modality
            or contract.certainty > min(content.certainty, matrix.certainty, relation.certainty)):
        raise ValueError("reporting scope, source or certainty is incompatible")
    # Explicitly reject additional outer constraints which this narrow path
    # has not matched; do not answer a time-qualified query by dropping time.
    if set(q.event.arguments) - {"agent", "subject", "experiencer", "patient", "content"}:
        raise ValueError("unmatched reporting question constraint")
    if any(getattr(r, "main_event_id", None) == matrix.event_id for r in memory.relations):
        raise ValueError("conditional/subordinate report is not unqualified evidence")
    required = {"source_entity_id": reporter.key, "matrix_predicate": matrix.predicate,
                "matrix_tense": matrix.tense, "relation_type": relation.relation_type.value}
    if any(contract.required_slots.get(k) != v for k,v in required.items()):
        raise ValueError("reporting qualification was altered")
    return relation, matrix, content
