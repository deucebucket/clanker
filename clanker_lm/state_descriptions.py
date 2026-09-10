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
from .cessation import CESSATION_ROLE, marker_of, withdrawal_matches
from .recurrence import RECURRENCE_ROLE, recurrence_of
from .continuation import CONTINUATION_ROLE, DISCOURSE_ROLE, continuation_of, discourse_of
from .model import (AnswerContract, AnswerStatus, EntityKind, Evidence, EventFrame,
                    HowKind, QuestionKind, RefKind, SemanticRef, SourceKind)
from .state_scope import ScopedStateReport, state_report

from .state_vocabulary import STATE_TERMS
DEICTICS = frozenset({"now", "today", "yesterday", "tomorrow", "tonight"})
MODIFIERS = frozenset({"very", "really", "quite", "so", "extremely"})
MAX_COMPONENTS = 4
SCHEMA = "qualified-state-components-v4-continuation"


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
    try:
        marker_of(event)
        recurrence = recurrence_of(event)
        ongoing = continuation_of(event)
        fronted = discourse_of(event)
    except ValueError:
        return ()
    subject_roles = [r for r in ("subject", "experiencer", "agent") if r in args]
    value_roles = [r for r in ("state", "value") if r in args]
    if (event.predicate not in {"be", "feel"} or event.aspect != "simple"
            or len(subject_roles) != 1 or len(value_roles) != 1
            or set(args) - {subject_roles[0], value_roles[0], "attribute", "time", CESSATION_ROLE, RECURRENCE_ROLE, CONTINUATION_ROLE, DISCOURSE_ROLE}
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
    if len(parts) > 1 and (not event.polarity or event.modality or recurrence or ongoing or fronted):
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
    blockers=unresolved_current_changes(memory,person) if temporal=='current' else []
    if blockers:
        return AnswerContract(status=AnswerStatus.UNKNOWN,question=question,certainty=0,
            source=SourceKind.UNKNOWN,response_goal="answer",
            reason="a recorded state change has unresolved conjunction scope",
            forbidden_claims=["invent_missing_fact"],
            required_slots={"appraisal_lookup":person,"unresolved_state_changes":stable_json(blockers)})
    rows = tuple(rows)
    if not rows:
        return None
    active: dict[tuple[str, str | None], tuple[dict, StateComponent, EventFrame]] = {}
    revisions=[]
    for row, part, event in _validated_appraisal_rows(rows,memory,person,temporal):
        scoped=part.report
        # For current-state recall, apply every eligible current revision
        # before the query's optional surface cue is considered. A denial
        # without 'now' must still retire an older 'now' report.
        if temporal == 'historical' and cue and part.temporal_cue != cue:
            continue
        key=(scoped.state_term,part.temporal_cue if temporal=='historical' else None)
        if not scoped.polarity:
            previous=active.get(key)
            if previous and withdrawal_matches(_state_ref(part).key,_state_ref(previous[1]).key):
                revisions.append({"withdrawal_id":row["id"], "retired_id":previous[0]["id"],
                                  "kind":marker_of(event) or "denial"})
                active.pop(key,None)
        else:
            active[key]=(row,part,event)
    selected = sorted(active.values(), key=lambda x: (x[2].turn_index, x[2].event_id, x[1].index))
    if not selected or len(selected) > MAX_COMPONENTS:
        return AnswerContract(status=AnswerStatus.UNKNOWN, question=question,
            certainty=0, source=SourceKind.UNKNOWN, response_goal="answer",
            reason="no complete bounded, source-qualified state set is available",
            forbidden_claims=["invent_missing_fact"],
            required_slots={"appraisal_lookup": person,
                            "appraisal_revisions":stable_json(revisions)})
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
            "appraisal_revisions":stable_json(revisions),
            "state_bundle": stable_json({"schema": SCHEMA, "question_hash": content_hash(question.to_dict()),
                                         "evidence_frontier":evidence_frontier(memory,person),
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
    if (not isinstance(data, dict) or set(data) != {"schema", "question_hash", "entries", "evidence_frontier"}
            or data["schema"] != SCHEMA or data["question_hash"] != content_hash(contract.question.to_dict())
            or not isinstance(data["entries"], list) or not 1 <= len(data["entries"]) <= MAX_COMPONENTS):
        raise ValueError("state bundle identity or bounds mismatch")
    scope = state_question_scope(contract.question)
    if scope is None:
        raise ValueError("unsupported state question")
    subject, temporal, cue = scope
    if data["evidence_frontier"] != evidence_frontier(memory,subject):
        raise ValueError("state answer became stale after new evidence")
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
                or (temporal == "historical" and cue and component.temporal_cue != cue)):
            raise ValueError("state component does not answer the qualified question")
        output.append(component.event.copy(discourse_role="main"))
    selected_values = {(e.arguments.get("state") or e.arguments["value"]).key for e in output}
    if selected_values != {v.key for v in contract.values}:
        raise ValueError("state answer values were dropped or added")
    if not contract.proposition or contract.proposition.to_dict() != output[0].to_dict():
        raise ValueError("state answer proposition changed")
    expected=answer_from_appraisals(contract.question,memory,original_state_rows(memory))
    if (expected is None or expected.status != AnswerStatus.ANSWERED
            or expected.required_slots.get("state_bundle") != stable_json(data)):
        raise ValueError("state selection is incomplete or stale against original evidence")
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


def direct_state_evidence(event, memory, *, allow_modal: bool = False) -> bool:
    """Eligibility shared by graph writers, readers and freshness checks."""
    if (event.source != SourceKind.USER or event.discourse_role not in {"main", "coordinate"}
            or (event.modality and not allow_modal) or event.inferred):
        return False
    if any(mark in event.raw_text for mark in ('"', '“', '”')):
        return False
    return not any((getattr(r, "main_event_id", None) == event.event_id
                    or getattr(r, "subordinate_event_id", None) == event.event_id)
                   and r.relation_type.value in {"condition", "exception_condition"}
                   for r in memory.relations)


def evidence_frontier(memory, person: str | None = None) -> str:
    """Bind a projection to relevant original evidence, including later denials.

    A receipt made before a new report must not be reused as a fresh answer.
    This identifies recorded evidence only, not unobserved psychological state.
    """
    return content_hash([e.to_dict() for e in memory.events if direct_state_evidence(e, memory)
                         and any(p.report.entity_id == person or person is None
                                 for p in state_components(e))])


def _validated_appraisal_rows(rows, memory, person=None, temporal="current"):
    candidates = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        e = memory.get_event(row.get("event_id", ""))
        if not e or not direct_state_evidence(e, memory) or row.get("turn") != e.turn_index:
            continue
        if content_hash(e.to_dict()) != row.get("event_signature_hash"):
            continue
        for p in state_components(e):
            s = p.report
            expected_id = f"appraisal:{e.turn_index}:{e.event_id}" + (f":component:{p.index}" if p.index else "")
            if (row.get("id") != expected_id or row.get("experiencer_id") != s.entity_id
                    or person is not None and person != s.entity_id
                    or s.temporal_scope != temporal or row.get("label") != s.state_term
                    or row.get("component_index", 0) != p.index
                    or row.get("polarity") is not s.polarity or row.get("tense") != s.tense
                    or row.get("temporal_cue") != p.temporal_cue
                    or row.get("source_kind") != e.source.value or row.get("reporter_ids") != ["user"]
                    or row.get("state_change") != marker_of(e)
                    or row.get("state_recurrence") != recurrence_of(e)
                    or row.get("state_continuation") != continuation_of(e)
                    or row.get("state_discourse") != discourse_of(e) or expected_id in seen):
                continue
            seen.add(expected_id)
            candidates.append((row,p,e))
    return sorted(candidates, key=lambda x: (x[2].turn_index, x[2].event_id, x[1].index))


def _state_ref(part):
    return part.event.arguments.get("state") or part.event.arguments["value"]


def select_current_claim(question, memory, rows):
    """Resolve simple current WHO/polar state queries from ordered evidence.

    Same experiencer and qualified target are mandatory. Positive reassertion
    after cessation can restore a label. Unknown people are not false people.
    """
    q = question
    if (q.unresolved or q.kind not in {QuestionKind.WHO, QuestionKind.YES_NO}
            or q.event.tense != "present" or q.event.modality
            or CESSATION_ROLE in q.event.arguments):
        return None
    roles = [k for k in ("subject","experiencer","agent") if k in q.event.arguments]
    if len(roles) != 1:
        return None
    role=roles[0]; subject=q.event.arguments[role]
    is_who=q.kind==QuestionKind.WHO
    if is_who:
        if q.requested_role != role or subject.kind != RefKind.VARIABLE:
            return None
        query_event=q.event.copy(arguments={**q.event.arguments,role:SemanticRef.entity("query_person", "query_person")})
        person=None
    else:
        if subject.kind!=RefKind.ENTITY:
            return None
        query_event=q.event;person=subject.key
    parts=state_components(query_event)
    if len(parts)!=1 or parts[0].report.temporal_scope!='current':
        return None
    wanted=_state_ref(parts[0]).key
    blockers=unresolved_current_changes(memory,person,parts[0].report.state_term)
    if blockers:
        return AnswerContract(status=AnswerStatus.UNKNOWN,question=q,source=SourceKind.UNKNOWN,
            certainty=0,response_goal="answer",reason="current state change remains unresolved",
            forbidden_claims=["invent_missing_fact"],required_slots={"unresolved_state_changes":stable_json(blockers)})
    candidates=_validated_appraisal_rows(rows,memory,person)
    if not candidates:
        return AnswerContract(status=AnswerStatus.UNKNOWN, question=q, source=SourceKind.UNKNOWN,
                              certainty=0,reason="no qualifying current-state evidence is available",
                              forbidden_claims=["invent_missing_fact"],response_goal="answer")
    latest={}
    for row,p,e in candidates:
        # A denied narrow degree does not resolve a question about the broad
        # state. A positive qualified report can support the broad state.
        description=_state_ref(p).key
        matches=(withdrawal_matches(wanted,description) if p.report.polarity
                 else withdrawal_matches(description,wanted))
        if matches:
            latest[p.report.entity_id]=(row,p,e)
    selected=[v for v in latest.values() if not is_who or v[1].report.polarity==q.event.polarity]
    requested_recurrence = recurrence_of(query_event)
    if requested_recurrence:
        # An unqualified positive report does not prove renewed occurrence.
        # A current denial of the state is sufficient to deny its recurrence.
        selected = [v for v in selected if not v[1].report.polarity
                    or recurrence_of(v[1].event) == requested_recurrence]
    requested_continuation = continuation_of(query_event)
    if requested_continuation:
        # A plain present-state assertion or a recurrence does not establish
        # uninterrupted continuation. An eligible current denial refutes it.
        selected = [v for v in selected if not v[1].report.polarity
                    or continuation_of(v[1].event) == requested_continuation]
    if len(selected)!=1:
        return AnswerContract(status=AnswerStatus.UNKNOWN,question=q,source=SourceKind.UNKNOWN,
                              certainty=0,reason="no single complete current-state answer is established",
                              forbidden_claims=["invent_missing_fact"],response_goal="answer")
    row,p,e=selected[0]
    status=AnswerStatus.ANSWERED if is_who else AnswerStatus.TRUE if p.report.polarity==q.event.polarity else AnswerStatus.FALSE
    event=p.event.copy(discourse_role="main")
    ref=next(v for k,v in event.arguments.items() if k in {"subject","experiencer","agent"})
    values=[ref] if is_who else []
    guard={"schema":"current-state-evidence-v1", "question_hash":content_hash(q.to_dict()),
           "evidence_frontier":evidence_frontier(memory,person), "person":person,
           "event_id":e.event_id,"event_hash":content_hash(e.to_dict()),"component_index":p.index,
           "status":status.value,"proposition_hash":content_hash(event.to_dict()),
           "values_hash":content_hash([v.to_dict() for v in values])}
    return AnswerContract(status=status,question=q,proposition=event,values=values,
        source=SourceKind.USER,certainty=e.certainty,response_goal="answer",
        evidence=[Evidence(e,matched_roles=[role,"state"],score=1.0)],
        reason="current state resolved through person-scoped report/revision history",
        required_slots={"current_state_guard":stable_json(guard),"appraisal_lookup":person or "person_query",
                        "retrieved_appraisal_ids":row["id"]})


def validate_current_state_guard(contract,memory):
    try:
        raw=contract.required_slots['current_state_guard']
        if len(raw)>16384:raise ValueError('state guard budget exceeded')
        guard=json.loads(raw)
        required={"schema","question_hash","evidence_frontier","person","event_id","event_hash",
                  "component_index","status","proposition_hash","values_hash"}
        if set(guard)!=required or guard['schema']!='current-state-evidence-v1':
            raise ValueError('invalid current-state guard')
        if (not contract.question or not contract.proposition or contract.source!=SourceKind.USER
                or guard['question_hash']!=content_hash(contract.question.to_dict())
                or guard['evidence_frontier']!=evidence_frontier(memory,guard['person'])
                or guard['status']!=contract.status.value
                or guard['proposition_hash']!=content_hash(contract.proposition.to_dict())
                or guard['values_hash']!=content_hash([v.to_dict() for v in contract.values])):
            raise ValueError('current-state answer binding or frontier changed')
        e=memory.get_event(guard['event_id']);idx=guard['component_index']
        parts=state_components(e) if e else ()
        if (not e or content_hash(e.to_dict())!=guard['event_hash']
                or not direct_state_evidence(e,memory) or type(idx) is not int or not 0<=idx<len(parts)
                or not any(ev.event.to_dict()==e.to_dict() for ev in contract.evidence)
                or contract.certainty>e.certainty
                or parts[idx].event.copy(discourse_role='main').to_dict()!=contract.proposition.to_dict()):
            raise ValueError('unmatched current-state evidence')
        # Re-run selection from original event-backed index entries. A forged
        # guard rehashed alongside a wrong status is still not valid evidence.
        expected=select_current_claim(contract.question,memory,original_state_rows(memory))
        if (expected is None or expected.status!=contract.status or not expected.proposition
                or expected.proposition.to_dict()!=contract.proposition.to_dict()
                or [v.to_dict() for v in expected.values]!=[v.to_dict() for v in contract.values]):
            raise ValueError('current-state evidence does not support selected answer')
    except (KeyError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('malformed current-state guard') from exc


def original_state_rows(memory):
    """Reconstruct the qualifying index solely for validation, never learning."""
    rows=[]
    for e in memory.events:
        if not direct_state_evidence(e,memory):
            continue
        for p in state_components(e):
            s=p.report
            rows.append({"id":f"appraisal:{e.turn_index}:{e.event_id}"+(f":component:{p.index}" if p.index else ""),
                         "event_id":e.event_id,"event_signature_hash":content_hash(e.to_dict()),
                         "experiencer_id":s.entity_id,"turn":e.turn_index,"label":s.state_term,
                         "component_index":p.index,"polarity":s.polarity,"tense":s.tense,
                         "temporal_cue":p.temporal_cue,"source_kind":e.source.value,
                         "reporter_ids":["user"],"state_change":marker_of(e),
                         "state_recurrence":recurrence_of(e),
                         "state_continuation":continuation_of(e),
                         "state_discourse":discourse_of(e)})
    return rows


def unresolved_current_changes(memory, person: str | None, label: str | None = None) -> list[str]:
    """A recognized but unresolved compound change cannot leave confident stale recall.

    It does not distribute the negation. Later explicit component evidence may
    resolve that component. Unsupported modal/historical/reporting scope is not
    used as a current direct revision.
    """
    pending: dict[tuple[str,str],str] = {}
    for e in memory.events:
        if not direct_state_evidence(e,memory) or e.tense!='present':
            continue
        subject=e.arguments.get('subject') or e.arguments.get('experiencer') or e.arguments.get('agent')
        if not subject or subject.kind!=RefKind.ENTITY or (person is not None and subject.key!=person):
            continue
        parts=state_components(e)
        if parts:
            for p in parts:
                if p.report.temporal_scope=='current':
                    pending.pop((subject.key,p.report.state_term),None)
        elif any(role in e.arguments for role in (CESSATION_ROLE, RECURRENCE_ROLE, CONTINUATION_ROLE, DISCOURSE_ROLE)):
            value=e.arguments.get('state') or e.arguments.get('value')
            if not value:continue
            words=[t.norm for t in lexicon.tokenize(value.surface or value.key)]
            # Only this bounded current conjunction ambiguity is blocked. It
            # does not invent a calendar interpretation of unknown dates.
            if any(t in words for t in ('and','or','but')) and not set(words)&{'yesterday','tomorrow'}:
                for term in set(words)&STATE_TERMS:
                    pending[(subject.key,term)]=e.event_id
    return sorted({eid for (who,term),eid in pending.items() if label is None or term==label})
