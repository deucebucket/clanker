"""Prerequisite scheduling and typed grading, with no student answer templates.

Readiness means success on the named authored tasks, not general intelligence,
independent human agreement, or a calibrated probability of competence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Skill:
    key: str
    requires: tuple[str, ...] = ()


SKILLS = (
    Skill("identity"),
    Skill("perspective", ("identity",)),
    Skill("time_attachment", ("identity",)),
    Skill("borrow_roles", ("identity", "time_attachment")),
    Skill("non_entailment", ("borrow_roles",)),
    Skill("event_continuity", ("borrow_roles", "perspective")),
    Skill("correction", ("event_continuity",)),
    Skill("dialogue_obligations", ("event_continuity",)),
    Skill("appraisal_separation", ("identity", "event_continuity")),
    Skill("academic_transfer", ("correction", "non_entailment",
                                 "dialogue_obligations", "appraisal_separation")),
)


class Curriculum:
    """A DAG with strict prerequisites and lineage-deduplicated observations."""
    def __init__(self, skills: Iterable[Skill] = SKILLS, *, minimum_contexts: int = 3):
        entries = tuple(skills)
        if (type(minimum_contexts) is not int or not 1 <= minimum_contexts <= 100
                or not 1 <= len(entries) <= 100):
            raise ValueError("bounded curriculum required")
        self.skills = {s.key: s for s in entries}
        if len(self.skills) != len(entries):
            raise ValueError("duplicate skill")
        self.minimum_contexts = minimum_contexts
        order: list[str] = []
        visiting: set[str] = set()
        def visit(key: str) -> None:
            if key not in self.skills:
                raise ValueError("missing prerequisite: " + key)
            if key in visiting:
                raise ValueError("cyclic prerequisites")
            if key in order:
                return
            visiting.add(key)
            for parent in self.skills[key].requires:
                visit(parent)
            visiting.remove(key)
            order.append(key)
        for key in self.skills:
            visit(key)
        self.order = tuple(order)
        self.records: list[dict[str, Any]] = []

    def record(self, *, student: str, skill: str, context: str, evidence: str,
               checks: list[dict[str, Any]]) -> bool:
        if skill not in self.skills or not student or not context or not evidence:
            raise ValueError("unbound observation")
        if not checks or len(checks) > 100 or len(self.records) >= 10000:
            raise ValueError("invalid observation budget")
        if any(set(c) != {"kind", "expected", "actual", "passed"}
               or type(c["passed"]) is not bool for c in checks):
            raise ValueError("invalid check record")
        # Latest result replaces the active judgment for an existing context;
        # historical revisions are retained, but replay never adds support.
        payload = dict(student=student, skill=skill, context=context,
                       evidence=evidence, checks=json.loads(canonical(checks)))
        for old in reversed(self.records):
            if (old["student"] == student and old["skill"] == skill
                    and old["evidence"] == evidence and old["context"] != context):
                raise ValueError("same evidence cannot become a new context")
            if all(old[k] == payload[k] for k in ("student", "skill", "context")):
                if all(old[k] == payload[k] for k in payload):
                    return False
                break
        previous = self.records[-1]["sha256"] if self.records else "0" * 64
        record = {**payload, "previous": previous}
        record["sha256"] = digest(record)
        self.records.append(record)
        return True

    def status(self, student: str) -> dict[str, dict[str, Any]]:
        active: dict[tuple[str, str], dict] = {}
        for item in self.records:
            if item["student"] == student:
                active[(item["skill"], item["context"])] = item
        result: dict[str, dict[str, Any]] = {}
        for key in self.order:
            rows = [r for (s, _), r in active.items() if s == key]
            passed = sum(all(c["passed"] for c in r["checks"]) for r in rows)
            blocked = [p for p in self.skills[key].requires
                       if result[p]["state"] != "demonstrated_on_authored_cases"]
            own_ready = len(rows) >= self.minimum_contexts and passed == len(rows)
            state = ("blocked" if blocked else "demonstrated_on_authored_cases" if own_ready
                     else "needs_practice" if rows else "unassessed")
            result[key] = {"state": state, "contexts": len(rows), "passed": passed,
                           "blocked_by": blocked, "own_checks_pass": own_ready}
        return result

    def next_lesson(self, student: str) -> str | None:
        states = self.status(student)
        # Ordered prerequisite roots first; an observed failure outranks a
        # merely unassessed peer. All order/tie rules are explicit and stable.
        eligible = [key for key in self.order if not states[key]["blocked_by"]
                    and states[key]["state"] != "demonstrated_on_authored_cases"]
        return min(eligible, key=lambda k: (states[k]["state"] != "needs_practice",
                                            self.order.index(k))) if eligible else None

    def to_dict(self) -> dict:
        return {"version": 1, "skills": [asdict(self.skills[k]) for k in self.order],
                "minimum_contexts": self.minimum_contexts,
                "records": json.loads(canonical(self.records))}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Curriculum":
        if set(data) != {"version", "skills", "minimum_contexts", "records"} or data["version"] != 1:
            raise ValueError("unsupported ledger")
        obj = cls([Skill(s["key"], tuple(s["requires"])) for s in data["skills"]],
                  minimum_contexts=data["minimum_contexts"])
        if len(data["records"]) > 10000:
            raise ValueError("ledger too large")
        for row in data["records"]:
            if set(row) != {"student", "skill", "context", "evidence", "checks", "previous", "sha256"}:
                raise ValueError("invalid ledger row")
            obj.record(**{k: row[k] for k in ("student", "skill", "context", "evidence", "checks")})
            if obj.records[-1] != row:
                raise ValueError("ledger chain mismatch")
        if len(obj.records) != len(data["records"]):
            raise ValueError("duplicate ledger row")
        return obj


@dataclass(frozen=True)
class Check:
    kind: str
    expected: str
    role: str = ""


@dataclass(frozen=True)
class Step:
    text: str
    checks: tuple[Check, ...] = ()


@dataclass(frozen=True)
class Episode:
    key: str
    skill: str
    context: str
    steps: tuple[Step, ...]


def ref_name(runtime: Any, ref: Any) -> str | None:
    if ref is None:
        return None
    if ref.kind.value == "entity":
        entity = runtime.memory.get_entity(ref.key)
        return entity.canonical_name.lower() if entity else None
    return str(ref.key).lower()


def grade(check: Check, result: Any, runtime: Any, memo: dict) -> dict:
    event = result.contract.proposition
    ref = event.arguments.get(check.role) if event else None
    if check.kind == "status":
        actual = result.contract.status.value
    elif check.kind == "requested_role":
        actual = result.contract.question.requested_role if result.contract.question else None
    elif check.kind == "value":
        actual = ref_name(runtime, result.contract.values[0]) if result.contract.values else None
    elif check.kind == "role":
        actual = ref_name(runtime, ref)
    elif check.kind == "owner":
        entity = runtime.memory.get_entity(ref.key) if ref and ref.kind.value == "entity" else None
        actual = entity.owner_id if entity else None
    elif check.kind == "perspective":
        # A narrow surface obligation, not a whole-answer string comparison.
        import re
        words = re.findall(r"[a-z]+", result.response.lower())
        required = ["your", check.expected]
        actual = (any(words[i:i + 2] == required for i in range(len(words) - 1))
                  and not any(words[i:i + 2] == ["my", check.expected] for i in range(len(words) - 1)))
        return dict(kind=check.kind, expected="addressee possession: " + check.expected,
                    actual=actual, passed=actual)
    elif check.kind == "same_entity":
        actual = bool(ref and ref.kind.value == "entity" and ref.key == memo.get(check.expected))
        return dict(kind=check.kind, expected=check.expected, actual=actual, passed=actual)
    elif check.kind == "no_question":
        actual = "?" not in result.response
        return dict(kind=check.kind, expected=True, actual=actual, passed=actual)
    elif check.kind == "appraisal":
        graph = getattr(runtime, "memory_web", None)
        subjects = sorted({n.get("experiencer_id") for n in graph.nodes.values()
                           if n.get("kind") == "appraisal" and n.get("experiencer_id")}) if graph else []
        names = [runtime.memory.get_entity(s).canonical_name.lower()
                 for s in subjects if runtime.memory.get_entity(s)]
        actual = names
        return dict(kind=check.kind, expected=check.expected, actual=actual,
                    passed=check.expected in actual)
    else:
        raise ValueError("unsupported grading operation: " + check.kind)
    return dict(kind=check.kind + (":" + check.role if check.role else ""),
                expected=check.expected, actual=actual, passed=actual == check.expected)


def freeze_learning(runtime: Any) -> None:
    """Assessment seam: memory works; lexical/usage/transition parameter writes do not.

    Calls the same process loop with write-side learning hooks replaced. This
    is an explicit assessment configuration, not unchanged live behavior.
    """
    from clanker_lm.learning import LearningOutcome
    from clanker_lm.trajectory import TrajectoryFinalization
    if getattr(runtime, "usage", None) is not None:
        raise ValueError("assessment requires usage learning disabled")
    runtime.learner.preparse = lambda *a, **kw: LearningOutcome()
    runtime.learner.postparse = lambda *a, **kw: LearningOutcome()
    runtime.learner.observe_known_context = lambda *a, **kw: {}
    runtime.trajectory.finalize_pending = lambda **kw: TrajectoryFinalization()
    # Existing pinned correction statistics may still be read by adjust_target.


def parameter_digest(runtime: Any) -> str:
    names = ("atoms", "grammar_rules", "gate_rules", "learned_terms", "learned_senses",
             "lexical_evidence", "transition_stats")
    return digest({name: sorted([tuple(row) for row in runtime.store.connection.execute(
        "SELECT * FROM " + name)], key=repr) for name in names})
