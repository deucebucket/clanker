"""Curriculum machinery tests; student shortcomings belong in diagnostic reports."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.school.core import (Check, Curriculum, Episode, Skill, Step, digest,
                                     freeze_learning, grade, parameter_digest)
from experiments.school.unit1 import unit_one
from experiments.school.__main__ import execute, write_atomic


def observe(book, skill, n, ok=True, student="v1"):
    return book.record(student=student, skill=skill, context=f"context-{n}",
        evidence=f"evidence-{skill}-{n}-{ok}",
        checks=[{"kind": "role", "expected": "a", "actual": "a" if ok else "b", "passed": ok}])


def test_dag_rejects_duplicate_missing_and_cyclic_skills():
    with pytest.raises(ValueError, match="duplicate"):
        Curriculum([Skill("a"), Skill("a")])
    with pytest.raises(ValueError, match="missing"):
        Curriculum([Skill("a", ("b",))])
    with pytest.raises(ValueError, match="cyclic"):
        Curriculum([Skill("a", ("b",)), Skill("b", ("a",))])


@pytest.mark.parametrize("limit", [False, 0, -1, 101, "3"])
def test_bounded_minimum_contexts(limit):
    with pytest.raises(ValueError):
        Curriculum(minimum_contexts=limit)


def test_prerequisites_change_actual_scheduler_and_can_regress():
    c = Curriculum([Skill("identity"), Skill("ownership", ("identity",))])
    assert c.next_lesson("v1") == "identity"
    for i in range(3):
        observe(c, "ownership", i)
    assert c.status("v1")["ownership"]["state"] == "blocked"
    for i in range(3):
        observe(c, "identity", i)
    assert c.next_lesson("v1") is None
    observe(c, "identity", 0, False)
    assert c.next_lesson("v1") == "identity"
    assert c.status("v1")["ownership"]["state"] == "blocked"
    assert c.status("v1")["identity"]["contexts"] == 3
    assert len(c.records) == 7  # Revision preserved; not seven independent contexts.


def test_replay_and_relabel_do_not_multiply_support():
    c = Curriculum([Skill("a")])
    observe(c, "a", 1)
    assert observe(c, "a", 1) is False
    old = c.records[0]
    with pytest.raises(ValueError, match="new context"):
        c.record(student="v1", skill="a", context="another label", evidence=old["evidence"],
                 checks=old["checks"])
    assert c.status("v1")["a"]["contexts"] == 1


def test_student_version_cannot_inherit_passes():
    c = Curriculum([Skill("a")])
    for i in range(3):
        observe(c, "a", i)
    assert c.next_lesson("v1") is None
    assert c.next_lesson("v2") == "a"


def test_ledger_roundtrip_and_tamper():
    c = Curriculum([Skill("a"), Skill("b", ("a",))])
    for i in range(4):
        observe(c, "a", i)
    observe(c, "a", 2, False)
    raw = c.to_dict()
    restored = Curriculum.from_dict(raw)
    assert restored.to_dict() == raw
    assert restored.next_lesson("v1") == c.next_lesson("v1")
    bad = copy.deepcopy(raw)
    bad["records"][0]["checks"][0]["passed"] = False
    with pytest.raises(ValueError, match="mismatch"):
        Curriculum.from_dict(bad)


def test_observations_require_real_boolean_checks():
    c = Curriculum([Skill("a")])
    with pytest.raises(ValueError):
        c.record(student="x", skill="a", context="y", evidence="z", checks=[])
    with pytest.raises(ValueError):
        c.record(student="x", skill="a", context="y", evidence="z",
                 checks=[{"kind":"a", "expected":"b", "actual":"b", "passed":1}])


def test_generated_transfer_worlds_are_disjoint_and_reproducible():
    dev, transfer = unit_one(), unit_one(phase="transfer")
    assert len(dev) == len(transfer) == 27
    assert dev == unit_one()
    assert not {e.context for e in dev} & {e.context for e in transfer}
    assert len({e.context for e in dev + transfer}) == 54
    with pytest.raises(ValueError):
        unit_one(phase="heldout")  # Never mislabel an authored exercise as sealed evaluation.


def test_frozen_assessment_preserves_memory_but_not_parameter_writes():
    from clanker_lm import ClankerLM
    with ClankerLM() as runtime:
        freeze_learning(runtime)
        before = parameter_digest(runtime)
        runtime.process("My brother bought a book yesterday.")
        answer = runtime.process("Who bought the book?")
        runtime.process("Glorp means negative and awful.")
        assert answer.contract.status.value == "answered"
        assert parameter_digest(runtime) == before
        assert runtime.memory.turn_index == 3
        assert runtime.memory.events
        assert not runtime.store.learned_senses("glorp")


def test_grader_rejects_wrong_role_even_if_words_appear_in_response():
    from clanker_lm import ClankerLM
    with ClankerLM() as runtime:
        runtime.process("Sarah called John.")
        out = runtime.process("Who called John?")
        assert grade(Check("role", "sarah", "agent"), out, runtime, {})["passed"]
        assert not grade(Check("role", "john", "agent"), out, runtime, {})["passed"]
        assert not grade(Check("same_entity", "object", "patient"), out, runtime,
                         {"object": "not-john"})["passed"]
        with pytest.raises(ValueError):
            grade(Check("invented-grader", "x"), out, runtime, {})


def test_surface_perspective_check_rejects_my():
    from clanker_lm import ClankerLM
    with ClankerLM() as runtime:
        runtime.process("My sister borrowed my car.")
        out = runtime.process("What did she borrow?")
        out.response = "Your sister borrowed my car."
        assert not grade(Check("perspective", "car"), out, runtime, {})["passed"]
        out.response = "Your sister borrowed your car."
        assert grade(Check("perspective", "car"), out, runtime, {})["passed"]


def test_runner_calls_persistent_real_runtime_and_identifies_absent_appraisals(tmp_path):
    from clanker_lm import ClankerLM
    episodes = [e for e in unit_one() if e.skill in {"identity", "appraisal_separation"}]
    report = execute(episodes, ClankerLM, {"test": "explicit-real-v8"})
    assert report["episode_count"] == 6
    assert report["passed_episodes"] == 3
    assert len(report["tutor_cards"]) == 3
    assert report["curriculum"]["identity"]["state"] == "demonstrated_on_authored_cases"
    assert report["curriculum"]["appraisal_separation"]["state"] == "blocked"
    path = tmp_path / "report.json"
    write_atomic(path, b"test")
    write_atomic(path, b"replacement")
    assert path.read_bytes() == b"replacement"


def test_grader_errors_are_not_mislabeled_student_failures():
    from clanker_lm import ClankerLM
    bad = Episode("x", "identity", "x", (Step("Hello.", (Check("bad-grader", "x"),)),))
    with pytest.raises(ValueError, match="grading"):
        execute([bad], ClankerLM, {"test": "invalid-grader"})


def test_runtime_error_is_visible_and_blocks_readiness():
    from clanker_lm import ClankerLM
    class Failing(ClankerLM):
        def process(self, text):
            raise ValueError("synthetic processing failure")
    report = execute([unit_one()[0]], Failing, {"test": "failure-test"})
    assert report["passed_episodes"] == 0
    assert all(c["kind"] == "execution" for c in report["tutor_cards"][0]["failures"])
