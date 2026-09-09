"""Run an authored, frozen-learning continuous-session curriculum diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any, Callable

from .core import Curriculum, canonical, digest, freeze_learning, grade, parameter_digest
from .unit1 import unit_one


def student_identity() -> dict:
    import clanker_lm
    root = Path(clanker_lm.__file__).resolve().parent.parent
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ("engine", "clanker_lm") for p in sorted((root / folder).rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts}
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-a-git-checkout"
    return {"commit": commit, "source_sha256": digest(files),
            "python": platform.python_version(), "source_files": len(files)}


def execute(episodes, factory: Callable[[], Any], identity: dict) -> dict:
    curriculum = Curriculum()
    results = []
    student = digest(identity)
    for episode in episodes:
        with factory() as runtime:
            if runtime.affect_backend_name != "clanker-v8":
                raise ValueError("school report requires the real V8 backend")
            freeze_learning(runtime)
            before = parameter_digest(runtime)
            memo: dict = {}
            turns = []
            checks = []
            for index, step in enumerate(episode.steps, 1):
                start = time.perf_counter_ns()
                try:
                    out = runtime.process(step.text)
                except Exception as exc:
                    # Exceptions are observed failures, never a hidden skipped turn.
                    graded = [{"kind": "execution", "expected": "completed turn",
                               "actual": type(exc).__name__ + ": " + str(exc), "passed": False}]
                    turn = {"turn": index, "input": step.text, "error": graded[0]["actual"],
                            "checks": graded}
                else:
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    event = out.contract.proposition
                    if event is not None and event.predicate == "borrow":
                        patient = event.arguments.get("patient")
                        if patient and patient.kind.value == "entity":
                            memo.setdefault("borrowed_object", patient.key)
                    # Grader/configuration errors are not student failures.
                    graded = [grade(check, out, runtime, memo) for check in step.checks]
                    turn = {"turn": index, "input": step.text, "response": out.response,
                            "status": out.contract.status.value, "act": out.gates.response_act,
                            "checks": graded, "milliseconds": elapsed,
                            "estimated_vadugwi": out.observed_state.to_dict(),
                            "event": event.to_dict() if event else None}
                turns.append(turn)
                checks.extend(graded)
            if parameter_digest(runtime) != before:
                raise RuntimeError("assessment changed a learned parameter table")
        evidence = digest([{k: v for k, v in t.items() if k != "milliseconds"} for t in turns])
        curriculum.record(student=student, skill=episode.skill, context=episode.context,
                          evidence=evidence, checks=checks)
        results.append({"episode": episode.key, "skill": episode.skill, "context": episode.context,
                        "evidence_sha256": evidence, "passed": all(c["passed"] for c in checks),
                        "turns": turns})
    statuses = curriculum.status(student)
    tutor_cards = []
    for result in results:
        failures = [{"turn": t["turn"], **c} for t in result["turns"]
                    for c in t["checks"] if not c["passed"]]
        if failures:
            tutor_cards.append({"episode": result["episode"], "skill": result["skill"],
                                "evidence_sha256": result["evidence_sha256"], "failures": failures,
                                "blocked_by": statuses[result["skill"]]["blocked_by"],
                                "action": "inspect prerequisite and execution path; no automatic weight update",
                                "teacher_sentence_stored_as_reply": False})
    return {"format": "clanker-school-v1", "student": identity, "student_id": student,
            "scope": "Authored development/transfer diagnostic; not independent held-out research evidence",
            "assessment_policy": "fresh runtime per episode; continuous memory inside episode; parameter writes frozen",
            "episode_count": len(results), "passed_episodes": sum(r["passed"] for r in results),
            "turn_count": sum(len(r["turns"]) for r in results),
            "check_count": sum(len(t["checks"]) for r in results for t in r["turns"]),
            "passed_checks": sum(c["passed"] for r in results for t in r["turns"] for c in t["checks"]),
            "curriculum": statuses, "next_lesson": curriculum.next_lesson(student),
            "tutor_cards": tutor_cards, "episodes": results, "ledger": curriculum.to_dict()}


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "transfer", "both"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder-pack", type=Path, help="Explicitly use the separately delivered decoder if installed")
    parser.add_argument("--require-ready", action="store_true", help="exit 2 if any exercised skill fails")
    parser.add_argument("--candidate", choices=("none", "temporal-suffix-v1"), default="none")
    args = parser.parse_args(argv)
    from clanker_lm import ClankerLM
    kwargs: dict[str, Any] = {"clock": lambda: datetime(2026, 9, 9, 12, tzinfo=timezone.utc)}
    identity = student_identity()
    identity["adapter"] = "production-default"
    identity["candidate"] = args.candidate
    if args.decoder_pack:
        # Explicit optional adapter; no fallback to a different runtime.
        from clanker_lm.decoding import AffinityPack, WordDecoder
        pack = AffinityPack.load_sqlite(args.decoder_pack)
        kwargs["word_decoder"] = WordDecoder(pack)
        identity.update(adapter="delivered-word-decoder", pack_sha256=pack.fingerprint)
    phases = ("development", "transfer") if args.phase == "both" else (args.phase,)
    episodes = tuple(e for phase in phases for e in unit_one(phase=phase))
    identity["curriculum_sha256"] = digest([e.context for e in episodes])
    identity["grader_sha256"] = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))})
    def factory():
        runtime = ClankerLM(**kwargs)
        if args.candidate == "temporal-suffix-v1":
            from .temporal_repair import TemporalSuffixParser
            runtime.parser = TemporalSuffixParser()
        return runtime
    report = execute(episodes, factory, identity)
    write_atomic(args.output, canonical(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("episode_count", "passed_episodes", "turn_count",
                                             "check_count", "passed_checks", "next_lesson")}, indent=2))
    return 2 if args.require_ready and report["passed_episodes"] != report["episode_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
