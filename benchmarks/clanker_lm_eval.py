#!/usr/bin/env python3
"""Deterministic end-to-end acceptance harness for Clanker-LM.

This is deliberately separate from pytest so a user can inspect the exact
conversation, expected semantic status, and generated answer from one command.
It exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clanker_lm import ClankerLM, HeuristicAffectBackend
from clanker_lm.model import AnswerStatus


FIXED_NOW = datetime(2026, 9, 3, 16, 5, tzinfo=timezone.utc)
FIXED_DATE_TEXT = f"{FIXED_NOW.strftime('%B')} {FIXED_NOW.day}, {FIXED_NOW.year}"


CASES: List[Dict[str, Any]] = [
    {
        "name": "typed conversational slots",
        "turns": [
            ("My sister bought a used Honda yesterday.", "acknowledged", None),
            ("Who bought the Honda?", "answered", "sister"),
            ("What did she buy?", "answered", "honda"),
            ("When did she buy it?", "answered", "yesterday"),
            ("Why did she buy it?", "unknown", "haven't told"),
            ("She bought it because her old car broke down.", "acknowledged", None),
            ("Why did she buy it?", "answered", "broke down"),
            ("Did my mother buy it?", "unknown", "don't know"),
        ],
    },
    {
        "name": "explicit false and conflict",
        "turns": [
            ("Sarah did not open the door.", "acknowledged", None),
            ("Did Sarah open the door?", "false", "no"),
            ("Sarah opened the door.", "acknowledged", None),
            ("Did Sarah open the door?", "conflict", "conflicting"),
        ],
    },
    {
        "name": "pronoun trap",
        "turns": [
            ("She pissed me off again.", "missing_reference", "who"),
            ("My sister called me.", "acknowledged", None),
            ("She pissed me off again.", "acknowledged", None),
            ("Who pissed me off again?", "answered", "sister"),
        ],
    },
    {
        "name": "why how and place roles",
        "turns": [
            ("Sarah went to the store to buy groceries.", "acknowledged", None),
            ("Where did Sarah go?", "answered", "store"),
            ("Why did Sarah go to the store?", "answered", "groceries"),
            ("Sarah opened the box with a key.", "acknowledged", None),
            ("How did Sarah open the box?", "answered", "key"),
        ],
    },
    {
        "name": "active lexical learning",
        "turns": [
            ("That movie was glorp.", "lexical_probe", "what does glorp mean"),
            ("Negative, like disappointing and overhyped.", "lexical_learned", "negative evaluation"),
            ("The sequel was glorp.", "acknowledged", None),
            ("What does glorp mean?", "answered", "negative evaluation"),
        ],
        "assert_learning": True,
    },
    {
        "name": "live semantic resolvers",
        "turns": [
            ("What time is it in Tokyo?", "answered", "1:05 AM"),
            ("What's today's date?", "answered", FIXED_DATE_TEXT),
            ("What is 2 + 3 * 4?", "answered", "14"),
        ],
        "assert_resolvers": True,
    },
    {
        "name": "collision masking",
        "turns": [
            ("Bruh, my mom is really sick.", "acknowledged", None),
        ],
        "assert_last_gate": {"masking": True, "register": "casual"},
    },
]


def run() -> int:
    failures: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []

    for case in CASES:
        runtime = ClankerLM(
            affect_backend=HeuristicAffectBackend(),
            clock=lambda: FIXED_NOW,
        )
        case_report: Dict[str, Any] = {"name": case["name"], "turns": []}
        try:
            for text, expected_status, required_text in case["turns"]:
                result = runtime.process(text)
                actual_status = result.contract.status.value
                answer_ok = required_text is None or required_text.lower() in result.response.lower()
                ok = actual_status == expected_status and answer_ok
                row = {
                    "input": text,
                    "response": result.response,
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "ok": ok,
                }
                case_report["turns"].append(row)
                if not ok:
                    failures.append({"case": case["name"], **row})

            if case.get("assert_learning"):
                learned = runtime.store.learned_senses("glorp")
                if not learned or learned[0].confidence < 0.68:
                    failures.append({"case": case["name"], "learning": "sense was not promoted"})
            if case.get("assert_resolvers"):
                count = runtime.store.schema_summary()["resolver_observations"]
                if count != len(case["turns"]):
                    failures.append({
                        "case": case["name"],
                        "resolver_observations": count,
                        "expected": len(case["turns"]),
                    })

            expected_gate = case.get("assert_last_gate")
            if expected_gate:
                assert runtime.last_result is not None
                actual_gate = runtime.last_result.gates.to_dict()
                for key, value in expected_gate.items():
                    if actual_gate.get(key) != value:
                        failures.append({
                            "case": case["name"],
                            "gate": key,
                            "expected": value,
                            "actual": actual_gate.get(key),
                        })
            if runtime.store.schema_summary()["template_tables"] != 0:
                failures.append({"case": case["name"], "template_free": False})
            reports.append(case_report)
        finally:
            runtime.close()

    output = {
        "cases": len(CASES),
        "turns": sum(len(case["turns"]) for case in CASES),
        "passed": not failures,
        "failures": failures,
        "results": reports,
    }
    print(json.dumps(output, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run())
