from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str, cwd: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd)
    return subprocess.run(
        [sys.executable, "-m", "clanker_lm", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_schema_command_reports_seeded_graph():
    root = Path(__file__).resolve().parents[1]
    result = run_cli("schema", cwd=root)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["atoms"] >= 90
    assert data["grammar_rules"] >= 10
    assert data["template_tables"] == 0


def test_once_json_contains_full_trace_contract():
    root = Path(__file__).resolve().parents[1]
    result = run_cli("once", "Hello", "--json", cwd=root)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["response"]
    assert "parse" in data and "contract" in data and "candidates" in data


def test_cli_memory_and_learn_commands_support_sourced_qa(tmp_path):
    root = Path(__file__).resolve().parents[1]
    memory = tmp_path / "session.json"
    learned = run_cli(
        "learn", "--text", "The", "launch", "is", "on", "Monday.",
        "--memory", str(memory), "--source", "retrieved", "--certainty", "212",
        cwd=root,
    )
    assert learned.returncode == 0, learned.stderr
    assert memory.exists()
    answered = run_cli("once", "When", "is", "the", "launch?", "--memory", str(memory), "--json", cwd=root)
    assert answered.returncode == 0, answered.stderr
    data = json.loads(answered.stdout)
    assert data["contract"]["status"] == "answered"
    assert data["contract"]["source"] == "retrieved"
    assert "monday" in data["response"].lower()
