from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from clanker_lm import web as web_module
from clanker_lm.__main__ import build_parser, cmd_web


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


def test_web_cli_reads_build_commit_from_deployment_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_commit = "c" * 40
    monkeypatch.setenv("CLANKER_LM_BUILD_COMMIT", build_commit)
    args = build_parser().parse_args(
        [
            "web",
            "--deployed",
            "--public-origin",
            "https://clanker.example.ts.net",
            "--allow-user",
            "owner@example.com",
        ]
    )
    captured: list[Any] = []

    def fake_run_server(config: Any, *, log_level: str) -> None:
        captured.append((config, log_level))

    monkeypatch.setattr(web_module, "run_server", fake_run_server)
    assert cmd_web(args) == 0
    assert captured[0][0].build_commit == build_commit
    assert captured[0][0].resolved_build_commit == build_commit


def test_web_cli_explicit_build_commit_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLANKER_LM_BUILD_COMMIT", "c" * 40)
    args = build_parser().parse_args(["web", "--build-commit", "d" * 40])
    assert args.build_commit == "d" * 40


def test_deployed_web_cli_fails_before_launch_without_build_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLANKER_LM_BUILD_COMMIT", raising=False)
    root = Path(__file__).resolve().parents[1]
    result = run_cli(
        "web",
        "--deployed",
        "--public-origin",
        "https://clanker.example.ts.net",
        "--allow-user",
        "owner@example.com",
        cwd=root,
    )
    assert result.returncode == 2
    assert "requires an exact build_commit" in result.stderr


def test_systemd_example_supplies_the_exact_build_environment() -> None:
    unit = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "systemd"
        / "clanker-lm-web.service.example"
    ).read_text(encoding="utf-8")
    assert "CLANKER_LM_BUILD_COMMIT=<full 40-character SHA" in unit
    assert "--build-commit ${CLANKER_LM_BUILD_COMMIT}" in unit
