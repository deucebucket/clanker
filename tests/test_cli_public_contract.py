"""Executable contract tests for every CLI-to-runtime dependency."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from clanker_lm import (
    ClankerLM,
    ConversationMemory,
    HeuristicAffectBackend,
    LanguageStore,
    validate_public_api_contract,
    validate_runtime_instance,
)
from clanker_lm.contracts import (
    CLI_MEMORY_METHODS,
    CLI_RUNTIME_METHODS,
    CLI_STORE_METHODS,
)
from clanker_lm.model import AnswerStatus


def run_cli(
    *args: str,
    cwd: Path,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd)
    return subprocess.run(
        [sys.executable, "-m", "clanker_lm", *args],
        cwd=cwd,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


def test_published_cli_api_contract_is_executable() -> None:
    validate_public_api_contract(ClankerLM, ConversationMemory, LanguageStore)
    for owner, names in (
        (ClankerLM, CLI_RUNTIME_METHODS),
        (ConversationMemory, CLI_MEMORY_METHODS),
        (LanguageStore, CLI_STORE_METHODS),
    ):
        assert all(callable(getattr(owner, name, None)) for name in names)

    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        validate_runtime_instance(runtime)
        assert callable(runtime.store.term_row)
        assert callable(runtime.store.list_corpus_profiles)
        assert callable(runtime.memory.dumps)
    finally:
        runtime.close()


def test_all_noninteractive_cli_routes_execute_against_real_api(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    memory = tmp_path / "contract-session.json"
    dialogue = tmp_path / "dialogue.txt"
    dialogue.write_text(
        'A: "I am here."\nB: "Why?"\nA: "Because I care."\n',
        encoding="utf-8",
    )
    script = tmp_path / "script.txt"
    script.write_text(
        "My sister bought a Honda yesterday.\nWho bought the Honda?\n",
        encoding="utf-8",
    )
    facts = tmp_path / "facts.txt"
    facts.write_text("The launch is on Monday.\n", encoding="utf-8")

    profile = run_cli(
        "profile",
        str(dialogue),
        "--name",
        "Contract Profile",
        "--profile-id",
        "contract-profile",
        "--memory",
        str(memory),
        cwd=root,
    )
    assert_ok(profile)
    assert json.loads(profile.stdout)["profile_id"] == "contract-profile"

    commands = (
        ("profiles", "--memory", str(memory)),
        ("match", str(dialogue), "--memory", str(memory)),
        ("tone", "contract-profile", "--memory", str(memory)),
        ("tone", "off", "--memory", str(memory)),
        ("learn", "--path", str(facts), "--memory", str(memory)),
        ("once", "When", "is", "the", "launch?", "--memory", str(memory), "--json"),
        ("parse", "Who", "bought", "the", "Honda?", "--memory", str(memory)),
        ("script", str(script), "--memory", str(memory), "--json"),
        ("lexicon", "--memory", str(memory)),
        ("schema",),
        ("demo",),
    )
    for command in commands:
        assert_ok(run_cli(*command, cwd=root))

    chat = run_cli(
        "chat",
        "--memory",
        str(memory),
        cwd=root,
        input_text=(
            "Hello\n"
            "/why\n"
            "/state\n"
            "/memory\n"
            "/lexicon\n"
            "/profiles\n"
            "/quit\n"
        ),
    )
    assert_ok(chat)


def test_default_store_is_intentionally_ephemeral_and_snapshot_restores_state() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        assert runtime.store.path == ":memory:"
        runtime.process("My sister bought a used Honda yesterday.")
        snapshot = runtime.dumps(indent=None)
    finally:
        runtime.close()

    fresh = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        unknown = fresh.process("Who bought the Honda?")
        assert unknown.contract.status != AnswerStatus.ANSWERED
    finally:
        fresh.close()

    restored = ClankerLM.loads(
        snapshot,
        affect_backend=HeuristicAffectBackend(),
    )
    try:
        answered = restored.process("Who bought the Honda?")
        assert answered.contract.status == AnswerStatus.ANSWERED
        assert "sister" in answered.response.lower()
    finally:
        restored.close()
