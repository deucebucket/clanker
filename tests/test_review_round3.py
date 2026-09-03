"""Regression tests for the seventh automated review."""

from __future__ import annotations

import pytest

import clanker_lm.affect as affect_module
import clanker_lm.runtime as runtime_module
from clanker_lm.__main__ import _load_runtime
from clanker_lm.affect import ClankerAffectBackend, HeuristicAffectBackend
from clanker_lm.contracts import validate_public_api_contract
from clanker_lm.database import LanguageStore
from clanker_lm.memory import ConversationMemory


def test_public_api_validation_is_deferred_to_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original = validate_public_api_contract

    def spy(runtime_type, memory_type, store_type):
        calls.append((runtime_type, memory_type, store_type))
        return original(runtime_type, memory_type, store_type)

    monkeypatch.setattr(runtime_module, "validate_public_api_contract", spy)
    runtime = runtime_module.ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        assert calls == [
            (runtime_module.ClankerLM, ConversationMemory, LanguageStore)
        ]
    finally:
        runtime.close()


def test_optional_v8_error_chain_is_safe_without_a_saved_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(affect_module, "_V8_COMPUTE_VADUG", None)
    monkeypatch.setattr(affect_module, "_V8_VADUG", None)
    monkeypatch.setattr(affect_module, "_V8_STATE_TRANSITION", None)
    monkeypatch.setattr(affect_module, "_V8_IMPORT_ERROR", None)
    with pytest.raises(ModuleNotFoundError, match="V8 engine imports are unavailable"):
        ClankerAffectBackend()


def test_missing_snapshot_policy_is_explicit(tmp_path) -> None:
    missing = tmp_path / "missing-session.json"
    with pytest.raises(ValueError, match="Memory snapshot does not exist"):
        _load_runtime(str(missing))

    runtime = _load_runtime(str(missing), create_if_missing=True)
    try:
        assert runtime.store.path == ":memory:"
    finally:
        runtime.close()


def test_language_store_uses_production_lock_timeout() -> None:
    assert LanguageStore.SQLITE_TIMEOUT_SECONDS == 30.0
