"""Executable public API contracts shared by the package and CLI tests.

The command-line layer intentionally consumes only the methods listed here.
Keeping this contract executable prevents a CLI command from drifting away from
``ClankerLM``, ``ConversationMemory``, or ``LanguageStore`` during refactors.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple, Type


CLI_RUNTIME_METHODS: Tuple[str, ...] = (
    "compile_corpus_profile",
    "dumps",
    "explain_last",
    "learn_many",
    "learned_lexicon",
    "load",
    "loads",
    "match_corpus",
    "process",
    "process_many",
    "save",
    "set_tone_profile",
)

CLI_MEMORY_METHODS: Tuple[str, ...] = (
    "begin_turn",
    "dumps",
)

CLI_STORE_METHODS: Tuple[str, ...] = (
    "get_corpus_profile",
    "learned_senses",
    "list_corpus_profiles",
    "term_row",
)

CLI_RUNTIME_ATTRIBUTES: Tuple[str, ...] = (
    "active_profile_id",
    "memory",
    "observed_state",
    "predicted_state",
    "store",
)


def _missing_callables(owner: Type[Any], names: Iterable[str]) -> Tuple[str, ...]:
    return tuple(
        name
        for name in names
        if not callable(getattr(owner, name, None))
    )


def validate_public_api_contract(
    runtime_type: Type[Any],
    memory_type: Type[Any],
    store_type: Type[Any],
) -> None:
    """Fail package import if a documented CLI dependency disappears."""

    missing: Dict[str, Tuple[str, ...]] = {
        runtime_type.__name__: _missing_callables(runtime_type, CLI_RUNTIME_METHODS),
        memory_type.__name__: _missing_callables(memory_type, CLI_MEMORY_METHODS),
        store_type.__name__: _missing_callables(store_type, CLI_STORE_METHODS),
    }
    failures = {
        owner: names
        for owner, names in missing.items()
        if names
    }
    if failures:
        detail = "; ".join(
            f"{owner}: {', '.join(names)}"
            for owner, names in sorted(failures.items())
        )
        raise RuntimeError(f"Clanker-LM public API contract is incomplete: {detail}")


def validate_runtime_instance(runtime: Any) -> None:
    """Verify instance-only state consumed by interactive CLI commands."""

    missing = tuple(
        name
        for name in CLI_RUNTIME_ATTRIBUTES
        if not hasattr(runtime, name)
    )
    if missing:
        raise RuntimeError(
            "Clanker-LM runtime instance is missing CLI state: "
            + ", ".join(missing)
        )
