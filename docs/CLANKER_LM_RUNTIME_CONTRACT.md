# Clanker-LM Runtime and CLI Contract

This document defines the persistence model and the executable API boundary used by `python -m clanker_lm`.

## Persistence is explicit

`ClankerLM()` deliberately starts with an in-memory `LanguageStore`. This is the safe library and CLI default:

```python
runtime = ClankerLM()
assert runtime.store.path == ":memory:"
```

Without a snapshot path, closing the process discards session entities, facts, learned lexical senses, resolver observations, transition statistics, and corpus profiles. This avoids silently writing conversational data to an unexpected filesystem location.

Cross-process persistence is opt-in through one of two mechanisms:

1. Pass `--memory /path/to/session.json` to CLI commands. The runtime loads that snapshot before the command and writes the updated snapshot afterward.
2. Library callers that need a continuously persistent SQL overlay may inject `LanguageStore("/path/to/clanker-lm.sqlite3")` into `ClankerLM(language_store=store)`.

Commands whose purpose depends on cross-session state—such as `learn`, `lexicon`, `profile`, `profiles`, `match`, and `tone`—require `--memory`. `once`, `chat`, and `script` accept it optionally. No hidden default state file is created.

A version-2 runtime snapshot contains:

- symbolic conversation memory;
- observed and predicted VADUGWI state;
- learned language overlay;
- lexical learner state;
- resolver and trajectory overlay data;
- active corpus profile;
- default timezone.

## Executable API contract

`clanker_lm.contracts.validate_public_api_contract()` runs during package import. It fails immediately if a refactor removes a method consumed by the CLI.

### `ClankerLM`

The CLI contract requires these callable methods:

```text
process
process_many
learn_many
learned_lexicon
compile_corpus_profile
match_corpus
set_tone_profile
explain_last
dumps
loads
save
load
```

Interactive commands also require these instance attributes:

```text
memory
store
observed_state
predicted_state
active_profile_id
```

### `ConversationMemory`

```text
begin_turn
dumps
```

### `LanguageStore`

```text
term_row
learned_senses
get_corpus_profile
list_corpus_profiles
```

The lexical learner's `term_row()` dependency is therefore part of the same executable package contract rather than an assumed private method.

## CLI-to-API mapping

| CLI route | Principal runtime API exercised |
|---|---|
| `once` | `process`, `loads`, `save` |
| `parse` | `memory.begin_turn`, semantic parser |
| `chat` | `process`, `explain_last`, state attributes, `memory.dumps`, `learned_lexicon`, `list_corpus_profiles` |
| `script` | `process_many` |
| `learn` | `learn_many` |
| `lexicon` | `learned_lexicon` |
| `profile` | `compile_corpus_profile` |
| `profiles` | `list_corpus_profiles` |
| `match` | `match_corpus` |
| `tone` | `set_tone_profile` |
| `schema` | `LanguageStore.schema_summary` |
| `demo` | `process` |

`tests/test_cli_public_contract.py` launches every command route against the installed package, exercises the interactive inspection commands, and verifies both ephemeral-default and snapshot-restored behavior.

## Design rationale

The runtime separates three forms of state:

- **Ephemeral process state:** the default for experimentation and privacy.
- **Portable snapshot state:** explicit JSON persistence through `--memory` or `save()`/`load()`.
- **Injected database state:** a caller-controlled SQLite path for services that need continuous persistence.

This is intentional behavior, not an accidental omission of a database path.

## Missing snapshot paths

Snapshot creation is an explicit command policy rather than a silent global
fallback:

- `once`, `chat`, `script`, `learn`, and `profile` may create a new snapshot at
  a supplied `--memory` path and save it when the command completes.
- `parse`, `lexicon`, `profiles`, `match`, and `tone` require the supplied
  snapshot to exist. A missing path produces a clear command-line error.
- Omitting `--memory` from commands where it is optional still creates an
  intentionally ephemeral in-memory runtime.

The internal helper expresses this choice as
`_load_runtime(path, create_if_missing=True|False)`, so each command's behavior
is visible at its call site and covered by tests.
