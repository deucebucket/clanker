# Modularity foundation — first decomposition

Date: 2026-09-12. Tracking issue: #161.

## Intent and scope

This is source organization, not a new reasoning or language capability. The
integrated parent is `e0757b6aabe7ba856e4031a7c1127fe93f092085`. A native adaptation
uses the same transformations against the independently verified main parser,
model, QA and realizer blobs. Do not conflate the native and integrated runtimes.
The original tuned engine and sealed evaluator are unchanged.

Four existing monoliths were split by responsibility. Public imports remain:

```python
from clanker_lm.model import EventFrame, AnswerContract, AffectVector
from clanker_lm.parser import SemanticParser
from clanker_lm.qa import QuestionAnswerer
from clanker_lm.realize import SurfaceRealizer
```

| Integrated entrypoint | Before (physical lines) | After |
|---|---:|---:|
| parser.py | 5,661 | 69 |
| model.py | 1,746 | 139 |
| qa.py | 2,435 | 30 |
| realize.py | 2,203 | 37 |

The replacement is 67 focused modules (including package initializers), not
four deleted capabilities. The largest extracted module is 484 lines. The
parser's 1,092-line input method is now 130 lines of ingress/preflight/dispatch;
its construction work lives in typed handlers. The largest extracted function
is an inherited 302-line gerund answer operation, not the former giant loop.
Further simplification of that operation can proceed independently.

## Dependency direction

```text
semantic_types/          enum, identity, relation, question, answer, affect records
        ↑               no parser, runtime, database, or UI imports
parsing/                lexical recognition and typed event construction
        ↓
parsing/steps/          ordered assembly through ParserPort + ParseContext
        ↓
answering/             evidence matching and answer contracts
        ↓
realization/           retained legacy composer and surface helpers

runtime.py             existing orchestrator (unchanged in this slice)
decoding/              existing word-level decoder (unchanged)
```

`semantic_types` is deliberately not named `contracts`: this application
already has a separate `contracts.py` public-API validator. An initial package
name collision was caught by the first smoke run and corrected, not hidden.

### Parser responsibilities

- `types.py`: local parse/split results and predicate profile types.
- `profiles.py`: existing reviewed construction tables, no session state.
- `transforms.py`: top-level token and coordination transformations.
- `nouns.py`, `predicate_roles.py`, `clauses.py`: reference binding, role
  attachment and finite clause parsing.
- `embedded.py`, `infinitives.py`, `gerunds.py`, `content.py`, `relatives.py`,
  `appositives.py`, `subordinates.py`: construction-specific recognition.
- `question_routing.py`, `wh_questions.py`: query scope and slot transforms.
- `context.py`: a fresh typed accumulator per input, including the supplied
  memory handle, output lists and counters; no global or cross-user state.
- `ports.py`: narrow structural interface consumed by the assembly handlers.
- `steps/`: one handler for each existing construction. Returns handled/not
  handled, with an explicit final ordinary-clause handler.
- `pipeline.py`: input preparation, ambiguity preflight and ordered dispatch.

The original order is preserved exactly: embedded interrogative, gerund,
infinitival, finite content, relative, appositive, subordinate, ordinary.
Reordering that sequence is a semantic change and requires its own evidence.

Recognition components are stateless mixins composed once by the public
SemanticParser. They retain existing private override seams used by experimental
subclasses. They are not independent microservices or new pluggable parsers.
Cross-component method calls still exist; the typed handler boundary and static
composition make that coupling explicit instead of pretending it disappeared.

### Answering and legacy realization

Answer modules separate routing, embedded questions, gerund evidence,
infinitive evidence, attribution, ordinary fact queries and matching helpers.
Legacy realization separates routing, assembly, factual responses, uncertainty,
conflicts, reference phrases, inflection, complements and validation. This
preserves existing behavior; it neither certifies legacy template freedom nor
makes legacy semantic validation stronger. The newer word decoder continues
to use its existing atomic grammar, VADUGWI and lexical scoring.

## Compatibility and evidence

Data class fields and methods were extracted without semantic changes. Public
aliases preserve existing imports and decoding of trusted older pickle class
references. New class implementation module paths are intentionally visible;
code which asserts old `__module__` strings needs an explicit update. Native
JSON schemas and record IDs do not change. Do not load untrusted pickle files.

Existing parser/answerer/realizer methods are not duplicated in facades. There
is one SemanticParser class. The AST-based extraction checks data definitions,
answer/legacy methods and segment transformations; ordinary tests then execute
the assembled implementation, including subclass overrides and handler order.
No generated code is evaluated dynamically by the runtime.

The integrated paired probe uses 164 literal inputs from named development
tests (not the sealed evaluator), plus a 31-turn persistent discussion. All
164 raw parser/memory results match. All 164 runtime results and all 31
continuous results match after excluding exactly two source-identity fields:
`decoding.runtime_assets_hash` and its enclosing `decoding.receipt_hash`.
Every word decision, score, binding, proof, state and diagnostic still matches.
Fifteen single-turn cases refuse in both builds; matching a failure is parity,
not newly acquired competence. Raw, unnormalized outputs remain in the delivery.

Source layout changes the code fingerprint even without behavioral changes.
Receipts remain version-bound: do not relabel old hashes to claim byte-identical
receipts, disable replay checks, or rewrite the frozen evaluator's identity.

## Executable architecture contract

Run:

```sh
python -m tools.architecture --output /tmp/architecture.json
python -m pytest tests/test_modular_architecture.py -q
```

For new Python modules the default ceilings are 600 physical lines, 32,000
source bytes and 320 lines per function. Byte limits catch compressed giant
single-line source payloads that line counts alone miss. These are project
engineering gates, not universal definitions of good design.

Layer rules reject forbidden dependency directions, static import cycles
involving the extracted packages, wildcard imports and dynamic exec/eval calls
inside those packages. Local function imports are inspected too. Static
analysis does not prove absence of every possible runtime/dynamic dependency.

Eight remaining integrated legacy modules have named, measured, non-growth
exceptions in architecture_policy.json: memory, database, learning, runtime,
web, associations, lexicon and state_descriptions. When an over-budget metric
shrinks its cap must shrink; deleting a module requires deleting its exception.
There is no blanket exclusion that permits another large new file.

These are explicit remaining debts, not a claim of a monolith-free repository.
Memory/persistence and the central runtime are the next responsibility-level
extractions. Existing engine sources are intentionally out of scope.

## Release

The exact-head architecture tests, applicable regressions, installed wheel and
paired evidence are required before promotion. A native PR may retain the
existing frozen-production binding failure because added/moved source changes
identity. That must be resolved through #123 rather than bypassing a check.
No deployment, full release readiness, calibrated confidence or new language
learning is established by this refactor.
