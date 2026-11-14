# Bounded goal workspace — first deliberation slice

Issue #151. This PR publishes the exact independently portable kernel used by
the separately delivered integrated concept/graph/word-decoder runtime. It does
not install that unmerged runtime on main, supply a second conversational
engine, or claim that an unused module alone improves default chat.

## Executable kernel

`solve(rows, scope, ledger_id, Goal(...), limits=..., find_bridge=...)` accepts
explicit scalar evidence for member/subclass relations and returns a bounded
executed derivation and optional clarification candidate. `verify` reruns the
computation against the same scoped evidence rather than only checking a hash.

1. Pin evidence and build directional indexes.
2. Traverse backward from the requested target to identify relevant subgoals.
3. Search supported forward paths, recording actual rule applications,
   counterevidence, cycles and irrelevant-path pruning.
4. Distinguish supported, explicitly denied, conflicting and unknown goals.
5. When a fully assessed goal is unknown, identify possible one-link bridges
   between supported anchors and trial them in a disposable evidence view.
6. Return an unaccepted clarification candidate only when the trial establishes
   how that missing premise could complete the requested proof.

A candidate assumption is never inserted into the caller's evidence. An absent
relationship does not mean false. A negative subclass assertion is not a
universal disjointness axiom. Intermediate opposing evidence is retained;
alternative clean paths are checked under the declared finite rule system.
The result is relative to supplied premises, not independent verification of
real-world facts. Explicit lookup scopes are necessary inputs, not a substitute
for a service's authentication and authorization.

## Bounds and receipts

Default limits: 1024 evidence records, 1024 instrumented work operations,
16-edge proof depth, at most16 bridge trials. All candidate-pair and traversal
operations count toward the work budget. Input provenance is bounded scalar
metadata; this module loads no prose, model, tools or external code.

Evidence indexing is linear in the admitted records and is rebuilt per call;
this is not a constant-time retrieval claim. Search budgets terminate work,
not merely logging. Exhausted absence checks report unknown; a pruned search
is not presented as proof that a fact is false. The trace is an operational
record, not subjective thought or a claim of frontier-model equivalence.

Bridge priority is an explicit structural utility (anchored paths and their
length), not a calibrated probability that the proposed premise is true.
Lexical likelihood and VADUGWI belong to the subsequent eligible realization
step and cannot establish the bridge as factual evidence.

## Integration already executed locally, not activated by this native PR

The restored integrated parent is306f9aac3dd9f720ebc310b6b6eb9def19e3783a.
Its new Deliberator stores a bounded pending goal per account compartment,
asks a typed missing-premise question, accepts only an adjacent still-bound
explicit yes/no or a relevant direct lesson, and resumes the original goal.
Questions/hypotheses and tutor-like suggestions are not automatically taught.
Expired or stale yes/no cannot mutate knowledge. Closure cancels the pending
question. A newly taught alternative route can also complete the goal.

Why-classification questions compose the actual supporting premises through
the existing atomic grammar and lexical/VADUGWI decoder. No replacement
sentence, finished reply lookup or legacy sentence-realizer fallback is added.
The semantic plan and rule families remain manually implemented: this is not
autonomous induction of arbitrary logic or language.

The Memory Atlas displays the same executed agenda and evidence. Withdrawal
marks the displayed prior proof stale immediately; the next answer recomputes
from active support. Scratch visualization is bounded, while source evidence
and receipts preserve their established history. These graph/word/controller
files remain in the complete local delivery pending branch reconciliation.

## Executed development evidence

Before/after on identical candidate code with the reasoning flag disabled or
enabled, same authored lesson generator and original count pack:
30 short continuous episodes,147 runtime calls per configuration plus3 explicit
withdrawals. Passing episodes3/30 ->30/30; query/control checks129/216 ->216/216.
The cases test missing premises, explicit confirmations, distractions, source
withdrawal, contradictory lessons, alternate bridges, stale replies and closure.
Three lexical worlds share a generator; this is authored transfer evidence,
not an independent held-out human or academic result.

A separate200-turn growing study completed all expected statuses. Saving at100
and restoring a second runtime reproduced all100 subsequent replies, contracts,
word receipts, workspace state, memory, graph, concept ledger and predicted
state. Original-user synthetic fixtures only; no sealed training data or
self-output reinforcement. Performance measurements are supplied separately
and are not a concurrency/large-corpus demonstration.

Portable native tests:23 passed locally. Additional integrated tests directly
exercise the real V8 runtime, empty count pack, disabled legacy sentence
generators, account isolation, rollback, snapshots and actual explanation output.
Only the portable23-test module is included in this native PR. Do not claim
that its isolated tests establish the full default chat integration.

## Development log and boundaries

The existing prover answered isolated queries; it did not retain the original
question while seeking missing evidence or compose a why-answer from its
actual derivation. This kernel and its local control adapter fill those seams.
A newly written account test initially used a nonexistent export_snapshot API;
it was corrected to the actual persistence API. That test repair is not a
student-quality improvement.

Only member/subclass reasoning is supported. The conversational adapter has
one pending question, an eight-turn expiry, a bounded interrogative grammar,
and six-premise/48-token explanation limits. Arbitrary compound goals,
quantifiers, multiword senses, arithmetic integration, open-domain truth and
new rule induction remain future work. Budgeted failure is explicit and does
not silently omit necessary premises. No external autonomous workflow runs.

## Release

This module adds production-package bytes but does not alter tuned engine/
or sealed evaluation/ files. Existing historical evaluator identity and data
must remain intact. Candidate-runtime binding #123, exact-head review/CI and
#125 reconciliation are prerequisites for default activation. Focused kernel
success must not be labelled a green full release. No existing failing gate
is removed or skipped by this PR.
