# Task-gated memory activation: first gray-thread slice

Date: 2026-09-11. Refs #154/#155. Companion issues #156 (same-name recognition),
#157 (calibration), #158 (session purpose), #159 (Atlas/report cards). Identity
and sense-lineage repairs remain #153; private service boundary remains #145.

## Native publication scope

This commit adds the exact portable activation kernel and its 29 tests to the
reviewed main lineage. It does not claim that main contains the separately
delivered MemoryWeb, CompartmentStore or word decoder. The local integrated
adoption is based on verified round-nine commit 07ed634088cf53a1cccee045a7841d1d497360b8.
No competing chat runtime is created. No engine/ or evaluation/ bytes change.

## Executable policy

Inputs: an already authorized metadata graph, typed roots/task, server-supplied
policy context/revision/allowed tasks, and depth/node/edge budgets. Personal,
semantic and simulation/auxiliary kinds are distinct. Unknown/unsupported
planes never silently become personal or academic evidence.

- PERSONAL_RECALL follows same-plane personal associations. A forward
  CONTEXT_REFERENCE may select a semantic target only as reference_only.
  That target is not queued for expansion.
- CONCEPT_EXPLANATION begins at a semantic root and follows semantic links.
  Incoming context references are not even indexed for reverse traversal.
- CROSS_DOMAIN_COMPARISON explicitly permits forward contextual expansion from
  a personal root into semantic knowledge, retaining the edge's reference type.
- An ordinary cross-plane association cannot bypass this policy. No numerical
  weight can override it; weights do not occur in the eligibility calculation.
- Gray links are not membership/subclass evidence, identity merges, consent or
  authentication. A selected node is not thereby a true proposition.

The index selects IDs; payloads remain in the owning store. A transient working
set is returned, not added to durable memories. Its receipt binds graph content,
policy, request, traversal decisions, selected/reference-only IDs, actual work
and explicit incomplete stop reasons. Replay recomputes rather than trusting a
rehash. Changed graph/policy invalidates the old receipt. This is integrity,
not authentication of a caller or proof of factual truth.

## Authorization and cost boundary

The caller MUST enforce scope before building the index. A context string is
not an authenticator. The kernel cannot discover or authorize other accounts.
The separate integrated service checks all scoped roots and policy context
before reading any compartment row. It calls the existing scoped snapshot
adapter, not a database containing everybody's nodes followed by filtering.

Index construction is O(N+E) over the current bounded authorized graph, plus
canonical sorting/hashing costs. The first adapter rebuilds the index per read;
it does not yet provide persistent incremental indexes. Traversal has separate
budgets. With 1,000 additional semantic leaves, the authored personal task still
expands only three personal nodes and inspects five adjacency entries, exposing
one reference-only concept. This excludes index-build work and is not a full
chat-latency, concurrency or huge-corpus claim.

## Connected local adoption

The integrated source adds MemoryWeb.add_context_reference/working_set,
ClankerLM.memory_working_set and CompartmentStore.add_context_reference/working_set.
References require existing endpoints and an existing event evidence ID. Scoped
writes are revision-checked, signed through the existing envelope, atomic and
idempotent; they do not add concept premises. A reference-only export contains
only ID, kind, label and activation/plane metadata, not the target's full record.
The existing Atlas can display only the actual selected graph and gray dashed
reference edges. No inaccessible nodes are fetched by the offline UI.

These are explicit trusted API operations. Automatic interpretation of arbitrary
natural-language task intent, whole-chat answer/planner gating, simulation
routing, consent enforcement, shared/public graph mounting and confidence
calibration are NOT implemented by this first slice. Existing broad inspector
neighborhood remains an explicitly separate inspection API, not this retrieval
policy. Do not claim that a pure kernel alone fixes conversational topic drift.

## Executed tests

Locally: 29 portable tests + 13 existing-runtime/graph/store integration tests =
42 passes. A wider relevant selection including memory compartments, concepts,
deliberation, web account binding and V8 reports 532 passes and two pre-existing
expected V8 failures. An attempted functional suite timed out at 32 seconds;
that run is not counted as success. No completed full release/browser result is
claimed here.

Coverage includes an independent reachability oracle on 60 small generated
graphs, cycles/self-links, policy-before-existence checks, task changes, limits,
rehashed tampering, changed generations, weight noninterference, reference-only
payload projection, snapshot replay and interleaved accounts with unchanged
signed states. The graph annotation does not change concept proof results.
Legacy sentence generators are disabled in the integrated chat smoke test.

An initial test fixture violated the existing fresh-runtime factory contract by
preloading a conversation. The fixture now teaches through the actual service
and creates the gray link through the explicit scoped annotation API. The
factory protection was retained, not weakened.

## Remaining work and release

The six linked issues stay open: a tested first policy is not full recognition,
learning, session-purpose or Atlas completion. No 'sureness percentage' has been
invented; no VADUGWI collision is treated as identity proof. Private appraisals
remain per experiencer/episode; that richer recognition is #156/#157.

No completed response template, external corpus, biometrics, automatic global
learning or covert account correlation is added. Native package bytes do gain a
new module, so the old frozen production identity remains a separate evaluation
gate (#123). Do not rewrite historical results or weaken checks to force green
CI. Keep draft until exact-head review and required checks permit integration.
