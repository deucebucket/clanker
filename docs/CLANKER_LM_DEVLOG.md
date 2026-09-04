# Clanker-LM Development Log

This is the running engineering record for the deterministic language runtime.
It complements the original V8 engine log in `docs/DEVLOG.md` and the concise
release history in `docs/CLANKER_LM_CHANGELOG.md`.

## Project boundary

Clanker-LM is not a replacement for the V8 engine.  It is an isolated runtime
layer that uses V8 as its affective controller:

```text
engine/       VADUGWI calculation, structures, force flow, solver
clanker_lm/   semantic parsing, symbolic memory, evidence, grammar, learning
```

The governing design rule is:

```text
semantic validity first
→ conversational target second
→ deterministic language realization third
→ VADUGWI candidate ranking last
```

A response that would move the affective state in a desirable direction is
still rejected when it invents an entity, binds unsupported information,
changes polarity, loses attribution, or answers the wrong semantic slot.

---

## 2026-09-03 — From scaffolding to a working runtime

### Starting observation

The existing Clanker engine already solved the central control problem:

- map language-derived signals into VADUGWI state;
- resolve competing forces and structural patterns;
- maintain conversational movement;
- calculate a target response range through the bidirectional solver;
- score candidate responses by predicted state transition.

The missing system was not another emotional model.  It was a deterministic
compiler around Clanker:

```text
language
→ symbolic proposition and discourse act
→ target state and response contract
→ grammatical construction
```

### Initial vertical architecture

The first Clanker-LM delivery established:

1. **Semantic parser**
   - entities and references;
   - predicates and semantic roles;
   - tense, polarity, modality, temporal markers, and repetition;
   - typed WH and polar questions;
   - explicit unresolved and ambiguous references.

2. **Conversation memory**
   - stable entity IDs and aliases;
   - salience and pronoun resolution;
   - event frames with certainty and provenance;
   - open-world matching;
   - contradiction preservation;
   - portable snapshots.

3. **Question/answer contract**
   - treat a question as a proposition containing a typed open slot;
   - bind only evidence-backed values;
   - distinguish answered, true, false, unknown, conflict, unsupported,
     missing-reference, and ambiguous-reference outcomes;
   - preserve the known portion of an event when the requested role is absent.

4. **Deterministic realization**
   - tense and morphology;
   - agreement and negation;
   - pronouns, articles, prepositions, and role ordering;
   - interrogative-to-declarative reconstruction;
   - candidate semantic plans and reverse checks.

5. **Affective controller**
   - use the real V8 `compute_vadug()` result;
   - choose a bounded target state;
   - analyze each semantically legal candidate;
   - apply `state_transition()`;
   - select the candidate nearest the target.

### No-template correction

The original prototype still used construction records containing completed
surface strings.  That violated the intended architecture.  The realization
system was therefore redesigned around:

- single-token lexical atoms;
- abstract grammar productions;
- runtime semantic values;
- deterministic morphology and punctuation;
- a hard schema audit rejecting legacy template tables or multiword atoms.

This invariant includes fallback behavior.  Unknown facts and unknown words do
not authorize canned sentences.  They authorize a different semantic act—such
as uncertainty or a probe—which is still composed through the atomic grammar.

### Active lexical learning

The learner was added as a versioned evidence overlay rather than direct edits
to the reviewed seed vocabulary.

Example interaction:

```text
That movie was glorp.
→ What does glorp mean?

Negative, like disappointing and overhyped.
→ Glorp means negative evaluation.
```

The learner records:

- unknown surface and normalized term;
- syntactic position and probable part of speech;
- surrounding semantic and VADUGWI context;
- user explanation and inferred semantic class;
- provisional VADUGWI estimate;
- support and contradiction weights;
- scope, status, and version;
- prior contexts requiring reinterpretation.

Later contradictory evidence can reduce confidence, deprecate a hypothesis, or
split one surface into conditioned senses.  The system retains history rather
than silently overwriting a bad entry.

A bounded clarification policy prevents recursive interrogation:

```text
one blocking unknown at a time
one-word discriminating follow-up when possible
remaining unknown descriptors retained as unresolved evidence
```

### Live semantic resolvers

Current time, current date, and safe arithmetic were implemented as typed
providers:

```text
QuestionFrame
→ resolver registry
→ typed observation
→ AnswerContract
→ compositional realization
```

Resolver observations carry source, certainty, observation time, expiration,
and metadata.  Transient values are not promoted into permanent semantic facts.

### Multi-turn transition learning

Clanker-LM began storing nontextual transition observations:

- prior conversational state;
- incoming dialogue act and VADUGWI vector;
- chosen response act and response vector;
- predicted resulting state;
- next observed state;
- signed residual and success score.

Similar contexts can supply bounded empirical corrections to future targets.
Hard truth, severity, and safety floors cannot be learned away.

### Dialogue and book profiles

Quoted or speaker-labelled dialogue can be compiled into:

- seven-byte VADUGWI vectors per turn;
- signed turn deltas;
- centroids and variance;
- dialogue-act distributions and transitions;
- multiresolution trajectory chunks;
- source and profile hashes.

These profiles can identify or condition affective trajectory.  They cannot
reconstruct exact prose from seven affective dimensions; the missing lexical
information is intentionally not encoded.

---

## 2026-09-03 — Deployment and API hardening

Automated review exposed deployment boundaries that were then made explicit:

- all CLI-to-runtime dependencies are represented by an executable API
  contract;
- every noninteractive CLI route and interactive inspection command is tested;
- the default `LanguageStore` is intentionally in-memory;
- cross-process persistence requires an explicit snapshot path or injected
  SQLite store;
- missing snapshot behavior is selected explicitly by each command;
- user-controlled file reads require a regular file and enforce byte limits;
- batch processing is bounded and fails rather than silently truncating;
- corpus chunk reads are bounded;
- SQLite uses an explicit lock timeout;
- optional V8 imports preserve isolated-package fallback;
- package import no longer requires the optional engine until a V8-backed
  runtime is constructed.

The initial runtime was squash-merged through PR #35 after exact-head review and
CI.  The tuned `engine/` tree remained untouched.

---

## 2026-09-03 to 2026-09-04 — Parser gap program

### Why sentence-local parsing was insufficient

The base runtime could answer bounded factual questions but still treated many
multi-clause inputs too shallowly.  That threatened the truth boundary because
one sentence can contain several propositions with different sources,
controllers, polarities, and entailment statuses.

The parser expansion program therefore adopted a strict rule:

```text
each proposition gets its own event frame
each semantic relationship gets its own typed relation
```

The parser must not flatten subordinate, modifier, reported, planned, desired,
questioned, or perceived content into the matrix event.

### Bounded-delivery discipline

Large grammar areas are divided into narrow slices.  Every slice must include:

- typed data structures;
- parser and memory integration;
- Q&A behavior;
- snapshot round-trip and migration;
- compositional realization;
- explicit ambiguity/failure diagnostics;
- hand-authored and generated conformance tests;
- complete repository regression;
- exact-head automated review;
- squash merge;
- zero changes to `engine/`.

### Structural groundwork

The first parser slices established:

- coordinated propositions;
- shared and independent arguments;
- subordinate cause, purpose, condition, concession, and temporal relations;
- matrix/subordinate event separation;
- subject and object relative clauses;
- restrictive and nonrestrictive modifiers;
- stable gap-role and head-entity binding;
- ambiguity rather than attachment guessing.

Possessive relatives and appositive identity remain separately scoped in issue
#84 so they do not contaminate the relative-clause foundation.

---

## 2026-09-04 — Finite attributed content complements (#86)

### Problem

Inputs such as:

```text
Sarah said that John left.
Sarah said John left.
```

contain a matrix communication event and attributed content.  Treating `John
left` as a direct global fact would lose source and truth status.

### Implementation

The first complement slice added:

- one-level finite declarative complements;
- explicit and zero complementizers;
- typed content relations;
- matrix source and content provenance;
- separate matrix/content polarity;
- source-qualified Q&A;
- explicit layered ambiguity;
- snapshot persistence;
- more than 160 conformance cases.

### Invariant

Reported, claimed, believed, remembered, or otherwise attributed content remains
attributed.  A direct later assertion may establish the event separately, but
the content complement alone does not become an unqualified fact.

Merged as commit:

```text
776c9053a5d3804fe1e64982b1b7ac392fe7e505
```

---

## 2026-09-04 — Infinitival control and raising (#88, PR #102)

### Problem

The surface sequence `NP + matrix verb + to + predicate` covers materially
different structures:

```text
Sarah planned to leave.          subject control
Sarah told John to leave.        object control
Sarah seemed to know.            raising/evidential
Sarah went to buy groceries.     purpose adjunct, not selected complement
```

The embedded event cannot be treated as completed merely because its predicate
is present.

### Representation

Added:

- `InfinitivalRelationType`;
- `InfinitivalContentStatus`;
- `InfinitivalRelation`;
- `InfinitivalAttachmentAmbiguity`;
- explicit matrix and embedded event indices/IDs;
- explicit subject controller, object controller, or raised subject;
- separate matrix and embedded polarity;
- attributed embedded provenance;
- version-4 snapshot migration.

Supported matrix classes:

- subject control: plan, intend, hope, and subject-control want;
- object control: tell, ask, and object-control want;
- raising: seem and appear.

### Truth boundary

```text
Sarah planned to leave.
```

does not prove:

```text
Sarah left.
```

Likewise, a directive does not prove compliance, and an evidential/raising
construction does not become an unqualified assertion.

### Testing

```text
2,140 passed, 2 xfailed
259 dedicated tests
200 generated control/raising cases
29/29 deterministic acceptance turns
```

The existing expected failures remained `CARETAKER_TRANSFER` and `BROADCAST` in
V8.

Merged through PR #102 as:

```text
72efa09438b23d34df0d4a072403090063599fb5
```

---

## 2026-09-04 — Embedded interrogative content (#89, PR #103)

### Problem

These are not direct questions with the same discourse status:

```text
Sarah asked who called.
Who called?
Do you remember when the meeting starts?
Tell me where John went.
```

The runtime must distinguish a statement reporting someone else's question,
the inner question frame, an outer polar question about knowledge, and a direct
request to answer the embedded question.

### Representation

Added:

- `EmbeddedInterrogativeType`;
- `EmbeddedInterrogativeStatus`;
- `EmbeddedInterrogativeRelation`;
- `EmbeddedInterrogativeAttachmentAmbiguity`;
- embedded WH requested roles;
- embedded polar proposition frames;
- matrix source and optional recipient;
- complementizer surface evidence;
- separate matrix and inner polarity;
- version-5 snapshot migration.

Supported inner content includes `who`, `whom`, `what`, `when`, `where`, `why`,
`how`, `which`, `whose`, `whether`, and complementizer `if`.

### Truth boundary

```text
Sarah asked who called.
```

does not establish any caller.  Similarly, wondering whether Mary left does not
establish that she left, and knowing a question with an absent answer does not
license invention.

Attributed inner questions cannot trigger live resolvers or mutate globally
trusted knowledge merely because they contain command-like or query-like text.

### Testing

```text
2,321 passed, 2 xfailed
181 dedicated tests
160 generated WH/polar cases
29/29 deterministic acceptance turns
```

Merged through PR #103 as:

```text
ca992c32e22cdaaf5239a689f11fffa176a31698
```

---

## 2026-09-04 — RR devlog: `-ing` complement implementation and local validation (#90)

- **Audience:** maintainers and reviewers of the deterministic language runtime
- **Date/window:** 2026-09-04, pre-merge branch validation
- **Scope:** issue #90 on `feature/clanker-lm-gerund-participial-complements`,
  starting from `ca992c32e22cdaaf5239a689f11fffa176a31698`; the V8 affect
  engine is explicitly excluded
- **Goal and success condition:** model the five issue-scoped `-ing` relation
  families without promoting attributed or noncompleted content to global fact;
  pass at least 160 generated cases, the full suite, deterministic acceptance,
  benchmark, compile, and repository-boundary checks before exact-head approval

### Rundown

**Proven:** the branch now implements `gerund_content`, `aspectual_start`,
`aspectual_stop`, `aspectual_continuation`, and
`perception_participial`. Matrix and complement events remain separate, with
explicit source, controller, polarity, content status, and stable relation
identity. Avoidance deliberately uses `GERUND_CONTENT` with `AVOIDED` status;
it is a noncompletion status, not a sixth relation family.

In plain language, Clanker-LM can preserve the difference between “Sarah
enjoys reading,” “Mary started working,” and “I saw Sarah leaving” instead of
flattening every `-ing` phrase into one asserted event. It also refuses to treat
ordinary nouns, progressive clauses, or free adjuncts as selected complements
when the grammar does not license that attachment.

The local acceptance evidence is green. **Proven:** independent correctness,
truth-boundary, and test/code reviewers accepted the corrected implementation
with no blockers. **Remote gate:** immutable-head CI, automated review, and
merge evidence are recorded on PR #105 after this devlog is committed; this
entry does not predict those later results.

### Current state

| Class | Claim | Evidence |
| --- | --- | --- |
| Proven | All five issue-scoped relation families are represented, including avoidance as `GERUND_CONTENT` + `AVOIDED`. | Model/parser/memory/Q&A/realization coverage in `clanker_lm/`; dedicated issue suite passes. |
| Proven | Factual phase answers are limited to positive, nonmodal, nonfuture simple/perfect clauses without forward-deictic time. | Truth-boundary regression cases in `tests/test_gerund_complements.py`. |
| Proven | Progressive, nominal, and free-adjunct `-ing` forms fail closed rather than becoming selected complement relations. | Parser boundary and ambiguity tests in the dedicated suite. |
| Proven | Version-6 snapshots load versions 1–6 and reject corrupt gerund bindings. | Snapshot round-trip, legacy-load, and corruption tests. |
| Proven | The implementation passes its stated local test and repository-boundary gates. | 285 dedicated passes; 2,606 full-suite passes and two expected xfails; 29/29 acceptance turns; clean compile/diff checks. |
| Proven | The corrected local implementation passed independent correctness, truth-boundary, and test/code review with no blockers. | Final local reviewer verdicts: ACCEPT / no blockers; full local evidence set rerun. |
| Unknown | Whether the committed branch head will pass every remote gate and merge unchanged. | CI, automated review, and merge evidence are authoritative on PR #105 because they run after this devlog commit. |

### Changes since the last entry

- Added separate matrix/complement event frames and typed relations for:
  `Sarah enjoys reading`, `Mary started working`, `John stopped smoking`,
  `They kept talking`, `I saw Sarah leaving`, and
  `Sarah avoided calling John`.
- Kept selected gerund content and perception participials attributed and
  nonassertive. Only qualified phase relations may support a factual answer;
  modal, future, negated, progressive, and forward-time contexts do not.
- Added explicit controller binding and preserved matrix versus embedded
  polarity, source, provenance, specificity, and conflict behavior through Q&A.
- Advanced symbolic-memory snapshots to version 6 with backward loading for
  versions 1–6 and validation that rejects inconsistent or corrupt bindings.
- Added compositional realization for direct and source-qualified answers; no
  completed response sentence or issue-specific whole-sentence template was
  introduced.
- Corrected modality/future leakage, specificity matching, provenance and
  conflict handling, snapshot validation, and atomic-realization defects found
  during independent review.

### How it works, in plain language

An `-ing` complement is stored as two linked facts: the matrix event (for
example, “Mary started”) and a source-qualified complement event (“Mary
working”). A typed relation records how they connect. The complement does not
become an ordinary global fact merely because it was mentioned.

Phase predicates are the narrow exception: a positive, nonmodal, nonfuture
simple/perfect “started,” “stopped,” or “kept/continued” relation can license a
qualified factual answer. This exception is shut off by negation, future or
modal framing, progressive phase aspect, or forward-deictic time. Perception,
enjoyment, and avoidance remain attributed/nonassertive, and avoidance never
proves completion.

### Evidence and conditions

- **Environment/version:** issue #90 branch based on
  `ca992c32e22cdaaf5239a689f11fffa176a31698`; memory/runtime snapshot version 6
  with compatible loading for versions 1–6.
- **Dedicated workload:** `python -m pytest -q tests/test_gerund_complements.py`
  — **285 passed**, including exactly **160 generated conformance cases**.
- **Full workload:** `python -m pytest -q` — **2,606 passed, 2 xfailed**. The
  two xfails are the pre-existing V8 `CARETAKER_TRANSFER` and `BROADCAST` gaps.
- **Acceptance and benchmark:** deterministic harness — **29/29 turns**;
  `benchmarks/clanker_lm_eval.py` — **7 cases / 29 turns**.
- **Static/repository checks:** package compilation and `git diff --check`
  completed cleanly; `engine/` and `clanker_engine.py` have no branch changes.
- **Review evidence:** independent review identified and drove fixes for
  modality, future-time, specificity, provenance, conflict, snapshot, and
  atomicity defects. Final local correctness and truth-boundary reviewers
  returned **ACCEPT / no blockers**, and test/code acceptance was clean.
- **Limitations:** local green evidence does not substitute for the exact-head
  CI and automated-review results recorded on PR #105.

### Negative results and open questions

- The full suite retains two expected V8 xfails. They predate issue #90 and do
  not exercise Clanker-LM complement behavior.
- No remote approval or merge result should be inferred from the local
  acceptance in this entry; PR #105 carries those post-commit results.

### Next step / blocker

Use PR #105 to run exact-head remote CI and automated review, and merge only if
both gates accept that same head. There is no local implementation, test, or
independent-review blocker.

---

## Backlog created for complement closure

The following issues convert the broad grammar gap into reviewable deliveries:

- **#84** — possessive relatives and appositive identity links;
- **#90** — gerund, participial, and aspectual complements;
- **#91** — immutable held-out complement attribution/control corpus;
- **#92** — ECM, bare-infinitive perception, causatives, and small-clause
  boundaries;
- **#93** — shared matrix-predicate catalog and entailment classes;
- **#94** — deterministic complement realization and reverse validation;
- **#95** — versioned complement signatures and snapshot migration;
- **#96** — generated truth-boundary and attachment fuzzing;
- **#97** — matrix/embedded trace graph and UI contract;
- **#98** — support matrix and user/developer documentation;
- **#99** — staged feature flags and compatibility checks;
- **#100** — bounded candidate growth and low-resource performance;
- **#101** — adversarial attribution and command/content boundaries.

Issues #52 and #54 remain the parent parser/modifier and complement roadmaps.
They should close only after their bounded children, held-out evaluation, and
support documentation are complete.

---

## Development rules going forward

1. Work from current `main`; never stack parser branches on unmerged parser
   branches.
2. Keep each grammar family in one bounded issue and one reviewable PR.
3. Commit all source, tests, migrations, documentation, and benchmark updates
   before requesting review.
4. Keep branch history clean; remove transfer workflows and staging payloads
   before final review.
5. Run exact-head CI and the complete repository suite.
6. Wait for an automated review tied to the exact head SHA.
7. Address every valid truth, persistence, interface, resource, or ambiguity
   finding with regression coverage.
8. Rebase/consolidate to one clean implementation commit when practical.
9. Squash merge only after green CI and approval.
10. Update this devlog and `docs/CLANKER_LM_CHANGELOG.md` with each bounded
    delivery.
11. Never change the original V8 engine as an incidental parser fix.
12. Never trade semantic correctness for a more emotionally attractive reply.

## Current status snapshot

```text
main head:        ca992c32e22cdaaf5239a689f11fffa176a31698
active branch:    feature/clanker-lm-gerund-participial-complements
active issue:     #90 — local independent review accepted
latest full test: 2,606 passed, 2 expected xfails
dedicated test:   285 passed; 160 generated conformance cases
acceptance:       29/29 turns
V8 engine files:  unchanged
response mode:    atomic/compositional; no whole-sentence templates
```

The current branch passed local independent correctness, truth-boundary, and
test/code review with no blockers: **2,606 passed, 2 expected xfails**, **285
dedicated passes** with **160 generated conformance cases**, and the deterministic
acceptance harness remains **29/29**. Exact-head CI, automated review, and merge
evidence are recorded on PR #105 after this devlog commit.

---

## 2026-09-04 — RR devlog: private browser workbench deployed

- **Audience:** the operator and maintainers of Clanker-LM
- **Scope:** the optional browser interface, its live user-service deployment,
  and its Tailnet boundary; no V8 engine changes
- **Success condition:** an allowlisted Tailnet user can use isolated Clanker-LM
  sessions through HTTPS while the application remains loopback-only, bounded,
  inspectable, and absent from the public internet

### Rundown

**Proven:** the Starlette/Uvicorn workbench is running under the
`clanker-lm-web.service` systemd user service on `127.0.0.1:8765`. Tailscale
Serve maps the tailnet-only URL
`https://bazzite.tail85f65f.ts.net:8444/` to that listener. This workbench does
not use Funnel and has no public route. Both the loopback and Tailnet health
checks return `{"status":"ok"}`.

In plain language, the approved user can chat with the same deterministic
runtime in a browser without opening a public website. Each browser gets its own
memory, and each answer shows **Answer**, **Truth**, **Source**, **Certainty**,
**Memory**, and **VADUG** instead of presenting an unexplained string.

### Current state

| Class | Claim | Evidence |
| --- | --- | --- |
| Proven | The service is deployed, enabled, and active. | `systemctl --user is-enabled` returned `enabled`; `is-active` returned `active`. |
| Proven | Port 8444 is tailnet-only and proxies to loopback port 8765. | `tailscale serve status` marks the exact 8444 route `tailnet only` and shows `http://127.0.0.1:8765`. |
| Proven | The live process and route are healthy. | Local and Tailnet `/healthz` requests both returned `{"status":"ok"}`. |
| Proven | Identity, origin, cookie, session-isolation, resource, logging, and browser security boundaries are covered by the dedicated suite. | 88 web tests passed on the current exact tree. |
| Proven | The web change preserved repository behavior and the V8 boundary. | 2,694 passed plus two expected xfails; benchmark 29/29; compile, JavaScript syntax, diff, and no-engine checks clean. |
| Proven | Independent review accepted the earlier web implementation. | Reviewer verdict before rebase and before `6f3c1bf`: ACCEPT. |
| Proven | The reported ACL-link regression is corrected in the live deployment. | Commit `6f3c1bf`; service restart; Jerry's successful live ACL-link retest; focused navigation regressions. |
| Inference | The corrected deployment is suitable for the current private, allowlisted workbench use. | Live boundary/link checks and the current suites cover the intended single-process scope; exact-head independent review remains outstanding. |
| Unknown | Long-duration availability across future host, user-session, or Tailnet maintenance. | The deployment is newly live; sustained uptime has not yet been measured. |
| Unknown | Whether an independent reviewer will accept the final exact head without further findings. | The recorded ACCEPT predates `6f3c1bf`; current exact-tree checks are green but are not a final reviewer verdict. |

### What changed

- Added an optional Starlette application and one-worker Uvicorn CLI adapter.
- Added a browser conversation surface with safe DOM rendering, accessible
  controls, session export, reset, and a six-field evidence rail: **Answer**,
  **Truth**, **Source**, **Certainty**, **Memory**, and **VADUG**.
- Added exact Tailscale identity and origin checks, strict session cookies,
  defensive response headers, no message-body/access logging, and resource
  ceilings for sessions, rates, turns, bodies, responses, and exports.
- Refined shell bootstrap after Jerry's ACL-link report: only a user-activated
  top-level document navigation may bootstrap from same/cross-site sources;
  iframe, subresource, fetch/CORS, and non-user cross-site attempts fail closed.
- Deployed the loopback process as `clanker-lm-web.service` and placed Tailscale
  Serve—not Funnel—in front of it on HTTPS port 8444.

### Evidence and conditions

```text
dedicated: 88 web tests passed
full:      2,694 passed, 2 expected xfails
benchmark: 29/29 deterministic turns
static:    compile, JavaScript syntax, and diff checks clean
review:    pre-rebase/pre-6f3c1bf web reviewer ACCEPT
boundary:  engine/ and clanker_engine.py unchanged
service:   enabled and active; loopback + Tailnet health checks healthy
route:     :8444 tailnet only -> http://127.0.0.1:8765
live fix:  6f3c1bf; Jerry's ACL-link retest succeeded
```

The independent verdict predates both the rebase and `6f3c1bf`. The green
exact-tree suites and live retest prove the exercised behavior after the fix,
but do not imply that reviewer examined the final exact commit.

### Risks and guardrails

- This is an in-memory, single-process workbench. A service restart clears
  unexported sessions.
- Tailnet membership is necessary but insufficient: deployed requests also need
  the exact allowlisted login. Local processes remain inside the host trust
  boundary.
- A minimal health response is intentionally unauthenticated, but it is reachable
  only through loopback or this tailnet-only route.
- Do not replace Serve with Funnel and do not use `tailscale serve reset` on the
  shared host; reset would affect unrelated routes.

### Next / blocker

No deployment blocker was known at this checkpoint. The later final exact-head
release gate did return **APPROVE** for
`780d77b4673aa45a692fc5a1f8af144a41f09fd0` before PR #106 was squash-merged.
Continue normal service and Tailnet monitoring; after host or Tailnet
maintenance, rerun the health and route checks in `docs/CLANKER_LM_WEB.md`.
Measure sustained availability only if an uptime claim becomes a requirement.

---

## 2026-09-04 — RR devlog: reviewed changes become visible

- **Audience:** the workbench operator and Clanker-LM maintainers
- **Scope:** issue #112's repository-backed browser changelog; no deployment,
  identity milestone, language-runtime, or V8 engine change
- **Goal and success condition:** a keyboard and mobile user can inspect an
  ordered record of reviewed releases, while current-live status, pending
  milestones, runtime build identity, provenance, evidence, limitations, and
  deployment state remain explicit and independently verifiable

### Rundown

**Proven:** the branch adds an authenticated, same-origin release endpoint and
an accessible native-dialog changelog to the existing workbench. Its packaged
feed names only reviewed and merged PR #106, package `0.2.0`, and merge commit
`9ae77f072f8afda0b1d2b757ab492757cabff0f8`. The corrected endpoint separately
reports the full build commit supplied by deployed configuration. Startup
validation fails closed on malformed, out-of-order, mismatched, externally
linked, or private-field data, while CI verifies merged PR state through GitHub.

In plain language, the interface now carries its own shipping receipt. A user
does not need ACL history or repository archaeology to distinguish the live
workbench baseline from work that is still under development.

### Current state

| Class | Claim | Evidence |
| --- | --- | --- |
| Proven | The changelog preserves the workbench's visual and evidence structure. | Existing six-field rail tests remain green; the new dialog reuses the graphite, warm-paper, amber, and teal system. |
| Proven | Runtime build and milestone provenance are distinct. | `WebConfig.build_commit` becomes API `deployed_build_commit` and UI **Runtime build**; feed rows carry separate `milestone_commit`. |
| Proven | Each `pr-N` row structurally matches its pull and commit evidence, and PR #106 is actually merged at the recorded commit. | Adversarial validator tests plus `python -m clanker_lm.web_release_verify`. |
| Proven | Dynamic release copy cannot be interpreted as HTML by this implementation. | Hostile-copy regression plus exclusive `textContent`/DOM-node construction; no `innerHTML`, `insertAdjacentHTML`, or `document.write`. |
| Proven | Reading the release endpoint requires the configured identity and does not create a runtime session. | Auth/no-allocation acceptance test. |
| Proven | No later roadmap milestone is claimed in the initial feed. | Packaged `releases.json` contains only `pr-106`; explicit regression assertion excludes `pr-107`. |
| Unknown | Whether the exact final branch will pass independent review and remote CI. | Those gates occur after the coherent local commit. |
| Unknown | Whether the new changelog is live. | This lane is explicitly not authorized to deploy the service. |

### Changes since the last entry

- Added a visible masthead entry point and native dialog with loading, success,
  retry/error, and empty/malformed boundaries.
- Added a mobile full-height reader, bounded scrolling, 44-pixel controls,
  close/Escape behavior, and focus restoration.
- Added the packaged `releases.json` source and strict server-side validator.
- Added a read-only `/api/releases` route that retains identity, CSP,
  no-store, same-origin, and no-session-allocation boundaries.
- Required a full nonzero build SHA in deployed mode and wired it through the
  CLI/environment, systemd example, API, UI, and live-verification procedure.
- Added a network-free runtime boundary plus a CI/release verifier that queries
  GitHub only outside the running service.
- Added adversarial tests for ordering, exact identity, escaping, accessibility,
  private-content exclusion, packaging metadata, and unchanged security rules.

### How it works, in plain language

The JSON file is a small milestone ledger checked into the repository. On
startup, the server validates its internal links and package agreement, then
adds the independently supplied runtime build commit to the API response. The
browser shows that build above the milestone cards and builds all dynamic copy
as text. CI separately checks GitHub's merged PR record. A bad ledger or missing
deployed build stops the app instead of producing a plausible-looking receipt.

### Evidence and conditions

```text
base:      main at 9ae77f072f8afda0b1d2b757ab492757cabff0f8
focused:   157 web/config/CLI/verifier/real-DOM tests passed
full:      2,760 passed, 2 expected xfails
benchmark: 29/29 deterministic turns
browser:   1440×1000, 360×800, and 300×700; no overflow; focus restored
identity:  injected runtime build differed visibly from PR #106 milestone
GitHub:    PR #106 verified merged at 9ae77f0
wheel:     verifier and all four packaged web assets present
static:    workflow YAML, Python compile, JavaScript syntax, and diff clean
boundary:  no engine/, clanker_engine.py, deployment, or service change
```

These are exact local-tree results before the corrected coherent commit. Remote
CI and independent exact-head review remain before merge or deployment.

### Negative results and open questions

- At that stage, the live service did not contain the changelog: the requested
  lane forbade deployment, and the feed itself excluded unshipped #112 work.
- A single initial release cannot visually demonstrate multiple-date ordering;
  synthetic validator coverage proves future out-of-order entries fail closed.
- The original `2c16f69` design was rejected: it validated PR #106's milestone
  commit only against the same ledger and mislabeled it as deployed code. That
  result rules out self-consistent metadata as proof of a running artifact. The
  corrected contract requires an independent deployed build input and shows
  both identities.

### Next step / blocker

Finish the corrected local gates, commit the coherent tree, then obtain
exact-head independent review and remote CI. After the implementation PR for
issue #112 merges, use that PR's actual number `N` and actual squash/merge SHA
`M` to add `pr-N` with matching `/pull/N` and `/commit/M` evidence in a
follow-up reviewed metadata artifact before any production deployment. Close
#112 only after runtime-build equality, milestone verification, and live UI
probes pass.

---

## 2026-09-04 — RR devlog: PR #113 release artifact

- **Audience:** workbench operator and Clanker-LM maintainers
- **Scope:** post-merge release metadata for issue #112; no language-runtime or
  V8 engine behavior change
- **Success condition:** the packaged ledger names the real implementation PR
  and merge commit, a separately identified artifact is deployed, and the live
  API and dialog agree on both identities

### Rundown

**Proven:** two independent reviewers accepted implementation head
`e4c25ae330d917132e496e700b1c2839e652a911`; both Python CI matrices passed;
GitHub merged the implementation as PR #113 at
`66b85de66337789fa83292ecf683c6b23cc0af55`.

**Proven:** this follow-up replaces the pre-merge PR #106-only ledger head with
an exact `pr-113` row. Its evidence URLs name PR #113, merge commit `66b85de`,
and exact-head CI run `33861328696`. It is explicitly `pending`; PR #106 remains
the current `live` marker rather than being retired before deployment.

**Proven:** the corrected ledger contract requires exactly one current-live row
at index zero, followed by pending releases newest-first within that group, then
newest-first retired or rolled-back history. A pending milestone dated after the
live baseline is valid; duplicate live rows and mixed lifecycle groups fail
closed. The dialog visibly says **Reviewed release record**, marks the current
live card, and gives the pending card distinct **What passed review** language
and amber treatment.

**Proven:** lifecycle badge copy is canonical rather than author-controlled.
Startup rejects pending/live, live/pending, and retired/live state-label
mismatches. The browser selects fixed badge text from structured state; a real
DOM test injects forged opposite-state labels and proves they are not visible
in either release card or the current-runtime summary.

**Inference:** the metadata is ready for release review because its identifiers
come from GitHub's completed merge and CI records, not a predicted PR number or
self-referential commit placeholder.

**Historical state at that stage:** this follow-up artifact was not live. Its
exact local gates passed, while independent review, remote CI, merge, service
restart, and live API/browser verification still remained. PR #114 and its
staging receipt below record how those gates later resolved.

### Evidence

```text
implementation PR:    113
implementation merge: 66b85de66337789fa83292ecf683c6b23cc0af55
accepted head:         e4c25ae330d917132e496e700b1c2839e652a911
PR CI run:             33861328696 (Python 3.10 and 3.12 passed)
live URL:              https://bazzite.tail85f65f.ts.net:8444/
runtime build:         supplied only after this follow-up artifact merges
focused:              157 web/config/CLI/verifier/real-DOM tests passed
full:                 2,760 passed, 2 expected xfails
benchmark:            29/29 deterministic turns
GitHub:               PR #106 at 9ae77f0; PR #113 at 66b85de
static:               compile, JavaScript syntax, and diff checks clean
boundary:             engine/ and clanker_engine.py unchanged
```

### Recorded next step at that stage

The recorded next step was to commit and review the two-entry ledger, merge and
deploy it with its exact SHA, run live probes, and then create a separate
reviewed artifact promoting PR #113 to `live` while retaining PR #106 history.

---

## 2026-09-04 — RR devlog: PR #113 final live-promotion candidate

- **Audience:** workbench operator, release reviewer, and Clanker-LM maintainer
- **Scope:** final issue #112 release metadata and browser presentation; no
  language-runtime, V8 engine, public-route, or service mutation
- **Success condition:** checked-in metadata promotes the already staged PR #113
  capability to the sole live milestone, preserves PR #106 as history, cites
  bounded deployment evidence, and remains explicit that runtime build identity
  comes only from the deployed artifact

### Rundown

**Proven:** staging PR #114 merged at
`2d736961d7db0510711a7ac54eb39a458446f5ee` after Python 3.10 and 3.12 passed in
Actions run `33863653170`. The exact detached artifact ran with the same SHA as
its `--build-commit`; loopback and Tailnet health returned `200`; the
authenticated release API returned `200`, `no-store`, no session cookie, and
the exact build. Chromium probes at 1440×1000, 360×800, and 300×700 saw one live
and one pending card, canonical labels, no overflow, restored Escape focus, and
zero console or page errors. The bounded [staging receipt](https://github.com/deucebucket/clanker/issues/112#issuecomment-5539229707)
records those conditions.

**Proven:** this candidate makes PR #113 the sole live row and keeps PR #106 as
retired history. Its PR #113 card links the implementation merge, staging PR
#114, staging merge, staging CI, and exact receipt. The validator and browser
allow only that exact issue-comment URL; hostile query, fragment, issue,
comment, and userinfo variants fail closed. Replacing any PR #114 pull, merge,
or CI URL with another otherwise allowed repository URL also fails. Generic
pending rows remain covered by server tests and a synthetic real-browser
rendering.

**Unknown at candidate validation time:** the final promotion artifact had no
merge SHA and had not been deployed. GitHub still reported issue #112 open.
This is a point-in-time release receipt, not an enduring status assertion;
`/api/releases` is authoritative after each deployment.

### Current state

| Class | Claim | Evidence |
| --- | --- | --- |
| Proven | PR #113's capability ran successfully from the merged staging artifact. | PR #114 at `2d73696`; CI `33863653170`; issue comment `5539229707`. |
| Proven | The promotion ledger has exactly one live row followed by retained history. | `latest_shipped_release=pr-113`; ordered states `[live, retired]`; validator and endpoint tests. |
| Proven | The browser shows the promoted semantics without losing pending coverage. | Real Chromium at three responsive sizes plus a synthetic pending-state pass. |
| Unknown | Final production identity after promotion. | Supplied only by the future merge SHA through `CLANKER_LM_BUILD_COMMIT`, then measured through `/api/releases`. |

### How it works, in plain language

PR #114 proved the changelog capability could run before the ledger called it
live. This separate artifact advances the pointer only after that proof and
moves the prior baseline into history. The milestone commit says which reviewed
feature shipped; the runtime build says which exact repository artifact is
currently executing. They intentionally answer different questions.

### Evidence and conditions

```text
base:          origin/main at 2d736961d7db0510711a7ac54eb39a458446f5ee
staging PR:    #114 merged at 2d736961d7db0510711a7ac54eb39a458446f5ee
staging CI:    33863653170 (Python 3.10 and 3.12 passed)
staging proof: https://github.com/deucebucket/clanker/issues/112#issuecomment-5539229707
focused:       164 web/config/CLI/verifier/real-DOM tests passed
full:          2,767 passed, 2 expected xfails
benchmark:     29/29 deterministic turns
browser:       1440×1000, 360×800, 300×700; no overflow/errors; focus restored
GitHub:        release PRs #113/#106 verified; PR #114 merge and CI verified
package:       verifier and all four web assets present in a fresh wheel
static:        Python compile, JavaScript syntax, and diff checks clean
boundary:      engine/ and clanker_engine.py unchanged
```

### Negative results and open questions

- The prior global newest-first rule rejected a valid newer pending milestone;
  lifecycle-group ordering replaced it before PR #114 merged.
- Free-form deployment labels could contradict structured state; canonical
  server validation and browser-derived badge copy now prevent that mismatch.
- Local and staging evidence cannot prove the final artifact is deployed. Only
  exact-SHA runtime equality and promoted live API/browser probes can do so.

### Next step / blocker

Obtain exact-head review and CI, merge the final promotion, deploy that merge
SHA as `CLANKER_LM_BUILD_COMMIT`, and verify health, authenticated API identity,
canonical live/history cards, responsive overflow, focus restoration, and
console/page errors. Issue closure depends on those measured results.

---

## 2026-09-04 — RR devlog: issue #41 held-out evaluation core candidate

- **Audience:** evaluation reviewer, Clanker-LM maintainer, and future #107
  tuning owner
- **Scope:** versioned conversation corpus, deterministic evaluator, integrity
  gates, CI, and documentation; no language-runtime behavior changes and no
  held-out baseline publication yet
- **Goal and success condition:** freeze at least 500 whole held-out turns with
  auditable rights, keep open development data separate, bind the measured
  production tree, and make every report/integrity boundary fail closed before
  any tuning can see held-out evidence

### Rundown

**Proven:** the selected `conversation-v1` generation contains 56 held-out
conversations / 520 turns and 10 open-development conversations / 60 turns.
The held-out split spans public-domain drama, public-domain novels, raw NASA
technical air-to-ground exchanges, and original CC0 synthetic structural
stress cases. The held-out and development split digests did not change while
the evaluator and release provenance were corrected.

**Proven:** corpus root
`7e0f03f335866c982c6225b805b2d1efb52a9de67e3c65dae5dd33cdf59582f0`
binds compiler, evaluator, schema, split policy, source provenance, and the
exact production/package tree at
`c8c0bf4ccd5e73b1bd6bbe99762c87c4a549665e`. Production Python, language seed,
and packaged web assets are all included in that tree digest. Evaluation code
is excluded from built packages and production has no corpus injection API.

**Unknown:** the baseline metrics are intentionally absent. Two detached
reviewers must accept this exact clean core before the 520-turn, three-mode
held-out run is allowed. Until that report is generated from the accepted core
commit and committed separately, issue #41 is not complete.

### Current state

| Class | Claim | Evidence |
| --- | --- | --- |
| Proven | Whole-conversation scale and split separation meet the corpus floor. | Held-out 56/520; development 10/60; compiler manifest under `evaluation/conversations/data/generations/7e0f03f…/`. |
| Proven | Real-human material is bounded to raw NASA technical transcript windows with authoritative public-use provenance. | NTRS `20160014392`, NASA raw transcript index/download locators, frozen raw digest, and source manifest. |
| Proven | Literary affect/outcome annotations remain weak supervision; synthetic structural gold is excluded from affect scoring. | Per-turn supervision fields and evaluator strata. |
| Proven | Integrity checks reject policy mutation, leakage, near-duplicates, source/schema corruption, split/report TOCTOU swaps, symlink escapes, partial publication, free-form response channels, and production-tree drift. | 94 evaluator tests passed with the baseline test deselected; `verify` reports zero overlap/reference/text-leak hits. |
| Unknown | Final semantic/entity/transition scores and confidence intervals. | Baseline run is held pending independent ACCEPT verdicts. |

### How it works, in plain language

`CURRENT` is one atomic pointer to one content-addressed corpus directory. A
reader hashes and parses the same bytes, validates the exact directory
inventory, and refuses training or replay access to held-out rows. Reports use
the same one-generation-pointer pattern, so a crash cannot select a report,
failure ledger, and checksum from different runs. Cluster bootstrap samples
whole conversations, preserving turn-weighted point estimates and paired mode
draws.

The truth boundary is intentionally conservative: literary and archival next
turns show what followed in their source, not what a Clanker response caused.
Synthetic conversations provide structural labels, not real-human outcome
ground truth. No private chat, ACL transcript, user-supplied conversation, or
copyright-ambiguous source is present.

### Evidence and conditions

```text
production base:     c8c0bf4ccd5e73b1bd6bbe99762c87c4a549665e
core milestone:      ed171f553d655056bc00c2279493bb4556bb00fa
corpus root:         7e0f03f335866c982c6225b805b2d1efb52a9de67e3c65dae5dd33cdf59582f0
production digest:   b6ea934649405392f655687eb0fca721f14afc1134ad0eac9db0577196a52b7f
held-out split:      05a2dfed6776ccdb53e191d7666c2a83b6fffa2f63c31492a24bea39cbe64f18
development split:   5018d7d3f2b60207333b6c50fec594b6b46411a6065962ec322038f0b75f0e43
focused evaluator:   94 passed, 1 baseline test deliberately deselected
full repository:     2,861 passed, 2 expected xfails, 1 expected missing-baseline failure
benchmark:           29/29 deterministic turns
integrity verify:    520 held-out + 60 development; 0 exact overlap; 0 near overlap;
                     0 production references; 0 production/test text leakage
visual evidence:     not applicable; this milestone has no UI surface
```

### Negative results and open questions

- Several earlier baseline attempts were invalidated before publication by
  scorer, annotation, provenance, and transactional-integrity findings. They
  are not evidence and are not present in the repository.
- The full suite's sole failure is the deliberately absent
  `post_113_heldout_v1.current`; this proves the release gate has not been
  bypassed, not that baseline acceptance passed.
- Static production-reference analysis covers resolvable literals,
  concatenation, constant f-strings, named constants, imports, loader calls,
  and path joins. It does not claim to analyze arbitrary dynamic code; the
  production API exposes no dynamic corpus path/module injection surface.

### Next step / blocker

Nominate the final clean core commit for both detached reviewers. Only after
both return ACCEPT: run all three modes over the immutable 520-turn held-out
split, transactionally publish the aggregate-only report/failure/checksum
generation, commit those baseline artifacts separately, rerun the full suite,
and create the documented immutable release tag after merge. Held-out text and
labels remain forbidden inputs to #107 training or teacher replay.
