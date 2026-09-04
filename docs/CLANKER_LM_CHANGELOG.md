# Clanker-LM Changelog

This file records the deterministic language-runtime line separately from the
original V8 affect engine changelog.  The two systems remain intentionally
isolated:

```text
engine/       tuned Clanker V8 affect engine
clanker_lm/   deterministic semantic, memory, learning, and realization layer
```

No entry in this file implies a modification to the tuned `engine/` tree unless
it explicitly says otherwise.  Every merged Clanker-LM parser slice to date has
left `engine/` unchanged.

---

## 2026-09-04 — Reviewed release feed

**Issue:** [#112](https://github.com/deucebucket/clanker/issues/112)
**PR:** [#113](https://github.com/deucebucket/clanker/pull/113)
**Merge commit:** `66b85de66337789fa83292ecf683c6b23cc0af55`

Added a visible **Changelog** entry point to the existing private semantic
workbench. The dialog keeps the graphite/warm-paper/amber-teal visual language,
works as a full-height mobile reading surface, uses native dialog keyboard
behavior, restores focus on close, and preserves the six-field answer evidence
rail.

The feed is repository-backed by a bounded packaged JSON artifact. Application
startup validates its schema, unique release IDs, exact package-version and
milestone agreement, `pr-N`/pull-request and milestone/commit evidence links,
URL allowlists, pinned private deployment URL, and absence of private-content
fields. It also requires exactly one current-live row first, followed by
newest-first pending rows and then newest-first history; a pending milestone may
be newer than the live baseline. Every state has one canonical badge label;
startup rejects contradictory state/label pairs, and the browser derives badge
copy from state rather than rendering ledger label text. A separate required
deployed configuration value reports the exact running build through the API
and UI. CI asks GitHub to prove every milestone PR is merged at its recorded
commit. The browser creates dynamic content with `textContent`; it uses no HTML
insertion or external assets.

The initial pre-merge feed contained only reviewed PR #106. After the
implementation merged as PR #113, staging PR #114 added the actual PR number
and merge commit as **Pending · live verification**. PR #114 then merged at
`2d736961d7db0510711a7ac54eb39a458446f5ee`; Actions run `33863653170` passed,
and that exact artifact passed live service, API, identity, and responsive
browser verification. The bounded [staging receipt](https://github.com/deucebucket/clanker/issues/112#issuecomment-5539229707)
records the conditions and results.

This final metadata artifact promotes PR #113 to **Live · private Tailnet** and
retains PR #106 as **Retired · release history**. That statement is backed by
the deployed staging artifact. The promotion procedure requires independent
review, CI, merge, deployment with the final artifact's exact merge SHA, and
live API/browser verification before issue closure. `/api/releases` is the
authority for whether that procedure has completed. Issue #107 and later
language-roadmap work remain absent until their own reviewed releases.

The first implementation commit `2c16f69` was rejected because it reused PR
#106's milestone commit as if that proved the running build. The corrected
contract keeps `milestone_commit` in the ledger and injects
`deployed_build_commit` from `WebConfig`, the CLI, and the deployment environment.
It rejects missing, abbreviated, uppercase, invalid, or all-zero deployed build
IDs. The checked-in feed contains no self-referential build SHA.

Local validation on the current tree:

```text
164 focused web/config/CLI/verifier/DOM tests passed
2,767 full-suite tests passed, 2 expected xfails
29/29 deterministic acceptance turns
real Chromium DOM wiring passed; hostile text remained text with no overflow
1440×1000, 360×800, and 300×700: runtime/milestone split visible; focus restored
GitHub verified PR #106 at 9ae77f0, PR #113 at 66b85de, and staging PR #114 at 2d73696
fresh wheel contains the verifier and all four web assets
workflow YAML, compile, JavaScript syntax, diff, and no-engine checks clean
```

Both independent exact-head reviewers accepted implementation head
`e4c25ae330d917132e496e700b1c2839e652a911`; GitHub squash-merged PR #113 as
`66b85de66337789fa83292ecf683c6b23cc0af55`. Staging PR #114 carried the
reviewed lifecycle correction and was separately merged, deployed, and probed.
The final promotion is a distinct artifact; the repeatable release gate is exact
review, CI, merge, exact-SHA deployment, and live API/browser verification.

---

## 2026-09-04 — Private browser workbench

**PR:** [#106](https://github.com/deucebucket/clanker/pull/106)
**Merge commit:** `9ae77f072f8afda0b1d2b757ab492757cabff0f8`

Added a small Starlette/Uvicorn interface for the deterministic runtime and
deployed it as the `clanker-lm-web.service` systemd user service.

- Live URL: `https://bazzite.tail85f65f.ts.net:8444/`.
- Exposure: Tailscale Serve on port 8444, **tailnet only**; no Funnel/public
  route for this workbench.
- Backend: one Uvicorn worker bound to `127.0.0.1:8765`.
- Access: exact `Tailscale-User-Login` allowlist for
  `jerrymares@gmail.com`, with exact-origin mutation checks.
- Sessions: isolated in-memory Clanker-LM runtimes behind opaque Secure,
  HttpOnly, SameSite=Strict cookies; export and reset controls are available.
- Evidence: every displayed answer includes **Answer**, **Truth**, **Source**,
  **Certainty**, **Memory**, and **VADUG**.
- Bounds: idle expiry, rate/turn/session limits, byte ceilings, bounded exports,
  fail-closed capacity, security headers, and no message-body/access logging.

Live use exposed one bootstrap regression after the earlier review: Jerry's ACL
link was blocked by the blanket cross-site rule. Commit `6f3c1bf` now permits
only authenticated, user-activated top-level document navigation to bootstrap
from same-site or cross-site links. Iframes, subresources, fetch/CORS, and
non-user cross-site attempts still fail closed without allocating a session.
The service was restarted and Jerry's live ACL-link retest succeeded.

Validation on the current exact tree:

```text
88 web tests passed
2,694 full-suite tests passed, 2 expected xfails
29/29 deterministic benchmark turns
compile, JavaScript syntax, and diff checks clean
engine/ and clanker_engine.py unchanged
```

The final exact-head release gate returned **APPROVE** on
`780d77b4673aa45a692fc5a1f8af144a41f09fd0`: Python 3.10/3.12 CI, automated
review, and independent security/UX judgment were green before the PR was
squash-merged. Operational details are in `docs/CLANKER_LM_WEB.md`.

---

## Unreleased — Gerund, participial, and aspectual complements

**Branch:** `feature/clanker-lm-gerund-participial-complements`
**Issue:** [#90](https://github.com/deucebucket/clanker/issues/90)
**Base:** `ca992c32e22cdaaf5239a689f11fffa176a31698`

**Status:** **Proven:** local independent correctness, truth-boundary, and
test/code review accepted with no blockers. **Remote gate:** immutable-head CI,
automated review, and merge evidence are recorded on
[PR #105](https://github.com/deucebucket/clanker/pull/105), because those checks
necessarily run after this changelog is committed.

Implemented the five issue-scoped relation families:

- `gerund_content`: enjoyment and avoidance, with avoidance represented as
  `GERUND_CONTENT` + `AVOIDED` rather than a sixth relation family;
- `aspectual_start`, `aspectual_stop`, and `aspectual_continuation`;
- `perception_participial` with an explicit embedded controller.

Matrix and embedded events remain separate and preserve controller, polarity,
source, provenance, specificity, and conflict boundaries. Gerund and perception
content stays attributed/nonassertive. Only positive, nonmodal, nonfuture
simple/perfect phase relations without forward-deictic time may support a
qualified factual answer; progressive, nominal, and free-adjunct `-ing` forms
fail closed.

Also added:

- version-6 snapshots with backward-compatible version 1–6 loading and corrupt
  binding rejection;
- direct and source-qualified Q&A over begun, stopped, continued, enjoyed,
  avoided, and perceived relations;
- compositional realization without completed-sentence templates;
- ambiguity, truth-boundary, persistence, provenance, and conflict regressions.

Local validation:

```text
285 dedicated tests passed
160 generated conformance cases included
2,606 full-suite tests passed, 2 pre-existing expected xfails
29/29 deterministic acceptance turns
7 benchmark cases / 29 turns
compile and diff checks clean
engine/ and clanker_engine.py unchanged
```

Independent review caught and drove fixes for modality/future leakage,
specificity, provenance, conflict, snapshot, and atomicity defects. Final local
correctness and truth-boundary reviewers returned **ACCEPT / no blockers**, and
test/code acceptance was clean. This pre-merge changelog does not predict the
post-commit CI, automated-review, or merge outcome; PR #105 is authoritative.

---

## 2026-09-04 — Embedded interrogative content and attribution

**Issue:** [#89](https://github.com/deucebucket/clanker/issues/89)
**PR:** [#103](https://github.com/deucebucket/clanker/pull/103)
**Merge commit:** `ca992c32e22cdaaf5239a689f11fffa176a31698`

Added deterministic one-level embedded WH and polar-question content while
preserving the outer event separately from the inner question.

Implemented:

- embedded `who`, `whom`, `what`, `when`, `where`, `why`, `how`, `which`, and
  `whose` content;
- embedded polar content through `whether` and complementizer `if`;
- matrix predicates including `ask`, `wonder`, `know`, `remember`, `discover`,
  `determine`, and command-form `tell`;
- direct answer requests such as `Tell me where John went`;
- outer epistemic questions such as `Do you remember when the meeting starts?`;
- typed inner requested roles and polar proposition frames;
- preservation of matrix source, optional recipient, complementizer surface,
  matrix polarity, inner polarity, certainty, provenance, and stable IDs;
- version-5 symbolic memory/runtime snapshots with backward-compatible loading;
- source-qualified Q&A and compositional realization.

Truth boundary:

```text
Sarah asked who called.
```

does **not** establish who called.  Likewise, wondering whether an event
happened does not assert the event, and attributed knowledge content does not
invent a missing answer.

Validation at merge:

```text
2,321 passed, 2 xfailed
181 dedicated #89 tests
160 generated WH/polar conformance cases
29/29 deterministic acceptance turns
```

The two expected failures remain the pre-existing V8 `CARETAKER_TRANSFER` and
`BROADCAST` gaps.

---

## 2026-09-04 — Infinitival control, raising, and desired/planned content

**Issue:** [#88](https://github.com/deucebucket/clanker/issues/88)
**PR:** [#102](https://github.com/deucebucket/clanker/pull/102)
**Merge commit:** `72efa09438b23d34df0d4a072403090063599fb5`

Added selected one-level infinitival complements with typed matrix/embedded
relations.

Implemented:

- subject control: `plan`, `intend`, `hope`, and subject-control `want`;
- object control: `tell`, `ask`, and object-control `want`;
- raising: `seem` and `appear`;
- planned, intended, hoped, desired, directed, requested, and evidential
  content statuses;
- explicit controller and raised-subject binding;
- separate matrix and embedded polarity;
- version-4 snapshot migration and round-trip;
- polarity-aware Q&A, source qualification, multi-controller aggregation, and
  compositional realization;
- fail-closed handling for unsupported nested `to`, competing attachments,
  coordination adjacency, and purpose-adjunct boundaries.

Truth boundary:

```text
Sarah planned to leave.
```

does not establish that Sarah left.  A completion question reports the plan and
returns an explicit unknown unless a later direct assertion establishes
completion.

Validation at merge:

```text
2,140 passed, 2 xfailed
259 dedicated #88 tests
200 generated control/raising cases
29/29 deterministic acceptance turns
```

---

## 2026-09-04 — Finite attributed content complements

**Issue:** [#86](https://github.com/deucebucket/clanker/issues/86)
**Merge commit:** `776c9053a5d3804fe1e64982b1b7ac392fe7e505`

First bounded delivery under complement roadmap [#54](https://github.com/deucebucket/clanker/issues/54).

Added:

- one-level finite declarative complements;
- explicit and zero complementizers;
- typed source attribution;
- separate matrix and content events;
- source-qualified Q&A;
- truth-boundary enforcement;
- snapshot persistence;
- explicit layered-structure ambiguity;
- more than 160 conformance cases.

Reported, claimed, remembered, believed, or otherwise attributed content remains
qualified by source and matrix polarity.  The parser does not convert attributed
content into an unqualified global fact.

---

## 2026-09-03 to 2026-09-04 — Parser structural groundwork

Before the complement-clause slices, the deterministic parser was expanded in
bounded deliveries covering:

- coordinated clauses and shared/independent argument structure;
- subordinate cause, purpose, condition, concession, and temporal relations;
- subject and object relative clauses;
- restrictive and nonrestrictive modifier relations;
- stable relative-gap binding;
- matrix/modifier event separation;
- appositive and possessive-relative roadmap boundaries;
- ambiguity diagnostics rather than silent attachment guesses.

The remaining possessive-relative and appositive identity work is tracked in
[#84](https://github.com/deucebucket/clanker/issues/84).

---

## 2026-09-03 — Adaptive deterministic Clanker-LM runtime

**PR:** [#35](https://github.com/deucebucket/clanker/pull/35)

Established the complete deterministic runtime around Clanker V8.

Core architecture:

```text
input text
→ semantic parsing
→ entity/event/reference memory
→ typed question or conversational act
→ evidence binding / explicit unknown
→ response semantic plan
→ atomic grammar and morphology
→ semantically valid candidates
→ VADUGWI target and candidate scoring
→ deterministic response
```

Included:

- typed entities, semantic references, event frames, question frames, answer
  contracts, certainty, provenance, polarity, and ambiguity states;
- persistent conversation memory and explicit pronoun probes;
- open-world `TRUE`, `FALSE`, `UNKNOWN`, and `CONFLICT` handling;
- active lexical learning with versioned evidence and provisional senses;
- live time, date, and bounded arithmetic resolvers;
- transition-residual learning over multi-turn conversational outcomes;
- nontextual dialogue/book trajectory profiles;
- an atomic language database and abstract grammar rules;
- hard rejection of whole-sentence response templates;
- contextual severity, family relevance, register, uncertainty, and collision
  masking gates;
- an explicit persistence contract and executable CLI/API contract;
- isolation from the tuned V8 engine.

Unknown knowledge is not an exception to the no-template rule.  Missing facts,
ambiguous references, conflicting evidence, and unknown words change the
permitted response act; their visible responses are still composed from atomic
lexemes and grammar operations.

---

## Active gap-closure roadmap

The current parser and complement program is tracked through these issues:

| Issue | Scope |
|---|---|
| [#52](https://github.com/deucebucket/clanker/issues/52) | Relative clauses, appositives, and entity-scoped modifiers |
| [#54](https://github.com/deucebucket/clanker/issues/54) | Complement-clause roadmap |
| [#84](https://github.com/deucebucket/clanker/issues/84) | Possessive relatives and appositive identity links |
| [#90](https://github.com/deucebucket/clanker/issues/90) | Gerund, participial, and aspectual complements |
| [#91](https://github.com/deucebucket/clanker/issues/91) | Held-out complement attribution/control evaluation |
| [#92](https://github.com/deucebucket/clanker/issues/92) | ECM, bare-infinitive perception, and reviewed small clauses |
| [#93](https://github.com/deucebucket/clanker/issues/93) | Shared matrix-predicate and entailment catalog |
| [#94](https://github.com/deucebucket/clanker/issues/94) | Complement realization and reverse validation |
| [#95](https://github.com/deucebucket/clanker/issues/95) | Versioned relation signatures and snapshot migration |
| [#96](https://github.com/deucebucket/clanker/issues/96) | Generated complement fuzzing |
| [#97](https://github.com/deucebucket/clanker/issues/97) | Matrix/embedded graph trace and UI contract |
| [#98](https://github.com/deucebucket/clanker/issues/98) | Complement support documentation |
| [#99](https://github.com/deucebucket/clanker/issues/99) | Staged feature flags and compatibility checks |
| [#100](https://github.com/deucebucket/clanker/issues/100) | Candidate-growth and low-resource performance bounds |
| [#101](https://github.com/deucebucket/clanker/issues/101) | Adversarial attribution and command/content boundaries |

## Permanent engineering invariants

- Do not modify `engine/` for Clanker-LM parser/runtime work.
- Do not store complete response sentences as templates.
- Semantic correctness is a hard gate before affective ranking.
- Mentioned, reported, planned, desired, questioned, perceived, and hypothetical
  content must retain its source and non-entailment status.
- Unknown means unknown; it never licenses invention.
- Ambiguity is represented explicitly rather than guessed away.
- Every new relation type must round-trip through snapshots.
- Every bounded slice receives dedicated tests, generated conformance coverage,
  exact-head CI, automated review, and squash-merge discipline.
