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

## Current branch — Gerund, participial, and aspectual complements (#90)

**Branch:** `feature/clanker-lm-gerund-participial-complements`  
**Starting commit:** `ca992c32e22cdaaf5239a689f11fffa176a31698`

### Intended forms

```text
Sarah enjoys reading.            gerund content
John stopped smoking.            aspectual stop
Mary started working.            aspectual start
They kept talking.               aspectual continuation
I saw Sarah leaving.             perception participial
Sarah avoided calling John.      noncompletion/avoidance
```

### Required distinctions

1. **Gerund complement versus noun phrase**
   - `Sarah enjoys reading` may select an event-like complement;
   - an ordinary nominal use must not be forced into an event.

2. **Complement versus progressive aspect**
   - `Sarah is reading` is a progressive matrix predicate;
   - `Sarah enjoys reading` contains a selected gerund complement.

3. **Complement versus free adjunct**
   - an `-ing` clause that is temporal, causal, manner, or discourse-adjacent
     must not be attached as selected content without licensing evidence.

4. **Entailment status**
   - `avoided calling` does not establish a call;
   - `started working` establishes an onset relation;
   - `stopped smoking` may support prior activity only as a typed derived
     inference;
   - `saw Sarah leaving` remains perception-attributed.

5. **Controller binding**
   - the matrix predicate catalog determines whether the matrix subject,
     object, or an explicit embedded subject controls the `-ing` event.

6. **Polarity and source**
   - matrix and embedded negation remain separate;
   - perceived or reported embedded content remains source-qualified.

### Planned acceptance gate

- typed relation families and ambiguity diagnostics;
- snapshot version update and migration;
- at least 160 dedicated/generated cases as scoped in #90;
- Q&A over begun, stopped, continued, avoided, enjoyed, and perceived events;
- compositional realization with no sentence templates;
- full regression and 29-turn acceptance harness;
- exact-head review;
- no `engine/` changes.

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
active issue:     #90
latest full test: 2,321 passed, 2 expected xfails
acceptance:       29/29 turns
V8 engine files:  unchanged
response mode:    atomic/compositional; no whole-sentence templates
```

The next code change on this branch should implement the typed `-ing` complement
schema and parser boundary tests before adding broad verb coverage.
