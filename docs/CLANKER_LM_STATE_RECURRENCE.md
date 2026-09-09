# State recurrence and observation history — school round 5

Refs #72, #141 and #125. Native subset stacked on #142, not main. The local
integrated adoption extends delivered commit 168a035...; do not confuse that
code line with the default runtime or experiment PR #124.

## Changelog

- Recognize a bounded positive AGAIN operator before person/NP parsing.
- Store state_recurrence=recurred with the licensed surface marker and normal
  event signature; never use a completed response phrase as the operation.
- Keep recurrence separate from cessation, negation and continuation.
- Preserve tense, person, degree, source and modality at shared state boundaries.
- Require explicit recurrence evidence for an affirmative recurrence-qualified
  query; an unqualified positive state alone does not prove recurrence.
- Retain cross-turn state observations rather than mutating an older identical
  state event. Same-turn storage stays idempotent and non-state facts retain
  their prior deduplication policy. Reusing an old observation ID at a new turn
  is rejected rather than creating two records with the same identity.
- Version component answer contracts; validate recurrence metadata and live
  evidence at selection and realization seams.
- Add 38 native regression cases and a focused Python 3.10/3.12 workflow.

## Integrated adoption (separately delivered)

The existing MemoryWeb stores the same operator on each appraisal. A current
recurrence creates a new report; it never reopens or rewrites the old event ID.
It links to compatible, earlier, source-validated report records and to an
available current withdrawal record. These are associations with recorded
observations, not proof of elapsed duration, a new cause, or exact episode count.
When no earlier report exists, prior_occurrence_unresolved is true. The word
'again' does not manufacture a dated prior event or an imaginary memory.

Generic typed grammar emits the atomic recurrence modifier after the state
complement and before a temporal adjunct. The existing word decoder performs
lexical/VADUGWI scoring and supplies the execution receipt. Legacy completed-
sentence generators are replaced by raising stubs in integration tests.

## Executed local report card

Real V8, CPython 3.13.5; original small authored affinity pack; same exam script
before and after. 51 authored continuous episodes / 288 runtime calls:
18/51 before -> 51/51 after. Families cover reactivation, missing prior
occurrences, person/source/time separation, renewed-then-ceased cycles,
explicit recurrence questions, historical and attributed recurrence, and degree.
Parameters are frozen during the exam; each episode retains conversational
memory. These are development/changed-person exercises, not independent human
or academic generalization evidence and not autonomous grammar learning.

The prior foundation216/216, composition18/18 and cessation45/45 remain intact.
The original 28-turn discussion completes. The 200-turn retention workload
completes, with 25/25 selected recall checks and 100/100 matching post-reload
turns (replies, states, receipts, semantic memory and graph). A ten-cycle
stop/restart test preserves every original state observation and event ID.
69 new native/integration tests pass locally. The native subset independently
runs without the separately delivered graph/decoder; graph-specific results
must not be presented as its default-main behavior.

## Development findings

AGAIN previously occupied a state value, a time slot, or even a person's name
('Jordan again'), depending on word order. The scoped parser operation removes
that ambiguity at the licensed local position before entity resolution.

Existing exact-fact deduplication also mutated earlier state observations when
a sequence returned to the same assertion. Preserving state observation turns
fixes repeated positive/ceased/renewed cycles without deleting earlier graph
references. It increases state-history storage linearly; compaction/retention
and old-snapshot migration are future work, not a claim of unlimited memory.

A newly written integration assertion initially compared whitespace-split
'again.' with 'again'. Twelve apparent failures were grader punctuation errors;
changing the check to actual lexical tokens fixes the test, not the student.
The unchanged inherited state/memory regressions passed throughout.

## Boundaries and release discipline

Positive simple BE/FEEL only; selected modifier positions. Negative recurrence,
combined cessation/recurrence, ambiguous conjunction scope, arbitrary fronted
AGAIN, general continuing STILL and absolute dates are not implemented here.
Historical/modal/reported/conditional/quoted content cannot revive a direct
current appraisal. The original input evidence remains qualified. A direct
report is not independently measured emotion or a clinical assessment.

The next harder probe remains: positive STILL after cessation is unsupported
by the state interpreter. AGAIN and STILL must not be conflated to pass a test.
'Stopped feeling', mixed time, nested reports and prior snapshot reinterpretation
also remain tracked boundaries. Operator absence does not prove its negation.

This native PR carries parser/memory/evidence primitives only; full graph and
word-grammar adoption is separately delivered until branch reconciliation.
Neither engine/ nor evaluation/ changes. The parent stack changes production
bytes, so #123 candidate-runtime binding, exact-head CI and independent review
remain required. No old manifest, baseline, held-out label or integrity check
was rewritten to manufacture a green release. Keep draft before those gates.
