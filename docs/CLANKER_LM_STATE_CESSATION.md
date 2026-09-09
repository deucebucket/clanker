# State cessation and revision — school round 4

Refs #141; stacked on #140 and #132. The native subset is not the separately
delivered integrated memory/decoder runtime. It does not install another chat
engine or activate a graph that is absent on native main.

## Changelog

- Interpret a bounded cessation operator before ordinary negation stripping.
- Carry the operation in EventFrame arguments and normal event signatures.
- Support reviewed no-longer and not-anymore forms for BE and FEEL states.
- Reject ambiguous/double-negated operators rather than flattening them.
- Preserve original evidence, source, subject, time, degree and modality.
- Share degree-aware withdrawal semantics across indexed state answers.
- Add evidence-frontier guards for current WHO, polar and HOW state answers.
- Check selection again at realization; changed or missing denial records cannot
  quietly resurrect an old positive answer.
- Unsupported current cessation compounds remain unresolved rather than
  distributing negation or treating an old affected label as confirmed.
- Preserve serious handling for an explicitly current self-denied safety state.
- New semantic tests and dedicated Python 3.10/3.12 native/engine CI.

## Development log

The previous local delivery (89f7545...) retained anger after a simple
'no longer angry' report. The parser had already stripped 'no', leaving a
negative 'longer angry' complement. A downstream string substitution would
also accept 'not no longer angry' incorrectly. The new clause-level scanner
identifies the phase operator before that information is lost.

The scanner recognizes operator tokens, not completed user or response
sentences. The native parser adds a typed state_change=ceased argument and
preserves the exact raw evidence separately. Identity binding occurs after
the operator is identified, preventing 'Jordan longer' becoming a person in
'Jordan no longer feels angry'.

The existing state-scope/component helpers consume the same representation.
Current appraisal selection and guarded WHO/polar queries follow the same
source-qualified revision frontier. Cessation removes only matching current
support; it neither deletes original events nor establishes an opposite mood.
A degree-limited denial (no longer VERY angry) is not proof of no anger at all.

## Native versus integrated scope

Native changes in this PR: cessation scanner, parser hook, state-scope helper,
state-description selection/guards and response-policy safety preservation.
They are independently tested without the delivered graph or word decoder.
No engine/ or evaluation/ files change.

The separately executed local adoption additionally wires the helper into
MemoryWeb, append-only revision links, query dispatch, and generic typed
BE/FEEL clause realization. It emits 'no' and 'longer' as atomic grammar output,
not stored reply templates. Tests prohibit legacy completed-sentence generation.
The local graph records which earlier current candidates a supported denial
withdraws, while retaining their historical evidence. Those integrated files
are delivered separately and are not silently included in this native subset.

## Executed local report card

Real V8, CPython 3.13.5, same original small authored affinity pack:
45 authored continuous episodes / 228 turns: 21/45 before -> 45/45 after.
The categories include same-label revision, unknown rather than invented
recovery, other-person/history/scope preservation, WHO/polar consistency,
reported cessation, degree-limited denial and simple positive reassertion.

Original foundation: 54/54 episodes, 216/216 checks. Previous state-composition
exam: 18/18 episodes. Original chat: 28/28 replies. Repetitive retention test:
200/200 replies, 25/25 selected recall checks and 100/100 resumed suffix matches
including graph, semantic memory, states and receipts.

99 new semantic/integrated tests pass. The applicable functional repository
selection reports 3341 passes and two pre-existing V8 xfails. It excludes the
frozen evaluator and browser-DOM module; it is NOT a full release claim.
A separate evaluator-module run reports 121 passes, six failures, five setup
errors and one skip: ten failures/errors are version binding, one failure is
unavailable local build tooling. Unfiltered aggregate attempts timed out in
this host and are not counted as completed successes.

These are authored development regressions, not independent human-outcome or
clinical evidence. Assessed parameters are frozen while episodic memory works.
No private chats, held-out training or automatic grammar induction is added.

## Bugs caught in this implementation

An initial stricter filter stopped recording modal appraisals. Existing tests
caught it; recording now retains their uncertainty while current-assertion
retrieval still rejects them. No old test was relaxed to conceal the regression.
The first native publication patch applied all modules except the parser,
because the local parser includes unrelated prior continuation changes. The
final native adaptation checks the exact parent blob and applies only the
cessation seams. The original patch hash and native baseline checks remain
intact. Staging payloads and publication workflow are not in the final tree.

## Limits and release gates

Not arbitrary temporal logic, recurrence grammar, absolute-date anchoring,
modifier semantics or proof of psychological recovery. 'Angry again' after
cessation is a newly reproduced gap: simple reassertion works, but the recurrence
qualifier is not yet understood by this state-component path. 'Stopped feeling'
and old snapshots that already lost cessation syntax require additional work.

Negative compounds do not distribute. Their typed unresolved status is not an
assertion that each component ended. Quoted, reported, modal, conditional and
other-person statements do not rewrite unrelated current appraisals.
The global observer/controller ordering audit remains #131.

Keep draft: #123 candidate-runtime evaluation binding, parent integration,
exact-head CI and independent review remain required. Do not change sealed
corpus labels, old benchmark identities or checksums to manufacture green CI.
