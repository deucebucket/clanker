# Clanker-LM state-scope repair and mathematical sequencing review

Date: 2026-09-09. Refs #131, #66, #125 and #129.

## Changelog

- Add a shared typed state-report adapter carrying participant, polarity, tense,
  modality, discourse/source and supported temporal scope.
- Consume denied-state evidence before ordinary affective keyword routing in
  the native ResponseActPlanner. Negated sadness is neither asserted sadness nor
  automatically happiness. No stored reply templates are introduced.
- Preserve existing critical/loss/serious gates ahead of this bounded rule.
- Abstain on multi-state/unsupported complements and unnormalized calendar
  references rather than pretending to observe a current self-state.
- Add native-runtime and typed-seam regression tests. No experimental decoder is
  required by this patch. No V8 engine or frozen evaluator bytes are changed.

## Devlog and measured scope

The audit traced main `29e2c5f5900279aced9f58111e64be7c07602802` and the separate
local memory/word-decoder delivery `dc7871f6a9f532d67fc0040f13863727a8a4e5ed`.
PR #124 is another, not-yet-merged experimental path. Do not assume parity.

An actual persistent real-V8 development run selected an empathic follow-up for
`I am not sad` even though its estimated observed valence was 154. This patch
corrects the native policy decision from parsed polarity, not from a new phrase
lookup. The separately tested local graph adoption uses the same helper to stop
linking past/denied feelings as current appraisals. That graph change is NOT in
this native PR; main does not yet contain that delivered graph implementation.

Local native functional tests: 2,792 passed, 2 inherited expected failures,
excluding frozen conversation evaluation and browser DOM modules. Environment:
CPython 3.13.5/Linux. A -S launcher and equivalent child-process shim bypassed
hosted startup instrumentation; production code and test assertions were not
modified by the launcher. Earlier unadapted runs timed out, not passed.

The graph adoption additionally passed 83 focused decoder/memory/scope tests.
These are authored development checks, not held-out emotion or human-outcome
validation. No Python 3.10/3.12 CI success is claimed by this local report.

Known boundary: the inspected parser does not produce supported frames for
some negated -ed copular complements (worried/excited). The helper does not add
that missing morphology coverage. Compound scope, full temporal grounding and
clinical affect interpretation are not supplied here.

## Remaining mathematical order problem (#131)

The current runtime updates state and finalizes the previous response before
it identifies the next input as a definition, query, historical report or
reaction. The same transition operator is being used for observation estimation
and response-effect prediction. This patch does NOT repair that entire loop.

Required dependency order:

1. Freeze versions and stage changes.
2. Interpret words with bounded memory retrieval; preserve competing meanings.
3. Establish participant/time/polarity/source and evidence purpose.
4. Estimate state only from appropriately scoped evidence; unknown axes are not
   neutral observations. Keep estimated and predicted state distinct.
5. Update semantic/appraisal memory and explicit dialogue obligations.
6. Choose the act, ideal target and reachable response constraints.
7. Generate through executable grammar and lexical/VADUGWI search.
8. Validate, atomically commit and deliver the supported response.
9. Assign feedback credit only to eligible, relevant later evidence.
10. Learn the appropriate scoped weights/abstractions at a new version boundary.

A concrete check of the existing solver: for a synthetic A_v=20, its maximum
one-step C_v is round(0.6*20+0.4*255)=114, whereas the critical target policy can
request 135. Exhaustive enumeration of all 65,536 B_v/B_i pairs confirmed the
bound. Other response axes cannot change output V in this exact operator.
This is a one-step property of current code, not a ceiling on intelligence or
an assertion about actual human emotional change.

## Release gate

Draft only. Production changes require the candidate-runtime evaluation binding
tracked in #123. Preserve old corpus/benchmark identities and reports; do not
rewrite held-out material, relax the integrity gate or claim the whole suite is
green. Request exact-head review and CI before merge or activation.
