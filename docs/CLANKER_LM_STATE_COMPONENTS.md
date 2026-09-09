# Qualified state components — school round 3

Refs #139, #125, #131/#132 and #94. This native subset is stacked on the
state-scope adapter in #132. It is deliberately a shared semantic primitive,
not a second chatbot, graph implementation or automatic default activation.

## Changelog

- Add bounded interpretation of simple BE/FEEL states and positive AND lists.
- Preserve source event, subject, intensifiers, polarity, modality, tense and
  supported session-local temporal cues without mutating original evidence.
- Resolve a graph-indexed current/historical state set into an AnswerContract,
  checking graph fields against live event IDs and content hashes.
- Validate all selected component references again at the realization boundary.
- Validate a reporter/matrix/finite-content chain for explicit reporting queries.
- Keep negated conjunction, OR/BUT, unsupported dates, modal conjunction,
  missing or changed source evidence and out-of-budget results explicit.
- Add 51 native/typed-seam regression tests and Python 3.10/3.12 focused CI.

## What is and is not wired here

`state_components(event)` is the single semantic projection used by the
separately tested graph extractor and the new word realizer. It returns source-
linked StateComponents rather than a completed surface sentence.

`answer_from_appraisals(question, memory, rows)` validates caller-supplied graph
records, handles per-label current denials within supported scope and produces
an answer bundle retaining every selected state component.

`resolve_state_bundle(contract, memory)` checks the query, originals, selected
values, component identities and temporal scope before a generator uses them.

`resolve_reported_state(contract, memory)` binds the reporter to an existing
matrix/content relation. It accepts only the supported positive reporting or
belief matrix and never removes the reporting qualification or inner negation.

The default main runtime does not yet contain the separately delivered
MemoryWeb or the same word decoder. This PR therefore does NOT claim that
adding the helper alone makes default chat answer these questions. Full
adoption must reconcile #123/#125 and the existing experiment code lines.
The local end-to-end integration changes the existing graph and decoder seams;
it does not retrieve example sentences or call the legacy completed-sentence
realizer to manufacture a receipted response.

## Executed observations

On delivered student d2d544b... plus these helpers and its graph/decoder adapter:
18 original authored episodes / 63 turns improve from 0/18 to 18/18. There are
six multiple-state, six historical-state and six attributed-state exercises.
These share an authored generator and are not independent held-out research.

Actual repaired outputs include:
- Jordan is angry and sad.
- Jordan was really angry yesterday.
- Sarah said that Jordan was not angry yesterday.

The transcript above is documentation of measured output, never production
reply data. Direct user reports are evidence records, not independently
measured emotions. VADUGWI estimates remain estimated; no clinical validity or
human benefit is established by these tests.

The original 54-episode foundation unit remains 216/216. The 28-turn discussion
and 200-turn stress workload complete without refusals; the 100-turn resumed
suffix matches replies, states, receipts and graph records. These are results
of the separate local integrated delivery, NOT the default native PR.

The native helper is tested without a graph/new decoder: 51 focused tests pass
locally on Python 3.13.5. Graph-index tests explicitly provide typed seam rows;
they do not pretend native main contains the full graph. The reporting tests
use real ClankerLM memory and its existing Q&A contracts.

## Development log

The old graph required exactly one current state, ignored historical FEEL
queries, and the delivered decoder rejected finite source qualification.
The replacement shares an interpretation rather than maintaining three
independent string heuristics. A graph/contract cannot substitute a different
experiencer or reporter while retaining the original evidence.

An early strict literal-only check regressed 'I feel better': the legacy parser
represents its bare complement as better_1. A bounded FEEL projection now
accepts fully licensed lower-case state constructions without changing the
stored entity, accepting arbitrary objects or resolving proper names as moods.
The existing recovery test remains unchanged and now passes again.

An early authored grader expected 'angry' even when the intended input was
'really angry'. The student was correctly retaining that modifier. The grader
was corrected to require the whole qualified state; that grader correction is
not presented as student learning or an accuracy gain. Another new test assumed
a nonexistent AnswerContract.from_dict; the final test uses actual bundle JSON
and runtime snapshot APIs instead. Harness errors are not student failures.

## Limits / review

Four components maximum. No general temporal calendar or deictic anchoring
across dates is supplied; literal session cues are compared explicitly.
A current-state set is the active recorded report set, not a measurement of
someone's mind or a claim that old reports never expire. Reported-only evidence
does not become a direct answer to a current-state question.

Negated conjunction is not distributed. No arbitrary nested content, mixed
conjunct time, broad grammar induction, autonomous tutoring or new language
corpus is added. Component schemas carry data, not executable code or completed
response templates. Content hashes detect mismatch; they do not authenticate
teachers or independently establish truth.

Production bytes are new, and the parent already changes response policy.
Full evaluator binding #123, exact-head CI and independent review remain merge
gates. No frozen source corpus, baseline result, checksum, engine equation or
integrity assertion has been modified. Keep the PR draft pending integration.
