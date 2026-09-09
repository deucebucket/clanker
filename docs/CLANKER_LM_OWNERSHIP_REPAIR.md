# School round 2: native ownership and polarity repair

Date: 2026-09-09. Issue #137; follows curriculum PR #135.

## Changelog

- Delay predicate-derived owner association until the parser knows that an
  outer positive present simple statement was asserted. Questions, negation,
  modality, attribution, past tense and conditional relations cannot assign it.
- Route explicit OWN evidence through the existing provenance/polarity matcher
  instead of bypassing conflicting or qualified evidence via owner_id.
- Require a WH binding to agree with the query's polarity. A negative-only
  source does not answer a positive missing slot.
- Compare opposing evidence on the same closed proposition after normalizing
  polarity; retain the requested value and aspect to avoid false conflicts.
- Recognize present has/does/goes and finite-auxiliary tense precedence.
- Add native regression cases requiring neither a word decoder nor memory graph.

## Devlog

The original authored foundation report card caught the missing possessor role
in the separate delivered decoder. Once that was connected, more demanding
questions exposed a native side effect: parsing a proposed owner wrote owner_id
before speech act, polarity or attribution had been established. A superficially
UNKNOWN reply could still leave a false owner behind for the next question.

Removing all ownership association broke the existing body-part association
test. Rather than remove its assertion, the repair defers legitimate positive
association until scope is known. This then exposed HAS misclassified as past
because its irregular lemma differs from its surface. The finite-tense repair
fixes that cause and preserves the association contract.

The next boundary test discovered that WH answering used positive_matches or
matches: explicit negative-only evidence could supply a positive answer. The
new path returns UNKNOWN with the contrary evidence retained. The conflict
comparison also retained polarity and therefore never intersected opposite
claims; it now compares the same bound proposition with polarity normalized.

## Executed validation

Native candidate based on main 29e2c5f... (the later school merge does not alter
its production bytes), real V8, CPython 3.13.5:

```
python -m pytest tests/test_ownership_evidence_boundaries.py tests/test_parser.py tests/test_qa_runtime.py -q
110 passed
```

The report is a manually implemented repair tested on authored development
cases, not autonomous grammar learning or general academic validation. The
separately delivered memory/word-decoder plus this repair reaches 216/216 checks
on its 54-episode foundation suite; that is a different code line, not the
native default's score. Its per-experiencer graph changes and reuse of the
state-scope adapter from PR #132 are not included in this narrow PR.

## Release boundaries

No changes to engine/, evaluation/, corpus generations, or old results.
Production parser/QA/lexicon bytes do change. The frozen evaluator therefore
correctly rejects running this candidate as the old baseline. Keep this PR
draft until candidate-runtime binding #123, exact-head full CI and independent
review are complete. No assertion is skipped or relaxed to conceal that gate.

This repair does not establish legal ownership, general truth maintenance,
quantity/plural ownership, arbitrary tense resolution or comprehensive
co-reference. Definite surface mentions may add aliases without assigning an
owner. Explicit possessive NPs retain the existing separately bound association
behavior. Source-qualified or conflicting claims remain evidence, not a
frequency vote. Existing failed-turn atomicity remains a separate concern.
