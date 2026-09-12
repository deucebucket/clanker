# Hypothesis core — issue #172

Stacked on #160 only for its existing identity/hash utilities. This publishes
six exact portable files and the 21 portable tests from the separately
committed integrated build. It does not publish its MemoryWeb adapter, modal
word-preview adapter, Atlas styling or accumulated runtime. The optional
.runtime.HypothesisLab described by the package docstring exists only in that
full delivery. Importing the portable facade does not import that adapter.

## Implemented

One bounded relational unifier proposes claims with ordered bindings, frame,
conditions, source records and unmet validation obligations. Three supplied
schemas exercise source-material property, category-member property and a
resource-blockage situation analogy. They are proposal schemas, not valid
universal implications, learned laws or answer templates. The source-material
schema does NOT establish that properties survive manufacturing in either
direction. Candidate discovery is not scientific confirmation.

Existing typed source records feed EvidenceView; no new database or chatbot.
Exact target statements under matching context/conditions support or oppose.
Adjacent records are listed but do not confirm the target. Overlapping declared
source roots count as one group; 100 copies of one origin do not produce 100
independent authorities. The host supplies those roots and source kinds; this
code does not independently verify source identity, copying or factual accuracy.
Hypothesis/self-output/analogy records are ineligible as external observations.

Assessments distinguish proposed, supported_by_recorded_evidence, contested,
opposed_by_recorded_evidence and stale. Probability is null/un-calibrated. Even
supported candidates are not automatically added to factual provers. Withdrawn
motivation loses eligibility; independent direct target support can remain.
Every candidate mapping is rechecked against its original records and rule,
including when its sources were withdrawn. Rehashed altered records do not
substitute for matching bindings. Empty or incomplete search is not a denial.

## Tests and separate local integration

Portable tests (21) explicitly construct source rows, not a simulated claim
that native main already contains the complete graph. They cover changed labels,
reversed roles, conditions/frames/polarity, dependent sources, counterevidence,
rule/evidence mutation, budgets, repeated review and an independent relation-
join oracle across 15 small generated graphs.

The separate local integration at fafddd9... stores observations, candidates,
withdrawals and review history in the existing MemoryWeb. Graph copies commit
under the runtime lock; failure does not expose partial graph edits. A read-only
property preview uses existing Clause grammar, WordDecoder and real V8, producing
'Wood might be flammable.' Its evidence did not prove that property. Nonempty
conditions and situation narration are declined by that preview rather than
silently dropped. Default process() is unchanged.

Local totals: 53 new tests (21 portable + 32 integrated); relevant regression
selection 642 passes / two inherited V8 xfails. Twelve authored lifecycle
exercises across material/category/situation families pass 144/144 checks.
All supplied confirming and contrary observations are SYNTHETIC test fixtures,
not physical tests, source research or scientific confidence measurements.
No broad first-grade post-test or independent human outcome claim is made.

Actual parser probing exposed existing made-from role/type errors and unknown
flammable vocabulary. The experiment therefore declares its trusted typed input
rather than silently repairing the student's parser and crediting it as learning.
This PR does not fix arbitrary language extraction or add autonomous research.

## Changelog / devlog

2026-09-12: add typed proposal matching and source-sensitive lifecycle; reuse
stable existing references; retain no-probability/non-proof boundaries. A new
receipt test initially used selected_path instead of the real winning_path key;
a duplicate-source fixture also initially changed its original source metadata.
These harness fixes are documented, not counted as learner improvement.

New code is modular; Clanker and Soul remain separate. No engine/ or evaluation/
bytes change. No continuous job, external action, new sentence library or default
model activation. Dependency indexing, learned proposal schemas, richer causal
verification, trusted promotion into Q&A and calibrated probabilities remain
future work under #130/#153/#157/#164/#170.

Run:

    python -m pytest tests/test_hypothesis_core.py tests/test_activation_core.py engine/tests -q

Keep draft pending exact-head review, parent integration and the existing
candidate-runtime evaluator binding. Do not alter old manifests, labels or
baseline reports to force the release checks green.
