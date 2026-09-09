# Clanker school: foundations diagnostic

Issues #133 and #134. This is development tooling, not another chatbot or a
production response library. Every student reply comes from ClankerLM.process.
No engine/, clanker_lm/, or evaluation/ bytes change; current package discovery
excludes this experiment from production distributions.

## Run

```sh
python -m experiments.school --phase development --output /tmp/school-dev.json
python -m experiments.school --phase transfer --output /tmp/school-transfer.json
python -m experiments.school --phase transfer --candidate temporal-suffix-v1 --output /tmp/school-candidate.json
python -m pytest tests/test_school_curriculum.py tests/test_school_temporal_repair.py -q
```

A diagnostic returns zero when it completes, even if the student fails tasks.
`--require-ready` returns 2 if an exercised skill fails. Tool success is not
student competence.

## Curriculum and evidence

Nine exercised skill families cover identity, perspective, time attachment,
borrower/source roles, ownership non-entailment, event continuity, correction,
dialogue obligations and appraisal separation. Academic transfer is an
unassessed downstream node; failed prerequisites keep it blocked.

Three original development worlds and three differently named transfer worlds
produce 54 short episodes, 180 turns, and 216 checks. Each episode has one
persistent runtime; episodes start fresh. This is local continuity testing,
not one 180-turn natural discussion. Both phases share an authored generator
and are NOT independent held-out research evidence.

Scenario state supplies expected role bindings independently of the student's
parser. Checks inspect roles, requested variables, selected answer values,
object identity and specific surface obligations, not equality to a finished
reply. The examples are development input, never production reply templates.
The appraisal check establishes only record coverage, not accurate feelings,
attribution, target attachment or clinical validity.

Assessment replaces write-side lexical and trajectory learning hooks while
retaining semantic memory and ordinary generation. Existing pinned parameters
may be read. Learned parameter tables are hashed before/after; unexpected
mutation stops the assessment. This is an explicit configuration, not a claim
that live learning behavior is unchanged.

An executable prerequisite DAG drives next-lesson scheduling. Three successful
distinct authored contexts is an engineering gate, not a calibrated mastery
probability. Duplicate context replay cannot add support; regression revokes
readiness and blocks dependents. Different student/configuration identities
need separate evidence. The bounded chained ledger preserves revisions;
hashes detect modification, not teacher authenticity or correctness.

Tutor cards identify failed checks, expected/actual values, episode and
evidence IDs, and missing prerequisites. They do not update weights, invent
new grammar, or insert a teacher's proposed sentence into user memory. An
autonomous tutor service is not implemented by this delivery.

## Narrow candidate repair

TemporalSuffixParser overrides two existing parser seams, not the final
response text. It separates a lower-case final yesterday/today/tomorrow/tonight
from a complete non-temporal prepositional phrase before entity resolution.
It also adds the missing WHO/from source-variable transformation while keeping
the direct object fixed. The original code can store 'Alex yesterday' as a
source and can ask for patient when the user requested a source person.

The candidate is manually implemented after observed failures, not a grammar
rule autonomously discovered by the student. Ambiguous time conflicts raise
an explicit candidate exception. It does not fix legacy failed-turn rollback,
general named-entity typing, arbitrary PP ambiguity, or broad temporal logic.
Proper-name/quote boundaries are conservative. Future-adverb stress tests
check syntax separation, not temporal coherence of every authored sentence.
After snapshot restoration the candidate must be explicitly reapplied; default
snapshots do not serialize an experimental parser class.

## Executed local results, 2026-09-09

Real V8, CPython 3.13.5. Each phase has 27 episodes / 90 turns / 108 checks.

| Student | Episodes passing in each phase | Checks passing in each phase |
|---|---:|---:|
| Main 29e2c5f... | 9/27 | 60/108 |
| Same main plus candidate | 15/27 | 78/108 |
| Delivered dc7871f... word decoder | 15/27 | 84/108 |
| Delivered decoder plus candidate | 21/27 | 102/108 |

The delivered code is a separate previously supplied Git bundle, not merged
main or an assertion about PR #124. `--decoder-pack PATH` explicitly enables
that adapter where clanker_lm.decoding is installed; absent support fails,
never silently falls back. Its original tiny count fixture is a wiring asset,
not broadly trained language.

The repaired delivered line schedules non_entailment next instead of
time_attachment. Ownership-question realization and other-person appraisals
still fail. The school keeps them visible and does not graduate the system.
47 dedicated tests pass locally. These counts are not the full release gate.

## Release boundary

Keep this opt-in until exact-head review and CI. Default parser integration
requires a separate candidate-runtime evaluation binding; do not overwrite an
old corpus, relabel old results or bypass frozen evaluation. No external
textbooks, private chats, sealed evaluation data or self-output training are
used. The primitive skill graph is a proposed curriculum, not established
human developmental science.
