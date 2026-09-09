# Dialogue corrections and scoped usage learning

## Identity and scope

This is a follow-up to PR #124 at `c58ed145f08c41b230d88b958ebb0110c692a16c`,
for issues #126 and #127. It repairs the **experimental** `ReceiptChat` path.
It changes no `engine/`, `clanker_lm/`, or frozen `evaluation/` code/data.
It is not an integration of the different, locally delivered `clanker_lm.decoding`
package or the separate associative-memory prototype described under #125.
That graph reconciliation and default-runtime promotion (#123) remain distinct.

## What now participates

- Owned-object NPs use the stored possessor ID: user -> `your`, assistant ->
  `my`, third party -> that party's possessive. Literal quotations are not
  globally rewritten. Attributed clauses are still coverage-gated.
- Pending word questions accept bounded explanatory fragments, not every short
  declarative. Independent facts continue to semantic storage while the probe
  remains pending. Speculative parsing uses a separate memory copy.
- Definition negation suppresses the negated clue instead of learning its
  opposite. `not good`/`not bad` alone remain unresolved; a supported contrast
  such as `not bad, actually excellent` contributes positive evidence.
- The active episode records entities, stored event IDs, an outstanding
  elaboration question, whether an explanation arrived, and a no-elaboration
  preference. These records constrain the actual response act and bind the
  decoder context receipt. They survive snapshots and failed-turn rollback.
- Narrow `return + it/them + late/early` parsing keeps the adverb out of the
  object identity. Spoken answer bindings refresh discourse salience.
- An explicit `Actually, ... new object, not old object` correction supersedes
  exactly one matching positive simple event in the answerer's active view.
  Original facts remain stored. Time may be inherited from that uniquely
  identified prior event, with the dependency recorded. Ambiguous, quoted,
  modal, negative, and embedded replacement forms abstain; ordinary conflicts
  are not treated as corrections.
- Gratitude and closure are actual social acts. A reviewed request to stop
  asking the generic elaboration question sets a stored preference. Output is
  composed through Clause/Slot operations; no finished reply is retrieved.

## Learning connections that did not exist

`UsageLedger` records scoped counts from explicitly admitted user/development
observations. It creates unigram, bigram, and trigram entries as evidence arrives.
An immutable per-turn `SessionAffinity` view combines those counts with the
reviewed base pack. The next decoder run uses that view, so the weights are
not merely displayed in a log.

```python
from experiments.lexical_decoder import ReceiptChat

with ReceiptChat(usage_scope="my-local-session") as chat:
    chat.observe_usage(
        "Go to the microdax.",
        evidence_id="example-1", source_id="speaker-1", consent=True,
    )
    result = chat.process("Hello.")
    # The new word/context edge now has support. The observation did not
    # establish a world fact or license inserting microdax into this greeting.
    chat.retract_usage("example-1", reason="Incorrect usage example")
```

Interactive equivalent:

```bash
python -m experiments.lexical_decoder --usage-scope local --snapshot /tmp/clanker-session.json
```

Use `/hear ID text` to explicitly submit usage evidence and `/retract ID reason`
to remove that evidence's active contribution. Normal chat is NOT automatically
captured for n-gram training. `/receipt` shows the actual last decision.

Every observation has its own identity, source identity, content hash,
count deltas, purpose and consent flags. Replaying an identical example from
one source is idempotent even under a different label. Retraction leaves the
original record intact; zero-support edges disappear from the active overlay.
It does not delete reviewed base counts or erase audit history.

There are at most 256 observations, 128 tokens per observation, and 16,000
observed tokens per ledger. The existing session/search/receipt limits remain.
Source and consent flags are CALLER ATTESTATIONS, not authentication or a
classifier that detects secretly relabelled held-out text. Official ingestion
rejects `assistant`, `heldout`, `evaluation`, `simulation`, unconsented and
marked quoted sources. No global promotion or implicit self-training exists.
Input-derived counts may reveal short phrases: this state and its receipts are
private-session data, not anonymized public telemetry.

## What the ablation proves

With identical semantic plan, conversational state, target, search settings and
real V8 scoring (affect weight 24), five original development examples change:

```text
That sounds difficult. What happened?
-> That sounds frustrating. What happened?
```

For `frustrating` after `that sounds`, the base count is 1 of 2 and the updated
count is 6 of 7. Add-half smoothing produces probabilities 3/88 and 13/103,
respectively, because vocabulary size also changes. These are lexical usage
probabilities, not truth/confidence scores. Retraction restores the original
preference. A separate test creates a previously absent `to the -> microdax`
edge, and another shows frequent incorrect content/agreement cannot change a
supported answer's Honda or its past-tense verb.

Evidence: `docs/evidence/usage-learning-ablation.json` and
`docs/evidence/dialogue-repairs-transcript.md`.

## Evaluator binding

The frozen conversation manifest identifies the exact production/evaluator/
corpus bytes behind its results. Its identity guards must not be disabled or
its held-out labels retuned to make changed code pass. A candidate-runtime
comparison needs an explicit protocol and new result identity (#123), while
preserving the baseline and held-out split. Keeping this patch experimental
allows the existing production-byte checks to keep operating unchanged.
The unrelated shallow-history CI defect is repaired on the parent PR.

## Validation and limits

Focused local validation: 97 tests passed (51 inherited decoder tests and 46 new
repair/learning cases). Native CPU development benchmark: 120 turns, 20 fresh
sessions, one selected CPU, 64 MiB RLIMIT_AS, CPython 3.13.5 with `-S`: mean
3.009 ms, p95 4.379 ms, p99 5.052 ms, process peak 31,842,304 bytes. This small
fixture has no browser/network/concurrency measurement and does not exercise a
maximal learning ledger. No artificial typing delays are used.

A broad intermediate local run passed 2,984 tests with one packaging failure
because `python -m build` was unavailable, two environment/data skips and two
pre-existing V8 xfails. That run is not substituted for exact-head supported-
Python CI; the final PR must report the final CI result separately.

Remaining limitations: generic acknowledgments remain repetitive; recovery
language does not yet fully drive new response acts; qualified/progressive
questions still have coverage limits; this episode bridge is not the complete
entity/event/appraisal graph; learned occurrence weights do not invent safe new
grammar or semantic predicates. Rare language is not automatically nonsense.
An explicit correction retracts evidence; lack of frequency is not a deletion
rule. Snapshot v2 is bound to the exact runtime, base pack, scope and policy;
old experimental snapshot v1 is not silently migrated.
