# Verified concept learning: foundation core

Issue #148; related #110, #125, #130 and #133. This PR publishes the exact
reusable core exercised by the separate integrated round-eight delivery. It
is not a new chatbot, automatic default activation, or a claim that native main
already contains that delivered account-scoped graph and word decoder.

## What is learned

The immutable seed declares six internal sorts and five inclusion relations:
person, object, event, state and concept are kinds of entity. These are explicit
internal ontology conventions, not a textbook corpus or externally validated
scientific facts. Private teaching cannot modify that seed.

The mutable per-compartment ledger learns new concept symbols, directed
inclusions and individual memberships. Two manually implemented inference
rules compose them: inclusion transitivity and membership inheritance.
Reflexivity for a known type is a logical convention, not proof of existence.
No new inference algorithm, general grammar, or equation is autonomously learned.

For example, taught dax -> tool -> object and an individual's dax membership
can support a new entity-membership answer through the seed's object -> entity
edge. Neither the complete answer sentence nor a precompiled dax relationship
is present in production. The examples here and in tests are authored inputs,
not response templates.

## Proof and correction policy

Every query has a scope, seed/ledger/limits generation, exact relation and
polarity. A bounded deterministic search produces supporting record IDs,
source-lineage attestations, derivation steps, negative evidence, blocked
conflict dependencies, search-budget status and a content receipt hash.
Verification reruns the computation; rehashing a fabricated result is insufficient.

Negative inclusion means NOT EVERY X IS Y, not NO X IS Y. It does not become a
negative statement about every individual. Missing support is UNKNOWN.
Conflicting endpoint evidence is CONFLICT; contradicted intermediate links
cannot support downstream claims. A separate uncontradicted path may still
support a result. This is a deliberately bounded policy, not complete first-order
logic or a proof that the supplied premises describe the external world.

Withdrawing an exact private premise appends evidence rather than deleting
history. Subsequent queries recompute from active premises. Existing receipts
become stale when their generation changes. Duplicate same-lineage claims do
not increase active support; lineage metadata is caller attestation, not proof
of source independence. Cycles never manufacture an individual or unrelated fact.
Instance IDs and concept names occupy different logical domains even when
spelled identically.

## Native use

```python
from clanker_lm.concept_learning import Claim, ConceptLedger
from clanker_lm.concept_learning.core import digest

brain = ConceptLedger('trusted-account-and-compartment-id')
claim = Claim('dax', 'is_a', 'object')
brain.teach(claim, evidence_id='lesson-1', lineage='reviewed-author',
            content_sha256=digest(claim.__dict__))
proof = brain.resolve(Claim('dax', 'is_a', 'entity'))
brain.verify(proof)
```

The caller must supply the trusted identity. This is not authentication or an
access-control service by itself. Hashes are consistency checks, not signatures.
The integration uses the existing owner-bound service envelope separately.

## Exact local evidence

36 core tests pass, including 480 query comparisons against independently
computed set closure over 30 small generated positive graphs. These are not 480
independent human judgments or 480 pytest test functions.

The separately delivered student based on 8226759... uses this same core through
its real ClankerLM.process(), typed grammar, V8/word decoder and Memory Atlas.
On six authored worlds, the same executable changes from 0/30 supported novel
answers before teaching (correct UNKNOWN) to 30/30 after explicit connecting
premises. It also exercises withdrawal, contradiction, unchanged other-account
snapshots and reload. The combined local native/integration selection passes
62 tests. These observations belong to that integrated delivery, not default
main or this standalone helper alone.

The integrated snapshot envelope is version 7 while existing symbolic memory
remains version 6. Old version-number assertions were updated specifically for
that declared schema migration; semantic round-trip assertions were retained.
That runtime migration is NOT silently included in this native subset.

## Changelog / devlog

- Introduce scoped, revisioned conceptual teaching and bounded derivations.
- Publish the exact seed identity and source-linked proof receipts.
- Preserve negative quantifier meaning, conflicts, withdrawal and alternatives.
- Detect a class/instance equal-spelling cycle bug and test separate identity domains.
- Keep private lessons outside immutable seed and lexical count packs.
- Add native proof, scope, replay, generated-graph and budget tests.

The first integrated exam had harness errors: absent optional learning metadata
and a control-memory question incorrectly included in the novel-answer
population. Those were corrected in the grader; they are not student gains.
A single early reload mismatch was not captured with a state diff; later
instrumented replay, every-prefix tests and fixed-hash-seed runs must be judged
on their retained evidence, not treated as a diagnosed fix of that observation.

## Boundaries and release gates

This native PR does not wire the default parser to new teaching constructions.
The full local integration additionally accepts a narrow explicit Every/kind-of
and named-member teaching grammar, composes quantified replies atomically,
revalidates proof contracts before word selection, exports the actual graph,
and supports operator-level premise withdrawal. Reconcile that delivery with
#123/#125 and existing experimental PRs before default activation.

No arbitrary textbook ingestion, broad lexical/grammar induction, spontaneous
source verification, automatic global promotion, assistant-output reinforcement,
or private/held-out training. Labels are bounded single-word symbols; inference
uses at most 16 edges and 2048 expansions by default, with 256 concepts/2048
ledger operations. Budget exhaustion is not a negative answer.

No engine/ or evaluation/ changes. New production assets still require a valid
candidate-runtime evaluation identity. Keep draft until exact-head independent
review, full CI, and existing branch/evaluator integration gates are satisfied.
