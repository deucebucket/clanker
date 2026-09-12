# Typed equation composition — #170

This optional provider adds scoped ground equations, explicit typed operation laws, bounded substitution/regrouping proofs and source withdrawal. Existing Clanker, clanker-soul, default Clanker-LM generation and sealed evaluation files are unchanged.

## Changelog / development record

2026-09-12: seven small composition modules implement the same source tested on the separately delivered integrated student. Largest module146lines;601lines total. New native calculus tests include numeric/abstract-domain composition, source/sort restrictions, missing-law controls, contrary equations, immutable evidence IDs, budget exhaustion, checkpoint and rehashed proof rejection. Twenty generated full proofs use an independent integer oracle; sparse problems must remain unknown when unsupported.

The existing calculator already answers 2+2+2+2 with8. The new path demonstrates something different: two supplied equations (2+2=4,4+4=8) plus the reviewed associativity law produce8 without evaluating the target. It regroups, applies the first premise twice and the second once. Two applications of one premise are not two independent sources. Numeric lessons are checked by the existing calculator at teaching/binding time, not during target inference.

Ground symbolic equations can also compose ordered sequence values under an explicitly supplied associative join operation. No reversal/commutativity is inferred. Domain tags can reject incompatible types, but do not establish physical applicability, disjoint collections or unit conversion by themselves.

## Native versus integrated tests

This PR publishes all seven new source modules and the portable test file. It does not carry the accumulated memory/word-decoder branch. Default main does not contain that decoder. Integrated tests and the authored teaching script are in the complete local delivery at69f77ef68e50e76ef076b721ae4eae2a504a4261.

Local integrated result:48newtests pass;703relevantregressionpasses and2inheritedV8xfails. Eight authored changed-number cases improve0/8->8/8 after two basic premises per case. RealV8 and original word decoder run with target calculation and both legacy sentence generators disabled. Paired runtime/equation restore matches subsequent response, proof and token receipt. No full-suite/evaluator/browser result is claimed. Native CI runs the portable tests and original engine separately.

## Use / limits

Bind explicitly through composition.resolver.bind(runtime, ledger). Teach through provider.teach(...). `Derive ...` selects the premise-only path; `What is ...?` keeps the existing calculator. Default runtime checkpoints do not include this addon; use the explicit composition.persistence capture/restore pair under the host's authorized scope. It is NOT integrated into CompartmentStore's signed service format.

UNKNOWN is not false; INCOMPLETE is not exhaustive absence. Pure symbolic proofs are conditional on the accepted operation/lesson specification. Numeric ingress rejects incorrect arithmetic lessons. Hashes bind content, not authority or human identity. Ground compound-to-atom rewriting is not arbitrary theorem proving; operation laws are host supplied, not autonomously induced. An analogy/gray-link source is not an equation premise.

No automatic broad classroom language, ontology migration, unknown-operator induction, physical grounding or general analogy engine is added. Existing compiler claim guards were preserved; an initial unsupported adapter marker was corrected to the established trusted resolver contract rather than weakening the compiler.

Keep draft pending independent review, branch reconciliation and candidate-runtime evaluator binding. Do not rewrite frozen labels/checksums to force green CI. Source changes are additive; no completed-response templates or external-model calls.
