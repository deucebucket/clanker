# Executed lexical-affinity / VADUGWI decoding experiment

**Opt-in research path for #120, supporting #109/#121. Not the default
production chatbot, a neural model, or a promoted global language pack.**

The implementation runs the real `ClankerLM.process()` pipeline. Its realizer
seam returns a typed, unlexicalized grammar frontier. After Clanker computes
its target state, the ranking seam expands that frontier one word at a time.
It does not generate a legacy response and then manufacture a token trace.
`SurfaceRealizer.realize()` and `render_event()` are not used. The existing
noun-reference renderer and morphology functions remain reused.

## Run from a repository checkout

No additional runtime package, network request, model download, or GPU is
needed. The main package's V8 backend is used by default.

```bash
python -m experiments.lexical_decoder
python -m experiments.lexical_decoder --once 'I am disappointed.' --trace
python -m experiments.lexical_decoder --snapshot ./my-private-session.json
python -m experiments.lexical_decoder --once 'Hello.' --receipt ./private-receipt.json
python -m pytest tests/test_lexical_affinity_decoder.py -q
```

`/quit` exits; `/receipt` displays the last actual receipt. Snapshot and receipt
files are created only when explicitly requested and use atomic replacement.
They can contain resolved user words and lexical evidence. They are NOT public
telemetry, training corpora, or anonymized artifacts. Do not publish private
sessions. No browser/network route is changed by this experiment.

## What executes

1. The existing affect analysis, learner, parser, memory, Q&A, resolver,
   response-policy and contextual gates run.
2. `Grammar.plan` translates the supported contract/response act into logical
   clauses or interjections. Generic clause operations execute tense,
   subject agreement, do-support, negation, interrogative ordering, role markers
   and punctuation. Each resulting slot retains eligible lexical alternatives.
3. Slot constraints remove wrong semantic senses, disallowed registers, locked
   pools, and pools not enabled by a nonempty pool allow-list. A lock wins over
   an allow. Bound reference and grammatical terminals are not lexical pools.
4. At each prefix, SQLite supplies observed count, context total, add-half
   smoothing, fallback context, and probability. An unknown token receives
   unknown-class mass, not an invented observed count.
5. Each extension is evaluated with a deterministic legal completion estimate
   through the real V8 transition. Candidate reads also apply the learned lexical
   overlay. This never adds evidence or teaches from generated output.
6. Integerized lexical and affect costs choose a bounded beam. Final tokens
   must match every executed slot's approved ID/surface pair. A final response
   and its receipt are committed together; resource exhaustion emits no partial
   response and the wrapper restores pre-turn symbolic AND SQLite state.

There are no finished response templates in the new decoder. Logical plans
still require reviewed meanings and grammar rules. This is a bounded
single-response word lattice, not universal English generation, full inverse
seven-axis solving, or the multi-turn simulation planner from #111.

## Score and receipt

```
lexical_prefix_cost = sum(round(-log(smoothed_probability) * 1_000_000))
affect_cost = round(weighted_VADUGWI_distance(predicted, target) / 255 * 1_000_000)
total = lexical_weight * lexical_prefix_cost + affect_weight * affect_cost
```

The prefix affect estimate fills the remaining slots with the first eligible
lexeme in deterministic ID order. It is explicitly labelled an estimate. It is
NOT an admissible reachability bound and never hard-rejects a word for emotional
unreachability. The last step evaluates the actual complete candidate.

Receipts bind the software tree, count pack and source attestation, grammar,
contract, relevant symbolic/lexical context, gates, observed/target state,
integer weights and resource policy. They expose:

- the options actually considered for each slot and each hard exclusion;
- every expanded parent/child, word ID, observed count and denominator,
  smoothing/backoff, raw probability and probability among eligible options;
- the suffix used for lookahead, projected and predicted VADUGWI vectors,
  signed axis errors, score components and total;
- semantic rejection separately from beam-budget pruning;
- the selected path, complete competitors, winner margin and deterministic tie
  break; and an exact receipt content digest.

`replay_last()` reruns the decoder from the captured pre-selection context.
`LexicalDecoder.replay()` compares the complete reconstructed receipt, not just
its self-reported digest. A hash is an integrity/replay mechanism, NOT a signed
human approval, proof of factual ground truth, or proof of psychological effect.

The output guard verifies the executed typed derivation. It is not a general
English equivalence verifier and does not close #66. Existing parsing and
knowledge-quality limits still apply.

## Initial conversational coverage

Supported: greetings, a bounded gratitude/closure parser extension, neutral
acknowledgment, negative/positive appraisal, empathetic follow-up, a safety-check
act, lexical probes/learned definitions, explicit uncertainty/conflict disclosure,
reference clarification, and simple closed factual event clauses. Live clock,
date and arithmetic observations use the same grammar. Learned definitions
queried as inferred hypotheses receive a modal qualification.

Complex attribution, complement relations, unsupported roles, open event
references, unsupported tense/aspect and missing prepositions decline explicitly
into a compositionally generated coverage probe. The returned contract is marked
`UNSUPPORTED`, retains the original status in diagnostics, and the receipt says
`coverage: declined:...`. Such a turn is not counted as successfully answering
its original question. There is no silent fallback to the legacy realizer.

The first reference-probe grammar is deliberately generic, rather than pretending
to have solved the full clarification-resumption program. Broader discourse,
identity/persona and correction semantics remain #37/#107/#108 work.

## Counts and licensing

`fixtures/development.txt` contains 24 original project-authored development
sentences, licensed under the repository's MIT license. `development_counts.json`
contains only their aggregate unigram/bigram/trigram counts and manifest. The
runtime loads the compiled counts, NEVER the text fixture. A test recompiles the
fixture and verifies byte-equivalent records. A source attestation is not
independent provenance verification.

The fixture is intentionally small and is ONLY a wiring demonstration, not a
trained conversational language model or evidence of generalization. #121 owns
licensed larger corpora, pruning and deployment packs; #110 owns promotion.
The development compiler accepts only its declared development purpose and
MIT/CC0 attestation. It does not claim to detect arbitrary private or held-out
text falsely relabelled by a caller. Do not feed such data to the developer API.
No automatic learning from generated output, online downloads, implicit pack
updates or production activation occurs.

## Performance and delivery

```bash
cpu=$(python -S -c 'import os; print(min(os.sched_getaffinity(0)))')
(ulimit -v 65536; taskset -c "$cpu" python -S -m experiments.lexical_decoder.benchmark)
```

On Linux this pins execution to one allowed logical CPU and caps virtual address
space at 64 MiB. `-S` excludes unrelated site startup hooks and proves that the
core path requires only the standard library and repository source. This is NOT
a 64-MiB container-total/RAM guarantee and does not include a browser/web server.
The benchmark reports initialization after import separately from construction,
first validated chunk, peak RSS, receipt size and token throughput. The workload
is a six-turn original development conversation repeated in 20 fresh sessions.
Long sessions, concurrent users and real network latency are separate #122 work.

Chunks are yielded immediately AFTER the complete response is validated. There
are no typing sleeps. This is low-latency committed delivery, not a claim that
unverified tokens are safely streamed from an unfinished beam. Time to first
validated chunk therefore includes full construction.

## Why it is outside the default package

The repository's `conversation-v1` binds exact production module bytes. Editing
that frozen data or suppressing integrity checks to accept a changed engine would
invalidate its baseline. This experiment lives outside those shipped production
paths, leaves them byte-identical, and is excluded by the explicit setuptools
package list. The full existing evaluation gates still run unchanged. It is
connected to the real runtime through two narrow injection seams, not a mock.

#123 tracks default hard-gate correction and the independently reviewed
candidate-runtime evaluation protocol. #109/#110/#111 dependencies must be met
before public/default promotion. No held-out corpus bytes, histories, compiler,
runner, schemas or labels are edited here.

## Known inherited limitations

The older lexical definition-negation and pending-probe routing issues (#40),
legacy surface validation (#66), and unconnected V8 anomaly/concern/inverse
components are not fixed by this decoder. Its wrapper fixes its own failed-turn
rollback and resumed post-response prior, but does not claim the underlying
unwrapped default has those fixes. No universal safety, truth, learning-quality,
or clinical-performance claim is made.
