# Versioned conversation evaluation

This directory is the executable contract for GitHub issue
[#41](https://github.com/deucebucket/clanker/issues/41). Corpus version
`conversation-v1` is compiled as one canonical JSON object per complete
conversation. It measures the exact production code at
`c8c0bf4ccd5e73b1bd6bbe99762c87c4a549665e`, the final reviewed live-promotion
main for the post-#106 web lineage through PR #115.

## Truth and copyright boundaries

- `public_domain_drama` and `public_domain_novel` contain public-domain source
  dialogue. Their state, act, and outcome labels are frozen weak supervision;
  the following source turn was not caused by a Clanker response.
- `public_domain_real_human` contains only raw United States government
  proceedings whose public-use status and speaker boundaries have authoritative
  evidence. It is a real-human exchange, but still not causal Clanker exposure.
- `synthetic_adversarial` contains original project-authored CC0 conversations.
  Its semantic annotations are gold structural labels; its outcomes are
  counterfactual.
- `development` is open and mutable only through a new content hash. It is the
  sole source of the frozen correction lookup. It is never blended into held-out
  results.

No private chat, ACL transcript, fabricated consent, copyrighted non-public-
domain prose, or personally supplied conversation is present. Reports contain
IDs and aggregates, not source text or generated responses.

## Immutable boundary

`data/CURRENT` atomically selects one immutable directory under
`data/generations/<corpus-root>/`. That directory contains `heldout_v1.jsonl`,
`development_v1.jsonl`, `manifest_v1.json`, and `ROOT.sha256` as one complete
generation. `data/HISTORY.json` binds the byte digest and `100644` mode of
every generation ever selected on this branch. The `verify-history` gate walks
the candidate's `HISTORY` changes and authenticates each generation at the
commit where it first appears. A first appearance may import a batch of branch
generations, so GitHub's squash-merge policy does not erase their lineage; the
final ledger must retain the union of all first introductions and their
original entries. Deleting, changing, or reintroducing different bytes still
fails. The manifest records the full split policy and counts, every
whole-conversation digest, raw-source digests, compiler/evaluator digests, and
a digest of the exact production code and runtime-data bytes at the named
release commit. `ROOT.sha256` anchors all of those constituents. The compiler
and runner fail if local production bytes differ.
A metadata-only `lineage-inventory` command authenticates the selected
manifest through that history ledger, verifies every whole-conversation
content address, and returns only source/conversation/lineage IDs, hashes,
counts, and split-policy flags. It exposes no turns, annotations, participants,
or text. This is the complete lineage/policy handoff for #110.
A correction requires `conversation-v2`; changing v1 in
place is forbidden. The release tag for the merge commit is the external,
immutable anchor; a digest stored beside its payload is not sufficient alone.

The loader requires an explicit purpose. Held-out permits only `evaluation` and
fails closed for `training`, `teacher_replay`, or `promotion`. Evaluation uses a
fresh store per whole conversation. Transition correction reads an immutable
development-only store while held-out observations go to a separate ephemeral
store, so an earlier held-out turn cannot tune a later one.
Development is intentionally eligible for supervised teacher replay and
proposal construction, but neither split permits the `promotion` purpose.
Under #110, promotion consumes an approved, provenance-bound correction or
preference artifact; it never promotes a raw conversation split directly.

The `evaluation` package is excluded from built distributions. A repository
gate also parses every declared production source and bounded executable asset
literal expressions—including conditional expressions, simple Boolean
choices, literal dictionary selection, and named expressions—rejecting
statically resolvable references to the
evaluation module, corpus paths, `CURRENT`, or a held-out loader call. Dynamic
Python import APIs are rejected outright in declared production sources. This
static gate does not claim to decide arbitrary Python or JavaScript obfuscation,
and it cannot govern arbitrary text or paths deliberately supplied by an
operator through generic production commands. Official teacher replay and
promotion must consume the content-addressed split policy and reject held-out
material; that integration boundary is owned by issue #110.

For original synthetic sources, `raw_source_sha256` records the project
authors' source attestation; there is no independent upstream download to
re-hash. It must not be described as externally verified. The compiler instead
reproducibly binds the exact repository source-document bytes and their Git
history in `source_document_sha256` and the constituent corpus root. The CC0
grant is explicit, but authorship and the raw-source attestation remain a
project claim rather than independent human-ground-truth provenance. In short,
the raw-source attestation remains a project claim.

## Reproduce

```bash
python -m evaluation.conversations compile
python -m evaluation.conversations verify
python -m evaluation.conversations verify-history --ref HEAD
python -m evaluation.conversations lineage-inventory
python -m evaluation.conversations run --split development
python -m evaluation.conversations run --split heldout \
  --output evaluation/conversations/baselines/post_113_heldout_v1.json \
  --failures evaluation/conversations/baselines/post_113_heldout_v1_failures.jsonl
```

The three modes are:

- `sentence_only`: new runtime for every turn; no memory or correction.
- `stateful`: new runtime per conversation; state and semantic memory retained;
  correction disabled.
- `transition_corrected`: same lifetime, with production correction math reading
  only the frozen development-derived statistics.

The runner reports mode/domain/outcome/supervision strata, per-axis next-state
MAE and direction accuracy, dialogue/response acts, semantic/status/truth and
entity resolution, UNKNOWN/CONFLICT precision and recall, calibration, target
attainment, drift, latency, candidate growth, memory growth, and SQLite growth.
Accuracy intervals use 10,000 fixed-seed whole-conversation cluster bootstrap
draws, stratified by domain for the overall result. Classification precision,
recall, and F1 intervals use the same fixed-seed, domain-stratified
whole-conversation cluster bootstrap. Latency/resource observations are
excluded from the deterministic semantic fingerprint; process max-RSS is an
observational process-lifetime peak and is not a paired mode comparison.
Every deterministic metric carries ID-only per-conversation/per-turn
sufficient statistics. Validation rebuilds point estimates and fixed-seed
cluster intervals from those observations, binds UNKNOWN/CONFLICT predictions
to gold statuses, and derives the failure ledger from the exact zero-valued
turn metrics. Latency, construction latency, memory growth, SQLite allocation,
row growth, maxima, counts, and slopes likewise carry bounded ID-only
observations; validation rebuilds their aggregates and binds every domain and
turn identity to the evaluated corpus. Drift rows are bound to the identical
next-state/MAE population, per-axis absolute residuals, normalized MAE, and the
weighted-RMS distance equation. No source utterance or generated response is
included.

`evaluation_commit` is not accepted merely because it is an ancestor. It must
equal the newest commit on the current ancestry that changes the measured
production/evaluator/corpus/schema paths. This identifies the executable
snapshot that could have produced the report while allowing a later,
artifact-only commit. Baseline publication is therefore a separate commit (or
PR) after the executable core is integrated. Report output cannot use a
`.jsonl` suffix, which is reserved for the failure-ledger role; ambiguous role
names fail before any directory, generation, symlink, or pointer is mutated.

VADUGWI labels are produced only by the frozen weak rule. Sources marked
`structural_only` are scored for their gold semantic/entity structure but are
excluded from affect and trajectory metrics, so those heuristic axes are never
reported as `gold_structural`. For the remaining affect rows, direction and
target-distance improvement use the declared whole-interaction estimand:
`g0 = vadugwi_before` (before the current utterance) and
`g1 = observed_next_state` (after the following response/reaction). This is not
a response-only `vadugwi_after -> observed_next_state` estimand.

Reports and ID-only failure ledgers are validated against every held-out turn.
The runner writes each report, ledger, and checksum as one immutable generation
and exposes the three conventional paths through fixed role symlinks beneath a
single atomically switched `.current` directory symlink. File and parent-directory
Final `100644` modes are applied before each file `fsync`; file and
parent-directory `fsync` calls make the generation durable before selection, so an exception or
abrupt process death cannot select a mixed generation. `load_published_artifacts`
hashes and parses the same buffers, then validates the aggregate schema, corpus
provenance, and exact producing-executable commit by default. The runner
captures that measured commit and all compiler/evaluator/production digests at
startup and fails if they or the
measured worktree change during the run.

The first baseline sets no accuracy threshold: it records the honest result at
the exact manifest-bound release commit. Later thresholds must name both the
corpus root and baseline fingerprint.
