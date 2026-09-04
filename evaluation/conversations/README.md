# Versioned conversation evaluation

This directory is the executable contract for GitHub issue
[#41](https://github.com/deucebucket/clanker/issues/41). Corpus version
`conversation-v1` is compiled as one canonical JSON object per complete
conversation. It measures the exact production code at
`9ae77f072f8afda0b1d2b757ab492757cabff0f8`, the merge result of #106.

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

`data/heldout_v1.jsonl` and its labels are frozen. `manifest_v1.json` records
the full split policy and counts, every whole-conversation digest, raw-source
digests, compiler/evaluator digests, and a digest of the exact production
module bytes at the named post-#106 commit. `ROOT.sha256` anchors all of those
constituents. The compiler and runner fail if local production bytes differ.
A correction requires `conversation-v2`; changing v1 in
place is forbidden. The release tag for the merge commit is the external,
immutable anchor; a digest stored beside its payload is not sufficient alone.

The loader requires an explicit purpose. Held-out permits only `evaluation` and
fails closed for `training`, `teacher_replay`, or `promotion`. Evaluation uses a
fresh store per whole conversation. Transition correction reads an immutable
development-only store while held-out observations go to a separate ephemeral
store, so an earlier held-out turn cannot tune a later one.

## Reproduce

```bash
python -m evaluation.conversations compile
python -m evaluation.conversations verify
python -m evaluation.conversations run --split development
python -m evaluation.conversations run --split heldout \
  --output evaluation/conversations/baselines/post_106_heldout_v1.json \
  --failures evaluation/conversations/baselines/post_106_heldout_v1_failures.jsonl
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

VADUGWI labels are produced only by the frozen weak rule. Sources marked
`structural_only` are scored for their gold semantic/entity structure but are
excluded from affect and trajectory metrics, so those heuristic axes are never
reported as `gold_structural`. For the remaining affect rows, direction and
target-distance improvement use the declared whole-interaction estimand:
`g0 = vadugwi_before` (before the current utterance) and
`g1 = observed_next_state` (after the following response/reaction). This is not
a response-only `vadugwi_after -> observed_next_state` estimand.

Reports and ID-only failure ledgers are validated against every held-out turn
and staged with their SHA-256 sidecar before publication. The runner captures
HEAD and all compiler/evaluator/production digests at startup and fails if they
or the measured worktree change during the run.

The first baseline sets no accuracy threshold: it records the honest post-#106
result. Later thresholds must name both the corpus root and baseline fingerprint.
