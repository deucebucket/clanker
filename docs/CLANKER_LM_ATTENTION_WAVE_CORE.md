# Attention wave core — issue #168

Stacked on #160's task-gated activation. The original Clanker engine and
clanker-soul are separate and unchanged. This native subset does not contain
the separately delivered MemoryWeb/word-decoder adapter or the new Atlas.
It must not be presented as automatic new default-chat behavior.

## Implemented primitive

ActivationIndex exposes its existing request and edge gates publicly while
retaining _allowed as a compatibility alias. The existing BFS select path
uses the same gates and retains its behavior. WaveIndex uses that same
metadata index for a bounded best-first traversal, not another memory store.

Eligible path salience uses integer arithmetic:

    next = current * edge_strength * decay // (1024 * 1024)

Default decay is 896; edge strengths are bounded 0..1024. Policy is checked
BEFORE the score. Same-plane associations may expand. Personal gray context
references expose a reference-only node but cannot expand its semantic
neighbors. Semantic explanation never reverse-looks up personal references.
Neither salience nor graph similarity is a probability, factual premise or
account credential.

The strongest path wins; repeated routes do not add support. Non-dominated
depth/salience paths remain eligible because a weaker shallower path may reach
an otherwise depth-blocked child. Stable heap ordering supports exact replay.
Existing Request caps bound depth, admitted nodes and inspected edges. Budget
exhaustion is explicit. A successful relevance selection is still NOT an
exhaustive truth search. Candidate pings read metadata; original payloads
remain the caller's responsibility. The immutable index must already represent
an authorized compartment; its context string does not authenticate a caller.

Receipts record actual seeds, queue admissions, expansions, reference-only
stops, rejected edges and budget/salience/dominance pruning. verify recomputes
the traversal and rejects rehashed edits or changed generations/gains. These
core traces contain permitted graph IDs; a public UI must still project/redact
records according to its own scope policy, as the separate adapter does.

## Tests

New core tests exercise all three task policies, invalid configuration,
policy-before-root-lookup, score-before-node-budget ordering, cyclic and repeated
routes, shallow/strong path tradeoffs, deterministic ordering, explicit stops,
and receipt mutation. Forty small generated graphs are checked against an
independent depth-based dynamic-program calculation. These are authored
engineering tests, not research on human recall or general understanding.

Run:

    python -m pytest tests/test_attention_wave_core.py tests/test_activation_core.py engine/tests -q

## Separate integrated experiment

The same core was tested against the accumulated integrated checkpoint
ab2ef855... via modular attention/cues.py, bridge.py, answers.py and atlas.py.
Those files and their teaching examples are NOT part of this native subset.
The adapter snapshots an existing authorized graph, resolves only bounded
metadata cues, and offers an opt-in read-only simple-event answer preview.
That preview refuses an answer when relevance filtering omits a matching or
opposing event; selected records alone do not license confident absence.
Its conservative coverage pass still examines the permitted event store.

The integrated experiment has 43 new core/adapter/view tests passing and a
relevant regression selection of 570 passes plus 2 inherited V8 expected
failures. Three authored conversations each exercise three task views, one
actual evidence-bound reply, reload, and no-write checks: 18/18 checks pass.
A synthetic fixture contains 1,000 event metadata records (not 1,000 acquired
human experiences): 1,004 nodes and 2,002 edges. The configured traversal
pings 1,140 adjacency entries, selects 47 active records plus one reference,
and explicitly reports incomplete because of its 48-node budget. Cold full-
graph snapshot/index construction, warm metadata selection and verified
projection are measured separately; do not label warm time full-chat latency.

The existing atomic word decoder supplies the inspected small-graph replies.
It does not use the stress fixture to claim broad factual accuracy. The Atlas
replays recorded steps at a slower presentation speed; its layout is aesthetic,
not neural sensing or a simulated claim of consciousness.

## Changelog / development notes

2026-09-12: expose shared gates, add fixed-point wave traversal and typed
configuration, and add independent-oracle/negative-control tests. The initial
integration cue index incorrectly interpreted an assistant output caption 'I'
as an input identity. It now indexes canonical entity labels; the regression
is explicit. No general identity recognizer or universal intent parser is
claimed. Edge gains are host-supplied, not autonomously learned this round.

The integrated source has no new monolith: the source modules remain well
below the existing LM architecture ceilings. A larger local functional run
timed out; it is not recorded as passing. The full release/evaluator/browser
results must remain separate from these focused tests.

## Release boundary

Keep draft pending parent integration, exact-head review and the unchanged
candidate-runtime evaluator binding. No engine/ or evaluation/ bytes changed.
No external corpus, stored reply template, neural dependency, background daemon,
or device action is introduced. Current native main and the accumulated
integrated delivery remain different code lines. Follow #153/#155/#159/#164 for
sense identity, broader context routing, display and persistent agenda work.
