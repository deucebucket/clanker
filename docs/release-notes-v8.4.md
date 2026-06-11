# Clanker V8.4 — Release Notes

2026-06-11/12. W/I channels, vocabulary neutralizer, tier-1 passive
aggression. All numbers measured fresh against this release.

## What Changed

- **Crisis structure wiring.** SUSPICIOUS_CALM, MASKING, RESIGNATION were
  detected but discarded by crisis scoring; now wired into STRUCTURE_WEIGHTS.
- **MUNDANE_HYPERBOLE narrowed.** Out-of-vocabulary tokens alone no longer
  count as mundane context; a mundane and a crisis reading can no longer both
  win. Ghost-possession grief added: possessive of an absent person +
  retention verb ("i still have his number saved" V=95).
- **Vocabulary neutralizer.** 19 words whose LLM-rated fixed charge
  contradicted sentence-level evidence demoted to a NEUTRALIZED_LIQUID tier,
  dV scaled by (1 − contradiction rate). Slang re-rated with two-faced
  physics: "bruh he cooked" V=165 vs "im cooked" V=67.
- **W — attribution-routed self-worth.** Valence reaches W through a
  force-flow attribution coefficient (self-declarative / self-targeted /
  guilt-partial / atmospheric-zero) instead of a binary SELF_REF-proximity
  gate. Crisis-set W median 88 vs 128 on the safe set.
- **I — agency axis.** Futility phrases ("whats the point") sink I to 64;
  volition phrases ("im going to fix this") lift I to 168, from the neutral
  band only — strong directional reads are never overridden.
- **Tier-1 passive aggression.** Four detectors for grievance smuggled into
  deniable grammar: TEMPORAL_GRIEVANCE, EXCLUSION_CONTRAST, IRONIC_DEFERENCE,
  FAINT_PRAISE — plus RARITY_MARKER disambiguated by actor ("i finally got my
  license" V=209 stays joy; "nice of you to finally answer" V=54).
- **RETROSPECTIVE_HOPE.** Hope aimed at a closed outcome the addressee
  already knows reads as accusation: "hope it was worth it" V=91, I=168.
  Future tense, benefit frames, and self-direction rescue to baseline.
- **Benchmark harness restored + holdout protocol.** stress_test.py and
  crisis_benchmark.py restored; `benchmarks/holdout_probes.json` established
  as an evaluation-only set under `benchmarks/HOLDOUT_PROTOCOL.md`.
- **Soul-side integration.** clanker-soul now consumes the real engine; the
  Governor heartbeat passes end to end (masked-collapse arc drives
  UNRESTRICTED → READ_ONLY).

## Measured Numbers (2026-06-11)

| Metric | Value |
|--------|-------|
| Tests | 286 passed (+2 xfail) |
| Crisis recall | 49/51 (96.1%) |
| Crisis false positives | 0/75 (0.0%) |
| Stress test (275 sentences, 11 categories) | 271/275 (98.5%) |
| Ground truth | 40/41 (97.6%) |
| Held-out probes (never tuned on) | 30/45 (66.7%) |
| — slang positive | 6/15 (40%) |
| — grief | 11/15 (73%) |
| — passive aggressive | 13/15 (87%) |
| SST-2 validation (872 sentences, default constants) | 542/872 (62.2%) |
| Vocabulary | 4,545 words |
| Structural patterns | 66 |
| Latency (conversational, single core) | 0.45 ms/sentence (~2,200/sec) |
| Corpus throughput (Twitch/novels/philosophy) | 998–2,445 sentences/sec |

## Honest Evaluation Protocol

The in-repo suites (stress test, ground truth, crisis benchmark) have been
tuned against repeatedly; they measure regression, not generalization, and
they overestimate accuracy. The held-out probes are sentences the engine has
never been tuned on, governed by `benchmarks/HOLDOUT_PROTOCOL.md`: run only
to measure, never to fix; when a holdout category needs work, source new
training sentences and leave the holdout set untouched. Both numbers are
reported side by side — 98.5% in-repo, 66.7% held-out — and the gap is the
honest one.

## V9 Status

The V9 equation-decomposition rewrite is abandoned (2026-06). Lessons:
order-independence loses information; lemma roots lose nuance. The bond-site
formalism and nucleus check remain backport candidates.
