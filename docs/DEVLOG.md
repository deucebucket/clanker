# Clanker Devlog

Play-by-play of engine development. Chronological; every session appends.
Earlier history (pre-devlog) is reconstructable from CHANGELOG.md, git log,
and docs/v8_audit_log.md.

---

## 2026-06-11 — Slang register overhaul + the council-v2 data pipeline

### Where the day started

V8.4 had just shipped (W/I channels, neutralizer, tier-1 PA detectors,
RETROSPECTIVE_HOPE, farewell double-fire fix). Canonical holdout stood at
30/45 (66.7%): slang_positive 6/15 (40%), grief 11/15 (73%), PA 13/15 (87%).
Slang was the worst measured number in the project, so it became the target.

### Step 1 — Baseline + reconnaissance (parallel agents)

- Confirmed baseline: slang_positive 6/15, total 30/45.
- An Explore agent mapped the slang path through the engine: SOLVENT
  dissolution in phase.py (zero-charge register markers flip negative LIQUID
  atoms), REGISTER_CASUAL role, danger-class words (sick/fire/cracked) that
  store one sense and rely on a nearby SOLVENT to flip.
- A second agent built `datasets/slang_training_candidates.json`: 80 fresh
  sentences from the Twitch/Reddit corpora + synthetics, labeled with
  direction bands, mechanically verified ZERO overlap with the holdout.
  The holdout file itself was never opened — HOLDOUT_PROTOCOL discipline.
- Agent fact-check caught two wrong claims before they cost anything:
  af/mid/W/cap were NOT missing from vocabulary, and the 1,671 duplicate
  dict keys in forces_curated.py are deterministic (last wins) — the
  append-to-override house idiom, not a bug.

### Step 2 — Engine probe on the fresh pool

52/80 (65%). Failures clustered into: hype/exclamatory under-read
(LETS GOOOOOO at dead 128), missing vocab (dope, dogwater, masterclass),
literal-sense overcorrection ("theres a fire in my kitchen" V=146 POSITIVE,
"i cooked dinner for the fam tonight" V=181), and mysteries ("you the goat
fr" V=97, "This shit slaps" V=33, "ordered it w extra cheese" V=51).

### Step 3 — Root-cause traces (diagnosis agent)

Ten distinct mechanisms, the big ones:
- Single stored sense per word: 'trap' (-77 entrapment) nuking "trap remix",
  'extra' (-52 drama) nuking "extra cheese", 'w' (+30 win) misfiring on
  "w extra cheese", 'fire' praise-sense making kitchen fires positive.
- Stemmer false positive: 'number' -> 'numb' injected grief forces into
  "she peaked at number three on the charts".
- Duplicate key killed 'clean': a late (0,-5,0,0,15) entry silently
  overwrote the positive praise grades.
- Elongation/caps carry zero signal; 'goooooo' fell out of vocabulary
  entirely ('go' isn't an entry).
- Discourse "no" ("no i'm sick with a sore throat") treated as logical
  negation -> proximity sign-flipped the illness words to V=149 positive.
- DIRECTED_POSITIVE (tier-1 PA from the same morning) punishing short
  second-person compliments: "you the goat fr" -> V=97.
- apply_w_coefficient amplifies below-center V by 1.5x even at NEUTRAL W,
  contradicting its own docstring. Flagged as "the highest-leverage fix".

### Step 4 — Batch 1 (vocab + token level, TDD, agent-implemented)

12 net-new slang entries (dope, rad, gnarly, elite, iconic, unreal, legend,
masterclass, ggs, dogwater, frfr, letsgo), regrades (clean, extra, trap),
'w' win-vs-with disambiguation, "lets go" compound + elongation collapse,
"as hell" amplifier compound, expletive rescue extended past index 0,
STEM_BLOCKLIST for 'number'. Agent also fixed "I am so happy for you"
(V=106 -> 174, DIRECTED_POSITIVE copula-chain guard) and added an
above-center-only inert-zero skip so trailing function words stop draining
positive V. Verified independently: pool 65% -> 79%, all 302 tests green,
stress 271/275, crisis FP 0/75.

### Step 5 — Batch 2 (sense guards, agent-implemented)

Discourse-no rescue (sentence-initial no/nah/nope + SELF_REF = answer
marker), SOLVENT literal-verb guard ('cooked dinner' no longer flips,
'fam' regraded from +40 to +12 to honor the zero-charge SOLVENT contract),
fire hazard/praise sense split ("theres a fire in my kitchen" V=108 U=58),
DIRECTED_POSITIVE praise-noun guard ("you the goat fr" V=182), "so back"
compound (WE ARE SO BACK V=205), degree-adverb slang ("stupid good" V=175),
filthy sports split, mid regrade + temporal split, obsessed -> LIQUID.
Verified: pool 77/80 (96%), 333 tests, benchmarks identical.

### Step 6 — The W-coupling experiment (failed, instructive)

Tested the docstring-faithful formula (no amplification at neutral W):
stress 271 -> 263, ground truth 40 -> 39, crisis recall 49 -> 46, pool
77 -> 75, five tests broke. Conclusion: the 1.5x neutral-W negative
amplification is a CALIBRATED negativity bias the whole suite is fitted
around, not a bug. Reverted; docstring corrected to say so, so nobody
"fixes" it again.

### Step 7 — Holdout measurement: the honest result

After all of the above: holdout 28/45 (62.2%) — slang_positive 5/15 (DOWN
one), grief 10/15 (DOWN one), PA 13/15 (held). The fixes are real on the
register we trained on (independent 80-sentence pool 65% -> 96%) and did
not transfer to the holdout's slang phrasings; something — suspects are
the discourse-no rescue or obsessed-as-LIQUID — cost one grief probe.
Recorded per protocol, tuned nothing against it. Lesson logged in README:
the register gap is wider than any one vocabulary pass. PA proves transfer
is possible (8/15 -> 13/15 across yesterday's structural work); slang needs
its own structural family, not more lexicon.

Shipped as commit 237830c (V8.5), docs updated with the honest numbers.

### Step 8 — Scale the data (the n=45 problem)

Decision: with the engine at 2,000+ sentences/sec, evaluation scale was
never compute-bound — it is LABEL-bound. At n=45 every probe is 2.2 points;
real deltas drown in noise. Built the council-v2 pipeline (tools/):

- `council_sample.py` — 2,144 candidates assembled: slang_casual 700 +
  mixed_safe 700 sampled from Twitch/Reddit (emote/url/"ago " cleaning,
  3-18 words), grief 350 + passive_aggressive 394 generated via DeepSeek
  across 11 prompt angles INCLUDING hard negatives (warm-memory grief
  lookalikes, sincere must-be-nice PA lookalikes). Deduped against holdout
  (mechanical), ground truth, stress test, slang pool, and every string
  literal in engine/tests/ (AST walk). 48 dupes dropped.
- `council_grade.py` — 2-model council: deepseek-chat (A) + deepseek-
  reasoner (B), batches of 20, strict JSON, crash-safe incremental writes,
  --resume. Smoke test: 20/20 inter-grader label agreement. A finished
  2,144/2,144 with zero failed batches at ~4s/batch; B ~15s/batch.
- `council_assemble.py` — agreement filter (disagreements ->
  council_v2_ambiguous.json), seeded (8) stratified 80/20 split into an
  OPEN train pool and a SEALED benchmarks/holdout_v2_probes.json that
  no human eyeballs, plus eval_holdout_v2.py and a protocol section.

Local llama.cpp server was down all session; DeepSeek via `ds` CLI carried
all generation and grading. Anthropic tokens spent only where tool-use
agents were genuinely needed.

### Step 9 — Results

Grading: A finished 2,144/2,144 (zero failed batches, ~4s/batch); B
finished 2,144/2,144 (zero failed batches, 16.5s/batch avg). Inter-grader
agreement 87.0% overall (grief 93.4%, PA 94.9%, mixed_safe 84.0%,
slang_casual 82.3%); 279 disagreements quarantined to
council_v2_ambiguous.json — useful later as a low-confidence test set.

Split (seed=8, stratified): train 1,492 open / holdout-v2 373 sealed.
Mechanical integrity verified: zero overlap train∩v2, v2∩v1, train∩v1.

**Holdout v2 baseline (V8.5): 154/373 (41.3%)** — grief 30/65 (46%),
mixed_safe 48/118 (41%), passive_aggressive 23/75 (31%), slang_casual
53/115 (46%). Holdout v1 unchanged at 28/45 (62.2%); all 333 tests green;
in-repo suites unchanged.

The v1-vs-v2 gap (62% vs 41%) is the small-n flattery of 45 hand-picked
probes meeting 373 council-graded ones. v2 is the honest yardstick now:
at n=373 each probe is 0.27%, so a 1-point delta is ~4 sentences of real
signal.

Open-pool diagnosis (1,492 sentences, legal to inspect): engine total
48.9%. Confusion matrix: 590 of 763 errors (77%) are boundary cases —
council commits to pos/neg where the engine sits in the neutral band
(neg->neutral 333 is the single biggest cell). Only 173 true inversions.
THE finding of the day: the engine under-commits; it rarely inverts.
Concrete new misses surfaced: "L" as loss ("such an L hacker"), "LOCKED
IN FR" (locked reads as trapped), W-prefix nouns ("w stream goodnight"),
PA temporal grievance not firing on "only took you three hours to reply".
Known data caveats: some emote spam survived cleaning ("EleGiggle" x6
labeled pos), and bittersweet grief sentences carry genuine label
ambiguity even after agreement filtering.

Next session's campaign, in priority order: (1) the under-commitment
pattern — why accumulated mild negative signal decays back into the
neutral band (suspects: per-word center-relaxation drain, static
friction threshold, the neutral band width itself vs council calibration);
(2) the new slang idiom classes; (3) PA structure recall on generated
phrasings the tier-1 detectors miss. All tuned against the OPEN pool
only, measured on v2.

Docs synced with measured numbers: README, CHANGELOG, CLAUDE.md,
CONTRIBUTING, HOLDOUT_PROTOCOL, this devlog.

---

## 2026-06-15 — Pet showroom QA surfaces 4 engine findings (flywheel working)

End-to-end QA of the clanker-pet-space M4 build (toys + physics + animation,
shipped live on deucebucket/clanker) doubled as a real-world engine audit: the
pet IS an eval harness with a face. Ran the four canonical demo sentences
through the engine and judged the reads for correctness — several came back
wrong, which is the point. Logged to docs/v8_audit_log.md (findings #7-#10) and
added 8 probes (the 4 sentences + diagnostic minimal pairs) UNLABELED to
datasets/council_v2_candidates.json (source pet:qa_audit) for council grading —
no truth labels invented.

Findings, sharpest first:
- ALL-CAPS / "!!!" invisible to Arousal: "THIS IS THE BEST DAY EVER" reads
  A=138, identical to lowercase. Orthographic intensity not modeled.
- "content/calm" sentence collapses to negative Valence: "content"=V218 alone
  but "i'm calm and content, everything is quietly fine"=V91. Compositional
  over-damping eats a SOLID positive atom; also contradicts a showcase claim.
- Stacked insults under-read in magnitude (V86/W78 for triple insult; aggression
  lowers A instead of raising it).
- Warm gratitude over-reads Gravity (G185 for light thanks).

Next: trace the "content" collapse stage; add caps/exclamation Arousal
amplification; all tuned against the OPEN pool only after council grades land.
