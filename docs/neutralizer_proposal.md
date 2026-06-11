# Neutralizer Proposal — data-driven de-biasing of EMOTIONAL_VOCABULARY

**Source:** `scripts/bias_audit.py` → `datasets/bias_audit.json` (8,484 labeled sentences:
3,645 human-verified + 4,839 Gemini; baseline pos=12% / neg=29% / neutral=59%).
For each word with |dV| ≥ 12: contradiction_rate (fraction of containing sentences whose
label sign opposes dV), neutral_rate, cross-source sign disagreement (current vs V1 46k,
WordNet-computed, anchors), and phase.

## Top-30 bias suspects (n ≥ 5 sentences)

| word | dV | phase | n | contra | neutral | score | remedy |
|---|---|---|---|---|---|---|---|
| adore | +48 | GAS | 17 | 1.00 | 0.00 | 1.42 | LIQUID |
| glorious | +45 | GAS | 9 | 1.00 | 0.00 | 1.42 | LIQUID |
| cherish | +37 | GAS | 16 | 1.00 | 0.00 | 1.31 | LIQUID |
| beat | -51 | GAS | 8 | 0.75 | 0.12 | 1.25 | LIQUID |
| congratulations | +49 | GAS | 10 | 0.90 | 0.10 | 1.22 | LIQUID |
| overwhelmed | -35 | GAS | 11 | 0.82 | 0.09 | 1.22 | LIQUID |
| fantastic | +30 | GAS | 27 | 1.00 | 0.00 | 1.07 | LIQUID |
| magical | +39 | GAS | 5 | 0.80 | 0.00 | 1.00 | LIQUID |
| survive | +28 | GAS | 11 | 1.00 | 0.00 | 0.99 | LIQUID |
| thrilled | +30 | GAS | 14 | 0.93 | 0.00 | 0.96 | LIQUID |
| thank | +50 | GAS | 73 | 0.77 | 0.15 | 0.96 | LIQUID |
| wonderful | +35 | GAS | 26 | 0.81 | 0.12 | 0.91 | LIQUID |
| crowded | +42 | GAS | 7 | 0.71 | 0.14 | 0.85 | LIQUID |
| breaking | -77 | GAS | 13 | 0.54 | 0.15 | 0.83 | LIQUID |
| secret | -24 | GAS | 17 | 0.77 | 0.06 | 0.77 | LIQUID |
| swallow | +50 | GAS | 6 | 0.67 | 0.33 | 0.75 | LIQUID |
| monster | -52 | GAS | 10 | 0.50 | 0.30 | 0.75 | LIQUID |
| response | +39 | GAS | 11 | 0.55 | 0.46 | 0.74 | LIQUID |
| truly | +28 | GAS | 70 | 0.81 | 0.03 | 0.73 | LIQUID |
| engagement | +20 | GAS | 8 | 0.88 | 0.12 | 0.71 | LIQUID |
| delighted | +49 | GAS | 14 | 0.64 | 0.29 | 0.71 | LIQUID |
| precious | +28 | GAS | 9 | 0.78 | 0.00 | 0.68 | LIQUID |
| flawless | +50 | GAS | 8 | 0.62 | 0.12 | 0.67 | LIQUID |
| enjoy | +24 | GAS | 13 | 0.85 | 0.15 | 0.67 | LIQUID |
| overjoyed | +20 | GAS | 18 | 0.94 | 0.00 | 0.66 | LIQUID |
| racing | -20 | GAS | 9 | 0.78 | 0.00 | 0.65 | LIQUID |
| wandering | -40 | GAS | 9 | 0.44 | 0.44 | 0.64 | LIQUID |
| vibes | +20 | GAS | 5 | 0.80 | 0.00 | 0.64 | LIQUID |
| major | +28 | GAS | 19 | 0.74 | 0.26 | 0.63 | LIQUID |
| project | +29 | GAS | 17 | 0.71 | 0.23 | 0.60 | LIQUID |

Also high: `love` (+40, **SOLID**, contra=0.57 over n=98 — 0.18 even in human-only labels),
`hope` (+45, contra=0.50, n=32), `information`/`remain`/`wooden`/`distant` (pure neutral
inflation → ZERO). Of 167 suspects with score ≥ 0.15, **159 are GAS, 3 SOLID, only 5 LIQUID**
— i.e. almost none of the contested words have any context protection today.

Evidence flavor (real pool sentences): *"Thank you so much for breaking my heart"*,
*"Congratulations on completely ruining my entire life"*, *"I truly cherish the two hours
I spend in traffic every day"*, *"I finally beat the final boss"* (neg word, pos sentence).

## Proposed math (recommended: confidence-gated phase demotion)

Two distinct bias classes appear in the data, so the neutralizer has two rules plus a
magnitude correction:

1. **Demote to LIQUID** when `excess_contradiction = contra_rate − baseline_opposing > 0.20`.
   The stored force becomes a *presumed* charge: it is applied only when context does not
   oppose it. This reuses the existing phase machinery — SOLVENT dissolution and contextual
   flip already exist for LIQUID; no new physics.
2. **Zero (true GAS)** when `neutral_inflation = neutral_rate − 0.59 > 0.20` and support is
   at/below baseline. The word's charge is population-average noise (the mass-zero class).
3. **Attenuate what remains**: for demoted words, `effective_force = base_force × c_w`,
   `c_w = 1 − contradiction_rate` (charge confidence). `thank` +50→+12, `hope` +45→+22,
   `love` +40→+17. So even when context confirms, the word no longer single-handedly drags
   a sentence positive.

Why demotion over pure scaling (option i alone): scaling keeps the **wrong sign** in the
27–100% of usages where context fights the word. The goal is "context, not the dictionary,
carries the charge" — only a phase change achieves that.

Against the verified failure sentences:
- *"thanks for finally texting back"* — `thanks` (+39, contra 0.27 raw; "thank" 0.77)
  demoted+attenuated: charge needs contextual confirmation; "finally" grievance frame blocks
  it → V no longer jumps to 179.
- *"first christmas without dad"* — `christmas` is already mostly drained (dV=+11) but
  keeps +24 dU/+20 dG celebration charge; demotion makes that conditional, so "without"
  (loss frame) blocks the festive injection and grief carries.
- *"hope it was worth it"* — `hope` contra=0.50, polarized-contradiction-share 0.76 →
  demoted; sarcasm/retribution context refuses confirmation → no +45.
- *"i still have his number saved"* — **not vocabulary bias**: `saved` shows contra=0.23,
  score 0 (KEEP). This failure is tense/counterfactual physics (v8_audit_log §6). The
  neutralizer should not be expected to fix it; honest limitation.

## Retroactive validation (V8 mass-zero, 646 words)

Replaying `forces_curated.py` up to `_V8_MASS_ZERO` recovers pre-zero forces. Of the 643
zeroed words with pre-zero |dV| ≥ 12, **210 have ≥5 sentences of evidence; 90 (43%) exceed
the 0.15 flag threshold and 145 (69%) show non-zero suspect signal** — without ever looking
at the mass-zero list. Top retro hits: attachment, roommate, volunteer, cereal, dedication,
mood, amazed, washing, important. Spot checks: `lawn` (pre-zero dV=+77) flagged at 0.29 via
neutral inflation; `cookie`/`socks` show contradiction=1.0 but n<5 (caught in the
low-evidence tier). The remainder are function words/emotes with no sentence coverage —
exactly the words a sentence-evidence method *should* stay silent on. The audit also
independently rediscovers the council-briefing `note`/`decision`/`wonder` class (now zeroed;
probe labels confirm neutral-dominant usage).

## Caveats / schema notes

- The Gemini half of the pool is sarcasm-adversarial **by construction**, so contradiction
  rates for positive words are upper bounds; the baseline-excess correction compensates, and
  human-only rates are reported (`contradiction_rate_human`) — e.g. `love` still contradicts
  in 18% of human-labeled sentences.
- `council_certain/majority` ids (0–499) all fall inside the human-verified prefix of
  `full_sentence_library.json` → they add labels but no unique sentences.
- **Zero coverage**: "christmas" and "dad" appear in *no* labeled sentence (n=0). The named
  grief failures are not in any dataset — targeted relationship/holiday-grief sentences are
  the single highest-value data collection next step.
- `fo76clanker-graded.json` is engine output (circular — excluded); fo76 texts already live
  in `verified_sentences.json` with human labels.

## Hygiene appendix — junk vocabulary entries (do not fix here; listed for later removal)

All-zero Twitch-emote/streamer-name garbage in `_NEW_WORDS` (engine/forces_curated.py
~lines 5782–6095), harmless but dead weight: `caseoh`, `caseohdailydoodledance`,
`caseohdailydoodlestwerk`, `dinodance`, `pogchamp`, `wutface`, `bigsad`, `bleedpurplehd`,
`goatemotey`, `goldplz`, `heyguys`, `jebasted`, `jinxlul`, `jynxzi`, `komodohype`,
`notlikethis`, `twitchconhype`, `votenay`, `voteyea`, plus streamer first names `casey`,
`chris`, `maddie`. (~77 all-zero entries in that section total; the rest are function words
that arguably belong as explicit zeros.)

## Implemented 2026-06-11

Confidence-gated demotion applied per the rule above (n ≥ 10, excess contradiction
> 0.20), highest evidence first. Each demoted word was added to
`NEUTRALIZED_LIQUID` in `engine/phase.py` (GAS → LIQUID) and its dV attenuated by
`(1 − contradiction_rate)` in `_NEUTRALIZER_DEMOTIONS` (`engine/forces_curated.py`);
other dimensions kept (the audit only measures V-sign contradiction).

### Demoted (19 words, old dV → new dV)

| word | old → new | contra | n | | word | old → new | contra | n |
|---|---|---|---|---|---|---|---|---|
| adore | +48 → 0 | 1.00 | 17 | | response | +39 → +18 | 0.55 | 11 |
| cherish | +37 → 0 | 1.00 | 16 | | truly | +28 → +5 | 0.81 | 70 |
| congratulations | +49 → +5 | 0.90 | 10 | | delighted | +49 → +18 | 0.64 | 14 |
| overwhelmed | −35 → −6 | 0.82 | 11 | | enjoy | +24 → +4 | 0.85 | 13 |
| fantastic | +30 → 0 | 1.00 | 27 | | overjoyed | +20 → +1 | 0.94 | 18 |
| survive | +28 → 0 | 1.00 | 11 | | major | +28 → +7 | 0.74 | 19 |
| thrilled | +30 → +2 | 0.93 | 14 | | project | +29 → +8 | 0.71 | 17 |
| thank | +50 → +12 | 0.77 | 73 | | hope | +45 → +22 | 0.50 | 32 |
| breaking | −77 → −35 | 0.54 | 13 | | secret | −24 → −6 | 0.77 | 17 |
| monster | −52 → −26 | 0.50 | 10 | | | | | |

### Skipped / conflicted

- **love** (+40 SOLID, kept): human-only contradiction 0.18 < 0.29 baseline; the
  headline 0.57 is inflated by Gemini's sarcasm-adversarial sentences.
- **wonderful** (+35 GAS, kept — conflicted): attenuating to +7 broke the
  structural sarcasm "what a ___" template (`structures.py` needs ≥ +15 charge as
  its contradiction signal); regressed `test_structures` + stress sarcasm.
  Demoting it requires teaching the sarcasm detector about presumed charges first.
- **beat** (−51, n=8): below the n ≥ 10 evidence bar; left for the next batch.
  It is why "bro he cooked on that beat" still reads negative.

### Slang re-rated (`_SLANG_RERATE_2026_06` + LIQUID additions in phase.py)

| word | old → new tuple | treatment |
|---|---|---|
| cooked | (0,5,0,0,5) → (−20,20,−12,8,−10) | LIQUID, safer doom reading; SOLVENT flips praise ("bruh he cooked" V=165, "im cooked" V=67). \|dV\| kept < 25 so MUNDANE_HYPERBOLE doesn't defuse "im cooked for this exam". |
| sheesh | (−47,6,−32,3,−21) → (−12,22,−5,5,−5) | LIQUID; exasperation OR awe — −47 was wildly hot for an interjection. |
| snapped | (−5,25,5,5,10) → (−12,28,0,8,0) | LIQUID; "snapped at me" negative, "lmao she snapped" flips positive. |
| drip | (0,5,0,0,10) → (+12,10,8,0,8) | +12 stays below the contrast-sarcasm strong-pos threshold (15) so "his drip is crazy" isn't misread as sarcasm. |
| w | (0,5,0,0,10) → (+30,15,10,0,10) | unambiguous win. |
| mid | (−15,0,0,0,−5) → (−18,5,0,0,−8) | unambiguous mild dismissal. |
| lowkey | (0,−5,0,0,0) → (0,0,0,0,0) | pure SOLVENT/hedge, zero charge. |
| ate | (−5,5,−5,0,−3) → (+5,8,2,0,4) | praise sense, kept sub-threshold because literal eating dominates frequency. |
| peak | absent → (+10,8,5,0,5) | new; kept small (US "peak" = best, UK slang "peak" = bad). |

`fr` (0,5,0,0,0), `goated` (+35), `bussin` (+40) were already correct via
`_V8_CORRECTIONS` — no change. Subject-aware disambiguation ("he cooked" vs
"im cooked" without a register marker) needs force_flow changes — out of scope;
the stored charge is the safer (negative) reading.

### Measurement (before → after, no regressions)

- `pytest engine/tests/`: 219 passed, 2 xfailed → **227 passed, 2 xfailed** (8 new neutralizer tests)
- `crisis_benchmark`: recall 44/51, FP 0/75 → **44/51, FP 0/75**
- `stress_test`: 271/275 → **271/275** (genuine_positive 25/25, slang_positive 25/25 held)
- `full_barrage` ground truth: 40/41 → **40/41** ("i love you" V=179, "i love my mom" V=184)

Dev probes (non-holdout) improved 14/24 → 19/24; notable wins: "sheesh he really
did that" 79→115, "im cooked for this exam" 128→108 (now negative),
"lmao she snapped on that verse" 124→140, "i truly cherish ... traffic" 152→119
(sarcasm no longer positive), "big w for the team" 128→135. Known remaining
misses: "i hope you feel better soon" (V≈87, pre-existing — "better"/"soon" carry
negative charge), "bro he cooked on that beat" ("beat" −51, below evidence bar),
"i ate breakfast at home" (pre-existing +V inflation from "breakfast"/"home").
