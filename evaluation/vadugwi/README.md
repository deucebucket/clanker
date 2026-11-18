# VADUGWI held-out evaluation

Measures the valence axis of `engine.pendulum.compute_vadug` on data the
engine was not fitted to, and compares two engine versions item by item.

## Data

The data is not in this repository (`benchmarks/` and `datasets/` are
gitignored, and holdout v2 is sealed). Point the harness at a private
checkout:

```bash
export CLANKER_VADUGWI_DATA=/path/to/private/clanker   # has benchmarks/, datasets/, training/data/
export CLANKER_VADUGWI_SST2=/path/to/sst2_validation.json   # optional; [{"text", "label": "pos"|"neg"}]
```

| Set | Role | n | What it is |
|---|---|---|---|
| `council_train` | fit | 1,492 | `datasets/council_v2_train.json`, the open 80% of the council-graded pool |
| `emobank_dev` | fit | 5,046 | half of EmoBank (human valence), split by a hash of the text |
| `council_heldout` | heldout | 373 | `benchmarks/holdout_v2_probes.json`, sealed 2026-06-11 |
| `heldout_v1` | heldout | 45 | `benchmarks/holdout_probes.json` |
| `emobank_test` | heldout | 5,010 | the other half of EmoBank |
| `sst2_validation` | heldout | 872 | SST-2 validation, a different domain |
| `stress` | regression | 275 | `benchmarks/stress_test.py`, the in-repo suite the engine was tuned on |

`core.load(name, "fit")` raises `PermissionError` for anything whose role is
not `fit`. The reliability fit goes through that call.

## Why the held-out sets are clean

- **Holdout v2** was split off on 2026-06-11 (seed 8, stratified) and
  sealed the same day; `HOLDOUT_PROTOCOL.md` forbids tuning on it or
  displaying it. The only engine commits since are two on 2026-06-15
  (bravado, orthographic arousal), neither driven by it. The harness
  reproduces the recorded baseline exactly: 154/373. It shares a source
  pool with `council_train`, so it is in-distribution for anything fitted
  on that pool. That is why the other two sets are here.
- **EmoBank** has human ratings. The V1/V2 engines were tuned against it
  in March 2026 and later deleted; I found no record of V8 constants or
  structures being fitted to it. The dev/test split is by text hash, and
  only the dev half was looked at while choosing rules.
- **SST-2 validation** was downloaded for this evaluation. It is not
  pristine: vocabulary words were added from SST-2 (`_BENCHMARK_WORDS`),
  and `benchmarks/sst2_ceiling.py` once searched constants on this split
  (not adopted). V8.6 changes neither the vocabulary nor those constants,
  and both versions are compared on the same items, so the paired
  difference is still meaningful. Treat the absolute level as a domain
  check.
- One disclosure: while exploring file formats, the first row of the holdout v2
  file was printed once. No rule or constant was chosen from it.

Every rule in V8.6 was chosen on the fit sets before the held-out sets
were measured. The "tried and not shipped" candidates were also measured on
held-out data, after the decision not to ship them was made on other
grounds, so that the table could report them.

## Metrics

Three-class accuracy at the repo's thresholds (neg < 110, pos >= 145),
balanced accuracy (mean per-class recall), macro-F1, the majority-class
accuracy for reference, and a threshold-free AUC: the probability that a
positive sentence gets a higher V than a negative one. Holdout v2 is 62%
negative, so accuracy alone rewards anything that pushes V down; read it
next to balanced accuracy and AUC. EmoBank uses Pearson r against the
human valence rating.

Intervals are 95% percentile bootstrap over items (2,000 draws, seed
20261118). `compare` resamples the same items for both runs, so its
interval is for the paired difference.

## Commands

```bash
python -m evaluation.vadugwi run --label my_run          # writes results/my_run.json (+ .items.json)
python -m evaluation.vadugwi compare results/0_baseline_v85.json results/my_run.json
python -m evaluation.vadugwi fit-reliability             # regenerates engine/structure_reliability.py from council_train
```

Result files hold aggregates and one V per item keyed by a hash of the
text. No sentence is written anywhere.

## Results on file

`results/0_baseline_v85` is `origin/main` 92bd354. `1_`, `2_` and `3_` are
the three V8.6 changes in commit order; `3_forward_negation` is V8.6. The
table and the candidates that were not shipped are in
`docs/vadug-calculation.md`, "V8.6: what changed and why".
