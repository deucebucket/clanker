# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Clanker-Lang detects **emotional stance**, not just emotion. "Whatever" alone reads as resignation (D=108). "Whatever makes you happy" reads as passive-aggressive (D=123). "Do whatever" reads as permission (D=129). Same word — context changes the Dominance dimension. A sentiment classifier says "neutral" for all three.

The engine computes 5D emotional coordinates (VADUG: Valence, Arousal, Dominance, Urgency, Gravity) using 26 mathematical forces. 300KB, 0.1ms/sentence, 66% on real-world text, knows its own limits (NULL confidence when it can't resolve meaning). The 34% it can't handle gets handed to a neural model.

Three components:
- **VADUG coordinate system**: 5 bytes encode 1.1 trillion emotional states (Valence, Arousal, Dominance, Urgency, Gravity)
- **Bytecode IR**: Opcodes (0x00-0xFF) with immutable meanings, decoded to any language via YAML dictionaries
- **Pendulum engine**: Rule-based emotional physics — word-by-word processing with momentum, 26 context-dependent forces, morphological decomposition, crisis detection

Key principle: **opcodes are forever** — never redefine an existing opcode.

## Commands

### Tests (264+ across 3 suites)

```bash
# Run ALL tests
pytest demo/ tokenizer/

# Decoder tests (requires install first)
cd decoder/python && pip install -e ".[dev]" && pytest tests/ -v

# Engine tests (demo/)
python3 -m pytest demo/tests/ -v
python3 -m pytest demo/tests/test_bigrams.py -v          # single file
python3 -m pytest demo/tests/test_intent.py::TestGreeting # single class

# Tokenizer tests
pytest tokenizer/tests/ -v
```

### Benchmarks & Evaluation

```bash
# Academic benchmark (Clanker V1/V2 vs VADER vs TextBlob vs RoBERTa)
python3 benchmarks/academic_benchmark.py --quick

# Essay benchmark (full essay emotional arc scoring)
python3 benchmarks/essay_benchmark.py
python3 benchmarks/essay_benchmark.py --verbose   # per-sentence traces

# Experiment tracker (NASA-style versioned logging)
python3 benchmarks/experiment_tracker.py --history
python3 benchmarks/experiment_tracker.py --best
python3 benchmarks/experiment_tracker.py --compare EXP-0001 EXP-0042

# GPU optimizer (genetic algorithm on RTX 3090, ~1M evals)
python3 benchmarks/gpu_optimizer_v2.py

# Find edge cases
python3 benchmarks/find_cracks.py

# Cross-language validation
python3 benchmarks/rosetta_stone.py

# Crisis recall benchmark
python3 benchmarks/crisis_benchmark.py
python3 benchmarks/crisis_benchmark.py --verbose   # show all misclassifications

# Ablation study (kill-switch force bloat detection)
python3 benchmarks/ablation_study.py
python3 benchmarks/ablation_study.py --workers 4

# EmoBank optimizer (human agreement tuning)
python3 benchmarks/emobank_optimizer.py
python3 benchmarks/emobank_optimizer.py --quick

# Human rating evaluation framework
python3 benchmarks/human_eval.py --generate          # generate evaluation set
python3 benchmarks/human_eval.py --analyze ratings.csv  # analyze human ratings
```

### Training Pipeline

```bash
# Train Clanker-Micro (~4.8M params, GPT-2 backbone, 5 VADUG heads, focal loss)
python3 training/train.py

# Generate training data
python3 training/generate_training_data.py          # Phase 1 (Rosetta) + Phase 2 (corpus)
python3 training/generate_scored_essays.py           # Essays with per-word VADUG traces
python3 training/generate_pangram_essays.py          # 8 lenses x 2,049 words = 16K sentences
python3 training/rosetta_triplets.py                 # A+B=C triplets, 5-band prism
python3 training/generate_triplets_bulk.py --generate   # LLM consensus triplets (4 steps)
python3 training/expand_training_data.py             # Fix neutral collapse + positive blindness
python3 training/convert_empathetic_dialogues.py     # Facebook EmpatheticDialogues → A+B=C

# Validation
python3 training/gap_finder.py                       # Find missing mechanics
python3 training/gap_finder.py --generate-fixes      # Generate engine patches
python3 training/idiom_discoverer.py                 # Find idioms from residuals (99.5K sentences)
python3 training/engine_vs_model.py                  # Adversarial engine vs model comparison
```

### Other

```bash
# Interactive demo (7-layer pipeline simulator)
python3 demo/simulator.py

# HuggingFace Space app
cd space && pip install -r requirements.txt && python3 app.py
```

## Architecture

### Modular Engine (`demo/`)

The engine is split into pipeline-ready modules. 35+ files grouped by function:

**Core pipeline:**

| Module | What |
|--------|------|
| `shared.py` | VADUG, MetadataHeader, PersonalityVector dataclasses |
| `forces_curated.py` | **V2 vocabulary** — `EMOTIONAL_VOCABULARY`, 2,154 curated words + 2,623 total mapped entries (replaced 46K `forces.py`) |
| `forces.py` | Legacy V1 vocabulary — 46K WORD_FORCES entries, 46K lines. Still imported by V1 modules. Do not read whole file. |
| `pendulum_v2.py` | **Active engine (V2)** — uses `EMOTIONAL_VOCABULARY`, 26 forces, 14 tunable params |
| `pendulum.py` | Legacy V1 pendulum — uses `WORD_FORCES`. Still imported by some paths, being phased out. |
| `personality.py` | apply_personality() — 8-knob resistance vector |
| `response.py` | ResponseBuilder, harmony math, emotion mapping |
| `arc.py` | ChunkedPipeline, run_pipeline — orchestrates everything |
| `simulator.py` | Backward-compatible shim + CLI entry point |
| `pipeline_config.py` | Pipeline configuration |

**Analysis & detection:**

| Module | What |
|--------|------|
| `chunker.py` | ChunkSplitter — paragraph-level arc detection |
| `grader.py` | SentenceGrader — 15-step emotional guardrails (A+ through F-) |
| `sarcasm.py` | SarcasmDetector — three-signal analysis |
| `classifier.py` | Text classification |
| `intent.py` | Intent detection |
| `nonsense.py` | Nonsense/gibberish detection |
| `entropy.py` | Entropy-based analysis |
| `tonal.py` | Tonal analysis |
| `anomaly.py` | AnomalyDetector — gravity wells, masking, velocity anomalies, resonance patterns |
| `preflight.py` | PreflightAnalyzer — digital prosody, environmental multipliers before word processing |

**Conversation & API:**

| Module | What |
|--------|------|
| `conversation.py` | ConversationEngine — trajectory tracking with TCI escalation detection |
| `clanker_api.py` | Three-layer API — sentence physics + conversation trajectory + Dark Matter |

**Language mechanics:**

| Module | What |
|--------|------|
| `morphemes.py` | Morphological decomposition roots |
| `bigrams.py` | Bigram force patterns |
| `word_roles.py` | Word role classification |
| `fuzzy.py` | Fuzzy matching for unknown words |
| `forces_curated.py` | (see Core pipeline — this is the V2 vocabulary) |
| `ring_forces.py` | Ring-based force composition |
| `bookend.py` | Sentence bookend detection |
| `idioms.py` | Standalone idiom dictionary — extracted from `pendulum.py` to avoid importing `forces.py` |

**Advanced systems:**

| Module | What |
|--------|------|
| `context_operators.py` | Context-dependent force operators |
| `dark_matter.py` | Unmeasured emotional influence |
| `memory.py` | Emotional memory across sentences |
| `multipath.py` | Multi-path emotional resolution |
| `outcome_optimizer.py` | A+B=C outcome prediction — Doctor Strange mode (standalone, not yet in main pipeline) |
| `trace.py` | Pipeline trace/debug output |

All imports via `from demo.simulator import X` still work.

**Tests (`demo/tests/`):** 11+ test files covering bigrams, bookends, density, fuzzy matching, intent, memory, nonsense, ramps, tonal analysis, word roles.

### Decoder (`decoder/python/clanker_decoder/`)

- **decoder.py** — Parses `.clank` instructions, renders templates from dictionaries
- **loader.py** — Loads/caches YAML dictionaries from `dictionaries/{human,code,other}/`
- **validator.py** — Validates `.clank` scripts against opcode specs

### Tokenizer (`tokenizer/`)

958-token direct-mapping tokenizer (not BPE). HuggingFace PreTrainedTokenizer compatible. Each opcode IS a token.

### Opcodes (`opcodes/`) and Dictionaries (`dictionaries/`)

Seven YAML files defining opcodes by range. Dictionaries map opcodes to language-specific templates. Adding a language = adding one YAML file.

### Rules (`rules/`)

YAML rule definitions for the engine:
- `composition.yaml` — How forces compose
- `constraints.yaml` — System constraints and bounds
- `types.yaml` — Type definitions

### Training (`training/`)

Model training pipeline. The engine reads English — the model learns to think in VADUG.

| Script | What |
|--------|------|
| `train.py` | Main training entry point |
| `generate_training_data.py` | Generate training pairs |
| `generate_scored_essays.py` | Generate essays with emotional scores |
| `generate_pangram_essays.py` | Pangram-style essay generation |
| `gap_finder.py` | Find coverage gaps in training data |
| `rosetta_triplets.py` | Rosetta Stone triplet generation |
| `generate_triplets_bulk.py` | Bulk triplet generation |
| `expand_training_data.py` | Expand existing training data |
| `convert_empathetic_dialogues.py` | Convert EmpatheticDialogues dataset |
| `idiom_discoverer.py` | Discover new idiom patterns |
| `engine_vs_model.py` | Adversarial engine vs model comparison pipeline |

Data lives in `training/data/`, checkpoints in `training/checkpoints/` (both gitignored for large files).

### Benchmarks (`benchmarks/`)

| Script | What |
|--------|------|
| `academic_benchmark.py` | Academic-grade evaluation (use `--quick` for fast run) |
| `essay_benchmark.py` | Essay-level emotional arc evaluation |
| `experiment_tracker.py` | Versioned experiment logging (NASA-style) |
| `gpu_optimizer_v2.py` | GPU-optimized parameter tuning |
| `rosetta_stone.py` | Cross-language validation |
| `find_cracks.py` | Find edge cases and failure modes |
| `crisis_benchmark.py` | Crisis recall at scale — targeting 99.9% recall |
| `ablation_study.py` | Kill-switch force bloat detection — disable one force at a time |
| `emobank_optimizer.py` | Human V/A/D agreement tuning via EmoBank |
| `human_eval.py` | Human rating evaluation framework — 5D correlation |

### HuggingFace Space (`space/`)

Gradio-based demo app for public-facing interaction. `app.py` + `requirements.txt`.

## Key Conventions

- Hex codes uppercase: `0xFF` not `0xff`
- YAML 2-space indentation
- Opcode names UPPER_SNAKE_CASE, parameters lowercase_with_underscores
- Instruction format: `@ 0xHH $target $source PARAM_COUNT {key: "value"}`
- IDIOM tuples: 5-element `(dv, da, dd, du, label)` or 6-element `(dv, da, dd, du, dg, label)` with gravity
- Force tuples: `(dv, da, dd, du, dg)` — deltas, not absolute values. V2 uses `EMOTIONAL_VOCABULARY`, V1 uses `WORD_FORCES`

## Key References

| Doc | What |
|-----|------|
| `docs/vadug-calculation.md` | Complete VADUG formula reference — the full sentence equation, step-by-step processing |
| `docs/tuning-notes.md` | All tuning decisions, ablation results, parameter values, personality defaults |
| `docs/THEORY.md` | Full theory — dark matter, outcome prediction, gravitational coupling |
| `docs/tci-cares-research.md` | TCI/CARE research connection to VADUG |
| `docs/linguistic-devices-taxonomy.md` | Taxonomy of linguistic devices the engine handles |
| `training/README.md` | Training data format, phases, model architecture, data sources |
| `benchmarks/rosetta_stone.py` | Rosetta Stone calibration sentences — the ground truth |
| `SPEC.md` | Full engine specification |
| `demo/clanker_api.py` | Three-layer API architecture — sentence + conversation + Dark Matter |
| `demo/anomaly.py` | Anomaly detection theory — gravity wells, masking, velocity, resonance |

## Gotchas

- **V2 is the active engine** — `pendulum_v2.py` + `forces_curated.py` (2,154 curated words, 26 forces). V1 (`pendulum.py` + `forces.py` 46K words) is legacy.
- **`forces.py` is 46K lines** — do not read the whole file. V2 doesn't use it. Only grep if debugging V1 paths.
- **Inactive modules** — `stone_correction.py`, `output_modes.py`, `word_factory.py` are not imported by active code. Archive candidates, not to be wired into new work.
- **Gravity and Dominance are the driving forces** — not Valence. The system's core insight.
- **Idioms are in `idioms.py`** — V2 imports idioms from `demo/idioms.py`, not from `pendulum.py`. The standalone module avoids pulling in the 46K-line `forces.py`.
- **Training data is gitignored** — large files like `discovered_idioms.jsonl` and `empathetic_dialogues.jsonl` won't be in the repo.
