# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Clanker-Lang is a **conversation state resolver** that detects emotional stance through structural pattern recognition. It reads text the way a chess player reads a board  --  recognizing patterns from piece positions, not memorizing specific games. "Whatever" alone reads as resignation (D=108). "Whatever makes you happy" reads as passive-aggressive (D=123). "Do whatever" reads as permission (D=129). Same word  --  context changes the Dominance dimension. A sentiment classifier says "neutral" for all three.

The engine computes 7D emotional coordinates (VADUGWI: Valence, Arousal, Dominance, Urgency, Gravity, Self-Worth, Intent) using structural pattern recognition. 4,475-word vocabulary, 45+ structural patterns, 0.17ms/sentence. Tested on 246K real sentences (novels, Twitch, Reddit, philosophy, game dialogue). Returns NULL confidence when it can't resolve meaning.

**Current verified numbers (V8.2, 2026-04-04):**
- Ground truth: 41/41 (100%) on developer-verified sentences
- Stress test: 201/275 (73.1%) on 275 real sentences across 11 categories
- Crisis recall: 70.6% (36/51 crisis sentences detected)
- Crisis false positive: 0/75 (0.0%)  --  zero false positives on safe text, dark humor, AND metaphor
- SST-2: 63.9% (beats VADER 55.7%, behind RoBERTa 69%)
- GoEmotions: 48.8% (28-emotion classification  --  different task than 3-way sentiment)
- Real-text spot-check: ~59% on 167 manually verified sentences across 5 corpora
- Throughput: 2,000-6,000 sentences/sec depending on corpus

**Known weaknesses:** slang positive (44%), grief (52%), passive aggressive (68%). Positive inflation reduced but not eliminated. SOLVENT dissolution needs stronger propagation.

**V8 is the current engine** (`engine/` directory). V2 is boxed at tag `v2.0` (`demo/` directory).

Four V8 systems:
- **Structure Recognition**: Words classified into 4 tiers (primary signal words, secondary signal words, operators, unclassified words). Connectors are math operators (and=+, but=-, or=><, of=/, if=?). 45+ structural patterns detected like chess positions. V8 adds: MUNDANE_HYPERBOLE, BOUNDARY_VIOLATION, SELF_ERASURE, DIVESTITURE, METHOD_FIXATION, RARITY_MARKER, ABANDONMENT, LIFE_ACHIEVEMENT.
- **A+B=C Bidirectional Solver**: Given state A + response B, predict outcome C. Or work backwards from target zone to find valid B.
- **Probe Calibration System**: Fire calibrated probes, measure vibration/distortion, triangulate hidden state.
- **Force Flow Resolver** (`engine/force_flow.py`): WHO does WHAT to WHOM  --  resolves directional force between entities in a sentence.

Additional features: absence scope ("havent had X" dampens absent events), compound phrase resolution ("no one" -> nobody, "no cap" -> nocap), RELIEF_ABSENCE / SELF_EXCLUDED / WITHHELD_POSITIVE patterns, Bayesian vocabulary corrections, forced choice cancellation, phase system (SOLID/LIQUID/GAS matter states), SOLVENT dissolution (casual register flips LIQUID negative to positive), mundane dampening (inert gas absorption of crisis energy), interpret_context layer (discourse markers, register detection, counterfactual inversion).

Additional components:
- **VADUGWI coordinate system**: 7 bytes encode 72 quadrillion emotional states (W=Self-Worth, I=Intent: withdraw/deflect/neutral/connect/control)
- **Bytecode IR**: Opcodes (0x00-0xFF) with immutable meanings, decoded to any language via YAML dictionaries

Key principle: **opcodes are forever**  --  never redefine an existing opcode.

## Commands

### Tests (167 in V8 engine, plus V2 and decoder/tokenizer suites)

```bash
# Run V8 engine tests (active)
python3 -m pytest engine/tests/ -v

# Run V2 engine tests (legacy, tagged v2.0)
python3 -m pytest demo/tests/ -v

# Decoder tests (requires install first)
cd decoder/python && pip install -e ".[dev]" && pytest tests/ -v

# Tokenizer tests
pytest tokenizer/tests/ -v

# Run ALL tests
pytest engine/ demo/ tokenizer/
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
# Train Clanker-Micro (~22.6M params, GPT-2 backbone, 7 VADUGWI heads, focal loss)
python3 training/train.py

# Generate training data
python3 training/generate_training_data.py          # Phase 1 (Rosetta) + Phase 2 (corpus)
python3 training/generate_scored_essays.py           # Essays with per-word VADUGWI traces
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

### V8 Engine (`engine/`)  --  Active

The V8 engine uses structural pattern recognition  --  words classified by role, proximity fields computed, then chess-like pattern detection on role sequences. No hardcoded word lists in pattern detection. 7D VADUGWI coordinates (W=Self-Worth, I=Intent), force flow resolution, absence scope, and Bayesian vocabulary corrections.

| Module | What |
|--------|------|
| `shared.py` | VADUGWI dataclass  --  7-byte emotional coordinate (V, A, D, U, G, W, I) |
| `word_classifier.py` | Layer 1  --  structural role classification (SELF_REF, EMOTIONAL, NEGATOR, CONNECTOR, etc.) |
| `vocabulary.py` | V8 vocabulary  --  imports 4,000+ curated words from `engine/forces_curated.py` |
| `proximity.py` | Layer 2  --  distance-based influence fields, exponential decay (0.7x per word) |
| `structures.py` | Layer 3  --  chess-like pattern detector (45+ structural patterns, role sequences -> structural matches). V8 adds MUNDANE_HYPERBOLE, BOUNDARY_VIOLATION, SELF_ERASURE, DIVESTITURE, METHOD_FIXATION, RARITY_MARKER, ABANDONMENT, LIFE_ACHIEVEMENT |
| `pendulum.py` | Fixed physics layer  --  9-stage pipeline: tokenize, classify, interpret_context (V8 discourse/register/counterfactual), coefficients, accumulate_forces (mundane dampening), structures, W→V coupling (asymmetric exponential), personality, saturate. Momentum, adaptive force application, tanh saturation |
| `phase.py` | Matter state system  --  SOLID (never flips), LIQUID (context flips), GAS (neutral). SOLVENT words dissolve LIQUID atoms |
| `crisis.py` | CrisisTracker  --  continuous 0.0-1.0 concern gradient, TCI-informed, trajectory accumulation |
| `anomaly.py` | AnomalyDetector  --  4 detectors: DEFLECTION, MASKING, VELOCITY, RESONANCE (7D VADUGWI) |
| `trace.py` | PipelineTrace  --  debug logging for every pipeline stage |
| `personality.py` | Personality vector application |
| `solver.py` | A+B=C bidirectional solver  --  forward (text->VADUGWI) and backward (target zone->valid B range) |
| `battleship.py` | Probe system  --  fire calibrated probes, measure vibration, triangulate hidden state |
| `force_flow.py` | Force flow resolver  --  WHO does WHAT to WHOM directional analysis |
| `forces_curated.py` | V8 vocabulary  --  4,475 curated emotional words with VADUGWI forces. V8 mass-zeroed 646 GAS atoms |
| `zones.py` | Zone classification (imports from V2) |
| `zones_impl.py` | Zone implementation details |
| `fuzzy.py` | Fuzzy matching for unknown words |

**Tests (`engine/tests/`):** 207 tests across 8 test files covering word classification, structures, proximity, pendulum, solver, battleship, scaffolding, and novel sentences.

### V2 Engine (`demo/`)  --  Legacy (tagged v2.0)

The V2 engine is boxed at tag `v2.0`. It uses a pendulum model with 26 conversational forces and curated vocabulary. Still functional, still importable via `from demo.simulator import X`.

**Core pipeline:**

| Module | What |
|--------|------|
| `shared.py` | VADUG, MetadataHeader, PersonalityVector dataclasses |
| `forces_curated.py` | **V2 vocabulary**  --  `EMOTIONAL_VOCABULARY`, legacy curated words (replaced 46K `forces.py`) |
| `forces.py` | Legacy V1 vocabulary  --  46K WORD_FORCES entries, 46K lines. Do not read whole file. |
| `pendulum_v2.py` | **V2 engine**  --  uses `EMOTIONAL_VOCABULARY`, 26 forces, 14 tunable params |
| `pendulum.py` | Legacy V1 pendulum  --  uses `WORD_FORCES` |
| `personality.py` | apply_personality()  --  8-knob resistance vector |
| `response.py` | ResponseBuilder, harmony math, emotion mapping |
| `arc.py` | ChunkedPipeline, run_pipeline  --  orchestrates everything |
| `simulator.py` | Backward-compatible shim + CLI entry point |
| `pipeline_config.py` | Pipeline configuration |

**Analysis & detection:**

| Module | What |
|--------|------|
| `chunker.py` | ChunkSplitter  --  paragraph-level arc detection |
| `grader.py` | SentenceGrader  --  15-step emotional guardrails (A+ through F-) |
| `sarcasm.py` | SarcasmDetector  --  three-signal analysis |
| `classifier.py` | Text classification |
| `intent.py` | Intent detection |
| `nonsense.py` | Nonsense/gibberish detection |
| `entropy.py` | Entropy-based analysis |
| `tonal.py` | Tonal analysis |
| `anomaly.py` | AnomalyDetector  --  gravity wells, masking, velocity anomalies, resonance patterns |
| `preflight.py` | PreflightAnalyzer  --  digital prosody, environmental multipliers before word processing |

**Conversation & API:**

| Module | What |
|--------|------|
| `conversation.py` | ConversationEngine  --  trajectory tracking with TCI escalation detection |
| `clanker_api.py` | Three-layer API  --  sentence physics + conversation trajectory + unclassified words |

**Language mechanics:**

| Module | What |
|--------|------|
| `morphemes.py` | Morphological decomposition roots |
| `bigrams.py` | Bigram force patterns |
| `word_roles.py` | Word role classification |
| `fuzzy.py` | Fuzzy matching for unknown words |
| `ring_forces.py` | Ring-based force composition |
| `bookend.py` | Sentence bookend detection |
| `idioms.py` | Standalone idiom dictionary |

**Advanced systems:**

| Module | What |
|--------|------|
| `context_operators.py` | Context-dependent force operators |
| `dark_matter.py` | Unmeasured emotional influence |
| `memory.py` | Emotional memory across sentences |
| `multipath.py` | Multi-path emotional resolution |
| `outcome_optimizer.py` | A+B=C outcome prediction  --  bidirectional solver mode |
| `trace.py` | Pipeline trace/debug output |

### Decoder (`decoder/python/clanker_decoder/`)

- **decoder.py**  --  Parses `.clank` instructions, renders templates from dictionaries
- **loader.py**  --  Loads/caches YAML dictionaries from `dictionaries/{human,code,other}/`
- **validator.py**  --  Validates `.clank` scripts against opcode specs

### Tokenizer (`tokenizer/`)

958-token direct-mapping tokenizer (not BPE). HuggingFace PreTrainedTokenizer compatible. Each opcode IS a token.

### Opcodes (`opcodes/`) and Dictionaries (`dictionaries/`)

Seven YAML files defining opcodes by range. Dictionaries map opcodes to language-specific templates. Adding a language = adding one YAML file.

### Rules (`rules/`)

YAML rule definitions for the engine:
- `composition.yaml`  --  How forces compose
- `constraints.yaml`  --  System constraints and bounds
- `types.yaml`  --  Type definitions

### Training (`training/`)

Model training pipeline. The engine reads English  --  the model learns to think in VADUGWI.

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
| `full_barrage.py` | One-command full test: ground truth + stress + crisis + throughput + conversations. The main benchmark. |
| `stress_test.py` | 275 sentences across 11 categories (sarcasm, slang, grief, betrayal, crisis, etc.). Real conversational text. |
| `crisis_benchmark.py` | 126 sentences: 51 crisis + 55 safe + 10 dark humor + 10 metaphor. Zero false positive target. |
| `academic_benchmark.py` | SST-2 + GoEmotions + TweetEval vs VADER vs TextBlob vs RoBERTa (use `--quick` for fast run) |
| `essay_benchmark.py` | Essay-level emotional arc evaluation |
| `experiment_tracker.py` | Versioned experiment logging (NASA-style) |
| `gpu_optimizer_v2.py` | GPU-optimized parameter tuning |
| `rosetta_stone.py` | Cross-language validation |
| `find_cracks.py` | Find edge cases and failure modes |
| `ablation_study.py` | Kill-switch force bloat detection  --  disable one force at a time |
| `emobank_optimizer.py` | Human V/A/D agreement tuning via EmoBank |
| `human_eval.py` | Human rating evaluation framework  --  7D correlation |

### HuggingFace Space (`space/`)

Gradio-based demo app for public-facing interaction. `app.py` + `requirements.txt`.

## Key Conventions

- Hex codes uppercase: `0xFF` not `0xff`
- YAML 2-space indentation
- Opcode names UPPER_SNAKE_CASE, parameters lowercase_with_underscores
- Instruction format: `@ 0xHH $target $source PARAM_COUNT {key: "value"}`
- IDIOM tuples: 7-element `(dv, da, dd, du, dg, dw, label)` or 8-element `(dv, da, dd, du, dg, dw, di, label)` with intent
- Force tuples: `(dv, da, dd, du, dg, dw, di)`  --  deltas, not absolute values. V8 uses `engine/forces_curated.py`, V2 uses `demo/forces_curated.py`, V1 uses `WORD_FORCES`

## Key References

| Doc | What |
|-----|------|
| `docs/v3-user-physics.md` | V3 structural rules  --  the laws the engine encodes |
| `docs/vadug-calculation.md` | Complete VADUGWI formula reference  --  the full sentence equation, step-by-step processing |
| `docs/tuning-notes.md` | All tuning decisions, ablation results, parameter values, personality defaults |
| `docs/THEORY.md` | Full theory  --  structural pattern recognition, unclassified words, outcome prediction |
| `docs/tci-cares-research.md` | TCI/CARE research connection to VADUGWI |
| `docs/linguistic-devices-taxonomy.md` | Taxonomy of linguistic devices the engine handles |
| `docs/v8_audit_log.md` | V8 real-data spot-check audit  --  59% accuracy on 167 sentences, 6 physics problems identified |
| `docs/council_round5_synthesis.md` | 4-LLM consensus  --  GPT/Claude/Gemini/Grok agree on 6 physics fixes |
| `docs/v7_physics_spec.md` | V7 physics spec  --  Grok's pure-physics solutions |
| `datasets/verified_sentences.json` | 71 human-verified correct sentences from spot-check audit |
| `training/README.md` | Training data format, phases, model architecture, data sources |
| `benchmarks/rosetta_stone.py` | Rosetta Stone calibration sentences  --  the ground truth |
| `SPEC.md` | Full engine specification |
| `demo/clanker_api.py` | V2 three-layer API architecture  --  sentence + conversation + unclassified words |
| `demo/anomaly.py` | Anomaly detection theory  --  gravity wells, masking, velocity, resonance |

## Gotchas

- **V8 is the active engine**  --  `engine/` directory. Structural pattern recognition (word roles + proximity + structure detection) + force flow + 7D VADUGWI. V2 (`demo/`) is boxed at tag `v2.0`.
- **V8 vocabulary is in engine/**  --  `engine/vocabulary.py` imports from `engine/forces_curated.py`. 4,475 curated words with 7D forces. V8 mass-zeroed 646 inflated GAS atoms.
- **`forces.py` is 46K lines**  --  do not read the whole file. Neither V3 nor V2 uses it directly. Only grep if debugging V1 paths.
- **Inactive modules**  --  `stone_correction.py`, `output_modes.py`, `word_factory.py` are not imported by active code. Archive candidates.
- **Gravity and Dominance appear to be the driving forces**  --  not Valence. The data suggests this but I am still testing it.
- **Idioms are in `idioms.py`**  --  V2 imports idioms from `demo/idioms.py`, not from `pendulum.py`. The standalone module avoids pulling in the 46K-line `forces.py`.
- **Training data is gitignored**  --  large files like `discovered_idioms.jsonl` and `empathetic_dialogues.jsonl` won't be in the repo.
- **207 tests in V8**  --  `engine/tests/` covers word classification, structures, proximity, pendulum, solver, battleship, scaffolding, and novel sentences.


## Distribution Rules

- **GGUF only** for public distribution (opaque neural weights, no source visible)
- **NEVER upload** engine source code, forces_curated.py, training scripts, or training data publicly
- **NEVER train** with canned/scripted responses -- teach the VADUGWI math, not memorized pairs
- The model is the free sample. The engine is the business.

