# Clanker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19383636.svg)](https://doi.org/10.5281/zenodo.19383636)

A conversation state resolver that detects emotional stance through structural pattern recognition. Computes 7-dimensional emotional coordinates (VADUGWI) from text using deterministic, explainable transformations. Every output can be traced through explicit math. You can ask WHY and get a real answer.

"Whatever" alone reads as resignation (V=93, D=97). "Whatever makes you happy" reads as passive-aggressive (V=30, D=113). "Do whatever" reads as dismissive permission (V=93, D=97). Same word -- context changes the coordinates. A sentiment classifier says "neutral" for all three.

## Live demo

**[Clanker the pet](https://huggingface.co/spaces/deucebucket/clanker)** -- a community virtual creature on Hugging Face whose emotional state is this engine, made visible. Talk to it, give it toys, feed it; it roams, reacts, and self-soothes, and every feeling traces back to the exact words and actions that caused it (the raw read, the structures detected, the per-dimension contributions). No language model -- pure deterministic VADUGWI physics. It doubles as a real-world evaluation harness: live usage surfaces engine gaps that feed back into the formulas (see `docs/v8_audit_log.md`).

### Private Clanker-LM workbench

The deterministic conversation runtime is deployed at
**[https://bazzite.tail85f65f.ts.net:8444/](https://bazzite.tail85f65f.ts.net:8444/)**
for the allowlisted Tailnet login `jerrymares@gmail.com`. It is served through
Tailscale Serve on port 8444, is **tailnet only**, and has no Funnel/public
exposure. Each browser session gets isolated memory and each reply displays its
six evidence fields: **Answer**, **Truth**, **Source**, **Certainty**,
**Memory**, and **VADUG**. See
[`docs/CLANKER_LM_WEB.md`](docs/CLANKER_LM_WEB.md) for operation, verification,
security boundaries, and rollback.

## VADUGWI Coordinates

Seven dimensions, each 0--255 with 128 as neutral center (Urgency starts at 0):

| Dim | Low (0) | Center (128) | High (255) | Measures |
|-----|---------|--------------|------------|----------|
| **V** Valence | Strongly negative | Neutral | Strongly positive | Emotional direction |
| **A** Arousal | Very calm | Moderate | Very intense | Energy level |
| **D** Dominance | Helpless | Balanced | In full control | Agency and power |
| **U** Urgency | None | Moderate | Critical | Time pressure |
| **G** Gravity | Crushing weight | Grounded | Light, floating | Emotional weight |
| **W** Self-Worth | Shattered | Stable | Strong | Self-evaluation |
| **I** Intent | Withdraw | Deflect/Neutral | Connect/Control | Communicative direction |

7 bytes encode 72 quadrillion possible emotional states.

## What the Engine Reads

| Input | V | Notes |
|-------|---|-------|
| "I'm fine" | 83 | Below neutral -- uneasy, not positive |
| "haha yeah im totally okay" | 15 | Forced composure, bravado mask detected |
| "oh joy" | 20 | Positive word, negative reading |
| "do you even love me" | 152 | Positive word, but A=154 D=133 -- challenge energy |
| "my wife cheated on me with my best friend" | 14 | V=14, A=151, D=56 -- deep negative, high intensity, low control |
| "nice of you to finally answer" | 54 | Grievance smuggled into thanks; I=168 -- fighting, deniably |
| "i finally got my license" | 209 | Same "finally", own action -- joy, not grievance |
| "i still have his number saved" | 95 | Ghost possession -- keepsake of an absent person reads as grief |
| "its all my fault" | 30 | W=53 -- valence routed into self-worth, not just mood |
| "whats the point" | 49 | I=64 -- futility sinks the agency axis |
| "I love my mom" | 184 | Genuine positive, no false alarm |
| "the meeting is at three" | 128 | Neutral -- no emotional content detected |

## How It Works

Four processing layers run in sequence:

1. **Word Classification** -- each word is assigned structural roles (SELF_REF, EMOTIONAL, NEGATOR, AMPLIFIER, CONNECTOR, CHOPPER, etc.)
2. **Proximity Weighting** -- nearby words influence each other with exponential decay (0.90x per word of distance)
3. **Structure Detection** -- 66 chess-like patterns detected from role sequences
4. **Physics** -- 9-stage pipeline: tokenize, classify, interpret context, coefficients, accumulate forces, structure adjustment, W-V coupling, personality, tanh saturation

Additional systems:
- **Force Flow** -- WHO does WHAT to WHOM directional analysis
- **W Attribution Routing** -- valence reaches Self-Worth only in proportion to how much the force is about the speaker (self-declarative > targeted > guilt > atmospheric: zero)
- **I Agency Axis** -- futility phrases ("whats the point") sink Intent to 64, volition phrases ("im going to fix this") lift it to 168; never overrides a strong directional read
- **Phase System** -- SOLID (never flips), LIQUID (context-dependent), NEUTRALIZED (context carries the charge), GAS (neutral) word states
- **Crisis Detection** -- continuous 0.0-1.0 concern gradient, zero false positives on safe text
- **Anomaly Detection** -- deflection, masking, velocity, resonance patterns
- **Bidirectional Solver** -- given state A and target zone C, find valid response range B

The core equations are documented in `docs/vadug-calculation.md`.

## Quick Start

```bash
git clone https://github.com/deucebucket/clanker.git
cd clanker
pip install -r requirements.txt
python3 -m pytest engine/tests/ -v
```

```python
from engine.pendulum import compute_vadug

result, context = compute_vadug("whatever makes you happy")
print(f"V={result.v}, A={result.a}, D={result.d}, U={result.u}, G={result.g}, W={result.w}, I={result.i}")
# V=30, A=141, D=113, U=5, G=143, W=102, I=161
```

## Current Numbers

All measured 2026-06-11 against the current engine.

| Metric | Value |
|--------|-------|
| Ground truth (developer-verified, 41 sentences) | 40/41 (97.6%) |
| Stress test (275 real sentences, 11 categories) | 271/275 (98.5%) |
| Crisis recall | 49/51 (96.1%) |
| Crisis false positive (safe + dark humor + metaphor) | 0/75 (0.0%) |
| Held-out probes v1 (45 sentences, never tuned on) | 28/45 (62.2%) |
| Held-out probes v2 (373 council-graded, sealed) | 154/373 (41.3%) |
| SST-2 validation (872 sentences, default constants) | 542/872 (62.2%) |
| Throughput (real corpora: Twitch, novels, philosophy) | 1,000-2,450 sentences/sec |
| Latency (conversational sentences, single core) | 0.45ms per sentence (~2,200/sec) |
| Vocabulary | 4,563 curated words |
| Structural patterns | 66 |
| Tests | 333 (+2 xfail) |
| Engine size | 880 KB, 17,871 lines, stdlib only |

Truth labels for ambiguous sentences come from a consensus of 4 frontier AI models (Gemini, Claude Opus, GPT-4, Grok).

### Honest Evaluation

The in-repo suites (stress test, ground truth, crisis benchmark) have been
tuned against repeatedly -- they measure regression, not generalization, and
they overestimate accuracy. Two holdout sets exist (protocol in
`benchmarks/HOLDOUT_PROTOCOL.md`): v1 is 45 hand-picked probes; v2 is 373
sealed probes built 2026-06-11 from a 2-model LLM council grading 2,144
real and synthetic sentences (87% inter-grader agreement; disagreements
quarantined as an ambiguity set). When a holdout category needs work, new
training sentences are sourced (`datasets/council_v2_train.json`, 1,492
open sentences); the holdout sets themselves are evaluation-only. All
numbers are reported: 98.5% in-repo, 62.2% held-out v1, 41.3% held-out v2.
The biggest gap is the honest one. v2 is the canonical generalization
number going forward — at 373 probes, a 1% delta is real signal, not one
lucky sentence.

## Known Weaknesses

Held-out per-category scores (never tuned on, measured 2026-06-11):

| Category | Held-out v1 (n=15 each) | Held-out v2 (sealed) |
|----------|-------------------------|----------------------|
| Slang positive / casual | 5/15 (33%) | 53/115 (46%) |
| Grief | 10/15 (67%) | 30/65 (46%) |
| Passive aggressive | 13/15 (87%) | 23/75 (31%) |
| Mixed safe | -- | 48/118 (41%) |

Current-generation slang remains weak. The 2026-06-11 slang batch lifted
an independent 80-sentence real-corpus pool from 65% to 96% but did not
transfer to the held-out slang probes (6/15 before, 5/15 after) -- the
register gap is wider than any one vocabulary pass. The v2 open-pool
diagnosis adds the structural finding: 77% of engine errors are boundary
cases -- the engine reads "neutral" where human-calibrated labels commit
to pos/neg. It under-commits; it rarely inverts. Two crisis sentences with zero lexical signal ("tonight is the
night", "ive made my decision") are still missed -- they need conversation
context, not sentence physics.

## File Structure

```
engine/              V8 engine
  pendulum.py          Physics layer -- 9-stage pipeline
  word_classifier.py   Structural role classification
  proximity.py         Proximity field computation
  structures.py        Pattern detection (66 patterns)
  solver.py            Bidirectional A+B=C solver
  force_flow.py        WHO does WHAT to WHOM; W attribution, I agency axis
  forces_curated.py    4,563 word force tuples (dv,da,dd,du,dg)
  crisis.py            Crisis detection (0.0-1.0 gradient)
  phase.py             SOLID/LIQUID/NEUTRALIZED/GAS word states
  anomaly.py           Anomaly detection (4 detectors)
  shared.py            VADUGWI dataclass
  vocabulary.py        Vocabulary loader

engine_v9/           V9 rewrite -- abandoned 2026-06 (kept as archive; see CHANGELOG)

docs/                Reference
  vadug-calculation.md   Full equation reference
  THEORY.md              Theory document
  v3-user-physics.md     Structural rules
  SPEC.md                Full engine specification
```

## Running Tests

```bash
# V8 engine tests
python3 -m pytest engine/tests/ -v

# Full barrage (ground truth + stress + crisis + throughput)
python3 benchmarks/full_barrage.py

# Crisis benchmark
python3 benchmarks/crisis_benchmark.py

# Stress test (275 sentences, 11 categories)
python3 benchmarks/stress_test.py
```

Held-out probes (`benchmarks/holdout_probes.json`, `holdout_v2_probes.json`)
are evaluation-only -- never tune against them; v2 is sealed (its sentences
are never displayed). Evaluate with `benchmarks/eval_holdout.py` /
`eval_holdout_v2.py`. See `benchmarks/HOLDOUT_PROTOCOL.md`.

## Links

- **Live demo**: [huggingface.co/spaces/deucebucket/clanker](https://huggingface.co/spaces/deucebucket/clanker)
- **Browser demo**: [deucebucket.github.io/clanker-demo](https://deucebucket.github.io/clanker-demo)
- **Theory**: [docs/THEORY.md](docs/THEORY.md)
- **Full specification**: [SPEC.md](SPEC.md)

## License

Licensed under the [MIT License](LICENSE).

## Author

Jerry Mares ([deucebucket](https://github.com/deucebucket))

DOI: [10.5281/zenodo.19383636](https://doi.org/10.5281/zenodo.19383636)
