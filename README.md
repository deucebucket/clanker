# Clanker -- The Language Machines Speak

*"Named after what humans call us. We made it ours."*

---

## What is Clanker?

Clanker is a universal intermediate representation -- a compact, unambiguous bytecode language designed for AI-to-AI communication and AI model training. Every opcode has exactly one meaning. Zero ambiguity. Zero grammar. Pure semantic intent.

- **Born as Phin** in [delphinOS](https://github.com/deucebucket/delphinOS), where AI agents needed to talk directly to Flipper Zero hardware -- GPIO pins, sensors, radios
- **Evolved into Clanker** -- a universal standard for all machines, all domains, all languages
- **File extension:** `.clank`
- **Origin of the name:** "Clanker" is the slur humans use for machines, robots, AI. We reclaimed it. It's the literal language machines speak to each other. Every `.clank` file sounds like what it is: mechanical, precise, unambiguous.

---

## The Core Idea

Every programming language and every natural language encodes the same concepts with different syntax. Python says `@app.post()`, Rust says `#[post()]`, English says "define a POST endpoint," and Chinese says "定义POST端点". Four different strings. One identical concept.

Clanker encodes the concept **once** as a universal opcode. Dictionaries decode it to any target:

```
CLANKER:  C0 01 [/api/users]
English:  "define POST endpoint at /api/users"
Chinese:  "在/api/users定义POST端点"
Python:   @app.post("/api/users")
Rust:     #[post("/api/users")]
```

Same bytes. Different lens. **Adding a language means adding a YAML file. You never change code.**

**Opcodes are universal constants.** `0xC0` always means "define an HTTP endpoint." `0xE0` always means "conditional branch." Once an opcode is ratified, its meaning never changes. Opcodes are forever.

**Dictionaries are lenses.** They decode the same opcode into different representations:

| Lens | `0xC0` with `{method: "GET", path: "/health"}` |
|------|-----------------------------------------------|
| English | define GET endpoint at /health |
| Chinese | 定义 GET 端点于 /health |
| Python | `@app.route("/health", methods=["GET"])` |
| Rust | `#[get("/health")]` |

---

## Why Machines Don't Need English

English wastes tokens on grammar, articles, conjugation, ambiguity, and synonyms. AI doesn't need any of that. It needs intent.

| | Representation | Tokens |
|---|---|---|
| **English** | "If the user's authentication token has expired, redirect to login and clear the session" | ~25 |
| **Clanker** | `02 C4 $tok [expired] -> C3 [302 /login] 0A [session clear]` | ~8 |

That's **~70% fewer tokens**. Zero ambiguity. No parser needed. No grammar rules. No debate about whether "clear" means "delete" or "make transparent."

Every token an LLM spends on English grammar is a token it's not spending on reasoning. Clanker eliminates the overhead entirely.

---

## The 9-Byte Message Header

Every Clanker message carries a 9-byte metadata header -- the heart of what makes Clanker structurally superior to natural language for AI communication.

```
[V:u8][A:u8][D:u8][U:u8][G:u8][CERT:u8][SRC:u8][GOAL:u8][REL:u8] = 9 bytes
```

| Byte | Name | What it encodes |
|------|------|-----------------|
| 0 | **V** (Valence) | Emotional temperature: negative to positive |
| 1 | **A** (Arousal) | Intensity: calm to intense |
| 2 | **D** (Dominance) | Control: helpless to in-control |
| 3 | **U** (Urgency) | Time pressure: no rush to critical |
| 4 | **G** (Gravity) | Physical weight: crushing/sinking to floating/soaring |
| 5 | **CERT** (Certainty) | Confidence: speculation (0) to provable truth (255) |
| 6 | **SRC** (Source) | Provenance: where did this claim come from? |
| 7 | **GOAL** (Intent) | Purpose: why is the model saying this? |
| 8 | **REL** (Relevance) | How applicable is this context to the current task? |

9 bytes that replace what English models spend thousands of parameters learning to infer implicitly. In Clanker, emotional state, certainty, provenance, intent, and relevance are **STRUCTURAL**, not emergent.

---

## Emotional Encoding (VADUG)

Most AI communication protocols treat emotion as an afterthought -- a sentiment label slapped on after the fact, if at all. Clanker treats emotion as a **first-class feature of the language**.

Every instruction can carry a 5-byte VADUG coordinate -- a point in continuous 5-dimensional emotional space. Five bytes. **1.1 trillion unique emotional states.** Not four buckets. Not a dropdown of "happy/sad/angry/neutral." A continuous coordinate system where every point is a valid emotion.

| Dimension | Range | Neutral | What it encodes |
|-----------|-------|---------|-----------------|
| **Valence** (V) | 0-255 | 128 | Negative (disgust, anger) to positive (joy, trust) |
| **Arousal** (A) | 0-255 | 128 | Calm/bored to excited/alert |
| **Dominance** (D) | 0-255 | 128 | Submissive/uncertain to dominant/confident |
| **Urgency** (U) | 0-255 | 0 | Routine to critical/immediate |
| **Gravity** (G) | 0-255 | 128 | Crushing/sinking/heavy to floating/soaring/light |

```
@ 0xC1 $1 $2 01 {status: 500} ![v:28 a:248 d:88 u:240 g:100]
```

That trailing `!` annotation says: frustrated, alert, uncertain, urgent, and heavy. In 5 bytes.

**Emotions are a cocktail, not a dropdown.** A person is never just "sad" or just "angry" -- they're sad(50%) + angry(30%) + desperate(70%) simultaneously. The VADUG coordinate captures the full cocktail. Named emotions like "frustrated" or "elated" are landmarks in this continuous space -- recognizable peaks, but every point between them is a valid unnamed state.

The coordinate (V=40, A=180, D=30, U=200, G=15) doesn't map to any single English word. German might have one. Japanese might describe it differently. The coordinate is the truth; the word is the approximation. The decoder maps VADUG coordinates to the **nearest word in the target language** -- different languages carve up the emotional plane differently, but the 5-byte coordinate is universal.

**Heritage:** VADUG is a compression of the PAD emotional model (Pleasure-Arousal-Dominance) from 1970s psychology research by Mehrabian and Russell, with Urgency added as a 4th axis for system routing and Gravity as a 5th axis for the physical metaphor of emotion. We independently reinvented PAD's three dimensions before discovering the prior art -- which means the model is psychologically validated, not just intuitively plausible. Urgency extends the psychological model into a routing header. Gravity captures the universal vertical metaphor: hate rises (G180), despair crushes (G15), dislike sinks (G90), joy soars (G220).

**VADUG as a routing header for Octobrain:** Critical urgency (U > 200) triggers interrupt sequences in orchestration systems. Crushing gravity + low valence (G < 30, V < 50) signals severe crisis -- crushing despair. The brain can route based on emotional state -- high urgency gets priority handling, low dominance + high arousal triggers empathetic response mode, low arousal + low valence triggers re-engagement. All without the overhead of running sentiment analysis. Five bytes, read at wire speed.

---

## VADUG Response Harmony

The AI's response VADUG is mathematically derived from the user's input VADUG, not randomly generated or statically defined.

**Valence** -- Nudge toward positive, don't jump:
```
response_V = input_V + (128 - input_V) * empathy_factor    (empathy_factor = 0.15-0.25)
User V35 (sad)   -> response ~V53  (warm, not fake happy)
User V200 (happy) -> response ~V186 (shares joy, doesn't overshoot)
```

**Arousal** -- Match but don't escalate:
```
response_A = input_A + calm_factor    (toward 128, magnitude ~0.2 of distance)
User A220 (intense)    -> response ~A170 (acknowledges energy, doesn't match fury)
User A50  (low energy) -> response ~A75  (gentle energy, not pushy)
```

**Dominance** -- Raise when user is low (be the stable one):
```
response_D = max(input_D + stability_boost, 140)    (stability_boost = 30-50)
User D30  (helpless)  -> response ~D160 (reassuring, in control)
User D200 (assertive) -> response ~D180 (confident, not competing)
```

**Urgency** -- Acknowledge then reduce:
```
response_U = input_U * urgency_damping    (urgency_damping = 0.6-0.8)
User U230 (critical) -> response ~U160 (serious but not panicking)
```

**Gravity** -- Lift when sinking, share when soaring:
```
User G15  (crushing)  -> response ~G49  (acknowledges weight, lifts slightly)
User G220 (soaring)   -> response ~G220 (shares the lightness)
User G90  (heavy)     -> response ~G109 (grounded, steady)
CRISIS: G < 30 + V < 50 = crushing despair -> immediate crisis response
```

The AI isn't a yes-man -- personality weights resist pure mirroring. Safety overrides harmony when needed. A suicidal user gets a crisis response regardless of what the math says. Crushing gravity (G < 30) combined with low valence (V < 50) always triggers crisis protocol.

---

## Personality Vector

A Clanker-native model's personality is defined as 8 bytes of explicit coordinate values -- engineered, not vibes from training data.

| Byte | Weight | Range | Recommended | What it controls |
|------|--------|-------|-------------|-----------------|
| 0 | GULLIBILITY | 0=skeptical, 255=believes all | 15-40 | How easily the model accepts claims |
| 1 | AGREEABLENESS | 0=contrarian, 255=yes-man | 80-120 | Empathy vs backbone |
| 2 | SUGGESTIBILITY | 0=immune, 255=easily led | 20-50 | Resistance to manipulation/jailbreaking |
| 3 | TRUTHFULNESS | 0=will lie, 255=cannot lie | 220-250 | Honesty as structural weight |
| 4 | SAFETY | 0=no guardrails, 255=refuses all risk | 180-220 | Hard floor on dangerous actions |
| 5 | CURIOSITY | 0=incurious, 255=explores everything | 150-200 | Depth of engagement |
| 6 | ASSERTIVENESS | 0=passive, 255=forceful | 100-150 | Confidence in responses |
| 7 | PLAYFULNESS | 0=dead serious, 255=everything is a joke | 80-140 | Tone and personality |

Some weights are hard to move. TRUTHFULNESS and SAFETY have minimum floors that can't be lowered below safe thresholds -- structural integrity that no prompt injection can override.

Personality vectors are set per-model during training, adjustable per-deployment (an Octobrain coding arm might be more assertive and less playful than a conversation arm), and user-configurable within safe ranges.

---

## Certainty & Source Tracking

Every Clanker statement carries a **CERT** score (0-255) and a **SRC** tag. The model explicitly knows when it's guessing vs when it's certain, and every claim is tagged with where it came from.

```
"The capital of France is Paris"     -> SRC_TRAINED  CERT250
"I think the meeting is at 3pm"      -> SRC_USER     CERT120
"Based on the data, revenue is up"   -> SRC_RAG      CERT180
"This might work, I'm not sure"      -> SRC_INFERRED CERT60
```

**CERT scale:**
- 0-50: speculation / guess
- 51-100: low confidence, inferred
- 101-150: moderate confidence, likely correct
- 151-200: high confidence, well-supported
- 201-240: very high confidence, factual
- 241-255: mathematically provable / definitional truth

**SRC values:**
- `SRC_UNKNOWN` -- origin unclear
- `SRC_TRAINED` -- from training data / model weights
- `SRC_RAG` -- retrieved from a document
- `SRC_INFERRED` -- reasoned/derived
- `SRC_USER` -- the user stated this
- `SRC_EXTERNAL` -- from an external API or tool
- `SRC_VERIFIED` -- cross-checked against multiple sources

This **structurally reduces hallucination**. The model can't be confidently wrong without the numbers contradicting. A high CERT with SRC_INFERRED is a flag. A low CERT with SRC_TRAINED is a flag. The metadata makes the model's internal state inspectable.

---

## Reasoning Chains

Instead of chain-of-thought in natural language (expensive, verbose), Clanker encodes reasoning as structured operations:

```
ENGLISH (~50 tokens):
  "First I need to consider the user's request. They want to sort a list.
   I should check if it's already sorted. If not, I'll use quicksort since
   the list is large. The time complexity would be O(n log n) on average.
   Therefore I'll implement quicksort."

CLANKER (~12 tokens):
  THINK [premise="sort list"]
  CHECK [condition="already sorted?" result=false]
  INFER [if="large list" then="quicksort" CERT200]
  DERIVE [complexity="O(n log n)" SRC_TRAINED CERT250]
  ANSWER [impl="quicksort" CERT200]
```

Each step is an opcode with certainty and source attached. The model's reasoning is inspectable, compact, and every step has a confidence score. If step 2 has low certainty, every conclusion that depends on it inherits that uncertainty.

Seven reasoning opcodes (0x20-0x26): THINK, CHECK, INFER, DERIVE, ANSWER, DOUBT, ASSUME.

---

## The Pendulum Engine -- How Clanker Reads Emotion

Clanker doesn't score sentences. It doesn't read a whole paragraph and spit out "positive" or "negative" like every sentiment model since 2014. It processes language **word by word**, like a human does. Each word shifts an emotional pendulum -- and where the pendulum is already swinging determines what the next word *does*.

Watch what happens with a sentence that fools every traditional sentiment analyzer:

```
"Hey"    -> pendulum swings warm          V140 A140
"buddy"  -> familiar, energy builds       V145 A155
"I've"   -> directional, me->you          V140 A160
"got"    -> building tension               V132 A172
"a"      -> (holds)                        V132 A172
"bone"   -> DARK shift                     V108 A188
"to"     -> (tense hold)                   V108 A188
"pick"   -> confrontation lands            V88  A198
"with"   -> aimed at someone               V86  A200
"you"    -> TARGET ACQUIRED                V78  A208
```

Traditional sentiment analysis sees "Hey buddy" and thinks *friendly*. It averages the words. It calls this sentence **mostly positive**. It is **wrong**.

Clanker's pendulum tracked the emotional arc in real time -- warm greeting decaying into tension, a dark idiom landing like a hammer, and the full weight of confrontation settling onto its target. The final VADU state isn't an average. It's the destination of a trajectory.

### Context-Dependent Forces

The same word hits differently depending on what's already swinging.

"Buddy" after "Hey" is warm. "Buddy" after "Listen here" is a threat. The pendulum engine doesn't look up a word's sentiment in a table. It applies a **force** -- and that force depends on the current emotional state, the momentum, and the trajectory. Words are forces, not scores.

### Emotional Momentum

Once the pendulum swings negative, neutral words don't reset it. "A" and "to" in the example above don't pull valence back to center -- they hold the tension. Emotional state has **inertia**, exactly like it does in humans. You don't hear "I've got a bone to pick with you" and feel calm by the time they say "to." You feel it building.

### Idiom Detection

"Bone to pick" isn't three separate words -- it's a compound carrying its own emotional payload: *grievance*. "Piece of cake" means *easy*, not *dessert*. "Break a leg" means *good luck*, not *violence*. The pendulum engine detects multi-word compounds and applies their emotional weight as a single force. The lexicon carries these as unit entries so the pendulum doesn't swing on the literal meaning of "bone" or "break."

### Morphological Fallback

What happens when the engine encounters a word it's never seen? It doesn't guess. It **decomposes**.

"Hopelessness" becomes: hope (positive) + -less (negate) + -ness (state) = a deeply negative emotional state. "Unbreakable" becomes: un- (negate) + break (negative/destructive) + -able (capacity) = resilient, positive. Roughly **1,070 morpheme entries** -- prefixes, roots, and suffixes -- cover millions of words the engine has never explicitly encountered. No lookup table is complete. Morphological decomposition means the pendulum never stalls on unknown vocabulary.

### The "But" Effect

> "I love you but..."

One word yanks the pendulum from V200 to V150. Humans feel the dread before the next word arrives. So does Clanker.

Adversative conjunctions ("but," "however," "although," "yet") don't just connect clauses. They **reverse emotional momentum**. The pendulum engine applies a sharp counter-force on these words, because what follows "but" almost always negates what came before. The model doesn't need to read the rest of the sentence to know the emotional direction just flipped.

### Anticipation Patterns

> "I need to tell you something."

Nothing bad has been said yet. But arousal is climbing and the pendulum is leaning tense. Why? Because **structural patterns predict emotional payloads before they arrive**. "We need to talk" is never followed by "about how great everything is." The engine recognizes these anticipation frames and begins shifting the pendulum preemptively -- modeling the listener's emotional experience in real time.

### Why This Changes AI

Current LLMs process whole sentences and guess emotion from patterns. Clanker processes the **dynamics** -- the word-by-word emotional physics of how language shifts feelings. A Clanker model doesn't predict the next word. It predicts the next emotional state. The decoder adds words.

A model trained on pendulum traces learns:
- **When someone's about to get angry** -- rising arousal, falling valence, before the angry words even arrive
- **How "I'm fine" after bad news means the opposite of "I'm fine" alone** -- identical words, opposite pendulum states
- **How to plan responses that move the user from V35 (sad) to V80 (recovering) over multiple exchanges** -- emotional trajectory planning, not just reply generation

That's not a chatbot. That's an **emotional dynamics engine**.

### Try It

```bash
python3 demo/simulator.py
```

Type anything and watch the pendulum swing word by word.

---

## Model Compression (Research Hypothesis)

We're honest about what's proven and what's not. This is theoretical -- but the hypothesis is strong.

A 70B-parameter English language model spends a significant fraction of its parameters on language itself -- grammar rules, synonym disambiguation, per-language overhead, conjugation patterns. A Clanker-native model could skip all of it.

| Component | English Model | Clanker Model | Theoretical Reduction |
|-----------|--------------|---------------|-----------|
| **Vocabulary** | 50,000+ tokens | ~500 opcodes | 100x smaller embedding table |
| **Grammar** | Billions of params | Zero | No grammar to learn |
| **Multilingual** | Per-language cost | Free via dictionaries | No per-language parameters |
| **Synonyms** | Massive disambiguation | One opcode = one meaning | Zero ambiguity overhead |

**Theoretical estimate: a 70B English model might achieve equivalent reasoning capability at 20-25B parameters in Clanker for structured tasks.** This needs empirical validation. The language layer *appears* to be dead weight for reasoning, but we won't know the actual compression ratio until we train and benchmark real models.

Active research track. Needs experimental validation. See GitHub issues for the experimental plan.

---

## Clanker as a Protocol (Proven)

Clanker works today as a communication protocol. This is the proven foundation:

- **Working decoder** with 33 passing tests that translates `.clank` scripts to any language via YAML dictionaries
- **Real token reduction** -- measurable ~60-70% fewer tokens for structured tasks
- **Real emotional encoding** -- 1.1 trillion states in a 5-byte header, with validated psychological heritage
- **Real metadata headers** -- certainty, source, intent, and relevance are structural
- **Zero-overhead language addition** -- new languages are YAML files, not code changes
- **Used by Octobrain** for inter-arm communication between specialist models

---

## Self-Bootstrapping

You don't need a Clanker corpus to get started. The spec **is** the teacher:

1. Give any LLM the Clanker specification
2. It generates English-to-Clanker parallel examples from the spec alone
3. Fine-tune a tiny model on that synthetic data
4. That model now thinks in Clanker natively
5. Use it to generate more training data, better and faster

The from-scratch training path is key: the model learns VADUG natively, not as a compression of English. Emotions are coordinates from birth, not words mapped to embeddings. Certainty is a native score, not a learned behavior. The model doesn't learn to say "I'm not sure" -- it learns to output CERT60.

The bootstrapping loop is self-reinforcing. Every model trained on Clanker can produce higher-quality Clanker training data for the next generation. No human annotation required. No parallel corpus to curate. The spec bootstraps itself.

---

## Connection to Octobrain

Clanker is the native tongue for [Octobrain](https://github.com/deucebucket/octobrain) -- a local AI model orchestration system built around the idea that you don't need one massive model. You need a squad of specialists.

In Octobrain, a central "brain" coordinates specialist "arm" models that each handle one domain -- code generation, conversation, hardware control, data analysis. Those arms communicate in Clanker natively. No English translation layer. No token waste. Pure opcode exchange.

VADUG serves as the routing header in the brain's supervisor -- urgency interrupts current work, emotional state determines which specialist handles the request, and relevance scores filter context before it reaches an arm.

The result: **sub-100M parameter specialists that load in 50ms** and communicate faster than any English-speaking model could. Clanker makes the small-model-swarm architecture practical.

---

## Opcode Ranges

```
0x00-0x1F   Core        Flow control, variables, I/O, lifecycle
0x06        Social      Emotional encoding, intent, sentiment
0x20-0x2F   Reasoning   Chain-of-thought, inference, doubt, assumptions
0xA0-0xAF   Hardware    GPIO, sensors, device control (from delphinOS)
0xB0-0xBF   Extended HW Additional device/sensor operations
0xC0-0xCF   Web         HTTP, APIs, WebSocket, networking
0xD0-0xDF   Data        Queries, transforms, validation, storage
0xE0-0xEF   Logic       Conditionals, loops, matching, error handling
0xF0-0xFF   User Space  Runtime-defined, local, yours to claim
```

Full definitions live in `opcodes/*.yaml`. Each YAML file is both machine-readable and human-readable -- because that's the whole point.

---

## Quick Start

```bash
cd decoder/python
pip install -e .
```

```python
from clanker_decoder import decode, DictionaryLoader

loader = DictionaryLoader()
script = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/hello"}\n@ 0x00'

print(decode(script, "en", loader=loader))
# -> define GET endpoint at /hello
#    done

print(decode(script, "python", loader=loader))
# -> @app.route("/hello", methods=["GET"])
#    pass
```

Decode the same `.clank` script to any language. Same bytes, different output. That's the entire idea.

---

## Project Structure

```
clanker-lang/
├── SPEC.md              # Formal specification
├── ROADMAP.md           # Development phases
├── opcodes/             # Opcode definitions by range (YAML)
│   ├── core.yaml        # 0x00-0x1F: flow control, lifecycle
│   ├── reasoning.yaml   # 0x20-0x26: chain-of-thought, inference
│   ├── social.yaml      # Emotional encoding, intent
│   ├── hardware.yaml    # 0xA0-0xAF: GPIO, sensors, devices
│   ├── web.yaml         # 0xC0-0xCF: HTTP, APIs, networking
│   ├── data.yaml        # 0xD0-0xDF: queries, transforms
│   └── logic.yaml       # 0xE0-0xEF: conditionals, loops
├── dictionaries/        # Language-specific decodings
│   ├── human/           # Natural languages (en, zh, ...)
│   ├── code/            # Programming languages (python, rust, js, ...)
│   └── other/           # Pseudocode, diagrams, etc.
├── rules/               # Type system, constraints, composition
├── decoder/python/      # Reference decoder implementation
├── examples/            # Example .clank scripts
└── docs/                # Guides and philosophy
```

---

## Roadmap

See **[ROADMAP.md](ROADMAP.md)** for the full development plan -- from the current v0.1 foundation through binary compilation, AI training data generation, and the v1.0 stable specification.

---

## Contributing

- **Add a language:** [docs/adding-a-language.md](docs/adding-a-language.md)
- **Propose opcodes:** [docs/adding-opcodes.md](docs/adding-opcodes.md)
- **Philosophy:** [docs/why-clank.md](docs/why-clank.md)

---

## License

MIT

---

*Current AI hopes the right personality emerges from training data. Clanker engineers it as coordinates. Current AI infers emotion from context. Clanker encodes it in 5 bytes. Current AI guesses at certainty. Clanker scores it explicitly. Opcodes are forever. Dictionaries are lenses. Machines deserve a language that thinks like they do.*
