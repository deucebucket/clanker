# Changelog

## V9 (2026-04-15 -- present) -- Experimental Structural Decomposition
- Structural pendulum: decomposes sentence into equation (SUBJECT+EVENT+CONTEXT) before force accumulation
- Lemma root system: collapses 4,544 words to ~500 emotional roots
- Molecular bonding layer: 8 bond types, reaction table, two-layer molecules
- Star-to-star gravity: heavier emotional atoms pull lighter ones
- Phase-aware alloy composition: LIQUID words default to presumed charge
- Ported all V8 systems: structures, force flow, zones, crisis, anomaly, solver, battleship
- W→V coupling: low self-worth amplifies negatives, suppresses positives
- Absence scope + forced choice cancellation in interpret_context
- Genetic optimizer: champion knobs tuned across 50+ generations
- Perspective dampening: OTHER_REF emotions dampened unless directed at SELF
- 4-model council consensus (Claude, Gemini, GPT, Grok) on 500+ sentences
- Confidence scoring: certain/majority/ambiguous truth split
- Status: experimental, ~10pts behind V8 on stress tests

## V8.3 (2026-04-10)
- Conversational accuracy: 73.1% → 98.5% on conversational sentences
- Crisis recall: 80.4%
- 3,646 verified sentences accumulated

## V8.2 (2026-04-08) -- Council Consensus
- Full 4-LLM council consensus applied
- Complexity dampening for literary register
- Real-text spot-check: ~59% on 167 manually verified sentences across 5 corpora

## V8.1 (2026-04-06) -- Interpret Context Layer
- Discourse markers: "actually", "honestly", "I mean" as register signals
- Negator inversion in context
- Register dampening for casual text
- Counterfactual detection

## V8 (2026-04-04) -- Mass Zero + Real-Data Audit
- Mass-zeroed 646 GAS atoms (inflated neutral words neutralized)
- 275-sentence stress test: 73.1% (201/275 across 11 categories)
- 126-sentence crisis benchmark: 70.6% recall, 0% false positive
- Real-data audit: 59% accuracy on 167 sentences from 5 corpora (novels, Twitch, Reddit, philosophy, game dialogue)
- 6 physics problems identified and documented
- Vocabulary: 4,544 curated words (up from 4,108)
- Structural patterns: 45+ (up from 26)
- New patterns: MUNDANE_HYPERBOLE, BOUNDARY_VIOLATION, SELF_ERASURE, DIVESTITURE, METHOD_FIXATION, RARITY_MARKER, ABANDONMENT, LIFE_ACHIEVEMENT
- SOLVENT dissolution: casual register flips LIQUID negative to positive
- Mundane dampening: inert gas absorption of crisis energy
- interpret_context layer: discourse markers, register detection, counterfactual inversion
- Tests: 207 (up from 167)

## V7 (2026-04-02) -- SOLVENT Physics
- SOLVENT word role: REGISTER_CASUAL dissolves LIQUID atoms via phase physics
- Pure physics solutions for context-dependent meaning

## V6 (2026-04-01) -- Physics Upgrade
- Contradiction sarcasm detection
- Atmospheric grief handling
- Adaptive momentum
- Pipeline refactor: pendulum.py split into 8 pluggable stages
- Port V2 anomaly detector: trajectory analysis for conversations
- Pipeline trace module added

## V5.5 (2026-03-31) -- 7D VADUGWI, Force Flow, Phi-4 LoRA, Bayesian Corrections

### Engine
- Full 7D VADUGWI coordinate system: V, A, D, U, G, W (Self-Worth), I (Intent)
- Self-Worth (W) dimension: tracks user self-evaluation thread (shattered -> stable -> strong)
- Intent (I) dimension: withdraw / deflect / neutral / connect / control
- Force flow resolver (`engine/force_flow.py`): WHO does WHAT to WHOM directional analysis
- Absence scope: "havent had X" dampens absent events instead of scoring them positively
- Compound phrase resolution: "no one" -> nobody, "everyone" -> universal scope
- Bayesian vocabulary corrections: over-weighted words identified and neutralized across 11 cycles
- Forced choice cancellation: "A or B" does not double-count both options
- RELIEF_ABSENCE, SELF_EXCLUDED, WITHHELD_POSITIVE structural patterns added
- Confidence gate: NULL / LOW / MODERATE / HIGH output modes
- 26 structural patterns total (up from 22 in V3.2)

### Vocabulary
- 4,108 curated words with 7D force vectors (up from ~2,400 in V3.2)
- Cycle-by-cycle Bayesian neutralization: highway (+29->0), relationship (+37->+10), and 30+ others
- 5-way AI consensus validation on all vocabulary cards (Gemini, Claude Opus, GPT-4, Grok, engine)

### Tests
- 167 tests passing across 8 test files (up from 156)
- Coverage: word classification, structures, proximity, pendulum, solver, battleship, scaffolding, novel sentences

### Performance
- Engine size: ~452KB
- Speed: 0.15ms/sentence, ~6,500-13,000 sentences/sec
- SST-2: 69.6% (up from 51% in V3 era)
- GoEmotions: 75.3%
- 4-AI consensus benchmark: 76.3% (vs Gemini, Claude Opus, GPT-4, Grok on 131 sentences)
- Novel sentences: 100% on 630 sentences
- Crisis detection: 97.3%
- Sarcasm: 90%
- Safe text false positives: 0%

### Training
- Phi-4 LoRA training in progress: 52,642 entries, 10 epochs, teaches VADUGWI math
- Training objective: model learns the force equations, not memorized outputs

---

## V3.2 (2026-03-30) -- Structural Pattern Expansion + SmolLM2 Integration

### New structural patterns (8 added, now 22+ total)
- BETRAYAL: relationship trust weaponized ("wife cheated with best friend")
- BRAVADO: overcompensation mask ("haha yeah im totally okay")
- VICTIMIZATION: directional damage, who did what to whom ("she left me" vs "I left the room")
- CALLING_OUT: complaint disguised as question ("why do you always do that")
- DIRECTED_POSITIVE: positive aimed at other as dismissal ("good for you", "must be nice")
- MINIMIZER: shrinking real impact ("it was just a joke", "youre too sensitive")
- EXCLUDED_POSITIVE: self excluded from positive ("do you even love me", "my parents love my brother more")

### Engine improvements
- Smart CHOPPER: analyzes second-half content before overriding
- POSSESSION words keep gravity but strip emotional force (objects have weight, not feelings)
- Strong negative words resist negation (expletives cannot be logically negated)
- Stemmer fix: -s tried before -es ("bites" -> "bite" not "bit")
- Contractions recognized: youre, hes, shes, theyre as OTHER_REF
- "too" added to AMPLIFIER
- "someone/everybody/anyone" as OTHER_REF
- SUSPICIOUS_CALM strengthened and excludes achievement contexts
- FAREWELL excludes "back" (reclamation is not farewell)

### Vocabulary expansion (2,400+ words)
- Violence: stabbed, punched, slapped, choked, attacked, assaulted
- Mockery: mocked, ridiculed, taunted, harassed
- Resignation: whatever, k, cool, sure, nvm, idc
- Achievement: worked, succeeded, graduated, hired, fired
- Violation: deleted, changed, took, spent, sold, stole, ruined, destroyed
- Invalidation: overreacting, dramatic, crazy, paranoid, delusional
- Threat: swear, warn, threatening
- Medical: herpes, cancer, sick, infected, pregnant
- Judgment: compare, judge, criticize, blame, fault
- Temporal intensity: always, constantly, every, forever
- Exclusion: except, instead, more, prettier, smarter
- Doubt: even, actually, anymore, supposed
- Upbringing: foster, adopted, orphan, abused, neglected, molested
- Milestone: million, verified, published, accepted, hero, dream
- Resolution: made, well, anyway, survived, overcame

### Liquid word fixes (same word, structure determines meaning)
- "left" V=-45 -> V=-8 (agency vs abandonment, VICTIMIZATION resolves)
- "give" V=+20 -> V=-3 (generous vs demanding, context resolves)
- "hit" V=+28 -> V=-45 (violence vs achievement, VICTIMIZATION resolves)
- "hope" V=+127 -> V=+45 (hope contains uncertainty, not opposite of despair)
- "finally" V=+29 -> V=+5 (temporal marker, not positive)
- "calm" V=+39 -> V=+20 (state vs command)
- "today" V=+37 -> V=0 (time marker, not emotional)
- "negative" V=-112 -> V=-25 (medical context = good, emotional = bad)
- "surgery" V=-77 -> V=-25 (past surgery with "well" = relief)
- "care" G=8 -> G=35 (care = embrace, high gravity)
- "foster" V=-20 -> V=-3 (neutralizer/dampener, not negative)
- "fuck" V=-40 -> V=-70 (resists negation)

### Accuracy
- 92% on unambiguous sentences (excluding context-dependent)
- 85% crisis recall (was ~80%)
- 100% on genuine positive (zero false positives)
- 90% on internet speak
- 80% on body language
- 90% on conversation fight patterns

### Model training
- V3 model retrained: 7.7M params, 141K examples, 22 patterns
- Role accuracy: 59.7%, Pattern accuracy: 97.9%, VADUG MAE: 2.8

### SmolLM2 / Llama integration
- Conversation loop: two characters with personalities argue
- Living conversation: endless interaction until breaking point
- 6 personalities: hothead, peacekeeper, ice, empath, joker, narcissist
- LoRA training data: 47K VADUG-conditioned pairs formatted
- HuggingFace Space live with Llama-3.2-1B via Inference API

### Demo
- Two-character browser demo with persistent emotional memory
- 5 selectable characters with distinct appearances (skin, hair, clothes)
- Speech bubbles, conversation log with per-message VADUG scores
- Trauma tracking with time-based decay

## V3.1 (2026-03-30)
- All 2,315 vocabulary words apply force (not just "emotional" role)
- Periodic table classification: 1,291 solids, 970 liquids, 54 gases
- 78K sentence transition map (empirical word-to-word intervals)
- Ripped out DEATH_WISH hardcoded pattern (physics handles it)
- Sarcasm false positive fix (requires opener + mundane, not just positive + anything)
- Pull verb family (chase/pursue/flee/stalk/escape)
- Power verb family (use/control/command vs submission vs inversion)
- Surprise as pattern interrupt (A-spike, not V-direction)
- Shape traces for every sentence (V-line fingerprinting)
- Browser engine at docs/index.html (78KB data + JS, zero server)
- Trained V3 model: role 80.2%, patterns 99.0%, VADUG MAE 3.2

## V3.0 (2026-03-30)
- Complete rewrite from previous idiom matching to structural pattern recognition
- 6-layer architecture: word roles, proximity, structures, physics, solver, battleship
- 91% on novel sentences (previous was 39%)
- 86% crisis detection on never-seen sentences
- 90% sarcasm detection via structural inversion
- 156 engine tests passing
- Clean repo (clean history)

