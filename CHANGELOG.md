# Changelog

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

