# V7 Physics Spec: New Roles and Interaction Rules

Based on Grok's physics-first response to Council Round 3.

## New Roles

### REGISTER_CASUAL (Solvent)
Words: bruh, bro, dude, fam, lol, lmao, haha, no cap, literally, lowkey, highkey
Properties: dV=0, high SOLVENT_MASS
Proximity rule: PHASE_CHANGE within 3 words
- LIQUID emotional atoms: dV sign flips, magnitude × 1.4
- SOLID emotional atoms: no change (can't dissolve rock)
- Momentum: casual phase carries forward 5 words

### EVALUATIVE_BARE (Context Vacuum)
Trigger: 2-4 word sentence, positive adjective head, no subject/ref/connector
Properties: CONTEXT_VACUUM field covers entire sentence
Effect: inverts all dV, multiplies by 1.8
Condition: total gravity (sum dG) below threshold (molecule too light for literal)

### RARITY_MARKER
Words/phrases: "for the first time", "in weeks/months/years", "today after", "finally"
Properties: dV=0, dU=+3.0 (urgency charge)
Proximity rule: within 6 words of neutral ACTION atom → phase shift to positive
Effect: +80 dV boost, tagged RECOVERY_VICTORY

### SCALE_ANCHOR (Number as Context)
Any numeric word + bonded noun
Properties: carries EXPECTED_RANGE for common nouns
Effect: DEVIATION_FIELD (inverse-square) warps bonded noun's dV
Direction from signed dU vector (too low vs too high)

## New Interaction Rules

### PHASE_CHANGE (Solvent dissolves liquid)
Trigger: REGISTER_CASUAL within 3 of EMOTIONAL
If EMOTIONAL.state == LIQUID: flip dV sign, × 1.4
If EMOTIONAL.state == SOLID: no change
Momentum: casual flag propagates 5 words forward

### HABITUAL_PERSISTENCE (Ghost atom grief)
Trigger: TEMPORAL("still","again","now") + ROUTINE_VERB("set","make","check","keep")
         + no explicit recipient in sentence
Effect: ABSENCE_GAP detected → dV × -1.6, tagged GRIEF_ATMOSPHERE

### GOVERNANCE_RULE (Contradiction resolution)
Trigger: positive + negative EMOTIONAL within 4 words, same momentum chain
HEAD_ATOM = main verb or sentence-final word
HEAD governs: its sign wins, all others × 0.6

### DOUBLE_NEGATION_RESCUE
Trigger: NEGATOR + NEGATIVE_OUTCOME + benefit to OTHER_REF
Effect: two flips = positive, magnitude × 1.5

### REVERSE_COMPRESSION (Understatement detection)
Trigger: COMPRESSOR + high total momentum but low surface dV
Effect: hidden magnitude × 2.2, compressor becomes amplifier

### DIRECTED_THREAT
Trigger: NEGATOR + TEMPORAL(future) + OTHER_REF within 5 words
Effect: negative dV × 2.5 directed at OTHER_REF
