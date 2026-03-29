# Emotional Chunking Architecture

## The Problem
Long text inputs dilute emotional signal in single-pass pendulum processing. A 70-word paragraph of despair resolves near-neutral because emotional words are spread across many neutral bridge words.

## The Solution: Emotional Chunking
Split input into emotional BEATS at natural boundaries. Each beat gets its own pendulum run and VADUG score. Responses are generated per-beat, then assembled by an arc-aware synthesizer.

## Chunk Detection
Split at:
- Sentence endings: `.` `!` `?`
- Conjunctive reversals: "but", "however", "although", "yet", "though"
- Causal links: "because", "since", "so", "therefore"
- Temporal shifts: "then", "after", "before", "meanwhile", "now"
- Additive: "and" (only when followed by subject change)

## Per-Chunk Processing
Each chunk runs through:
1. Its own SequentialPendulum instance (fresh start)
2. Produces its own VADUG coordinates
3. Gets its own nearest emotion label
4. Gets its own response via harmony formula

## Arc Analysis
After all chunks are scored, analyze the trajectory:
- **Descending**: each chunk more negative than last → things getting worse
- **Ascending**: each chunk more positive than last → things getting better
- **Valley**: negative → positive → "bittersweet" or "recovery"
- **Peak**: positive → negative → "bad news" or "decline"
- **Flat negative**: sustained negative → persistent problem
- **Flat positive**: sustained positive → sustained joy
- **Oscillating**: mixed → complex situation, address each beat

## Response Assembly
1. Generate response per emotional chunk via harmony
2. Add transitions between chunk responses ("but", "and", "though")
3. Add arc-aware closer:
   - Valley arc → encouraging finish ("congrats!", "that's amazing")
   - Peak arc → empathetic finish ("I'm here if you need me")
   - Descending → supportive finish ("we'll figure this out")
   - Ascending → celebratory finish ("you deserve this")
   - Flat negative → solidarity finish ("you're not alone in this")

## Three-Tier Precision
- **Computation** (internal): f32 per axis, 20 bytes, used during pendulum math
- **Transmission** (wire): u16 per axis, 10 bytes, model-to-model communication
- **Compact** (constrained): u8 per axis, 5 bytes, for tiny devices/arms

Per-chunk VADUG uses u8 (each chunk is short enough for 0-255 precision).
Arc-level metadata (overall trajectory) uses f32 for smooth interpolation.

## Example

Input: "I am sad, because I'm leaving my job, I'm going to miss everyone. But my new job is my dream job, so I'm also happy."

```
Chunk 1: "I am sad"                    → V60  A100 D80  U10 G60  [sad, sinking]
Chunk 2: "because I'm leaving my job"  → V70  A120 D90  U30 G70  [loss, transition]
Chunk 3: "I'm going to miss everyone"  → V55  A110 D70  U15 G50  [grief, social loss]
Chunk 4: "But my new job is my dream"  → V200 A180 D170 U10 G210 [excitement, soaring]
Chunk 5: "so I'm also happy"           → V180 A150 D150 U5  G190 [settled joy]

Arc: VALLEY (descent → reversal → ascent)
Pattern: bittersweet transition

Response:
  Chunk 1+2: "That's a big transition."
  Chunk 3:   "I bet they're going to miss you too."
  Chunk 4:   "But a dream job? That's incredible."
  Chunk 5:   [ARC CLOSER - valley → encouraging]
             "How many people get to work their dream? Congrats."
```
