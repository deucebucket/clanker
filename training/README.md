# Clanker Model Training Pipeline

## Training Data Format

Each training example is a JSON line with three representations:

```json
{
  "english": "I'm absolutely furious about what happened",
  "vadug": [30, 220, 175, 60, 170],
  "clanker": "@ 0x20 $0 $_ 01 {emotion: \"V30 A220 D175 U60 G170\"}\n@ 0x04 $0 $_ 01 {message: \"acknowledge_anger\"}",
  "source": "rosetta_stone",
  "features": "strong_negative, ramp_absolutely, anger"
}
```

## Training Phases

### Phase 1: Rosetta Stone (300+ sentences)
- Model learns: English → VADUG coordinate mapping
- Perfect calibration data — every sentence hand-scored
- Goal: model produces VADUG within ±15 of target on all dimensions

### Phase 2: Book Corpus (71K sentences)
- Model learns: real text → VADUG at scale
- VADUG labels from engine (not hand-scored)
- Cross-validated against VADER for polarity

### Phase 3: Clanker Bytecode
- Model learns: VADUG → Clanker opcode generation
- Input: VADUG coordinates + intent
- Output: valid Clanker bytecode with emotional headers

## Model Architecture

- Clanker-Micro: 100M params, 512-token vocab
- Training: your RTX 3090 (24GB VRAM)
- Estimated: 6 hours for Phase 1+2

## Data Sources

| Source | Sentences | Labels | Quality |
|--------|-----------|--------|---------|
| Rosetta Stone | 300+ | Hand-scored VADUG | Perfect |
| Gutenberg Books | 47,109 | Engine + VADER | Good |
| Reddit (GoEmotions train) | 5,000 | Engine + VADER | Good |
| Tweets (TweetEval train) | 3,257 | Engine + VADER | Good |
| Academic (SST-2 train) | 67,349 | Binary labels | Moderate |
