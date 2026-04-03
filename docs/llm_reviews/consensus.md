# LLM Consensus: Engine Physics Upgrade (2026-04-03)

## Reviewed by: GPT-4, Claude, Gemini, Grok

## UNANIMOUS (4/4 agree):
1. Replace fixed momentum with adaptive (per-dimension, per-pattern)
2. Sarcasm = contradiction detection (surface polarity vs context polarity)
3. Atmospheric grief = object permanence + absence
4. History/context gates ambiguous sentence interpretation
5. Self-worth affects interpretation bidirectionally

## STRONG CONSENSUS (3/4):
6. Accumulate sentence evidence first, THEN blend to state (GPT-4, Claude, Gemini)
7. Friction/tension memory across turns (GPT-4, Grok, Claude)
8. Negativity bias: negative states persist longer (Claude, Grok, Gemini)
9. Brevity as sarcasm signal (GPT-4, Claude, Grok)

## KEY EQUATIONS:

### Sarcasm (GPT-4):
S_sarc = σ(w1*F_mock_praise + w2*F_dismissive_assent + w3*F_permission_hostility + w4*C_surface_context)

### Atmospheric Grief (all 4):
G_total = 1 - (1-G_explicit)(1-η*G_atmospheric)
G_atmospheric = possessive_object * persistence_marker * absent_agent

### Adaptive Momentum (Claude):
M_effective = M_base + (A_current - CENTER) * M_AROUSAL_SCALE
Negative direction: M *= 1.15 (stickier)
Positive direction: M *= 0.90 (easier to leave)

### Sentence Evidence (GPT-4):
E_t = Σ f_i + Σ g_ij + Σ h_p
target_t = CENTER + Φ(E_t) where Φ(E) = s * tanh(E/s)

### Friction Memory (GPT-4):
F_t = ρ*F_{t-1} + κ1*A_t + κ2*D_neg + κ3*S_sarc + κ4*P_passive

### Context Gate (GPT-4):
λ_t = C_literal / (C_literal + C_ambiguous + C_contextual)
score = λ*score_sentence + (1-λ)*score_context

## IMPLEMENTATION PRIORITY:
1. Contradiction sarcasm (replaces pattern-based)
2. Atmospheric grief (residue detection)
3. Sentence-level accumulation before blend
4. Adaptive vector momentum
5. Friction memory
6. Context gate for ambiguous sentences
7. tanh saturation to prevent clipping
