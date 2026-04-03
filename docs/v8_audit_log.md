# V8 Audit Log — Real Data Spot-Check Results

## Date: 2026-04-03
## Engine: V8 (post-council, post-mass-zero)

## Verified Accuracy

| Corpus | Sentences | Accuracy | Top Failure |
|--------|----------|----------|-------------|
| Gutenberg (my check) | 50 | ~64% | Positive inflation on neutral narration |
| Twitch (agent) | 30 | 60% | Slang hype invisible, "no" as negation kills positive |
| Reddit (agent) | 30 | 60% | Sarcasm/anger reads positive, expletive-as-intensifier |
| Philosophy (agent) | 32 | 56% | Abstract/instructional prose inflated positive |
| FO76 (agent) | 25 | 56% | Procedural quest text inflated positive |
| **Combined** | **167** | **~59%** | **Positive inflation dominant** |

## After Mass Zero (646 words zeroed)

Novel distribution shifted from pos=33.5% to pos=26.0% (more balanced).
9/13 changed sentences moved toward correct neutral.
Ground truth held at 41/41.

## Critical Bugs Found

1. **"raping children" read V=153 POSITIVE** — word "raping" was NOT IN VOCABULARY. Fixed: added raping/raped/rape as SOLID negative.
2. **"no we good" read V=17 NEGATIVE** — "no" as discourse marker (= "nah, we're fine") triggers NEGATOR. Engine has no discourse marker handling.
3. **"The intense suffering" read V=160 POSITIVE** — word "suffering" has dV=-30 but surrounded by other words with residual positive charge.

## Physics Problems (NOT vocabulary — need council)

These failures are in the MATH, not the word weights:

### 1. Discourse Markers
"no we good" — "no" is an interjection confirming positive state, not negating it.
"no way thats our family" — "no way" is hype/disbelief, not negation.
The engine treats ALL instances of "no" as NEGATOR. Needs contextual role assignment.

### 2. Expletive-as-Intensifier
"Shit you are right! Thanks for pointing out" V=26 — "shit" as "wow" crashes V.
The engine reads expletives as negative regardless of pragmatic function.
Needs: when expletive is sentence-initial and followed by positive content, treat as AMPLIFIER not EMOTIONAL.

### 3. Literary Complexity Penalty
Long subordinate-clause sentences trigger false structure detections.
"I do not know whether, in this combination of the fresh and vigorous projection of youth..." V=23
The NEGATOR "not" + complex phrasing = false SARCASM/SOCIAL_NULLITY.
Needs: sentence length/complexity should DAMPEN structure detection confidence.

### 4. Negated Positive ≠ Negative Enough
"the script isn't very good" V=149 (positive!) — the "isn't" should flip "good" but doesn't override.
Current NEGATOR proximity modifier is -2.464 but "good" at dV=50 * FORCE_SCALE=1.405 = 70 points.
After negation: still positive because the raw force is so strong.
Needs: NEGATOR + positive EMOTIONAL should FULLY invert, not just dampen.

### 5. Register Detection
Expository/instructional prose ("Please head to the Intelligence Center") reads positive.
The engine has no concept of REGISTER — formal/instructional vs conversational vs literary.
Needs: sentence-level register detection that dampens forces in non-emotional registers.

### 6. Counterfactual/Past Tense
"I trusted him with everything" V=129 — past tense "trusted" implies trust is BROKEN.
"we were supposed to grow old together" V=206 — counterfactual positive = grief.
The engine reads present-tense meaning regardless of tense markers.

## Vocabulary Stats After Mass Zero
- Total: 4,473 words
- 646 GAS atoms zeroed (residual positive from V1)
- 28 corrections (restored wrongly-zeroed + added SOLID words)
- Distribution on novels: neg=26.1% neut=47.9% pos=26.0%

## Next Steps for Council Round 5
1. Discourse marker physics (not vocabulary)
2. Register detection (expository dampening)
3. Tense-aware force application
4. Sentence complexity → structure confidence dampening
5. Full NEGATOR inversion on positive emotional words
