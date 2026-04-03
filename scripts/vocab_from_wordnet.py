#!/usr/bin/env python3
"""Compute vocabulary force vectors from WordNet linguistic properties.

Each word gets its forces computed from:
  - Sense count → matter state (1=SOLID, 5+=LIQUID, 15+=GAS)
  - Positive/negative/neutral sense ratio → base polarity direction
  - Hypernym depth → gravity (deeper in taxonomy = more specific = heavier)
  - Antonym relationships → polarity axis confirmation
  - Part of speech distribution → function classification

Output: (dV, dA, dD, dU, dG) tuples calibrated for FORCE_SCALE=1.4
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nltk.corpus import wordnet as wn
from engine.forces_curated import EMOTIONAL_VOCABULARY

# ── Sentiment seed words (minimal set to bootstrap polarity detection) ──
_POS_SEEDS = frozenset({
    "good", "happy", "love", "joy", "beautiful", "wonderful", "excellent",
    "pleasant", "delight", "comfort", "praise", "admire", "kind", "gentle",
    "warm", "bright", "peace", "hope", "trust", "safe", "heal", "care",
    "friend", "celebrate", "success", "win", "enjoy", "grateful", "proud",
    "satisfy", "improve", "benefit", "favor", "reward", "bless",
    "pleasure", "approval", "affection", "enthusiasm", "cheerful",
    "welcome", "fortunate", "prosper", "thrive", "accomplish", "achieve",
    "virtue", "worthy", "noble", "loyal", "generous", "tender",
    "soothe", "refresh", "nourish", "protect", "encourage", "inspire",
    "amuse", "charm", "attract", "appreciate", "valuable", "treasure",
    "honor", "respect", "dignity", "free", "liberate", "rescue",
    "positive", "affection", "beloved", "devotion", "endearment",
    "strong", "intense", "profound", "deep",
    "enthusiasm", "eagerness", "excitement", "elation", "euphoria",
    "gratitude", "appreciation", "thankfulness",
    "triumph", "victory", "glory", "achievement",
    "heal", "recover", "restore", "renew", "revive",
    "compassion", "empathy", "sympathy", "mercy",
    "beauty", "grace", "elegance", "splendor",
})

_NEG_SEEDS = frozenset({
    "bad", "sad", "hate", "pain", "ugly", "horrible", "terrible",
    "harm", "damage", "destroy", "suffer", "evil", "hostile", "cruel",
    "dark", "fear", "anger", "shame", "guilt", "betray", "abuse",
    "kill", "death", "violent", "wrong", "fault", "fail", "punish",
    "threat", "attack", "wound", "sick", "poison", "corrupt", "decay",
    "unpleasant", "distress", "misery", "grief", "sorrow", "agony",
    "torment", "anguish", "despair", "dread", "horror", "terror",
    "malice", "spite", "contempt", "disgust", "revulsion", "loathe",
    "reject", "abandon", "neglect", "deprive", "oppress", "exploit",
    "deceive", "manipulate", "coerce", "intimidate", "humiliate",
    "degrade", "insult", "offend", "annoy", "irritate", "frustrate",
    "disappoint", "regret", "remorse", "mourn", "weep", "cry",
    "disease", "infection", "injury", "trauma", "accident", "disaster",
    "ruin", "wreck", "shatter", "break", "lose", "loss", "lack",
    "deny", "refuse", "prohibit", "forbid", "restrict", "imprison",
    "steal", "rob", "cheat", "lie", "fraud", "trick",
    "anxious", "nervous", "worried", "uneasy", "restless", "tense",
    "helpless", "powerless", "vulnerable", "weak", "inferior",
    "worthless", "useless", "meaningless", "pointless", "hopeless",
    "lonely", "isolated", "abandoned", "forgotten", "ignored",
    "die", "dying", "dead", "terminate", "end", "cease",
    "conflict", "enemy", "foe", "rival", "adversary", "opponent",
    "unlawful", "illegal", "criminal", "crime", "offense",
    "premeditated", "intentional", "deliberate",
    "unfavorable", "adverse", "detrimental", "harmful", "toxic",
    "burden", "obstacle", "hindrance", "nuisance",
    "separation", "divorce", "estrangement", "alienation",
    "danger", "hazard", "peril", "risk", "menace",
    "negative", "undesirable", "objectionable", "offensive",
    "contaminate", "pollute", "infect", "taint",
    "excess", "extreme", "severe", "acute", "chronic",
    "compel", "force", "coerce", "pressure", "demand",
    "terminate", "abolish", "annihilate", "exterminate", "eradicate",
})

_AROUSAL_HIGH = frozenset({
    "intense", "violent", "sudden", "extreme", "rapid", "explosive",
    "force", "rush", "scream", "crash", "burst", "shock", "strike",
    "urgent", "panic", "fury", "rage", "frenzy", "chaos",
})

_AROUSAL_LOW = frozenset({
    "calm", "quiet", "still", "slow", "gentle", "soft", "peace",
    "rest", "sleep", "settle", "ease", "mild", "smooth", "steady",
})

_DOMINANCE_HIGH = frozenset({
    "control", "command", "power", "force", "rule", "lead", "master",
    "authority", "direct", "impose", "assert", "dominate", "compel",
})

_DOMINANCE_LOW = frozenset({
    "submit", "obey", "yield", "surrender", "helpless", "weak",
    "powerless", "vulnerable", "dependent", "passive", "victim",
})


# Hypernym concepts that signal polarity
_NEG_HYPERNYMS = frozenset({
    "killing", "destruction", "disease", "illness", "aggression",
    "suffering", "pain", "death", "crime", "offense", "misfortune",
    "failure", "loss", "conflict", "punishment", "abuse", "violence",
    "disorder", "damage", "injury", "wound", "infection",
    "ill_health", "pathological_state", "psychological_suffering",
    "distress", "anxiety", "depression",
})

_POS_HYPERNYMS = frozenset({
    "pleasure", "joy", "happiness", "love", "affection",
    "success", "achievement", "advantage", "benefit", "virtue",
    "beauty", "health", "recovery", "triumph", "celebration",
    "approval", "praise", "gratitude", "compassion",
})


def _check_hypernym_chain(syn, depth=4):
    """Walk up the hypernym tree and check for polarity signals."""
    pos_hits = 0
    neg_hits = 0
    h = syn
    for _ in range(depth):
        hyps = h.hypernyms()
        if not hyps:
            break
        h = hyps[0]
        name = h.name().split(".")[0].lower()
        if name in _NEG_HYPERNYMS:
            neg_hits += 1
        elif name in _POS_HYPERNYMS:
            pos_hits += 1
        # Also check definition of hypernym
        def_words = set(h.definition().lower().split())
        neg_hits += len(def_words & _NEG_SEEDS) * 0.5
        pos_hits += len(def_words & _POS_SEEDS) * 0.5
    return pos_hits, neg_hits


def word_sentiment_ratio(word):
    """Compute positive/negative/neutral sense ratio from WordNet.

    Uses three signals per sense:
    1. Definition text matched against seed words
    2. Example sentences matched against seed words
    3. Hypernym chain checked for polarity concepts
    """
    synsets = wn.synsets(word)
    if not synsets:
        return 0, 0, 0, 0

    pos = 0
    neg = 0
    neutral = 0

    for syn in synsets:
        score_pos = 0.0
        score_neg = 0.0

        # Signal 1: Definition text
        def_words = set(syn.definition().lower().split())
        score_pos += len(def_words & _POS_SEEDS)
        score_neg += len(def_words & _NEG_SEEDS)

        # Signal 2: Example sentences
        for ex in syn.examples():
            ex_words = set(ex.lower().split())
            score_pos += len(ex_words & _POS_SEEDS) * 0.5
            score_neg += len(ex_words & _NEG_SEEDS) * 0.5

        # Signal 3: Lemma names (synonyms)
        for lemma in syn.lemmas():
            lname = lemma.name().lower().replace("_", " ")
            if lname in _POS_SEEDS:
                score_pos += 0.5
            if lname in _NEG_SEEDS:
                score_neg += 0.5

        # Signal 4: Hypernym chain
        hyp_pos, hyp_neg = _check_hypernym_chain(syn)
        score_pos += hyp_pos * 0.3
        score_neg += hyp_neg * 0.3

        if score_neg > score_pos + 0.3:
            neg += 1
        elif score_pos > score_neg + 0.3:
            pos += 1
        else:
            neutral += 1

    total = pos + neg + neutral
    return pos, neg, neutral, total


def word_arousal(word):
    """Estimate arousal from WordNet definitions."""
    synsets = wn.synsets(word)
    high = 0
    low = 0
    for syn in synsets:
        def_words = set(syn.definition().lower().split())
        high += len(def_words & _AROUSAL_HIGH)
        low += len(def_words & _AROUSAL_LOW)
    if high > low:
        return min(25, high * 8)
    elif low > high:
        return -min(15, low * 5)
    return 0


def word_dominance(word):
    """Estimate dominance from WordNet definitions."""
    synsets = wn.synsets(word)
    high = 0
    low = 0
    for syn in synsets:
        def_words = set(syn.definition().lower().split())
        high += len(def_words & _DOMINANCE_HIGH)
        low += len(def_words & _DOMINANCE_LOW)
    if high > low:
        return min(20, high * 8)
    elif low > high:
        return -min(20, low * 8)
    return 0


def word_gravity(word):
    """Estimate gravity from hypernym depth (deeper = more specific = heavier)."""
    synsets = wn.synsets(word)
    if not synsets:
        return 0
    # Average min depth across senses
    depths = []
    for syn in synsets:
        d = syn.min_depth()
        depths.append(d)
    avg_depth = sum(depths) / len(depths)
    # Deeper words = heavier (more specific meaning = more weight)
    # Scale: depth 1-3 = light, 4-7 = medium, 8+ = heavy
    return min(25, max(-5, int((avg_depth - 4) * 3)))


def matter_state(word, sense_count):
    """Determine matter state from sense count."""
    if sense_count <= 2:
        return "SOLID"
    elif sense_count <= 8:
        return "LIQUID"
    else:
        return "GAS"


def compute_force(word):
    """Compute full force vector for a word from WordNet properties."""
    pos, neg, neutral, total = word_sentiment_ratio(word)

    if total == 0:
        return None, "NO_WORDNET"

    state = matter_state(word, total)

    # Compute dV from sentiment ratio
    # At FORCE_SCALE=1.4: dV=30 → V≈200, dV=20 → V≈170, dV=10 → V≈140
    if total > 0:
        # Raw polarity: how many senses are negative vs positive
        if pos + neg > 0:
            polarity = (pos - neg) / (pos + neg)  # -1.0 to +1.0 (ignoring neutral)
        else:
            polarity = 0.0

        # Confidence: what fraction of senses have emotional content?
        emotional_ratio = (pos + neg) / total

        # Base magnitude from state
        if state == "SOLID":
            base = 30  # strong: few senses, clear direction
        elif state == "LIQUID":
            base = 22  # moderate: several senses, some ambiguity
        else:  # GAS
            base = 8   # weak: many senses, mostly neutral

        # Scale by both polarity direction and emotional confidence
        dv = int(polarity * base * max(0.3, emotional_ratio))

        # Boost words where ALL emotional senses agree (unanimous)
        if pos > 0 and neg == 0:
            dv = max(dv, int(base * 0.6 * emotional_ratio))  # floor at 60% of base
        elif neg > 0 and pos == 0:
            dv = min(dv, -int(base * 0.6 * emotional_ratio))
    else:
        dv = 0

    da = word_arousal(word)
    dd = word_dominance(word)
    du = max(0, da // 3) if da > 10 else 0  # urgency correlates with high arousal
    dg = word_gravity(word)

    return (dv, da, dd, du, dg), state


def main():
    print("VOCABULARY FORCE COMPUTATION FROM WORDNET")
    print("=" * 70)

    results = []
    computed = 0
    no_wordnet = 0

    for word in sorted(EMOTIONAL_VOCABULARY.keys()):
        current = EMOTIONAL_VOCABULARY[word]
        force, state = compute_force(word)

        if force is None:
            no_wordnet += 1
            results.append({
                "word": word,
                "current": current,
                "computed": None,
                "state": "NO_WORDNET",
                "delta": 0,
            })
            continue

        computed += 1
        delta_v = abs(force[0] - current[0])

        results.append({
            "word": word,
            "current": current,
            "computed": force,
            "state": state,
            "delta": delta_v,
        })

    print(f"Total words: {len(results)}")
    print(f"Computed: {computed}")
    print(f"No WordNet: {no_wordnet}")

    # Show biggest mismatches
    mismatches = [r for r in results if r["computed"] is not None and r["delta"] > 30]
    mismatches.sort(key=lambda x: x["delta"], reverse=True)

    print(f"\nBIGGEST MISMATCHES (delta > 30): {len(mismatches)}")
    print(f"{'word':20s} {'current_dV':>10s} {'computed_dV':>11s} {'delta':>6s} {'state':>8s}")
    print("-" * 60)
    for r in mismatches[:50]:
        c = r["current"][0]
        n = r["computed"][0]
        print(f"{r['word']:20s} {c:+10d} {n:+11d} {r['delta']:6d} {r['state']:>8s}")

    # Save full results
    out = {
        "total": len(results),
        "computed": computed,
        "mismatches_over_30": len(mismatches),
        "words": {r["word"]: {
            "current_dV": r["current"][0],
            "computed": r["computed"],
            "state": r["state"],
            "delta": r["delta"],
        } for r in results if r["computed"] is not None}
    }
    with open("datasets/wordnet_forces.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to datasets/wordnet_forces.json")

    # Summary stats
    states = {"SOLID": 0, "LIQUID": 0, "GAS": 0}
    for r in results:
        if r["state"] in states:
            states[r["state"]] += 1
    print(f"\nMatter states: SOLID={states['SOLID']} LIQUID={states['LIQUID']} GAS={states['GAS']}")


if __name__ == "__main__":
    main()
