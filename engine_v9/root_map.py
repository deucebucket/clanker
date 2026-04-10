"""
V9 Word-to-Root Mapper — three-tier lookup.

Tier 1: Static hash (WORD_TO_ROOT dict) — O(1) direct lookup.
Tier 2: Morphological suffix stripping — tries stem in Tier 1.
Tier 3: Fallback to OBJECT_GENERIC (GAS, zero charge).

Usage:
    from engine_v9.root_map import map_to_root
    root = map_to_root("happiness")   # -> Root(name="HAPPY", ...)
"""

from __future__ import annotations
import string
from engine_v9.roots import ROOTS, Root, RootCategory

# ---------------------------------------------------------------------------
# Static word → root-name mapping (lowercase surface forms only)
# ---------------------------------------------------------------------------

WORD_TO_ROOT: dict[str, str] = {

    # ── Self / Other / Relation ──────────────────────────────────────────────
    "i":          "SELF",
    "me":         "SELF",
    "my":         "SELF",
    "myself":     "SELF",
    "im":         "SELF",
    "ive":        "SELF",
    "id":         "SELF",
    "ill":        "SELF",
    "mine":       "SELF",

    "you":        "OTHER",
    "your":       "OTHER",
    "yours":      "OTHER",
    "yourself":   "OTHER",
    "he":         "OTHER",
    "him":        "OTHER",
    "his":        "OTHER",
    "she":        "OTHER",
    "her":        "OTHER",
    "hers":       "OTHER",
    "they":       "OTHER",
    "them":       "OTHER",
    "their":      "OTHER",
    "we":         "OTHER",
    "us":         "OTHER",
    "our":        "OTHER",
    "it":         "OTHER",
    "its":        "OTHER",

    "someone":    "PERSON_GENERIC",
    "somebody":   "PERSON_GENERIC",
    "everyone":   "PERSON_GENERIC",
    "everybody":  "PERSON_GENERIC",
    "anyone":     "PERSON_GENERIC",
    "anybody":    "PERSON_GENERIC",
    "people":     "PERSON_GENERIC",

    "mom":        "RELATION",
    "mother":     "RELATION",
    "dad":        "RELATION",
    "father":     "RELATION",
    "brother":    "RELATION",
    "sister":     "RELATION",
    "friend":     "RELATION",
    "friends":    "RELATION",
    "family":     "RELATION",
    "husband":    "RELATION",
    "wife":       "RELATION",
    "partner":    "RELATION",
    "boyfriend":  "RELATION",
    "girlfriend": "RELATION",
    "son":        "RELATION",
    "daughter":   "RELATION",
    "child":      "RELATION",
    "children":   "RELATION",
    "kids":       "RELATION",
    "kid":        "RELATION",
    "boss":       "RELATION",
    "teacher":    "RELATION",
    "dog":        "RELATION",
    "cat":        "RELATION",
    "baby":       "RELATION",
    "ex":         "RELATION",

    # ── Positive states ──────────────────────────────────────────────────────
    "happy":      "HAPPY",
    "glad":       "HAPPY",
    "pleased":    "HAPPY",
    "joy":        "HAPPY",
    "joyful":     "HAPPY",
    "joyous":     "HAPPY",

    "excited":    "EXCITED",
    "thrilled":   "EXCITED",
    "pumped":     "EXCITED",
    "stoked":     "EXCITED",
    "ecstatic":   "EXCITED",
    "elated":     "EXCITED",
    "overjoyed":  "EXCITED",

    "grateful":   "GRATEFUL",
    "thankful":   "GRATEFUL",
    "appreciative": "GRATEFUL",

    "relieved":   "RELIEVED",

    "proud":      "PROUD",

    "content":    "CONTENT",
    "satisfied":  "CONTENT",
    "comfortable": "CONTENT",

    "amused":     "AMUSED",
    "entertained": "AMUSED",

    "hopeful":    "HOPEFUL",
    "optimistic": "HOPEFUL",

    "love":       "LOVED",
    "loved":      "LOVED",
    "adore":      "LOVED",
    "cherish":    "LOVED",

    # ── Negative states ──────────────────────────────────────────────────────
    "sad":        "SAD",
    "unhappy":    "SAD",
    "miserable":  "SAD",
    "down":       "SAD",
    "depressed":  "SAD",
    "depression": "SAD",
    "grief":      "SAD",

    "angry":      "ANGRY",
    "furious":    "ANGRY",
    "livid":      "ANGRY",
    "enraged":    "ANGRY",
    "mad":        "ANGRY",
    "pissed":     "ANGRY",
    "irate":      "ANGRY",
    "hate":       "ANGRY",
    "hatred":     "ANGRY",
    "loathe":     "ANGRY",

    "afraid":     "AFRAID",
    "scared":     "AFRAID",
    "terrified":  "AFRAID",
    "fearful":    "AFRAID",
    "dread":      "AFRAID",

    "disgusted":  "DISGUSTED",
    "repulsed":   "DISGUSTED",
    "revolted":   "DISGUSTED",

    "ashamed":    "ASHAMED",
    "humiliated": "ASHAMED",
    "embarrassed": "ASHAMED",

    "lonely":     "LONELY",
    "isolated":   "LONELY",
    "alone":      "LONELY",

    "anxious":    "ANXIOUS",
    "nervous":    "ANXIOUS",
    "worried":    "ANXIOUS",
    "anxiety":    "ANXIOUS",
    "panic":      "ANXIOUS",

    "guilty":     "GUILTY",

    "jealous":    "JEALOUS",
    "envious":    "JEALOUS",

    "frustrated": "FRUSTRATED",
    "annoyed":    "FRUSTRATED",
    "irritated":  "FRUSTRATED",

    "exhausted":  "EXHAUSTED",
    "drained":    "EXHAUSTED",
    "burned":     "EXHAUSTED",
    "tired":      "EXHAUSTED",
    "fatigued":   "EXHAUSTED",

    "devastated": "DEVASTATED",
    "shattered":  "DEVASTATED",
    "crushed":    "DEVASTATED",

    "desperate":  "DESPAIR",
    "hopeless":   "DESPAIR",
    "despair":    "DESPAIR",

    # ── Positive qualities ───────────────────────────────────────────────────
    "good":        "POS_QUALITY",
    "great":       "POS_QUALITY",
    "nice":        "POS_QUALITY",
    "wonderful":   "POS_QUALITY",
    "amazing":     "POS_QUALITY",
    "awesome":     "POS_QUALITY",
    "excellent":   "POS_QUALITY",
    "fantastic":   "POS_QUALITY",
    "incredible":  "POS_QUALITY",
    "fine":        "POS_QUALITY",
    "perfect":     "POS_QUALITY",
    "brilliant":   "POS_QUALITY",

    "beautiful":   "BEAUTIFUL",
    "gorgeous":    "BEAUTIFUL",
    "stunning":    "BEAUTIFUL",
    "pretty":      "BEAUTIFUL",
    "lovely":      "BEAUTIFUL",

    "strong":      "STRONG",
    "powerful":    "STRONG",
    "mighty":      "STRONG",

    # ── Negative qualities ───────────────────────────────────────────────────
    "bad":         "NEG_QUALITY",
    "terrible":    "NEG_QUALITY",
    "awful":       "NEG_QUALITY",
    "horrible":    "NEG_QUALITY",
    "dreadful":    "NEG_QUALITY",
    "atrocious":   "NEG_QUALITY",
    "wrong":       "NEG_QUALITY",
    "poor":        "NEG_QUALITY",

    "ugly":        "UGLY",

    "weak":        "WEAK",
    "pathetic":    "WEAK",
    "useless":     "WEAK",

    # ── Events ──────────────────────────────────────────────────────────────
    "success":     "ACHIEVEMENT",
    "succeeded":   "ACHIEVEMENT",
    "accomplished": "ACHIEVEMENT",
    "won":         "ACHIEVEMENT",
    "winning":     "ACHIEVEMENT",
    "promotion":   "ACHIEVEMENT",
    "graduated":   "ACHIEVEMENT",
    "achievement": "ACHIEVEMENT",

    "healed":      "HEALING",
    "recovered":   "HEALING",
    "cured":       "HEALING",

    "lost":        "LOSS",
    "losing":      "LOSS",
    "loss":        "LOSS",
    "missed":      "LOSS",
    "died":        "LOSS",
    "death":       "LOSS",
    "dead":        "LOSS",

    "hurt":        "HARM",
    "injured":     "HARM",
    "wounded":     "HARM",
    "pain":        "HARM",
    "abuse":       "HARM",
    "abused":      "HARM",
    "assault":     "HARM",

    "murder":      "MURDER",
    "murdered":    "MURDER",
    "killed":      "MURDER",

    "suicide":     "CRISIS",
    "suicidal":    "CRISIS",
    "rape":        "CRISIS",
    "raped":       "CRISIS",
    "torture":     "CRISIS",
    "tortured":    "CRISIS",

    "betrayed":    "BETRAYAL",
    "betrayal":    "BETRAYAL",
    "cheated":     "BETRAYAL",
    "stabbed":     "BETRAYAL",

    # ── Social evaluation ────────────────────────────────────────────────────
    "brave":       "SOC_EVAL_POS",
    "generous":    "SOC_EVAL_POS",
    "kind":        "SOC_EVAL_POS",
    "loyal":       "SOC_EVAL_POS",
    "honest":      "SOC_EVAL_POS",
    "caring":      "SOC_EVAL_POS",

    "selfish":     "SOC_EVAL_NEG",
    "cruel":       "SOC_EVAL_NEG",
    "mean":        "SOC_EVAL_NEG",
    "liar":        "SOC_EVAL_NEG",
    "traitor":     "SOC_EVAL_NEG",
    "coward":      "SOC_EVAL_NEG",

    # ── Surprise ────────────────────────────────────────────────────────────
    "surprised":   "SURPRISE",
    "shocked":     "SURPRISE",
    "unexpected":  "SURPRISE",
    "astonished":  "SURPRISE",
    "stunned":     "SURPRISE",

    # ── Actions ─────────────────────────────────────────────────────────────
    "go":       "MOTION",
    "going":    "MOTION",
    "went":     "MOTION",
    "gone":     "MOTION",
    "run":      "MOTION",
    "running":  "MOTION",
    "ran":      "MOTION",
    "walk":     "MOTION",
    "walking":  "MOTION",
    "walked":   "MOTION",
    "come":     "MOTION",
    "came":     "MOTION",
    "coming":   "MOTION",
    "move":     "MOTION",
    "moved":    "MOTION",
    "moving":   "MOTION",

    "have":     "POSSESS",
    "had":      "POSSESS",
    "has":      "POSSESS",
    "own":      "POSSESS",
    "keep":     "POSSESS",
    "kept":     "POSSESS",

    "give":     "TRANSFER",
    "gave":     "TRANSFER",
    "giving":   "TRANSFER",
    "take":     "TRANSFER",
    "took":     "TRANSFER",
    "taking":   "TRANSFER",
    "get":      "TRANSFER",
    "got":      "TRANSFER",
    "getting":  "TRANSFER",
    "send":     "TRANSFER",
    "sent":     "TRANSFER",

    "see":      "PERCEIVE",
    "saw":      "PERCEIVE",
    "seeing":   "PERCEIVE",
    "hear":     "PERCEIVE",
    "heard":    "PERCEIVE",
    "feel":     "PERCEIVE",
    "felt":     "PERCEIVE",
    "notice":   "PERCEIVE",
    "noticed":  "PERCEIVE",

    "say":      "COMMUNICATE",
    "said":     "COMMUNICATE",
    "tell":     "COMMUNICATE",
    "told":     "COMMUNICATE",
    "speak":    "COMMUNICATE",
    "spoke":    "COMMUNICATE",
    "ask":      "COMMUNICATE",
    "asked":    "COMMUNICATE",

    "think":       "THINK",
    "thought":     "THINK",
    "know":        "THINK",
    "knew":        "THINK",
    "believe":     "THINK",
    "understand":  "THINK",
    "understood":  "THINK",
    "remember":    "THINK",
    "remembered":  "THINK",

    # ── Operators ────────────────────────────────────────────────────────────
    "very":        "INTENSIFY",
    "really":      "INTENSIFY",
    "extremely":   "INTENSIFY",
    "absolutely":  "INTENSIFY",
    "totally":     "INTENSIFY",
    "completely":  "INTENSIFY",
    "incredibly":  "INTENSIFY",
    "deeply":      "INTENSIFY",
    "truly":       "INTENSIFY",
    "super":       "INTENSIFY",
    "so":          "INTENSIFY",
    "hella":       "INTENSIFY",
    "fucking":     "INTENSIFY",
    "damn":        "INTENSIFY",
    "too":         "INTENSIFY",

    "not":         "NEGATE",
    "no":          "NEGATE",
    "never":       "NEGATE",
    "nobody":      "NEGATE",
    "nothing":     "NEGATE",
    "nowhere":     "NEGATE",
    "none":        "NEGATE",
    "nor":         "NEGATE",
    "neither":     "NEGATE",
    "dont":        "NEGATE",
    "doesnt":      "NEGATE",
    "didnt":       "NEGATE",
    "cant":        "NEGATE",
    "wont":        "NEGATE",
    "isnt":        "NEGATE",
    "wasnt":       "NEGATE",
    "arent":       "NEGATE",
    "havent":      "NEGATE",
    "hasnt":       "NEGATE",
    "wouldnt":     "NEGATE",
    "couldnt":     "NEGATE",
    "shouldnt":    "NEGATE",

    "and":         "CONNECT",
    "but":         "CONNECT",
    "or":          "CONNECT",
    "if":          "CONNECT",
    "because":     "CONNECT",
    "since":       "CONNECT",
    "although":    "CONNECT",
    "though":      "CONNECT",
    "yet":         "CONNECT",
    "however":     "CONNECT",
    "while":       "CONNECT",

    "maybe":       "HEDGE_OP",
    "perhaps":     "HEDGE_OP",
    "somewhat":    "HEDGE_OP",
    "kinda":       "HEDGE_OP",
    "sorta":       "HEDGE_OP",
    "probably":    "HEDGE_OP",
    "might":       "HEDGE_OP",

    "just":        "TIME",
    "now":         "TIME",
    "then":        "TIME",
    "already":     "TIME",
    "still":       "TIME",
    "finally":     "TIME",
    "recently":    "TIME",
    "today":       "TIME",
    "yesterday":   "TIME",
    "tomorrow":    "TIME",

    # ── Formulaic ────────────────────────────────────────────────────────────
    "hello":       "GREETING",
    "hi":          "GREETING",
    "hey":         "GREETING",
    "goodbye":     "GREETING",
    "bye":         "GREETING",

    "thanks":      "THANKS",
    "thank":       "THANKS",

    "sorry":       "APOLOGY",
    "apologize":   "APOLOGY",

    "please":      "FILLER_WORD",
    "um":          "FILLER_WORD",
    "uh":          "FILLER_WORD",
    "like":        "FILLER_WORD",
    "well":        "FILLER_WORD",
    "ok":          "FILLER_WORD",
    "okay":        "FILLER_WORD",

    # ── Function words (prevent suffix-strip misclassification) ─────────────
    "is":          "FILLER_WORD",
    "was":         "FILLER_WORD",
    "are":         "FILLER_WORD",
    "were":        "FILLER_WORD",
    "been":        "FILLER_WORD",
    "being":       "FILLER_WORD",
    "am":          "FILLER_WORD",
    "do":          "FILLER_WORD",
    "does":        "FILLER_WORD",
    "did":         "FILLER_WORD",
    "will":        "FILLER_WORD",
    "would":       "FILLER_WORD",
    "could":       "FILLER_WORD",
    "should":      "FILLER_WORD",
    "can":         "FILLER_WORD",
    "may":         "FILLER_WORD",
    "shall":       "FILLER_WORD",
    "must":        "FILLER_WORD",
    "the":         "FILLER_WORD",
    "a":           "FILLER_WORD",
    "an":          "FILLER_WORD",
    "this":        "FILLER_WORD",
    "that":        "FILLER_WORD",
    "there":       "FILLER_WORD",
    "here":        "FILLER_WORD",
    "what":        "FILLER_WORD",
    "which":       "FILLER_WORD",
    "who":         "FILLER_WORD",
    "where":       "FILLER_WORD",
    "when":        "FILLER_WORD",
    "how":         "FILLER_WORD",
    "why":         "FILLER_WORD",
    "for":         "FILLER_WORD",
    "from":        "FILLER_WORD",
    "with":        "FILLER_WORD",
    "about":       "FILLER_WORD",
    "into":        "FILLER_WORD",
    "through":     "FILLER_WORD",
    "during":      "FILLER_WORD",
    "before":      "FILLER_WORD",
    "after":       "FILLER_WORD",
    "above":       "FILLER_WORD",
    "below":       "FILLER_WORD",
    "between":     "FILLER_WORD",
    "under":       "FILLER_WORD",
    "over":        "FILLER_WORD",
    "up":          "FILLER_WORD",
    "out":         "FILLER_WORD",
    "on":          "FILLER_WORD",
    "off":         "FILLER_WORD",
    "at":          "FILLER_WORD",
    "to":          "FILLER_WORD",
    "by":          "FILLER_WORD",
    "of":          "FILLER_WORD",
    "in":          "FILLER_WORD",
    "as":          "FILLER_WORD",
    "than":        "FILLER_WORD",
    "also":        "FILLER_WORD",
    "such":        "FILLER_WORD",
    "more":        "FILLER_WORD",
    "most":        "FILLER_WORD",
    "much":        "FILLER_WORD",
    "many":        "FILLER_WORD",
    "some":        "FILLER_WORD",
    "any":         "FILLER_WORD",
    "each":        "FILLER_WORD",

    # ── Ambiguous / inflated words — override V8 auto-map ──────────────────
    # These words have high |dV| in V8 but are contextually neutral in most usage.
    # Without these overrides they steal nucleus and cause false classifications.

    # Function/structural words (should never carry charge)
    "means":       "FILLER_WORD",
    "meaning":     "FILLER_WORD",
    "kind":        "FILLER_WORD",
    "whom":        "FILLER_WORD",
    "quite":       "FILLER_WORD",
    "little":      "FILLER_WORD",
    "fact":        "FILLER_WORD",
    "instead":     "FILLER_WORD",
    "behind":      "FILLER_WORD",
    "enough":      "FILLER_WORD",
    "yes":         "FILLER_WORD",
    "passed":      "FILLER_WORD",
    "direction":   "FILLER_WORD",
    "exactly":     "FILLER_WORD",
    "information": "FILLER_WORD",
    "large":       "FILLER_WORD",
    "quickly":     "FILLER_WORD",
    "count":       "FILLER_WORD",
    "present":     "FILLER_WORD",
    "ground":      "FILLER_WORD",
    "sight":       "FILLER_WORD",
    "turned":      "FILLER_WORD",
    "turning":     "FILLER_WORD",
    "turn":        "FILLER_WORD",
    "fell":        "FILLER_WORD",
    "fall":        "FILLER_WORD",
    "falling":     "FILLER_WORD",
    "rid":         "FILLER_WORD",
    "fart":        "FILLER_WORD",
    "farts":       "FILLER_WORD",
    "riding":      "FILLER_WORD",
    "ride":        "FILLER_WORD",

    # Context-dependent (GAS phase — charge too strong for default mapping)
    # These should be near-neutral; context determines their meaning.
    "down":        "FILLER_WORD",     # "sit down" vs "feeling down" — structure decides
    "home":        "FILLER_WORD",     # "go home" is neutral
    "light":       "FILLER_WORD",     # "light" as noun/adj is neutral
    "heart":       "FILLER_WORD",     # "heart attack" vs "heart of gold" — ambiguous
    "soul":        "FILLER_WORD",     # "soul" in narrative is descriptive
    "strange":     "FILLER_WORD",     # observation, not emotion
    "wish":        "FILLER_WORD",     # "I wish" is mild, not strongly negative

    # Words V8 inflated — real charge is near zero
    "solitary":    "FILLER_WORD",     # V8: dV=-90 — absurd for a descriptive word
    "separated":   "FILLER_WORD",     # V8: dV=-89 — "they separated" is often neutral
    "distracted":  "FILLER_WORD",     # V8: dV=-86 — mild annoyance at most
    "treacherous": "NEG_QUALITY",     # keep negative but not DEVASTATED
    "repulsive":   "NEG_QUALITY",     # keep negative but not ANGRY

    # Casual/slang that strips to negative stems
    "chilling":    "FILLER_WORD",     # "just chilling" = relaxing, not crisis
    "chill":       "FILLER_WORD",     # V8: dV=-70 — absurd for "chill out"
    "chills":      "FILLER_WORD",
    "hanging":     "FILLER_WORD",     # "hanging out" is neutral
    "hang":        "FILLER_WORD",
    "killing":     "HARM",            # strong negative — LIQUID phase handles "killing it" = great
    "tripping":    "FILLER_WORD",     # "tripping" = overreacting (slang)
    "trip":        "FILLER_WORD",

    # V8 auto-import overrides — words wrongly mapped by charge bucketing
    "lay":         "FILLER_WORD",     # V8: dV=+55 — "lay down" is neutral
    "became":      "FILLER_WORD",     # V8: dV=+25 — past tense of become, neutral
    "trees":       "FILLER_WORD",     # V8: dV=+25 — a tree is not emotional
    "tree":        "FILLER_WORD",
    "woke":        "FILLER_WORD",     # ambiguous slang, not emotional
    "career":      "FILLER_WORD",     # strips to "care" → POS_QUALITY (wrong)
    "remain":      "FILLER_WORD",     # V8: dV=+35 — "remained" is neutral
    "remaining":   "FILLER_WORD",
    "remained":    "FILLER_WORD",
    "remains":     "FILLER_WORD",
    "kiss":        "SLIGHT_POS",      # emotional but too strong as HAPPY nucleus
    "kissed":      "SLIGHT_POS",
    "smile":       "SLIGHT_POS",      # emotional coloring, not nucleus material
    "smiled":      "SLIGHT_POS",
    "smiling":     "SLIGHT_POS",
    "laugh":       "SLIGHT_POS",
    "laughed":     "SLIGHT_POS",
    "laughing":    "SLIGHT_POS",
    "swim":        "FILLER_WORD",     # V8: dV=+90 — swimming is not ecstatic
    "swimming":    "FILLER_WORD",
    "bacon":       "FILLER_WORD",     # V8: dV=+66 — bacon is not emotional
    "candy":       "FILLER_WORD",     # V8: dV=+62
    "cash":        "FILLER_WORD",     # V8: dV=+80
    "basic":       "FILLER_WORD",     # V8: dV=-77
    "taste":       "FILLER_WORD",     # V8: dV=+71
    "profit":      "FILLER_WORD",     # V8: dV=+66
    "lift":        "FILLER_WORD",     # V8: dV=+62
    "bid":         "FILLER_WORD",     # V8: dV=+58
    "bout":        "FILLER_WORD",     # V8: dV=+62

    # Narrative words — neutral in literary text
    "passing":     "FILLER_WORD",     # "passing by" is neutral, not SAD
    "ended":       "FILLER_WORD",     # "the chapter ended" is neutral
    "thrown":      "FILLER_WORD",     # "thrown into the mix" is neutral
    "cried":       "SLIGHT_NEG",      # emotional but often neutral in narrative

    # -less words that strip wrong or need explicit mapping
    "heartless":   "SOC_EVAL_NEG",   # "heart" overridden to FILLER, but "heartless" = cruel
    "painless":    "SLIGHT_POS",     # strips to "pain" → HARM (wrong! painless = good)
    "jobless":     "SLIGHT_NEG",     # not in V8
    "homeless":    "NEG_QUALITY",    # override V8's mapping through "home"
    "blameless":   "SLIGHT_POS",     # innocent
    "flawless":    "POS_QUALITY",    # perfect

    # Words missing from V8 entirely
    "fatal":       "NEG_QUALITY",
    "ecstasy":     "EXCITED",
    "reckless":    "NEG_QUALITY",
    "speechless":  "SURPRISE",
    "ruthless":    "SOC_EVAL_NEG",
    "relentless":  "NEG_QUALITY",
    "senseless":   "NEG_QUALITY",
    "mindless":    "NEG_QUALITY",
    "listless":    "EXHAUSTED",
    "numb":        "EXHAUSTED",
    "stuck":       "FRUSTRATED",
    "trapped":     "AFRAID",
    "empty":       "SAD",
    "pointless":   "NEG_QUALITY",
    "hollow":      "SAD",
    "drained":     "EXHAUSTED",
    "suffocating": "AFRAID",
    "drowning":    "AFRAID",
    "spiraling":   "ANXIOUS",
    "overwhelmed": "ANXIOUS",
    "overthinking":"ANXIOUS",
    "ghosting":    "SLIGHT_NEG",
    "ghosted":     "SLIGHT_NEG",

    # Ambiguous action words — neutral in most contexts
    "bumped":      "FILLER_WORD",     # "bumped up" = positive, "bumped into" = neutral
    "bump":        "FILLER_WORD",
    "beat":        "FILLER_WORD",     # "beat the boss" = positive, "beat him" = negative — ambiguous
    "beating":     "FILLER_WORD",
    "stray":       "FILLER_WORD",     # "stray dog" = neutral context
    "grade":       "FILLER_WORD",     # academic word, not emotional
    "grades":      "FILLER_WORD",
    "insanely":    "INTENSIFY",       # intensifier, not negative
    "insane":      "INTENSIFY",       # often used as positive intensifier in casual speech

    # Positive achievement/life words missing from V8
    "confident":   "POS_QUALITY",
    "confidence":  "POS_QUALITY",
    "approved":    "SLIGHT_POS",
    "approval":    "SLIGHT_POS",
    "promoted":    "ACHIEVEMENT",
    "promotion":   "ACHIEVEMENT",
    "secured":     "SLIGHT_POS",
    "funded":      "SLIGHT_POS",
    "funding":     "SLIGHT_POS",
    "agreed":      "SLIGHT_POS",
    "agreement":   "SLIGHT_POS",
    "repaired":    "SLIGHT_POS",
    "fixed":       "SLIGHT_POS",
    "cleared":     "SLIGHT_POS",
    "matched":     "SLIGHT_POS",
    "earned":      "ACHIEVEMENT",
    "graduated":   "ACHIEVEMENT",
    "accepted":    "SLIGHT_POS",
    "pregnant":    "SURPRISE",        # high-impact life event, direction depends on context
    "engaged":     "SLIGHT_POS",
    "married":     "SLIGHT_POS",
    "hired":       "ACHIEVEMENT",
    "startup":     "FILLER_WORD",
    "outfit":      "FILLER_WORD",
    "concert":     "FILLER_WORD",     # venue, not emotion — override V8's dV=+40
    "stepped":     "FILLER_WORD",

    # ── Strong evaluative words — too strong for POS/NEG_QUALITY catch-all ──
    # These are definitive judgments that need named emotional roots
    "great":       "HAPPY",
    "wonderful":   "HAPPY",
    "amazing":     "HAPPY",
    "awesome":     "HAPPY",
    "fantastic":   "HAPPY",
    "incredible":  "HAPPY",
    "brilliant":   "HAPPY",
    "magnificent": "HAPPY",
    "outstanding": "HAPPY",
    "superb":      "HAPPY",
    "excellent":   "HAPPY",
    "perfect":     "HAPPY",
    "terrible":    "SAD",
    "awful":       "SAD",
    "horrible":    "SAD",
    "dreadful":    "SAD",
    "atrocious":   "SAD",
    "disgusting":  "DISGUSTED",
    "revolting":   "DISGUSTED",
    "pathetic":    "SOC_EVAL_NEG",
    "worthless":   "SOC_EVAL_NEG",

    # ── Domain: Financial ────────────────────────────────────────────────────
    "funded":      "FIN_POS",
    "funding":     "FIN_POS",
    "approved":    "FIN_POS",
    "approval":    "FIN_POS",
    "profit":      "FIN_POS",
    "profitable":  "FIN_POS",
    "raise":       "FIN_POS",
    "bonus":       "FIN_POS",
    "savings":     "FIN_POS",
    "invested":    "FIN_POS",
    "salary":      "FILLER_WORD",
    "debt":        "FIN_NEG",
    "bankrupt":    "FIN_NEG",
    "bankruptcy":  "FIN_NEG",
    "overdue":     "FIN_NEG",
    "foreclosure": "FIN_NEG",
    "eviction":    "FIN_NEG",
    "evicted":     "FIN_NEG",
    "broke":       "FIN_NEG",         # "I'm broke" = financial
    "overdraft":   "FIN_NEG",
    "bills":       "FIN_NEG",
    "rent":        "FILLER_WORD",     # neutral unless paired with neg

    # ── Domain: Medical ──────────────────────────────────────────────────────
    "healed":      "MED_POS",
    "remission":   "MED_POS",
    "recovered":   "MED_POS",
    "recovery":    "MED_POS",
    "cured":       "MED_POS",
    "cleared":     "MED_POS",        # "scans cleared" = medical relief
    "diagnosed":   "MED_NEG",
    "diagnosis":   "MED_NEG",
    "terminal":    "MED_NEG",
    "relapsed":    "MED_NEG",
    "relapse":     "MED_NEG",
    "tumor":       "MED_NEG",
    "chemo":       "MED_NEG",
    "chemotherapy":"MED_NEG",
    "surgery":     "FILLER_WORD",    # neutral — context determines
    "hospital":    "FILLER_WORD",

    # ── Domain: Academic ─────────────────────────────────────────────────────
    "graduated":   "ACAD_POS",
    "honors":      "ACAD_POS",
    "scholarship": "ACAD_POS",
    "valedictorian":"ACAD_POS",
    "dean":        "FILLER_WORD",
    "failed":      "ACAD_NEG",
    "expelled":    "ACAD_NEG",
    "flunked":     "ACAD_NEG",
    "dropout":     "ACAD_NEG",
    "suspended":   "ACAD_NEG",

    # ── Domain: Legal ────────────────────────────────────────────────────────
    "acquitted":   "LEGAL_POS",
    "pardoned":    "LEGAL_POS",
    "innocent":    "LEGAL_POS",
    "exonerated":  "LEGAL_POS",
    "arrested":    "LEGAL_NEG",
    "convicted":   "LEGAL_NEG",
    "sentenced":   "LEGAL_NEG",
    "sued":        "LEGAL_NEG",
    "lawsuit":     "LEGAL_NEG",
    "indicted":    "LEGAL_NEG",
    "prison":      "LEGAL_NEG",
    "jail":        "LEGAL_NEG",

    # ── Domain: Career ───────────────────────────────────────────────────────
    "promoted":    "CAREER_POS",
    "promotion":   "CAREER_POS",
    "hired":       "CAREER_POS",
    "recognized":  "CAREER_POS",
    "awarded":     "CAREER_POS",
    "earned":      "CAREER_POS",
    "demoted":     "CAREER_NEG",
    "terminated":  "CAREER_NEG",
    "downsized":   "CAREER_NEG",
    "outsourced":  "CAREER_NEG",

    # ── Domain: Relationship ─────────────────────────────────────────────────
    "engaged":     "REL_POS",
    "married":     "REL_POS",
    "dating":      "REL_POS",
    "matched":     "REL_POS",
    "reconnected": "REL_POS",
    "divorced":    "REL_NEG",
    "dumped":      "REL_NEG",
    "ghosted":     "REL_NEG",
    "rejected":    "REL_NEG",
    "cheated":     "REL_NEG",
    "unfaithful":  "REL_NEG",
    "separated":   "REL_NEG",        # override FILLER_WORD — relationship context

    # Relationship verbs — need structural context to determine meaning
    "left":        "SLIGHT_NEG",      # "she left me" = abandonment, "I left the room" = neutral
    "leave":       "SLIGHT_NEG",
    "leaving":     "SLIGHT_NEG",
    "gone":        "SLIGHT_NEG",      # "she's gone" = loss context
    "lost":        "LOSS",            # override auto-map — this IS loss
    "miss":        "SLIGHT_NEG",      # "I miss her" vs "I missed the bus"
    "missing":     "SLIGHT_NEG",

    # ── Compound events (pre-joined tokens) ──────────────────────────────────
    "laidoff":      "EMPLOYMENT_LOSS",
    "firedoff":     "EMPLOYMENT_LOSS",
    "cancerfree":   "MEDICAL_RELIEF",
    "debtfree":     "MEDICAL_RELIEF",
    "painfree":     "MEDICAL_RELIEF",
    "brokeup":      "RELATIONSHIP_END",
    "passedaway":   "DEATH_EUPHEMISM",
    "brokedown":    "BREAKDOWN",
    "burnedout":    "BURNOUT",
    "pulledthrough": "RECOVERY",
    "workedout":    "RECOVERY",
    "paidoff":      "RECOVERY",
    "turnedaround": "RECOVERY",
    "pulledoff":    "ACHIEVEMENT",

    # Relationship compounds
    "leftme":       "ABANDONMENT",
    "lefthim":      "ABANDONMENT",
    "lefther":      "ABANDONMENT",
    "leftus":       "ABANDONMENT",
    "dumpedme":     "ABANDONMENT",
    "dumpedhim":    "ABANDONMENT",
    "dumpedher":    "ABANDONMENT",
    # Failure/surrender compounds
    "gaveup":       "SURRENDER",
    "givenup":      "SURRENDER",
    "letdown":      "LETDOWN",
    "messedup":     "FAILURE",
    "screwedup":    "FAILURE",
    "ranaway":      "SLIGHT_NEG",
    "movedon":      "MOVING_ON",

    # ── V8 inflation overrides (Gemini-4000 gap analysis, v9.2) ──────────
    # Descriptive words V8 over-charged — neutral in most contexts
    "wooden":      "FILLER_WORD",     # V8 dV=-30 — material descriptor
    "distant":     "FILLER_WORD",     # V8 dV=-77 — spatial descriptor
    "smoke":       "FILLER_WORD",     # V8 dV=-32 — physical substance
    "fragile":     "SLIGHT_NEG",      # V8 dV=-35 — can be emotional but mild
    "critical":    "SLIGHT_NEG",      # V8 dV=-55 — context-dependent (critical thinking vs critical condition)
    "medication":  "FILLER_WORD",     # V8 dV=-32 — medical item, not emotion
    "secure":      "FILLER_WORD",     # V8 dV=+39 — state descriptor
    "current":     "FILLER_WORD",     # V8 dV=+28 — temporal descriptor
    "concert":     "FILLER_WORD",     # V8 dV=+44 — event venue (already overridden but ensuring)
    "leaves":      "FILLER_WORD",     # V8 dV=? — plant part or verb, both neutral
    "somewhere":   "FILLER_WORD",     # spatial
    "currently":   "FILLER_WORD",     # temporal
    "horizon":     "FILLER_WORD",     # spatial
    "boots":       "FILLER_WORD",     # clothing

    # ── Missing positive words (Gemini-4000 gap analysis, v9.2) ──────────
    # Words that should carry positive charge but V8 has zero/low
    "supportive":  "SOC_EVAL_POS",    # V8 dV=0 — strong social positive
    "miracle":     "HAPPY",           # V8 dV=+37 — keep strong but via named root
    "miracles":    "HAPPY",
    "bonus":       "FIN_POS",         # V8 dV=+10 — financial positive event
    "massive":     "INTENSIFY",       # amplifier not a charge carrier
    "tickets":     "SLIGHT_POS",      # event attendance = mild positive
    "scholarship": "ACAD_POS",        # already domain but ensure override
    "job":         "CAREER_POS",      # "got the job" = career positive
    "found":       "SLIGHT_POS",      # "finally found" = recovery/relief
    "received":    "SLIGHT_POS",      # acquiring something
    "pizza":       "FILLER_WORD",     # food item
    "ring":        "FILLER_WORD",     # object
    "nocap":       "INTENSIFY",       # slang intensifier, not a word to evaluate

    # ── Sarcasm false positive fixes (v9.2) ──────────────────────────────
    # These triggered false sarcasm detection on genuine positive sentences
    "supportive":  "SOC_EVAL_POS",
    "always":      "FILLER_WORD",     # "you always are" is filler, not neg context

    # Words that need stronger positive charge
    "best":        "HAPPY",           # V8 dV=+25 too weak — "best" is strong positive eval
    "insane":      "HAPPY",           # slang positive — "that was insane" = amazing (LIQUID)
    "found":       "SLIGHT_POS",      # recovery — "finally found" = relief
    "lost":        "SLIGHT_NEG",      # override LOSS — "lost my ring" is mild, not grief

    # ── Domain overrides (AFTER auto-import — last-write-wins) ───────────
    # These override coarse auto-mappings with domain-specific roots.
    # Must be LAST in the dict to take priority over auto-import.

    # Financial
    "funded": "FIN_POS", "funding": "FIN_POS", "approved": "FIN_POS",
    "profit": "FIN_POS", "bonus": "FIN_POS", "savings": "FIN_POS",
    "invested": "FIN_POS", "raise": "FIN_POS",
    "debt": "FIN_NEG", "bankrupt": "FIN_NEG", "bankruptcy": "FIN_NEG",
    "overdue": "FIN_NEG", "foreclosure": "FIN_NEG", "evicted": "FIN_NEG",
    "overdraft": "FIN_NEG", "broke": "FIN_NEG", "bills": "FIN_NEG",

    # Medical
    "healed": "MED_POS", "remission": "MED_POS", "recovered": "MED_POS",
    "recovery": "MED_POS", "cured": "MED_POS", "cleared": "MED_POS",
    "diagnosed": "MED_NEG", "diagnosis": "MED_NEG", "terminal": "MED_NEG",
    "relapsed": "MED_NEG", "relapse": "MED_NEG", "tumor": "MED_NEG",
    "cancer": "MED_NEG", "chemo": "MED_NEG", "chemotherapy": "MED_NEG",

    # Academic
    "graduated": "ACAD_POS", "honors": "ACAD_POS", "scholarship": "ACAD_POS",
    "failed": "ACAD_NEG", "expelled": "ACAD_NEG", "flunked": "ACAD_NEG",
    "suspended": "ACAD_NEG",

    # Legal
    "acquitted": "LEGAL_POS", "pardoned": "LEGAL_POS", "innocent": "LEGAL_POS",
    "exonerated": "LEGAL_POS",
    "arrested": "LEGAL_NEG", "convicted": "LEGAL_NEG", "sentenced": "LEGAL_NEG",
    "sued": "LEGAL_NEG", "indicted": "LEGAL_NEG", "prison": "LEGAL_NEG",
    "jail": "LEGAL_NEG",

    # Career
    "promoted": "CAREER_POS", "promotion": "CAREER_POS", "hired": "CAREER_POS",
    "recognized": "CAREER_POS", "awarded": "CAREER_POS", "earned": "CAREER_POS",
    "fired": "CAREER_NEG", "demoted": "CAREER_NEG", "terminated": "CAREER_NEG",
    "downsized": "CAREER_NEG",

    # Relationship
    "engaged": "REL_POS", "married": "REL_POS", "dating": "REL_POS",
    "reconnected": "REL_POS",
    "divorced": "REL_NEG", "dumped": "REL_NEG", "ghosted": "REL_NEG",
    "rejected": "REL_NEG", "cheated": "REL_NEG", "separated": "REL_NEG",

    # ── Promoted from SLIGHT_NEG — too weak for negative sentences ───────
    "grave":       "NEG_QUALITY",
    "disease":     "MED_NEG",
    "sick":        "MED_NEG",
    "illness":     "MED_NEG",
    "divorce":     "REL_NEG",
    "stupid":      "SOC_EVAL_NEG",
    "lying":       "SOC_EVAL_NEG",
    "screaming":   "NEG_QUALITY",
    "screams":     "NEG_QUALITY",
    "battle":      "NEG_QUALITY",
    "fight":       "NEG_QUALITY",
    "fighting":    "NEG_QUALITY",
    "solitude":    "LONELY",
    "pretended":   "NEG_QUALITY",
    "pretending":  "NEG_QUALITY",
    "threw":       "FILLER_WORD",
    "muffled":     "FILLER_WORD",
    "judgment":    "NEG_QUALITY",
    "tempted":     "NEG_QUALITY",
    "spent":       "FILLER_WORD",
    "end":         "FILLER_WORD",

    # ── Narrative/descriptive words that shouldn't carry emotional charge ──
    "staff":       "FILLER_WORD",     # noun, not emotional
    "personal":    "FILLER_WORD",     # adjective, descriptive
    "quick":       "FILLER_WORD",     # speed descriptor
    "quickly":     "FILLER_WORD",
    "number":      "FILLER_WORD",     # math/count, not emotional
    "become":      "FILLER_WORD",     # verb of state change, neutral
    "becoming":    "FILLER_WORD",
    "became":      "FILLER_WORD",
    "wandering":   "FILLER_WORD",     # motion descriptor
    "wander":      "FILLER_WORD",
    "spirit":      "FILLER_WORD",     # context-dependent — "team spirit" vs "spiritual"
    "recognized":  "FILLER_WORD",     # override CAREER_POS — neutral in narrative
    "fine":        "FILLER_WORD",     # too ambiguous ("I'm fine" = masking OR genuine)
    "hope":        "SLIGHT_POS",      # override POS_QUALITY — mild, not definitive
    "hoping":      "SLIGHT_POS",
    "pay":         "FILLER_WORD",     # override FIN_NEG — "pay" alone is neutral
    "mean":        "FILLER_WORD",     # override SOC_EVAL_NEG — "what does it mean" is neutral
    "wound":       "FILLER_WORD",     # override AFRAID — "wound" in narrative is descriptive
    "drowning":    "FILLER_WORD",     # override AFRAID — often metaphorical in narrative
    "prison":      "FILLER_WORD",     # setting, not emotion — "he went to prison" is descriptive

    # ── Achievement / success words (novel-500 gap analysis) ─────────────
    "figured":     "ACHIEVEMENT",     # "figured out the bug"
    "managed":     "SLIGHT_POS",      # "managed to do X"
    "finally":     "SLIGHT_POS",      # "finally X" = relief/achievement
    "blessed":     "GRATEFUL",        # "so blessed"
    "rested":      "CONTENT",         # "feel rested"
    "slayed":      "POS_QUALITY",     # slang positive
    "slay":        "POS_QUALITY",
    "incredible":  "HAPPY",           # strong positive eval
    "insanely":    "INTENSIFY",       # override — intensifier not neg
    "perfectly":   "POS_QUALITY",
    "maple":       "FILLER_WORD",     # food item, not emotional
    "pancakes":    "FILLER_WORD",
    "syrup":       "FILLER_WORD",
    "breeze":      "FILLER_WORD",     # weather descriptor
    "alarm":       "FILLER_WORD",
    "cloud":       "FILLER_WORD",
    "race":        "FILLER_WORD",
    "crumbs":      "FILLER_WORD",     # slang: "left no crumbs" = did well

    # ── Negative event/situation words (novel-500 gap) ───────────────────
    "cancelled":   "SLIGHT_NEG",     # context-dependent: force flow determines negativity
    "canceled":    "SLIGHT_NEG",     # "they cancelled" = bad, "I cancelled" = neutral
    "sideswiped":  "NEG_QUALITY",
    "leaking":     "NEG_QUALITY",
    "scratched":   "SLIGHT_NEG",
    "dropped":     "SLIGHT_NEG",
    "audited":     "NEG_QUALITY",
    "blocked":     "NEG_QUALITY",
    "soaked":      "SLIGHT_NEG",
    "bleeding":    "NEG_QUALITY",
    "failing":     "ACAD_NEG",
    "poison":      "MED_NEG",
    "closed":      "SLIGHT_NEG",
    "backed":      "SLIGHT_NEG",
    "refuses":     "NEG_QUALITY",
    "refuse":      "NEG_QUALITY",
    "refused":     "NEG_QUALITY",
    "tolerate":    "SLIGHT_NEG",
    "another":     "SLIGHT_NEG",      # "another X" = repetition frustration in context
    "flop":        "NEG_QUALITY",
    "dark":        "SLIGHT_NEG",
    "terrible":    "SAD",             # already mapped but ensure override
    "extra":       "FILLER_WORD",     # override if auto-imported

    # ── Direction words — need DIRECTIONAL bond sites ────────────────────
    # These are FILLER for charge but carry structural bonding potential.
    # Re-mapped to MOTION so they get DIRECTIONAL bond sites from CATEGORY_BONDS.
    # Financial context words — need charge for sarcasm detection
    "paying":      "FIN_NEG",         # spending money = financial negative context
    "pay":         "FIN_NEG",
    "cost":        "FIN_NEG",
    "costs":       "FIN_NEG",
    "spend":       "FIN_NEG",
    "spending":    "FIN_NEG",
    "expensive":   "FIN_NEG",
    "price":       "FIN_NEG",
    "fee":         "FIN_NEG",
    "charge":      "FIN_NEG",
    "dollars":     "FILLER_WORD",     # override V8 dV=+35 (neutral unit of currency)
    "dollar":      "FILLER_WORD",
    "money":       "FILLER_WORD",

    # Negative context words for sarcasm detection
    "delayed":     "SLIGHT_NEG",
    "delay":       "SLIGHT_NEG",
    "traffic":     "SLIGHT_NEG",
    "parking":     "SLIGHT_NEG",
    "waiting":     "SLIGHT_NEG",
    "wait":        "SLIGHT_NEG",
    "forgot":      "NEG_QUALITY",
    "forgotten":   "NEG_QUALITY",
    "forget":      "FILLER_WORD",     # imperative "forget it" is different
    "broke":       "FIN_NEG",
    "broken":      "NEG_QUALITY",
    "tiny":        "FILLER_WORD",     # override V8 — "tiny" is descriptive, not emotional
    "small":       "FILLER_WORD",
    "empty":       "NEG_QUALITY",

    "off":         "MOTION",
    "away":        "MOTION",
    "out":         "MOTION",
    "back":        "MOTION",
    "down":        "MOTION",          # override the earlier FILLER_WORD
    "up":          "MOTION",
    "over":        "MOTION",
    "through":     "MOTION",
}

# ---------------------------------------------------------------------------
# Suffix stripping — try longer suffixes first; min stem length enforced
# ---------------------------------------------------------------------------

# Each entry: (suffix, min_stem_length)
_SUFFIXES: list[tuple[str, int]] = [
    # 7-char suffixes
    ("fulness",  7),
    ("lessly",   6),
    ("ingness",  7),
    # 6-char suffixes
    # (none beyond above at 6)
    # 4-5 char suffixes
    ("ness",     4),
    ("ment",     4),
    ("tion",     4),
    ("sion",     4),
    ("less",     4),
    ("able",     4),
    ("ible",     4),
    # 3-char suffixes
    ("ful",      3),
    ("ous",      3),
    ("ive",      3),
    ("ity",      3),
    ("ing",      3),
    ("est",      3),
    ("ling",     3),
    # 2-char suffixes
    ("ly",       2),
    ("ed",       2),
    ("er",       2),
    ("es",       2),
    # 1-char suffixes
    ("s",        1),
]


def _strip_suffix(word: str) -> str | None:
    """
    Try removing known suffixes from longest to shortest.
    Returns the stem if a suffix was removed and the stem meets minimum
    length; returns None if no suffix matched.
    """
    for suffix, min_stem in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[: -len(suffix)]
    return None


# Punctuation characters to strip from token edges
_PUNCT = string.punctuation + "\u2026\u2013\u2014"  # … – —
_PUNCT_TABLE = str.maketrans("", "", _PUNCT)


def _candidate_stems(word: str) -> list[str]:
    """
    Return candidate stems to try after suffix stripping.
    Handles English i→y spelling restoration (happiness → happi → happy).
    """
    stem = _strip_suffix(word)
    if stem is None:
        return []
    candidates = [stem]
    # i → y restoration: "happi" → "happy", "anxi" → "anxy" (won't match, harmless)
    if stem.endswith("i"):
        candidates.append(stem[:-1] + "y")
    # doubled consonant removal: "running" → "runn" → "run"
    if len(stem) >= 2 and stem[-1] == stem[-2]:
        candidates.append(stem[:-1])
    return candidates


def map_to_root(word: str) -> Root:
    """
    Map a single English word to its V9 Root via three-tier lookup.

    Tier 1 — Direct: lowercase + strip punctuation, look up in WORD_TO_ROOT.
    Tier 2 — Morphological: strip one suffix, look up stem in WORD_TO_ROOT.
    Tier 3 — Fallback: return ROOTS["OBJECT_GENERIC"] (GAS, zero charge).
    """
    # Clean: lowercase, remove punctuation
    clean = word.lower().translate(_PUNCT_TABLE).strip()

    # Tier 1: direct lookup
    if clean in WORD_TO_ROOT:
        return ROOTS[WORD_TO_ROOT[clean]]

    # Tier 2: morphological stripping (with spelling normalization)
    for stem in _candidate_stems(clean):
        if stem in WORD_TO_ROOT:
            return ROOTS[WORD_TO_ROOT[stem]]

    # Tier 3: V8 calibrated vocabulary fallback
    # If the word isn't manually mapped but IS in V8's vocabulary,
    # create a per-word root with V8's calibrated charge (5D → 7D).
    from .vocabulary import _V8_VOCAB
    if clean in _V8_VOCAB:
        v8_charge = _V8_VOCAB[clean]
        # Only use if it has meaningful charge (|dV| >= 5)
        if abs(v8_charge[0]) >= 5:
            charge_7d = v8_charge + (0, 0)  # extend 5D → 7D (dW=0, dI=0)
            # Determine category from charge polarity
            cat = RootCategory.POSITIVE_QUALITY if v8_charge[0] > 0 else RootCategory.NEGATIVE_QUALITY
            return Root(
                name=f"V8:{clean}",
                category=cat,
                charge=charge_7d,
                phase="GAS",
            )
    # Also check stems against V8
    for stem in _candidate_stems(clean):
        if stem in _V8_VOCAB:
            v8_charge = _V8_VOCAB[stem]
            if abs(v8_charge[0]) >= 5:
                charge_7d = v8_charge + (0, 0)
                cat = RootCategory.POSITIVE_QUALITY if v8_charge[0] > 0 else RootCategory.NEGATIVE_QUALITY
                return Root(
                    name=f"V8:{stem}",
                    category=cat,
                    charge=charge_7d,
                    phase="GAS",
                )

    # Tier 4: generic fallback — truly unknown word
    return ROOTS["OBJECT_GENERIC"]
