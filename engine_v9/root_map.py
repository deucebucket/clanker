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
from engine_v9.roots import ROOTS, Root

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

    # Tier 3: generic fallback
    return ROOTS["OBJECT_GENERIC"]
