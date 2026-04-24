"""
Fuzzy word matcher for the Clanker pipeline.

Fast preprocessing layer between dictionary lookup and morpheme decomposition.
Catches typos, elongation (e.g. "happyyyy"), text speak (e.g. "u", "gr8"),
and Cambridge-effect misspellings (scrambled middles, same first/last).

Flow:
    word -> exact match in _VOCAB? -> YES -> use force
                                                 -> NO  -> fuzzy match? -> YES -> use matched force
                                                                         -> NO  -> morpheme decomposition
"""

import re
from collections import defaultdict

from .forces_curated import EMOTIONAL_VOCABULARY

_VOCAB = EMOTIONAL_VOCABULARY
_VOCAB_REF = _VOCAB

# ── Strategy 1: Text speak / internet slang mapping ───────────────

TEXT_SPEAK = {
    # Abbreviations → closest emotional equivalent
    "u": "you", "ur": "your", "r": "are", "b": "be", "c": "see",
    "k": "okay", "ok": "okay", "thx": "thanks", "ty": "thank",
    "pls": "please", "plz": "please", "rn": "now", "fr": "for",
    "ngl": "honestly", "tbh": "honestly", "imo": "think",
    "idk": "unsure", "smh": "disappointed", "af": "very",
    "nvm": "nevermind", "jk": "joking", "irl": "really",
    "fyi": "know", "brb": "wait", "gtg": "leaving",
    "luv": "love", "boi": "boy", "gurl": "girl",
    "dat": "that", "dis": "this", "dey": "they",
    "wut": "what", "wat": "what", "wth": "what",
    "cuz": "because", "bcuz": "because", "bc": "because",
    "tho": "though", "thru": "through",
    "yr": "year", "yrs": "years", "govt": "government",
    "w": "with", "wo": "without", "b4": "before",
    "2day": "today", "2nite": "tonight", "4ever": "forever",
    "gr8": "great", "l8r": "later", "h8": "hate",
    "sum1": "someone", "ne1": "anyone", "no1": "nobody",
    # Common misspellings of emotional words
    "depresed": "depressed", "deppressed": "depressed",
    "anxious": "anxious", "anixous": "anxious",
    "suicidal": "suicidal", "suicidel": "suicidal",
    "lonley": "lonely", "lonly": "lonely",
    "scred": "scared", "scarred": "scared",
    "happines": "happiness", "hapiness": "happiness",
    "dissapointed": "disappointed", "disapointed": "disappointed",
    "exausted": "exhausted", "exhuasted": "exhausted",
    "overwelmed": "overwhelmed", "overwhelmd": "overwhelmed",
    "fustrated": "frustrated", "frustated": "frustrated",
    "embarassed": "embarrassed", "embarased": "embarrassed",
    "definately": "definitely", "definatly": "definitely",
    "desparate": "desperate", "despirate": "desperate",
    "awfull": "awful", "terible": "terrible",
    "beautifull": "beautiful", "wonderfull": "wonderful",
    "gd": "good", "bd": "bad",
}

# Filter to only mappings whose target is in _VOCAB
_TEXT_SPEAK_VALID = {k: v for k, v in TEXT_SPEAK.items() if v in _VOCAB}


# ── Strategy 2: Character deduplication (elongation) ──────────────

def _deduplicate(word):
    """Collapse character elongation and check vocabulary.

    "happyyyy" -> "happyy" -> "happy"
    "nooooo"   -> "noo"    -> "no"
    "soooo"    -> "soo"    -> "so"

    Only activates when the word contains a run of 3+ identical characters.
    """
    if not re.search(r'(.)\1{2,}', word):
        return None

    # Try 1: collapse runs of 3+ identical chars down to 2
    reduced2 = re.sub(r'(.)\1{2,}', r'\1\1', word)
    if reduced2 in _VOCAB:
        return reduced2

    # Try 2: collapse runs of 3+ identical chars down to 1
    reduced1 = re.sub(r'(.)\1{2,}', r'\1', word)
    if reduced1 in _VOCAB:
        return reduced1

    # Try 3: collapse 3+ to 2, then try each remaining double as single
    doubles = [(m.start(), m.group()[0]) for m in re.finditer(r'(.)\1', reduced2)]
    for pos, char in doubles:
        trial = reduced2[:pos] + char + reduced2[pos + 2:]
        if trial in _VOCAB:
            return trial

    return None


# ── Strategy 3: Cambridge-effect matching ─────────────────────────
# People can read words with scrambled middles if first/last letters match.
# "hpapy" -> "happy", "sicskenig" -> "sickening"
# This catches a class of typos that edit-distance misses.

def _cambridge_match(word):
    """Match words by first letter, last letter, and sorted middle.

    Only for words 6+ chars to avoid false positives on small vocab.
    """
    if len(word) < 6:
        return None

    key = (word[0], word[-1], tuple(sorted(word[1:-1])))
    return _cambridge_index.get(key)


# Build Cambridge index lazily
_cambridge_index = None

def _build_cambridge_index():
    idx = {}
    for w in _VOCAB:
        if len(w) >= 5:
            key = (w[0], w[-1], tuple(sorted(w[1:-1])))
            # Only store first match (avoid collisions overwriting)
            if key not in idx:
                idx[key] = w
    return idx

def _get_cambridge_index():
    global _cambridge_index
    if _cambridge_index is None:
        _cambridge_index = _build_cambridge_index()
    return _cambridge_index


# ── Strategy 4: Edit distance search ─────────────────────────────

_fuzzy_index = None

def _build_fuzzy_index():
    """Index words by (length, first_char) for fast approximate lookup."""
    by_len_and_char = defaultdict(set)
    for word in _VOCAB:
        if len(word) >= 4:
            by_len_and_char[(len(word), word[0])].add(word)
            if len(word) > 1:
                by_len_and_char[(len(word), word[1])].add(word)
    return by_len_and_char

def _get_fuzzy_index():
    global _fuzzy_index
    if _fuzzy_index is None:
        _fuzzy_index = _build_fuzzy_index()
    return _fuzzy_index

def _levenshtein(s1, s2):
    """Levenshtein edit distance — O(min(m,n)) space."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1,
                              prev_row[j] + (c1 != c2)))
        prev_row = curr_row
    return prev_row[-1]

def _edit_distance_match(word):
    """Find vocabulary entry within edit distance 1. Words 7+ chars only.

    Higher threshold than before because the vocabulary is 2.2K words,
    making false positives much more likely on short words.
    """
    if len(word) < 7 or word in _VOCAB:
        return None

    index = _get_fuzzy_index()
    candidates = set()
    for length_offset in (-1, 0, 1):
        target_len = len(word) + length_offset
        if target_len < 4:
            continue
        candidates.update(index.get((target_len, word[0]), set()))

    for candidate in candidates:
        if _levenshtein(word, candidate) == 1:
            return candidate
    return None




# ── Strategy 5: Simple stemmer ──────────────────────────────────
# Strip common English suffixes and look up the root.
# "hates" -> "hate", "loving" -> "love", "cries" -> "cry"
# NOT a full linguistic stemmer -- just common patterns that matter.

def _stem_match(word):
    """Try stripping common suffixes to find a vocabulary match."""
    if len(word) < 4:
        return None
    
    # Try each suffix pattern
    patterns = [
        # (suffix_to_strip, possible_replacements_to_try)
        # ORDER MATTERS: -s before -es so "bites"->bite not bit
        ("ies", ["y", "ie"]),          # cries->cry, dies->die
        ("s", [""]),                     # bites->bite, kills->kill
        ("es", ["", "e"]),              # watches->watch, hates->hate
        ("ing", ["", "e"]),             # loving->lov? no, loving->love
        ("ed", ["", "e"]),              # walked->walk, loved->love
        ("ly", [""]),                    # sadly->sad
        ("ness", [""]),                  # sadness->sad
        ("ment", [""]),                  # abandonment->abandon
        ("ful", [""]),                   # hopeful->hope
        ("less", [""]),                  # hopeless->hope
        ("er", ["", "e"]),              # hater->hate, lover->love
        ("est", ["", "e"]),             # saddest->sad
    ]
    
    for suffix, replacements in patterns:
        if word.endswith(suffix):
            root = word[:-len(suffix)]
            if len(root) < 2:
                continue
            for repl in replacements:
                candidate = root + repl
                if candidate in _VOCAB_REF:
                    return candidate
    
    return None

# ── Cache + public API ────────────────────────────────────────────

_fuzzy_cache = {}

def fuzzy_match(word):
    """Find an _VOCAB match for *word* using fuzzy strategies.

    Returns the matched vocabulary key, or None if no match found.
    Results are cached for O(1) repeated lookups.

    Strategy order:
      1. Deduplication (elongation): "happyyyy" -> "happy"
      2. Text speak mapping: "tbh" -> "honestly"
      3. Cambridge-effect: scrambled middles -> correct word
      4. Edit distance (5+ chars): 1-edit typos
    """
    if word in _fuzzy_cache:
        return _fuzzy_cache[word]

    # Strategy 1: deduplication
    result = _deduplicate(word)
    if result is not None:
        _fuzzy_cache[word] = result
        return result

    # Strategy 2: text speak
    mapped = _TEXT_SPEAK_VALID.get(word)
    if mapped is not None:
        _fuzzy_cache[word] = mapped
        return mapped

    # Strategy 3: Cambridge effect (5+ chars)
    _get_cambridge_index()  # ensure built
    result = _cambridge_match(word)
    if result is not None:
        _fuzzy_cache[word] = result
        return result

    # Strategy 4: stemmer (conjugations)
    result = _stem_match(word)
    if result is not None:
        _fuzzy_cache[word] = result
        return result

    # Strategy 5: edit distance — DISABLED (2.2K vocab too small,
    # causes false positives like "degrees"→"degree", "committee"→"committed")
    # TODO: re-enable when vocab is larger or with a suffix-aware matcher
    # result = _edit_distance_match(word)
    # if result is not None:
    #     _fuzzy_cache[word] = result
    #     return result

    _fuzzy_cache[word] = None
    return None


def clear_cache():
    """Clear the fuzzy match cache (useful for testing)."""
    _fuzzy_cache.clear()
