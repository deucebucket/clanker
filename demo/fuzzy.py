"""
Fuzzy word matcher for the Clanker pipeline.

Fast preprocessing layer between dictionary lookup and morpheme decomposition.
Catches typos, elongation (e.g. "happyyyy"), and text speak (e.g. "u", "gr8")
before falling back to morpheme analysis.

Flow:
    word -> exact match in WORD_FORCES? -> YES -> use force
                                        -> NO  -> fuzzy match? -> YES -> use matched force
                                                                -> NO  -> morpheme decomposition
"""

import re
from collections import defaultdict

from .forces import WORD_FORCES

# ── Strategy 2: Text speak mapping ──────────────────────────────────

TEXT_SPEAK = {
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
}

# Filter to only mappings whose target is actually in WORD_FORCES
_TEXT_SPEAK_VALID = {k: v for k, v in TEXT_SPEAK.items() if v in WORD_FORCES}

# ── Strategy 3: Edit distance index ────────────────────────────────

_fuzzy_index = None


def _build_fuzzy_index():
    """Build index for fast approximate lookup.

    Index words by (length, first_char) and (length, second_char) so we
    only need to check a small subset during edit-distance search.
    """
    by_len_and_char = defaultdict(set)
    for word in WORD_FORCES:
        if len(word) >= 4:  # Only index words 4+ chars (targets for 5+ char inputs)
            by_len_and_char[(len(word), word[0])].add(word)
            if len(word) > 1:
                by_len_and_char[(len(word), word[1])].add(word)
    return by_len_and_char


def _get_fuzzy_index():
    """Lazy-init the fuzzy index."""
    global _fuzzy_index
    if _fuzzy_index is None:
        _fuzzy_index = _build_fuzzy_index()
    return _fuzzy_index


def _levenshtein(s1, s2):
    """Compute Levenshtein edit distance between two strings.

    Optimised single-row DP — O(min(m,n)) space.
    """
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dele = curr_row[j] + 1
            sub = prev_row[j] + (c1 != c2)
            curr_row.append(min(ins, dele, sub))
        prev_row = curr_row
    return prev_row[-1]


# ── Strategy 1: Character deduplication ─────────────────────────────

def _deduplicate(word):
    """Collapse character elongation and check WORD_FORCES.

    "happyyyy" -> "happyy" -> "happy"
    "nooooo"   -> "noo"    -> "no"

    Only activates when the word contains a run of 3+ identical characters,
    so legitimate words with natural doubles (e.g. "happy") are never touched.
    """
    # Bail out early if there are no elongated runs (3+ of same char)
    if not re.search(r'(.)\1{2,}', word):
        return None

    # Try 1: collapse runs of 3+ identical chars down to 2
    reduced2 = re.sub(r'(.)\1{2,}', r'\1\1', word)
    if reduced2 in WORD_FORCES:
        return reduced2

    # Try 2: collapse runs of 3+ identical chars down to 1
    reduced1 = re.sub(r'(.)\1{2,}', r'\1', word)
    if reduced1 in WORD_FORCES:
        return reduced1

    # Try 3: collapse 3+ to 2, then try each remaining double as single
    # Handles "happyyyy" -> "happyy" -> try "happy" (remove yy->y)
    doubles = [(m.start(), m.group()[0]) for m in re.finditer(r'(.)\1', reduced2)]
    for pos, char in doubles:
        trial = reduced2[:pos] + char + reduced2[pos + 2:]
        if trial in WORD_FORCES:
            return trial

    return None


# ── Strategy 3: Edit distance search ───────────────────────────────

def _edit_distance_match(word):
    """Find a WORD_FORCES entry within edit distance 1 of *word*.

    Only applied to words of 5+ characters to avoid false positives.
    Words that are already exact matches in WORD_FORCES are skipped
    (the caller should check exact match first).
    """
    if len(word) < 5:
        return None
    if word in WORD_FORCES:
        return None

    index = _get_fuzzy_index()
    candidates = set()

    # Gather candidates: same length +-1, sharing first char only
    # (first char is almost always correct in typos)
    for length_offset in (-1, 0, 1):
        target_len = len(word) + length_offset
        if target_len < 4:
            continue
        candidates.update(index.get((target_len, word[0]), set()))

    for candidate in candidates:
        if _levenshtein(word, candidate) == 1:
            return candidate

    return None


# ── Cache + public API ──────────────────────────────────────────────

_fuzzy_cache = {}


def _try_fuzzy(word):
    """Try all fuzzy strategies in order. Returns matched key or None."""
    # Strategy 1: deduplication
    result = _deduplicate(word)
    if result is not None:
        return result

    # Strategy 2: text speak
    mapped = _TEXT_SPEAK_VALID.get(word)
    if mapped is not None:
        return mapped

    # Strategy 3: edit distance (5+ chars only)
    result = _edit_distance_match(word)
    if result is not None:
        return result

    return None


def fuzzy_match(word):
    """Find a WORD_FORCES match for *word* using fuzzy strategies.

    Returns the matched WORD_FORCES key, or None if no match found.
    Results are cached for O(1) repeated lookups.
    """
    if word in _fuzzy_cache:
        return _fuzzy_cache[word]
    result = _try_fuzzy(word)
    _fuzzy_cache[word] = result
    return result


def clear_cache():
    """Clear the fuzzy match cache (useful for testing)."""
    _fuzzy_cache.clear()
