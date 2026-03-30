"""Context Operators for the Clanker Pendulum Engine.

Bridge words are NOT zero-weight words. They are context multipliers that
determine HOW MUCH the next emotional word should weigh.

Research basis: Mining 99.5K EmpatheticDialogues showed:
  - "I am sad" (self-report) hits harder than "it was sad" (observation)
  - "my sad life" (personal) > "a sad movie" (describing)
  - "do you hate" (question) != "I hate" (statement)
  - "I was angry" (past) < "I am angry" (present)

Usage:
    from demo.context_operators import get_context_coefficient, is_question, detect_tense

    words = "i am really sad".split()
    idx = 3  # "sad"
    coeff = get_context_coefficient(words, idx)
    # self(1.8) * present(1.0) * intensifier(1.4) = 2.52x
"""


# ---------------------------------------------------------------------------
# Context Coefficients — 84 operators across 17 categories
#
# Values represent how much a bridge word amplifies/dampens the NEXT
# emotional word. 1.0 = neutral (no change). >1.0 = amplify. <1.0 = dampen.
#
# RULE: Operators in the SAME category do NOT multiply — take the NEAREST.
#       Operators in DIFFERENT categories DO multiply.
#       Total coefficient is capped at 3.0 and floored at 0.1.
# ---------------------------------------------------------------------------

# Each entry: word -> (coefficient, category_name) or (coefficient, category_name, d_offset)
# Category names used to prevent intra-category multiplication.
# Optional d_offset shifts Dominance independently of the coefficient scaling.
# This is how hedges express uncertainty: dampen magnitude AND lower confidence.

CONTEXT_OPERATORS = {
    # --- Self-reference high ---
    "i":        (1.8, "self_high"),
    "me":       (1.6, "self_high"),
    "myself":   (1.8, "self_high"),
    "im":       (1.7, "self_high"),
    "i'm":      (1.7, "self_high"),
    "i've":     (1.6, "self_high"),
    "i'll":     (1.5, "self_high"),
    "i'd":      (1.4, "self_high"),

    # --- Possessive self ---
    "my":       (1.5, "possessive_self"),
    "mine":     (1.5, "possessive_self"),
    "our":      (1.3, "possessive_self"),
    "ours":     (1.3, "possessive_self"),

    # --- Self medium ---
    "we":       (1.3, "self_medium"),
    "us":       (1.2, "self_medium"),

    # --- Other close ---
    "you":      (0.9, "other_close"),
    "your":     (0.9, "other_close"),
    "yours":    (0.9, "other_close"),
    "you're":   (0.9, "other_close"),

    # --- Other far ---
    "he":       (0.7, "other_far"),
    "she":      (0.7, "other_far"),
    "his":      (0.8, "other_far"),
    "her":      (0.8, "other_far"),
    "they":     (0.6, "other_far"),
    "their":    (0.7, "other_far"),
    "them":     (0.6, "other_far"),

    # --- Abstract ---
    "it":       (0.4, "abstract"),
    "it's":     (0.4, "abstract"),
    "its":      (0.4, "abstract"),

    # --- Articles --- (0.3 was too aggressive — crushed movie reviews to V=2)
    "a":        (0.6, "articles"),
    "an":       (0.6, "articles"),
    "the":      (0.7, "articles"),

    # --- Demonstrative near ---
    "this":     (1.2, "demo_near"),
    "these":    (1.0, "demo_near"),
    "here":     (1.1, "demo_near"),

    # --- Demonstrative far ---
    "that":     (0.6, "demo_far"),
    "those":    (0.5, "demo_far"),
    "there":    (0.5, "demo_far"),

    # --- Amplifiers ---
    "so":       (1.4, "amplifiers"),
    "very":     (1.3, "amplifiers"),
    "really":   (1.4, "amplifiers"),
    "extremely":    (1.6, "amplifiers"),
    "absolutely":   (1.6, "amplifiers"),
    "totally":      (1.4, "amplifiers"),
    "completely":   (1.5, "amplifiers"),
    "incredibly":   (1.5, "amplifiers"),
    "deeply":       (1.4, "amplifiers"),
    "truly":        (1.3, "amplifiers"),
    "super":        (1.5, "amplifiers"),
    "hella":        (1.5, "amplifiers"),
    "mad":          (1.4, "amplifiers"),   # "mad angry" = very angry (slang)
    "fucking":      (1.6, "amplifiers"),   # spice word — amplifies direction of next word
    "freakin":      (1.4, "amplifiers"),   # softened fucking
    "freaking":     (1.4, "amplifiers"),
    # "damn" excluded — dual-use: "damn good"=amplifier, "damn you"=curse, "don't give a damn"=indifference

    # --- Diminishers ---
    "just":     (0.7, "diminishers"),
    "only":     (0.7, "diminishers"),
    "barely":   (0.5, "diminishers"),
    "slightly": (0.6, "diminishers"),
    "somewhat": (0.7, "diminishers"),
    "kinda":    (0.7, "diminishers"),
    "sorta":    (0.7, "diminishers"),

    # --- Hedging qualifiers (reduce certainty/commitment + lower Dominance) ---
    # Third element is D-offset: how much to depress Dominance independently.
    # Hedging = "I feel this but I'm not confident about it."
    "generally":    (0.70, "hedging", -5),
    "usually":      (0.75, "hedging", -5),
    "sometimes":    (0.50, "hedging", -10),
    "occasionally": (0.40, "hedging", -10),
    "often":        (0.80, "hedging", -5),
    "rarely":       (0.30, "hedging", -10),
    "typically":    (0.70, "hedging", -5),
    "maybe":        (0.50, "hedging", -15),
    "perhaps":      (0.50, "hedging", -15),
    "probably":     (0.60, "hedging", -10),
    "possibly":     (0.40, "hedging", -15),
    "arguably":     (0.60, "hedging", -10),
    "seemingly":    (0.60, "hedging", -15),
    "apparently":   (0.60, "hedging", -15),
    "supposedly":   (0.50, "hedging", -15),
    # Cognitive hedges — "I think", "I guess", "I feel like"
    # These dampen after "I" (self_high 1.8x), net: 1.8 * 0.7 = 1.26x
    "think":        (0.70, "hedging", -10),
    "guess":        (0.50, "hedging", -20),
    "suppose":      (0.60, "hedging", -15),
    # NOTE: "believe" excluded — too context-dependent. "I believe it's X" = hedge,
    # but "I cannot believe this" = exclamation. Let it fall through to vocabulary.
    "reckon":       (0.60, "hedging", -10),
    "assume":       (0.60, "hedging", -10),
    "wonder":       (0.50, "hedging", -15),
    "feel":         (0.90, "hedging", -5),
    # Glass breakers — "honestly" / "to be honest" = laying it bare
    # Not dishonesty signal — emphasizes what follows. Like "feel" but stronger.
    # Often precedes negative truth ("honestly, this sucks")
    "honestly":     (1.20, "glass_breaker", -5),
    "truthfully":   (1.20, "glass_breaker", -5),
    "felt":         (0.85, "hedging", -5),

    # --- Present tense ---
    "am":       (1.0, "present_tense"),
    "is":       (0.9, "present_tense"),
    "are":      (0.9, "present_tense"),

    # --- Past tense ---
    "was":      (0.85, "past_tense"),
    "were":     (0.85, "past_tense"),
    "had":      (0.8, "past_tense"),
    "did":      (0.85, "past_tense"),
    "been":     (0.8, "past_tense"),

    # --- Hypothetical (modals reduce certainty + lower Dominance) ---
    "would":    (0.60, "hypothetical", -10),
    "could":    (0.60, "hypothetical", -10),
    "might":    (0.50, "hypothetical", -15),
    "should":   (0.70, "hypothetical", -5),
    "will":     (0.70, "hypothetical", 0),
    "can":      (0.80, "hypothetical", 0),
    "may":      (0.60, "hypothetical", -10),

    # --- Deflection shields (masking real feelings with apathy) ---
    # "Whatever" gates emotion: surrenders agency (D-25), feigns calm (A dampened)
    "whatever":     (0.50, "deflection", -25),
    "whatevs":      (0.50, "deflection", -20),
    "idc":          (0.40, "deflection", -25),
    "idk":          (0.50, "deflection", -20),
    "meh":          (0.40, "deflection", -20),
    "nvm":          (0.45, "deflection", -20),
    "nevermind":    (0.45, "deflection", -20),
    "dunno":        (0.50, "deflection", -20),
    # Exasperation starters — "man..." "dude..." at sentence start
    "man":          (0.80, "exasperation", -10),
    "dude":         (0.80, "exasperation", -10),
    "bro":          (0.80, "exasperation", -10),
    # Temporal urgency slang
    "rn":           (1.2, "amplifiers", -5),



    # --- Evidential markers (reported speech, distance from experience) ---
    "allegedly":    (0.40, "evidential", -20),
    "reportedly":   (0.40, "evidential", -20),
    "presumably":   (0.50, "evidential", -15),
}

# Legacy alias for backward compatibility
CONTEXT_COEFFICIENTS = {k: v[0] for k, v in CONTEXT_OPERATORS.items()}


def _parse_operator(entry):
    """Parse an operator entry — handles both 2-tuple and 3-tuple formats."""
    if len(entry) == 3:
        return entry[0], entry[1], entry[2]
    return entry[0], entry[1], 0


# ---------------------------------------------------------------------------
# Question words — sentences that ASK about emotion rather than FEEL it.
# Asking "are you sad?" is fundamentally different from stating "I am sad."
# ---------------------------------------------------------------------------

QUESTION_STARTERS = {
    "do", "does", "did",
    "is", "are", "was", "were",
    "will", "can", "could", "would", "should",
    "why", "how", "what", "where", "when", "who",
}

QUESTION_DAMPENER = 0.4


# ---------------------------------------------------------------------------
# Tense markers — kept for backward compatibility with detect_tense().
# The new system uses categories in CONTEXT_OPERATORS instead.
# ---------------------------------------------------------------------------

TENSE_MARKERS = {
    # Past tense
    "was":    0.85,
    "were":   0.85,
    "had":    0.85,
    "did":    0.85,

    # Present tense
    "am":     1.0,
    "is":     1.0,
    "are":    1.0,

    # Hypothetical/future
    "will":   0.6,
    "would":  0.6,
    "could":  0.6,
    "might":  0.6,
    "may":    0.6,
}


# Coefficient bounds
_COEFF_CAP = 3.0
_COEFF_FLOOR = 0.1


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def is_question(words):
    """Detect if a sentence is a question.

    Two signals:
      1. Starts with a question word (inverted syntax)
      2. Ends with "?" (any declarative turned question)

    Questions dampen emotional force because asking about an emotion
    is not the same as feeling it.

    Args:
        words: list of lowercase word strings (one sentence)

    Returns:
        True if the sentence is a question, False otherwise.

    Examples:
        >>> is_question(["do", "you", "hate", "me"])
        True
        >>> is_question(["i", "hate", "this"])
        False
        >>> is_question(["you", "hate", "me", "?"])
        True
        >>> is_question(["why", "am", "i", "so", "sad"])
        True
    """
    if not words:
        return False

    # Check first word against question starters
    first = words[0].lower().strip("\"'(")
    if first in QUESTION_STARTERS:
        return True

    # Check if any word ends with or is "?"
    last = words[-1]
    if last == "?" or last.endswith("?"):
        return True

    return False


def detect_tense(words, current_idx):
    """Detect the tense of the clause surrounding the emotional word.

    Scans backwards from current_idx to find the nearest tense marker.
    Returns a multiplier: 1.0 for present, 0.85 for past, 0.6 for hypothetical.

    Only looks back up to 3 words — tense markers further back likely
    belong to a different clause.

    Args:
        words:       list of lowercase word strings
        current_idx: index of the emotional word being evaluated

    Returns:
        float multiplier (0.6 to 1.0)

    Examples:
        >>> detect_tense(["i", "was", "angry"], 2)
        0.85
        >>> detect_tense(["i", "am", "angry"], 2)
        1.0
        >>> detect_tense(["i", "would", "be", "angry"], 3)
        0.6
        >>> detect_tense(["angry"], 0)
        1.0
    """
    # Look back up to 3 words for a tense marker
    search_start = max(0, current_idx - 3)

    for i in range(current_idx - 1, search_start - 1, -1):
        w = words[i].lower().strip("\"'()")
        if w in TENSE_MARKERS:
            return TENSE_MARKERS[w]

    # No tense marker found — default to full impact (present/imperative)
    return 1.0


def get_context_coefficient(words, current_idx):
    """Calculate the combined context multiplier for an emotional word.

    Scans backwards from current_idx to find context operators. Operators
    in the SAME category do NOT stack — only the nearest one counts.
    Operators in DIFFERENT categories multiply together.

    Question dampener (0.4x) is applied when the first word is a question
    starter.

    The final multiplier is capped at 3.0 and floored at 0.1.

    Args:
        words:       list of lowercase word strings (one sentence)
        current_idx: index of the emotional word being evaluated

    Returns:
        float — the combined multiplier to apply to the emotional word's force.
        1.0 means no change. >1.0 = amplify. <1.0 = dampen.

    Examples:
        >>> get_context_coefficient(["i", "am", "sad"], 2)
        1.8
        >>> get_context_coefficient(["a", "sad", "movie"], 1)
        0.3
        >>> round(get_context_coefficient(["i", "am", "very", "sad"], 3), 2)
        2.34
    """
    coeff = _find_combined_context(words, current_idx)
    question_mult = QUESTION_DAMPENER if is_question(words) else 1.0

    result = coeff * question_mult
    return max(_COEFF_FLOOR, min(_COEFF_CAP, result))


def _find_combined_context(words, current_idx):
    """Scan backwards from current_idx collecting one operator per category.

    Within the same category, only the NEAREST operator is used.
    Across different categories, multipliers are combined via multiplication.

    Scans up to 5 positions back to catch multi-word preambles like
    "I am very truly deeply sad".

    Args:
        words:       list of lowercase word strings
        current_idx: index of the emotional word

    Returns:
        float combined coefficient, or 1.0 if no operators found.
    """
    search_start = max(0, current_idx - 5)
    seen_categories = {}  # category -> coefficient (nearest wins)

    for i in range(current_idx - 1, search_start - 1, -1):
        w = words[i].lower().strip("\"'()")
        if w in CONTEXT_OPERATORS:
            coeff, category, _ = _parse_operator(CONTEXT_OPERATORS[w])
            if category not in seen_categories:
                seen_categories[category] = coeff

    if not seen_categories:
        return 1.0

    result = 1.0
    for coeff in seen_categories.values():
        result *= coeff

    return result


# Backward-compatible alias
def _find_nearest_context(words, current_idx):
    """Legacy: find nearest single context operator coefficient.

    Kept for backward compatibility. New code should use
    _find_combined_context() instead.
    """
    search_start = max(0, current_idx - 3)
    for i in range(current_idx - 1, search_start - 1, -1):
        w = words[i].lower().strip("\"'()")
        if w in CONTEXT_COEFFICIENTS:
            return CONTEXT_COEFFICIENTS[w]
    return 1.0


# ---------------------------------------------------------------------------
# Convenience: analyze a full sentence
# ---------------------------------------------------------------------------

def analyze_sentence(text, emotional_word_indices=None):
    """Analyze context multipliers for all words in a sentence.

    If emotional_word_indices is None, computes coefficients for every
    word position (useful for debugging/visualization).

    Args:
        text: raw sentence string
        emotional_word_indices: optional list of int indices to analyze

    Returns:
        list of (word, index, coefficient) tuples

    Example:
        >>> analyze_sentence("I am really sad")
        [('i', 0, 1.0), ('am', 1, 1.8), ('really', 2, 1.8), ('sad', 3, 2.52)]
    """
    words = text.lower().split()
    if emotional_word_indices is None:
        emotional_word_indices = range(len(words))

    results = []
    for idx in emotional_word_indices:
        if idx < len(words):
            coeff = get_context_coefficient(words, idx)
            results.append((words[idx], idx, round(coeff, 4)))

    return results


# ---------------------------------------------------------------------------
# Test / demo section
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("CONTEXT OPERATORS — 84 Operators, 17 Categories")
    print("=" * 70)
    print()

    # Count operators and categories
    categories = set(cat for _, cat in CONTEXT_OPERATORS.values())
    print(f"  Total operators: {len(CONTEXT_OPERATORS)}")
    print(f"  Categories: {len(categories)}")
    print(f"  + Question starters: {len(QUESTION_STARTERS)} (dampener at position 0)")
    print()

    test_cases = [
        # (sentence, emotional word index, description)
        ("I am sad", 2, "Self-report, present tense — maximum impact"),
        ("it was sad", 2, "Abstract + past tense — dampened"),
        ("a sad movie", 1, "Article context — describing, not feeling"),
        ("my sad life", 1, "Possessive personal — strong ownership"),
        ("his sad story", 1, "Possessive other — observation"),
        ("this pain is unbearable", 1, "Demonstrative near — immediate"),
        ("that pain was unbearable", 1, "Demonstrative far — removed"),
        ("do you hate me", 2, "Question — asking, not feeling"),
        ("I hate you", 1, "Statement — direct emotional report"),
        ("I was angry", 2, "Past tense self-report"),
        ("I am angry", 2, "Present tense self-report"),
        ("I would be angry", 3, "Hypothetical — not real yet"),
        ("they were devastated", 2, "Third person past — distant observation"),
        ("why am I so sad", 4, "Question with self-reference + amplifier"),
        ("the cat is happy", 3, "Article + present — mild observation"),
        ("my heart is broken", 3, "Personal possessive + present"),
        ("I am very sad", 3, "Self + present + intensifier = multi-category"),
        ("I am really truly deeply sad", 5, "Self + present + amplifier (nearest only)"),
        ("just barely sad", 2, "Diminisher (nearest only, same category)"),
        ("I'm extremely sad", 2, "Contraction self + amplifier"),
    ]

    for sentence, emo_idx, description in test_cases:
        words = sentence.lower().split()
        coeff = get_context_coefficient(words, emo_idx)
        combined = _find_combined_context(words, emo_idx)
        q = is_question(words)

        print(f"  \"{sentence}\"")
        print(f"    target word: '{words[emo_idx]}'")
        print(f"    combined context: {combined:.4f}  question: {q}")
        print(f"    FINAL MULTIPLIER: {coeff:.4f}")
        print(f"    ({description})")
        print()

    # Comparison demo
    print("=" * 70)
    print("COMPARISON: Same emotional word, different contexts")
    print("=" * 70)
    print()

    comparisons = [
        ("I am sad",            2, "self + present"),
        ("I was sad",           2, "self + past"),
        ("I would be sad",      3, "self + hypothetical"),
        ("I am very sad",       3, "self + present + intensifier"),
        ("a sad movie",         1, "article"),
        ("my sad life",         1, "personal possessive"),
        ("his sad story",       1, "other possessive"),
        ("this sad thing",      1, "near demonstrative"),
        ("that sad thing",      1, "far demonstrative"),
        ("are you sad",         2, "question"),
        ("just sad",            1, "diminisher"),
        ("extremely sad",       1, "amplifier"),
    ]

    # Sort by coefficient descending
    results = []
    for sentence, idx, label in comparisons:
        words = sentence.lower().split()
        coeff = get_context_coefficient(words, idx)
        results.append((coeff, sentence, label))

    results.sort(reverse=True)

    for coeff, sentence, label in results:
        bar = "#" * int(coeff * 20)
        print(f"  {coeff:6.3f}x  {bar:<60s}  \"{sentence}\" ({label})")

    print()
    print("1.0x = baseline (no context modifier)")
    print(">1.0 = amplified (personal, present, self-reference, intensifier)")
    print("<1.0 = dampened (question, article, past, hypothetical, diminisher)")
    print(f"Cap: {_COEFF_CAP}x  Floor: {_COEFF_FLOOR}x")
