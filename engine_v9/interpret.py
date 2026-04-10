"""V9 Interpret Context — pragmatic reinterpretation of word roles.

Ported from V8's interpret_context (Stage 2.5). Runs AFTER word classification,
BEFORE composition. Modifies roles and computes context flags.

Features:
  1. Discourse markers: "no we good" → "no" becomes FILLER (affirm, not negate)
  2. Expletive-as-intensifier: "shit you are right" → "shit" becomes AMPLIFIER
  3. Register detection: CONVERSATIONAL / LITERARY / EXPOSITORY
  4. Counterfactual marking: "supposed to" → positive forces inverted
  5. Hard negator inversion: NEGATOR + strong positive → flip sign
  6. SOLVENT dissolution: casual register flips LIQUID atoms
  7. Sarcasm field: ironic onset + flat affect → sarcasm penalty
"""

from .word_classifier import WordRole
from .vocabulary import VOCABULARY
from .phase import get_phase, is_solvent


# ── Word sets ────────────────────────────────────────────────────

_DISCOURSE_AFFIRMS = frozenset({"no", "nah", "nope"})
_DISCOURSE_LOOKAHEAD_POS = frozenset({
    "good", "great", "fine", "cool", "yeah", "yep", "alright",
    "okay", "ok", "right", "thanks", "chill", "bet", "word",
    "cap", "fair", "true", "facts", "valid", "solid",
    "we", "its", "all",
})

_EXPLETIVE_WORDS = frozenset({
    "shit", "fuck", "damn", "hell", "crap", "ass", "god",
    "jesus", "christ", "holy", "dammit", "goddamn",
})

_INSTRUCTIONAL_CUES = frozenset({
    "please", "proceed", "navigate", "locate", "head",
    "report", "follow", "enter", "exit", "return",
    "submit", "register", "verify", "confirm",
})

_COUNTERFACTUAL_MARKERS = frozenset({
    "supposed", "wouldve", "couldve", "shouldve",
    "wouldnt", "couldnt", "shouldnt",
    "wished", "imagined",
})

_PAST_TRUST_VERBS = frozenset({
    "trusted", "believed", "thought", "assumed", "expected",
})


def interpret_context(word_roles, text=""):
    """Reinterpret word roles based on pragmatic context.

    Args:
        word_roles: list of WordRole from classify_sentence
        text: original text (for quote detection)

    Returns:
        dict with:
          roles: modified word_roles
          register: "CONVERSATIONAL" | "LITERARY" | "EXPOSITORY"
          register_dampener: float (1.0 = full force, 0.35 = heavy dampen)
          counterfactual: bool
          sarcasm_inversion: bool
          sarcasm_penalty: float
          solvent_active: bool
    """
    if not word_roles:
        return {
            "roles": word_roles,
            "register": "CONVERSATIONAL",
            "register_dampener": 1.0,
            "counterfactual": False,
            "sarcasm_inversion": False,
            "sarcasm_penalty": 0.0,
            "solvent_active": False,
        }

    roles = word_roles
    words = [r.word for r in roles]
    n = len(roles)

    # ── 1. Discourse markers ────────────────────────────────────
    for i, wr in enumerate(roles):
        if wr.word in _DISCOURSE_AFFIRMS and wr.role == "NEGATOR":
            lookahead = words[i+1:i+4]
            has_positive_ahead = any(w in _DISCOURSE_LOOKAHEAD_POS for w in lookahead)
            has_emo_pos = any(
                roles[j].force and roles[j].force[0] > 10
                for j in range(i+1, min(i+4, n))
            )
            if has_positive_ahead or has_emo_pos:
                wr.role = "FILLER"
                wr.base_role = "FILLER"

    # ── 2. Expletive-as-intensifier ─────────────────────────────
    _AFFIRM_WORDS = frozenset({
        "right", "true", "yes", "exactly", "correct", "agreed",
        "good", "great", "nice", "thanks", "thank",
    })
    if roles[0].word in _EXPLETIVE_WORDS:
        pos_ahead = sum(
            1 for j in range(1, min(6, n))
            if (roles[j].force and roles[j].force[0] > 10)
            or roles[j].word in _AFFIRM_WORDS
        )
        if pos_ahead > 0:
            roles[0].role = "AMPLIFIER"
            roles[0].force = None

    # ── 3. Register detection ───────────────────────────────────
    _CASUAL_SIGNALS = frozenset({
        "im", "ive", "youre", "dont", "cant", "wont",
        "gonna", "wanna", "gotta", "lol", "lmao", "bruh", "bro",
        "dude", "omg", "tbh", "ngl", "fr", "nah", "yeah", "yep",
        "hey", "yo", "haha", "ok", "okay",
    })
    _ARTICLES = frozenset({"the", "a", "an", "this", "these", "those"})
    _THIRD_PERSON = frozenset({
        "he", "she", "they", "it", "his", "her", "its", "their", "him", "them",
    })
    _FIRST_PERSON = frozenset({"i", "im", "ive", "my", "me", "we", "us", "our"})
    _LITERARY_VERBS = frozenset({
        "seized", "seizing", "walked", "strode", "gazed",
        "muttered", "exclaimed", "remarked", "observed",
        "whispered", "cried", "replied", "declared",
    })

    instructional_count = sum(1 for w in words if w in _INSTRUCTIONAL_CUES)
    casual_count = sum(1 for w in words if w in _CASUAL_SIGNALS)
    has_exclamation = any(w.endswith('!') for w in words)
    has_quotes = text.count('"') >= 2

    c_art = sum(1 for w in words if w in _ARTICLES)
    c_3rd = sum(1 for w in words if w in _THIRD_PERSON)
    c_1st = sum(1 for w in words if w in _FIRST_PERSON)
    c_lit_verb = sum(1 for w in words if w in _LITERARY_VERBS)
    past_count = sum(1 for w in words if w.endswith("ed") or w in {"was", "were", "had", "been"})

    dx = (c_art + c_3rd) / max(1, n)
    past_ratio = past_count / max(1, n)
    obs_score = dx + 0.3 * past_ratio + (0.1 if casual_count == 0 else 0)
    if c_lit_verb >= 1:
        obs_score += 0.2

    if instructional_count >= 1:
        register = "EXPOSITORY"
        register_dampener = 0.35
    elif casual_count >= 1 or has_exclamation or has_quotes:
        register = "CONVERSATIONAL"
        register_dampener = 1.0
    elif obs_score >= 0.50 and c_1st == 0 and casual_count == 0 and n >= 10:
        register = "LITERARY"
        register_dampener = 0.55
    else:
        register = "CONVERSATIONAL"
        register_dampener = 1.0

    # ── 4. Counterfactual marking ───────────────────────────────
    has_counterfactual = any(w in _COUNTERFACTUAL_MARKERS for w in words)
    has_past_trust = any(w in _PAST_TRUST_VERBS for w in words)

    # ── 5. Hard negator inversion ───────────────────────────────
    negators_consumed = set()
    for i, wr in enumerate(roles):
        if wr.role == "NEGATOR":
            for j in range(i+1, min(n, i+4)):
                jr = roles[j]
                f = jr.force or VOCABULARY.get(jr.word)
                if f and f[0] > 25:
                    jr.force = (-f[0], f[1], -f[2], f[3], f[4])
                    negators_consumed.add(i)
                    break
    for i in negators_consumed:
        roles[i].role = "FILLER"

    # ── 6. SOLVENT dissolution ──────────────────────────────────
    has_solvent = any(is_solvent(wr.word) for wr in roles)
    if has_solvent:
        for wr in roles:
            if get_phase(wr.word) == "LIQUID":
                f = wr.force or VOCABULARY.get(wr.word)
                if f and f[0] < -5:
                    flipped = (-f[0], f[1], abs(f[2]), f[3], abs(f[4]))
                    wr.force = flipped
    solvent_active = has_solvent

    # ── 7. Sarcasm field ────────────────────────────────────────
    _IRONIC_ONSETS = frozenset({
        "clearly", "oh", "wow", "sure", "right", "great",
        "nice", "yeah", "gee", "wonderful", "brilliant", "lovely",
    })
    _AMPLIFIERS = frozenset({
        "so", "really", "very", "super", "extremely", "absolutely",
        "honestly", "seriously", "genuinely", "totally",
    })
    first_word = words[0] if words else ""
    two_word = " ".join(words[:2]) if len(words) >= 2 else ""
    has_ironic_onset = first_word in _IRONIC_ONSETS or two_word in ("what a", "oh great", "oh cool", "oh nice")
    has_amplifier = any(w in _AMPLIFIERS for w in words) or any(w.endswith("!") for w in words)

    sarcasm_inversion = False
    sarcasm_penalty = 0.0

    if has_ironic_onset and not has_amplifier and not has_solvent:
        total_charge = 0
        pos_count = 0
        neg_count = 0
        for wr in roles:
            f = wr.force or VOCABULARY.get(wr.word)
            if f:
                total_charge += f[0]
                if f[0] > 0:
                    pos_count += 1
                elif f[0] < 0:
                    neg_count += 1
        if neg_count >= pos_count and total_charge < 0 and neg_count >= 1:
            sarcasm_inversion = True
            sarcasm_penalty = -15.0

    # 7b. Contrast sarcasm — DISABLED
    # Tested: fires on genuine sentences like "I finally found my lost ring"
    # because both positive and negative words coexist naturally.
    # The ironic onset detector (7) is more precise.
    # Re-enable when bonding layer can distinguish genuine contrast from sarcasm.

    # ── 8. Absence scope ──────────────────────────────────────────
    # "haven't felt happy" → "happy" is in absence scope → dampen
    # "without love" → "love" dampened
    _ABSENCE_MARKERS = frozenset({"without", "havent", "hasnt", "hadnt", "didnt"})
    _ABSENCE_FOLLOWERS = frozenset({"had", "been", "felt", "seen", "gotten", "experienced"})

    absence_scope = set()
    for idx, wr in enumerate(roles):
        if wr.word in _ABSENCE_MARKERS:
            if wr.word == "without":
                for j in range(idx + 1, min(idx + 4, n)):
                    absence_scope.add(j)
            elif idx + 1 < n and roles[idx + 1].word in _ABSENCE_FOLLOWERS:
                for j in range(idx + 2, min(idx + 7, n)):
                    absence_scope.add(j)

    # Apply absence scope: dampen absent words' charges
    for idx in absence_scope:
        if idx < n:
            wr = roles[idx]
            f = wr.force or VOCABULARY.get(wr.word)
            if f and f[0] < -10:
                wr.force = (int(f[0] * 0.2), int(f[1] * 0.3), int(f[2] * 0.3), f[3], f[4])
            elif f and f[0] > 10:
                # Absent positive → slight negative (missing something good = sad)
                wr.force = (int(-f[0] * 0.3), f[1], int(-f[2] * 0.2), f[3], f[4])

    # ── 9. Forced choice cancellation ───────────────────────────
    _CHOICE_VERBS = frozenset({"choose", "chose", "choosing", "decide", "pick", "picked"})
    forced_choice_scope = set()
    for idx, wr in enumerate(roles):
        if wr.word == "between":
            has_choice = any(roles[j].word in _CHOICE_VERBS for j in range(max(0, idx - 3), idx))
            if has_choice:
                for j in range(idx + 1, min(idx + 6, n)):
                    forced_choice_scope.add(j)
    for idx in forced_choice_scope:
        if idx < n:
            wr = roles[idx]
            if wr.force:
                wr.force = (0, 0, 0, 0, 0)  # cancel emotional charge in forced choice

    return {
        "roles": roles,
        "register": register,
        "register_dampener": register_dampener,
        "counterfactual": has_counterfactual,
        "past_trust": has_past_trust,
        "sarcasm_inversion": sarcasm_inversion,
        "sarcasm_penalty": sarcasm_penalty,
        "solvent_active": solvent_active,
        "absence_scope": absence_scope,
    }
