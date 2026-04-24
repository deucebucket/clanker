"""V3 Layer 1: Word Role Classifier.

Every word gets a STRUCTURAL ROLE based on position, not just dictionary
definition. "Give" in "give me a hug" is different from "give" in
"I gave my dog to my neighbor." Same word, different structural position
= different meaning.

Roles are assigned in two passes:
  Pass 1: Base classification from word sets + position overrides
  Pass 2: Fill neighbor information (left_role, right_role)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from engine.vocabulary import VOCABULARY


# ── Structural roles ──────────────────────────────────────────────
ROLES = [
    "INVERSION", "SURPRISE",
    "SUBMISSION",
    "POWER",
    "PULL_TOWARD", "PULL_AWAY", "PULL_RESOLVED",
    "SELF_REF", "OTHER_REF", "RELATION_REF",
    "TRANSFER", "ACQUIRE",
    "EMOTIONAL",
    "AMPLIFIER", "NEGATOR", "COMPRESSOR", "REGISTER_CASUAL", "TEMPORAL", "HEDGE",
    "CONNECTOR", "CHOPPER",
    "POSSESSION", "METHOD", "FINALITY", "PEACE",
    "FILLER", "NEUTRAL",
]

# ── Word sets for each role ───────────────────────────────────────
# These are BASE classifications. Position and neighbors can OVERRIDE.
ROLE_WORDS = {
    "SELF_REF": frozenset({
        "i", "me", "my", "myself", "im", "i'm", "ive", "i've",
        "mine", "id", "i'd", "ill", "i'll",
    }),
    "OTHER_REF": frozenset({
        "you", "your", "yours", "yourself", "youre", "youve", "youd", "youll",
        "they", "them", "their", "theyre", "theyve", "theyd", "theyll",
        "he", "him", "his", "hes", "she", "her", "hers", "shes",
        "it", "its",
        "we", "us", "our", "were", "weve",
        "someone", "somebody", "everyone", "everybody", "anyone", "anybody",
    }),
    "RELATION_REF": frozenset({
        "mom", "mother", "dad", "father", "parent", "parents",
        "brother", "sister", "son", "daughter", "child", "children",
        "kids", "kid",
        "family", "friend", "friends", "husband", "wife", "partner",
        "boyfriend", "girlfriend", "neighbor", "boss", "teacher",
        "boo", "bae", "fam", "bestie", "homie",
        "dog", "cat", "pet", "puppy", "kitten", "baby",
        "grandma", "grandpa", "grandmother", "grandfather",
        "uncle", "aunt", "cousin", "niece", "nephew",
        "fiancee", "fiance", "ex", "coworker",
    }),
    "TRANSFER": frozenset({
        "give", "gave", "giving", "leave", "left", "leaving",
        "hand", "pass", "passed", "send", "sent", "donate",
        "return", "returned", "distribute", "share", "shared",
    }),
    "ACQUIRE": frozenset({
        "buy", "bought", "buying", "get", "got", "getting",
        "find", "found", "finding", "take", "took", "taking",
        "order", "ordered", "search", "searched", "grab", "grabbed",
        "have", "had", "has", "holding", "held", "carry", "carrying",
    }),
    "AMPLIFIER": frozenset({
        "very", "really", "extremely", "absolutely", "totally",
        "completely", "incredibly", "deeply", "truly", "super",
        "hella", "so", "fucking", "freaking", "damn", "too",
        "everything", "everyone", "everybody", "everywhere",
        "all", "whole", "entire", "entirely",
    }),
    "NEGATOR": frozenset({
        "not", "no", "never", "nobody", "nothing", "nowhere",
        "neither", "nor", "none", "dont", "don't", "doesnt",
        "doesn't", "didnt", "didn't", "cant", "can't", "wont",
        "won't", "isnt", "isn't", "wasnt", "wasn't", "havent",
        "haven't", "shouldnt", "shouldn't", "wouldnt", "wouldn't",
        "stopped", "quit",
    }),
    "TEMPORAL": frozenset({
        "tonight", "tomorrow", "today", "soon", "now", "forever",
        "permanently", "anymore", "always", "never", "finally",
        "eventually", "lately", "recently", "still", "already",
        "morning", "evening", "night",
    }),
    "COMPRESSOR": frozenset({
        "only", "just", "merely", "barely", "simply", "hardly",
    }),
    "REGISTER_CASUAL": frozenset({
        "bruh", "bro", "dude", "fam", "bestie",
        "lol", "lmao", "lmfao", "rofl", "haha", "hahaha",
        "omg", "istg", "fr", "ngl", "tbh",
        "lowkey", "highkey", "deadass",
        "literally", "nocap",
    }),
    "HEDGE": frozenset({
        "maybe", "perhaps", "possibly", "probably", "potentially",
        "generally", "sometimes", "occasionally", "arguably",
        "seemingly", "apparently", "supposedly", "might", "could",
        "somewhat", "slightly", "guess", "suppose", "wonder",
    }),
    "CHOPPER": frozenset({
        "but", "however", "although", "though", "yet",
        "instead", "whereas", "nevertheless", "despite",
    }),
    "CONNECTOR": frozenset({
        "and", "or", "because", "since", "so", "then",
        "also", "plus", "while", "when", "if", "after", "before",
        "with", "without", "for", "from", "to", "into", "about",
    }),
    "POSSESSION": frozenset({
        "things", "stuff", "belongings", "possessions", "keys",
        "car", "phone", "clothes", "money", "wallet", "purse",
        "account", "passwords", "ring", "journal", "laptop", "computer",
        "plants", "guitar", "collection", "remote", "tv", "console",
        "house", "apartment", "room", "bed", "desk", "chair",
        "bag", "backpack", "shoes", "jacket", "hoodie",
        "food", "lunch", "dinner", "drink", "coffee",
        "bike", "skateboard", "headphones", "charger",
        "book", "notebook", "pen", "pencil",
    }),
    "METHOD": frozenset({
        "pills", "pill", "gun", "pistol", "rope", "bridge",
        "knife", "blade", "razor", "noose", "overdose", "poison",
        "ledge", "rail", "tracks", "height", "tower",
    }),
    "FINALITY": frozenset({
        "last", "final", "goodbye", "farewell", "bye",
        "done", "complete", "goodbyes",
        # "end" removed -- too liquid. "end of the table" = spatial.
        # "over", "through", "finished" removed -- too liquid
        # "over the weekend" = temporal. "it's over" = finality.
        # "through with this" = finality. "drove through" = movement.
    }),
    "PEACE": frozenset({
        "peace", "peaceful", "calm", "ready", "free",
        "relief", "relieved", "serene", "quiet", "rest",
        "accepted", "settled", "okay", "fine",
    }),
    "FILLER": frozenset({
        "um", "uh", "like", "just", "basically", "literally",
        "actually", "honestly", "well", "anyway", "anyways",
    }),
}


def _clean(word: str) -> str:
    """Strip punctuation and lowercase."""
    return word.lower().strip(".,!?;:'\"")


# ── WordRole dataclass ────────────────────────────────────────────

@dataclass
class WordRole:
    """A word with its classified structural role."""
    word: str
    role: str
    base_role: str          # role from word set (before position override)
    position: int           # index in sentence
    neighbors: tuple        # (left_role, right_role) or None at edges
    force: Optional[tuple] = None  # (dV, dA, dD, dU, dG) if EMOTIONAL


# ── Single-word classifier ────────────────────────────────────────

def classify_word(word: str, position: int, words: List[str],
                  roles_so_far: List[str]) -> str:
    """Classify a single word's structural role.

    Uses the word itself + its position + its neighbors to determine role.
    Position overrides dictionary classification when context demands it.
    """
    w = _clean(word)

    # Check each role set
    for role_name, word_set in ROLE_WORDS.items():
        if w in word_set:
            # -- Position-based overrides --

            # "just" before acquire verb = TEMPORAL ("just bought" = recently)
            if w == "just" and position + 1 < len(words):
                next_w = _clean(words[position + 1])
                if next_w in ROLE_WORDS.get("ACQUIRE", frozenset()):
                    return "TEMPORAL"

            # "still" is always TEMPORAL (persistence/freshness marker)
            if w == "still":
                return "TEMPORAL"

            # "never" is primarily NEGATOR (not TEMPORAL)
            if w == "never" and role_name == "TEMPORAL":
                continue  # skip TEMPORAL, let NEGATOR win

            # "fine" after SELF_REF = PEACE (minimization — "im fine")
            if w == "fine" and position > 0:
                prev_role = (roles_so_far[position - 1]
                             if position - 1 < len(roles_so_far) else None)
                if prev_role == "SELF_REF":
                    return "PEACE"

            # "last" + temporal word = TEMPORAL, not FINALITY
            # "last night" / "last week" / "last time" = temporal
            # BUT "my last night" / "his last day" = FINALITY (possessive before)
            _TEMPORAL_FOLLOWERS = frozenset({
                "night", "week", "month", "year", "day",
                "summer", "winter", "spring", "fall", "semester",
                "tuesday", "wednesday", "thursday", "friday",
                "saturday", "sunday", "monday",
            })
            if w == "last" and role_name == "FINALITY":
                if position + 1 < len(words):
                    next_w = _clean(words[position + 1])
                    prev_role = (roles_so_far[position - 1]
                                 if position > 0 and position - 1 < len(roles_so_far) else None)
                    # "my last" / "his last" = possessive → FINALITY stays
                    if next_w in _TEMPORAL_FOLLOWERS and prev_role not in ("SELF_REF", "POSSESSION", "OTHER_REF"):
                        return "TEMPORAL"

            # "so" before emotional/amplifier = AMPLIFIER, else CONNECTOR
            if w == "so" and role_name == "CONNECTOR":
                continue  # skip CONNECTOR, AMPLIFIER already matched first

            return role_name

    # "end" is liquid — "end it" = finality, "end of the table" = spatial
    # Only classify as FINALITY when followed by pronoun/blanket
    _END_FINALITY_FOLLOWERS = {"it", "this", "everything", "things", "all", "myself"}
    if w in ("end", "ending", "stopping", "finishing") and position + 1 < len(words):
        next_w = _clean(words[position + 1])
        if next_w in _END_FINALITY_FOLLOWERS:
            return "FINALITY"

    # Check if it's an emotional vocabulary word with significant V-force
    if w in VOCABULARY:
        force = VOCABULARY[w]
        if abs(force[0]) >= 15:  # |dV| >= 15 = emotionally significant
            return "EMOTIONAL"
        # Heavy neutral: high gravity but low valence. Still carries weight.
        # "adopted", "pregnant", "diagnosed" -- these matter even at dV=0.
        if abs(force[4]) >= 15:  # |dG| >= 15 = gravitationally significant
            return "EMOTIONAL"  # gets force attached, physics handles the rest

    return "NEUTRAL"


# ── Sentence classifier (two-pass) ───────────────────────────────

def classify_sentence(words: List[str]) -> List[WordRole]:
    """Classify all words in a sentence into structural roles.

    Two-pass:
      Pass 1: Base role classification left-to-right
      Pass 2: Fill in neighbor information (left_role, right_role)
    """
    cleaned = [_clean(w) for w in words]

    # Pass 1: Base role classification
    roles: List[WordRole] = []
    role_names: List[str] = []

    for i, word in enumerate(cleaned):
        role = classify_word(word, i, cleaned, role_names)
        role_names.append(role)

        force = None
        if role == "EMOTIONAL" and word in VOCABULARY:
            force = VOCABULARY[word]

        roles.append(WordRole(
            word=word,
            role=role,
            base_role=role,
            position=i,
            neighbors=(role_names[i - 1] if i > 0 else None, None),
            force=force,
        ))

    # Pass 2: Fill in right neighbors
    for i in range(len(roles)):
        left = roles[i - 1].role if i > 0 else None
        right = roles[i + 1].role if i + 1 < len(roles) else None
        roles[i].neighbors = (left, right)

    return roles

# -- Pull verb family (chase/pursue/flee variants) --
# These are gravitational verbs - the target has mass, the actor orbits
PULL_TOWARD = frozenset({
    'chase', 'chased', 'chasing',
    'pursue', 'pursued', 'pursuing',
    'hunt', 'hunted', 'hunting',
    'seek', 'sought', 'seeking',
    'track', 'tracked', 'tracking',
    'follow', 'followed', 'following',
    'stalk', 'stalked', 'stalking',
    'attract', 'attracted', 'attracting',
    'drawn',
})

PULL_AWAY = frozenset({
    'flee', 'fled', 'fleeing',
    'run', 'ran', 'running',
    'escape', 'escaped', 'escaping',
    'avoid', 'avoided', 'avoiding',
    'evade', 'evaded', 'evading',
    'hide', 'hid', 'hiding',
    'retreat', 'retreated', 'retreating',
})

PULL_RESOLVED = frozenset({
    'catch', 'caught', 'catching',
    'capture', 'captured', 'capturing',
    'corner', 'cornered', 'cornering',
    'trap', 'trapped', 'trapping',
    'lose', 'lost', 'losing',
    'miss', 'missed', 'missing',
    'free', 'freed', 'freeing',
})

# Add to ROLE_WORDS for classification
ROLE_WORDS["PULL_TOWARD"] = PULL_TOWARD
ROLE_WORDS["PULL_AWAY"] = PULL_AWAY
ROLE_WORDS["PULL_RESOLVED"] = PULL_RESOLVED

# -- Power verb family (use/control/command) --
# These redistribute D-axis. User has power, used has none.
POWER_VERBS = frozenset({
    'use', 'used', 'using', 'uses',
    'control', 'controlled', 'controlling',
    'command', 'commanded', 'commanding',
    'direct', 'directed', 'directing',
    'manage', 'managed', 'managing',
    'lead', 'led', 'leading',
    # drive/drove removed -- too ambiguous (driving a car vs driving someone)
    'manipulate', 'manipulated', 'manipulating',
    'exploit', 'exploited', 'exploiting',
})

SUBMISSION_VERBS = frozenset({
    'obey', 'obeyed', 'obeying',
    'serve', 'served', 'serving',
    'submit', 'submitted', 'submitting',
    'surrender', 'surrendered', 'surrendering',
    'comply', 'complied', 'complying',
    'yield', 'yielded', 'yielding',
})

INVERSION_VERBS = frozenset({
    'addicted', 'addiction', 'obsessed', 'obsession',
    'trapped', 'captive', 'enslaved', 'dependent',
    'hooked', 'consumed', 'possessed',
})

ROLE_WORDS["POWER"] = POWER_VERBS
ROLE_WORDS["SUBMISSION"] = SUBMISSION_VERBS
ROLE_WORDS["INVERSION"] = INVERSION_VERBS

# -- Surprise family (pattern interrupts) --
# Surprise is an A-spike + D-drop. Not V-directional.
# The content AFTER the surprise determines V.
SURPRISE_WORDS = frozenset({
    'surprised', 'shocking', 'shocked', 'stunned',
    'unexpected', 'unexpectedly', 'suddenly', 'whoa',
    'omg', 'seriously', 'unbelievable',
    'astonished', 'astounded', 'flabbergasted',
    'speechless', 'dumbfounded', 'blindsided',
})

ROLE_WORDS["SURPRISE"] = SURPRISE_WORDS
