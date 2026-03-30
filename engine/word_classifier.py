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
    "SELF_REF", "OTHER_REF", "RELATION_REF",
    "TRANSFER", "ACQUIRE",
    "EMOTIONAL",
    "AMPLIFIER", "NEGATOR", "TEMPORAL", "HEDGE",
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
        "you", "your", "yourself", "they", "them", "their",
        "he", "him", "his", "she", "her", "it", "its",
        "we", "us", "our",
    }),
    "RELATION_REF": frozenset({
        "mom", "mother", "dad", "father", "parent", "parents",
        "brother", "sister", "son", "daughter", "child", "children",
        "family", "friend", "friends", "husband", "wife", "partner",
        "boyfriend", "girlfriend", "neighbor", "boss", "teacher",
        "boo", "bae", "fam", "bestie", "homie",
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
    }),
    "AMPLIFIER": frozenset({
        "very", "really", "extremely", "absolutely", "totally",
        "completely", "incredibly", "deeply", "truly", "super",
        "hella", "so", "fucking", "freaking", "damn",
    }),
    "NEGATOR": frozenset({
        "not", "no", "never", "nobody", "nothing", "nowhere",
        "neither", "nor", "none", "dont", "don't", "doesnt",
        "doesn't", "didnt", "didn't", "cant", "can't", "wont",
        "won't", "isnt", "isn't", "wasnt", "wasn't", "havent",
        "haven't", "shouldnt", "shouldn't", "wouldnt", "wouldn't",
    }),
    "TEMPORAL": frozenset({
        "tonight", "tomorrow", "today", "soon", "now", "forever",
        "permanently", "anymore", "always", "never", "finally",
        "eventually", "lately", "recently", "still", "already",
        "morning", "evening", "night",
    }),
    "HEDGE": frozenset({
        "maybe", "perhaps", "possibly", "probably", "potentially",
        "generally", "sometimes", "occasionally", "arguably",
        "seemingly", "apparently", "supposedly", "might", "could",
        "somewhat", "slightly", "guess", "suppose", "wonder",
    }),
    "CHOPPER": frozenset({
        "but", "however", "although", "though", "yet",
        "instead", "whereas", "nevertheless",
    }),
    "CONNECTOR": frozenset({
        "and", "or", "because", "since", "so", "then",
        "also", "plus", "while", "when", "if", "after", "before",
    }),
    "POSSESSION": frozenset({
        "things", "stuff", "belongings", "possessions", "keys",
        "dog", "cat", "car", "phone", "clothes", "money",
        "account", "passwords", "ring", "journal", "laptop",
        "pet", "plants", "guitar", "collection",
    }),
    "METHOD": frozenset({
        "pills", "pill", "gun", "pistol", "rope", "bridge",
        "knife", "blade", "razor", "noose", "overdose", "poison",
        "ledge", "rail", "tracks", "height", "tower",
    }),
    "FINALITY": frozenset({
        "last", "final", "end", "goodbye", "farewell", "bye",
        "done", "finished", "over", "through", "complete",
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

            # "so" before emotional/amplifier = AMPLIFIER, else CONNECTOR
            if w == "so" and role_name == "CONNECTOR":
                continue  # skip CONNECTOR, AMPLIFIER already matched first

            return role_name

    # Check if it's an emotional vocabulary word with significant V-force
    if w in VOCABULARY:
        force = VOCABULARY[w]
        if abs(force[0]) > 15:  # |dV| > 15 = emotionally significant
            return "EMOTIONAL"

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
