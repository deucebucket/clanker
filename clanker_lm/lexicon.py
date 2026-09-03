"""Deterministic English lexicon and morphology used by Clanker-LM.

This is intentionally not a statistical tagger.  It contains the closed-class
words and high-value verb semantics needed to turn common conversational
English into predicate/argument frames.  Unknown content words remain usable as
entities or predicates through conservative morphology.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .model import EntityKind, Gender, GrammaticalNumber


@dataclass(frozen=True)
class Token:
    text: str
    norm: str
    index: int


TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?|[$€£]\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)


CONTRACTIONS: Dict[str, Tuple[str, ...]] = {
    "i'm": ("i", "am"),
    "im": ("i", "am"),
    "i've": ("i", "have"),
    "ive": ("i", "have"),
    "i'll": ("i", "will"),
    "ill": ("i", "will"),
    "i'd": ("i", "would"),
    "id": ("i", "would"),
    "you're": ("you", "are"),
    "youre": ("you", "are"),
    "you've": ("you", "have"),
    "youve": ("you", "have"),
    "you'll": ("you", "will"),
    "youll": ("you", "will"),
    "you'd": ("you", "would"),
    "youd": ("you", "would"),
    "he's": ("he", "is"),
    "hes": ("he", "is"),
    "she's": ("she", "is"),
    "shes": ("she", "is"),
    "it's": ("it", "is"),
    "that's": ("that", "is"),
    "thats": ("that", "is"),
    "what's": ("what", "is"),
    "whats": ("what", "is"),
    "who's": ("who", "is"),
    "whos": ("who", "is"),
    "where's": ("where", "is"),
    "wheres": ("where", "is"),
    "how's": ("how", "is"),
    "hows": ("how", "is"),
    "there's": ("there", "is"),
    "theres": ("there", "is"),
    "we're": ("we", "are"),
    "they're": ("they", "are"),
    "theyre": ("they", "are"),
    "don't": ("do", "not"),
    "dont": ("do", "not"),
    "doesn't": ("does", "not"),
    "doesnt": ("does", "not"),
    "didn't": ("did", "not"),
    "didnt": ("did", "not"),
    "isn't": ("is", "not"),
    "isnt": ("is", "not"),
    "aren't": ("are", "not"),
    "arent": ("are", "not"),
    "wasn't": ("was", "not"),
    "wasnt": ("was", "not"),
    "weren't": ("were", "not"),
    "werent": ("were", "not"),
    "can't": ("can", "not"),
    "cant": ("can", "not"),
    "couldn't": ("could", "not"),
    "couldnt": ("could", "not"),
    "won't": ("will", "not"),
    "wont": ("will", "not"),
    "wouldn't": ("would", "not"),
    "wouldnt": ("would", "not"),
    "shouldn't": ("should", "not"),
    "shouldnt": ("should", "not"),
    "hasn't": ("has", "not"),
    "hasnt": ("has", "not"),
    "haven't": ("have", "not"),
    "havent": ("have", "not"),
    "hadn't": ("had", "not"),
    "hadnt": ("had", "not"),
    "could've": ("could", "have"),
    "would've": ("would", "have"),
    "should've": ("should", "have"),
    "let's": ("let", "us"),
    "lets": ("let", "us"),
}


PUNCTUATION = {".", "?", "!", ",", ";", ":", "…"}
ARTICLES = {"a", "an", "the"}
DETERMINERS = ARTICLES | {
    "this", "that", "these", "those", "some", "any", "each", "every",
    "another", "either", "neither", "both", "all", "no",
}
POSSESSIVES = {"my", "your", "his", "her", "its", "our", "their"}
AUX_DO = {"do", "does", "did"}
COPULAS = {"am", "is", "are", "was", "were", "be", "been", "being"}
AUX_HAVE = {"have", "has", "had"}
MODALS = {"can", "could", "will", "would", "shall", "should", "may", "might", "must"}
AUXILIARIES = AUX_DO | COPULAS | AUX_HAVE | MODALS
NEGATORS = {"not", "never", "no", "nobody", "nothing", "none", "neither", "nor"}
PREPOSITIONS = {
    "to", "from", "at", "in", "on", "into", "onto", "inside", "outside",
    "near", "beside", "behind", "under", "over", "through", "across", "around",
    "with", "without", "by", "for", "of", "about", "before", "after", "during",
    "since", "until", "toward", "towards", "against", "between", "among",
}
CLAUSE_MARKERS = {"because", "although", "though", "while", "when", "if", "unless", "since"}
CONJUNCTIONS = {"and", "or", "but", "yet", "so"}
FILLERS = {"um", "uh", "well", "like", "basically", "actually", "honestly", "anyway", "anyways"}
CASUAL_MARKERS = {
    "bruh", "bro", "dude", "fam", "bestie", "lol", "lmao", "lmfao", "haha",
    "omg", "fr", "ngl", "tbh", "lowkey", "highkey", "deadass", "nocap", "yo",
}
INTENSIFIERS = {"very", "really", "extremely", "absolutely", "totally", "so", "too", "super", "deeply"}

QUESTION_WORDS = {"who", "whom", "what", "when", "where", "why", "how", "which", "whose"}
YES_NO_STARTERS = AUXILIARIES


RELATIONS: Dict[str, Tuple[Gender, GrammaticalNumber, EntityKind]] = {
    "mom": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "mother": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "mama": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "dad": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "father": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "sister": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "brother": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "daughter": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "son": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "wife": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "husband": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "girlfriend": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "boyfriend": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "aunt": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "uncle": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "grandma": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "grandmother": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "grandpa": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "grandfather": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "niece": (Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "nephew": (Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "cousin": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "friend": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "bestie": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "boss": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "teacher": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "coworker": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "partner": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "parent": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "parents": (Gender.NEUTRAL, GrammaticalNumber.PLURAL, EntityKind.PERSON),
    "kids": (Gender.NEUTRAL, GrammaticalNumber.PLURAL, EntityKind.PERSON),
    "children": (Gender.NEUTRAL, GrammaticalNumber.PLURAL, EntityKind.PERSON),
    "family": (Gender.NEUTRAL, GrammaticalNumber.PLURAL, EntityKind.PERSON),
    "dog": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.THING),
    "cat": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.THING),
    "pet": (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.THING),
}

RELATION_CANONICAL: Dict[str, str] = {
    "mother": "mom", "mama": "mom", "father": "dad",
    "grandmother": "grandma", "grandfather": "grandpa",
}

FEMALE_NAMES = {
    "sarah", "mary", "jane", "anna", "emma", "olivia", "sophia", "mia", "ava",
    "marissa", "jenna", "selena", "aurora", "alice", "dorothy", "lisa", "jessica",
}
MALE_NAMES = {
    "john", "james", "michael", "david", "robert", "william", "daniel", "joseph",
    "jerry", "alex", "mark", "chris", "tom", "jack", "henry", "liam", "noah",
}

PRONOUN_FEATURES: Dict[str, Tuple[Optional[str], Gender, GrammaticalNumber, EntityKind]] = {
    "i": ("user", Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "me": ("user", Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "myself": ("user", Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "you": ("assistant", Gender.UNKNOWN, GrammaticalNumber.UNKNOWN, EntityKind.PERSON),
    "yourself": ("assistant", Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "he": (None, Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "him": (None, Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "himself": (None, Gender.MALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "she": (None, Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "her": (None, Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "herself": (None, Gender.FEMALE, GrammaticalNumber.SINGULAR, EntityKind.PERSON),
    "they": (None, Gender.NEUTRAL, GrammaticalNumber.PLURAL, EntityKind.PERSON),
    "them": (None, Gender.NEUTRAL, GrammaticalNumber.PLURAL, EntityKind.PERSON),
    "themselves": (None, Gender.NEUTRAL, GrammaticalNumber.PLURAL, EntityKind.PERSON),
    "it": (None, Gender.NEUTRAL, GrammaticalNumber.SINGULAR, EntityKind.THING),
    "itself": (None, Gender.NEUTRAL, GrammaticalNumber.SINGULAR, EntityKind.THING),
    "we": (None, Gender.NEUTRAL, GrammaticalNumber.PLURAL, EntityKind.PERSON),
    "us": (None, Gender.NEUTRAL, GrammaticalNumber.PLURAL, EntityKind.PERSON),
    "this": (None, Gender.NEUTRAL, GrammaticalNumber.SINGULAR, EntityKind.UNKNOWN),
    "that": (None, Gender.NEUTRAL, GrammaticalNumber.SINGULAR, EntityKind.UNKNOWN),
}


NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "a": 1, "an": 1,
}

DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
RELATIVE_TIMES = {
    "today", "yesterday", "tomorrow", "tonight", "now", "later", "earlier",
    "recently", "lately", "soon", "already", "still", "eventually", "again",
    "repeatedly",
}
TIME_UNITS = {"second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years"}
TIME_WORDS = DAYS | MONTHS | RELATIVE_TIMES | TIME_UNITS | {"morning", "afternoon", "evening", "night", "noon", "midnight"}

LOCATION_PREPOSITIONS = {"at", "in", "on", "inside", "outside", "near", "beside", "behind", "under", "over", "between", "among"}
TIME_PREPOSITIONS = {"at", "on", "in", "before", "after", "during", "since", "until"}
METHOD_PREPOSITIONS = {"with", "by", "through", "using", "via"}
DIRECTION_PREPOSITIONS = {"to", "into", "onto", "toward", "towards"}
SOURCE_PREPOSITIONS = {"from", "out"}


# Surface form -> lemma.  Include the verbs used by the vertical slice plus
# common conversational verbs.  Unknown regular forms are handled below.
IRREGULAR_LEMMAS: Dict[str, str] = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
    "has": "have", "had": "have",
    "does": "do", "did": "do", "done": "do",
    "went": "go", "gone": "go",
    "left": "leave",
    # ``-ied`` usually maps to ``-y`` (tried -> try), but these high-frequency
    # verbs retain a final silent e.  Keep the exceptions explicit rather than
    # guessing from suffix shape.
    "died": "die", "lied": "lie", "tied": "tie", "vied": "vie",
    "bought": "buy",
    "brought": "bring",
    "caught": "catch",
    "taught": "teach",
    "thought": "think",
    "felt": "feel",
    "found": "find",
    "got": "get", "gotten": "get",
    "gave": "give", "given": "give",
    "told": "tell",
    "said": "say",
    "saw": "see", "seen": "see",
    "heard": "hear",
    "made": "make",
    "took": "take", "taken": "take",
    "came": "come",
    "ran": "run",
    "ate": "eat", "eaten": "eat",
    "drank": "drink", "drunk": "drink",
    "drove": "drive", "driven": "drive",
    "wrote": "write", "written": "write",
    "read": "read",
    "spoke": "speak", "spoken": "speak",
    "broke": "break", "broken": "break",
    "fell": "fall", "fallen": "fall",
    "lost": "lose",
    "won": "win",
    "sent": "send",
    "paid": "pay",
    "met": "meet",
    "kept": "keep",
    "held": "hold",
    "built": "build",
    "sold": "sell",
    "stood": "stand",
    "sat": "sit",
    "slept": "sleep",
    "woke": "wake", "woken": "wake",
    "knew": "know", "known": "know",
    "meant": "mean",
    "hurt": "hurt",
    "hit": "hit",
    "put": "put",
    "cut": "cut",
    "let": "let",
    "upset": "upset",
    "set": "set",
    "spread": "spread",
    "cost": "cost",
    "belonged": "belong",
}

IRREGULAR_PAST: Dict[str, str] = {
    "be": "was", "have": "had", "do": "did", "go": "went", "leave": "left",
    "buy": "bought", "bring": "brought", "catch": "caught", "teach": "taught",
    "think": "thought", "feel": "felt", "find": "found", "get": "got",
    "give": "gave", "tell": "told", "say": "said", "see": "saw", "hear": "heard",
    "make": "made", "take": "took", "come": "came", "run": "ran", "eat": "ate",
    "drink": "drank", "drive": "drove", "write": "wrote", "read": "read",
    "speak": "spoke", "break": "broke", "fall": "fell", "lose": "lost",
    "win": "won", "send": "sent", "pay": "paid", "meet": "met", "keep": "kept",
    "hold": "held", "build": "built", "sell": "sold", "stand": "stood", "sit": "sat",
    "sleep": "slept", "wake": "woke", "know": "knew", "mean": "meant",
    "hurt": "hurt", "hit": "hit", "put": "put", "cut": "cut", "let": "let",
    "upset": "upset", "set": "set", "spread": "spread", "cost": "cost",
}

IRREGULAR_PARTICIPLE: Dict[str, str] = {
    **IRREGULAR_PAST,
    "be": "been", "do": "done", "go": "gone", "give": "given", "see": "seen",
    "take": "taken", "eat": "eaten", "drink": "drunk", "drive": "driven",
    "write": "written", "speak": "spoken", "break": "broken", "fall": "fallen",
    "wake": "woken", "know": "known", "get": "gotten",
}

KNOWN_VERBS: Set[str] = {
    "be", "have", "do", "go", "leave", "buy", "purchase", "bring", "catch", "teach",
    "think", "believe", "feel", "find", "get", "give", "tell", "say", "see", "hear",
    "make", "take", "come", "run", "eat", "drink", "drive", "write", "read", "speak",
    "break", "fall", "lose", "win", "send", "pay", "meet", "keep", "hold", "build",
    "sell", "stand", "sit", "sleep", "wake", "know", "mean", "hurt", "hit", "put",
    "cut", "let", "love", "hate", "like", "dislike", "want", "need", "choose",
    "decide", "plan", "try", "help", "fix", "work", "stop", "start", "finish",
    "open", "close", "enter", "exit", "arrive", "travel", "move", "live", "stay",
    "visit", "call", "text", "email", "ask", "answer", "explain", "show", "lend",
    "borrow", "wear", "use", "unlock", "calculate", "compute", "process", "sort",
    "create", "delete", "change", "turn", "look", "watch", "play", "study", "learn",
    "record", "schedule", "drop", "tie", "vie", "wait",
    "remember", "forget", "happen", "occur", "rain", "snow", "cost", "weigh",
    "measure", "seem", "become", "own", "apologize", "argue", "fight", "cheat",
    "lie", "laugh", "cry", "smile", "recover", "heal", "die", "kill", "save",
    "propose", "marry", "graduate", "pass", "fail", "receive", "order", "pick",
    "upset", "belong", "offer", "hand", "walk", "fly", "return", "collapse",
    "explode", "melt", "freeze", "function", "operate", "notice", "possess",
    "claim", "owe", "contain", "include", "need", "anger", "piss",
}

PHRASAL_VERBS: Dict[Tuple[str, str], str] = {
    ("piss", "off"): "anger",
    ("cheer", "up"): "cheer",
    ("calm", "down"): "calm",
    ("give", "up"): "quit",
    ("break", "up"): "separate",
    ("break", "down"): "break",
    ("find", "out"): "discover",
}
PHRASAL_PARTICLES = {particle for _, particle in PHRASAL_VERBS}

DITRANSITIVE_VERBS = {"give", "tell", "send", "show", "lend", "teach", "bring", "offer", "hand", "pay"}
MOVEMENT_VERBS = {"go", "come", "leave", "arrive", "travel", "move", "enter", "exit", "run", "drive", "walk", "fly", "visit", "return"}
VOLITIONAL_VERBS = {"leave", "go", "choose", "decide", "buy", "give", "send", "call", "text", "apologize", "visit", "quit", "start", "stop", "marry", "move"}
PURPOSE_LIKELY_VERBS = {"go", "come", "visit", "call", "text", "buy", "use", "open", "enter", "study", "work", "save"}
PHYSICAL_EVENT_VERBS = {"break", "fall", "rain", "snow", "happen", "occur", "collapse", "explode", "melt", "freeze", "stop", "fail"}
PROCESS_VERBS = {"calculate", "compute", "process", "sort", "build", "create", "work", "function", "operate"}
UNACCUSATIVE_VERBS = {"break", "fall", "die", "happen", "occur", "arrive", "collapse", "melt", "freeze"}
POSSESSION_VERBS = {"have", "own", "keep", "hold", "possess"}
COMMUNICATION_VERBS = {"say", "tell", "ask", "answer", "explain", "call", "text", "email", "write", "speak"}
PERCEPTION_VERBS = {"see", "hear", "watch", "notice", "feel"}
EXCLUSIVE_STATE_PREDICATES = {"be"}

ADJECTIVE_DIMENSIONS: Dict[str, str] = {
    "tall": "height", "short": "height", "high": "height", "low": "height",
    "old": "age", "young": "age", "fast": "speed", "slow": "speed",
    "heavy": "weight", "light": "weight", "far": "distance", "close": "distance",
    "long": "length", "wide": "width", "deep": "depth", "hot": "temperature",
    "cold": "temperature", "happy": "state", "sad": "state", "angry": "state",
    "sick": "health", "healthy": "health", "safe": "safety", "ready": "readiness",
}
ATTRIBUTE_NOUNS = {"color", "age", "height", "weight", "size", "name", "job", "price", "cost", "speed", "distance", "length", "width", "depth", "temperature", "owner", "reason"}
COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray", "grey", "brown", "silver", "gold", "beige", "teal", "navy"}

SOCIAL_QUESTIONS = {
    ("how", "are", "you"): "wellbeing_check",
    ("how", "is", "it", "going"): "wellbeing_check",
    ("what", "is", "up"): "greeting",
    ("what", "are", "you", "doing"): "activity_check",
}
GREETINGS = {"hi", "hello", "hey", "yo", "greetings", "sup"}


def normalize_apostrophes(text: str) -> str:
    return text.replace("’", "'").replace("`", "'")


def tokenize(text: str, *, include_punctuation: bool = True) -> List[Token]:
    """Tokenize and expand common contractions while retaining useful case."""

    text = normalize_apostrophes(text.strip())
    raw = TOKEN_RE.findall(text)
    expanded: List[Tuple[str, str]] = []
    for item in raw:
        norm = item.lower()
        parts = CONTRACTIONS.get(norm)
        if parts:
            # Preserve the original token's capitalization on the first part only.
            for idx, part in enumerate(parts):
                surface = part.capitalize() if idx == 0 and item[:1].isupper() else part
                expanded.append((surface, part))
        else:
            expanded.append((item, norm))
    tokens = [Token(surface, norm, idx) for idx, (surface, norm) in enumerate(expanded)]
    if not include_punctuation:
        tokens = [token for token in tokens if token.norm not in PUNCTUATION]
        tokens = [Token(token.text, token.norm, idx) for idx, token in enumerate(tokens)]
    return tokens


def clean_words(tokens: Sequence[Token]) -> List[str]:
    return [token.norm for token in tokens if token.norm not in PUNCTUATION]


def strip_discourse_prefix(tokens: Sequence[Token]) -> List[Token]:
    """Remove fillers/casual address that do not participate in the proposition."""

    items = list(tokens)
    while items and (items[0].norm in FILLERS or items[0].norm in CASUAL_MARKERS or items[0].norm in {",", ":"}):
        items.pop(0)
    return [Token(token.text, token.norm, idx) for idx, token in enumerate(items)]


def lemma(word: str) -> str:
    w = normalize_apostrophes(word.lower()).strip(".,!?;:'\"")
    if w in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[w]
    if w in KNOWN_VERBS:
        return w
    if w.endswith("ies") and len(w) > 4:
        candidate = w[:-3] + "y"
        if candidate in KNOWN_VERBS:
            return candidate
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        candidates = [stem, stem + "e"]
        if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
            candidates.append(stem[:-1])
        for candidate in candidates:
            if candidate in KNOWN_VERBS:
                return candidate
        return stem
    if w.endswith("ied") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ed") and len(w) > 3:
        stem = w[:-2]
        if stem.endswith("i"):
            stem = stem[:-1] + "y"
        if stem in KNOWN_VERBS:
            return stem
        if stem + "e" in KNOWN_VERBS:
            return stem + "e"
        if len(stem) > 2 and stem[-1:] == stem[-2:-1] and stem[:-1] in KNOWN_VERBS:
            return stem[:-1]
        return stem
    if w.endswith("es") and len(w) > 3:
        for candidate in (w[:-2], w[:-1]):
            if candidate in KNOWN_VERBS:
                return candidate
    if w.endswith("s") and len(w) > 3 and w[:-1] in KNOWN_VERBS:
        return w[:-1]
    return w


def detect_tense(surface_verb: str, auxiliary: Optional[str] = None) -> str:
    aux = (auxiliary or "").lower()
    word = surface_verb.lower()
    if aux == "did" or word in IRREGULAR_LEMMAS and IRREGULAR_LEMMAS[word] != word and word not in COPULAS:
        return "past"
    if word in {"was", "were", "had"} or word.endswith("ed"):
        return "past"
    if word in {"upset", "hurt", "hit", "put", "cut", "let", "read", "set", "spread", "cost"}:
        return "past"
    if aux == "will" or aux == "shall":
        return "future"
    return "present"


def past_form(base: str, *, plural_subject: bool = False) -> str:
    base = lemma(base)
    if base == "be":
        return "were" if plural_subject else "was"
    if base in IRREGULAR_PAST:
        return IRREGULAR_PAST[base]
    if base.endswith("e"):
        return base + "d"
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ied"
    # English consonant doubling depends on stress, which cannot be inferred
    # from spelling alone (``stop`` -> ``stopped`` but ``open`` -> ``opened``).
    # Keep the deterministic morphology honest with an explicit high-frequency
    # set rather than over-generalizing every CVC ending.
    if base in {
        "stop", "plan", "drop", "rob", "hop", "chat", "nod", "hug",
        "beg", "drag", "slip", "trip", "grab", "admit", "commit",
        "prefer", "occur",
    }:
        return base + base[-1] + "ed"
    return base + "ed"


def present_form(base: str, *, third_person_singular: bool = False, plural_subject: bool = False, first_person: bool = False) -> str:
    base = lemma(base)
    if base == "be":
        if first_person:
            return "am"
        if plural_subject:
            return "are"
        return "is" if third_person_singular else "are"
    if base == "have":
        return "has" if third_person_singular else "have"
    if base == "do":
        return "does" if third_person_singular else "do"
    if not third_person_singular:
        return base
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ies"
    if base.endswith(("s", "sh", "ch", "x", "z", "o")):
        return base + "es"
    return base + "s"


def participle_form(base: str) -> str:
    base = lemma(base)
    if base in IRREGULAR_PARTICIPLE:
        return IRREGULAR_PARTICIPLE[base]
    return past_form(base)


def is_probable_verb(word: str, previous: Optional[str] = None, following: Optional[str] = None) -> bool:
    w = word.lower()
    base = lemma(w)
    # Content words immediately governed by a determiner/possessive are noun
    # phrases in the supported grammar (``the work``, ``my plan``, ``the
    # meeting``), even when the same spelling can be a verb.
    if previous in DETERMINERS | POSSESSIVES and w not in AUXILIARIES:
        return False
    # A gerund between a determiner and a finite copula is normally a noun:
    # ``the meeting is at three``.  Without this guard, ``meeting`` is
    # incorrectly selected as the sentence predicate.
    if w.endswith("ing") and previous in DETERMINERS | POSSESSIVES and following in COPULAS:
        return False
    if w in AUXILIARIES or base in KNOWN_VERBS:
        return True
    if w in DETERMINERS | PREPOSITIONS | CONJUNCTIONS | NEGATORS | POSSESSIVES:
        return False
    if previous in AUX_DO | MODALS:
        return True
    if previous in AUX_HAVE and (w in IRREGULAR_LEMMAS or w.endswith(("ed", "en", "ing"))):
        return True
    # An auxiliary supplies independent grammatical evidence for an unknown
    # participle/progressive predicate.  Bare suffixes do not: ``did florb``
    # and ``is florbing`` are recoverable, while an isolated ``florbed`` is not
    # promoted to a verb solely because of its spelling.
    if previous in COPULAS and w.endswith(("ed", "en", "ing")) and len(w) > 4:
        return True
    # Regular finite and progressive forms have already succeeded through
    # ``base in KNOWN_VERBS`` above.  Unknown suffix-shaped tokens remain
    # unknown until syntax or lexical learning supplies stronger evidence.
    return False


def infer_name_gender(name: str) -> Gender:
    n = name.lower()
    if n in FEMALE_NAMES:
        return Gender.FEMALE
    if n in MALE_NAMES:
        return Gender.MALE
    return Gender.UNKNOWN


def parse_number(words: Sequence[str]) -> Optional[float]:
    if not words:
        return None
    compact = "".join(words).replace(",", "")
    compact = compact.lstrip("$€£")
    try:
        return float(compact) if "." in compact else int(compact)
    except ValueError:
        pass
    total = 0
    used = False
    for word in words:
        if word in NUMBER_WORDS:
            total += NUMBER_WORDS[word]
            used = True
        elif word == "hundred" and used:
            total *= 100
        elif word == "thousand" and used:
            total *= 1000
        elif word == "and":
            continue
        else:
            return None
    return total if used else None


def is_time_phrase(words: Sequence[str]) -> bool:
    clean = [word.lower() for word in words if word not in PUNCTUATION]
    if not clean:
        return False
    if any(word in TIME_WORDS for word in clean):
        return True
    if any(re.fullmatch(r"\d{1,2}(:\d{2})?(am|pm)?", word) for word in clean):
        return True
    if clean[0] in {"last", "next", "this"} and len(clean) > 1 and clean[1] in TIME_WORDS:
        return True
    return False


def is_clock_phrase(words: Sequence[str]) -> bool:
    """Return whether a phrase can denote a clock time after ``at``.

    This deliberately does not classify every bare number as time globally;
    the caller supplies the temporal/prepositional context.
    """

    clean = [word.lower() for word in words if word not in PUNCTUATION]
    if not clean:
        return False
    if len(clean) <= 3 and all(
        word in NUMBER_WORDS or word in {"hundred", "noon", "midnight", "am", "pm"}
        or re.fullmatch(r"\d{1,2}(:\d{2})?(am|pm)?", word)
        for word in clean
    ):
        return True
    return False


def normalize_phrase(words: Iterable[str]) -> str:
    return " ".join(word.lower().strip() for word in words if word and word not in PUNCTUATION).strip()


def relation_features(relation: str) -> Tuple[str, Gender, GrammaticalNumber, EntityKind]:
    normalized = RELATION_CANONICAL.get(relation.lower(), relation.lower())
    gender, number, kind = RELATIONS.get(normalized, (Gender.UNKNOWN, GrammaticalNumber.SINGULAR, EntityKind.PERSON))
    return normalized, gender, number, kind


def classify_unknown_noun(words: Sequence[str], *, preposition: Optional[str] = None) -> EntityKind:
    clean = [word.lower() for word in words if word not in DETERMINERS and word not in POSSESSIVES]
    if not clean:
        return EntityKind.UNKNOWN
    if is_time_phrase(clean):
        return EntityKind.TIME
    if preposition in LOCATION_PREPOSITIONS:
        return EntityKind.PLACE
    if clean[-1] in RELATIONS:
        return RELATIONS[clean[-1]][2]
    if clean[-1] in {"school", "home", "work", "store", "hospital", "office", "house", "room", "city", "town", "park", "restaurant", "dealership", "airport"}:
        return EntityKind.PLACE
    return EntityKind.THING
