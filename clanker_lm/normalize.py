"""Small deterministic normalization and morphology helpers.

This is intentionally not a statistical tagger.  It supplies only the stable
surface mechanics needed by the semantic frame parser and realizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator, List, Mapping, Sequence, Tuple

from .models import Gender, GrammaticalNumber


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?|[^\w\s]", re.UNICODE)
_ALIAS_RE = re.compile(r"[^a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class LexToken:
    surface: str
    norm: str
    index: int


CONTRACTIONS: Mapping[str, Tuple[str, ...]] = {
    "i'm": ("i", "am"),
    "im": ("i", "am"),
    "i've": ("i", "have"),
    "ive": ("i", "have"),
    "i'd": ("i", "would"),
    "id": ("i", "would"),
    "i'll": ("i", "will"),
    "ill": ("i", "will"),
    "you're": ("you", "are"),
    "youre": ("you", "are"),
    "you've": ("you", "have"),
    "youve": ("you", "have"),
    "you'd": ("you", "would"),
    "youd": ("you", "would"),
    "you'll": ("you", "will"),
    "youll": ("you", "will"),
    "he's": ("he", "is"),
    "hes": ("he", "is"),
    "she's": ("she", "is"),
    "shes": ("she", "is"),
    "it's": ("it", "is"),
    "its": ("it", "is"),
    "we're": ("we", "are"),
    "they're": ("they", "are"),
    "theyre": ("they", "are"),
    "what's": ("what", "is"),
    "whats": ("what", "is"),
    "who's": ("who", "is"),
    "whos": ("who", "is"),
    "where's": ("where", "is"),
    "wheres": ("where", "is"),
    "when's": ("when", "is"),
    "whens": ("when", "is"),
    "why's": ("why", "is"),
    "whys": ("why", "is"),
    "how's": ("how", "is"),
    "hows": ("how", "is"),
    "isn't": ("is", "not"),
    "isnt": ("is", "not"),
    "aren't": ("are", "not"),
    "arent": ("are", "not"),
    "wasn't": ("was", "not"),
    "wasnt": ("was", "not"),
    "weren't": ("were", "not"),
    "werent": ("were", "not"),
    "don't": ("do", "not"),
    "dont": ("do", "not"),
    "doesn't": ("does", "not"),
    "doesnt": ("does", "not"),
    "didn't": ("did", "not"),
    "didnt": ("did", "not"),
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
    "let's": ("let", "us"),
    "lets": ("let", "us"),
}


DETERMINERS = frozenset(
    {"a", "an", "the", "this", "that", "these", "those", "some", "any"}
)
POSSESSIVES = frozenset({"my", "your", "his", "her", "our", "their"})
FIRST_PERSON = frozenset({"i", "me", "myself", "we", "us", "ourselves"})
SECOND_PERSON = frozenset({"you", "yourself", "yourselves"})
FEMALE_PRONOUNS = frozenset({"she", "her", "hers"})
MALE_PRONOUNS = frozenset({"he", "him", "his"})
PLURAL_PRONOUNS = frozenset({"they", "them", "theirs"})
OBJECT_PRONOUNS = frozenset({"it", "this", "that"})


# canonical relation -> (aliases, gender, number)
_RELATION_ROWS = {
    "mother": ({"mother", "mom", "mum", "mama"}, Gender.FEMALE, GrammaticalNumber.SINGULAR),
    "father": ({"father", "dad", "papa"}, Gender.MALE, GrammaticalNumber.SINGULAR),
    "sister": ({"sister", "sis"}, Gender.FEMALE, GrammaticalNumber.SINGULAR),
    "brother": ({"brother", "bro"}, Gender.MALE, GrammaticalNumber.SINGULAR),
    "daughter": ({"daughter"}, Gender.FEMALE, GrammaticalNumber.SINGULAR),
    "son": ({"son"}, Gender.MALE, GrammaticalNumber.SINGULAR),
    "wife": ({"wife"}, Gender.FEMALE, GrammaticalNumber.SINGULAR),
    "husband": ({"husband"}, Gender.MALE, GrammaticalNumber.SINGULAR),
    "girlfriend": ({"girlfriend"}, Gender.FEMALE, GrammaticalNumber.SINGULAR),
    "boyfriend": ({"boyfriend"}, Gender.MALE, GrammaticalNumber.SINGULAR),
    "aunt": ({"aunt"}, Gender.FEMALE, GrammaticalNumber.SINGULAR),
    "uncle": ({"uncle"}, Gender.MALE, GrammaticalNumber.SINGULAR),
    "grandmother": ({"grandmother", "grandma"}, Gender.FEMALE, GrammaticalNumber.SINGULAR),
    "grandfather": ({"grandfather", "grandpa"}, Gender.MALE, GrammaticalNumber.SINGULAR),
    "cousin": ({"cousin"}, Gender.UNKNOWN, GrammaticalNumber.SINGULAR),
    "friend": ({"friend", "bestie", "buddy", "pal", "homie"}, Gender.UNKNOWN, GrammaticalNumber.SINGULAR),
    "boss": ({"boss", "manager", "supervisor"}, Gender.UNKNOWN, GrammaticalNumber.SINGULAR),
    "coworker": ({"coworker", "colleague"}, Gender.UNKNOWN, GrammaticalNumber.SINGULAR),
    "teacher": ({"teacher"}, Gender.UNKNOWN, GrammaticalNumber.SINGULAR),
    "partner": ({"partner", "spouse"}, Gender.UNKNOWN, GrammaticalNumber.SINGULAR),
    "parents": ({"parents"}, Gender.NEUTRAL, GrammaticalNumber.PLURAL),
    "siblings": ({"siblings"}, Gender.NEUTRAL, GrammaticalNumber.PLURAL),
    "friends": ({"friends"}, Gender.NEUTRAL, GrammaticalNumber.PLURAL),
    "children": ({"children", "kids"}, Gender.NEUTRAL, GrammaticalNumber.PLURAL),
}
RELATION_ALIASES: dict[str, str] = {}
RELATION_INFO: dict[str, tuple[Gender, GrammaticalNumber]] = {}
for _canonical, (_aliases, _gender, _number) in _RELATION_ROWS.items():
    RELATION_INFO[_canonical] = (_gender, _number)
    for _alias in _aliases:
        RELATION_ALIASES[_alias] = _canonical


BODY_PARTS = frozenset(
    {
        "head",
        "stomach",
        "tummy",
        "belly",
        "back",
        "arm",
        "leg",
        "hand",
        "foot",
        "feet",
        "chest",
        "heart",
        "throat",
        "tooth",
        "teeth",
        "neck",
        "knee",
        "shoulder",
        "ear",
        "ears",
        "eye",
        "eyes",
    }
)

CASUAL_MARKERS = frozenset(
    {
        "bruh",
        "bro",
        "dude",
        "fam",
        "bestie",
        "lol",
        "lmao",
        "lmfao",
        "haha",
        "omg",
        "fr",
        "ngl",
        "tbh",
        "lowkey",
        "highkey",
        "deadass",
        "nocap",
        "yo",
    }
)
PROFANITY = frozenset(
    {
        "fuck",
        "fucking",
        "fucked",
        "shit",
        "shitty",
        "damn",
        "bitch",
        "ass",
        "pissed",
        "piss",
    }
)
REPETITION_MARKERS = frozenset({"again", "still", "always", "keeps", "repeatedly"})

TIME_SINGLE = frozenset(
    {
        "today",
        "yesterday",
        "tomorrow",
        "tonight",
        "now",
        "later",
        "recently",
        "lately",
        "already",
        "soon",
    }
)
TIME_UNITS = frozenset(
    {
        "morning",
        "afternoon",
        "evening",
        "night",
        "day",
        "week",
        "month",
        "year",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }
)

SEVERE_MARKERS = frozenset(
    {
        "dying",
        "dead",
        "death",
        "cancer",
        "hospital",
        "emergency",
        "critical",
        "overdose",
        "suicide",
        "suicidal",
        "killed",
        "murdered",
        "missing",
        "terminal",
        "unconscious",
        "bleeding",
        "stroke",
        "heartattack",
        "heart",
        "sick",
        "ill",
        "hurt",
        "injured",
        "pain",
    }
)
HIGH_SEVERITY_MARKERS = frozenset(
    {
        "dying",
        "dead",
        "death",
        "cancer",
        "hospital",
        "emergency",
        "critical",
        "overdose",
        "suicide",
        "suicidal",
        "killed",
        "murdered",
        "terminal",
        "unconscious",
        "bleeding",
        "stroke",
        "heartattack",
    }
)

AUXILIARIES = frozenset(
    {
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
    }
)
MODALS = frozenset({"can", "could", "will", "would", "shall", "should", "may", "might", "must"})
COPULAS = frozenset({"am", "is", "are", "was", "were", "be", "been", "being"})
DO_AUX = frozenset({"do", "does", "did"})
HAVE_AUX = frozenset({"have", "has", "had"})


_IRREGULAR = {
    "am": ("be", "was", "is"),
    "is": ("be", "was", "is"),
    "are": ("be", "were", "are"),
    "was": ("be", "was", "is"),
    "were": ("be", "were", "are"),
    "been": ("be", "was", "is"),
    "bought": ("buy", "bought", "buys"),
    "buy": ("buy", "bought", "buys"),
    "buys": ("buy", "bought", "buys"),
    "left": ("leave", "left", "leaves"),
    "leave": ("leave", "left", "leaves"),
    "leaves": ("leave", "left", "leaves"),
    "went": ("go", "went", "goes"),
    "go": ("go", "went", "goes"),
    "goes": ("go", "went", "goes"),
    "gave": ("give", "gave", "gives"),
    "give": ("give", "gave", "gives"),
    "gives": ("give", "gave", "gives"),
    "saw": ("see", "saw", "sees"),
    "see": ("see", "saw", "sees"),
    "sees": ("see", "saw", "sees"),
    "ate": ("eat", "ate", "eats"),
    "eat": ("eat", "ate", "eats"),
    "eats": ("eat", "ate", "eats"),
    "made": ("make", "made", "makes"),
    "make": ("make", "made", "makes"),
    "makes": ("make", "made", "makes"),
    "took": ("take", "took", "takes"),
    "take": ("take", "took", "takes"),
    "takes": ("take", "took", "takes"),
    "got": ("get", "got", "gets"),
    "get": ("get", "got", "gets"),
    "gets": ("get", "got", "gets"),
    "found": ("find", "found", "finds"),
    "find": ("find", "found", "finds"),
    "finds": ("find", "found", "finds"),
    "sent": ("send", "sent", "sends"),
    "send": ("send", "sent", "sends"),
    "sends": ("send", "sent", "sends"),
    "wrote": ("write", "wrote", "writes"),
    "write": ("write", "wrote", "writes"),
    "writes": ("write", "wrote", "writes"),
    "broke": ("break", "broke", "breaks"),
    "broken": ("break", "broke", "breaks"),
    "break": ("break", "broke", "breaks"),
    "breaks": ("break", "broke", "breaks"),
    "did": ("do", "did", "does"),
    "done": ("do", "did", "does"),
    "does": ("do", "did", "does"),
    "had": ("have", "had", "has"),
    "has": ("have", "had", "has"),
    "said": ("say", "said", "says"),
    "say": ("say", "said", "says"),
    "says": ("say", "said", "says"),
    "told": ("tell", "told", "tells"),
    "tell": ("tell", "told", "tells"),
    "tells": ("tell", "told", "tells"),
    "hurt": ("hurt", "hurt", "hurts"),
    "hurts": ("hurt", "hurt", "hurts"),
    "felt": ("feel", "felt", "feels"),
    "feel": ("feel", "felt", "feels"),
    "feels": ("feel", "felt", "feels"),
    "thought": ("think", "thought", "thinks"),
    "think": ("think", "thought", "thinks"),
    "thinks": ("think", "thought", "thinks"),
    "knew": ("know", "knew", "knows"),
    "know": ("know", "knew", "knows"),
    "knows": ("know", "knew", "knows"),
    "ran": ("run", "ran", "runs"),
    "run": ("run", "ran", "runs"),
    "runs": ("run", "ran", "runs"),
    "came": ("come", "came", "comes"),
    "come": ("come", "came", "comes"),
    "comes": ("come", "came", "comes"),
    "paid": ("pay", "paid", "pays"),
    "pay": ("pay", "paid", "pays"),
    "pays": ("pay", "paid", "pays"),
    "spoke": ("speak", "spoke", "speaks"),
    "speak": ("speak", "spoke", "speaks"),
    "speaks": ("speak", "spoke", "speaks"),
    "won": ("win", "won", "wins"),
    "win": ("win", "won", "wins"),
    "wins": ("win", "won", "wins"),
    "lost": ("lose", "lost", "loses"),
    "lose": ("lose", "lost", "loses"),
    "loses": ("lose", "lost", "loses"),
    "lied": ("lie", "lied", "lies"),
    "lie": ("lie", "lied", "lies"),
    "lies": ("lie", "lied", "lies"),
    "pissed": ("piss", "pissed", "pisses"),
    "piss": ("piss", "pissed", "pisses"),
    "pisses": ("piss", "pissed", "pisses"),
    "upset": ("upset", "upset", "upsets"),
    "upsets": ("upset", "upset", "upsets"),
    "hit": ("hit", "hit", "hits"),
    "hits": ("hit", "hit", "hits"),
    "killed": ("kill", "killed", "kills"),
    "kills": ("kill", "killed", "kills"),
    "loved": ("love", "loved", "loves"),
    "loves": ("love", "loved", "loves"),
    "hated": ("hate", "hated", "hates"),
    "hates": ("hate", "hated", "hates"),
}

# Forms not covered by inflection heuristics but useful to the first semantic domain.
_BASE_VERBS = {
    "answer",
    "apologize",
    "arrive",
    "ask",
    "believe",
    "calculate",
    "call",
    "cause",
    "cheat",
    "close",
    "cry",
    "decide",
    "drive",
    "enter",
    "explain",
    "fix",
    "happen",
    "help",
    "hit",
    "kill",
    "learn",
    "live",
    "love",
    "move",
    "need",
    "open",
    "piss",
    "purchase",
    "remember",
    "reply",
    "score",
    "start",
    "stop",
    "study",
    "upset",
    "use",
    "visit",
    "want",
    "work",
}

PHRASAL_VERBS = {
    ("piss", "off"): "piss_off",
    ("pass", "away"): "die",
    ("break", "down"): "break_down",
    ("calm", "down"): "calm_down",
    ("give", "up"): "give_up",
    ("figure", "out"): "figure_out",
    ("find", "out"): "find_out",
    ("show", "up"): "show_up",
    ("pick", "up"): "pick_up",
}

MOTION_VERBS = frozenset({"go", "leave", "come", "arrive", "travel", "move", "run", "drive", "visit", "enter"})
TRANSFER_VERBS = frozenset({"give", "send", "tell", "show", "hand", "pay", "write"})
VOLITIONAL_VERBS = frozenset(
    {
        "buy",
        "leave",
        "go",
        "give",
        "send",
        "write",
        "enter",
        "visit",
        "call",
        "apologize",
        "help",
        "cheat",
        "hit",
        "piss_off",
        "say",
        "tell",
        "decide",
        "use",
        "make",
    }
)
PHYSICAL_EVENT_VERBS = frozenset({"break", "break_down", "fall", "happen", "die", "stop", "start", "hurt"})


def tokenize(text: str) -> Tuple[LexToken, ...]:
    """Tokenize and expand common contractions while retaining surface forms."""

    output: List[LexToken] = []
    for raw in _TOKEN_RE.findall(text.replace("’", "'")):
        lower = raw.lower()
        expanded = CONTRACTIONS.get(lower)
        if expanded:
            for item in expanded:
                output.append(LexToken(surface=item, norm=item, index=len(output)))
        else:
            output.append(LexToken(surface=raw, norm=lower, index=len(output)))
    return tuple(output)


def words(tokens: Sequence[LexToken]) -> Tuple[str, ...]:
    return tuple(token.norm for token in tokens if token.norm not in {"?", "!", ".", ",", ";", ":"})


def split_sentences(text: str) -> Tuple[str, ...]:
    parts = [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]
    return tuple(parts or ([text.strip()] if text.strip() else []))


def normalize_alias(text: str) -> str:
    cleaned = _ALIAS_RE.sub(" ", text.lower()).strip()
    pieces = [piece for piece in cleaned.split() if piece not in DETERMINERS]
    return " ".join(pieces)


def canonical_relation(word: str) -> str | None:
    return RELATION_ALIASES.get(normalize_alias(word))


def relation_features(relation: str) -> tuple[Gender, GrammaticalNumber]:
    return RELATION_INFO.get(
        relation, (Gender.UNKNOWN, GrammaticalNumber.SINGULAR)
    )


def detect_time_phrase(items: Sequence[str]) -> tuple[str | None, tuple[int, int] | None]:
    """Return a normalized time phrase and its half-open token span."""

    for index, token in enumerate(items):
        if token in TIME_SINGLE:
            return token, (index, index + 1)
        if token in {"last", "this", "next"} and index + 1 < len(items):
            if items[index + 1] in TIME_UNITS:
                return f"{token} {items[index + 1]}", (index, index + 2)
        if token in {"at", "on"} and index + 1 < len(items):
            next_token = items[index + 1]
            if next_token in TIME_UNITS or next_token.isdigit():
                end = index + 2
                if end < len(items) and items[end] in {"am", "pm"}:
                    end += 1
                return " ".join(items[index:end]), (index, end)
    return None, None


def lemmatize_verb(word: str) -> str:
    token = normalize_alias(word).replace(" ", "")
    if token in _IRREGULAR:
        return _IRREGULAR[token][0]
    if token.endswith("ied") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if stem.endswith("mak") or stem.endswith("tak") or stem.endswith("giv"):
            stem += "e"
        return stem
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if stem.endswith("i"):
            return stem[:-1] + "y"
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if stem.endswith("at") or stem.endswith("iz") or stem.endswith("us"):
            stem += "e"
        return stem
    if token.endswith("es") and len(token) > 4:
        possible = token[:-2]
        if possible in _BASE_VERBS:
            return possible
    if token.endswith("s") and len(token) > 3:
        possible = token[:-1]
        if possible in _BASE_VERBS or possible in {row[0] for row in _IRREGULAR.values()}:
            return possible
    return token


def is_known_verb(word: str) -> bool:
    token = normalize_alias(word).replace(" ", "")
    if token in _IRREGULAR or token in _BASE_VERBS:
        return True
    lemma = lemmatize_verb(token)
    return lemma in _BASE_VERBS or lemma in {row[0] for row in _IRREGULAR.values()}



def infer_tense(word: str) -> str:
    """Infer present/past from a lexical verb surface form."""

    lower = word.lower()
    irregular = _IRREGULAR.get(lower)
    if irregular is not None and lower == irregular[1]:
        return "past"
    if lower.endswith("ed") and len(lower) > 3:
        return "past"
    return "present"

def past_tense(lemma: str) -> str:
    lemma = lemmatize_verb(lemma)
    for _, (base, past, _) in _IRREGULAR.items():
        if base == lemma:
            return past
    if lemma.endswith("e"):
        return lemma + "d"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ied"
    return lemma + "ed"


def third_person(lemma: str) -> str:
    lemma = lemmatize_verb(lemma)
    for _, (base, _, third) in _IRREGULAR.items():
        if base == lemma:
            return third
    if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        return lemma + "es"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ies"
    return lemma + "s"


def choose_article(noun_phrase: str) -> str:
    stripped = normalize_alias(noun_phrase)
    if not stripped:
        return "a"
    return "an" if stripped[0] in "aeiou" else "a"


def clean_phrase(items: Iterable[str]) -> str:
    values = [item for item in items if item not in {",", ".", "!", "?", ";", ":"}]
    text = " ".join(values).strip()
    return re.sub(r"\s+", " ", text)


def title_if_name(surface: str) -> str:
    if not surface:
        return surface
    if surface.lower() in RELATION_ALIASES or surface.lower() in DETERMINERS:
        return surface.lower()
    return surface[0].upper() + surface[1:] if surface[0].isalpha() else surface
