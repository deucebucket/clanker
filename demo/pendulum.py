"""Sequential Pendulum Engine for the Clanker pipeline."""

import re
import math
from dataclasses import dataclass

from .shared import VADUG, VADU
from .forces import WORD_FORCES
from .morphemes import decompose_word, ROOTS, PREFIXES, SUFFIXES
from .pipeline_config import PipelineConfig
from .bookend import BookendParser
from .word_roles import WordRoleDetector
from .bigrams import BigramDetector


# Negation words flip the valence of the NEXT emotional word
NEGATORS = {"not", "don't", "didn't", "can't", "won't", "never", "no",
            "isn't", "aren't", "wasn't", "weren't", "hardly", "barely"}


# ---------------------------------------------------------------------------
# Intensity ramps — amplifiers as physical slopes (#72)
# A ramp affects multiple subsequent words with decaying multiplier.
# Replaces the old single-word INTENSIFIERS system.
# ---------------------------------------------------------------------------

@dataclass
class Ramp:
    """A physical slope that amplifies subsequent word forces."""
    multiplier: float      # how much amplification (1.0 = flat, 2.0 = cliff)
    length: int            # how many words the ramp affects (1-3)
    decay: float = 0.6     # each subsequent word gets decay * previous effect


RAMPS = {
    # Extreme amplifiers (steep ramp, long)
    "extremely": Ramp(1.6, 3),
    "absolutely": Ramp(1.7, 3),
    "utterly": Ramp(1.6, 2),
    "completely": Ramp(1.5, 2),
    "totally": Ramp(1.5, 2),

    # Strong amplifiers
    "very": Ramp(1.3, 2),
    "really": Ramp(1.35, 2),
    "so": Ramp(1.25, 2),
    "truly": Ramp(1.3, 2),
    "deeply": Ramp(1.35, 2),
    "incredibly": Ramp(1.5, 3),
    "insanely": Ramp(1.5, 2),
    "super": Ramp(1.4, 2),
    "pretty": Ramp(1.15, 1),

    # Mild amplifiers (gentle slope)
    "quite": Ramp(1.15, 1),
    "fairly": Ramp(1.1, 1),
    "rather": Ramp(1.15, 1),
    "somewhat": Ramp(0.8, 1),   # dampener, not amplifier
    "slightly": Ramp(0.7, 1),   # dampener
    "barely": Ramp(0.5, 1),     # heavy dampener
    "kinda": Ramp(0.8, 1),
    "sorta": Ramp(0.8, 1),

    # Superlatives (steep but short)
    "most": Ramp(1.5, 1),
    "least": Ramp(0.5, 1),
}

# Backward-compatible alias: old code that imports INTENSIFIERS still works.
# Maps each ramp word to its base multiplier (single-word view).
INTENSIFIERS = {word: ramp.multiplier for word, ramp in RAMPS.items()}


# -------------------------------------------------------------
# Idiom detection — multi-word expressions with fixed emotional meaning
# Each idiom: (tuple of words, v_force, a_force, d_force, u_force, label)
# The words are checked as a sliding window against previous_words + current
# -------------------------------------------------------------

IDIOMS = {
    # Confrontation / grievance
    ("bone", "to", "pick", "with"): (-45, +45, +35, +35, "confrontation"),
    ("fed", "up"):                (-30, +35, +20, +20, "frustrated/fed up"),
    ("shut", "up"):               (-35, +45, +35, +20, "hostile silencing"),
    ("pissed", "off"):            (-40, +50, +25, +25, "angry"),
    ("ticked", "off"):            (-30, +35, +20, +15, "irritated"),

    # Defeat / disappointment
    ("let", "down"):              (-30, +10, -20, +10, "disappointed"),
    ("give", "up"):               (-40, -15, -40, +10, "surrender"),
    ("gave", "up"):               (-35, +5, -40, +10, "defeat/surrender"),
    ("no", "way"):                (0, +40, +10, +15, "disbelief"),
    ("worn", "out"):              (-20, -15, -20, +5, "exhausted"),

    # Positive idioms
    ("piece", "of", "cake"):      (+30, -10, +30, -5, "easy/confident"),
    ("break", "a", "leg"):        (+25, +20, +15, 0, "good luck"),
    ("on", "fire"):               (+35, +45, +30, 0, "doing great"),
    ("knocked", "it", "out"):     (+40, +35, +30, 0, "nailed it"),
    ("over", "the", "moon"):      (+50, +40, +20, 0, +55, "metaphor: over the moon"),
    ("on", "top", "of", "the", "world"): (+50, +40, +35, 0, "elated"),

    # Anticipation / tension builders
    ("look", "forward"):          (+25, +20, +15, +10, "anticipation"),
    ("looking", "forward"):       (+25, +20, +15, +10, "anticipation"),
    ("can't", "wait"):            (+30, +35, +15, +20, "eager anticipation"),

    # Panic / crisis
    ("freak", "out"):             (-35, +60, -30, +40, "panic"),
    ("freaked", "out"):           (-40, +55, -30, +35, "panicked"),
    ("freaking", "out"):          (-40, +60, -30, +40, "panicking"),
    ("melt", "down"):             (-45, +50, -40, +35, "meltdown"),
    ("break", "down"):            (-45, +35, -35, +25, "collapse/cry"),
    ("end", "of", "the", "world"): (-50, +50, -50, +40, "catastrophizing"),

    # De-escalation
    ("calm", "down"):             (+10, -30, +10, -10, "de-escalation attempt"),
    ("take", "a", "breath"):      (+15, -25, +15, -10, "grounding"),
    ("it's", "okay"):             (+20, -15, +15, -5, "reassurance"),
    ("no", "worries"):            (+20, -20, +15, -5, "reassurance"),
    ("hang", "in", "there"):      (+15, +5, +10, +5, "encouragement"),

    # Death/grief idioms — override individual word scores (#41)
    ("passed", "away"):           (-55, +30, -35, +20, "death/loss"),
    ("passed", "on"):             (-50, +25, -30, +15, "death/loss"),
    ("rest", "in", "peace"):      (-30, -20, +20, 0, "memorial"),
    ("better", "place"):          (-15, -10, +15, 0, "consolation (often hollow)"),
    ("gone", "too", "soon"):      (-60, +35, -40, +20, "premature death"),
    ("taken", "from", "us"):      (-65, +40, -45, +25, "unjust death"),
    ("lost", "the", "battle"):    (-60, +30, -40, +20, "death after illness"),
    ("laid", "to", "rest"):       (-40, -15, +15, +5, "funeral/burial"),
    ("at", "peace", "now"):       (-20, -25, +20, 0, "consolation"),
    ("in", "our", "hearts"):      (+20, +10, +15, 0, "memorial positive"),
    ("deepest", "sympathy"):      (-30, -10, +10, +5, "condolence"),
    ("thoughts", "and", "prayers"): (-15, -10, +5, +5, "standard condolence"),

    # Threat/aggression idioms (#37)
    ("eat", "alive"):             (-40, +50, +40, +30, "threat/dominance"),
    ("rip", "apart"):             (-50, +55, +35, +40, "aggressive destruction"),
    ("beat", "up"):               (-55, +55, +30, +45, "physical threat"),
    ("mess", "up"):               (-35, +30, -15, +20, "ruin/damage"),
    ("screw", "over"):            (-50, +40, -20, +25, "betrayal"),
    ("throw", "shade"):           (-30, +35, +20, +15, "disrespect"),
    ("talk", "shit"):             (-45, +40, +15, +20, "gossip/insult"),
    ("lose", "it"):               (-40, +55, -30, +35, "emotional breakdown"),
    ("burn", "out"):              (-50, -20, -40, +10, "exhaustion"),

    # Direct threats (#37)
    ("knock", "out"):             (-45, +55, +35, +40, +20, "threat: knock out"),
    ("take", "down"):             (-40, +45, +30, +35, +15, "threat: take down"),
    ("shut", "down"):             (-35, +40, +30, +25, +10, "threat: shut down"),
    ("wipe", "out"):              (-50, +50, +30, +40, +15, "threat: wipe out"),
    ("take", "out"):              (-40, +45, +30, +35, +15, "threat: take out"),
    ("cut", "off"):               (-35, +30, +25, +20, +5, "threat: cut off"),
    ("throw", "down"):            (-40, +50, +30, +35, +20, "threat: throw down"),

    # Intimidation (#37)
    ("watch", "out"):             (-25, +35, +20, +30, +10, "intimidation: watch out"),
    ("back", "off"):              (-30, +40, +35, +25, +15, "intimidation: back off"),
    ("step", "up"):               (-15, +35, +30, +20, +15, "intimidation: step up"),
    ("bring", "it"):              (-10, +45, +35, +25, +20, "intimidation: bring it"),
    ("come", "at", "me"):         (-10, +50, +40, +30, +25, "intimidation: come at me"),

    # Emotional aggression (#37)
    ("tear", "apart"):            (-50, +50, +25, +35, -10, "aggression: tear apart"),
    ("rip", "off"):               (-40, +40, -15, +25, +5, "aggression: rip off/scam"),
    ("sell", "out"):              (-35, +30, -20, +20, -10, "aggression: sell out/betray"),
    ("stab", "in", "the", "back"): (-55, +45, -25, +35, -15, "aggression: betrayal"),
    ("throw", "under", "the", "bus"): (-50, +40, -20, +30, -15, "aggression: betrayal"),
    ("hang", "out", "to", "dry"): (-40, +30, -25, +25, -20, "aggression: abandonment"),
    ("leave", "hanging"):         (-30, +20, -20, +20, -15, "aggression: abandonment"),
    ("string", "along"):          (-35, +25, -15, +20, -10, "aggression: manipulation"),

    # Dismissal/contempt (#37)
    ("blow", "off"):              (-30, +25, +20, +15, +10, "contempt: blow off/dismiss"),
    ("write", "off"):             (-25, +15, +15, +10, 0, "contempt: write off"),
    ("brush", "off"):             (-25, +20, +20, +10, +5, "contempt: brush off"),
    ("look", "down", "on"):       (-30, +15, +30, +10, +20, "contempt: look down on"),
    ("talk", "down", "to"):       (-35, +20, +30, +15, +20, "contempt: condescend"),
    ("put", "down"):              (-40, +25, +25, +15, +10, "contempt: insult/belittle"),

    # Crisis / suicidal ideation idioms (6-element tuples with gravity)
    ("want", "to", "die"):        (-95, +35, -60, +70, -65, "crisis: want to die"),
    ("don't", "want", "to", "be", "here"): (-85, +25, -50, +60, -60, "crisis: don't want to be here"),
    ("don't", "want", "to", "be"): (-80, +25, -50, +55, -55, "crisis: don't want to be"),
    ("end", "it", "all"):         (-90, +30, -55, +65, -65, "crisis: end it all"),
    ("ending", "it", "all"):      (-90, +30, -55, +65, -65, "crisis: ending it all"),
    ("ending", "it"):             (-80, +25, -50, +55, -55, "crisis: ending it"),
    ("want", "to", "disappear"):  (-80, +20, -50, +50, -55, "crisis: want to disappear"),
    ("better", "off", "without", "me"): (-85, +25, -55, +55, -60, "crisis: better off without me"),
    ("can't", "go", "on"):        (-80, +20, -55, +50, -55, "crisis: can't go on"),
    ("the", "pain", "to", "stop"): (-70, +20, -45, +50, -50, "crisis: the pain to stop"),
    ("want", "it", "to", "stop"): (-60, +15, -40, +45, -45, "crisis: want it to stop"),
    ("want", "it", "to", "end"):  (-65, +20, -45, +50, -50, "crisis: want it to end"),
    ("no", "point"):              (-45, -10, -25, +10, -35, "crisis: no point"),
    ("what's", "the", "point"):   (-75, -5, -35, +18, -50, "crisis: what's the point"),
    ("the", "point", "of"):       (-40, -5, -20, +10, -30, "crisis: the point of"),
    ("nobody", "would", "care"):  (-75, +12, -40, +35, -48, "crisis: nobody would care"),
    ("nobody", "would", "miss"):  (-75, +12, -40, +35, -48, "crisis: nobody would miss"),
    ("would", "care"):            (-50, +10, -25, +20, -30, "crisis: would care (neg context)"),
    ("would", "miss"):            (-50, +10, -25, +20, -30, "crisis: would miss (neg context)"),
    ("a", "burden"):              (-75, +15, -38, +22, -50, "crisis: a burden"),
    ("i'm", "a", "burden"):       (-80, +15, -40, +25, -55, "crisis: I'm a burden"),
    ("tired", "of", "everything"): (-45, -15, -25, +10, -35, "crisis: tired of everything"),

    # Gravity metaphors — sinking/heavy
    ("weight", "of", "the", "world"): (-60, +30, -40, +30, -70, "metaphor: weight of the world"),
    ("heavy", "heart"):              (-45, +15, -20, +10, -50, "metaphor: heavy heart"),
    ("heart", "feels", "heavy"):     (-50, +20, -25, +15, -55, "metaphor: heart feels heavy"),
    ("heart", "feels", "so", "heavy"): (-50, +20, -25, +15, -55, "metaphor: heart feels so heavy"),
    ("feels", "so", "heavy"):        (-40, +15, -20, +10, -45, "metaphor: feels so heavy"),
    ("feels", "heavy"):              (-40, +15, -20, +10, -45, "metaphor: feels heavy"),
    ("sinking", "feeling"):          (-45, +25, -30, +20, -50, "metaphor: sinking feeling"),
    ("drowning", "in"):              (-50, +30, -35, +25, -55, "metaphor: drowning in"),
    ("crushed", "by"):               (-55, +25, -40, +20, -60, "metaphor: crushed by"),
    ("weighed", "down"):             (-45, +15, -25, +10, -50, "metaphor: weighed down"),
    ("buried", "under"):             (-50, +20, -35, +15, -55, "metaphor: buried under"),
    ("world", "on", "my", "shoulders"): (-55, +25, -35, +25, -65, "metaphor: world on shoulders"),
    ("carrying", "the", "weight"):   (-45, +20, -25, +15, -50, "metaphor: carrying weight"),
    ("rock", "bottom"):              (-60, +20, -45, +15, -70, "metaphor: rock bottom"),

    # Gravity metaphors — floating/rising
    ("walking", "on", "air"):        (+55, +25, +25, 0, +65, "metaphor: walking on air"),
    ("on", "cloud", "nine"):         (+55, +20, +20, 0, +60, "metaphor: cloud nine"),
    ("spirits", "lifted"):           (+40, +15, +20, -5, +50, "metaphor: spirits lifted"),
    ("spirits", "soaring"):          (+50, +25, +25, 0, +55, "metaphor: spirits soaring"),
    ("spirits", "are", "soaring"):   (+50, +25, +25, 0, +55, "metaphor: spirits are soaring"),
    ("light", "as", "a", "feather"): (+35, -10, +15, 0, +50, "metaphor: light as feather"),
    ("floating", "on"):              (+40, +15, +15, 0, +45, "metaphor: floating on"),
    ("weight", "lifted"):            (+45, -15, +25, -10, +55, "metaphor: weight lifted"),
    ("load", "off"):                 (+35, -10, +20, -10, +45, "metaphor: load off"),

    # Slang / Internet
    ("hits", "different"):           (+25, +20, +10, 0, +20, "slang: resonates emotionally"),
    ("no", "cap"):                   (+5, +10, +15, 0, +5, "slang: for real/honest"),
    ("living", "my", "best", "life"): (+45, +30, +20, 0, +35, "slang: thriving"),
    ("catch", "feelings"):           (+20, +25, -10, +10, +15, "slang: developing emotions"),
    ("low", "key"):                  (-5, -10, +5, 0, -5, "slang: subtly/secretly"),
    ("high", "key"):                 (+5, +15, +10, 0, +5, "slang: openly/obviously"),
    ("spill", "the", "tea"):        (+5, +25, +15, +10, +10, "slang: share gossip"),
    ("main", "character"):           (+25, +20, +25, 0, +25, "slang: protagonist energy"),
    ("villain", "arc"):              (-20, +25, +20, +10, +15, "slang: turning bad"),
    ("plot", "twist"):               (+5, +35, -5, +15, +10, "slang: unexpected turn"),

    # Sarcasm — engine CANNOT reliably detect sarcasm from text alone.
    # Most "sarcastic" phrases can also be genuine. Sarcasm detection
    # requires context/tone that text doesn't carry (Mehrabian: 7% words).
    # This is where the TRAINED MODEL should beat the engine — the model
    # can learn contextual sarcasm patterns the rule engine cannot.
    # Sarcasm idioms REMOVED after benchmark showed 3.5pp drop on SST-2.

    # Passive Aggressive / Workplace
    ("per", "my", "last", "email"):  (-20, +15, +25, +15, +10, "passive: per my last email"),
    ("as", "previously", "mentioned"): (-15, +10, +20, +10, +5, "passive: already said this"),
    ("going", "forward"):            (+5, +10, +15, +10, +5, "corporate: direction change"),
    ("circle", "back"):              (0, +5, +10, +5, 0, "corporate: revisit later"),
    ("above", "my", "pay", "grade"): (-15, +10, -20, +5, -10, "workplace: not my problem"),
    ("glass", "ceiling"):            (-25, +15, -20, +10, -15, "workplace: invisible barrier"),

    # Relationship
    ("we", "need", "to", "talk"):    (-35, +30, -15, +40, -25, "relationship: serious incoming"),
    ("it's", "not", "you"):          (-30, +20, +15, +15, -15, "relationship: breakup opening"),
    ("seeing", "someone"):           (+10, +20, +5, +10, +10, "relationship: dating"),
    ("catching", "feelings"):        (+15, +25, -10, +15, +15, "relationship: falling for someone"),
    ("on", "a", "break"):            (-25, +15, -15, +15, -15, "relationship: separation"),
    ("second", "chance"):            (+20, +15, +10, +10, +15, "relationship: forgiveness"),
    ("growing", "apart"):            (-30, +10, -10, +10, -20, "relationship: drifting"),

    # Physical / Embodied Emotion
    ("stomach", "in", "knots"):      (-35, +30, -20, +25, -25, "physical: anxiety"),
    ("lump", "in", "throat"):        (-30, +20, -15, +15, -20, "physical: about to cry"),
    ("blood", "boiling"):            (-45, +50, +30, +30, +25, "physical: extreme anger"),
    ("skin", "crawling"):            (-35, +35, -15, +20, +10, "physical: disgust/fear"),
    ("butterflies", "in"):           (+15, +25, -10, +10, +15, "physical: nervous excitement"),
    ("gut", "feeling"):              (-10, +20, +10, +15, +5, "physical: intuition"),
    ("seeing", "red"):               (-50, +55, +30, +30, +25, "physical: blind rage"),
    ("cold", "feet"):                (-20, +20, -20, +15, -10, "physical: fear/hesitation"),

    # Hope / Recovery
    ("light", "at", "the", "end"):   (+30, +15, +15, +5, +30, "hope: light at end of tunnel"),
    ("fresh", "start"):              (+25, +15, +20, +5, +25, "hope: new beginning"),
    ("bounce", "back"):              (+25, +20, +25, +5, +25, "recovery: resilience"),
    ("pick", "myself", "up"):        (+20, +20, +20, +10, +20, "recovery: self-recovery"),
    ("turn", "over", "a", "new", "leaf"): (+25, +15, +20, +5, +25, "recovery: change behavior"),
    ("silver", "lining"):            (+20, +10, +15, 0, +20, "hope: positive in bad situation"),
    ("blessing", "in", "disguise"):  (+15, +15, +15, 0, +15, "hope: hidden good"),

    # More Slang / Internet
    ("dead", "inside"):              (-50, -20, -30, 0, -40, "slang: emotionally numb"),
    ("rent", "free"):                (-10, +15, -10, 0, +10, "slang: living in your head"),
    ("vibe", "check"):               (+5, +20, +10, +10, +5, "slang: assessing energy"),
    ("big", "mood"):                 (+10, +20, +5, 0, +10, "slang: relatable feeling"),
    ("go", "off"):                   (+10, +40, +20, +15, +15, "slang: let loose/rant"),
    ("absolutely", "slaying"):       (+30, +25, +25, 0, +20, "slang: doing amazing"),
    ("understood", "the", "assignment"): (+30, +20, +25, 0, +20, "slang: nailed it"),
    ("ate", "that", "up"):           (+30, +25, +25, 0, +20, "slang: did perfectly"),

    # More Workplace / Professional
    ("touch", "base"):               (0, +5, +10, +5, 0, "corporate: check in"),
    ("move", "the", "needle"):       (+10, +15, +15, +10, +10, "corporate: make progress"),
    ("run", "it", "up", "the", "flagpole"): (0, +10, -10, +5, 0, "corporate: get approval"),
    ("think", "outside", "the", "box"): (+10, +15, +15, 0, +10, "corporate: be creative"),
    ("drink", "the", "kool", "aid"): (-20, +10, -20, 0, -10, "corporate: blind loyalty"),
    ("golden", "handcuffs"):         (-15, +10, -15, +5, -10, "workplace: trapped by money"),

    # More Relationship / Social
    ("love", "at", "first", "sight"): (+50, +40, -10, +15, +40, "relationship: instant love"),
    ("head", "over", "heels"):       (+45, +35, -15, +10, +35, "relationship: deeply in love"),
    ("cold", "shoulder"):            (-35, +15, +20, +10, -15, "relationship: deliberate ignoring"),
    ("mixed", "signals"):            (-20, +25, -15, +20, -10, "relationship: confusing behavior"),
    ("red", "flag"):                 (-25, +25, +10, +20, -10, "relationship: warning sign"),
    ("green", "flag"):               (+25, +15, +10, 0, +15, "relationship: positive sign"),

    # More Physical / Embodied
    ("hair", "standing"):            (-25, +35, -15, +20, +10, "physical: fear/alarm"),
    ("jaw", "dropped"):              (+5, +40, -15, +10, +15, "physical: shock/surprise"),
    ("heart", "racing"):             (-10, +40, -10, +20, +10, "physical: anxiety/excitement"),
    ("knees", "weak"):               (-15, +25, -20, +10, -10, "physical: overwhelmed"),
    ("tears", "of", "joy"):          (+40, +35, -10, 0, +30, "physical: happy crying"),
    ("sick", "to", "my", "stomach"): (-40, +30, -20, +20, -25, "physical: disgust/dread"),
}


# Relationship gravity — intimate connections amplify emotional weight
RELATIONSHIP_GRAVITY = {
    # Intimate — reduced from 2.0-2.5x to 1.4-1.7x to prevent false amplification
    "wife": 1.5, "husband": 1.5, "daughter": 1.7, "son": 1.7,
    "mother": 1.5, "mom": 1.6, "mommy": 1.6, "father": 1.5, "dad": 1.6, "daddy": 1.6,
    "baby": 1.7, "child": 1.7, "children": 1.5, "kid": 1.4, "kids": 1.4,
    # Family
    "brother": 1.4, "sister": 1.4, "grandma": 1.4, "grandmother": 1.4,
    "grandpa": 1.4, "grandfather": 1.4, "family": 1.4, "uncle": 1.2, "aunt": 1.2,
    "cousin": 1.2, "nephew": 1.2, "niece": 1.2,
    # Close
    "friend": 1.3, "bestie": 1.3, "boyfriend": 1.3, "girlfriend": 1.3,
    "partner": 1.3, "fiance": 1.3, "fiancee": 1.3, "soulmate": 1.4,
    # Moderate
    "neighbor": 1.1, "coworker": 1.05, "colleague": 1.05, "classmate": 1.1,
    "roommate": 1.15, "teammate": 1.15, "mentor": 1.15, "teacher": 1.1,
    # Distant
    "someone": 1.0, "person": 1.0, "people": 0.95, "stranger": 0.9,
    "guy": 0.9, "dude": 0.95, "man": 0.95, "woman": 0.95,
    # Antagonist
    "enemy": 1.3, "bully": 1.3, "abuser": 1.5, "ex": 1.15,
    "stalker": 1.4, "predator": 1.4, "attacker": 1.4,
    # Pets
    "dog": 1.3, "cat": 1.2, "pet": 1.25, "puppy": 1.4, "kitten": 1.3,
}

# Build a lookup: first word -> list of (full_tuple, forces)
# for efficient scanning
_IDIOM_STARTERS = {}
for words_tuple, *forces_and_label in IDIOMS.items():
    first = words_tuple[0]
    if first not in _IDIOM_STARTERS:
        _IDIOM_STARTERS[first] = []
    _IDIOM_STARTERS[first].append((words_tuple, IDIOMS[words_tuple]))


# Anticipation patterns: sequences that build tension/arousal
ANTICIPATION_PATTERNS = {
    ("i've", "got"):          (0, +15, +10, +15, "something coming"),
    ("i", "need", "to", "tell"): (-5, +20, +10, +20, "serious incoming"),
    ("i", "need", "to", "tell", "you"): (-10, +25, +10, +25, "serious targeted"),
    ("we", "need", "to", "talk"): (-15, +25, +15, +25, "serious conversation"),
    ("there's", "something"):  (-5, +15, +5, +15, "something brewing"),
    ("i", "have", "to", "say"): (-5, +15, +10, +15, "forthcoming"),
}


# Context-dependent word modifiers: (condition_fn, v_delta, a_delta, d_delta, u_delta, label)
# These adjust a word's force based on the current pendulum state
def _ctx_buddy(pend):
    """'buddy' is friendly when positive, confrontational when tense."""
    if pend.v < 110 or pend.a > 160:
        return (-20, +15, +10, +10, "confrontational 'buddy'")
    return None

def _ctx_you(pend):
    """'you' in high arousal = targeted/threatening."""
    if pend.a > 155:
        return (-15, +12, +10, +10, "targeted 'you'")
    return None

def _ctx_but(pend):
    """'but' after positive = massive dread yank; after negative = slight relief."""
    if pend.v > 140:
        return (-35, +25, -10, +15, "dread: 'but' after positive")
    elif pend.v < 100:
        return (+10, -5, +5, -5, "relief: 'but' after negative")
    return (-8, +10, 0, +5, "pivot: 'but'")

def _ctx_however(pend):
    """'however' = reversal, similar to 'but'."""
    if pend.v > 140:
        return (-25, +20, -5, +10, "reversal: 'however' after positive")
    elif pend.v < 100:
        return (+8, -5, +5, -3, "relief: 'however' after negative")
    return (-5, +8, 0, +3, "pivot: 'however'")

def _ctx_right(pend):
    """'right' can be agreement or challenge depending on arousal."""
    if pend.a > 160:
        return (-10, +10, +15, +5, "challenging 'right?!'")
    return (+5, 0, +5, 0, "agreeable 'right'")

def _ctx_please(pend):
    """'please' — polite vs desperate vs sarcastic depending on context."""
    if pend.v < 60:  # crisis territory
        return (-20, +20, -30, +30, "desperate 'please'")
    elif pend.v < 100 and pend.a > 150:  # angry
        return (-10, +15, +20, +10, "sarcastic/demanding 'please'")
    elif pend.u > 40 or pend.a > 160:  # urgent
        return (-10, +10, -15, +10, "urgent 'please'")
    return (+10, -5, +5, 0, "polite 'please'")

def _ctx_friend(pend):
    """'friend' when tense = passive-aggressive."""
    if pend.v < 110 or pend.a > 155:
        return (-15, +10, +10, +10, "passive-aggressive 'friend'")
    return None

def _ctx_fine(pend):
    """'fine' after negative = passive-aggressive; neutral otherwise."""
    if pend.v < 110:
        return (-15, +10, +5, +5, "passive-aggressive 'fine'")
    return None

def _ctx_sure(pend):
    """'sure' after negative = sarcastic agreement."""
    if pend.v < 105:
        return (-10, +8, +5, +5, "sarcastic 'sure'")
    return None

def _ctx_okay(pend):
    """'okay' after strongly negative = resignation."""
    if pend.v < 90:
        return (-10, -5, -10, +5, "resigned 'okay'")
    return None

def _ctx_man(pend):
    """'man' as filler when tense = exasperation."""
    if pend.a > 150:
        return (-5, +8, +5, +5, "exasperated 'man'")
    return None

def _ctx_love(pend):
    """'love' — sarcastic vs genuine based on trajectory."""
    if pend.v < 100:  # negative trajectory
        return (-20, +15, -5, +5, "sarcastic 'love'")
    return (+40, +25, +15, 0, "genuine 'love'")

def _ctx_really(pend):
    """'really' — emphasis vs disbelief depending on context."""
    if pend.v < 110 and pend.a > 140:
        return (-15, +20, +10, +5, "skeptical 'really'")
    return (+5, +10, +5, 0, "emphatic 'really'")

def _ctx_interesting(pend):
    """'interesting' — genuine curiosity vs dismissive."""
    if pend.v < 110:  # negative context
        return (-10, -5, +10, 0, "dismissive 'interesting'")
    return (+15, +10, +10, 0, "genuinely 'interesting'")

def _ctx_sorry(pend):
    """'sorry' — empathy vs sarcastic/dismissive."""
    if pend.a > 160 and pend.v < 100:  # heated negative context
        return (-5, +10, +15, +5, "dismissive 'sorry'")
    return (-15, -10, -10, 0, "genuine 'sorry'")

def _ctx_good(pend):
    """'good' — genuine vs passive aggressive."""
    if pend.v < 100:  # negative context
        return (-10, -5, +5, 0, "passive-aggressive 'good'")
    return (+20, +10, +10, 0, "genuine 'good'")

def _ctx_actually(pend):
    """'actually' — confrontational correction vs mild clarification."""
    if pend.v < 110:
        return (-10, +15, +20, +5, "confrontational 'actually'")
    return (+5, +10, +15, 0, "clarifying 'actually'")

def _ctx_just(pend):
    """'just' — minimizer that downplays pain in negative context."""
    if pend.v < 90:  # negative context, "just" minimizes
        return (+5, -10, -10, 0, "downplaying 'just'")
    return (-5, -5, -5, 0, "neutral 'just'")

def _ctx_oh(pend):
    """'oh' — dread or pleasant surprise depending on context."""
    if pend.v < 100:
        return (-10, +20, -10, +10, "dread 'oh'")
    return (+10, +20, +5, +5, "surprised 'oh'")

def _ctx_wow(pend):
    """'wow' — shocked/appalled vs genuinely impressed."""
    if pend.v < 100:
        return (-15, +30, -10, +10, "shocked 'wow'")
    return (+25, +30, +10, +5, "impressed 'wow'")



def _ctx_lol(pend):
    """lol is a chameleon — meaning depends on current state."""
    if pend.v < 60: return (-5, -5, -5, 0, -5)       # crisis + lol = shield, amplify pain
    elif pend.v < 100: return (-3, -3, 0, 0, -3)      # negative + lol = deflection
    elif pend.v > 170: return (+5, +5, 0, 0, +5)      # very positive + lol = shared joy
    elif pend.a > 160: return (+5, +5, 0, 0, +5)      # high energy + lol = genuine amusement
    return (0, 0, 0, 0, 0)                             # neutral + lol = filler

def _ctx_haha(pend):
    """haha — similar to lol but slightly more genuine."""
    if pend.v < 60: return (-5, -5, -5, 0, -5)
    elif pend.v < 100: return (-2, -2, 0, 0, -2)
    elif pend.v > 150: return (+8, +8, 0, 0, +8)
    return (+3, +3, 0, 0, +3)

def _ctx_ok(pend):
    """ok — agreeable, defeated, or cold depending on state."""
    if pend.v < 80: return (-10, -10, -10, 0, -10)    # defeated ok
    elif pend.v > 150: return (+5, +5, +5, 0, +5)     # enthusiastic ok
    elif pend.a < 100: return (-5, -5, 0, 0, -5)      # flat/cold ok
    return (0, 0, 0, 0, 0)

def _ctx_so(pend):
    """so — transition, emphasis, or exasperation."""
    if pend.v < 90: return (-5, +10, 0, +5, 0)        # exasperation
    elif pend.v > 160: return (+5, +5, 0, 0, +5)      # emphasis
    return (0, +5, 0, 0, 0)                            # transition

def _ctx_anyway(pend):
    """anyway — redirect, often after something painful."""
    if pend.v < 100: return (+5, -10, +5, -5, +5)     # pulling away from pain (redirect)
    return (0, -5, 0, 0, 0)                            # neutral redirect

def _ctx_nvm(pend):
    """nvm/nevermind — retraction of vulnerability."""
    if pend.v < 80: return (-8, -5, -10, 0, -8)       # retracting pain = amplify (they shouldn't have to hide)
    return (0, -5, 0, 0, 0)


def _ctx_or(pend):
    """or = uncertainty operator. User is comparing, not declaring."""
    # Pull V toward center — dampens whatever direction we were heading
    if pend.v > 140:
        return (-10, -5, 0, 0, 0)  # pull back from positive
    elif pend.v < 115:
        return (+10, -5, 0, 0, 0)  # pull back from negative
    return (0, -5, 0, 0, 0)  # already neutral, just reduce arousal

CONTEXT_MODIFIERS = {
    "buddy": _ctx_buddy,
    "pal": _ctx_buddy,
    "friend": _ctx_friend,
    "you": _ctx_you,
    "your": _ctx_you,
    "but": _ctx_but,
    "however": _ctx_however,
    "right": _ctx_right,
    "please": _ctx_please,
    "fine": _ctx_fine,
    "sure": _ctx_sure,
    "okay": _ctx_okay,
    "man": _ctx_man,
    "love": _ctx_love,
    # "really" handled by RAMPS system (step 4), not context modifiers
    "interesting": _ctx_interesting,
    "sorry": _ctx_sorry,
    "good": _ctx_good,
    "actually": _ctx_actually,
    "just": _ctx_just,
    "oh": _ctx_oh,
    "wow": _ctx_wow,
    "lol": _ctx_lol,
    "lmao": _ctx_lol,
    "haha": _ctx_haha,
    "hahaha": _ctx_haha,
    "ok": _ctx_ok,
    "okay": _ctx_ok,
    # "so" handled by RAMPS system (step 4), not context modifiers
    "anyway": _ctx_anyway,
    "anyways": _ctx_anyway,
    "nvm": _ctx_nvm,
    "nevermind": _ctx_nvm,
    "or": _ctx_or,
}


class SequentialPendulum:
    """Word-by-word emotional pendulum with momentum, context, and idiom detection.

    Each word shifts the pendulum based on what's ALREADY swinging.
    The pendulum has momentum (inertia) — once it starts swinging negative,
    neutral words don't instantly reset it. It drifts back slowly.
    """

    # First-word center calibration — shifts the starting point based on
    # what kind of sentence is coming.  Applied BEFORE the word's own force.
    STARTER_CALIBRATION = {
        # Self-referential → emotional content expected, lower V center
        "i": {"v": 120, "a": 132, "d": 124, "u": 5, "g": 124},
        "i'm": {"v": 118, "a": 135, "d": 122, "u": 8, "g": 122},
        "i've": {"v": 116, "a": 133, "d": 122, "u": 8, "g": 122},
        "my": {"v": 118, "a": 130, "d": 124, "u": 5, "g": 124},
        "we": {"v": 122, "a": 130, "d": 126, "u": 5, "g": 126},

        # Questions → stay neutral, practical
        "can": {"v": 128, "a": 130, "d": 128, "u": 10, "g": 128},
        "could": {"v": 128, "a": 128, "d": 128, "u": 8, "g": 128},
        "would": {"v": 128, "a": 128, "d": 128, "u": 8, "g": 128},
        "do": {"v": 128, "a": 130, "d": 128, "u": 10, "g": 128},
        "does": {"v": 128, "a": 128, "d": 128, "u": 8, "g": 128},
        "what": {"v": 126, "a": 132, "d": 126, "u": 12, "g": 126},
        "why": {"v": 122, "a": 135, "d": 120, "u": 15, "g": 122},
        "how": {"v": 126, "a": 130, "d": 126, "u": 10, "g": 126},
        "where": {"v": 128, "a": 128, "d": 128, "u": 12, "g": 128},
        "when": {"v": 128, "a": 128, "d": 128, "u": 10, "g": 128},
        "who": {"v": 126, "a": 130, "d": 126, "u": 10, "g": 126},

        # Urgent/help → shift negative-expecting
        "help": {"v": 110, "a": 145, "d": 100, "u": 30, "g": 110},
        "please": {"v": 115, "a": 140, "d": 105, "u": 25, "g": 115},
        "stop": {"v": 108, "a": 150, "d": 110, "u": 35, "g": 112},
        "wait": {"v": 120, "a": 135, "d": 115, "u": 20, "g": 120},

        # Greetings → warm, positive-leaning
        "hey": {"v": 138, "a": 130, "d": 132, "u": 0, "g": 135},
        "hi": {"v": 140, "a": 125, "d": 132, "u": 0, "g": 135},
        "hello": {"v": 140, "a": 125, "d": 132, "u": 0, "g": 135},
        "yo": {"v": 135, "a": 135, "d": 130, "u": 0, "g": 132},
        "sup": {"v": 135, "a": 132, "d": 130, "u": 0, "g": 132},

        # Positive openers
        "wow": {"v": 140, "a": 150, "d": 130, "u": 5, "g": 140},
        "omg": {"v": 138, "a": 155, "d": 125, "u": 10, "g": 138},
        "yes": {"v": 142, "a": 135, "d": 140, "u": 0, "g": 138},

        # Negative openers
        "no": {"v": 110, "a": 140, "d": 120, "u": 15, "g": 115},
        "ugh": {"v": 108, "a": 125, "d": 115, "u": 5, "g": 112},
        "fuck": {"v": 100, "a": 160, "d": 125, "u": 20, "g": 118},
        "shit": {"v": 105, "a": 150, "d": 120, "u": 15, "g": 115},

        # Referencing others
        "he": {"v": 124, "a": 130, "d": 126, "u": 5, "g": 126},
        "she": {"v": 124, "a": 130, "d": 126, "u": 5, "g": 126},
        "they": {"v": 126, "a": 128, "d": 126, "u": 5, "g": 126},
        "it": {"v": 126, "a": 128, "d": 128, "u": 5, "g": 128},

        # Negation openers
        "nothing": {"v": 108, "a": 120, "d": 110, "u": 5, "g": 108},
        "nobody": {"v": 105, "a": 125, "d": 108, "u": 10, "g": 105},
        "never": {"v": 106, "a": 130, "d": 112, "u": 8, "g": 108},
        "everything": {"v": 120, "a": 140, "d": 120, "u": 10, "g": 120},
    }

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.v = 128.0  # start neutral
        self.a = 128.0
        self.d = 128.0
        self.u = 0.0
        self.g = 128.0  # gravity: start grounded
        self.momentum = 0.99  # tuned via bracket tournament (benchmarks/optimal_config.json)
        self.drift_rate = 0.01  # tuned from 0.02 based on benchmark analysis  # how fast pendulum drifts toward center per tick
        self.shift_marker_dampening = 0.6  # tunable: how much shift markers reduce momentum
        self.idiom_momentum = 0.50  # tunable: idioms break through harder
        self.idiom_push = 0.7       # tunable: stronger direct push for idioms
        self.crisis_momentum_cap = 0.99  # tunable: momentum cap during crisis lock
        self.crisis_direct_push = 0.05   # tunable: direct push scale during crisis lock
        self.history = []  # (word, v, a, d, u, g, state_label) per step
        self.previous_words = []  # for idiom/context detection
        self.negate_next = False
        self.intensity = 1.0
        self._relationship_mult = 1.0  # relationship gravity multiplier
        self._personality = None  # set externally for willingness-based force scaling
        self._active_ramp = None  # (multiplier, words_remaining, decay) or None
        self.idiom_consumed = set()  # indices of words consumed by idiom detection
        self._word_index = 0
        self._crisis_momentum = 0  # remaining words to lock after crisis idiom fires
        self._last_spike_strength = 0  # exponential decay: strength of last strong word
        self._words_since_spike = 0    # exponential decay: words elapsed since last spike
        self._decay_lambda = 0.15      # exponential decay rate (swept: 0.15 > 0.3)
        self._spike_threshold = 10     # force magnitude to count as a "spike" (swept: 10 > 20)
        self._momentum_floor = 0.5     # minimum momentum during recovery
        self._momentum_range = 0.49    # momentum range above floor
        self._bookend_parser = BookendParser()
        self._word_role_detector = WordRoleDetector()
        self._bigram_detector = BigramDetector()
        self._bookend_result = None

    def _calibrate_start(self, first_word):
        """Shift the pendulum's initial center based on the first word.

        Called before processing begins.  Sets the STARTING POINT, then the
        word's own force still applies on top during process_word.
        """
        word = first_word.lower()
        if word in self.STARTER_CALIBRATION:
            cal = self.STARTER_CALIBRATION[word]
            self.v = float(cal["v"])
            self.a = float(cal["a"])
            self.d = float(cal["d"])
            self.u = float(cal["u"])
            self.g = float(cal["g"])

    @property
    def _pend_state(self):
        """Quick snapshot for context functions."""
        return type('Pend', (), {'v': self.v, 'a': self.a, 'd': self.d, 'u': self.u, 'g': self.g})()

    def _clamp(self):
        """Clamp all values to valid range."""
        self.v = max(0.0, min(255.0, self.v))
        self.a = max(0.0, min(255.0, self.a))
        self.d = max(0.0, min(255.0, self.d))
        self.u = max(0.0, min(255.0, self.u))
        self.g = max(0.0, min(255.0, self.g))


    def _compute_willingness(self):
        """Compute personality willingness (W) from Rackliffe's emotional gravity.

        Empath (high agree, low assert) → forces 1.3x (feels more)
        Stoic (low agree, high assert) → forces 0.8x (dampened)
        Default → 1.0x
        """
        if self._personality is None:
            return 1.0
        p = self._personality
        W = (p.assertiveness + (255 - p.agreeableness)) / 510.0
        return 1.3 - W * 0.5

    def _drift_toward_center(self):
        """Pendulum drifts 10% toward neutral each tick (unless strong force)."""
        self.v += (128.0 - self.v) * self.drift_rate
        self.a += (128.0 - self.a) * self.drift_rate
        self.d += (128.0 - self.d) * self.drift_rate
        self.u += (0.0 - self.u) * self.drift_rate
        self.g += (128.0 - self.g) * self.drift_rate

    def _state_label(self) -> str:
        """Describe the current pendulum state in a few words."""
        labels = []

        # Valence
        if self.v < 60:
            labels.append("DARK")
        elif self.v < 90:
            labels.append("negative")
        elif self.v < 115:
            labels.append("slightly tense")
        elif self.v < 142:
            labels.append("neutral")
        elif self.v < 175:
            labels.append("warm")
        elif self.v < 210:
            labels.append("positive")
        else:
            labels.append("euphoric")

        # Arousal
        if self.a > 200:
            labels.append("INTENSE")
        elif self.a > 165:
            labels.append("charged")
        elif self.a > 140:
            labels.append("alert")
        elif self.a < 80:
            labels.append("subdued")
        elif self.a < 100:
            labels.append("quiet")

        # Dominance
        if self.d > 185:
            labels.append("dominant")
        elif self.d < 70:
            labels.append("vulnerable")
        elif self.d < 90:
            labels.append("uncertain")

        # Urgency
        if self.u > 60:
            labels.append("urgent")
        elif self.u > 30:
            labels.append("building")

        return ", ".join(labels) if labels else "resting"

    def check_idiom(self, words, current_idx):
        """Check if current position completes an idiom.

        Returns (v, a, d, u, label, length, start_idx) if idiom found, else None.
        Prefers the LONGEST matching idiom to avoid double-triggering.
        """
        best_match = None
        best_len = 0

        window_back = 5  # look back up to 5 words

        for back_offset in range(min(window_back, current_idx + 1)):
            check_start = current_idx - back_offset
            if check_start < 0:
                continue

            first_word = words[check_start]
            if first_word not in _IDIOM_STARTERS:
                continue

            for idiom_words, forces_tuple in _IDIOM_STARTERS[first_word]:
                idiom_len = len(idiom_words)
                # Current word must be the LAST word of the idiom
                if check_start + idiom_len - 1 != current_idx:
                    continue
                # Check all words match
                match = True
                for j, iw in enumerate(idiom_words):
                    if check_start + j >= len(words) or words[check_start + j] != iw:
                        match = False
                        break
                if match and idiom_len > best_len:
                    # Handle both 5-element (dv,da,dd,du,label) and 6-element (dv,da,dd,du,dg,label) tuples
                    if len(forces_tuple) == 6:
                        vf, af, df, uf, gf, label = forces_tuple
                    else:
                        vf, af, df, uf, label = forces_tuple
                        gf = 0  # no gravity override for old-style idioms
                    best_match = (vf, af, df, uf, label, idiom_len, check_start, gf)
                    best_len = idiom_len

        return best_match

    def check_anticipation(self, words, current_idx):
        """Check if recent words match an anticipation pattern."""
        for pattern_words, (vf, af, df, uf, label) in ANTICIPATION_PATTERNS.items():
            plen = len(pattern_words)
            # Pattern ends at current_idx
            check_start = current_idx - plen + 1
            if check_start < 0:
                continue
            match = True
            for j, pw in enumerate(pattern_words):
                if words[check_start + j] != pw:
                    match = False
                    break
            if match:
                return (vf, af, df, uf, label)
        return None

    def get_contextual_force(self, word):
        """Get context-dependent force modifier for a word.

        Returns (v_delta, a_delta, d_delta, u_delta, label) or None.
        """
        if word in CONTEXT_MODIFIERS:
            result = CONTEXT_MODIFIERS[word](self._pend_state)
            return result
        return None

    # Common function/bridge words that should NEVER be morpheme-decomposed
    BRIDGE_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "am", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "could", "must", "to", "of", "in", "on",
        "at", "by", "for", "with", "from", "as", "into", "about", "up",
        "out", "off", "over", "after", "under", "between", "through",
        "during", "before", "that", "this", "these", "those", "it", "its",
        "he", "she", "him", "her", "his", "them", "their", "our", "us",
        "who", "which", "whose", "whom", "if", "then", "than", "or", "and",
        "but", "so", "yet", "nor", "both", "each", "all", "any", "some",
        "just", "also", "too", "even", "still", "here", "there", "where",
        "very", "really", "quite", "rather", "much", "more", "most",
        "own", "other", "another", "such", "only", "same",
        "got", "get", "gets", "getting", "go", "goes", "going", "went",
        "come", "comes", "came", "coming", "take", "takes", "took", "taken",
        "put", "let", "say", "said", "says", "tell", "told", "talk",
        "give", "gave", "given", "see", "saw", "seen", "feel", "felt",
    }

    def get_word_force(self, word):
        """Get the base force for a word. Tries WORD_FORCES, then morphemes.

        Returns (vf, af, df, uf, source_label) or None for bridge words.
        """
        if word in WORD_FORCES:
            vf, af, df, uf, gf = WORD_FORCES[word]
            return (vf, af, df, uf, gf, None)

        # Skip morpheme decomposition for common function words
        if word in self.BRIDGE_WORDS:
            return None

        # Skip very short words (1-2 chars) -- too ambiguous for morphemes
        if len(word) <= 2:
            return None

        # Try morphological decomposition
        if not self.config.morpheme_decomposition:
            return None
        result = decompose_word(word)
        if result["found"]:
            vf = result["v"]
            af = result["a"]
            df = result["d"]
            uf = result["u"]
            gf = result["g"]
            parts = []
            if result["prefix"]:
                parts.append(result["prefix"] + "-")
            parts.append(result["root"])
            if result["suffix"]:
                parts.append("-" + result["suffix"])
            return (vf, af, df, uf, gf, f"morpheme:{''.join(parts)}")

        return None

    # Temporal shift markers — these words signal a narrative turn.
    # When encountered, momentum is reduced by 40% to allow reversals.
    SHIFT_MARKERS = {"then", "suddenly", "until"}
    SHIFT_BIGRAMS = {("but", "then"), ("after", "that"), ("and", "then")}

    def process_word(self, word, words, current_idx):
        """Process a single word in sequence. Returns a trace dict."""
        state_label = ""
        force_source = ""
        applied_force = False
        idiom_hit = False

        # Dynamic neutral center: calibrate starting point from first word
        if current_idx == 0:
            self._calibrate_start(word)

            # Bookend priming: parse full sentence and prime the pendulum
            if self.config.bookend_parsing:
                self._bookend_result = self._bookend_parser.parse(' '.join(words))
                pattern = self._bookend_result.get('pattern')
                if pattern == 'isolation':
                    self.v -= 15  # prime negative — absence->self
                    self.g -= 10
                elif pattern == 'self_loop':
                    self.v -= 20  # prime very negative — self->self is dangerous
                    self.g -= 15
                elif pattern == 'accusation':
                    self.v -= 10
                    self.a += 15
                elif pattern == 'plea':
                    self.u += 20  # prime urgency

        # Recency weighting: later words in a sequence get more influence
        if self.config.recency_weighting:
            n_words = len(words)
            recency_weight = 0.9 + 0.4 * (current_idx / max(n_words - 1, 1))  # tuned: 0.9-1.3 range
        else:
            recency_weight = 1.0

        # Temporal shift marker detection — reduce momentum to allow reversals
        if self.config.shift_markers:
            is_shift = False
            if word in self.SHIFT_MARKERS:
                is_shift = True
            if current_idx > 0:
                bigram = (words[current_idx - 1], word)
                if bigram in self.SHIFT_BIGRAMS:
                    is_shift = True
            if is_shift:
                self.momentum *= (1.0 - self.shift_marker_dampening)  # reduce momentum

        # 1. Drift toward center ONLY happens AFTER emotional words (see below)
        # Bridge/filler words are "zero mass" — they don't pull the pendulum
        # This prevents "the", "is", "a" from diluting emotional payload

        # 2. Check for idiom completion at this word
        idiom = self.check_idiom(words, current_idx) if self.config.idiom_detection else None
        if idiom:
            vf, af, df, uf, label, idiom_len, idiom_start, gf = idiom
            # Mark previous words in the idiom as consumed
            for j in range(idiom_start, current_idx):
                self.idiom_consumed.add(j)

            # Apply idiom force — idioms hit HARDER than single words because
            # they represent a recognized multi-word expression with clear intent.
            # Reduced momentum (0.7 vs 0.9) and stronger direct push (0.5 vs 0.3).
            force_scale = 1.0 * self.intensity * recency_weight
            # Apply active ramp to idiom force as well
            if self._active_ramp:
                ramp_mult, ramp_remaining, ramp_decay = self._active_ramp
                force_scale *= ramp_mult
                ramp_remaining -= 1
                if ramp_remaining <= 0:
                    self._active_ramp = None
                else:
                    self._active_ramp = (ramp_mult * ramp_decay, ramp_remaining, ramp_decay)
            if self.negate_next:
                vf = -vf
                df = -df
                self.negate_next = False

            im = self.idiom_momentum   # tuned: idioms break through harder
            ip = self.idiom_push       # tuned: stronger direct push
            self.v = self.v * im + (128.0 + vf * force_scale) * (1.0 - im) + vf * ip
            self.a = self.a * im + (128.0 + af * force_scale) * (1.0 - im) + af * ip
            self.d = self.d * im + (128.0 + df * force_scale) * (1.0 - im) + df * ip
            self.u = self.u * im + (uf * force_scale) * (1.0 - im) + uf * ip
            self.g = self.g * im + (128.0 + gf * force_scale) * (1.0 - im) + gf * ip
            self._clamp()
            self.intensity = 1.0
            force_source = f"IDIOM: \"{label}\""
            applied_force = True
            idiom_hit = True

            # Crisis idioms lock momentum to prevent tail-word dilution.
            # Strong crisis idioms (vf <= -80) always lock.
            # Weaker crisis idioms only lock with first-person self-reference.
            if self.config.crisis_momentum_lock and label.startswith("crisis:"):
                if vf <= -80:
                    self._crisis_momentum = len(words) - current_idx - 1
                    # Strong crisis idiom penalty: ensure V stays deep in crisis range
                    self.v = max(0, self.v - 15)
                    self.g = max(0, self.g - 10)
                else:
                    first_person = {"i", "i'm", "i've", "me", "my", "myself", "i'd", "i'll"}
                    crisis_context = {"die", "died", "death", "dead", "kill", "suicide",
                                      "disappear", "disappeared", "vanish", "hopeless",
                                      "worthless", "helpless", "alone", "empty", "numb",
                                      "living", "anymore", "everyone", "everything",
                                      "nobody", "nothing", "never", "always", "forever"}
                    has_self = bool(first_person & set(words))
                    has_crisis = bool(crisis_context & set(words))
                    existential = {"living", "alive", "life", "exist", "anymore",
                                   "disappeared", "disappear", "vanish", "gone"}
                    has_existential = bool(existential & set(words))
                    if (has_self and has_crisis) or has_existential:
                        self._crisis_momentum = len(words) - current_idx - 1
                        # Context-triggered penalty: push V further into crisis range
                        self.v = max(0, self.v - 20)
                        self.g = max(0, self.g - 15)

        # If this word was already consumed by an earlier idiom, it "holds"
        if current_idx in self.idiom_consumed and not idiom_hit:
            self._clamp()
            state_label = self._state_label()
            trace_entry = {
                "word": word,
                "v": int(self.v), "a": int(self.a),
                "d": int(self.d), "u": int(self.u), "g": int(self.g),
                "state": f"(idiom part) {state_label}",
            }
            self.history.append(trace_entry)
            self.previous_words.append(word)
            return trace_entry

        if not applied_force:
            # 3. Check for negators
            if word in NEGATORS:
                self.negate_next = True
                # Negators still have their own mild force
                if word in WORD_FORCES:
                    vf, af, df, uf, gf = WORD_FORCES[word]
                    self.v += vf * 0.3
                    self.a += af * 0.3
                    self.d += df * 0.3
                    self.u += uf * 0.3
                    self.g += gf * 0.3
                self._clamp()
                state_label = self._state_label()
                trace_entry = {
                    "word": word,
                    "v": int(self.v), "a": int(self.a),
                    "d": int(self.d), "u": int(self.u), "g": int(self.g),
                    "state": f"NEGATE next | {state_label}",
                }
                self.history.append(trace_entry)
                self.previous_words.append(word)
                return trace_entry

            # 4. Check for ramp words (intensity amplifiers / dampeners)
            if word in RAMPS:
                ramp = RAMPS[word]
                if self._active_ramp:
                    # Chain: compound the ramps (e.g. "really really" → 1.35 * 1.35)
                    existing_mult, existing_remaining, existing_decay = self._active_ramp
                    new_mult = existing_mult * ramp.multiplier
                    new_length = max(existing_remaining, ramp.length)
                    new_decay = min(existing_decay, ramp.decay)  # keep the steeper decay
                    self._active_ramp = (new_mult, new_length, new_decay)
                else:
                    self._active_ramp = (ramp.multiplier, ramp.length, ramp.decay)
                # The ramp word itself is a modifier, not content — minimal force.
                # Still set self.intensity for backward compat with idiom path.
                self.intensity = 1.0
                self._clamp()
                state_label = self._state_label()
                mult_display = self._active_ramp[0]
                remaining_display = self._active_ramp[1]
                trace_entry = {
                    "word": word,
                    "v": int(self.v), "a": int(self.a),
                    "d": int(self.d), "u": int(self.u), "g": int(self.g),
                    "state": f"RAMP x{mult_display:.2f} ({remaining_display}w) | {state_label}",
                }
                self.history.append(trace_entry)
                self.previous_words.append(word)
                return trace_entry

            # 5. Check anticipation patterns
            anticipation = self.check_anticipation(words, current_idx)

            # 5b. Bigram detection — check if current word forms a bigram pair
            bigram = self._bigram_detector.detect(words, current_idx) if self.config.bigram_detection else None

            # 6. Get base word force (bigram overrides individual word force)
            if bigram:
                vf, af, df, uf, gf = bigram[:5]
                label = bigram[5] if len(bigram) > 5 else "bigram"
                word_force = (vf, af, df, uf, gf, f"BIGRAM: {label}")
            else:
                # Check for relationship gravity words
                if word in RELATIONSHIP_GRAVITY:
                    self._relationship_mult = RELATIONSHIP_GRAVITY[word]

                word_force = self.get_word_force(word)

            # 7. Get contextual modifier
            ctx_mod = self.get_contextual_force(word) if self.config.context_modifiers else None

            if word_force is not None:
                vf, af, df, uf, gf, source = word_force

                # Word role detection: descriptors inherit emotion from subject
                if self.config.word_role_detection:
                    resolved = self._word_role_detector.resolve_force(word, (vf, af, df, uf, gf), words, current_idx)
                    if resolved != (vf, af, df, uf, gf):
                        vf, af, df, uf, gf = resolved[:5]
                        source = f"ROLE_RESOLVED({source})"

                if self.negate_next:
                    vf = -vf
                    df = -df
                    gf = -gf  # negate flips gravity too
                    self.negate_next = False
                    force_source = "NEGATED"
                elif source:
                    force_source = source

                willingness_mult = self._compute_willingness()
                # Context operators: bridge words scale emotional impact
                from demo.context_operators import get_context_coefficient
                context_coeff = get_context_coefficient(words, current_idx)
                force_scale = self.intensity * recency_weight * self._relationship_mult * willingness_mult * context_coeff
                # Cap combined multiplier — context can push higher now
                force_scale = min(force_scale, 4.0)
                # Decay relationship gravity — only amplifies next 2 words then fades
                if self._relationship_mult != 1.0:
                    self._relationship_mult = 1.0 + (self._relationship_mult - 1.0) * 0.3
                    if abs(self._relationship_mult - 1.0) < 0.05:
                        self._relationship_mult = 1.0

                # Apply active ramp (multi-word intensity slope)
                if self._active_ramp:
                    ramp_mult, ramp_remaining, ramp_decay = self._active_ramp
                    force_scale *= ramp_mult
                    ramp_remaining -= 1
                    if ramp_remaining <= 0:
                        self._active_ramp = None
                    else:
                        # Decay the ramp for next word
                        self._active_ramp = (ramp_mult * ramp_decay, ramp_remaining, ramp_decay)

                # Apply context modifier — only when trajectory is CLEARLY directional
                # In neutral zone (V 100-155), use base WORD_FORCES instead
                ctx_label = ""
                if ctx_mod and (self.v < 100 or self.v > 155):
                    cv, ca, cd, cu, cl = ctx_mod
                    vf = cv
                    af = ca
                    df = cd
                    uf = cu
                    gf = ctx_mod[4] if len(ctx_mod) > 4 and isinstance(ctx_mod[4], (int, float)) and not isinstance(ctx_mod[4], str) else gf
                    ctx_label = cl

                # Apply anticipation boost
                ant_label = ""
                if anticipation:
                    av, aa, ad, au, al = anticipation
                    af += aa  # anticipation mainly affects arousal/urgency
                    uf += au
                    ant_label = al

                # Momentum blending: new state = momentum * old + (1-momentum) * target + direct push
                # The "direct push" is what makes strong words override momentum
                # Weak words (low force) use reduced blending to avoid diluting emotional state
                total_force = abs(vf) + abs(af)
                push_strength = min(1.0, total_force / 60.0)  # stronger words push harder
                direct_push = push_strength * 0.6  # up to 60% direct force

                target_v = 128.0 + vf * force_scale
                target_a = 128.0 + af * force_scale
                target_d = 128.0 + df * force_scale
                target_u = uf * force_scale
                target_g = 128.0 + gf * force_scale

                # Hybrid exponential decay (Harrison 2024):
                # Strong words use momentum (emotional continuation).
                # Weak words after a spike recover exponentially toward center.
                # Gated behind config.exponential_decay for A/B testing.
                if self.config.exponential_decay and total_force > self._spike_threshold:
                    # STRONG word — use flat momentum, track spike
                    effective_momentum = self.momentum
                    self._last_spike_strength = total_force
                    self._words_since_spike = 0
                elif self.config.exponential_decay:
                    # WEAK word — exponential recovery toward center
                    self._words_since_spike += 1
                    if self._last_spike_strength > 0:
                        decay_factor = math.exp(-self._decay_lambda * self._words_since_spike)
                        effective_momentum = self._momentum_floor + self._momentum_range * decay_factor
                    else:
                        # No spike history — gentle drift
                        effective_momentum = 0.85
                else:
                    # Legacy flat momentum path
                    blend_scale = min(1.0, total_force / 30.0)
                    effective_momentum = 1.0 - (1.0 - self.momentum) * blend_scale

                # Crisis momentum lock: dampen non-strongly-negative words
                if self._crisis_momentum > 0:
                    if vf >= -30:
                        effective_momentum = max(effective_momentum, self.crisis_momentum_cap)
                        direct_push = direct_push * self.crisis_direct_push
                    self._crisis_momentum -= 1

                blend = 1.0 - effective_momentum
                self.v = self.v * effective_momentum + target_v * blend + vf * direct_push * force_scale
                self.a = self.a * effective_momentum + target_a * blend + af * direct_push * force_scale
                self.d = self.d * effective_momentum + target_d * blend + df * direct_push * force_scale
                self.u = self.u * effective_momentum + target_u * blend + uf * direct_push * force_scale
                self.g = self.g * effective_momentum + target_g * blend + gf * direct_push * force_scale

                # Suppress drift during crisis lock
                if self._crisis_momentum <= 0:
                    self._drift_toward_center()
                self.intensity = 1.0
                applied_force = True

                if ctx_label and force_source:
                    force_source = f"{force_source} + {ctx_label}"
                elif ctx_label:
                    force_source = ctx_label
                if ant_label:
                    force_source = f"{force_source} + {ant_label}" if force_source else ant_label

            else:
                # Bridge word — no force, just check context modifier
                if ctx_mod:
                    cv, ca, cd, cu, cl = ctx_mod
                    self.v += cv * 0.5
                    self.a += ca * 0.5
                    self.d += cd * 0.5
                    self.u += cu * 0.5
                    force_source = cl
                    applied_force = True
                else:
                    force_source = "(bridge)"
                    # Even bridge words still get anticipation if matched
                    if anticipation:
                        av, aa, ad, au, al = anticipation
                        self.a += aa * 0.3
                        self.u += au * 0.3
                        force_source = f"(bridge) + {al}"

                self.negate_next = False
                self.intensity = 1.0

        self._clamp()
        state_label = self._state_label()
        if force_source and force_source != "(bridge)":
            state_label = f"{force_source} | {state_label}"
        elif force_source == "(bridge)":
            state_label = f"(holds) {state_label}"

        trace_entry = {
            "word": word,
            "v": int(self.v), "a": int(self.a),
            "d": int(self.d), "u": int(self.u), "g": int(self.g),
            "state": state_label,
            "source": force_source or "(bridge)",
        }
        self.history.append(trace_entry)
        self.previous_words.append(word)
        return trace_entry

    def compute_density(self):
        """Compute emotional density — length-normalized VADUG.

        Uses sqrt of word count to normalize accumulated forces back to
        per-word emotional impact.  This makes short and long sentences
        comparable when the core emotion is the same but padding words
        cause accumulation drift.

        The scale factor is capped at 4.0 (16 words) to prevent
        over-normalization of very long inputs.
        """
        n = max(len(self.history), 1)
        sqrt_n = math.sqrt(n)

        # Cap scale to prevent over-normalization of long sentences
        scale = min(sqrt_n, 4.0)

        # Deviation from center, normalized by sqrt of word count
        dv = (self.v - 128) / scale
        da = (self.a - 128) / scale
        dd = (self.d - 128) / scale
        du = self.u / scale
        dg = (self.g - 128) / scale

        return {
            'v': max(0, min(255, int(128 + dv))),
            'a': max(0, min(255, int(128 + da))),
            'd': max(0, min(255, int(128 + dd))),
            'u': max(0, min(255, int(du))),
            'g': max(0, min(255, int(128 + dg))),
        }

    def apply_ending_weight(self, n_words):
        """After all words processed, weight the ending more heavily.

        Humans remember how things ENDED.  Re-run fresh pendulums over
        the first 30% and last 30% of words to detect the emotional arc
        direction, then blend the final V/G toward the ending's reading.

        The fresh pendulums start at neutral (128) so their states
        reflect only their words' forces, free of cross-sentence
        momentum contamination.
        """
        if n_words < 5:
            return  # too short for ending adjustment

        if not self.history or len(self.history) < n_words:
            return

        end_cutoff = int(n_words * 0.7)
        ending_words = [e['word'] for e in self.history[end_cutoff:]]
        if not ending_words:
            return

        # Run a FRESH pendulum on just the ending words so momentum
        # from earlier sentences doesn't contaminate the reading.
        fresh = SequentialPendulum(config=self.config)
        for idx, word in enumerate(ending_words):
            fresh.process_word(word, ending_words, idx)

        # Also run a fresh pendulum on the opening words to detect arc direction.
        start_cutoff = int(n_words * 0.3)
        opening_words = [e['word'] for e in self.history[:start_cutoff]]
        if opening_words:
            fresh_start = SequentialPendulum(config=self.config)
            for idx, word in enumerate(opening_words):
                fresh_start.process_word(word, opening_words, idx)
            start_v = fresh_start.v
            start_g = fresh_start.g
        else:
            start_v = 128.0
            start_g = 128.0

        # Detect arc: how far apart are the opening and ending reads?
        arc_span = abs(fresh.v - start_v)

        # Only apply ending weight when a real arc is detected.
        # Flat inputs (arc_span < 15) get no adjustment — they aren't arcs.
        # Moderate arcs (15-50) ramp from 0% to 30% ending pull.
        # Strong arcs (> 50) get a full 30% ending pull.
        if arc_span < 15:
            return  # no arc detected, skip adjustment
        elif arc_span > 50:
            ending_weight = 0.3
        else:
            ending_weight = 0.3 * ((arc_span - 15) / 35.0)

        current_weight = 1.0 - ending_weight
        self.v = self.v * current_weight + fresh.v * ending_weight
        self.g = self.g * current_weight + fresh.g * ending_weight
        self._clamp()

    def process_text(self, text):
        """Process full text word by word. Returns (VADU, history)."""
        words = re.findall(r"[a-z']+", text.lower())
        for idx, word in enumerate(words):
            self.process_word(word, words, idx)

        self.apply_ending_weight(len(words))

        final_vadug = VADUG(
            v=int(self.v),
            a=int(self.a),
            d=int(self.d),
            u=int(self.u),
            g=int(self.g)
        )
        return final_vadug, self.history

    def render_trace(self) -> str:
        """Render the word-by-word visual pendulum trace table."""
        lines = []
        lines.append("")
        lines.append("  SEQUENTIAL PENDULUM TRACE:")
        lines.append(f"  {'Word':<16} {'V':>4} {'A':>4} {'D':>4} {'U':>4} {'G':>4}   {'Visual':<24} State")
        lines.append(f"  {'─'*96}")

        for entry in self.history:
            word = entry["word"]
            v, a, d, u, g = entry["v"], entry["a"], entry["d"], entry["u"], entry["g"]
            state = entry["state"]

            # Build visual bar (16 chars wide)
            # Positive valence = filled blocks, arousal = mid blocks, empty = rest
            bar_width = 16
            # V determines how many solid blocks (left side = positive)
            v_ratio = v / 255.0
            a_ratio = a / 255.0

            # Solid blocks for valence (higher V = more solid)
            solid = int(v_ratio * bar_width * 0.6)
            # Mid blocks for arousal (higher A = more mid)
            mid = int(a_ratio * (bar_width - solid) * 0.7)
            empty = bar_width - solid - mid

            visual = "\u2588" * solid + "\u2593" * mid + "\u2591" * empty

            # Truncate state for display
            if len(state) > 40:
                state = state[:37] + "..."

            lines.append(f"  \"{word}\"{'':>{14-len(word)}} {v:>4} {a:>4} {d:>4} {u:>4} {g:>4}   {visual:<24} {state}")

        lines.append(f"  {'─'*96}")

        # Final summary
        if self.history:
            last = self.history[-1]
            lines.append(f"  FINAL:       V{last['v']} A{last['a']} D{last['d']} U{last['u']} G{last['g']}")

        return "\n".join(lines)


# Legacy wrapper for backward compatibility
def pendulum_parse(text: str):
    """Parse English text into VADUG using the sequential pendulum engine.

    Returns (VADUG, trace_lines) for compatibility with existing pipeline.
    """
    pend = SequentialPendulum()
    vadug, history = pend.process_text(text)
    # Build legacy-format trace lines
    trace = []
    for entry in history:
        trace.append(
            f"  '{entry['word']}' → V{entry['v']} A{entry['a']} "
            f"D{entry['d']} U{entry['u']} G{entry['g']}  [{entry['state']}]"
        )
    return vadug, trace, pend


