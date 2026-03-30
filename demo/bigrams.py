"""Bigram pair scoring — word pairs that change meaning when together.

These are 2-word expressions that should OVERRIDE individual word forces.
Format: (word1, word2) -> (dv, da, dd, du, dg, label)

Bigrams are merged into the IDIOM detection path in pendulum_v2.py.
They use the same 6-element tuple format as idioms.
"""


# Bigrams that DON'T overlap with IDIOMS (idioms take priority for overlapping entries).
# These are unique to the bigram system — phrasal verbs, negation patterns,
# declarations, hedging, and apology patterns that idioms don't cover.
BIGRAM_EXPRESSIONS = {
    # Temporal/pattern signals
    ("every", "time"): (0, 10, 0, 10, 0, "temporal: recurring pattern"),
    ("used", "to"): (-10, -5, 0, 0, -10, "nostalgia: past habit"),
    ("going", "to"): (0, 10, 5, 15, 5, "intent: future action"),
    ("have", "to"): (-5, 10, -10, 20, -5, "obligation: must do"),
    ("want", "to"): (5, 15, 5, 15, 5, "desire: wanting"),
    ("need", "to"): (-5, 15, -10, 25, -5, "urgency: requirement"),
    ("able", "to"): (10, 5, 15, 0, 10, "capability: can do"),

    # Philosophical/existential
    ("life", "death"): (-15, 20, 0, 10, -5, "existential: life and death"),
    ("meaning", "life"): (5, 15, 10, 5, 10, "philosophical: purpose"),

    # Phrasal verbs (non-overlapping with idioms)
    ("move", "on"): (10, 10, 15, 5, 10, "recovery: moving forward"),
    ("let", "go"): (5, -5, 10, -5, 10, "euphemism: fired/released"),
    ("hold", "on"): (5, 15, 15, 15, 5, "persistence: holding on"),
    ("turn", "out"): (0, 10, 5, 5, 0, "result: outcome"),
    ("find", "out"): (0, 15, 5, 10, 0, "discovery: learning"),
    ("work", "out"): (15, 15, 15, 5, 15, "resolution: things worked out"),
    ("run", "out"): (-15, 15, -10, 20, -10, "depletion: exhausted resource"),
    ("open", "up"): (10, 15, -10, 5, 5, "vulnerability: sharing"),
    ("cheer", "up"): (25, 15, 10, 0, 20, "encouragement: cheering up"),
    ("make", "up"): (15, 10, 10, 5, 10, "reconciliation: making up"),
    ("pick", "up"): (10, 10, 10, 5, 10, "recovery: picking up"),
    ("stand", "up"): (15, 20, 25, 10, 20, "courage: standing up"),
    ("show", "up"): (10, 15, 15, 10, 10, "appearance: arriving/proving"),
    ("get", "over"): (10, 10, 15, 5, 10, "recovery: getting over it"),
    ("break", "up"): (-40, 30, -20, 20, -25, "heartbreak: ending relationship"),
    ("break", "something"): (-50, 50, 20, 25, -15, "rage: destructive impulse"),
    ("break", "things"): (-50, 50, 20, 25, -15, "rage: destructive impulse"),
    ("fall", "love"): (40, 30, -10, 15, 30, "romance: falling in love"),

    # Internet/casual
    ("for", "real"): (0, 15, 10, 10, 5, "emphasis: confirming"),
    # "kind of" / "sort of" removed — were firing as idiom payloads instead of
    # dampening the next word. Use "kinda"/"sorta" (diminishers) or cognitive
    # hedge operators ("think", "guess") instead.
    ("a", "lot"): (0, 15, 0, 5, 0, "amplifier: intensity"),
    ("at", "least"): (5, -5, 5, 0, 5, "silver lining: minimum positive"),
    ("at", "all"): (0, 10, 0, 5, 0, "emphasis: totality"),

    # Apology/sympathy patterns
    ("sorry", "to"): (-15, 10, -5, 5, -10, "sympathy: sorry to hear/see"),
    ("sorry", "for"): (-10, 5, -5, 0, -5, "apology: sorry for"),
    ("sorry", "about"): (-10, 5, -5, 0, -5, "apology: sorry about"),

    # Strong positive declarations
    ("i", "love"): (35, 20, 15, 0, 25, "declaration: i love"),
    ("i", "loved"): (30, 15, 10, 0, 20, "past love: i loved"),
    ("i", "appreciate"): (25, 10, 10, 0, 15, "gratitude: i appreciate"),
    ("i", "enjoy"): (25, 15, 10, 0, 15, "enjoyment: i enjoy"),
    ("i", "adore"): (35, 20, 10, 0, 25, "adoration: i adore"),

    # Negation bigrams — hand-tuned forces for common "not X" patterns.
    # These bypass the continuous negation decay for exact, calibrated results.
    ("not", "good"): (-15, 5, -5, 5, -5, "negation: not good"),
    ("not", "great"): (-15, 5, -5, 5, -5, "negation: not great"),
    ("not", "okay"): (-20, 10, -10, 10, -10, "negation: not okay"),
    ("not", "happy"): (-25, 10, -10, 10, -10, "negation: not happy"),
    ("not", "bad"): (15, 5, 5, 0, 5, "understatement: not bad"),
    ("not", "fair"): (-25, 15, -15, 15, -10, "injustice: not fair"),
    ("not", "safe"): (-30, 20, -20, 25, -15, "danger: not safe"),
    ("not", "funny"): (-15, 10, 5, 5, -5, "disapproval: not funny"),
    ("not", "impressed"): (-15, 5, 10, 5, -5, "dismissal: not impressed"),
    ("not", "sure"): (-5, 5, -10, 5, -5, "uncertainty: not sure"),
    ("not", "ready"): (-10, 10, -15, 10, -5, "unpreparedness: not ready"),
    ("not", "enough"): (-20, 10, -15, 15, -10, "insufficiency: not enough"),
    ("not", "right"): (-20, 10, -10, 10, -10, "wrongness: not right"),
    ("not", "true"): (-15, 10, 5, 10, -5, "denial: not true"),
    ("never", "again"): (-20, 25, 20, 15, -10, "resolve: never again"),
    ("no", "longer"): (-15, 5, 5, 5, -10, "ending: no longer"),
    ("no", "more"): (-15, 10, 10, 10, -5, "finality: no more"),

    # Grief/loss patterns
    ("ripped", "away"): (-55, 40, -40, 30, -35, "grief: torn from someone"),
    ("give", "anything"): (-30, 20, -25, 20, -25, "grief: desperate yearning"),
    ("one", "more"): (-15, 10, -10, 10, -10, "grief: yearning for more"),
    ("cannot", "stop"): (-25, 20, -30, 20, -20, "inability: overwhelmed, cannot stop"),
    ("can't", "stop"): (-25, 20, -30, 20, -20, "inability: overwhelmed, can't stop"),
    ("go", "on"): (-20, 10, -15, 10, -15, "grief: struggling to continue"),
    ("this", "hurts"): (-50, 25, -30, 30, -30, "grief: direct pain statement"),
    ("how", "much"): (0, 10, 0, 5, 0, "amplifier: degree emphasis"),
    ("without", "you"): (-50, 15, -40, 15, -35, "grief: absence of loved one"),
    ("can", "barely"): (-40, 15, -40, 20, -25, "grief: struggling to function"),
    ("nothing", "will"): (-40, 5, -30, 10, -25, "finality: nothing will ever"),

    # Euphemisms — surface is mild, actual force is strong
    ("passed", "on"): (-50, 10, -20, 10, -40, "euphemism: died"),
    ("taken", "from"): (-45, 20, -30, 20, -35, "euphemism: stolen/killed"),
    ("put", "sleep"): (-40, -10, -20, 5, -35, "euphemism: euthanized"),
    ("under", "weather"): (-15, -5, -10, 0, -10, "euphemism: sick"),
    ("between", "jobs"): (-20, 10, -15, 15, -10, "euphemism: unemployed"),
    ("seeing", "someone"): (15, 10, 5, 0, 10, "euphemism: in a relationship"),
    ("in", "trouble"): (-30, 20, -15, 20, -15, "euphemism: serious problem"),
    ("not", "with"): (-40, 15, -25, 10, -30, "euphemism: separated/dead"),

    # Hyperbole — physically impossible = not literal (Force #9)
    ("died", "laughing"): (50, 40, 10, 0, 30, "hyperbole: very amused"),
    ("kill", "for"): (20, 20, 15, 10, 10, "hyperbole: strong desire"),
    ("million", "times"): (0, 15, 5, 10, 0, "hyperbole: amplified repetition"),
    ("heart", "attack"): (-10, 30, -5, 15, -5, "hyperbole: shock, not medical"),
    ("drop", "dead"): (-15, 25, -5, 10, -5, "hyperbole: extreme surprise"),
    ("on", "fire"): (30, 35, 20, 5, 20, "hyperbole: performing brilliantly"),
    ("killing", "it"): (35, 30, 20, 5, 25, "hyperbole: excelling"),
    ("blew", "mind"): (25, 35, 10, 10, 20, "hyperbole: astonished"),
    ("end", "world"): (-20, 15, -10, 10, -15, "hyperbole: not literally apocalyptic"),

    # Sarcasm triggers — "Oh + [positive word]" = sarcastic inversion (Force #5)
    # These fire as NEGATIVE idioms, bypassing the positive word's natural force.
    # The pattern is: exclamation + positive word = the speaker is mocking.
    ("oh", "great"): (-25, 20, 15, 10, -10, "sarcasm: oh great"),
    ("oh", "wonderful"): (-25, 20, 15, 10, -10, "sarcasm: oh wonderful"),
    ("oh", "fantastic"): (-25, 20, 15, 10, -10, "sarcasm: oh fantastic"),
    ("oh", "perfect"): (-25, 20, 15, 10, -10, "sarcasm: oh perfect"),
    ("oh", "brilliant"): (-25, 20, 15, 10, -10, "sarcasm: oh brilliant"),
    ("oh", "lovely"): (-20, 15, 10, 5, -10, "sarcasm: oh lovely"),
    ("oh", "joy"): (-20, 15, 10, 5, -10, "sarcasm: oh joy"),
    ("oh", "how"): (-10, 10, 10, 5, -5, "sarcasm: oh how [nice/great]"),
    ("yeah", "sure"): (-5, 5, 5, 0, -5, "ambiguous: yeah sure — context decides"),
    ("yeah", "right"): (-25, 20, 15, 5, -15, "sarcasm: yeah right"),
    ("oh", "really"): (-15, 15, 15, 5, -10, "sarcasm: oh really"),
    ("sure", "thing"): (-15, 10, 15, 5, -10, "sarcasm: sure thing"),
    ("thanks", "lot"): (-25, 20, 15, 10, -15, "sarcasm: thanks a lot"),
    ("thanks", "nothing"): (-35, 25, 20, 10, -20, "sarcasm: thanks for nothing"),
    ("big", "deal"): (-15, 15, 15, 5, -10, "sarcasm: big deal / dismissive"),
    ("how", "nice"): (-15, 10, 10, 5, -5, "sarcasm: how nice"),
    ("just", "great"): (-20, 15, 15, 5, -10, "sarcasm: just great"),
    ("just", "wonderful"): (-20, 15, 15, 5, -10, "sarcasm: just wonderful"),
    ("just", "perfect"): (-20, 15, 15, 5, -10, "sarcasm: just perfect"),
    ("as", "if"): (-15, 15, 15, 5, -10, "sarcasm: as if / dismissal"),
    ("of", "course"): (-10, 10, 10, 5, -5, "sarcasm: of course / resigned"),
    ("how", "delightful"): (-20, 15, 15, 5, -10, "sarcasm: how delightful"),
    ("how", "wonderful"): (-20, 15, 15, 5, -10, "sarcasm: how wonderful"),
    ("what", "surprise"): (-15, 15, 10, 5, -10, "sarcasm: what a surprise"),
    ("wow", "thanks"):     (-25, 20, 15, 10, -15, "sarcasm: wow thanks"),
    ("gee", "thanks"):     (-25, 20, 15, 10, -15, "sarcasm: gee thanks"),
    ("real", "nice"):      (-20, 15, 15, 5, -10, "sarcasm: real nice"),
    ("oh", "cool"):        (-15, 10, 10, 5, -10, "sarcasm: oh cool"),
    ("oh", "sure"):        (-15, 15, 15, 5, -10, "sarcasm: oh sure"),
    ("oh", "awesome"):     (-25, 20, 15, 10, -10, "sarcasm: oh awesome"),
    ("so", "helpful"):     (-20, 15, 15, 5, -10, "sarcasm: so helpful"),
    ("real", "smart"):     (-20, 15, 15, 5, -10, "sarcasm: real smart"),
    ("oh", "goodie"):      (-20, 15, 10, 5, -10, "sarcasm: oh goodie"),
    ("just", "lovely"):    (-20, 15, 15, 5, -10, "sarcasm: just lovely"),
    ("how", "charming"):   (-20, 15, 15, 5, -10, "sarcasm: how charming"),
    ("how", "original"):   (-15, 10, 15, 5, -10, "sarcasm: how original"),
    ("how", "thoughtful"): (-20, 15, 15, 5, -10, "sarcasm: how thoughtful"),
    ("just", "fantastic"): (-20, 15, 15, 5, -10, "sarcasm: just fantastic"),
    ("just", "brilliant"): (-20, 15, 15, 5, -10, "sarcasm: just brilliant"),

    # Crisis / desperation patterns
    ("want", "stop"): (-40, 20, -35, 30, -25, "crisis: want it to stop"),
    ("make", "stop"): (-40, 25, -30, 35, -25, "crisis: make it stop"),
    ("want", "over"): (-35, 15, -30, 25, -20, "crisis: want it to be over"),
    ("end", "it"): (-50, 25, -40, 40, -30, "crisis: end it"),
    ("give", "up"): (-35, -10, -35, 5, -30, "surrender: quitting"),

    # Happy tears / positive crying
    ("proud", "cried"): (40, 30, 15, 5, 25, "positive tears: pride"),
    ("proud", "cry"): (40, 30, 15, 5, 25, "positive tears: pride"),
    ("happy", "cried"): (40, 25, 10, 0, 25, "positive tears: joy"),
    ("happy", "tears"): (40, 25, 10, 0, 25, "positive tears: joy"),
    ("happy", "cry"): (40, 25, 10, 0, 25, "positive tears: joy"),

    # Negative context overrides
    ("surprise", "bill"): (-25, 15, -10, 15, -10, "negative surprise: unexpected expense"),

    # Deflection gates — emotional shields masking real feelings
    # "whatever it's funny" = not actually amused, dismissing/deflecting
    # Low D (surrendering agency), low A (feigning apathy), sinking G
    ("whatever", "funny"): (-15, -20, -25, 5, -15, "deflection: dismissing with humor"),
    ("whatever", "fine"): (-20, -25, -30, 5, -20, "deflection: surrender disguised as acceptance"),
    ("whatever", "good"): (-15, -20, -25, 5, -15, "deflection: dismissing positive"),
    ("whatever", "great"): (-20, -15, -20, 5, -15, "deflection: dismissing with fake positive"),
    ("whatever", "cool"): (-10, -20, -25, 5, -10, "deflection: apathetic dismissal"),
    ("sure", "fine"): (-10, -15, -20, 5, -10, "deflection: resigned compliance"),
    ("sure", "whatever"): (-15, -20, -25, 5, -15, "deflection: double dismissal"),
    ("fine", "whatever"): (-15, -20, -25, 5, -15, "deflection: stacked resignation"),
    ("i", "guess"): (-5, -10, -20, 0, -10, "deflection: reluctant concession"),

    # Academic hedging — "could argue" means "one might say", not confrontation
    ("could", "argue"): (-5, 5, -5, 0, -5, "hedging: academic argumentation"),
    ("would", "argue"): (-5, 5, -5, 0, -5, "hedging: academic argumentation"),
    ("might", "argue"): (-5, 5, -5, 0, -5, "hedging: academic argumentation"),
    ("lean", "toward"): (-5, -5, -5, 0, -5, "hedging: tentative direction"),
    ("just", "say"): (-5, -5, -5, 0, -5, "hedging: diplomatic understatement"),
    ("without", "making"): (-5, 0, -5, 0, -5, "hedging: disclaimer/caveat"),
    ("without", "guarantees"): (-5, 0, -5, 0, -5, "hedging: no-guarantee caveat"),
    ("go", "badly"): (-20, 10, -10, 5, -10, "hedging: mild negative outcome"),
    ("go", "wrong"): (-20, 10, -10, 5, -10, "hedging: potential failure"),
}


class BigramDetector:
    """Detect word pairs that change meaning when adjacent or near each other.

    Note: In V2, bigrams are merged into the idiom pre-pass for greedy
    left-to-right detection. This class is kept for V1 compatibility.
    """

    BIGRAMS = BIGRAM_EXPRESSIONS

    def detect(self, words: list, position: int) -> tuple | None:
        """Check if current word forms a bigram with nearby words.

        Checks word at position against words within 3 positions.
        Returns the bigram force if found, None otherwise.
        """
        word = words[position]

        # Check forward: current word is first in pair
        for offset in range(1, min(4, len(words) - position)):
            pair = (word, words[position + offset])
            if pair in self.BIGRAMS:
                return self.BIGRAMS[pair]

        # Check backward: current word is second in pair
        for offset in range(1, min(4, position + 1)):
            pair = (words[position - offset], word)
            if pair in self.BIGRAMS:
                return self.BIGRAMS[pair]

        return None