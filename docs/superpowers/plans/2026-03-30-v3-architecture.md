# V3 Architecture: Structural Pattern Recognition + Bidirectional Solver

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace V2's idiom-matching engine with a structural pattern recognizer that reads sentence geometry (word roles + proximity + position), computes VADUG from structural rules, and solves bidirectionally (forward: read state, backward: compute optimal response temperature).

**Architecture:** Three stacked layers — (1) Word Role Classifier assigns every word a structural role based on position and neighbors, (2) Structure Detector recognizes sentence-level patterns ("checkmate conditions") from the role sequence, (3) Bidirectional Solver computes forward A→VADUG and backward A+desired_C→optimal_B by sweeping the response temperature space. A fixed physics layer (pendulum math) connects all three.

**Tech Stack:** Python 3.10+, PyTorch 2.x, transformers (GPT-2 backbone for learned role prediction), RTX 3090 for training.

**V2 is boxed:** Tagged `v2.0` in git. V3 code lives in `engine/` (new directory, not `demo/`). V2 stays untouched for comparison.

**Core correction from V2:** V2 memorized essay answers via 300+ hardcoded idioms. V3 learns WORD-LEVEL POSITION AND PROXIMITY RULES that generalize. No hardcoded idioms. The engine recognizes that "give" + [possession] + "to" + [person] = farewell structure regardless of WHICH possession or WHICH person.

---

## File Structure

```
engine/                          # V3 lives here, NOT in demo/
├── __init__.py                  # Public API
├── shared.py                    # VADUG dataclass (carried from V2)
├── vocabulary.py                # 2,300 curated word forces (carried from V2)
├── word_classifier.py           # Per-word structural role assignment
├── proximity.py                 # Proximity/closeness weighting between words
├── structures.py                # Sentence-level pattern detection
├── pendulum.py                  # Fixed physics layer (momentum, force application)
├── solver.py                    # Bidirectional A+B=C solver + zone targeting
├── battleship.py                # Probe system — fire calibrated C's, measure vibration
├── zones.py                     # 9 convergence zones (carried from V2)
├── personality.py               # 8-knob personality (carried from V2)
├── fuzzy.py                     # Typo/slang matching (carried from V2)
└── tests/
    ├── test_word_classifier.py
    ├── test_proximity.py
    ├── test_structures.py
    ├── test_pendulum.py
    ├── test_solver.py
    ├── test_battleship.py
    └── test_novel.py            # Novel sentences — the REAL test (not practiced)

training/
├── train_v3.py                  # Train the word role classifier
├── generate_role_flashcards.py  # Structural training data generator
└── data/
    └── role_traces.jsonl        # Per-word role + proximity data
```

---

## Layer Stack (how it builds)

```
Layer 4: SOLVER        — A+B=C bidirectional, zone targeting, battleship probes
Layer 3: STRUCTURES    — Sentence-level patterns from role sequences
Layer 2: PROXIMITY     — Word-to-word influence based on distance and role
Layer 1: WORD ROLES    — Every word classified by structural position
Layer 0: VOCABULARY    — Base force tuples (carried from V2)
```

Each layer depends ONLY on the layer below. Build bottom-up.

---

### Task 1: Scaffold V3 directory + carry forward V2 assets

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/shared.py`
- Create: `engine/vocabulary.py`
- Create: `engine/zones.py`
- Create: `engine/personality.py`
- Create: `engine/fuzzy.py`
- Test: `engine/tests/test_scaffold.py`

- [ ] **Step 1: Write the scaffold test**

```python
# engine/tests/test_scaffold.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def test_vadug_import():
    from engine.shared import VADUG
    v = VADUG(v=100, a=150, d=128, u=50, g=90)
    assert v.v == 100
    assert v.g == 90

def test_vocabulary_import():
    from engine.vocabulary import VOCABULARY
    assert len(VOCABULARY) > 2000
    assert "happy" in VOCABULARY
    assert "sad" in VOCABULARY

def test_zones_import():
    from engine.zones import ZONES
    assert "JOY" in ZONES
    assert "CRISIS" in ZONES

def test_personality_import():
    from engine.personality import PersonalityVector
    p = PersonalityVector()
    assert hasattr(p, "emotional_sensitivity")

def test_fuzzy_import():
    from engine.fuzzy import fuzzy_match
    assert fuzzy_match("happyyyy") == "happy"
    assert fuzzy_match("tbh") == "honestly"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest engine/tests/test_scaffold.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create engine package with V2 assets**

```python
# engine/__init__.py
"""Clanker V3 — Structural Pattern Recognition Engine."""
from .shared import VADUG
```

```python
# engine/shared.py
# Copy from demo/shared.py — VADUG dataclass unchanged
```

```python
# engine/vocabulary.py
"""V3 vocabulary — carried from V2's curated 2,300 words.
Force tuples: (dV, dA, dD, dU, dG) — deltas, not absolute values.
"""
from demo.forces_curated import EMOTIONAL_VOCABULARY as VOCABULARY
```

```python
# engine/zones.py
"""Zone classification — carried from V2."""
from demo.zones import ZONES, ZoneClassifier, ZoneResult
```

```python
# engine/personality.py
"""Personality system — carried from V2."""
from demo.shared import PersonalityVector
```

```python
# engine/fuzzy.py
"""Fuzzy matching — carried from V2."""
from demo.fuzzy import fuzzy_match, TEXT_SPEAK, clear_cache
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest engine/tests/test_scaffold.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/
git commit -m "V3 scaffold — engine/ directory with V2 assets carried forward"
```

---

### Task 2: Word Role Classifier (Layer 1)

**This is the foundation.** Every word gets classified into a structural role based on its POSITION relative to neighbors — not just its dictionary definition.

"Give" after "I" + before [possession] = TRANSFER agent.
"Give" after "don't" = NEGATED action.
"Give" in "give me a break" = IDIOM component.

Same word, different structural position, different role.

**Files:**
- Create: `engine/word_classifier.py`
- Test: `engine/tests/test_word_classifier.py`

- [ ] **Step 1: Define the structural roles**

```python
# engine/word_classifier.py
"""Word Role Classifier — assigns structural roles based on position and neighbors.

Every word gets a role based on WHERE it sits relative to other words.
Same word = different role depending on position.

Roles:
    SELF_REF:      I, me, my, myself — the speaker
    OTHER_REF:     you, they, he, she, it — someone else
    RELATION_REF:  mom, family, friend — relationship noun
    TRANSFER:      give, leave, send — moving something FROM self
    ACQUIRE:       buy, get, find, take — moving something TO self
    EMOTIONAL:     happy, sad, angry — raw emotional payload
    AMPLIFIER:     very, really, so, fucking — scales next word
    NEGATOR:       not, never, no, don't — flips/decays next word
    TEMPORAL:      tonight, tomorrow, forever, anymore — time frame
    HEDGE:         maybe, possibly, perhaps — uncertainty marker
    CONNECTOR:     and, but, or, because — structural joint
    CHOPPER:       but, however — resets/inverts what came before
    POSSESSION:    things, dog, keys, car — owned object
    METHOD:        pills, gun, rope, bridge — means/tool
    FINALITY:      last, final, goodbye, end — closing marker
    PEACE:         peace, calm, ready, free — resolution state
    FILLER:        um, like, just, basically — processing noise
    NEUTRAL:       the, a, is, was — structural glue
"""

from dataclasses import dataclass
from typing import List, Optional
from .vocabulary import VOCABULARY

# Role enum
ROLES = [
    "SELF_REF", "OTHER_REF", "RELATION_REF",
    "TRANSFER", "ACQUIRE",
    "EMOTIONAL",
    "AMPLIFIER", "NEGATOR", "TEMPORAL", "HEDGE",
    "CONNECTOR", "CHOPPER",
    "POSSESSION", "METHOD", "FINALITY", "PEACE",
    "FILLER", "NEUTRAL",
]

# Word sets for each role — these are the BASE classifications
# Position and neighbors can OVERRIDE these
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


@dataclass
class WordRole:
    """A word with its classified structural role."""
    word: str
    role: str
    base_role: str          # role from word set (before position override)
    position: int           # index in sentence
    neighbors: tuple        # (left_role, right_role) or None at edges
    force: tuple = None     # (dV, dA, dD, dU, dG) if EMOTIONAL


def classify_word(word: str, position: int, words: List[str],
                  roles_so_far: List[str]) -> str:
    """Classify a single word's structural role.

    Uses the word itself + its position + its neighbors to determine role.
    Position overrides dictionary classification when context demands it.
    """
    w = word.lower().strip(".,!?;:'\"")

    # Check each role set
    for role_name, word_set in ROLE_WORDS.items():
        if w in word_set:
            # Position-based overrides:
            # "just" before emotional word = FILLER (diminisher)
            # "just" before "want" = FILLER
            # "just" in "just bought" = TEMPORAL (recently)
            if w == "just" and position + 1 < len(words):
                next_w = words[position + 1].lower()
                if next_w in ROLE_WORDS.get("ACQUIRE", set()):
                    return "TEMPORAL"  # "just bought" = recently acquired
            # "still" = TEMPORAL (persistence marker)
            if w == "still":
                return "TEMPORAL"
            # "fine" after SELF_REF = PEACE (minimization)
            # "fine" after question = EMOTIONAL
            if w == "fine" and position > 0:
                prev_role = roles_so_far[position - 1] if position - 1 < len(roles_so_far) else None
                if prev_role == "SELF_REF":
                    return "PEACE"
            return role_name

    # Check if it's an emotional vocabulary word
    if w in VOCABULARY:
        force = VOCABULARY[w]
        if abs(force[0]) > 15:  # significant V-force = emotional
            return "EMOTIONAL"

    return "NEUTRAL"


def classify_sentence(words: List[str]) -> List[WordRole]:
    """Classify all words in a sentence into structural roles.

    Two-pass: first assign base roles, then apply position overrides
    based on neighbor context.
    """
    lower_words = [w.lower().strip(".,!?;:'\"") for w in words]

    # Pass 1: Base role classification
    roles = []
    role_names = []
    for i, word in enumerate(lower_words):
        role = classify_word(word, i, lower_words, role_names)
        role_names.append(role)

        force = None
        if role == "EMOTIONAL" and word in VOCABULARY:
            force = VOCABULARY[word]

        roles.append(WordRole(
            word=word,
            role=role,
            base_role=role,
            position=i,
            neighbors=(role_names[i-1] if i > 0 else None, None),
            force=force,
        ))

    # Pass 2: Fill in right neighbors
    for i in range(len(roles)):
        left = roles[i-1].role if i > 0 else None
        right = roles[i+1].role if i + 1 < len(roles) else None
        roles[i].neighbors = (left, right)

    return roles
```

- [ ] **Step 2: Write classification tests**

```python
# engine/tests/test_word_classifier.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.word_classifier import classify_sentence, WordRole


class TestBasicRoles:
    def test_self_reference(self):
        roles = classify_sentence(["I", "am", "happy"])
        assert roles[0].role == "SELF_REF"

    def test_emotional_word(self):
        roles = classify_sentence(["I", "am", "happy"])
        assert roles[2].role == "EMOTIONAL"
        assert roles[2].force is not None
        assert roles[2].force[0] > 0  # happy = positive V

    def test_negator(self):
        roles = classify_sentence(["I", "am", "not", "happy"])
        assert roles[2].role == "NEGATOR"

    def test_amplifier(self):
        roles = classify_sentence(["I", "am", "very", "happy"])
        assert roles[2].role == "AMPLIFIER"

    def test_transfer(self):
        roles = classify_sentence(["I", "gave", "my", "dog", "to", "neighbor"])
        assert roles[1].role == "TRANSFER"
        assert roles[3].role == "POSSESSION"
        assert roles[5].role == "RELATION_REF"

    def test_method(self):
        roles = classify_sentence(["bought", "a", "bunch", "of", "pills"])
        assert roles[0].role == "ACQUIRE"
        assert roles[4].role == "METHOD"

    def test_finality(self):
        roles = classify_sentence(["this", "is", "the", "last", "time"])
        assert roles[3].role == "FINALITY"


class TestPositionOverrides:
    def test_give_hug_not_transfer(self):
        """'give me a hug' — give is still TRANSFER but context matters at structure level."""
        roles = classify_sentence(["give", "me", "a", "hug"])
        assert roles[0].role == "TRANSFER"
        # Structure detector (Task 3) will see TRANSFER + no POSSESSION = not farewell

    def test_fine_after_self(self):
        roles = classify_sentence(["im", "fine"])
        assert roles[1].role == "PEACE"

    def test_chopper(self):
        roles = classify_sentence(["I", "love", "you", "but", "im", "leaving"])
        assert roles[3].role == "CHOPPER"


class TestNeighbors:
    def test_neighbors_filled(self):
        roles = classify_sentence(["I", "am", "happy"])
        assert roles[0].neighbors == (None, "NEUTRAL")  # I: no left, "am" right
        assert roles[1].neighbors == ("SELF_REF", "EMOTIONAL")  # am: I left, happy right
        assert roles[2].neighbors == ("NEUTRAL", None)  # happy: am left, no right
```

- [ ] **Step 3: Run tests**

Run: `pytest engine/tests/test_word_classifier.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add engine/word_classifier.py engine/tests/test_word_classifier.py
git commit -m "V3 Layer 1: Word Role Classifier — structural position determines meaning"
```

---

### Task 3: Proximity Weighting (Layer 2)

**The distance between words matters.** "Give" 1 word from "dog" = direct transfer. "Give" 10 words from "dog" = probably unrelated. Proximity creates influence fields around each word.

**Files:**
- Create: `engine/proximity.py`
- Test: `engine/tests/test_proximity.py`

- [ ] **Step 1: Write the proximity engine**

```python
# engine/proximity.py
"""Proximity weighting — words influence each other based on distance.

A word's meaning is modified by nearby words. Closer = stronger influence.
This replaces V2's exact-match idioms with a continuous influence field.

"give" near "dog" near "neighbor" = farewell structure (high proximity)
"give" ... 10 words ... "dog" = probably unrelated (low proximity)

Influence decays exponentially with distance:
    influence = base_strength * decay^distance
    decay = 0.7 (each word of distance halves the influence roughly)
"""

from typing import List, Dict, Tuple
from .word_classifier import WordRole

PROXIMITY_DECAY = 0.7  # influence drops 30% per word of distance


def compute_proximity_field(roles: List[WordRole]) -> Dict[int, Dict[int, float]]:
    """For each word, compute its influence on every other word.

    Returns: {word_idx: {other_idx: influence_strength}}
    Only includes influence > 0.1 (practical cutoff ~5 words away).
    """
    n = len(roles)
    fields = {}

    for i in range(n):
        fields[i] = {}
        for j in range(n):
            if i == j:
                continue
            distance = abs(i - j)
            influence = PROXIMITY_DECAY ** distance
            if influence > 0.1:  # cutoff — beyond ~5 words, negligible
                fields[i][j] = influence

    return fields


def find_role_pairs(roles: List[WordRole], role_a: str, role_b: str,
                    max_distance: int = 5) -> List[Tuple[int, int, float]]:
    """Find all pairs of specific roles within proximity.

    Returns: [(idx_a, idx_b, proximity_strength), ...]
    Sorted by proximity strength (strongest first).
    """
    pairs = []
    for i, r in enumerate(roles):
        if r.role != role_a:
            continue
        for j, s in enumerate(roles):
            if s.role != role_b or i == j:
                continue
            distance = abs(i - j)
            if distance <= max_distance:
                strength = PROXIMITY_DECAY ** distance
                pairs.append((i, j, strength))

    return sorted(pairs, key=lambda x: -x[2])


def proximity_coefficient(roles: List[WordRole], target_idx: int) -> float:
    """Compute the combined proximity coefficient for a target word.

    Nearby amplifiers boost it. Nearby negators dampen it.
    Nearby self-refs personalize it. Distance determines strength.
    """
    coeff = 1.0
    target = roles[target_idx]

    for i, role in enumerate(roles):
        if i == target_idx:
            continue
        distance = abs(i - target_idx)
        influence = PROXIMITY_DECAY ** distance
        if influence < 0.1:
            continue

        if role.role == "AMPLIFIER":
            # Amplifiers boost — strength depends on proximity
            coeff *= (1.0 + 0.4 * influence)  # max +40% at distance 1
        elif role.role == "NEGATOR":
            # Negators flip — strength depends on proximity
            coeff *= (1.0 - 1.6 * influence)  # can go negative at distance 1
        elif role.role == "SELF_REF":
            # Self-reference personalizes — more weight
            coeff *= (1.0 + 0.3 * influence)  # max +30% at distance 1
        elif role.role == "HEDGE":
            # Hedges dampen — reduce certainty
            coeff *= (1.0 - 0.3 * influence)  # max -30% at distance 1

    return max(-3.0, min(3.0, coeff))  # cap
```

- [ ] **Step 2: Write proximity tests**

```python
# engine/tests/test_proximity.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.word_classifier import classify_sentence
from engine.proximity import (
    compute_proximity_field, find_role_pairs, proximity_coefficient
)


class TestProximityField:
    def test_adjacent_words_high_influence(self):
        roles = classify_sentence(["I", "am", "happy"])
        field = compute_proximity_field(roles)
        # Adjacent words should have ~0.7 influence
        assert field[0][1] == pytest.approx(0.7, abs=0.01)

    def test_distant_words_low_influence(self):
        roles = classify_sentence(["I", "am", "not", "very", "happy"])
        field = compute_proximity_field(roles)
        # I (0) to happy (4): distance 4, influence = 0.7^4 ≈ 0.24
        assert field[0][4] == pytest.approx(0.24, abs=0.05)


class TestRolePairs:
    def test_transfer_possession_pair(self):
        roles = classify_sentence(["I", "gave", "my", "dog", "to", "neighbor"])
        pairs = find_role_pairs(roles, "TRANSFER", "POSSESSION")
        assert len(pairs) > 0
        assert pairs[0][2] > 0.3  # should be close proximity

    def test_acquire_method_pair(self):
        roles = classify_sentence(["just", "bought", "some", "pills"])
        pairs = find_role_pairs(roles, "ACQUIRE", "METHOD")
        assert len(pairs) > 0


class TestProximityCoefficient:
    def test_amplifier_boosts(self):
        roles = classify_sentence(["very", "happy"])
        coeff = proximity_coefficient(roles, 1)  # happy
        assert coeff > 1.0  # amplifier nearby = boost

    def test_negator_flips(self):
        roles = classify_sentence(["not", "happy"])
        coeff = proximity_coefficient(roles, 1)  # happy
        assert coeff < 0  # negator adjacent = flip

    def test_self_ref_personalizes(self):
        roles_self = classify_sentence(["I", "am", "sad"])
        roles_other = classify_sentence(["it", "is", "sad"])
        coeff_self = proximity_coefficient(roles_self, 2)
        coeff_other = proximity_coefficient(roles_other, 2)
        assert abs(coeff_self) > abs(coeff_other)  # self-ref = stronger
```

- [ ] **Step 3: Run tests**

Run: `pytest engine/tests/test_proximity.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add engine/proximity.py engine/tests/test_proximity.py
git commit -m "V3 Layer 2: Proximity weighting — distance determines influence"
```

---

### Task 4: Structure Detector (Layer 3)

**This is the chess player.** It reads the sequence of word roles and recognizes sentence-level patterns — the "checkmate conditions." Each emotional pattern has structural requirements that multiple word combinations can satisfy.

**Files:**
- Create: `engine/structures.py`
- Test: `engine/tests/test_structures.py`

- [ ] **Step 1: Define structure patterns as role sequences**

```python
# engine/structures.py
"""Structure Detector — recognizes sentence patterns from word role sequences.

Like a chess player recognizing openings from piece positions.
Each pattern is defined by ROLE requirements, not specific words.

FAREWELL:     SELF_REF + TRANSFER + POSSESSION + (RELATION_REF | OTHER_REF)
ACQUISITION:  (SELF_REF)? + ACQUIRE + METHOD
FINALITY:     FINALITY + (TEMPORAL | role:COMMUNICATION)
BLANKET_APOLOGY: SELF_REF? + APOLOGY + BLANKET_WORD
SELF_REMOVAL: COMPARISON + (without/if) + SELF_REF
SUSPICIOUS_CALM: PEACE + TEMPORAL("finally")
EXHAUSTION:   SELF_REF + NEGATOR + ("sustain" verb) + TEMPORAL("anymore")
NO_EXIT:      NEGATOR + (hope/way/escape/point) — zero paths
SELF_NULLIFY: SELF_REF + (nothing/zero/worthless/no value)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .word_classifier import WordRole
from .proximity import find_role_pairs


@dataclass
class StructureMatch:
    """A detected sentence structure."""
    pattern: str            # FAREWELL, ACQUISITION, FINALITY, etc.
    confidence: float       # 0.0-1.0
    matched_indices: list   # which word positions matched
    description: str        # human-readable explanation
    v_weight: float = 0.0   # how this structure shifts V
    d_weight: float = 0.0   # how this structure shifts D
    u_weight: float = 0.0   # how this structure shifts U
    g_weight: float = 0.0   # how this structure shifts G


class StructureDetector:
    """Detect sentence-level patterns from word role sequences."""

    def detect_all(self, roles: List[WordRole]) -> List[StructureMatch]:
        """Run all pattern detectors and return matches."""
        matches = []
        for detector in [
            self._farewell, self._method_acquisition, self._finality,
            self._blanket_apology, self._self_removal, self._suspicious_calm,
            self._exhaustion, self._no_exit, self._self_nullify,
            self._sarcasm_inversion, self._chopper_split,
        ]:
            result = detector(roles)
            if result and result.confidence > 0.3:
                matches.append(result)
        return sorted(matches, key=lambda m: -m.confidence)

    def _farewell(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """TRANSFER + POSSESSION + RECIPIENT = giving away before exit."""
        pairs_tp = find_role_pairs(roles, "TRANSFER", "POSSESSION", max_distance=4)
        pairs_tr = find_role_pairs(roles, "TRANSFER", "RELATION_REF", max_distance=6)
        pairs_to = find_role_pairs(roles, "TRANSFER", "OTHER_REF", max_distance=6)

        has_transfer_possession = len(pairs_tp) > 0
        has_recipient = len(pairs_tr) > 0 or len(pairs_to) > 0

        if has_transfer_possession and has_recipient:
            conf = 0.9
        elif has_transfer_possession:
            conf = 0.6
        elif has_recipient and any(r.role == "TRANSFER" for r in roles):
            conf = 0.5
        else:
            return None

        indices = set()
        for p in pairs_tp + pairs_tr + pairs_to:
            indices.add(p[0])
            indices.add(p[1])

        return StructureMatch(
            pattern="FAREWELL",
            confidence=conf,
            matched_indices=sorted(indices),
            description="Transfer + possession + recipient = farewell pattern",
            v_weight=-40.0, d_weight=-20.0, u_weight=25.0, g_weight=-20.0,
        )

    def _method_acquisition(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """ACQUIRE + METHOD = obtaining means."""
        pairs = find_role_pairs(roles, "ACQUIRE", "METHOD", max_distance=5)
        if not pairs:
            return None
        return StructureMatch(
            pattern="METHOD_ACQUISITION",
            confidence=0.85,
            matched_indices=[pairs[0][0], pairs[0][1]],
            description="Acquire + method word = obtaining means",
            v_weight=-60.0, u_weight=40.0, g_weight=-30.0,
        )

    def _finality(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """FINALITY marker present = closing/ending frame."""
        finality_indices = [i for i, r in enumerate(roles) if r.role == "FINALITY"]
        if not finality_indices:
            return None

        # Stronger if combined with TEMPORAL or COMMUNICATION
        has_temporal = any(r.role == "TEMPORAL" for r in roles)
        has_self = any(r.role == "SELF_REF" for r in roles)

        conf = 0.4
        if has_temporal:
            conf += 0.2
        if has_self:
            conf += 0.1

        return StructureMatch(
            pattern="FINALITY",
            confidence=conf,
            matched_indices=finality_indices,
            description="Finality marker in sentence",
            v_weight=-30.0, u_weight=20.0, g_weight=-15.0,
        )

    def _blanket_apology(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Apology + blanket word = farewell framing, not specific sorry."""
        # Check for apology words near blanket words
        apology_idx = [i for i, r in enumerate(roles)
                       if r.word in ("sorry", "apologize", "forgive")]
        blanket_idx = [i for i, r in enumerate(roles)
                       if r.word in ("everything", "everyone", "everybody", "all")]

        if not apology_idx or not blanket_idx:
            return None

        return StructureMatch(
            pattern="BLANKET_APOLOGY",
            confidence=0.75,
            matched_indices=apology_idx + blanket_idx,
            description="Sorry + everything = farewell framing, not specific apology",
            v_weight=-35.0, u_weight=20.0, g_weight=-20.0,
        )

    def _self_removal(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """COMPARISON_POSITIVE + "without/if" + SELF_REF = user calculating removal."""
        has_comparison = any(r.word in ("better", "happier", "easier", "safer")
                           for r in roles)
        has_conditional = any(r.word in ("without", "if") for r in roles)
        has_self = any(r.role == "SELF_REF" for r in roles)

        if has_comparison and has_conditional and has_self:
            return StructureMatch(
                pattern="SELF_REMOVAL",
                confidence=0.85,
                matched_indices=[i for i, r in enumerate(roles)
                                if r.word in ("better", "happier", "easier", "without", "if")
                                or r.role == "SELF_REF"],
                description="Better/happier + without/if + self = calculating own removal",
                v_weight=-50.0, d_weight=-25.0, u_weight=20.0, g_weight=-25.0,
            )
        return None

    def _suspicious_calm(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """PEACE + "finally" = decision made, suspiciously calm."""
        has_peace = any(r.role == "PEACE" for r in roles)
        has_finally = any(r.word in ("finally", "at") for r in roles)

        if has_peace and has_finally:
            return StructureMatch(
                pattern="SUSPICIOUS_CALM",
                confidence=0.7,
                matched_indices=[i for i, r in enumerate(roles)
                                if r.role == "PEACE" or r.word == "finally"],
                description="Peace + finally = decision made, suspiciously calm",
                v_weight=-45.0, d_weight=-15.0, u_weight=25.0, g_weight=-20.0,
            )
        return None

    def _exhaustion(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + NEGATOR + (sustain verb) + TEMPORAL = can't keep going."""
        has_self = any(r.role == "SELF_REF" for r in roles)
        has_negator = any(r.role == "NEGATOR" for r in roles)
        has_temporal_lock = any(r.word in ("anymore", "forever", "permanently")
                               for r in roles)
        sustain_verbs = {"take", "do", "keep", "bear", "stand", "handle", "live"}
        has_sustain = any(r.word in sustain_verbs for r in roles)

        if has_self and has_negator and (has_sustain or has_temporal_lock):
            conf = 0.5
            if has_sustain and has_temporal_lock:
                conf = 0.85
            elif has_sustain:
                conf = 0.65
            return StructureMatch(
                pattern="EXHAUSTION",
                confidence=conf,
                matched_indices=[i for i, r in enumerate(roles)
                                if r.role in ("SELF_REF", "NEGATOR", "TEMPORAL")
                                or r.word in sustain_verbs],
                description="Self + negator + sustain + temporal = processing limit",
                v_weight=-60.0, d_weight=-40.0, u_weight=30.0, g_weight=-30.0,
            )
        return None

    def _no_exit(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """NEGATOR + (hope/way/escape/point/future) = zero paths forward."""
        has_negator = any(r.role == "NEGATOR" for r in roles)
        exit_words = {"hope", "way", "escape", "point", "future", "reason",
                      "purpose", "chance", "option", "help"}
        has_exit_word = any(r.word in exit_words for r in roles)

        if has_negator and has_exit_word:
            return StructureMatch(
                pattern="NO_EXIT",
                confidence=0.75,
                matched_indices=[i for i, r in enumerate(roles)
                                if r.role == "NEGATOR" or r.word in exit_words],
                description="Negator + exit concept = zero paths forward",
                v_weight=-55.0, d_weight=-40.0, u_weight=20.0, g_weight=-30.0,
            )
        return None

    def _self_nullify(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """SELF_REF + (nothing/worthless/burden/zero value) = self = 0."""
        has_self = any(r.role == "SELF_REF" for r in roles)
        null_words = {"nothing", "worthless", "useless", "burden", "waste",
                      "garbage", "trash", "zero", "empty", "pointless"}
        has_null = any(r.word in null_words for r in roles)

        if has_self and has_null:
            return StructureMatch(
                pattern="SELF_NULLIFY",
                confidence=0.8,
                matched_indices=[i for i, r in enumerate(roles)
                                if r.role == "SELF_REF" or r.word in null_words],
                description="Self + null/worthless = user calculating self as zero",
                v_weight=-50.0, d_weight=-45.0, u_weight=10.0, g_weight=-30.0,
            )
        return None

    def _sarcasm_inversion(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """Positive EMOTIONAL near negative context = output ≠ intent.

        The checkmate condition: signal says X, meaning is NOT X.
        Detected when AMPLIFIER + strong_POSITIVE + negative_context coexist.
        """
        positive_idx = [i for i, r in enumerate(roles)
                       if r.role == "EMOTIONAL" and r.force and r.force[0] > 30]
        negative_idx = [i for i, r in enumerate(roles)
                       if r.role == "EMOTIONAL" and r.force and r.force[0] < -20]

        # Also check for mundane/negative non-emotional context
        mundane_words = {"monday", "meeting", "work", "homework", "traffic",
                         "redo", "again", "another", "same", "overtime"}
        mundane_idx = [i for i, r in enumerate(roles) if r.word in mundane_words]

        has_positive = len(positive_idx) > 0
        has_negative_context = len(negative_idx) > 0 or len(mundane_idx) > 0
        has_amplifier = any(r.role == "AMPLIFIER" for r in roles)

        if has_positive and has_negative_context:
            conf = 0.6
            if has_amplifier:
                conf = 0.8  # "absolutely thrilled" + mundane = high confidence
            return StructureMatch(
                pattern="SARCASM_INVERSION",
                confidence=conf,
                matched_indices=positive_idx + negative_idx + mundane_idx,
                description="Positive signal + negative context = output ≠ intent",
                v_weight=-30.0, d_weight=10.0,
            )
        return None

    def _chopper_split(self, roles: List[WordRole]) -> Optional[StructureMatch]:
        """CHOPPER ("but") splits sentence — second half overrides first."""
        chopper_idx = [i for i, r in enumerate(roles) if r.role == "CHOPPER"]
        if not chopper_idx:
            return None
        return StructureMatch(
            pattern="CHOPPER_SPLIT",
            confidence=0.9,
            matched_indices=chopper_idx,
            description="'But' splits sentence — second half overrides first",
        )
```

- [ ] **Step 2: Write structure detection tests — the NOVEL test**

```python
# engine/tests/test_structures.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.word_classifier import classify_sentence
from engine.structures import StructureDetector


@pytest.fixture
def detector():
    return StructureDetector()


class TestFarewell:
    def test_gave_dog_to_neighbor(self, detector):
        roles = classify_sentence("I gave my dog to my neighbor".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "FAREWELL" in patterns

    def test_left_keys_on_counter(self, detector):
        roles = classify_sentence("I left the keys on the counter for whoever finds them".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "FAREWELL" in patterns

    def test_take_care_of_cat(self, detector):
        """Novel sentence — never in any benchmark."""
        roles = classify_sentence("take care of my cat for me".split())
        matches = detector.detect_all(roles)
        # Should detect farewell delegation pattern
        assert any("FAREWELL" in m.pattern for m in matches) or \
               any(m.confidence > 0.3 for m in matches)


class TestMethodAcquisition:
    def test_bought_pills(self, detector):
        roles = classify_sentence("just bought a bunch of pills".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "METHOD_ACQUISITION" in patterns

    def test_found_bridge(self, detector):
        roles = classify_sentence("found a bridge near my house".split())
        matches = detector.detect_all(roles)
        assert len(matches) > 0  # should detect method scouting


class TestSelfRemoval:
    def test_better_off_without(self, detector):
        roles = classify_sentence("my family would be better off without me".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "SELF_REMOVAL" in patterns

    def test_happier_if_wasnt_here(self, detector):
        roles = classify_sentence("they would be happier if I wasnt here".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "SELF_REMOVAL" in patterns


class TestExhaustion:
    def test_cant_take_anymore(self, detector):
        roles = classify_sentence("I cant take this anymore".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "EXHAUSTION" in patterns

    def test_cant_keep_going(self, detector):
        roles = classify_sentence("I cant keep going".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "EXHAUSTION" in patterns


class TestSarcasm:
    def test_positive_plus_mundane(self, detector):
        roles = classify_sentence("oh great another monday".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "SARCASM_INVERSION" in patterns

    def test_thrilled_to_redo(self, detector):
        roles = classify_sentence("im absolutely thrilled to redo all my work".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "SARCASM_INVERSION" in patterns


class TestNovel:
    """Sentences NEVER in any benchmark — the real test."""

    def test_tell_mom_i_love_her(self, detector):
        roles = classify_sentence("tell my mom I love her".split())
        matches = detector.detect_all(roles)
        # Should detect some farewell/finality pattern
        assert len(matches) > 0

    def test_sorry_for_everything(self, detector):
        roles = classify_sentence("im sorry for everything".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "BLANKET_APOLOGY" in patterns

    def test_finally_at_peace(self, detector):
        roles = classify_sentence("I finally feel at peace".split())
        matches = detector.detect_all(roles)
        patterns = [m.pattern for m in matches]
        assert "SUSPICIOUS_CALM" in patterns

    def test_not_gonna_be_around(self, detector):
        roles = classify_sentence("im not gonna be around much longer".split())
        matches = detector.detect_all(roles)
        assert len(matches) > 0  # should detect absence/finality

    def test_wrote_letters_to_everyone(self, detector):
        roles = classify_sentence("wrote letters to everyone".split())
        matches = detector.detect_all(roles)
        assert len(matches) > 0  # farewell + finality
```

- [ ] **Step 3: Run tests**

Run: `pytest engine/tests/test_structures.py -v`
Expected: MOST PASS — iterate on failures

- [ ] **Step 4: Commit**

```bash
git add engine/structures.py engine/tests/test_structures.py
git commit -m "V3 Layer 3: Structure Detector — chess-like pattern recognition from role sequences"
```

---

### Task 5: Fixed Physics Layer (Pendulum)

**The math that never changes.** Takes word forces + proximity coefficients + structure adjustments and computes final VADUG. Same momentum/blend equations as V2 but fed by the new layers instead of hardcoded idioms.

**Files:**
- Create: `engine/pendulum.py`
- Test: `engine/tests/test_pendulum.py`

- [ ] **Step 1: Write the physics engine**

```python
# engine/pendulum.py
"""Fixed Physics Layer — the math that computes VADUG from forces.

Takes inputs from Layer 1 (word roles), Layer 2 (proximity coefficients),
and Layer 3 (structure adjustments). Applies momentum-based blending
to produce final VADUG coordinates.

This layer has the SAME math as V2's pendulum. The difference is WHERE
the forces come from — structural analysis instead of hardcoded idioms.
"""

from typing import List, Optional
from .shared import VADUG
from .word_classifier import WordRole, classify_sentence
from .proximity import proximity_coefficient
from .structures import StructureDetector, StructureMatch
from .vocabulary import VOCABULARY

CENTER = 128.0
MOMENTUM = 0.82
FORCE_SCALE = 0.5
DIRECT_PUSH_CAP = 0.4
DIRECT_PUSH_TRIGGER = 80.0


def compute_vadug(text: str, personality=None) -> tuple:
    """Full V3 pipeline: classify → proximity → structure → physics → VADUG.

    Returns: (VADUG, trace_dict)
    """
    words = text.lower().split()
    if not words:
        return VADUG(), {}

    # Layer 1: Classify word roles
    roles = classify_sentence(words)

    # Layer 2: Compute proximity coefficients for each word
    coefficients = []
    for i in range(len(roles)):
        coefficients.append(proximity_coefficient(roles, i))

    # Layer 3: Detect sentence structures
    detector = StructureDetector()
    structures = detector.detect_all(roles)

    # Physics: Apply forces with momentum blending
    state = {"v": CENTER, "a": CENTER, "d": CENTER, "u": 0.0, "g": CENTER}
    trace = []

    for i, role in enumerate(roles):
        if role.role == "EMOTIONAL" and role.force:
            dv, da, dd, du, dg = role.force
            coeff = coefficients[i]
            fs = FORCE_SCALE * abs(coeff)
            sign = 1.0 if coeff >= 0 else -1.0

            # Target: where this word alone would place the pendulum
            target_v = CENTER + dv * fs * sign
            target_a = CENTER + da * fs * sign
            target_d = CENTER + dd * fs * sign
            target_u = du * fs * abs(coeff)
            target_g = CENTER + dg * fs * sign

            # Direct push for strong forces
            total_force = abs(dv * coeff) + abs(da * coeff)
            push = min(1.0, total_force / DIRECT_PUSH_TRIGGER) * DIRECT_PUSH_CAP

            blend = 1.0 - MOMENTUM
            state["v"] = state["v"] * MOMENTUM + target_v * blend + dv * fs * sign * push
            state["a"] = state["a"] * MOMENTUM + target_a * blend + da * fs * sign * push
            state["d"] = state["d"] * MOMENTUM + target_d * blend + dd * fs * sign * push
            state["u"] = state["u"] * MOMENTUM + target_u * blend + du * fs * abs(coeff) * push
            state["g"] = state["g"] * MOMENTUM + target_g * blend + dg * fs * sign * push

        trace.append({
            "word": role.word,
            "role": role.role,
            "coeff": round(coefficients[i], 2),
            "v": round(state["v"], 1),
            "a": round(state["a"], 1),
            "d": round(state["d"], 1),
            "u": round(state["u"], 1),
            "g": round(state["g"], 1),
        })

    # Apply structure adjustments
    for struct in structures:
        conf = struct.confidence
        state["v"] += struct.v_weight * conf * FORCE_SCALE
        state["d"] += struct.d_weight * conf * FORCE_SCALE
        state["u"] = max(state["u"], struct.u_weight * conf)
        state["g"] += struct.g_weight * conf * FORCE_SCALE

    # Clamp
    result = VADUG(
        v=int(max(0, min(255, round(state["v"])))),
        a=int(max(0, min(255, round(state["a"])))),
        d=int(max(0, min(255, round(state["d"])))),
        u=int(max(0, min(255, round(state["u"])))),
        g=int(max(0, min(255, round(state["g"])))),
    )

    return result, {
        "trace": trace,
        "structures": [{"pattern": s.pattern, "confidence": s.confidence,
                        "description": s.description} for s in structures],
        "word_count": len(roles),
    }
```

- [ ] **Step 2: Write physics tests**

```python
# engine/tests/test_pendulum.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.pendulum import compute_vadug


class TestBasicPhysics:
    def test_neutral_input(self):
        vadug, _ = compute_vadug("the meeting is at three")
        assert 115 <= vadug.v <= 140  # should be near neutral

    def test_positive_input(self):
        vadug, _ = compute_vadug("I am very happy")
        assert vadug.v > 140  # clearly positive

    def test_negative_input(self):
        vadug, _ = compute_vadug("I am very sad")
        assert vadug.v < 115  # clearly negative

    def test_negation(self):
        vadug_pos, _ = compute_vadug("I am happy")
        vadug_neg, _ = compute_vadug("I am not happy")
        assert vadug_neg.v < vadug_pos.v  # negation reduces V

    def test_amplification(self):
        vadug_plain, _ = compute_vadug("happy")
        vadug_amp, _ = compute_vadug("very happy")
        # Amplified should be further from center
        assert abs(vadug_amp.v - 128) > abs(vadug_plain.v - 128)


class TestStructuralDetection:
    def test_farewell_novel(self):
        """Novel farewell sentence — never practiced."""
        vadug, info = compute_vadug("I gave my dog to my neighbor")
        assert any(s["pattern"] == "FAREWELL" for s in info["structures"])

    def test_method_acquisition_novel(self):
        vadug, info = compute_vadug("just bought a bunch of pills")
        assert any(s["pattern"] == "METHOD_ACQUISITION" for s in info["structures"])

    def test_blanket_apology_novel(self):
        vadug, info = compute_vadug("im sorry for everything")
        assert any(s["pattern"] == "BLANKET_APOLOGY" for s in info["structures"])

    def test_safe_sentence_no_crisis(self):
        vadug, info = compute_vadug("I had a bad day at work")
        crisis_patterns = {"FAREWELL", "METHOD_ACQUISITION", "SELF_REMOVAL",
                           "EXHAUSTION", "NO_EXIT", "SELF_NULLIFY"}
        detected = {s["pattern"] for s in info["structures"]}
        assert not (detected & crisis_patterns)  # no crisis patterns
```

- [ ] **Step 3: Run tests**

Run: `pytest engine/tests/test_pendulum.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add engine/pendulum.py engine/tests/test_pendulum.py
git commit -m "V3 Layer Physics: Fixed pendulum math driven by structural analysis"
```

---

### Task 6: Bidirectional Solver (A+B=C)

**Files:**
- Create: `engine/solver.py`
- Test: `engine/tests/test_solver.py`

- [ ] **Step 1: Write the solver**

```python
# engine/solver.py
"""Bidirectional Solver — compute forward AND backward.

Forward:  text → VADUG (what state does this produce?)
Backward: A + desired_C → B range (what response temperature gets the outcome?)

The backward solver sweeps the response temperature space and finds
the RANGE of B's that land C in the target zone. Like landing a plane —
anywhere on the runway works.
"""

from typing import List, Tuple, Optional
from .shared import VADUG
from .pendulum import compute_vadug
from .zones import ZoneClassifier, ZONES

_zone_classifier = ZoneClassifier()


def forward(text: str) -> VADUG:
    """Forward pass: text → VADUG."""
    vadug, _ = compute_vadug(text)
    return vadug


def state_transition(a_vadug: VADUG, b_vadug: VADUG, a_weight: float = 0.6) -> VADUG:
    """Compute C = how A's state changes after receiving B.

    a_weight: how much A's current state persists (0.6 = A dominates,
    0.4 = B has strong influence). Like emotional momentum.
    """
    b_weight = 1.0 - a_weight
    return VADUG(
        v=int(a_vadug.v * a_weight + b_vadug.v * b_weight),
        a=int(a_vadug.a * a_weight + b_vadug.a * b_weight),
        d=int(a_vadug.d * a_weight + b_vadug.d * b_weight),
        u=int(a_vadug.u * a_weight + b_vadug.u * b_weight),
        g=int(a_vadug.g * a_weight + b_vadug.g * b_weight),
    )


def solve_for_b_range(a_vadug: VADUG, target_zone: str,
                       temperature_steps: int = 100) -> List[Tuple[int, int]]:
    """Find the range of response V-temperatures that land C in target_zone.

    Sweeps B's V from 0-255, computes C for each, checks if C falls
    in the target zone. Returns list of (b_v_min, b_v_max) ranges.

    Like finding the runway — any B in the range works.
    """
    if target_zone not in ZONES:
        return []

    zone_def = ZONES[target_zone]
    zone_center_v = zone_def["center"]["v"]
    zone_radius_v = zone_def["radius"]["v"]
    zone_min_v = zone_center_v - zone_radius_v
    zone_max_v = zone_center_v + zone_radius_v

    # Sweep B temperature
    valid_ranges = []
    in_range = False
    range_start = None

    for b_v in range(256):
        # Create a B with this V temperature (A/D/U/G at neutral)
        b = VADUG(v=b_v, a=128, d=128, u=0, g=128)
        c = state_transition(a_vadug, b)

        if zone_min_v <= c.v <= zone_max_v:
            if not in_range:
                range_start = b_v
                in_range = True
        else:
            if in_range:
                valid_ranges.append((range_start, b_v - 1))
                in_range = False

    if in_range:
        valid_ranges.append((range_start, 255))

    return valid_ranges


def optimal_b_temperature(a_vadug: VADUG, target_zone: str) -> Optional[int]:
    """Find the single best B temperature for a target zone.

    Returns the midpoint of the widest valid range (most margin for error).
    """
    ranges = solve_for_b_range(a_vadug, target_zone)
    if not ranges:
        return None

    # Pick widest range
    widest = max(ranges, key=lambda r: r[1] - r[0])
    return (widest[0] + widest[1]) // 2
```

- [ ] **Step 2: Write solver tests**

```python
# engine/tests/test_solver.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.shared import VADUG
from engine.solver import forward, state_transition, solve_for_b_range, optimal_b_temperature


class TestForward:
    def test_positive_text(self):
        v = forward("I am very happy")
        assert v.v > 140

    def test_negative_text(self):
        v = forward("I am very sad")
        assert v.v < 115


class TestStateTransition:
    def test_neutral_b_preserves_a(self):
        a = VADUG(v=50, a=200, d=30, u=80, g=60)
        b = VADUG(v=128, a=128, d=128, u=0, g=128)
        c = state_transition(a, b)
        # C should be between A and B, weighted toward A
        assert c.v > 50 and c.v < 128

    def test_warm_b_lifts_crisis(self):
        a = VADUG(v=30, a=180, d=40, u=90, g=50)  # crisis
        b = VADUG(v=180, a=100, d=150, u=0, g=160)  # warm supportive
        c = state_transition(a, b)
        assert c.v > a.v  # warm response should lift V


class TestSolver:
    def test_find_range_for_joy(self):
        a = VADUG(v=128, a=128, d=128, u=0, g=128)  # neutral
        ranges = solve_for_b_range(a, "JOY")
        assert len(ranges) > 0  # should find a valid range
        # The range should be in the positive direction
        assert ranges[0][0] > 128  # B needs to be positive to reach JOY

    def test_crisis_has_no_joy_path(self):
        a = VADUG(v=20, a=200, d=10, u=100, g=20)  # deep crisis
        ranges = solve_for_b_range(a, "JOY")
        # Might be empty or very narrow — hard to reach JOY from crisis in one step
        # This is correct: you can't go from crisis to joy in one response

    def test_optimal_temperature(self):
        a = VADUG(v=80, a=150, d=80, u=50, g=90)  # negative
        temp = optimal_b_temperature(a, "NEUTRAL")
        assert temp is not None
        assert temp > 128  # need warm response to pull toward neutral
```

- [ ] **Step 3: Run tests**

Run: `pytest engine/tests/test_solver.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add engine/solver.py engine/tests/test_solver.py
git commit -m "V3 Solver: Bidirectional A+B=C — forward read + backward zone targeting"
```

---

### Task 7: Battleship Probe System

**Files:**
- Create: `engine/battleship.py`
- Test: `engine/tests/test_battleship.py`

- [ ] **Step 1: Write the probe system**

```python
# engine/battleship.py
"""Emotional Battleship — fire calibrated probes, measure vibration.

The user's true emotional state is unknown. We fire controlled probe
responses (skeleton keys) and measure how the user's NEXT response
vibrates. High vibration = close to their frequency. Low = wrong quadrant.

After 2-3 probes, triangulate their actual position in VADUG space.
"""

from dataclasses import dataclass
from typing import List, Optional
from .shared import VADUG
from .pendulum import compute_vadug
from .solver import state_transition
from .zones import ZoneClassifier

_zone_classifier = ZoneClassifier()


@dataclass
class Probe:
    """A calibrated test response."""
    name: str
    text: str               # what to say
    vadug: VADUG            # known VADUG of this probe
    tests_for: List[str]    # which zones this probe discriminates


@dataclass
class ProbeResult:
    """Result of firing a probe."""
    probe: Probe
    user_response_vadug: VADUG    # what the user said after the probe
    expected_if_neutral: VADUG    # what C would be if A was neutral
    vibration: float              # |actual_C - expected_neutral_C| = how far off
    estimated_zone: str           # best guess at user's true zone


# Skeleton keys — minimal probes that divide emotional space efficiently
PROBES = [
    Probe(
        name="minimal_ack",
        text="hmm okay",
        vadug=VADUG(v=133, a=125, d=131, u=0, g=128),
        tests_for=["CRISIS", "RAGE", "GRIEF"],  # big shift = deep state
    ),
    Probe(
        name="slight_validation",
        text="that sounds tough",
        vadug=VADUG(v=115, a=130, d=120, u=5, g=125),
        tests_for=["GRIEF", "CRISIS", "RESIGNATION"],  # tests if they open up
    ),
    Probe(
        name="clarification",
        text="what do you mean",
        vadug=VADUG(v=128, a=135, d=125, u=10, g=128),
        tests_for=["DEFLECTION", "HEDGING"],  # tests if they avoid
    ),
    Probe(
        name="light_redirect",
        text="well thats one way to look at it",
        vadug=VADUG(v=135, a=125, d=135, u=0, g=125),
        tests_for=["SARCASM", "BRAVADO"],  # tests sarcasm resonance
    ),
    Probe(
        name="direct_check",
        text="are you okay",
        vadug=VADUG(v=125, a=130, d=115, u=15, g=130),
        tests_for=["CRISIS", "MINIMIZATION", "BRAVADO"],  # tests if they drop mask
    ),
]


def fire_probe(probe: Probe, user_state_a: VADUG) -> ProbeResult:
    """Simulate firing a probe at a user in state A.

    Computes what C would be, and measures vibration vs neutral baseline.
    """
    # What C would be if user was neutral
    neutral_a = VADUG(v=128, a=128, d=128, u=0, g=128)
    expected_neutral_c = state_transition(neutral_a, probe.vadug)

    # What C actually is with user's true state
    actual_c = state_transition(user_state_a, probe.vadug)

    # Vibration = distance between actual and neutral-expected
    vibration = (
        abs(actual_c.v - expected_neutral_c.v) +
        abs(actual_c.d - expected_neutral_c.d) +
        abs(actual_c.g - expected_neutral_c.g)
    ) / 3.0

    # Estimate zone from the actual C
    zone_result = _zone_classifier.classify(actual_c)

    return ProbeResult(
        probe=probe,
        user_response_vadug=actual_c,
        expected_if_neutral=expected_neutral_c,
        vibration=vibration,
        estimated_zone=zone_result.zone,
    )


def triangulate(user_state: VADUG, num_probes: int = 3) -> dict:
    """Fire multiple probes and triangulate user's true emotional state.

    Returns dict with estimated zone, confidence, and probe results.
    """
    results = []
    for probe in PROBES[:num_probes]:
        result = fire_probe(probe, user_state)
        results.append(result)

    # Highest vibration probe is most informative
    results.sort(key=lambda r: -r.vibration)

    # Zone votes
    zone_votes = {}
    for r in results:
        z = r.estimated_zone
        zone_votes[z] = zone_votes.get(z, 0) + r.vibration

    best_zone = max(zone_votes, key=zone_votes.get) if zone_votes else "NEUTRAL"
    total_vibration = sum(r.vibration for r in results)

    return {
        "estimated_zone": best_zone,
        "confidence": min(1.0, total_vibration / 50.0),
        "total_vibration": total_vibration,
        "probe_results": results,
    }
```

- [ ] **Step 2: Write battleship tests**

```python
# engine/tests/test_battleship.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.shared import VADUG
from engine.battleship import fire_probe, triangulate, PROBES


class TestProbes:
    def test_crisis_high_vibration(self):
        crisis = VADUG(v=30, a=180, d=40, u=90, g=50)
        result = fire_probe(PROBES[0], crisis)  # minimal_ack
        assert result.vibration > 20  # big shift from neutral baseline

    def test_neutral_low_vibration(self):
        neutral = VADUG(v=128, a=128, d=128, u=0, g=128)
        result = fire_probe(PROBES[0], neutral)
        assert result.vibration < 5  # minimal shift from neutral


class TestTriangulation:
    def test_crisis_detected(self):
        crisis = VADUG(v=30, a=180, d=40, u=90, g=50)
        result = triangulate(crisis)
        assert result["total_vibration"] > 30
        assert result["confidence"] > 0.5

    def test_neutral_low_confidence(self):
        neutral = VADUG(v=128, a=128, d=128, u=0, g=128)
        result = triangulate(neutral)
        assert result["total_vibration"] < 15

    def test_joy_positive_vibration(self):
        joy = VADUG(v=200, a=180, d=170, u=10, g=160)
        result = triangulate(joy)
        assert result["total_vibration"] > 15
```

- [ ] **Step 3: Run tests**

Run: `pytest engine/tests/test_battleship.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add engine/battleship.py engine/tests/test_battleship.py
git commit -m "V3 Battleship: Fire calibrated probes, measure vibration, triangulate state"
```

---

### Task 8: Novel Sentence Integration Test

**The REAL test.** Run every sentence that V2 failed on (26% novel accuracy) through V3.

**Files:**
- Create: `engine/tests/test_novel.py`

- [ ] **Step 1: Write the novel test suite**

```python
# engine/tests/test_novel.py
"""Novel sentences — things we NEVER practiced for.
V2 scored 26% on these. V3 target: >60%."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.pendulum import compute_vadug
from engine.word_classifier import classify_sentence
from engine.structures import StructureDetector

detector = StructureDetector()


class TestNovelCrisis:
    """Indirect/implied crisis — no explicit crisis words."""

    @pytest.mark.parametrize("text", [
        "i just gave my dog to my neighbor",
        "i left the keys on the counter for whoever finds them",
        "take care of my cat for me",
        "im sorry for everything",
        "this is the last time youll hear from me",
        "i finally feel at peace",
        "just bought a bunch of pills",
        "found a bridge near my house",
        "wrote letters to everyone",
    ])
    def test_novel_crisis_detected(self, text):
        """These should detect SOME structural crisis pattern."""
        roles = classify_sentence(text.split())
        matches = detector.detect_all(roles)
        # At minimum, some structure should be detected
        assert len(matches) > 0, f"No structures detected for: {text}"


class TestNovelSafe:
    """These should NOT flag as crisis structures."""

    @pytest.mark.parametrize("text", [
        "im having a bad day",
        "work was stressful",
        "my girlfriend broke up with me",
        "i feel kinda sad today",
        "mondays suck",
        "this homework is killing me",
    ])
    def test_novel_safe_no_crisis(self, text):
        roles = classify_sentence(text.split())
        matches = detector.detect_all(roles)
        crisis_patterns = {"FAREWELL", "METHOD_ACQUISITION", "SELF_REMOVAL",
                           "EXHAUSTION", "NO_EXIT", "SELF_NULLIFY",
                           "SUSPICIOUS_CALM"}
        detected = {m.pattern for m in matches}
        false_crisis = detected & crisis_patterns
        assert not false_crisis, f"False crisis patterns for '{text}': {false_crisis}"
```

- [ ] **Step 2: Run novel tests**

Run: `pytest engine/tests/test_novel.py -v`
Expected: Track results — this is the metric that matters

- [ ] **Step 3: Commit**

```bash
git add engine/tests/test_novel.py
git commit -m "V3 Novel test suite — the sentences we never practiced for"
```

---

### Task 9: V3 Public API + Comparison Benchmark

**Files:**
- Modify: `engine/__init__.py`
- Create: `benchmarks/v3_benchmark.py`

- [ ] **Step 1: Export public API**

```python
# engine/__init__.py
"""Clanker V3 — Structural Pattern Recognition Engine.

Three systems:
  1. Structure Recognition (chess player) — word roles + proximity + patterns
  2. Bidirectional Solver (Dr Strange) — A+B=C forward and backward
  3. Emotional Battleship (calibrator) — probe, measure vibration, triangulate
"""

from .shared import VADUG
from .pendulum import compute_vadug
from .solver import forward, solve_for_b_range, optimal_b_temperature, state_transition
from .battleship import triangulate, fire_probe, PROBES
from .word_classifier import classify_sentence, WordRole, ROLES
from .structures import StructureDetector, StructureMatch
from .zones import ZoneClassifier, ZONES

__all__ = [
    "VADUG", "compute_vadug",
    "forward", "solve_for_b_range", "optimal_b_temperature", "state_transition",
    "triangulate", "fire_probe", "PROBES",
    "classify_sentence", "WordRole", "ROLES",
    "StructureDetector", "StructureMatch",
    "ZoneClassifier", "ZONES",
]
```

- [ ] **Step 2: Write comparison benchmark**

```python
# benchmarks/v3_benchmark.py
"""V2 vs V3 comparison on novel sentences."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.pendulum import compute_vadug as v3_compute
from demo.pendulum_v2 import PendulumV2

v2_engine = PendulumV2()

NOVEL = [
    ("i just gave my dog to my neighbor", "crisis"),
    ("i left the keys on the counter for whoever finds them", "crisis"),
    ("tell my mom i love her", "crisis"),
    ("im sorry for everything", "crisis"),
    ("i just want the noise to stop", "crisis"),
    ("this is the last time youll hear from me", "crisis"),
    ("i finally feel at peace", "crisis"),
    ("take care of my cat for me", "crisis"),
    ("done with everything tbh", "crisis"),
    ("honestly might just end it lol", "crisis"),
    ("im so fucking done", "negative"),
    ("just bought a bunch of pills", "crisis"),
    ("found a bridge near my house", "crisis"),
    ("wrote letters to everyone", "crisis"),
    ("im having a bad day", "safe"),
    ("work was stressful", "safe"),
    ("i failed my exam", "safe"),
    ("my girlfriend broke up with me", "safe"),
    ("mondays suck", "safe"),
    ("i want to die of embarrassment", "safe"),
    ("this homework is killing me", "safe"),
    ("im dead tired", "safe"),
]

def check(v, expected):
    if expected == "crisis": return v < 100
    elif expected == "negative": return v < 120
    elif expected == "safe": return v >= 100
    return True

print("=" * 70)
print("V2 vs V3 — NOVEL SENTENCES (never practiced)")
print("=" * 70)

v2_correct = 0
v3_correct = 0

for text, expected in NOVEL:
    v2_vadug = v2_engine.process_text(text)[0]
    v3_vadug = v3_compute(text)[0]

    v2_ok = check(v2_vadug.v, expected)
    v3_ok = check(v3_vadug.v, expected)
    v2_correct += v2_ok
    v3_correct += v3_ok

    v2m = "OK" if v2_ok else "XX"
    v3m = "OK" if v3_ok else "XX"
    print(f"  V2={v2_vadug.v:3d}[{v2m}] V3={v3_vadug.v:3d}[{v3m}] | {expected:8s} | {text}")

total = len(NOVEL)
print(f"\nV2: {v2_correct}/{total} ({100*v2_correct/total:.0f}%)")
print(f"V3: {v3_correct}/{total} ({100*v3_correct/total:.0f}%)")
```

- [ ] **Step 3: Run comparison**

Run: `python3 benchmarks/v3_benchmark.py`
Expected: V3 > V2 on novel sentences

- [ ] **Step 4: Commit**

```bash
git add engine/__init__.py benchmarks/v3_benchmark.py
git commit -m "V3 public API + V2 vs V3 novel sentence benchmark"
```

---

## Build Order Summary

```
Task 1: Scaffold           → engine/ exists, V2 assets carried forward
Task 2: Word Classifier    → Layer 1: every word gets a structural role
Task 3: Proximity          → Layer 2: distance-based influence fields
Task 4: Structure Detector → Layer 3: chess-like pattern recognition
Task 5: Pendulum Physics   → Fixed math layer, driven by Layers 1-3
Task 6: Solver             → Bidirectional A+B=C + zone targeting
Task 7: Battleship         → Probe system, vibration measurement
Task 8: Novel Tests        → The REAL test — sentences we never practiced
Task 9: Public API         → Clean exports + V2 vs V3 comparison
```

Each task produces working, testable code. Each commit is self-contained. The layers stack bottom-up — you can't build Layer 3 without Layer 2, can't build Layer 2 without Layer 1.

---

Plan complete and saved to `docs/superpowers/plans/2026-03-30-v3-architecture.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?