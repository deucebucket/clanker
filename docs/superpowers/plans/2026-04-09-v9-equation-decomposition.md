# V9 Equation Decomposition Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace V8's sequential force accumulation with equation decomposition — find the event nucleus first, assign structural roles, compose the result. Order-independent, O(n), deterministic.

**Architecture:** Every sentence is an equation: `V = f(SUBJECT, EVENT_NUCLEUS, CONTEXT[], OPERATORS[])`. The decomposer finds the highest-gravity atom (event nucleus), assigns roles relative to it, and the composer resolves charges based on structural position (EVENT=1.0, CONTEXT=0.3, FORMULAIC=0). Root lemmatization collapses 2,315 vocabulary words to ~500 emotional roots. Compounds are pre-mapped molecular roots.

**Tech Stack:** Python 3, pytest, no external dependencies. All data structures are dataclasses. All operations O(n).

**Council consensus (Gemini/Grok/Claude/GPT):**
- Event nucleus = highest gravity atom (max operation, not a parse)
- Root lemmatization before decomposition
- Compounds as pre-mapped molecular roots
- Shadow pipeline migration: roots first, decomposition second
- Don't blend bidirectional — pick the stronger signal
- Context atoms at reduced weight vs event nucleus at full weight

**V8 baseline numbers to beat:**
- Conversational: 98.5% (275 stress test sentences)
- Blind holdout: 61% (the problem V9 solves)
- Crisis recall: 80.4%
- Crisis false positive: 0%

**Clean break rules:**
- engine_v9/ is independent — no imports from engine/ (V8)
- Copy what we need (VADUG type, phase system, compound bonds)
- New test suite, new benchmarks
- Verified sentences carry forward as ground truth
- V8 benchmark numbers carry forward as baseline

---

### Task 1: Project Skeleton + Shared Types

**Files:**
- Create: `engine_v9/__init__.py`
- Create: `engine_v9/shared.py`
- Create: `engine_v9/phase.py`
- Create: `engine_v9/tests/__init__.py`
- Create: `engine_v9/tests/test_shared.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p engine_v9/tests
```

- [ ] **Step 2: Write test for VADUG type**

```python
# engine_v9/tests/test_shared.py
from engine_v9.shared import VADUG, PersonalityVector

def test_vadug_defaults_to_neutral():
    v = VADUG()
    assert v.v == 128
    assert v.a == 128
    assert v.d == 128
    assert v.u == 0
    assert v.g == 128
    assert v.w == 128
    assert v.i == 128

def test_vadug_clamps():
    v = VADUG(v=300, a=-10)
    assert v.v == 255
    assert v.a == 0

def test_vadug_to_bytes():
    v = VADUG(v=100, a=200, d=50, u=30, g=180, w=90, i=160)
    b = v.to_bytes()
    assert len(b) == 7
    assert b[0] == 100

def test_personality_sensitivity_range():
    p = PersonalityVector()
    assert 0.5 <= p.emotional_sensitivity <= 2.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest engine_v9/tests/test_shared.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 4: Create shared.py — copy VADUG + PersonalityVector from V8**

Copy `engine/shared.py` → `engine_v9/shared.py` verbatim. These are data types, not engine logic.

- [ ] **Step 5: Create phase.py — copy phase system from V8**

Copy `engine/phase.py` → `engine_v9/phase.py` verbatim. SOLID/LIQUID/GAS is physics, not engine-specific.

- [ ] **Step 6: Create __init__.py files**

```python
# engine_v9/__init__.py
from engine_v9.shared import VADUG
```

```python
# engine_v9/tests/__init__.py
```

- [ ] **Step 7: Run tests**

Run: `python3 -m pytest engine_v9/tests/test_shared.py -v`
Expected: 4 PASSED

- [ ] **Step 8: Commit**

```bash
git add engine_v9/
git commit -m "V9: project skeleton with shared types and phase system"
```

---

### Task 2: Root Definitions — Categories + Base Charges

**Files:**
- Create: `engine_v9/roots.py`
- Create: `engine_v9/tests/test_roots.py`

Each root is a named emotional atom with a 7D charge vector (dV, dA, dD, dU, dG, dW, dI). Multiple English words map to the same root. The root's charge is the "average meaning" of its word cluster.

- [ ] **Step 1: Write tests for root lookup**

```python
# engine_v9/tests/test_roots.py
from engine_v9.roots import Root, ROOTS, RootCategory

def test_root_has_charge_vector():
    r = ROOTS["HAPPY"]
    assert len(r.charge) == 7
    assert r.charge[0] > 0  # positive valence

def test_root_has_category():
    r = ROOTS["HAPPY"]
    assert r.category == RootCategory.POSITIVE_STATE

def test_negative_root():
    r = ROOTS["SAD"]
    assert r.charge[0] < 0  # negative valence

def test_root_has_phase():
    r = ROOTS["MURDER"]
    assert r.phase == "SOLID"

def test_formulaic_root_zero_charge():
    r = ROOTS["GREETING"]
    assert r.charge[0] == 0
    assert r.charge[1] == 0

def test_compound_event_root():
    r = ROOTS["EMPLOYMENT_LOSS"]
    assert r.charge[0] < -40  # strong negative
    assert r.category == RootCategory.COMPOUND_EVENT

def test_all_roots_have_7d_charge():
    for name, root in ROOTS.items():
        assert len(root.charge) == 7, f"Root {name} has {len(root.charge)}D charge"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest engine_v9/tests/test_roots.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Create roots.py with categories, Root dataclass, and starter set**

```python
# engine_v9/roots.py
"""Emotional root system — ~500 roots that 2,315+ words collapse into.

Each root is a named emotional atom with a 7D charge vector.
Charge tuple: (dV, dA, dD, dU, dG, dW, dI)

Categories determine physics rules (structure-dependent charge scaling).
Roots determine specific charge values.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple


class RootCategory(Enum):
    # Emotional states
    POSITIVE_STATE = auto()     # happy, joy, excited, grateful
    NEGATIVE_STATE = auto()     # sad, depressed, miserable, lonely
    POSITIVE_QUALITY = auto()   # good, great, wonderful, beautiful
    NEGATIVE_QUALITY = auto()   # bad, terrible, awful, ugly
    POSITIVE_EVENT = auto()     # achievement, success, promotion
    NEGATIVE_EVENT = auto()     # loss, failure, death, accident
    SOCIAL_EVAL_POS = auto()    # brave, generous, kind, loyal
    SOCIAL_EVAL_NEG = auto()    # selfish, cruel, coward, traitor
    EXPECTATION_VIOLATION = auto()  # surprised, shocked, disbelief

    # Actions (mostly neutral charge, structural role matters)
    MOTION = auto()
    POSSESSION = auto()
    TRANSFER = auto()
    PERCEPTION = auto()
    COMMUNICATION = auto()
    COGNITION = auto()

    # Entities
    SELF_REF = auto()
    OTHER_REF = auto()
    RELATION_REF = auto()
    PERSON = auto()
    OBJECT = auto()

    # Operators (modify other roots, no standalone charge)
    INTENSIFIER = auto()
    NEGATOR = auto()
    CONNECTOR = auto()
    HEDGE = auto()
    COMPRESSOR = auto()
    TEMPORAL = auto()

    # Special
    FORMULAIC = auto()          # "good morning", "thank you" — always 0
    COMPOUND_EVENT = auto()     # "laid_off", "cancer_free" — pre-mapped
    FILLER = auto()             # "um", "uh", "like" — zero charge


@dataclass(frozen=True)
class Root:
    name: str
    category: RootCategory
    charge: Tuple[int, int, int, int, int, int, int]  # (dV, dA, dD, dU, dG, dW, dI)
    phase: str = "GAS"  # SOLID, LIQUID, GAS


# ── Root definitions ─────────────────────────────────────────────
# Charge tuple: (dV, dA, dD, dU, dG, dW, dI)
# These are DELTAS from neutral (128), not absolute values.
# Positive dV = positive emotion, negative dV = negative emotion.

ROOTS = {
    # ── Positive states ──
    "HAPPY":        Root("HAPPY",        RootCategory.POSITIVE_STATE,  (+35, -5, +15, 0, +25, +15, 0)),
    "EXCITED":      Root("EXCITED",      RootCategory.POSITIVE_STATE,  (+40, +60, +20, +15, +30, +10, 0)),
    "GRATEFUL":     Root("GRATEFUL",     RootCategory.POSITIVE_STATE,  (+45, -10, -5, 0, +20, +20, +30)),
    "RELIEVED":     Root("RELIEVED",     RootCategory.POSITIVE_STATE,  (+35, -30, +10, -20, +30, +15, 0)),
    "PROUD":        Root("PROUD",        RootCategory.POSITIVE_STATE,  (+40, +10, +30, 0, +25, +35, 0)),
    "CONTENT":      Root("CONTENT",      RootCategory.POSITIVE_STATE,  (+25, -20, +10, 0, +15, +15, 0)),
    "AMUSED":       Root("AMUSED",       RootCategory.POSITIVE_STATE,  (+25, +20, +15, 0, +20, +5, 0)),
    "HOPEFUL":      Root("HOPEFUL",      RootCategory.POSITIVE_STATE,  (+30, +10, +5, +10, +20, +15, +20)),
    "LOVED":        Root("LOVED",        RootCategory.POSITIVE_STATE,  (+50, +10, +5, 0, +25, +40, +30), "SOLID"),

    # ── Negative states ──
    "SAD":          Root("SAD",          RootCategory.NEGATIVE_STATE,  (-35, -10, -20, 0, -25, -15, 0)),
    "ANGRY":        Root("ANGRY",        RootCategory.NEGATIVE_STATE,  (-80, +100, +80, +60, +50, 0, +50)),
    "AFRAID":       Root("AFRAID",       RootCategory.NEGATIVE_STATE,  (-60, +50, -80, +70, +20, -20, -30)),
    "DISGUSTED":    Root("DISGUSTED",    RootCategory.NEGATIVE_STATE,  (-70, +50, +30, +20, +10, 0, -20)),
    "ASHAMED":      Root("ASHAMED",      RootCategory.NEGATIVE_STATE,  (-80, +60, -100, +30, -50, -60, -40)),
    "LONELY":       Root("LONELY",       RootCategory.NEGATIVE_STATE,  (-50, -20, -40, 0, -30, -30, -30)),
    "ANXIOUS":      Root("ANXIOUS",      RootCategory.NEGATIVE_STATE,  (-50, +80, -100, +90, +20, -20, 0)),
    "GUILTY":       Root("GUILTY",       RootCategory.NEGATIVE_STATE,  (-70, +40, -80, +30, -40, -50, 0)),
    "JEALOUS":      Root("JEALOUS",      RootCategory.NEGATIVE_STATE,  (-60, +70, -30, +40, +20, -40, +30)),
    "FRUSTRATED":   Root("FRUSTRATED",   RootCategory.NEGATIVE_STATE,  (-50, +80, -20, +50, +30, -10, 0)),
    "EXHAUSTED":    Root("EXHAUSTED",    RootCategory.NEGATIVE_STATE,  (-30, -60, -40, 0, -40, -15, -20)),
    "DEVASTATED":   Root("DEVASTATED",   RootCategory.NEGATIVE_STATE,  (-100, +80, -100, +60, -70, -50, -30), "SOLID"),
    "DESPAIR":      Root("DESPAIR",      RootCategory.NEGATIVE_STATE,  (-110, +40, -120, +50, -80, -60, -40), "SOLID"),

    # ── Positive qualities ──
    "POS_QUALITY":  Root("POS_QUALITY",  RootCategory.POSITIVE_QUALITY, (+25, 0, +10, 0, +10, +5, 0)),
    "BEAUTIFUL":    Root("BEAUTIFUL",    RootCategory.POSITIVE_QUALITY, (+35, +10, +5, 0, +20, +5, 0)),
    "STRONG":       Root("STRONG",       RootCategory.POSITIVE_QUALITY, (+20, +15, +40, 0, +15, +20, 0)),

    # ── Negative qualities ──
    "NEG_QUALITY":  Root("NEG_QUALITY",  RootCategory.NEGATIVE_QUALITY, (-25, 0, -10, 0, -10, -5, 0)),
    "UGLY":         Root("UGLY",         RootCategory.NEGATIVE_QUALITY, (-30, +10, -10, 0, -15, -15, 0)),
    "WEAK":         Root("WEAK",         RootCategory.NEGATIVE_QUALITY, (-15, -10, -40, 0, -15, -25, 0)),

    # ── Positive events ──
    "ACHIEVEMENT":  Root("ACHIEVEMENT",  RootCategory.POSITIVE_EVENT,  (+50, +30, +40, 0, +30, +40, 0)),
    "HEALING":      Root("HEALING",      RootCategory.POSITIVE_EVENT,  (+40, -10, +15, -10, +25, +20, +20)),

    # ── Negative events ──
    "LOSS":         Root("LOSS",         RootCategory.NEGATIVE_EVENT,  (-60, +30, -50, +20, -40, -30, -20)),
    "HARM":         Root("HARM",         RootCategory.NEGATIVE_EVENT,  (-80, +80, -60, +70, -20, -40, 0), "SOLID"),
    "CRISIS":       Root("CRISIS",       RootCategory.NEGATIVE_EVENT,  (-100, +100, -100, +100, -60, -50, -40), "SOLID"),
    "BETRAYAL":     Root("BETRAYAL",     RootCategory.NEGATIVE_EVENT,  (-90, +80, -70, +50, -30, -60, -40), "SOLID"),

    # ── Social evaluation ──
    "SOC_EVAL_POS": Root("SOC_EVAL_POS", RootCategory.SOCIAL_EVAL_POS, (+20, 0, +15, 0, +10, +15, +15)),
    "SOC_EVAL_NEG": Root("SOC_EVAL_NEG", RootCategory.SOCIAL_EVAL_NEG, (-20, +20, -15, +10, -10, -20, -15)),

    # ── Expectation violation ──
    "SURPRISE":     Root("SURPRISE",     RootCategory.EXPECTATION_VIOLATION, (0, +60, -30, +30, +10, 0, 0)),

    # ── Actions (near-zero charge, structural role matters) ──
    "MOTION":       Root("MOTION",       RootCategory.MOTION,       (0, +5, 0, 0, 0, 0, 0)),
    "POSSESS":      Root("POSSESS",      RootCategory.POSSESSION,   (0, 0, +5, 0, 0, 0, 0)),
    "TRANSFER":     Root("TRANSFER",     RootCategory.TRANSFER,     (0, 0, -5, 0, 0, 0, 0)),
    "PERCEIVE":     Root("PERCEIVE",     RootCategory.PERCEPTION,   (0, +5, 0, 0, 0, 0, 0)),
    "COMMUNICATE":  Root("COMMUNICATE",  RootCategory.COMMUNICATION,(0, +5, 0, 0, 0, 0, 0)),
    "THINK":        Root("THINK",        RootCategory.COGNITION,    (0, 0, +5, 0, 0, 0, 0)),

    # ── Entities (zero charge, structural position matters) ──
    "SELF":         Root("SELF",         RootCategory.SELF_REF,     (0, 0, 0, 0, 0, 0, 0)),
    "OTHER":        Root("OTHER",        RootCategory.OTHER_REF,    (0, 0, 0, 0, 0, 0, 0)),
    "RELATION":     Root("RELATION",     RootCategory.RELATION_REF, (0, 0, 0, 0, 0, 0, 0)),
    "PERSON_GENERIC": Root("PERSON_GENERIC", RootCategory.PERSON,   (0, 0, 0, 0, 0, 0, 0)),
    "OBJECT_GENERIC": Root("OBJECT_GENERIC", RootCategory.OBJECT,   (0, 0, 0, 0, 0, 0, 0)),

    # ── Operators (structural modifiers, not standalone) ──
    "INTENSIFY":    Root("INTENSIFY",    RootCategory.INTENSIFIER,  (0, +15, 0, +5, 0, 0, 0)),
    "NEGATE":       Root("NEGATE",       RootCategory.NEGATOR,      (0, 0, 0, 0, 0, 0, 0)),
    "CONNECT":      Root("CONNECT",      RootCategory.CONNECTOR,    (0, 0, 0, 0, 0, 0, 0)),
    "HEDGE_OP":     Root("HEDGE_OP",     RootCategory.HEDGE,        (0, -5, 0, 0, 0, 0, 0)),
    "COMPRESS":     Root("COMPRESS",     RootCategory.COMPRESSOR,   (0, -10, 0, 0, 0, 0, 0)),
    "TIME":         Root("TIME",         RootCategory.TEMPORAL,     (0, 0, 0, 0, 0, 0, 0)),

    # ── Formulaic (always zero, never contributes) ──
    "GREETING":     Root("GREETING",     RootCategory.FORMULAIC,    (0, 0, 0, 0, 0, 0, 0)),
    "THANKS":       Root("THANKS",       RootCategory.FORMULAIC,    (0, 0, 0, 0, 0, 0, 0)),
    "APOLOGY":      Root("APOLOGY",      RootCategory.FORMULAIC,    (0, 0, 0, 0, 0, 0, 0)),
    "FILLER_WORD":  Root("FILLER_WORD",  RootCategory.FILLER,       (0, 0, 0, 0, 0, 0, 0)),

    # ── Compound events (pre-mapped molecular roots) ──
    "EMPLOYMENT_LOSS":   Root("EMPLOYMENT_LOSS",   RootCategory.COMPOUND_EVENT, (-60, +40, -60, +50, -40, -30, -20), "SOLID"),
    "MEDICAL_RELIEF":    Root("MEDICAL_RELIEF",    RootCategory.COMPOUND_EVENT, (+55, -10, +30, -20, +40, +30, +20), "SOLID"),
    "RELATIONSHIP_END":  Root("RELATIONSHIP_END",  RootCategory.COMPOUND_EVENT, (-55, +50, -30, +30, -30, -40, -30), "SOLID"),
    "DEATH_EUPHEMISM":   Root("DEATH_EUPHEMISM",   RootCategory.COMPOUND_EVENT, (-65, +20, -50, +20, -50, -20, -20), "SOLID"),
    "BREAKDOWN":         Root("BREAKDOWN",         RootCategory.COMPOUND_EVENT, (-45, +40, -40, +30, -30, -20, -10)),
    "RECOVERY":          Root("RECOVERY",          RootCategory.COMPOUND_EVENT, (+40, +10, +20, -10, +25, +20, +15)),
    "BURNOUT":           Root("BURNOUT",           RootCategory.COMPOUND_EVENT, (-40, -30, -50, +20, -35, -25, -20)),
}
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest engine_v9/tests/test_roots.py -v`
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add engine_v9/roots.py engine_v9/tests/test_roots.py
git commit -m "V9: root definitions — categories, charges, starter set of ~55 roots"
```

---

### Task 3: Word-to-Root Mapper

**Files:**
- Create: `engine_v9/root_map.py`
- Create: `engine_v9/tests/test_root_map.py`

Three-tier lookup: static hash → morphological strip → fallback to OBJECT_GENERIC (GAS, zero charge). The static hash is built from V8's forces_curated.py words, each manually assigned to a root.

- [ ] **Step 1: Write tests for word→root mapping**

```python
# engine_v9/tests/test_root_map.py
from engine_v9.root_map import map_to_root
from engine_v9.roots import RootCategory

def test_direct_lookup():
    root = map_to_root("happy")
    assert root.name == "HAPPY"

def test_case_insensitive():
    root = map_to_root("HAPPY")
    assert root.name == "HAPPY"

def test_morphological_strip():
    root = map_to_root("happiness")
    assert root.category == RootCategory.POSITIVE_STATE

def test_ly_suffix():
    root = map_to_root("sadly")
    assert root.category == RootCategory.NEGATIVE_STATE

def test_unknown_word_returns_generic():
    root = map_to_root("splendiferous")
    assert root.name == "OBJECT_GENERIC"
    assert root.phase == "GAS"
    assert root.charge[0] == 0  # zero charge

def test_self_ref():
    root = map_to_root("i")
    assert root.category == RootCategory.SELF_REF

def test_negator():
    root = map_to_root("not")
    assert root.category == RootCategory.NEGATOR

def test_intensifier():
    root = map_to_root("very")
    assert root.category == RootCategory.INTENSIFIER

def test_compound_event():
    root = map_to_root("laidoff")
    assert root.category == RootCategory.COMPOUND_EVENT
    assert root.charge[0] < -40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest engine_v9/tests/test_root_map.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Create root_map.py with three-tier lookup**

```python
# engine_v9/root_map.py
"""Word → Root mapping.

Three-tier lookup:
  1. Static hash for known words, compounds, slang, irregulars (O(1))
  2. Morphological strip (-ly, -ing, -ed, -ness, -tion, -ment, -ful, -less) then re-lookup
  3. Fallback: OBJECT_GENERIC (GAS, zero charge)
"""

import re
from engine_v9.roots import Root, ROOTS, RootCategory


# ── Tier 1: Static word → root name mapping ──────────────────────
# Each English word maps to exactly one root name (key in ROOTS dict).
# This is the primary lookup. Populated from V8 forces_curated.py clusters.

WORD_TO_ROOT = {
    # ── Self/Other/Relation references ──
    "i": "SELF", "me": "SELF", "my": "SELF", "myself": "SELF",
    "im": "SELF", "ive": "SELF", "id": "SELF", "ill": "SELF",
    "mine": "SELF",

    "you": "OTHER", "your": "OTHER", "yours": "OTHER", "yourself": "OTHER",
    "he": "OTHER", "him": "OTHER", "his": "OTHER", "she": "OTHER",
    "her": "OTHER", "hers": "OTHER", "they": "OTHER", "them": "OTHER",
    "their": "OTHER", "we": "OTHER", "us": "OTHER", "our": "OTHER",
    "it": "OTHER", "its": "OTHER",
    "someone": "PERSON_GENERIC", "somebody": "PERSON_GENERIC",
    "everyone": "PERSON_GENERIC", "everybody": "PERSON_GENERIC",
    "anyone": "PERSON_GENERIC", "anybody": "PERSON_GENERIC",
    "people": "PERSON_GENERIC",

    "mom": "RELATION", "mother": "RELATION", "dad": "RELATION",
    "father": "RELATION", "brother": "RELATION", "sister": "RELATION",
    "friend": "RELATION", "friends": "RELATION", "family": "RELATION",
    "husband": "RELATION", "wife": "RELATION", "partner": "RELATION",
    "boyfriend": "RELATION", "girlfriend": "RELATION",
    "son": "RELATION", "daughter": "RELATION", "child": "RELATION",
    "children": "RELATION", "kids": "RELATION", "kid": "RELATION",
    "boss": "RELATION", "teacher": "RELATION", "dog": "RELATION",
    "cat": "RELATION", "baby": "RELATION", "ex": "RELATION",

    # ── Positive states ──
    "happy": "HAPPY", "glad": "HAPPY", "pleased": "HAPPY",
    "excited": "EXCITED", "thrilled": "EXCITED", "pumped": "EXCITED", "stoked": "EXCITED",
    "grateful": "GRATEFUL", "thankful": "GRATEFUL", "appreciative": "GRATEFUL",
    "relieved": "RELIEVED",
    "proud": "PROUD",
    "content": "CONTENT", "satisfied": "CONTENT", "comfortable": "CONTENT",
    "amused": "AMUSED", "entertained": "AMUSED",
    "hopeful": "HOPEFUL", "optimistic": "HOPEFUL",
    "love": "LOVED", "loved": "LOVED", "adore": "LOVED", "cherish": "LOVED",
    "joy": "HAPPY", "joyful": "HAPPY", "joyous": "HAPPY",
    "ecstatic": "EXCITED", "elated": "EXCITED", "overjoyed": "EXCITED",

    # ── Negative states ──
    "sad": "SAD", "unhappy": "SAD", "miserable": "SAD", "down": "SAD",
    "angry": "ANGRY", "furious": "ANGRY", "livid": "ANGRY", "enraged": "ANGRY",
    "mad": "ANGRY", "pissed": "ANGRY", "irate": "ANGRY",
    "afraid": "AFRAID", "scared": "AFRAID", "terrified": "AFRAID", "fearful": "AFRAID",
    "disgusted": "DISGUSTED", "repulsed": "DISGUSTED", "revolted": "DISGUSTED",
    "ashamed": "ASHAMED", "humiliated": "ASHAMED", "embarrassed": "ASHAMED",
    "lonely": "LONELY", "isolated": "LONELY", "alone": "LONELY",
    "anxious": "ANXIOUS", "nervous": "ANXIOUS", "worried": "ANXIOUS",
    "guilty": "GUILTY",
    "jealous": "JEALOUS", "envious": "JEALOUS",
    "frustrated": "FRUSTRATED", "annoyed": "FRUSTRATED", "irritated": "FRUSTRATED",
    "exhausted": "EXHAUSTED", "drained": "EXHAUSTED", "burned": "EXHAUSTED",
    "tired": "EXHAUSTED", "fatigued": "EXHAUSTED",
    "devastated": "DEVASTATED", "shattered": "DEVASTATED", "crushed": "DEVASTATED",
    "desperate": "DESPAIR", "hopeless": "DESPAIR", "despair": "DESPAIR",
    "depressed": "SAD", "depression": "SAD", "grief": "SAD",
    "hate": "ANGRY", "hatred": "ANGRY", "loathe": "ANGRY",
    "anxiety": "ANXIOUS", "panic": "ANXIOUS", "dread": "AFRAID",

    # ── Positive qualities ──
    "good": "POS_QUALITY", "great": "POS_QUALITY", "nice": "POS_QUALITY",
    "wonderful": "POS_QUALITY", "amazing": "POS_QUALITY", "awesome": "POS_QUALITY",
    "excellent": "POS_QUALITY", "fantastic": "POS_QUALITY", "incredible": "POS_QUALITY",
    "fine": "POS_QUALITY", "perfect": "POS_QUALITY", "brilliant": "POS_QUALITY",
    "beautiful": "BEAUTIFUL", "gorgeous": "BEAUTIFUL", "stunning": "BEAUTIFUL",
    "pretty": "BEAUTIFUL", "lovely": "BEAUTIFUL",
    "strong": "STRONG", "powerful": "STRONG", "mighty": "STRONG",

    # ── Negative qualities ──
    "bad": "NEG_QUALITY", "terrible": "NEG_QUALITY", "awful": "NEG_QUALITY",
    "horrible": "NEG_QUALITY", "dreadful": "NEG_QUALITY", "atrocious": "NEG_QUALITY",
    "wrong": "NEG_QUALITY", "poor": "NEG_QUALITY",
    "ugly": "UGLY",
    "weak": "WEAK", "pathetic": "WEAK", "useless": "WEAK",

    # ── Events ──
    "success": "ACHIEVEMENT", "succeeded": "ACHIEVEMENT", "accomplished": "ACHIEVEMENT",
    "won": "ACHIEVEMENT", "winning": "ACHIEVEMENT", "promotion": "ACHIEVEMENT",
    "graduated": "ACHIEVEMENT", "achievement": "ACHIEVEMENT",
    "healed": "HEALING", "recovered": "HEALING", "cured": "HEALING",
    "lost": "LOSS", "losing": "LOSS", "loss": "LOSS", "missed": "LOSS",
    "died": "LOSS", "death": "LOSS", "dead": "LOSS",
    "hurt": "HARM", "injured": "HARM", "wounded": "HARM", "pain": "HARM",
    "murder": "HARM", "murdered": "HARM", "killed": "HARM",
    "abuse": "HARM", "abused": "HARM", "assault": "HARM",
    "suicide": "CRISIS", "suicidal": "CRISIS",
    "rape": "CRISIS", "raped": "CRISIS",
    "torture": "CRISIS", "tortured": "CRISIS",
    "betrayed": "BETRAYAL", "betrayal": "BETRAYAL", "cheated": "BETRAYAL",
    "stabbed": "BETRAYAL",

    # ── Social evaluation ──
    "brave": "SOC_EVAL_POS", "generous": "SOC_EVAL_POS", "kind": "SOC_EVAL_POS",
    "loyal": "SOC_EVAL_POS", "honest": "SOC_EVAL_POS", "caring": "SOC_EVAL_POS",
    "selfish": "SOC_EVAL_NEG", "cruel": "SOC_EVAL_NEG", "mean": "SOC_EVAL_NEG",
    "liar": "SOC_EVAL_NEG", "traitor": "SOC_EVAL_NEG", "coward": "SOC_EVAL_NEG",

    # ── Surprise ──
    "surprised": "SURPRISE", "shocked": "SURPRISE", "unexpected": "SURPRISE",
    "astonished": "SURPRISE", "stunned": "SURPRISE",

    # ── Actions ──
    "go": "MOTION", "going": "MOTION", "went": "MOTION", "gone": "MOTION",
    "run": "MOTION", "running": "MOTION", "ran": "MOTION",
    "walk": "MOTION", "walking": "MOTION", "walked": "MOTION",
    "come": "MOTION", "came": "MOTION", "coming": "MOTION",
    "move": "MOTION", "moved": "MOTION", "moving": "MOTION",
    "have": "POSSESS", "had": "POSSESS", "has": "POSSESS",
    "own": "POSSESS", "keep": "POSSESS", "kept": "POSSESS",
    "give": "TRANSFER", "gave": "TRANSFER", "giving": "TRANSFER",
    "take": "TRANSFER", "took": "TRANSFER", "taking": "TRANSFER",
    "get": "TRANSFER", "got": "TRANSFER", "getting": "TRANSFER",
    "send": "TRANSFER", "sent": "TRANSFER",
    "see": "PERCEIVE", "saw": "PERCEIVE", "seeing": "PERCEIVE",
    "hear": "PERCEIVE", "heard": "PERCEIVE", "feel": "PERCEIVE",
    "felt": "PERCEIVE", "notice": "PERCEIVE", "noticed": "PERCEIVE",
    "say": "COMMUNICATE", "said": "COMMUNICATE", "tell": "COMMUNICATE",
    "told": "COMMUNICATE", "speak": "COMMUNICATE", "spoke": "COMMUNICATE",
    "ask": "COMMUNICATE", "asked": "COMMUNICATE",
    "think": "THINK", "thought": "THINK", "know": "THINK",
    "knew": "THINK", "believe": "THINK", "understand": "THINK",
    "understood": "THINK", "remember": "THINK", "remembered": "THINK",

    # ── Operators ──
    "very": "INTENSIFY", "really": "INTENSIFY", "extremely": "INTENSIFY",
    "absolutely": "INTENSIFY", "totally": "INTENSIFY", "completely": "INTENSIFY",
    "incredibly": "INTENSIFY", "deeply": "INTENSIFY", "truly": "INTENSIFY",
    "super": "INTENSIFY", "so": "INTENSIFY", "hella": "INTENSIFY",
    "fucking": "INTENSIFY", "damn": "INTENSIFY", "too": "INTENSIFY",

    "not": "NEGATE", "no": "NEGATE", "never": "NEGATE",
    "nobody": "NEGATE", "nothing": "NEGATE", "nowhere": "NEGATE",
    "none": "NEGATE", "nor": "NEGATE", "neither": "NEGATE",
    "dont": "NEGATE", "doesnt": "NEGATE", "didnt": "NEGATE",
    "cant": "NEGATE", "wont": "NEGATE", "isnt": "NEGATE",
    "wasnt": "NEGATE", "arent": "NEGATE", "havent": "NEGATE",
    "hasnt": "NEGATE", "wouldnt": "NEGATE", "couldnt": "NEGATE",
    "shouldnt": "NEGATE",

    "and": "CONNECT", "but": "CONNECT", "or": "CONNECT",
    "if": "CONNECT", "because": "CONNECT", "since": "CONNECT",
    "although": "CONNECT", "though": "CONNECT", "yet": "CONNECT",
    "however": "CONNECT", "while": "CONNECT",

    "maybe": "HEDGE_OP", "perhaps": "HEDGE_OP", "somewhat": "HEDGE_OP",
    "kinda": "HEDGE_OP", "sorta": "HEDGE_OP", "probably": "HEDGE_OP",
    "might": "HEDGE_OP",

    "just": "TIME", "now": "TIME", "then": "TIME", "already": "TIME",
    "still": "TIME", "yet": "TIME", "finally": "TIME", "recently": "TIME",
    "today": "TIME", "yesterday": "TIME", "tomorrow": "TIME",

    # ── Formulaic ──
    "hello": "GREETING", "hi": "GREETING", "hey": "GREETING",
    "goodbye": "GREETING", "bye": "GREETING",
    "thanks": "THANKS", "thank": "THANKS",
    "sorry": "APOLOGY", "apologize": "APOLOGY",
    "please": "FILLER_WORD",
    "um": "FILLER_WORD", "uh": "FILLER_WORD", "like": "FILLER_WORD",
    "well": "FILLER_WORD", "ok": "FILLER_WORD", "okay": "FILLER_WORD",

    # ── Compound events (pre-bonded molecules) ──
    "laidoff": "EMPLOYMENT_LOSS",
    "firedoff": "EMPLOYMENT_LOSS",
    "cancerfree": "MEDICAL_RELIEF",
    "debtfree": "MEDICAL_RELIEF",  # same recovery shape
    "painfree": "MEDICAL_RELIEF",
    "brokeup": "RELATIONSHIP_END",
    "passedaway": "DEATH_EUPHEMISM",
    "brokedown": "BREAKDOWN",
    "burnedout": "BURNOUT",
    "pulledthrough": "RECOVERY",
    "workedout": "RECOVERY",
    "paidoff": "RECOVERY",
    "turnedaround": "RECOVERY",
    "pulledoff": "ACHIEVEMENT",
}

# ── Tier 2: Morphological suffix stripping ───────────────────────

_SUFFIXES = [
    ("fulness", 7), ("lessly", 6), ("ingness", 7),
    ("ness", 4), ("ment", 4), ("tion", 4), ("sion", 4),
    ("ling", 3), ("ful", 3), ("less", 4), ("able", 4), ("ible", 4),
    ("ous", 3), ("ive", 3), ("ity", 3),
    ("ly", 2), ("ing", 3), ("ed", 2), ("er", 2), ("est", 3),
    ("es", 2), ("s", 1),
]


def _strip_suffix(word: str) -> str:
    """Strip common English suffixes to find stem."""
    for suffix, min_stem in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[:-len(suffix)]
    return word


def map_to_root(word: str) -> Root:
    """Map an English word to its emotional root.

    Three-tier lookup:
      1. Direct lookup in WORD_TO_ROOT (covers known words, compounds, slang)
      2. Strip morphological suffix, then re-lookup
      3. Fallback: OBJECT_GENERIC (GAS phase, zero charge)
    """
    w = word.lower().strip(".,!?;:'\"()[]{}…–—")

    # Tier 1: direct lookup
    if w in WORD_TO_ROOT:
        return ROOTS[WORD_TO_ROOT[w]]

    # Tier 2: morphological strip then lookup
    stem = _strip_suffix(w)
    if stem != w and stem in WORD_TO_ROOT:
        return ROOTS[WORD_TO_ROOT[stem]]

    # Tier 3: fallback — unknown word, zero charge, GAS phase
    return ROOTS["OBJECT_GENERIC"]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest engine_v9/tests/test_root_map.py -v`
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add engine_v9/root_map.py engine_v9/tests/test_root_map.py
git commit -m "V9: word-to-root mapper — 3-tier lookup with morphological fallback"
```

---

### Task 4: Tokenizer — Compound Resolution + Cleaning

**Files:**
- Create: `engine_v9/tokenizer.py`
- Create: `engine_v9/tests/test_tokenizer.py`

Reuse V8's compound bond tables and tokenization logic, adapted for V9.

- [ ] **Step 1: Write tests**

```python
# engine_v9/tests/test_tokenizer.py
from engine_v9.tokenizer import tokenize

def test_basic_split():
    tokens = tokenize("I am happy")
    assert tokens == ["i", "am", "happy"]

def test_compound_bond():
    tokens = tokenize("I got laid off from work")
    assert "laidoff" in tokens
    assert "laid" not in tokens

def test_trigram_compound():
    tokens = tokenize("I got laid off yesterday")
    assert "laidoff" in tokens

def test_passed_away():
    tokens = tokenize("she passed away last night")
    assert "passedaway" in tokens

def test_punctuation_stripped():
    tokens = tokenize("I'm happy!")
    assert "happy" in tokens

def test_no_one_becomes_nobody():
    tokens = tokenize("no one cares")
    assert "nobody" in tokens

def test_empty_string():
    tokens = tokenize("")
    assert tokens == []

def test_preserves_contractions():
    tokens = tokenize("I can't believe it")
    assert "cant" in tokens

def test_cancer_free():
    tokens = tokenize("finally cancer free")
    assert "cancerfree" in tokens
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest engine_v9/tests/test_tokenizer.py -v`
Expected: FAIL

- [ ] **Step 3: Create tokenizer.py**

```python
# engine_v9/tokenizer.py
"""V9 Tokenizer — splits text into atoms, resolves compound bonds.

Compound bonds are multi-word phrases that form a single emotional atom.
"laid off" → "laidoff" (one atom with its own root charge).
Resolved BEFORE root mapping so the mapper sees molecular roots.
"""

import re

# ── Compound bond tables ─────────────────────────────────────────

COMPOUND_BONDS_TRI = {
    ("got", "laid", "off"): "laidoff",
    ("get", "laid", "off"): "laidoff",
    ("got", "kicked", "out"): "kickedout",
    ("got", "locked", "out"): "lockedout",
    ("got", "ripped", "off"): "rippedoff",
    ("got", "thrown", "out"): "thrownout",
    ("got", "cut", "off"): "cutoff",
    ("got", "burned", "out"): "burnedout",
    ("got", "wiped", "out"): "wipedout",
}

COMPOUND_BONDS = {
    ("laid", "off"): "laidoff",
    ("food", "poisoning"): "foodpoisoning",
    ("broke", "down"): "brokedown",
    ("locked", "out"): "lockedout",
    ("kicked", "out"): "kickedout",
    ("passed", "away"): "passedaway",
    ("cut", "off"): "cutoff",
    ("thrown", "out"): "thrownout",
    ("ripped", "off"): "rippedoff",
    ("wiped", "out"): "wipedout",
    ("burned", "out"): "burnedout",
    ("shut", "down"): "shutdown",
    ("backed", "out"): "backedout",
    ("dropped", "out"): "droppedout",
    ("sold", "out"): "soldout",
    ("stressed", "out"): "stressedout",
    ("freaked", "out"): "freakedout",
    ("cancer", "free"): "cancerfree",
    ("debt", "free"): "debtfree",
    ("pain", "free"): "painfree",
    ("pulled", "off"): "pulledoff",
    ("pulled", "through"): "pulledthrough",
    ("worked", "out"): "workedout",
    ("paid", "off"): "paidoff",
    ("turned", "around"): "turnedaround",
    ("broke", "up"): "brokeup",
}

SPECIAL_PAIRS = {
    ("no", "one"): "nobody",
    ("no", "cap"): "nocap",
    ("come", "on"): "comeon",
}


def _clean(word: str) -> str:
    """Lowercase and strip punctuation."""
    return re.sub(r"[^\w']+", "", word.lower()).replace("'", "")


def tokenize(text: str) -> list:
    """Tokenize text into atoms with compound bond resolution.

    Returns list of cleaned, lowercased tokens with compounds fused.
    """
    if not text or not text.strip():
        return []

    raw = text.split()
    words = [_clean(w) for w in raw if _clean(w)]

    if not words:
        return []

    # Trigram compounds first (longer match wins)
    result = []
    i = 0
    while i < len(words):
        matched = False

        # Try trigram
        if i + 2 < len(words):
            tri = (words[i], words[i+1], words[i+2])
            if tri in COMPOUND_BONDS_TRI:
                result.append(COMPOUND_BONDS_TRI[tri])
                i += 3
                matched = True

        # Try bigram
        if not matched and i + 1 < len(words):
            bi = (words[i], words[i+1])
            if bi in COMPOUND_BONDS:
                result.append(COMPOUND_BONDS[bi])
                i += 2
                matched = True
            elif bi in SPECIAL_PAIRS:
                result.append(SPECIAL_PAIRS[bi])
                i += 2
                matched = True

        if not matched:
            result.append(words[i])
            i += 1

    return result
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest engine_v9/tests/test_tokenizer.py -v`
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add engine_v9/tokenizer.py engine_v9/tests/test_tokenizer.py
git commit -m "V9: tokenizer with compound bond resolution"
```

---

### Task 5: Decomposer — Event Nucleus Detection

**Files:**
- Create: `engine_v9/decomposer.py`
- Create: `engine_v9/tests/test_decomposer.py`

The decomposer finds the highest-gravity atom (event nucleus) and assigns structural roles relative to it. This is the core V9 innovation — meaning is a field around a mass, not a timeline.

- [ ] **Step 1: Write tests for nucleus detection**

```python
# engine_v9/tests/test_decomposer.py
from engine_v9.decomposer import decompose, Equation
from engine_v9.roots import RootCategory

def test_simple_negative_event():
    eq = decompose("I just got laid off from work")
    assert eq.nucleus.root.name == "EMPLOYMENT_LOSS"

def test_simple_positive():
    eq = decompose("I am so happy")
    assert eq.nucleus.root.category == RootCategory.POSITIVE_STATE

def test_yoda_order_same_nucleus():
    eq1 = decompose("I just got laid off from work")
    eq2 = decompose("Laid off from work I just got")
    assert eq1.nucleus.root.name == eq2.nucleus.root.name

def test_subject_is_self():
    eq = decompose("I am happy")
    assert eq.subject.root.category == RootCategory.SELF_REF

def test_implicit_self():
    """No explicit subject → implicit SELF."""
    eq = decompose("feeling sad today")
    assert eq.subject.root.category == RootCategory.SELF_REF

def test_negator_in_operators():
    eq = decompose("I am not happy")
    operator_cats = [a.root.category for a in eq.operators]
    assert RootCategory.NEGATOR in operator_cats

def test_context_atoms():
    eq = decompose("I got laid off from work yesterday")
    context_cats = [a.root.category for a in eq.context]
    # "work" and "yesterday" should be context
    assert len(eq.context) >= 1

def test_tie_breaks_to_later_atom():
    """When two atoms have equal gravity, later one wins (punchline rule)."""
    eq = decompose("I am happy but devastated")
    # "devastated" has higher gravity AND comes later
    assert eq.nucleus.root.charge[0] < 0

def test_equation_has_all_parts():
    eq = decompose("I am really happy")
    assert eq.subject is not None
    assert eq.nucleus is not None
    assert isinstance(eq.context, list)
    assert isinstance(eq.operators, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest engine_v9/tests/test_decomposer.py -v`
Expected: FAIL

- [ ] **Step 3: Create decomposer.py**

```python
# engine_v9/decomposer.py
"""V9 Equation Decomposer — dissects sentence into SUBJECT + EVENT + CONTEXT + OPERATORS.

Core insight: meaning is a field around a mass, not a timeline.
The Event Nucleus is the highest-gravity atom. Everything else is
positioned relative to it: subject, operators, context.

Two passes, O(n):
  Pass 1: Map words to roots, compute gravity for each atom
  Pass 2: Find nucleus (max gravity), assign roles
"""

from dataclasses import dataclass, field
from typing import List, Optional

from engine_v9.roots import Root, ROOTS, RootCategory
from engine_v9.root_map import map_to_root
from engine_v9.tokenizer import tokenize


# Root categories that can be a SUBJECT
_SUBJECT_CATEGORIES = frozenset({
    RootCategory.SELF_REF,
    RootCategory.OTHER_REF,
    RootCategory.RELATION_REF,
    RootCategory.PERSON,
})

# Root categories that are OPERATORS (modify the event, don't carry standalone meaning)
_OPERATOR_CATEGORIES = frozenset({
    RootCategory.NEGATOR,
    RootCategory.INTENSIFIER,
    RootCategory.HEDGE,
    RootCategory.COMPRESSOR,
})

# Root categories that carry emotional charge (candidates for nucleus)
_CHARGED_CATEGORIES = frozenset({
    RootCategory.POSITIVE_STATE, RootCategory.NEGATIVE_STATE,
    RootCategory.POSITIVE_QUALITY, RootCategory.NEGATIVE_QUALITY,
    RootCategory.POSITIVE_EVENT, RootCategory.NEGATIVE_EVENT,
    RootCategory.SOCIAL_EVAL_POS, RootCategory.SOCIAL_EVAL_NEG,
    RootCategory.EXPECTATION_VIOLATION,
    RootCategory.COMPOUND_EVENT,
})


@dataclass
class Atom:
    """A word mapped to its root, with position info."""
    word: str
    root: Root
    position: int
    gravity: float = 0.0  # computed from charge magnitude


@dataclass
class Equation:
    """Decomposed sentence equation."""
    subject: Atom
    nucleus: Atom
    context: List[Atom] = field(default_factory=list)
    operators: List[Atom] = field(default_factory=list)
    all_atoms: List[Atom] = field(default_factory=list)


# Implicit self atom for sentences without explicit subject
_IMPLICIT_SELF = Atom(
    word="[self]",
    root=ROOTS["SELF"],
    position=-1,
    gravity=0.0,
)


def _compute_gravity(root: Root) -> float:
    """Gravity = magnitude of charge vector. Higher charge = stronger pull.

    Uses |dV| as primary signal, weighted by sum of other dimensions.
    Council consensus: nucleus is the atom with highest emotional mass.
    """
    c = root.charge
    # Primary: absolute valence
    v_mag = abs(c[0])
    # Secondary: sum of absolute values of other dimensions (A, D, U, G, W, I)
    other_mag = sum(abs(x) for x in c[1:])
    # Gravity = valence magnitude + secondary contribution
    return v_mag + other_mag * 0.2


def decompose(text: str) -> Equation:
    """Decompose text into equation: SUBJECT + EVENT_NUCLEUS + CONTEXT + OPERATORS.

    The event nucleus is the highest-gravity atom — the emotional center of mass.
    Everything else is positioned relative to it.

    Handles Yoda order: "Laid off from work, I just got" produces the same
    nucleus as "I just got laid off from work".
    """
    tokens = tokenize(text)
    if not tokens:
        return Equation(
            subject=_IMPLICIT_SELF,
            nucleus=Atom(word="[empty]", root=ROOTS["OBJECT_GENERIC"], position=0),
        )

    # Pass 1: map words to roots, compute gravity
    atoms = []
    for i, token in enumerate(tokens):
        root = map_to_root(token)
        gravity = _compute_gravity(root)
        atoms.append(Atom(word=token, root=root, position=i, gravity=gravity))

    # Pass 2: find event nucleus (highest gravity, ties → later atom wins)
    nucleus_idx = 0
    max_gravity = atoms[0].gravity
    for i, atom in enumerate(atoms):
        if atom.gravity > max_gravity or (atom.gravity == max_gravity and i > nucleus_idx):
            max_gravity = atom.gravity
            nucleus_idx = i

    nucleus = atoms[nucleus_idx]

    # Pass 3: assign roles relative to nucleus
    subject = None
    operators = []
    context = []

    for i, atom in enumerate(atoms):
        if i == nucleus_idx:
            continue

        if atom.root.category in _SUBJECT_CATEGORIES and subject is None:
            # Nearest person-like atom becomes subject
            subject = atom
        elif atom.root.category in _OPERATOR_CATEGORIES:
            # Operators modify the event
            operators.append(atom)
        elif atom.root.category not in (RootCategory.FILLER, RootCategory.FORMULAIC,
                                         RootCategory.CONNECTOR, RootCategory.TEMPORAL):
            # Remaining charged atoms are context
            context.append(atom)

    # Default subject = implicit self if none found
    if subject is None:
        subject = _IMPLICIT_SELF

    return Equation(
        subject=subject,
        nucleus=nucleus,
        context=context,
        operators=operators,
        all_atoms=atoms,
    )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest engine_v9/tests/test_decomposer.py -v`
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add engine_v9/decomposer.py engine_v9/tests/test_decomposer.py
git commit -m "V9: equation decomposer — gravity-based nucleus detection + role assignment"
```

---

### Task 6: Composer — Charge Resolution from Equation

**Files:**
- Create: `engine_v9/composer.py`
- Create: `engine_v9/tests/test_composer.py`

The composer takes a decomposed equation and resolves it to a 7D VADUGWI coordinate. Structure-dependent charge weighting: EVENT=1.0, CONTEXT=0.3, FORMULAIC=0. Operators (negators, intensifiers) modify the nucleus charge before composition.

- [ ] **Step 1: Write tests**

```python
# engine_v9/tests/test_composer.py
from engine_v9.composer import compose
from engine_v9.decomposer import decompose

CENTER = 128

def test_positive_sentence():
    eq = decompose("I am happy")
    result = compose(eq)
    assert result.v > CENTER  # positive

def test_negative_sentence():
    eq = decompose("I am sad")
    result = compose(eq)
    assert result.v < CENTER  # negative

def test_negation_flips():
    eq_pos = decompose("I am happy")
    eq_neg = decompose("I am not happy")
    v_pos = compose(eq_pos)
    v_neg = compose(eq_neg)
    assert v_pos.v > CENTER
    assert v_neg.v < CENTER

def test_intensifier_amplifies():
    eq_base = decompose("I am happy")
    eq_intense = decompose("I am very happy")
    v_base = compose(eq_base)
    v_intense = compose(eq_intense)
    assert v_intense.v > v_base.v

def test_strong_negative_event():
    eq = decompose("I just got laid off from work")
    result = compose(eq)
    assert result.v < 90  # clearly negative
    assert result.d < CENTER  # loss of control

def test_yoda_same_result():
    eq1 = decompose("I just got laid off from work")
    eq2 = decompose("Laid off from work I just got")
    v1 = compose(eq1)
    v2 = compose(eq2)
    # Should produce similar V (within 15 points)
    assert abs(v1.v - v2.v) <= 15

def test_neutral_sentence():
    eq = decompose("the cat sat on the mat")
    result = compose(eq)
    # Should be near neutral
    assert 100 < result.v < 156

def test_result_is_clamped():
    eq = decompose("I am absolutely devastated destroyed shattered")
    result = compose(eq)
    assert 0 <= result.v <= 255
    assert 0 <= result.a <= 255
    assert 0 <= result.d <= 255

def test_context_weighted_less_than_nucleus():
    """Context atoms contribute less than the nucleus."""
    # "happy" as nucleus vs "happy" as context with a stronger nucleus
    eq1 = decompose("I am happy")
    eq2 = decompose("I am devastated but the weather is happy")
    v1 = compose(eq1)
    v2 = compose(eq2)
    # eq2 should still be negative because devastated is nucleus
    assert v2.v < CENTER
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest engine_v9/tests/test_composer.py -v`
Expected: FAIL

- [ ] **Step 3: Create composer.py**

```python
# engine_v9/composer.py
"""V9 Equation Composer — resolves decomposed equation to VADUGWI.

The equation: V = f(SUBJECT, EVENT_NUCLEUS, CONTEXT[], OPERATORS[])

Structure-dependent charge weighting:
  EVENT_NUCLEUS: 1.0 (full charge — this IS the meaning)
  CONTEXT:       0.3 (coloring, not driving)
  FORMULAIC:     0.0 (zero contribution)

Operators modify the nucleus BEFORE composition:
  NEGATOR:      inverts charge (×-1)
  INTENSIFIER:  amplifies charge (×1.5)
  HEDGE:        dampens charge (×0.5)
  COMPRESSOR:   reduces arousal
"""

from math import tanh
from engine_v9.shared import VADUG
from engine_v9.decomposer import Equation, Atom
from engine_v9.roots import RootCategory


# ── Composition constants ────────────────────────────────────────

CENTER = 128.0
FORCE_SCALE = 1.4           # how much charges move the needle
EVENT_WEIGHT = 1.0          # nucleus gets full weight
CONTEXT_WEIGHT = 0.3        # context atoms color, don't drive
SATURATION = 120.0          # tanh compression range
NEGATOR_FACTOR = -1.0       # full inversion
INTENSIFIER_FACTOR = 1.5    # 50% amplification
HEDGE_FACTOR = 0.5          # 50% dampening
SUBJECT_SELF_BONUS = 1.2    # self-reference amplifies charge


def _apply_operators(charge: list, operators: list) -> list:
    """Apply operator effects to a charge vector.

    Negators invert, intensifiers amplify, hedges dampen.
    Multiple operators stack multiplicatively.
    """
    multiplier = 1.0
    for op in operators:
        if op.root.category == RootCategory.NEGATOR:
            multiplier *= NEGATOR_FACTOR
        elif op.root.category == RootCategory.INTENSIFIER:
            multiplier *= INTENSIFIER_FACTOR
        elif op.root.category == RootCategory.HEDGE:
            multiplier *= HEDGE_FACTOR
        elif op.root.category == RootCategory.COMPRESSOR:
            multiplier *= 0.7

    return [c * multiplier for c in charge]


def compose(equation: Equation) -> VADUG:
    """Compose equation into VADUGWI coordinate.

    Steps:
      1. Start from center (128, 128, 128, 0, 128, 128, 128)
      2. Apply operators to nucleus charge
      3. Add weighted nucleus charge
      4. Add weighted context charges
      5. Apply subject modifier
      6. Saturate (tanh compression) and clamp
    """
    # Start at neutral
    state = [CENTER, CENTER, CENTER, 0.0, CENTER, CENTER, CENTER]

    # Step 1: get nucleus charge, apply operators
    nuc_charge = list(equation.nucleus.root.charge)
    nuc_charge = _apply_operators(nuc_charge, equation.operators)

    # Step 2: apply nucleus to state (full weight)
    for dim in range(7):
        state[dim] += nuc_charge[dim] * EVENT_WEIGHT * FORCE_SCALE

    # Step 3: apply context atoms (reduced weight)
    for ctx_atom in equation.context:
        ctx_charge = list(ctx_atom.root.charge)
        for dim in range(7):
            state[dim] += ctx_charge[dim] * CONTEXT_WEIGHT * FORCE_SCALE

    # Step 4: subject modifier — self-reference amplifies emotional impact
    if equation.subject.root.category == RootCategory.SELF_REF:
        for dim in range(7):
            deviation = state[dim] - (CENTER if dim != 3 else 0.0)
            state[dim] = (CENTER if dim != 3 else 0.0) + deviation * SUBJECT_SELF_BONUS

    # Step 5: tanh saturation (smooth compression, prevents runaway)
    for dim in range(7):
        center = CENTER if dim != 3 else 0.0
        deviation = state[dim] - center
        state[dim] = center + SATURATION * tanh(deviation / SATURATION)

    # Step 6: clamp to 0-255
    return VADUG(
        v=max(0, min(255, int(round(state[0])))),
        a=max(0, min(255, int(round(state[1])))),
        d=max(0, min(255, int(round(state[2])))),
        u=max(0, min(255, int(round(state[3])))),
        g=max(0, min(255, int(round(state[4])))),
        w=max(0, min(255, int(round(state[5])))),
        i=max(0, min(255, int(round(state[6])))),
    )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest engine_v9/tests/test_composer.py -v`
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add engine_v9/composer.py engine_v9/tests/test_composer.py
git commit -m "V9: equation composer — structure-dependent charge resolution"
```

---

### Task 7: Pipeline — Full V9 Entry Point

**Files:**
- Create: `engine_v9/pipeline.py`
- Modify: `engine_v9/__init__.py`
- Create: `engine_v9/tests/test_pipeline.py`

Wire up the full pipeline: tokenize → root map → decompose → compose → VADUGWI. This is the public API.

- [ ] **Step 1: Write tests**

```python
# engine_v9/tests/test_pipeline.py
from engine_v9.pipeline import compute_vadug

CENTER = 128

def test_basic_positive():
    result, trace = compute_vadug("I am happy")
    assert result.v > CENTER

def test_basic_negative():
    result, trace = compute_vadug("I am sad")
    assert result.v < CENTER

def test_negation():
    result, trace = compute_vadug("I am not happy")
    assert result.v < CENTER

def test_strong_negative_event():
    result, trace = compute_vadug("I just got laid off from work")
    assert result.v < 100

def test_yoda_order():
    v1, _ = compute_vadug("I just got laid off from work")
    v2, _ = compute_vadug("Laid off from work I just got")
    assert abs(v1.v - v2.v) <= 15

def test_neutral():
    result, _ = compute_vadug("the weather is cloudy today")
    assert 100 < result.v < 156

def test_trace_has_equation():
    _, trace = compute_vadug("I am happy")
    assert "equation" in trace
    assert "nucleus" in trace["equation"]
    assert "subject" in trace["equation"]

def test_trace_has_tokens():
    _, trace = compute_vadug("I am happy")
    assert "tokens" in trace

def test_empty_string():
    result, _ = compute_vadug("")
    assert result.v == CENTER

def test_returns_vadug_type():
    result, _ = compute_vadug("hello world")
    assert hasattr(result, 'v')
    assert hasattr(result, 'a')
    assert hasattr(result, 'd')
    assert hasattr(result, 'u')
    assert hasattr(result, 'g')
    assert hasattr(result, 'w')
    assert hasattr(result, 'i')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest engine_v9/tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Create pipeline.py**

```python
# engine_v9/pipeline.py
"""V9 Pipeline — the full equation decomposition engine.

Pipeline:
  text → tokenize → decompose (find nucleus, assign roles) → compose → VADUGWI

Public API: compute_vadug(text) → (VADUG, trace_dict)
"""

from typing import Optional, Tuple

from engine_v9.shared import VADUG, PersonalityVector
from engine_v9.tokenizer import tokenize
from engine_v9.decomposer import decompose
from engine_v9.composer import compose


def compute_vadug(
    text: str,
    personality: Optional[PersonalityVector] = None,
    perspective: str = "speaker",
) -> Tuple[VADUG, dict]:
    """Compute VADUGWI coordinates for text using equation decomposition.

    Args:
        text: Input sentence/text
        personality: Optional personality vector (applied as final modifier)
        perspective: "speaker", "listener", or "bystander"

    Returns:
        (VADUG, trace_dict) where trace contains decomposition details
    """
    # Stage 1: tokenize
    tokens = tokenize(text)

    # Stage 2: decompose into equation
    equation = decompose(text)

    # Stage 3: compose equation into VADUGWI
    result = compose(equation)

    # Stage 4: apply personality (if provided)
    if personality is not None:
        sensitivity = personality.emotional_sensitivity
        CENTER = 128
        result = VADUG(
            v=max(0, min(255, int(CENTER + (result.v - CENTER) * sensitivity))),
            a=max(0, min(255, int(CENTER + (result.a - CENTER) * sensitivity))),
            d=max(0, min(255, int(CENTER + (result.d - CENTER) * sensitivity + personality.dominance_baseline))),
            u=max(0, min(255, int(result.u * sensitivity))),
            g=max(0, min(255, int(CENTER + (result.g - CENTER) * sensitivity + personality.gravity_bias))),
            w=max(0, min(255, int(CENTER + (result.w - CENTER) * sensitivity))),
            i=result.i,
        )

    # Build trace for debugging
    trace = {
        "tokens": tokens,
        "equation": {
            "subject": equation.subject.word,
            "nucleus": equation.nucleus.word,
            "nucleus_root": equation.nucleus.root.name,
            "nucleus_gravity": equation.nucleus.gravity,
            "operators": [a.word for a in equation.operators],
            "context": [a.word for a in equation.context],
        },
        "word_count": len(tokens),
    }

    return result, trace
```

- [ ] **Step 4: Update __init__.py**

```python
# engine_v9/__init__.py
from engine_v9.shared import VADUG, PersonalityVector
from engine_v9.pipeline import compute_vadug
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest engine_v9/tests/test_pipeline.py -v`
Expected: 10 PASSED

- [ ] **Step 6: Commit**

```bash
git add engine_v9/pipeline.py engine_v9/__init__.py engine_v9/tests/test_pipeline.py
git commit -m "V9: full pipeline — tokenize → decompose → compose → VADUGWI"
```

---

### Task 8: Shadow Comparison Harness

**Files:**
- Create: `benchmarks/v9_shadow.py`

Run both V8 and V9 on the same input, log divergences. This is how we measure V9 progress without breaking V8.

- [ ] **Step 1: Write the shadow comparison script**

```python
# benchmarks/v9_shadow.py
"""V9 Shadow Pipeline — run V8 and V9 side-by-side, log divergences.

Usage:
  python3 benchmarks/v9_shadow.py                    # quick comparison on ground truth
  python3 benchmarks/v9_shadow.py --verbose           # show every sentence
  python3 benchmarks/v9_shadow.py --threshold 20      # only show divergences > 20 points
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.pendulum import compute_vadug as v8_compute
from engine_v9.pipeline import compute_vadug as v9_compute


def load_verified_sentences():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "datasets", "verified_sentences.json")
    with open(path) as f:
        data = json.load(f)
    return data["sentences"]


def classify(v):
    if v >= 145:
        return "positive"
    elif v < 110:
        return "negative"
    return "neutral"


def run_shadow(threshold=15, verbose=False):
    sentences = load_verified_sentences()
    print(f"V9 Shadow Pipeline — {len(sentences)} verified sentences")
    print(f"Divergence threshold: {threshold} points")
    print("=" * 80)

    divergences = []
    v8_correct = 0
    v9_correct = 0
    agree = 0

    for s in sentences:
        text = s["text"]
        human = s["human_label"]

        v8_result, _ = v8_compute(text)
        v9_result, v9_trace = v9_compute(text)

        v8_label = classify(v8_result.v)
        v9_label = classify(v9_result.v)

        v8_ok = (v8_label == human)
        v9_ok = (v9_label == human)

        if v8_ok:
            v8_correct += 1
        if v9_ok:
            v9_correct += 1
        if v8_label == v9_label:
            agree += 1

        delta_v = abs(v8_result.v - v9_result.v)

        if delta_v > threshold or verbose:
            marker = ""
            if v9_ok and not v8_ok:
                marker = " ← V9 WINS"
            elif v8_ok and not v9_ok:
                marker = " ← V8 WINS"
            elif not v8_ok and not v9_ok:
                marker = " ← BOTH WRONG"

            print(f"\n  \"{text[:70]}...\"" if len(text) > 70 else f"\n  \"{text}\"")
            print(f"  Human: {human}")
            print(f"  V8: V={v8_result.v} ({v8_label}) {'OK' if v8_ok else 'WRONG'}")
            print(f"  V9: V={v9_result.v} ({v9_label}) {'OK' if v9_ok else 'WRONG'}")
            print(f"  Nucleus: {v9_trace['equation']['nucleus']} "
                  f"(root={v9_trace['equation']['nucleus_root']})")
            print(f"  Delta V: {delta_v}{marker}")

            if delta_v > threshold:
                divergences.append({
                    "text": text,
                    "human": human,
                    "v8_v": v8_result.v,
                    "v9_v": v9_result.v,
                    "v8_label": v8_label,
                    "v9_label": v9_label,
                    "nucleus": v9_trace["equation"]["nucleus"],
                })

    print("\n" + "=" * 80)
    print(f"RESULTS:")
    print(f"  V8 accuracy: {v8_correct}/{len(sentences)} ({v8_correct/len(sentences)*100:.1f}%)")
    print(f"  V9 accuracy: {v9_correct}/{len(sentences)} ({v9_correct/len(sentences)*100:.1f}%)")
    print(f"  Agreement:   {agree}/{len(sentences)} ({agree/len(sentences)*100:.1f}%)")
    print(f"  Divergences: {len(divergences)} (>{threshold} points)")

    v9_wins = sum(1 for d in divergences
                  if classify(d["v9_v"]) == d["human"] and classify(d["v8_v"]) != d["human"])
    v8_wins = sum(1 for d in divergences
                  if classify(d["v8_v"]) == d["human"] and classify(d["v9_v"]) != d["human"])
    print(f"  V9 wins: {v9_wins}  |  V8 wins: {v8_wins}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V9 Shadow Pipeline")
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_shadow(threshold=args.threshold, verbose=args.verbose)
```

- [ ] **Step 2: Run it**

Run: `python3 benchmarks/v9_shadow.py`
Expected: Output showing V8 vs V9 accuracy on verified sentences. V9 will likely be lower initially — that's the baseline.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/v9_shadow.py
git commit -m "V9: shadow comparison harness — V8 vs V9 side-by-side"
```

---

### Task 9: Run Against Verified Sentences — Record V9 Baseline

**Files:**
- Create: `benchmarks/v9_baseline.py`

Standalone V9 benchmark against verified sentences. Records the starting accuracy so we know what to improve.

- [ ] **Step 1: Write the baseline benchmark**

```python
# benchmarks/v9_baseline.py
"""V9 Baseline Benchmark — measure V9 accuracy on verified sentences.

Records the starting point. Every future V9 change must improve or maintain this.

Usage:
  python3 benchmarks/v9_baseline.py
  python3 benchmarks/v9_baseline.py --verbose
  python3 benchmarks/v9_baseline.py --category grief
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v9.pipeline import compute_vadug


def load_verified():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "datasets", "verified_sentences.json")
    with open(path) as f:
        data = json.load(f)
    return data


def classify(v):
    if v >= 145:
        return "positive"
    elif v < 110:
        return "negative"
    return "neutral"


def run_baseline(verbose=False, category=None):
    data = load_verified()
    sentences = data["sentences"]

    if category:
        sentences = [s for s in sentences if s.get("source", "") == category]

    print(f"V9 Baseline Benchmark")
    print(f"Engine: V9 Equation Decomposition")
    print(f"Sentences: {len(sentences)}")
    print("=" * 70)

    correct = 0
    wrong = []
    by_label = {"positive": [0, 0], "negative": [0, 0], "neutral": [0, 0]}

    for s in sentences:
        text = s["text"]
        human = s["human_label"]
        result, trace = compute_vadug(text)
        v9_label = classify(result.v)

        is_correct = (v9_label == human)
        if is_correct:
            correct += 1
        else:
            wrong.append({
                "text": text,
                "human": human,
                "v9_label": v9_label,
                "v9_v": result.v,
                "nucleus": trace["equation"]["nucleus"],
                "nucleus_root": trace["equation"]["nucleus_root"],
            })

        by_label[human][0] += 1
        if is_correct:
            by_label[human][1] += 1

    accuracy = correct / len(sentences) * 100 if sentences else 0

    print(f"\nOVERALL: {correct}/{len(sentences)} ({accuracy:.1f}%)")
    print(f"\nBy label:")
    for label, (total, ok) in by_label.items():
        pct = ok / total * 100 if total else 0
        print(f"  {label:>8}: {ok}/{total} ({pct:.1f}%)")

    if verbose or len(wrong) <= 20:
        print(f"\nMISSES ({len(wrong)}):")
        for w in wrong:
            print(f"  \"{w['text'][:60]}\"")
            print(f"    Human={w['human']} V9={w['v9_label']} (V={w['v9_v']}) "
                  f"Nucleus={w['nucleus']} ({w['nucleus_root']})")

    print(f"\n{'=' * 70}")
    print(f"V9 BASELINE: {accuracy:.1f}%")
    print(f"V8 BASELINE: see benchmarks/full_barrage.py for comparison")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()
    run_baseline(verbose=args.verbose, category=args.category)
```

- [ ] **Step 2: Run it**

Run: `python3 benchmarks/v9_baseline.py --verbose`
Expected: V9 baseline accuracy printed. This is our starting number.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/v9_baseline.py
git commit -m "V9: baseline benchmark — records starting accuracy on verified sentences"
```

---

### Task 10: Expand Root Coverage

**Files:**
- Modify: `engine_v9/roots.py` (add more roots)
- Modify: `engine_v9/root_map.py` (add more word→root mappings)

After running the baseline, examine the misses. Add roots and mappings for the most common failure patterns. This task is iterative — repeat until baseline stabilizes.

- [ ] **Step 1: Run baseline and identify top failure categories**

Run: `python3 benchmarks/v9_baseline.py --verbose`
Look at the MISSES section. Group failures by pattern:
- Missing words (mapped to OBJECT_GENERIC when they should have charge)
- Wrong nucleus (a context word stole the nucleus role)
- Missing compounds (multi-word phrases not fused)

- [ ] **Step 2: Add roots for top failure patterns**

For each failure pattern, add the missing root to `engine_v9/roots.py` and the word mappings to `engine_v9/root_map.py`. Example:

If "frustrated" maps to OBJECT_GENERIC, add it to WORD_TO_ROOT:
```python
"frustrated": "FRUSTRATED",
```

If "broke up" isn't fusing, add it to `engine_v9/tokenizer.py`:
```python
("broke", "up"): "brokeup",
```

- [ ] **Step 3: Write a test for each new root/mapping**

Add to `engine_v9/tests/test_root_map.py`:
```python
def test_frustrated_maps_to_negative():
    root = map_to_root("frustrated")
    assert root.charge[0] < 0
```

- [ ] **Step 4: Run tests and baseline**

Run: `python3 -m pytest engine_v9/tests/ -v && python3 benchmarks/v9_baseline.py`
Expected: Tests pass, baseline accuracy improves.

- [ ] **Step 5: Commit**

```bash
git add engine_v9/roots.py engine_v9/root_map.py engine_v9/tests/test_root_map.py
git commit -m "V9: expand root coverage — [describe what was added]"
```

- [ ] **Step 6: Repeat steps 1-5 until baseline stabilizes**

Target: V9 baseline should be within 10% of V8 on verified sentences before moving to Phase 2 (structure detection integration).

---

## File Map Summary

```
engine_v9/
├── __init__.py          # Public API: compute_vadug, VADUG
├── shared.py            # VADUG + PersonalityVector (copied from V8)
├── phase.py             # SOLID/LIQUID/GAS (copied from V8)
├── roots.py             # Root categories + charge definitions (~55 starter, grow to ~500)
├── root_map.py          # Word → Root mapping (static + morphological + fallback)
├── tokenizer.py         # Tokenize + compound bond resolution
├── decomposer.py        # Equation decomposition (nucleus + roles)
├── composer.py          # Equation composition (charge resolution → VADUGWI)
├── pipeline.py          # Full pipeline orchestrator
└── tests/
    ├── __init__.py
    ├── test_shared.py
    ├── test_roots.py
    ├── test_root_map.py
    ├── test_tokenizer.py
    ├── test_decomposer.py
    ├── test_composer.py
    └── test_pipeline.py

benchmarks/
├── v9_shadow.py         # V8 vs V9 side-by-side comparison
└── v9_baseline.py       # V9 standalone accuracy benchmark
```

## What Comes After This Plan

This plan builds the V9 core: roots + decomposition + composition. Once the baseline stabilizes, the next phase adds:

1. **Phase system integration** — SOLID/LIQUID/GAS + SOLVENT dissolution
2. **Structure detection** — port V8's 45+ patterns to work on decomposed equations
3. **W→V coupling** — self-worth modulating valence (asymmetric)
4. **Crisis detection** — continuous concern gradient
5. **Bidirectional validation** — forward/backward as sanity check on decomposition
6. **Force flow** — WHO does WHAT to WHOM on top of equation structure

Each of these is a separate plan. This plan gets V9 running and measured.
