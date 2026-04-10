"""V9 vocabulary — merges V8's calibrated per-word charges with V9 root charges.

Two sources:
  1. V8's EMOTIONAL_VOCABULARY (4,544 individually calibrated words, 5-tuple)
  2. V9's root system (manually classified words with 7D charges)

V9 manual classifications take priority (last-write-wins in the merge).
V8 words provide the broad coverage. V9 roots provide the precise overrides.

Used by word_classifier.py, proximity.py, structures.py, and interpret.py.
"""
import importlib
import sys
import os

# ── Source 1: V8 calibrated per-word charges ─────────────────────
# Import V8's forces_curated.py directly — these are REAL calibrated charges
_V8_VOCAB = {}
try:
    # Add parent directory to path temporarily to import from engine/
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from engine.forces_curated import EMOTIONAL_VOCABULARY as _EV
    _V8_VOCAB = dict(_EV)
except ImportError:
    # Standalone mode (e.g., inside PMS) — V8 not available
    # Fall back to a bundled copy if available
    pass

# ── Source 2: V9 root-derived charges ────────────────────────────
from .root_map import WORD_TO_ROOT
from .roots import ROOTS

_V9_VOCAB = {}
for word, root_name in WORD_TO_ROOT.items():
    root = ROOTS.get(root_name)
    if root and any(root.charge[i] != 0 for i in range(5)):
        _V9_VOCAB[word] = root.charge[:5]  # (dV, dA, dD, dU, dG)

# ── Merge: V8 base + V9 overrides ───────────────────────────────
# V8 provides 4,544 calibrated words as the foundation.
# V9 manual mappings override where we've explicitly reclassified.
VOCABULARY = {}
VOCABULARY.update(_V8_VOCAB)   # V8 base (4,544 words, per-word calibrated)
VOCABULARY.update(_V9_VOCAB)   # V9 overrides (manually classified words take priority)
