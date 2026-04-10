"""V9 vocabulary shim — provides VOCABULARY dict for word_classifier compatibility.

V9 uses the root system (roots.py + root_map.py) for emotional charges.
This shim generates a VOCABULARY dict from roots so that word_classifier.py
and proximity.py can work unchanged.
"""
from engine_v9.root_map import WORD_TO_ROOT
from engine_v9.roots import ROOTS

# Build VOCABULARY: word -> (dV, dA, dD, dU, dG) 5-tuple
# word_classifier only uses VOCABULARY to check if a word is EMOTIONAL
# and to get its force tuple. We extract the first 5 dims from the 7D root charge.
VOCABULARY = {}
for word, root_name in WORD_TO_ROOT.items():
    root = ROOTS.get(root_name)
    if root and any(root.charge[i] != 0 for i in range(5)):
        VOCABULARY[word] = root.charge[:5]  # (dV, dA, dD, dU, dG)
