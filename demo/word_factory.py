"""
Morpheme Word Factory — compose words from roots+prefixes+suffixes to hit VADUG targets.

This is the INVERSE of decomposition: instead of breaking a word into parts,
you build a word from parts to reach a specific emotional coordinate.
"""

import math
from demo.morphemes import ROOTS, PREFIXES, SUFFIXES


def _apply_prefix(v, a, d, u, g, prefix_key):
    """Apply a prefix modifier to base VADUG values using morpheme rules."""
    p = PREFIXES[prefix_key]
    rule = p["rule"]
    if rule == "negate":
        v = int(v * p["v"])
        d = int(d + d * p["d"])
        a = int(a + abs(a) * p["a"])
        u = int(u + 10 * p["u"])
        g = int(g + g * p.get("g", 0))
    elif rule == "amplify":
        v = int(v + v * p["v"])
        a = int(a + abs(a) * p["a"])
        d = int(d + d * p["d"])
        u = int(u + 10 * p["u"])
        g = int(g + g * p.get("g", 0))
    elif rule in ("reduce", "modify"):
        v = int(v + 10 * p["v"])
        a = int(a + 10 * p["a"])
        d = int(d + 10 * p["d"])
        u = int(u + 10 * p["u"])
        g = int(g + 10 * p.get("g", 0))
    return v, a, d, u, g


def _apply_suffix(v, a, d, u, g, suffix_key):
    """Apply a suffix modifier to base VADUG values using morpheme rules."""
    s = SUFFIXES[suffix_key]
    rule = s["rule"]
    if rule == "negate":
        v = int(v * s["v"])
        d = int(d + d * s["d"])
        a = int(a + abs(a) * s["a"])
        g = int(g + g * s.get("g", 0))
    elif rule == "amplify":
        v = int(v + v * s["v"])
        a = int(a + abs(a) * s["a"])
        d = int(d + d * s["d"])
        g = int(g + g * s.get("g", 0))
    elif rule in ("reduce", "modify"):
        v = int(v + 10 * s["v"])
        a = int(a + 10 * s["a"])
        d = int(d + 10 * s["d"])
        u = int(u + 10 * s.get("u", 0))
        g = int(g + 10 * s.get("g", 0))
    return v, a, d, u, g


def _root_vadug(root_data):
    """Extract (v, a, d, u, g) from a root entry (tuple or dict)."""
    if isinstance(root_data, tuple):
        v, a, d, u = root_data[0], root_data[1], root_data[2], root_data[3]
        g = root_data[4] if len(root_data) > 4 else 0
        return v, a, d, u, g
    if isinstance(root_data, dict):
        return (
            root_data.get("v", 0),
            root_data.get("a", 0),
            root_data.get("d", 0),
            root_data.get("u", 0),
            root_data.get("g", 0),
        )
    return (0, 0, 0, 0, 0)


class WordFactory:
    """Compose words from morpheme parts to hit VADUG targets."""

    # Valence is weighted 2x in distance calculations (primary emotional dimension)
    VALENCE_WEIGHT = 2.0

    def __init__(self):
        self._combos = []
        self._build_combinations()

    def _build_combinations(self):
        """Precompute VADUG vectors for all valid morpheme combinations."""
        combos = self._combos

        # Root only
        for root_key, root_data in ROOTS.items():
            force = _root_vadug(root_data)
            combos.append((root_key, force))

        # Prefix + root
        for prefix_key in PREFIXES:
            for root_key, root_data in ROOTS.items():
                v, a, d, u, g = _root_vadug(root_data)
                v, a, d, u, g = _apply_prefix(v, a, d, u, g, prefix_key)
                word = prefix_key + root_key
                combos.append((word, (v, a, d, u, g)))

        # Root + suffix
        for root_key, root_data in ROOTS.items():
            v, a, d, u, g = _root_vadug(root_data)
            for suffix_key in SUFFIXES:
                nv, na, nd, nu, ng = _apply_suffix(v, a, d, u, g, suffix_key)
                word = root_key + suffix_key
                combos.append((word, (nv, na, nd, nu, ng)))

        # Prefix + root + suffix
        for prefix_key in PREFIXES:
            for root_key, root_data in ROOTS.items():
                rv, ra, rd, ru, rg = _root_vadug(root_data)
                pv, pa, pd, pu, pg = _apply_prefix(rv, ra, rd, ru, rg, prefix_key)
                for suffix_key in SUFFIXES:
                    sv, sa, sd, su, sg = _apply_suffix(pv, pa, pd, pu, pg, suffix_key)
                    word = prefix_key + root_key + suffix_key
                    combos.append((word, (sv, sa, sd, su, sg)))

    def _distance(self, force, target):
        """Weighted Euclidean distance in 5D VADUG space (valence weighted 2x)."""
        w = self.VALENCE_WEIGHT
        return math.sqrt(
            w * (force[0] - target[0]) ** 2
            + (force[1] - target[1]) ** 2
            + (force[2] - target[2]) ** 2
            + (force[3] - target[3]) ** 2
            + (force[4] - target[4]) ** 2
        )

    def build_word(self, target_dv, target_da, target_dd, target_du, target_dg, top_n=5):
        """Find the top N morpheme combinations closest to the target force.

        Returns list of (word, force_tuple, distance) sorted by distance (closest first).
        """
        target = (target_dv, target_da, target_dd, target_du, target_dg)
        scored = []
        for word, force in self._combos:
            dist = self._distance(force, target)
            scored.append((word, force, dist))
        scored.sort(key=lambda x: x[2])
        return scored[:top_n]

    def build_positive(self, intensity=0.5, top_n=5):
        """Shortcut: build a word with positive valence at given intensity (0-1)."""
        dv = int(intensity * 55)
        return self.build_word(dv, 10, 10, 0, int(intensity * 25), top_n=top_n)

    def build_negative(self, intensity=0.5, top_n=5):
        """Shortcut: build a word with negative valence."""
        dv = int(-intensity * 55)
        return self.build_word(
            dv,
            int(intensity * 20),
            -int(intensity * 15),
            0,
            -int(intensity * 25),
            top_n=top_n,
        )

    def build_for_emotion(self, emotion_name, top_n=5):
        """Build a word matching a named emotion's typical VADUG profile."""
        EMOTION_TARGETS = {
            "joy": (40, 25, 15, 0, 30),
            "sadness": (-35, -10, -20, 0, -25),
            "anger": (-40, 45, 25, 15, 20),
            "fear": (-45, 40, -30, 30, -20),
            "love": (45, 20, 15, 0, 30),
            "disgust": (-35, 20, 10, 10, 10),
            "surprise": (5, 35, 0, 15, 15),
            "calm": (15, -20, 15, -10, 5),
        }
        if emotion_name not in EMOTION_TARGETS:
            return []
        return self.build_word(*EMOTION_TARGETS[emotion_name], top_n=top_n)


if __name__ == "__main__":
    factory = WordFactory()
    print(f"Precomputed {len(factory._combos)} morpheme combinations\n")

    for emotion in ("joy", "sadness", "anger", "fear", "love", "calm", "surprise"):
        print(f"--- {emotion.upper()} ---")
        results = factory.build_for_emotion(emotion, top_n=5)
        for word, force, dist in results:
            print(f"  {word:25s} VADUG={force}  dist={dist:.1f}")
        print()

    print("--- POSITIVE (0.7) ---")
    for word, force, dist in factory.build_positive(intensity=0.7, top_n=5):
        print(f"  {word:25s} VADUG={force}  dist={dist:.1f}")

    print("\n--- NEGATIVE (0.8) ---")
    for word, force, dist in factory.build_negative(intensity=0.8, top_n=5):
        print(f"  {word:25s} VADUG={force}  dist={dist:.1f}")
