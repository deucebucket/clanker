"""Rosetta Stone correction — compare engine output against known-good examples.

The Stone acts as a GPS landmark system. The engine computes position (VADUG),
then the Stone corrects drift by pulling toward the nearest verified reading.

More Stone entries = more landmarks = more accurate corrections.
"""

import re
import math


class StoneCorrector:
    """Correct pendulum VADUG output using Rosetta Stone as reference landmarks."""

    def __init__(self):
        self._entries = []
        self._loaded = False

    def _load_stone(self):
        """Lazy-load Rosetta Stone entries."""
        if self._loaded:
            return
        try:
            from benchmarks.rosetta_stone import ROSETTA_STONE
            for text, tv, ta, td, tu, tg, features in ROSETTA_STONE:
                words = set(re.findall(r"[a-z']+", text.lower()))
                self._entries.append({
                    "words": words,
                    "v": tv, "a": ta, "d": td, "u": tu, "g": tg,
                    "features": features,
                })
        except ImportError:
            pass
        self._loaded = True

    def correct(self, v, a, d, u, g, input_words):
        """Correct VADUG by blending toward nearest Rosetta Stone match.

        Args:
            v, a, d, u, g: raw pendulum output
            input_words: list of words from the input text

        Returns:
            (corrected_v, corrected_a, corrected_d, corrected_u, corrected_g, match_info)
        """
        self._load_stone()

        if not self._entries:
            return v, a, d, u, g, None

        input_set = set(input_words)
        best_match = None
        best_score = -1

        for entry in self._entries:
            # Word overlap (Jaccard similarity)
            overlap = len(input_set & entry["words"])
            union = len(input_set | entry["words"])
            word_sim = overlap / max(union, 1)

            # VADUG proximity (is our reading close to this Stone entry?)
            v_dist = abs(v - entry["v"])
            a_dist = abs(a - entry["a"])
            d_dist = abs(d - entry["d"])
            g_dist = abs(g - entry["g"])
            vadug_dist = (v_dist + a_dist * 0.5 + d_dist * 0.5 + g_dist * 0.5) / 4
            vadug_sim = 1.0 / (1.0 + vadug_dist / 40.0)

            # Combined: word similarity matters more than VADUG proximity
            # (we want to find sentences ABOUT the same thing, not just same numbers)
            score = word_sim * 0.7 + vadug_sim * 0.3

            if score > best_score:
                best_score = score
                best_match = entry

        if best_match is None or best_score < 0.1:
            return v, a, d, u, g, None

        # Blend toward Stone's values — strength proportional to match confidence
        blend = min(best_score * 0.4, 0.25)  # max 25% pull toward Stone

        cv = int(v * (1 - blend) + best_match["v"] * blend)
        ca = int(a * (1 - blend) + best_match["a"] * blend)
        cd = int(d * (1 - blend) + best_match["d"] * blend)
        cu = int(u * (1 - blend) + best_match["u"] * blend)
        cg = int(g * (1 - blend) + best_match["g"] * blend)

        match_info = {
            "features": best_match["features"],
            "score": round(best_score, 3),
            "blend": round(blend, 3),
            "stone_v": best_match["v"],
            "correction": cv - v,
        }

        return cv, ca, cd, cu, cg, match_info
