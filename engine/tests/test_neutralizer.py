"""Tests for the 2026-06-11 neutralizer (docs/neutralizer_proposal.md).

Covers:
  1. Audit-driven demotions: contested words are LIQUID with attenuated dV,
     while protected words (love) keep their SOLID charge.
  2. Demoted positive words resolve by context instead of dragging
     sentences positive on their own.
  3. Two-faced slang (cooked/snapped/sheesh) doom-vs-praise duality via
     SOLVENT dissolution.
"""

from engine.forces_curated import EMOTIONAL_VOCABULARY as VOCAB
from engine.pendulum import compute_vadug
from engine.phase import get_phase


def _v(text: str) -> float:
    result, _ = compute_vadug(text)
    return result.v


class TestNeutralizerDemotions:

    def test_demoted_words_are_liquid_and_attenuated(self):
        """Audit suspects are LIQUID with dV scaled by (1 - contradiction)."""
        assert get_phase("thank") == "LIQUID"
        assert VOCAB["thank"][0] == 12  # +50 x (1 - 0.77)
        assert get_phase("hope") == "LIQUID"
        assert VOCAB["hope"][0] == 22   # +45 x (1 - 0.50)
        assert get_phase("fantastic") == "LIQUID"
        assert VOCAB["fantastic"][0] == 0  # contradiction 1.00 -> fully drained

    def test_protected_words_untouched(self):
        """'love' (human contra 0.18 < baseline) stays SOLID and charged;
        'wonderful' was dropped as conflicted with the sarcasm detector."""
        assert get_phase("love") == "SOLID"
        assert VOCAB["love"][0] == 40
        assert get_phase("wonderful") == "GAS"
        assert VOCAB["wonderful"][0] == 35

    def test_demoted_positive_resolves_by_context(self):
        """A demoted word no longer single-handedly drags a hostile
        sentence positive, but genuine usage still reads positive."""
        assert _v("thank you for ruining everything") < 110
        assert _v("congratulations on completely ruining my entire life") < 110
        assert _v("thank you so much") > 128  # genuine gratitude stays warm
        assert _v("i love you") >= 140        # love untouched

    def test_demoted_word_stops_inflating_sarcasm(self):
        """'i truly cherish...' sarcasm no longer reads positive
        (truly +28->+5, cherish +37->0)."""
        assert _v("i truly cherish the two hours i spend in traffic every day") < 140


class TestSlangDuality:

    def test_cooked_doom_vs_praise(self):
        """'im cooked' = doomed (negative); casual register flips it
        to praise via SOLVENT dissolution."""
        assert _v("im cooked") < 110
        assert _v("im cooked for this exam") < 110
        assert _v("bruh he cooked") >= 140

    def test_snapped_anger_vs_praise(self):
        """'snapped at me' = anger; 'lmao she snapped' = performed
        brilliantly (SOLVENT flip)."""
        assert _v("he snapped at me in front of everyone") < 110
        assert _v("lmao she snapped on that verse") > 128

    def test_sheesh_no_longer_nuclear(self):
        """'sheesh' was -47 (exasperation-only); as mild LIQUID it must not
        sink an awe sentence deep negative."""
        assert _v("sheesh he really did that") >= 110

    def test_unambiguous_slang_positive(self):
        """goated / bussin / w carry direct positive charge."""
        assert _v("that fit is goated") >= 140
        assert _v("that meal was bussin fr") >= 140
        assert _v("thats a big w") >= 140
