"""V8.6: structure reliability, axis inertness, forward negation, beat rung.

These test the rules themselves, not V bands on particular sentences.
"""

import pytest

from engine import pendulum
from engine.ladder import compose_beats, split_beats
from engine.pendulum import compute_vadug
from engine.proximity import (
    NEGATION_DECAY, NEGATION_FLIP, NEGATION_WINDOW, proximity_coefficient,
)
from engine.shared import VADUG
from engine.solver import A_WEIGHT, blend_axes, state_transition
from engine.vocabulary import VOCABULARY
from engine.word_classifier import classify_sentence


class TestAxisInertness:

    @pytest.mark.parametrize("text", [
        "i failed my exam again",
        "she smiled at the keys in the car",
        "they had to fire him",
        "the funeral was on a tuesday in the rain",
    ])
    def test_zero_valence_word_leaves_v_alone(self, text):
        _, meta = compute_vadug(text)
        previous_v = 128
        for entry in meta["trace"]:
            force = VOCABULARY.get(entry["word"])
            if force is not None and force[0] == 0:
                assert entry["v"] == previous_v, entry["word"]
            previous_v = entry["v"]

    def test_trailing_neutral_words_do_not_erase_valence(self):
        short, _ = compute_vadug("i failed")
        longer, _ = compute_vadug("i failed my exam again")
        assert short.v < 128
        assert longer.v <= short.v + 5


class TestForwardNegation:

    def _coeff(self, words, target):
        return proximity_coefficient(classify_sentence(words), target)

    def test_negator_flips_the_next_word(self):
        plain = self._coeff(["happy"], 0)
        negated = self._coeff(["not", "happy"], 1)
        assert negated == pytest.approx(plain * NEGATION_FLIP)

    def test_flip_fades_with_the_gap(self):
        gap2 = self._coeff(["not", "the", "happy"], 2)
        expected = 1.0 + (NEGATION_FLIP - 1.0) * NEGATION_DECAY
        assert gap2 == pytest.approx(expected)

    def test_negator_after_the_word_does_nothing(self):
        assert self._coeff(["happy", "not"], 0) == pytest.approx(self._coeff(["happy"], 0))

    def test_negator_beyond_the_window_does_nothing(self):
        filler = ["the"] * NEGATION_WINDOW
        far = self._coeff(["not", *filler, "happy"], NEGATION_WINDOW + 1)
        assert far == pytest.approx(self._coeff([*filler, "happy"], NEGATION_WINDOW))

    def test_sentence_level_flip(self):
        assert compute_vadug("i am not happy")[0].v < 128 < compute_vadug("i am happy")[0].v


class TestStructureReliability:

    TEXT = "homework is killing me"

    def test_gated_structure_is_still_detected(self, monkeypatch):
        monkeypatch.setitem(pendulum.STRUCTURE_RELIABILITY, "MUNDANE_HYPERBOLE", 0.0)
        _, meta = compute_vadug(self.TEXT)
        assert "MUNDANE_HYPERBOLE" in {s.pattern for s in meta["structures"]}

    def test_reliability_scales_only_the_valence_push(self, monkeypatch):
        monkeypatch.setitem(pendulum.STRUCTURE_RELIABILITY, "MUNDANE_HYPERBOLE", 1.0)
        trusted, _ = compute_vadug(self.TEXT)
        monkeypatch.setitem(pendulum.STRUCTURE_RELIABILITY, "MUNDANE_HYPERBOLE", 0.0)
        gated, _ = compute_vadug(self.TEXT)
        assert gated.v < trusted.v           # the +V push is gone
        assert gated.g == trusted.g          # other channels untouched
        assert gated.u == trusted.u

    def test_table_only_lowers_trust(self):
        assert all(0.0 <= r <= 1.0 for r in pendulum.STRUCTURE_RELIABILITY.values())


class TestBeatRung:

    def test_single_beat_equals_compute_vadug(self):
        text = "i miss my dog so much"
        assert compose_beats(text)[0] == compute_vadug(text)[0]

    def test_two_beats_fold_with_the_turn_blend(self):
        first, second = "I lost my job today.", "My friends took me out for dinner."
        a, b = compute_vadug(first)[0], compute_vadug(second)[0]
        expected = VADUG(*(int(round(x)) for x in blend_axes(a, b, A_WEIGHT)))
        result, meta = compose_beats(f"{first} {second}")
        assert result == expected
        assert [beat for beat, _ in meta["beats"]] == [first, second]

    def test_state_transition_starts_from_the_same_blend(self):
        a, b = VADUG(v=100, i=128), VADUG(v=150, i=128)   # no directional branch fires
        c = state_transition(a, b)
        assert c.v == int(round(blend_axes(a, b)[0]))

    def test_clause_split(self):
        assert split_beats("it was fine, but i hated the ending", clauses=True) == [
            "it was fine,", "but i hated the ending"]
        assert split_beats("ok") == ["ok"]
