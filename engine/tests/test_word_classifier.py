"""Tests for V3 Layer 1: Word Role Classifier."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.word_classifier import classify_sentence, classify_word, WordRole


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
        assert roles[3].role == "RELATION_REF"  # dog is relationship
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

    def test_fine_without_self_ref(self):
        """'fine' not after SELF_REF should stay in its base role."""
        roles = classify_sentence(["that", "is", "fine"])
        # "fine" is in PEACE set, so it's PEACE regardless — position override
        # only matters if it could be EMOTIONAL vs PEACE
        assert roles[2].role == "PEACE"

    def test_chopper(self):
        roles = classify_sentence(["I", "love", "you", "but", "im", "leaving"])
        assert roles[3].role == "CHOPPER"

    def test_still_is_temporal(self):
        """'still' is always TEMPORAL — freshness/persistence marker."""
        roles = classify_sentence(["I", "still", "feel", "bad"])
        assert roles[1].role == "TEMPORAL"

    def test_never_is_negator(self):
        """'never' should be NEGATOR, not TEMPORAL."""
        roles = classify_sentence(["I", "never", "said", "that"])
        assert roles[1].role == "NEGATOR"

    def test_just_before_acquire_is_temporal(self):
        """'just bought' = recently acquired → TEMPORAL."""
        roles = classify_sentence(["I", "just", "bought", "a", "car"])
        assert roles[1].role == "TEMPORAL"
        assert roles[2].role == "ACQUIRE"

    def test_just_is_compressor(self):
        """'just' = COMPRESSOR — squeezes magnitude of nearby words."""
        roles = classify_sentence(["I", "just", "want", "to", "sleep"])
        assert roles[1].role == "COMPRESSOR"


class TestNeighbors:
    def test_neighbors_filled(self):
        roles = classify_sentence(["I", "am", "happy"])
        assert roles[0].neighbors == (None, "NEUTRAL")     # I: no left, "am" right
        assert roles[1].neighbors == ("SELF_REF", "EMOTIONAL")  # am: I left, happy right
        assert roles[2].neighbors == ("NEUTRAL", None)      # happy: am left, no right

    def test_single_word(self):
        roles = classify_sentence(["happy"])
        assert roles[0].neighbors == (None, None)

    def test_two_words(self):
        roles = classify_sentence(["very", "happy"])
        assert roles[0].neighbors == (None, "EMOTIONAL")
        assert roles[1].neighbors == ("AMPLIFIER", None)


class TestAmplifiers:
    def test_fucking_is_amplifier(self):
        """'fucking' is a 1.6x spice word amplifier."""
        roles = classify_sentence(["thats", "fucking", "amazing"])
        assert roles[1].role == "AMPLIFIER"

    def test_honestly_is_filler(self):
        """'honestly' is a glass breaker — classified as FILLER (processing marker)."""
        roles = classify_sentence(["honestly", "this", "sucks"])
        assert roles[0].role == "FILLER"

    def test_so_is_amplifier(self):
        """'so' before emotional word = AMPLIFIER."""
        roles = classify_sentence(["im", "so", "happy"])
        assert roles[1].role == "AMPLIFIER"


class TestEdgeCases:
    def test_empty_sentence(self):
        roles = classify_sentence([])
        assert roles == []

    def test_all_neutral(self):
        roles = classify_sentence(["the", "a", "is"])
        for r in roles:
            assert r.role == "NEUTRAL"

    def test_emotional_from_vocabulary(self):
        """Words not in any role set but in VOCABULARY with |dV| > 15 = EMOTIONAL."""
        roles = classify_sentence(["I", "am", "sad"])
        # "sad" is not in any role set, but is in VOCABULARY with big dV
        assert roles[2].role == "EMOTIONAL"
        assert roles[2].force is not None
        assert roles[2].force[0] < 0  # sad = negative V

    def test_hedge_words(self):
        roles = classify_sentence(["maybe", "I", "should"])
        assert roles[0].role == "HEDGE"

    def test_relation_ref(self):
        roles = classify_sentence(["my", "mom", "is", "angry"])
        assert roles[0].role == "SELF_REF"
        assert roles[1].role == "RELATION_REF"

    def test_word_role_dataclass(self):
        """WordRole stores all expected fields."""
        roles = classify_sentence(["I", "am", "happy"])
        r = roles[0]
        assert r.word == "i"
        assert r.role == "SELF_REF"
        assert r.base_role == "SELF_REF"
        assert r.position == 0
        assert r.force is None  # SELF_REF has no force

    def test_punctuation_stripped(self):
        """Punctuation should not affect classification."""
        roles = classify_sentence(["I'm", "fine!"])
        assert roles[0].role == "SELF_REF"
        assert roles[1].role == "PEACE"
