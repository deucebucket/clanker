"""Tests for word role detection — descriptors inherit emotion from subjects."""

import pytest
from demo.word_roles import WordRoleDetector
from demo.forces import WORD_FORCES


@pytest.fixture
def detector():
    return WordRoleDetector()


class TestDescriptorInheritance:
    """Descriptors should inherit valence direction from nearby subjects."""

    def test_moist_cake_positive(self, detector):
        """'moist cake' — moist inherits positive from cake."""
        words = ["moist", "cake"]
        base = WORD_FORCES["moist"]  # (-6, ...)
        result = detector.resolve_force("moist", base, words, 0)
        assert result[0] > 0, "moist should become positive near cake"

    def test_moist_wound_negative(self, detector):
        """'moist wound' — moist stays/becomes negative from wound."""
        words = ["moist", "wound"]
        base = WORD_FORCES["moist"]
        result = detector.resolve_force("moist", base, words, 0)
        assert result[0] < 0, "moist should be negative near wound"

    def test_dark_night_negative(self, detector):
        """'dark night' — dark stays negative (night is negative)."""
        words = ["dark", "night"]
        base = WORD_FORCES["dark"]  # (-28, ...)
        result = detector.resolve_force("dark", base, words, 0)
        assert result[0] < 0, "dark should stay negative near night"

    def test_dark_chocolate_positive(self, detector):
        """'dark chocolate' — dark inherits positive from chocolate."""
        words = ["dark", "chocolate"]
        base = WORD_FORCES["dark"]  # (-28, ...)
        result = detector.resolve_force("dark", base, words, 0)
        assert result[0] > 0, "dark should become positive near chocolate"

    def test_heavy_heart(self, detector):
        """'heavy heart' — heavy inherits from heart (positive subject)."""
        words = ["heavy", "heart"]
        base = WORD_FORCES["heavy"]  # (-28, ...)
        result = detector.resolve_force("heavy", base, words, 0)
        assert result[0] > 0, "heavy should become positive near heart"

    def test_the_cake_is_moist_lookbehind(self, detector):
        """'the cake is moist' — look-behind finds cake → positive."""
        words = ["the", "cake", "is", "moist"]
        base = WORD_FORCES["moist"]
        result = detector.resolve_force("moist", base, words, 3)
        assert result[0] > 0, "moist should become positive when cake is behind via bridge"


class TestNonDescriptors:
    """Non-descriptor words should pass through unchanged."""

    def test_non_descriptor_unchanged(self, detector):
        """Words not in DESCRIPTORS pass through with original force."""
        words = ["happy", "cake"]
        force = (50, 10, 5, 3, -2)
        result = detector.resolve_force("happy", force, words, 0)
        assert result == force

    def test_none_force_passthrough(self, detector):
        """Descriptor with None force returns None."""
        words = ["moist", "cake"]
        result = detector.resolve_force("moist", None, words, 0)
        assert result is None


class TestBridgeDetection:
    """Bridge words should be correctly identified."""

    def test_common_bridges(self, detector):
        for word in ["is", "are", "the", "a", "of", "with"]:
            assert detector.is_bridge(word), f"{word} should be a bridge"

    def test_non_bridges(self, detector):
        for word in ["cake", "happy", "moist", "run"]:
            assert not detector.is_bridge(word), f"{word} should not be a bridge"


class TestDescriptorDetection:
    """Descriptor words should be correctly identified."""

    def test_known_descriptors(self, detector):
        for word in ["moist", "dark", "heavy", "soft", "cold"]:
            assert detector.is_descriptor(word), f"{word} should be a descriptor"

    def test_non_descriptors(self, detector):
        for word in ["cake", "run", "happy", "the"]:
            assert not detector.is_descriptor(word), f"{word} should not be a descriptor"


class TestEdgeCases:
    """Edge cases should not crash."""

    def test_word_not_in_forces(self, detector):
        """Subject not in WORD_FORCES should not crash."""
        words = ["moist", "xyzzy"]
        base = WORD_FORCES["moist"]
        result = detector.resolve_force("moist", base, words, 0)
        # Should return without crashing — dv unchanged since xyzzy not in forces
        assert result is not None

    def test_descriptor_at_end_of_sentence(self, detector):
        """Descriptor at end with no lookahead subject."""
        words = ["the", "moist"]
        base = WORD_FORCES["moist"]
        result = detector.resolve_force("moist", base, words, 1)
        assert result is not None

    def test_other_tuple_elements_preserved(self, detector):
        """Only dv changes — other elements stay the same."""
        words = ["dark", "chocolate"]
        base = WORD_FORCES["dark"]  # (-28, 11, -18, 13, -22)
        result = detector.resolve_force("dark", base, words, 0)
        assert result[1:] == base[1:], "Only dv should change, rest preserved"
