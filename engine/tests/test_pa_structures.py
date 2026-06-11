"""Tests for Tier-1 passive aggression structures — embedded grievance.

Four phrasing classes where the complaint is smuggled into deniable
grammar via presupposition markers:

  TEMPORAL_GRIEVANCE — "finally"/"for once"/"about time" on ANOTHER
      person's action inside acknowledgment framing
  EXCLUSION_CONTRAST — "one of us"/"some of us"/"must be nice to"
      contrasting speaker deprivation vs target enjoyment
  IRONIC_DEFERENCE   — capitulation-as-attack ("youre clearly the expert")
  FAINT_PRAISE       — minimal praise + dismissive concession

All test sentences are NOVEL — not in any benchmark. Context-dependent
PA ("hope it was worth it", bare "k") is deliberately out of scope and
must stay untouched.
"""

import pytest

from engine.word_classifier import classify_sentence
from engine.structures import StructureDetector
from engine.pendulum import compute_vadug


# ── Helpers ──────────────────────────────────────────────────────

def _detect(sentence: str):
    roles = classify_sentence(sentence.split())
    return StructureDetector().detect_all(roles)


def _has_pattern(matches, pattern_name):
    return any(m.pattern == pattern_name for m in matches)


# ── TEMPORAL_GRIEVANCE ───────────────────────────────────────────

class TestTemporalGrievance:

    def test_thanks_for_finally_replying(self):
        """Gratitude frame + 'finally' = resented delay, not milestone."""
        matches = _detect("thanks for finally replying to my email")
        assert _has_pattern(matches, "TEMPORAL_GRIEVANCE")
        # The grievance reading suppresses the milestone reading
        assert not _has_pattern(matches, "RARITY_MARKER")

    def test_nice_of_you_to_finally_join(self):
        """'nice of you to finally join us' — politeness frame + delay marker."""
        matches = _detect("nice of you to finally join us")
        assert _has_pattern(matches, "TEMPORAL_GRIEVANCE")

    def test_about_time_second_person(self):
        """'about time you took out the trash' — presupposed resentment."""
        matches = _detect("about time you took out the trash")
        assert _has_pattern(matches, "TEMPORAL_GRIEVANCE")

    def test_for_once_second_person(self):
        """'for once you showed up on schedule' — rarity as complaint."""
        matches = _detect("for once you showed up on schedule")
        assert _has_pattern(matches, "TEMPORAL_GRIEVANCE")

    def test_self_finally_stays_milestone(self):
        """'i finally passed my driving test' — SELF actor keeps RARITY_MARKER."""
        matches = _detect("i finally passed my driving test")
        assert not _has_pattern(matches, "TEMPORAL_GRIEVANCE")
        assert _has_pattern(matches, "RARITY_MARKER")

    def test_you_finally_achievement_is_shared_joy(self):
        """'you finally got the promotion' — no acknowledgment frame, no
        deliberation verb: congratulation, not grievance."""
        matches = _detect("you finally got the promotion")
        assert not _has_pattern(matches, "TEMPORAL_GRIEVANCE")

    def test_temporal_grievance_reads_negative_v(self):
        vadug, _ = compute_vadug("thanks for finally texting me back")
        assert vadug.v < 110
        assert vadug.i > 150  # deniable jab = control-side intent


# ── EXCLUSION_CONTRAST ───────────────────────────────────────────

class TestExclusionContrast:

    def test_glad_one_of_us(self):
        """'glad one of us got to relax' — positive surface + partitive."""
        matches = _detect("glad one of us got to relax this weekend")
        assert _has_pattern(matches, "EXCLUSION_CONTRAST")

    def test_some_of_us_obligation(self):
        """'some of us have to clean this up' — partitive + obligation."""
        matches = _detect("some of us have to clean this up")
        assert _has_pattern(matches, "EXCLUSION_CONTRAST")

    def test_must_be_nice_to(self):
        """'must be nice to never check your bank account' — false envy."""
        matches = _detect("must be nice to never check your bank account")
        assert _has_pattern(matches, "EXCLUSION_CONTRAST")

    def test_must_be_nice_weather_guard(self):
        """'must be nice out there this morning' — impersonal, no target."""
        matches = _detect("must be nice out there this morning")
        assert not _has_pattern(matches, "EXCLUSION_CONTRAST")
        assert not _has_pattern(matches, "MARTYRDOM_FIELD")

    def test_one_of_us_logistics_guard(self):
        """'one of us can take the early shift' — logistics, no contrast."""
        matches = _detect("one of us can take the early shift")
        assert not _has_pattern(matches, "EXCLUSION_CONTRAST")

    def test_exclusion_contrast_reads_negative_v(self):
        vadug, _ = compute_vadug("glad one of us is enjoying the holidays")
        assert vadug.v < 110


# ── IRONIC_DEFERENCE ─────────────────────────────────────────────

class TestIronicDeference:

    def test_youre_clearly_the_expert(self):
        matches = _detect("youre clearly the expert on parenting")
        assert _has_pattern(matches, "IRONIC_DEFERENCE")

    def test_doubled_marker_intensifies(self):
        """'no no youre obviously the genius here' — doubled discourse marker."""
        matches = _detect("no no youre obviously the genius here")
        assert _has_pattern(matches, "IRONIC_DEFERENCE")

    def test_fine_youre_right(self):
        matches = _detect("fine youre right as always")
        assert _has_pattern(matches, "IRONIC_DEFERENCE")

    def test_genuine_deference_question_guard(self):
        """'youre clearly the expert so what should we order' — a genuine
        question after the grant = real deference, no attack."""
        matches = _detect("youre clearly the expert so what should we order")
        assert not _has_pattern(matches, "IRONIC_DEFERENCE")

    def test_sincere_concession_guard(self):
        """'fine youre right i should have asked first' — elaboration after
        the grant = sincere concession."""
        matches = _detect("fine youre right i should have asked first")
        assert not _has_pattern(matches, "IRONIC_DEFERENCE")

    def test_ironic_deference_d_stays_up(self):
        """The surrender is an assertion: V negative, D not collapsed."""
        vadug, _ = compute_vadug("youre clearly the expert")
        assert vadug.v < 110
        assert vadug.d >= 128


# ── FAINT_PRAISE ─────────────────────────────────────────────────

class TestFaintPraise:

    def test_interesting_strategy_but_ok(self):
        matches = _detect("interesting strategy but ok")
        assert _has_pattern(matches, "FAINT_PRAISE")

    def test_bold_strategy_standalone(self):
        matches = _detect("bold strategy")
        assert _has_pattern(matches, "FAINT_PRAISE")

    def test_thats_ellipsis_something(self):
        matches = _detect("thats... something")
        assert _has_pattern(matches, "FAINT_PRAISE")

    def test_well_thats_one_way(self):
        matches = _detect("well thats one way to handle it")
        assert _has_pattern(matches, "FAINT_PRAISE")

    def test_bare_interesting_choice_guard(self):
        """Bare 'interesting choice' is ambiguous — stays neutral."""
        matches = _detect("interesting choice")
        assert not _has_pattern(matches, "FAINT_PRAISE")

    def test_bare_one_way_guard(self):
        """Bare 'thats one way to do it' without dismissive opener — neutral."""
        matches = _detect("thats one way to do it")
        assert not _has_pattern(matches, "FAINT_PRAISE")

    def test_sincere_curiosity_guard(self):
        """'interesting choice what made you pick it' — real question."""
        matches = _detect("interesting choice what made you pick it")
        assert not _has_pattern(matches, "FAINT_PRAISE")

    def test_something_not_terminal_guard(self):
        """'thats something to be proud of' — 'something' not utterance-final."""
        matches = _detect("thats something to be proud of")
        assert not _has_pattern(matches, "FAINT_PRAISE")


# ── Cross-class invariants ───────────────────────────────────────

class TestPAInvariants:

    def test_context_dependent_pa_stays_out_of_scope(self):
        """Bare context-dependent PA must NOT fire sentence-level patterns."""
        for s in ("hope it was worth it", "sounds fun"):
            matches = _detect(s)
            for pat in ("TEMPORAL_GRIEVANCE", "EXCLUSION_CONTRAST",
                        "IRONIC_DEFERENCE", "FAINT_PRAISE"):
                assert not _has_pattern(matches, pat), (s, pat)

    def test_pa_lifts_intent_toward_control(self):
        """All four classes push I above 150 — the speaker is fighting."""
        for s in ("about time you answered your texts",
                  "some of us have to be up at six",
                  "no no youre clearly the expert here",
                  "bold strategy"):
            vadug, _ = compute_vadug(s)
            assert vadug.i > 150, s
