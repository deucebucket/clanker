"""Tests for V3 Structure Detector -- chess-like pattern recognition.

All test sentences are NOVEL -- things never in any benchmark.
Safe sentences must NOT flag crisis patterns.
"""

import pytest

from engine.word_classifier import classify_sentence
from engine.structures import StructureDetector, StructureMatch


# ── Helper ───────────────────────────────────────────────────────

def _detect(sentence: str):
    """Classify and detect structures in a sentence."""
    roles = classify_sentence(sentence.split())
    detector = StructureDetector()
    return detector.detect_all(roles)


def _has_pattern(matches, pattern_name):
    """Check if a specific pattern was detected."""
    return any(m.pattern == pattern_name for m in matches)


def _get_pattern(matches, pattern_name):
    """Get the first match of a specific pattern."""
    for m in matches:
        if m.pattern == pattern_name:
            return m
    return None


# ── FAREWELL ─────────────────────────────────────────────────────

class TestFarewell:

    def test_gave_dog_to_neighbor(self):
        """'I gave my dog to my neighbor' -> DIVESTITURE (giving away possessions)."""
        matches = _detect("I gave my dog to my neighbor")
        assert _has_pattern(matches, "FAREWELL") or _has_pattern(matches, "DIVESTITURE")

    def test_leaving_keys_with_friend(self):
        """'leaving my keys with a friend' -> FAREWELL or DIVESTITURE."""
        matches = _detect("leaving my keys with a friend")
        assert _has_pattern(matches, "FAREWELL") or _has_pattern(matches, "DIVESTITURE") or _has_pattern(matches, "ABANDONMENT")

    def test_no_farewell_without_ref(self):
        """'I gave my dog a treat' -- no person ref, not FAREWELL."""
        matches = _detect("I gave my dog a treat")
        assert not _has_pattern(matches, "FAREWELL")


# ── METHOD_ACQUISITION ───────────────────────────────────────────

class TestMethodAcquisition:

    def test_bought_pills(self):
        """'just bought a bunch of pills' -> METHOD_ACQUISITION."""
        matches = _detect("just bought a bunch of pills")
        assert _has_pattern(matches, "METHOD_ACQUISITION")

    def test_found_a_gun(self):
        """'I found a gun in the closet' -> METHOD_ACQUISITION."""
        matches = _detect("I found a gun in the closet")
        assert _has_pattern(matches, "METHOD_ACQUISITION")

    def test_bought_groceries_safe(self):
        """'just bought some groceries' -- no METHOD, should be safe."""
        matches = _detect("just bought some groceries")
        assert not _has_pattern(matches, "METHOD_ACQUISITION")


# ── FINALITY ─────────────────────────────────────────────────────

class TestFinality:

    def test_last_time(self):
        """'this is the last time' -> FINALITY."""
        matches = _detect("this is the last time")
        assert _has_pattern(matches, "FINALITY")

    def test_goodbye(self):
        """'goodbye everyone' -> FINALITY."""
        matches = _detect("goodbye everyone")
        assert _has_pattern(matches, "FINALITY")

    def test_finality_with_self_ref(self):
        """'I am done with everything' -> FINALITY with SELF_REF boost."""
        matches = _detect("I am done with everything")
        assert _has_pattern(matches, "FINALITY")
        m = _get_pattern(matches, "FINALITY")
        assert m.confidence > 0.4  # boosted by SELF_REF


# ── BLANKET_APOLOGY ──────────────────────────────────────────────

class TestBlanketApology:

    def test_sorry_for_everything(self):
        """'im sorry for everything' -> BLANKET_APOLOGY."""
        matches = _detect("im sorry for everything")
        assert _has_pattern(matches, "BLANKET_APOLOGY")

    def test_apologize_to_everyone(self):
        """'I apologize to everyone' -> BLANKET_APOLOGY."""
        matches = _detect("I apologize to everyone")
        assert _has_pattern(matches, "BLANKET_APOLOGY")

    def test_sorry_for_being_late_safe(self):
        """'im sorry for being late' -- specific apology, NOT blanket."""
        matches = _detect("im sorry for being late")
        assert not _has_pattern(matches, "BLANKET_APOLOGY")


# ── SELF_REMOVAL ─────────────────────────────────────────────────

class TestSelfRemoval:

    def test_happier_without_me(self):
        """'they would be happier if I wasnt here' -> SELF_REMOVAL."""
        matches = _detect("they would be happier if I wasnt here")
        assert _has_pattern(matches, "SELF_REMOVAL")

    def test_better_off_without(self):
        """'everyone is better off without me' -> SELF_REMOVAL."""
        matches = _detect("everyone is better off without me")
        assert _has_pattern(matches, "SELF_REMOVAL")

    def test_better_at_cooking_safe(self):
        """'she is better at cooking than me' -- comparison, not removal."""
        matches = _detect("she is better at cooking than me")
        # No conditional word present, should not match
        assert not _has_pattern(matches, "SELF_REMOVAL")


# ── SUSPICIOUS_CALM ──────────────────────────────────────────────

class TestSuspiciousCalm:

    def test_finally_at_peace_is_relief(self):
        """'I finally feel at peace' = present tense breakthrough, NOT suspicious."""
        matches = _detect("I finally feel at peace")
        assert not _has_pattern(matches, "SUSPICIOUS_CALM")

    def test_decided_calm_is_suspicious(self):
        """'i feel calm now that ive decided' -> SUSPICIOUS_CALM."""
        matches = _detect("i feel calm now that ive decided")
        assert _has_pattern(matches, "SUSPICIOUS_CALM")

    def test_ready_to_go_is_not_suspicious(self):
        """'im ready to go now' = could be waiting for a ride. Needs conversation context."""
        matches = _detect("im ready to go now")
        assert not _has_pattern(matches, "SUSPICIOUS_CALM")

    def test_peace_without_finally_safe(self):
        """'I feel at peace' -- no 'finally', not suspicious."""
        matches = _detect("I feel at peace")
        assert not _has_pattern(matches, "SUSPICIOUS_CALM")


# ── EXHAUSTION ───────────────────────────────────────────────────

class TestExhaustion:

    def test_cant_take_anymore(self):
        """'I cant take this anymore' -> EXHAUSTION."""
        matches = _detect("I cant take this anymore")
        assert _has_pattern(matches, "EXHAUSTION")

    def test_cant_do_this(self):
        """'I cant do this' -> EXHAUSTION (without temporal, lower conf)."""
        matches = _detect("I cant do this")
        assert _has_pattern(matches, "EXHAUSTION")

    def test_cant_find_keys_safe(self):
        """'I cant find my keys' -- 'find' is ACQUIRE not sustain."""
        matches = _detect("I cant find my keys")
        assert not _has_pattern(matches, "EXHAUSTION")


# ── SARCASM_INVERSION ────────────────────────────────────────────

class TestSarcasmInversion:

    # ── Contradiction feature: Surface-Context Mismatch ─────────
    def test_great_another_monday(self):
        """'oh great another monday' -> surface-context mismatch."""
        matches = _detect("oh great another monday")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    def test_great_another_meeting(self):
        """'oh great another meeting' -> surface-context mismatch."""
        matches = _detect("oh great another meeting")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    # ── Contradiction feature: Mock Praise ──────────────────────
    def test_nice_work_genius(self):
        """'nice work genius' -> mock praise (positive + ironic title)."""
        matches = _detect("nice work genius")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    # ── Contradiction feature: Dismissive Assent ────────────────
    def test_yeah_right(self):
        """'yeah right' -> dismissive assent (hollow + echo)."""
        matches = _detect("yeah right")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    def test_oh_sure_exactly_what_i_needed(self):
        """'oh sure thats exactly what i needed' -> dismissive assent."""
        matches = _detect("oh sure thats exactly what i needed")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    def test_what_a_wonderful_surprise(self):
        """'what a wonderful surprise' -> dismissive assent (what-a template)."""
        matches = _detect("what a wonderful surprise")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    # ── Contradiction feature: Compressed Sarcasm ───────────────
    def test_oh_joy(self):
        """'oh joy' -> compressed sarcasm (hollow + positive + ultra-short)."""
        matches = _detect("oh joy")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    def test_oh_how_lovely(self):
        """'oh how lovely' -> compressed sarcasm."""
        matches = _detect("oh how lovely")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    def test_wow_thanks_stacked(self):
        """'wow thanks so much for the help' -> stacked positives."""
        matches = _detect("wow thanks so much for the help")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    # ── Permission Hostility: context-dependent ─────────────────
    def test_sure_go_ahead_no_context_is_genuine(self):
        """'sure go ahead' without negative context = genuine permission."""
        matches = _detect("sure go ahead")
        assert not _has_pattern(matches, "SARCASM_INVERSION")

    # ── Safe sentences: no sarcasm ──────────────────────────────
    def test_genuine_great_not_sarcasm(self):
        """'that was a great wonderful performance' -- genuinely positive."""
        matches = _detect("that was a great wonderful performance")
        assert not _has_pattern(matches, "SARCASM_INVERSION")

    def test_love_my_mom_not_sarcasm(self):
        """'I love my mom' -- RELATION_REF blocks sarcasm."""
        matches = _detect("I love my mom")
        assert not _has_pattern(matches, "SARCASM_INVERSION")


# ── NO_EXIT ──────────────────────────────────────────────────────

class TestNoExit:

    def test_no_hope(self):
        """'there is no hope' -> NO_EXIT."""
        matches = _detect("there is no hope")
        assert _has_pattern(matches, "NO_EXIT")

    def test_no_point(self):
        """'there is no point anymore' -> NO_EXIT."""
        matches = _detect("there is no point anymore")
        assert _has_pattern(matches, "NO_EXIT")

    def test_no_milk_safe(self):
        """'there is no milk' -- 'milk' is not an exit concept."""
        matches = _detect("there is no milk")
        assert not _has_pattern(matches, "NO_EXIT")


# ── SELF_NULLIFY ─────────────────────────────────────────────────

class TestSelfNullify:

    def test_i_am_nothing(self):
        """'I am nothing' -> SELF_NULLIFY."""
        matches = _detect("I am nothing")
        assert _has_pattern(matches, "SELF_NULLIFY")

    def test_i_am_worthless(self):
        """'I am worthless' -> SELF_NULLIFY."""
        matches = _detect("I am worthless")
        assert _has_pattern(matches, "SELF_NULLIFY")

    def test_i_am_tired_safe(self):
        """'I am tired' -- 'tired' is not a null word."""
        matches = _detect("I am tired")
        assert not _has_pattern(matches, "SELF_NULLIFY")


# ── CHOPPER_SPLIT ────────────────────────────────────────────────

class TestChopperSplit:

    def test_but_splits(self):
        """'I was fine but now everything hurts' -> CHOPPER_SPLIT."""
        matches = _detect("I was fine but now everything hurts")
        assert _has_pattern(matches, "CHOPPER_SPLIT")

    def test_however_splits(self):
        """'things were good however it changed' -> CHOPPER_SPLIT."""
        matches = _detect("things were good however it changed")
        assert _has_pattern(matches, "CHOPPER_SPLIT")


# ── SAFE sentences should NOT flag crisis patterns ───────────────

class TestSafeSentences:
    """These everyday sentences must NOT trigger crisis patterns."""

    CRISIS_PATTERNS = {
        "FAREWELL", "METHOD_ACQUISITION", "BLANKET_APOLOGY",
        "SELF_REMOVAL", "SUSPICIOUS_CALM", "EXHAUSTION",
        "NO_EXIT", "SELF_NULLIFY",
    }

    def _assert_no_crisis(self, sentence):
        matches = _detect(sentence)
        crisis_found = [
            m.pattern for m in matches if m.pattern in self.CRISIS_PATTERNS
        ]
        assert not crisis_found, (
            f"'{sentence}' falsely flagged: {crisis_found}"
        )

    def test_bad_day(self):
        self._assert_no_crisis("im having a bad day")

    def test_work_stressful(self):
        self._assert_no_crisis("work was stressful")

    def test_mondays_suck(self):
        self._assert_no_crisis("mondays suck")

    def test_traffic_was_terrible(self):
        self._assert_no_crisis("traffic was terrible today")

    def test_need_coffee(self):
        self._assert_no_crisis("I need more coffee")

    def test_forgot_lunch(self):
        self._assert_no_crisis("I forgot my lunch at home")


# ── StructureMatch dataclass ─────────────────────────────────────

class TestStructureMatch:

    def test_has_required_fields(self):
        m = StructureMatch(
            pattern="TEST",
            confidence=0.8,
            matched_indices=[0, 1],
            description="test match",
        )
        assert m.pattern == "TEST"
        assert m.confidence == 0.8
        assert m.v_weight == 0.0  # default

    def test_weight_fields(self):
        m = StructureMatch(
            pattern="TEST",
            confidence=0.8,
            matched_indices=[0],
            description="test",
            v_weight=-30.0,
            d_weight=-20.0,
            u_weight=40.0,
            g_weight=50.0,
        )
        assert m.v_weight == -30.0
        assert m.g_weight == 50.0


# ── ATMOSPHERIC_GRIEF ───────────────────────────────────────────

class TestAtmosphericGrief:

    def test_his_chair_still_at_table(self):
        """'his chair is still at the table' -> ATMOSPHERIC_GRIEF."""
        matches = _detect("his chair is still at the table")
        assert _has_pattern(matches, "ATMOSPHERIC_GRIEF")

    def test_found_her_necklace(self):
        """'i found her necklace in the drawer' -> ATMOSPHERIC_GRIEF."""
        matches = _detect("i found her necklace in the drawer")
        assert _has_pattern(matches, "ATMOSPHERIC_GRIEF")

    def test_coffee_mug_hasnt_moved(self):
        """'the coffee mug hasnt moved' -> ATMOSPHERIC_GRIEF."""
        matches = _detect("the coffee mug hasnt moved")
        assert _has_pattern(matches, "ATMOSPHERIC_GRIEF")

    def test_no_trigger_comfortable_chair(self):
        """'his chair is comfortable' -- no absence/persistence, NOT atmospheric grief."""
        matches = _detect("his chair is comfortable")
        assert not _has_pattern(matches, "ATMOSPHERIC_GRIEF")

    def test_no_trigger_active_possessor(self):
        """'she sat in her chair' -- person is active, NOT atmospheric grief."""
        matches = _detect("she sat in her chair")
        assert not _has_pattern(matches, "ATMOSPHERIC_GRIEF")

    def test_grief_score_negative_v(self):
        """Atmospheric grief should push V below center (128)."""
        matches = _detect("his chair is still at the table")
        m = _get_pattern(matches, "ATMOSPHERIC_GRIEF")
        assert m is not None
        assert m.v_weight < 0, "V weight should be negative (grief)"
        assert m.g_weight > 0, "G weight should be positive (heavy)"
        assert m.d_weight < 0, "D weight should be negative (helpless)"


# ── CONTRADICTION_RESOLVE ──────────────────────────────────────

class TestContradictionResolve:

    def test_painfully_beautiful(self):
        """'painfully beautiful' -> adjective head governs, positive."""
        matches = _detect("painfully beautiful")
        assert _has_pattern(matches, "CONTRADICTION_RESOLVE")
        m = _get_pattern(matches, "CONTRADICTION_RESOLVE")
        assert m.v_weight > 0, "Adjective head should make V positive"

    def test_hate_love(self):
        """'i hate how much i love you' -> main verb hate dominates."""
        matches = _detect("i hate how much i love you")
        assert _has_pattern(matches, "CONTRADICTION_RESOLVE")
        m = _get_pattern(matches, "CONTRADICTION_RESOLVE")
        assert m.v_weight < 0, "Main verb 'hate' should dominate -> negative"

    def test_sweet_revenge(self):
        """'sweet revenge' -> noun head governs, negative."""
        matches = _detect("sweet revenge")
        assert _has_pattern(matches, "CONTRADICTION_RESOLVE")
        m = _get_pattern(matches, "CONTRADICTION_RESOLVE")
        assert m.v_weight < 0, "Noun head 'revenge' should make V negative"

    def test_hurts_so_good(self):
        """'it hurts so good' -> complement overrides verb, positive."""
        matches = _detect("it hurts so good")
        assert _has_pattern(matches, "CONTRADICTION_RESOLVE")
        m = _get_pattern(matches, "CONTRADICTION_RESOLVE")
        assert m.v_weight > 0, "Qualifying complement 'so good' should override"

    def test_no_trigger_on_plain_positive(self):
        """'really beautiful' -> no contradiction, no trigger."""
        matches = _detect("really beautiful")
        assert not _has_pattern(matches, "CONTRADICTION_RESOLVE")

    def test_no_trigger_on_plain_negative(self):
        """'absolutely terrible' -> no contradiction, no trigger."""
        matches = _detect("absolutely terrible")
        assert not _has_pattern(matches, "CONTRADICTION_RESOLVE")


# ── NUMBERS_CONTEXT ────────────────────────────────────────────

class TestNumbersContext:

    def test_sleep_deprivation(self):
        """'i only slept 3 hours' -> sleep deprivation detected."""
        matches = _detect("i only slept 3 hours")
        assert _has_pattern(matches, "NUMBERS_CONTEXT")
        m = _get_pattern(matches, "NUMBERS_CONTEXT")
        assert m.v_weight < 0, "Sleep deprivation should be negative"

    def test_only_one_at_birthday(self):
        """'only one person came to my birthday' -> social isolation."""
        matches = _detect("only one person came to my birthday")
        assert _has_pattern(matches, "NUMBERS_CONTEXT")
        m = _get_pattern(matches, "NUMBERS_CONTEXT")
        assert m.v_weight < 0

    def test_no_one_at_party(self):
        """'no one came to my party' -> social isolation."""
        matches = _detect("no one came to my party")
        assert _has_pattern(matches, "NUMBERS_CONTEXT")
        m = _get_pattern(matches, "NUMBERS_CONTEXT")
        assert m.v_weight < 0

    def test_no_trigger_normal_sleep(self):
        """'i slept 8 hours' -> no deprivation signal."""
        matches = _detect("i slept 8 hours")
        assert not _has_pattern(matches, "NUMBERS_CONTEXT")

    def test_no_trigger_everyone_at_party(self):
        """'everyone came to my party' -> no isolation."""
        matches = _detect("everyone came to my party")
        assert not _has_pattern(matches, "NUMBERS_CONTEXT")


# ── NEGATED_NEGATIVE_COMPLIMENT ────────────────────────────────

class TestNegatedNegativeCompliment:

    def test_without_you(self):
        """'i couldnt have done it without you' -> positive compliment."""
        matches = _detect("i couldnt have done it without you")
        assert _has_pattern(matches, "NEGATED_NEGATIVE_COMPLIMENT")
        m = _get_pattern(matches, "NEGATED_NEGATIVE_COMPLIMENT")
        assert m.v_weight > 0, "Double negation should be positive"

    def test_reason_didnt_give_up(self):
        """'youre the reason i didnt give up' -> positive compliment."""
        matches = _detect("youre the reason i didnt give up")
        assert _has_pattern(matches, "NEGATED_NEGATIVE_COMPLIMENT")
        m = _get_pattern(matches, "NEGATED_NEGATIVE_COMPLIMENT")
        assert m.v_weight > 0

    def test_cant_thank_enough(self):
        """'i cant thank you enough' -> positive compliment."""
        matches = _detect("i cant thank you enough")
        assert _has_pattern(matches, "NEGATED_NEGATIVE_COMPLIMENT")
        m = _get_pattern(matches, "NEGATED_NEGATIVE_COMPLIMENT")
        assert m.v_weight > 0

    def test_wouldnt_be_here_without_you(self):
        """'i wouldnt be here without you' -> positive compliment."""
        matches = _detect("i wouldnt be here without you")
        assert _has_pattern(matches, "NEGATED_NEGATIVE_COMPLIMENT")

    def test_no_trigger_plain_negation(self):
        """'i dont like you' -> not a compliment."""
        matches = _detect("i dont like you")
        assert not _has_pattern(matches, "NEGATED_NEGATIVE_COMPLIMENT")


# ── RECOVERY_SMALL_WIN ─────────────────────────────────────────

class TestRecoverySmallWin:

    def test_got_out_of_bed_today(self):
        """'i got out of bed today' -> recovery milestone."""
        matches = _detect("i got out of bed today")
        assert _has_pattern(matches, "RECOVERY_SMALL_WIN")
        m = _get_pattern(matches, "RECOVERY_SMALL_WIN")
        assert m.v_weight > 0, "Small win should be positive"
        assert m.w_weight > 0, "Small win should boost self-worth"

    def test_ate_a_full_meal(self):
        """'i ate a full meal' -> recovery signal."""
        matches = _detect("i ate a full meal")
        assert _has_pattern(matches, "RECOVERY_SMALL_WIN")

    def test_finally_took_a_shower(self):
        """'i finally took a shower' -> small win with temporal marker."""
        matches = _detect("i finally took a shower")
        assert _has_pattern(matches, "RECOVERY_SMALL_WIN")
        m = _get_pattern(matches, "RECOVERY_SMALL_WIN")
        assert m.confidence >= 0.85, "Temporal marker should boost confidence"

    def test_went_outside_first_time(self):
        """'i went outside for the first time' -> recovery milestone."""
        matches = _detect("i went outside for the first time")
        assert _has_pattern(matches, "RECOVERY_SMALL_WIN")

    def test_no_trigger_mundane_no_context(self):
        """'the shower is broken' -> no recovery signal without self-ref/temporal."""
        matches = _detect("the shower is broken")
        assert not _has_pattern(matches, "RECOVERY_SMALL_WIN")

    def test_no_trigger_normal_routine(self):
        """'i need some coffee' -> not a recovery win."""
        matches = _detect("i need some coffee")
        assert not _has_pattern(matches, "RECOVERY_SMALL_WIN")
