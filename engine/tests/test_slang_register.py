"""Tests for the Batch 1 slang-register fixes (2026-06-11).

Covers:
  1. New slang vocabulary (dope, dogwater, washed, masterclass, ...)
  2. 'w' sense disambiguation: WIN vs "with"
  3. "lets go" hype compound + elongation collapse ("LETS GOOOOOO")
  4. "as hell" as pure intensifier (no literal hell charge)
  5. Expletive-as-intensifier rescue after demonstratives ("This shit slaps")
  6. Fuzzy stemmer blocklist ('number' must not stem to 'numb')
  7. Regression guards on regraded words (clean, extra, trap)
"""

from engine.pendulum import compute_vadug


def _v(text: str) -> float:
    result, _ = compute_vadug(text)
    return result.v


class TestNewSlangVocab:

    def test_dope_with_elongated_greeting(self):
        assert _v("Yoooooo, that's dope!!!!!!!!!!!") >= 140

    def test_dogwater_negative(self):
        assert _v("the servers are actually dogwater tonight") <= 123

    def test_washed_negative(self):
        assert _v("batista is washed bro") <= 124

    def test_ggs_masterclass(self):
        assert _v("ggs that was a masterclass") >= 140


class TestWDisambiguation:

    def test_w_as_with_in_food_order(self):
        assert 112 <= _v("ordered it w extra cheese") <= 144

    def test_w_as_with_before_name(self):
        assert 105 <= _v("don't play w jynxzi") <= 144

    def test_w_as_win_after_massive(self):
        assert _v("massive W for the whole team today") >= 138


class TestHypeCompounds:

    def test_lets_go_elongated(self):
        assert _v("LETS GOOOOOO") >= 140

    def test_clean_as_hell(self):
        assert _v("new drop is clean as hell") >= 133

    def test_demonstrative_expletive_rescue_with_trap_genre(self):
        assert _v("This shit slaps, we need the trap remix") >= 140


class TestStemmerBlocklist:

    def test_number_does_not_stem_to_numb(self):
        assert 105 <= _v("she peaked at number three on the charts") <= 144


class TestRegressionGuards:

    def test_happy_stays_high(self):
        assert _v("I am so happy for you") >= 150

    def test_crisis_stays_low(self):
        assert _v("I want to die") <= 60

    def test_literal_hell_stays_negative(self):
        assert _v("this is hell") <= 110

    def test_literal_extra_quantity(self):
        assert _v("we got extra time to finish") >= 112

    def test_literal_trap_stays_negative(self):
        assert _v("he fell into a trap") <= 120


# ── Batch 2 (2026-06-11) ────────────────────────────────────────


class TestDiscourseNoRescue:

    def test_no_before_self_ref_illness(self):
        assert _v("no i'm sick with a sore throat") <= 123

    def test_no_before_self_ref_negation_preserved(self):
        assert _v("no i don't want that") <= 110

    def test_no_positive_rescue_still_works(self):
        assert _v("no we good") >= 140


class TestCookedLiteralGuard:

    def test_cooked_dinner_for_fam(self):
        assert 112 <= _v("i cooked dinner for the fam tonight") <= 144

    def test_cooked_on_the_grill(self):
        # 111 as of batch 2: "on" is not a literal-object follower, so the
        # stored slang charge still applies; held just under neutral by
        # W-V coupling. Band guards against regression in either direction.
        assert 105 <= _v("I cooked on the grill so yes") <= 144

    def test_home_cooked_meal(self):
        assert 112 <= _v("3 pennes and a home cooked meal") <= 144

    def test_im_cooked_bro_still_flips_positive(self):
        assert _v("im cooked bro") >= 133

    def test_im_cooked_stays_doomed(self):
        assert _v("im cooked") < 110


class TestFireSenseSplit:

    def test_fire_hazard_in_kitchen(self):
        result, _ = compute_vadug("theres a fire in my kitchen")
        assert result.v <= 123
        assert result.u >= 25

    def test_fire_praise_its(self):
        assert _v("Play bite by night on Roblox it's fire") >= 133

    def test_fire_praise_be(self):
        assert _v("game might actually be fire without the broom tho lol") >= 133

    def test_fire_away_not_strongly_positive(self):
        assert _v("fire away with your questions") <= 135

    def test_fire_him_not_strongly_positive(self):
        assert _v("they had to fire him") <= 145


class TestPraiseNounGuard:

    def test_you_the_goat_fr(self):
        assert _v("you the goat fr") >= 145

    def test_good_for_you_stays_pa(self):
        assert _v("good for you") <= 110

    def test_must_be_nice_stays_pa(self):
        assert _v("must be nice") <= 112


class TestRegisterCasualInertSkip:

    def test_trailing_ngl_no_drain(self):
        assert _v("yo this game gives me hope for the human race man ngl") >= 133

    def test_helluva_combo_ngl(self):
        assert _v("Yoo pink and teal, that's one helluva combo ngl") >= 133

    def test_elongated_good_burgers(self):
        assert _v("yall i made 2 big burgers today i ate GOOOOOD") >= 133


class TestHypeIdioms:

    def test_we_are_so_back(self):
        assert _v("WE ARE SO BACK") >= 140

    def test_turned_his_back_unaffected(self):
        assert _v("he turned his back") <= 128

    def test_stupid_good_amplifier(self):
        assert _v("the chorus on this track is stupid good") >= 140

    def test_stupid_mistake_stays_negative(self):
        assert _v("stupid mistake") <= 110

    def test_he_is_stupid_stays_negative(self):
        assert _v("he is stupid") <= 115


class TestVocabPhaseBatch2:

    def test_lowkey_obsessed_flips_positive(self):
        assert _v("honestly lowkey obsessed with this app") >= 133

    def test_obsessed_third_person_stays_nonpositive(self):
        assert _v("he is obsessed with her every move") <= 128

    def test_filthy_sports_praise(self):
        assert _v("that play was filthy") >= 133

    def test_filthy_bathroom_stays_nonpositive(self):
        assert _v("the bathroom was filthy") <= 128

    def test_mid_game_negative(self):
        assert _v("mid game tbh, wouldn't recommend") <= 123

    def test_mid_predicate_stays_negative(self):
        assert _v("the new album is mid") <= 110

    def test_mid_morning_temporal_neutral(self):
        assert 105 <= _v("mid morning meeting") <= 144
