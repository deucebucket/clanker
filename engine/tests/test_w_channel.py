"""Board 2: W (self-worth) attribution routing + I (intent) agency axis.

W is valence routed through WHO it lands on:
  - self-declarative ("i am worthless")           -> full attribution, W drops hard
  - emotional force targeting self ("he told me") -> strong attribution
  - self as actor of negative act (guilt)         -> partial attribution, W dips
  - atmospheric / other-directed                  -> no attribution, W stable

I gains an agency axis on top of the directional (withdraw/connect/control)
logic: volition markers lift I, futility markers sink it.
"""

import pytest

from engine.pendulum import compute_vadug


def _vadug(text):
    result, _ = compute_vadug(text)
    return result


class TestWAttribution:
    def test_self_declarative_drops_w_hard(self):
        """SELF as subject + emotional predicate = full attribution."""
        r = _vadug("i am worthless")
        assert r.w < 80, f"W={r.w} should drop hard for self-declaration"

    def test_target_self_drops_w(self):
        """Emotional force whose target is self lands on W."""
        r = _vadug("he told me im nothing")
        assert r.w < 100, f"W={r.w} should drop when self is the target"

    def test_atmospheric_w_stable(self):
        """Other-directed/atmospheric forces say nothing about MY worth."""
        r = _vadug("the weather is awful")
        assert r.w >= 110, f"W={r.w} should stay stable for atmospheric negativity"
        r2 = _vadug("they announced layoffs")
        assert r2.w >= 110, f"W={r2.w} should stay stable for other-directed news"

    def test_guilt_partial_attribution(self):
        """Self harming another dips W, but far less than self-declaration."""
        guilt = _vadug("i hurt her badly")
        declarative = _vadug("i am worthless")
        assert guilt.w < 128, f"W={guilt.w} should dip below neutral for guilt"
        assert guilt.w >= 95, f"W={guilt.w} should not collapse for guilt"
        assert guilt.w > declarative.w, (
            f"guilt W={guilt.w} must stay above self-declarative W={declarative.w}"
        )

    def test_masked_collapse_reads_low_w(self):
        """Crisis patterns with damaged self-worth push W into the 80-100 band."""
        for text in (
            "cant do this shit anymore",
            "theres no way out",
            "i just want it to be over",
            "i stopped making plans because whats the point",
        ):
            r = _vadug(text)
            assert r.w <= 100, f"W={r.w} should be <= 100 for: {text!r}"


class TestIntentAgency:
    def test_agency_raises_intent(self):
        """Volition marker + action verb lifts I toward control."""
        for text in ("im going to fix this tomorrow", "i need to finish this project"):
            r = _vadug(text)
            assert r.i >= 160, f"I={r.i} should be >= 160 for agency: {text!r}"

    def test_futility_drops_intent(self):
        """Powerlessness/futility markers sink I toward deflect."""
        for text in ("whats the point", "i give up", "why bother trying"):
            r = _vadug(text)
            assert r.i <= 80, f"I={r.i} should be <= 80 for futility: {text!r}"

    def test_agency_travel_not_boosted(self):
        """'im going to the store' is travel, not agency."""
        r = _vadug("im going to the store")
        assert r.i < 160, f"I={r.i} should stay neutral for destinations"

    def test_negated_give_up_not_futility(self):
        """'didnt give up' is perseverance — must not sink I."""
        r = _vadug("she didnt give up on me")
        assert r.i > 64, f"I={r.i} should not read negated give-up as futility"
