"""Tests for V3 proximity weighting — distance-based influence fields."""

import pytest

from engine.word_classifier import WordRole, classify_sentence
from engine.proximity import (
    PROXIMITY_DECAY,
    compute_proximity_field,
    find_role_pairs,
    proximity_coefficient,
)


# ── Helper ───────────────────────────────────────────────────────

def _roles(sentence: str):
    """Classify a sentence into WordRoles."""
    return classify_sentence(sentence.split())


# ── compute_proximity_field tests ────────────────────────────────

class TestProximityField:

    def test_adjacent_words_have_decay_influence(self):
        """Adjacent words (distance 1) should have ~0.7 influence."""
        roles = _roles("I am happy")
        field = compute_proximity_field(roles)
        # word 0 -> word 1 = distance 1
        assert abs(field[0][1] - PROXIMITY_DECAY) < 1e-9

    def test_distant_words_have_decayed_influence(self):
        """Words at distance 4 should have ~0.7^4 = ~0.24 influence."""
        roles = _roles("I gave my dog to neighbor")
        field = compute_proximity_field(roles)
        # word 0 -> word 4 = distance 4
        expected = PROXIMITY_DECAY ** 4
        assert abs(field[0][4] - expected) < 1e-9

    def test_influence_is_symmetric(self):
        """Influence from A->B should equal B->A."""
        roles = _roles("very happy today")
        field = compute_proximity_field(roles)
        assert abs(field[0][2] - field[2][0]) < 1e-9

    def test_no_self_influence(self):
        """A word should not appear in its own influence map."""
        roles = _roles("I am fine")
        field = compute_proximity_field(roles)
        for idx, influences in field.items():
            assert idx not in influences

    def test_empty_sentence(self):
        """Empty input produces empty field."""
        field = compute_proximity_field([])
        assert field == {}

    def test_single_word(self):
        """Single word has no influences."""
        roles = _roles("hello")
        field = compute_proximity_field(roles)
        assert field[0] == {}

    def test_cutoff_applied(self):
        """Distant words eventually fall below influence cutoff."""
        from engine.proximity import PROXIMITY_DECAY, INFLUENCE_CUTOFF
        # Find the distance where influence drops below cutoff
        import math
        cutoff_dist = int(math.log(INFLUENCE_CUTOFF) / math.log(PROXIMITY_DECAY)) + 1
        roles = _roles(" ".join(f"w{i}" for i in range(cutoff_dist + 5)))
        field = compute_proximity_field(roles)
        # distance cutoff_dist-2 should be included
        assert cutoff_dist - 2 in field[0]
        # distance cutoff_dist+1 should be excluded
        assert cutoff_dist + 1 not in field[0]


# ── find_role_pairs tests ────────────────────────────────────────

class TestFindRolePairs:

    def test_transfer_relation_found(self):
        """'I gave my dog to neighbor' should find TRANSFER+RELATION_REF."""
        roles = _roles("I gave my dog to neighbor")
        # dog is RELATION_REF now (relationship, not possession)
        pairs = find_role_pairs(roles, "TRANSFER", "RELATION_REF")
        assert len(pairs) > 0
        transfer_idxs = [r.position for r in roles if r.role == "TRANSFER"]
        relation_idxs = [r.position for r in roles if r.role == "RELATION_REF"]
        pair_as = [p[0] for p in pairs]
        pair_bs = [p[1] for p in pairs]
        assert any(a in transfer_idxs for a in pair_as)
        assert any(b in relation_idxs for b in pair_bs)

    def test_acquire_method_found(self):
        """'just bought some pills' should find ACQUIRE+METHOD."""
        roles = _roles("just bought some pills")
        pairs = find_role_pairs(roles, "ACQUIRE", "METHOD")
        assert len(pairs) > 0
        # "bought" is ACQUIRE, "pills" is METHOD
        acquire_idxs = [r.position for r in roles if r.role == "ACQUIRE"]
        method_idxs = [r.position for r in roles if r.role == "METHOD"]
        pair_as = [p[0] for p in pairs]
        pair_bs = [p[1] for p in pairs]
        assert any(a in acquire_idxs for a in pair_as)
        assert any(b in method_idxs for b in pair_bs)

    def test_sorted_strongest_first(self):
        """Pairs should be sorted by proximity strength, strongest first."""
        roles = _roles("I gave my dog to neighbor")
        pairs = find_role_pairs(roles, "TRANSFER", "POSSESSION")
        if len(pairs) > 1:
            for i in range(len(pairs) - 1):
                assert pairs[i][2] >= pairs[i + 1][2]

    def test_max_distance_respected(self):
        """Pairs beyond max_distance should not be returned."""
        roles = _roles("gave a b c d e f g dog")
        pairs = find_role_pairs(roles, "TRANSFER", "POSSESSION", max_distance=3)
        for _, _, strength in pairs:
            # All returned pairs must have distance <= 3
            assert strength >= PROXIMITY_DECAY ** 3

    def test_no_pairs_returns_empty(self):
        """If no matching role pair exists, return empty list."""
        roles = _roles("hello world")
        pairs = find_role_pairs(roles, "TRANSFER", "METHOD")
        assert pairs == []


# ── proximity_coefficient tests ──────────────────────────────────

class TestProximityCoefficient:

    def test_amplifier_boosts(self):
        """Nearby AMPLIFIER should boost coefficient above 1.0."""
        roles = _roles("very happy")
        # "very" = AMPLIFIER at idx 0, target "happy" at idx 1
        coeff = proximity_coefficient(roles, 1)
        assert coeff > 1.0

    def test_negator_flips(self):
        """Nearby NEGATOR should flip coefficient below 0."""
        roles = _roles("not happy")
        # "not" = NEGATOR at idx 0, target "happy" at idx 1
        coeff = proximity_coefficient(roles, 1)
        assert coeff < 0.0

    def test_self_ref_increases_magnitude(self):
        """SELF_REF nearby should increase |coefficient| vs OTHER_REF."""
        roles_self = _roles("I am happy")
        roles_other = _roles("they are happy")
        # "happy" is the last word in both
        coeff_self = proximity_coefficient(roles_self, 2)
        coeff_other = proximity_coefficient(roles_other, 2)
        assert abs(coeff_self) > abs(coeff_other)

    def test_hedge_dampens(self):
        """Nearby HEDGE should dampen coefficient below 1.0."""
        roles = _roles("maybe happy")
        # "maybe" = HEDGE at idx 0, target "happy" at idx 1
        coeff = proximity_coefficient(roles, 1)
        assert coeff < 1.0

    def test_no_modifiers_returns_one(self):
        """Without modifier roles nearby, coefficient should be 1.0."""
        roles = _roles("the dog barked")
        # All NEUTRAL/POSSESSION — no AMPLIFIER/NEGATOR/SELF_REF/HEDGE
        coeff = proximity_coefficient(roles, 0)
        assert coeff == 1.0

    def test_coefficient_capped(self):
        """Coefficient should never exceed [-COEFFICIENT_CAP, COEFFICIENT_CAP]."""
        from engine.proximity import COEFFICIENT_CAP
        roles = _roles("very really extremely totally absolutely happy")
        coeff = proximity_coefficient(roles, 5)
        assert -COEFFICIENT_CAP <= coeff <= COEFFICIENT_CAP

    def test_empty_roles(self):
        """Empty roles list returns 1.0."""
        assert proximity_coefficient([], 0) == 1.0

    def test_single_word(self):
        """Single word has no modifiers — coefficient is 1.0."""
        roles = _roles("happy")
        assert proximity_coefficient(roles, 0) == 1.0

    def test_out_of_bounds_index(self):
        """Out-of-bounds target index returns 1.0."""
        roles = _roles("hello world")
        assert proximity_coefficient(roles, 99) == 1.0
        assert proximity_coefficient(roles, -1) == 1.0
