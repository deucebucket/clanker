from engine_v9.roots import Root, ROOTS, RootCategory


def test_root_has_charge_vector():
    r = ROOTS["HAPPY"]
    assert len(r.charge) == 7
    assert r.charge[0] > 0  # positive valence


def test_root_has_category():
    r = ROOTS["HAPPY"]
    assert r.category == RootCategory.POSITIVE_STATE


def test_negative_root():
    r = ROOTS["SAD"]
    assert r.charge[0] < 0  # negative valence


def test_root_has_phase():
    r = ROOTS["MURDER"]
    assert r.phase == "SOLID"


def test_formulaic_root_zero_charge():
    r = ROOTS["GREETING"]
    assert r.charge[0] == 0
    assert r.charge[1] == 0


def test_compound_event_root():
    r = ROOTS["EMPLOYMENT_LOSS"]
    assert r.charge[0] < -40  # strong negative
    assert r.category == RootCategory.COMPOUND_EVENT


def test_all_roots_have_7d_charge():
    for name, root in ROOTS.items():
        assert len(root.charge) == 7, f"Root {name} has {len(root.charge)}D charge"
