"""The VADUGWI held-out evaluation helpers (no data files needed)."""

import pytest

from evaluation.vadugwi import core
from evaluation.vadugwi.fit_reliability import posterior_below_half


def test_classify_uses_the_repo_thresholds():
    assert [core.classify(v) for v in (109, 110, 144, 145)] == ["neg", "neutral", "neutral", "pos"]


def test_auc_counts_ties_as_half():
    assert core.auc_pos_vs_neg([200, 100], ["pos", "neg"]) == 1.0
    assert core.auc_pos_vs_neg([128, 128], ["pos", "neg"]) == 0.5
    assert core.auc_pos_vs_neg([100, 200], ["pos", "neg"]) == 0.0


def test_class_metrics_report_the_majority_baseline():
    m = core.class_metrics([90, 90, 128, 200], ["neg", "neg", "neutral", "neg"])
    assert m["accuracy"] == 0.75
    assert m["majority_class_accuracy"] == 0.75
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1.0) / 2)


@pytest.mark.parametrize("name", ["council_heldout", "heldout_v1", "emobank_test",
                                  "sst2_validation", "stress"])
def test_fitting_on_non_fit_data_fails_closed(name):
    with pytest.raises(PermissionError):
        core.load(name, "fit")


def test_paired_bootstrap_of_identical_runs_is_zero():
    truth = ["neg", "pos", "neutral", "neg"] * 5
    values = [90, 160, 128, 120] * 5
    ci = core.bootstrap("class", truth, values, other=values, draws=50)
    assert ci["accuracy"] == [0.0, 0.0]


def test_posterior_below_half():
    assert posterior_below_half(0, 0) == pytest.approx(0.5, abs=1e-3)
    assert posterior_below_half(87, 224) > 0.999
    assert posterior_below_half(87, 11) < 0.001
