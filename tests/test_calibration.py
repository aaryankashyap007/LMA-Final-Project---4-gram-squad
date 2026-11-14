import numpy as np

from swasthya.calibration import (apply_temperature, bootstrap_temperature, calibration_report, fit_temperature,
                                  nll)
from swasthya.schema import LABELS


def overconfident(n=3000, true_T=3.0, seed=0):
    """Labels drawn from softmax(z); the 'model' reports softmax(true_T * z), i.e. is over-confident by true_T."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, size=(n, 4))
    p_true = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = [LABELS[rng.choice(4, p=p)] for p in p_true]
    zs = z * true_T
    return np.exp(zs) / np.exp(zs).sum(1, keepdims=True), y


def test_temperature_one_is_identity_and_rows_sum_to_one():
    p, _ = overconfident(50)
    q = apply_temperature(p, 1.0)
    assert np.allclose(q, p, atol=1e-9) and np.allclose(apply_temperature(p, 2.5).sum(1), 1.0)


def test_argmax_is_unchanged_by_scaling():
    p, _ = overconfident(500)
    for T in (0.5, 2.0, 7.0):
        assert (apply_temperature(p, T).argmax(1) == p.argmax(1)).all()


def test_fit_recovers_the_overconfidence_and_lowers_nll_and_ece():
    p, y = overconfident()
    T = fit_temperature(p, y)
    assert 2.4 < T < 3.7
    rep = calibration_report(p, y, p, y, T)
    assert rep["nll_test_after"] < rep["nll_test_before"] and rep["ece_test_after"] < rep["ece_test_before"]


def test_well_calibrated_probabilities_get_t_near_one():
    p, y = overconfident(true_T=1.0, seed=3)
    assert 0.8 < fit_temperature(p, y) < 1.25


def test_underconfident_model_gets_t_below_one():
    p, y = overconfident(true_T=0.4, seed=5)
    assert fit_temperature(p, y) < 0.7


def test_bootstrap_interval_brackets_the_point_estimate():
    p, y = overconfident(600, seed=7)
    groups = [f"c{i // 3}" for i in range(len(y))]                          # three renderings per case
    T = fit_temperature(p, y)
    lo, hi = bootstrap_temperature(p, y, groups, n=30)
    assert lo <= T <= hi and lo < hi


def test_nll_is_zero_for_perfect_confident_predictions():
    p = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    assert nll(p, ["EMERGENCY", "URGENT"]) < 1e-9
