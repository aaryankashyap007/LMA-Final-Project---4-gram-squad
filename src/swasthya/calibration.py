"""Temperature scaling for the gate's class probabilities (fit on dev, applied to test).

The gate's probabilities are a softmax over four letter logits, so log p equals the logits up to a constant and
dividing log p by T is exactly temperature scaling. T > 1 softens over-confident probabilities. The per-sample
argmax is unchanged for a fixed T (accuracy, macro-F1 and the default decisions do not move); only ECE and any
threshold-based analysis (e.g. the H1 matched over-triage budgets) can change.
"""
import numpy as np

from .metrics import ece
from .schema import LABEL2ID

GRID = np.exp(np.linspace(np.log(0.2), np.log(10.0), 241))


def _clip(probs):
    return np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)


def apply_temperature(probs, T):
    z = np.log(_clip(probs)) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs, y_true):
    idx = np.array([LABEL2ID[y] for y in y_true])
    return float(-np.mean(np.log(_clip(probs)[np.arange(len(idx)), idx])))


def fit_temperature(probs, y_true, grid=GRID):
    """Grid search for the T that minimises negative log-likelihood of the gold labels."""
    scores = [nll(apply_temperature(probs, T), y_true) for T in grid]
    return float(grid[int(np.argmin(scores))])


def bootstrap_temperature(probs, y_true, groups, n=200, seed=0):
    """95% interval of the fitted T when dev CASES are resampled (a case's script renderings stay together)."""
    rng = np.random.default_rng(seed)
    probs, y, groups = np.asarray(probs), np.asarray(y_true), np.asarray(groups)
    uniq = np.unique(groups)
    members = [np.where(groups == u)[0] for u in uniq]
    fits = []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        fits.append(fit_temperature(probs[idx], list(y[idx])))
    return float(np.percentile(fits, 2.5)), float(np.percentile(fits, 97.5))


def calibration_report(dev_probs, dev_y, test_probs, test_y, T):
    return {"temperature": T,
            "ece_dev_before": ece(dev_probs, dev_y), "ece_dev_after": ece(apply_temperature(dev_probs, T), dev_y),
            "ece_test_before": ece(test_probs, test_y), "ece_test_after": ece(apply_temperature(test_probs, T), test_y),
            "nll_test_before": nll(test_probs, test_y), "nll_test_after": nll(apply_temperature(test_probs, T), test_y)}
